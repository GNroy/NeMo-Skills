#!/bin/bash
# Phase B (Q2 co-location): ONE batch GPU node runs BOTH a registered sglang worker
# (on the 4 GPUs) AND the gym driver for one seed (on the idle CPU cores). CPU
# footprint of the whole 5-seed run drops to the 2 shared-service nodes (router +
# sandbox); the drivers no longer occupy CPU nodes and the GPUs are NOT wasted.
#
# Reuses: gpu_worker.sh's sglang launch+register (proven) + the NeMo-Run-staged
# gym driver scripts nemo-run-{0,1}.sh (static-path parameterized, proven on the
# cpu validation run). Routes the driver's LLM + delegate children to the shared
# router; the co-located server registers INTO that same router/pool.
#   SEED=<k> WORKER_PORT=<p> sbatch -J lazyalloc_q2co_s<k> cojob_q2.sh
#SBATCH -p batch
#SBATCH -A nemotron_reason_science
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
set -uo pipefail

A=/lustre/fsw/portfolios/nemotron/users/alaptev
SEED="${SEED:?set SEED}"
WORKER_PORT="${WORKER_PORT:-30001}"
POOL="$A/data/lazypool_q2"
SGLANG_IMG="$A/containers/sglang-v0.5.11.sqsh"
GYM_IMG="/lustre/fsw/portfolios/nemotron/users/igitman/images/nemo-skills-dc43f3e.sqsh"
MODEL="/hf_models/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
HF="/lustre/fsw/portfolios/nemotron/users/igitman/hf_models"
SWE="/lustre/fsw/portfolios/nemotron/users/igitman/images/swe-bench"

# Reuse the NeMo-Run-staged gym driver scripts + code mount for this seed (proven
# on the cpu run; static /alaptev paths, no job-specific state baked in).
NRUN="$(ls -dt $A/nemo-run/lazyalloc_q2_s${SEED}/*/nemo-run 2>/dev/null | head -1)"
[ -d "$NRUN/scripts" ] || { echo "[cojob] ERROR: no staged nemo-run for seed $SEED ($NRUN)"; exit 1; }
GYM_MOUNTS="${HF}:/hf_models,${SWE}:/swe-bench-images,${A}:/alaptev,${A}/data:/data,${NRUN}:/nemo_run"

# Env the gym driver expects (mirrors cluster_config env_vars + hermes path).
export HERMES_AGENT_PATH=/alaptev/hermes-agent
export HF_HOME=/alaptev/hf-cache TRANSFORMERS_CACHE=/alaptev/hf-cache HUGGINGFACE_HUB_CACHE=/alaptev/hf-cache
export TIKTOKEN_ENCODINGS_BASE=/alaptev/tiktoken-encodings TIKTOKEN_RS_CACHE_DIR=/alaptev/tiktoken-encodings/cache
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=true SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS=16 SGLANG_DG_CACHE_DIR=/alaptev/deep_gemm_cache
export HF_HUB_OFFLINE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
GENV="HERMES_AGENT_PATH,HF_HOME,TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE,TIKTOKEN_ENCODINGS_BASE,TIKTOKEN_RS_CACHE_DIR,SGLANG_JIT_DEEPGEMM_FAST_WARMUP,SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS,SGLANG_DG_CACHE_DIR,HF_HUB_OFFLINE,VLLM_ALLOW_LONG_MAX_MODEL_LEN"

ROUTER_URL="$(cat $POOL/router_url)"
HOST="$(hostname -f)"; SELF="http://${HOST}:${WORKER_PORT}"
WID_FILE="$POOL/cojob_s${SEED}_${WORKER_PORT}.id"
echo "[cojob] seed=$SEED node=$HOST self=$SELF router=$ROUTER_URL nrun=$NRUN"

# Deregister the co-located worker from the router on ANY exit.
cleanup(){ WID=$(cat "$WID_FILE" 2>/dev/null||true); echo "[cojob] dereg $SELF id=$WID"; [ -n "$WID" ] && curl -s -X DELETE "${ROUTER_URL}/workers/${WID}" || true; rm -f "$WID_FILE" 2>/dev/null||true; }
trap cleanup EXIT

# --- 1. hermes_home bootstrap (gym container, no GPU) ---
# NOTE: nemo-run-0.sh does its work (copytree hermes_home) in seconds then
# `sleep infinity` (NeMo-Run's parallel-srun model: any exit kills the group, so
# the bootstrap must linger). So run it in the BACKGROUND and poll for the
# materialized hermes_home; the lingering sleep is harmless (killed at job end).
echo "[cojob] === bootstrap (nemo-run-0.sh, background) ==="
srun --overlap --ntasks=1 --nodes=1 --gpus-per-node=0 --no-container-mount-home \
     --container-image="$GYM_IMG" --container-mounts="$GYM_MOUNTS" --container-workdir /nemo_run/code \
     --container-env="$GENV" bash /nemo_run/scripts/nemo-run-0.sh &
BOOT=$!
HH="$A/exp/a0py_q2_s${SEED}/agents/scientist/hermes_home/config.yaml"
for i in $(seq 1 120); do
  [ -f "$HH" ] && { echo "[cojob] hermes_home ready ($HH)"; break; }
  kill -0 $BOOT 2>/dev/null || { echo "[cojob] bootstrap srun exited before hermes_home ready"; break; }
  sleep 2
done
[ -f "$HH" ] || { echo "[cojob] ERROR: hermes_home not materialized"; exit 1; }

# --- 2. co-located sglang server on the 4 GPUs (background) ---
echo "[cojob] === launching co-located sglang server :$WORKER_PORT ==="
srun --overlap --ntasks=1 --nodes=1 \
     --container-image="$SGLANG_IMG" --container-mounts="${A}:/alaptev,${HF}:/hf_models" \
     bash -c "python3 -m sglang.launch_server --model-path ${MODEL} --host 0.0.0.0 --port ${WORKER_PORT} \
        --tp-size 4 --expert-parallel-size 4 --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 \
        --context-length 262144 --max-running-requests 8 --mem-fraction-static 0.82 --cuda-graph-max-bs 16 \
        --mamba-scheduler-strategy no_buffer --disable-piecewise-cuda-graph --disable-radix-cache \
        --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 5 --reasoning-parser nemotron_3 --tool-call-parser qwen3_coder" &
SRV=$!

# --- 3. registrar (background): wait local server healthy, POST /workers ---
(
  for i in $(seq 1 240); do
    if curl -sf "${SELF}/health" >/dev/null 2>&1; then
      RESP=$(curl -s -X POST -H 'Content-Type: application/json' -d "{\"url\":\"${SELF}\"}" "${ROUTER_URL}/workers")
      echo "[cojob] register resp: $RESP"
      echo "$RESP" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("worker_id",""))' > "$WID_FILE" 2>/dev/null||true
      echo "[cojob] registered id=$(cat "$WID_FILE" 2>/dev/null)"; exit 0
    fi
    kill -0 $SRV 2>/dev/null || { echo "[cojob] server died before healthy"; exit 1; }
    sleep 10
  done
  echo "[cojob] server never healthy; not registered"
) &

# --- 4. gym driver (gym container, no GPU, foreground) — routes to the shared router ---
echo "[cojob] === gym driver (nemo-run-1.sh) ==="
srun --overlap --ntasks=1 --nodes=1 --gpus-per-node=0 --no-container-mount-home \
     --container-image="$GYM_IMG" --container-mounts="$GYM_MOUNTS" --container-workdir /nemo_run/code \
     --container-env="$GENV" bash /nemo_run/scripts/nemo-run-1.sh
echo "[cojob] gym driver exited rc=$?"
