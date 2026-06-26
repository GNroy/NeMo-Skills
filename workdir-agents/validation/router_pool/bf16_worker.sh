#!/bin/bash
# BF16 worker: ONE 2-node sglang server (TP=8 over 2 nodes) serving the GA-BF16
# Ultra checkpoint, registering into the standalone router. Self-deregisters on exit.
# Mirrors the proven co-tenant recipe (serve_sglang --num_nodes 2, EAGLE spec-decode).
#SBATCH -p batch
#SBATCH -A nemotron_reason_science
#SBATCH -t 04:00:00
#SBATCH -J lazyalloc_bf16worker
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
set -uo pipefail

POOL_DIR_HOST="${POOL_DIR_HOST:-/lustre/fsw/portfolios/nemotron/users/alaptev/data/lazypool_bf16}"
WORKER_PORT="${WORKER_PORT:-5656}"
MODEL="${MODEL:-/hf_models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16}"
SGLANG_IMG="${SGLANG_IMG:-/lustre/fsw/portfolios/nemotron/users/alaptev/containers/sglang-v0.5.11.sqsh}"
CODE_HOST=/lustre/fsw/portfolios/nemotron/users/alaptev/NeMo-Skills-clean

# Multinode: head node = where this batch script runs (node_rank 0).
export SLURM_MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD=$(hostname -f)
SELF_URL="http://${HEAD}:${WORKER_PORT}"

# Discover the router URL (published by start_router.sh).
for i in $(seq 1 180); do
  [ -f "${POOL_DIR_HOST}/router_url" ] && break; sleep 5
done
ROUTER_URL=$(cat "${POOL_DIR_HOST}/router_url" 2>/dev/null || true)
if [ -z "$ROUTER_URL" ]; then echo "[bf16worker] no router_url; abort"; exit 1; fi
echo "[bf16worker] job=$SLURM_JOB_ID master=$SLURM_MASTER_NODE head=$HEAD self=$SELF_URL router=$ROUTER_URL"

WID_FILE="${POOL_DIR_HOST}/bf16worker_${SLURM_JOB_ID}.id"

cleanup() {
  WID=$(cat "$WID_FILE" 2>/dev/null || true)
  echo "[bf16worker] deregistering $SELF_URL (id=$WID)"
  [ -n "$WID" ] && curl -s -X DELETE "${ROUTER_URL}/workers/${WID}" || true
  rm -f "$WID_FILE" 2>/dev/null || true
}
trap cleanup EXIT

# Background registrar: wait until the head server answers /health, then POST /workers.
(
  for i in $(seq 1 360); do
    if curl -sf "${SELF_URL}/health" >/dev/null 2>&1; then
      echo "[bf16worker] healthy; POST /workers -> router"
      RESP=$(curl -s -X POST -H "Content-Type: application/json" \
                  -d "{\"url\":\"${SELF_URL}\"}" "${ROUTER_URL}/workers")
      echo "[bf16worker] register resp: $RESP"
      echo "$RESP" | python3 -c "import sys,json;print(json.load(sys.stdin).get(\"worker_id\",\"\"))" \
                   > "$WID_FILE" 2>/dev/null || true
      echo "[bf16worker] registered id=$(cat "$WID_FILE" 2>/dev/null)"
      exit 0
    fi
    sleep 10
  done
  echo "[bf16worker] never became healthy; not registering"
) &

# 2-node serve via the proven serve_sglang wrapper (binds 0.0.0.0:WORKER_PORT, TP=8,
# dist-init MASTER:20000). node_rank from SLURM_PROCID (one task per node).
srun --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
     --container-image="$SGLANG_IMG" \
     --container-mounts=/lustre/fsw/portfolios/nemotron/users/alaptev:/alaptev,/lustre/fsw/portfolios/nemotron/users/igitman/hf_models:/hf_models \
     --container-env=SLURM_MASTER_NODE,SLURM_PROCID \
     bash -c "
       export PYTHONPATH=/alaptev/NeMo-Skills-clean:\$PYTHONPATH
       cd /alaptev/NeMo-Skills-clean
       python3 -m nemo_skills.inference.server.serve_sglang \
         --model ${MODEL} \
         --num_gpus 4 --num_nodes 2 --port ${WORKER_PORT} \
         --dist_init_addr \$SLURM_MASTER_NODE --node_rank \$SLURM_PROCID \
         --dtype bfloat16 --expert-parallel-size 8 --kv-cache-dtype fp8_e4m3 \
         --context-length 262144 --mem-fraction-static 0.85 --chunked-prefill-size 16384 \
         --mamba-scheduler-strategy no_buffer --disable-piecewise-cuda-graph --disable-radix-cache \
         --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 \
         --speculative-num-draft-tokens 5 --reasoning-parser nemotron_3 --tool-call-parser qwen3_coder
     "
