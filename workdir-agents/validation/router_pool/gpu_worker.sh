#!/bin/bash
# L1: ONE independent 1-node sglang server (1 replica) that registers into the
# standalone router. Submitted N times (chunked) -> each backfills separately,
# landing far sooner than a monolithic 16-node reservation. Self-deregisters on exit.
#SBATCH -p batch
#SBATCH -A nemotron_reason_science
#SBATCH -t 04:00:00
#SBATCH -J lazyalloc_worker
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
set -uo pipefail

POOL_DIR_HOST="${POOL_DIR_HOST:-/lustre/fsw/portfolios/nemotron/users/alaptev/data/lazypool}"
WORKER_PORT="${WORKER_PORT:-30001}"
MODEL="${MODEL:-/hf_models/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4}"
SGLANG_IMG="${SGLANG_IMG:-/lustre/fsw/portfolios/nemotron/users/alaptev/containers/sglang-v0.5.11.sqsh}"

# Discover the router URL (published by start_router.sh).
for i in $(seq 1 120); do
  [ -f "${POOL_DIR_HOST}/router_url" ] && break; sleep 5
done
ROUTER_URL=$(cat "${POOL_DIR_HOST}/router_url" 2>/dev/null || true)
if [ -z "$ROUTER_URL" ]; then echo "[worker] no router_url; abort"; exit 1; fi
HOST=$(hostname -f)
SELF_URL="http://${HOST}:${WORKER_PORT}"
echo "[worker] node=$HOST self=$SELF_URL router=$ROUTER_URL"

# sgl-model-gateway REST API: POST /workers {url} (202, async health-gated) returns a
# worker_id; DELETE /workers/<worker_id> removes it. We stash the id under POOL_DIR.
WID_FILE="${POOL_DIR_HOST}/worker_${WORKER_PORT}.id"

# Deregister on ANY exit (preemption, walltime, crash) so the router stops routing here.
cleanup() {
  WID=$(cat "$WID_FILE" 2>/dev/null || true)
  echo "[worker] deregistering $SELF_URL (id=$WID)"
  [ -n "$WID" ] && curl -s -X DELETE "${ROUTER_URL}/workers/${WID}" || true
  rm -f "$WID_FILE" 2>/dev/null || true
}
trap cleanup EXIT

# Background registrar: wait until THIS server answers /health, then POST /workers.
(
  for i in $(seq 1 240); do
    if curl -sf "${SELF_URL}/health" >/dev/null 2>&1; then
      echo "[worker] healthy; POST /workers -> router"
      RESP=$(curl -s -X POST -H 'Content-Type: application/json' \
                  -d "{\"url\":\"${SELF_URL}\"}" "${ROUTER_URL}/workers")
      echo "[worker] register resp: $RESP"
      echo "$RESP" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("worker_id",""))' \
                   > "$WID_FILE" 2>/dev/null || true
      echo "[worker] registered id=$(cat "$WID_FILE" 2>/dev/null)"
      exit 0
    fi
    sleep 10
  done
  echo "[worker] never became healthy; not registering"
) &

# Worker server_args MIRROR the current worker pool (EAGLE, fp4, 256k, mrr=8).
srun --container-image="$SGLANG_IMG" \
     --container-mounts=/lustre/fsw/portfolios/nemotron/users/alaptev:/alaptev,/lustre/fsw/portfolios/nemotron/users/igitman/hf_models:/hf_models \
     bash -c "
       python3 -m sglang.launch_server \
         --model-path ${MODEL} \
         --host 0.0.0.0 --port ${WORKER_PORT} \
         --tp-size 4 \
         --expert-parallel-size 4 \
         --quantization modelopt_fp4 \
         --kv-cache-dtype fp8_e4m3 \
         --context-length 262144 \
         --max-running-requests 8 \
         --mem-fraction-static 0.82 \
         --cuda-graph-max-bs 16 \
         --mamba-scheduler-strategy no_buffer \
         --disable-piecewise-cuda-graph \
         --disable-radix-cache \
         --speculative-algorithm EAGLE \
         --speculative-num-steps 5 \
         --speculative-eagle-topk 1 \
         --speculative-num-draft-tokens 5 \
         --reasoning-parser nemotron_3 \
         --tool-call-parser qwen3_coder
     "
