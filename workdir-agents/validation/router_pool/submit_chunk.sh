#!/bin/bash
# Submit N independent 1-node GPU worker jobs. Each backfills separately and
# registers into the running router. Call repeatedly to grow the pool.
#   ./submit_chunk.sh 4          # add 4 replicas
#   ./submit_chunk.sh 4 30010    # base worker port 30010 (avoid collisions across waves)
#   ./submit_chunk.sh 16 30001 s2  # RUN_LABEL=s2 -> names lazyalloc_worker_s2_pNNNNN
# For N-seed parallel runs, ALWAYS pass a distinct RUN_LABEL per seed so worker job
# names are unique across seeds (ports repeat across seeds on different nodes).
set -euo pipefail
N="${1:?usage: submit_chunk.sh N [base_port] [run_label]}"
BASE_PORT="${2:-30001}"
RUN_LABEL="${3:-}"
LBL=""; [ -n "$RUN_LABEL" ] && LBL="_${RUN_LABEL}"
HERE="$(cd "$(dirname "$0")" && pwd)"
for i in $(seq 0 $((N-1))); do
  PORT=$((BASE_PORT + i))
  JID=$(WORKER_PORT=$PORT sbatch --parsable \
        -J "lazyalloc_worker${LBL}_p${PORT}" \
        --export=ALL,WORKER_PORT=$PORT \
        "${HERE}/gpu_worker.sh")
  echo "submitted replica $i  port=$PORT  job=$JID"
done
