#!/bin/bash
# Top the BF16 worker pool up to TARGET live (RUNNING+PENDING) 2-node workers.
# Run on the cluster login node (pure sbatch; no conda needed). Workers self-
# deregister on exit (4h batch cap) and re-register on relaunch; the router
# (cpu-long) persists so the pool heals seamlessly. Usage: topup_workers.sh [TARGET]
set -uo pipefail
TARGET="${1:-16}"
A=/lustre/fsw/portfolios/nemotron/users/alaptev; RP=$A/router_pool
ALIVE=$(squeue --me -h -o "%j %T" | grep -c '^lazyalloc_bf16worker_' || true)
NEED=$(( TARGET - ALIVE ))
echo "[topup] alive=$ALIVE target=$TARGET need=$NEED"
[ "$NEED" -le 0 ] && { echo "[topup] pool full"; exit 0; }
# find a free index suffix (max existing +1)
BASE=$(squeue --me -h -o "%j" | grep '^lazyalloc_bf16worker_w' | sed 's/.*_w//' | sort -n | tail -1)
START=$(( ${BASE:--1} + 1 ))
for i in $(seq 0 $((NEED-1))); do
  IDX=$(( START + i ))
  JID=$(POOL_DIR_HOST=$A/data/lazypool_bf16 WORKER_PORT=5656 sbatch --parsable \
        -J "lazyalloc_bf16worker_w${IDX}" \
        --export=ALL,POOL_DIR_HOST=$A/data/lazypool_bf16,WORKER_PORT=5656 \
        $RP/bf16_worker.sh)
  echo "[topup] submitted w${IDX} job=$JID"
  sleep 1
done
