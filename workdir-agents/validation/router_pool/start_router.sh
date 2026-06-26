#!/bin/bash
# L1: standalone sglang_router on a CPU node (cpu partition, long walltime).
# Boots with ZERO workers; GPU server jobs register dynamically via /add_worker.
# Publishes its URL to a shared lustre file other jobs read.
#SBATCH -p cpu
#SBATCH -A nemotron_reason_science
#SBATCH -t 12:00:00
#SBATCH -J lazyalloc_router
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail

POOL_DIR="${POOL_DIR:-/alaptev/data/lazypool}"          # container-mount path
POOL_DIR_HOST="${POOL_DIR_HOST:-/lustre/fsw/portfolios/nemotron/users/alaptev/data/lazypool}"
ROUTER_PORT="${ROUTER_PORT:-20000}"
SGLANG_IMG="${SGLANG_IMG:-/lustre/fsw/portfolios/nemotron/users/alaptev/containers/sglang-v0.5.11.sqsh}"
POLICY="${POLICY:-round_robin}"   # cache_aware also fine; rr is simplest for parity

mkdir -p "$POOL_DIR_HOST"
HOST=$(hostname -f)
echo "[router] node=$HOST port=$ROUTER_PORT pool_dir=$POOL_DIR_HOST"

# Publish the router URL atomically so worker jobs + the driver can find it.
echo "http://${HOST}:${ROUTER_PORT}" > "${POOL_DIR_HOST}/router_url.tmp"
mv "${POOL_DIR_HOST}/router_url.tmp" "${POOL_DIR_HOST}/router_url"
echo "[router] published router_url -> ${POOL_DIR_HOST}/router_url"

# Launch the router with no workers. --worker-startup-check-interval keeps it
# polling so freshly-added workers are health-gated before traffic.
srun --container-image="$SGLANG_IMG" \
     --container-mounts=/lustre/fsw/portfolios/nemotron/users/alaptev:/alaptev \
     bash -c "
       python3 -m sglang_router.launch_router \
         --host 0.0.0.0 --port ${ROUTER_PORT} \
         --policy ${POLICY} \
         --worker-startup-timeout-secs 1200 \
         --worker-startup-check-interval 10
     "
