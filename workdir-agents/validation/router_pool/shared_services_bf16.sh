#!/bin/bash
# BF16 run: co-locate shared SANDBOX + shared ROUTER in ONE cpu-long job (7-day,
# footprint=1 CPU node, stable URLs for the whole multi-day run, leaves cpu-normal
# free for the co-tenant). Robust variant: STAGGER the two srun steps (sandbox first,
# settle, then router) to avoid the simultaneous dual-step-creation race that 0-byte'd
# the router last time; cold-container-generous health waits.
#   sbatch --qos=cpu-long --time=72:00:00 -J lazyalloc_bf16_services shared_services_bf16.sh
#SBATCH -p cpu
#SBATCH -A nemotron_reason_science
#SBATCH -t 72:00:00
#SBATCH -J lazyalloc_bf16_services
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
set -uo pipefail

A=/lustre/fsw/portfolios/nemotron/users/alaptev
SANDBOX_IMG="${SANDBOX_IMG:-/lustre/fsw/portfolios/nemotron/users/igitman/images/nemo-skills-sandbox-dc43f3e.sqsh}"
SGLANG_IMG="${SGLANG_IMG:-$A/containers/sglang-v0.5.11.sqsh}"
POOL_DIR_HOST="${POOL_DIR_HOST:-$A/data/lazypool_bf16}"
SANDBOX_PORT="${SANDBOX_PORT:-6000}"
ROUTER_PORT="${ROUTER_PORT:-20000}"
UWSGI_PROCESSES="${UWSGI_PROCESSES:-96}"
SESSION_TIMEOUT="${NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT:-600}"
POLICY="${POLICY:-round_robin}"

mkdir -p "$POOL_DIR_HOST" "$POOL_DIR_HOST/sandbox_run"
HOST=$(hostname -f)
echo "[services] node=$HOST sandbox=:$SANDBOX_PORT router=:$ROUTER_PORT uwsgi=$UWSGI_PROCESSES session_timeout=${SESSION_TIMEOUT}s"

# --- sandbox (background) ---
export UWSGI_PROCESSES NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT="$SESSION_TIMEOUT"
srun --overlap --ntasks=1 --gpus-per-node=0 \
     --container-env=UWSGI_PROCESSES,NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT \
     --container-image="$SANDBOX_IMG" \
     --container-mounts="$A:/alaptev,$POOL_DIR_HOST/sandbox_run:/nemo_run" \
     bash -lc "/start-with-nginx.sh" >"$POOL_DIR_HOST/shared_sandbox.log" 2>&1 &
SBX=$!
echo "[services] sandbox srun launched (pid $SBX); settling 30s before router step"
sleep 30
kill -0 $SBX 2>/dev/null || { echo "[services] sandbox srun died during settle"; tail -30 "$POOL_DIR_HOST/shared_sandbox.log"; }

# --- router (background, staggered after sandbox step is established) ---
srun --overlap --ntasks=1 --gpus-per-node=0 \
     --container-image="$SGLANG_IMG" \
     --container-mounts="$A:/alaptev" \
     bash -c "python3 -m sglang_router.launch_router --host 0.0.0.0 --port ${ROUTER_PORT} --policy ${POLICY} --worker-startup-timeout-secs 1200 --worker-startup-check-interval 10" \
     >"$POOL_DIR_HOST/shared_router.log" 2>&1 &
RTR=$!
echo "[services] router srun launched (pid $RTR)"

# --- router up? (cold sglang import can take minutes) ---
for i in $(seq 1 600); do
  if curl -sf -o /dev/null "http://localhost:${ROUTER_PORT}/health" 2>/dev/null || curl -sf -o /dev/null "http://localhost:${ROUTER_PORT}/workers" 2>/dev/null; then
    echo "http://${HOST}:${ROUTER_PORT}" > "${POOL_DIR_HOST}/router_url.tmp" && mv "${POOL_DIR_HOST}/router_url.tmp" "${POOL_DIR_HOST}/router_url"
    echo "[services] router up after ${i}s -> $(cat ${POOL_DIR_HOST}/router_url)"; break
  fi
  kill -0 $RTR 2>/dev/null || { echo "[services] router srun died"; tail -40 "$POOL_DIR_HOST/shared_router.log"; exit 1; }
  sleep 1
done
[ -f "${POOL_DIR_HOST}/router_url" ] || { echo "[services] router never came up"; tail -40 "$POOL_DIR_HOST/shared_router.log"; exit 1; }

# --- sandbox up? ---
for i in $(seq 1 600); do
  if curl -sf -o /dev/null "http://localhost:${SANDBOX_PORT}/health" 2>/dev/null; then
    echo "${HOST}:${SANDBOX_PORT}" > "${POOL_DIR_HOST}/sandbox_addr.tmp" && mv "${POOL_DIR_HOST}/sandbox_addr.tmp" "${POOL_DIR_HOST}/sandbox_addr"
    echo "[services] sandbox up after ${i}s -> $(cat ${POOL_DIR_HOST}/sandbox_addr)"; break
  fi
  kill -0 $SBX 2>/dev/null || { echo "[services] sandbox srun died"; tail -40 "$POOL_DIR_HOST/shared_sandbox.log"; exit 1; }
  sleep 1
done
[ -f "${POOL_DIR_HOST}/sandbox_addr" ] || { echo "[services] sandbox never came up"; tail -40 "$POOL_DIR_HOST/shared_sandbox.log"; exit 1; }

echo "[services] BOTH up: router=$(cat ${POOL_DIR_HOST}/router_url) sandbox=$(cat ${POOL_DIR_HOST}/sandbox_addr)"
wait
