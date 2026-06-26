#!/bin/bash
# 5-seed run: co-locate the shared SANDBOX + shared ROUTER in ONE CPU job (saves a
# CPU-job slot so more seeds get long walltime — CPU QOS ceiling is ~7 jobs:
# cpu-long node=1, cpu-normal node=2, cpu-short MaxSubmit=4). Both are light HTTP
# services on distinct ports of the same node; both publish to the pool dir the
# drivers' sidecars (sandbox) and the launcher prehosted_url (router) read.
# Submit on cpu-long for a long walltime (outlives all seeds):
#   sbatch --qos=cpu-long --time=24:00:00 -J lazyalloc_services_5seed shared_services.sh
#SBATCH -p cpu
#SBATCH -A nemotron_reason_science
#SBATCH -t 24:00:00
#SBATCH -J lazyalloc_services
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
set -uo pipefail

A=/lustre/fsw/portfolios/nemotron/users/alaptev
SANDBOX_IMG="${SANDBOX_IMG:-/lustre/fsw/portfolios/nemotron/users/igitman/images/nemo-skills-sandbox-dc43f3e.sqsh}"
SGLANG_IMG="${SGLANG_IMG:-$A/containers/sglang-v0.5.11.sqsh}"
POOL_DIR_HOST="${POOL_DIR_HOST:-$A/data/lazypool_5seed}"
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
     bash -lc '/start-with-nginx.sh' >"$POOL_DIR_HOST/shared_sandbox.log" 2>&1 &
SBX=$!

# --- router (background) ---
srun --overlap --ntasks=1 --gpus-per-node=0 \
     --container-image="$SGLANG_IMG" \
     --container-mounts="$A:/alaptev" \
     bash -c "python3 -m sglang_router.launch_router --host 0.0.0.0 --port ${ROUTER_PORT} --policy ${POLICY} --worker-startup-timeout-secs 1200 --worker-startup-check-interval 10" \
     >"$POOL_DIR_HOST/shared_router.log" 2>&1 &
RTR=$!

# --- publish addresses once each is healthy ---
# router: publishes as soon as it binds (it boots with 0 workers, health-gates them)
for i in $(seq 1 60); do
  if curl -sf -o /dev/null "http://localhost:${ROUTER_PORT}/health" 2>/dev/null || curl -sf -o /dev/null "http://localhost:${ROUTER_PORT}/workers" 2>/dev/null; then
    echo "http://${HOST}:${ROUTER_PORT}" > "${POOL_DIR_HOST}/router_url.tmp" && mv "${POOL_DIR_HOST}/router_url.tmp" "${POOL_DIR_HOST}/router_url"
    echo "[services] router up -> ${POOL_DIR_HOST}/router_url"; break
  fi
  kill -0 $RTR 2>/dev/null || { echo "[services] router died"; tail -20 "$POOL_DIR_HOST/shared_router.log"; exit 1; }
  sleep 2
done
# sandbox: health then publish
for i in $(seq 1 180); do
  if curl -sf -o /dev/null "http://localhost:${SANDBOX_PORT}/health" 2>/dev/null; then
    echo "${HOST}:${SANDBOX_PORT}" > "${POOL_DIR_HOST}/sandbox_addr.tmp" && mv "${POOL_DIR_HOST}/sandbox_addr.tmp" "${POOL_DIR_HOST}/sandbox_addr"
    echo "[services] sandbox up -> ${POOL_DIR_HOST}/sandbox_addr"; break
  fi
  kill -0 $SBX 2>/dev/null || { echo "[services] sandbox died"; tail -20 "$POOL_DIR_HOST/shared_sandbox.log"; exit 1; }
  sleep 1
done
[ -f "${POOL_DIR_HOST}/router_url" ] && [ -f "${POOL_DIR_HOST}/sandbox_addr" ] || { echo "[services] one service never came up"; exit 1; }
echo "[services] BOTH up: router=$(cat ${POOL_DIR_HOST}/router_url) sandbox=$(cat ${POOL_DIR_HOST}/sandbox_addr)"

# hold both for the life of the job
wait
