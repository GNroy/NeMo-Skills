#!/bin/bash
# Stage 2: a STANDALONE, long-lived code-execution sandbox shared by all seeds on
# (eventually) a node. One sandbox's ~110 GB uwsgi preload (96 workers each loading
# scipy/sympy) is the dominant driver-memory cost; sharing it across N seeds pays it
# ONCE instead of N times (per-session cost is only ~20 MB — see
# sandbox_session_measure.log). Publishes its host:port to a pool file the per-seed
# drivers' python sidecars read (gym_sidecars.sh NS_SIDECAR_SANDBOX_*). Modeled on
# start_router.sh (CPU service that publishes a URL).
#
# UWSGI_PROCESSES is the memory lever: fewer workers => less preload. The
# tiangolo/uwsgi-nginx-flask entrypoint regenerates uwsgi.ini from these env vars,
# so they override the image's build-time defaults at runtime (scoped to THIS job,
# not the shared cluster sandbox config).
#SBATCH -p cpu
#SBATCH -A nemotron_reason_science
#SBATCH -t 12:00:00
#SBATCH -J lazyalloc_shared_sandbox
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
set -uo pipefail

A=/lustre/fsw/portfolios/nemotron/users/alaptev
SANDBOX_IMG="${SANDBOX_IMG:-/lustre/fsw/portfolios/nemotron/users/igitman/images/nemo-skills-sandbox-dc43f3e.sqsh}"
POOL_DIR_HOST="${POOL_DIR_HOST:-$A/data/lazypool_stage2}"
PORT="${SANDBOX_PORT:-6000}"
UWSGI_PROCESSES="${UWSGI_PROCESSES:-96}"   # memory lever; 96 = image default
UWSGI_CHEAPER="${UWSGI_CHEAPER:-}"          # optional lazy-scale floor (e.g. 8)
# Idle-session reaping: local_sandbox_server.py reaps Jupyter kernels idle past this
# many seconds (cleanup_expired_sessions, triggered each execute). DEFAULT in the
# image is 0 = DISABLED -> kernels (~73 MB each) accumulate for the whole run (the
# growth seen in stage1/s0). For a SHARED pool serving N seeds we MUST bound the live
# set to the working set, so default it ON here. 0 to disable.
SESSION_TIMEOUT="${NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT:-600}"

mkdir -p "$POOL_DIR_HOST" "$A/data/lazypool_stage2/sandbox_run"
HOST=$(hostname -f)
echo "[sandbox] node=$HOST port=$PORT UWSGI_PROCESSES=$UWSGI_PROCESSES cheaper=${UWSGI_CHEAPER:-none} session_timeout=${SESSION_TIMEOUT}s"

# Publish address atomically once the server is healthy (below). Pre-write the
# intended address; consumers also health-check before use.
ADDR="${HOST}:${PORT}"

# Launch the sandbox; --container-env overrides the worker count at runtime.
ENVFLAGS="--container-env=UWSGI_PROCESSES,NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT"
export UWSGI_PROCESSES
export NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT="$SESSION_TIMEOUT"
if [ -n "$UWSGI_CHEAPER" ]; then export UWSGI_CHEAPER; ENVFLAGS="$ENVFLAGS,UWSGI_CHEAPER"; fi

srun --overlap --ntasks=1 --gpus-per-node=0 $ENVFLAGS \
     --container-image="$SANDBOX_IMG" \
     --container-mounts="$A:/alaptev,$A/data/lazypool_stage2/sandbox_run:/nemo_run" \
     bash -lc '/start-with-nginx.sh' >"$POOL_DIR_HOST/shared_sandbox.log" 2>&1 &
SRV=$!

# Health-gate, then publish.
for i in $(seq 1 180); do
  if curl -sf -o /dev/null "http://localhost:${PORT}/health" 2>/dev/null; then
    echo "$ADDR" > "${POOL_DIR_HOST}/sandbox_addr.tmp"
    mv "${POOL_DIR_HOST}/sandbox_addr.tmp" "${POOL_DIR_HOST}/sandbox_addr"
    echo "[sandbox] healthy after ${i}s; published ${POOL_DIR_HOST}/sandbox_addr = $ADDR"
    break
  fi
  kill -0 $SRV 2>/dev/null || { echo "[sandbox] server died early"; tail -20 "$POOL_DIR_HOST/shared_sandbox.log"; exit 1; }
  sleep 1
done
[ -f "${POOL_DIR_HOST}/sandbox_addr" ] || { echo "[sandbox] never became healthy"; exit 1; }

# Hold for the life of the job.
wait $SRV
