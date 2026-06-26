#!/bin/bash
# CONSOLIDATED SINGLE-CPU-NODE LAUNCHER (SCI-548 lazy allocation, 5-seed full-HLE).
# Runs EVERYTHING for an N-seed full-2158 run in ONE cpu-long sbatch on ONE node:
# shared sandbox + shared sglang router + N gym drivers, all via `srun --overlap`.
# WHY one job: bypasses the 7-CPU-job QOS ceiling AND gives ALL seeds 7-day walltime
# (cpu-long), vs separate-jobs capping 3 seeds at cpu-short 4h.
#
# Parameterized by SEEDS (default "0 1 2 3 4"); use SEEDS="0 1" for the 2-driver
# de-risk (measure combined RSS + FD/ephemeral-ports under load before the full 5x).
# Reuses the VALIDATED generated driver/bootstrap scripts (hand-templated per seed
# with OFFSET sidecar ports 9101+10k) + the proven shared_services sandbox/router.
#   sbatch --qos=cpu-long -J lazyalloc_5n cons5n_launch.sh        # full 5-seed
#   SEEDS="0 1" sbatch --qos=cpu-long -J lazyalloc_2n cons5n_launch.sh   # de-risk
#SBATCH -p cpu
#SBATCH -A nemotron_reason_science
#SBATCH -t 7-00:00:00
#SBATCH -J lazyalloc_5n
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=0
#SBATCH --open-mode=append
set -uo pipefail

SEEDS="${SEEDS:-0 1 2 3 4}"
A=/lustre/fsw/portfolios/nemotron/users/alaptev
POOL="$A/data/lazypool_5seed"
NEMORUN="$A/nemo-run/lazyalloc_5n"
SCRIPTS="$NEMORUN/scripts"                            # host path (unused at runtime)
SCRIPTS_C="/alaptev/nemo-run/lazyalloc_5n/scripts"   # IN-CONTAINER path (only /alaptev is mounted, not /lustre/...)
CODE_MOUNT="$A/nemo-run/lazyalloc_mA/lazyalloc_mA_1782351074/nemo-run"   # reuse rsynced /nemo_run code
GYM_IMG="/lustre/fsw/portfolios/nemotron/users/igitman/images/nemo-skills-dc43f3e.sqsh"
SBX_IMG="/lustre/fsw/portfolios/nemotron/users/igitman/images/nemo-skills-sandbox-dc43f3e.sqsh"
SGLANG_IMG="$A/containers/sglang-v0.5.11.sqsh"
SANDBOX_PORT="${SANDBOX_PORT:-6000}"
ROUTER_PORT="${ROUTER_PORT:-20000}"
POLICY="${POLICY:-round_robin}"
UWSGI_PROCESSES="${UWSGI_PROCESSES:-48}"           # cap shared-sandbox memory on the shared node
SESSION_TIMEOUT="${NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT:-600}"

MOUNTS="/lustre/fsw/portfolios/nemotron/users/igitman/hf_models:/hf_models,/lustre/fsw/portfolios/nemotron/users/igitman/images/swe-bench:/swe-bench-images,$A:/alaptev,$A/data:/data,$CODE_MOUNT:/nemo_run"

# --- gym driver env (mirrors the validated NeMo-Run sbatch) ---
export HF_HOME=/alaptev/hf-cache TRANSFORMERS_CACHE=/alaptev/hf-cache HUGGINGFACE_HUB_CACHE=/alaptev/hf-cache
export TIKTOKEN_ENCODINGS_BASE=/alaptev/tiktoken-encodings TIKTOKEN_RS_CACHE_DIR=/alaptev/tiktoken-encodings/cache
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=true SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS=16 SGLANG_DG_CACHE_DIR=/alaptev/deep_gemm_cache
export HF_HUB_OFFLINE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 HERMES_AGENT_PATH=/alaptev/hermes-agent
# N drivers share the gym server .venvs read-only (skip_venv_if_present), but Python
# still writes .pyc into the shared site-packages on first import — concurrent drivers
# race on that (partially-initialized-module / missing-stdlib import errors; found in
# de-risk: s1 won, s0 crashed importing markdown_it/ray). Disabling bytecode writes
# removes the shared-write entirely (existing .pyc are still read).
export PYTHONDONTWRITEBYTECODE=1
# Isolated gym checkout whose venvs are re-homed to a STABLE lustre Python
# (/alaptev/uv_python) instead of the volatile container /root/.local (which
# corrupts under churn + clashes across co-located containers). Pin UV_PYTHON_INSTALL_DIR
# so any incidental uv call resolves the lustre Python, never re-downloading to /root/.local.
export UV_PYTHON_INSTALL_DIR=/alaptev/uv_python
export SLURM_MASTER_NODE="$(hostname)"; export SLURM_GROUP_NODES="$(hostname)"
GENV="HF_HOME,TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE,TIKTOKEN_ENCODINGS_BASE,TIKTOKEN_RS_CACHE_DIR,SGLANG_JIT_DEEPGEMM_FAST_WARMUP,SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS,SGLANG_DG_CACHE_DIR,HF_HUB_OFFLINE,VLLM_ALLOW_LONG_MAX_MODEL_LEN,HERMES_AGENT_PATH,PYTHONDONTWRITEBYTECODE,UV_PYTHON_INSTALL_DIR,SLURM_MASTER_NODE,SLURM_GROUP_NODES"

HOST="$(hostname -f)"
mkdir -p "$POOL" "$POOL/sandbox_run"
rm -f "$POOL/router_url" "$POOL/sandbox_addr"     # clear stale addrs from prior runs
echo "[cons5n] node=$HOST seeds='$SEEDS' uwsgi=$UWSGI_PROCESSES session_timeout=${SESSION_TIMEOUT}s"
ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
echo "[cons5n] login-shell open-file limit: $(ulimit -Sn)/$(ulimit -Hn)"

SR="srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=0 --kill-on-bad-exit=0"

# 1) verify the pre-built lustre-Python venv (do NOT run `uv venv`/`uv sync` — the
# isolated NeMo-Gym-5seed venvs are already built + re-homed to the lustre Python;
# re-running uv could re-point them back at the volatile /root/.local).
echo "[cons5n] === verifying pre-built lustre-Python venv ==="
$SR --container-image="$GYM_IMG" --container-mounts="$MOUNTS" --container-workdir=/nemo_run/code --container-env=$GENV \
    bash -lc 'cd /alaptev/NeMo-Gym-5seed && source .venv/bin/activate && python -c "import sys,nemo_gym; print(\"venv OK\", sys.version.split()[0], sys.executable, nemo_gym.__file__)"' \
    > "$POOL/cons5n_presync.log" 2>&1
echo "[cons5n] venv verify rc=$? -> $(tail -1 "$POOL/cons5n_presync.log")"

# 1b) per-seed uv caches: ng_run invokes `uv` to launch its servers and writes the
# shared cache; N drivers starting near-simultaneously on ONE node race on it
# (No such file or directory on tmp files) and crash all-but-one ng_run. Give each
# seed a PRIVATE warm cache (hardlink copy = fast, ~no extra space) + UV_OFFLINE=1
# in the driver scripts so uv never refreshes the index. (Non-issue on separate
# nodes; single-node specific — found in the 2-driver de-risk.)
echo "[cons5n] === preparing per-seed uv caches ==="
for k in $SEEDS; do
  CDIR="$A/NeMo-Gym-5seed/cache/uv_s${k}"
  # gate on a completion marker, NOT mere dir existence — a partial copy left a dir
  # that the old check skipped, yielding a broken cache (seen in de-risk).
  if [ ! -f "$CDIR/.cache_complete" ]; then
    rm -rf "$CDIR"
    cp -al "$A/NeMo-Gym-5seed/cache/uv" "$CDIR" 2>/dev/null || cp -a "$A/NeMo-Gym-5seed/cache/uv" "$CDIR"
    touch "$CDIR/.cache_complete"
  fi
  echo "[cons5n]   uv_s${k}: $(du -sh "$CDIR" 2>/dev/null | cut -f1)"
done

# 2) bootstrap barrier: materialize each seed's hermes_home (offset mcp ports baked in)
echo "[cons5n] === bootstrapping hermes_home (barrier) ==="
bpids=()
for k in $SEEDS; do
  $SR --container-image="$GYM_IMG" --container-mounts="$MOUNTS" --container-workdir=/nemo_run/code --container-env=$GENV \
      bash "$SCRIPTS_C/nemo-run-0-s${k}.sh" > "$POOL/cons5n_bootstrap_s${k}.log" 2>&1 &
  bpids+=($!)
done
brc=0; for p in "${bpids[@]}"; do wait "$p" || brc=1; done
if [ "$brc" -ne 0 ]; then echo "[cons5n] FATAL: a bootstrap step failed; aborting"; for k in $SEEDS; do echo "--- bootstrap s$k ---"; tail -5 "$POOL/cons5n_bootstrap_s${k}.log"; done; exit 1; fi
echo "[cons5n] bootstrap barrier done (all hermes_home materialized)"

# 3) shared sandbox (background)
export UWSGI_PROCESSES NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT="$SESSION_TIMEOUT"
$SR --container-env=UWSGI_PROCESSES,NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT \
    --container-image="$SBX_IMG" --container-mounts="$A:/alaptev,$POOL/sandbox_run:/nemo_run" \
    bash -lc '/start-with-nginx.sh' > "$POOL/shared_sandbox.log" 2>&1 &
SBX=$!

# 4) shared sglang router (background)
$SR --container-image="$SGLANG_IMG" --container-mounts="$A:/alaptev" \
    bash -c "python3 -m sglang_router.launch_router --host 0.0.0.0 --port ${ROUTER_PORT} --policy ${POLICY} --worker-startup-timeout-secs 1200 --worker-startup-check-interval 10" \
    > "$POOL/shared_router.log" 2>&1 &
RTR=$!

# 5) publish router_url + sandbox_addr once healthy (drivers + GPU workers read these).
# Generous timeouts: a COLD node must enroot-import the 29 GB sglang / 14 GB sandbox
# sqsh before the server even starts (minutes); 180s was too short on a cold node and
# the router "failed to publish" while still importing (seen in de-risk on cpu-0007).
for i in $(seq 1 360); do   # ~12 min: cold sglang-container import + router bind
  if curl -sf -o /dev/null "http://localhost:${ROUTER_PORT}/workers" 2>/dev/null; then
    echo "http://${HOST}:${ROUTER_PORT}" > "$POOL/router_url.tmp" && mv "$POOL/router_url.tmp" "$POOL/router_url"
    echo "[cons5n] router up -> $POOL/router_url (after $((i*2))s)"; break
  fi
  kill -0 $RTR 2>/dev/null || { echo "[cons5n] router died"; tail -30 "$POOL/shared_router.log"; exit 1; }
  sleep 2
done
for i in $(seq 1 600); do
  if curl -sf -o /dev/null "http://localhost:${SANDBOX_PORT}/health" 2>/dev/null; then
    echo "${HOST}:${SANDBOX_PORT}" > "$POOL/sandbox_addr.tmp" && mv "$POOL/sandbox_addr.tmp" "$POOL/sandbox_addr"
    echo "[cons5n] sandbox up -> $POOL/sandbox_addr"; break
  fi
  kill -0 $SBX 2>/dev/null || { echo "[cons5n] sandbox died"; tail -30 "$POOL/shared_sandbox.log"; exit 1; }
  sleep 1
done
[ -f "$POOL/router_url" ] && [ -f "$POOL/sandbox_addr" ] || { echo "[cons5n] services failed to publish"; exit 1; }
echo "[cons5n] services up: router=$(cat $POOL/router_url) sandbox=$(cat $POOL/sandbox_addr)"

# 6) launch N gym drivers (background). Each blocks on the router /models gate until a
#    GPU worker registers, then starts its offset-port sidecars + ng_run + ng_collect.
pids=()
for k in $SEEDS; do
  $SR --container-image="$GYM_IMG" --container-mounts="$MOUNTS" --container-workdir=/nemo_run/code --container-env=$GENV \
      bash "$SCRIPTS_C/nemo-run-1-s${k}.sh" > "$POOL/cons5n_driver_s${k}.log" 2>&1 &
  last=$!
  pids+=($last)
  echo "[cons5n] launched driver s${k} pid=$last; staggering until its servers are ready"
  # STAGGER: hold until THIS driver's ng_run finishes its import-heavy server startup
  # before launching the next. N drivers importing the SAME shared lustre venv at the
  # same instant trigger transient ModuleNotFound / 'partially initialized module'
  # (a lustre concurrent-import pathology — the files exist + are NOT mutated; whichever
  # driver loses the burst crashes). Serializing the startup burst avoids the overlap;
  # steady-state children fork-inherit imports so they don't re-contend.
  for w in $(seq 1 240); do
    grep -q 'All servers ready' "$POOL/cons5n_driver_s${k}.log" 2>/dev/null && { echo "[cons5n] driver s${k} servers ready (after $((w*5))s)"; break; }
    grep -q 'ng_run process exited' "$POOL/cons5n_driver_s${k}.log" 2>/dev/null && { echo "[cons5n] WARN driver s${k} ng_run exited during startup"; break; }
    kill -0 "$last" 2>/dev/null || { echo "[cons5n] WARN driver s${k} process gone during startup"; break; }
    sleep 5
  done
done

# 7) wait for all drivers; tear services down when the last finishes
echo "[cons5n] waiting on ${#pids[@]} drivers (pids: ${pids[*]})"
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[cons5n] all drivers exited (rc=$rc); stopping shared services"
kill $SBX $RTR 2>/dev/null || true
exit $rc
