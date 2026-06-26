#!/bin/bash
# Stage 2 MEASURE-FIRST gate: per-Jupyter-session memory cost in the sandbox.
# A shared per-node sandbox would hold up to N_seeds*width sessions (e.g. 5*128=640).
# This quantifies the per-session RSS so we know whether that fits 370 GB, and
# whether DELETE /sessions/<id> reclaims it (session lifecycle for the shared pool).
# CPU-only, no GPU. Starts the real sandbox container, drives sessions, measures.
#SBATCH -p cpu
#SBATCH -A nemotron_reason_science
#SBATCH -t 00:25:00
#SBATCH -J lazyalloc_sandbox_measure
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
set -uo pipefail

A=/lustre/fsw/portfolios/nemotron/users/alaptev
SANDBOX_IMG=/lustre/fsw/portfolios/nemotron/users/igitman/images/nemo-skills-sandbox-dc43f3e.sqsh
PORT=6000
OUT=$A/grading/sandbox_session_measure.log
: > "$OUT"

log(){ echo "$@" | tee -a "$OUT"; }
memfree(){ free -m | awk '/Mem:/{print $3}'; }   # used MB

log "[measure] node=$(hostname) start=$(date +%H:%M:%S)"

# 1. start the sandbox container in the background (shares host netns -> localhost:6000)
mkdir -p "$A/grading/sandbox_run"
srun --overlap --ntasks=1 --gpus-per-node=0 \
     --container-image="$SANDBOX_IMG" \
     --container-mounts="$A:/alaptev,$A/grading/sandbox_run:/nemo_run" \
     bash -lc '/start-with-nginx.sh' >"$A/grading/sandbox_measure_server.log" 2>&1 &
SRV=$!

# 2. wait for health
for i in $(seq 1 120); do
  curl -sf -o /dev/null "http://localhost:${PORT}/health" 2>/dev/null && { log "[measure] sandbox healthy after ${i}s"; break; }
  kill -0 $SRV 2>/dev/null || { log "[measure] sandbox server died early"; tail -20 "$A/grading/sandbox_measure_server.log" | tee -a "$OUT"; exit 1; }
  sleep 1
done

exec_session(){  # $1 = session id
  curl -sf -X POST "http://localhost:${PORT}/execute" \
    -H 'Content-Type: application/json' -H "X-Session-ID: $1" \
    -d '{"generated_code":"import numpy as np, sympy, math\nx=np.zeros((200,200))\ndf=[i*i for i in range(2000)]\nprint(sum(df))","std_input":"","timeout":30,"language":"ipython","max_output_characters":200,"traceback_verbosity":"Plain"}' \
    >/dev/null 2>&1
}
del_session(){ curl -sf -X DELETE "http://localhost:${PORT}/sessions/$1" -H "X-Session-ID: $1" >/dev/null 2>&1; }
kcount(){ ps -eo cmd --no-headers 2>/dev/null | grep -icE 'ipykernel|ipython|jupyter|kernel'; }

BASE=$(memfree); log "[measure] baseline used=${BASE}MB kernels=$(kcount) uwsgi=$(ps -eo cmd --no-headers|grep -c [u]wsgi)"

N=0
for batch in 25 25 50 100 200; do
  for j in $(seq 1 $batch); do N=$((N+1)); exec_session "sess-$N"; done
  U=$(memfree); K=$(kcount)
  log "[measure] sessions=$N used=${U}MB delta_from_base=$((U-BASE))MB per_session=$(( (U-BASE) / (N>0?N:1) ))MB kernels=$K"
done

log "[measure] --- now DELETE all $N sessions, check reclaim ---"
for i in $(seq 1 $N); do del_session "sess-$i"; done
sleep 8
U=$(memfree); log "[measure] after_delete used=${U}MB delta_from_base=$((U-BASE))MB kernels=$(kcount)"

log "[measure] done=$(date +%H:%M:%S)"
kill $SRV 2>/dev/null || true
