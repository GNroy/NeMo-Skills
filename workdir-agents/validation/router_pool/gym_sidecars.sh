#!/bin/bash
# Lazy-allocation Stage 1: launch ONE set of shared HTTP MCP services (worklog +
# benchmark + python) on THIS node, instead of one stdio MCP server per delegate
# child. The gym driver script runs this in the background (gated by
# NS_GYM_SIDECAR_SCRIPT) and waits for the "<this>.ready" sentinel before starting
# the rollout. Children + the orchestrator reach the sidecars at 127.0.0.1:<port>
# (process-mode keeps every child on the driver node).
#
# Inherits the gym script's environment: the activated Gym venv (so `python` has
# mcp/starlette/uvicorn + nemo_skills on PYTHONPATH) and NEMO_SKILLS_SANDBOX_HOST/
# PORT (so the python sidecar reaches the in-job sandbox). Per-run paths are env
# overridable with sane defaults baked below.
set -u

NS_CLEAN="${NS_SIDECAR_PYTHONPATH:-/alaptev/NeMo-Skills-clean}"
export PYTHONPATH="${NS_CLEAN}:${PYTHONPATH:-}"

# Use the SAME interpreter the stdio MCP servers used in production
# (/usr/bin/python3 with PYTHONPATH=NeMo-Skills-clean) — known to have mcp,
# nemo_skills, the sandbox client, and FastMCP's uvicorn/starlette. The gym uv
# venv `python` is a different env and may lack them.
PYBIN="${NS_SIDECAR_PYTHON:-/usr/bin/python3}"

OUT_DIR="${NS_SIDECAR_OUT_DIR:-/alaptev/exp/a0py_lazy_stage1}"
RUN_ID="${NS_SIDECAR_RUN_ID:-nhwstage1}"
AGENT_ID="${NS_SIDECAR_AGENT_ID:-scientist}"
BENCH_PATH="${NS_SIDECAR_BENCH_PATH:-/alaptev/data/hle_full2158.ng.jsonl}"

WORKLOG_PORT="${NS_SIDECAR_WORKLOG_PORT:-9101}"
BENCH_PORT="${NS_SIDECAR_BENCH_PORT:-9102}"
PY_PORT="${NS_SIDECAR_PY_PORT:-9103}"
HOST=127.0.0.1

# Sentinel the gym hook waits on. Defaults to this script's path + .ready; a wrapper
# (e.g. gym_sidecars_stage2.sh) that execs this script overrides it to ITS own
# .ready so the hook (which keys off NS_GYM_SIDECAR_SCRIPT) matches.
READY_FILE="${NS_SIDECAR_READY_FILE:-${BASH_SOURCE[0]}.ready}"
LOG_DIR="${OUT_DIR}/sidecar_logs"
mkdir -p "$LOG_DIR" "${OUT_DIR}/worklogs" "${OUT_DIR}/batch_plans"

echo "[sidecars] PYTHONPATH=$PYTHONPATH"
echo "[sidecars] OUT_DIR=$OUT_DIR RUN_ID=$RUN_ID BENCH=$BENCH_PATH"
echo "[sidecars] sandbox=${NEMO_SKILLS_SANDBOX_HOST:-?}:${NEMO_SKILLS_SANDBOX_PORT:-?}"
echo "[sidecars] ports worklog=$WORKLOG_PORT benchmark=$BENCH_PORT python=$PY_PORT"

pids=()

# worklog (Tool subclass via generic HTTP wrapper)
NS_WORKLOG_DIR="${OUT_DIR}/worklogs" NS_WORKLOG_RUN_ID="$RUN_ID" NS_WORKLOG_AGENT_ID="$AGENT_ID" \
  "$PYBIN" -m nemo_skills.mcp.http_serve \
    nemo_skills.mcp.servers.agentic.worklog_tool:WorklogTool \
    --host "$HOST" --port "$WORKLOG_PORT" --path /mcp \
    >"$LOG_DIR/worklog.log" 2>&1 &
pids+=($!)

# benchmark (Tool subclass via generic HTTP wrapper)
# NS_WORKLOG_DIR/RUN_ID are passed so plan_batch can read the worklog for RESUME
# (skip already-completed ids on a restart) when NS_BATCH_RESUME is set. Resume is
# OFF by default — set NS_BATCH_RESUME=1 (env) on a restart to enable it.
NS_BENCHMARK_PATH="$BENCH_PATH" NS_BATCH_PLAN_DIR="${OUT_DIR}/batch_plans" \
NS_WORKLOG_DIR="${OUT_DIR}/worklogs" NS_WORKLOG_RUN_ID="$RUN_ID" \
NS_BATCH_RESUME="${NS_BATCH_RESUME:-}" \
  "$PYBIN" -m nemo_skills.mcp.http_serve \
    nemo_skills.mcp.servers.agentic.benchmark_tool:BenchmarkTool \
    --host "$HOST" --port "$BENCH_PORT" --path /mcp \
    >"$LOG_DIR/benchmark.log" 2>&1 &
pids+=($!)

# python (FastMCP server over streamable-http; reads sandbox host/port from env).
# Stage 2: point it at a SHARED external sandbox if one is published, else use the
# inherited in-job sandbox (NEMO_SKILLS_SANDBOX_HOST/PORT, localhost). A shared
# sandbox is given either as NS_SIDECAR_SANDBOX_ADDR=host:port or via a pool file
# NS_SIDECAR_SANDBOX_POOL/sandbox_addr (written by shared_sandbox.sh).
PY_SBX_HOST="${NEMO_SKILLS_SANDBOX_HOST:-127.0.0.1}"
PY_SBX_PORT="${NEMO_SKILLS_SANDBOX_PORT:-6000}"
SBX_ADDR="${NS_SIDECAR_SANDBOX_ADDR:-}"
if [ -z "$SBX_ADDR" ] && [ -n "${NS_SIDECAR_SANDBOX_POOL:-}" ]; then
  for i in $(seq 1 120); do
    [ -f "${NS_SIDECAR_SANDBOX_POOL}/sandbox_addr" ] && { SBX_ADDR=$(cat "${NS_SIDECAR_SANDBOX_POOL}/sandbox_addr"); break; }
    sleep 2
  done
fi
if [ -n "$SBX_ADDR" ]; then
  PY_SBX_HOST="${SBX_ADDR%%:*}"; PY_SBX_PORT="${SBX_ADDR##*:}"
  echo "[sidecars] python -> SHARED sandbox $PY_SBX_HOST:$PY_SBX_PORT"
else
  echo "[sidecars] python -> in-job sandbox $PY_SBX_HOST:$PY_SBX_PORT"
fi
NEMO_SKILLS_SANDBOX_HOST="$PY_SBX_HOST" NEMO_SKILLS_SANDBOX_PORT="$PY_SBX_PORT" \
"$PYBIN" -m nemo_skills.mcp.servers.python_tool \
    --transport streamable-http --host "$HOST" --port "$PY_PORT" \
    >"$LOG_DIR/python.log" 2>&1 &
pids+=($!)

echo "[sidecars] launched pids: ${pids[*]}"

# Readiness: do a real MCP initialize+list_tools handshake against each endpoint.
ready_check() {
  "$PYBIN" - "$@" <<'PY'
import sys, asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
urls = sys.argv[1:]
async def one(u):
    async with streamablehttp_client(u) as (r,w,_):
        async with ClientSession(r,w) as s:
            await s.initialize()
            t = await s.list_tools()
            return [x.name for x in t.tools]
async def main():
    for u in urls:
        names = await one(u)
        if not names:
            print("EMPTY", u); sys.exit(1)
    print("ALL_READY")
asyncio.run(main())
PY
}

URLS=("http://$HOST:$WORKLOG_PORT/mcp" "http://$HOST:$BENCH_PORT/mcp" "http://$HOST:$PY_PORT/mcp")
for attempt in $(seq 1 120); do
  # bail early if any sidecar died
  for p in "${pids[@]}"; do
    kill -0 "$p" 2>/dev/null || { echo "[sidecars] ERROR pid $p died; see $LOG_DIR/*.log"; tail -20 "$LOG_DIR"/*.log; exit 1; }
  done
  if ready_check "${URLS[@]}" 2>/dev/null | grep -q ALL_READY; then
    echo "[sidecars] all 3 MCP endpoints ready (attempt $attempt)"
    touch "$READY_FILE"
    break
  fi
  sleep 1
done

if [ ! -f "$READY_FILE" ]; then
  echo "[sidecars] ERROR: endpoints not ready in time"; tail -20 "$LOG_DIR"/*.log; exit 1
fi

# Hold the children for the life of the job (SLURM cgroup reaps on job end).
wait
