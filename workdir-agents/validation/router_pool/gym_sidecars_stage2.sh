#!/bin/bash
# Stage 2 sidecar wrapper: same 3 HTTP MCP sidecars as gym_sidecars.sh, but the
# python sidecar points at the EXTERNAL shared sandbox (lazypool_stage2/sandbox_addr,
# published by shared_sandbox.sh) instead of an in-job sandbox. Set NS_GYM_SIDECAR_SCRIPT
# to THIS file; it injects the stage2 paths + sandbox pool then execs gym_sidecars.sh.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NS_SIDECAR_OUT_DIR="${NS_SIDECAR_OUT_DIR:-/alaptev/exp/a0py_lazy_stage2}"
export NS_SIDECAR_RUN_ID="${NS_SIDECAR_RUN_ID:-nhwstage2}"
export NS_SIDECAR_SANDBOX_POOL="${NS_SIDECAR_SANDBOX_POOL:-/alaptev/data/lazypool_stage2}"
# The gym hook waits on <this-script>.ready; make gym_sidecars.sh write exactly that.
export NS_SIDECAR_READY_FILE="${BASH_SOURCE[0]}.ready"
exec bash "${HERE}/gym_sidecars.sh"
