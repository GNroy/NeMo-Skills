#!/bin/bash
# Consolidated single-node 5-seed: seed 4 sidecars on OFFSET ports (all seeds
# share ONE node -> 127.0.0.1 ports must not collide). Shared external sandbox.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NS_SIDECAR_OUT_DIR="/alaptev/exp/a0py_5seedN_s4"
export NS_SIDECAR_RUN_ID="nhw5n4"
export NS_SIDECAR_SANDBOX_POOL="/alaptev/data/lazypool_5seed"
export NS_SIDECAR_WORKLOG_PORT=9141
export NS_SIDECAR_BENCH_PORT=9142
export NS_SIDECAR_PY_PORT=9143
export NS_SIDECAR_READY_FILE="${BASH_SOURCE[0]}.ready"
exec bash "${HERE}/gym_sidecars.sh"
