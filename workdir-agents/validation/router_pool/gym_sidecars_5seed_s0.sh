#!/bin/bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NS_SIDECAR_OUT_DIR="/alaptev/exp/a0py_5seed_s0"
export NS_SIDECAR_RUN_ID="nhw5s0"
export NS_SIDECAR_SANDBOX_POOL="/alaptev/data/lazypool_5seed"
export NS_SIDECAR_READY_FILE="${BASH_SOURCE[0]}.ready"
exec bash "${HERE}/gym_sidecars.sh"
