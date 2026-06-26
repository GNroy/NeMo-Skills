#!/bin/bash
# 2-seed multi run: seed mA. Per-seed worklog/output, SHARED sandbox (lazypool_multi).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NS_SIDECAR_OUT_DIR="/alaptev/exp/a0py_lazy_mA"
export NS_SIDECAR_RUN_ID="nhwmA"
export NS_SIDECAR_SANDBOX_POOL="/alaptev/data/lazypool_multi"
export NS_SIDECAR_READY_FILE="${BASH_SOURCE[0]}.ready"
exec bash "${HERE}/gym_sidecars.sh"
