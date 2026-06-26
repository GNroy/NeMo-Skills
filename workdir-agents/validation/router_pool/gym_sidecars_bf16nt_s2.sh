#!/bin/bash
# BF16 NO-TOOL arm: seed s2. Same shared sandbox/router; python sidecar still
# launched but unused (manifest disables mcp-python). Resume ON.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NS_SIDECAR_OUT_DIR="/alaptev/exp/a0py_bf16nt_s2"
export NS_SIDECAR_RUN_ID="nhwbf16nts2"
export NS_SIDECAR_SANDBOX_POOL="/alaptev/data/lazypool_bf16"
export NS_SIDECAR_READY_FILE="${BASH_SOURCE[0]}.ready"
export NS_BATCH_RESUME=1
exec bash "${HERE}/gym_sidecars.sh"
