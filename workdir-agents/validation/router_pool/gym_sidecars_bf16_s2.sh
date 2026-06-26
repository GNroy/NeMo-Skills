#!/bin/bash
# BF16 5-seed run: seed s2. Per-seed worklog/output, SHARED sandbox (lazypool_bf16).
# Driver runs on its OWN cpu node -> sidecar ports default (no offset). RESUME ON:
# a resume-restart skips already-completed ids (NS_BATCH_RESUME=1).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NS_SIDECAR_OUT_DIR="/alaptev/exp/a0py_bf16_s2"
export NS_SIDECAR_RUN_ID="nhwbf16s2"
export NS_SIDECAR_SANDBOX_POOL="/alaptev/data/lazypool_bf16"
export NS_SIDECAR_READY_FILE="${BASH_SOURCE[0]}.ready"
export NS_BATCH_RESUME=1
exec bash "${HERE}/gym_sidecars.sh"
