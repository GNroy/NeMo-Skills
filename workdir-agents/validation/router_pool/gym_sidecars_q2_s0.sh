#!/bin/bash
# Q2 5-seed run: seed s0. Per-seed worklog/output, SHARED sandbox (lazypool_q2).
# Driver runs on its OWN GPU batch node, so sidecar ports use defaults (no offset).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NS_SIDECAR_OUT_DIR="/alaptev/exp/a0py_q2_s0"
export NS_SIDECAR_RUN_ID="nhwq2s0"
export NS_SIDECAR_SANDBOX_POOL="/alaptev/data/lazypool_q2"
export NS_SIDECAR_READY_FILE="${BASH_SOURCE[0]}.ready"
exec bash "${HERE}/gym_sidecars.sh"
