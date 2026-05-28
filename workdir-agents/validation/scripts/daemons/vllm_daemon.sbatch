#!/bin/bash
#SBATCH --account=nemotron_reason_science
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output=%x_%j.log

# Generic long-lived vLLM serve daemon.  Held up only as long as needed
# by its phase — the launcher scancels this job as soon as the dependent
# client phase finishes (cluster has tight GPU-utilization rules).
#
# Required env (set via --export when sbatch'ing):
#   MODEL_PATH        absolute path to the HF model dir, e.g. /hf_models/Kimi-K2.6
#   SERVED_NAME       value for --served-model-name (what the client passes as `model:`)
#   PORT              listen port (host:port returned to the client)
#   ENDPOINT_FILE     where to write `http://<host>:<port>/v1` once /v1/models is healthy
#   CONTAINER         sqsh container path
#   LOG_DIR           where to dump the inner vllm log
#   EXTRA_VLLM_ARGS   appended to `vllm serve` (e.g. tool-call-parser, max-model-len)
#
# The endpoint file is removed on exit (trap) so a dead daemon can't be
# mistaken for a live one.

set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH required}"
: "${SERVED_NAME:?SERVED_NAME required}"
: "${PORT:?PORT required}"
: "${ENDPOINT_FILE:?ENDPOINT_FILE required}"
: "${CONTAINER:?CONTAINER required}"
: "${LOG_DIR:?LOG_DIR required}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

mkdir -p "$(dirname "${ENDPOINT_FILE}")" "${LOG_DIR}"
trap 'rm -f "${ENDPOINT_FILE}"' EXIT

# Mirror the aws-cmh env_vars block from cluster_configs/aws-cmh.yaml.
# Without these the gpt-oss-120b harmony tokenizer can't find its
# pre-staged vocab file and tries to download it under
# HF_HUB_OFFLINE — which fails on compute nodes with no internet.
# NS's pipeline sets these via its cluster config; raw sbatch scripts
# don't get them for free, so we set them here explicitly.
export HF_HOME=/alaptev/hf-cache
export TRANSFORMERS_CACHE=/alaptev/hf-cache
export HUGGINGFACE_HUB_CACHE=/alaptev/hf-cache
export TIKTOKEN_ENCODINGS_BASE=/alaptev/tiktoken-encodings
export TIKTOKEN_RS_CACHE_DIR=/alaptev/tiktoken-encodings/cache
export HF_HUB_OFFLINE=1
# Kimi-K2.6 cold-load can exceed vLLM's 600s engine-ready default when
# lustre I/O is contended.  Bump to 30 min so the engine has slack.  Pure
# multi-threaded weight loading on a warm cache lands in ~5 min; this
# only matters for the cold first request of a session.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"

# Identity-mount the lustre user dir alongside the /alaptev alias so paths
# work both inside the container (what srun sees) and on the login node
# (where the launcher does cp/mkdir).  Without this, the launcher's
# /lustre/... defaults wouldn't resolve inside the container.
CONTAINER_MOUNTS="/lustre/fsw/portfolios/nemotron/users/igitman/hf_models:/hf_models,/lustre/fsw/portfolios/nemotron/users/alaptev:/alaptev,/lustre/fsw/portfolios/nemotron/users/alaptev:/lustre/fsw/portfolios/nemotron/users/alaptev,/lustre/fsw/portfolios/nemotron/users/alaptev/data:/data"

INNER_LOG="${LOG_DIR}/vllm_${SLURM_JOB_ID}_inner.log"

INNER_CMD=$(cat <<INNER
set -euo pipefail
echo "[\$(date -Is)] starting vllm serve ${MODEL_PATH}"
vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --host 0.0.0.0 \
    --port ${PORT} \
    --trust-remote-code \
    ${EXTRA_VLLM_ARGS} \
    > "${INNER_LOG}" 2>&1 &
VLLM_PID=\$!
echo "[\$(date -Is)] vllm serve PID \${VLLM_PID}; polling /v1/models"

ATTEMPTS=0
MAX_ATTEMPTS=720  # 60 min ceiling on boot
while true; do
    if ! kill -0 \$VLLM_PID 2>/dev/null; then
        echo "[\$(date -Is)] ERROR: vllm serve died before /v1/models came up"
        tail -200 "${INNER_LOG}" || true
        exit 1
    fi
    if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        break
    fi
    ATTEMPTS=\$((ATTEMPTS+1))
    if [ \$ATTEMPTS -ge \$MAX_ATTEMPTS ]; then
        echo "[\$(date -Is)] ERROR: vllm /v1/models did not come up in 60 min"
        kill \$VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 5
done

HOST_IP=\$(hostname -i | awk '{print \$1}')
URL="http://\${HOST_IP}:${PORT}/v1"
echo "[\$(date -Is)] vllm READY at \${URL}; writing ${ENDPOINT_FILE}"
echo "\${URL}" > "${ENDPOINT_FILE}"

wait \$VLLM_PID
echo "[\$(date -Is)] vllm exited; daemon shutting down"
INNER
)

srun \
    --container-image "${CONTAINER}" \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --no-container-mount-home \
    --container-env=HF_HOME,TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE,TIKTOKEN_ENCODINGS_BASE,TIKTOKEN_RS_CACHE_DIR,HF_HUB_OFFLINE,VLLM_ENGINE_READY_TIMEOUT_S \
    --gpus-per-node=4 \
    --ntasks-per-node=1 \
    bash -c "${INNER_CMD}"
