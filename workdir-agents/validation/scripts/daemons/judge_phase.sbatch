#!/bin/bash
#SBATCH --account=nemotron_reason_science
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x_%j.log

# Judge phase — re-grades the three rollouts.jsonl files via the
# frontierscience judge prompt against a separate gpt-oss-120b vLLM
# daemon.  The judge daemon is the only other GPU job running while
# this script executes; the Kimi daemon was scancelled by the launcher
# before this phase starts.
#
# Required env:
#   JUDGE_ENDPOINT_FILE   path the judge daemon writes its `http://...:port/v1` to
#   EXP_DIR               experiment root containing run0/, run1/, run2/
#   JUDGE_MODEL_NAME      served-model-name on the judge daemon
#   JUDGE_PROMPT_YAML     path to NeMo-Gym frontierscience_judge prompt YAML
#   JUDGE_SCRIPT          path to judge_rollouts.py
#   CONTAINER             client container

set -euo pipefail

: "${JUDGE_ENDPOINT_FILE:?required}"
: "${EXP_DIR:?required}"
: "${JUDGE_MODEL_NAME:?required}"
: "${JUDGE_PROMPT_YAML:?required}"
: "${JUDGE_SCRIPT:?required}"
: "${CONTAINER:?required}"

# Identity-mount the lustre user dir (see rollout_phase.sbatch for why).
CONTAINER_MOUNTS="/lustre/fsw/portfolios/nemotron/users/igitman/hf_models:/hf_models,/lustre/fsw/portfolios/nemotron/users/alaptev:/alaptev,/lustre/fsw/portfolios/nemotron/users/alaptev:/lustre/fsw/portfolios/nemotron/users/alaptev,/lustre/fsw/portfolios/nemotron/users/alaptev/data:/data"

# Standard cluster env_vars (see cluster_configs/aws-cmh.yaml).
export HF_HOME=/alaptev/hf-cache
export TRANSFORMERS_CACHE=/alaptev/hf-cache
export HUGGINGFACE_HUB_CACHE=/alaptev/hf-cache
export TIKTOKEN_ENCODINGS_BASE=/alaptev/tiktoken-encodings
export TIKTOKEN_RS_CACHE_DIR=/alaptev/tiktoken-encodings/cache
export HF_HUB_OFFLINE=1

INNER_CMD=$(cat <<'INNER_EOF'
set -euo pipefail

echo "[$(date -Is)] judge phase booting"

echo "[$(date -Is)] waiting for judge endpoint at $JUDGE_ENDPOINT_FILE"
WAIT_DEADLINE=$(( $(date +%s) + 3600 ))
until [ -s "$JUDGE_ENDPOINT_FILE" ]; do
    if [ "$(date +%s)" -ge "$WAIT_DEADLINE" ]; then
        echo "ERROR: judge endpoint never appeared in 60 min"
        exit 1
    fi
    # Bail out fast if the daemon's SLURM job has already exited —
    # otherwise we'd burn the full 60 min polling for a file that
    # will never be written.
    if [ -n "${DAEMON_JOB_ID:-}" ]; then
        st=$(squeue -j "$DAEMON_JOB_ID" -h -o '%T' 2>/dev/null || true)
        if [ -z "$st" ]; then
            echo "ERROR: daemon job $DAEMON_JOB_ID is no longer in the queue (likely failed)"
            sacct -j "$DAEMON_JOB_ID" -X -n -o State,ExitCode 2>/dev/null | head -1 || true
            exit 1
        fi
    fi
    sleep 10
done
JUDGE_URL=$(cat "$JUDGE_ENDPOINT_FILE")
echo "[$(date -Is)] judge endpoint: $JUDGE_URL"

# judge_rollouts.py uses stdlib only (urllib + json + re) with the
# frontierscience judge prompt inlined.  No pyyaml install needed.

for run in run0 run1 run2; do
    in="${EXP_DIR}/${run}/rollouts.jsonl"
    out="${EXP_DIR}/${run}/judged.jsonl"
    if [ ! -s "$in" ]; then
        echo "WARN: missing $in — skipping ${run}"
        continue
    fi
    echo "[$(date -Is)] === judging ${run} (${in}) ==="
    python3 "$JUDGE_SCRIPT" \
        "$in" "$out" \
        --judge-url "$JUDGE_URL" \
        --judge-model "$JUDGE_MODEL_NAME" \
        --prompt-yaml "$JUDGE_PROMPT_YAML" \
        2>&1 | tee "${EXP_DIR}/${run}/judge.log"
done

touch "${EXP_DIR}/.judge_phase_done"
echo "[$(date -Is)] judge phase complete"
INNER_EOF
)

srun \
    --container-image "${CONTAINER}" \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --no-container-mount-home \
    --container-env=JUDGE_ENDPOINT_FILE,EXP_DIR,JUDGE_MODEL_NAME,JUDGE_PROMPT_YAML,JUDGE_SCRIPT,DAEMON_JOB_ID \
    --ntasks-per-node=1 \
    bash -c "${INNER_CMD}"
