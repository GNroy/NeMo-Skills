#!/bin/bash
# Chunked A/B/C launcher.  Replaces launch_abc_smoke.sh.
#
# Layout:
#   1. sbatch Kimi vLLM daemon    → kimi.endpoint   (stays up across all passes)
#   2. sbatch judge daemon        → judge.endpoint  (stays up across all passes)
#   3. For pass in 0, 1, 2:
#        a. sbatch NUM_CHUNKS chunk-workers in parallel
#        b. wait for all chunks
#        c. if pass == 1: mergeback all chunk hermes_homes → TEMPLATE_RUN1
#        d. if pass == 1: snapshot TEMPLATE_RUN1 → TEMPLATE_RUN2
#   4. scancel both daemons
#   5. aggregate per-pass metrics (avg reward / pass@1)
#
# Verifying happens in-loop via NeMo-Gym's frontierscience_judge resources
# server, so there is no separate "judge phase".  This requires the judge
# daemon to be alive *during* every rollout — both daemons stay up for
# the duration of the run.
#
# Override defaults via env or `--key value` flags.

set -euo pipefail

# -----------------------------------------------------------------------------
# Defaults.  Identical to launch_abc_smoke.sh where the underlying knob is
# the same; new defaults are NUM_CHUNKS, judge endpoint wiring, etc.
# -----------------------------------------------------------------------------
EXP_DIR_DEFAULT="/lustre/fsw/portfolios/nemotron/users/alaptev/exp/abc_chunked/$(date -u +%Y%m%dT%H%M%SZ)"
EXP_DIR="${EXP_DIR:-${EXP_DIR_DEFAULT}}"
INPUT_FILE="${INPUT_FILE:-/lustre/fsw/portfolios/nemotron/users/alaptev/data/smoke_ng.jsonl}"
NUM_CHUNKS="${NUM_CHUNKS:-4}"
LIMIT="${LIMIT:-0}"  # 0 = all rows; >0 = cap input (smoke convenience)

TEMPLATE_RUN0="${TEMPLATE_RUN0:-/lustre/fsw/portfolios/nemotron/users/alaptev/hermes_home/frontier_seed}"
TEMPLATE_RUN1="${TEMPLATE_RUN1:-/lustre/fsw/portfolios/nemotron/users/alaptev/hermes_home/run1_template}"
TEMPLATE_RUN2="${TEMPLATE_RUN2:-/lustre/fsw/portfolios/nemotron/users/alaptev/hermes_home/run2_template}"

POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/hf_models/Kimi-K2.6}"
POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-/hf_models/Kimi-K2.6}"
POLICY_PORT="${POLICY_PORT:-35041}"
POLICY_CONTAINER="${POLICY_CONTAINER:-/lustre/fsw/portfolios/nemotron/users/alaptev/containers/vllm-glm51-cu130-ray.sqsh}"
POLICY_EXTRA_VLLM_ARGS="${POLICY_EXTRA_VLLM_ARGS:---enable-expert-parallel --distributed-executor-backend=ray --data-parallel-size 4 --tensor-parallel-size 1 --language-model-only --compilation-config '{\"pass_config\": {\"fuse_allreduce_rms\": false}}' --model-loader-extra-config '{\"enable_multithread_load\": true, \"num_threads\": 96}' --max-model-len 131072 --enable-auto-tool-choice --tool-parser-plugin /lustre/fsw/portfolios/nemotron/users/alaptev/reasoning_parsers/kimi_k26_tool_parser.py --tool-call-parser kimi_k26 --reasoning-parser-plugin /lustre/fsw/portfolios/nemotron/users/alaptev/reasoning_parsers/kimi_k26_reasoning_parser.py --reasoning-parser kimi_k26}"

JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/hf_models/gpt-oss-120b}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-/hf_models/gpt-oss-120b}"
JUDGE_PORT="${JUDGE_PORT:-35042}"
JUDGE_CONTAINER="${JUDGE_CONTAINER:-/lustre/fsw/portfolios/nemotron/users/igitman/images/nemo-skills-vllm-latest.sqsh}"
JUDGE_EXTRA_VLLM_ARGS="${JUDGE_EXTRA_VLLM_ARGS:---tensor-parallel-size 4 --max-model-len 131072}"

GYM_PATH="${GYM_PATH:-/lustre/fsw/portfolios/nemotron/users/alaptev/NeMo-Gym}"
HERMES_AGENT_PATH="${HERMES_AGENT_PATH:-/lustre/fsw/portfolios/nemotron/users/alaptev/hermes-agent}"
CLIENT_CONTAINER="${CLIENT_CONTAINER:-/lustre/fsw/portfolios/nemotron/users/igitman/images/nemo-skills-0a3c03f.sqsh}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATOR_SCRIPT_DEFAULT="${GYM_PATH}/responses_api_agents/hermes_agent/scripts/merge_hermes_home.py"
CURATOR_SCRIPT="${CURATOR_SCRIPT:-${CURATOR_SCRIPT_DEFAULT}}"

# CLI overrides.
while [ $# -gt 0 ]; do
    case "$1" in
        --exp-dir)        EXP_DIR="$2"; shift 2 ;;
        --input-file)     INPUT_FILE="$2"; shift 2 ;;
        --num-chunks)     NUM_CHUNKS="$2"; shift 2 ;;
        --limit)          LIMIT="$2"; shift 2 ;;
        --template-run0)  TEMPLATE_RUN0="$2"; shift 2 ;;
        --template-run1)  TEMPLATE_RUN1="$2"; shift 2 ;;
        --template-run2)  TEMPLATE_RUN2="$2"; shift 2 ;;
        --policy-port)    POLICY_PORT="$2"; shift 2 ;;
        --judge-port)     JUDGE_PORT="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "${EXP_DIR}" "${EXP_DIR}/daemons" "${EXP_DIR}/logs"
POLICY_ENDPOINT_FILE="${EXP_DIR}/daemons/kimi.endpoint"
JUDGE_ENDPOINT_FILE="${EXP_DIR}/daemons/judge.endpoint"
rm -f "${POLICY_ENDPOINT_FILE}" "${JUDGE_ENDPOINT_FILE}"

# Track every SLURM job we submit so a Ctrl+C / disconnect scancels them —
# leaving a 4h GPU daemon orphaned would violate cluster rules.
SUBMITTED_JOBS=()
cleanup_jobs () {
    echo
    echo "=== launcher exiting; scancelling tracked SLURM jobs ==="
    for jid in "${SUBMITTED_JOBS[@]:-}"; do
        [ -n "$jid" ] || continue
        echo "  scancel $jid"
        scancel "$jid" 2>/dev/null || true
    done
}
trap cleanup_jobs INT TERM EXIT

echo "=== abc_chunked ==="
echo "EXP_DIR=${EXP_DIR}"
echo "INPUT_FILE=${INPUT_FILE}"
echo "NUM_CHUNKS=${NUM_CHUNKS}"
echo "POLICY_ENDPOINT_FILE=${POLICY_ENDPOINT_FILE}"
echo "JUDGE_ENDPOINT_FILE=${JUDGE_ENDPOINT_FILE}"

# Pre-flight: TEMPLATE_RUN0 must exist; TEMPLATE_RUN1 is reset from
# frontier_seed so a previous mutation doesn't bleed in; TEMPLATE_RUN2 is
# staged by this launcher after pass-1 mergeback.
if [ ! -d "${TEMPLATE_RUN0}" ]; then
    echo "ERROR: TEMPLATE_RUN0 missing: ${TEMPLATE_RUN0}"
    exit 1
fi
echo "=== pre-flight: refreshing TEMPLATE_RUN1 from TEMPLATE_RUN0 ==="
rm -rf "${TEMPLATE_RUN1}"
cp -a "${TEMPLATE_RUN0}" "${TEMPLATE_RUN1}"
rm -rf "${TEMPLATE_RUN2}"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "(dry-run; sbatch commands would have been issued — exiting)"
    exit 0
fi

# -----------------------------------------------------------------------------
# Prep INPUT_FILE into rollout-ready shape (idempotent — rows that
# already have agent_ref + responses_create_params pass through unchanged).
# Source rows must carry top-level ``question`` and ``expected_answer``
# (the frontierscience_judge reads those off the verify request body).
# -----------------------------------------------------------------------------
PREPARED_INPUT="${EXP_DIR}/input.jsonl"
python3 "${SCRIPT_DIR}/prepare_frontierscience_input.py" \
    --input "${INPUT_FILE}" --output "${PREPARED_INPUT}" --limit "${LIMIT}"
INPUT_FILE="${PREPARED_INPUT}"
echo "INPUT_FILE (prepared) = ${INPUT_FILE}"

# -----------------------------------------------------------------------------
# Daemons — both stay up for the duration; verify happens in-loop.
# -----------------------------------------------------------------------------
echo
echo "=== launching Kimi vLLM daemon ==="
KIMI_JOB=$(sbatch --parsable \
    --job-name=kimi_chunked \
    --chdir="${EXP_DIR}/logs" \
    --export=ALL,MODEL_PATH="${POLICY_MODEL_PATH}",SERVED_NAME="${POLICY_MODEL_NAME}",PORT="${POLICY_PORT}",ENDPOINT_FILE="${POLICY_ENDPOINT_FILE}",CONTAINER="${POLICY_CONTAINER}",LOG_DIR="${EXP_DIR}/logs",EXTRA_VLLM_ARGS="${POLICY_EXTRA_VLLM_ARGS}" \
    "${SCRIPT_DIR}/vllm_daemon.sbatch")
echo "Kimi daemon SLURM job: ${KIMI_JOB}"
SUBMITTED_JOBS+=("${KIMI_JOB}")

echo "=== launching gpt-oss-120b judge daemon ==="
JUDGE_JOB=$(sbatch --parsable \
    --job-name=judge_chunked \
    --chdir="${EXP_DIR}/logs" \
    --export=ALL,MODEL_PATH="${JUDGE_MODEL_PATH}",SERVED_NAME="${JUDGE_MODEL_NAME}",PORT="${JUDGE_PORT}",ENDPOINT_FILE="${JUDGE_ENDPOINT_FILE}",CONTAINER="${JUDGE_CONTAINER}",LOG_DIR="${EXP_DIR}/logs",EXTRA_VLLM_ARGS="${JUDGE_EXTRA_VLLM_ARGS}" \
    "${SCRIPT_DIR}/vllm_daemon.sbatch")
echo "Judge daemon SLURM job: ${JUDGE_JOB}"
SUBMITTED_JOBS+=("${JUDGE_JOB}")

# -----------------------------------------------------------------------------
# Pass loop.
# -----------------------------------------------------------------------------
submit_pass_chunks () {
    local pass_id="$1"
    local template_home="$2"
    local persist_memory="$3"
    local save_trajectories="$4"

    echo
    echo "=== pass ${pass_id}: submitting ${NUM_CHUNKS} chunk-workers ==="
    PASS_CHUNK_JOBS=()
    for (( chunk_id = 0; chunk_id < NUM_CHUNKS; chunk_id++ )); do
        local jid
        jid=$(sbatch --parsable \
            --job-name="abc_p${pass_id}_c${chunk_id}" \
            --chdir="${EXP_DIR}/logs" \
            --export=ALL,POLICY_ENDPOINT_FILE="${POLICY_ENDPOINT_FILE}",JUDGE_ENDPOINT_FILE="${JUDGE_ENDPOINT_FILE}",EXP_DIR="${EXP_DIR}",PASS_ID="${pass_id}",CHUNK_ID="${chunk_id}",NUM_CHUNKS="${NUM_CHUNKS}",INPUT_FILE="${INPUT_FILE}",TEMPLATE_HOME="${template_home}",PERSIST_MEMORY="${persist_memory}",SAVE_TRAJECTORIES="${save_trajectories}",POLICY_MODEL_NAME="${POLICY_MODEL_NAME}",JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME}",GYM_PATH="${GYM_PATH}",HERMES_AGENT_PATH="${HERMES_AGENT_PATH}",CONTAINER="${CLIENT_CONTAINER}",POLICY_DAEMON_JOB_ID="${KIMI_JOB}",JUDGE_DAEMON_JOB_ID="${JUDGE_JOB}" \
            "${SCRIPT_DIR}/chunk_worker.sbatch")
        echo "  pass${pass_id} chunk${chunk_id} → SLURM ${jid}"
        PASS_CHUNK_JOBS+=("${jid}")
        SUBMITTED_JOBS+=("${jid}")
    done
}

wait_for_chunks () {
    local pass_id="$1"
    echo "=== pass ${pass_id}: waiting for ${#PASS_CHUNK_JOBS[@]} chunks ==="
    local jids
    jids=$(IFS=,; echo "${PASS_CHUNK_JOBS[*]}")
    while squeue -j "${jids}" -h -o '%T' 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING'; do
        sleep 30
    done
    echo "=== pass ${pass_id}: chunk exit states ==="
    for jid in "${PASS_CHUNK_JOBS[@]}"; do
        local st
        st=$(sacct -j "$jid" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
        echo "  ${jid}: ${st}"
    done
}

# ---- Pass 0: cold baseline ----
submit_pass_chunks 0 "$TEMPLATE_RUN0" false false
wait_for_chunks 0

# ---- Pass 1: learning ----
submit_pass_chunks 1 "$TEMPLATE_RUN1" true true
wait_for_chunks 1

# ---- Mergeback: fold all pass-1 chunk homes into TEMPLATE_RUN1 ----
echo
echo "=== mergeback: pass-1 chunk homes → ${TEMPLATE_RUN1} ==="
PASS1_HOMES=()
for (( chunk_id = 0; chunk_id < NUM_CHUNKS; chunk_id++ )); do
    home="${EXP_DIR}/pass1/chunk${chunk_id}/hermes_home"
    [ -d "$home" ] && PASS1_HOMES+=("$home")
done
if [ "${#PASS1_HOMES[@]}" -gt 0 ]; then
    python3 "$CURATOR_SCRIPT" \
        --template "$TEMPLATE_RUN1" \
        --audit "${EXP_DIR}/pass1/merge_audit.json" \
        -- "${PASS1_HOMES[@]}"
else
    echo "WARNING: no pass-1 hermes_homes found — skipping mergeback"
fi

# ---- Stage TEMPLATE_RUN2 from the mutated TEMPLATE_RUN1 ----
echo "=== stage TEMPLATE_RUN2: ${TEMPLATE_RUN1} → ${TEMPLATE_RUN2} ==="
rm -rf "$TEMPLATE_RUN2"
cp -a "$TEMPLATE_RUN1" "$TEMPLATE_RUN2"

# ---- Pass 2: evaluation against warmed template ----
submit_pass_chunks 2 "$TEMPLATE_RUN2" false false
wait_for_chunks 2

# -----------------------------------------------------------------------------
# Tear down daemons.
# -----------------------------------------------------------------------------
echo
echo "=== scancel daemons (free GPUs) ==="
scancel "${KIMI_JOB}" 2>/dev/null || true
scancel "${JUDGE_JOB}" 2>/dev/null || true

# -----------------------------------------------------------------------------
# Aggregate per-pass: union all chunk rollouts.jsonl and compute pass@1.
# Reward is in each row (frontierscience_judge sets it during /verify),
# so we don't need a separate judge phase or aggregation step.
# -----------------------------------------------------------------------------
echo
echo "=== Summary ==="
for pass_id in 0 1 2; do
    pass_dir="${EXP_DIR}/pass${pass_id}"
    glob_pattern="${pass_dir}/chunk*/rollouts.jsonl"
    # shellcheck disable=SC2086
    files=( $(ls ${glob_pattern} 2>/dev/null || true) )
    if [ "${#files[@]}" -eq 0 ]; then
        echo "pass${pass_id}: no rollouts files found"
        continue
    fi
    python3 - "${files[@]}" <<'PY'
import json, sys
n = ok = 0
for path in sys.argv[1:]:
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            r = row.get("reward")
            if r is None:
                continue
            n += 1
            if float(r) >= 1.0:
                ok += 1
pct = 100.0 * ok / n if n else 0.0
print(f"  ok={ok}/{n} ({pct:.1f}%)")
PY
done

echo
echo "Done.  EXP_DIR=${EXP_DIR}"
