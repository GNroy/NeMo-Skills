#!/bin/bash
# Orchestrate the A/B/C smoke with sequential daemon lifecycles.
#
#   1. sbatch Kimi vLLM daemon            → kimi.endpoint
#   2. sbatch rollout-phase client        → waits for endpoint, runs 3 passes
#   3. on rollout-phase done: scancel Kimi daemon (release GPUs)
#   4. sbatch judge (gpt-oss-120b) daemon → judge.endpoint
#   5. sbatch judge-phase client          → waits for endpoint, grades 3 jsonl
#   6. on judge-phase done: scancel judge daemon
#   7. print pass-rate summary
#
# This script is invoked from the aws-cmh login node (no SLURM heartbeat
# itself).  It manages job lifecycles via squeue / sacct / scancel.
#
# Override defaults via env or CLI flags (`--exp-dir`, `--input-file`).

set -euo pipefail

# -----------------------------------------------------------------------------
# Defaults.  Override via env or `key=value` style positional args.
# -----------------------------------------------------------------------------
EXP_DIR_DEFAULT="/lustre/fsw/portfolios/nemotron/users/alaptev/exp/abc_smoke/$(date -u +%Y%m%dT%H%M%SZ)"
EXP_DIR="${EXP_DIR:-${EXP_DIR_DEFAULT}}"
INPUT_FILE="${INPUT_FILE:-/lustre/fsw/portfolios/nemotron/users/alaptev/data/smoke_ng.jsonl}"
TEMPLATE_RUN0="${TEMPLATE_RUN0:-/lustre/fsw/portfolios/nemotron/users/alaptev/hermes_home/frontier_seed}"
TEMPLATE_RUN1="${TEMPLATE_RUN1:-/lustre/fsw/portfolios/nemotron/users/alaptev/hermes_home/run1_template}"
TEMPLATE_RUN2="${TEMPLATE_RUN2:-/lustre/fsw/portfolios/nemotron/users/alaptev/hermes_home/run2_template}"

POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-/hf_models/Kimi-K2.6}"
POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-/hf_models/Kimi-K2.6}"
POLICY_PORT="${POLICY_PORT:-35041}"
POLICY_CONTAINER="${POLICY_CONTAINER:-/lustre/fsw/portfolios/nemotron/users/alaptev/containers/vllm-glm51-cu130-ray.sqsh}"
POLICY_EXTRA_VLLM_ARGS="${POLICY_EXTRA_VLLM_ARGS:---enable-expert-parallel --distributed-executor-backend=ray --tensor-parallel-size 4 --compilation-config '{\"pass_config\": {\"fuse_allreduce_rms\": false}}' --max-model-len 131072 --enable-auto-tool-choice --tool-parser-plugin /lustre/fsw/portfolios/nemotron/users/alaptev/abc_smoke/scripts/kimi_k26_tool_parser.py --tool-call-parser kimi_k26 --reasoning-parser kimi_k2}"

JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/hf_models/gpt-oss-120b}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-/hf_models/gpt-oss-120b}"
JUDGE_PORT="${JUDGE_PORT:-35042}"
JUDGE_CONTAINER="${JUDGE_CONTAINER:-/lustre/fsw/portfolios/nemotron/users/alaptev/containers/vllm-glm51-cu130-ray.sqsh}"
JUDGE_EXTRA_VLLM_ARGS="${JUDGE_EXTRA_VLLM_ARGS:---tensor-parallel-size 4 --max-model-len 131072}"

GYM_PATH="${GYM_PATH:-/lustre/fsw/portfolios/nemotron/users/alaptev/NeMo-Gym}"
HERMES_AGENT_PATH="${HERMES_AGENT_PATH:-/lustre/fsw/portfolios/nemotron/users/alaptev/hermes-agent}"
CLIENT_CONTAINER="${CLIENT_CONTAINER:-/lustre/fsw/portfolios/nemotron/users/igitman/images/nemo-skills-0a3c03f.sqsh}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JUDGE_SCRIPT_DEFAULT="${SCRIPT_DIR}/judge_rollouts.py"
JUDGE_PROMPT_YAML_DEFAULT="${GYM_PATH}/resources_servers/frontierscience_judge/prompts/judge.yaml"
CURATOR_SCRIPT_DEFAULT="${SCRIPT_DIR}/curator.py"
JUDGE_SCRIPT="${JUDGE_SCRIPT:-${JUDGE_SCRIPT_DEFAULT}}"
JUDGE_PROMPT_YAML="${JUDGE_PROMPT_YAML:-${JUDGE_PROMPT_YAML_DEFAULT}}"
CURATOR_SCRIPT="${CURATOR_SCRIPT:-${CURATOR_SCRIPT_DEFAULT}}"

# CLI overrides (simple --key value parser).
while [ $# -gt 0 ]; do
    case "$1" in
        --exp-dir)        EXP_DIR="$2"; shift 2 ;;
        --input-file)     INPUT_FILE="$2"; shift 2 ;;
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

# Track every SLURM job we submit so a Ctrl+C / disconnect can scancel
# them — leaving a 4 h GPU daemon orphaned would violate cluster rules.
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
trap cleanup_jobs INT TERM

echo "=== abc_smoke ==="
echo "EXP_DIR=${EXP_DIR}"
echo "INPUT_FILE=${INPUT_FILE}"
echo "POLICY_ENDPOINT_FILE=${POLICY_ENDPOINT_FILE}"
echo "JUDGE_ENDPOINT_FILE=${JUDGE_ENDPOINT_FILE}"

# Pre-flight: TEMPLATE_RUN0 must exist; TEMPLATE_RUN1 is reset from
# frontier_seed each run so a previous mutation doesn't bleed in;
# TEMPLATE_RUN2 is staged by the rollout phase after Run 1's mergeback,
# so we delete any leftover here.
if [ ! -d "${TEMPLATE_RUN0}" ]; then
    echo "ERROR: TEMPLATE_RUN0 missing: ${TEMPLATE_RUN0}"
    exit 1
fi
echo "=== pre-flight: refreshing TEMPLATE_RUN1 from TEMPLATE_RUN0 ==="
rm -rf "${TEMPLATE_RUN1}"
cp -a "${TEMPLATE_RUN0}" "${TEMPLATE_RUN1}"
echo "  ${TEMPLATE_RUN0} -> ${TEMPLATE_RUN1}"
rm -rf "${TEMPLATE_RUN2}"
echo "  cleared ${TEMPLATE_RUN2} (will be staged after Run 1 mergeback)"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "(dry-run; sbatch commands would have been issued — exiting)"
    exit 0
fi

# -----------------------------------------------------------------------------
# Phase 1 — Kimi daemon + rollout client
# -----------------------------------------------------------------------------
echo
echo "=== Phase 1: launching Kimi vLLM daemon ==="
KIMI_JOB=$(sbatch --parsable \
    --job-name=kimi_abc \
    --chdir="${EXP_DIR}/logs" \
    --export=ALL,MODEL_PATH="${POLICY_MODEL_PATH}",SERVED_NAME="${POLICY_MODEL_NAME}",PORT="${POLICY_PORT}",ENDPOINT_FILE="${POLICY_ENDPOINT_FILE}",CONTAINER="${POLICY_CONTAINER}",LOG_DIR="${EXP_DIR}/logs",EXTRA_VLLM_ARGS="${POLICY_EXTRA_VLLM_ARGS}" \
    "${SCRIPT_DIR}/vllm_daemon.sbatch")
echo "Kimi daemon SLURM job: ${KIMI_JOB}"
SUBMITTED_JOBS+=("${KIMI_JOB}")

echo "=== Phase 1: launching rollout-phase client (waits for Kimi endpoint) ==="
ROLLOUT_JOB=$(sbatch --parsable \
    --job-name=abc_rollouts \
    --chdir="${EXP_DIR}/logs" \
    --export=ALL,POLICY_ENDPOINT_FILE="${POLICY_ENDPOINT_FILE}",EXP_DIR="${EXP_DIR}",INPUT_FILE="${INPUT_FILE}",GYM_PATH="${GYM_PATH}",HERMES_AGENT_PATH="${HERMES_AGENT_PATH}",TEMPLATE_RUN0="${TEMPLATE_RUN0}",TEMPLATE_RUN1="${TEMPLATE_RUN1}",TEMPLATE_RUN2="${TEMPLATE_RUN2}",CONTAINER="${CLIENT_CONTAINER}",POLICY_MODEL_NAME="${POLICY_MODEL_NAME}",CURATOR_SCRIPT="${CURATOR_SCRIPT}",DAEMON_JOB_ID="${KIMI_JOB}" \
    "${SCRIPT_DIR}/rollout_phase.sbatch")
echo "Rollout client SLURM job: ${ROLLOUT_JOB}"
SUBMITTED_JOBS+=("${ROLLOUT_JOB}")

echo
echo "=== Phase 1: waiting for rollout client to finish ==="
while squeue -j "${ROLLOUT_JOB}" -h -o '%T' 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING'; do
    sleep 30
done
ROLLOUT_STATE=$(sacct -j "${ROLLOUT_JOB}" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
echo "Rollout client exit state: ${ROLLOUT_STATE}"

echo "=== Phase 1: scancelling Kimi daemon (free GPUs) ==="
scancel "${KIMI_JOB}" || true
sleep 5
sacct -j "${KIMI_JOB}" -X -n -o State 2>/dev/null | head -1 || true

if [ ! -f "${EXP_DIR}/.rollout_phase_done" ]; then
    echo "WARNING: ${EXP_DIR}/.rollout_phase_done missing — rollout phase did not complete cleanly"
    exit 1
fi

# -----------------------------------------------------------------------------
# Phase 2 — Judge daemon + judging client
# -----------------------------------------------------------------------------
echo
echo "=== Phase 2: launching gpt-oss-120b judge daemon ==="
JUDGE_JOB=$(sbatch --parsable \
    --job-name=judge_abc \
    --chdir="${EXP_DIR}/logs" \
    --export=ALL,MODEL_PATH="${JUDGE_MODEL_PATH}",SERVED_NAME="${JUDGE_MODEL_NAME}",PORT="${JUDGE_PORT}",ENDPOINT_FILE="${JUDGE_ENDPOINT_FILE}",CONTAINER="${JUDGE_CONTAINER}",LOG_DIR="${EXP_DIR}/logs",EXTRA_VLLM_ARGS="${JUDGE_EXTRA_VLLM_ARGS}" \
    "${SCRIPT_DIR}/vllm_daemon.sbatch")
echo "Judge daemon SLURM job: ${JUDGE_JOB}"
SUBMITTED_JOBS+=("${JUDGE_JOB}")

echo "=== Phase 2: launching judge-phase client ==="
JUDGE_CLIENT_JOB=$(sbatch --parsable \
    --job-name=abc_judging \
    --chdir="${EXP_DIR}/logs" \
    --export=ALL,JUDGE_ENDPOINT_FILE="${JUDGE_ENDPOINT_FILE}",EXP_DIR="${EXP_DIR}",JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME}",JUDGE_PROMPT_YAML="${JUDGE_PROMPT_YAML}",JUDGE_SCRIPT="${JUDGE_SCRIPT}",CONTAINER="${CLIENT_CONTAINER}",DAEMON_JOB_ID="${JUDGE_JOB}" \
    "${SCRIPT_DIR}/judge_phase.sbatch")
echo "Judge client SLURM job: ${JUDGE_CLIENT_JOB}"
SUBMITTED_JOBS+=("${JUDGE_CLIENT_JOB}")

echo
echo "=== Phase 2: waiting for judge client to finish ==="
while squeue -j "${JUDGE_CLIENT_JOB}" -h -o '%T' 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING'; do
    sleep 30
done
JUDGE_STATE=$(sacct -j "${JUDGE_CLIENT_JOB}" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
echo "Judge client exit state: ${JUDGE_STATE}"

echo "=== Phase 2: scancelling judge daemon (free GPUs) ==="
scancel "${JUDGE_JOB}" || true
sleep 5
sacct -j "${JUDGE_JOB}" -X -n -o State 2>/dev/null | head -1 || true

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo
echo "=== Summary ==="
for run in run0 run1 run2; do
    j="${EXP_DIR}/${run}/judged.jsonl"
    if [ -s "$j" ]; then
        python3 - <<PY
import json
n=0; ok=0
for line in open("$j"):
    e=json.loads(line); n+=1
    if (e.get("judge_verdict") or "").upper()=="YES":
        ok+=1
print(f"$run: {ok}/{n} ({100*ok/n if n else 0:.1f}%)")
PY
    else
        echo "$run: judged.jsonl missing"
    fi
done

echo
echo "Done.  EXP_DIR=${EXP_DIR}"
