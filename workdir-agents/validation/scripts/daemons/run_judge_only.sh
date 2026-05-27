#!/bin/bash
# Submit just Phase 2 (judge daemon + judge client) for an existing EXP_DIR
# that already has run0/rollouts.jsonl, run1/rollouts.jsonl, run2/rollouts.jsonl
# but never ran Phase 2 (e.g. Phase 1 hit walltime).  Mirrors the Phase 2
# block of launch_abc_smoke.sh.
set -euo pipefail
EXP_DIR="${1:?usage: run_judge_only.sh <EXP_DIR>}"
SCRIPT_DIR=/lustre/fsw/portfolios/nemotron/users/alaptev/abc_smoke/scripts
GYM_PATH=/lustre/fsw/portfolios/nemotron/users/alaptev/NeMo-Gym
JUDGE_ENDPOINT_FILE="${EXP_DIR}/daemons/judge.endpoint"
JUDGE_MODEL_PATH=/hf_models/gpt-oss-120b
JUDGE_MODEL_NAME=/hf_models/gpt-oss-120b
JUDGE_PORT=35042
JUDGE_CONTAINER=/lustre/fsw/portfolios/nemotron/users/alaptev/containers/vllm-glm51-cu130-ray.sqsh
JUDGE_EXTRA_VLLM_ARGS="--tensor-parallel-size 4 --max-model-len 131072"
CLIENT_CONTAINER=/lustre/fsw/portfolios/nemotron/users/igitman/images/nemo-skills-0a3c03f.sqsh
JUDGE_PROMPT_YAML="${GYM_PATH}/resources_servers/frontierscience_judge/prompts/judge.yaml"
JUDGE_SCRIPT="${SCRIPT_DIR}/judge_rollouts.py"
mkdir -p "${EXP_DIR}/daemons" "${EXP_DIR}/logs"
rm -f "${JUDGE_ENDPOINT_FILE}"

JUDGE_JOB=$(sbatch --parsable \
  --job-name=judge_abc \
  --chdir="${EXP_DIR}/logs" \
  --export=ALL,MODEL_PATH="${JUDGE_MODEL_PATH}",SERVED_NAME="${JUDGE_MODEL_NAME}",PORT="${JUDGE_PORT}",ENDPOINT_FILE="${JUDGE_ENDPOINT_FILE}",CONTAINER="${JUDGE_CONTAINER}",LOG_DIR="${EXP_DIR}/logs",EXTRA_VLLM_ARGS="${JUDGE_EXTRA_VLLM_ARGS}" \
  "${SCRIPT_DIR}/vllm_daemon.sbatch")
echo "JUDGE_JOB=${JUDGE_JOB}"

CLIENT_JOB=$(sbatch --parsable \
  --job-name=abc_judging \
  --chdir="${EXP_DIR}/logs" \
  --export=ALL,JUDGE_ENDPOINT_FILE="${JUDGE_ENDPOINT_FILE}",EXP_DIR="${EXP_DIR}",JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME}",JUDGE_PROMPT_YAML="${JUDGE_PROMPT_YAML}",JUDGE_SCRIPT="${JUDGE_SCRIPT}",CONTAINER="${CLIENT_CONTAINER}",DAEMON_JOB_ID="${JUDGE_JOB}" \
  "${SCRIPT_DIR}/judge_phase.sbatch")
echo "CLIENT_JOB=${CLIENT_JOB}"
