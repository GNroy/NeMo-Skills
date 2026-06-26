#!/bin/bash
# Launch ONE BF16 agentic driver (seed k) on the CPU partition, routing to the
# shared lazypool_bf16 router + sandbox. Resume is ON (sidecar wrapper sets
# NS_BATCH_RESUME=1) so a restart skips already-completed ids. Reused by the
# supervisor for resume-restarts.
#   launch_bf16_driver.sh <seed_k> <qos> [variant=tool|notool]
# variant=tool   : python tool enabled (mcp-python), tool-mode goal prompt
# variant=notool : python disabled, pure-reasoning goal prompt (apples-to-apples arm)
set -uo pipefail
K="${1:?seed index}"; QOS="${2:?qos (cpu-short|cpu-normal)}"; VARIANT="${3:-tool}"
NS_DIR=/home/alaptev/Projects/NeMo-Skills
IG=/lustre/fsw/portfolios/nemotron/users/igitman/images
A=/lustre/fsw/portfolios/nemotron/users/alaptev

if [ "$VARIANT" = "notool" ]; then
  MANIFEST=workdir-agents/validation/batch_solve_a0py_bf16_notool_manifest.yaml
  INPUT=/alaptev/data/batch_solve_a0py_bf16_notool_input.jsonl
  SIDECAR=/alaptev/router_pool/gym_sidecars_bf16nt_s${K}.sh
  OUTDIR=/alaptev/exp/a0py_bf16nt_s${K}
  EXPNAME=lazyalloc_bf16nt_s${K}
else
  MANIFEST=workdir-agents/validation/batch_solve_a0py_bf16_manifest.yaml
  INPUT=/alaptev/data/batch_solve_a0py_lazy_full_input.jsonl
  SIDECAR=/alaptev/router_pool/gym_sidecars_bf16_s${K}.sh
  OUTDIR=/alaptev/exp/a0py_bf16_s${K}
  EXPNAME=lazyalloc_bf16_s${K}
fi

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nemo-skills
cd "$NS_DIR"
NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 \
NS_GYM_SIDECAR_SCRIPT="$SIDECAR" \
ns hermes_agent_rollouts --cluster aws-cmh \
  --agent_manifest "$MANIFEST" \
  --input_file "$INPUT" \
  --output_dir "$OUTDIR" \
  --expname "$EXPNAME" \
  --no-merge_back --no-with_sandbox \
  --qos "${QOS}" \
  --server_container ${A}/containers/sglang-v0.5.11.sqsh \
  --gym_container ${IG}/nemo-skills-dc43f3e.sqsh \
  --sandbox_container ${IG}/nemo-skills-sandbox-dc43f3e.sqsh \
  --hermes_agent_path /alaptev/hermes-agent \
  --gym_path /alaptev/NeMo-Gym
