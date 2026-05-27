#!/bin/bash
#SBATCH --account=nemotron_reason_science
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x_%j.log

# A/B/C rollout phase using a long-lived Kimi vLLM daemon.
#
# Required env (set via --export when sbatch'ing):
#   POLICY_ENDPOINT_FILE   path the Kimi daemon writes its `http://...:port/v1` to
#   EXP_DIR                top-level experiment dir (run0/, run1/, run2/ created under it)
#   INPUT_FILE             input JSONL (same for all 3 passes)
#   GYM_PATH               NeMo-Gym checkout to run from
#   HERMES_AGENT_PATH      hermes-agent checkout to prepend to PYTHONPATH
#   TEMPLATE_RUN0          HERMES_HOME template for Run 0 (cold)
#   TEMPLATE_RUN1          HERMES_HOME template for Run 1 (learning, mutated by mergeback)
#   TEMPLATE_RUN2          HERMES_HOME template for Run 2 (= staged copy of post-mergeback TEMPLATE_RUN1)
#   CONTAINER              client container (nemo-skills-0a3c03f.sqsh)
#   POLICY_MODEL_NAME      what the daemon serves under --served-model-name (e.g. /hf_models/Kimi-K2.6)
#   CURATOR_SCRIPT         path to curator.py (standalone merge_hermes_home; bundled in daemons/)
#
# Passthrough verifier — judging happens in the separate judge_phase job
# so we don't keep Kimi GPUs and judge GPUs co-resident.

set -euo pipefail

: "${POLICY_ENDPOINT_FILE:?required}"
: "${EXP_DIR:?required}"
: "${INPUT_FILE:?required}"
: "${GYM_PATH:?required}"
: "${HERMES_AGENT_PATH:?required}"
: "${TEMPLATE_RUN0:?required}"
: "${TEMPLATE_RUN1:?required}"
: "${TEMPLATE_RUN2:?required}"
: "${CONTAINER:?required}"
: "${POLICY_MODEL_NAME:?required}"
: "${CURATOR_SCRIPT:?required}"

# Identity-mount the lustre user dir so /lustre/... paths work inside
# the container too — the launcher uses lustre paths everywhere (the
# /alaptev alias doesn't exist on the login node).
CONTAINER_MOUNTS="/lustre/fsw/portfolios/nemotron/users/igitman/hf_models:/hf_models,/lustre/fsw/portfolios/nemotron/users/alaptev:/alaptev,/lustre/fsw/portfolios/nemotron/users/alaptev:/lustre/fsw/portfolios/nemotron/users/alaptev,/lustre/fsw/portfolios/nemotron/users/alaptev/data:/data"

# Standard cluster env_vars (see cluster_configs/aws-cmh.yaml).  Set
# explicitly because raw sbatch doesn't pick them up from NS pipeline.
export HF_HOME=/alaptev/hf-cache
export TRANSFORMERS_CACHE=/alaptev/hf-cache
export HUGGINGFACE_HUB_CACHE=/alaptev/hf-cache
export TIKTOKEN_ENCODINGS_BASE=/alaptev/tiktoken-encodings
export TIKTOKEN_RS_CACHE_DIR=/alaptev/tiktoken-encodings/cache
export HF_HUB_OFFLINE=1

INNER_CMD=$(cat <<'INNER_EOF'
set -euo pipefail

echo "[$(date -Is)] rollout phase booting"

# 1. Wait for the policy daemon to write its endpoint.
echo "[$(date -Is)] waiting for policy endpoint at $POLICY_ENDPOINT_FILE"
WAIT_DEADLINE=$(( $(date +%s) + 3600 ))  # 60 min ceiling
until [ -s "$POLICY_ENDPOINT_FILE" ]; do
    if [ "$(date +%s)" -ge "$WAIT_DEADLINE" ]; then
        echo "ERROR: policy endpoint never appeared in 60 min"
        exit 1
    fi
    # Daemon-died fast fail — only when squeue is on PATH (container
    # doesn't ship SLURM CLI; see judge_phase.sbatch for the full reason).
    if [ -n "${DAEMON_JOB_ID:-}" ] && command -v squeue >/dev/null 2>&1; then
        st=$(squeue -j "$DAEMON_JOB_ID" -h -o '%T' 2>/dev/null || true)
        if [ -z "$st" ]; then
            echo "ERROR: policy daemon job $DAEMON_JOB_ID is no longer in the queue"
            sacct -j "$DAEMON_JOB_ID" -X -n -o State,ExitCode 2>/dev/null | head -1 || true
            exit 1
        fi
    fi
    sleep 10
done
POLICY_URL=$(cat "$POLICY_ENDPOINT_FILE")
echo "[$(date -Is)] policy endpoint: $POLICY_URL"

# 2. Install / sync the Gym venv (idempotent fast path).
cd "$GYM_PATH"
uv venv --python 3.12 --allow-existing .venv
source .venv/bin/activate
uv sync --active --extra dev

# 3. Stage hermes-agent on PYTHONPATH (avoids the openai version conflict).
export PYTHONPATH="$HERMES_AGENT_PATH${PYTHONPATH:+:$PYTHONPATH}"
echo "[$(date -Is)] PYTHONPATH=$PYTHONPATH"

# Helper: bootstrap a HERMES_HOME from a template, merge mcp_servers into
# config.yaml, and drop a hermes_agent_overlay.json with the per-pass
# HermesAgentConfig overrides (read by app.py's _apply_overlay_file at
# server-init time — cleaner than threading lists/dicts through Hydra +=).
bootstrap_home () {
    local template="$1"
    local home="$2"
    local persist_memory="$3"
    local save_trajectories="$4"
    python3 - "$template" "$home" "$persist_memory" "$save_trajectories" <<'PY'
import json, shutil, sys, yaml
from pathlib import Path

template_arg, home_arg, persist_arg, save_arg = sys.argv[1:5]
persist_memory = persist_arg.lower() == "true"
save_trajectories = save_arg.lower() == "true"

template = Path(template_arg)
home = Path(home_arg)
if home.exists():
    shutil.rmtree(home)
shutil.copytree(template, home)

# Manifest-supplied MCP servers — merged into Hermes's config.yaml.
mcp = {"ns_python_tool": {"command": "python", "args": ["-m", "nemo_skills.mcp.servers.python_tool"]}}
cfg_path = home / "config.yaml"
cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
existing = (cfg.get("mcp_servers") or {})
existing.update(mcp)
cfg["mcp_servers"] = existing
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True))

# Per-pass overlay (HermesAgent._apply_overlay_file picks this up at
# server init from <hermes_home>/hermes_agent_overlay.json).
#
# max_turns capped low (30) for the 100-problem run so a single rollout
# hitting the cap can't blow past the 4h walltime; concurrency bumped to
# 64 to drain the batch in fewer rounds.  Both can be overridden via
# HERMES_MAX_TURNS / HERMES_CONCURRENCY env vars without re-editing.
import os
overlay = {
    "agent_name": "scientist",
    "max_turns": int(os.environ.get("HERMES_MAX_TURNS", "30")),
    "concurrency": int(os.environ.get("HERMES_CONCURRENCY", "64")),
    "enabled_toolsets": [
        "terminal", "file", "code_execution", "web",
        "skills", "memory", "browser",
    ],
    "disabled_toolsets": ["send_message", "cronjob", "kanban", "delegate"],
    "persist_memory": persist_memory,
    "persist_skills": persist_memory,
    "persist_session": False,
    "save_trajectories": save_trajectories,
    "enable_compression": False,
    # Leave thinking ON for K2.6 — needed for difficult multi-tool agentic
    # work.  K2.6 + --reasoning-parser kimi_k2 routes the full visible reply
    # (chain-of-thought ending with the final answer) through ``reasoning``
    # and leaves ``content=""``; ``promote_reasoning_to_content`` below
    # copies that into content so Hermes' empty-content / thinking-prefill
    # retries don't fire and the LLM judge has populated assistant text to
    # grade.  Tool extraction is handled by our custom kimi_k26 tool-call
    # parser (loaded via --tool-parser-plugin) regardless of thinking state.
    "chat_template_kwargs": {},
    "disable_streaming": True,
    "promote_reasoning_to_content": True,
}
(home / "hermes_agent_overlay.json").write_text(json.dumps(overlay, indent=2))
print(f"bootstrapped {home} from {template}; mcp={sorted(existing)}; overlay_keys={list(overlay)}")
PY
}

# Helper: run one A/B/C pass.
run_pass () {
    local pass_name="$1"
    local hermes_home="$2"
    local persist_memory="$3"
    local save_trajectories="$4"
    local output_dir="$5"

    local rollouts_jsonl="${output_dir}/rollouts.jsonl"
    mkdir -p "$output_dir"

    echo "[$(date -Is)] === pass ${pass_name}: ng_run + ng_collect_rollouts ==="

    # Launch ng_run in background.  Compose three Gym configs: hermes_agent
    # (the orchestrator), passthrough (no-op verifier — real judging comes
    # later via the judge phase), vllm_model (lets us override
    # policy_base_url).
    # Hydra overrides cover (a) cross-config interpolation targets
    # (policy_base_url etc.) and (b) the few HermesAgentConfig fields
    # that must be set BEFORE the overlay merge — hermes_home itself
    # (the overlay's location) and the resources_server.name link.
    # All other per-pass settings live in the overlay JSON written by
    # bootstrap_home, so the ng_run line stays terse.
    ng_run "+config_paths=[\
responses_api_agents/hermes_agent/configs/hermes_agent.yaml,\
resources_servers/passthrough/configs/passthrough.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml\
]" \
        "+policy_base_url=${POLICY_URL}" \
        "+policy_api_key=dummy" \
        "+policy_model_name=${POLICY_MODEL_NAME}" \
        "+agent_name=hermes_agent" \
        "+hermes_agent.responses_api_agents.hermes_agent.resources_server.name=passthrough" \
        "+hermes_agent.responses_api_agents.hermes_agent.hermes_home=${hermes_home}" \
        > "${output_dir}/ng_run.log" 2>&1 &
    NG_PID=$!
    echo "[$(date -Is)] ng_run PID ${NG_PID}"

    # Wait for ng_status all-healthy.
    LAST=""
    while true; do
        if ! kill -0 $NG_PID 2>/dev/null; then
            echo "ERROR: ng_run died before all-healthy"
            wait $NG_PID 2>/dev/null
            tail -100 "${output_dir}/ng_run.log" || true
            return 1
        fi
        STATUS=$(ng_status 2>&1 || true)
        if echo "$STATUS" | grep -q "healthy, 0 unhealthy"; then
            echo "[$(date -Is)] ng_run all-healthy"
            break
        fi
        CURR=$(echo "$STATUS" | grep -oE '[0-9]+ healthy' | head -1 || echo starting)
        if [ "$CURR" != "$LAST" ]; then
            echo "[$(date -Is)] status: $CURR"
            LAST="$CURR"
        fi
        sleep 10
    done

    # Collect rollouts.
    ng_collect_rollouts \
        "+input_jsonl_fpath=${INPUT_FILE}" \
        "+output_jsonl_fpath=${rollouts_jsonl}" \
        "+agent_name=hermes_agent" \
        2>&1 | tee -a "${output_dir}/ng_collect_rollouts.log"

    # Tear down ng_run for this pass — next pass needs to start one with
    # a different HERMES_HOME / persist_memory.
    kill $NG_PID 2>/dev/null || true
    wait $NG_PID 2>/dev/null || true
    echo "[$(date -Is)] pass ${pass_name} done; rollouts at ${rollouts_jsonl}"
}

# ---- Pass 0: cold baseline, no memory writes, no merge-back ----
HERMES_HOME_0="${EXP_DIR}/run0/agents/scientist/hermes_home"
bootstrap_home "$TEMPLATE_RUN0" "$HERMES_HOME_0" false false
run_pass run0 "$HERMES_HOME_0" false false "${EXP_DIR}/run0"

# ---- Pass 1: learning, memory writes ON, merge-back ON ----
HERMES_HOME_1="${EXP_DIR}/run1/agents/scientist/hermes_home"
bootstrap_home "$TEMPLATE_RUN1" "$HERMES_HOME_1" true true
run_pass run1 "$HERMES_HOME_1" true true "${EXP_DIR}/run1"

# ---- Mergeback for Run 1 → mutates TEMPLATE_RUN1 in place ----
# Uses the standalone curator.py shipped alongside this script; the
# Gym venv doesn't have nemo_skills importable so we invoke by path.
echo "[$(date -Is)] === mergeback: ${HERMES_HOME_1} → ${TEMPLATE_RUN1} ==="
python3 "$CURATOR_SCRIPT" \
    --template "$TEMPLATE_RUN1" \
    --audit "${EXP_DIR}/run1/merge_audit.json" \
    -- "$HERMES_HOME_1"

# ---- Stage Run 2 template: snapshot the mutated Run 1 template ----
echo "[$(date -Is)] === stage TEMPLATE_RUN2: ${TEMPLATE_RUN1} → ${TEMPLATE_RUN2} ==="
python3 - <<PY
import shutil
from pathlib import Path
src, dst = Path("$TEMPLATE_RUN1"), Path("$TEMPLATE_RUN2")
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f"staged {dst}")
PY

# ---- Pass 2: evaluation against mutated template, no memory writes ----
HERMES_HOME_2="${EXP_DIR}/run2/agents/scientist/hermes_home"
bootstrap_home "$TEMPLATE_RUN2" "$HERMES_HOME_2" false false
run_pass run2 "$HERMES_HOME_2" false false "${EXP_DIR}/run2"

echo "[$(date -Is)] rollout phase complete; all three rollouts.jsonl present"
touch "${EXP_DIR}/.rollout_phase_done"
INNER_EOF
)

srun \
    --container-image "${CONTAINER}" \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --no-container-mount-home \
    --container-env=HF_HOME,TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE,TIKTOKEN_ENCODINGS_BASE,TIKTOKEN_RS_CACHE_DIR,HF_HUB_OFFLINE,POLICY_ENDPOINT_FILE,EXP_DIR,INPUT_FILE,GYM_PATH,HERMES_AGENT_PATH,TEMPLATE_RUN0,TEMPLATE_RUN1,TEMPLATE_RUN2,POLICY_MODEL_NAME,CURATOR_SCRIPT,DAEMON_JOB_ID,HERMES_MAX_TURNS,HERMES_CONCURRENCY,HERMES_DEBUG_MESSAGES \
    --ntasks-per-node=1 \
    bash -c "${INNER_CMD}"
