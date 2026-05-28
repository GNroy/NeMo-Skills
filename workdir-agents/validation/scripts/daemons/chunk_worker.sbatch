#!/bin/bash
#SBATCH --account=nemotron_reason_science
#SBATCH --partition=cpu
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x_%j.log

# One pass × one chunk of the A/B/C rollout, with judging done in-loop
# by NeMo-Gym's frontierscience_judge resources server (no separate
# judge phase).  Replaces rollout_phase.sbatch + judge_phase.sbatch.
#
# Required env (set via --export when sbatch'ing):
#   POLICY_ENDPOINT_FILE   path the Kimi daemon writes its base URL to
#   JUDGE_ENDPOINT_FILE    path the gpt-oss-120b daemon writes its base URL to
#   EXP_DIR                top-level experiment dir
#   PASS_ID                0 | 1 | 2 (A/B/C pass index)
#   CHUNK_ID               this chunk's index in [0, NUM_CHUNKS)
#   NUM_CHUNKS             how many chunks per pass
#   INPUT_FILE             full input JSONL (worker slices its own chunk)
#   TEMPLATE_HOME          HERMES_HOME template to start from for this pass
#   PERSIST_MEMORY         "true" | "false" — persist_memory + persist_skills
#   SAVE_TRAJECTORIES      "true" | "false" — save_trajectories
#   POLICY_MODEL_NAME      what the policy daemon serves under --served-model-name
#   JUDGE_MODEL_NAME       what the judge daemon serves under --served-model-name
#   GYM_PATH               NeMo-Gym checkout to run from
#   HERMES_AGENT_PATH      hermes-agent checkout to prepend to PYTHONPATH
#   CONTAINER              client container (CPU-only OK)
#
# Optional env:
#   POLICY_DAEMON_JOB_ID   policy daemon job (for fast-fail if it dies)
#   JUDGE_DAEMON_JOB_ID    judge daemon job (for fast-fail)
#   HERMES_MAX_TURNS       per-rollout max turns (default 30)
#   HERMES_CONCURRENCY     in-flight rollouts per worker (default 25)

set -euo pipefail

: "${POLICY_ENDPOINT_FILE:?required}"
: "${JUDGE_ENDPOINT_FILE:?required}"
: "${EXP_DIR:?required}"
: "${PASS_ID:?required}"
: "${CHUNK_ID:?required}"
: "${NUM_CHUNKS:?required}"
: "${INPUT_FILE:?required}"
: "${TEMPLATE_HOME:?required}"
: "${PERSIST_MEMORY:?required}"
: "${SAVE_TRAJECTORIES:?required}"
: "${POLICY_MODEL_NAME:?required}"
: "${JUDGE_MODEL_NAME:?required}"
: "${GYM_PATH:?required}"
: "${HERMES_AGENT_PATH:?required}"
: "${CONTAINER:?required}"

# Identity-mount the lustre user dir so /lustre/... paths work inside
# the container.  Mirrors the rollout_phase.sbatch convention.
CONTAINER_MOUNTS="/lustre/fsw/portfolios/nemotron/users/igitman/hf_models:/hf_models,/lustre/fsw/portfolios/nemotron/users/alaptev:/alaptev,/lustre/fsw/portfolios/nemotron/users/alaptev:/lustre/fsw/portfolios/nemotron/users/alaptev,/lustre/fsw/portfolios/nemotron/users/alaptev/data:/data"

# Standard cluster env vars (formerly cluster_configs/aws-cmh.yaml).
export HF_HOME=/alaptev/hf-cache
export TRANSFORMERS_CACHE=/alaptev/hf-cache
export HUGGINGFACE_HUB_CACHE=/alaptev/hf-cache
export TIKTOKEN_ENCODINGS_BASE=/alaptev/tiktoken-encodings
export TIKTOKEN_RS_CACHE_DIR=/alaptev/tiktoken-encodings/cache
export HF_HUB_OFFLINE=1

INNER_CMD=$(cat <<'INNER_EOF'
set -euo pipefail

export CHUNK_DIR="${EXP_DIR}/pass${PASS_ID}/chunk${CHUNK_ID}"
export HERMES_HOME="${CHUNK_DIR}/hermes_home"
export CHUNK_INPUT="${CHUNK_DIR}/input.jsonl"
export ROLLOUTS_JSONL="${CHUNK_DIR}/rollouts.jsonl"
mkdir -p "${CHUNK_DIR}"

echo "[$(date -Is)] chunk_worker pass=${PASS_ID} chunk=${CHUNK_ID}/${NUM_CHUNKS}"

# 1. Wait for both daemons' endpoint files.
wait_endpoint () {
    local label="$1" file="$2" job_id_var="$3"
    echo "[$(date -Is)] waiting for ${label} endpoint at ${file}"
    local deadline=$(( $(date +%s) + 3600 ))
    until [ -s "$file" ]; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "ERROR: ${label} endpoint never appeared in 60 min"
            exit 1
        fi
        # Fast-fail if the daemon job left the queue (squeue is in the
        # client container only when running on a SLURM-aware base; gate
        # on availability).
        local jid="${!job_id_var:-}"
        if [ -n "$jid" ] && command -v squeue >/dev/null 2>&1; then
            local st
            st=$(squeue -j "$jid" -h -o '%T' 2>/dev/null || true)
            if [ -z "$st" ]; then
                echo "ERROR: ${label} daemon job $jid is no longer in the queue"
                sacct -j "$jid" -X -n -o State,ExitCode 2>/dev/null | head -1 || true
                exit 1
            fi
        fi
        sleep 10
    done
}
wait_endpoint policy "$POLICY_ENDPOINT_FILE" POLICY_DAEMON_JOB_ID
wait_endpoint judge  "$JUDGE_ENDPOINT_FILE"  JUDGE_DAEMON_JOB_ID
POLICY_URL=$(cat "$POLICY_ENDPOINT_FILE")
JUDGE_URL=$(cat "$JUDGE_ENDPOINT_FILE")
echo "[$(date -Is)] policy=${POLICY_URL} judge=${JUDGE_URL}"

# Route Hermes' auxiliary client (compression, session_search, ...) at
# the policy endpoint so it doesn't drop middle turns silently.
export OPENAI_BASE_URL="$POLICY_URL"
export OPENAI_API_KEY="gym"
# Bump Hermes per-API-call timeout so hard problems with long thinking
# don't trigger spurious retries.
export HERMES_API_TIMEOUT="${HERMES_API_TIMEOUT:-3600}"

# 2. Slice the input.  Stable hash-free split: row i goes to chunk
# (i % NUM_CHUNKS).  Deterministic across reruns of the same INPUT_FILE.
python3 - <<PY
import os
src = os.environ["INPUT_FILE"]
dst = os.environ["CHUNK_INPUT"]
cid = int(os.environ["CHUNK_ID"])
n   = int(os.environ["NUM_CHUNKS"])
kept = 0
with open(src) as fi, open(dst, "w") as fo:
    for i, line in enumerate(fi):
        if i % n == cid:
            fo.write(line)
            kept += 1
print(f"[chunk_input] {kept} rows -> {dst}")
PY

# 3. Bootstrap a per-chunk HERMES_HOME from the template.  Identical to
# rollout_phase.sbatch's bootstrap, just simplified for one pass.
POLICY_URL_FOR_BOOTSTRAP="$POLICY_URL" \
POLICY_MODEL_NAME_FOR_BOOTSTRAP="$POLICY_MODEL_NAME" \
python3 - <<PY
import json, os, shutil, yaml
from pathlib import Path

template = Path(os.environ["TEMPLATE_HOME"])
home = Path(os.environ["HERMES_HOME"])
persist_memory = os.environ["PERSIST_MEMORY"].lower() == "true"
save_trajectories = os.environ["SAVE_TRAJECTORIES"].lower() == "true"

if home.exists():
    shutil.rmtree(home)
shutil.copytree(template, home)

# Pre-stage Node.js so Hermes' install.sh skips the 50 MB download.
node_cache = Path("/alaptev/cache/hermes_node/node")
if node_cache.exists() and not (home / "node").exists():
    os.symlink(node_cache, home / "node")

# MCP servers — still pulled from the NS package as a library dep.
mcp = {"ns_python_tool": {"command": "python", "args": ["-m", "nemo_skills.mcp.servers.python_tool"]}}
cfg_path = home / "config.yaml"
cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
existing = (cfg.get("mcp_servers") or {})
existing.update(mcp)
cfg["mcp_servers"] = existing

# Auxiliary client routed at the policy endpoint (silences the
# "main provider unavailable" warning).
policy_url = os.environ.get("POLICY_URL_FOR_BOOTSTRAP", "").strip()
policy_model = os.environ.get("POLICY_MODEL_NAME_FOR_BOOTSTRAP", "").strip()
if policy_url:
    aux = cfg.get("auxiliary") or {}
    for task in ("compression", "session_search", "web_extract", "vision"):
        t = aux.get(task) or {}
        t["provider"] = "custom"
        t["base_url"] = policy_url
        t["api_key"] = "gym"
        if policy_model:
            t["model"] = policy_model
        aux[task] = t
    cfg["auxiliary"] = aux
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True))

# Per-pass overlay (HermesAgent._apply_overlay_file picks this up).
overlay = {
    "agent_name": "scientist",
    "max_turns": int(os.environ.get("HERMES_MAX_TURNS", "30")),
    "concurrency": int(os.environ.get("HERMES_CONCURRENCY", "25")),
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
    "chat_template_kwargs": {},
    "disable_streaming": True,
    "promote_reasoning_to_content": True,
}
(home / "hermes_agent_overlay.json").write_text(json.dumps(overlay, indent=2))
print(f"[bootstrap] {home} <- {template}; mcp={sorted(existing)}")
PY

# 4. Activate the pre-built Gym venv.  Setup (uv venv + uv sync) is done
# ONCE in the launcher before any chunks are submitted (see
# launch_chunked.sh) — re-running uv sync concurrently across chunks
# corrupts in-flight ng_run binaries on lustre and crashes random chunks
# with "bad interpreter: Bad address".
cd "$GYM_PATH"
if [ ! -x ".venv/bin/ng_run" ]; then
    echo "ERROR: $GYM_PATH/.venv missing or stale (no ng_run binary)."
    echo "       Run \`cd $GYM_PATH && uv sync --active --extra dev\` once"
    echo "       on the login node or via the launcher's setup step."
    exit 1
fi
source .venv/bin/activate

# 5. Stage hermes-agent on PYTHONPATH.
export PYTHONPATH="$HERMES_AGENT_PATH${PYTHONPATH:+:$PYTHONPATH}"

# 6. Launch ng_run with frontierscience_judge as the resources server.
# Verify happens in-loop; reward is in the rollouts JSONL.  No separate
# judge phase.
ng_run "+config_paths=[\
responses_api_agents/hermes_agent/configs/hermes_agent.yaml,\
resources_servers/frontierscience_judge/configs/frontierscience_judge.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml\
]" \
    "+policy_base_url=${POLICY_URL}" \
    "+policy_api_key=gym" \
    "+policy_model_name=${POLICY_MODEL_NAME}" \
    "+judge_base_url=${JUDGE_URL}" \
    "+judge_api_key=gym" \
    "+judge_model_name=${JUDGE_MODEL_NAME}" \
    "+agent_name=hermes_agent" \
    "+hermes_agent.responses_api_agents.hermes_agent.resources_server.name=frontierscience_judge" \
    "+hermes_agent.responses_api_agents.hermes_agent.hermes_home=${HERMES_HOME}" \
    > "${CHUNK_DIR}/ng_run.log" 2>&1 &
NG_PID=$!
echo "[$(date -Is)] ng_run PID ${NG_PID}"

# 7. Wait for ng_status all-healthy.
LAST=""
while true; do
    if ! kill -0 "$NG_PID" 2>/dev/null; then
        echo "ERROR: ng_run died before all-healthy"
        wait "$NG_PID" 2>/dev/null || true
        tail -100 "${CHUNK_DIR}/ng_run.log" || true
        exit 1
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

# 8. Collect rollouts on this chunk's slice.  ng_collect_rollouts already
# streams per-rollout to disk and supports resume_from_cache; if the job
# is requeued, completed rows survive.
ng_collect_rollouts \
    "+input_jsonl_fpath=${CHUNK_INPUT}" \
    "+output_jsonl_fpath=${ROLLOUTS_JSONL}" \
    "+agent_name=hermes_agent" \
    "+resume_from_cache=true" \
    2>&1 | tee -a "${CHUNK_DIR}/ng_collect_rollouts.log"

# 9. Tear down ng_run.
kill "$NG_PID" 2>/dev/null || true
wait "$NG_PID" 2>/dev/null || true

touch "${CHUNK_DIR}/.chunk_done"
echo "[$(date -Is)] chunk_worker done; rollouts at ${ROLLOUTS_JSONL}"
INNER_EOF
)

srun \
    --container-image "${CONTAINER}" \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --no-container-mount-home \
    --container-env=HF_HOME,TRANSFORMERS_CACHE,HUGGINGFACE_HUB_CACHE,TIKTOKEN_ENCODINGS_BASE,TIKTOKEN_RS_CACHE_DIR,HF_HUB_OFFLINE,POLICY_ENDPOINT_FILE,JUDGE_ENDPOINT_FILE,EXP_DIR,PASS_ID,CHUNK_ID,NUM_CHUNKS,INPUT_FILE,TEMPLATE_HOME,PERSIST_MEMORY,SAVE_TRAJECTORIES,POLICY_MODEL_NAME,JUDGE_MODEL_NAME,GYM_PATH,HERMES_AGENT_PATH,POLICY_DAEMON_JOB_ID,JUDGE_DAEMON_JOB_ID,HERMES_MAX_TURNS,HERMES_CONCURRENCY,HERMES_API_TIMEOUT,OPENAI_BASE_URL,OPENAI_API_KEY \
    --ntasks-per-node=1 \
    bash -c "${INNER_CMD}"
