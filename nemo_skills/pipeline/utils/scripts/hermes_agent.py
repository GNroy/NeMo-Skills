# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pipeline scripts for ``ns hermes_agent_rollouts``.

Three job-script classes:

* ``HermesHomeBootstrapScript`` — rsync the bundled template HERMES_HOME
  into a per-agent job-local dir and render a config overlay (model URL,
  peer agent URLs, trace_dir, terminal backend).  Runs *before* the agent
  server in the same het-group so the directory exists by the time
  ``ng_run`` starts.

* ``HermesAgentHeadScript`` — run ``ng_run`` against a one-line config
  that hosts only this agent's ``hermes_agent``.  Each agent gets its
  own port, its own HERMES_HOME, and references its assigned LLM server
  (lazy hostname resolution against the het-group master).

* ``HermesHomeMergebackScript`` — invoke
  ``python -m nemo_skills.scripts.merge_hermes_home`` after the
  orchestrator's rollout collection finishes, copying the allowlisted
  paths back to the template and emitting an audit JSON.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nemo_skills.pipeline.utils.scripts.base import BaseJobScript
from nemo_skills.pipeline.utils.scripts.server import SandboxScript, ServerScript


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class HermesHomeBootstrapScript(BaseJobScript):
    """Materialize a per-agent HERMES_HOME snapshot from the template.

    The script:
      1. rsyncs ``<template>/`` → ``<agent_home>/``  (preserves mtimes;
         idempotent so a re-run picks up template updates safely).
      2. Renames ``env.example`` → ``.env`` (the template ships
         ``env.example`` because ``.env`` is gitignored).
      3. Writes ``<agent_home>/hermes_agent_overlay.json`` — a JSON file
         the ng_run config picks up via Hydra to override the bundled
         ``hermes_agent.yaml`` per agent.

    The overlay carries everything that varies per agent: peer URLs,
    LLM server URL, trace_dir, terminal backend, persistence flags.
    Keeping it as a JSON file (rather than building a long Hydra
    command-line) avoids quoting hell when peer URLs contain ``=``.

    Attributes:
        template_path: Path to the bundled hermes_home_template/ directory.
        agent_home: Target per-agent HERMES_HOME path (job-local).
        overlay: Dict that will be written verbatim as the overlay JSON.
        log_prefix: SLURM log filename prefix.
    """

    template_path: str
    agent_home: str
    overlay: Dict[str, object] = field(default_factory=dict)
    # Phase 4: when set, symlink ``<agent_home>/kanban.db`` to this path so
    # every agent in the fleet sees the same kanban board.  The dispatcher
    # het-group reads/writes the same file.  No-op when ``None``.
    shared_kanban_db: Optional[str] = None
    log_prefix: str = field(default="hermes_bootstrap", init=False)
    span_group_nodes: bool = False  # one-shot setup on the master node

    def __post_init__(self):
        # ``shlex.quote`` everywhere — paths may contain spaces when
        # users mount Lustre/NFS roots with non-trivial names.
        template_q = shlex.quote(self.template_path)
        agent_home_q = shlex.quote(self.agent_home)
        overlay_json = json.dumps(self.overlay, sort_keys=True)
        overlay_q = shlex.quote(overlay_json)

        if self.shared_kanban_db:
            kanban_q = shlex.quote(self.shared_kanban_db)
            kanban_snippet = (
                f'mkdir -p "$(dirname {kanban_q})"\n'
                f"touch {kanban_q}\n"
                f"ln -sf {kanban_q} {agent_home_q}/kanban.db\n"
            )
        else:
            kanban_snippet = ""

        cmd = f"""set -euo pipefail
echo "=== Hermes bootstrap: {self.agent_home} ==="
mkdir -p {agent_home_q}
# Trailing slash on source so we copy *contents* into the target.
rsync -a --delete --exclude='.git' {template_q}/ {agent_home_q}/

# env.example -> .env  (the template ships env.example to dodge .gitignore)
if [ -f {agent_home_q}/env.example ] && [ ! -f {agent_home_q}/.env ]; then
    mv {agent_home_q}/env.example {agent_home_q}/.env
fi

# Write the per-agent overlay JSON the HermesAgent reads at startup.
mkdir -p {agent_home_q}
printf '%s' {overlay_q} > {agent_home_q}/hermes_agent_overlay.json
{kanban_snippet}echo "Bootstrap done."
"""
        self.set_inline(cmd)
        super().__post_init__()


# ---------------------------------------------------------------------------
# Per-agent head server (ng_run hosting hermes_agent only)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class HermesAgentHeadScript(BaseJobScript):
    """Run ``ng_run`` for one Hermes agent inside a het-group.

    Spins up a NeMo-Gym head server hosting *only* this agent's
    ``hermes_agent``.  Other agents in the fleet have their own
    HermesAgentHeadScript instances (typically in other het-groups,
    or co-located in the same het-group for shared-model agents).

    The orchestrator additionally has a sibling ``NemoGymRolloutsScript``
    that calls ``ng_collect_rollouts`` against this server's endpoint —
    the head script just keeps the server alive.

    Attributes:
        config_paths: NG_run config_paths arg (list of relative YAML paths
            inside the mounted NeMo-Gym tree).
        agent_home: HERMES_HOME for this agent (already populated by
            HermesHomeBootstrapScript).
        agent_port: HTTP port the agent's FastAPI binds to.  ``CallAgentTool``
            on the orchestrator uses this for ``{host,port}`` resolution.
        agent_name: Logical name of the agent (matches manifest key).
        server: ServerScript handle for the LLM server in the same
            het-group (lazy hostname/port resolution).
        sandbox: SandboxScript handle for the sandbox in the same het-group.
        gym_path: Optional Gym checkout override (else uses container default).
        keep_alive: When True, ``ng_run`` runs in the foreground until SIGTERM
            (workers).  When False, the script returns once servers are healthy
            (orchestrator pairs this with a NemoGymRolloutsScript that does the
            collect-then-cleanup dance).
        log_prefix: SLURM log filename prefix.
    """

    config_paths: List[str]
    agent_home: str
    agent_port: int
    agent_name: str
    server: Optional[ServerScript] = None
    server_address: Optional[str] = None
    sandbox: Optional[SandboxScript] = None
    gym_path: Optional[str] = None
    policy_api_key: str = "dummy"  # pragma: allowlist secret
    policy_model_name: Optional[str] = None
    keep_alive: bool = True

    log_prefix: str = field(default="hermes_agent_head", init=False)
    span_group_nodes: bool = False  # agent process is single-node; LLM server spans

    def __post_init__(self):
        if self.server is not None and self.server_address is not None:
            raise ValueError("Specify only one of `server` or `server_address`.")

        config_paths_str = ",".join(self.config_paths)
        agent_home_q = shlex.quote(self.agent_home)

        def build_cmd() -> Tuple[str, Dict]:
            ng_run_parts = [
                "ng_run",
                f'"+config_paths=[{config_paths_str}]"',
                f"+hermes_agent.responses_api_agents.hermes_agent.port={self.agent_port}",
                f'+hermes_agent.responses_api_agents.hermes_agent.host="0.0.0.0"',
                f'+hermes_agent.responses_api_agents.hermes_agent.agent_name="{self.agent_name}"',
                f'+hermes_agent.responses_api_agents.hermes_agent.hermes_home="{self.agent_home}"',
            ]

            if self.server is not None:
                server_url = f"http://{self.server.hostname_ref()}:{self.server.port}/v1"
                ng_run_parts.append(f'+policy_base_url="{server_url}"')
            elif self.server_address is not None:
                ng_run_parts.append(f'+policy_base_url="{self.server_address}"')

            ng_run_parts.append(f'+policy_api_key="{self.policy_api_key}"')
            if self.policy_model_name:
                ng_run_parts.append(f'+policy_model_name="{self.policy_model_name}"')

            ng_run_cmd = " ".join(ng_run_parts)

            # Resolve a Gym checkout the same way nemo_gym.py does (so the
            # mounted dev tree is used when available).
            if self.gym_path is None:
                resolve_gym_path = """GYM_PATH=$(python - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("nemo_gym")
if spec is not None:
    for candidate in Path(spec.origin).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            print(candidate)
            raise SystemExit(0)

print("/opt/Gym")
PY
)"""
            else:
                resolve_gym_path = f"GYM_PATH={shlex.quote(str(self.gym_path))}"

            # The agent's own server will be at http://localhost:{agent_port};
            # we wait on that, not the LLM server (the LLM is awaited downstream
            # by ng_run's HeadServer when the hermes_agent first issues a call).
            if self.keep_alive:
                tail = f"""
echo "=== Worker {self.agent_name} keeping ng_run alive (foreground) ==="
wait $NG_RUN_PID
"""
            else:
                tail = f"""
echo "=== Orchestrator {self.agent_name} ng_run ready ==="
# Returns control to the surrounding script; the orchestrator's
# NemoGymRolloutsScript runs after this in the same CommandGroup.
# ng_run remains as PID $NG_RUN_PID for the duration of the job.
"""

            cmd = f"""set -e
set -o pipefail

echo "=== Installing NeMo Gym ==="
{resolve_gym_path}
echo "Using NeMo Gym path: $GYM_PATH"
cd "$GYM_PATH"
uv venv --python 3.12 --allow-existing .venv
source .venv/bin/activate
uv sync --active --extra dev
echo "NeMo Gym installed."

export HERMES_HOME={agent_home_q}
echo "HERMES_HOME=$HERMES_HOME"

set +o pipefail
echo "=== Starting Hermes agent {self.agent_name!r} on port {self.agent_port} ==="
{ng_run_cmd} &
NG_RUN_PID=$!
echo "ng_run PID: $NG_RUN_PID"

# Wait for *this agent's* HTTP endpoint to come up (not all servers — we
# only care that hermes_agent is reachable for peers / the orchestrator).
for i in $(seq 1 120); do
    if curl -fs "http://localhost:{self.agent_port}/global_config_dict_yaml" >/dev/null 2>&1; then
        echo "hermes_agent {self.agent_name!r} is responding."
        break
    fi
    if ! kill -0 $NG_RUN_PID 2>/dev/null; then
        echo "ERROR: ng_run died while we were waiting"
        wait $NG_RUN_PID 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

{tail}
"""
            env_vars = {"HERMES_HOME": self.agent_home}
            if self.sandbox is not None:
                env_vars["NEMO_SKILLS_SANDBOX_HOST"] = self.sandbox.hostname_ref()
                env_vars["NEMO_SKILLS_SANDBOX_PORT"] = str(self.sandbox.port)
            return cmd.strip(), {"environment": env_vars}

        self.set_inline(build_cmd)
        super().__post_init__()


# ---------------------------------------------------------------------------
# Merge-back (curator)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class HermesHomeMergebackScript(BaseJobScript):
    """Run the curator step that promotes per-run learnings back to the template.

    Wraps ``python -m nemo_skills.scripts.merge_hermes_home`` with the
    allowlist policy described in the integration plan: only ``MEMORY.md``,
    ``USER.md`` and ``skills/user/`` are copied.

    Attributes:
        sources: List of per-agent HERMES_HOME paths to harvest from.
        template_path: Template directory the writable paths get merged into.
        audit_path: Where to write the audit JSON
            (before/after sha256 per file, size delta).
        dry_run: When True the script enumerates the diff without writing.
    """

    sources: List[str]
    template_path: str
    audit_path: str
    dry_run: bool = False
    log_prefix: str = field(default="hermes_mergeback", init=False)
    span_group_nodes: bool = False

    def __post_init__(self):
        template_q = shlex.quote(self.template_path)
        audit_q = shlex.quote(self.audit_path)
        sources_q = " ".join(shlex.quote(s) for s in self.sources)
        dry_flag = "--dry-run" if self.dry_run else ""

        cmd = f"""set -euo pipefail
echo "=== Hermes merge-back ==="
mkdir -p "$(dirname {audit_q})"
python -m nemo_skills.scripts.merge_hermes_home \\
    --template {template_q} \\
    --audit {audit_q} \\
    {dry_flag} \\
    -- {sources_q}
echo "Merge-back done. Audit: {self.audit_path}"
"""
        self.set_inline(cmd)
        super().__post_init__()


# ---------------------------------------------------------------------------
# Phase 4 — Kanban dispatcher
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class HermesKanbanDispatcherScript(BaseJobScript):
    """Run the Hermes kanban dispatcher daemon in a het-group.

    Used in place of ``HermesAgentHeadScript`` for manifest entries with
    ``kind: dispatcher``.  The dispatcher has no LLM dependency — it
    just polls the shared kanban DB and routes new tasks to workers via
    Hermes's built-in kanban tools.  We launch it via the hermes-agent
    CLI so future changes to the daemon (locking, log fan-out, etc.)
    arrive without pipeline edits.

    Attributes:
        agent_home: HERMES_HOME for the dispatcher (already populated by
            HermesHomeBootstrapScript, including the kanban.db symlink
            when ``shared_kanban_db`` is set).
        agent_name: Logical name (used only for log prefix / readability).
        log_prefix: SLURM log filename prefix.
    """

    agent_home: str
    agent_name: str
    log_prefix: str = field(default="hermes_kanban_dispatcher", init=False)
    span_group_nodes: bool = False  # dispatcher is a single CPU process

    def __post_init__(self):
        agent_home_q = shlex.quote(self.agent_home)
        cmd = f"""set -euo pipefail
echo "=== Hermes kanban dispatcher: {self.agent_name} ==="
export HERMES_HOME={agent_home_q}

# Make sure the kanban plugin is enabled in this HERMES_HOME before the
# dispatcher tries to attach to its tables — enabling is idempotent.
hermes plugins enable kanban
exec hermes kanban dispatcher run
"""
        self.set_inline(cmd)
        super().__post_init__()
