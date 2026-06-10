# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``ns hermes_agent_debug`` — attach an INTERACTIVE Hermes CLI to a running
rollout's model endpoint, configured identically to the headless run (SCI-548).

Motivation: you monitor a headless ``hermes_agent_rollouts`` job and, when the
agent misbehaves (e.g. Kimi-K2.6 narrates ``delegate_task`` but never emits it,
or finishes early), you want to *take the wheel* — watch every tool call live
and steer mid-run ("issue the delegate_task now", "you skipped get_problem").

The headless gym ``responses()`` path has no human-steer channel, so we don't
hijack the live agent process. Instead we **reproduce its scenario** in the
hermes-agent interactive CLI, which already does what we need for free:
  * discovers MCP servers from ``<HERMES_HOME>/config.yaml`` natively,
  * renders ``tool_start`` / ``tool_complete`` live,
  * runs ``delegate_task``,
  * supports ``/steer <text>`` to inject guidance after the next tool call.

This command does the one thing the CLI doesn't: translate a rollout *manifest*
(+ an input row) into an equivalent interactive session. It materializes a
HERMES_HOME (the manifest's ``mcp_servers`` + ``delegation`` + provider stub),
extracts the orchestrator's system prompt + first task from the input JSONL, and
writes an ``attach.sh`` that points the CLI at the run's vLLM endpoint via
``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` (exactly how the pipeline wires the
endpoint).

Endpoint mode is ATTACH-TO-EXISTING: you pass ``--base_url`` of an
already-running server (the monitored run's vLLM, reachable from where you launch
``hermes`` — typically an ``srun --overlap --pty`` shell into the job's gym
container, or a tunnel).

Typical use::

    # On the cluster (where the vLLM endpoint is reachable):
    python -m nemo_skills.pipeline.hermes_agent_debug \\
        --agent_manifest batch_solve_kimi_manifest.yaml \\
        --input_file /alaptev/data/batch_solve_smoke.jsonl \\
        --base_url http://<server-host>:<port>/v1 \\
        --model /hf_models/Kimi-K2.6 \\
        --hermes_home /alaptev/exp/debug_kimi/hermes_home \\
        --launch

Without ``--launch`` it only materializes + prints the ready-to-run command, so
you can inspect/tweak the HERMES_HOME first. Drop ``--problem_id`` to attach an
empty session (type your own task), or set it to seed a specific row's task.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import yaml

from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.hermes_agent_rollouts import _extract_mcp_servers
from nemo_skills.pipeline.utils.hermes_manifest import HermesAgentSpec, parse_manifest
from nemo_skills.utils import setup_logging

import logging

LOG = logging.getLogger(__file__)


# ---------------------------------------------------------------------------
# Input-row prompt extraction
# ---------------------------------------------------------------------------


def _content_to_text(content: Any) -> str:
    """Flatten an OpenAI-style message ``content`` (str or list of parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                # {"type": "text"/"input_text", "text": ...} and friends.
                parts.append(str(part.get("text") or part.get("content") or ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def _row_messages(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    rcp = row.get("responses_create_params") or {}
    msgs = rcp.get("input")
    if isinstance(msgs, list):
        return [m for m in msgs if isinstance(m, dict)]
    return []


def _select_row(input_file: Path, problem_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the requested input row (by id) or the first row; None if empty."""
    if not input_file.exists():
        raise FileNotFoundError(f"input_file not found: {input_file}")
    first: Optional[Dict[str, Any]] = None
    with input_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if first is None:
                first = row
            if problem_id is None:
                return row
            # Match a problem id against common id fields used in our datasets.
            for key in ("id", "problem_id", "task_id", "uuid", "qid"):
                if str(row.get(key)) == str(problem_id):
                    return row
    if problem_id is not None and first is not None:
        LOG.warning("problem_id %r not found; falling back to the first row", problem_id)
    return first


def _split_prompt(row: Dict[str, Any]) -> tuple[str, str]:
    """Return (system_prompt, first_user_task) from an input row."""
    system_parts: List[str] = []
    user_parts: List[str] = []
    for m in _row_messages(row):
        role = m.get("role")
        text = _content_to_text(m.get("content"))
        if role == "system":
            system_parts.append(text)
        elif role == "user":
            user_parts.append(text)
    return "\n\n".join(p for p in system_parts if p), "\n\n".join(p for p in user_parts if p)


# ---------------------------------------------------------------------------
# HERMES_HOME materialization
# ---------------------------------------------------------------------------


def _materialize_hermes_home(
    *,
    template_path: Path,
    hermes_home: Path,
    agent: HermesAgentSpec,
    output_dir: str,
    system_prompt: str,
    overwrite: bool,
) -> Dict[str, Any]:
    """Copy the template HERMES_HOME and overlay the manifest's MCP/delegation.

    Returns the resolved config dict (for logging). Mirrors what
    ``HermesHomeBootstrapScript`` does headlessly, minus the gym overlay JSON
    (the interactive CLI reads config.yaml directly).
    """
    if hermes_home.exists():
        if not overwrite:
            raise FileExistsError(
                f"hermes_home already exists: {hermes_home} (pass --overwrite to replace)"
            )
        shutil.rmtree(hermes_home)
    shutil.copytree(template_path, hermes_home)

    config_path = hermes_home / "config.yaml"
    config: Dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            config = loaded

    # MCP servers: the manifest's block with {output_dir} substituted, exactly
    # as the headless pipeline does.  This is what the CLI discovers natively.
    mcp_servers = _extract_mcp_servers(agent, {"output_dir": output_dir})
    if mcp_servers:
        config["mcp_servers"] = mcp_servers

    # Delegation: manifest overrides win over the template defaults.
    manifest_delegation = agent.hermes.get("delegation")
    if isinstance(manifest_delegation, dict):
        config.setdefault("delegation", {})
        config["delegation"].update(manifest_delegation)

    # Keep the provider stub so the aux client resolves; the real endpoint is
    # injected via OPENAI_BASE_URL/OPENAI_API_KEY in attach.sh.
    config.setdefault("model", {}).setdefault("provider", "custom")

    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    if system_prompt:
        (hermes_home / "AGENTS.md").write_text(system_prompt, encoding="utf-8")
    return config


def _write_attach_script(
    *,
    hermes_home: Path,
    base_url: str,
    api_key: str,
    model: str,
    toolsets: List[str],
    task_path: Optional[Path],
) -> Path:
    """Write an ``attach.sh`` that exports the endpoint env and execs hermes."""
    toolsets_flag = ""
    if toolsets:
        toolsets_flag = f" -t {shlex.quote(','.join(toolsets))}"
    task_hint = ""
    if task_path is not None:
        task_hint = (
            f'echo "Seed task written to: {task_path}"\n'
            f'echo "Paste it as your first message (or type your own)."\n'
        )
    script = f"""#!/usr/bin/env bash
# Auto-generated by `ns hermes_agent_debug`. Attaches an interactive Hermes CLI
# to an already-running model endpoint, configured like the headless rollout.
# Run this where {base_url} is reachable (e.g. inside the job's gym container).
set -uo pipefail
export HERMES_HOME={shlex.quote(str(hermes_home))}
export OPENAI_BASE_URL={shlex.quote(base_url)}
export OPENAI_API_KEY={shlex.quote(api_key)}
{task_hint}echo "HERMES_HOME=$HERMES_HOME  endpoint=$OPENAI_BASE_URL  model={shlex.quote(model)}"
echo "Tip: while the agent runs, type  /steer <guidance>  to nudge it after the next tool call."
exec hermes --model {shlex.quote(model)}{toolsets_flag} "$@"
"""
    path = hermes_home / "attach.sh"
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def hermes_agent_debug(
    agent_manifest: str = typer.Option(..., help="Path to the YAML manifest (same one used for rollouts)."),
    base_url: str = typer.Option(
        ...,
        help="Endpoint of an ALREADY-RUNNING model server to attach to "
        "(e.g. the monitored rollout's vLLM): http://<host>:<port>/v1.",
    ),
    model: Optional[str] = typer.Option(
        None,
        help="Model name to send to the endpoint. Defaults to the chosen agent's manifest model.",
    ),
    hermes_home: str = typer.Option(
        ...,
        help="Where to materialize the interactive HERMES_HOME (use a path reachable "
        "from where you'll run hermes — e.g. a lustre dir for cluster attach).",
    ),
    agent: Optional[str] = typer.Option(
        None,
        help="Which manifest agent to reproduce. Defaults to the orchestrator.",
    ),
    input_file: Optional[str] = typer.Option(
        None,
        help="Input JSONL whose row seeds the first task + system prompt. Omit to attach an empty session.",
    ),
    problem_id: Optional[str] = typer.Option(
        None,
        help="Seed a specific row by id (id/problem_id/task_id/uuid/qid); default = first row.",
    ),
    output_dir: str = typer.Option(
        "/tmp/hermes_debug",
        help="Value substituted for {output_dir} in the manifest's mcp_servers env "
        "(e.g. worklog/trace paths). Point at the monitored run's output_dir to share its dirs.",
    ),
    api_key: str = typer.Option("dummy", help="API key for the endpoint (dummy for local vLLM)."),  # pragma: allowlist secret
    template_hermes_home: Optional[str] = typer.Option(
        None,
        help="Override the HERMES_HOME template. Defaults to the manifest's template_hermes_home.",
    ),
    overwrite: bool = typer.Option(False, help="Replace --hermes_home if it already exists."),
    launch: bool = typer.Option(
        False,
        help="Exec the interactive session now (requires hermes on PATH + base_url reachable here). "
        "Default: only materialize + print the ready-to-run command.",
    ),
):
    """Materialize an interactive Hermes debugging session from a rollout manifest.

    See the module docstring for the full workflow. Designed for the
    attach-to-existing-endpoint case: monitor a run, attach, drive/steer.
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)

    manifest = parse_manifest(Path(agent_manifest))
    spec = manifest.by_name.get(agent) if agent else manifest.orchestrator
    if spec is None:
        raise ValueError(f"agent {agent!r} not found in manifest (have: {list(manifest.by_name)})")

    effective_model = model or spec.model
    if not effective_model:
        raise ValueError(
            f"agent {spec.name!r} has no manifest model; pass --model explicitly."
        )

    template = Path(template_hermes_home or manifest.template_hermes_home or "")
    if not template or not template.exists():
        raise ValueError(
            f"HERMES_HOME template not found: {template!r}. Set --template_hermes_home "
            "or the manifest's template_hermes_home to a path valid where this runs."
        )

    home = Path(hermes_home)

    system_prompt, task_text = "", ""
    if input_file:
        row = _select_row(Path(input_file), problem_id)
        if row is not None:
            system_prompt, task_text = _split_prompt(row)

    config = _materialize_hermes_home(
        template_path=template,
        hermes_home=home,
        agent=spec,
        output_dir=output_dir,
        system_prompt=system_prompt,
        overwrite=overwrite,
    )

    task_path: Optional[Path] = None
    if task_text:
        task_path = home / "debug_task.md"
        task_path.write_text(task_text, encoding="utf-8")

    toolsets = list(spec.hermes.get("enabled_toolsets") or [])
    attach = _write_attach_script(
        hermes_home=home,
        base_url=base_url,
        api_key=api_key,
        model=effective_model,
        toolsets=toolsets,
        task_path=task_path,
    )

    LOG.info("Materialized interactive HERMES_HOME at %s", home)
    LOG.info("  endpoint : %s", base_url)
    LOG.info("  model    : %s", effective_model)
    LOG.info("  toolsets : %s", ", ".join(toolsets) or "(config default)")
    LOG.info("  mcp      : %s", ", ".join((config.get("mcp_servers") or {}).keys()) or "(none)")
    if task_path:
        LOG.info("  seed task: %s", task_path)
    print()
    print("To attach an interactive, steerable Hermes session, run (where the endpoint is reachable):")
    print(f"    {attach}")
    print("Then type the task as your first message and watch the tool calls. Use  /steer <guidance>  mid-run.")

    if launch:
        if shutil.which("hermes") is None:
            LOG.error("--launch given but `hermes` is not on PATH here; run %s on a host that has it.", attach)
            raise typer.Exit(code=1)
        LOG.info("Launching interactive session...")
        os.execv("/bin/bash", ["/bin/bash", str(attach)])

    return str(attach)


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
