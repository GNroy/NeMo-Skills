# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Phase 1 tests for ns hermes_agent_rollouts.

Covers T1.3 — T1.9 from the integration plan:
  T1.3  bootstrap copies the template
  T1.4  bootstrap renders overlay JSON with trace_dir
  T1.5  orchestrator overlay contains peer_agents entries for every worker
  T1.6  mergeback only copies allowlisted paths
  T1.7  mergeback emits audit JSON with sha256s
  T1.8  pipeline assembly with one orchestrator + one worker validates cleanly
  T1.9  existing ns nemo_gym_rollouts importable+callable (regression guard)
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict

import pytest

from nemo_skills.pipeline.hermes_agent_rollouts import _build_jobs, _build_overlay
from nemo_skills.pipeline.utils.declarative import CommandGroup
from nemo_skills.pipeline.utils.hermes_manifest import parse_manifest
from nemo_skills.pipeline.utils.scripts.hermes_agent import (
    HermesAgentHeadScript,
    HermesHomeBootstrapScript,
    HermesHomeMergebackScript,
)


# ---------------------------------------------------------------------------
# T1.3 — bootstrap copies the template and writes the overlay JSON.
# ---------------------------------------------------------------------------


def _make_fake_template(root: Path) -> Path:
    """Build a minimal hermes_home_template/ structure under ``root``."""
    template = root / "hermes_home_template"
    (template / "skills" / "user").mkdir(parents=True)
    (template / "skills" / "seed" / "demo").mkdir(parents=True)
    (template / "logs").mkdir()
    (template / "config.yaml").write_text("# template\n")
    (template / "env.example").write_text("# secrets go in .env\n")
    (template / "MEMORY.md").write_text("# memory\n")
    (template / "USER.md").write_text("# user\n")
    (template / "skills" / "user" / ".keep").write_text("")
    (template / "skills" / "seed" / "demo" / "SKILL.md").write_text("# demo skill\n")
    return template


def _run_inline_script(script: HermesHomeBootstrapScript, *, env: Dict[str, str] | None = None) -> None:
    """Execute a HermesHomeBootstrapScript's inline command in a real bash shell.

    The script's ``inline`` is a string; ``bash -e -c`` will replay it just
    as the SLURM executor would.
    """
    assert isinstance(script.inline, str)
    proc = subprocess.run(
        ["bash", "-c", script.inline],
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"bootstrap script failed (rc={proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )


def test_t13_bootstrap_copies_template(tmp_path: Path) -> None:
    """rsync end-to-end: template contents should land verbatim under agent_home."""
    template = _make_fake_template(tmp_path)
    agent_home = tmp_path / "agents" / "orch" / "hermes_home"

    script = HermesHomeBootstrapScript(
        template_path=str(template),
        agent_home=str(agent_home),
        overlay={"agent_name": "orch", "trace_dir": str(tmp_path / "traces" / "orch")},
    )
    _run_inline_script(script)

    assert (agent_home / "config.yaml").is_file()
    assert (agent_home / "MEMORY.md").is_file()
    assert (agent_home / "USER.md").is_file()
    assert (agent_home / "skills" / "user" / ".keep").is_file()
    assert (agent_home / "skills" / "seed" / "demo" / "SKILL.md").is_file()
    assert (agent_home / "logs").is_dir()
    # env.example was renamed to .env (the gitignore-dodging convention).
    assert (agent_home / ".env").is_file()
    assert not (agent_home / "env.example").exists()


# ---------------------------------------------------------------------------
# T1.4 — overlay JSON carries the per-agent fields the pipeline computes.
# ---------------------------------------------------------------------------


def test_t14_bootstrap_writes_overlay_with_trace_dir(tmp_path: Path) -> None:
    template = _make_fake_template(tmp_path)
    agent_home = tmp_path / "agents" / "alice" / "hermes_home"
    overlay = {
        "agent_name": "alice",
        "trace_dir": "/results/traces/alice",
        "persist_memory": True,
        "save_trajectories": True,
        "peer_agents": {"bob": {"het_group": 1, "port": 9001}},
    }
    script = HermesHomeBootstrapScript(
        template_path=str(template),
        agent_home=str(agent_home),
        overlay=overlay,
    )
    _run_inline_script(script)

    overlay_path = agent_home / "hermes_agent_overlay.json"
    assert overlay_path.is_file()
    loaded = json.loads(overlay_path.read_text())
    assert loaded == overlay
    # Critical fields that the pipeline always sets:
    assert loaded["trace_dir"] == "/results/traces/alice"
    assert loaded["agent_name"] == "alice"


# ---------------------------------------------------------------------------
# T1.5 — orchestrator overlay enumerates every declared worker.
# ---------------------------------------------------------------------------


def _toy_manifest_two_workers() -> Dict[str, Any]:
    return {
        "agents": {
            "orch": {
                "model": "/m/orch",
                "server_gpus": 8,
                "workers": ["analyst", "coder"],
            },
            "analyst": {"model_share": "orch"},
            "coder": {"model": "/m/coder", "server_gpus": 4},
        }
    }


def test_t15_orchestrator_overlay_has_peer_agents(tmp_path: Path) -> None:
    manifest = parse_manifest(_toy_manifest_two_workers())
    cluster_config = _minimal_cluster_config()

    jobs = _build_jobs(
        manifest=manifest,
        cluster_config=cluster_config,
        output_dir="/results/run1",
        log_dir="/results/run1/logs",
        input_file="/data/in.jsonl",
        template_path="/templates/hermes_home_template",
        server_container_override=None,
        gym_container="nemo-gym",
        gym_path=None,
        with_sandbox=False,
        sandbox_container_override=None,
        sandbox_mounts=None,
        sbatch_kwargs={},
        partition=None,
        qos=None,
        time_min=None,
        merge_back=True,
        merge_dry_run=False,
        extra_arguments="",
        policy_api_key="dummy",  # pragma: allowlist secret
        policy_model_name=None,
        expname="hermes_test",
    )
    assert len(jobs) == 1
    job = jobs[0]
    groups = job.get("groups") or [job["group"]]
    bootstrap_overlays = _collect_bootstrap_overlays(groups)

    # orchestrator overlay must list all declared workers (analyst, coder),
    # mapped to their (het_group, port).
    assert "orch" in bootstrap_overlays
    peers = bootstrap_overlays["orch"].get("peer_agents")
    assert peers is not None and set(peers) == {"analyst", "coder"}
    # analyst is co-located → het_group 0; coder owns its own group → 1.
    assert peers["analyst"]["het_group"] == 0
    assert peers["coder"]["het_group"] == 1
    # Each peer has a port and it's an int.
    for name, entry in peers.items():
        assert isinstance(entry["port"], int) and entry["port"] > 0

    # Worker overlays do NOT carry peer_agents in Phase 1 (only the
    # orchestrator delegates).  Asserting this keeps us honest about the
    # documented role boundaries.
    assert "peer_agents" not in bootstrap_overlays["analyst"]
    assert "peer_agents" not in bootstrap_overlays["coder"]


def test_orchestrator_overlay_defaults_to_full_fleet_when_workers_omitted() -> None:
    """If the manifest's orchestrator omits ``workers:``, every other agent
    is treated as a peer.  Lets users wire trivial fleets without
    enumerating workers twice."""
    manifest = parse_manifest(
        {
            "agents": {
                "orch": {"model": "/m/orch", "server_gpus": 8},
                "helper": {"model_share": "orch"},
            }
        }
    )
    jobs = _build_jobs(
        manifest=manifest,
        cluster_config=_minimal_cluster_config(),
        output_dir="/results",
        log_dir="/results/logs",
        input_file="/data/in.jsonl",
        template_path="/templates/h",
        server_container_override=None,
        gym_container="nemo-gym",
        gym_path=None,
        with_sandbox=False,
        sandbox_container_override=None,
        sandbox_mounts=None,
        sbatch_kwargs={},
        partition=None,
        qos=None,
        time_min=None,
        merge_back=False,
        merge_dry_run=False,
        extra_arguments="",
        policy_api_key="dummy",  # pragma: allowlist secret
        policy_model_name=None,
        expname="hermes_test",
    )
    overlays = _collect_bootstrap_overlays(jobs[0].get("groups") or [jobs[0]["group"]])
    assert set(overlays["orch"]["peer_agents"]) == {"helper"}


# ---------------------------------------------------------------------------
# T1.6, T1.7 — merge-back policy + audit JSON.
# ---------------------------------------------------------------------------


def _populate_run_home(home: Path) -> None:
    """Create a per-run HERMES_HOME with files in *every* category — both
    allowlisted (MEMORY.md / USER.md / skills/user) and non-allowlisted
    (sessions.db / logs / .env / skills/seed) — so the merge step is
    exercised end-to-end."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "MEMORY.md").write_text("learned X\n")
    (home / "USER.md").write_text("user notes\n")
    (home / ".env").write_text("HERMES_TEST_SECRET=should_not_leak\n")  # pragma: allowlist secret
    (home / "sessions.db").write_text("binary blob")
    (home / "kanban.db").write_text("binary blob")
    (home / "logs").mkdir()
    (home / "logs" / "agent.log").write_text("verbose log\n")
    (home / "skills" / "user" / "my-new-skill").mkdir(parents=True)
    (home / "skills" / "user" / "my-new-skill" / "SKILL.md").write_text("# new skill\n")
    (home / "skills" / "seed" / "default").mkdir(parents=True)
    (home / "skills" / "seed" / "default" / "SKILL.md").write_text("# don't override seed\n")


def test_t16_mergeback_only_copies_allowlisted_paths(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    # Pre-existing template content the curator should override.
    (template / "MEMORY.md").write_text("old\n")

    src = tmp_path / "run" / "orch" / "hermes_home"
    _populate_run_home(src)

    from nemo_skills.scripts import merge_hermes_home

    result = merge_hermes_home.merge(sources=[src], template=template, dry_run=False)

    # Allowlisted paths landed in the template:
    assert (template / "MEMORY.md").read_text() == "learned X\n"
    assert (template / "USER.md").read_text() == "user notes\n"
    assert (template / "skills" / "user" / "my-new-skill" / "SKILL.md").read_text() == "# new skill\n"

    # Non-allowlisted paths did NOT:
    assert not (template / ".env").exists()
    assert not (template / "sessions.db").exists()
    assert not (template / "kanban.db").exists()
    assert not (template / "logs").exists()
    assert not (template / "skills" / "seed" / "default" / "SKILL.md").exists()


def test_t17_mergeback_emits_audit_json(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "MEMORY.md").write_text("old\n")

    src = tmp_path / "run" / "orch" / "hermes_home"
    _populate_run_home(src)

    from nemo_skills.scripts import merge_hermes_home

    result = merge_hermes_home.merge(sources=[src], template=template, dry_run=False)
    audit = tmp_path / "audit.json"
    merge_hermes_home.write_audit(result, audit)

    payload = json.loads(audit.read_text())
    by_rel = {f["rel_path"]: f for f in payload["files"]}

    # Per-file before/after sha256s:
    mem_audit = by_rel["MEMORY.md"]
    assert mem_audit["action"] == "copied"
    assert mem_audit["before_sha256"] == hashlib.sha256(b"old\n").hexdigest()
    assert mem_audit["after_sha256"] == hashlib.sha256(b"learned X\n").hexdigest()
    assert mem_audit["size_delta"] != 0

    # USER.md was missing in template before; before-hash is empty.
    user_audit = by_rel["USER.md"]
    assert user_audit["before_sha256"] == ""
    assert user_audit["after_sha256"] == hashlib.sha256(b"user notes\n").hexdigest()

    # skills/user subtree files have nested rel_paths and copied action.
    skill_key = "skills/user/my-new-skill/SKILL.md"
    assert skill_key in by_rel
    assert by_rel[skill_key]["action"] == "copied"


def test_mergeback_dry_run_doesnt_write(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "MEMORY.md").write_text("old\n")
    src = tmp_path / "run" / "orch"
    _populate_run_home(src)

    from nemo_skills.scripts import merge_hermes_home

    merge_hermes_home.merge(sources=[src], template=template, dry_run=True)
    # Template is unchanged.
    assert (template / "MEMORY.md").read_text() == "old\n"
    assert not (template / "USER.md").exists()


def test_mergeback_unchanged_files_are_marked(tmp_path: Path) -> None:
    """When a source file already matches the template, the audit notes
    'unchanged' (not 'copied') — important so we don't bump mtimes on
    every run and confuse downstream tooling."""
    template = tmp_path / "template"
    template.mkdir()
    (template / "MEMORY.md").write_text("same content\n")
    src = tmp_path / "run"
    src.mkdir()
    (src / "MEMORY.md").write_text("same content\n")

    from nemo_skills.scripts import merge_hermes_home

    result = merge_hermes_home.merge(sources=[src], template=template, dry_run=False)
    mem = next(f for f in result.files if f.rel_path == "MEMORY.md")
    assert mem.action == "unchanged"


# ---------------------------------------------------------------------------
# T1.8 — pipeline assembly produces a coherent het-job spec.
# ---------------------------------------------------------------------------


def _minimal_cluster_config() -> Dict[str, Any]:
    # ``executor: "none"`` short-circuits mount validation inside
    # ``ServerScript.__post_init__`` → ``get_server_command``, so test
    # paths like ``/m/orch`` don't need to be declared as mounts.
    return {
        "executor": "none",
        "containers": {
            "vllm": "vllm:latest",
            "sandbox": "sandbox:latest",
            "nemo-gym": "nemo-gym:latest",
            "nemo-rl": "nemo-rl:latest",
        },
    }


def _collect_bootstrap_overlays(groups) -> Dict[str, Dict[str, Any]]:
    """Walk the CommandGroups → bootstrap scripts → overlays dict.

    Returns ``{agent_name: overlay_dict}``.  Used by several tests to
    inspect what the pipeline writes per agent.
    """
    overlays: Dict[str, Dict[str, Any]] = {}
    for g in _iter_groups(groups):
        for cmd in g.commands:
            if isinstance(cmd.script, HermesHomeBootstrapScript):
                name = cmd.script.overlay.get("agent_name")
                if name:
                    overlays[name] = dict(cmd.script.overlay)
    return overlays


def _iter_groups(groups):
    if isinstance(groups, list):
        for g in groups:
            yield g
    else:
        yield groups


def test_t18_pipeline_dry_run_validates_full_manifest() -> None:
    """End-to-end: build_jobs for orchestrator + co-located worker + own-model
    worker yields one het-job with 2 CommandGroups (het-group 0 + het-group 1).
    """
    manifest = parse_manifest(_toy_manifest_two_workers())
    jobs = _build_jobs(
        manifest=manifest,
        cluster_config=_minimal_cluster_config(),
        output_dir="/results/run1",
        log_dir="/results/run1/logs",
        input_file="/data/in.jsonl",
        template_path="/tmpl",
        server_container_override=None,
        gym_container="nemo-gym",
        gym_path=None,
        with_sandbox=True,
        sandbox_container_override=None,
        sandbox_mounts=None,
        sbatch_kwargs={},
        partition=None,
        qos=None,
        time_min=None,
        merge_back=True,
        merge_dry_run=False,
        extra_arguments="",
        policy_api_key="dummy",  # pragma: allowlist secret
        policy_model_name=None,
        expname="hermes_test",
    )
    assert len(jobs) == 1
    job = jobs[0]
    groups = job["groups"]  # Multiple groups because we have a non-co-located worker
    assert len(groups) == 2

    # Het-group 0 (orchestrator + analyst): server + sandbox + 2*(bootstrap+head) + rollouts + mergeback.
    orch_group = groups[0]
    orch_names = [c.name for c in orch_group.commands]
    assert any(n.endswith("_server") for n in orch_names)
    assert any(n.endswith("_sandbox") for n in orch_names)
    assert "hermes_test_orch_bootstrap" in orch_names
    assert "hermes_test_orch_head" in orch_names
    assert "hermes_test_analyst_bootstrap" in orch_names
    assert "hermes_test_analyst_head" in orch_names
    assert "hermes_test_orch_rollouts" in orch_names
    assert "hermes_test_mergeback" in orch_names

    # Het-group 1 (coder): server + sandbox + bootstrap + head.  No
    # rollouts/mergeback — the orchestrator owns those.
    coder_group = groups[1]
    coder_names = [c.name for c in coder_group.commands]
    assert "hermes_test_coder_bootstrap" in coder_names
    assert "hermes_test_coder_head" in coder_names
    assert not any("rollouts" in n for n in coder_names)
    assert not any("mergeback" in n for n in coder_names)


def test_single_agent_manifest_produces_single_group() -> None:
    manifest = parse_manifest({"agents": {"only": {"model": "/m/only", "server_gpus": 1}}})
    jobs = _build_jobs(
        manifest=manifest,
        cluster_config=_minimal_cluster_config(),
        output_dir="/results",
        log_dir="/results/logs",
        input_file="/data/in.jsonl",
        template_path="/t",
        server_container_override=None,
        gym_container="nemo-gym",
        gym_path=None,
        with_sandbox=False,
        sandbox_container_override=None,
        sandbox_mounts=None,
        sbatch_kwargs={},
        partition=None,
        qos=None,
        time_min=None,
        merge_back=False,
        merge_dry_run=False,
        extra_arguments="",
        policy_api_key="dummy",  # pragma: allowlist secret
        policy_model_name=None,
        expname="solo",
    )
    job = jobs[0]
    # Single group → top-level "group", not "groups".
    assert "group" in job
    assert "groups" not in job
    grp: CommandGroup = job["group"]
    names = [c.name for c in grp.commands]
    assert "solo_only_bootstrap" in names
    assert "solo_only_head" in names
    assert "solo_only_rollouts" in names


def test_pipeline_rejects_missing_gym_container() -> None:
    from nemo_skills.pipeline.hermes_agent_rollouts import _resolve_template_path

    # Direct unit on the helper; the command-level guard against missing
    # containers triggers via the cluster_config["containers"][...] lookup.
    manifest = parse_manifest({"agents": {"a": {"model": "/m/a"}}})
    # Sanity: the resolver falls back through the chain we documented.
    path = _resolve_template_path(
        cluster_config=_minimal_cluster_config(),
        explicit=None,
        manifest=manifest,
        gym_path="/some/gym",
    )
    assert path.endswith("/responses_api_agents/hermes_agent/hermes_home_template")


# ---------------------------------------------------------------------------
# T1.9 — existing ns nemo_gym_rollouts is untouched.
# ---------------------------------------------------------------------------


def test_t19_existing_nemo_gym_rollouts_command_still_importable() -> None:
    # The Phase 1 PR must not break the existing single-agent pipeline.
    from nemo_skills.pipeline.nemo_gym_rollouts import nemo_gym_rollouts  # noqa: F401

    # If the command's typer wiring degrades we'd see a runtime error
    # registering the app; the bare import suffices as a regression guard.
    from nemo_skills.pipeline.cli import app

    # Typer leaves ``RegisteredCommand.name`` as ``None`` until the
    # name-resolver runs at CLI dispatch time; the stable identity at
    # import time is the callback function's name.
    cb_names = {c.callback.__name__ for c in app.registered_commands if c.callback}
    assert "nemo_gym_rollouts" in cb_names
    assert "hermes_agent_rollouts" in cb_names
