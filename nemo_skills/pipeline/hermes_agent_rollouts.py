# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""``ns hermes_agent_rollouts`` — multi-agent Hermes rollout collection.

For a manifest declaring N Hermes agents (one orchestrator + M workers):

  * Deduplicates agents by LLM model into het-groups (orchestrator always
    het-group 0; workers sharing a model co-located; otherwise own group).
  * Bootstraps a per-agent ``HERMES_HOME`` from the bundled template at
    ``NeMo-Gym/responses_api_agents/hermes_agent/hermes_home_template/``.
  * Starts one ng_run head server per agent, hosting only that agent's
    ``hermes_agent`` (on its own port).
  * Runs ``ng_collect_rollouts`` against the orchestrator's endpoint;
    input JSONL flows there.  The orchestrator's Hermes can delegate to
    workers via their ``/task`` endpoints (peer_agents wiring lives in
    the orchestrator's overlay JSON; resolved at startup against
    ``SLURM_MASTER_NODE_HET_GROUP_N`` env vars).
  * After the orchestrator returns, runs the curator (merge_hermes_home)
    to promote ``MEMORY.md`` / ``USER.md`` / ``skills/user/`` back to the
    template — opt-out via ``--no-merge-back``.

This is Phase 1 of the Hermes-agent integration; see
``workdir-agents/hermes_integration_plan.md`` for the broader plan.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils.cluster import parse_kwargs
from nemo_skills.pipeline.utils.declarative import (
    Command,
    CommandGroup,
    HardwareConfig,
    Pipeline,
)
from nemo_skills.pipeline.utils.hermes_manifest import (
    HermesFleetManifest,
    ServerGroup,
    build_server_groups,
    parse_manifest,
)
from nemo_skills.pipeline.utils.scripts import (
    HermesAgentHeadScript,
    HermesHomeBootstrapScript,
    HermesHomeMergebackScript,
    HermesKanbanDispatcherScript,
    NemoGymRolloutsScript,
    SandboxScript,
    ServerScript,
)
from nemo_skills.pipeline.utils.server import get_free_port
from nemo_skills.utils import get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))

# Where the bundled template ships in the NeMo-Gym source tree.  This path
# is *inside* the mounted NeMo-Gym repo at job time (Phase 1 ships the
# template inside the Gym repo so each cluster always has a known-good
# starting point).
_DEFAULT_TEMPLATE_REL = "responses_api_agents/hermes_agent/hermes_home_template"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_template_path(
    cluster_config: Dict[str, Any],
    explicit: Optional[str],
    manifest: HermesFleetManifest,
    gym_path: Optional[str],
) -> str:
    """Resolve the HERMES_HOME template path.

    Order of precedence:
      1. ``--template_hermes_home`` CLI flag.
      2. ``template_hermes_home:`` in the manifest.
      3. ``<gym_path>/responses_api_agents/hermes_agent/hermes_home_template``.

    The path is returned verbatim — the pipeline does not validate
    existence here because the resolved path lives on the remote, which
    may be unreachable at submit time.  The bootstrap script will fail
    loudly if the path is missing.
    """
    if explicit:
        return explicit
    if manifest.template_hermes_home:
        return manifest.template_hermes_home
    if gym_path:
        return f"{gym_path.rstrip('/')}/{_DEFAULT_TEMPLATE_REL}"
    # Last resort: assume the container's prebaked Gym path.  Matches the
    # ``/opt/Gym`` fallback used by NemoGymRolloutsScript.
    return f"/opt/Gym/{_DEFAULT_TEMPLATE_REL}"


def _build_overlay(
    agent_spec: "HermesAgentSpec",  # noqa: F821 - forward-ref for readability
    *,
    is_orchestrator: bool,
    trace_dir: str,
    peer_agents: Optional[Dict[str, Dict[str, Any]]] = None,
    sandbox_inside_container: bool = True,
) -> Dict[str, Any]:
    """Build the per-agent HermesAgentConfig overlay JSON.

    Returns the dict that ``HermesHomeBootstrapScript`` will write as
    ``<agent_home>/hermes_agent_overlay.json``.  The Phase 0 hermes_agent
    treats this file as authoritative overrides on top of the bundled
    ``hermes_agent.yaml``.

    The peer_agents map carries *het_group + port* (not URLs) because
    SLURM hostnames are only set at job start; the hermes_agent resolves
    URLs lazily from ``SLURM_MASTER_NODE_HET_GROUP_N`` env vars.
    """
    overlay: Dict[str, Any] = {
        "agent_name": agent_spec.name,
        "trace_dir": trace_dir,
        # Defaults: orchestrator persists everything; workers persist memory
        # too (research continuity) but trajectories only on the
        # orchestrator (where rollouts are collected).  These are flipped
        # off by passing ``hermes: {persist_memory: false}`` etc. in the
        # manifest.
        "persist_memory": True,
        "persist_skills": True,
        "persist_session": False,
        "save_trajectories": is_orchestrator,
        "enable_compression": False,
        "terminal_backend": "docker" if sandbox_inside_container else "local",
    }
    # Manifest overlay wins over our defaults.
    overlay.update(agent_spec.hermes)
    # Peer agents (only meaningful for the orchestrator in Phase 1, but
    # kept generic so future fan-out patterns work without a schema change).
    if peer_agents:
        overlay["peer_agents"] = peer_agents
    return overlay


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


def _build_jobs(
    manifest: HermesFleetManifest,
    cluster_config: Dict[str, Any],
    *,
    output_dir: str,
    log_dir: str,
    input_file: str,
    template_path: str,
    server_container_override: Optional[str],
    gym_container: str,
    gym_path: Optional[str],
    with_sandbox: bool,
    sandbox_container_override: Optional[str],
    sandbox_mounts: Optional[List[str]],
    sbatch_kwargs: Dict[str, Any],
    partition: Optional[str],
    qos: Optional[str],
    time_min: Optional[str],
    merge_back: bool,
    merge_dry_run: bool,
    extra_arguments: str,
    policy_api_key: str,
    policy_model_name: Optional[str],
    expname: str,
) -> List[Dict[str, Any]]:
    """Build the het-job spec; returns ``jobs=[...]`` for ``Pipeline``."""
    groups = build_server_groups(manifest)
    # Pre-allocate one agent port per kind=agent entry (random free ports
    # on submit host; the agents bind 0.0.0.0:<port> inside their het-group
    # node).  Dispatchers run no HTTP endpoint so they're skipped.
    agent_ports: Dict[str, int] = {
        a.name: get_free_port(strategy="random")
        for a in manifest.agents
        if a.kind == "agent"
    }

    # Per-agent home dir (job-local under output_dir).
    def home_for(name: str) -> str:
        return f"{output_dir}/agents/{name}/hermes_home"

    # Trace dirs (per-agent).
    def trace_dir_for(name: str) -> str:
        return f"{output_dir}/{manifest.trace_dir_subpath}/{name}"

    # ------------------------------------------------------------------
    # Containers
    # ------------------------------------------------------------------
    containers = cluster_config["containers"]
    sandbox_container = sandbox_container_override or containers.get("sandbox", "")

    # ------------------------------------------------------------------
    # Compute peer_agents map (orchestrator only in Phase 1).
    # ------------------------------------------------------------------
    orchestrator = manifest.orchestrator
    orchestrator_group = next(g for g in groups if g.owner.name == orchestrator.name)

    peer_agents_for_orch: Dict[str, Dict[str, Any]] = {}
    # Default worker list: every other kind=agent entry.  Dispatchers are
    # excluded because they expose no /task endpoint (their interface is
    # the kanban DB).
    declared_workers = orchestrator.workers or [
        a.name for a in manifest.agents
        if a.name != orchestrator.name and a.kind == "agent"
    ]
    for worker_name in declared_workers:
        if worker_name == orchestrator.name:
            continue
        worker_spec = manifest.by_name.get(worker_name)
        if worker_spec is None:
            raise ValueError(f"orchestrator references unknown worker {worker_name!r}")
        if worker_spec.kind != "agent":
            raise ValueError(
                f"orchestrator worker {worker_name!r} has kind={worker_spec.kind!r}; "
                f"only kind=agent peers can host /task endpoints"
            )
        # Find which het-group hosts this worker.
        for g in groups:
            if any(a.name == worker_name for a in g.agents):
                peer_agents_for_orch[worker_name] = {
                    "het_group": g.het_group,
                    "port": agent_ports[worker_name],
                }
                break

    # ------------------------------------------------------------------
    # Build a CommandGroup per het-group
    # ------------------------------------------------------------------
    command_groups: List[CommandGroup] = []
    server_script_for_group: List[ServerScript] = []
    sandbox_script_for_group: List[Optional[SandboxScript]] = []

    for group in groups:
        commands: List[Command] = []
        server_script: Optional[ServerScript] = None
        sandbox_script: Optional[SandboxScript] = None

        if not group.is_dispatcher:
            # 1. LLM server for this group.
            server_type = group.server_type
            server_container = server_container_override or containers.get(server_type)
            if server_container is None:
                raise ValueError(
                    f"cluster_config['containers'] has no entry for server type {server_type!r} "
                    f"(group {group.het_group}); set --server_container or extend the config."
                )
            server_script = ServerScript(
                server_type=server_type,
                model_path=group.model,
                cluster_config=cluster_config,
                num_gpus=group.server_gpus,
                num_nodes=group.server_nodes,
                server_args=group.server_args,
                allocate_port=True,
            )
            commands.append(
                Command(
                    script=server_script,
                    container=server_container,
                    name=f"{expname}_g{group.het_group}_server",
                )
            )

            # 2. Sandbox per non-dispatcher group (workers in non-orchestrator
            # groups need their own sandbox because they're on a different node).
            if with_sandbox:
                sandbox_script = SandboxScript(
                    cluster_config=cluster_config,
                    allocate_port=True,
                )
                commands.append(
                    Command(
                        script=sandbox_script,
                        container=sandbox_container,
                        name=f"{expname}_g{group.het_group}_sandbox",
                        mounts=sandbox_mounts,
                    )
                )

        server_script_for_group.append(server_script)  # type: ignore[arg-type]
        sandbox_script_for_group.append(sandbox_script)

        # 3. Per-agent bootstrap + head/dispatcher process.
        for agent in group.agents:
            is_orch = agent.name == orchestrator.name
            is_dispatcher = agent.kind == "dispatcher"
            # Dispatchers don't emit traces or accept peer calls — keep
            # the overlay minimal so the dispatcher process doesn't try to
            # mount HermesAgentConfig at startup.
            if is_dispatcher:
                overlay: Dict[str, Any] = {
                    "agent_name": agent.name,
                    # The kanban toolset is enabled at runtime by the
                    # dispatcher script — but echoing it here keeps the
                    # overlay self-describing for human readers.
                    "enabled_toolsets": list(agent.hermes.get("enabled_toolsets", ["kanban"])),
                }
            else:
                overlay = _build_overlay(
                    agent,
                    is_orchestrator=is_orch,
                    trace_dir=trace_dir_for(agent.name),
                    peer_agents=peer_agents_for_orch if is_orch else None,
                    sandbox_inside_container=with_sandbox,
                )
            commands.append(
                Command(
                    script=HermesHomeBootstrapScript(
                        template_path=template_path,
                        agent_home=home_for(agent.name),
                        overlay=overlay,
                        shared_kanban_db=manifest.shared_kanban_db,
                    ),
                    container=gym_container,
                    name=f"{expname}_{agent.name}_bootstrap",
                )
            )
            if is_dispatcher:
                commands.append(
                    Command(
                        script=HermesKanbanDispatcherScript(
                            agent_home=home_for(agent.name),
                            agent_name=agent.name,
                        ),
                        container=gym_container,
                        name=f"{expname}_{agent.name}_dispatcher",
                    )
                )
                continue
            commands.append(
                Command(
                    script=HermesAgentHeadScript(
                        config_paths=["responses_api_agents/hermes_agent/configs/hermes_agent.yaml"],
                        agent_home=home_for(agent.name),
                        agent_port=agent_ports[agent.name],
                        agent_name=agent.name,
                        server=server_script,
                        sandbox=sandbox_script,
                        gym_path=gym_path,
                        policy_api_key=policy_api_key,
                        policy_model_name=policy_model_name or group.model,
                        keep_alive=not is_orch,  # orchestrator yields to ng_collect_rollouts
                    ),
                    container=gym_container,
                    name=f"{expname}_{agent.name}_head",
                )
            )

        # 4. Orchestrator group additionally runs ng_collect_rollouts and
        #    the optional merge-back step.
        if any(a.name == orchestrator.name for a in group.agents):
            output_file = f"{output_dir}/rollouts.jsonl"
            commands.append(
                Command(
                    script=NemoGymRolloutsScript(
                        # Re-use the standard rollouts script — it polls
                        # ng_status until ready then runs ng_collect_rollouts
                        # against the orchestrator's local ng_run instance.
                        config_paths=[
                            "responses_api_agents/hermes_agent/configs/hermes_agent.yaml",
                        ],
                        input_file=input_file,
                        output_file=output_file,
                        # Force the rollouts script to talk to the orchestrator's
                        # hermes_agent (override the agent_name).
                        extra_arguments=(
                            f"+agent_name={orchestrator.name}_hermes_agent " + extra_arguments
                        ).strip(),
                        server=server_script,
                        sandbox=sandbox_script,
                        gym_path=gym_path,
                        policy_api_key=policy_api_key,
                        policy_model_name=policy_model_name or group.model,
                    ),
                    container=gym_container,
                    name=f"{expname}_{orchestrator.name}_rollouts",
                )
            )
            if merge_back:
                # Dispatchers don't author curated content; their HERMES_HOME
                # is just a launch shell.  Skip them from merge-back sources.
                merge_sources = [home_for(a.name) for a in manifest.agents if a.kind == "agent"]
                commands.append(
                    Command(
                        script=HermesHomeMergebackScript(
                            sources=merge_sources,
                            template_path=template_path,
                            audit_path=f"{output_dir}/merge_audit.json",
                            dry_run=merge_dry_run,
                        ),
                        container=gym_container,
                        name=f"{expname}_mergeback",
                    )
                )

        # 5. HardwareConfig for this group.
        if group.is_dispatcher:
            # Dispatcher het-group: no GPUs, single CPU task.  ``num_tasks=1``
            # mirrors how a plain CPU-only NeMo-Run job is sized.
            hardware = HardwareConfig(
                partition=partition,
                num_gpus=0,
                num_nodes=1,
                num_tasks=1,
                sbatch_kwargs=sbatch_kwargs,
            )
        else:
            hardware = HardwareConfig(
                partition=partition,
                num_gpus=group.server_gpus,
                num_nodes=group.server_nodes,
                num_tasks=server_script.num_tasks,  # type: ignore[union-attr]
                sbatch_kwargs=sbatch_kwargs,
            )
        command_groups.append(
            CommandGroup(
                commands=commands,
                hardware=hardware,
                name=f"{expname}_g{group.het_group}",
                log_dir=log_dir,
            )
        )

    job_spec: Dict[str, Any] = {"name": expname}
    if len(command_groups) > 1:
        job_spec["groups"] = command_groups
    else:
        job_spec["group"] = command_groups[0]

    return [job_spec]


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def hermes_agent_rollouts(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="Cluster config name (NEMO_SKILLS_CONFIG_DIR or ./cluster_configs).",
    ),
    agent_manifest: str = typer.Option(..., help="Path to the YAML manifest declaring the agent fleet."),
    input_file: str = typer.Option(..., help="Path to the input JSONL file (the orchestrator's task list)."),
    output_dir: str = typer.Option(..., help="Output dir. Per-agent homes land at <output_dir>/agents/<name>/hermes_home."),
    expname: str = typer.Option("hermes_agents", help="NeMo-Run experiment name."),
    template_hermes_home: Optional[str] = typer.Option(
        None,
        help="Override the HERMES_HOME template path. Defaults to the manifest's "
        "template_hermes_home, then the bundled hermes_home_template/ in NeMo-Gym.",
    ),
    with_sandbox: bool = typer.Option(True, help="Start a sandbox container per het-group."),
    sandbox_mounts: List[str] | None = typer.Option(None, help="Extra mounts for the sandbox."),
    sandbox_container: Optional[str] = typer.Option(None, help="Override sandbox container."),
    server_container: Optional[str] = typer.Option(None, help="Override LLM server container."),
    gym_container: Optional[str] = typer.Option(
        None,
        help="Container for the per-agent ng_run head servers and the orchestrator's "
        "ng_collect_rollouts.  Defaults to cluster_config['containers']['nemo-gym'], "
        "falling back to 'nemo-rl'.",
    ),
    gym_path: Optional[str] = typer.Option(None, help="Path to NeMo-Gym checkout (overrides auto-resolve)."),
    policy_api_key: str = typer.Option(
        "dummy",
        help="API key for the policy server (dummy works for local vLLM).",  # pragma: allowlist secret
    ),
    policy_model_name: Optional[str] = typer.Option(
        None,
        help="Model name override for the policy server. Required for pre-hosted; "
        "for self-hosted, defaults to the model path.",
    ),
    merge_back: bool = typer.Option(
        True,
        help="Run the curator step at job end to promote MEMORY.md/USER.md/skills/user/ to the template.",
    ),
    merge_back_dry_run: bool = typer.Option(
        False,
        help="When --merge-back is on, only audit the diff without writing to the template.",
    ),
    partition: Optional[str] = typer.Option(None, help="SLURM partition."),
    qos: Optional[str] = typer.Option(None, help="SLURM QoS."),
    time_min: Optional[str] = typer.Option(None, help="SLURM time-min."),
    exclusive: bool | None = typer.Option(None, help="Add --exclusive to the SLURM job."),
    config_dir: Optional[str] = typer.Option(None, help="Cluster configs directory."),
    log_dir: Optional[str] = typer.Option(None, help="Custom SLURM log directory."),
    dry_run: bool = typer.Option(False, help="Validate arguments without submitting."),
    sbatch_kwargs: str = typer.Option("", help="Extra sbatch kwargs as JSON string."),
):
    """Launch a multi-agent Hermes rollout job on SLURM.

    See ``workdir-agents/hermes_integration_plan.md`` for the design.  The
    manifest schema is documented in
    ``nemo_skills.pipeline.utils.hermes_manifest``.
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = " ".join(ctx.args)
    LOG.info("Starting Hermes multi-agent rollouts pipeline")
    LOG.info(f"Manifest: {agent_manifest}")
    LOG.info(f"Extra arguments: {extra_arguments}")

    manifest = parse_manifest(Path(agent_manifest))

    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)

    if not log_dir:
        log_dir = f"{output_dir}/logs"

    # Container resolution (mirror nemo_gym_rollouts.py).
    containers = cluster_config["containers"]
    if gym_container is not None:
        resolved_gym_container = gym_container
    elif "nemo-gym" in containers:
        resolved_gym_container = containers["nemo-gym"]
    else:
        resolved_gym_container = containers.get("nemo-rl", "")
    if not resolved_gym_container:
        raise ValueError(
            "Could not resolve a Gym container — set --gym_container or add "
            "'nemo-gym' (or 'nemo-rl') to cluster_config['containers']."
        )
    LOG.info(f"Using gym container: {resolved_gym_container}")

    template_path = _resolve_template_path(
        cluster_config=cluster_config,
        explicit=template_hermes_home,
        manifest=manifest,
        gym_path=gym_path,
    )
    LOG.info(f"HERMES_HOME template: {template_path}")

    sbatch_kwargs_dict = parse_kwargs(sbatch_kwargs, exclusive=exclusive, qos=qos, time_min=time_min)

    jobs = _build_jobs(
        manifest=manifest,
        cluster_config=cluster_config,
        output_dir=output_dir,
        log_dir=log_dir,
        input_file=input_file,
        template_path=template_path,
        server_container_override=server_container,
        gym_container=resolved_gym_container,
        gym_path=gym_path,
        with_sandbox=with_sandbox,
        sandbox_container_override=sandbox_container,
        sandbox_mounts=sandbox_mounts,
        sbatch_kwargs=sbatch_kwargs_dict,
        partition=partition,
        qos=qos,
        time_min=time_min,
        merge_back=merge_back,
        merge_dry_run=merge_back_dry_run,
        extra_arguments=extra_arguments,
        policy_api_key=policy_api_key,
        policy_model_name=policy_model_name,
        expname=expname,
    )

    pipeline = Pipeline(
        name=expname,
        cluster_config=cluster_config,
        jobs=jobs,
    )

    sequential = cluster_config["executor"] in ["local", "none"]
    result = pipeline.run(dry_run=dry_run, sequential=sequential)
    LOG.info("Pipeline %s successfully", "validated" if dry_run else "submitted")
    return result


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
