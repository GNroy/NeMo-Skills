# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Pipeline command for multi-agent SLURM jobs.

ns agent creates a heterogeneous SLURM job with:
  Het-group 0  — orchestrator: LLM server A + agent_generate (batch mode)
  Het-group 1  — worker "name_0": LLM server B + agent_server (HTTP mode)
  Het-group N  — worker "name_N-1": LLM server N + agent_server (HTTP mode)

For single-agent usage (no --worker-model), this degrades to a simple
generate job using nemo_skills.inference.agent_generate instead of
nemo_skills.inference.generate, with trajectory saving enabled.

Worker address wiring
---------------------
`ns agent` automatically appends Hydra overrides for CallAgentTool so the
orchestrator knows how to reach each worker:

    ++tool_overrides.CallAgentTool.agents.{name}.het_group={N}
    ++tool_overrides.CallAgentTool.agents.{name}.port={port}

These are added to extra_arguments for the orchestrator script.
The worker's actual hostname is resolved at job startup from
SLURM_MASTER_NODE_HET_GROUP_N (exported by NeMo-Run).

Example — single agent with code execution:
    ns agent \\
        --cluster my_cluster \\
        --model /models/llama-3-8b --server-type vllm --server-gpus 8 \\
        --input-file /data/problems.jsonl --output-dir /results \\
        ++tool_modules=["nemo_skills.mcp.servers.python_tool::DirectPythonTool"]

Example — multi-agent:
    ns agent \\
        --cluster my_cluster \\
        --model /models/orchestrator --server-type vllm --server-gpus 8 \\
        --worker-model /models/analyst /models/coder \\
        --worker-server-type vllm --worker-server-gpus 4 \\
        --worker-names analyst coder \\
        --input-file /data/problems.jsonl --output-dir /results \\
        ++tool_modules=["nemo_skills.mcp.servers.agent_tool::CallAgentTool", \\
                        "nemo_skills.mcp.servers.python_tool::DirectPythonTool"]
"""

import logging
from typing import Dict, List, Optional

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
from nemo_skills.pipeline.utils.scripts import (
    GenerationClientScript,
    SandboxScript,
    ServerScript,
)
from nemo_skills.pipeline.utils.scripts.agent import AgentWorkerScript
from nemo_skills.utils import (
    compute_chunk_ids,
    get_logger_name,
    setup_logging,
    str_ids_to_list,
)

LOG = logging.getLogger(get_logger_name(__file__))


def _build_worker_agent_overrides(
    worker_names: List[str],
    worker_scripts: List[AgentWorkerScript],
) -> str:
    """Return Hydra++ args that wire CallAgentTool to the worker agents.

    Each worker gets:
      ++tool_overrides.CallAgentTool.agents.{name}.het_group={N}
      ++tool_overrides.CallAgentTool.agents.{name}.port={port}

    Het-group indices start at 1 (group 0 is the orchestrator).
    """
    parts = []
    for i, (name, worker) in enumerate(zip(worker_names, worker_scripts)):
        het_group = i + 1  # workers start at het-group 1
        parts.append(f"++tool_overrides.CallAgentTool.agents.{name}.het_group={het_group}")
        parts.append(f"++tool_overrides.CallAgentTool.agents.{name}.port={worker.agent_port}")
    return " ".join(parts)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def agent(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="Cluster config name (inside config_dir or NEMO_SKILLS_CONFIG_DIR). "
        "Can also set NEMO_SKILLS_CONFIG.",
    ),
    input_file: str = typer.Option(None, help="Path to the input JSONL file."),
    input_dir: str = typer.Option(None, help="Path to the input directory."),
    output_dir: str = typer.Option(..., help="Where to put results."),
    expname: str = typer.Option("agent", help="NeMo-Run experiment name."),
    # Orchestrator LLM
    model: str = typer.Option(..., help="Path/name of the orchestrator LLM."),
    server_address: str = typer.Option(None, help="Address of a pre-hosted orchestrator LLM."),
    server_type: pipeline_utils.SupportedServers = typer.Option(..., help="Orchestrator server type."),
    server_gpus: int = typer.Option(None, help="GPUs for orchestrator LLM."),
    server_nodes: int = typer.Option(1, help="Nodes for orchestrator LLM."),
    server_args: str = typer.Option("", help="Extra args for orchestrator LLM server."),
    server_container: str = typer.Option(None, help="Container for orchestrator LLM."),
    main_container: str = typer.Option(None, help="Container for the orchestrator agent script."),
    sandbox_container: str = typer.Option(None, help="Container for the sandbox."),
    # Worker agents (optional — omit for single-agent mode)
    worker_model: List[str] = typer.Option(
        None,
        help="Model path(s) for worker agent(s). Repeat flag for multiple workers. "
        "Single value broadcasts to all workers.",
    ),
    worker_server_type: List[pipeline_utils.SupportedServers] = typer.Option(
        None,
        help="Server type(s) for worker(s). Single value broadcasts.",
    ),
    worker_server_gpus: List[int] = typer.Option(
        None,
        help="GPUs per worker. Single value broadcasts.",
    ),
    worker_server_nodes: List[int] = typer.Option(
        [1],
        help="Nodes per worker. Single value broadcasts.",
    ),
    worker_server_args: List[str] = typer.Option(
        [""],
        help="Extra LLM server args per worker. Single value broadcasts.",
    ),
    worker_names: List[str] = typer.Option(
        None,
        help="Names for workers (used in tool names: call_{name}). "
        "Defaults to worker_0, worker_1, …",
    ),
    worker_extra_args: List[str] = typer.Option(
        [""],
        help="Extra Hydra++ args forwarded to each worker agent_server. "
        "Single value broadcasts.",
    ),
    worker_agent_port: int = typer.Option(
        7777,
        help="Base agent HTTP port for workers. Each worker gets this port "
        "(workers are on different nodes so no collision).",
    ),
    # Common pipeline options
    with_sandbox: bool = typer.Option(False, help="Start a sandbox container alongside this job."),
    sandbox_env_overrides: List[str] = typer.Option(
        None, help="Extra env vars for the sandbox container (KEY=VALUE)."
    ),
    keep_mounts_for_sandbox: bool = typer.Option(False, help="Keep filesystem mounts for the sandbox."),
    partition: str = typer.Option(None, help="SLURM partition."),
    account: str = typer.Option(None, help="SLURM account."),
    qos: str = typer.Option(None, help="SLURM QoS."),
    time_min: str = typer.Option(None, help="SLURM time-min."),
    run_after: List[str] = typer.Option(None, help="Experiment names that must complete first."),
    reuse_code: bool = typer.Option(True, help="Reuse code from last submitted experiment."),
    reuse_code_exp: str = typer.Option(None, help="Reuse code from this specific experiment."),
    config_dir: str = typer.Option(None, help="Directory to search for cluster configs."),
    log_dir: str = typer.Option(None, help="Custom SLURM log directory."),
    exclusive: bool | None = typer.Option(None, help="Add --exclusive to the SLURM job."),
    num_random_seeds: int = typer.Option(None, help="Number of random seeds for repeated runs."),
    random_seeds: str = typer.Option(None, help="Explicit random seeds (comma or range)."),
    starting_seed: int = typer.Option(0, help="Starting seed."),
    num_chunks: int = typer.Option(None, help="Number of chunks to split the dataset into."),
    chunk_ids: str = typer.Option(None, help="Explicit chunk ids to run."),
    rerun_done: bool = typer.Option(False, help="Re-run even if .done files exist."),
    dry_run: bool = typer.Option(False, help="Validate arguments without submitting."),
    mount_paths: str = typer.Option(None, help="Comma-separated paths to mount on the remote."),
    check_mounted_paths: bool = typer.Option(False, help="Check mounted paths exist on remote."),
    sbatch_kwargs: str = typer.Option("", help="Extra sbatch kwargs as JSON string."),
    skip_hf_home_check: bool | None = typer.Option(None, help="Skip HF_HOME env var check."),
    installation_command: str | None = typer.Option(
        None, help="Installation command to run before the orchestrator."
    ),
    _reuse_exp: str = typer.Option(None, hidden=True),
    _task_dependencies: List[str] = typer.Option(None, hidden=True),
):
    """Run a single or multi-agent job on a SLURM cluster.

    Single-agent mode (no --worker-model):
      Equivalent to `ns generate` but uses agent_generate.py with trajectory
      saving enabled by default.

    Multi-agent mode (--worker-model specified):
      Creates a het-job with one het-group per agent.  Workers run as HTTP
      servers; the orchestrator calls them via CallAgentTool.  Agent names
      are used as tool names: call_{name}.
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = " ".join(ctx.args)
    LOG.info("Starting agent job (extra_arguments: %s)", extra_arguments)

    # ------------------------------------------------------------------
    # Normalise orchestrator server config
    # ------------------------------------------------------------------
    server_type_str = server_type.value if hasattr(server_type, "value") else server_type
    orch_server_config, orch_server_address, orch_extra_args = pipeline_utils.configure_client(
        model=model,
        server_type=server_type_str,
        server_address=server_address,
        server_gpus=server_gpus,
        server_nodes=server_nodes,
        server_args=server_args,
        server_entrypoint=None,
        server_container=server_container,
        extra_arguments=extra_arguments,
        get_random_port=pipeline_utils.should_get_random_port(server_gpus, exclusive),
    )

    # ------------------------------------------------------------------
    # Normalise worker configs
    # ------------------------------------------------------------------
    num_workers = len(worker_model) if worker_model else 0
    have_workers = num_workers > 0

    if have_workers:
        def _conv(v):
            return v.value if hasattr(v, "value") else v

        worker_server_types = pipeline_utils.normalize_parameter(
            [_conv(t) for t in worker_server_type] if worker_server_type else ["vllm"],
            num_workers,
            "worker_server_type",
        )
        worker_gpus = pipeline_utils.normalize_parameter(
            worker_server_gpus or [8],
            num_workers,
            "worker_server_gpus",
        )
        worker_nodes = pipeline_utils.normalize_parameter(
            worker_server_nodes,
            num_workers,
            "worker_server_nodes",
        )
        worker_args = pipeline_utils.normalize_parameter(
            worker_server_args,
            num_workers,
            "worker_server_args",
        )
        worker_xargs = pipeline_utils.normalize_parameter(
            worker_extra_args,
            num_workers,
            "worker_extra_args",
        )
        # Default worker names
        if not worker_names:
            worker_names = [f"worker_{i}" for i in range(num_workers)]
        if len(worker_names) != num_workers:
            raise ValueError(
                f"--worker-names has {len(worker_names)} entries but "
                f"{num_workers} workers were specified via --worker-model."
            )

    # ------------------------------------------------------------------
    # Cluster & paths
    # ------------------------------------------------------------------
    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)
    cluster_config = pipeline_utils.resolve_mount_paths(
        cluster_config, mount_paths, create_remote_dir=check_mounted_paths
    )
    if not log_dir:
        log_dir = f"{output_dir}/agent-logs"
    output_dir, log_dir = pipeline_utils.check_mounts(
        cluster_config,
        log_dir=log_dir,
        mount_map={output_dir: None},
        check_mounted_paths=check_mounted_paths,
    )

    # ------------------------------------------------------------------
    # Random seeds / chunks
    # ------------------------------------------------------------------
    if random_seeds and num_random_seeds:
        raise ValueError("Cannot specify both --random-seeds and --num-random-seeds.")
    if num_random_seeds:
        random_seeds = list(range(starting_seed, starting_seed + num_random_seeds))
    if isinstance(random_seeds, str):
        random_seeds = str_ids_to_list(random_seeds)
    if num_chunks:
        chunk_ids = compute_chunk_ids(chunk_ids, num_chunks)
    if chunk_ids is None:
        chunk_ids = [None]

    remaining_jobs = pipeline_utils.get_remaining_jobs(
        cluster_config=cluster_config,
        output_dir=output_dir,
        random_seeds=random_seeds or [None],
        chunk_ids=chunk_ids,
        rerun_done=rerun_done,
    )

    sbatch_kwargs = parse_kwargs(sbatch_kwargs, exclusive=exclusive, qos=qos, time_min=time_min)
    if _task_dependencies is None:
        _task_dependencies = []

    # ------------------------------------------------------------------
    # Build jobs
    # ------------------------------------------------------------------
    jobs = []

    for seed_idx, (seed, seed_chunk_ids) in enumerate(remaining_jobs.items()):
        for chunk_id in seed_chunk_ids:
            task_name = f"{expname}-rs{seed}" if seed is not None else expname
            if chunk_id is not None:
                task_name += f"-chunk{chunk_id}"

            groups: List[CommandGroup] = []

            # ----------------------------------------------------------
            # Worker het-groups (1 .. N)
            # ----------------------------------------------------------
            worker_scripts: List[AgentWorkerScript] = []

            if have_workers:
                for i, w_model in enumerate(worker_model):
                    w_name = worker_names[i]
                    w_server_type = worker_server_types[i]
                    w_gpus = worker_gpus[i]
                    w_nodes = worker_nodes[i]
                    w_args = worker_args[i]
                    w_xargs = worker_xargs[i]

                    w_container = cluster_config["containers"].get(w_server_type, "")

                    w_server = ServerScript(
                        server_type=w_server_type,
                        model_path=w_model,
                        cluster_config=cluster_config,
                        num_gpus=w_gpus,
                        num_nodes=w_nodes,
                        server_args=w_args,
                        allocate_port=True,
                    )

                    w_worker = AgentWorkerScript(
                        server_script=w_server,
                        agent_name=w_name,
                        agent_port=worker_agent_port,
                        extra_arguments=w_xargs,
                    )

                    worker_scripts.append(w_worker)

                    w_server_cmd = Command(
                        script=w_server,
                        container=w_container,
                        name=f"{task_name}_{w_name}_server",
                    )
                    w_worker_cmd = Command(
                        script=w_worker,
                        container=main_container or cluster_config["containers"]["nemo-skills"],
                        name=f"{task_name}_{w_name}",
                    )
                    groups.append(
                        CommandGroup(
                            commands=[w_server_cmd, w_worker_cmd],
                            hardware=HardwareConfig(
                                partition=partition,
                                account=account,
                                num_gpus=w_gpus,
                                num_nodes=w_nodes,
                                num_tasks=w_server.num_tasks,
                                sbatch_kwargs=sbatch_kwargs,
                            ),
                            name=f"{task_name}_{w_name}_group",
                            log_dir=log_dir,
                        )
                    )

            # ----------------------------------------------------------
            # Orchestrator het-group (group 0)
            # ----------------------------------------------------------
            orch_extra = orch_extra_args

            # Auto-inject CallAgentTool worker address overrides
            if have_workers:
                orch_extra += " " + _build_worker_agent_overrides(worker_names, worker_scripts)

            # Build orchestrator server (may be None if pre-hosted)
            orch_server_script = None
            if orch_server_config is not None and int(orch_server_config.get("num_gpus", 0)) > 0:
                orch_container = orch_server_config.get("container") or cluster_config["containers"][server_type_str]
                orch_server_script = ServerScript(
                    server_type=server_type_str,
                    model_path=orch_server_config["model_path"],
                    cluster_config=cluster_config,
                    num_gpus=orch_server_config["num_gpus"],
                    num_nodes=orch_server_config["num_nodes"],
                    server_args=orch_server_config.get("server_args", ""),
                    port=orch_server_config.get("server_port"),
                    allocate_port=(orch_server_config.get("server_port") is None),
                )

            sandbox_script = None
            orch_components = []

            if orch_server_script is not None:
                orch_components.append(
                    Command(
                        script=orch_server_script,
                        container=orch_container,
                        name=f"{task_name}_server",
                    )
                )

            if with_sandbox:
                sandbox_script = SandboxScript(
                    cluster_config=cluster_config,
                    keep_mounts=keep_mounts_for_sandbox,
                    allocate_port=True,
                    env_overrides=sandbox_env_overrides,
                )
                orch_components.append(
                    Command(
                        script=sandbox_script,
                        container=sandbox_container or cluster_config["containers"]["sandbox"],
                        name=f"{task_name}_sandbox",
                    )
                )

            orch_client = GenerationClientScript(
                output_dir=output_dir,
                input_file=input_file,
                input_dir=input_dir,
                extra_arguments=orch_extra,
                random_seed=seed,
                chunk_id=chunk_id,
                num_chunks=num_chunks,
                with_sandbox=with_sandbox,
                # Use agent_generate instead of the default generate script
                script="nemo_skills.inference.agent_generate",
                servers=[orch_server_script] if orch_server_script else None,
                server_addresses_prehosted=[orch_server_address] if not orch_server_script else None,
                model_names=[model],
                server_types=[server_type_str],
                sandbox=sandbox_script,
                installation_command=installation_command,
            )
            orch_components.append(
                Command(
                    script=orch_client,
                    container=main_container or cluster_config["containers"]["nemo-skills"],
                    name=task_name,
                )
            )

            orch_hardware = HardwareConfig(
                partition=partition,
                account=account,
                num_gpus=orch_server_script.num_gpus if orch_server_script else 0,
                num_nodes=orch_server_script.num_nodes if orch_server_script else 1,
                num_tasks=orch_server_script.num_tasks if orch_server_script else 1,
                sbatch_kwargs=sbatch_kwargs,
            )
            # Orchestrator group is group 0 — workers were appended as groups 1..N above.
            # We prepend the orchestrator group so het_group indices match.
            orch_group = CommandGroup(
                commands=orch_components,
                hardware=orch_hardware,
                name=task_name,
                log_dir=log_dir,
            )
            # Groups list: [worker_0, worker_1, ...] so far; prepend orchestrator → [orch, w0, w1, ...]
            groups.insert(0, orch_group)

            # ----------------------------------------------------------
            # Build dependencies
            # ----------------------------------------------------------
            dependencies = list(_task_dependencies)
            if not dependencies and run_after:
                dependencies.extend(run_after if isinstance(run_after, list) else [run_after])

            job_spec = {
                "name": task_name,
                "dependencies": dependencies or None,
            }
            if len(groups) > 1:
                job_spec["groups"] = groups
            else:
                job_spec["group"] = groups[0]

            jobs.append(job_spec)

    if not jobs:
        return None

    pipeline = Pipeline(
        name=expname,
        cluster_config=cluster_config,
        jobs=jobs,
        reuse_code=reuse_code,
        reuse_code_exp=reuse_code_exp,
        skip_hf_home_check=skip_hf_home_check,
    )

    sequential = cluster_config["executor"] in ["local", "none"]
    return pipeline.run(dry_run=dry_run, _reuse_exp=_reuse_exp, sequential=sequential)


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
