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

"""Manifest parser for ``ns hermes_agent_rollouts``.

The manifest declares a fleet of Hermes-backed agents that should be
scheduled together in a single heterogeneous SLURM job.  Distinct LLM
models become distinct het-groups; agents sharing a model are
co-located in the same het-group sharing one LLM server (this is what
the existing ``ns agent`` command does for ``call_<worker>`` agents,
mirrored here for parity).

The schema is intentionally small — anything provider/cluster-specific
belongs in the pipeline's CLI flags or the cluster_config, not in the
manifest, so the same manifest can be portable across deployments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HermesAgentSpec:
    """One agent entry from the manifest.

    For ``kind="agent"`` (the default), either ``model`` or
    ``model_share`` must be set (exactly one):
      - ``model`` → this agent owns an LLM server with the given path.
      - ``model_share`` → this agent piggybacks on another agent's LLM
        server (co-located in the same het-group, no extra GPU spend).

    For ``kind="dispatcher"`` (Phase 4 kanban hooks), no model is
    required: the entry stands up a separate het-group running the
    Hermes kanban dispatcher daemon (``hermes kanban dispatcher run``)
    instead of a hermes_agent server.  See the Phase 4 design notes in
    ``workdir-agents/hermes_integration_plan.md``.
    """

    name: str
    role: str = "worker"  # "orchestrator" or "worker"; exactly one orchestrator per manifest

    # Phase 4: ``"agent"`` (default) runs a hermes_agent server.
    # ``"dispatcher"`` runs the kanban dispatcher daemon and has no LLM.
    kind: str = "agent"

    # LLM placement — exactly one of model / model_share is set
    # (only required when ``kind == "agent"``).
    model: Optional[str] = None
    model_share: Optional[str] = None

    # vLLM/server settings (only meaningful when ``model`` is set).
    server_type: str = "vllm"
    server_gpus: int = 8
    server_nodes: int = 1
    server_args: str = ""

    # Pass-through HermesAgentConfig overlay merged into the agent's
    # config.yaml at bootstrap time (max_turns, enabled_toolsets,
    # mcp_servers, etc.).
    hermes: Dict[str, Any] = field(default_factory=dict)

    # Names of other agents this agent should be able to call (as peer
    # workers via the /task endpoint).  Only meaningful for the
    # orchestrator in Phase 1; ignored otherwise.
    workers: List[str] = field(default_factory=list)

    # CPU/memory hints for dispatcher het-groups (ignored when ``kind == "agent"``).
    dispatcher_cpus: int = 2
    dispatcher_mem_gb: int = 8


@dataclass
class HermesFleetManifest:
    """Top-level manifest object."""

    agents: List[HermesAgentSpec]
    template_hermes_home: Optional[str] = None
    trace_dir_subpath: str = "traces"
    sandbox_mounts: List[str] = field(default_factory=list)
    # Phase 4 (deferred runtime, design-only here): when set, the bootstrap
    # script symlinks ``<hermes_home>/kanban.db`` to this shared path so
    # every agent in the fleet sees the same kanban board.  Pair with a
    # ``kind: dispatcher`` agent entry to actually run the daemon.
    shared_kanban_db: Optional[str] = None

    @property
    def orchestrator(self) -> HermesAgentSpec:
        for a in self.agents:
            if a.role == "orchestrator":
                return a
        # parse_manifest guarantees there is exactly one orchestrator, so
        # this branch is unreachable in valid manifests.
        raise ValueError("manifest has no orchestrator (this should have been caught at parse time)")

    @property
    def by_name(self) -> Dict[str, HermesAgentSpec]:
        return {a.name: a for a in self.agents}


@dataclass
class ServerGroup:
    """One het-group worth of agents that share an LLM server.

    Workers in the same group run on the same node(s) as the LLM server,
    with the orchestrator always in het-group 0.

    Dispatcher groups (``is_dispatcher=True``) carry a single
    ``kind=dispatcher`` owner and have no LLM server — ``model``,
    ``server_type``, ``server_gpus``, ``server_nodes``, ``server_args``
    all return safe placeholders so the pipeline can branch on the flag
    without re-reading the agent spec.
    """

    het_group: int
    # The agent whose ``model`` field defines the LLM server for this group.
    # All other agents in this group have ``model_share == owner.name``.
    # For dispatcher groups this is the dispatcher entry itself.
    owner: HermesAgentSpec
    agents: List[HermesAgentSpec]

    @property
    def is_dispatcher(self) -> bool:
        return self.owner.kind == "dispatcher"

    @property
    def model(self) -> str:
        # ``owner.model`` is guaranteed non-empty by parse_manifest for
        # kind=agent owners; dispatcher groups have no LLM and callers
        # must check ``is_dispatcher`` before using this.
        return self.owner.model or ""

    @property
    def server_type(self) -> str:
        return self.owner.server_type

    @property
    def server_gpus(self) -> int:
        return 0 if self.is_dispatcher else self.owner.server_gpus

    @property
    def server_nodes(self) -> int:
        return self.owner.server_nodes

    @property
    def server_args(self) -> str:
        return self.owner.server_args


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_manifest(source: Union[str, Path, Dict[str, Any]]) -> HermesFleetManifest:
    """Parse a manifest from a YAML file path or an already-loaded dict.

    Validation:
      - exactly one orchestrator
      - every agent has either ``model`` or ``model_share``, never both
      - ``model_share`` resolves to an existing agent that itself owns a model
      - no name collisions
      - ``workers`` entries (if any) reference declared agents
    """
    if isinstance(source, (str, Path)):
        data = yaml.safe_load(Path(source).read_text())
    elif isinstance(source, dict):
        data = source
    else:
        raise TypeError(f"parse_manifest expects path or dict, got {type(source).__name__}")

    if not isinstance(data, dict) or "agents" not in data:
        raise ValueError("manifest must be a mapping with an 'agents' field")

    raw_agents = data["agents"]
    if not isinstance(raw_agents, dict) or not raw_agents:
        raise ValueError("'agents' must be a non-empty mapping {name: spec}")

    agents: List[HermesAgentSpec] = []
    seen_names: set = set()
    for name, spec in raw_agents.items():
        if name in seen_names:
            raise ValueError(f"duplicate agent name: {name!r}")
        seen_names.add(name)

        if not isinstance(spec, dict):
            raise ValueError(f"agent {name!r}: spec must be a mapping")

        agents.append(_spec_from_dict(name, spec))

    # Validate orchestrator cardinality.  Dispatcher entries are excluded
    # from the orchestrator-pick fallback — they cannot orchestrate rollouts.
    runnable_agents = [a for a in agents if a.kind == "agent"]
    if not runnable_agents:
        raise ValueError("manifest has no kind=agent entries; nothing to orchestrate")
    orchestrators = [a for a in runnable_agents if a.role == "orchestrator"]
    if not orchestrators:
        # Default rule: first runnable agent in the manifest is the
        # orchestrator unless someone explicitly named one.  Lets short
        # single-orchestrator manifests omit `role:` boilerplate.
        runnable_agents[0].role = "orchestrator"
        orchestrators = [runnable_agents[0]]
    if len(orchestrators) > 1:
        names = [a.name for a in orchestrators]
        raise ValueError(f"manifest has multiple orchestrators: {names!r}")

    # Validate model placement and model_share targets.
    by_name = {a.name: a for a in agents}
    for a in agents:
        if a.kind == "dispatcher":
            # Dispatcher: no LLM, no peer-routing.  Reject keys that only
            # make sense for a kind=agent entry so typos become loud.
            if a.model or a.model_share:
                raise ValueError(
                    f"agent {a.name!r}: kind=dispatcher must not declare 'model' or 'model_share'"
                )
            if a.role == "orchestrator":
                raise ValueError(f"agent {a.name!r}: dispatcher cannot also be the orchestrator")
            if a.workers:
                raise ValueError(f"agent {a.name!r}: dispatcher cannot declare 'workers'")
            continue
        if a.kind != "agent":
            raise ValueError(
                f"agent {a.name!r}: unknown kind={a.kind!r}; expected 'agent' or 'dispatcher'"
            )
        if a.model and a.model_share:
            raise ValueError(f"agent {a.name!r}: set exactly one of 'model' / 'model_share'")
        if not a.model and not a.model_share:
            raise ValueError(f"agent {a.name!r}: missing 'model' or 'model_share'")
        if a.model_share:
            owner = by_name.get(a.model_share)
            if owner is None:
                raise ValueError(
                    f"agent {a.name!r}: model_share={a.model_share!r} is not a declared agent"
                )
            if owner.kind != "agent":
                raise ValueError(
                    f"agent {a.name!r}: model_share={a.model_share!r} resolves to a "
                    f"{owner.kind!r} entry; only kind=agent peers can host a shared LLM"
                )
            if not owner.model:
                raise ValueError(
                    f"agent {a.name!r}: model_share={a.model_share!r} must itself own a model"
                    " (chained model_share is not supported)"
                )
        # Server fields only meaningful for owners — quietly ignore on shared agents,
        # but if explicitly set on a shared agent reject so the manifest is unambiguous.
        if a.model_share:
            for f in ("server_gpus", "server_nodes", "server_args", "server_type"):
                # server_type/gpus/nodes default to the dataclass defaults; we treat
                # a non-default value on a model_share agent as a configuration smell.
                default = HermesAgentSpec.__dataclass_fields__[f].default
                if getattr(a, f) != default:
                    raise ValueError(
                        f"agent {a.name!r}: cannot set '{f}' when 'model_share' is used "
                        f"(server config belongs to the owner {a.model_share!r})"
                    )

    # Validate workers references.
    for a in agents:
        for w in a.workers:
            if w not in by_name:
                raise ValueError(
                    f"agent {a.name!r}: workers entry {w!r} does not match a declared agent"
                )
            if w == a.name:
                raise ValueError(f"agent {a.name!r}: cannot list itself in 'workers'")

    return HermesFleetManifest(
        agents=agents,
        template_hermes_home=data.get("template_hermes_home"),
        trace_dir_subpath=data.get("trace_dir_subpath", "traces"),
        sandbox_mounts=list(data.get("sandbox", {}).get("mounts", []) or []),
        shared_kanban_db=data.get("shared_kanban_db"),
    )


def _spec_from_dict(name: str, spec: Dict[str, Any]) -> HermesAgentSpec:
    """Convert a YAML mapping to a HermesAgentSpec.

    Tolerates unknown keys by raising — the manifest is small and explicit;
    silent ignore would mask typos like ``moedl: ...``.
    """
    known = {
        "role",
        "kind",
        "model",
        "model_share",
        "server_type",
        "server_gpus",
        "server_nodes",
        "server_args",
        "hermes",
        "workers",
        "dispatcher_cpus",
        "dispatcher_mem_gb",
    }
    extra = set(spec.keys()) - known
    if extra:
        raise ValueError(f"agent {name!r}: unknown manifest keys: {sorted(extra)}")

    return HermesAgentSpec(
        name=name,
        role=spec.get("role", "worker"),
        kind=spec.get("kind", "agent"),
        model=spec.get("model"),
        model_share=spec.get("model_share"),
        server_type=spec.get("server_type", "vllm"),
        server_gpus=int(spec.get("server_gpus", 8)),
        server_nodes=int(spec.get("server_nodes", 1)),
        server_args=spec.get("server_args", ""),
        hermes=dict(spec.get("hermes", {}) or {}),
        workers=list(spec.get("workers", []) or []),
        dispatcher_cpus=int(spec.get("dispatcher_cpus", 2)),
        dispatcher_mem_gb=int(spec.get("dispatcher_mem_gb", 8)),
    )


# ---------------------------------------------------------------------------
# Server-group planning
# ---------------------------------------------------------------------------


def build_server_groups(manifest: HermesFleetManifest) -> List[ServerGroup]:
    """Compute the het-group plan.

    Algorithm:
      0. Orchestrator's owner group is always het-group 0.
      1. For every other model-owning agent, allocate a new het-group.
      2. Each model_share agent is appended to the group of its owner.

    Returns the list ordered by het_group ascending so the orchestrator
    group is always first.
    """
    orchestrator = manifest.orchestrator
    if orchestrator.model_share:
        # The orchestrator cannot piggyback on someone else — it must own
        # its group so its het-group index is 0 by convention.
        raise ValueError(
            f"orchestrator {orchestrator.name!r} must declare its own 'model', not 'model_share'"
        )

    by_name = manifest.by_name
    groups: Dict[str, ServerGroup] = {}

    # Owner of het-group 0 = orchestrator.
    groups[orchestrator.name] = ServerGroup(het_group=0, owner=orchestrator, agents=[orchestrator])

    next_het = 1
    for a in manifest.agents:
        if a is orchestrator:
            continue
        if a.kind == "dispatcher":
            # Each dispatcher gets its own het-group with no LLM.
            groups[a.name] = ServerGroup(het_group=next_het, owner=a, agents=[a])
            next_het += 1
            continue
        if a.model:
            # New owner group.
            groups[a.name] = ServerGroup(het_group=next_het, owner=a, agents=[a])
            next_het += 1
        else:
            owner_name = a.model_share  # guaranteed non-None by parse_manifest
            owner_group = groups.get(owner_name)
            if owner_group is None:
                # Owner appears later in the manifest — build out-of-order then merge.
                owner_spec = by_name[owner_name]
                groups[owner_name] = ServerGroup(
                    het_group=next_het if owner_spec is not orchestrator else 0,
                    owner=owner_spec,
                    agents=[owner_spec, a],
                )
                if owner_spec is not orchestrator:
                    next_het += 1
            else:
                owner_group.agents.append(a)

    return sorted(groups.values(), key=lambda g: g.het_group)


def find_group_for_agent(groups: List[ServerGroup], agent_name: str) -> ServerGroup:
    """Lookup the het-group containing the named agent."""
    for g in groups:
        if any(a.name == agent_name for a in g.agents):
            return g
    raise KeyError(f"agent {agent_name!r} not found in any server group")
