"""Wasted-compute invariants — vendored stub for Hovhannes's
run_hle_mcp_ablation_*.

The original module (in his mcp/ checkout) enforces a handful of SLURM
allocation hygiene rules derived from a "wasted compute" RCA.  For our
baseline runs on aws-cmh with full-node TP=4 configs (the only thing we
exercise here), the relevant invariants are trivially satisfied, so this
stub mostly defers to "yes, looks fine".

If/when we start submitting partial-node jobs or running on aws-dfw with
its co-tenant placement policy, replace this with the real module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_TIMEOUT_SEC = 300


@dataclass
class AllocationVerdict:
    """What kind of SLURM allocation are we asking for?"""

    server_gpus: int
    server_nodes: int
    exclusive: bool  # True for full-node; False for partial-node on shared clusters
    partial_node: bool = False
    note: str = ""


def classify_allocation(*, cluster: str, server_gpus: int, server_nodes: int) -> AllocationVerdict:
    """Return a verdict describing the allocation shape.

    Hovhannes's original distinguishes full-node from partial-node based on
    the cluster's GPU/node count.  For aws-cmh (GB300, 4 GPUs/node), the
    only configs we care about are full-node TP=4 (one node) or multi-node
    full allocations.  We treat both as exclusive.
    """
    full_node = server_gpus in (4, 8) and server_nodes >= 1
    return AllocationVerdict(
        server_gpus=server_gpus,
        server_nodes=server_nodes,
        exclusive=full_node,
        partial_node=not full_node,
        note=("full-node" if full_node else "partial-node — review before submitting"),
    )


def check_tp_matches_alloc(server_args: str, server_gpus: int, server_nodes: int) -> None:
    """Best-effort check that ``--tensor-parallel-size`` matches the
    allocation shape.  Raises ``SystemExit`` on hard mismatch."""
    m = re.search(r"--tensor-parallel-size\s+(\d+)", server_args)
    if not m:
        return  # no TP flag → skip
    tp = int(m.group(1))
    total_gpus = server_gpus * server_nodes
    if total_gpus % tp != 0:
        raise SystemExit(
            f"check_tp_matches_alloc: TP={tp} does not divide allocation "
            f"({server_gpus} GPUs x {server_nodes} nodes = {total_gpus})."
        )


def require_direct_mcp_tools(tool_module_strs) -> None:
    """Validate that every retrieval-style MCP tool is the in-process
    ``Direct*`` variant.  The non-Direct variants leak async transports
    in some failure modes.

    Only applied to retrieval tools (libretexts/arxiv/wikipedia).  Compute
    tools and the python_tool can stay non-Direct.
    """
    retrieval_keywords = ("libretexts", "arxiv", "wikipedia")
    for mod in tool_module_strs or []:
        lower = mod.lower()
        if any(kw in lower for kw in retrieval_keywords) and "Direct" not in mod:
            raise SystemExit(
                f"require_direct_mcp_tools: retrieval tool {mod!r} must use the "
                "Direct* in-process variant.  See wasted_compute_root_cause Pattern 5."
            )


def retrieval_timeout_env() -> str:
    """Shell snippet that exports the retrieval-tool timeout."""
    return f"export MCP_TOOL_TIMEOUT_SEC={_TIMEOUT_SEC}"


def sbatch_kwargs_for_cluster(cluster: str) -> dict:
    """Per-cluster sbatch kwargs.  We return an empty dict; aws-cmh has no
    special reaper exemption needs in our account."""
    return {}


def print_verdict(verdict: AllocationVerdict, *, header: str = "") -> None:
    """Pretty-print a verdict line."""
    tag = verdict.note or ("full-node" if verdict.exclusive else "partial-node")
    print(
        f"{header} {verdict.server_gpus} GPUs x {verdict.server_nodes} node(s) "
        f"({tag}, exclusive={verdict.exclusive})"
    )
