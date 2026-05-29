"""Monotonic run-ID counter, vendored stub for Hovhannes's run_hle_mcp_ablation_*.

Hovhannes's original lives in a separate checkout that mounts a JSON
counter under ``experiments/.exp_run_ids.json``.  We reproduce the same
contract with a per-prefix counter stored next to this file.
"""

from __future__ import annotations

import json
from pathlib import Path

_STATE = Path(__file__).resolve().parent / ".exp_run_ids.json"


def next_run_id(prefix: str) -> int:
    """Return the next run number for *prefix* and persist it."""
    data: dict[str, int] = {}
    if _STATE.exists():
        try:
            data = json.loads(_STATE.read_text())
        except json.JSONDecodeError:
            data = {}
    nxt = int(data.get(prefix, 0)) + 1
    data[prefix] = nxt
    _STATE.write_text(json.dumps(data, indent=2, sort_keys=True))
    return nxt
