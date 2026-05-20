# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Phase 5 tests for per-agent trajectory export.

Covers:
  - T5.1 per-agent trajectory JSONL carries token IDs + reward
  - T5.2 hermes-agent trajectory_compressor consumes the exported entry
        without crashing
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


# hermes-agent lives in a sibling repo; T5.2 skips cleanly if it's not
# importable.  We add candidates lazily to sys.path to keep the rest of
# the module loadable.
_HERMES_AGENT_CANDIDATES = [
    os.environ.get("HERMES_AGENT_PATH", ""),
    "/home/alaptev/Projects/hermes-agent",
    str(Path(__file__).resolve().parents[3] / "hermes-agent"),
]


from nemo_skills.scripts.export_hermes_trajectories import (
    export_trajectories,
    _build_trajectory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_event(path: Path, event: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _ev(
    ts: float,
    agent: str,
    kind: str,
    session_id: str,
    payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "ts": ts,
        "agent": agent,
        "session_id": session_id,
        "kind": kind,
        "payload": payload or {},
    }


def _synthesize_two_agent_run(root: Path, ticket: str = "tkt0001") -> None:
    """Create a synthetic traces/ tree mimicking a small two-agent run."""
    base = root / "traces"

    # Orchestrator session: session_start → step (with token IDs) →
    # assistant → step → session_end.
    orch_sid = "orchA"
    _write_event(
        base / "orchestrator" / f"{orch_sid}.jsonl",
        _ev(
            10.0,
            "orchestrator",
            "session_start",
            orch_sid,
            {"model": "orch-model", "user_message_preview": "solve task X"},
        ),
    )
    _write_event(
        base / "orchestrator" / f"{orch_sid}.jsonl",
        _ev(
            11.0,
            "orchestrator",
            "step",
            orch_sid,
            {
                "api_call_count": 1,
                "prev_tools": [],
                "prompt_token_ids": [101, 102, 103],
                "generation_token_ids": [201, 202],
                "generation_log_probs": [-0.1, -0.2],
            },
        ),
    )
    _write_event(
        base / "orchestrator" / f"{orch_sid}.jsonl",
        _ev(11.5, "orchestrator", "assistant", orch_sid, {"text": "I'll delegate to analyst."}),
    )
    _write_event(
        base / "orchestrator" / f"{orch_sid}.jsonl",
        _ev(20.0, "orchestrator", "session_end", orch_sid, {}),
    )

    # Side-channel delegations file: ticket = analyst session id.
    _write_event(
        base / "orchestrator" / "delegations.jsonl",
        _ev(
            12.0,
            "orchestrator",
            "delegation",
            "delegations",
            {"mode": "blocking", "worker": "analyst", "ticket_id": ticket, "task_preview": "go"},
        ),
    )
    _write_event(
        base / "orchestrator" / "delegations.jsonl",
        _ev(
            17.0,
            "orchestrator",
            "delegation",
            "delegations",
            {"mode": "result", "worker": "analyst", "ticket_id": ticket, "ok": True, "result_preview": "done"},
        ),
    )

    # Analyst session — session_id IS the ticket (worker's session_id =
    # incoming task_id per HermesAgent._resolve_session_id).
    _write_event(
        base / "analyst" / f"{ticket}.jsonl",
        _ev(13.0, "analyst", "session_start", ticket, {"model": "analyst-model", "user_message_preview": "go"}),
    )
    _write_event(
        base / "analyst" / f"{ticket}.jsonl",
        _ev(
            14.0,
            "analyst",
            "step",
            ticket,
            {"api_call_count": 1, "prev_tools": [], "prompt_token_ids": [1], "generation_token_ids": [2]},
        ),
    )
    _write_event(
        base / "analyst" / f"{ticket}.jsonl",
        _ev(15.0, "analyst", "tool_complete", ticket, {"tool_name": "calculator", "result": "42"}),
    )
    _write_event(
        base / "analyst" / f"{ticket}.jsonl",
        _ev(16.0, "analyst", "assistant", ticket, {"text": "result: 42"}),
    )
    _write_event(
        base / "analyst" / f"{ticket}.jsonl",
        _ev(16.5, "analyst", "session_end", ticket, {}),
    )


# ---------------------------------------------------------------------------
# T5.1 — per-agent trajectory carries token IDs + reward field
# ---------------------------------------------------------------------------


def test_t51_per_agent_trajectory_export(tmp_path: Path) -> None:
    """Given a synthesized multi-agent run, exporter produces one JSONL per
    agent, each with token_ids and a reward field (propagated from the
    orchestrator's rollouts.jsonl)."""
    output_dir = tmp_path / "rollout_out"
    out_dir = tmp_path / "trajectories"
    _synthesize_two_agent_run(output_dir, ticket="tkt0001")

    # Rollouts file with one task → reward.
    rollouts_path = output_dir / "rollouts.jsonl"
    rollouts_path.parent.mkdir(parents=True, exist_ok=True)
    rollouts_path.write_text(json.dumps({"task_index": 0, "reward": 1.0}) + "\n")

    written = export_trajectories(output_dir, out_dir, rollouts_path)

    assert set(written) == {"orchestrator", "analyst"}

    orch_entries = [json.loads(line) for line in (out_dir / "orchestrator.jsonl").open()]
    assert len(orch_entries) == 1
    entry = orch_entries[0]
    assert entry["agent"] == "orchestrator"
    assert entry["session_id"] == "orchA"
    assert entry["model"] == "orch-model"
    assert entry["reward"] == 1.0
    # ShareGPT-shaped conversations:
    roles = [t["from"] for t in entry["conversations"]]
    assert "human" in roles
    assert "gpt" in roles
    # Token IDs survived the step event:
    assert entry["token_ids"]["prompt_token_ids"] == [[101, 102, 103]]
    assert entry["token_ids"]["generation_token_ids"] == [[201, 202]]
    assert entry["token_ids"]["generation_log_probs"] == [[-0.1, -0.2]]

    analyst_entries = [json.loads(line) for line in (out_dir / "analyst.jsonl").open()]
    assert len(analyst_entries) == 1
    a = analyst_entries[0]
    assert a["session_id"] == "tkt0001"
    # Worker inherits the orchestrator's reward (propagate-terminal mode).
    assert a["reward"] == 1.0
    a_roles = [t["from"] for t in a["conversations"]]
    assert "tool" in a_roles
    assert "gpt" in a_roles


def test_t51_no_rollouts_means_reward_none(tmp_path: Path) -> None:
    """Without --rollouts, exporter still writes trajectories but reward=None."""
    output_dir = tmp_path / "rollout_out"
    out_dir = tmp_path / "trajectories"
    _synthesize_two_agent_run(output_dir)

    written = export_trajectories(output_dir, out_dir, rollouts_path=None)
    assert set(written) == {"orchestrator", "analyst"}

    for agent in ("orchestrator", "analyst"):
        entries = [json.loads(line) for line in (out_dir / f"{agent}.jsonl").open()]
        for entry in entries:
            assert entry["reward"] is None
            assert "conversations" in entry


def test_build_trajectory_skips_unknown_kinds() -> None:
    """Unrecognized event kinds are dropped from conversations rather than
    crashing (keeps the exporter forward-compatible with future Hermes
    callbacks)."""
    events = [
        {"ts": 1.0, "agent": "a", "session_id": "s", "kind": "session_start", "payload": {"model": "m", "user_message_preview": "hi"}},
        {"ts": 2.0, "agent": "a", "session_id": "s", "kind": "future_kind_xyz", "payload": {"data": 1}},
        {"ts": 3.0, "agent": "a", "session_id": "s", "kind": "assistant", "payload": {"text": "hi"}},
    ]
    traj = _build_trajectory("a", "s", events, reward=0.5)
    kinds_in_conv = [t["from"] for t in traj["conversations"]]
    assert kinds_in_conv == ["human", "gpt"]


# ---------------------------------------------------------------------------
# T5.2 — hermes-agent trajectory_compressor consumes the export
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _trajectory_compressor():
    """Lazily import the compressor module, skipping if unavailable."""
    for candidate in _HERMES_AGENT_CANDIDATES:
        if candidate and Path(candidate).is_dir() and candidate not in sys.path:
            sys.path.insert(0, candidate)
    try:
        import trajectory_compressor  # type: ignore
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"hermes-agent trajectory_compressor not importable: {exc}")
    return trajectory_compressor


def test_t52_trajectory_compressor_handles_export(
    tmp_path: Path, _trajectory_compressor
) -> None:
    """The compressor's schema-handling code path accepts a Hermes export.

    We don't instantiate the full ``TrajectoryCompressor`` here because
    its constructor reaches out to the configured LLM provider (and an
    HF tokenizer) on import — none of which is available in unit-test
    environments.  Instead we exercise the pure ``_find_protected_indices``
    code path directly, with a minimal config stub.  That is enough to
    prove the entry's ``conversations`` shape — role tags + value field —
    is the one the compressor's downstream paths expect.
    """
    output_dir = tmp_path / "rollout_out"
    out_dir = tmp_path / "trajectories"
    _synthesize_two_agent_run(output_dir)
    export_trajectories(output_dir, out_dir, rollouts_path=None)

    entries = [json.loads(line) for line in (out_dir / "orchestrator.jsonl").open()]
    assert entries, "expected at least one trajectory"
    entry = entries[0]
    trajectory = entry["conversations"]

    class _StubConfig:
        protect_first_system = True
        protect_first_human = True
        protect_first_gpt = True
        protect_first_tool = True
        protect_last_n_turns = 4

    class _StubCompressor:
        config = _StubConfig()

    protected, start, end = _trajectory_compressor.TrajectoryCompressor._find_protected_indices(
        _StubCompressor(),  # type: ignore[arg-type]
        trajectory,
    )

    # The compressor must recognise at least one of our turn roles — if
    # the export had drifted to roles the compressor doesn't know about
    # (e.g. "assistant" instead of "gpt"), ``protected`` would be empty.
    assert protected, (
        "compressor did not recognise any role in the exported trajectory; "
        "schema drift between exporter and trajectory_compressor"
    )
    # Sanity: the compressible window is bounded by the trajectory length.
    assert 0 <= start <= len(trajectory)
    assert 0 <= end <= len(trajectory)
