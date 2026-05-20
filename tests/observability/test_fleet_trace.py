# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Phase 3 tests for the multi-agent Hermes JSONL trace pipeline.

Covers:
  - T3.1 every event written by Phase-0 callbacks has the required fields
  - T3.2 delegation events link orchestrator and worker via ticket_id
  - T3.3 fleet renderer orders by ts across agents
  - T3.4 token_ids preserved in step events when save_trajectories is on
  - T3.5 trace files truncate safely on mid-write process kill
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# NeMo-Gym lives in a sibling repo; tests that touch the Phase 0 trace
# writer skip cleanly when it's not on the path.  The conditional path
# probe lets a local dev run them without explicit PYTHONPATH setup.
_NEMO_GYM_CANDIDATES = [
    os.environ.get("NEMO_GYM_PATH", ""),
    "/home/alaptev/Projects/NeMo-Gym",
    str(Path(__file__).resolve().parents[3] / "NeMo-Gym"),
]
for _candidate in _NEMO_GYM_CANDIDATES:
    if _candidate and Path(_candidate).is_dir() and _candidate not in sys.path:
        sys.path.insert(0, _candidate)
        break

try:
    from responses_api_agents.hermes_agent import trace_writer as _trace_writer  # type: ignore
except Exception:  # noqa: BLE001 - any import failure → skip Phase-0 cases
    _trace_writer = None


from nemo_skills.scripts.render_hermes_fleet_trace import (
    _events_linked_to_ticket,
    _iter_trace_events,
    main as render_main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_event(path: Path, event: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _make_event(
    agent: str,
    kind: str,
    *,
    ts: float,
    session_id: str = "sess_test",
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "ts": ts,
        "agent": agent,
        "session_id": session_id,
        "kind": kind,
        "payload": payload or {},
    }


# ---------------------------------------------------------------------------
# T3.1 — Phase-0 callbacks emit the required fields
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_trace_writer is None, reason="NeMo-Gym trace_writer not on sys.path")
def test_t31_event_schema_fields_required(tmp_path: Path) -> None:
    """Every event from build_callbacks(...) must carry ts/agent/session_id/kind/payload."""
    _trace_writer._reset_writers_for_tests()
    session_id = "sess_t31"
    agent_name = "orchestrator"
    trace_dir = tmp_path / "traces" / agent_name
    cb = _trace_writer.build_callbacks(trace_dir, agent_name, session_id)

    cb["tool_start_callback"]("tc1", "echo", {"msg": "hi"})
    cb["tool_complete_callback"]("tc1", "echo", {"msg": "hi"}, "result")
    cb["step_callback"](3, ["echo"])
    cb["thinking_callback"]("planning")
    cb["status_callback"]("info", "ok")
    _trace_writer.emit_event(trace_dir, agent_name, session_id, "session_start", {"model": "x"})
    _trace_writer.emit_event(trace_dir, agent_name, session_id, "session_end", {})

    trace_file = trace_dir / f"{session_id}.jsonl"
    assert trace_file.exists()
    required = {"ts", "agent", "session_id", "kind", "payload"}
    with trace_file.open() as f:
        events = [json.loads(line) for line in f if line.strip()]
    assert events, "expected non-empty trace"
    for event in events:
        assert required.issubset(event.keys()), event
        assert event["agent"] == agent_name
        assert event["session_id"] == session_id
        assert isinstance(event["ts"], (int, float))


# ---------------------------------------------------------------------------
# T3.2 — delegation events link orchestrator and worker
# ---------------------------------------------------------------------------


def test_t32_delegation_events_link_orchestrator_and_worker(tmp_path: Path) -> None:
    """The ticket_id on orchestrator's delegation matches the worker's session_id."""
    ticket = uuid.uuid4().hex
    other_ticket = uuid.uuid4().hex
    base = tmp_path / "traces"

    # Orchestrator emits delegation dispatch + result with ticket_id
    _write_event(
        base / "orchestrator" / "delegations.jsonl",
        _make_event(
            "orchestrator",
            "delegation",
            ts=10.0,
            session_id="delegations",
            payload={"mode": "blocking", "worker": "analyst", "ticket_id": ticket, "task_preview": "analyze X"},
        ),
    )
    _write_event(
        base / "orchestrator" / "delegations.jsonl",
        _make_event(
            "orchestrator",
            "delegation",
            ts=20.0,
            session_id="delegations",
            payload={"mode": "result", "worker": "analyst", "ticket_id": ticket, "ok": True, "result_preview": "X is foo"},
        ),
    )
    # Unrelated delegation in the same file
    _write_event(
        base / "orchestrator" / "delegations.jsonl",
        _make_event(
            "orchestrator",
            "delegation",
            ts=25.0,
            session_id="delegations",
            payload={"mode": "blocking", "worker": "coder", "ticket_id": other_ticket, "task_preview": "code Y"},
        ),
    )

    # Worker uses ticket as its session_id (Hermes ``/task`` adapter behaviour)
    _write_event(
        base / "analyst" / f"{ticket}.jsonl",
        _make_event("analyst", "session_start", ts=11.0, session_id=ticket, payload={"model": "m"}),
    )
    _write_event(
        base / "analyst" / f"{ticket}.jsonl",
        _make_event("analyst", "assistant", ts=15.0, session_id=ticket, payload={"text": "X is foo"}),
    )
    _write_event(
        base / "analyst" / f"{ticket}.jsonl",
        _make_event("analyst", "session_end", ts=19.0, session_id=ticket, payload={}),
    )

    events = list(_iter_trace_events(tmp_path))
    linked = _events_linked_to_ticket(events, ticket)

    agents_in_linked = {e["agent"] for e in linked}
    assert "orchestrator" in agents_in_linked
    assert "analyst" in agents_in_linked
    # Exactly the analyst session events plus the two orchestrator delegations
    # for THIS ticket — the coder delegation is excluded.
    kinds = sorted((e["agent"], e["kind"]) for e in linked)
    assert ("orchestrator", "delegation") in kinds
    assert ("analyst", "session_start") in kinds
    assert ("analyst", "session_end") in kinds
    # The unrelated ticket's delegation must not leak into the linked view
    assert all(
        (e.get("payload") or {}).get("ticket_id") != other_ticket
        for e in linked
        if e["kind"] == "delegation"
    )


# ---------------------------------------------------------------------------
# T3.3 — renderer orders by ts
# ---------------------------------------------------------------------------


def test_t33_fleet_renderer_orders_by_ts(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Out-of-order writes across two agents merge into a ts-monotonic stream."""
    base = tmp_path / "traces"
    # Write deliberately out of order.
    _write_event(base / "orchestrator" / "s.jsonl", _make_event("orchestrator", "step", ts=30.0))
    _write_event(base / "orchestrator" / "s.jsonl", _make_event("orchestrator", "step", ts=10.0))
    _write_event(base / "analyst" / "s.jsonl", _make_event("analyst", "step", ts=20.0))
    _write_event(base / "analyst" / "s.jsonl", _make_event("analyst", "step", ts=5.0))

    rc = render_main([str(tmp_path), "--format", "json", "--no-color"])
    assert rc == 0
    captured = capsys.readouterr()
    rendered_events = [json.loads(line) for line in captured.out.splitlines() if line.strip()]
    timestamps = [e["ts"] for e in rendered_events]
    assert timestamps == sorted(timestamps)
    assert timestamps == [5.0, 10.0, 20.0, 30.0]
    # JSON mode strips the internal provenance key
    for event in rendered_events:
        assert "_source_file" not in event


# ---------------------------------------------------------------------------
# T3.4 — token_ids preserved in step events
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_trace_writer is None, reason="NeMo-Gym trace_writer not on sys.path")
def test_t34_token_ids_preserved_in_step_events(tmp_path: Path) -> None:
    """When Hermes passes prompt_token_ids / generation_token_ids, they survive in the trace."""
    _trace_writer._reset_writers_for_tests()
    session_id = "sess_t34"
    agent_name = "orchestrator"
    trace_dir = tmp_path / "traces" / agent_name
    cb = _trace_writer.build_callbacks(trace_dir, agent_name, session_id)

    cb["step_callback"](
        2,
        ["echo"],
        prompt_token_ids=[1, 2, 3, 4],
        generation_token_ids=[10, 11, 12],
        generation_log_probs=[-0.1, -0.2, -0.3],
    )

    trace_file = trace_dir / f"{session_id}.jsonl"
    with trace_file.open() as f:
        events = [json.loads(line) for line in f if line.strip()]
    step_events = [e for e in events if e["kind"] == "step"]
    assert len(step_events) == 1
    payload = step_events[0]["payload"]
    assert payload["api_call_count"] == 2
    assert payload["prev_tools"] == ["echo"]
    assert payload["prompt_token_ids"] == [1, 2, 3, 4]
    assert payload["generation_token_ids"] == [10, 11, 12]
    assert payload["generation_log_probs"] == [-0.1, -0.2, -0.3]


# ---------------------------------------------------------------------------
# T3.5 — trace files truncate safely
# ---------------------------------------------------------------------------


def test_t35_trace_files_truncate_safely(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A mid-write kill leaves a partial trailing line; the renderer skips it and continues."""
    trace_file = tmp_path / "traces" / "orchestrator" / "s.jsonl"
    trace_file.parent.mkdir(parents=True, exist_ok=True)

    good_a = _make_event("orchestrator", "step", ts=1.0, payload={"i": 1})
    good_b = _make_event("orchestrator", "step", ts=2.0, payload={"i": 2})
    # Simulate a kill in the middle of writing the next event: write a
    # complete line, then a complete line, then a truncated line.
    with trace_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(good_a) + "\n")
        f.write(json.dumps(good_b) + "\n")
        f.write('{"ts": 3.0, "agent": "orchestrator", "session_id": "sess_test", "kind": "step", "payload": {')

    # The renderer must yield the good events and warn (not raise) on the
    # partial line.
    rc = render_main([str(tmp_path), "--format", "json", "--no-color"])
    assert rc == 0
    captured = capsys.readouterr()
    lines = [json.loads(line) for line in captured.out.splitlines() if line.strip()]
    assert [event["payload"]["i"] for event in lines] == [1, 2]
    assert "malformed" in captured.err.lower() or "skipping" in captured.err.lower()


# ---------------------------------------------------------------------------
# CallAgentTool delegation emission
# ---------------------------------------------------------------------------


def test_call_agent_tool_emits_delegation_events(tmp_path: Path) -> None:
    """When delegation_trace_dir is set, dispatch + injection write delegation events."""
    from nemo_skills.mcp.servers.agent_tool import CallAgentTool

    tool = CallAgentTool()
    tool.configure(
        {
            "agents": {"analyst": {"host": "127.0.0.1", "port": 65530}},
            "delegation_trace_dir": str(tmp_path / "traces"),
            "delegation_agent_name": "orchestrator",
        }
    )
    tool._emit_delegation(
        {"mode": "blocking", "worker": "analyst", "ticket_id": "abc123", "task_preview": "go"}
    )
    tool._emit_delegation(
        {"mode": "result", "worker": "analyst", "ticket_id": "abc123", "ok": True, "result_preview": "done"}
    )

    delegation_file = tmp_path / "traces" / "orchestrator" / "delegations.jsonl"
    assert delegation_file.exists()
    with delegation_file.open() as f:
        events = [json.loads(line) for line in f if line.strip()]
    assert len(events) == 2
    assert all(event["kind"] == "delegation" for event in events)
    assert all(event["agent"] == "orchestrator" for event in events)
    assert [event["payload"]["mode"] for event in events] == ["blocking", "result"]
    assert all(event["payload"]["ticket_id"] == "abc123" for event in events)


def test_call_agent_tool_emit_is_noop_without_trace_dir() -> None:
    """When delegation_trace_dir is unset, _emit_delegation is a silent no-op."""
    from nemo_skills.mcp.servers.agent_tool import CallAgentTool

    tool = CallAgentTool()
    tool.configure({"agents": {"analyst": {"host": "127.0.0.1", "port": 65530}}})
    # Must not raise even when called repeatedly.
    tool._emit_delegation({"mode": "blocking", "worker": "analyst", "ticket_id": "x"})
    tool._emit_delegation({"mode": "result", "worker": "analyst", "ticket_id": "x", "ok": True})
