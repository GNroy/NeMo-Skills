# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""SCI-548 — ``worklog_enrich``: join trajectory traces into worklog reports.

Coverage:
  E1  — summarize_session derives task_id from the clock_in event + counts tools
  E2  — error results are counted; args arriving as a JSON string are parsed
  E3  — end-to-end enrich(): matches a worker trace to its report, appends the
        observed table + writes the .tools.json sidecar
  E4  — claim-vs-observed diff (claimed_not_observed / observed_not_claimed)
  E5  — concurrent workers: each session's own file maps to its own task_id
        (the merged-file failure mode the per-child trace wiring avoids)
  E6  — idempotent: re-running replaces the block, never appends a 2nd copy
  E7  — a trace with no clock_in is reported unmatched (can't attribute)
  E8  — trust boundary: long tool results are truncated in the preview
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from nemo_skills.mcp.servers.agentic.worklog_enrich import (
    _ENRICH_BEGIN,
    _ENRICH_END,
    enrich,
    summarize_session,
)


# ---------------------------------------------------------------------------
# Helpers — build trace files + worklog reports the way the real stack does.
# ---------------------------------------------------------------------------


def _ev(session_id: str, kind: str, payload: Dict[str, Any], agent: str = "scientist") -> Dict[str, Any]:
    return {"ts": 1.0, "agent": agent, "session_id": session_id, "kind": kind, "payload": payload}


def _write_trace(trace_dir: Path, session_id: str, events: List[Dict[str, Any]]) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return path


def _write_report(worklog_dir: Path, task_id: str, *, tools_used: Any = None, body: str = "did stuff") -> Path:
    worklog_dir.mkdir(parents=True, exist_ok=True)
    fm: Dict[str, Any] = {"worklog_version": 1, "task_id": task_id, "status": "completed"}
    if tools_used is not None:
        fm["tools_used"] = tools_used
    text = f"---\n{yaml.safe_dump(fm, sort_keys=False)}---\n\n{body}\n"
    path = worklog_dir / f"{task_id}.md"
    path.write_text(text, encoding="utf-8")
    return path


def _clock_in_events(session_id: str, task_id: str) -> List[Dict[str, Any]]:
    return [
        _ev(session_id, "tool_start", {"tc_id": "c1", "tool_name": "mcp_worklog_clock_in",
                                       "args": {"task_id": task_id}}),
        _ev(session_id, "tool_complete", {"tc_id": "c1", "tool_name": "mcp_worklog_clock_in",
                                          "args": {"task_id": task_id}, "result": '{"status": "clocked_in"}'}),
    ]


# ---------------------------------------------------------------------------
# E1 / E2 — summarize_session
# ---------------------------------------------------------------------------


def test_e1_summarize_derives_task_and_counts() -> None:
    sid = "20260610_aaaaaa"
    events = _clock_in_events(sid, "bs-0") + [
        _ev(sid, "tool_start", {"tc_id": "g1", "tool_name": "mcp_benchmark_get_problem",
                                "args": {"id": "bs-0"}}),
        _ev(sid, "tool_complete", {"tc_id": "g1", "tool_name": "mcp_benchmark_get_problem",
                                   "args": {"id": "bs-0"}, "result": '{"id": "bs-0", "prompt": "17+25"}'}),
        _ev(sid, "tool_start", {"tc_id": "g2", "tool_name": "mcp_benchmark_get_problem",
                                "args": {"id": "bs-0"}}),
        _ev(sid, "tool_complete", {"tc_id": "g2", "tool_name": "mcp_benchmark_get_problem",
                                   "args": {"id": "bs-0"}, "result": "ok"}),
    ]
    s = summarize_session(events)
    assert s["task_id"] == "bs-0"
    assert s["session_id"] == sid
    by_name = {o["name"]: o for o in s["observed"]}
    assert by_name["mcp_benchmark_get_problem"]["calls"] == 2
    assert by_name["mcp_benchmark_get_problem"]["errors"] == 0
    assert "17+25" in by_name["mcp_benchmark_get_problem"]["first_result_preview"]


def test_e1b_task_from_clock_off_when_no_clock_in() -> None:
    # The orchestrator may clock_off "batch" without a matching clock_in
    # (lenient close). The join must fall back to the clock_off task_id.
    sid = "orch1"
    events = [
        _ev(sid, "tool_start", {"tc_id": "lb", "tool_name": "mcp_benchmark_load_benchmark", "args": {}}),
        _ev(sid, "tool_start", {"tc_id": "co", "tool_name": "mcp_worklog_clock_off",
                                "args": {"task_id": "batch", "status": "completed"}}),
    ]
    s = summarize_session(events)
    assert s["task_id"] == "batch"  # derived from clock_off, no clock_in present


def test_e2_error_counted_and_string_args_parsed() -> None:
    sid = "s2"
    events = _clock_in_events(sid, "bs-1") + [
        # args delivered as a JSON STRING (some tool-callers do this) — must parse.
        _ev(sid, "tool_start", {"tc_id": "x1", "tool_name": "mcp_benchmark_get_problem",
                                "args": '{"id": "bs-1"}'}),
        _ev(sid, "tool_complete", {"tc_id": "x1", "tool_name": "mcp_benchmark_get_problem",
                                   "args": '{"id": "bs-1"}', "result": "Error: no such problem"}),
    ]
    s = summarize_session(events)
    assert s["task_id"] == "bs-1"
    gp = {o["name"]: o for o in s["observed"]}["mcp_benchmark_get_problem"]
    assert gp["calls"] == 1
    assert gp["errors"] == 1
    assert "bs-1" in gp["first_args_preview"]


# ---------------------------------------------------------------------------
# E3 / E4 — end-to-end enrich + diff
# ---------------------------------------------------------------------------


def test_e3_enrich_matches_and_writes(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces" / "scientist"
    worklog_dir = tmp_path / "worklogs" / "run1"
    _write_trace(trace_dir, "sessA", _clock_in_events("sessA", "bs-0") + [
        _ev("sessA", "tool_start", {"tc_id": "g", "tool_name": "mcp_benchmark_get_problem", "args": {"id": "bs-0"}}),
        _ev("sessA", "tool_complete", {"tc_id": "g", "tool_name": "mcp_benchmark_get_problem",
                                       "args": {"id": "bs-0"}, "result": "prompt text"}),
    ])
    report = _write_report(worklog_dir, "bs-0")

    result = enrich(tmp_path / "traces", worklog_dir)
    assert result["n_matched"] == 1
    text = report.read_text()
    assert "## Tools used (observed)" in text
    assert _ENRICH_BEGIN in text and _ENRICH_END in text
    assert "mcp_benchmark_get_problem" in text

    sidecar = worklog_dir / "bs-0.tools.json"
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text())
    assert payload["task_id"] == "bs-0"
    assert payload["session_id"] == "sessA"
    assert any(o["name"] == "mcp_benchmark_get_problem" for o in payload["observed"])


def test_e4_claim_vs_observed_diff(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    worklog_dir = tmp_path / "worklogs" / "run1"
    _write_trace(trace_dir / "scientist", "sB", _clock_in_events("sB", "bs-2") + [
        _ev("sB", "tool_start", {"tc_id": "g", "tool_name": "mcp_benchmark_get_problem", "args": {"id": "bs-2"}}),
    ])
    # Agent CLAIMS it used a web_search it never actually called, and does NOT
    # mention get_problem which it did call.
    _write_report(worklog_dir, "bs-2", tools_used=[{"name": "web_search", "helped": True}])

    result = enrich(trace_dir, worklog_dir)
    rec = next(r for r in result["records"] if r.get("task_id") == "bs-2")
    assert "web_search" in rec["diff"]["claimed_not_observed"]
    assert "mcp_benchmark_get_problem" in rec["diff"]["observed_not_claimed"]


# ---------------------------------------------------------------------------
# E5 — concurrent workers, one file each (the join the trace wiring enables)
# ---------------------------------------------------------------------------


def test_e5_concurrent_workers_distinct_files(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces" / "scientist"
    worklog_dir = tmp_path / "worklogs" / "run1"
    # Two workers ran concurrently — each wrote its OWN <session_id>.jsonl
    # (per-child trace wiring), so each clock_in→task_id stays unambiguous.
    _write_trace(trace_dir, "w0", _clock_in_events("w0", "bs-0"))
    _write_trace(trace_dir, "w1", _clock_in_events("w1", "bs-1"))
    _write_report(worklog_dir, "bs-0")
    _write_report(worklog_dir, "bs-1")

    result = enrich(tmp_path / "traces", worklog_dir)
    assert result["n_matched"] == 2
    matched = {r["task_id"]: r["session_id"] for r in result["records"] if r["matched"]}
    assert matched == {"bs-0": "w0", "bs-1": "w1"}


# ---------------------------------------------------------------------------
# E6 — idempotency
# ---------------------------------------------------------------------------


def test_e6_idempotent_reenrich(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    worklog_dir = tmp_path / "worklogs" / "run1"
    _write_trace(trace_dir / "scientist", "s", _clock_in_events("s", "bs-0"))
    report = _write_report(worklog_dir, "bs-0")

    enrich(trace_dir, worklog_dir)
    enrich(trace_dir, worklog_dir)  # second pass
    text = report.read_text()
    assert text.count(_ENRICH_BEGIN) == 1  # exactly one block, not two
    assert text.count("## Tools used (observed)") == 1


# ---------------------------------------------------------------------------
# E7 — trace with no clock_in is unmatched
# ---------------------------------------------------------------------------


def test_e7_no_clock_in_is_unmatched(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    worklog_dir = tmp_path / "worklogs" / "run1"
    _write_trace(trace_dir / "scientist", "orphan", [
        _ev("orphan", "tool_start", {"tc_id": "g", "tool_name": "some_tool", "args": {}}),
    ])
    _write_report(worklog_dir, "bs-0")

    result = enrich(trace_dir, worklog_dir)
    assert result["n_matched"] == 0
    assert result["n_unmatched_traces"] == 1


# ---------------------------------------------------------------------------
# E8 — long results are truncated (bounded previews / trust boundary)
# ---------------------------------------------------------------------------


def test_e9_failed_write_leaves_report_intact(tmp_path: Path, monkeypatch) -> None:
    # A write failure (e.g. disk-quota EDQUOT) must NOT truncate/corrupt the
    # agent's existing report — the atomic temp+replace guarantees it.
    import nemo_skills.mcp.servers.agentic.worklog_enrich as we

    trace_dir = tmp_path / "traces"
    worklog_dir = tmp_path / "worklogs" / "run1"
    _write_trace(trace_dir / "scientist", "s", _clock_in_events("s", "bs-0"))
    report = _write_report(worklog_dir, "bs-0", body="ORIGINAL REPORT BODY")
    original = report.read_text()

    real_replace = os.replace

    def boom(src, dst):  # simulate EDQUOT on the atomic swap
        raise OSError(122, "Disk quota exceeded")

    monkeypatch.setattr(we.os, "replace", boom)
    enrich(trace_dir, worklog_dir)  # must not raise, must not corrupt
    monkeypatch.setattr(we.os, "replace", real_replace)

    assert report.read_text() == original  # untouched
    assert "ORIGINAL REPORT BODY" in report.read_text()
    # No leftover temp files.
    assert not list(worklog_dir.glob("*.enrich.tmp.*"))


def test_e8_result_preview_truncated(tmp_path: Path) -> None:
    sid = "s8"
    big = "Z" * 5000
    events = _clock_in_events(sid, "bs-0") + [
        _ev(sid, "tool_start", {"tc_id": "g", "tool_name": "t", "args": {}}),
        _ev(sid, "tool_complete", {"tc_id": "g", "tool_name": "t", "args": {}, "result": big}),
    ]
    s = summarize_session(events, max_preview=120)
    preview = {o["name"]: o for o in s["observed"]}["t"]["first_result_preview"]
    assert len(preview) < 200
    assert "chars]" in preview  # truncation marker present
