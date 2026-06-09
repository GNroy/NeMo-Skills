# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""SCI-548 P1 — ``worklog`` clock-in/clock-off MCP tool.

Coverage:
  In-process (fast, direct Tool calls):
    W1  — clock_in → clock_off writes a report with correct frontmatter + body
    W2  — frontmatter is valid YAML and round-trips; body is recoverable
    W3  — run_id / agent_id / subdir overrides shape the path + frontmatter
    W4  — per-call agent_id override lands in the frontmatter
    W5  — invalid status raises (surfaces as a tool error via the wrapper)
    W6  — missing/empty task_id raises
    W7  — clock_off with no prior clock_in is lenient (no timing, warning)
    W8  — double clock_off → already_closed, report not overwritten
    W9  — re-entrant clock_in → already_open, original start preserved
    W10 — _sweep_sync flushes an open timer to disk as error/shutdown_sweep
    W11 — async shutdown() hook performs the same sweep
    W12 — hostile task_id is sanitized; file stays inside the run dir
    W13 — list_tools advertises clock_in/clock_off with the right shape
    W14 — composes under ToolManager (qualified names)
  Over real stdio (ns-mcp-serve wrapper subprocess):
    W15 — list_tools round-trip
    W16 — clock_in + clock_off round-trip writes the file on disk
    W17 — un-closed timer is swept to disk when the server process exits (EOF)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import yaml
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from nemo_skills.mcp.servers.agentic.worklog_tool import VALID_STATUSES, WorklogTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _parse_worklog(text: str) -> Tuple[Dict[str, Any], str]:
    """Split a worklog markdown file into (frontmatter dict, body str).

    maxsplit=2 consumes only the opening/closing frontmatter fences, so any
    ``---`` inside the body is preserved.
    """
    assert text.startswith("---\n"), f"missing frontmatter fence: {text[:40]!r}"
    _, fm_text, body = text.split("---\n", 2)
    return yaml.safe_load(fm_text), body


def _make_tool(tmp_path: Path, **overrides: Any) -> WorklogTool:
    tool = WorklogTool()
    cfg = {
        "worklog_dir": str(tmp_path),
        "run_id": "r1",
        "agent_id": "a1",
        "install_signal_handlers": False,  # never touch process signals in-process
    }
    cfg.update(overrides)
    tool.configure(cfg, {})
    return tool


# ---------------------------------------------------------------------------
# W1 / W2 — happy path + frontmatter integrity
# ---------------------------------------------------------------------------


def test_w1_clock_in_off_writes_report(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    ci = _run(tool.execute("clock_in", {"task_id": "fs-0042", "task_description": "solve problem 42"}))
    assert ci["status"] == "clocked_in"
    assert ci["task_id"] == "fs-0042"

    report = "## What I did\nTried X.\n\n## Final answer\n42"
    co = _run(tool.execute("clock_off", {"task_id": "fs-0042", "status": "completed", "report": report}))
    assert co["status"] == "clocked_off"
    assert co["self_reported_status"] == "completed"
    assert co["elapsed_s"] is not None and co["elapsed_s"] >= 0

    path = Path(co["report_path"])
    assert path == tmp_path / "r1" / "fs-0042.md"
    assert path.is_file()


def test_w2_frontmatter_roundtrips(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    _run(tool.execute("clock_in", {"task_id": "t1", "task_description": "desc: with colon"}))
    report = "body line\n---\nstill body after an hr"
    co = _run(tool.execute("clock_off", {"task_id": "t1", "status": "completed", "report": report}))

    fm, body = _parse_worklog(Path(co["report_path"]).read_text())
    assert fm["task_id"] == "t1"
    assert fm["agent_id"] == "a1"
    assert fm["run_id"] == "r1"
    assert fm["status"] == "completed"
    assert fm["closed_by"] == "self"
    assert fm["worklog_version"] == 1
    assert fm["task_description"] == "desc: with colon"  # colon survived YAML quoting
    assert isinstance(fm["elapsed_s"], (int, float))
    assert fm["started_at"] and fm["started_at"].endswith("Z")
    assert fm["ended_at"] and fm["ended_at"].endswith("Z")
    # Body preserved verbatim, including the inner '---'.
    assert "still body after an hr" in body


# ---------------------------------------------------------------------------
# W3 / W4 — config + per-call identity shape the path/frontmatter
# ---------------------------------------------------------------------------


def test_w3_run_subdir_agent_overrides(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path, run_id="run9", agent_id="worker-7", subdir="workers/chunk_3")
    _run(tool.execute("clock_in", {"task_id": "p9"}))
    co = _run(tool.execute("clock_off", {"task_id": "p9", "status": "completed", "report": "ok"}))
    path = Path(co["report_path"])
    assert path == tmp_path / "run9" / "workers" / "chunk_3" / "p9.md"
    fm, _ = _parse_worklog(path.read_text())
    assert fm["agent_id"] == "worker-7"
    assert fm["run_id"] == "run9"


def test_w4_per_call_agent_id(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    _run(tool.execute("clock_in", {"task_id": "p1", "agent_id": "worker-99"}))
    co = _run(tool.execute("clock_off", {"task_id": "p1", "status": "completed", "report": "ok"}))
    assert co["agent_id"] == "worker-99"
    fm, _ = _parse_worklog(Path(co["report_path"]).read_text())
    assert fm["agent_id"] == "worker-99"


# ---------------------------------------------------------------------------
# W5 / W6 — contract validation
# ---------------------------------------------------------------------------


def test_w5_invalid_status_raises(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    _run(tool.execute("clock_in", {"task_id": "p1"}))
    with pytest.raises(ValueError, match="status must be one of"):
        _run(tool.execute("clock_off", {"task_id": "p1", "status": "done", "report": "x"}))


def test_w6_missing_task_id_raises(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    with pytest.raises(ValueError, match="task_id is required"):
        _run(tool.execute("clock_in", {"task_description": "no id"}))
    with pytest.raises(ValueError, match="task_id is required"):
        _run(tool.execute("clock_off", {"task_id": "  ", "status": "completed"}))


@pytest.mark.parametrize("status", list(VALID_STATUSES))
def test_w5b_all_valid_statuses_accepted(tmp_path: Path, status: str) -> None:
    tool = _make_tool(tmp_path)
    _run(tool.execute("clock_in", {"task_id": status}))
    co = _run(tool.execute("clock_off", {"task_id": status, "status": status.upper(), "report": "x"}))
    assert co["self_reported_status"] == status  # normalized to lowercase


# ---------------------------------------------------------------------------
# W7 / W8 / W9 — edge cases
# ---------------------------------------------------------------------------


def test_w7_clock_off_without_clock_in_is_lenient(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    co = _run(tool.execute("clock_off", {"task_id": "orphan", "status": "error", "report": "huh"}))
    assert co["status"] == "clocked_off"
    assert co["elapsed_s"] is None
    assert "no matching clock_in" in co["warning"]
    fm, _ = _parse_worklog(Path(co["report_path"]).read_text())
    assert fm["started_at"] is None
    assert fm["elapsed_s"] is None


def test_w8_double_clock_off_does_not_overwrite(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    _run(tool.execute("clock_in", {"task_id": "p1"}))
    co1 = _run(tool.execute("clock_off", {"task_id": "p1", "status": "completed", "report": "first"}))
    original = Path(co1["report_path"]).read_text()

    co2 = _run(tool.execute("clock_off", {"task_id": "p1", "status": "error", "report": "second"}))
    assert co2["status"] == "already_closed"
    assert co2["report_path"] == co1["report_path"]
    assert Path(co1["report_path"]).read_text() == original  # untouched


def test_w9_reentrant_clock_in_keeps_start(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    ci1 = _run(tool.execute("clock_in", {"task_id": "p1"}))
    ci2 = _run(tool.execute("clock_in", {"task_id": "p1", "task_description": "again"}))
    assert ci2["status"] == "already_open"
    assert ci2["started_at"] == ci1["started_at"]


# ---------------------------------------------------------------------------
# W10 / W11 — guaranteed close via sweep
# ---------------------------------------------------------------------------


def test_w10_sweep_flushes_open_timer(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    _run(tool.execute("clock_in", {"task_id": "stuck", "task_description": "never closed"}))
    written = tool._sweep_sync("shutdown_sweep")
    assert len(written) == 1

    path = Path(written[0])
    assert path == tmp_path / "r1" / "stuck.md"
    fm, body = _parse_worklog(path.read_text())
    assert fm["status"] == "error"
    assert fm["closed_by"] == "shutdown_sweep"
    assert fm["elapsed_s"] is not None  # had a clock_in, so timing is known
    assert "Auto-generated stub" in body
    assert "never closed" in body  # description preserved

    # Idempotent: a second sweep writes nothing more.
    assert tool._sweep_sync("shutdown_sweep") == []


def test_w11_async_shutdown_sweeps(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    _run(tool.execute("clock_in", {"task_id": "stuck"}))
    _run(tool.shutdown())
    fm, _ = _parse_worklog((tmp_path / "r1" / "stuck.md").read_text())
    assert fm["status"] == "error"
    assert fm["closed_by"] == "shutdown_sweep"


def test_w11b_sweep_skips_already_closed(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    _run(tool.execute("clock_in", {"task_id": "done"}))
    _run(tool.execute("clock_off", {"task_id": "done", "status": "completed", "report": "ok"}))
    # Nothing open → sweep writes nothing and does not clobber the report.
    assert tool._sweep_sync("shutdown_sweep") == []
    fm, _ = _parse_worklog((tmp_path / "r1" / "done.md").read_text())
    assert fm["status"] == "completed"
    assert fm["closed_by"] == "self"


# ---------------------------------------------------------------------------
# W12 — path-traversal safety
# ---------------------------------------------------------------------------


def test_w12_task_id_path_traversal_is_contained(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    _run(tool.execute("clock_in", {"task_id": "../../etc/passwd"}))
    co = _run(tool.execute("clock_off", {"task_id": "../../etc/passwd", "status": "error", "report": "x"}))
    path = Path(co["report_path"]).resolve()
    run_dir = (tmp_path / "r1").resolve()
    assert run_dir in path.parents, f"{path} escaped {run_dir}"
    # Real task_id is still recorded faithfully in the frontmatter.
    fm, _ = _parse_worklog(path.read_text())
    assert fm["task_id"] == "../../etc/passwd"


# ---------------------------------------------------------------------------
# W13 / W14 — schema + ToolManager composition
# ---------------------------------------------------------------------------


def test_w13_list_tools_shape(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    entries = {e["name"]: e for e in _run(tool.list_tools())}
    assert set(entries) == {"clock_in", "clock_off"}
    assert entries["clock_in"]["input_schema"]["required"] == ["task_id"]
    off = entries["clock_off"]["input_schema"]
    assert off["required"] == ["task_id", "status"]
    assert off["properties"]["status"]["enum"] == list(VALID_STATUSES)


def test_w14_composes_under_tool_manager(tmp_path: Path) -> None:
    from nemo_skills.mcp.tool_manager import ToolManager

    # ToolManager.locate() uses the '::' separator (in-process convention),
    # unlike the ns-mcp-serve stdio wrapper which uses ':'.
    mgr = ToolManager(
        ["nemo_skills.mcp.servers.agentic.worklog_tool::WorklogTool"],
        overrides={"WorklogTool": {"worklog_dir": str(tmp_path), "run_id": "r1", "install_signal_handlers": False}},
    )
    listed = _run(mgr.list_all_tools())
    names = sorted(t["name"] for t in listed)
    assert names == ["clock_in", "clock_off"]
    _run(mgr.execute_tool("clock_in", {"task_id": "p1"}))
    _run(mgr.execute_tool("clock_off", {"task_id": "p1", "status": "completed", "report": "ok"}))
    assert (tmp_path / "r1" / "p1.md").is_file()


# ---------------------------------------------------------------------------
# W15 / W16 / W17 — real stdio round-trips through ns-mcp-serve
# ---------------------------------------------------------------------------

_SPEC = "nemo_skills.mcp.servers.agentic.worklog_tool:WorklogTool"


def _server_params(overrides: Dict[str, Any]) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "nemo_skills.mcp.stdio_serve", _SPEC, "--overrides", json.dumps(overrides)],
        env=os.environ.copy(),
    )


def _wait_for_file(path: Path, timeout: float = 25.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            return True
        time.sleep(0.1)
    return False


async def _list_over_stdio(overrides: Dict[str, Any]):
    async with stdio_client(_server_params(overrides)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.list_tools()


def test_w15_list_tools_over_stdio(tmp_path: Path) -> None:
    overrides = {"worklog_dir": str(tmp_path), "run_id": "s1", "agent_id": "stdio"}
    listed = asyncio.run(_list_over_stdio(overrides))
    assert sorted(t.name for t in listed.tools) == ["clock_in", "clock_off"]


async def _happy_path_over_stdio(overrides: Dict[str, Any]):
    async with stdio_client(_server_params(overrides)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool("clock_in", {"task_id": "t1", "task_description": "d"})
            return await session.call_tool("clock_off", {"task_id": "t1", "status": "completed", "report": "done"})


def test_w16_clock_off_over_stdio_writes_file(tmp_path: Path) -> None:
    overrides = {"worklog_dir": str(tmp_path), "run_id": "s2", "agent_id": "stdio"}
    result = asyncio.run(_happy_path_over_stdio(overrides))
    text = next(b.text for b in result.content if b.type == "text")
    payload = json.loads(text)
    assert payload["status"] == "clocked_off"

    path = tmp_path / "s2" / "t1.md"
    assert path.is_file()
    fm, _ = _parse_worklog(path.read_text())
    assert fm["status"] == "completed" and fm["agent_id"] == "stdio"


async def _open_then_disconnect(overrides: Dict[str, Any]):
    async with stdio_client(_server_params(overrides)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool("clock_in", {"task_id": "t1", "task_description": "left open"})
    # Context exit closes stdin → server EOFs and exits → atexit/signal sweep.


def test_w17_unclosed_timer_swept_on_process_exit(tmp_path: Path) -> None:
    overrides = {"worklog_dir": str(tmp_path), "run_id": "s3", "agent_id": "stdio"}
    asyncio.run(_open_then_disconnect(overrides))

    path = tmp_path / "s3" / "t1.md"
    assert _wait_for_file(path), "sweep did not write the report after the server exited"
    fm, body = _parse_worklog(path.read_text())
    assert fm["status"] == "error"
    assert fm["closed_by"] == "shutdown_sweep"
    assert "left open" in body
