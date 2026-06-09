# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``worklog`` — a clock-in / clock-off "punch clock" MCP tool (SCI-548 P1).

Every agent (orchestrator *and*, later, swarm workers) brackets a unit of
work with two calls:

- ``clock_in(task_id, task_description)`` — open a timer.
- ``clock_off(task_id, status, report)`` — close it; write a markdown report.

On ``clock_off`` the tool writes one markdown file per task with YAML
frontmatter (the machine-readable record) followed by the agent's free-form
report (the human/LLM-readable narration). ``reflect`` (P3) reads these files
off disk, so they must be persisted — not just returned inline.

**Guaranteed close.** A ``clock_in`` must never dangle. Defence in depth, from
the most robust to the most graceful:

- **Eager stub (survives ``SIGKILL``).** ``clock_in`` immediately writes the
  report file with ``status: in_progress`` / ``closed_by: pending``. If the
  process is killed abruptly before ``clock_off`` — e.g. the MCP SDK spawns the
  stdio server with ``setsid()`` so it escapes the parent cgroup, and an
  orphaned subprocess gets ``SIGKILL``ed without a catchable signal when the
  container is torn down — the stub still records that the task started. This is
  the *only* mechanism that survives ``SIGKILL``; the two below are graceful
  upgrades that finalise the record.
- ``atexit`` — covers normal interpreter exit, including the common case where
  the MCP client closes the stdio connection (EOF → ``main()`` returns → atexit);
  flushes still-open timers to ``status: error`` / ``closed_by: shutdown_sweep``.
- a ``SIGTERM`` / ``SIGHUP`` handler — covers walltime kills, which would not
  otherwise run ``atexit``.

(``batch_solve`` adds a fourth mechanism in P2: a per-worker timeout that
backfills a ``clock_off`` with ``closed_by: batch_solve_backfill``.)

A task's on-disk record thus progresses: ``in_progress`` (clock_in) →
``completed``/``early_exit``/… (clock_off) or ``error``/``shutdown_sweep``
(sweep). The file is rewritten atomically at each step.

Report layout (``§3`` of the design doc)::

    <worklog_dir>/<run_id>/<subdir>/<task_id>.md

``subdir`` is configurable so P2 can group worker reports under
``workers/chunk_<k>/`` while the orchestrator writes to ``orchestrator/``.

Run as a Group-B in-process Tool via the generic stdio wrapper::

    ns-mcp-serve nemo_skills.mcp.servers.agentic.worklog_tool:WorklogTool \\
        --overrides '{"worklog_dir": "/path/run_artifacts", "run_id": "run1", "agent_id": "orchestrator"}'

Config (``--overrides`` JSON; each also falls back to an env var):

==================  =========================  ============================================
key                 env fallback               default
==================  =========================  ============================================
``worklog_dir``     ``NS_WORKLOG_DIR``         ``"worklogs"`` (cwd-relative)
``run_id``          ``NS_WORKLOG_RUN_ID``      ``"run-<UTC timestamp>"`` (auto)
``agent_id``        ``NS_WORKLOG_AGENT_ID``    ``"agent"``
``subdir``          ``NS_WORKLOG_SUBDIR``      ``""`` (flat under the run dir)
``install_...``     ``-``                      ``True`` (atexit + signal sweep)
==================  =========================  ============================================

Set ``run_id`` explicitly (env or override) when several worklog servers must
share one run directory; the auto default is per-process.
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from nemo_skills.mcp.tool_manager import Tool

logger = logging.getLogger(__name__)

WORKLOG_VERSION = 1
VALID_STATUSES = ("completed", "early_exit", "timeout", "error")
# Written on the eager stub at clock_in (before any clock_off). NOT a valid
# clock_off self-assessment: it only remains on disk if the worker was killed
# between clock_in and clock_off/sweep.
STATUS_IN_PROGRESS = "in_progress"
CLOSED_BY_PENDING = "pending"
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _utcnow_iso() -> str:
    """Wall-clock UTC timestamp, ISO-8601 with a ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_run_id() -> str:
    return "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_component(value: Any, fallback: str = "unnamed", maxlen: int = 200) -> str:
    """Sanitize an arbitrary string into one safe path component.

    Keeps ``[A-Za-z0-9._-]``; everything else collapses to ``_``. Strips
    leading/trailing dots/underscores and rejects ``.``/``..`` so a hostile
    ``task_id`` (e.g. ``"../../etc/passwd"``) cannot escape the run directory.
    """
    s = _SAFE_RE.sub("_", str(value if value is not None else "").strip())
    s = s.strip("._")
    if not s or s in (".", ".."):
        s = fallback
    return s[:maxlen]


class WorklogTool(Tool):
    """In-process ``Tool`` exposing ``clock_in`` / ``clock_off``.

    State (open timers, closed-task ledger) lives on the instance and persists
    for the life of the MCP server process. Open timers are keyed by
    ``task_id``; within one server process task ids are assumed unique per
    concurrently-open timer (true for our design: problem ids are unique and
    the orchestrator's own tasks are named ``"batch"`` / ``"reflect"``).
    """

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {
            "worklog_dir": None,
            "run_id": None,
            "agent_id": None,
            "subdir": None,
            "install_signal_handlers": None,
        }
        self._configured = False
        self._open: Dict[str, Dict[str, Any]] = {}
        self._closed: set[str] = set()
        self._closed_paths: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._sweeping = False
        self._handlers_installed = False
        self._prev_handlers: Dict[int, Any] = {}

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def default_config(self) -> Dict[str, Any]:
        return dict(self._config)

    def configure(self, overrides: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(self._config)
        if overrides:
            cfg.update(overrides)

        cfg["worklog_dir"] = cfg.get("worklog_dir") or os.environ.get("NS_WORKLOG_DIR") or "worklogs"
        cfg["run_id"] = cfg.get("run_id") or os.environ.get("NS_WORKLOG_RUN_ID") or _default_run_id()
        cfg["agent_id"] = cfg.get("agent_id") or os.environ.get("NS_WORKLOG_AGENT_ID") or "agent"
        subdir = cfg.get("subdir")
        if subdir is None:
            subdir = os.environ.get("NS_WORKLOG_SUBDIR") or ""
        cfg["subdir"] = subdir
        if cfg.get("install_signal_handlers") is None:
            cfg["install_signal_handlers"] = True

        self._config = cfg
        self._configured = True

    def post_configure(self) -> None:
        self._ensure_configured()
        if self._config.get("install_signal_handlers"):
            self._install_signal_handlers()

    async def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "clock_in",
                "description": (
                    "Open a work timer for a task. Call this BEFORE starting a unit of "
                    "work, then call clock_off when done. Records the start time so "
                    "elapsed time is measured for you."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Stable identifier for this unit of work (e.g. a problem id, or 'batch'/'reflect').",
                        },
                        "task_description": {
                            "type": "string",
                            "description": "Short human-readable description of what this task is.",
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "Optional id of the agent doing the work; defaults to the server's configured agent_id.",
                        },
                    },
                    "required": ["task_id"],
                },
            },
            {
                "name": "clock_off",
                "description": (
                    "Close the work timer for a task and write its markdown report to "
                    "disk. Provide an honest self-assessed status and a report covering "
                    "what you did, what worked/failed (including which tools helped vs. "
                    "were useless/broken), and your final answer."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The same task_id used at clock_in.",
                        },
                        "status": {
                            "type": "string",
                            "enum": list(VALID_STATUSES),
                            "description": "Your self-assessment: completed | early_exit | timeout | error.",
                        },
                        "report": {
                            "type": "string",
                            "description": (
                                "Free-form markdown report: what you did, what worked / "
                                "didn't (incl. tool usefulness), and the final answer."
                            ),
                        },
                    },
                    "required": ["task_id", "status"],
                },
            },
        ]

    async def execute(
        self, tool_name: str, arguments: Dict[str, Any], extra_args: Optional[Dict[str, Any]] = None
    ) -> Any:
        args = dict(arguments or {})
        if tool_name == "clock_in":
            return self._clock_in(
                task_id=args.get("task_id"),
                task_description=args.get("task_description"),
                agent_id=args.get("agent_id"),
            )
        if tool_name == "clock_off":
            return self._clock_off(
                task_id=args.get("task_id"),
                status=args.get("status"),
                report=args.get("report"),
                agent_id=args.get("agent_id"),
            )
        return f"Error: unknown tool '{tool_name}'"

    async def shutdown(self) -> None:  # honored by ToolManager; harmless under the stdio wrapper
        self._sweep_sync("shutdown_sweep")

    # ------------------------------------------------------------------
    # clock_in / clock_off
    # ------------------------------------------------------------------

    def _clock_in(
        self, task_id: Any, task_description: Optional[str] = None, agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        self._ensure_configured()
        task_id = self._require(task_id, "task_id")
        a_id = (str(agent_id).strip() if agent_id else "") or self._config["agent_id"]
        with self._lock:
            existing = self._open.get(task_id)
            if existing is not None:
                return {
                    "status": "already_open",
                    "task_id": task_id,
                    "agent_id": existing["agent_id"],
                    "started_at": existing["started_at"],
                    "handle": task_id,
                    "message": "timer already open for this task_id; keeping original start time",
                }
            started_at = _utcnow_iso()
            self._open[task_id] = {
                "agent_id": a_id,
                "task_id": task_id,
                "task_description": task_description,
                "started_at": started_at,
                "started_monotonic": time.monotonic(),
            }
            # Persist an in_progress stub NOW (under the lock, so it cannot race
            # a concurrent clock_off/sweep that keys on self._open). This is the
            # record that survives an abrupt SIGKILL of an orphaned stdio
            # subprocess, which would defeat the atexit/SIGTERM sweep.
            stub_path = self._write_stub(
                task_id=task_id, agent_id=a_id, started_at=started_at, description=task_description
            )
        return {
            "status": "clocked_in",
            "task_id": task_id,
            "agent_id": a_id,
            "started_at": started_at,
            "report_path": stub_path,
            "handle": task_id,
        }

    def _write_stub(
        self, *, task_id: str, agent_id: str, started_at: str, description: Optional[str]
    ) -> Optional[str]:
        """Best-effort eager stub at clock_in time.

        Returns the path on success, ``None`` on failure. A failure here must
        NOT break ``clock_in`` — the in-memory timer + shutdown sweep remain the
        graceful-exit path; the stub is the extra floor that survives ``SIGKILL``.
        """
        try:
            return self._write_report(
                task_id=task_id,
                agent_id=agent_id,
                status=STATUS_IN_PROGRESS,
                closed_by=CLOSED_BY_PENDING,
                started_at=started_at,
                ended_at=None,
                elapsed_s=None,
                description=description,
                report=None,
            )
        except Exception:  # noqa: BLE001 — clock_in must not fail on a disk hiccup
            logger.warning("worklog: failed to persist in_progress stub for task %r", task_id, exc_info=True)
            return None

    def _clock_off(
        self,
        task_id: Any,
        status: Any,
        report: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._ensure_configured()
        task_id = self._require(task_id, "task_id")
        status = self._normalize_status(status)

        with self._lock:
            entry = self._open.pop(task_id, None)
            already_closed = entry is None and task_id in self._closed

        if already_closed:
            return {
                "status": "already_closed",
                "task_id": task_id,
                "report_path": self._closed_paths.get(task_id),
                "message": "clock_off ignored; this task was already closed (report not overwritten)",
            }

        warning = None
        if entry is not None:
            started_at = entry["started_at"]
            elapsed_s = round(time.monotonic() - entry["started_monotonic"], 3)
            a_id = entry["agent_id"]
            description = entry.get("task_description")
        else:
            # Lenient: a clock_off with no prior clock_in still writes a report
            # (better data for reflect than dropping it), just without timing.
            started_at = None
            elapsed_s = None
            a_id = (str(agent_id).strip() if agent_id else "") or self._config["agent_id"]
            description = None
            warning = "no matching clock_in; wrote report without timing"

        path = self._write_report(
            task_id=task_id,
            agent_id=a_id,
            status=status,
            closed_by="self",
            started_at=started_at,
            ended_at=_utcnow_iso(),
            elapsed_s=elapsed_s,
            description=description,
            report=report,
        )
        with self._lock:
            self._closed.add(task_id)
            self._closed_paths[task_id] = path

        out: Dict[str, Any] = {
            "status": "clocked_off",
            "task_id": task_id,
            "agent_id": a_id,
            "report_path": path,
            "elapsed_s": elapsed_s,
            "self_reported_status": status,
        }
        if warning:
            out["warning"] = warning
        return out

    # ------------------------------------------------------------------
    # Report writing
    # ------------------------------------------------------------------

    def _write_report(
        self,
        *,
        task_id: str,
        agent_id: str,
        status: str,
        closed_by: str,
        started_at: Optional[str],
        ended_at: Optional[str],
        elapsed_s: Optional[float],
        description: Optional[str],
        report: Optional[str],
    ) -> str:
        run_id = _safe_component(self._config["run_id"], fallback="run")
        base = Path(self._config["worklog_dir"]).expanduser() / run_id
        rel_dir = self._safe_subdir(self._config.get("subdir"))
        if rel_dir:
            base = base / rel_dir
        base.mkdir(parents=True, exist_ok=True)

        fname = _safe_component(task_id, fallback="task") + ".md"
        path = base / fname

        frontmatter: Dict[str, Any] = {
            "worklog_version": WORKLOG_VERSION,
            "run_id": self._config["run_id"],
            "agent_id": agent_id,
            "task_id": task_id,
            "status": status,
            "closed_by": closed_by,
            "started_at": started_at,
            "ended_at": ended_at,
            "elapsed_s": elapsed_s,
        }
        if description:
            frontmatter["task_description"] = description

        body = self._build_body(report, closed_by, description)
        fm_yaml = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True)
        content = f"---\n{fm_yaml}---\n\n{body}\n"

        # Atomic write: a SIGKILL mid-write leaves the old file (or nothing),
        # never a half-written report a reflect child might choke on.
        tmp = base / f"{fname}.tmp.{os.getpid()}"
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
        return str(path)

    @staticmethod
    def _build_body(report: Optional[str], closed_by: str, description: Optional[str]) -> str:
        if report and str(report).strip():
            return str(report).rstrip()

        # Stub body for sweeps / backfills / empty reports so the report always
        # has the expected sections for reflect to scan.
        lines: List[str] = []
        if closed_by == CLOSED_BY_PENDING:
            lines.append(
                "> Work IN PROGRESS — eager `clock_in` stub. This file is overwritten by "
                "`clock_off` (final report) or the shutdown sweep. If this text remains, the "
                "worker was killed before either ran."
            )
        elif closed_by != "self":
            lines.append(f"> Auto-generated stub: timer closed by `{closed_by}` without a self-reported `clock_off`.")
        else:
            lines.append("> No report text was provided at clock_off.")
        lines += [
            "",
            "## What I did",
            (f"Task: {description}" if description else "(unknown)"),
            "",
            "## What worked / didn't (incl. tool usefulness)",
            "(not reported)",
            "",
            "## Final answer",
            "(not reported)",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Shutdown sweep (guaranteed close)
    # ------------------------------------------------------------------

    def _sweep_sync(self, closed_by: str = "shutdown_sweep") -> List[str]:
        """Flush every still-open timer to disk as ``status: error``.

        Idempotent and best-effort: safe to call from ``atexit``, a signal
        handler, and the async ``shutdown`` hook (all of which may fire). A
        re-entrancy guard prevents a signal landing mid-sweep from double-work.
        """
        with self._lock:
            if self._sweeping:
                return []
            self._sweeping = True
            pending = list(self._open.items())
            self._open.clear()

        written: List[str] = []
        try:
            for task_id, entry in pending:
                try:
                    path = self._write_report(
                        task_id=task_id,
                        agent_id=entry["agent_id"],
                        status="error",
                        closed_by=closed_by,
                        started_at=entry["started_at"],
                        ended_at=_utcnow_iso(),
                        elapsed_s=round(time.monotonic() - entry["started_monotonic"], 3),
                        description=entry.get("task_description"),
                        report=None,
                    )
                    with self._lock:
                        self._closed.add(task_id)
                        self._closed_paths[task_id] = path
                    written.append(path)
                except Exception:  # noqa: BLE001 — sweep must not raise on shutdown
                    logger.exception("worklog sweep failed for task %r", task_id)
        finally:
            with self._lock:
                self._sweeping = False
        return written

    def _install_signal_handlers(self) -> None:
        if self._handlers_installed:
            return
        try:
            atexit.register(self._sweep_sync, "shutdown_sweep")
        except Exception:  # noqa: BLE001
            logger.debug("worklog: could not register atexit sweep", exc_info=True)
        # SIGINT is intentionally left alone: it raises KeyboardInterrupt, which
        # unwinds normally and triggers the atexit sweep. SIGTERM/SIGHUP would
        # NOT run atexit, so they need an explicit handler.
        for signame in ("SIGTERM", "SIGHUP"):
            sig = getattr(signal, signame, None)
            if sig is None:
                continue
            try:
                self._prev_handlers[int(sig)] = signal.getsignal(sig)
                signal.signal(sig, self._signal_handler)
            except (ValueError, OSError, RuntimeError):
                # Not in the main thread (e.g. a ThreadPoolExecutor worker) or
                # unsupported platform — atexit still covers normal exit.
                logger.debug("worklog: could not install %s handler", signame, exc_info=True)
        self._handlers_installed = True

    def _signal_handler(self, signum: int, frame: Any) -> None:
        try:
            self._sweep_sync("shutdown_sweep")
        finally:
            restore = self._prev_handlers.get(signum)
            if restore is None:
                restore = signal.SIG_DFL
            try:
                signal.signal(signum, restore)
            except Exception:  # noqa: BLE001
                pass
            try:
                os.kill(os.getpid(), signum)
            except Exception:  # noqa: BLE001
                os._exit(0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_configured(self) -> None:
        if not self._configured:
            self.configure(None, None)

    @staticmethod
    def _require(value: Any, name: str) -> str:
        s = str(value).strip() if value is not None else ""
        if not s:
            raise ValueError(f"{name} is required and must be a non-empty string")
        return s

    @staticmethod
    def _normalize_status(status: Any) -> str:
        s = str(status if status is not None else "").strip().lower()
        if s not in VALID_STATUSES:
            raise ValueError(f"status must be one of {list(VALID_STATUSES)}; got {status!r}")
        return s

    @staticmethod
    def _safe_subdir(subdir: Optional[str]) -> str:
        if not subdir:
            return ""
        parts: List[str] = []
        for raw in str(subdir).replace("\\", "/").split("/"):
            raw = raw.strip()
            if not raw or raw in (".", ".."):
                continue
            parts.append(_safe_component(raw))
        return "/".join(parts)
