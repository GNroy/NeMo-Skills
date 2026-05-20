# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Render a multi-agent Hermes JSONL trace as a chronological fleet timeline.

Usage::

    python -m nemo_skills.scripts.render_hermes_fleet_trace <output_dir>
        [--format text|json]
        [--no-color]
        [--filter-agents agent1,agent2]
        [--filter-kinds tool_start,delegation]
        [--ticket TICKET_ID]

Walks ``<output_dir>/traces/**/*.jsonl`` (or any directory containing
``*.jsonl`` files), parses each event, merges by ``ts``, and emits a
single chronologically-sorted stream.  In ``text`` mode each event is
colored by agent and rendered as a single line with the kind, agent,
and a compact payload summary.  In ``json`` mode each event is emitted
as a JSON line — useful as input to ``jq`` or downstream analytics.

Delegation events (kind=delegation) emitted by the orchestrator carry a
``ticket_id`` that equals the worker's ``session_id`` (the wire
``task_id`` in the Phase 0 ``/task`` adapter).  Pass
``--ticket TICKET_ID`` to filter the merged timeline to just that
delegation and the matching worker session — the renderer joins them
on that key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


# Six ANSI color codes, picked to be readable on dark and light terminals.
# Agents are assigned in order of first appearance.
_PALETTE = ["31", "32", "33", "34", "35", "36", "91", "92", "93", "94", "95", "96"]
_RESET = "\033[0m"


def _iter_trace_events(root: Path) -> Iterable[Dict[str, Any]]:
    """Yield every parseable JSON object from any ``*.jsonl`` under ``root``.

    Accepts either ``<output_dir>`` (which contains a ``traces/`` subdir)
    or ``<output_dir>/traces`` directly — the renderer probes for the
    ``traces`` folder before falling back to a plain recursive walk.

    Malformed lines are skipped with a stderr warning rather than aborting
    the run; an in-flight kill of an agent process can truncate the last
    line of its file and we'd rather still render the rest of the fleet.
    """
    search_root = root / "traces" if (root / "traces").is_dir() else root
    for path in sorted(search_root.rglob("*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as f:
                for lineno, raw in enumerate(f, start=1):
                    raw = raw.rstrip("\n")
                    if not raw.strip():
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        sys.stderr.write(
                            f"warn: skipping malformed line {path}:{lineno}: {exc}\n"
                        )
                        continue
                    if not isinstance(event, dict):
                        continue
                    # Provenance is opt-in (text mode never prints it; JSON
                    # mode strips it) but useful when debugging gaps.
                    event.setdefault("_source_file", str(path))
                    yield event
        except OSError as exc:
            sys.stderr.write(f"warn: cannot read {path}: {exc}\n")


def _ticket_of(event: Dict[str, Any]) -> Optional[str]:
    """Return the ticket_id this event participates in, if any.

    Two ways an event can carry a ticket:
      * ``kind=delegation``: payload['ticket_id'] explicitly.
      * any kind on the worker side: ``session_id`` equals the wire
        ``task_id`` (see HermesAgent._resolve_session_id) — and the
        orchestrator's delegation event sets ticket_id to the same
        value.  So treat session_id as a fallback ticket.
    """
    payload = event.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("ticket_id"), str):
        return payload["ticket_id"]
    sid = event.get("session_id")
    if isinstance(sid, str) and sid and sid != "delegations":
        return sid
    return None


def _events_linked_to_ticket(
    events: Sequence[Dict[str, Any]], ticket_id: str
) -> List[Dict[str, Any]]:
    """Filter ``events`` down to those tied to ``ticket_id``."""
    return [e for e in events if _ticket_of(e) == ticket_id]


def _short(value: Any, max_len: int = 80) -> str:
    s = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    s = s.replace("\n", " ")
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _payload_summary(kind: str, payload: Any) -> str:
    """One-line, terminal-friendly summary for a payload, per event kind."""
    if not isinstance(payload, dict):
        return _short(payload)
    if kind == "tool_start":
        return f"{payload.get('tool_name', '?')}({_short(payload.get('args'))})"
    if kind == "tool_complete":
        return f"{payload.get('tool_name', '?')} → {_short(payload.get('result'))}"
    if kind == "delegation":
        mode = payload.get("mode")
        ticket = (payload.get("ticket_id") or "")[:8]
        worker = payload.get("worker", "?")
        if mode == "result":
            ok = payload.get("ok", True)
            badge = "ok" if ok else "ERR"
            return f"[{badge}] result ← {worker} ticket={ticket} {_short(payload.get('result_preview'))}"
        return f"{mode} → {worker} ticket={ticket} {_short(payload.get('task_preview'))}"
    if kind == "assistant" or kind == "assistant_interim":
        return _short(payload.get("text") or payload)
    if kind == "thinking":
        return _short(payload.get("msg") or payload)
    if kind == "step":
        return f"iter={payload.get('iteration', payload.get('api_call_count', '?'))}"
    if kind == "session_start" or kind == "session_end":
        return _short({k: v for k, v in payload.items() if k != "history"})
    return _short(payload)


def render(events: Sequence[Dict[str, Any]], use_color: bool, fmt: str) -> Iterable[str]:
    """Yield one rendered line per event in the order given."""
    if fmt == "json":
        for event in events:
            event = {k: v for k, v in event.items() if k != "_source_file"}
            yield json.dumps(event, default=str, ensure_ascii=False)
        return

    palette: Dict[str, str] = {}
    for event in events:
        agent = str(event.get("agent", ""))
        if agent not in palette:
            palette[agent] = _PALETTE[len(palette) % len(_PALETTE)]

    for event in events:
        ts = float(event.get("ts", 0.0))
        agent = str(event.get("agent", "?"))
        kind = str(event.get("kind", "?"))
        summary = _payload_summary(kind, event.get("payload"))
        line = f"{ts:14.3f} {agent:<14} {kind:<18} {summary}"
        if use_color:
            code = palette.get(agent, "0")
            line = f"\033[{code}m{line}{_RESET}"
        yield line


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="render_hermes_fleet_trace",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("output_dir", help="Directory containing traces/*.jsonl (or a traces dir directly)")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors even on a TTY")
    p.add_argument(
        "--filter-agents",
        default=None,
        help="Comma-separated agent names to include (default: all)",
    )
    p.add_argument(
        "--filter-kinds",
        default=None,
        help="Comma-separated event kinds to include (default: all)",
    )
    p.add_argument(
        "--ticket",
        default=None,
        help="Restrict output to one ticket_id and its linked worker session",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        sys.stderr.write(f"error: path does not exist: {output_dir}\n")
        return 2

    agent_filter = set(args.filter_agents.split(",")) if args.filter_agents else None
    kind_filter = set(args.filter_kinds.split(",")) if args.filter_kinds else None

    events = list(_iter_trace_events(output_dir))

    if args.ticket:
        events = _events_linked_to_ticket(events, args.ticket)
    if agent_filter:
        events = [e for e in events if e.get("agent") in agent_filter]
    if kind_filter:
        events = [e for e in events if e.get("kind") in kind_filter]

    # Stable sort by ts; missing ts sorts first (best-effort robustness).
    events.sort(key=lambda e: float(e.get("ts", 0.0)))

    use_color = (not args.no_color) and sys.stdout.isatty() and args.format == "text"
    for line in render(events, use_color, args.format):
        print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
