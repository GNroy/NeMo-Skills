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

"""``worklog_enrich`` — join trajectory traces into worklog reports (SCI-548).

Worklogs are *self-reported* (what the agent says it did); JSONL trajectory
traces are *ground truth* (what the agent's tool calls actually were). This
module bridges the two: for each agent session it reconstructs a factual
"tools used" table from the trace and writes it back onto the matching worklog
report — plus a structured sidecar and a claim-vs-observed diff that downstream
``reflect`` can read.

**The join key.** Traces are keyed by ``session_id`` (one ``<session_id>.jsonl``
per agent — orchestrator *and*, since the delegate-child trace wiring, each
worker). Worklog reports are keyed by ``task_id`` (``<task_id>.md``). There is
no external map between them — but every agent calls ``mcp_worklog_clock_in``
with its ``task_id``, and that call appears *in its own trace* as a
``tool_start`` event. So we derive the session→task mapping straight from the
trace: the first ``mcp_worklog_clock_in``'s ``args.task_id`` names the task the
session belongs to. No coordination, no shared state.

**Trust boundary.** Tool *result* previews are truncated hard and never expose
full payloads; the benchmark trust-boundary tool already withholds reference
answers from ``get_problem`` results, so a reference answer cannot reach a trace
in the first place — this is belt-and-suspenders.

Run standalone over any past run::

    python -m nemo_skills.mcp.servers.agentic.worklog_enrich \\
        --trace-dir  <output_dir>/traces \\
        --worklog-dir <output_dir>/worklogs/<run_id>

or call :func:`enrich` from a pipeline post-step.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# Mangled MCP tool name (server ``worklog`` + tool ``clock_in``); see the
# worklog_tool / NeMo-Gym integration notes.
CLOCK_IN_TOOL = "mcp_worklog_clock_in"
CLOCK_OFF_TOOL = "mcp_worklog_clock_off"

# Idempotency markers — re-running enrichment replaces the block in place
# rather than appending a second copy.
_ENRICH_BEGIN = "<!-- worklog_enrich:begin -->"
_ENRICH_END = "<!-- worklog_enrich:end -->"
_ENRICH_HEADING = "## Tools used (observed)"
_BLOCK_RE = re.compile(
    re.escape(_ENRICH_HEADING) + r"\s*\n" + re.escape(_ENRICH_BEGIN) + r".*?" + re.escape(_ENRICH_END),
    re.DOTALL,
)

# Heuristic error detection on a tool result preview. Conservative: anchored
# at the start or an obvious traceback so a result that merely mentions the
# word "error" mid-text isn't misflagged.
_ERROR_PREFIXES = ("error", "exception", "traceback", "failed", "invalid")

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_jsonl_events(path: Path) -> List[Dict[str, Any]]:
    """Parse a trace file into a list of event dicts; tolerate junk lines."""
    events: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("worklog_enrich: cannot read trace %s: %s", path, exc)
        return events
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _coerce_args(args: Any) -> Dict[str, Any]:
    """Tool args arrive as a dict, occasionally as a JSON string — normalize."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            v = json.loads(args)
            return v if isinstance(v, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _preview(value: Any, limit: int) -> str:
    """One-line, length-bounded preview of an arbitrary value."""
    if value is None:
        return ""
    if isinstance(value, str):
        s = value
    else:
        try:
            s = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            s = str(value)
    s = " ".join(s.split())
    if len(s) > limit:
        return s[:limit] + f"...[+{len(s) - limit} chars]"
    return s


def _looks_error(result: Any) -> bool:
    if result is None:
        return False
    s = result if isinstance(result, str) else _preview(result, 400)
    low = s.strip().lower()
    if not low:
        return False
    if low.startswith(_ERROR_PREFIXES):
        return True
    return "traceback (most recent call last)" in low


def summarize_session(events: List[Dict[str, Any]], *, max_preview: int = 160) -> Dict[str, Any]:
    """Reduce one session's events to a factual tool-usage summary.

    Returns ``{session_id, task_id, observed: [{name, calls, errors,
    first_args_preview, first_result_preview}], n_events}``. ``task_id`` is the
    first ``clock_in``'s ``args.task_id`` (``None`` if the session never clocked
    in — e.g. a crashed worker — in which case it can't be matched to a report).
    """
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    task_id_fallback: Optional[str] = None  # from clock_off, if no clock_in
    order: List[str] = []
    tools: Dict[str, Dict[str, Any]] = {}

    def _tool(name: str) -> Dict[str, Any]:
        t = tools.get(name)
        if t is None:
            t = {"calls": 0, "errors": 0, "first_args_preview": None, "first_result_preview": None}
            tools[name] = t
            order.append(name)
        return t

    for ev in events:
        if session_id is None:
            session_id = ev.get("session_id")
        kind = ev.get("kind")
        payload = ev.get("payload") or {}
        if kind == "tool_start":
            name = payload.get("tool_name")
            if not name:
                continue
            a = _coerce_args(payload.get("args"))
            if name == CLOCK_IN_TOOL and task_id is None:
                tid = a.get("task_id")
                if tid:
                    task_id = str(tid)
            elif name == CLOCK_OFF_TOOL and task_id_fallback is None:
                # An agent may clock_off without a matching clock_in (the
                # orchestrator's lenient "batch" close). clock_off also carries
                # task_id, so it's a sound fallback join key when clock_in is absent.
                tid = a.get("task_id")
                if tid:
                    task_id_fallback = str(tid)
            t = _tool(name)
            t["calls"] += 1
            if t["first_args_preview"] is None:
                t["first_args_preview"] = _preview(a, max_preview)
        elif kind == "tool_complete":
            name = payload.get("tool_name")
            if not name:
                continue
            t = _tool(name)
            res = payload.get("result")
            if _looks_error(res):
                t["errors"] += 1
            if t["first_result_preview"] is None:
                t["first_result_preview"] = _preview(res, max_preview)

    observed = [dict(name=n, **tools[n]) for n in order]
    return {
        "session_id": session_id,
        "task_id": task_id if task_id is not None else task_id_fallback,
        "observed": observed,
        "n_events": len(events),
    }


def _parse_frontmatter(md_text: str) -> Dict[str, Any]:
    m = _FRONTMATTER_RE.match(md_text)
    if not m:
        return {}
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _index_worklogs(worklog_dir: Path) -> Dict[str, Path]:
    """Map ``task_id`` → report path. Frontmatter ``task_id`` wins; else stem."""
    index: Dict[str, Path] = {}
    for md in sorted(worklog_dir.rglob("*.md")):
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        fm = _parse_frontmatter(text)
        tid = str(fm.get("task_id") or md.stem)
        # First writer wins; reports are 1:1 with task_id by construction.
        index.setdefault(tid, md)
    return index


def _diff_claims(self_reported: Any, observed: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Compare the agent's ``tools_used`` self-report against observed tools."""
    claimed = set()
    if isinstance(self_reported, (list, tuple)):
        for item in self_reported:
            if isinstance(item, dict) and item.get("name"):
                claimed.add(str(item["name"]))
    seen = {o["name"] for o in observed}
    return {
        "claimed_not_observed": sorted(claimed - seen),
        "observed_not_claimed": sorted(seen - claimed),
    }


def _render_block(summary: Dict[str, Any], diff: Dict[str, List[str]]) -> str:
    """Render the machine-derived markdown block (between idempotency markers)."""
    lines = [_ENRICH_HEADING, _ENRICH_BEGIN]
    lines.append(f"*Reconstructed from trajectory trace `{summary.get('session_id')}` by worklog_enrich.*")
    lines.append("")
    observed = summary.get("observed") or []
    if observed:
        lines.append("| tool | calls | errors | first args | first result |")
        lines.append("|------|------:|------:|------------|--------------|")
        for o in observed:
            args_prev = (o.get("first_args_preview") or "").replace("|", "\\|")
            res_prev = (o.get("first_result_preview") or "").replace("|", "\\|")
            lines.append(
                f"| `{o['name']}` | {o['calls']} | {o['errors']} | {args_prev} | {res_prev} |"
            )
    else:
        lines.append("_No tool calls observed in the trace for this task._")
    lines.append("")
    cno = diff.get("claimed_not_observed") or []
    onc = diff.get("observed_not_claimed") or []
    if cno or onc:
        lines.append("**Self-report vs. observed:**")
        if cno:
            lines.append(f"- claimed but NOT observed: {', '.join('`%s`' % t for t in cno)}")
        if onc:
            lines.append(f"- observed but NOT claimed: {', '.join('`%s`' % t for t in onc)}")
    else:
        lines.append("_Self-report matches observed tools (or no self-report provided)._")
    lines.append(_ENRICH_END)
    return "\n".join(lines)


def _apply_to_report(md_path: Path, block: str, *, inplace: bool) -> None:
    if not inplace:
        return
    try:
        text = md_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("worklog_enrich: cannot read report %s: %s", md_path, exc)
        return
    body = text.rstrip("\n")
    if _BLOCK_RE.search(body):
        new = _BLOCK_RE.sub(block, body)
    else:
        new = body + "\n\n" + block
    # Atomic write: a failed write (e.g. disk-quota EDQUOT mid-write) must never
    # truncate/corrupt the agent's existing report. Write to a temp sibling and
    # os.replace only on success; on any error the original file is untouched.
    tmp = md_path.with_suffix(md_path.suffix + f".enrich.tmp.{os.getpid()}")
    try:
        tmp.write_text(new + "\n", encoding="utf-8")
        os.replace(tmp, md_path)
    except OSError as exc:
        logger.warning("worklog_enrich: cannot write report %s (left intact): %s", md_path, exc)
        try:
            tmp.unlink()
        except OSError:
            pass


def enrich(
    trace_dir: str | Path,
    worklog_dir: str | Path,
    *,
    inplace: bool = True,
    write_sidecar: bool = True,
    max_preview: int = 160,
) -> Dict[str, Any]:
    """Enrich every worklog report under ``worklog_dir`` from traces under
    ``trace_dir``.

    ``trace_dir`` may be the per-agent dir (``.../traces/scientist``) or the
    trace root (``.../traces``) — we glob ``*.jsonl`` recursively either way.
    Returns a summary dict (counts + per-task records); never raises on a single
    bad file.
    """
    trace_dir = Path(trace_dir)
    worklog_dir = Path(worklog_dir)

    report_index = _index_worklogs(worklog_dir) if worklog_dir.exists() else {}
    trace_files = sorted(trace_dir.rglob("*.jsonl")) if trace_dir.exists() else []

    records: List[Dict[str, Any]] = []
    matched = 0
    unmatched_sessions: List[str] = []

    for tf in trace_files:
        events = _load_jsonl_events(tf)
        if not events:
            continue
        summary = summarize_session(events, max_preview=max_preview)
        task_id = summary.get("task_id")
        rec: Dict[str, Any] = {
            "trace_file": str(tf),
            "session_id": summary.get("session_id"),
            "task_id": task_id,
            "observed": summary.get("observed"),
        }
        md_path = report_index.get(str(task_id)) if task_id else None
        if md_path is None:
            unmatched_sessions.append(str(tf.name))
            rec["matched"] = False
            records.append(rec)
            continue

        matched += 1
        rec["matched"] = True
        rec["report"] = str(md_path)
        fm = _parse_frontmatter(md_path.read_text(encoding="utf-8", errors="replace"))
        diff = _diff_claims(fm.get("tools_used"), summary.get("observed") or [])
        rec["diff"] = diff

        block = _render_block(summary, diff)
        _apply_to_report(md_path, block, inplace=inplace)

        if write_sidecar:
            sidecar = md_path.with_suffix(".tools.json")
            payload = {
                "task_id": task_id,
                "session_id": summary.get("session_id"),
                "generated_at": _utcnow_iso(),
                "observed": summary.get("observed"),
                "self_reported": fm.get("tools_used"),
                "diff": diff,
            }
            try:
                sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            except OSError as exc:
                logger.warning("worklog_enrich: cannot write sidecar %s: %s", sidecar, exc)

        records.append(rec)

    result = {
        "trace_dir": str(trace_dir),
        "worklog_dir": str(worklog_dir),
        "n_traces": len(trace_files),
        "n_reports": len(report_index),
        "n_matched": matched,
        "n_unmatched_traces": len(unmatched_sessions),
        "unmatched_traces": unmatched_sessions,
        "records": records,
    }
    logger.info(
        "worklog_enrich: %d trace(s), %d report(s), %d matched, %d unmatched trace(s)",
        result["n_traces"],
        result["n_reports"],
        result["n_matched"],
        result["n_unmatched_traces"],
    )
    return result


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m nemo_skills.mcp.servers.agentic.worklog_enrich",
        description="Join trajectory traces into worklog reports (factual tools-used + claim diff).",
    )
    p.add_argument("--trace-dir", required=True, help="Trace dir or root (globs *.jsonl recursively).")
    p.add_argument("--worklog-dir", required=True, help="Worklog run dir (globs *.md recursively).")
    p.add_argument("--no-inplace", action="store_true", help="Do not append the section to the .md reports.")
    p.add_argument("--no-sidecar", action="store_true", help="Do not write <task_id>.tools.json sidecars.")
    p.add_argument("--max-preview", type=int, default=160, help="Max chars for args/result previews.")
    p.add_argument("--quiet", action="store_true", help="Only print the final summary line.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    result = enrich(
        args.trace_dir,
        args.worklog_dir,
        inplace=not args.no_inplace,
        write_sidecar=not args.no_sidecar,
        max_preview=args.max_preview,
    )
    print(
        f"worklog_enrich: matched {result['n_matched']}/{result['n_reports']} report(s) "
        f"from {result['n_traces']} trace(s); {result['n_unmatched_traces']} trace(s) unmatched."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
