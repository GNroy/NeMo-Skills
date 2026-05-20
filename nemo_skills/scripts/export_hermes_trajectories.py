# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Export per-agent Hermes trajectories from a multi-agent rollout.

Phase 5 surface for RL preparation.  Given an ``ns hermes_agent_rollouts``
``output_dir``, this script walks ``traces/**/*.jsonl`` and emits one
JSONL file per agent in ShareGPT shape::

    <out_dir>/<agent>.jsonl

Each line carries:

* ``agent``, ``session_id``, ``model``  — provenance
* ``conversations`` — list of ``{"from": role, "value": text}`` turns,
  reconstructed from the trace events (system from session_start prompt,
  human/gpt/tool from assistant / tool_complete / thinking events).
  This is the shape ``hermes-agent/trajectory_compressor.py`` consumes.
* ``token_ids`` — per-step ``prompt_token_ids``, ``generation_token_ids``,
  ``generation_log_probs`` extracted from ``step`` events when
  ``save_trajectories=True`` was on.
* ``reward`` — propagated from the orchestrator's rollout reward
  (when ``--rollouts`` is provided); worker trajectories inherit the
  same value via the dispatch ticket join.  ``None`` when no rollout
  file is available — Phase 5 RL design defers per-worker judges until
  baseline runs.

Usage::

    python -m nemo_skills.scripts.export_hermes_trajectories \\
        <output_dir>                                         \\
        --out <trajectories_dir>                             \\
        [--rollouts <output_dir>/rollouts.jsonl]

The exporter is conservative: missing or malformed trace files produce
warnings on stderr but never abort the run, because trajectories are an
offline analytics surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from nemo_skills.scripts.render_hermes_fleet_trace import _iter_trace_events


# Map the trace event kinds we know about onto ShareGPT roles.  Events not
# in this map are ignored for trajectory reconstruction (they may still
# show up in the fleet renderer for debugging).
_KIND_TO_ROLE = {
    "thinking": "gpt",
    "assistant": "gpt",
    "assistant_interim": "gpt",
    "tool_start": "gpt",
    "tool_complete": "tool",
    "tool_progress": "tool",
    "delegation": "tool",
}


def _system_value(payload: Any) -> str:
    if isinstance(payload, dict):
        prompt = payload.get("user_message_preview")
        if isinstance(prompt, str) and prompt:
            return prompt
    return ""


def _turn_value(kind: str, payload: Any) -> str:
    if isinstance(payload, dict):
        if kind == "assistant" or kind == "assistant_interim":
            return str(payload.get("text") or payload)
        if kind == "thinking":
            return str(payload.get("msg") or payload)
        if kind == "tool_start":
            return json.dumps(
                {"tool_name": payload.get("tool_name"), "args": payload.get("args")},
                default=str,
                ensure_ascii=False,
            )
        if kind == "tool_complete":
            return json.dumps(
                {"tool_name": payload.get("tool_name"), "result": payload.get("result")},
                default=str,
                ensure_ascii=False,
            )
        if kind == "tool_progress":
            return json.dumps(payload, default=str, ensure_ascii=False)
        if kind == "delegation":
            mode = payload.get("mode")
            if mode == "result":
                return f"[delegation result ← {payload.get('worker')} ticket={payload.get('ticket_id')}] {payload.get('result_preview', '')}"
            return f"[delegation {mode} → {payload.get('worker')} ticket={payload.get('ticket_id')}] {payload.get('task_preview', '')}"
    return json.dumps(payload, default=str, ensure_ascii=False)


def _bucket_by_session(
    events: Iterable[Dict[str, Any]],
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Group events by (agent, session_id), preserving original ts order."""
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        agent = str(event.get("agent", "unknown"))
        sid = str(event.get("session_id", ""))
        if not sid or sid == "delegations":
            # The orchestrator's delegation file is a side-channel — its
            # events are attached to the per-session bucket via ticket_id
            # matching below, not as their own trajectory.
            continue
        buckets[(agent, sid)].append(event)
    return buckets


def _load_rollouts(rollouts_path: Optional[Path]) -> Dict[int, float]:
    """Load reward per task_index from a NeMo-Gym ng_collect_rollouts file."""
    rewards: Dict[int, float] = {}
    if rollouts_path is None or not rollouts_path.is_file():
        return rewards
    try:
        with rollouts_path.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    sys.stderr.write(
                        f"warn: skipping malformed rollouts line {rollouts_path}:{lineno}: {exc}\n"
                    )
                    continue
                idx = obj.get("task_index")
                if isinstance(idx, int) and "reward" in obj:
                    rewards[idx] = float(obj["reward"])
    except OSError as exc:
        sys.stderr.write(f"warn: cannot read {rollouts_path}: {exc}\n")
    return rewards


def _build_trajectory(
    agent: str,
    session_id: str,
    events: List[Dict[str, Any]],
    reward: Optional[float],
) -> Dict[str, Any]:
    """Convert a per-session list of trace events into a ShareGPT entry."""
    events_sorted = sorted(events, key=lambda e: float(e.get("ts", 0.0)))

    model = ""
    conversations: List[Dict[str, str]] = []
    prompt_token_ids: List[Any] = []
    generation_token_ids: List[Any] = []
    generation_log_probs: List[Any] = []

    for event in events_sorted:
        kind = event.get("kind")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}

        if kind == "session_start":
            user_msg = _system_value(payload)
            if not model and isinstance(payload, dict):
                model = str(payload.get("model") or "")
            if user_msg:
                conversations.append({"from": "human", "value": user_msg})
            continue

        if kind == "session_end":
            continue

        if kind == "step":
            if "prompt_token_ids" in payload:
                prompt_token_ids.append(payload["prompt_token_ids"])
            if "generation_token_ids" in payload:
                generation_token_ids.append(payload["generation_token_ids"])
            if "generation_log_probs" in payload:
                generation_log_probs.append(payload["generation_log_probs"])
            continue

        role = _KIND_TO_ROLE.get(kind)
        if role is None:
            continue
        conversations.append({"from": role, "value": _turn_value(kind, payload)})

    entry: Dict[str, Any] = {
        "agent": agent,
        "session_id": session_id,
        "model": model,
        "conversations": conversations,
        "reward": reward,
    }
    # Only emit a token_ids subobject when at least one tensor survived;
    # avoids burning bytes on every trajectory in non-RL runs.
    if prompt_token_ids or generation_token_ids or generation_log_probs:
        entry["token_ids"] = {
            "prompt_token_ids": prompt_token_ids,
            "generation_token_ids": generation_token_ids,
            "generation_log_probs": generation_log_probs,
        }
    return entry


def export_trajectories(
    output_dir: Path,
    out_dir: Path,
    rollouts_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Walk traces/*.jsonl, emit one trajectory JSONL per agent.

    Returns a mapping ``{agent_name: trajectory_jsonl_path}``.

    Reward propagation: the orchestrator's first session in time order
    is matched to ``task_index=0`` in the rollouts file, the second to
    ``task_index=1``, etc.  Worker sessions inherit the matching
    orchestrator's reward via ticket_id ↔ session_id (workers'
    session_id = the orchestrator's wire task_id).  This is the
    "propagate orchestrator's terminal reward" path from the Phase 5
    design — per-worker judges are deferred until a baseline is in hand.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    events = list(_iter_trace_events(output_dir))
    buckets = _bucket_by_session(events)
    rewards = _load_rollouts(rollouts_path)

    # 1. Identify orchestrator sessions and assign rewards in ts order.
    orch_session_to_reward: Dict[str, Optional[float]] = {}
    orch_buckets = sorted(
        [(sid, evs) for (agent, sid), evs in buckets.items() if agent != "orchestrator" and False],
        key=lambda x: min(float(e.get("ts", 0.0)) for e in x[1]),
    )
    # The orchestrator agent_name is whatever the manifest set — we don't
    # know it a priori.  Use a heuristic: any agent that appears as the
    # ``agent`` field on a ``delegation`` event is the orchestrator.
    orchestrator_name = ""
    for event in events:
        if event.get("kind") == "delegation":
            orchestrator_name = str(event.get("agent", ""))
            break
    if not orchestrator_name:
        # Fallback: pick the agent whose first session timestamp is earliest.
        first_ts: Dict[str, float] = {}
        for (agent, _sid), evs in buckets.items():
            ts0 = min(float(e.get("ts", 0.0)) for e in evs)
            if agent not in first_ts or ts0 < first_ts[agent]:
                first_ts[agent] = ts0
        if first_ts:
            orchestrator_name = min(first_ts.items(), key=lambda kv: kv[1])[0]

    orch_sessions_ordered = sorted(
        [sid for (agent, sid) in buckets if agent == orchestrator_name],
        key=lambda sid: min(float(e.get("ts", 0.0)) for e in buckets[(orchestrator_name, sid)]),
    )
    for idx, sid in enumerate(orch_sessions_ordered):
        orch_session_to_reward[sid] = rewards.get(idx)

    # 2. Build worker → orchestrator session map via delegation events.
    # Workers' session_id == ticket_id == orchestrator's wire task_id.
    ticket_to_orch_sid: Dict[str, str] = {}
    for event in events:
        if event.get("kind") != "delegation":
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        ticket = payload.get("ticket_id")
        # We do not directly know which orchestrator session emitted this
        # delegation (the delegation file uses session_id="delegations").
        # Pair each delegation ts to the nearest orchestrator session ts.
        if not isinstance(ticket, str):
            continue
        ts_d = float(event.get("ts", 0.0))
        if not orch_sessions_ordered:
            continue
        nearest_sid = min(
            orch_sessions_ordered,
            key=lambda sid: abs(ts_d - min(float(e.get("ts", 0.0)) for e in buckets[(orchestrator_name, sid)])),
        )
        ticket_to_orch_sid[ticket] = nearest_sid

    # 3. Write per-agent JSONL files.
    written: Dict[str, Path] = {}
    by_agent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for (agent, sid), evs in buckets.items():
        if agent == orchestrator_name:
            reward = orch_session_to_reward.get(sid)
        else:
            # Worker session — inherit reward from its orchestrator's session.
            orch_sid = ticket_to_orch_sid.get(sid)
            reward = orch_session_to_reward.get(orch_sid) if orch_sid else None
        trajectory = _build_trajectory(agent, sid, evs, reward)
        by_agent[agent].append(trajectory)

    for agent, entries in by_agent.items():
        out_path = out_dir / f"{agent}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
        written[agent] = out_path
    return written


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="export_hermes_trajectories",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("output_dir", help="ns hermes_agent_rollouts output_dir (contains traces/)")
    p.add_argument("--out", required=True, help="Directory to write per-agent JSONLs into")
    p.add_argument(
        "--rollouts",
        default=None,
        help="Optional rollouts.jsonl file for reward propagation",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        sys.stderr.write(f"error: path does not exist: {output_dir}\n")
        return 2
    out_dir = Path(args.out)
    rollouts = Path(args.rollouts) if args.rollouts else None
    written = export_trajectories(output_dir, out_dir, rollouts)
    for agent, path in sorted(written.items()):
        print(f"{agent}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
