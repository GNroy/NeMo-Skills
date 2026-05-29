#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the raw frontierscience-olympiad JSONL into a rollout-ready
file for ng_collect_rollouts + the hermes_agent + frontierscience_judge
stack.

Source rows look like::

    {"id": "olympiad-0", "question": "...", "expected_answer": "...",
     "subset_for_metrics": "physics", "task_group_id": "..."}

We add the two fields ng_collect_rollouts needs to dispatch the row:

  * ``agent_ref`` — selects the hermes_agent server
  * ``responses_create_params.input`` — the seed user turn the policy sees

We leave ``question`` and ``expected_answer`` at top level so
frontierscience_judge can read them off the verify request body.
Idempotent: if a row already has both fields shaped correctly, it is
passed through unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_AGENT_REF = {"type": "responses_api_agents", "name": "hermes_agent"}

# Earlier experiment: prepending a FINAL ANSWER nudge as a system message
# (see Phase 8 follow-up notes in hermes_integration_plan.md).  Falsified
# on the 25-problem v4 run — 48% vs 62.5% on the same slice without it.
# The nudge pushes the model to commit confident-but-wrong answers
# instead of working through the problem; removing it.  Keep this file
# free of system-prompt manipulation; if we need to influence agent
# termination behaviour, fix Hermes's continuation heuristic directly
# (option (A) in the Phase 8 follow-up triage).


def transform(row: dict) -> dict:
    out = dict(row)

    # Hoist verifier_metadata.{expected_answer,id,subset_for_metrics,...} to
    # top level — frontierscience_judge reads these off the verify request
    # body directly, not nested under verifier_metadata.
    vm = out.get("verifier_metadata") or {}
    if isinstance(vm, dict):
        for key in ("expected_answer", "id", "subset_for_metrics", "task_group_id", "subject"):
            if key in vm and key not in out:
                out[key] = vm[key]

    # Derive ``question`` from the seed user turn if missing.  Rows from
    # the NS dataset (all.jsonl) have ``question`` at top level; rows from
    # the pre-converted NG smoke_ng.jsonl don't — but they always have a
    # user message in responses_create_params.input.
    if "question" not in out:
        msgs = (out.get("responses_create_params") or {}).get("input") or []
        for m in msgs:
            if m.get("role") == "user":
                out["question"] = m.get("content", "")
                break

    # Force agent_ref to hermes_agent regardless of what the source row
    # says.  This prep is hermes+frontierscience-specific.
    out["agent_ref"] = dict(_AGENT_REF)

    # Ensure responses_create_params.input exists.
    rcp = out.get("responses_create_params")
    if not isinstance(rcp, dict):
        out["responses_create_params"] = {
            "input": [{"role": "user", "content": out.get("question", "")}],
        }
        rcp = out["responses_create_params"]
    elif "input" not in rcp or not rcp["input"]:
        rcp["input"] = [{"role": "user", "content": out.get("question", "")}]

    # If a previous prep added the FINAL ANSWER nudge (now reverted), strip
    # it so rerunning prep on cached rollout-ready inputs gives the same
    # shape as a fresh prep.  Detect by the unique phrase combination.
    inp = rcp["input"]
    rcp["input"] = [
        m for m in inp
        if not (
            isinstance(m, dict)
            and m.get("role") == "system"
            and isinstance(m.get("content"), str)
            and "FINAL ANSWER:" in m["content"]
            and "planning prose" in m["content"]
        )
    ]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="source JSONL")
    p.add_argument("--output", required=True, help="rollout-ready JSONL")
    p.add_argument("--limit", type=int, default=0, help="cap rows (0 = all)")
    args = p.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with src.open() as fi, dst.open("w") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = transform(row)
            # After hoisting, the judge needs question + expected_answer.
            if not t.get("question") or not t.get("expected_answer"):
                continue
            fo.write(json.dumps(t) + "\n")
            kept += 1
            if args.limit and kept >= args.limit:
                break
    print(f"wrote {kept} rows to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
