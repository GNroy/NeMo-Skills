#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert a NeMo-Skills frontier-science JSONL to NeMo-Gym input format.

Input  (NS frontierscience-olympiad):
    {"id": "...", "question": "...", "expected_answer": "...",
     "subset_for_metrics": "physics", "task_group_id": "..."}

Output (NeMo-Gym responses_create_params shape):
    {"responses_create_params": {"input": [{"role": "user", "content": "..."}]},
     "verifier_metadata": {"id": "...", "expected_answer": "...",
                            "subset_for_metrics": "...", "task_group_id": "..."}}

The verifier_metadata is preserved verbatim so the downstream NS judge
step (gpt-oss-120b via ``ns eval``) has the reference answer when it
runs against the NG-shape rollouts file converted back via
``convert_ng_to_ns.py``.

Optional ``--limit N`` truncates to the first N entries (smoke runs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


_SYSTEM_PROMPT = (
    "You are an expert scientific research assistant.  You have access to a "
    "set of tools — a Python interpreter, the web, scientific databases, and "
    "long-term skill/memory storage.  Use the tools as needed to solve the "
    "problem.  When you have the final answer, state it clearly and concisely "
    "on its own line, prefixed with 'Final Answer:'."
)


def convert_entry(ns_entry: Dict[str, Any]) -> Dict[str, Any]:
    question = ns_entry.get("question", "")
    if not question:
        raise ValueError(f"NS entry missing 'question': {ns_entry}")
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        },
        "verifier_metadata": {
            "id": ns_entry.get("id", ""),
            "expected_answer": ns_entry.get("expected_answer", ""),
            "subset_for_metrics": ns_entry.get("subset_for_metrics", ""),
            "task_group_id": ns_entry.get("task_group_id", ""),
        },
    }


def convert_file(input_path: Path, output_path: Path, limit: Optional[int] = None) -> int:
    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ns_entry = json.loads(line)
            ng_entry = convert_entry(ns_entry)
            fout.write(json.dumps(ng_entry, ensure_ascii=False) + "\n")
            written += 1
            if limit is not None and written >= limit:
                break
    return written


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="convert_ns_to_ng", description=__doc__)
    p.add_argument("input", help="NS frontierscience-olympiad JSONL")
    p.add_argument("output", help="NeMo-Gym input JSONL to write")
    p.add_argument("--limit", type=int, default=None, help="Keep only the first N entries")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    n = convert_file(Path(args.input), Path(args.output), args.limit)
    print(f"wrote {n} entries to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
