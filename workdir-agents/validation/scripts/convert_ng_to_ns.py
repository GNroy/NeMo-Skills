#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert a NeMo-Gym rollouts JSONL back to NeMo-Skills-judgeable format.

ng_collect_rollouts writes lines of the shape::

    {
      "task_index": 0,
      "reward": 0.0,                        # passthrough verifier
      "response": {"output_text": "Final Answer: 42", ...},
      "responses_create_params": {"input": [...]},
      "verifier_metadata": {"id": "...", "expected_answer": "...", ...}
    }

The NS frontier-science judge (``ns eval --benchmarks frontierscience-olympiad``)
expects per-line entries shaped like::

    {
      "id": "...",
      "question": "...",
      "expected_answer": "...",
      "generation": "<final assistant text>",
      "subset_for_metrics": "..."
    }

This script extracts the assistant text from ``response.output_text``
(or falls back to walking ``response.output`` for the last assistant
message) and stitches the verifier metadata back in.  Each NG line maps
to one NS line; ``--seed-tag`` annotates the output so downstream eval
machinery can distinguish multiple runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _extract_generation(response: Any) -> str:
    if not isinstance(response, dict):
        return ""
    # NeMo-Gym sets ``output_text`` directly on the response for convenience.
    text = response.get("output_text")
    if isinstance(text, str) and text:
        return text
    # Fallback: walk ``output`` for the last assistant message.
    last = ""
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        for chunk in item.get("content") or []:
            chunk_text = chunk.get("text") if isinstance(chunk, dict) else None
            if isinstance(chunk_text, str) and chunk_text:
                last = chunk_text
    return last


def _extract_question(entry: Dict[str, Any]) -> str:
    rcp = entry.get("responses_create_params") or {}
    messages = rcp.get("input") or []
    # The user-content message we care about is the *last* user turn —
    # the converter prepends a system prompt, the user question follows.
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def convert_entry(ng_entry: Dict[str, Any], seed_tag: Optional[str] = None) -> Dict[str, Any]:
    meta = ng_entry.get("verifier_metadata") or {}
    out = {
        "id": meta.get("id", ""),
        "question": _extract_question(ng_entry),
        "expected_answer": meta.get("expected_answer", ""),
        "subset_for_metrics": meta.get("subset_for_metrics", ""),
        "task_group_id": meta.get("task_group_id", ""),
        "generation": _extract_generation(ng_entry.get("response")),
        "ng_reward": ng_entry.get("reward"),
        "task_index": ng_entry.get("task_index"),
    }
    if seed_tag is not None:
        out["seed_tag"] = seed_tag
    return out


def convert_file(input_path: Path, output_path: Path, seed_tag: Optional[str] = None) -> int:
    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ng_entry = json.loads(line)
            ns_entry = convert_entry(ng_entry, seed_tag=seed_tag)
            fout.write(json.dumps(ns_entry, ensure_ascii=False) + "\n")
            written += 1
    return written


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="convert_ng_to_ns", description=__doc__)
    p.add_argument("input", help="NeMo-Gym rollouts JSONL")
    p.add_argument("output", help="NS-judgeable JSONL to write")
    p.add_argument(
        "--seed-tag",
        default=None,
        help="Annotate every output entry with this seed_tag (e.g. run0/run1/run2)",
    )
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    n = convert_file(Path(args.input), Path(args.output), args.seed_tag)
    print(f"wrote {n} entries to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
