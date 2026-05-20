#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the three Hermes validation runs and print the deltas.

For each of ``run0_baseline``, ``run1_learning``, ``run2_evaluation``:

  1. Reads ``<run_dir>/rollouts.jsonl`` (NeMo-Gym shape).
  2. Converts each rollout to NS-judge shape via ``convert_ng_to_ns``.
  3. Walks the corresponding ``<run_dir>/eval/`` directory (populated by
     ``ns eval``) for ``judgement: YES|NO`` and computes pass@1.
  4. Falls back to a cheap regex/exact-match against ``expected_answer``
     if no judge output is available — useful during smoke runs before
     the gpt-oss-120b judge has been wired in.

Outputs a small table to stdout and a JSON summary at
``<output-base>/validation_summary.json``.  The interesting numbers are:

  * ``run2.pass1 - run0.pass1``  — net effect of the memory update
  * ``run1.pass1 - run0.pass1``  — intra-run learning during the
                                    learning run itself
  * ``run2.pass1 - run1.pass1``  — stability across two warm runs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_ANSWER_RE = re.compile(r"Final Answer:\s*(?P<answer>.+)", re.IGNORECASE)


def _extract_final_answer(generation: str) -> str:
    """Pull a 'Final Answer: ...' line out of the model's generation.

    Falls back to the whole generation when the prefix is missing.  The
    cheap exact-match path uses this to compare against
    ``expected_answer`` — it never overrides the judge's verdict when
    one is available.
    """
    match = _ANSWER_RE.search(generation or "")
    if match:
        return match.group("answer").strip()
    return (generation or "").strip()


def _exact_match(answer: str, expected: str) -> bool:
    if not expected:
        return False
    a = re.sub(r"\s+", " ", answer.strip().lower())
    e = re.sub(r"\s+", " ", expected.strip().lower())
    return a == e or e in a


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                sys.stderr.write(f"warn: skipping malformed line in {path}: {exc}\n")


def _load_judge_results(run_dir: Path) -> Dict[str, bool]:
    """Walk the ``ns eval`` output tree for per-id judgement YES/NO."""
    results: Dict[str, bool] = {}
    eval_root = run_dir / "eval"
    if not eval_root.is_dir():
        return results
    for path in eval_root.rglob("*.jsonl"):
        for row in _read_jsonl(path):
            rid = row.get("id") or row.get("task_index")
            judgement = row.get("judgement")
            if rid is None or judgement is None:
                continue
            results[str(rid)] = str(judgement).strip().upper().startswith("YES")
    return results


def _score_run(run_dir: Path) -> Dict[str, Any]:
    rollouts = list(_read_jsonl(run_dir / "rollouts.jsonl"))
    judge = _load_judge_results(run_dir)

    by_subset: Dict[str, Dict[str, int]] = {}
    n_judged = 0
    n_em_fallback = 0
    n_correct = 0

    for entry in rollouts:
        meta = entry.get("verifier_metadata") or {}
        rid = str(meta.get("id") or entry.get("task_index") or "")
        subset = meta.get("subset_for_metrics") or "unknown"
        expected = meta.get("expected_answer") or ""
        response = entry.get("response") or {}
        generation = response.get("output_text") or ""

        if rid in judge:
            correct = judge[rid]
            n_judged += 1
        else:
            correct = _exact_match(_extract_final_answer(generation), expected)
            n_em_fallback += 1

        by_subset.setdefault(subset, {"n": 0, "correct": 0})
        by_subset[subset]["n"] += 1
        if correct:
            by_subset[subset]["correct"] += 1
            n_correct += 1

    n = len(rollouts)
    pass1 = n_correct / n if n else 0.0
    return {
        "run_dir": str(run_dir),
        "n_rollouts": n,
        "n_correct": n_correct,
        "pass1": pass1,
        "n_judged": n_judged,
        "n_em_fallback": n_em_fallback,
        "by_subset": {
            sub: {"n": stats["n"], "correct": stats["correct"], "pass1": stats["correct"] / stats["n"]}
            for sub, stats in by_subset.items()
        },
    }


def _print_delta(label: str, lhs: float, rhs: float) -> None:
    delta = rhs - lhs
    print(f"  Δ {label:<24} = {delta:+.3f}  ({lhs:.3f} → {rhs:.3f})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="compare_runs", description=__doc__)
    p.add_argument("output_base", help="The --output-base used by run_validation.py")
    p.add_argument("--summary", default=None, help="Where to write the JSON summary (default: <output_base>/validation_summary.json)")
    args = p.parse_args(argv)

    base = Path(args.output_base)
    state_path = base / "state.json"
    if not state_path.is_file():
        sys.stderr.write(f"error: {state_path} not found; run run_validation.py first\n")
        return 2
    state = json.loads(state_path.read_text())
    runs_by_name = {r["name"]: r for r in state.get("runs", [])}

    summary: Dict[str, Any] = {"input": state.get("input_ns"), "cluster": state.get("cluster"), "runs": {}}
    for name in ("run0_baseline", "run1_learning", "run2_evaluation"):
        run_info = runs_by_name.get(name)
        if not run_info:
            print(f"[skip] {name} not in state.json")
            continue
        run_dir = Path(run_info["output_dir"])
        score = _score_run(run_dir)
        summary["runs"][name] = score
        print(f"\n=== {name} ===")
        print(f"  dir:                  {score['run_dir']}")
        print(f"  rollouts:             {score['n_rollouts']}")
        print(f"  correct (overall):    {score['n_correct']}")
        print(f"  pass@1:               {score['pass1']:.3f}")
        print(f"  judged vs em-fallback:{score['n_judged']} / {score['n_em_fallback']}")
        for sub, stats in score["by_subset"].items():
            print(f"    [{sub:>10}] pass@1 = {stats['pass1']:.3f}  ({stats['correct']}/{stats['n']})")

    r0 = summary["runs"].get("run0_baseline")
    r1 = summary["runs"].get("run1_learning")
    r2 = summary["runs"].get("run2_evaluation")
    if r0 and r2:
        print("\n=== deltas ===")
        if r0 and r1:
            _print_delta("run1 - run0 (in-run)", r0["pass1"], r1["pass1"])
        if r0 and r2:
            _print_delta("run2 - run0 (net)", r0["pass1"], r2["pass1"])
        if r1 and r2:
            _print_delta("run2 - run1 (stability)", r1["pass1"], r2["pass1"])

    summary_path = Path(args.summary) if args.summary else base / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[+] summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
