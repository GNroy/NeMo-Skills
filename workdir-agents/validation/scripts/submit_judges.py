#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submit the on-cluster gpt-oss-120b judge against each run's rollouts.

The validation driver writes ``rollouts.jsonl`` (NeMo-Gym shape) under
each run's output dir.  This script:

  1. Converts each ``rollouts.jsonl`` to NS-judgeable shape via
     ``convert_ng_to_ns.py`` (preserving ``id`` and ``expected_answer``).
  2. Calls ``nemo_skills.pipeline.cli.eval`` with ``num_jobs=0`` so it
     only runs the judge + summarize steps (no generation re-run) —
     matching the pattern in
     ``inference_scripts/agentic/run_nano_physics_agent.py``.

Run after all three jobs have produced their ``rollouts.jsonl``.  The
judge runs sequentially over the three runs (gpt-oss-120b spins up a
single shared server, so parallel submission is wasted).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Mirrors run_nano_physics_agent.py.
JUDGE_MODEL = "/hf_models/gpt-oss-120b"
JUDGE_SERVER_TYPE = "vllm"
JUDGE_SERVER_ARGS = "--async-scheduling --max-model-len 131072"


def _submit_one(
    *,
    cluster: str,
    expname: str,
    run_dir: Path,
    judge_server_gpus: int,
    extra_args: List[str],
    dry_run: bool,
) -> None:
    # Import lazily so the validation driver can stage NS files without
    # forcing the heavy NS imports at top of module.
    from nemo_skills.pipeline.cli import eval as ns_eval, wrap_arguments

    rollouts_ns = run_dir / "rollouts_ns.jsonl"
    if not rollouts_ns.is_file():
        # Late-bind the converter — it lives alongside this file; the
        # parent dir name has a hyphen so we go through importlib.
        import importlib.util
        from pathlib import Path as _Path

        converter_path = _Path(__file__).resolve().parent / "convert_ng_to_ns.py"
        _spec = importlib.util.spec_from_file_location("convert_ng_to_ns", converter_path)
        _mod = importlib.util.module_from_spec(_spec)
        assert _spec.loader is not None
        _spec.loader.exec_module(_mod)
        _mod.convert_file(run_dir / "rollouts.jsonl", rollouts_ns, seed_tag=run_dir.name)

    output_dir = str(run_dir)
    kwargs = dict(
        ctx=wrap_arguments(" ".join(extra_args)),
        cluster=cluster,
        expname=expname,
        output_dir=output_dir,
        benchmarks="frontierscience-olympiad",
        split="all",
        # The agent path-style: num_jobs=0 → skip generation, only judge+summarize.
        judge_model=JUDGE_MODEL,
        judge_server_type=JUDGE_SERVER_TYPE,
        judge_server_gpus=judge_server_gpus,
        judge_server_args=JUDGE_SERVER_ARGS,
        num_jobs=0,
    )
    print(f"\n[+] judge submission for {run_dir.name}:")
    for k, v in kwargs.items():
        print(f"    {k} = {v!r}")
    if dry_run:
        return
    ns_eval(**kwargs)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="submit_judges", description=__doc__)
    p.add_argument("output_base", help="Validation run base dir (the one with state.json)")
    p.add_argument("--cluster", default="aws-cmh")
    p.add_argument("--judge-gpus", type=int, default=4, help="Per-node GPU count for the judge")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args forwarded to `ns eval`")
    args = p.parse_args(argv)

    base = Path(args.output_base)
    state_path = base / "state.json"
    if not state_path.is_file():
        p.error(f"{state_path} not found; run run_validation.py first")
    state = json.loads(state_path.read_text())

    extra: List[str] = list(args.extra_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]

    for run_info in state.get("runs", []):
        run_dir = Path(run_info["output_dir"])
        if not (run_dir / "rollouts.jsonl").is_file() and not args.dry_run:
            print(f"[skip] {run_dir} has no rollouts.jsonl yet — leave it for a later submit pass.")
            continue
        _submit_one(
            cluster=args.cluster,
            expname=f"judge_{run_info['name']}",
            run_dir=run_dir,
            judge_server_gpus=args.judge_gpus,
            extra_args=extra,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
