#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Three-run frontier-science validation driver.

This script orchestrates the A/B/C comparison for the Hermes integration
plan's Phase 6 (validation):

  Run 0 — *baseline*.  Frozen template; ``persist_memory=False`` and
          no merge-back.  Measures the cold-start performance of the
          single Hermes agent with all tools/MCPs available.

  Run 1 — *learning*.  Same frozen template at start; the run uses
          ``persist_memory=True`` and ``merge_back=True`` so per-task
          learnings flow into the orchestrator's HERMES_HOME and then
          back into the template at job end.

  Run 2 — *evaluation*.  Starts from Run 1's updated template;
          ``persist_memory=True`` to read but ``merge_back=False`` to
          keep the template stable for the comparison.  Run 0 vs Run 2
          isolates the memory-update effect from intra-run drift.

Each run produces a ``rollouts.jsonl`` (NeMo-Gym shape) under its own
output dir.  After all runs land, the driver invokes
``convert_ng_to_ns.py`` so the offline judge step (``ns eval``) can
consume them.  Judging is dispatched in a separate command (see README)
because it has its own SLURM job dependency on the rollouts.

Usage (dry-run on submit host)::

    python -m workdir-agents.validation.scripts.run_validation \\
        --manifest workdir-agents/validation/frontierscience_manifest.yaml \\
        --input-ns /alaptev/data/frontierscience/all.jsonl \\
        --output-base /alaptev/exp/hermes_validation/smoke \\
        --cluster aws-cmh \\
        --limit 5 \\
        --dry-run

Drop ``--dry-run`` to actually submit the three SLURM jobs.  The
template's *post-Run-1* state is the key artefact — the driver records
its absolute path in ``<output-base>/state.json`` so a later re-run or
debug session can resume cleanly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RunSpec:
    name: str
    output_dir: str
    persist_memory: bool
    merge_back: bool
    template_hermes_home: str
    description: str


def _read_yaml(path: Path) -> Dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(data: Dict, path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _materialize_manifest(
    base_manifest: Path,
    target: Path,
    *,
    template_hermes_home: str,
    persist_memory: bool,
) -> None:
    """Write a manifest variant with run-specific overrides applied."""
    data = _read_yaml(base_manifest)
    data["template_hermes_home"] = template_hermes_home
    agents = data.get("agents") or {}
    for spec in agents.values():
        hermes = spec.setdefault("hermes", {})
        hermes["persist_memory"] = persist_memory
        # Skills + trajectories follow memory: a run with no memory persistence
        # also has no skill curation; a memory-bearing run carries both.
        hermes["persist_skills"] = persist_memory
        hermes["save_trajectories"] = persist_memory
    _write_yaml(data, target)


def _run_command(cmd: List[str], *, dry_run: bool) -> None:
    pretty = " ".join(cmd)
    print(f"\n$ {pretty}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _stage_template(source: Path, target: Path) -> None:
    """Snapshot a HERMES_HOME template directory tree."""
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, symlinks=True)


def _build_plan(args: argparse.Namespace) -> List[RunSpec]:
    base = Path(args.output_base)
    runs: List[RunSpec] = [
        RunSpec(
            name="run0_baseline",
            output_dir=str(base / "run0_baseline"),
            persist_memory=False,
            merge_back=False,
            template_hermes_home=str(base / "templates" / "run0_template"),
            description="frozen template, persist_memory=False, no merge-back",
        ),
        RunSpec(
            name="run1_learning",
            output_dir=str(base / "run1_learning"),
            persist_memory=True,
            merge_back=True,
            template_hermes_home=str(base / "templates" / "run1_template"),
            description="frozen template at start, persist_memory=True, merge-back ON",
        ),
        RunSpec(
            name="run2_evaluation",
            output_dir=str(base / "run2_evaluation"),
            persist_memory=True,
            merge_back=False,
            # The driver copies Run 1's *post-mergeback* template into this slot
            # before Run 2 starts.
            template_hermes_home=str(base / "templates" / "run2_template"),
            description="starts from Run 1's merged template, no merge-back",
        ),
    ]
    return runs


def _build_pipeline_cmd(
    *,
    cluster: str,
    expname: str,
    manifest: Path,
    input_file: Path,
    output_dir: str,
    merge_back: bool,
    dry_run: bool,
    extra_args: List[str],
) -> List[str]:
    cmd = [
        "ns",
        "hermes_agent_rollouts",
        "--cluster",
        cluster,
        "--agent_manifest",
        str(manifest),
        "--input_file",
        str(input_file),
        "--output_dir",
        output_dir,
        "--expname",
        expname,
    ]
    # Typer keeps underscores in the bool flag names for this command.
    cmd.append("--merge_back" if merge_back else "--no-merge_back")
    if dry_run:
        cmd.append("--dry_run")
    cmd.extend(extra_args)
    return cmd


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="run_validation", description=__doc__)
    p.add_argument("--manifest", required=True, help="Base Hermes manifest YAML")
    p.add_argument("--input-ns", required=True, help="NS frontier-science JSONL (post-prepare_data)")
    p.add_argument("--output-base", required=True, help="Output base dir for all three runs + templates")
    p.add_argument("--cluster", default="aws-cmh", help="NS cluster config name")
    p.add_argument("--limit", type=int, default=None, help="Smoke flag: limit input to N entries")
    p.add_argument(
        "--seed-template",
        default=None,
        help="Source HERMES_HOME template to seed all three runs from (defaults to the manifest's template_hermes_home).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands without submitting")
    p.add_argument("--skip-run0", action="store_true", help="Skip the baseline run (assumes it already landed)")
    p.add_argument("--skip-run1", action="store_true", help="Skip the learning run (assumes Run 1 template exists)")
    p.add_argument("--skip-run2", action="store_true", help="Skip the evaluation run")
    p.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to `ns hermes_agent_rollouts` (after a --)",
    )
    args = p.parse_args(argv)

    base = Path(args.output_base)
    base.mkdir(parents=True, exist_ok=True)
    print(f"[+] output base: {base}")

    # 1. Convert the NS input → NG input (single conversion shared by all three runs).
    # The directory name contains a hyphen so we can't import as a package;
    # load the converter via importlib from its absolute file path instead.
    import importlib.util

    ng_input = base / "input_ng.jsonl"
    converter_path = Path(__file__).resolve().parent / "convert_ns_to_ng.py"
    _spec = importlib.util.spec_from_file_location("convert_ns_to_ng", converter_path)
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_mod)
    n_in = _mod.convert_file(Path(args.input_ns), ng_input, args.limit)
    print(f"[+] converted {n_in} entries → {ng_input}")

    # 2. Seed the Run 0 / Run 1 template from the manifest's declared
    # template_hermes_home (or --seed-template).
    base_manifest = Path(args.manifest)
    manifest_data = _read_yaml(base_manifest)
    seed = args.seed_template or manifest_data.get("template_hermes_home")
    if not seed:
        p.error("manifest has no template_hermes_home; pass --seed-template")
    seed_path = Path(seed)
    if not args.dry_run and not seed_path.exists():
        p.error(f"seed template does not exist: {seed_path}")

    runs = _build_plan(args)
    run0, run1, run2 = runs

    # Snapshot the seed into Run 0 + Run 1 template slots.
    for rs in (run0, run1):
        if args.dry_run:
            print(f"[dry] would copy {seed_path} → {rs.template_hermes_home}")
        else:
            _stage_template(seed_path, Path(rs.template_hermes_home))

    # 3. Materialize the per-run manifest variants with the right
    # template path + persist_memory flag baked in.
    extra_args: List[str] = list(args.extra_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    submitted: List[Dict] = []
    for rs, skip in (
        (run0, args.skip_run0),
        (run1, args.skip_run1),
        (run2, args.skip_run2),
    ):
        if skip:
            print(f"[skip] {rs.name}")
            continue
        manifest_path = base / f"manifest_{rs.name}.yaml"
        _materialize_manifest(
            base_manifest,
            manifest_path,
            template_hermes_home=rs.template_hermes_home,
            persist_memory=rs.persist_memory,
        )
        # Run 2 is dependent on Run 1's mergeback landing.  The driver
        # only stages Run 2's template after Run 1 finishes.  In dry-run
        # mode we just describe the wiring.
        if rs is run2:
            if args.dry_run:
                print(
                    f"[dry] would copy Run 1 merged template "
                    f"{run1.template_hermes_home} → {rs.template_hermes_home}"
                )
            else:
                print(
                    "[!] Run 2 expects Run 1's merged template at "
                    f"{run1.template_hermes_home}.  Re-run this script with "
                    "--skip-run0 --skip-run1 once Run 1's SLURM job has "
                    "finished and its merge_audit.json is present."
                )
                _stage_template(Path(run1.template_hermes_home), Path(rs.template_hermes_home))

        cmd = _build_pipeline_cmd(
            cluster=args.cluster,
            expname=f"hermes_val_{rs.name}",
            manifest=manifest_path,
            input_file=ng_input,
            output_dir=rs.output_dir,
            merge_back=rs.merge_back,
            dry_run=args.dry_run,
            extra_args=extra_args,
        )
        _run_command(cmd, dry_run=args.dry_run)
        submitted.append(
            {
                "name": rs.name,
                "description": rs.description,
                "manifest": str(manifest_path),
                "output_dir": rs.output_dir,
                "template_hermes_home": rs.template_hermes_home,
                "merge_back": rs.merge_back,
                "persist_memory": rs.persist_memory,
            }
        )

    state_path = base / "state.json"
    state = {
        "manifest": str(base_manifest),
        "input_ns": str(args.input_ns),
        "input_ng": str(ng_input),
        "n_input": n_in,
        "cluster": args.cluster,
        "limit": args.limit,
        "seed_template": str(seed_path),
        "runs": submitted,
        "extra_args": extra_args,
        "ts": int(time.time()),
    }
    state_path.write_text(json.dumps(state, indent=2))
    print(f"\n[+] state recorded at {state_path}")
    print(
        "Next: once all three runs have produced rollouts.jsonl, run\n"
        f"  python workdir-agents/validation/scripts/compare_runs.py {base}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
