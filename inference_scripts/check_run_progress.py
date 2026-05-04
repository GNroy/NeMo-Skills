"""
check_run_progress.py — Stage and progress for NeMo-Skills eval runs.

Shows per-run:
  Stage       generation / judge / summarize / DONE
  Per-seed    lines done out of N (generation) or files present (judge)
  Speed       avg gen_seconds per problem (from completed records)
  ETA         estimated time remaining for generation stage

Run ID formats
  r17 r18          agentic runs   ({bench_short}_nano-agent/agent_{bench}-nano-agent-rN)
  rec2:no-tool     non-agentic    (run_nano_physics_fs_official/eval_{bench}-nano-rec-no-tool-r2)
  rec2:python      non-agentic    (run_nano_physics_fs_official/eval_{bench}-nano-rec-python-r2)

Usage:
  source ~/miniconda3/bin/activate
  # aws-cmh cluster, frontierscience-olympiad (defaults)
  python inference_scripts/check_run_progress.py r17 r18 rec2:no-tool rec2:python

  # specify cluster or benchmark explicitly
  python inference_scripts/check_run_progress.py --cluster oci --bench physics r9 r10
  python inference_scripts/check_run_progress.py --bench hle-physics r19

  # single run
  python inference_scripts/check_run_progress.py r17
"""

import argparse
import subprocess
import sys
import textwrap

# ── cluster configs ───────────────────────────────────────────────────────────

CLUSTERS = {
    "aws-cmh": {
        "ssh_key":  "/home/alaptev/.ssh/clusters/aws-cmh/id_ecdsa",
        "ssh_host": "alaptev@aws-cmh-slurm-1-login-01.nvidia.com",
        "lustre":   "/lustre/fsw/portfolios/nemotron/users/alaptev",
    },
    "oci": {
        "ssh_key":  "/home/alaptev/.ssh/clusters/oci/id_ecdsa",
        "ssh_host": "alaptev@draco-oci-login-01.draco-oci-iad.nvidia.com",
        "lustre":   "/lustre/fsw/portfolios/llmservice/users/alaptev",
    },
}

# ── benchmark configs ─────────────────────────────────────────────────────────

BENCH_CONFIGS = {
    "fs": {
        "bench_short":       "fs",
        "benchmark":         "frontierscience-olympiad",
        "problems_per_seed": 100,
    },
    "physics": {
        "bench_short":       "physics",
        "benchmark":         "physics",
        "problems_per_seed": 100,
    },
    "hle-physics": {
        "bench_short":       "hle-physics",
        "benchmark":         "hle",
        "problems_per_seed": 100,
    },
}

# ── run-id resolution ─────────────────────────────────────────────────────────

def resolve_run(run_id: str, cluster_cfg: dict, bench_cfg: dict) -> tuple[str, str]:
    """Return (label, output_dir) for a run ID string."""
    lustre      = cluster_cfg["lustre"]
    bench_short = bench_cfg["bench_short"]
    agent_base    = f"{lustre}/exp/eval/nano_physics_agent"
    official_base = f"{lustre}/exp/eval/run_nano_physics_fs_official"

    if run_id.startswith("rec"):
        # recN:arm  e.g.  rec2:no-tool
        parts    = run_id.split(":", 1)
        rec_part = parts[0]          # rec2
        arm      = parts[1] if len(parts) > 1 else "no-tool"
        n        = rec_part[3:]      # 2
        odir     = f"{official_base}/eval_{bench_short}-nano-rec-{arm}-r{n}"
        return run_id, odir
    else:
        # rN  e.g.  r17
        n    = run_id.lstrip("r")
        odir = f"{agent_base}/agent_{bench_short}-nano-agent-r{n}"
        return run_id, odir


# ── remote analysis snippet ───────────────────────────────────────────────────

# Runs on the cluster; prints one-line-per-field for easy parsing.
_REMOTE_SNIPPET = textwrap.dedent(r"""
import json, glob, os, time, statistics

base  = {base!r}
bench = {bench!r}
n_per = {n_per}

tmpdir   = f"{{base}}/tmp-eval-results/{{bench}}"
evaldir  = f"{{base}}/eval-results/{{bench}}"
sumdir   = f"{{evaldir}}/summarized-results"

# ── summarize done? ────────────────────────────────────────────────────────
sum_logs = glob.glob(f"{{sumdir}}/main_*_srun.log")
if sum_logs:
    print("stage=DONE")
    sys.exit(0)

# ── generation file inventory ─────────────────────────────────────────────
done_files  = sorted(glob.glob(f"{{tmpdir}}/output-rs*.jsonl.done"))
async_files = sorted(glob.glob(f"{{tmpdir}}/output-rs*.jsonl-async"))
done_count  = len(done_files)
total_seeds = max(done_count + len(async_files), 5)   # fallback 5

# ── judge files ───────────────────────────────────────────────────────────
judge_files = glob.glob(f"{{evaldir}}/output-rs*.jsonl") + \
              glob.glob(f"{{evaldir}}/output-rs*.jsonl.done")

if done_count == total_seeds and done_count > 0:
    # All generation done; check judge stage
    if judge_files:
        j_done = len([f for f in judge_files if f.endswith(".done")])
        j_total = len(set(f.replace(".done","") for f in judge_files))
        print(f"stage=judge")
        print(f"judge_done={{j_done}}/{{j_total}}")
    else:
        print("stage=judge")
        print(f"judge_done=0/{{total_seeds}}")
    # show completed seeds
    for df in done_files:
        name = os.path.basename(df).replace(".jsonl.done","")
        print(f"seed={{name}}:100/{{n_per}}:done")
    sys.exit(0)

# ── generation in progress ────────────────────────────────────────────────
print("stage=generation")
print(f"seeds_done={{done_count}}/{{total_seeds}}")

all_speeds = []
now = time.time()

# completed seeds
for df in done_files:
    name = os.path.basename(df).replace(".jsonl.done","")
    jsonl = df.replace(".done","")
    speeds = []
    if os.path.exists(jsonl):
        with open(jsonl) as f:
            for line in f:
                try:
                    gs = json.loads(line).get("gen_seconds")
                    if gs and gs > 0: speeds.append(gs)
                except Exception: pass
    all_speeds.extend(speeds)
    print(f"seed={{name}}:{{n_per}}/{{n_per}}:done")

# in-progress seeds
for af in sorted(async_files):
    name = os.path.basename(af).replace(".jsonl-async","")
    lines, speeds = 0, []
    try:
        with open(af) as f:
            for line in f:
                lines += 1
                try:
                    gs = json.loads(line).get("gen_seconds")
                    if gs and gs > 0: speeds.append(gs)
                except Exception: pass
    except Exception: pass
    mtime_ago = (now - os.path.getmtime(af)) / 60 if os.path.exists(af) else -1
    all_speeds.extend(speeds)
    last_str = f"last_upd={{mtime_ago:.1f}}m_ago"
    print(f"seed={{name}}:{{lines}}/{{n_per}}:running:{{last_str}}")

# overall speed summary
if all_speeds:
    med = statistics.median(all_speeds)
    mn  = min(all_speeds)
    mx  = max(all_speeds)
    # problems remaining across all active seeds
    remaining = sum(
        n_per - int(open(af).read().count("\n"))
        for af in async_files
        if os.path.exists(af)
    )
    # ETA = remaining * median_speed / active_seeds (parallel)
    active = max(len(async_files), 1)
    eta_s  = remaining * med / active
    eta_m  = eta_s / 60
    print(f"speed_median={{med:.0f}}s/prob speed_min={{mn:.0f}}s speed_max={{mx:.0f}}s")
    print(f"remaining={{remaining}} eta={{eta_m:.0f}}m")
else:
    print("speed=unknown")
""")


def build_snippet(output_dir: str, benchmark: str, problems_per_seed: int) -> str:
    return (
        "import sys\n"
        + _REMOTE_SNIPPET.format(
            base=output_dir,
            bench=benchmark,
            n_per=int(problems_per_seed),
        )
    )


def ssh_run(snippet: str, cluster_cfg: dict) -> str:
    result = subprocess.run(
        ["ssh", "-i", cluster_cfg["ssh_key"], cluster_cfg["ssh_host"],
         f"python3 -c {snippet!r}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


# ── formatting ────────────────────────────────────────────────────────────────

def parse_and_print(label: str, raw: str) -> None:
    lines = raw.splitlines()
    kv = {}
    seed_lines = []
    for line in lines:
        if line.startswith("seed="):
            seed_lines.append(line[5:])   # strip "seed="
        else:
            k, _, v = line.partition("=")
            kv[k] = v

    stage = kv.get("stage", "unknown")
    print(f"\n{'─'*54}")
    print(f"  {label}   [{stage.upper()}]")
    print(f"{'─'*54}")

    if stage == "DONE":
        print("  All stages complete.")
        return

    if stage == "judge":
        print(f"  Judge: {kv.get('judge_done', '?')}")
        for sl in seed_lines:
            parts = sl.split(":")
            print(f"    {parts[0]:12s}  {parts[1]}")
        return

    # generation
    print(f"  Seeds done : {kv.get('seeds_done', '?')}")
    for sl in seed_lines:
        parts = sl.split(":", 3)
        name   = parts[0]
        prog   = parts[1]
        status = parts[2] if len(parts) > 2 else ""
        extra  = parts[3].replace("_", " ") if len(parts) > 3 else ""
        marker = "✓" if status == "done" else "…"
        print(f"    {marker} {name:12s}  {prog:8s}  {extra}")

    if "speed_median" in kv:
        print(f"\n  Speed  : {kv['speed_median']}  "
              f"(min {kv.get('speed_min','?')}  max {kv.get('speed_max','?')})")
        print(f"  Remain : {kv.get('remaining','?')} problems")
        eta = kv.get("eta", "?")
        if eta != "?":
            eta_val = int(eta)
            hours, mins = divmod(eta_val, 60)
            eta_str = f"{hours}h {mins}m" if hours else f"{mins}m"
            print(f"  ETA    : ~{eta_str}")
    else:
        print(f"\n  Speed  : {kv.get('speed', 'unknown (no completed problems yet)')}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Check stage/progress of NeMo-Skills eval runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            clusters : aws-cmh (default), oci
            benches  : fs (default), physics, hle-physics

            examples:
              python inference_scripts/check_run_progress.py r17 r18
              python inference_scripts/check_run_progress.py rec2:no-tool rec2:python
              python inference_scripts/check_run_progress.py --bench physics r9 r10
              python inference_scripts/check_run_progress.py --cluster oci --bench physics r5
        """),
    )
    ap.add_argument(
        "runs",
        nargs="+",
        metavar="RUN_ID",
        help="Run IDs: r17, r18, rec2:no-tool, rec2:python, …",
    )
    ap.add_argument(
        "--cluster", "-c",
        default="aws-cmh",
        choices=list(CLUSTERS),
        help="Cluster to connect to (default: aws-cmh)",
    )
    ap.add_argument(
        "--bench", "-b",
        default="fs",
        choices=list(BENCH_CONFIGS),
        help="Benchmark to check (default: fs)",
    )
    args = ap.parse_args()

    cluster_cfg = CLUSTERS[args.cluster]
    bench_cfg   = BENCH_CONFIGS[args.bench]

    for run_id in args.runs:
        label, odir = resolve_run(run_id, cluster_cfg, bench_cfg)
        snippet = build_snippet(odir, bench_cfg["benchmark"], bench_cfg["problems_per_seed"])
        raw = ssh_run(snippet, cluster_cfg)
        if not raw:
            print(f"\n  {label}: no output (dir may not exist yet)")
        else:
            parse_and_print(label, raw)

    print()


if __name__ == "__main__":
    main()
