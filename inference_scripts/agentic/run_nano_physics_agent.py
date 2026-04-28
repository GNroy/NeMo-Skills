"""
Nemotron-Nano-30B — Physics + FrontierScience, agent evaluation
===============================================================
Single-agent run using PythonTool.  Model and sampling parameters match
run_nano_physics_fs_official.py (tool-calling preset: temperature=0.6,
top_p=0.95 per NVIDIA model card).

Pipeline:
    agent generation  →  judge (gpt-oss-120b)  →  summarize

Generation and judge/summarize are submitted as separate NeMo-Run
experiments.  The generation step uses nemo_skills.inference.agent_generate
(same module as ns agent).  eval() chains the judge and summarize steps
after generation completes, with skip_filled=True ensuring already-generated
outputs are never re-processed.

Worker agents can be added via --worker-model / --worker-names to test
multi-agent setups without changing the orchestrator config.

Usage:
    # Single agent, physics benchmark
    python inference_scripts/agentic/run_nano_physics_agent.py --run-id 1

    # One benchmark, dry run
    python inference_scripts/agentic/run_nano_physics_agent.py --run-id 1 \\
        --benchmarks physics --dry-run

    # Multi-agent: orchestrator + one worker
    python inference_scripts/agentic/run_nano_physics_agent.py --run-id 1 \\
        --worker-model /hf_models/worker-model \\
        --worker-gpus 4 --worker-names analyst
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nemo_skills.pipeline.cli import agent, eval, wrap_arguments
from nemo_skills.pipeline.utils import get_cluster_config
from nemo_skills.pipeline.utils.eval import add_default_args

CLUSTER = "dfw"

OUTPUT_DIR_BASE = "/alaptev/exp/eval/nano_physics_agent"

# Judge: on-cluster gpt-oss-120b (same as run_nano_physics_fs_official.py)
JUDGE_MODEL = "/hf_models/gpt-oss-120b"
JUDGE_SERVER_TYPE = "vllm"
JUDGE_SERVER_GPUS = 8
JUDGE_SERVER_ARGS = "--async-scheduling --max-model-len 131072"

MODEL = {
    "short": "nano-agent",
    "path": "/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "server_type": "vllm",
    "server_gpus": 8,
    "server_nodes": 1,
    "server_args": (
        "--dtype auto "
        "--kv-cache-dtype fp8 "
        "--enable-expert-parallel "
        "--attention-backend FLASH_ATTN "
        "--trust-remote-code "
        "--gpu-memory-utilization 0.9 "
        "--enable-chunked-prefill "
        "--mamba-ssm-cache-dtype float16 "
        "--enable-auto-tool-choice "
        "--tool-call-parser qwen3_coder "
        "--reasoning-parser-plugin "
        "/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/nano_v3_reasoning_parser.py "
        "--reasoning-parser nano_v3 "
        "--max-num-seqs 256 "
    ),
    # Tool-calling preset per NVIDIA model card
    "sampling_args": (
        "++inference.temperature=0.6 "
        "++inference.top_p=0.95 "
    ),
    "inference_args": (
        "++inference.tokens_to_generate=131072 "
        "++chat_template_kwargs.enable_thinking=True "
        "++parse_reasoning=False "
    ),
}

# Keys map to registered NeMo-Skills benchmark names + optional split overrides.
BENCHMARKS = {
    "physics": {
        "name": "physics",
        "split": None,          # defaults to EVAL_SPLIT = "test" from the benchmark module
        "num_random_seeds": 5,
        "num_chunks": 10,
    },
    "fs": {
        "name": "frontierscience-olympiad",
        "split": "all",
        "num_random_seeds": 5,
        "num_chunks": 1,
    },
    "hle-physics": {
        "name": "hle",
        "split": "phy",
        "num_random_seeds": 5,
        "num_chunks": 1,
    },
}

PYTHON_TOOL = '++tool_modules=["nemo_skills.mcp.servers.python_tool::PythonTool"] '


def main():
    ap = argparse.ArgumentParser(
        description="Nemotron-Nano agent evaluation — Physics + FrontierScience",
    )
    ap.add_argument("--output-dir", default=OUTPUT_DIR_BASE)
    ap.add_argument("--run-id", type=int, required=True)
    ap.add_argument("--benchmarks", default="physics,fs", help="Comma-sep benchmark keys")
    ap.add_argument("--cluster", default=CLUSTER)
    ap.add_argument("--dry-run", action="store_true")
    # Worker agents (omit for single-agent mode)
    ap.add_argument(
        "--worker-model",
        nargs="+",
        default=[],
        help="Model path(s) for worker agent(s). Enables multi-agent mode.",
    )
    ap.add_argument("--worker-server-type", nargs="+", default=["vllm"])
    ap.add_argument("--worker-gpus", nargs="+", type=int, default=[8])
    ap.add_argument("--worker-names", nargs="+", default=None)
    ap.add_argument(
        "--max-tool-calls",
        type=int,
        default=50,
        help="Max tool calls per problem (caps runaway agentic loops)",
    )
    args = ap.parse_args()

    bench_keys = [b.strip() for b in args.benchmarks.split(",")]

    # cluster_config is used to resolve benchmark input paths in multi-agent mode
    cluster_config = get_cluster_config(args.cluster)

    results = []
    for bkey in bench_keys:
        bcfg = BENCHMARKS[bkey]

        name = f"agent_{bkey}-{MODEL['short']}-r{args.run_id}"
        odir = f"{args.output_dir}/agent_{bkey}-{MODEL['short']}-r{args.run_id}"

        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"  benchmark:  {bcfg['name']}  ({bcfg['num_random_seeds']} seeds)")
        print(f"  cluster:    {args.cluster}")
        print(f"  output:     {odir}")
        print(f"  workers:    {args.worker_model or '(none — single agent)'}")
        print(f"{'=' * 60}")

        extra_args = (
            MODEL["sampling_args"]
            + MODEL["inference_args"]
            + PYTHON_TOOL
            + f"++max_tool_calls={args.max_tool_calls} "
        )

        if args.dry_run:
            print("  [DRY] extra args:")
            for token in extra_args.strip().split():
                print(f"    {token}")
            results.append((name, odir))
            continue

        # ----------------------------------------------------------------
        # Common eval() kwargs for judge + summarize.
        # eval() also handles generation (agent_generate module) when no
        # pre-existing outputs are found; otherwise skip_filled=True makes
        # the generation step exit immediately.
        # ----------------------------------------------------------------
        eval_kwargs = dict(
            ctx=wrap_arguments(extra_args),
            cluster=args.cluster,
            expname=name,
            output_dir=odir,
            benchmarks=f"{bcfg['name']}:{bcfg['num_random_seeds']}",
            split=bcfg["split"],
            num_chunks=bcfg["num_chunks"] if bcfg["num_chunks"] > 1 else None,
            generation_module="nemo_skills.inference.agent_generate",
            model=MODEL["path"],
            server_type=MODEL["server_type"],
            server_gpus=MODEL["server_gpus"],
            server_nodes=MODEL["server_nodes"],
            server_args=MODEL["server_args"],
            with_sandbox=True,
            judge_model=JUDGE_MODEL,
            judge_server_type=JUDGE_SERVER_TYPE,
            judge_server_gpus=JUDGE_SERVER_GPUS,
            judge_server_args=JUDGE_SERVER_ARGS,
        )

        if args.worker_model:
            # ----------------------------------------------------------
            # Multi-agent mode: agent() runs the het-job for generation,
            # eval() chains judge + summarize afterwards.
            # ----------------------------------------------------------
            # Resolve benchmark → input_file for the agent() call.
            benchmark_args_list = add_default_args(
                cluster_config=cluster_config,
                benchmark_or_group=bcfg["name"],
                split=bcfg["split"],
                data_dir=None,
                eval_requires_judge=True,
            )
            benchmark_args = benchmark_args_list[0]
            # Agent writes to the same subdir that eval() expects, so that
            # eval's skip_filled logic finds the .done files after agent finishes.
            agent_odir = f"{odir}/{benchmark_args.eval_subfolder}"

            agent_kwargs = dict(
                ctx=wrap_arguments(extra_args),
                cluster=args.cluster,
                expname=name,
                model=MODEL["path"],
                server_type=MODEL["server_type"],
                server_gpus=MODEL["server_gpus"],
                server_nodes=MODEL["server_nodes"],
                server_args=MODEL["server_args"],
                input_file=benchmark_args.input_file,
                output_dir=agent_odir,
                with_sandbox=True,
                num_random_seeds=bcfg["num_random_seeds"],
                num_chunks=bcfg["num_chunks"] if bcfg["num_chunks"] > 1 else None,
                worker_model=args.worker_model,
                worker_server_type=args.worker_server_type,
                worker_server_gpus=args.worker_gpus,
                worker_names=args.worker_names,
            )
            agent(**agent_kwargs)

            # eval() for judge + summarize waits for agent to finish.
            # It will also submit (and immediately skip) generation jobs
            # if .done files are not yet present at submission time.
            eval_kwargs["run_after"] = [name]
            eval_kwargs["expname"] = f"{name}-eval"

        eval(**eval_kwargs)
        results.append((name, odir))

    print(f"\n{'=' * 60}")
    if args.dry_run:
        print(f"  [DRY RUN] Would submit {len(results)} jobs")
    else:
        print(f"  Submitted {len(results)} jobs on {args.cluster}")
    print(f"{'=' * 60}\n")
    for name, odir in results:
        print(f"  {name}")
        print(f"    ns summarize_results --cluster {args.cluster} {odir}")
    print()


if __name__ == "__main__":
    main()
