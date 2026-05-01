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
multi-agent setups.  --gpt-worker additionally attaches gpt-oss-120b as a
second worker so the orchestrator can fire non-blocking parallel calls to
both workers and pick the best answer (or fall back if one returns nothing).

Usage:
    # Single agent, physics benchmark
    python inference_scripts/agentic/run_nano_physics_agent.py --run-id 1

    # One benchmark, dry run
    python inference_scripts/agentic/run_nano_physics_agent.py --run-id 1 \\
        --benchmarks physics --dry-run

    # Multi-agent: orchestrator + one nano worker
    python inference_scripts/agentic/run_nano_physics_agent.py --run-id 1 \\
        --worker-model /hf_models/worker-model --worker-names nano_solver

    # Multi-agent: orchestrator + nano worker + gpt-oss-120b worker
    python inference_scripts/agentic/run_nano_physics_agent.py --run-id 1 \\
        --worker-model /hf_models/worker-model --worker-names nano_solver \\
        --gpt-worker
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nemo_skills.pipeline.cli import agent, eval, wrap_arguments
from nemo_skills.pipeline.utils import get_cluster_config
from nemo_skills.pipeline.utils.eval import add_default_args

CLUSTER = "oci"

OUTPUT_DIR_BASE = "/alaptev/exp/eval/nano_physics_agent"

# Judge: on-cluster gpt-oss-120b (same as run_nano_physics_fs_official.py)
JUDGE_MODEL = "/hf_models/gpt-oss-120b"
JUDGE_SERVER_TYPE = "vllm"
JUDGE_SERVER_ARGS = "--async-scheduling --max-model-len 131072"

# vLLM flags that require NVIDIA compute capability ≥ 9.0 (H100/H200/GB300).
# OCI uses A100 (cc 8.0) — these are appended only for clusters in _H100_CLUSTERS.
# Note: FLASH_ATTN backend does NOT support fp8 kv-cache in vLLM 0.18.x; omit it
# and let vLLM auto-select an fp8-compatible backend (e.g. FlashInfer on H100/GB300).
_H100_SERVER_ARGS = "--kv-cache-dtype fp8 "
_H100_CLUSTERS = {"dfw", "eos", "aws-cmh"}  # GB300 also supports fp8 kv-cache

# Per-cluster hardware parameters.  Each entry overrides the corresponding
# MODEL/judge defaults for that cluster.
CLUSTER_PARAMS = {
    "oci": {
        # A100 80 GB, 8 GPUs/node
        "server_gpus": 8,
        "server_nodes": 1,
        "worker_gpus": 8,
        "judge_server_gpus": 8,
        "model_path": "/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        # nano_v3_reasoning_parser.py ships alongside the model in igitman's /hf_models.
        "reasoning_parser_path": (
            "/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/nano_v3_reasoning_parser.py"
        ),
    },
    "aws-cmh": {
        # GB300 288 GB HBM/GPU, 4 GPUs/node — single node is sufficient for
        # both the 30B model (nano) and the 120B judge (4 × 288 GB = 1152 GB).
        "server_gpus": 4,
        "server_nodes": 1,
        "worker_gpus": 4,
        "judge_server_gpus": 4,
        # Model loaded from HF cache (copied from OCI); Hub ID triggers cache lookup.
        "model_path": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        # Parser file copied from OCI (igitman's llm/hf_models) to alaptev's home.
        "reasoning_parser_path": "/alaptev/reasoning_parsers/nano_v3_reasoning_parser.py",
    },
}
_DEFAULT_CLUSTER_PARAMS = CLUSTER_PARAMS["oci"]

MODEL = {
    "short": "nano-agent",
    "server_type": "vllm",
    # Base server args without the reasoning-parser-plugin path, which is
    # cluster-specific and injected at runtime from CLUSTER_PARAMS.
    "server_args": (
        "--dtype auto "
        "--enable-expert-parallel "
        "--trust-remote-code "
        "--gpu-memory-utilization 0.9 "
        "--enable-chunked-prefill "
        "--mamba-ssm-cache-dtype float16 "
        "--enable-auto-tool-choice "
        "--tool-call-parser qwen3_coder "
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
        "num_chunks": 4,
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

# Single-agent: the one agent handles everything with PythonTool.
PYTHON_TOOL = '++tool_modules=["nemo_skills.mcp.servers.python_tool::PythonTool"] '

# Multi-agent: orchestrator delegates via CallAgentTool (no direct Python access).
ORCHESTRATOR_TOOL = '++tool_modules=["nemo_skills.mcp.servers.agent_tool::CallAgentTool"] '

# Nano worker: DirectPythonTool + code-execution system prompt + thinking enabled.
# tokens_to_generate is capped to prevent per-call queue time from exceeding
# the orchestrator's HTTP timeout when many concurrent problems fire simultaneously.
# Thinking is re-enabled: within the 32k budget it helps code quality without
# exhausting the generation limit (the issue in r6–r11 was the *orchestrator*
# thinking, not the worker).
NANO_WORKER_EXTRA_ARGS = (
    '++tool_modules=["nemo_skills.mcp.servers.python_tool::DirectPythonTool"] '
    '++system_message_yaml=agents/code_agent '
    '++inference.tokens_to_generate=32768 '
    '++max_tool_output_tokens=2000 '
    '++chat_template_kwargs.enable_thinking=True '
)

# GPT-OSS-120B worker: same code-execution setup; no thinking flags (not supported).
GPT_WORKER_EXTRA_ARGS = (
    '++tool_modules=["nemo_skills.mcp.servers.python_tool::DirectPythonTool"] '
    '++system_message_yaml=agents/code_agent '
    '++inference.tokens_to_generate=32768 '
    '++max_tool_output_tokens=2000 '
)

# vLLM server args for the gpt-oss-120b worker (same model as judge).
GPT_WORKER_SERVER_ARGS = JUDGE_SERVER_ARGS


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
    ap.add_argument("--worker-names", nargs="+", default=None)
    ap.add_argument(
        "--max-tool-calls",
        type=int,
        default=15,
        help="Max tool calls per problem (caps runaway agentic loops). "
             "Multi-agent orchestrator typically uses 4-8 calls; 15 gives "
             "headroom while preventing runaway loops that exhaust the 4h SLURM timeout.",
    )
    ap.add_argument(
        "--no-fp8",
        action="store_true",
        help="Disable fp8 kv-cache even on H100/GB300 clusters (use for ablation).",
    )
    ap.add_argument(
        "--gpt-worker",
        action="store_true",
        help=(
            "Add gpt-oss-120b as an additional worker alongside any --worker-model "
            "workers. The orchestrator can dispatch to both in parallel and pick the "
            "best answer. Forces worker_reuse_orchestrator_server=False."
        ),
    )
    ap.add_argument(
        "--gpt-worker-name",
        default="gpt_solver",
        help="Agent name for the gpt-oss-120b worker (default: gpt_solver).",
    )
    args = ap.parse_args()

    # Resolve cluster-specific hardware parameters.
    cparams = CLUSTER_PARAMS.get(args.cluster, _DEFAULT_CLUSTER_PARAMS)
    server_gpus = cparams["server_gpus"]
    server_nodes = cparams["server_nodes"]
    worker_gpus = cparams["worker_gpus"]
    judge_server_gpus = cparams["judge_server_gpus"]
    model_path = cparams["model_path"]

    bench_keys = [b.strip() for b in args.benchmarks.split(",")]

    # Inject cluster-specific reasoning-parser-plugin path, then append
    # H100-only vLLM flags when targeting an H100/H200/GB300 cluster.
    # --no-fp8 suppresses fp8 kv-cache for ablation runs.
    h100_args = "" if args.no_fp8 else (_H100_SERVER_ARGS if args.cluster in _H100_CLUSTERS else "")
    server_args = (
        MODEL["server_args"]
        + f"--reasoning-parser-plugin {cparams['reasoning_parser_path']} "
        + h100_args
    )

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
        all_workers = list(args.worker_model) + ([JUDGE_MODEL] if args.gpt_worker else [])
        print(f"  workers:    {all_workers or '(none — single agent)'}")
        print(f"{'=' * 60}")

        base_args = MODEL["sampling_args"] + MODEL["inference_args"] + f"++max_tool_calls={args.max_tool_calls} "

        have_workers = bool(args.worker_model) or args.gpt_worker
        if have_workers:
            # Multi-agent: orchestrator delegates via CallAgentTool with orchestrator
            # system prompt; workers handle computation with PythonTool.
            # Disable thinking for the orchestrator: the worker does the heavy
            # reasoning; the orchestrator only needs to delegate and synthesize.
            # Without this, the model exhausts its full 131k token budget in the
            # <think> block before emitting a tool call (finish_reason=length).
            extra_args = (
                base_args
                + ORCHESTRATOR_TOOL
                + "++system_message_yaml=agents/orchestrator "
                + "++inference.extra_body.tool_choice=required "
                + "++chat_template_kwargs.enable_thinking=False "
                # Allow up to 20 min per worker call: 100 concurrent problems all fire
                # call_solver_async at once; worker semaphore=64 means up to 36 queue
                # at any time, each waiting for a 32k-token worker call to free a slot.
                + "++tool_overrides.CallAgentTool.timeout_s=1200 "
                + "++tool_overrides.CallAgentTool.max_injection_tokens=4000 "
            )
        else:
            # Single-agent: one agent handles everything with PythonTool.
            extra_args = base_args + PYTHON_TOOL

        if args.dry_run:
            print("  [DRY] orchestrator args:")
            for token in extra_args.strip().split():
                print(f"    {token}")
            if have_workers:
                for i, (wm, wargs) in enumerate(zip(
                    list(args.worker_model) + ([JUDGE_MODEL] if args.gpt_worker else []),
                    [NANO_WORKER_EXTRA_ARGS] * len(args.worker_model)
                    + ([GPT_WORKER_EXTRA_ARGS] if args.gpt_worker else []),
                )):
                    print(f"  [DRY] worker {i} ({wm}) args:")
                    for token in wargs.strip().split():
                        print(f"    {token}")
            results.append((name, odir))
            continue

        # ----------------------------------------------------------------
        # agent() runs generation for both single- and multi-agent modes.
        # eval() chains judge + summarize afterwards (run_after=name).
        # ----------------------------------------------------------------
        # Resolve benchmark → input_file and generation_args (e.g.
        # ++prompt_config, ++eval_type) defined by the benchmark module.
        benchmark_args_list = add_default_args(
            cluster_config=cluster_config,
            benchmark_or_group=bcfg["name"],
            split=bcfg["split"],
            data_dir=None,
            eval_requires_judge=True,
        )
        benchmark_args = benchmark_args_list[0]
        # agent() writes to the subdir that eval() expects, so eval's
        # skip_filled logic finds the .done files after agent finishes.
        agent_odir = f"{odir}/{benchmark_args.eval_subfolder}"

        agent_kwargs = dict(
            # benchmark_args.generation_args carries ++prompt_config and
            # ++eval_type defined by the benchmark module.
            ctx=wrap_arguments(benchmark_args.generation_args + " " + extra_args),
            cluster=args.cluster,
            expname=name,
            model=model_path,
            server_type=MODEL["server_type"],
            server_gpus=server_gpus,
            server_nodes=server_nodes,
            server_args=server_args,
            input_file=benchmark_args.input_file,
            output_dir=agent_odir,
            # Keep agent-logs at the top-level odir (same level as eval-logs).
            log_dir=f"{odir}/agent-logs",
            with_sandbox=True,
            num_random_seeds=bcfg["num_random_seeds"],
            num_chunks=bcfg["num_chunks"] if bcfg["num_chunks"] > 1 else None,
        )

        if have_workers:
            # Build per-worker lists.  agent() auto-deduplicates LLM servers:
            # workers whose model matches the orchestrator are co-located (het-group 0);
            # workers sharing a non-orchestrator model share one server in one het-group.
            # No manual reuse flag needed.
            worker_models = list(args.worker_model)
            worker_types = list(args.worker_server_type)
            worker_xargs = [NANO_WORKER_EXTRA_ARGS] * len(worker_models)
            # Nano worker: same server args as orchestrator (triggers co-location).
            worker_sargs = [server_args] * len(worker_models)
            worker_gpus_list = [worker_gpus] * len(worker_models)
            worker_names = list(args.worker_names) if args.worker_names else None

            if args.gpt_worker:
                worker_models.append(JUDGE_MODEL)
                worker_types.append(JUDGE_SERVER_TYPE)
                worker_xargs.append(GPT_WORKER_EXTRA_ARGS)
                worker_sargs.append(GPT_WORKER_SERVER_ARGS)
                worker_gpus_list.append(judge_server_gpus)
                if worker_names is not None:
                    worker_names.append(args.gpt_worker_name)
                else:
                    worker_names = [args.gpt_worker_name]

            agent_kwargs.update(
                worker_model=worker_models,
                worker_server_type=worker_types,
                worker_server_gpus=worker_gpus_list,
                worker_server_args=worker_sargs,
                worker_names=worker_names,
                worker_extra_args=worker_xargs,
            )

        agent(**agent_kwargs)

        # eval() for judge + summarize only; waits for agent() to finish.
        # num_jobs=0 causes prepare_eval_commands() to return immediately with
        # no generation job batches (early-return path), so no generation SLURM
        # job is submitted.  Judge and summarize tasks still depend on the agent
        # experiment via run_after=[name].
        eval_kwargs = dict(
            ctx=wrap_arguments(extra_args),
            cluster=args.cluster,
            expname=f"{name}-eval",
            output_dir=odir,
            benchmarks=f"{bcfg['name']}:{bcfg['num_random_seeds']}",
            split=bcfg["split"],
            num_chunks=bcfg["num_chunks"] if bcfg["num_chunks"] > 1 else None,
            # generation params kept for reference but unused when num_jobs=0
            generation_module="nemo_skills.inference.agent_generate",
            model=model_path,
            server_type=MODEL["server_type"],
            server_gpus=server_gpus,
            server_nodes=server_nodes,
            server_args=server_args,
            with_sandbox=True,
            judge_model=JUDGE_MODEL,
            judge_server_type=JUDGE_SERVER_TYPE,
            judge_server_gpus=judge_server_gpus,
            judge_server_args=JUDGE_SERVER_ARGS,
            run_after=[name],
            num_jobs=0,
        )
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
