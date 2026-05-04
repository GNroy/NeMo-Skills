"""
Nemotron-Nano-30B — PhysicsTest + FrontierScience (NVIDIA OFFICIAL params)
=============================================================================
Same benchmarks and 5-seed protocol as `run_nano_physics_fs.py` (r7), but with
all generation parameters set to NVIDIA's officially recommended values from
the model card:

    https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

Per the card:
    "temperature=1.0 and top_p=1.0 are recommended for reasoning tasks,
     while temperature=0.6 and top_p=0.95 are recommended for tool calling."

So each arm gets its own sampling config:
    Arm 1 (no-tool, "reasoning task"):  temperature=1.0, top_p=1.0
    Arm 2 (python,  "tool calling"):    temperature=0.6, top_p=0.95

We also enable the model's official reasoning parser. The model card prescribes
two extra vLLM flags that we did NOT have in r7:

    --reasoning-parser-plugin /hf_models/.../nano_v3_reasoning_parser.py
    --reasoning-parser nano_v3

The plugin file ships inside the HF model directory, so it's already mounted
at the path above on the EOS cluster (verified via clausius MCP, not pulled
from network).

Difference vs r7:
    Param                  r7 (matched HLE baseline)   r-rec (NVIDIA official)
    ----------------------- --------------------------- ------------------------------
    temperature             1.0  (both arms)             1.0 no-tool / 0.6 python
    top_p                   0.95 (both arms)             1.0 no-tool / 0.95 python
    reasoning-parser        (none)                       nano_v3 + plugin
    tokens_to_generate      131072                       131072 (unchanged: judge cap)
    max_tool_calls          50                           50 (unchanged: spiral safety)
    enable_thinking         True                         True (unchanged)
    parse_reasoning         True                         True (unchanged)
    server tool-call-parser qwen3_coder                  qwen3_coder (unchanged)

Why we still keep tokens_to_generate=131072: the on-cluster gpt-oss-120b judge
runs with `--max-model-len 131072`, and pushing the python arm's conversation
past that breaks ~30% of judge jobs (we hit this on r5). Model card just says
"high (e.g., 10,000)"; 131K is well above that.

Why we still keep max_tool_calls=50: model card has no opinion. We've observed
the model spiral past 100 calls on hard chemistry; 50 is generous for the long
tail and only kicks in for the worst few percent of question-seeds.

Output: separate run-id namespace from r7 thanks to the new short name
`nano-rec`. First call creates `eval_physics-nano-rec-{arm}-r1` etc.

Usage:
  conda activate hle-dev
  cd ~/workspace/hle

  # Both benchmarks, both arms (4 generation jobs + 4 judge + 4 summarize):
  CLAUSIUS_URL=http://localhost:7272 \
  NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 \
    python experiments_v7/run_nano_physics_fs_official.py

  # Dry run first to inspect the resolved configs:
  CLAUSIUS_URL=http://localhost:7272 \
  NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 \
    python experiments_v7/run_nano_physics_fs_official.py --dry-run

  # Restrict to one arm/benchmark (useful for debugging):
  CLAUSIUS_URL=http://localhost:7272 \
  NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 \
    python experiments_v7/run_nano_physics_fs_official.py \
        --arms 1 --benchmarks physics
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nemo_skills.pipeline.cli import eval, wrap_arguments

# CLUSTER = "eos"
CLUSTER = "aws-cmh"

OUTPUT_DIR_BASE = "/alaptev/exp/eval/run_nano_physics_fs_official"

# Judge: same as r7 (on-cluster gpt-oss-120b, no external API key needed)
JUDGE_MODEL = "/hf_models/gpt-oss-120b"
JUDGE_SERVER_TYPE = "vllm"
JUDGE_SERVER_ARGS = "--async-scheduling --max-model-len 131072"

# Per-cluster hardware parameters.
CLUSTER_PARAMS = {
    "eos": {
        # A100 80 GB, 8 GPUs/node
        "server_gpus": 8,
        "server_nodes": 1,
        "judge_server_gpus": 8,
        "model_path": "/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "reasoning_parser_path": (
            "/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/nano_v3_reasoning_parser.py"
        ),
        "extra_server_args": "--kv-cache-dtype fp8 ",
    },
    "aws-cmh": {
        # GB300 288 GB HBM/GPU, 4 GPUs/node
        "server_gpus": 4,
        "server_nodes": 1,
        "judge_server_gpus": 4,
        "model_path": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "reasoning_parser_path": "/alaptev/reasoning_parsers/nano_v3_reasoning_parser.py",
        "extra_server_args": "--kv-cache-dtype fp8 ",
    },
}
_DEFAULT_CLUSTER_PARAMS = CLUSTER_PARAMS["eos"]

MODEL = {
    "label": "Nemotron-3-Nano-30B-A3B (official params)",
    "short": "nano-rec",
    "server_type": "vllm",
    # Base server args — cluster-specific reasoning-parser-plugin path and
    # optional fp8 flag are injected at runtime from CLUSTER_PARAMS.
    # Note: --attention-backend FLASH_ATTN is NOT included here because
    # FLASH_ATTN does not support fp8 kv-cache in vLLM 0.18.x.  vLLM
    # auto-selects FlashInfer on GB300 when fp8 is enabled.
    "server_args_base": (
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
    "thinking_args": (
        "++chat_template_kwargs.enable_thinking=True "
        "++parse_reasoning=False "
    ),
    # Shared inference args (NOT temperature/top_p — those are per-arm below)
    "inference_args_shared": (
        "++inference.tokens_to_generate=131072 "
    ),
}

BENCHMARKS = {
    "physics": {
        "benchmarks_str": "physics:5",
        "num_chunks": 10,
        "eval_split": "test",
    },
    "fs": {
        "benchmarks_str": "frontierscience-olympiad:5",
        "num_chunks": 1,
        "eval_split": "all",
    },
    "hle-physics": {
        "benchmarks_str": "hle:5",
        "num_chunks": 1,
        "eval_split": "phy",
    }
}

PYTHON_TOOL = '++tool_modules=["nemo_skills.mcp.servers.python_tool::PythonTool"] '

ARMS = [
    {
        "key": "no-tool",
        "desc": "Baseline reasoning. NVIDIA reasoning preset: temp=1.0, top_p=1.0",
        # Reasoning task per NVIDIA model card.
        "sampling_args": (
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
        ),
        "extra_args": "",
        "sandbox": False,
    },
    {
        "key": "python",
        # "desc": "PythonTool. NVIDIA tool-calling preset: temp=0.6, top_p=0.95",
        "desc": "PythonTool. NVIDIA tool-calling preset: temp=1.0, top_p=1.0",
        # Tool calling task per NVIDIA model card.
        "sampling_args": (
            # "++inference.temperature=0.6 "
            # "++inference.top_p=0.95 "
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
        ),
        "extra_args": PYTHON_TOOL + "++max_tool_calls=50 ",
        "sandbox": True,
    },
]


def main():
    ap = argparse.ArgumentParser(
        description="Nemotron-Nano — PhysicsTest + FrontierScience (NVIDIA official params)",
    )
    ap.add_argument("--arms", default="1,2", help="Arm indices (1=no-tool, 2=python)")
    ap.add_argument("--benchmarks", default="physics,fs", help="Comma-sep benchmark keys")
    ap.add_argument("--cluster", default=CLUSTER)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--run-id",
        type=int,
        required=True,
        help="Run ID for output directory naming (e.g. 1 → eval_fs-nano-rec-no-tool-r1).",
    )
    args = ap.parse_args()

    arm_indices = [int(x) for x in args.arms.split(",")]
    bench_keys = [b.strip() for b in args.benchmarks.split(",")]

    cparams = CLUSTER_PARAMS.get(args.cluster, _DEFAULT_CLUSTER_PARAMS)
    server_args = (
        MODEL["server_args_base"]
        + f"--reasoning-parser-plugin {cparams['reasoning_parser_path']} "
        + cparams["extra_server_args"]
    )

    results = []
    for bkey in bench_keys:
        bcfg = BENCHMARKS[bkey]
        run_id = args.run_id

        for i in arm_indices:
            arm = ARMS[i - 1]
            name = f"eval_{bkey}-{MODEL['short']}-{arm['key']}-r{run_id}"
            odir = (
                f"{OUTPUT_DIR_BASE}/eval_{bkey}-"
                f"{MODEL['short']}-{arm['key']}-r{run_id}"
            )

            print(f"\n{'=' * 60}")
            print(f"  {name}")
            print(f"  {arm['desc']}")
            print(f"  benchmark: {bcfg['benchmarks_str']}")
            print(f"  cluster:   {args.cluster}")
            print(f"  output:    {odir}")
            print(f"{'=' * 60}")

            all_extra_args = (
                arm["sampling_args"]
                + MODEL["inference_args_shared"]
                + MODEL["thinking_args"]
                + arm["extra_args"]
            )

            if args.dry_run:
                print("  [DRY] resolved extra args:")
                for line in all_extra_args.strip().split(" "):
                    if line:
                        print(f"    {line}")
                results.append((name, odir))
                continue

            eval_kwargs = dict(
                ctx=wrap_arguments(all_extra_args),
                cluster=args.cluster,
                expname=name,
                model=cparams["model_path"],
                server_type=MODEL["server_type"],
                server_gpus=cparams["server_gpus"],
                server_nodes=cparams["server_nodes"],
                server_args=server_args,
                benchmarks=bcfg["benchmarks_str"],
                num_chunks=bcfg["num_chunks"],
                split=bcfg["eval_split"],
                output_dir=odir,
                with_sandbox=arm["sandbox"],
                judge_model=JUDGE_MODEL,
                judge_server_type=JUDGE_SERVER_TYPE,
                judge_server_gpus=cparams["judge_server_gpus"],
                judge_server_args=JUDGE_SERVER_ARGS,
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
