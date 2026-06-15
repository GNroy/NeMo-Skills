"""
HLE MCP Ablation v12 — DeepSeek v4 + Kimi K2.6
=============================================================================
Same arm structure as experiments_v11 (28 MCP/skill arms on HLE), now with:

  * Two new generator configs: ``deepseek_v4`` and ``kimi_k26``.
  * Pre-submit waste-compute invariants (see ``waste_invariants.py``) derived
    from mpsf/wasted_compute_root_cause.md — every Slurm job is now either
    a full single node or a multi-node-full allocation, the port policy is
    classified per allocation, and partial-node submissions are forbidden
    on clusters where co-tenant placement is possible.
  * Per-cluster ``sbatch_kwargs`` with the OccupiedIdleGPUsJobReaper
    exemption baked in for aws-dfw.
  * Retrieval arms export ``MCP_TOOL_TIMEOUT_SEC`` so a stuck CPU bi-encoder
    cannot strand the GPU server for hours.
  * ``exclusive`` is no longer always-True — it follows the verdict from
    ``classify_allocation`` so partial-node configs do not strand resources.

Run ID counter is shared with v11 (``experiments/run_id.py`` ->
``experiments/.exp_run_ids.json``) so numbering continues monotonically.

Usage:
  conda activate hle-dev
  cd ~/workspace/mcp

  # DeepSeek v4 — full sweep:
  CLAUSIUS_URL=http://localhost:7272 \\
  NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 \\
    python experiments/experiments_v12/run_hle_mcp_ablation.py \\
      --model deepseek_v4 --cluster aws-cmh --arms 1,2

  # Kimi K2.6 — single arm dry run on aws-cmh:
  ... --model kimi_k26 --cluster aws-cmh --arms 1 --dry-run
"""

import argparse
import re
import sys
from pathlib import Path

# Shared run-ID counter with v11.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
from run_id import next_run_id

for parent in Path(__file__).resolve().parents:
    if (parent / "NeMo-Skills" / "nemo_skills").is_dir():
        sys.path.insert(0, str(parent / "NeMo-Skills"))
        break

from nemo_skills.pipeline import utils as pipeline_utils
from nemo_skills.pipeline.cli import eval, wrap_arguments

from waste_invariants import (
    AllocationVerdict,
    classify_allocation,
    check_tp_matches_alloc,
    require_direct_mcp_tools,
    retrieval_timeout_env,
    sbatch_kwargs_for_cluster,
    print_verdict,
)

# ── Cluster ──────────────────────────────────────────────────────────────

CLUSTER = "aws-cmh"
BENCHMARKS = "hle:5"
SPLIT = "phy"

# ── Per-cluster server args ──────────────────────────────────────────────

# aws-cmh GB300 (4 GPUs/node, 288 GB HBM). Full-node TP=4 keeps every
# arm safely inside the wasted_compute_root_cause "full-node serving"
# invariant.
AWS_CMH_NEMOTRON_PATH = "/workspace/hf_models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
AWS_CMH_NEMOTRON_SERVER_ARGS = (
    "--dtype auto "
    "--tensor-parallel-size 4 "
    "--pipeline-parallel-size 1 "
    "--max-model-len 262144 "
    "--reasoning-parser nemotron_v3 "
    "--enable-expert-parallel "
    "--attention-backend FLASH_ATTN "
    "--trust-remote-code "
    "--gpu-memory-utilization 0.9 "
    "--enable-chunked-prefill "
    "--mamba-ssm-cache-dtype float16 "
    "--enable-auto-tool-choice "
    "--tool-call-parser qwen3_coder "
    "--max-num-seqs 128 "
)
# Nemotron-3-Ultra-550B-A55B-NVFP4 (hybrid Mamba/Transformer + LatentMoE,
# ModelOpt NVFP4 weights).  Full-node TP=4 on one GB300 node (4x288 GB =
# 1152 GB; NVFP4 weights ~275 GB fit with room for fp8 KV cache).  Args
# follow NVIDIA's "AA low-latency" deployment recipe (kv fp8, mamba triton
# backend, nemotron_v3 reasoning parser) — see docs_general/Nemotron.  The
# checkpoint ships hf_quant_config.json so vLLM auto-detects the NVFP4
# quantization; container nemo-skills-vllm-dc43f3e (vLLM 0.22.0) verified to
# register NemotronHForCausalLM + modelopt_fp4 + the nemotron_v3 parser.
AWS_CMH_NEMOTRON3_ULTRA_PATH = "/hf_models/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
AWS_CMH_NEMOTRON3_ULTRA_CONTAINER = (
    "/lustre/fsw/portfolios/nemotron/users/igitman/images/nemo-skills-vllm-dc43f3e.sqsh"
)
AWS_CMH_NEMOTRON3_ULTRA_SERVER_ARGS = (
    "--tensor-parallel-size 4 "
    "--pipeline-parallel-size 1 "
    "--kv-cache-dtype fp8 "
    "--max-model-len 262144 "
    "--max-num-seqs 16 "
    "--max-num-batched-tokens 16384 "
    "--gpu-memory-utilization 0.9 "
    "--enable-chunked-prefill "
    "--enable-prefix-caching "
    "--enable-flashinfer-autotune "
    "--async-scheduling "
    "--mamba-backend triton "
    "--mamba-ssm-cache-dtype float32 "
    "--trust-remote-code "
    "--reasoning-parser nemotron_v3 "
    "--enable-auto-tool-choice "
    "--tool-call-parser qwen3_coder "
    # MTP speculative decoding (model ships shared-weight MTP heads) — the
    # purpose-built decode accelerator for this NVFP4 checkpoint.  JSON form
    # per the HF model card; survives NS verbatim-interpolation like the
    # model-loader-extra-config JSON below.
    """--speculative-config '{"method": "nemotron_h_mtp", "num_speculative_tokens": 5}' """
    """--model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}' """
)
# Nemotron-3-Ultra NVFP4 on SGLang (speed A/B vs vLLM).  Container =
# lmsysorg/sglang:v0.5.11 (the version NVIDIA validated for Ultra NVFP4+MTP),
# imported to alaptev/containers.  NS serve_sglang FORCES --tensor-parallel-size
# (=server_gpus) + --trust-remote-code + --model/--served-model-name/--host/
# --port, so we pass only the extras.  Args follow the deployment doc's proven
# "8xB200 NVFP4 + MTP" sbatch, adapted to one GB300 node (TP4 + EP4):
#   --quantization modelopt_fp4 is REQUIRED for EP>1 (auto modelopt_mixed
#   crashes in fused-MoE weight post-processing); NEXTN MTP needs
#   --disable-radix-cache under mamba no_buffer.
AWS_CMH_NEMOTRON3_ULTRA_SGLANG_CONTAINER = (
    "/lustre/fsw/portfolios/nemotron/users/alaptev/containers/sglang-v0.5.11.sqsh"
)
AWS_CMH_NEMOTRON3_ULTRA_SGLANG_SERVER_ARGS = (
    "--quantization modelopt_fp4 "
    "--expert-parallel-size 4 "
    "--kv-cache-dtype fp8_e4m3 "
    "--context-length 262144 "
    "--mem-fraction-static 0.85 "
    "--chunked-prefill-size 16384 "
    "--mamba-scheduler-strategy no_buffer "  # extra_buffer is rejected for NemotronH
    "--disable-piecewise-cuda-graph "
    "--disable-radix-cache "
    # EAGLE 5/5 per NVIDIA's Nemotron-3-Super deployment guide (same MTP arch;
    # reported accept-len ~3.45) — deeper than the NEXTN 3/4 first try
    # (accept-len 3.1).  spec-v2 is default-on for EAGLE in v0.5.11.
    "--speculative-algorithm EAGLE "
    "--speculative-num-steps 5 "
    "--speculative-eagle-topk 1 "
    "--speculative-num-draft-tokens 5 "
    "--reasoning-parser nemotron_3 "
    "--tool-call-parser qwen3_coder "
)
AWS_CMH_GPTOSS_SERVER_ARGS = (
    "--async-scheduling "
    "--tensor-parallel-size 4 "
    "--max-model-len 131072 "
)
AWS_CMH_GPTOSS_SERVER_ARGS_TOOL = (
    "--async-scheduling "
    "--tensor-parallel-size 4 "
    "--max-model-len 131072 "
    "--enable-auto-tool-choice "
    "--tool-call-parser openai "
    "--reasoning-parser openai_gptoss "
)
AWS_CMH_NEMOTRON_NANO_PATH = "/workspace/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
AWS_CMH_NEMOTRON_NANO_SERVER_ARGS = (
    "--dtype auto "
    "--tensor-parallel-size 1 "
    "--data-parallel-size 4 "
    "--max-model-len 131072 "
    "--attention-backend FLASH_ATTN "
    "--trust-remote-code "
    "--gpu-memory-utilization 0.85 "
    "--mamba-ssm-cache-dtype float32 "
    "--enable-auto-tool-choice "
    "--tool-call-parser qwen3_coder "
    "--max-num-seqs 256 "
)
AWS_CMH_MINIMAX_SERVER_ARGS = (
    "--distributed-executor-backend ray "
    "--trust-remote-code "
    "--enable-expert-parallel "
    "--enable-auto-tool-choice "
    "--tool-call-parser minimax_m2 "
    "--reasoning-parser minimax_m2_append_think "
    "--tensor-parallel-size 4 "
    "--max-model-len 131072 "
)
AWS_CMH_QWEN35_122B_SERVER_ARGS = (
    "--enable-expert-parallel "
    "--trust-remote-code "
    "--enable-auto-tool-choice "
    "--tool-call-parser qwen3_coder "
    "--reasoning-parser qwen3 "
    "--kv-cache-dtype fp8 "
    "--gpu-memory-utilization 0.85 "
    "--tensor-parallel-size 4 "
    "--max-model-len 131072 "
    "--max-num-seqs 256 "
    "--dtype bfloat16 "
)
AWS_DFW_QWEN35_122B_FP8_SERVER_ARGS = (
    "--enable-expert-parallel "
    "--trust-remote-code "
    "--enable-auto-tool-choice "
    "--tool-call-parser qwen3_coder "
    "--reasoning-parser qwen3 "
    "--kv-cache-dtype fp8 "
    "--gpu-memory-utilization 0.85 "
    "--tensor-parallel-size 4 "
    "--max-model-len 131072 "
    "--max-num-seqs 256 "
)
HLE_EVAL_CONFIG_OVERRIDE = "++eval_type=hle_math "
AWS_CMH_QWEN35_397B_SERVER_ARGS = (
    "--distributed-executor-backend ray "
    "--enable-expert-parallel "
    "--trust-remote-code "
    "--enable-auto-tool-choice "
    "--tool-call-parser qwen3_coder "
    "--reasoning-parser qwen3 "
    "--kv-cache-dtype fp8 "
    "--gpu-memory-utilization 0.85 "
    "--enforce-eager "
    "--tensor-parallel-size 16 "
    "--max-model-len 131072 "
    "--max-num-seqs 256 "
    "--dtype bfloat16 "
)

# DeepSeek-V3.2 (legacy) — sglang EP/DP across 16 GPUs (4 nodes x 4).
AWS_CMH_DEEPSEEK_SERVER_ARGS = (
    "--enable-dp-attention "
    "--ep-size 16 "
    "--dp 16 "
    "--tool-call-parser deepseekv32 "
    "--reasoning-parser deepseek-v3 "
    "--log-requests "
    "--mem-fraction-static=0.8 "
    """--model-loader-extra-config '{"enable_multithread_load":true,"num_threads":112}' """
)
AWS_CMH_DEEPSEEK_TOKENIZER = "/hf_models/Qwen3.5-122B-A10B"

# DeepSeek-V4-Pro (new). Server args follow Jiacheng's GB300 DeepGEMM recipe:
# TP=4 x DP=2, with one DP rank per 4-GPU node (8 GPUs total).
#
# Container: jiachengx's prebuilt DeepGEMM-enabled vllm 0.21 image on
# aws-cmh nemotron portfolio (mode 755 jiachengx:dip as of 2026-05-26,
# group-readable for the dip team).  Same image is also at
# /lustre/fsw/portfolios/llmservice/users/jiachengx/images/ for aws-dfw.
AWS_CMH_DEEPSEEK_V4_CONTAINER = (
    # Jiacheng's GB300 DeepGEMM vLLM image. This is a server-only image:
    # keep clients/judges in the normal nemo-skills image.
    "/lustre/fsw/portfolios/nemotron/users/jiachengx/images/"
    "nemo-skills-vllm21-deepgemm-arm64.sqsh"
)
# Server config: jiachengx's recipe from the GB300 benchmark
# (~1.4x throughput vs Sanyam's recipe: 914 -> 1270 tok/s on GPQA-diamond
# COT/no-tool).  Layout: TP=4 x DP=2, local-DP=1 -> one DP rank per node,
# each rank doing TP=4 across the node's 4 GPUs = 8 GPUs total.
AWS_CMH_DEEPSEEK_V4_SERVER_ARGS = (
    # Layout: TP=4 x DP=2, local-DP=1 -> one DP rank per node,
    # each rank doing TP=4 across 4 GPUs = 8 GPUs total (2 nodes x 4 GPUs).
    # Matches Sanyam's nemo-skills-stem model_deepseek_v4_pro_gb config
    # AND jiachengx's GB300 benchmark recipe (~1270 tok/s on GPQA-diamond).
    "--tensor-parallel-size 4 "
    "--data-parallel-size 2 "
    "--data-parallel-size-local 1 "
    "--data-parallel-backend ray "
    "--distributed-executor-backend ray "
    "--enable-expert-parallel "
    "--kv-cache-dtype fp8 "
    "--block-size 256 "
    "--max-model-len 131072 "
    """--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"]}' """
    "--attention_config.use_fp4_indexer_cache=True "
    "--tokenizer-mode deepseek_v4 "
    "--tool-call-parser deepseek_v4 "
    "--enable-auto-tool-choice "
    "--reasoning-parser deepseek_v4 "
    """--model-loader-extra-config '{"enable_multithread_load":true,"num_threads":96}' """
    "--max-num-seqs 512 "
    "--moe-backend deep_gemm_mega_moe "
)
# Sanyam's DSv4 path uses the HF tokenizer that ships with the model, so
# we no longer remap to a Qwen tokenizer.  Set to empty so launcher
# omits ++tokenizer=...
AWS_CMH_DEEPSEEK_V4_TOKENIZER = ""

# vllm needs a longer engine-ready timeout to load DeepSeek-V4-Pro's 64
# shards from /hf_models; threaded via env vars rather than CLI flags.
AWS_CMH_DEEPSEEK_V4_SERVER_ENV = {
    "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
}

# Kimi-K2.6 (new).  Aleksandr's vllm container is pre-baked under
# alaptev's portfolio on aws-cmh; both portfolios live under
# /lustre/fsw/portfolios/nemotron/ so the path is reachable from our
# nemo-run sync.
#
# TP=4 = 1 full GB300 node = full-node allocation per the waste
# invariants.  ``--enable-expert-parallel --distributed-executor-backend
# ray`` is required because K2.6 is MoE; Aleksandr also reports the
# stock kimi_k2 tool-call parser does not match what K2.6 actually
# emits, so we ship a corrected parser plugin (kimi_k26_tool_parser.py)
# alongside this launcher and reference it via ``--tool-parser-plugin``.
# Boot takes ~20 min per Aleksandr; the 4h job wall is enough provided
# we don't pile the boot on top of long retrieval chunks.
# Initially used alaptev/containers/vllm-glm51-cu130-ray.sqsh per
# Aleksandr's note, but that container's binary kernels lack sm_103
# (GB300) support — server crashes at cudagraph capture with
# `cudaErrorNoKernelImageForDevice`.  jiachengx's vllm 0.21 + DeepGEMM
# container was explicitly benchmarked on GB300 (1270 tok/s on DSv4-Pro),
# so it has sm_103 kernels.  Both containers have the `kimi_k2` parser.
AWS_CMH_KIMI_K26_CONTAINER = (
    "/lustre/fsw/portfolios/nemotron/users/alaptev/containers/"
    "vllm-glm51-cu130-ray.sqsh"
)
# Plugin path inside the container.  Our /workspace mount on aws-cmh is
# /lustre/fsw/portfolios/nemotron/users/htamoyan -> /workspace, so any
# file under ~/workspace/mcp/experiments/experiments_v12/ that has been
# rsynced to lustre is reachable here.
KIMI_K26_TOOL_PARSER_PLUGIN = (
    "/alaptev/reasoning_parsers/kimi_k26_tool_parser.py"
)
AWS_CMH_KIMI_K26_SERVER_ARGS = (
    # Aleksandr's actual working recipe (from his minimal sbatch in
    # /lustre/fsw/portfolios/nemotron/users/alaptev/share/kimi_k26_minimal/):
    # DP=4 TP=1 (4 independent replicas, Moonshot's recommendation for
    # K2.6 throughput), --language-model-only to skip the unused vision
    # tower.  Earlier r11/r12/r13/r21 on aws-cmh failed because we used
    # the wrong TP layout or the wrong container; the working combo is
    # this + alaptev/containers/vllm-glm51-cu130-ray.sqsh.
    "--enable-expert-parallel "
    "--distributed-executor-backend=ray "
    "--data-parallel-size 4 "
    "--tensor-parallel-size 1 "
    "--language-model-only "
    "--max-model-len 131072 "
    # fuse_allreduce_rms compilation pass miscompiles on GB300 -> disable
    "--compilation-config '{\"pass_config\": {\"fuse_allreduce_rms\": false}}' "
    "--enable-auto-tool-choice "
    # Custom K2.6 tool-call parser (the stock kimi_k2 regex doesn't match
    # K2.6's emit order).  Plugin lives at our /workspace mount, copied
    # from alaptev's reasoning_parsers/ portfolio.
    "--tool-parser-plugin /alaptev/reasoning_parsers/kimi_k26_tool_parser.py "
    "--tool-call-parser kimi_k26 "
    "--reasoning-parser-plugin /alaptev/reasoning_parsers/kimi_k26_reasoning_parser.py "
    "--reasoning-parser kimi_k26 "
    # Aggressive multi-thread loader: 192 threads (vs 96) cuts the 64-
    # shard weight load from ~6.5 min to ~3 min on lustre.
    """--model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 192}' """
)

# Kimi K2.6 on aws-iad (H100, sm_90):
# - Marlin INT4 MoE kernel works natively (sm_90 binaries present).
# - Single 8-GPU node fits the model with TP=8, matching Moonshot's
#   official deploy guide: https://recipes.vllm.ai/moonshotai/Kimi-K2.6
#   and huggingface.co/moonshotai/Kimi-K2.6/blob/main/docs/deploy_guidance.md
# - No --moe-backend override needed; Marlin is the right backend on H100.
# - Use team-default vllm container (igitman/images/nemo-skills-vllm-latest.sqsh)
#   instead of alaptev's; no need for the GB300-specific DeepGEMM build.
AWS_IAD_KIMI_K26_SERVER_ARGS = (
    # K2.6 INT4 weights (~555 GB on disk, same in memory) do not fit on
    # 8x80 GB H100s with workable headroom: TP=8 puts ~70 GB of weights
    # on each GPU, leaving <10 GB for KV cache + sampler.  r15 (default
    # gpu_memory_utilization=0.9) OOMed at sampler warmup; r16 with 0.85
    # OOMed before cache allocation.  Solution: 2-node multi-node with
    # TP=16, ~35 GB weights / GPU, ~45 GB headroom.
    "--tensor-parallel-size 16 "
    "--distributed-executor-backend ray "
    "--enable-expert-parallel "
    # Drop max-model-len from 131072 to 65536: HLE prompts are <2k
    # tokens, generation cap is 100k, so 65k is well above what any
    # row actually needs.  Frees ~half the per-token KV cache budget,
    # which is what enables max-num-seqs > 64 below.
    "--max-model-len 65536 "
    "--language-model-only "
    "--enable-auto-tool-choice "
    "--tool-call-parser kimi_k2 "
    # Stock `kimi_k2` reasoning parser only handles a single <think>
    # block at the *start* of the output.  K2.6 with Hermes/large system
    # prompts emits mid-stream and multiple <think> blocks; the stock
    # parser leaves them in the visible response → judge scores noise.
    # Aleksandr's `kimi_k26` shim does multi-block regex extraction.
    "--reasoning-parser-plugin /alaptev/reasoning_parsers/kimi_k26_reasoning_parser.py "
    "--reasoning-parser kimi_k26 "
    # Throughput knobs (r19 ran at 211 tok/s = 127h ETA with the old
    # max-num-seqs 64; KV cache was only 72% full, indicating headroom):
    # - max-num-seqs 128 doubles concurrent batch size
    # - async-scheduling overlaps CPU step prep with GPU compute (~10-20%
    #   throughput uplift)
    "--max-num-seqs 128 "
    "--async-scheduling "
    """--model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}' """
)

# ── Judge ────────────────────────────────────────────────────────────────

JUDGE_CONFIGS = {
    "gpt-oss": {
        "judge_model": "/hf_models/gpt-oss-120b",
        "judge_server_type": "vllm",
        "judge_server_gpus": 8,
        "judge_server_args": "--async-scheduling --max-model-len 131072",
        "judge_server_address": None,
        "extra_judge_args": None,
    },
}

# ── Model configs ────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "nemotron": {
        "label": "Nemotron-3-Super-120B-A12B",
        "short": "nemotron120b",
        "path": "/hf_models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        "server_type": "vllm",
        "server_gpus": 8,
        "server_nodes": 1,
        "server_args": (
            "--async-scheduling "
            "--dtype auto "
            "--kv-cache-dtype fp8 "
            "--tensor-parallel-size 4 "
            "--pipeline-parallel-size 1 "
            "--data-parallel-size 2 "
            "--max-model-len 240000 "
            "--enable-expert-parallel "
            "--attention-backend FLASH_ATTN "
            "--trust-remote-code "
            "--gpu-memory-utilization 0.9 "
            "--enable-chunked-prefill "
            "--mamba-ssm-cache-dtype float16 "
            "--enable-auto-tool-choice "
            "--tool-call-parser qwen3_coder "
            "--max-num-seqs 256 "
        ),
        "thinking_args": "++chat_template_kwargs.enable_thinking=True ",
        "inference_args_no_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=131072 ",
        "inference_args_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=80000 ",
        "judge": "gpt-oss",
    },
    "nemotron3_ultra": {
        "label": "Nemotron-3-Ultra-550B-A55B-NVFP4",
        "short": "nem3ultra",
        "path": AWS_CMH_NEMOTRON3_ULTRA_PATH,
        "server_type": "vllm",
        "server_gpus": 4,
        "server_nodes": 1,
        "server_args": AWS_CMH_NEMOTRON3_ULTRA_SERVER_ARGS,
        "server_container": AWS_CMH_NEMOTRON3_ULTRA_CONTAINER,
        "thinking_args": "++chat_template_kwargs.enable_thinking=True ",
        "inference_args_no_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=250000 ",
        "inference_args_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=250000 ",
        "judge": "gpt-oss",
    },
    "nemotron3_ultra_sglang": {
        "label": "Nemotron-3-Ultra-550B-A55B-NVFP4 (SGLang)",
        "short": "nem3ultrasgl",
        "path": AWS_CMH_NEMOTRON3_ULTRA_PATH,
        "server_type": "sglang",
        "server_gpus": 4,
        "server_nodes": 1,
        "server_args": AWS_CMH_NEMOTRON3_ULTRA_SGLANG_SERVER_ARGS,
        "server_container": AWS_CMH_NEMOTRON3_ULTRA_SGLANG_CONTAINER,
        "thinking_args": "++chat_template_kwargs.enable_thinking=True ",
        "inference_args_no_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=250000 ",
        "inference_args_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=250000 ",
        "judge": "gpt-oss",
    },
    "nemotron_nano": {
        "label": "Nemotron-3-Nano-30B-A3B",
        "short": "nemnano30b",
        "path": "/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "server_type": "vllm",
        "server_gpus": 4,
        "server_nodes": 1,
        "server_args": (
            "--dtype auto "
            "--tensor-parallel-size 1 "
            "--data-parallel-size 4 "
            "--max-model-len 131072 "
            "--attention-backend FLASH_ATTN "
            "--trust-remote-code "
            "--gpu-memory-utilization 0.85 "
            "--mamba-ssm-cache-dtype float32 "
            "--enable-auto-tool-choice "
            "--tool-call-parser qwen3_coder "
            "--max-num-seqs 256 "
        ),
        "thinking_args": "++chat_template_kwargs.enable_thinking=True ",
        "inference_args_no_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=131072 ",
        "inference_args_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=80000 ",
        "judge": "gpt-oss",
    },
    "deepseek": {
        "label": "DeepSeek-V3.2",
        "short": "dsv32",
        "path": "/hf_models/DeepSeek-V3.2",
        "tokenizer": "/hf_models/Qwen2.5-32B-Instruct",
        "server_type": "sglang",
        "server_gpus": 4,
        "server_nodes": 4,
        "server_args": AWS_CMH_DEEPSEEK_SERVER_ARGS,
        "thinking_args": "++chat_template_kwargs.thinking=true ",
        "inference_args_no_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=100000 ",
        "inference_args_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=100000 ",
        "judge": "gpt-oss",
    },
    "deepseek_v4": {
        # DeepSeek-V4-Pro (verified on aws-cmh: 806G, 65 safetensors at
        # /lustre/fsw/portfolios/nemotron/users/igitman/hf_models/
        # DeepSeek-V4-Pro -> /hf_models/DeepSeek-V4-Pro inside the
        # container). Server config is Jiacheng's GB300 DeepGEMM recipe:
        # TP=4 x DP=2, 4 GPUs/node x 2 nodes = 8 GPUs total.
        # The custom image is server-only; the eval client and judges stay
        # on the normal nemo-skills image.
        "label": "DeepSeek-V4-Pro",
        "short": "dsv4pro",
        "path": "/hf_models/DeepSeek-V4-Pro",
        # tokenizer="" means launcher does NOT pass ++tokenizer= (vllm uses
        # the HF tokenizer that ships with the model).
        "tokenizer": AWS_CMH_DEEPSEEK_V4_TOKENIZER,
        "server_type": "vllm",
        "server_gpus": 4,
        "server_nodes": 2,
        "server_args": AWS_CMH_DEEPSEEK_V4_SERVER_ARGS,
        "server_container": AWS_CMH_DEEPSEEK_V4_CONTAINER,
        "server_env": AWS_CMH_DEEPSEEK_V4_SERVER_ENV,
        # Per Sanyam's inference_deepseek_v4: thinking=true,
        # reasoning_effort=high, top_p=1.0, tokens_to_generate=100k.
        "thinking_args": "++chat_template_kwargs.thinking=true ++chat_template_kwargs.reasoning_effort=high ",
        "inference_args_no_tool": "++inference.temperature=1.0 ++inference.top_p=1.0 ++inference.tokens_to_generate=100000 ",
        "inference_args_tool": "++inference.temperature=1.0 ++inference.top_p=1.0 ++inference.tokens_to_generate=100000 ",
        "judge": "gpt-oss",
    },
    "kimi_k26": {
        # Kimi-K2.6 (verified on aws-cmh: 555G, 65 safetensors at
        # /lustre/fsw/portfolios/nemotron/users/igitman/hf_models/
        # Kimi-K2.6 -> /hf_models/Kimi-K2.6 inside the container).
        # config.json reports model_type=kimi_k25 (same architecture as
        # K2.5); only the tool-call emit format differs, which the in-
        # tree kimi_k26_tool_parser.py shim handles.
        #
        # Container: alaptev's vllm-glm51-cu130-ray.sqsh on lustre (19.8
        # GB) — verified present.  Full-node TP=4 on GB300; boots ~20m.
        "label": "Kimi-K2.6",
        "short": "kimi-k26",
        "path": "/hf_models/Kimi-K2.6",
        "server_type": "vllm",
        "server_gpus": 4,
        "server_nodes": 1,
        "server_args": AWS_CMH_KIMI_K26_SERVER_ARGS,
        "server_container": AWS_CMH_KIMI_K26_CONTAINER,
        # Aleksandr's snippet for disabling Kimi K2.6 thinking is
        # ``++inference.extra_body.chat_template_kwargs.thinking=false``.
        # We default to enabling thinking (true) for parity with our
        # DeepSeek path; if the model rejects the flag, drop this to "".
        "thinking_args": "++inference.extra_body.chat_template_kwargs.thinking=true ",
        "inference_args_no_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=100000 ",
        "inference_args_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=100000 ",
        "judge": "gpt-oss",
    },
    "gptoss": {
        "label": "gpt-oss-120b",
        "short": "gptoss120b",
        "path": "/hf_models/gpt-oss-120b",
        "server_type": "vllm",
        "server_gpus": 8,
        "server_nodes": 1,
        "server_args": "--async-scheduling --max-model-len 131072 ",
        "server_args_tool": (
            "--async-scheduling --max-model-len 131072 "
            "--enable-auto-tool-choice --tool-call-parser openai "
            "--reasoning-parser openai_gptoss "
        ),
        "thinking_args": "++chat_template_kwargs.reasoning_effort=high ",
        "inference_args_no_tool": "++inference.temperature=1.0 ++inference.top_p=1.0 ++inference.tokens_to_generate=100000 ",
        "inference_args_tool": "++inference.temperature=1.0 ++inference.top_p=1.0 ++inference.tokens_to_generate=100000 ++inference.endpoint_type=responses ",
        "judge": "gpt-oss",
    },
    "minimax": {
        "label": "MiniMax-M2.5",
        "short": "mm25",
        "path": "/hf_models/MiniMax-M2.5",
        "server_type": "vllm",
        "server_gpus": 8,
        "server_nodes": 1,
        "server_args": (
            "--trust-remote-code "
            "--enable-expert-parallel "
            "--enable-auto-tool-choice "
            "--tool-call-parser minimax_m2 "
            "--reasoning-parser minimax_m2_append_think "
            "--max-model-len 131072 "
        ),
        "thinking_args": "",
        "inference_args_no_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=100000 ",
        "inference_args_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=100000 ",
        "judge": "gpt-oss",
    },
    "qwen35": {
        "label": "Qwen3.5-122B-A10B",
        "short": "qwen35-122b",
        "path": "/hf_models/Qwen3.5-122B-A10B",
        "server_type": "vllm",
        "server_gpus": 8,
        "server_nodes": 1,
        "server_args": (
            "--enable-expert-parallel "
            "--trust-remote-code "
            "--enable-auto-tool-choice "
            "--tool-call-parser qwen3_coder "
            "--reasoning-parser qwen3 "
            "--kv-cache-dtype fp8 "
            "--gpu-memory-utilization 0.85 "
            "--max-model-len 131072 "
            "--max-num-seqs 256 "
            "--dtype bfloat16 "
        ),
        "thinking_args": "++chat_template_kwargs.enable_thinking=True ",
        "inference_args_no_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=100000 ",
        "inference_args_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=100000 ",
        "judge": "gpt-oss",
    },
    "qwen35_397b": {
        "label": "Qwen3.5-397B-A17B",
        "short": "qwen35-397b",
        "path": "/hf_models/Qwen3.5-397B-A17B",
        "server_type": "vllm",
        "server_gpus": 8,
        "server_nodes": 2,
        "server_args": (
            "--distributed-executor-backend ray "
            "--enable-expert-parallel "
            "--trust-remote-code "
            "--enable-auto-tool-choice "
            "--tool-call-parser qwen3_coder "
            "--reasoning-parser qwen3 "
            "--kv-cache-dtype fp8 "
            "--gpu-memory-utilization 0.85 "
            "--enforce-eager "
            "--max-model-len 131072 "
            "--max-num-seqs 256 "
            "--dtype bfloat16 "
        ),
        "thinking_args": "++chat_template_kwargs.enable_thinking=True ",
        "inference_args_no_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=100000 ",
        "inference_args_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=100000 ",
        "judge": "gpt-oss",
    },
}

# ── Tool module paths (Direct* in-process variants only) ────────────────

_PYTHON = "nemo_skills.mcp.servers.python_tool::PythonTool"
_PARTICLE = "nemo_skills.mcp.servers.physics.particle_tool::ParticleTool"
_RADIOACTIVEDECAY = (
    "nemo_skills.mcp.servers.physics.radioactivedecay_tool::RadioactivedecayTool"
)
_COOLPROP = "nemo_skills.mcp.servers.physics.coolprop_tool::CoolPropTool"
_PERIODICTABLE = "nemo_skills.mcp.servers.chemistry.periodictable_tool::PeriodictableTool"
# Wasted-compute Pattern 5 (MCP stdio cleanup): every retrieval tool path
# uses the Direct* in-process variant so async transports cannot leak.
_LIBRETEXTS = "nemo_skills.mcp.servers.web.libretexts_tool::DirectLibreTextsTool"
_ARXIV = "nemo_skills.mcp.servers.web.arxiv_local_tool::DirectArxivTool"
_WIKIPEDIA = "nemo_skills.mcp.servers.web.wikipedia_local_tool::DirectWikipediaTool"
_MCP_GUIDE = "nemo_skills.mcp.servers.mcp_skill_tool::MCPGuideTool"

_LOCAL_MCP_TOOLS = (_PARTICLE, _RADIOACTIVEDECAY, _COOLPROP, _PERIODICTABLE)
_RETRIEVAL_MCP_TOOLS = (_LIBRETEXTS, _ARXIV, _WIKIPEDIA)

_LOCAL_MCP_INSTALL = "pip install particle radioactivedecay CoolProp periodictable"
_RETRIEVAL_INSTALL = (
    "pip install bm25s sentence-transformers faiss-cpu && "
    f"{retrieval_timeout_env()} && "
    "export ARXIV_INDEX_DIR=/workspace/mcp-retrieval-indices/arxiv && "
    "export WIKIPEDIA_INDEX_DIR=/workspace/mcp-retrieval-indices/wikipedia && "
    "export LIBRETEXTS_INDEX_DIR=/workspace/libretexts_data/index-gte-large"
)
_ALL_MCP_INSTALL = f"{_LOCAL_MCP_INSTALL} && {_RETRIEVAL_INSTALL}"

_KNOWLEDGE_PATH_FILTERS = (
    "++tool_overrides.WikipediaLocalTool.enabled_tools="
    "[wikipedia-search,wikipedia-summary] "
    "++tool_overrides.LibreTextsTool.enabled_tools=[libretexts-search] "
)
_WIKIPEDIA_METADATA_FILTERS = (
    "++tool_overrides.WikipediaLocalTool.enabled_tools="
    "[wikipedia-search,wikipedia-summary] "
)
_LIBRETEXTS_METADATA_FILTERS = "++tool_overrides.LibreTextsTool.enabled_tools=[libretexts-search] "

# Helper used by every retrieval arm's install block.
_RETRIEVAL_THREAD_PINS = (
    "export OPENBLAS_NUM_THREADS=1 && "
    "export OMP_NUM_THREADS=1 && "
    "export MKL_NUM_THREADS=1 && "
    "export OPENBLAS_VERBOSE=0 && "
    f"{retrieval_timeout_env()}"
)

# ── Arms (preserved from v11; install blocks now thread MCP_TOOL_TIMEOUT_SEC) ─

def _tool_modules(*modules, max_tool_calls=50):
    joined = ",".join(f'"{m}"' for m in modules)
    return f"++tool_modules=[{joined}] ++max_tool_calls={max_tool_calls} "


def _guide_skills(*skills):
    return f"++tool_overrides.MCPGuideTool.enabled_skills=[{','.join(skills)}] "


ARMS = [
    {
        "key": "no-tool",
        "desc": "Baseline: reasoning only",
        "extra_args": "",
        "prompt_config": None,
        "sandbox": False,
        "use_tool_inference": False,
        "install": None,
    },
    {
        "key": "python",
        "desc": "+ PythonTool",
        "extra_args": _tool_modules(_PYTHON),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": None,
    },
    {
        "key": "python+particle",
        "desc": "+ PythonTool + ParticleTool (PDG particle data)",
        "extra_args": _tool_modules(_PYTHON, _PARTICLE),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": "pip install particle",
    },
    {
        "key": "python+radioactivedecay",
        "desc": "+ PythonTool + RadioactivedecayTool",
        "extra_args": _tool_modules(_PYTHON, _RADIOACTIVEDECAY),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": "pip install radioactivedecay",
    },
    {
        "key": "python+coolprop",
        "desc": "+ PythonTool + CoolPropTool",
        "extra_args": _tool_modules(_PYTHON, _COOLPROP),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": "pip install CoolProp",
    },
    {
        "key": "python+periodictable",
        "desc": "+ PythonTool + PeriodictableTool",
        "extra_args": _tool_modules(_PYTHON, _PERIODICTABLE),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": "pip install periodictable",
    },
    {
        "key": "python+libretexts",
        "desc": "+ PythonTool + DirectLibreTextsTool (in-process)",
        "extra_args": _tool_modules(_PYTHON, _LIBRETEXTS, max_tool_calls=25),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": (
            "pip install bm25s sentence-transformers faiss-cpu && "
            f"{_RETRIEVAL_THREAD_PINS} && "
            "export LIBRETEXTS_INDEX_DIR=/workspace/mcp-retrieval-indices/libretexts"
        ),
        "num_chunks": 5,
    },
    {
        "key": "python+arxiv",
        "desc": "+ PythonTool + DirectArxivTool (in-process)",
        "extra_args": _tool_modules(_PYTHON, _ARXIV, max_tool_calls=12),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": (
            "pip install bm25s sentence-transformers faiss-cpu && "
            f"{_RETRIEVAL_THREAD_PINS} && "
            "export ARXIV_INDEX_DIR=/workspace/mcp-retrieval-indices/arxiv"
        ),
        "num_chunks": 5,
    },
    {
        "key": "python+wikipedia",
        "desc": "+ PythonTool + DirectWikipediaTool (in-process)",
        "extra_args": _tool_modules(_PYTHON, _WIKIPEDIA, max_tool_calls=25),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": (
            "pip install bm25s sentence-transformers faiss-cpu && "
            f"{_RETRIEVAL_THREAD_PINS} && "
            "export WIKIPEDIA_INDEX_DIR=/workspace/mcp-retrieval-indices/wikipedia"
        ),
        "num_chunks": 5,
    },
    {
        "key": "python+mcp-local-policy",
        "desc": "+ deterministic MCPs + concise policy prompt",
        "extra_args": _tool_modules(_PYTHON, *_LOCAL_MCP_TOOLS, max_tool_calls=30),
        "prompt_config": "generic/hle-mcp-policy",
        "sandbox": True,
        "use_tool_inference": True,
        "install": _LOCAL_MCP_INSTALL,
    },
    {
        "key": "python+mcp-local-skill",
        "desc": "+ deterministic MCPs + lazy MCP guide activation",
        "extra_args": _tool_modules(_PYTHON, _MCP_GUIDE, *_LOCAL_MCP_TOOLS, max_tool_calls=30),
        "prompt_config": "generic/hle-mcp-skill",
        "sandbox": True,
        "use_tool_inference": True,
        "install": _LOCAL_MCP_INSTALL,
    },
    {
        "key": "python+mcp-local-inline-docs",
        "desc": "+ deterministic MCPs + same guide inlined in prompt",
        "extra_args": _tool_modules(_PYTHON, *_LOCAL_MCP_TOOLS, max_tool_calls=30),
        "prompt_config": "generic/hle-mcp-inline-docs",
        "sandbox": True,
        "use_tool_inference": True,
        "install": _LOCAL_MCP_INSTALL,
    },
    {
        "key": "knowledge-path",
        "desc": "+ retrieval MCPs + lazy guide + strict budgets",
        "extra_args": _tool_modules(_PYTHON, _MCP_GUIDE, *_RETRIEVAL_MCP_TOOLS, max_tool_calls=12)
        + _KNOWLEDGE_PATH_FILTERS,
        "prompt_config": "generic/hle-mcp-skill",
        "sandbox": True,
        "use_tool_inference": True,
        "install": _RETRIEVAL_INSTALL,
        "num_chunks": 5,
    },
    {
        "key": "overloaded-tools",
        "desc": "+ all seven MCPs without prompt/skill filtering",
        "extra_args": _tool_modules(_PYTHON, *_LOCAL_MCP_TOOLS, *_RETRIEVAL_MCP_TOOLS, max_tool_calls=50),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": _ALL_MCP_INSTALL,
        "num_chunks": 5,
    },
    {
        "key": "python+mcp-all-skill",
        "desc": "+ all seven MCPs + lazy MCP guide activation",
        "extra_args": _tool_modules(
            _PYTHON, _MCP_GUIDE, *_LOCAL_MCP_TOOLS, *_RETRIEVAL_MCP_TOOLS, max_tool_calls=50
        )
        + _KNOWLEDGE_PATH_FILTERS,
        "prompt_config": "generic/hle-mcp-skill",
        "sandbox": True,
        "use_tool_inference": True,
        "install": _ALL_MCP_INSTALL,
        "num_chunks": 10,
    },
    {
        "key": "python+arxiv-metadata-skill-max1",
        "desc": "arXiv-only metadata+skill, max_tool_calls=1",
        "extra_args": _tool_modules(_PYTHON, _MCP_GUIDE, _ARXIV, max_tool_calls=1),
        "prompt_config": "generic/hle-mcp-arxiv-metadata-skill",
        "sandbox": True,
        "use_tool_inference": True,
        "install": "unset OPENALEX_API_KEY; export ARXIV_REQUEST_INTERVAL=4.0; export ARXIV_RATE_LIMIT_LOCK=/workspace/mcp-experiments/arxiv_api_rate_limit.lock; pip install httpx",
        "num_chunks": 5,
    },
    {
        "key": "python+arxiv-metadata-skill-max5",
        "desc": "arXiv-only metadata+skill, max_tool_calls=5",
        "extra_args": _tool_modules(_PYTHON, _MCP_GUIDE, _ARXIV, max_tool_calls=5),
        "prompt_config": "generic/hle-mcp-arxiv-metadata-skill",
        "sandbox": True,
        "use_tool_inference": True,
        "install": "unset OPENALEX_API_KEY; export ARXIV_REQUEST_INTERVAL=4.0; export ARXIV_RATE_LIMIT_LOCK=/workspace/mcp-experiments/arxiv_api_rate_limit.lock; pip install httpx",
        "num_chunks": 5,
    },
    {
        "key": "python+wikipedia-metadata-skill-max100",
        "desc": "Wikipedia-only metadata+skill, max_tool_calls=100",
        "extra_args": _tool_modules(_PYTHON, _MCP_GUIDE, _WIKIPEDIA, max_tool_calls=100)
        + _guide_skills("science-tools", "python-computation", "python", "wikipedia")
        + _WIKIPEDIA_METADATA_FILTERS,
        "prompt_config": "generic/hle-mcp-wikipedia-metadata-skill",
        "sandbox": True,
        "use_tool_inference": True,
        "install": "pip install httpx",
        "num_chunks": 10,
    },
    {
        "key": "python+arxiv-metadata-skill-max100",
        "desc": "arXiv-only metadata+skill, max_tool_calls=100",
        "extra_args": _tool_modules(_PYTHON, _MCP_GUIDE, _ARXIV, max_tool_calls=100)
        + _guide_skills("science-tools", "python-computation", "python", "arxiv"),
        "prompt_config": "generic/hle-mcp-arxiv-metadata-skill",
        "sandbox": True,
        "use_tool_inference": True,
        "install": "unset OPENALEX_API_KEY; export ARXIV_REQUEST_INTERVAL=4.0; export ARXIV_RATE_LIMIT_LOCK=/workspace/mcp-experiments/arxiv_api_rate_limit.lock; pip install httpx",
        "num_chunks": 10,
    },
    {
        "key": "python+libretexts-metadata-skill-max100",
        "desc": "LibreTexts-only metadata+skill, max_tool_calls=100",
        "extra_args": _tool_modules(_PYTHON, _MCP_GUIDE, _LIBRETEXTS, max_tool_calls=100)
        + _guide_skills("science-tools", "python-computation", "python", "libretexts")
        + _LIBRETEXTS_METADATA_FILTERS,
        "prompt_config": "generic/hle-mcp-libretexts-metadata-skill",
        "sandbox": True,
        "use_tool_inference": True,
        "install": "pip install bm25s sentence-transformers faiss-cpu",
        "num_chunks": 10,
    },
    {
        "key": "python+particle-skill-max100",
        "desc": "Particle-only skill-guided MCP, max_tool_calls=100",
        "extra_args": _tool_modules(_PYTHON, _MCP_GUIDE, _PARTICLE, max_tool_calls=100)
        + _guide_skills("science-tools", "python-computation", "python", "particle"),
        "prompt_config": "generic/hle-mcp-skill",
        "sandbox": True,
        "use_tool_inference": True,
        "install": "pip install particle",
        "num_chunks": 10,
    },
    {
        "key": "python+radioactivedecay-skill-max100",
        "desc": "Radioactivedecay-only skill-guided MCP, max_tool_calls=100",
        "extra_args": _tool_modules(_PYTHON, _MCP_GUIDE, _RADIOACTIVEDECAY, max_tool_calls=100)
        + _guide_skills(
            "science-tools",
            "python-computation",
            "python",
            "radioactivedecay",
        ),
        "prompt_config": "generic/hle-mcp-skill",
        "sandbox": True,
        "use_tool_inference": True,
        "install": "pip install radioactivedecay",
        "num_chunks": 10,
    },
    {
        "key": "python+coolprop-skill-max100",
        "desc": "CoolProp-only skill-guided MCP, max_tool_calls=100",
        "extra_args": _tool_modules(_PYTHON, _MCP_GUIDE, _COOLPROP, max_tool_calls=100)
        + _guide_skills("science-tools", "python-computation", "python", "coolprop"),
        "prompt_config": "generic/hle-mcp-skill",
        "sandbox": True,
        "use_tool_inference": True,
        "install": "pip install CoolProp",
        "num_chunks": 10,
    },
    {
        "key": "python+periodictable-skill-max100",
        "desc": "Periodictable-only skill-guided MCP, max_tool_calls=100",
        "extra_args": _tool_modules(_PYTHON, _MCP_GUIDE, _PERIODICTABLE, max_tool_calls=100)
        + _guide_skills("science-tools", "python-computation", "python", "periodictable"),
        "prompt_config": "generic/hle-mcp-skill",
        "sandbox": True,
        "use_tool_inference": True,
        "install": "pip install periodictable",
        "num_chunks": 10,
    },
    # ── Canonical-grid arms (max_tool_calls=50, match mcp_experiment_grid.csv) ─
    # Use these for filling the cross-model canonical grid; chunked so a
    # single chunk fits under the 4h walltime even on the slow retrieval arms.
    {
        "key": "python-max50",
        "desc": "PythonTool (canonical grid, max_tool_calls=50)",
        "extra_args": _tool_modules(_PYTHON, max_tool_calls=50),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": None,
    },
    {
        "key": "python+arxiv-max50",
        "desc": "PythonTool + DirectArxivTool (canonical, max=50)",
        "extra_args": _tool_modules(_PYTHON, _ARXIV, max_tool_calls=50),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": (
            "pip install bm25s sentence-transformers faiss-cpu && "
            f"{_RETRIEVAL_THREAD_PINS} && "
            "export ARXIV_INDEX_DIR=/workspace/mcp-retrieval-indices/arxiv"
        ),
        "num_chunks": 10,
    },
    {
        "key": "python+libretexts-max50",
        "desc": "PythonTool + DirectLibreTextsTool (canonical, max=50)",
        "extra_args": _tool_modules(_PYTHON, _LIBRETEXTS, max_tool_calls=50),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": (
            "pip install bm25s sentence-transformers faiss-cpu && "
            f"{_RETRIEVAL_THREAD_PINS} && "
            "export LIBRETEXTS_INDEX_DIR=/workspace/mcp-retrieval-indices/libretexts"
        ),
        "num_chunks": 10,
    },
    {
        "key": "python+wikipedia-max50",
        "desc": "PythonTool + DirectWikipediaTool (canonical, max=50)",
        "extra_args": _tool_modules(_PYTHON, _WIKIPEDIA, max_tool_calls=50),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": (
            "pip install bm25s sentence-transformers faiss-cpu && "
            f"{_RETRIEVAL_THREAD_PINS} && "
            "export WIKIPEDIA_INDEX_DIR=/workspace/mcp-retrieval-indices/wikipedia"
        ),
        "num_chunks": 10,
    },
    # ── Moonshot-hypothesis arm (added 2026-06-04): code + LIVE web search ──
    # Tests whether tools HELP once the model has search (not python-only).
    # Uses the in-repo live WikipediaSearchTool (REST/MediaWiki API, no key,
    # needs internet — aws-cmh compute nodes have it). No local index / bm25s
    # needed, so no install. Pair with --context-strategy blank to mimic
    # Moonshot's "retain only the most recent tool round" context management.
    {
        "key": "python+wikisearch",
        "desc": "+ PythonTool + WikipediaSearchTool (live web search)",
        "extra_args": _tool_modules(
            _PYTHON,
            "nemo_skills.mcp.servers.web.wikipedia_tool::WikipediaSearchTool",
            max_tool_calls=50,
        ),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": None,
        "num_chunks": 10,
    },
    # Live arXiv + Wikipedia retrieval baseline (SCI-575 web-search direction). Both are in-process
    # Tool classes hitting export.arxiv.org / wikipedia.org; no API key, deps (httpx) already in image,
    # compute nodes have egress. Paired with the python-degradation discipline (cap=8 + run-flag
    # force_final_answer + timeout=60) so search/read/compute spirals still resolve to a final answer.
    {
        "key": "python+arxiv-wiki-live",
        "desc": "+ PythonTool + live arXiv + live Wikipedia search (no API key; compute-node egress)",
        "extra_args": _tool_modules(
            _PYTHON,
            "nemo_skills.mcp.servers.web.arxiv_tool::ArxivSearchTool",
            "nemo_skills.mcp.servers.web.wikipedia_tool::WikipediaSearchTool",
            max_tool_calls=8,
        ),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": None,
    },
]

# Canonical arm-index quintet used by run_canonical_sweep.sh — 5 arms per
# (model, benchmark) cell in the same order they appear in
# mcp_experiment_grid.csv.  Indices are 1-based and point into ARMS above.
# ARMS has 28 entries; the canonical block is the last 4 (25-28).
CANONICAL_ARM_INDICES = {
    "no-tool":                1,   # baseline (no tools)
    "python-max50":          25,
    "python+arxiv-max50":    26,
    "python+libretexts-max50": 27,
    "python+wikipedia-max50":  28,
}
CANONICAL_ARM_CLI = ",".join(str(v) for v in CANONICAL_ARM_INDICES.values())

# ── Naming ───────────────────────────────────────────────────────────────


def _expname(model_short, arm_key, run_id, ctx_tag=""):
    suffix = f"-{ctx_tag}" if ctx_tag else ""
    return f"ablation_{model_short}-{arm_key}{suffix}-r{run_id}"


def _output_dir(model_short, arm_key, run_id, ctx_tag=""):
    suffix = f"-{ctx_tag}" if ctx_tag else ""
    # Adapted from Hovhannes's /workspace/mcp-experiments/ convention to
    # our /alaptev/exp/baselines/ tree (lustre-backed, container-mounted
    # under /alaptev).
    return f"/alaptev/exp/baselines/ablation-{model_short}-{arm_key}{suffix}-r{run_id}"


CONTEXT_STRATEGY_MAP = {
    "drop": "drop_earlier_turns",
    "blank": "drop_earlier_tool_outputs_only",
    "summary": "summarize_earlier_turns",
}


# Benchmark + split -> expname/output_dir tag.  ``hle/phy`` stays empty
# to preserve the historical canonical-grid naming; every other
# (benchmark, split) pair gets a unique tag so the canonical sweep can
# put 3 benchmark cells under the same model+arm without colliding.
def _benchmark_tag(benchmark: str, split: str) -> str:
    if benchmark == "hle":
        return "" if split == "phy" else "hletext"
    if benchmark == "physics":
        return "physen" if split == "test" else "physics"
    if benchmark == "frontierscience-olympiad":
        return "fsolymphys" if split == "physics" else "fsolymp"
    if benchmark == "ugphysics":
        return "ugphys"
    return benchmark


def _replace_or_append_hydra_arg(arg_string: str, key: str, value) -> str:
    """Set a Hydra CLI override while preserving the surrounding arg string."""
    import shlex

    prefix = f"{key}="
    plus_prefix = f"++{key}="
    replacement = f"++{key}={value}"
    tokens = shlex.split(arg_string or "")
    replaced = False
    for idx, token in enumerate(tokens):
        if token.startswith(prefix) or token.startswith(plus_prefix):
            tokens[idx] = replacement
            replaced = True
    if not replaced:
        tokens.append(replacement)
    return " ".join(shlex.quote(token) for token in tokens) + " "


def _require_mcp_cluster(cluster):
    cluster_cfg = pipeline_utils.get_cluster_config(cluster)
    prefix = cluster_cfg.get("job_name_prefix", "")
    if prefix != "mcp_":
        raise SystemExit(
            f"Refusing to submit MCP jobs with cluster config {cluster!r}: "
            f"job_name_prefix is {prefix!r}, expected 'mcp_'."
        )
    return prefix


def _apply_cluster_overrides(args, model_cfg, judge_cfg):
    """Adjust path/server fields for the cluster.  Mirrors v11 with two
    additions: deepseek_v4 and kimi_k26 are aws-cmh-only by default."""

    if args.cluster.startswith("aws-dfw"):
        if args.model == "gptoss":
            model_cfg.update({
                "path": "/shared_hf_models/gpt-oss-120b",
                "tokenizer": "/shared_hf_models/gpt-oss-120b",
                "server_gpus": 4,
                "server_nodes": 1,
                "server_args": AWS_CMH_GPTOSS_SERVER_ARGS,
                "server_args_tool": AWS_CMH_GPTOSS_SERVER_ARGS_TOOL,
            })
        elif args.model == "qwen35":
            model_cfg.update({
                "path": "/shared_hf_models/Qwen3.5-122B-A10B-FP8",
                "tokenizer": "/shared_hf_models/Qwen3.5-122B-A10B-FP8",
                "server_gpus": 4,
                "server_nodes": 1,
                "server_args": AWS_DFW_QWEN35_122B_FP8_SERVER_ARGS,
            })
        elif args.model in ("deepseek_v4", "kimi_k26"):
            raise SystemExit(
                f"{args.model!r} not yet wired for aws-dfw; submit on "
                "aws-cmh where the containers live."
            )
        if judge_cfg.get("judge_model") == "/hf_models/gpt-oss-120b":
            judge_cfg["judge_model"] = "/shared_hf_models/gpt-oss-120b"
        if judge_cfg.get("judge_server_gpus") == 8:
            judge_cfg["judge_server_gpus"] = 4
            judge_cfg["judge_server_args"] = AWS_CMH_GPTOSS_SERVER_ARGS

    elif args.cluster.startswith("aws-iad"):
        if args.model == "minimax":
            model_cfg.update({
                "path": "/workspace/hf_models/MiniMax-M2.5",
                "tokenizer": "/workspace/hf_models/MiniMax-M2.5",
                "server_gpus": 8,
                "server_nodes": 1,
                "server_args": AWS_CMH_MINIMAX_SERVER_ARGS.replace(
                    "--tensor-parallel-size 4 ",
                    "--tensor-parallel-size 8 ",
                ),
            })
        elif args.model == "kimi_k26":
            # 2 nodes x 8 H100 = 16 GPUs total (TP=16), needed because
            # K2.6 INT4 weights at TP=8 leave no room for KV cache.
            # No Marlin-on-GB300 problem here: H100 is sm_90 native.
            # Use team-default vllm container (not jiachengx's GB300 image).
            model_cfg.update({
                "path": "/hf_models/Kimi-K2.6",
                "server_gpus": 8,
                "server_nodes": 2,
                "server_args": AWS_IAD_KIMI_K26_SERVER_ARGS,
                "main_container": None,  # let cluster_config default win
            })
        elif args.model == "deepseek_v4":
            raise SystemExit(
                "deepseek_v4 not yet wired for aws-iad; jiachengx container "
                "is aws-cmh-only. Submit DSv4 on aws-cmh."
            )

    elif args.cluster.startswith("aws-cmh"):
        if args.model == "nemotron":
            model_cfg.update({
                "path": AWS_CMH_NEMOTRON_PATH,
                "server_gpus": 4, "server_nodes": 1,
                "server_args": AWS_CMH_NEMOTRON_SERVER_ARGS,
            })
        elif args.model == "nemotron_nano":
            model_cfg.update({
                "path": AWS_CMH_NEMOTRON_NANO_PATH,
                "server_gpus": 4, "server_nodes": 1,
                "server_args": AWS_CMH_NEMOTRON_NANO_SERVER_ARGS,
            })
        elif args.model == "gptoss":
            model_cfg.update({
                "server_gpus": 4, "server_nodes": 1,
                "server_args": AWS_CMH_GPTOSS_SERVER_ARGS,
                "server_args_tool": AWS_CMH_GPTOSS_SERVER_ARGS_TOOL,
            })
        elif args.model == "minimax":
            model_cfg.update({
                "server_gpus": 4, "server_nodes": 2,
                "server_args": AWS_CMH_MINIMAX_SERVER_ARGS,
            })
        elif args.model == "qwen35":
            model_cfg.update({
                "server_gpus": 4, "server_nodes": 1,
                "server_args": AWS_CMH_QWEN35_122B_SERVER_ARGS,
            })
        elif args.model == "qwen35_397b":
            model_cfg.update({
                "server_gpus": 4, "server_nodes": 4,
                "server_args": AWS_CMH_QWEN35_397B_SERVER_ARGS,
            })
        elif args.model == "deepseek":
            model_cfg.update({
                "tokenizer": AWS_CMH_DEEPSEEK_TOKENIZER,
                "server_gpus": 4, "server_nodes": 4,
                "server_args": AWS_CMH_DEEPSEEK_SERVER_ARGS,
            })
        elif args.model == "deepseek_v4":
            if not AWS_CMH_DEEPSEEK_V4_CONTAINER:
                raise SystemExit(
                    "deepseek_v4: AWS_CMH_DEEPSEEK_V4_CONTAINER is empty. "
                    "Set AWS_CMH_DEEPSEEK_V4_CONTAINER to the DeepGEMM vLLM "
                    "server image before submitting."
                )
            model_cfg.update({
                "tokenizer": AWS_CMH_DEEPSEEK_V4_TOKENIZER,
                # GB300 config: TP=4 x DP=2 across 2 full GB300 nodes.
                "server_gpus": 4, "server_nodes": 2,
                "server_args": AWS_CMH_DEEPSEEK_V4_SERVER_ARGS,
                "server_container": AWS_CMH_DEEPSEEK_V4_CONTAINER,
            })
        elif args.model == "kimi_k26":
            model_cfg.update({
                "server_gpus": 4, "server_nodes": 1,
                "server_args": AWS_CMH_KIMI_K26_SERVER_ARGS,
                "server_container": AWS_CMH_KIMI_K26_CONTAINER,
            })
        if judge_cfg.get("judge_server_gpus") is not None:
            judge_cfg["judge_server_gpus"] = 4

    return model_cfg, judge_cfg


def _preflight(model_cfg, arm, verdict: AllocationVerdict) -> None:
    """Run every waste invariant before submission.  Raises SystemExit on failure."""
    server_args = model_cfg.get("server_args_tool") if arm["use_tool_inference"] and model_cfg.get("server_args_tool") else model_cfg["server_args"]
    check_tp_matches_alloc(server_args, verdict.server_gpus, verdict.server_nodes)

    # Pattern 5: enforce in-process Direct* retrieval tools.
    tool_module_strs = []
    extra = arm.get("extra_args", "")
    if "tool_modules=" in extra:
        # Strings live inside a Hydra list like ++tool_modules=["a","b"].
        # Pull each quoted module path.
        import re
        tool_module_strs = re.findall(r'"([^"]+)"', extra)
    require_direct_mcp_tools(tool_module_strs)


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="HLE MCP Ablation v12 — DeepSeek v4 + Kimi K2.6 with waste invariants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--arms", default=",".join(str(i) for i in range(1, len(ARMS) + 1)))
    ap.add_argument("--run-id", type=int, default=None)
    ap.add_argument("--cluster", default=CLUSTER)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--benchmark", default="hle")
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--num-chunks", type=int, default=None)
    ap.add_argument(
        "--num-jobs",
        type=int,
        default=None,
        help=(
            "Number of parallel SLURM jobs to spread the (seed x chunk) eval "
            "units across.  Defaults to --num-chunks (legacy coupling).  With "
            "the default single chunk, set --num-jobs=<#seeds> to get one model "
            "server per seed (e.g. --seeds 5 --num-jobs 5 -> 5 generation jobs)."
        ),
    )
    ap.add_argument(
        "--context-strategy",
        choices=["none", "drop", "blank", "summary"],
        default="none",
    )
    ap.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="kimi_k26")
    ap.add_argument("--judge", choices=sorted(JUDGE_CONFIGS), default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--server-extra", default="")
    ap.add_argument(
        "--tokens-to-generate",
        type=int,
        default=None,
        help="Override ++inference.tokens_to_generate for faster smoke/large-split runs.",
    )
    ap.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=None,
        help="Override ++max_concurrent_requests; useful for long-output models with large KV pressure.",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override ++max_samples for smoke runs.",
    )
    ap.add_argument(
        "--max-tool-calls",
        type=int,
        default=None,
        help=(
            "Override ++max_tool_calls for tool arms (default per-arm is 50). "
            "Used to study/curb tool-call spirals in the python-degradation work."
        ),
    )
    ap.add_argument(
        "--sandbox-container",
        default=None,
        help=(
            "Override the sandbox container image (e.g. one with extra libs baked in: "
            "sklearn/z3/periodictable/pywt/control). Default None uses the cluster config sandbox."
        ),
    )
    ap.add_argument(
        "--direct-python",
        action="store_true",
        default=False,
        help=(
            "Use the in-process DirectPythonTool (calls the sandbox directly, no MCP subprocess) "
            "instead of the MCP PythonTool. Matches Jiacheng's SCI-453/455 recipe. Swaps the "
            "python_tool::PythonTool module string in tool arms."
        ),
    )
    ap.add_argument(
        "--server-args-extra",
        default="",
        help="Append extra raw args to the spawned vLLM/SGLang server command.",
    )
    ap.add_argument(
        "--server-address",
        default=None,
        help=(
            "If set, do NOT spawn a new model server; point the eval client "
            "at this URL (e.g. 'http://10.67.26.85:35041/v1' from "
            "scratch/kimi_k26_minimal/kimi.endpoint).  Use this to bypass the "
            "NeMo-Skills wrapper for Kimi K2.6, which has a plugin-loading bug; "
            "boot the server via experiments_v12/launch_kimi_k26.sbatch and pass "
            "the URL here for each arm."
        ),
    )
    args = ap.parse_args()

    job_prefix = _require_mcp_cluster(args.cluster)
    model_cfg = dict(MODEL_CONFIGS[args.model])
    model_short = model_cfg["short"]
    judge_cfg = dict(JUDGE_CONFIGS[args.judge or model_cfg["judge"]])

    model_cfg, judge_cfg = _apply_cluster_overrides(args, model_cfg, judge_cfg)

    # Classify allocation up-front so the verdict gates exclusive +
    # get_random_port for every arm.  Per the wasted_compute root-cause
    # doc this is non-optional: a partial-node submission MUST use a
    # random port AND must NOT carry --exclusive.
    verdict = classify_allocation(
        cluster=args.cluster,
        server_gpus=model_cfg["server_gpus"],
        server_nodes=model_cfg["server_nodes"],
    )
    sbatch_kwargs = sbatch_kwargs_for_cluster(args.cluster)

    run_id = (
        args.run_id
        if args.run_id is not None
        else next_run_id(f"mcp-ablation-{model_short}")
    )
    arm_indices = [int(x) for x in args.arms.split(",")]

    print(f"\n{'=' * 70}")
    print(f"  HLE MCP Ablation v12 — {model_cfg['label']} — run {run_id}")
    print(f"  model:      {model_cfg['path']}")
    print(f"  server:     {model_cfg['server_type']} "
          f"({model_cfg['server_gpus']} GPUs x {model_cfg['server_nodes']} node(s))")
    print(f"  judge:      {judge_cfg['judge_model']}")
    print(f"  cluster:    {args.cluster} (job prefix: {job_prefix})")
    print_verdict(verdict, header="  alloc:")
    if sbatch_kwargs:
        print(f"  sbatch:     {sbatch_kwargs}")
    server_container = model_cfg.get("server_container") or model_cfg.get("main_container")
    if server_container:
        print(f"  server container: {server_container}")
    benchmarks = f"{args.benchmark}:{args.seeds}"
    print(f"  benchmark:  {benchmarks} (split={args.split})")
    print(f"  arms:       {arm_indices} ({len(arm_indices)} jobs)")
    print(f"{'=' * 70}")
    if args.server_extra.strip():
        print(f"  server-extra: {args.server_extra.strip()}")
        print(f"{'=' * 70}")
    if args.tokens_to_generate is not None:
        print(f"  tokens cap:  {args.tokens_to_generate}")
    if args.max_concurrent_requests is not None:
        print(f"  max conc:    {args.max_concurrent_requests}")
    if args.max_samples is not None:
        print(f"  max samples: {args.max_samples}")
    if args.server_args_extra.strip():
        print(f"  server args extra: {args.server_args_extra.strip()}")

    # Compose the (benchmark, split, ctx-strategy) tag once; reuse in
    # both the dry-run preview and the actual submission loop so the
    # two paths cannot drift.
    _bshort = _benchmark_tag(args.benchmark, args.split)
    _ctx_short = args.context_strategy if args.context_strategy != "none" else ""
    ctx_tag = "-".join(t for t in (_bshort, _ctx_short) if t)

    if args.dry_run:
        for i in arm_indices:
            arm = ARMS[i - 1]
            _preflight(model_cfg, arm, verdict)
            print(f"\n  Arm {i} ({arm['key']}): {arm['desc']}")
            print(f"    expname:    {_expname(model_short, arm['key'], run_id, ctx_tag)}")
            print(f"    slurm name: {job_prefix}{_expname(model_short, arm['key'], run_id, ctx_tag)}")
            print(f"    output_dir: {_output_dir(model_short, arm['key'], run_id, ctx_tag)}")
            print(f"    chunks:     {args.num_chunks if args.num_chunks is not None else arm.get('num_chunks', 1)}")
            print(f"    jobs:       {args.num_jobs if args.num_jobs is not None else (args.num_chunks if args.num_chunks is not None else arm.get('num_jobs', arm.get('num_chunks', 1)))}")
        print(f"\n[DRY RUN] Would submit {len(arm_indices)} eval job(s) — exiting.\n")
        return

    ctx_server_extra = ""
    if args.context_strategy != "none":
        strategy = CONTEXT_STRATEGY_MAP[args.context_strategy]
        ctx_server_extra = (
            "++server.enable_soft_fail=True "
            f"++server.context_limit_retry_strategy={strategy} "
            "++server.context_limit_retry_max_context_length=131072 "
            "++server.num_special_tokens_budget=512 "
        )

    results = []
    for i in arm_indices:
        arm = ARMS[i - 1]
        _preflight(model_cfg, arm, verdict)

        name = _expname(model_short, arm["key"], run_id, ctx_tag)
        odir = _output_dir(model_short, arm["key"], run_id, ctx_tag)
        print(f"\n{'─' * 60}\n  Arm {i}: {name}\n  {arm['desc']}\n{'─' * 60}")

        inference_args = (
            model_cfg["inference_args_tool"]
            if arm["use_tool_inference"]
            else model_cfg["inference_args_no_tool"]
        )
        if args.tokens_to_generate is not None:
            inference_args = _replace_or_append_hydra_arg(
                inference_args,
                "inference.tokens_to_generate",
                args.tokens_to_generate,
            )
        runtime_extra_args = ""
        if args.max_concurrent_requests is not None:
            runtime_extra_args += f"++max_concurrent_requests={args.max_concurrent_requests} "
        if args.max_samples is not None:
            runtime_extra_args += f"++max_samples={args.max_samples} "
        # Empty-string tokenizer means "no override" (vllm uses the
        # HF tokenizer that ships with the model).
        tokenizer_args = (
            f"++tokenizer={model_cfg['tokenizer']} "
            if model_cfg.get("tokenizer") else ""
        )
        # server_env: optional dict of vllm env vars (e.g. ENGINE_READY
        # timeout for slow loaders).  Wired through wrap_arguments via
        # the standard Hydra namespace; nemo-run forwards these to the
        # server-side launch wrapper.
        # (Kept off-band for now -- enable when nemo-run exposes a
        #  server_env path; the multithread loader already brings DSv4
        #  boot to ~1 min so the 3600s timeout is rarely load-bearing.)
        all_extra_args = (
            inference_args
            + tokenizer_args
            + model_cfg["thinking_args"]
            + (f"++prompt_config={arm['prompt_config']} " if arm["prompt_config"] else "")
            + arm["extra_args"]
            + runtime_extra_args
            + ctx_server_extra
            + (args.server_extra.strip() + " " if args.server_extra.strip() else "")
        )
        # Optional: swap the MCP PythonTool for the in-process DirectPythonTool (SCI-453 recipe).
        # python_tool::PythonTool is NOT a substring of python_tool::DirectPythonTool, so this is safe
        # and idempotent. Note: tool_overrides for the timeout must then target DirectPythonTool
        # (tool_overrides keyed by class name) — pass that in --server-extra at launch.
        if args.direct_python and arm["use_tool_inference"]:
            all_extra_args = all_extra_args.replace(
                "python_tool::PythonTool", "python_tool::DirectPythonTool"
            )

        # Optional max_tool_calls override (replaces the per-arm default set via _tool_modules).
        # Default None -> no change, so other users of this harness are unaffected. Use a targeted
        # regex (not the shlex helper) so the ++tool_modules=[...] list's quotes are left intact.
        if args.max_tool_calls is not None and arm["use_tool_inference"]:
            if re.search(r"\+\+max_tool_calls=\S+", all_extra_args):
                all_extra_args = re.sub(
                    r"\+\+max_tool_calls=\S+", f"++max_tool_calls={args.max_tool_calls}", all_extra_args
                )
            else:
                all_extra_args += f"++max_tool_calls={args.max_tool_calls} "

        # Common eval kwargs (independent of whether we spawn our own
        # server or point at an external one).
        eval_kwargs = dict(
            ctx=wrap_arguments(all_extra_args),
            cluster=args.cluster,
            expname=name,
            model=model_cfg["path"],
            server_type=model_cfg["server_type"],
            benchmarks=benchmarks,
            split=args.split,
            output_dir=odir,
            with_sandbox=arm["sandbox"],
            **({"sandbox_container": args.sandbox_container} if args.sandbox_container else {}),
            num_chunks=(args.num_chunks if args.num_chunks is not None else arm.get("num_chunks", 1)),
            num_jobs=(
                args.num_jobs
                if args.num_jobs is not None
                else args.num_chunks
                if args.num_chunks is not None
                else arm.get("num_jobs", arm.get("num_chunks", 1))
            ),
            judge_model=judge_cfg["judge_model"],
            exclusive=verdict.exclusive,
        )

        if args.server_address:
            # Point at an externally-managed server (e.g. Kimi K2.6 booted
            # via experiments_v12/launch_kimi_k26.sbatch).  NeMo-Skills'
            # wrapper has a plugin-loading bug for kimi_k26 that makes
            # spawning the server through it fail; bypassing the spawn
            # with --server-address sidesteps the bug entirely.
            eval_kwargs["server_address"] = args.server_address
        else:
            # Spawn our own server via NeMo-Skills.
            eval_kwargs["server_gpus"] = model_cfg["server_gpus"]
            eval_kwargs["server_nodes"] = model_cfg["server_nodes"]
            eval_kwargs["server_args"] = (
                model_cfg.get("server_args_tool")
                if arm["use_tool_inference"] and model_cfg.get("server_args_tool")
                else model_cfg["server_args"]
            )
            if args.server_args_extra.strip():
                eval_kwargs["server_args"] += " " + args.server_args_extra.strip() + " "
            if server_container:
                # This is a serving image only.  Keep the eval client and
                # judge in the regular nemo-skills container; several custom
                # vLLM images expose python3 but not the `python` executable
                # used by NeMo-Skills' generated client/summarize commands.
                eval_kwargs["server_container"] = server_container
        if sbatch_kwargs:
            eval_kwargs["sbatch_kwargs"] = sbatch_kwargs
        if judge_cfg.get("judge_server_address"):
            eval_kwargs["judge_server_address"] = judge_cfg["judge_server_address"]
        if judge_cfg.get("judge_server_type"):
            eval_kwargs["judge_server_type"] = judge_cfg["judge_server_type"]
        if judge_cfg.get("judge_server_gpus") is not None:
            eval_kwargs["judge_server_gpus"] = judge_cfg["judge_server_gpus"]
        if judge_cfg.get("judge_server_args"):
            eval_kwargs["judge_server_args"] = judge_cfg["judge_server_args"]
        if judge_cfg.get("extra_judge_args"):
            eval_kwargs["extra_judge_args"] = judge_cfg["extra_judge_args"]
        if arm["install"]:
            eval_kwargs["installation_command"] = arm["install"]

        eval(**eval_kwargs)
        results.append((name, odir))

    print(f"\n{'=' * 70}")
    print(f"  Submitted {len(results)} eval job(s) on {args.cluster}")
    print(f"  Model: {model_cfg['label']} | Benchmark: HLE ({args.split}, hle:{args.seeds})")
    print(f"{'=' * 70}\n")
    for name, odir in results:
        print(f"  {name}")
        print(f"    ns summarize_results --cluster {args.cluster} {odir}")
    print()


if __name__ == "__main__":
    main()
