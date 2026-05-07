"""
HLE MCP Ablation — Per-Tool Contribution on Physics Subset
=============================================================================
Model-selectable HLE physics ablation (202 questions, 5 seeds).
24 arms: 9 legacy controls plus best-practice prompt/skill/retrieval variants.

Arms:
  1: no-tool       — baseline reasoning only
  2: python        — PythonTool only
  3: python+particle       — + ParticleTool (PDG data)
  4: python+radioactivedecay — + RadioactivedecayTool (decay chains)
  5: python+coolprop       — + CoolPropTool (thermophysical properties)
  6: python+periodictable  — + PeriodictableTool (element/isotope data)
  7: python+libretexts     — + LibreTextsTool (textbook search)
  8: python+arxiv          — + ArxivSearchTool (paper search)
  9: python+wikipedia      — + WikipediaSearchTool (encyclopedia)
 10: python+mcp-local-policy       — deterministic MCPs + concise policy prompt
 11: python+mcp-local-skill        — deterministic MCPs + lazy guide activation
 12: python+mcp-local-inline-docs  — deterministic MCPs + inline rich docs
 13: knowledge-path                — retrieval MCPs + lazy guide + strict budgets
 14: overloaded-tools              — all MCPs exposed without prompt/skill filtering
 15: python+mcp-all-skill          — all MCPs + lazy guide activation
16: python+arxiv-metadata-skill-max1 — arXiv-only metadata+skill, one tool call total
17: python+arxiv-metadata-skill-max5 — arXiv-only metadata+skill, five tool calls total
18: python+wikipedia-metadata-skill-max100 — Wikipedia-only metadata+skill, 100 tool calls
19: python+arxiv-metadata-skill-max100 — arXiv-only metadata+skill, 100 tool calls
20: python+libretexts-metadata-skill-max100 — LibreTexts-only metadata+skill, 100 tool calls
21: python+particle-skill-max100 — Particle-only skill-guided MCP, 100 tool calls
22: python+radioactivedecay-skill-max100 — Radioactivedecay-only skill-guided MCP, 100 tool calls
23: python+coolprop-skill-max100 — CoolProp-only skill-guided MCP, 100 tool calls
24: python+periodictable-skill-max100 — Periodictable-only skill-guided MCP, 100 tool calls

Usage:
  conda activate hle-dev
  cd ~/workspace/hle

  # All 9 arms:
  CLAUSIUS_URL=http://localhost:7272 \\
  NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 \\
    python experiments/run_hle_mcp_ablation.py

  # MCP arms only (skip baselines):
  CLAUSIUS_URL=http://localhost:7272 \\
  NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 \\
    python experiments/run_hle_mcp_ablation.py --arms 3,4,5,6,7,8,9

  # Single arm:
  ... --arms 3

  # Dry run:
  ... --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_id import next_run_id

for parent in Path(__file__).resolve().parents:
    if (parent / "NeMo-Skills" / "nemo_skills").is_dir():
        sys.path.insert(0, str(parent / "NeMo-Skills"))
        break

from nemo_skills.pipeline import utils as pipeline_utils
from nemo_skills.pipeline.cli import eval, wrap_arguments

# ── Cluster ──────────────────────────────────────────────────────────────

CLUSTER = "dfw-science"
BENCHMARKS = "hle:5"
SPLIT = "phy"
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

# aws-cmh nodes are GB300, 4 GPUs/node. Models that fit in one node at TP=4
# stay single-node; models needing more memory use 2 nodes x 4 GPUs (TP=8) or
# 4 nodes x 4 GPUs (TP=16) via ray.
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
# Nemotron-3-Nano-30B-A3B-BF16 — hybrid Mamba/transformer MoE, 3B active params.
# 30B BF16 fits in 1 GPU on aws-cmh GB300 (288 GB HBM/GPU); we run 4 DP replicas
# on a single 4-GPU node for throughput. Drops the server-side reasoning parser
# (matches v7 / r10 Nemotron-Super winning config; client-side ``parse_reasoning``
# handles ``<think>`` extraction).
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
# DeepSeek-V3.2 needs sglang with EP/DP across 16 GPUs (4 nodes x 4 GPUs).
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

# ── Judge ────────────────────────────────────────────────────────────────

JUDGE_CONFIGS = {
    # Default: fast on-cluster judge used throughout the MCP/Nemotron runs.
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
        # The generation server can load DeepSeek-V3.2, but the client prompt
        # layer in the eval container has an older Transformers build that does
        # not recognize model_type=deepseek_v32. Use a compatible chat tokenizer
        # only for prompt/tool accounting; chat requests still go to DeepSeek.
        "tokenizer": "/hf_models/Qwen2.5-32B-Instruct",
        "server_type": "sglang",
        "server_gpus": 8,
        "server_nodes": 2,
        "server_args": (
            "--enable-dp-attention "
            "--ep-size 8 "
            "--dp 8 "
            "--tool-call-parser deepseekv32 "
            "--reasoning-parser deepseek-v3 "
            "--log-requests "
            "--mem-fraction-static=0.8 "
            """--model-loader-extra-config '{"enable_multithread_load":true,"num_threads":112}' """
        ),
        "thinking_args": "++chat_template_kwargs.thinking=true ",
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
        # vLLM GPT-OSS custom-tool parsing uses the OpenAI/Harmony parser.
        "server_args_tool": "--async-scheduling --max-model-len 131072 --enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss ",
        "thinking_args": "++chat_template_kwargs.reasoning_effort=high ",
        "inference_args_no_tool": "++inference.temperature=1.0 ++inference.top_p=1.0 ++inference.tokens_to_generate=100000 ",
        "inference_args_tool": "++inference.temperature=1.0 ++inference.top_p=1.0 ++inference.tokens_to_generate=100000 ",
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
        ),
        "thinking_args": "",
        "inference_args_no_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=100000 ",
        "inference_args_tool": "++inference.temperature=1.0 ++inference.top_p=0.95 ++inference.tokens_to_generate=100000 ",
        "judge": "gpt-oss",
    },
    "qwen35": {
        # Qwen3.5-122B-A10B — fits on 1 node x 8 GPUs.
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
        # Qwen3.5-397B-A17B — needs 2 nodes x 8 GPUs (TP=16) via Ray.
        # Server args mirror experiments_v7/run_qwen35_397b_eos.py which is the
        # validated path used for the HLE text-split baselines.
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

# ── Tool module paths ────────────────────────────────────────────────────

_PYTHON = "nemo_skills.mcp.servers.python_tool::PythonTool"
_PARTICLE = "nemo_skills.mcp.servers.physics.particle_tool::ParticleTool"
_RADIOACTIVEDECAY = (
    "nemo_skills.mcp.servers.physics.radioactivedecay_tool::RadioactivedecayTool"
)
_COOLPROP = "nemo_skills.mcp.servers.physics.coolprop_tool::CoolPropTool"
_PERIODICTABLE = "nemo_skills.mcp.servers.chemistry.periodictable_tool::PeriodictableTool"
_LIBRETEXTS = "nemo_skills.mcp.servers.web.libretexts_tool::LibreTextsTool"
_ARXIV = "nemo_skills.mcp.servers.web.arxiv_tool::ArxivSearchTool"
_WIKIPEDIA = "nemo_skills.mcp.servers.web.wikipedia_tool::WikipediaSearchTool"
_MCP_GUIDE = "nemo_skills.mcp.servers.mcp_skill_tool::MCPGuideTool"

_LOCAL_MCP_TOOLS = (_PARTICLE, _RADIOACTIVEDECAY, _COOLPROP, _PERIODICTABLE)
_RETRIEVAL_MCP_TOOLS = (_LIBRETEXTS, _ARXIV, _WIKIPEDIA)

_LOCAL_MCP_INSTALL = "pip install particle radioactivedecay CoolProp periodictable"
_RETRIEVAL_INSTALL = (
    "unset OPENALEX_API_KEY && "
    "export ARXIV_REQUEST_INTERVAL=4.0 && "
    "export ARXIV_RATE_LIMIT_LOCK=/workspace/mcp-experiments/arxiv_api_rate_limit.lock && "
    "pip install httpx bm25s sentence-transformers faiss-cpu"
)
_ALL_MCP_INSTALL = f"{_LOCAL_MCP_INSTALL} && {_RETRIEVAL_INSTALL}"
_KNOWLEDGE_PATH_FILTERS = (
    "++tool_overrides.WikipediaSearchTool.enabled_tools="
    "[wikipedia-search,wikipedia-summary,wikipedia-sections,wikipedia-query-summary,"
    "wikipedia-key-facts,wikipedia-section] "
    "++tool_overrides.LibreTextsTool.enabled_tools=[libretexts-search] "
)
_WIKIPEDIA_METADATA_FILTERS = (
    "++tool_overrides.WikipediaSearchTool.enabled_tools="
    "[wikipedia-search,wikipedia-summary,wikipedia-sections,wikipedia-query-summary,"
    "wikipedia-key-facts,wikipedia-section] "
)
_LIBRETEXTS_METADATA_FILTERS = "++tool_overrides.LibreTextsTool.enabled_tools=[libretexts-search] "

# ── Arms ─────────────────────────────────────────────────────────────────


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
        "desc": "+ PythonTool + RadioactivedecayTool (nuclear decay chains)",
        "extra_args": _tool_modules(_PYTHON, _RADIOACTIVEDECAY),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": "pip install radioactivedecay",
    },
    {
        "key": "python+coolprop",
        "desc": "+ PythonTool + CoolPropTool (thermophysical properties)",
        "extra_args": _tool_modules(_PYTHON, _COOLPROP),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": "pip install CoolProp",
    },
    {
        "key": "python+periodictable",
        "desc": "+ PythonTool + PeriodictableTool (element/isotope data)",
        "extra_args": _tool_modules(_PYTHON, _PERIODICTABLE),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": "pip install periodictable",
    },
    {
        "key": "python+libretexts",
        "desc": "+ PythonTool + LibreTextsTool (textbook search, dense bge-m3 + bm25s + RRF hybrid)",
        "extra_args": _tool_modules(_PYTHON, _LIBRETEXTS, max_tool_calls=25),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": "pip install bm25s sentence-transformers faiss-cpu",
        # Retrieval arms are generation-throughput bound; split the 5-seed
        # physics set across chunks so a single slow job does not hit 4h.
        "num_chunks": 5,
    },
    {
        "key": "python+arxiv",
        "desc": "+ PythonTool + ArxivSearchTool (arXiv paper search, OpenAlex backend)",
        "extra_args": _tool_modules(_PYTHON, _ARXIV, max_tool_calls=12),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        # arxiv-search now uses native export.arxiv.org with a shared lock file
        # rate limiter; unset OpenAlex to ensure no metadata-search fallback uses
        # the exhausted DFW key.
        "install": "unset OPENALEX_API_KEY; export ARXIV_REQUEST_INTERVAL=4.0; export ARXIV_RATE_LIMIT_LOCK=/workspace/mcp-experiments/arxiv_api_rate_limit.lock; pip install httpx",
        "num_chunks": 5,
    },
    {
        "key": "python+wikipedia",
        "desc": "+ PythonTool + WikipediaSearchTool (encyclopedia, httpx + MediaWiki API)",
        "extra_args": _tool_modules(_PYTHON, _WIKIPEDIA, max_tool_calls=25),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": "pip install httpx",
        "num_chunks": 5,
    },
    {
        "key": "python+mcp-local-policy",
        "desc": "+ PythonTool + deterministic MCPs + concise tool-use policy prompt",
        "extra_args": _tool_modules(_PYTHON, *_LOCAL_MCP_TOOLS, max_tool_calls=30),
        "prompt_config": "generic/hle-mcp-policy",
        "sandbox": True,
        "use_tool_inference": True,
        "install": _LOCAL_MCP_INSTALL,
    },
    {
        "key": "python+mcp-local-skill",
        "desc": "+ PythonTool + deterministic MCPs + lazy MCP guide activation",
        "extra_args": _tool_modules(_PYTHON, _MCP_GUIDE, *_LOCAL_MCP_TOOLS, max_tool_calls=30),
        "prompt_config": "generic/hle-mcp-skill",
        "sandbox": True,
        "use_tool_inference": True,
        "install": _LOCAL_MCP_INSTALL,
    },
    {
        "key": "python+mcp-local-inline-docs",
        "desc": "+ PythonTool + deterministic MCPs + same guide inlined in prompt",
        "extra_args": _tool_modules(_PYTHON, *_LOCAL_MCP_TOOLS, max_tool_calls=30),
        "prompt_config": "generic/hle-mcp-inline-docs",
        "sandbox": True,
        "use_tool_inference": True,
        "install": _LOCAL_MCP_INSTALL,
    },
    {
        "key": "knowledge-path",
        "desc": "+ PythonTool + retrieval MCPs + lazy guide + strict retrieval budgets",
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
        "desc": "+ PythonTool + all seven MCPs exposed without prompt/skill filtering",
        "extra_args": _tool_modules(_PYTHON, *_LOCAL_MCP_TOOLS, *_RETRIEVAL_MCP_TOOLS, max_tool_calls=50),
        "prompt_config": None,
        "sandbox": True,
        "use_tool_inference": True,
        "install": _ALL_MCP_INSTALL,
        "num_chunks": 5,
    },
    {
        "key": "python+mcp-all-skill",
        "desc": "+ PythonTool + all seven MCPs + lazy MCP guide activation",
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
        "desc": "+ PythonTool + MCPGuideTool + ArxivSearchTool, metadata-skill prompt, max_tool_calls=1",
        "extra_args": _tool_modules(_PYTHON, _MCP_GUIDE, _ARXIV, max_tool_calls=1),
        "prompt_config": "generic/hle-mcp-arxiv-metadata-skill",
        "sandbox": True,
        "use_tool_inference": True,
        "install": "unset OPENALEX_API_KEY; export ARXIV_REQUEST_INTERVAL=4.0; export ARXIV_RATE_LIMIT_LOCK=/workspace/mcp-experiments/arxiv_api_rate_limit.lock; pip install httpx",
        "num_chunks": 5,
    },
    {
        "key": "python+arxiv-metadata-skill-max5",
        "desc": "+ PythonTool + MCPGuideTool + ArxivSearchTool, metadata-skill prompt, max_tool_calls=5",
        "extra_args": _tool_modules(_PYTHON, _MCP_GUIDE, _ARXIV, max_tool_calls=5),
        "prompt_config": "generic/hle-mcp-arxiv-metadata-skill",
        "sandbox": True,
        "use_tool_inference": True,
        "install": "unset OPENALEX_API_KEY; export ARXIV_REQUEST_INTERVAL=4.0; export ARXIV_RATE_LIMIT_LOCK=/workspace/mcp-experiments/arxiv_api_rate_limit.lock; pip install httpx",
        "num_chunks": 5,
    },
    {
        "key": "python+wikipedia-metadata-skill-max100",
        "desc": "+ PythonTool + MCPGuideTool + WikipediaSearchTool, metadata-skill prompt, max_tool_calls=100",
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
        "desc": "+ PythonTool + MCPGuideTool + ArxivSearchTool, metadata-skill prompt, max_tool_calls=100",
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
        "desc": "+ PythonTool + MCPGuideTool + LibreTextsTool, metadata-skill prompt, max_tool_calls=100",
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
        "desc": "+ PythonTool + MCPGuideTool + ParticleTool, skill prompt, max_tool_calls=100",
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
        "desc": "+ PythonTool + MCPGuideTool + RadioactivedecayTool, skill prompt, max_tool_calls=100",
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
        "desc": "+ PythonTool + MCPGuideTool + CoolPropTool, skill prompt, max_tool_calls=100",
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
        "desc": "+ PythonTool + MCPGuideTool + PeriodictableTool, skill prompt, max_tool_calls=100",
        "extra_args": _tool_modules(_PYTHON, _MCP_GUIDE, _PERIODICTABLE, max_tool_calls=100)
        + _guide_skills("science-tools", "python-computation", "python", "periodictable"),
        "prompt_config": "generic/hle-mcp-skill",
        "sandbox": True,
        "use_tool_inference": True,
        "install": "pip install periodictable",
        "num_chunks": 10,
    },
]

# ── Naming ───────────────────────────────────────────────────────────────


def _expname(model_short, arm_key, run_id):
    return f"ablation_{model_short}-{arm_key}-r{run_id}"


def _output_dir(model_short, arm_key, run_id):
    return f"/workspace/mcp-experiments/ablation-{model_short}-{arm_key}-r{run_id}"


def _require_mcp_cluster(cluster):
    cluster_cfg = pipeline_utils.get_cluster_config(cluster)
    prefix = cluster_cfg.get("job_name_prefix", "")
    if prefix != "mcp_":
        raise SystemExit(
            f"Refusing to submit MCP jobs with cluster config {cluster!r}: "
            f"job_name_prefix is {prefix!r}, expected 'mcp_'."
        )
    return prefix


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="HLE MCP Ablation — Per-Tool Contribution (Nemotron-120B, physics subset)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--arms",
        default=",".join(str(i) for i in range(1, len(ARMS) + 1)),
        help="Comma-separated arm indices (1=no-tool, 2=python, 3-9=legacy MCP arms, 10-15=best-practice arms)",
    )
    ap.add_argument("--run-id", type=int, default=None, help="Override run ID")
    ap.add_argument("--cluster", default=CLUSTER, help="Cluster config name")
    ap.add_argument("--seeds", type=int, default=5, help="Number of HLE seeds to run")
    ap.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS),
        default="nemotron",
        help="Generator model to evaluate",
    )
    ap.add_argument(
        "--judge",
        choices=sorted(JUDGE_CONFIGS),
        default=None,
        help="Override judge config (default comes from model config)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Print config without submitting"
    )
    args = ap.parse_args()
    job_prefix = _require_mcp_cluster(args.cluster)

    model_cfg = dict(MODEL_CONFIGS[args.model])
    model_short = model_cfg["short"]
    judge_cfg = dict(JUDGE_CONFIGS[args.judge or model_cfg["judge"]])
    if args.cluster.startswith("aws-cmh"):
        # aws-cmh nodes are GB300 with 4 GPUs/node. Each model needs custom
        # tensor-parallel and node count to fit. Per-model overrides:
        if args.model == "nemotron":
            model_cfg.update(
                {
                    "path": AWS_CMH_NEMOTRON_PATH,
                    "server_gpus": 4,
                    "server_nodes": 1,
                    "server_args": AWS_CMH_NEMOTRON_SERVER_ARGS,
                }
            )
        elif args.model == "nemotron_nano":
            model_cfg.update(
                {
                    "path": AWS_CMH_NEMOTRON_NANO_PATH,
                    "server_gpus": 4,
                    "server_nodes": 1,
                    "server_args": AWS_CMH_NEMOTRON_NANO_SERVER_ARGS,
                }
            )
        elif args.model == "gptoss":
            model_cfg.update(
                {
                    "server_gpus": 4,
                    "server_nodes": 1,
                    "server_args": AWS_CMH_GPTOSS_SERVER_ARGS,
                    "server_args_tool": AWS_CMH_GPTOSS_SERVER_ARGS_TOOL,
                }
            )
        elif args.model == "qwen35":
            model_cfg.update(
                {
                    "server_gpus": 4,
                    "server_nodes": 1,
                    "server_args": AWS_CMH_QWEN35_122B_SERVER_ARGS,
                }
            )
        elif args.model == "qwen35_397b":
            # 4 nodes x 4 GPUs = 16 GPUs total via ray.
            model_cfg.update(
                {
                    "server_gpus": 4,
                    "server_nodes": 4,
                    "server_args": AWS_CMH_QWEN35_397B_SERVER_ARGS,
                }
            )
        elif args.model == "deepseek":
            # 4 nodes x 4 GPUs = 16 GPUs total via sglang DP+EP.
            model_cfg.update(
                {
                    "tokenizer": AWS_CMH_DEEPSEEK_TOKENIZER,
                    "server_gpus": 4,
                    "server_nodes": 4,
                    "server_args": AWS_CMH_DEEPSEEK_SERVER_ARGS,
                }
            )
        if judge_cfg.get("judge_server_gpus") is not None:
            judge_cfg["judge_server_gpus"] = 4

    run_id = (
        args.run_id
        if args.run_id is not None
        else next_run_id(f"mcp-ablation-{model_short}")
    )
    arm_indices = [int(x) for x in args.arms.split(",")]

    print(f"\n{'=' * 70}")
    print(f"  HLE MCP Ablation — {model_cfg['label']} — run {run_id}")
    print(f"  model:      {model_cfg['path']}")
    print(f"  server:     {model_cfg['server_type']} ({model_cfg['server_gpus']} GPUs x {model_cfg['server_nodes']} node(s))")
    print(f"  judge:      {judge_cfg['judge_model']}")
    print(f"  cluster:    {args.cluster} (job prefix: {job_prefix})")
    benchmarks = f"hle:{args.seeds}"
    print(f"  benchmark:  {benchmarks} (split={SPLIT}, 202 physics questions)")
    print(f"  arms:       {arm_indices} ({len(arm_indices)} jobs)")
    print(f"{'=' * 70}")

    if args.dry_run:
        for i in arm_indices:
            arm = ARMS[i - 1]
            print(f"\n  Arm {i} ({arm['key']}): {arm['desc']}")
            print(f"    expname:    {_expname(model_short, arm['key'], run_id)}")
            print(f"    slurm name: {job_prefix}{_expname(model_short, arm['key'], run_id)}")
            print(f"    output_dir: {_output_dir(model_short, arm['key'], run_id)}")
            print(f"    sandbox:    {arm['sandbox']}")
            print(f"    prompt:     {arm['prompt_config'] or '(default)'}")
            print(f"    install:    {arm['install'] or '(none)'}")
            print(f"    chunks:     {arm.get('num_chunks', 1)}")
            if arm["extra_args"]:
                print(f"    tool_args:  {arm['extra_args'].strip()[:120]}...")
        print(f"\n[DRY RUN] Would submit {len(arm_indices)} eval job(s) — exiting.\n")
        return

    results = []
    for i in arm_indices:
        arm = ARMS[i - 1]
        name = _expname(model_short, arm["key"], run_id)
        odir = _output_dir(model_short, arm["key"], run_id)

        print(f"\n{'─' * 60}")
        print(f"  Arm {i}: {name}")
        print(f"  {arm['desc']}")
        print(f"{'─' * 60}")

        inference_args = (
            model_cfg["inference_args_tool"]
            if arm["use_tool_inference"]
            else model_cfg["inference_args_no_tool"]
        )
        tokenizer_args = (
            f"++tokenizer={model_cfg['tokenizer']} " if model_cfg.get("tokenizer") else ""
        )
        all_extra_args = (
            inference_args
            + tokenizer_args
            + model_cfg["thinking_args"]
            + (f"++prompt_config={arm['prompt_config']} " if arm["prompt_config"] else "")
            + arm["extra_args"]
        )

        eval_kwargs = dict(
            ctx=wrap_arguments(all_extra_args),
            cluster=args.cluster,
            expname=name,
            model=model_cfg["path"],
            server_type=model_cfg["server_type"],
            server_gpus=model_cfg["server_gpus"],
            server_nodes=model_cfg["server_nodes"],
            server_args=model_cfg.get("server_args_tool", model_cfg["server_args"])
            if arm["use_tool_inference"]
            else model_cfg["server_args"],
            benchmarks=benchmarks,
            split=SPLIT,
            output_dir=odir,
            with_sandbox=arm["sandbox"],
            num_chunks=arm.get("num_chunks", 1),
            num_jobs=arm.get("num_jobs", arm.get("num_chunks", 1)),
            judge_model=judge_cfg["judge_model"],
        )
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
    print(f"  Model: {model_cfg['label']} | Benchmark: HLE physics ({SPLIT}, hle:{args.seeds})")
    print(f"{'=' * 70}\n")
    for name, odir in results:
        print(f"  {name}")
        print(f"    ns summarize_results --cluster {args.cluster} {odir}")
    print()


if __name__ == "__main__":
    main()
