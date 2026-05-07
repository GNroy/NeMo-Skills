# Experiment r32 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r32 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: **gpt-oss-120b orchestrator** + nano worker (own server) + gpt-oss-120b worker (co-located with orchestrator) |
| Model | `/hf_models/gpt-oss-120b` orchestrator + gpt worker (co-located); `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` nano worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, fp8, FlashInfer backend) |
| Key config changes | **Fix r31**: add `--tool-call-parser openai --reasoning-parser openai_gptoss` to GPT orchestrator server args; `--max-tool-calls 3`; also add `--reasoning-parser openai_gptoss` to GPT worker server args |
| Commit | `504c1cf5 fix: add --reasoning-parser openai_gptoss to GPT worker server args` |
| SLURM generation jobs | 156511 (rs0), 156513 (rs1), 156517 (rs2), 156515 (rs3), 156519 (rs4) |
| SLURM eval jobs | 156521–156525 (judge rs0–rs4), 156526 (summarize) |

## Motivation

r31 failed (3.2% pass@1, 0 tool calls) because `GPT_ORCHESTRATOR_MODEL["server_args"]` was copied from `JUDGE_SERVER_ARGS` which omits tool-calling flags. The judge never calls tools; the orchestrator does.

Root cause: gpt-oss-120b uses a proprietary "harmony" format for tool calls, requiring:
- `--enable-auto-tool-choice`
- `--tool-call-parser openai`
- `--reasoning-parser openai_gptoss`

These are already present in the nemo-skills-vllm container (confirmed from `run_hle_mcp_ablation.py`). r31 simply did not pass them.

Additional fix: `--max-tool-calls 3` enforces the intended 3-call workflow (call_nano_async + call_gpt_async + collect_results), making retry loops structurally impossible even if GPT orchestrator exhibits sequential fallback.

GPT worker server args also updated to include `--reasoning-parser openai_gptoss` so gpt-oss reasoning tokens are separated from the final code output (cleaner DirectPythonTool extraction).

Key changes vs r31:
- `--tool-call-parser openai --reasoning-parser openai_gptoss` in orchestrator server args (the actual fix)
- `--reasoning-parser openai_gptoss` in GPT worker server args
- `--max-tool-calls 3` (safety cap)
- `tokens_to_generate`: 32768 → 65536 (headroom for reasoning tokens)

## Results

*(Pending — jobs running)*

| Run | Mode | pass@1 | majority | pass@N | avg_tokens | gen_seconds |
|-----|------|--------|----------|--------|------------|-------------|
| rec1 | non-agentic, fp8, t=1.0 | 48.80% ± 3.03% [5] | 49.90% | 67.00% [5] | 40,952 | 728 |
| r24 | nano orch (no thinking), nano+gpt workers, tc≤15 | 31.25% ± 2.22% [4] | 33.25% | 57.00% [4] | 3,329 | 3,166 |
| r29 | nano orch (thinking=True), nano+gpt workers, timeout=2400s | 31.0% ± 1.7% [3] | 26.7% | 50.0% [3] | 70,156 | 5,859 |
| r30 | same + session_timeout=7200s (2/5 seeds; rs4: 33% session timeouts) | 30.0% ± 4.2% [2] | 30.0% | 45.0% [2] | 44,672 | 7,310 |
| r31 | gpt orch (no tool-call-parser — BROKEN) | 3.2% ± 1.3% [5] | 2.4% | 7.0% [5] | 499 | 61 |
| r32 | **gpt orch** (tool-call-parser fixed), nano+gpt workers, tc=3, session=7200s | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

## Analysis

*(Pending)*

Key metrics to watch:
- Tool call count: should be exactly 3 per problem (call_nano + call_gpt + collect)
- Session timeout rate: target near-zero (tc=3 cap prevents retry loops)
- pass@1: target > 31.25% (r24 baseline)
- avg_tokens: moderate (GPT reasoning at default "medium" effort + worker outputs injected)
- gen_seconds: target < 4h walltime (with tc=3 cap, worst case = 2× worker_timeout = 4800s per problem, but pipelined)

## Next steps

*(To be filled after results)*
