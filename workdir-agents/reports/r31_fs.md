# Experiment r31 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r31 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: **gpt-oss-120b orchestrator** + nano worker (own server) + gpt-oss-120b worker (co-located with orchestrator) |
| Model | `/hf_models/gpt-oss-120b` orchestrator + gpt worker (co-located); `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` nano worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, fp8, FlashInfer backend) |
| Key config changes | **--gpt-orchestrator**: GPT-120B as orchestrator instead of Nano; same workers (nano + gpt), same session_timeout=7200s, worker_timeout=2400s, concurrency 128/agent |
| Commit | `09ae5ccf feat: add --gpt-orchestrator flag to use gpt-oss-120b as orchestrator` |
| SLURM generation jobs | 156220 (rs0), 156222 (rs1), 156224 (rs2), 156226 (rs3), 156228 (rs4) |
| SLURM eval jobs | 156230–156234 (judge rs0–rs4), 156235 (summarize) |

## Motivation

r29/r30 with Nano orchestrator (thinking=True) showed a sequential fallback pattern:
the orchestrator calls nano alone → waits 2400s for timeout → then calls GPT, instead
of firing both workers simultaneously. This caused ~33% session timeouts (rs4) even
with the 7200s cap.

The hypothesis: GPT-120B is better at following structured multi-step instructions
("fire all workers simultaneously, then collect") without deviating into sequential
retry loops. GPT has no thinking mode, so it cannot exhaust its token budget in a
`<think>` block, and its tool-calling reliability is generally higher.

Key changes vs r30:
- `--gpt-orchestrator` → GPT-120B as orchestrator (32768 token budget, no thinking)
- Nano worker still on its own server (thinking=True, 131072 token budget)
- GPT worker co-located with GPT orchestrator (same server, auto-split 128/agent)
- `HF_HUB_OFFLINE=1` now active → no HF 429 startup failures

## Results

| Seed | pass@1 | non-empty | tool_calls |
|------|--------|-----------|------------|
| rs0 | 3.0% | 11/100 | 0 |
| rs1 | 4.0% | 7/100 | 0 |
| rs2 | 4.0% | 9/100 | 0 |
| rs3 | 4.0% | 13/100 | 0 |
| rs4 | 1.0% | 5/100 | 0 |

| Run | Mode | pass@1 | majority | pass@N | avg_tokens | gen_seconds |
|-----|------|--------|----------|--------|------------|-------------|
| rec1 | non-agentic, fp8, t=1.0 | 48.80% ± 3.03% [5] | 49.90% | 67.00% [5] | 40,952 | 728 |
| r24 | nano orch (no thinking), nano+gpt workers, tc≤15 | 31.25% ± 2.22% [4] | 33.25% | 57.00% [4] | 3,329 | 3,166 |
| r29 | nano orch (thinking=True), nano+gpt workers, timeout=2400s | 31.0% ± 1.7% [3] | 26.7% | 50.0% [3] | 70,156 | 5,859 |
| r30 | same + session_timeout=7200s (2/5 seeds; rs4: 33% session timeouts) | 30.0% ± 4.2% [2] | 30.0% | 45.0% [2] | 44,672 | 7,310 |
| r31 | **gpt orch** (no thinking), nano+gpt workers, timeout=2400s, session=7200s | **3.2% ± 1.3% [5]** | **2.4%** | **7.0% [5]** | **499** | **61** |

pass@1 ± std_dev computed from per-seed [3%, 4%, 4%, 4%, 1%]. gen_seconds = 61s — generation completed nearly instantly.

## Analysis

**Root cause: missing vLLM tool-call-parser for gpt-oss-120b.**

The `GPT_ORCHESTRATOR_MODEL["server_args"]` used `JUDGE_SERVER_ARGS` = `"--async-scheduling --max-model-len 131072"`, which does NOT include `--enable-auto-tool-choice` or `--tool-call-parser`. Without these flags, vLLM does not parse tool-call responses from the model and the orchestrator receives no tool calls.

gpt-oss-120b uses a proprietary "harmony" format for tool calls:
```
<|start|>assistant to=functions.{name}<|channel|>commentary json<|message|>{args_json}<|call|>
```
The standard vLLM nemo-skills container does not include a parser for this format. The model's own README specifies a specially-patched vLLM build: `vllm==0.10.1+gptoss` from `https://wheels.vllm.ai/gpt-oss/`.

**Observed behavior:**
- 0 tool calls across all 500 problem-seed combinations
- 91% empty outputs: orchestrator session exited immediately after getting no tool call from the model
- 9% text-only outputs: for a few problems, the model's first response was a direct answer (not a tool call), which was captured before the session ended
- 3.2% pass@1: the ~35% of text-only responses that happened to be judged correct (GPT-OSS is a strong model even without tools)
- 499 avg_tokens, 61s gen_seconds: nearly instant — no worker calls, no thinking, no timeouts

**Why no session timeouts:** with 0 tool calls, every session exited in seconds, so the 7200s cap was never reached. This confirmed the orchestrator itself was the problem.

## Next steps

The GPT-orchestrator approach requires the gpt-oss-patched vLLM (`vllm==0.10.1+gptoss`) or a custom tool-call-parser plugin matching the harmony `<|call|>` format. This is a non-trivial container change.

**r32 plan: Nano orchestrator + max_tool_calls=3**

Return to Nano as orchestrator but fix the sequential fallback at the source: hard-cap `max_tool_calls=3`. With 2 workers (nano + gpt), the correct workflow is exactly 3 tool calls:
1. `call_nano_async`
2. `call_gpt_async`
3. `collect_results`

Setting `max_tool_calls=3` makes retry loops structurally impossible — the orchestrator is forced to fire both workers then collect, regardless of what its `<think>` block says. This directly addresses the root cause identified in r30 (thinking=True allows 3× sequential retries hitting the 7200s cap).
