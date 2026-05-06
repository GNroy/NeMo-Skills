# Experiment r25 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r25 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + nano worker (co-located) + gpt-oss-120b worker (own server + own sandbox) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` orchestrator + nano worker; `/hf_models/gpt-oss-120b` gpt worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, fp8, FlashInfer backend) |
| Key config changes | Same as r24 + **orchestrator thinking enabled** (`enable_thinking=True`) + **max_tool_calls=4** |
| Commit | `f89d1ed6 feat: add --orchestrator-thinking flag to run_nano_physics_agent.py` |
| SLURM generation jobs | 140979, 140981, 140983, 140985, 140987 (rs0–rs4, het-jobs +0/+1) |
| SLURM eval jobs | 140989–140993 (judge rs0–rs4), 140994 (summarize) |

## Motivation

r24 established that the collect_results protocol works (TC mode=3, max_tc rate=11.6%,
pass@1=31.25% avg-of-4). The remaining 17.5 pp gap to rec1 (48.8%) likely reflects
orchestrator synthesis quality — Nano-30B with thinking disabled comparing two long
solutions is a hard task.

r25 tests two changes simultaneously:

1. **Orchestrator thinking enabled** (`enable_thinking=True`) — allows the orchestrator
   to reason in a `<think>` block before deciding how to synthesize worker answers.
   Previously disabled to prevent the model from exhausting its 131k token budget
   reasoning before emitting a tool call. With max_tool_calls=4 enforcing the compact
   protocol, the orchestrator should spend thinking on synthesis rather than delegation.

2. **max_tool_calls=4** — enforces the compact protocol: nano_async (1) + gpt_async (2)
   + collect_results (3) = 3 calls, with 1 spare for retry. In r24, 91.6% of sessions
   used ≤4 tool calls, so cutting off at 4 should have minimal impact while ensuring
   the model cannot loop into sub-task decomposition.

Expected outcome: thinking enables better synthesis of worker results → pass@1 > 31.25%.
Risk: if nano times out and collect_results blocks, the model may spend its thinking
budget waiting and still fail to synthesize (empty generation from max_tc at 4).

## Results

*Pending.*

| Run | Mode | pass@1 | majority | pass@N | avg_tokens | gen_seconds |
|-----|------|--------|----------|--------|------------|-------------|
| rec1 | non-agentic, fp8, t=1.0 | 48.80% ± 3.03% [5] | 49.90% | 67.00% [5] | 40,952 | 728 |
| r20 | nano+gpt, all tools, thinking=False, tc≤15 | 28.80% ± 2.28% [5] | 29.20% | 58.00% [5] | 3,232 | 4,098 |
| r24 | fp8, no blocking, collect_results, thinking=False, tc≤15 | 31.25% ± 2.22% [4] | 33.25% | 57.00% [4] | 3,329 | 3,166 |
| r25 | same as r24 + thinking=True + tc≤4 | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

## Next steps

*To be filled after results arrive.*
