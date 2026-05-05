# Experiment r20 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r20 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + nano worker (co-located) + gpt-oss-120b worker (own server + own sandbox) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` orchestrator + nano worker; `/hf_models/gpt-oss-120b` gpt worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, no fp8, FLASH_ATTN backend) |
| Key config changes | Same as r19 + **orchestrator prompt v3**: enforce all workers called, ban collect_results and follow-up calls, explicit empty-result handling |
| Commit | `de2e5394 fix: orchestrator prompt — enforce all workers called, forbid collect_results and follow-ups` |
| SLURM generation jobs | 128663, 128665, 128667, 128669, 128671 (rs0–rs4, het-jobs +0/+1) |
| SLURM eval jobs | 128673–128678 (judge 128673–128677, summarize 128678) |

## Motivation

r19 analysis revealed two root causes of the remaining convergence failures (15.8% max_tool_calls=15):

1. **Orchestrator called only one worker 59% of the time** — the "call EVERY agent" instruction
   was not followed; the model would pick GPT (the stronger worker) and ignore nano.
2. **Empty nano responses triggered follow-up loops** — when nano returned empty, the
   orchestrator entered a multi-turn clarification Q&A with GPT (707 follow-up GPT calls
   after turn 6), consuming all remaining tool budget.

Additionally, 632 follow-up `collect_results` calls after turn 6 showed the orchestrator
using `collect_results` as an extra tool call that auto-injection would have handled for free.

r20 tests an updated orchestrator prompt that addresses all three:
- Explicitly requires ALL agents called in turn 1 ("You MUST call ALL of them")
- Bans `collect_results` and blocking `call_<name>` entirely
- States total tool calls = exactly N (one per agent)
- Explicit empty-result rule: "ignore it and use the others"

Primary question: do these constraints eliminate the follow-up loop and push most sessions
to exactly 2 tool calls (one per worker)?

## Results

*Pending.*

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r12 | single-agent, GB300 (no fp8) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| rec1 | non-agentic, GB300 fp8, t=1.0 | 48.80% ± 3.03% | 49.90% | 67.00% | 40,952 | 728 |
| r18 | nano + gpt workers, old prompt | 20.60% ± 2.88% | 15.60% | 52.00% | 8,236 | 7,562 |
| r19 | nano + gpt workers, delegate-once prompt | 28.20% ± 2.05% | 30.20% | 56.00% | 3,701 | 3,571 |
| r20 | nano + gpt workers, enforce-all + no-followup prompt | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

## Next steps

*To be filled after results arrive.*
