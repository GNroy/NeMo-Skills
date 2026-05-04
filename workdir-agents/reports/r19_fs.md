# Experiment r19 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r19 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + nano worker (co-located) + gpt-oss-120b worker (own server + own sandbox) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` orchestrator + nano worker; `/hf_models/gpt-oss-120b` gpt worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, no fp8, FLASH_ATTN backend) |
| Key config changes | Same as r18 + **orchestrator prompt fix**: delegate once per agent then answer directly (was: sub-task decomposition + tie-breaker re-delegation loop) |
| Commit | `0d987603 fix: orchestrator prompt — delegate once per agent, then answer directly` |
| SLURM generation jobs | 127692, 127694, 127696, 127698, 127700 (rs0–rs4, het-jobs +0/+1) |
| SLURM eval jobs | 127704–127709 (judge 127704–127708, summarize 127709) |

## Motivation

r17 and r18 both showed ~49% and ~28% of sessions hitting `max_tool_calls=15` respectively,
with >30% empty generations. Analysis showed the orchestrator was cycling through worker calls
without converging to a final answer — the "sub-task decomposition + tie-breaker" workflow
in the old prompt encouraged this loop.

r19 tests a simpler orchestrator prompt: fire `call_<name>_async` exactly once per available
worker, wait for the auto-injected results, then immediately produce the final answer with no
further tool calls.

Primary question: does restricting the orchestrator to one delegation per worker eliminate
the convergence failure and recover accuracy toward r12/rec1 levels?

## Results

*Pending.*

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent, OCI A100 | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r12 | single-agent, GB300 (no fp8) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| rec1 | non-agentic, GB300 fp8, t=1.0 | 48.80% ± 3.03% | 49.90% | 67.00% | 40,952 | 728 |
| r18 | nano + gpt workers, GB300 (token budget fix, old prompt) | 20.60% ± 2.88% | 15.60% | 52.00% | 8,236 | 7,562 |
| r19 | nano + gpt workers, GB300 (delegate-once prompt) | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

## Next steps

*To be filled after results arrive.*
