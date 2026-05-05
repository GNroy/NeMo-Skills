# Experiment r24 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r24 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + nano worker (co-located) + gpt-oss-120b worker (own server + own sandbox) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` orchestrator + nano worker; `/hf_models/gpt-oss-120b` gpt worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, fp8, FlashInfer backend) |
| Key config changes | Re-enable `collect_results` via `expose_collect_results=true`; keep `expose_blocking_calls=False` |
| Commit | `<pending>` |
| SLURM generation jobs | TBD |
| SLURM eval jobs | TBD |

## Motivation

r23 analysis identified that removing collect_results (r21–r23) was the wrong fix.
Without collect_results, the orchestrator has no explicit "stop delegating, get results"
operation, so it loops indefinitely — avg 17 async tool calls/session, 49% hit max_tool_calls.

In r20 (collect_results exposed, all tools exposed), the model followed a clean 3-step
protocol: call_async × 2 → collect_results → synthesize. TC mode = 4, max_tc rate = 15.8%.
That overhead (1.84 extra collect_results calls/session) was far less harmful than the
unbounded retry loops observed in r21–r23.

r24 re-enables collect_results while keeping blocking tools hidden:

Tool list:
- `call_nano_solver_0_async` — fire nano
- `call_gpt_solver_async` — fire gpt
- `collect_results` — explicit wait + retrieve all results

This combines:
- r20's collect_results protocol (model understands delegate+collect)
- r21's fp8 + FlashInfer nano fix (no nano blocking timeouts)
- r23's hidden blocking tools (no 1,200s blocking nano calls)

Expected outcome:
- TC: ~3-4 per session (matching r20)
- max_tc rate: ~15% (matching r20)
- gen_seconds: lower than r20 (no blocking timeouts pulling up the mean)
- pass@1: ≥ 28.8% (r20 baseline, potentially higher with faster nano execution)

## Results

*Pending.*

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| rec1 | non-agentic, GB300 fp8, t=1.0 | 48.80% ± 3.03% | 49.90% | 67.00% | 40,952 | 728 |
| r20 | nano+gpt, blocking+async+collect_results, t=0.6 | 28.80% ± 2.28% | 29.20% | 58.00% | 3,232 | 4,098 |
| r23 | fp8 + no blocking + no collect_results, t=0.6 | 16.20% ± 2.05% | 13.30% | 39.00% | 7,157 | 1,390 |
| r24 | fp8 + no blocking + collect_results, t=0.6 | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

## Next steps

*To be filled after results arrive.*
