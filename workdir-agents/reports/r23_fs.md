# Experiment r23 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r23 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + nano worker (co-located) + gpt-oss-120b worker (own server + own sandbox) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` orchestrator + nano worker; `/hf_models/gpt-oss-120b` gpt worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, fp8, FlashInfer backend) |
| Key config changes | Hide blocking call_{name} variants from tool list; update async tool description to "EXACTLY ONCE, complete problem" |
| Commit | `<pending>` |
| SLURM generation jobs | TBD |
| SLURM eval jobs | TBD |

## Motivation

r21 and r22 both regressed from r20 (19.2% and 15.2% vs 28.8%), caused by a new
failure mode introduced when collect_results was hidden:

1. **Blocking tools exposed** — Both `call_{name}` (blocking) and `call_{name}_async`
   appear in the tool list. The orchestrator calls the blocking variants 923 times per
   500 records (616 GPT + 216 nano), despite an explicit prompt ban. Text-level bans
   are ineffective for Nano-30B (same lesson as collect_results in r20). Many blocking
   nano calls hit the 1,200s timeout.

2. **Async description invited sub-task decomposition** — `call_{name}_async` was
   described as "Use for independent sub-tasks that can run while you continue reasoning."
   The model read this as: decompose the physics problem into steps and call GPT for each.
   Result: avg 6.49 GPT async calls per session (r21) and 7.78 (r22) vs the expected 1.
   This exhausts the 15-call budget: 46.2% of sessions hit max_tool_calls (r21), 53.6% (r22).

r23 applies the same schema-removal fix as the collect_results change:
- `CallAgentTool.expose_blocking_calls=False` (new default) — blocking `call_{name}`
  variants are omitted from `list_tools()`, leaving exactly 2 tools:
  `call_nano_solver_0_async` and `call_gpt_solver_async`.
- Updated async description: "Call this tool EXACTLY ONCE — pass the full, original
  problem statement and let the agent handle the entire solution. No further action
  is needed to retrieve the result."

Running at t=0.6 (back to default; t=1.0 was strictly worse in r22).

Expected outcome: tool calls settle at exactly 2 per session, max_tc hits drop to
near zero, empty generation rate drops from 48%+ to near zero, and accuracy recovers
to at least r20 (28.8%) or above.

## Results

*Pending.*

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| rec1 | non-agentic, GB300 fp8, t=1.0 | 48.80% ± 3.03% | 49.90% | 67.00% | 40,952 | 728 |
| r20 | nano + gpt workers, t=0.6 | 28.80% ± 2.28% | 29.20% | 58.00% | 3,232 | 4,098 |
| r21 | fp8 + no collect_results, t=0.6 | 19.20% ± 5.93% | 13.30% | 46.00% | 4,530 | 4,638 |
| r22 | fp8 + no collect_results, t=1.0 | 15.20% ± 3.27% | 7.60% | 37.00% | 5,773 | 3,435 |
| r23 | fp8 + no blocking + ONCE description, t=0.6 | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

## Next steps

*To be filled after results arrive.*
