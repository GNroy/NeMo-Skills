# Experiment r21 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r21 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + nano worker (co-located) + gpt-oss-120b worker (own server + own sandbox) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` orchestrator + nano worker; `/hf_models/gpt-oss-120b` gpt worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, **fp8, FlashInfer backend**) |
| Key config changes | fp8 + FlashInfer (matching rec1); `collect_results` hidden from tool list (auto-injection only) |
| Commit | TBD |
| SLURM generation jobs | TBD |
| SLURM eval jobs | TBD |

## Motivation

r20 analysis identified two root causes:

1. **Nano worker times out 39.4% of the time** — r18/r19/r20 ran with
   `--no-fp8 --attention-backend FLASH_ATTN`, making the nano worker ~37%
   slower than its rec1 baseline (FLASH_ATTN is incompatible with fp8 kv-cache
   in vLLM 0.18.x; the correct fix — remove FLASH_ATTN, use fp8 + FlashInfer —
   was applied to the non-agentic script in commit `aa53c760` but not to the
   agentic run invocations).  With a ~1,000 s no-fp8 baseline and a 1,200 s
   timeout, ~40% of problems in the upper token tail time out.

2. **`collect_results` called in every session despite the prompt ban** — the
   model averages 1.84 explicit `collect_results` calls per session, inflating
   tool call counts from 2 (expected) to 4 (mode in r20) and making the
   auto-injection mechanism irrelevant.  Text-level bans do not work; removing
   the tool from the schema does.

r21 fixes both:
- Submit without `--no-fp8` and `--attention-backend FLASH_ATTN` → fp8 +
  FlashInfer matches rec1's ~728 s per problem, giving a 472 s margin before
  the 1,200 s timeout.
- `CallAgentTool.expose_collect_results=False` (new default) — `collect_results`
  is omitted from `list_tools()`, so the LLM cannot see or call it.  Results
  arrive only via auto-injection.

Expected outcome: tool calls settle at exactly 2 per session (call_nano_async
+ call_gpt_async), nano timeouts drop from 39.4% to near zero, and both workers
produce usable results in most sessions.  If these mechanical fixes hold,
accuracy should improve over r20 (28.8%).

## Results

*Pending.*

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r12 | single-agent, GB300 (no fp8) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| rec1 | non-agentic, GB300 fp8, t=1.0 | 48.80% ± 3.03% | 49.90% | 67.00% | 40,952 | 728 |
| r19 | nano + gpt workers, delegate-once prompt | 28.20% ± 2.05% | 30.20% | 56.00% | 3,701 | 3,571 |
| r20 | nano + gpt workers, enforce-all + no-followup prompt | 28.80% ± 2.28% | 29.20% | 58.00% | 3,232 | 4,098 |
| r21 | nano + gpt workers, fp8 + no collect_results | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

## Next steps

*To be filled after results arrive.*
