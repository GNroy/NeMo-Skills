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

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r12 | single-agent, GB300 (no fp8) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| rec1 | non-agentic, GB300 fp8, t=1.0 | 48.80% ± 3.03% | 49.90% | 67.00% | 40,952 | 728 |
| r18 | nano + gpt workers, old prompt | 20.60% ± 2.88% | 15.60% | 52.00% | 8,236 | 7,562 |
| r19 | nano + gpt workers, delegate-once prompt | 28.20% ± 2.05% | 30.20% | 56.00% | 3,701 | 3,571 |
| r20 | nano + gpt workers, enforce-all + no-followup prompt | **28.80% ± 2.28%** | **29.20%** | **58.00%** | **3,232** | **4,098** |

### Subject breakdown

| Subject | pass@1 | majority@5 | pass@5 | avg_tokens | n problems |
|---------|--------|------------|--------|------------|------------|
| Physics | 26.00% ± 3.74% | 25.40% | 54.00% | 4,564 | 50 |
| Chemistry | 31.50% ± 6.75% | 33.25% | 62.50% | 2,264 | 40 |
| Biology | 32.00% ± 8.37% | 32.00% | 60.00% | 446 | 10 |

### Worker behavior (diagnostic)

| Metric | r19 | r20 | Change |
|--------|-----|-----|--------|
| Records analyzed | 500 | 500 | — |
| Empty generations | 90 / 500 (18.0%) | 86 / 500 (17.2%) | −0.8 pp |
| `finish_reason=length` hits | 4 / 500 (0.8%) | 6 / 500 (1.2%) | +0.4 pp |
| Hit `max_tool_calls=15` | 79 / 500 (15.8%) | 79 / 500 (15.8%) | 0 pp |
| Called both workers | ~202 / 500 (40.4%) | 335 / 500 (67.0%) | **+26.6 pp** |
| Called GPT only | ~294 / 500 (58.8%) | 149 / 500 (29.8%) | −29.0 pp |
| nano timeouts | — | 197 / 500 (39.4%) | — |
| nano timeout + GPT empty | — | 96 / 500 (19.2%) | — |
| `collect_results` calls (total) | ~632+ | 921 | — |
| avg `collect_results` per record | ~1.3 | 1.84 | +0.5 |

Tool call distribution: {0:6, 1:5, 2:25, 3:82, 4:190, 5:46, 6:27, 7:15, 8:6, 9:6, 10:4, 11:4, 12:2, 13:2, 14:1, 15:79}

## Analysis

### Primary question: no — collect_results was never suppressed

The r20 prompt explicitly bans `collect_results`, but the orchestrator calls it in every
session (921 calls / 500 records = 1.84× average). The model completely ignores the
text-level prohibition. This is the dominant effect of r20: instead of waiting for
auto-injection, the model always polls explicitly, adding 1–2 extra tool calls per session.

Instruction-level bans on tool calls are ineffective for Nano-30B. The only reliable
fix is to **remove `collect_results` from the tool schema entirely**.

### Tool call mode shifted from 2 → 4, explained by collect_results

r19 typical session (GPT-only): call_gpt_async (1) + collect_results (1) = 2 tool calls → mode at 2.

r20 typical session (both workers): call_nano_async + call_gpt_async (2) + collect_results × 1.71 ≈ 3.7 tool calls → mode at 4.

GPT-only sessions in r20 average 2.28 collect_results calls (the model re-polls after
getting results), pushing them to tc=4–5.

Conversation trace confirmed: model makes async calls sequentially (one per turn),
then explicitly calls collect_results with all ticket IDs. Auto-injection fires as
well, but the explicit collect_results call always precedes or races it.

### "Call ALL agents" instruction succeeded on worker diversity

Both workers called: 67.0% (r20) vs 40.4% (r19), a +26.6 pp improvement. The
explicit "You MUST call ALL of them" instruction is partially effective — the majority
of sessions now include nano. However, this did not improve accuracy (see below).

### Accuracy flat: +0.6 pp over r19, well below baselines

pass@1 = 28.80% — not meaningfully different from r19's 28.20% (within the ±2.3 pp
confidence interval). Despite calling both workers 67% of the time (vs 40% in r19),
there is no accuracy benefit. Two explanations:

1. **Nano times out 39.4% of the time** — nano's answer is absent in ~40% of sessions,
   so calling it has no effect on the output.
2. **Orchestrator synthesis still discards nano** — even when nano returns a result, the
   orchestrator (a Nano-30B model) likely cannot meaningfully compare or combine two
   solutions and falls back on GPT's answer.

The 96 double-failure sessions (nano timeout + GPT empty = 19.2%) are likely the
primary driver of the persistent 15.8% max_tool_calls pile, as the orchestrator has
no valid result to synthesize and retries GPT.

### Nano timeout rate (39.4%) is a critical unaddressed bottleneck

Nearly 40% of nano worker calls time out. This was not measured in r19 but is likely
similar. Every timeout forces the orchestrator to work with GPT's answer alone
(when GPT has one), or to retry (when GPT is also empty). The 1200s timeout is too
short for nano on many problems — or nano is simply unable to produce a valid solution
for frontier-science-level problems within that budget.

### gen_seconds increased despite fewer tokens

gen_seconds = 4,098 (up from r19's 3,571, +15%). avg_tokens = 3,232 (down from
3,701, −13%). The token count fell (fewer GPT tokens reused after better orchestration)
but wall-clock time increased — consistent with the nano timeout cost (1200s timeouts
in 39.4% of sessions inflate total wall time).

## Next steps

### Highest priority: remove `collect_results` from the tool list (r21)

Since instruction-level bans are ignored, removing the tool schema entry is the only
reliable way to eliminate explicit polling. With collect_results unavailable:
- Sessions would settle at tc=2 (2 async calls, auto-injection, final answer)
- The retry loops driven by explicit collect_results loops would stop
- ETA: ~−1.8 tool calls per session, likely reducing gen_seconds by ~30%

This is the single most impactful change available with no code changes to the prompt.

### Address nano timeout rate

39.4% nano timeouts is too high. Options:
- Increase the nano timeout limit from 1200s to a larger value
- Reduce nano's `max_tokens` (thinking tokens) so it finishes faster
- Accept that nano adds no value on fs problems and test GPT-only as a single worker

A GPT-solo worker run (no orchestrator overhead) would establish the ceiling for
the current two-model setup. If GPT-solo matches or exceeds the two-worker pass@1,
the orchestrator layer is purely overhead.

### Diagnose the orchestrator synthesis gap

Even when both workers produce valid answers, the pass@5 = 58% (5 independent GPT
answers, no orchestration) is only marginally above rec1's pass@5 = 67%. The
orchestrator is not adding value through synthesis. Consider:
- Logging which worker's answer the orchestrator chose and whether it was correct
- Testing whether the orchestrator discards correct nano answers in favor of GPT
