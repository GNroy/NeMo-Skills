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

r25 tests: **orchestrator thinking=True** + **max_tool_calls=4**.

## Results

| Run | Mode | pass@1 | majority | pass@N | avg_tokens | gen_seconds |
|-----|------|--------|----------|--------|------------|-------------|
| rec1 | non-agentic, fp8, t=1.0 | 48.80% ± 3.03% [5] | 49.90% | 67.00% [5] | 40,952 | 728 |
| r24 | fp8, no blocking, collect_results, thinking=False, tc≤15 | 31.25% ± 2.22% [4] | 33.25% | 57.00% [4] | 3,329 | 3,166 |
| r25 | same + thinking=True, tc≤4 | **28.00% ± 4.42% [5]** | **21.10%** | **59.00% [5]** | **70,735** | **7,576** |

### Subject breakdown

| Subject | pass@1[avg-of-5] | majority@5 | pass@5 | avg_tokens | n problems |
|---------|-----------------|------------|--------|------------|------------|
| Physics | 24.40% ± 2.97% | 16.00% | 52.00% | 76,086 | 50 |
| Chemistry | 33.50% ± 8.94% | 27.75% | 70.00% | 71,528 | 40 |
| Biology | 24.00% ± 11.40% | 20.00% | 50.00% | 40,803 | 10 |

### Worker behavior (diagnostic)

| Metric | r24 | r25 | Change |
|--------|-----|-----|--------|
| Records analyzed | 500 | 500 | — |
| Empty gen | 66 (13.2%) | **219 (43.8%)** | +30.6 pp |
| finish_reason=length | ~0 | **191 (38.2%)** | +38.2 pp |
| max_tc_limit | 58 (11.6%) | **8 (1.6%)** | −10 pp |
| Nano called | 490 (98.0%) | **313 (62.6%)** | −35.4 pp |
| GPT called | 493 (98.6%) | **242 (48.4%)** | −50.2 pp |
| Both called | 489 (97.8%) | **237 (47.4%)** | −50.4 pp |
| collect_results avg/rec | 1.66 | **0.44** | −1.22 |
| avg tokens/record | 3,329 | **70,735** | +67k |
| gen_time median | 1,002s | **2,282s** | +1,280s |
| gen_time max | 3,091s | **7,498s** | +4,407s |

Tool call distribution: {0:182, 1:48, 2:89, 3:118, 4:63}

## Analysis

### Root cause: thinking exhausts 131k token budget before tool calls

38.2% of sessions hit `finish_reason=length` — the orchestrator spent all 131,072
tokens in its `<think>` block reasoning about the problem before emitting any tool calls
or output. This was the exact failure mode described in the original code comment:

> "Disable thinking for the orchestrator: the worker does the heavy reasoning;
> the orchestrator only needs to delegate and synthesize. Without this, the model
> exhausts its full 131k token budget in the `<think>` block before emitting a tool
> call (finish_reason=length)."

TC distribution shows the consequence: tc=0 for 182 sessions (36.4%) — the orchestrator
never emitted a single tool call. avg_tokens = 70,735 (vs r24's 3,329), with a median
of 52,518 and max of 131,072 (the hard limit).

### max_tool_calls=4 worked as intended — but wasn't the bottleneck

max_tc hits dropped from 11.6% (r24) to 1.6% (r25). The tc=4 cap cleanly constrains
the correct sessions. However, the failure mode is now token exhaustion, not tool
call looping. max_tool_calls=4 is a good default for future runs.

### Sessions that did complete suggest thinking may help when budgeted correctly

Of 500 records, 281 (56.2%) produced non-empty generations. The overall pass@1 = 28.0%.
Conditioned on non-empty output: 28.0% / 56.2% ≈ **49.8% effective accuracy** — nearly
matching rec1's 48.8%. This implies that sessions where the orchestrator successfully
delegated and synthesized (the tc=3 modal path, 118/500 = 23.6%) performed substantially
better than r24's non-thinking sessions.

pass@5 also improved to 59% (vs r24's 57%), consistent with better synthesis quality
in the completing sessions.

Chemistry pass@1 jumped to 33.5% (vs r24's 30.6%), which may reflect thinking helping
with multi-step synthesis.

### Fix: cap orchestrator thinking tokens, not just total generation tokens

The issue is that `tokens_to_generate=131072` is shared between thinking and output.
The orchestrator needs only a small token budget for its task (delegate + brief synthesis).
Reducing to ~8,192 tokens would cap the thinking block and still leave ample budget for
tool calls and a final answer.

## Next steps

### r26: orchestrator thinking=True with reduced tokens_to_generate (~8k)

Add `--orchestrator-max-tokens` flag to cap orchestrator `tokens_to_generate` at ~8,192.
With thinking=True but budget ~8k, the model must reason briefly (not exhaustively) and
still have tokens for tool calls and synthesis. Keep max_tool_calls=4.

Expected: length exhaustion rate drops from 38.2% to near zero; sessions that complete
should show better synthesis than r24 (based on the ~50% effective accuracy observed
in completing r25 sessions). Target: pass@1 > 31.25% (r24 best).

Commit: `<pending r26 commit>`
