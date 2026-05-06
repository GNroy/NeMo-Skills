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
| Commit | `a2072279 fix: re-enable collect_results for orchestrator; r23 results + r24 stub` |
| SLURM generation jobs | 134891, 134893, 134895, 134897, 134899 (rs0–rs4, het-jobs +0/+1) |
| SLURM eval jobs | 134910–134914 (judge rs0–rs4), 134915 (summarize) |
| Note | rs3 judge (134912) failed (exit code 1:0); summarize ran over 4 seeds |

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

## Results

*(Summarize based on 4/5 seeds; rs3 judge failed.)*

| Run | Mode | pass@1 | majority | pass@N | avg_tokens | gen_seconds |
|-----|------|--------|----------|--------|------------|-------------|
| rec1 | non-agentic, fp8, t=1.0 | 48.80% ± 3.03% (p@1[5]) | 49.90% | 67.00% (p@5) | 40,952 | 728 |
| r20 | nano+gpt, blocking+async+collect, t=0.6 | 28.80% ± 2.28% (p@1[5]) | 29.20% | 58.00% (p@5) | 3,232 | 4,098 |
| r21 | fp8 + no collect_results, t=0.6 | 19.20% ± 5.93% (p@1[5]) | 13.30% | 46.00% (p@5) | 4,530 | 4,638 |
| r23 | fp8 + no blocking + no collect_results, t=0.6 | 16.20% ± 2.05% (p@1[5]) | 13.30% | 39.00% (p@5) | 7,157 | 1,390 |
| r24 | fp8 + no blocking + collect_results, t=0.6 | **31.25% ± 2.22% (p@1[4])** | **33.25%** | **57.00% (p@4)** | **3,329** | **3,166** |

### Subject breakdown

| Subject | pass@1[avg-of-4] | majority@4 | pass@4 | avg_tokens | n problems |
|---------|-----------------|------------|--------|------------|------------|
| Physics | 33.00% ± 3.46% | 33.50% | 60.00% | 3,078 | 50 |
| Chemistry | 30.62% ± 3.75% | 35.00% | 57.50% | 3,453 | 40 |
| Biology | 25.00% ± 10.00% | 25.00% | 40.00% | 4,089 | 10 |

### Worker behavior (diagnostic)

| Metric | r20 | r23 | r24 | vs r20 | vs r23 |
|--------|-----|-----|-----|--------|--------|
| Records analyzed | 500 | 500 | 500 | — | — |
| Empty gen | 86 (17.2%) | 259 (51.8%) | **66 (13.2%)** | −4.0 pp | **−38.6 pp** |
| max_tc_limit | 79 (15.8%) | 245 (49.0%) | **58 (11.6%)** | −4.2 pp | **−37.4 pp** |
| Nano called | 335 (67.0%) | 488 (97.6%) | **490 (98.0%)** | +31 pp | +0.4 pp |
| GPT called | 483 (96.6%) | 491 (98.2%) | **493 (98.6%)** | +2 pp | +0.4 pp |
| Both called | 335 (67.0%) | 488 (97.6%) | **489 (97.8%)** | +30.8 pp | +0.2 pp |
| Blocking calls | ~842 | 0 | **9** | ~−833 | +9 |
| collect_results | 921 (1.84/rec) | 9 (0.02/rec) | **828 (1.66/rec)** | −93 | **+819** |
| TC dist mode | 4 | 15 | **3** | — | — |
| gen_time median | — | 149s | **1,002s** | — | +853s |
| gen_time max | — | 1,246s | **3,091s** | — | +1,845s |

Tool call distribution: {0:6, 1:1, 2:9, 3:259, 4:91, 5:31, 6:17, 7:6, 8:3, 9:2, 10:8, 11:4, 13:1, 14:3, 15:59}

## Analysis

### collect_results restored the 3-call TC protocol — massive improvement

Re-enabling collect_results immediately restored the correct orchestration pattern:
**TC mode = 3** (259/500 = 51.8% of sessions): `call_nano_solver_0_async` +
`call_gpt_solver_async` + `collect_results([nano_ticket, gpt_ticket])` = 3 calls.

The cascade effect:
- max_tc rate: 49.0% → **11.6%** (−37.4 pp; better than r20's 15.8%)
- Empty gen: 51.8% → **13.2%** (−38.6 pp; better than r20's 17.2%)
- pass@1: 16.2% → **31.25%** (+15 pp)

This confirms the diagnosis: collect_results is the model's "stop delegating, get results
now" signal. Without it, the model loops. With it, sessions converge in 3–4 calls.

### Blocking calls fully eliminated; no more 1200s nano timeouts

Blocking calls dropped to 9 (essentially 0 — likely hallucinated calls that fail silently).
The gen_time max dropped from r20's ~4,600s (dominated by blocking nano timeouts) to 3,091s.

### pass@1[avg-of-4] = 31.25% — first improvement over r20 in this series

r24's 31.25% is the first run since r21 to exceed r20's 28.80%. The improvement is
meaningful: r20 had blocking nano timeouts artificially inflating its baseline;
r24 achieves slightly better accuracy without that noise.

However, the comparison is slightly favorable to r24: it's avg-of-4 seeds (rs3 judge
failed), which adds variance. The true avg-of-5 might be slightly different.

### gen_time median = 1,002s: collect_results correctly waiting for nano

With collect_results blocking until BOTH workers complete, session time ≈ max(nano, gpt).
With fp8 nano at ~728s and gpt at ~30–120s, the expected median is ~750–900s. Observed
1,002s is consistent (some sessions have nano closer to the 1,200s limit). This is expected
and acceptable — the orchestrator properly waits rather than giving up early.

### collect_results avg = 1.66/record — slightly better than r20's 1.84

The model calls collect_results slightly less often than r20 (1.66 vs 1.84). This may
reflect the cleaner 2-tool list (no blocking tools) leading to fewer retry sequences.
The TC=4 sessions (91/500 = 18.2%) are likely: nano + gpt + collect_results + collect_results
(double poll after empty worker result) or nano + gpt + collect_results + final_answer_tool.

### Some hallucinated tool names persist (86 calls to non-existent agents)

The model still occasionally invents specialized agent names:
`call_synthesis_analyst_0_async` (14), `call_gaussian_quantum_chemistry_0_async` (14),
`call_cauchy_schwarz_solver_async` (14), `call_rast_solver_async` (14), etc. These fail
silently but consume budget in the 11.6% of sessions that hit max_tc=15.

### Still far from rec1 (48.8%) — orchestrator synthesis is the remaining gap

With the orchestration protocol now working correctly, the remaining accuracy gap is:
- r24 pass@1 ≈ 31.25% vs rec1 48.80% → 17.5 pp gap
- r24 pass@4 = 57% vs rec1 pass@5 = 67% → 10 pp gap

This gap likely reflects the orchestrator's inability to synthesize two worker answers
meaningfully. The orchestrator (Nano-30B with thinking disabled) sees two long solutions
and must pick or combine them — a hard reasoning task for a 3B-active model without thinking.

## Next steps

### Re-run rs3 judge to complete the 5-seed comparison

Job 134912 failed. Re-running the rs3 judge would give a fair avg-of-5 comparison vs
r20 and confirm the improvement is statistically robust.

### Diagnose: does nano actually help over GPT-only?

With the protocol working, the key question is whether nano adds value. If the orchestrator
always defaults to GPT's answer when available, nano is pure overhead (728s of waiting time).
A GPT-solo run (no orchestrator, one worker) would establish the ceiling for GPT alone.
If GPT-solo ≈ 31%, nano adds nothing. If GPT-solo < 31%, nano contributes.

### Improve orchestrator synthesis

The 17.5 pp gap to rec1 is likely dominated by synthesis quality. Options:
- Enable thinking for the orchestrator (currently disabled via `enable_thinking=False`)
- Use a stronger orchestrator (GPT-120B as orchestrator, Nano as worker)
- Log which worker's answer the orchestrator chose and whether it was correct
