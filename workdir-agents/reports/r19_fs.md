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

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent, OCI A100 | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r12 | single-agent, GB300 (no fp8) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| rec1 | non-agentic, GB300 fp8, t=1.0 | 48.80% ± 3.03% | 49.90% | 67.00% | 40,952 | 728 |
| r18 | nano + gpt workers, GB300 (old prompt) | 20.60% ± 2.88% | 15.60% | 52.00% | 8,236 | 7,562 |
| r19 | nano + gpt workers, GB300 (delegate-once prompt) | **28.20% ± 2.05%** | **30.20%** | **56.00%** | **3,701** | **3,571** |

### Subject breakdown

| Subject | pass@1 | majority@5 | pass@5 | avg_tokens | n problems |
|---------|--------|------------|--------|------------|------------|
| Physics | 25.60% ± 4.98% | 24.40% | 52.00% | 3,376 | 50 |
| Chemistry | 34.50% ± 4.81% | 41.00% | 65.00% | 4,318 | 40 |
| Biology | 16.00% ± 11.40% | 16.00% | 40.00% | 2,858 | 10 |

### Worker behavior (diagnostic)

| Metric | r18 (old prompt) | r19 (delegate-once) | Change |
|--------|-----------------|---------------------|--------|
| Records analyzed | 500 | 500 | — |
| Empty generations | 161 / 500 (32.2%) | 90 / 500 (18.0%) | −14.2 pp |
| `finish_reason=length` hits | 17 / 500 (3.4%) | 4 / 500 (0.8%) | −2.6 pp |
| Hit `max_tool_calls=15` | 138 / 500 (27.6%) | 79 / 500 (15.8%) | −11.8 pp |
| Peak tool call count | 15 (138 records) | 2–3 (240 records) | shifted left |

Tool call distribution: {0:4, 1:24, 2:137, 3:103, 4:78, 5:23, 6:14, 7:11, 8:5, 9:7, 10:6, 11:2, 12:5, 13:1, 14:1, 15:79}

## Analysis

### Delegate-once prompt substantially reduces convergence failure

The prompt fix cut the `max_tool_calls=15` rate from 27.6% to 15.8% and empty generations
from 32.2% to 18.0%. The tool call distribution shifted from a bimodal (1–5 shoulder + 15
spike) to a mode at 2–3 (the expected count for 2 workers each called once). Most
orchestrators now follow the intended workflow.

### Accuracy improved +7.6 pp vs r18, but still well below baselines

pass@1 = 28.20% — up from r18's 20.60% but still 8.8 pp below r12 (37%) and 20.6 pp below
rec1 (48.8%). The prompt fix is directionally correct but insufficient on its own. The
accuracy gap to single-agent baselines remains large.

### Speed improved 2× vs r18

gen_seconds = 3,571 (vs r18's 7,562). avg_tokens = 3,701 (vs r18's 8,236). With ~2–3 tool
calls per problem instead of the r18 loop, the orchestrator overhead is substantially
reduced. The two-worker setup is now only ~3.6× slower than r12 (vs r18's 7.6×).

### Residual 15.8% max_tool_calls pile

79/500 = 15.8% of sessions still hit the cap despite the "call once" instruction. The
most likely cause: `tool_choice=required` forces a tool call even after the orchestrator
should stop. When both workers have been called and results injected, `tool_choice=required`
may push the orchestrator to fire another call instead of writing a final answer. Removing
`tool_choice=required` (or switching to `tool_choice=auto`) after initial delegation could
eliminate this residual loop.

### Biology avg_tokens recovered but accuracy did not improve

Biology avg_tokens jumped from 711 (r18) to 2,858 — the orchestrator is now actually
engaging with biology problems (the biology anomaly from r17/r18 is resolved). However
pass@1 = 16.0% (r18 was 24.0%). The additional tokens don't translate to correct answers;
biology problems may require domain knowledge the workers lack.

### majority@5 > pass@1 for the first time

majority@5 = 30.20% exceeds pass@1 = 28.20% — the first time in the agentic runs. This
indicates the orchestrator has enough diversity across seeds to benefit from majority vote,
a healthy sign. Chemistry in particular: majority@5 = 41.00% vs pass@1 = 34.50%.

## Next steps

- **Remove `tool_choice=required` for orchestrator**: with the delegate-once prompt, the
  orchestrator should not need forced tool use after the initial delegation. Try
  `tool_choice=auto` to allow it to stop calling tools and write the final answer.
- **The pass@1 gap to rec1 (−20.6 pp) is still large.** Diagnose a sample of correct
  rec1 / incorrect r19 problems to understand whether the orchestrator's synthesis step
  is discarding or misinterpreting correct worker answers.
- Consider enabling `enable_thinking=True` for the orchestrator in the synthesis step
  only (e.g., a second-pass model call after worker results arrive) to improve final
  answer quality without the looping problem.
