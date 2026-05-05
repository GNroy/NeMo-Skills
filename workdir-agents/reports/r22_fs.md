# Experiment r22 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r22 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + nano worker (co-located) + gpt-oss-120b worker (own server + own sandbox) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` orchestrator + nano worker; `/hf_models/gpt-oss-120b` gpt worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, fp8, FlashInfer backend) |
| Key config changes | Same as r21 + **temperature=1.0, top_p=0.95** for orchestrator and both workers |
| Commit | `631c72c7 feat: add --temperature / --top-p flags to run_nano_physics_agent.py` |
| SLURM generation jobs | 133569, 133571, 133573, 133575, 133577 (rs0–rs4, het-jobs +0/+1) |
| SLURM eval jobs | 133582–133586 (judge rs0–rs4), 133587 (summarize) |

## Motivation

r21 (fp8 + no collect_results) is the mechanical fix run; r22 runs in parallel to
test whether the temperature=1.0 sampling preset (matching rec1's top-performing
non-agentic baseline) improves accuracy over the default t=0.6 preset.

rec1 used temperature=1.0, top_p=1.0 and achieved 48.8% pass@1 — the best result
on this benchmark. r18–r21 all used the NVIDIA tool-calling preset (t=0.6, top_p=0.95).
If the accuracy gap to rec1 is partly a sampling issue rather than purely an
orchestration issue, r22 should close it.

Key difference from r21: temperature=1.0, top_p=0.95 applied to orchestrator,
nano worker, and GPT worker (via --temperature 1.0 --top-p 0.95 CLI flags).

## Results

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| rec1 | non-agentic, GB300 fp8, t=1.0 | 48.80% ± 3.03% | 49.90% | 67.00% | 40,952 | 728 |
| r20 | nano + gpt workers, t=0.6 | 28.80% ± 2.28% | 29.20% | 58.00% | 3,232 | 4,098 |
| r21 | nano + gpt workers, fp8 + no collect_results, t=0.6 | 19.20% ± 5.93% | 13.30% | 46.00% | 4,530 | 4,638 |
| r22 | nano + gpt workers, fp8 + no collect_results, t=1.0 | **15.20% ± 3.27%** | **7.60%** | **37.00%** | **5,773** | **3,435** |

### Subject breakdown

| Subject | pass@1 | majority@5 | pass@5 | avg_tokens | n problems |
|---------|--------|------------|--------|------------|------------|
| Physics | 11.60% ± 4.98% | 2.40% | 28.00% | 6,216 | 50 |
| Chemistry | 19.50% ± 4.81% | 13.50% | 45.00% | 6,359 | 40 |
| Biology | 16.00% ± 11.40% | 10.00% | 50.00% | 1,211 | 10 |

### Worker behavior (diagnostic)

| Metric | r21 (t=0.6) | r22 (t=1.0) | Change |
|--------|-------------|-------------|--------|
| Records analyzed | 500 | 500 | — |
| Empty generations | 241 / 500 (48.2%) | 277 / 500 (55.4%) | +7.2 pp |
| Hit max_tool_calls=15 | 231 / 500 (46.2%) | 268 / 500 (53.6%) | +7.4 pp |
| Called both workers | 361 / 500 (72.2%) | 298 / 500 (59.6%) | −12.6 pp |
| collect_results calls | 15 | 67 | +52 |
| Blocking tool calls (total) | 923 | 619 (486 GPT + 132 nano) | −304 |
| Avg GPT async calls/record | 6.49 | 7.78 | +1.29 |
| Tool call dist (mode) | 15 (233/500) | 15 (262/500) | worse |

## Analysis

### Temperature=1.0 made every metric worse

Every metric degraded from r21:
- pass@1: 15.2% vs 19.2% (−4.0 pp)
- majority@5: 7.6% vs 13.3% (−5.7 pp)
- pass@5: 37.0% vs 46.0% (−9.0 pp)
- max_tc_limit rate: 53.6% vs 46.2% (+7.4 pp)
- Empty generations: 55.4% vs 48.2% (+7.2 pp)
- Avg GPT async calls per session: 7.78 vs 6.49 (+1.29)

Higher temperature increases the orchestrator's tendency to decompose problems and retry, exacerbating the exact failure mode identified in r21. The temperature=1.0 hypothesis (matching rec1's best non-agentic baseline) does not transfer to the orchestrator role — the orchestrator needs structured, predictable behavior, not diverse sampling.

Note: r22 avg_tokens (5,773) is higher than r21 (4,530), while gen_seconds (3,435) is lower. This is because higher temperature leads to faster divergence toward simpler/shorter completions but more tokens spent in the multi-call GPT sub-task loop before hitting max_tc.

### The underlying failure mode is identical to r21

Both r21 and r22 suffer from: blocking tools + sub-task language in async description → GPT called 6-8× per session → max_tc=15 exhausted → empty generation. Temperature does not change this failure mode; it only changes how quickly it manifests.

## Next steps

Same as r21: r23 hides blocking tools and updates async tool description to eliminate sub-task language. See r21 Analysis → Next steps and r23_fs.md.
