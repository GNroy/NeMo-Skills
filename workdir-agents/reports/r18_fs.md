# Experiment r18 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r18 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + nano worker (co-located) + gpt-oss-120b worker (own server + own sandbox) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` orchestrator + worker; `/hf_models/gpt-oss-120b` second worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, no fp8, FLASH_ATTN backend) |
| Key config changes | Same as r16 + **worker token budget fix**: `tokens_to_generate=131072` for nano worker (was 32768) |
| Commit | `d130c7f6 fix: raise nano worker token budget to 131072; add r15-r18 reports` |
| SLURM generation jobs | 124728, 124730, 124734, 124732, 124736 (rs0–rs4, het-jobs +0/+1) |
| SLURM eval jobs | 124738–124743 (judge 124738–124742, summarize 124743) |

## Motivation

r16 was partially invalidated because the nano worker still had the token budget bug:
`tokens_to_generate=32768` with `enable_thinking=True` caused empty responses from the
nano worker. Only the GPT worker (which lacks thinking mode) produced useful answers,
explaining the modest improvement over r15 (24.4% vs 20.8%).

r18 fixes the nano worker budget (`tokens_to_generate=131072`) so both workers contribute.
This is the first valid two-worker measurement.

Primary question: does the two-worker setup (nano + gpt) improve accuracy over single-worker
(r17)?

## Results

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent, OCI A100 | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r12 | single-agent, GB300 (no fp8) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| r16 | nano + gpt workers, GB300 (nano token budget bug) | 24.40% ± 3.71% | 20.20% | 56.00% | 7,548 | 2,694 |
| r17 | nano worker, GB300 (no fp8, FLASH_ATTN, token budget fix) | 15.60% ± 5.37% | 8.40% | 42.00% | 6,465 | 13,364 |
| r18 | nano + gpt workers, GB300 (no fp8, FLASH_ATTN, token budget fix) | **20.60% ± 2.88%** | **15.60%** | **52.00%** | **8,236** | **7,562** |

### Subject breakdown

| Subject | pass@1 | majority@5 | pass@5 | avg_tokens | n problems |
|---------|--------|------------|--------|------------|------------|
| Physics | 19.20% ± 4.82% | 12.80% | 48.00% | 6,360 | 50 |
| Chemistry | 21.50% ± 3.79% | 17.00% | 60.00% | 12,463 | 40 |
| Biology | 24.00% ± 11.40% | 24.00% | 40.00% | 711 | 10 |

### Worker behavior (diagnostic)

| Metric | Value |
|--------|-------|
| Records analyzed | 500 (5 seeds × 100 problems) |
| Empty generations | 161 / 500 (32.2%) |
| `finish_reason=length` hits | 17 / 500 (3.4%) |
| Hit `max_tool_calls=15` | 138 / 500 (27.6%) |
| Tool call distribution | 0:17, 1:28, 2:52, 3:76, 4:53, 5:44, 6:26, 7:28, 8:9, 9:9, 10:6, 11:10, 12:1, 13:2, 14:1, 15:138 |

## Analysis

### Token budget fix did not recover accuracy

r18 scores 20.60% pass@1 — essentially unchanged from r16 (24.40%) and r15 (20.80%), and
far below r12 (37%) and rec1 (48.8%). Fixing the nano worker token budget was necessary but
not sufficient to make the two-worker agentic setup competitive with the non-agentic baseline.

### Nano worker IS generating tokens now

avg_tokens rose from 7,548 (r16, nano mostly empty) to 8,236 — a modest increase confirming
the nano worker is now contributing thinking tokens. `finish_reason=length` dropped to only
3.4% (vs ~100% in r15 for nano), confirming the 131k budget fix works. However the quality
of those contributions is not translating into better final answers.

### Orchestrator convergence failure is the dominant problem

138/500 = 27.6% of records hit `max_tool_calls=15`. Combined with 161/500 = 32.2% empty
generations (which exceeds the 138 cap-hits), many orchestrators are spending all their
tool budget without converging to an extractable answer. This is the proximate cause of
the low accuracy.

The tool call distribution is heavy-tailed toward `15` — a bimodal pattern with a cluster
of quick solutions (1–4 calls) and a large pile-up at the cap. The pile-up suggests the
orchestrator enters a loop where it keeps querying workers without making progress.

### Biology avg_tokens anomaly persists

Biology avg_tokens = 711 (r16 was 354). Despite the token budget fix, biology problems
are still generating far fewer tokens than physics (6,360) or chemistry (12,463). This
suggests the orchestrator is not engaging deeply with biology problems — possibly returning
early or the worker responses for biology are very short.

### Two-worker advantage not demonstrated

R17 (single nano worker) results are still pending. However, r18 is essentially identical
to r16 in pass@1 (both ~20–24%), which had the nano worker fully disabled by the token
budget bug. The GPT-120B second worker does not appear to be adding meaningful signal
beyond what the nano worker alone provides once it is functional.

### Gen_seconds regression

gen_seconds = 7,562 in r18 vs 1,000 for r12 (non-agentic, same cluster). The agentic
two-worker setup is 7.5× slower and significantly less accurate. The speed penalty reflects
sequential worker call overhead plus GPT-120B server startup and sandbox execution.

## Next steps

- Wait for r17 results to complete the single-worker vs two-worker comparison; both are
  clearly well below the non-agentic baseline (rec1: 48.8%).
- Root-cause the orchestrator convergence failure: log full conversation traces for a
  sample of tool_calls=15 records to understand what the orchestrator does after workers
  respond. Is it asking for clarification repeatedly? Failing to extract the answer?
- Investigate the biology avg_tokens anomaly: are biology worker responses shorter, or
  is the orchestrator stopping early?
- The multi-agent agentic approach is currently ~28 pp below rec1. Before running more
  agentic experiments, understand what the orchestrator is actually doing with worker
  responses — it may need a better system prompt or answer-extraction step.
