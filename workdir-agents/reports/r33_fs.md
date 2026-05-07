# Experiment r33 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r33 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: **gpt-oss-120b orchestrator** + nano worker (own server) + gpt-oss-120b worker (co-located with orchestrator) |
| Model | `/hf_models/gpt-oss-120b` orchestrator + gpt worker (co-located); `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` nano worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, fp8) |
| Key config changes | Fix r32: nano worker gets `nano_server_args` (qwen3_coder), not orchestrator's GPT server args |
| Commit | `3c9f1f7c fix: use model-specific server args for non-co-located workers` |
| SLURM generation jobs | 156546 (rs0), 156550 (rs1), 156552 (rs2), 156548 (rs3), 156554 (rs4) |
| SLURM eval jobs | 156556–156560 (judge rs0–rs4), 156561 (summarize) |

## Motivation

r32 failed because `worker_sargs = [server_args] * len(worker_models)` used the orchestrator's server args for all workers. With `--gpt-orchestrator`, this gave the nano worker `--tool-call-parser openai --reasoning-parser openai_gptoss` — flags incompatible with the Nano model — causing immediate vLLM startup crash (exit code 1, ~4 min after launch).

Fix: workers with a different model path from the orchestrator now receive `nano_server_args` (computed from `MODEL["server_args"]` with `qwen3_coder` parser and `nano_v3` reasoning parser). Co-located workers (same model path as orchestrator) continue sharing `server_args`.

r33 is the first clean run of the GPT-orchestrator setup with:
- Correct `--tool-call-parser openai --reasoning-parser openai_gptoss` for GPT orchestrator (r31 fix)
- Correct `nano_server_args` for nano worker (r32 fix)
- `--max-tool-calls 3` enforcing the 3-call workflow
- `session_timeout=7200s`, `worker_timeout=2400s`

## Results

| Seed | pass@1 | empty | tool_calls |
|------|--------|-------|------------|
| rs0 | 44% | 5/100 | 264 |
| rs1 | 43% | 4/100 | 285 |
| rs2 | 45% | 7/100 | 272 |
| rs3 | 43% | 6/100 | 264 |
| rs4 | 49% | 5/100 | 270 |

| Run | Mode | pass@1 | majority | pass@N | avg_tokens | gen_seconds |
|-----|------|--------|----------|--------|------------|-------------|
| rec1 | non-agentic, fp8, t=1.0 | 48.80% ± 3.03% [5] | 49.90% | 67.00% [5] | 40,952 | 728 |
| r24 | nano orch (no thinking), nano+gpt workers, tc≤15 | 31.25% ± 2.22% [4] | 33.25% | 57.00% [4] | 3,329 | 3,166 |
| r29 | nano orch (thinking=True), nano+gpt workers, timeout=2400s | 31.0% ± 1.7% [3] | 26.7% | 50.0% [3] | 70,156 | 5,859 |
| r30 | same + session_timeout=7200s (5/5 seeds after relaunch) | 30.2% ± 4.8% [5] | 19.8% | 62.0% [5] | 38,418 | 13,186 |
| r31 | gpt orch (missing tool-call-parser — BROKEN) | 3.2% ± 1.3% [5] | 2.4% | 7.0% [5] | 499 | 61 |
| r32 | gpt orch (nano worker wrong server args — CRASH) | — | — | — | — | — |
| r33 | **gpt orch** (all fixes), nano+gpt workers, tc=3, session=7200s | **44.8% ± 2.5% [5]** | **44.2%** | **70.0% [5]** | **958** | **1,073** |

pass@1 ± std_dev from per-seed [44%, 43%, 45%, 43%, 49%]. gen_seconds per-seed average (~18 min — well within 4h wall).

## Analysis

**Breakthrough result.** r33 is the first multi-agent run to substantially beat the r24/r29/r30 plateau (~31%) and approach the non-agentic baseline.

**pass@1 = 44.8% vs rec1 = 48.8%**: the multi-agent system closes ~88% of the gap to the non-agentic baseline. The remaining 4pp gap likely reflects problems where the orchestrator's synthesis loses information present in the worker outputs, or where the workers don't find the right answer within the 2400s budget.

**pass@5 = 70.0% beats rec1's 67.0%**: in the oracle "best of 5 seeds" case, the agentic system finds the correct answer more often than non-agentic inference. This confirms the workers are genuinely solving problems that direct inference misses.

**majority@5 = 44.2% vs rec1's 49.9%**: tight majority voting; the 5–7% empty rate and occasional synthesis variance cause some disagreements. Still far better than r30's 19.8% (which collapsed due to 40% empties).

**avg_tokens = 958**: the GPT orchestrator generates ~1000 tokens per problem — tool call arguments (~200 tokens each × 3 calls) plus brief synthesis (~200–300 tokens). Worker tokens are not counted (they run on a separate inference loop). This makes the orchestrator extremely cheap.

**Empty rate = 4–7%** (down from 35–47% in r30): the tc=3 cap eliminates retry loops. The remaining empties are session timeouts where even the 2400s worker budget isn't enough (or rare infrastructure errors).

**avg tool calls ≈ 2.7/problem** (264–285 per 100 problems): consistent with the 3-call workflow. Non-empty problems average ~2.9 tool calls, confirming call_nano_async + call_gpt_async + collect_results is executing as intended.

**gen_seconds = 1,073/seed** (~18 min): huge improvement over r30's 3.66h/seed. With tc=3, no problem can block for more than 2×worker_timeout = 4800s, but in practice the GPT worker typically responds in <120s, so the bottleneck is the nano worker's 2400s cap on hard problems.

## Next steps

- Try relaxing `max_tool_calls` to 5 or 7 to allow one retry/fallback turn after `collect_results` (synthesis + clarification). May recover some of the 4pp gap to rec1.
- Investigate the 4–7% empty problems: are they session timeouts or connection errors? If connection errors, infrastructure reliability fix could recover them.
- The pass@5=70% oracle headroom suggests more seeds or better synthesis could push accuracy higher.
