# Experiment r17 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r17 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + 1 nano worker (co-located, shared server) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (HF cache, aws-cmh) |
| Cluster | aws-cmh (GB300, 4 GPUs/node, no fp8, FLASH_ATTN backend) |
| Key config changes | Same as r15 + **worker token budget fix**: `tokens_to_generate=131072` for nano worker (was 32768) |
| Commit | `d130c7f6 fix: raise nano worker token budget to 131072; add r15-r18 reports` |
| SLURM generation jobs | 124716, 124718, 124719, 124720, 124721 (rs0–rs4) |
| SLURM eval jobs | 124722–124727 (judge 124722–124726, summarize 124727) |

## Motivation

r15 was invalidated by a second pipeline bug: `NANO_WORKER_EXTRA_ARGS` set
`tokens_to_generate=32768` with `enable_thinking=True`. Hard frontier-science problems
exhaust the 32k token budget entirely in the `<think>` block (`finish_reason=length`),
leaving `choice.message.content` empty. The worker returned `generation=""` on every
call; the orchestrator hit `max_tool_calls=15` without a useful response.

r17 fixes this by raising the nano worker budget to `tokens_to_generate=131072` — the
same limit as the orchestrator — so the model can complete its thinking phase and emit
an answer.

Primary questions:
1. Does the token budget fix restore accuracy to at least r12 levels (37%)?
2. Does `--attention-backend FLASH_ATTN` on GB300 (no fp8) close the ~4 pp gap to r5 (41%)?

## Results

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent, OCI A100 | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r12 | single-agent, GB300 (no fp8, auto-backend) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| r15 | nano worker, GB300 (token budget bug) | 20.80% ± 2.39% | 19.50% | 44.00% | 5,915 | 3,499 |
| r17 | nano worker, GB300 (no fp8, FLASH_ATTN, token budget fix) | **15.60% ± 5.37%** | **8.40%** | **42.00%** | **6,465** | **13,364** |

### Subject breakdown

| Subject | pass@1 | majority@5 | pass@5 | avg_tokens | n problems |
|---------|--------|------------|--------|------------|------------|
| Physics | 14.00% ± 7.21% | 6.60% | 38.00% | 5,228 | 50 |
| Chemistry | 17.00% ± 6.71% | 8.25% | 50.00% | 9,465 | 40 |
| Biology | 18.00% ± 4.47% | 18.00% | 30.00% | 652 | 10 |

### Worker behavior (diagnostic)

| Metric | Value |
|--------|-------|
| Records analyzed | 500 (5 seeds × 100 problems) |
| Empty generations | 257 / 500 (51.4%) |
| `finish_reason=length` hits | 10 / 500 (2.0%) |
| Hit `max_tool_calls=15` | 247 / 500 (49.4%) |
| Tool call distribution | 0:10, 1:13, 2:35, 3:23, 4:20, 5:21, 6:24, 7:19, 8:17, 9:16, 10:13, 11:13, 12:11, 13:9, 14:9, 15:247 |

## Analysis

### Token budget fix did not recover accuracy

r17 scores 15.60% pass@1 — below r15 (20.80%) and far below r12 (37%). Given the large
uncertainty (±5.37%), r17 and r15 are not significantly different; both are fundamentally
broken. The fix resolved the `finish_reason=length` symptom (down to 2%) but did nothing
for the underlying problem: the orchestrator fails to converge to an answer.

### Orchestrator convergence failure dominates

247/500 = 49.4% of records hit `max_tool_calls=15` — the hard cap. With 257/500 = 51.4%
empty generations (more than the cap-hits), even some sessions that did not exhaust the
tool budget still produced no extractable answer. This is a near-identical pattern to r18
(27.6% cap-hits, 32.2% empty), confirming the problem is structural to the orchestration
approach rather than specific to the number of workers.

The tool call distribution is strongly bimodal: a long shoulder of 1–14 calls and then a
massive spike at 15. The orchestrator is cycling through worker calls without extracting
a final answer — it likely keeps asking the worker for more information or reformulations
without ever committing to a response.

### avg_tokens confirms worker is generating now

avg_tokens rose from 5,915 (r15, worker mostly empty) to 6,465, and `finish_reason=length`
fell from effectively 100% to 2%. The worker is completing its thinking and emitting
answers. The failure is in the orchestrator, not the worker.

### gen_seconds regression is severe

gen_seconds = 13,364 — nearly 4× r15 (3,499) and 13× r12 (1,000). With 49.4% of problems
exhausting 15 tool calls at ~131k tokens each on a shared co-located server, wall-clock
time explodes. The agentic single-worker setup is 13× slower and 59% less accurate than
the non-agentic baseline (r12).

### Biology avg_tokens anomaly

Biology avg_tokens = 652 — roughly the same as r18 (711). Despite the token budget fix,
biology problems generate almost no tokens. The orchestrator appears to bail out on biology
very early, possibly because the worker returns short answers or the orchestrator's internal
reasoning for biology problems is terse.

### Comparison with r18 (two workers)

r18 (nano + gpt, same token budget fix) scored 20.6% vs r17's 15.6%. The GPT worker's
addition modestly helped, presumably because it produces more directly useful answers
without thinking overhead. Neither run is competitive with the non-agentic baselines.

## Next steps

- **Diagnose orchestrator behavior directly**: sample 10–20 conversations from
  tool_calls=15 records and read the full turn-by-turn trace to understand what the
  orchestrator is doing after each worker response.
- **The multi-agent approach is ~22–25 pp below rec1 (48.8%)**. Before more agentic
  runs, determine whether the orchestrator prompt, the answer-extraction step, or the
  delegation strategy is the root cause of convergence failure.
- **Consider disabling thinking for the orchestrator**: with `enable_thinking=True`,
  the orchestrator may be over-deliberating before each tool call, leading to verbose
  inner monologue that cycles without output. r12 used a direct single-agent setup with
  no orchestrator — that is still the strongest configuration.
