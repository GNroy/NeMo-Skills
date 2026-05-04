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

*Pending.*

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent, OCI A100 | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r12 | single-agent, GB300 (no fp8, auto-backend) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| r15 | nano worker, GB300 (token budget bug) | 20.80% ± 2.39% | 19.50% | 44.00% | 5,915 | 3,499 |
| r17 | nano worker, GB300 (no fp8, FLASH_ATTN, token budget fix) | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

## Next steps

*To be filled after results arrive.*
