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
| Commit | TBD |
| SLURM generation jobs | TBD |
| SLURM eval jobs | TBD |

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

*Pending.*

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent, OCI A100 | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r12 | single-agent, GB300 (no fp8) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| r16 | nano + gpt workers, GB300 (nano token budget bug) | 24.40% ± 3.71% | 20.20% | 56.00% | 7,548 | 2,694 |
| r17 | nano worker, GB300 (no fp8, FLASH_ATTN, token budget fix) | TBD | TBD | TBD | TBD | TBD |
| r18 | nano + gpt workers, GB300 (no fp8, FLASH_ATTN, token budget fix) | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

## Next steps

*To be filled after results arrive.*
