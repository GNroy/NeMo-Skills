# Experiment r14 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r14 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + nano worker (co-located) + gpt-oss-120b worker (own server) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` orchestrator + worker; `/hf_models/gpt-oss-120b` second worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, no fp8, FLASH_ATTN backend) |
| Key config changes | `--no-fp8 --attention-backend FLASH_ATTN --gpt-worker`; 2 het-groups (orch+nano in group 0, gpt in group 1) |
| Commit | TBD |
| SLURM generation jobs | TBD |
| SLURM eval jobs | TBD |

## Motivation

First test of the two-worker setup: nano orchestrator dispatches to a nano worker (same
model, co-located) and a gpt-oss-120b worker (separate server). The orchestrator calls
both non-blockingly and picks the best answer or falls back if one produces nothing.
Uses same GB300 config as r13 (no fp8, FLASH_ATTN) so the two-worker effect can be
isolated. Resource cost: 2 nodes/seed (orch+nano share one node, gpt gets another).

## Results

*Pending.*

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent, OCI A100 | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r12 | single-agent, GB300 (no fp8) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| r13 | nano worker, GB300 (no fp8, FLASH_ATTN) | TBD | TBD | TBD | TBD | TBD |
| r14 | nano + gpt workers, GB300 (no fp8, FLASH_ATTN) | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

## Next steps

*To be filled after results arrive.*
