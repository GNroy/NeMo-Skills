# Experiment r15 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r15 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + 1 nano worker (co-located, shared server) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (HF cache, aws-cmh) |
| Cluster | aws-cmh (GB300, 4 GPUs/node, no fp8, FLASH_ATTN backend) |
| Key config changes | Same as r13 + **sandbox port fix** (`NEMO_SKILLS_SANDBOX_PORT` propagated to co-located workers) |
| Commit | TBD |
| SLURM generation jobs | TBD |
| SLURM eval jobs | TBD |

## Motivation

r13 was invalidated by a pipeline bug: the nano worker's container did not receive
`NEMO_SKILLS_SANDBOX_PORT`, causing all `DirectPythonTool` calls to time out. r15 is the
corrected re-run with the bug fixed.

Primary question: does `--attention-backend FLASH_ATTN` on GB300 (no fp8) close the
~4 pp gap between r12 (37.0%) and r5 (41.0%)?

## Results

*Pending.*

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent, OCI A100 | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r12 | single-agent, GB300 (no fp8, auto-backend) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| r15 | multi-agent (nano worker), GB300 (no fp8, FLASH_ATTN, sandbox fix) | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

If r15 ≈ r5: FLASH_ATTN explains the residual gap; use it for all future GB300 runs.
If r15 ≈ r12: attention backend is not the cause; investigate further.

## Next steps

*To be filled after results arrive.*
