# Experiment r16 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r16 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + nano worker (co-located) + gpt-oss-120b worker (own server + own sandbox) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` orchestrator + worker; `/hf_models/gpt-oss-120b` second worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, no fp8, FLASH_ATTN backend) |
| Key config changes | Same as r14 + **sandbox fix**: nano worker shares orch sandbox; gpt worker gets dedicated sandbox in het-group 1 |
| Commit | TBD |
| SLURM generation jobs | TBD |
| SLURM eval jobs | TBD |

## Motivation

r14 was invalidated by the same sandbox port bug as r13. r16 is the corrected re-run.
Both workers now have functional code execution:
- Nano worker: shares orchestrator's sandbox (co-located, same node)
- GPT worker: dedicated `SandboxScript` started in het-group 1 (its own node)

Primary question: does the two-worker setup (nano + gpt) improve accuracy over
single-worker (r15)?

## Results

*Pending.*

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent, OCI A100 | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r12 | single-agent, GB300 (no fp8) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| r15 | nano worker, GB300 (no fp8, FLASH_ATTN, sandbox fix) | TBD | TBD | TBD | TBD | TBD |
| r16 | nano + gpt workers, GB300 (no fp8, FLASH_ATTN, sandbox fix) | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

## Next steps

*To be filled after results arrive.*
