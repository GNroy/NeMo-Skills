# Experiment r13 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r13 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + 1 nano worker (co-located, shared server) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (HF cache, aws-cmh) |
| Cluster | aws-cmh (GB300, 4 GPUs/node, no fp8, FLASH_ATTN backend) |
| Key config changes | `--no-fp8 --attention-backend FLASH_ATTN`; nano worker (thinking=True); auto server deduplication (co-located) |
| Commit | TBD |
| SLURM generation jobs | TBD |
| SLURM eval jobs | TBD |

## Motivation

r12 (no fp8) recovered ~2 pp vs r11 (fp8) but remained ~4 pp below r5 (OCI A100).
The remaining gap is hypothesised to be the attention backend: OCI A100 uses FLASH_ATTN,
GB300 auto-selects FlashInfer. r13 forces `--attention-backend FLASH_ATTN` on GB300 to
isolate this effect. It also introduces the first multi-agent run with the new
auto-deduplication: the nano worker co-locates with the orchestrator (same model → shared
server, zero extra GPU cost).

## Results

*Pending.*

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent, OCI A100 | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r12 | single-agent, GB300 (no fp8, auto-backend) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| r13 | multi-agent (nano worker), GB300 (no fp8, FLASH_ATTN) | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

If r13 ≈ r5: FLASH_ATTN explains the residual gap; use it for all future GB300 runs.
If r13 ≈ r12: attention backend is not the cause; investigate further.

## Next steps

*To be filled after results arrive.*
