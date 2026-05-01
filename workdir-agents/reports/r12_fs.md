# Experiment r12 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r12 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Single-agent (PythonTool) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (HF cache, aws-cmh) |
| Cluster | aws-cmh (GB300, 4 GPUs/node, **no fp8 kv-cache**) |
| Key config changes | `--no-fp8` flag added; `--kv-cache-dtype fp8` suppressed via `--no-fp8` |
| Commit | TBD |
| SLURM generation jobs | TBD |
| SLURM eval jobs | TBD |

## Motivation

r11 showed a significant accuracy drop vs r5 (34.8% vs 41.0% pass@1) despite the same model and single-agent config. Two candidates:
1. `--kv-cache-dtype fp8` (new on GB300) — quantization artifacts on hard problems
2. FlashInfer attention backend (auto-selected on GB300 vs OCI default)

r12 isolates hypothesis (1) by disabling fp8 kv-cache entirely (`--no-fp8`). All other settings match r11.

## Results

*Pending.*

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent, OCI A100 (no fp8) | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r11 | single-agent, aws-cmh GB300 (fp8) | 34.80% ± 3.05% | 30.20% | 57.00% | 55,371 | 903 |
| r12 | single-agent, aws-cmh GB300 (no fp8) | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

If r12 ≈ r5: fp8 quantization explains the drop; disable fp8 for future GB300 runs.
If r12 ≈ r11: fp8 is not the cause; investigate FlashInfer backend or other GB300 differences.

## Next steps

*To be filled after results arrive.*
