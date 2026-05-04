# Experiment rec1 — fs (non-agentic baseline)

## Setup

| Field | Value |
|-------|-------|
| Run ID | rec1 |
| Benchmark | frontierscience-olympiad (fs) |
| Script | `inference_scripts/run_nano_physics_fs_official.py` |
| Cluster | aws-cmh (GB300, 4 GPUs/node, fp8, FLASH_ATTN) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (HF cache) |
| Commit | `4b5a3188 feat: add aws-cmh cluster support to run_nano_physics_fs_official.py` |

### Arms

| Arm | Sampling | Tool |
|-----|----------|------|
| no-tool | temp=1.0, top_p=1.0 | none |
| python  | temp=1.0, top_p=1.0 | PythonTool, max_tool_calls=50 |

Both arms: `tokens_to_generate=131072`, `enable_thinking=True`, `parse_reasoning=False`

### SLURM jobs

| Arm | Gen jobs | Eval jobs |
|-----|----------|-----------|
| no-tool | 124863–124867 (rs0–rs4) | 124868–124873 (judge 124868–124872, summarize 124873) |
| python  | 124874–124878 (rs0–rs4) | 124879–124884 (judge 124879–124883, summarize 124884) |

## Motivation

Non-agentic re-run on aws-cmh (GB300, fp8, FLASH_ATTN) with temperature=1.0, top_p=1.0
for both arms (NVIDIA recommended reasoning preset). Provides a direct comparison
baseline for the multi-agent r17/r18 runs on the same cluster.

Key questions:
- How does GB300 + fp8 + FLASH_ATTN compare to r5 (OCI A100, same script)?
- Does the python arm (PythonTool) with t=1.0 match or exceed r5 (t=1.0)?

## Results

*Pending.*

| Run | Arm | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|-----|----------------|------------|--------|------------|-------------|
| r5 (OCI A100) | python, t=1.0 | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r12 (GB300, no fp8) | python, t=0.6 | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| rec1 (GB300, fp8) | no-tool, t=1.0 | TBD | TBD | TBD | TBD | TBD |
| rec1 (GB300, fp8) | python, t=1.0  | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

## Next steps

*To be filled after results arrive.*
