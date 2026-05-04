# Experiment rec1 — fs (non-agentic baseline)

## Setup

| Field | Value |
|-------|-------|
| Run ID | rec1 |
| Benchmark | frontierscience-olympiad (fs) |
| Script | `inference_scripts/run_nano_physics_fs_official.py` |
| Cluster | aws-cmh (GB300, 4 GPUs/node, fp8, auto attention backend) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (HF cache) |
| Commit | `bd4497fe fix: remove --attention-backend FLASH_ATTN from non-agentic baseline script` |

### Arms

| Arm | Sampling | Tool |
|-----|----------|------|
| no-tool | temp=1.0, top_p=1.0 | none |
| python  | temp=1.0, top_p=1.0 | PythonTool, max_tool_calls=50 |

Both arms: `tokens_to_generate=131072`, `enable_thinking=True`, `parse_reasoning=False`

### SLURM jobs

rec1 (run-id 1) failed at vLLM startup: `FLASH_ATTN` + `fp8 kv-cache` are mutually
exclusive in vLLM 0.18.x. Fixed and resubmitted as run-id 2 (rec2 below).

| Arm | Gen jobs | Eval jobs |
|-----|----------|-----------|
| no-tool (failed rec1) | 124863–124867 | 124868–124873 |
| python  (failed rec1) | 124874–124878 | 124879–124884 |
| no-tool (rec2) | 124888–124892 (rs0–rs4) | 124893–124898 (judge 124893–124897, summarize 124898) |
| python  (rec2) | 124899–124903 (rs0–rs4) | 124904–124909 (judge 124904–124908, summarize 124909) |

## Motivation

Non-agentic re-run on aws-cmh (GB300, fp8, auto attention backend) with temperature=1.0,
top_p=1.0 for both arms (NVIDIA recommended reasoning preset). Provides a direct
comparison baseline for the multi-agent r17/r18 runs on the same cluster.

Key questions:
- How does GB300 + fp8 compare to r5 (OCI A100, same script)?
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
