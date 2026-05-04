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
exclusive in vLLM 0.18.x. Fixed and resubmitted as run-id 2.

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
- Does the python arm (PythonTool) with t=1.0 add value over no-tool?

## Results

| Run | Arm | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|-----|----------------|------------|--------|------------|-------------|
| r5 (OCI A100) | python, t=1.0 | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r12 (GB300, no fp8) | python, t=0.6 | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| rec1 (GB300, fp8) | no-tool, t=1.0 | **48.80% ± 3.03%** | **49.90%** | **67.00%** | 40,952 | 728 |
| rec1 (GB300, fp8) | python, t=1.0  | **48.40% ± 2.88%** | **50.00%** | **67.00%** | 36,782 | 846 |

### Subject breakdown

| Subject | no-tool pass@1 | no-tool pass@5 | python pass@1 | python pass@5 | n problems |
|---------|----------------|----------------|---------------|---------------|------------|
| Physics | 50.00% ± 4.00% | 64.00% | 48.40% ± 2.61% | 64.00% | 50 |
| Chemistry | 54.00% ± 3.79% | 75.00% | 55.00% ± 6.85% | 75.00% | 40 |
| Biology | 22.00% ± 8.37% | 50.00% | 22.00% ± 8.37% | 50.00% | 10 |

## Analysis

### Large accuracy gain over all prior baselines

rec1 achieves 48.8% pass@1 (no-tool) and 48.4% (python) — both ~8 pp above r5 (41%)
and ~12 pp above r12 (37%). This is the highest accuracy seen on this benchmark.

The most likely drivers of the gain:
1. **fp8 kv-cache on GB300**: vLLM 0.18.x on GB300 with fp8 may execute the model more
   faithfully than the A100 container used in r5 (fp8 kv-cache quantization effects can
   go either way but here appear to help).
2. **Newer vLLM / container**: the aws-cmh container (vLLM 0.18.x) has improved
   NemotronH support compared to what ran on OCI for r5.
3. **Sampling variance**: 48.8% ± 3.0% overlaps 41.0% ± 3.2% at ~1.8 SE — the
   difference is suggestive but not definitively significant from 5 seeds alone.

### PythonTool adds no benefit at t=1.0

Both arms score identically (48.8% vs 48.4%, within noise; pass@5 both 67%). The model
solves these problems through reasoning alone — PythonTool does not provide additional
signal at this temperature. avg_tokens is slightly lower for the python arm (36,782 vs
40,952), suggesting the tool calls save some reasoning tokens but don't change the answer.

### Speed improvement

gen_seconds dropped dramatically vs all prior runs: 728 s (no-tool) and 846 s (python)
vs 1,891 s for r5 and 1,000 s for r12. GB300 with fp8 is substantially faster.

### Biology remains hard

Biology scores 22% pass@1 in both arms — the same as r12 and r15. Physics (50%) and
Chemistry (54%) benefit the most from the new configuration.

## Next steps

- Compare rec1 (non-agentic, 48.8%) vs r17/r18 (multi-agent, once results arrive) to
  assess whether the agentic setup adds value over the strong non-agentic baseline.
- The ~8 pp improvement vs r5 on the same script warrants investigation: rerun r5 config
  on aws-cmh (same no-fp8 + A100-equivalent settings) to isolate hardware vs fp8 effect.
