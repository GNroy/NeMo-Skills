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
| Commit | `0d26915b feat: add --attention-backend flag; reports for r13/r14` |
| SLURM generation jobs | 92447, 92449, 92451, 92453, 92455 (rs0–rs4, het-jobs +0/+1) |
| SLURM eval jobs | 92457–92462 (judge 92457–92461, summarize 92462) |
| Seeds completed | **4 of 5** (rs4 / job 92455 appears to have failed) |

## Motivation

First test of the two-worker setup: nano orchestrator dispatches to a nano worker (same
model, co-located) and a gpt-oss-120b worker (separate server). The orchestrator calls
both non-blockingly and picks the best answer or falls back if one produces nothing.
Uses same GB300 config as r13 (no fp8, FLASH_ATTN) so the two-worker effect can be
isolated. Resource cost: 2 nodes/seed (orch+nano share one node, gpt gets another).

## Results

| Run | Mode | pass@1 (judge) | majority@N | pass@N | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent, OCI A100 | 41.00% ± 3.16% | 40.10% (N=5) | 63.00% | 50,296 | 1,891 |
| r12 | single-agent, GB300 (no fp8) | 37.00% ± 2.12% | 33.00% (N=5) | 59.00% | 56,174 | 1,000 |
| r13 | nano worker (sandbox bug) | 18.40% ± 2.24% | 17.80% (N=5) | 41.00% | 3,703 | 3,189 |
| r14 | nano + gpt workers (sandbox bug) | **20.50% ± 2.89%** | **17.50% (N=4)** | **44.00%** | 6,823 | 2,806 |

### Subject breakdown (r14, 4 seeds)

| Subject | pass@1 | n problems |
|---------|--------|------------|
| Physics | 18.0% | 50 |
| Chemistry | 23.1% | 40 |
| Biology | 22.5% | 10 |

## Analysis

### Root cause: same sandbox port bug as r13

Both workers (nano co-located, gpt separate het-group) received no `NEMO_SKILLS_SANDBOX_PORT`
in their containers. The nano worker fell back to port 6000 (wrong), and the gpt worker in
het-group 1 had the same issue — plus no sandbox was started in its het-group at all.
Every tool call timed out for both workers.

r14 is marginally better than r13 (20.5% vs 18.4%) likely because gpt-oss-120b is a better
reasoner and produced slightly better stub answers even without code execution. The higher
avg_tokens (6,823 vs 3,703) reflects gpt generating longer text responses.

Chemistry improved most in r14 (23.1% vs 19.5% in r13) — consistent with gpt-oss-120b
having stronger chemistry knowledge for pure text reasoning.

### Fix (r15/r16)

In addition to propagating `NEMO_SKILLS_SANDBOX_PORT` to co-located workers, the fix also
starts a dedicated `SandboxScript` in each non-orchestrator het-group (one per unique worker
model group). This ensures every worker that uses `DirectPythonTool` or `PythonTool` can
reach a sandbox on its own node.

## Next steps

- r16: re-run r14 config with sandbox fix → first valid two-worker measurement
- Compare r15 vs r16 to quantify two-worker benefit (if any) over single-worker
