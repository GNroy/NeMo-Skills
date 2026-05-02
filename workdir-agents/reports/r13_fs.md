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
| Commit | `0d26915b feat: add --attention-backend flag; reports for r13/r14` |
| SLURM generation jobs | 92435–92439 (rs0–rs4) |
| SLURM eval jobs | 92440–92445 (judge 92440–92444, summarize 92445) |

## Motivation

r12 (no fp8) recovered ~2 pp vs r11 (fp8) but remained ~4 pp below r5 (OCI A100).
The remaining gap is hypothesised to be the attention backend: OCI A100 uses FLASH_ATTN,
GB300 auto-selects FlashInfer. r13 forces `--attention-backend FLASH_ATTN` on GB300 to
isolate this effect. It also introduces the first multi-agent run with the new
auto-deduplication: the nano worker co-locates with the orchestrator (same model → shared
server, zero extra GPU cost).

## Results

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent, OCI A100 | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r12 | single-agent, GB300 (no fp8, auto-backend) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| r13 | multi-agent (nano worker), GB300 (no fp8, FLASH_ATTN) | **18.40% ± 2.24%** | **17.80%** | **41.00%** | 3,703 | 3,189 |

### Subject breakdown (r13)

| Subject | pass@1 | n problems |
|---------|--------|------------|
| Physics | 17.6% | 50 |
| Chemistry | 19.5% | 40 |
| Biology | 18.0% | 10 |

## Analysis

### Root cause: sandbox port not propagated to worker

r13 is **severely degraded** vs r12 (18.4% vs 37.0% pass@1), but the cause is **not**
FLASH_ATTN or the multi-agent setup — it is a pipeline bug:

The nano worker's `srun` command was missing `NEMO_SKILLS_SANDBOX_PORT` in
`--container-env`. The sandbox was running on port 50811 (randomly assigned), but the
worker's container fell back to the default port 6000. Every `stateful_python_code_exec`
call returned "Client timed out", so the worker produced empty or stub answers:

```
entry 0 — generation: "FINAL ANSWER\n\boxed{U = C\,T + \frac{K'}{T}}"  (44 chars)
entry 1 — generation: ""  (0 chars)
entry 2 — generation: "FINAL ANSWER\n3.26e33 K"  (22 chars)
```

The extremely low avg_tokens (3,703 vs 56,174 for r12) and long gen_seconds (3,189 vs
1,000) are consistent with this: the worker runs (taking time), tool calls all timeout,
and the orchestrator synthesises a near-empty response.

### Fix

`AgentWorkerScript` now accepts a `sandbox` field. When set, `NEMO_SKILLS_SANDBOX_PORT`
is exported into the worker container's environment, mirroring how `GenerationClientScript`
does it for the orchestrator. Workers in separate het-groups (non-co-located) get their own
dedicated `SandboxScript` in their het-group. Committed in `<r15-r16 commit>`.

### FLASH_ATTN effect: unknown

r13/r14 cannot answer the FLASH_ATTN hypothesis — the sandbox bug dominated. r15 (same
config as r13 with the fix) will give the first valid data point.

## Next steps

- r15: re-run r13 config with sandbox fix → isolates FLASH_ATTN effect
- r16: re-run r14 config with sandbox fix → validates two-worker setup
