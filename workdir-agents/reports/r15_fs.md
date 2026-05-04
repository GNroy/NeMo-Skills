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
| Commit | `16ef2ee8 fix: propagate NEMO_SKILLS_SANDBOX_PORT to worker containers in multi-agent jobs` |
| SLURM generation jobs | 93991–93995 (rs0–rs4) |
| SLURM eval jobs | 93996–94001 (judge 93996–94000, summarize 94001) |

## Motivation

r13 was invalidated by a pipeline bug: the nano worker's container did not receive
`NEMO_SKILLS_SANDBOX_PORT`, causing all `DirectPythonTool` calls to time out. r15 is the
corrected re-run with the bug fixed.

Primary question: does `--attention-backend FLASH_ATTN` on GB300 (no fp8) close the
~4 pp gap between r12 (37.0%) and r5 (41.0%)?

## Results

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent, OCI A100 | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r12 | single-agent, GB300 (no fp8, auto-backend) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| r15 | multi-agent (nano worker), GB300 (no fp8, FLASH_ATTN, sandbox fix) | **20.80% ± 2.39%** | **19.50%** | **44.00%** | 5,915 | 3,499 |

### Subject breakdown (r15)

| Subject | pass@1 | pass@5 | n problems |
|---------|--------|--------|------------|
| Physics | 17.60% ± 2.61% | 38.00% | 50 |
| Chemistry | 24.50% ± 4.47% | 52.50% | 40 |
| Biology | 22.00% ± 13.04% | 40.00% | 10 |

## Analysis

### Root cause: worker token budget exhausted in thinking phase

The sandbox fix worked — zero sandbox connection errors in the worker logs and
tool-call activity was observed (4,570 tool-related lines in the worker log). The fix
from r13 landed correctly. However, accuracy barely improved over r13 (20.8% vs 18.4%)
and remains far below r12 (37.0%).

The new failure mode: `NANO_WORKER_EXTRA_ARGS` sets
`++inference.tokens_to_generate=32768` with `++chat_template_kwargs.enable_thinking=True`.
Hard frontier-science problems exhaust the full 32k token budget entirely inside the
`<think>` block (`finish_reason=length`). When this happens, `choice.message.content`
is `None`/empty — the thinking tokens are in `reasoning_content`, not `content`. The
`agent_server.py` returns `generation=result.get("generation", "")` → empty string.

The orchestrator receives `'content': ''` from `call_nano_solver`, treats it as a
failed call, and retries up to `max_tool_calls=15`. It exhausts the retry budget and
either returns empty or a raw fallback. The low avg_tokens (5,915 vs 56,174 for r12)
confirms the orchestrator is doing very little work — it never gets a useful worker
response to synthesize from.

### FLASH_ATTN effect: still unknown

r15 cannot answer the FLASH_ATTN hypothesis because the worker token-exhaustion bug
dominated. The question remains open for r17.

### Chemistry vs Physics

Chemistry (24.5%) outperforms Physics (17.6%) — consistent with earlier runs. Chemistry
problems may require slightly less extended reasoning, fitting within 32k thinking tokens
more often.

## Next steps

- r17: re-run r15 config with `tokens_to_generate=131072` for the nano worker → first
  valid measurement of FLASH_ATTN effect and multi-agent nano-only setup
