# Experiment r5 — fs, physics, hle-physics

## Setup

| Field | Value |
|-------|-------|
| Run ID | r5 |
| Benchmarks | frontierscience-olympiad (fs), physics, hle-physics |
| Mode | Multi-agent: orchestrator + solver worker (same model, `worker_reuse_orchestrator_server=True`) |
| Model | `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (nano-agent) |
| Key config changes | First multi-agent run with `call_*_async` support; async injection of completed worker results |
| Commit | `8afd6fed feat(agent): non-blocking async agent delegation via call_{name}_async` |
| SLURM jobs | 9484049 (fs eval), others for physics/hle |

## Results

### frontierscience-olympiad

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5 | multi-agent | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |

### physics (r5 reference only)

| Run | pass@1 (judge) | majority@5 | pass@5 |
|-----|----------------|------------|--------|
| r5 | (see physics report) | — | — |

## Analysis

Tool usage across all benchmarks: **0% tool calls** in every seed.  The
orchestrator never delegated to the worker.  Root cause: `enable_thinking=True`
gives the model a 131k-token internal scratchpad; the model self-solved every
problem entirely within `<think>` without ever emitting a tool call.
`call_*_async` infrastructure was therefore not exercised at all.

Average gen time was fast (1,891 s) because the single-turn "think + answer"
pattern is efficient — no network round-trips to the worker.

## Notes

- The 41% fs pass@1 is the **single-agent thinking baseline** despite being a
  multi-agent submission.  Future multi-agent runs must beat this to justify
  the overhead.
- physics and hle-physics showed the same 0% tool-call pattern.

## Next steps

- Force the orchestrator to call workers on its first turn via `tool_choice=required`
- Mandate async delegation in the system prompt (`orchestrator.yaml`)
