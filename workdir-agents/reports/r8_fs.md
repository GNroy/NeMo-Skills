# Experiment r8 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r8 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + solver worker |
| Model | `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (nano-agent) |
| Key config changes | (1) `tool_call.py`: flatten `finish_reason` list to final element — fixes pydantic validation error in `AgentTaskResponse`; (2) `WORKER_EXTRA_ARGS`: `++inference.tokens_to_generate=32768` — caps worker per-call latency, prevents 600s HTTP timeout under 100 concurrent problems; (3) orchestrator: `++tool_overrides.CallAgentTool.timeout_s=1200` |
| Commit | `a38bf993 fix worker timeout + finish_reason serialization` |
| SLURM generation jobs | 9505826–9505831 (rs0–rs4) |
| SLURM eval jobs | 9505835–9505841 |

## Results

*Pending — jobs submitted 2026-04-30 14:10.*

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent (thinking) | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r6  | multi-agent (worker crashed) | 29.40% ± 5.59% | 23.70% | 53.00% | 66,893 | 4,843 |
| r7  | multi-agent (killed — timeout) | — | — | — | — | — |
| r8  | multi-agent | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

## Notes

*Pending.*

## Next steps

*To be filled after results arrive.*
