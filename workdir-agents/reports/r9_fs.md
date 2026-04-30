# Experiment r9 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r9 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + solver worker |
| Model | `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (nano-agent) |
| Key config changes | (1) `agent_server.py`: `require_tokenizer=True` when `tokens_to_generate` set — fixes tokenizer crash on worker code exec; (2) `tool_call.py`: `KeyError` handler with actionable error listing available tools — fixes silent `"solver"` tool-name mistake; (3) Tier 1 compaction: `++max_tool_output_tokens=2000` for workers (caps code exec output), `++tool_overrides.CallAgentTool.max_injection_tokens=4000` for orchestrator (caps worker result injections to ~16 KB) |
| Commit | `d6479da2 fix: r9 — Tier 1 context compaction + worker reliability fixes` |
| SLURM generation jobs | 9507540–9507544 (rs0–rs4) |
| SLURM eval jobs | 9507517–9507518 |

## Results

*Pending — submitted 2026-04-30 ~14:57.*

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent (thinking) | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r6  | multi-agent (worker crashed) | 29.40% ± 5.59% | 23.70% | 53.00% | 66,893 | 4,843 |
| r7  | multi-agent (killed — timeout) | — | — | — | — | — |
| r8  | multi-agent (killed — tokenizer crash + KeyError) | — | — | — | — | — |
| r9  | multi-agent (Tier 1 compaction) | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

## Notes

*Pending.*

## Next steps

*To be filled after results arrive.*
