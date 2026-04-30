# Experiment r10 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r10 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + solver worker |
| Model | `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (nano-agent) |
| Key config changes | Fix `KeyError` handler to list actual tool names from `_raw_to_qualified_map` instead of `schema_mappings` keys (which returned `['tool_names', 'parameters']`) |
| Commit | `79563856 fix: KeyError handler lists actual tool names, not schema_mappings keys` |
| SLURM generation jobs | 9508718 (rs0), 9508719 (rs1), 9508720 (rs2), 9508721 (rs3), 9508724 (rs4) |
| SLURM eval jobs | 9508734–9508738 (judge rs0–rs4), 9508740 (summarize) |

## Results

*Pending — submitted 2026-04-30 ~17:30.*

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent (thinking) | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r6  | multi-agent (worker crashed) | 29.40% ± 5.59% | 23.70% | 53.00% | 66,893 | 4,843 |
| r7  | multi-agent (killed — timeout) | — | — | — | — | — |
| r8  | multi-agent (killed — tokenizer crash + KeyError) | — | — | — | — | — |
| r9  | multi-agent (Tier 1 compaction; bad KeyError msg) | TBD | TBD | TBD | TBD | TBD |
| r10 | multi-agent (fixed KeyError tool listing) | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

## Notes

*Pending.*

## Next steps

*To be filled after results arrive.*
