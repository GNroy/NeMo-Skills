# Experiment r22 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r22 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + nano worker (co-located) + gpt-oss-120b worker (own server + own sandbox) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` orchestrator + nano worker; `/hf_models/gpt-oss-120b` gpt worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, fp8, FlashInfer backend) |
| Key config changes | Same as r21 + **temperature=1.0, top_p=0.95** for orchestrator and both workers |
| Commit | `631c72c7 feat: add --temperature / --top-p flags to run_nano_physics_agent.py` |
| SLURM generation jobs | 133569, 133571, 133573, 133575, 133577 (rs0–rs4, het-jobs +0/+1) |
| SLURM eval jobs | 133582–133586 (judge rs0–rs4), 133587 (summarize) |

## Motivation

r21 (fp8 + no collect_results) is the mechanical fix run; r22 runs in parallel to
test whether the temperature=1.0 sampling preset (matching rec1's top-performing
non-agentic baseline) improves accuracy over the default t=0.6 preset.

rec1 used temperature=1.0, top_p=1.0 and achieved 48.8% pass@1 — the best result
on this benchmark. r18–r21 all used the NVIDIA tool-calling preset (t=0.6, top_p=0.95).
If the accuracy gap to rec1 is partly a sampling issue rather than purely an
orchestration issue, r22 should close it.

Key difference from r21: temperature=1.0, top_p=0.95 applied to orchestrator,
nano worker, and GPT worker (via --temperature 1.0 --top-p 0.95 CLI flags).

## Results

*Pending.*

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| rec1 | non-agentic, GB300 fp8, t=1.0 | 48.80% ± 3.03% | 49.90% | 67.00% | 40,952 | 728 |
| r20 | nano + gpt workers, t=0.6 | 28.80% ± 2.28% | 29.20% | 58.00% | 3,232 | 4,098 |
| r21 | nano + gpt workers, fp8 + no collect_results, t=0.6 | TBD | TBD | TBD | TBD | TBD |
| r22 | nano + gpt workers, fp8 + no collect_results, t=1.0 | TBD | TBD | TBD | TBD | TBD |

## Analysis

*Pending.*

## Next steps

*To be filled after results arrive.*
