# Experiment r30 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r30 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + nano worker (co-located) + gpt-oss-120b worker (own server + own sandbox) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` orchestrator + nano worker; `/hf_models/gpt-oss-120b` gpt worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, fp8, FlashInfer backend) |
| Key config changes | Same as r29 + **session_timeout=7200s** (3× worker_timeout) |
| Commit | `0671cc0c feat: add --session-timeout to cap per-problem orchestrator wall time` |
| SLURM generation jobs | 143949 (rs0), 143951 (rs1), 143953 (rs2), 143955 (rs3), 143957 (rs4) |
| SLURM eval jobs | 143960–143964 (judge rs0–rs4), 143965 (summarize) |
| Note | rs1 (143951), rs2 (143953), rs3 (143955) cancelled at startup within 2–4 min — HuggingFace 429 rate-limit. rs1/rs2/rs3 relaunched after `HF_HUB_OFFLINE=1` fix (jobs 156112/156114/156116). All 5 seeds now complete. |

## Motivation

r29 revealed two problems:
1. **35% nano timeout rate** even with 2400s — hard problems genuinely exceed 40 min.
2. **Sequential fallback pattern**: the thinking=True orchestrator sometimes calls nano alone → waits 2400s for timeout → then calls GPT. This causes problems to take 40+ min end-to-end, exhausting the 4h SLURM walltime before all 100 problems complete (rs1 and rs4 cancelled with 61/65 done).

r30 adds `session_timeout=7200s`: each problem's full orchestrator session (all tool-call turns + synthesis) is capped at 2h via `asyncio.wait_for`. On expiry the output is saved as empty and the slot freed immediately, preventing walltime exhaustion.

Key changes vs r29:
- `--session-timeout 7200` → `session_timeout=7200` in GenerationTaskConfig
- Everything else identical to r29 (thinking=True, worker_timeout=2400s, concurrency 128/agent)

## Results

| Run | Mode | pass@1 | majority | pass@N | avg_tokens | gen_seconds |
|-----|------|--------|----------|--------|------------|-------------|
| rec1 | non-agentic, fp8, t=1.0 | 48.80% ± 3.03% [5] | 49.90% | 67.00% [5] | 40,952 | 728 |
| r24 | fp8, no blocking, collect_results, thinking=False, tc≤15 | 31.25% ± 2.22% [4] | 33.25% | 57.00% [4] | 3,329 | 3,166 |
| r25 | same + thinking=True, tc≤4 | 28.00% ± 4.42% [5] | 21.10% | 59.00% [5] | 70,735 | 7,576 |
| r26 | same + brief-thinking prompt (KILLED — nano timeouts ~72%) | — | — | — | — | — |
| r27 | same + concurrency 256/256 (KILLED — nano timeouts ~60%) | — | — | — | — | — |
| r28 | same + concurrency 64/64 (KILLED — nano timeouts ~45–76%) | — | — | — | — | — |
| r29 | same + timeout=2400s, concurrency 128/128 (rs1/rs4 partial) | 31.0% ± 1.7% [3] | 26.7% | 50.0% [3] | 70,156 | 5,859 |
| r30 | same + session_timeout=7200s *(all 5 seeds after relaunch)* | **30.2% ± 4.8% [5]** | **19.8%** | **62.0% [5]** | **38,418** | **13,186** |

Per-seed: rs0=27%, rs1=25%, rs2=29%, rs3=37%, rs4=33%. Empty rate 36–47%/seed. pass@1 ± std_dev from per-seed values. gen_seconds per-seed average (~3h40m — within 4h wall ✓).

## Analysis

**Startup failures (original run)**: rs1/rs2/rs3 hit the HF 429 rate limit during vLLM startup. Fixed via `HF_HUB_OFFLINE=1` in cluster config; all 3 seeds relaunched successfully.

**Session timeout counts (original 2 successful seeds):**

| Seed | Nano timeouts | Session timeouts | Empty rate | pass@1 |
|------|--------------|------------------|------------|--------|
| rs0 | 42/100 (42%) | 2/100 (2%) | 44% | 27% |
| rs1 | — | — | 47% | 25% |
| rs2 | — | — | 42% | 29% |
| rs3 | — | — | 35% | 37% |
| rs4 | 45/100 (45%) | 33/100 (33%) | 36% | 33% |

Nano/session timeout fields were not captured for the relaunched seeds (rs1/rs2/rs3); empty rate is the best proxy. High empty rates (35–47%) confirm the session timeout problem persists across all seeds.

**rs4: 33% session timeouts** from the retry loop pattern: thinking=True orchestrator calls nano 3+ times sequentially (3 × 2400s = 7200s hits session cap exactly). max_tool_calls=15 allows this; the orchestrator ignores the "fire both workers simultaneously" instruction in its prompt.

**avg_tokens = 38,418** (full 5-seed average; down from 44,672 for 2 seeds): the relaunched seeds have slightly more empties pulling the average down.

**majority@5 = 19.8%** is strikingly low vs r24's 33.25%. High empty rate means many problems are answered correctly in 1–2 seeds but not the majority, so majority voting collapses. This is the retry loop's second-order effect.

**Accuracy flat**: 30.2% ≈ r29's 31.0% ≈ r24's 31.25%. The thinking=True setup provides no gain; the retry loop converts problems that would have been answered into empties.

## Next steps

Two independent problems to fix:

1. **HF startup failures**: model should be pre-loaded from local path on the cluster. Check if `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` is available locally (e.g. `/hf_models/`) and use that path instead of the HF Hub ID.

2. **Retry loop causing session timeouts**: reduce `max_tool_calls` from 15 to 3 (N workers + 1 collect_results = 3 for this setup). The orchestrator prompt already says total tool calls should be N+1 = 3, but the model ignores it under thinking=True. A hard tc=3 cap enforces the intended 3-turn workflow and eliminates the retry loop.
