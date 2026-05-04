# Experiment r16 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r16 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + nano worker (co-located) + gpt-oss-120b worker (own server + own sandbox) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` orchestrator + worker; `/hf_models/gpt-oss-120b` second worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, no fp8, FLASH_ATTN backend) |
| Key config changes | Same as r14 + **sandbox fix**: nano worker shares orch sandbox; gpt worker gets dedicated sandbox in het-group 1 |
| Commit | `16ef2ee8 fix: propagate NEMO_SKILLS_SANDBOX_PORT to worker containers in multi-agent jobs` |
| SLURM generation jobs | 94002, 94004, 94006, 94008, 94010 (rs0–rs4, het-jobs +0/+1) |
| SLURM eval jobs | 94012–94017 (judge 94012–94016, summarize 94017) |

## Motivation

r14 was invalidated by the same sandbox port bug as r13. r16 is the corrected re-run.
Both workers now have functional code execution:
- Nano worker: shares orchestrator's sandbox (co-located, same node)
- GPT worker: dedicated `SandboxScript` started in het-group 1 (its own node)

Primary question: does the two-worker setup (nano + gpt) improve accuracy over
single-worker (r15)?

## Results

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent, OCI A100 | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r12 | single-agent, GB300 (no fp8) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| r15 | nano worker, GB300 (no fp8, FLASH_ATTN, sandbox fix) | 20.80% ± 2.39% | 19.50% | 44.00% | 5,915 | 3,499 |
| r16 | nano + gpt workers, GB300 (no fp8, FLASH_ATTN, sandbox fix) | **24.40% ± 3.71%** | **20.20%** | **56.00%** | 7,548 | 2,694 |

### Subject breakdown (r16)

| Subject | pass@1 | pass@5 | n problems |
|---------|--------|--------|------------|
| Physics | 22.00% ± 4.00% | 56.00% | 50 |
| Chemistry | 28.50% ± 6.52% | 60.00% | 40 |
| Biology | 20.00% ± 10.00% | 40.00% | 10 |

## Analysis

### GPT worker partially rescued by not having thinking enabled

r16 (24.4%) modestly outperforms r15 (20.8%) because the GPT-OSS-120B worker has
`tokens_to_generate=32768` but **no** `enable_thinking=True`. Without a thinking phase,
32k tokens is sufficient for the text response, so `choice.message.content` is non-empty.
The orchestrator could use GPT's answers on the subset of problems where the nano worker
returned empty. This explains the +3.6 pp gain in pass@1 and the larger +12 pp gain in
pass@5 (56% vs 44%).

The nano worker still has the same token-exhaustion bug as r15 — its responses are still
mostly empty. Only the GPT worker's contributions lifted the score.

### Biology avg_tokens anomaly

Biology avg_tokens in r16 is 354 (vs 3,008 in r15). This is likely because GPT's direct
text answers to biology problems are short (no code needed) and the orchestrator quickly
synthesized them without retrying the nano worker. Physics/Chemistry avg_tokens remains
high (7,087 / 9,923) — code execution is needed for those.

### gen_seconds improvement

r16 completed faster than r15 (2,694 s vs 3,499 s) despite two workers. The GPT worker
provided answers for many problems without the orchestrator waiting through nano worker
retries up to `max_tool_calls=15`.

### Nano worker bug: same as r15

The nano worker token-exhaustion bug was NOT fixed in r16 — the fix landed in
`NANO_WORKER_EXTRA_ARGS` (increasing `tokens_to_generate` to 131072) after r16 was
submitted. r18 will be the first valid two-worker run.

### Two-worker effect: cannot be measured yet

The true benefit of the two-worker setup (if any) is still unknown because the nano
worker was broken in both r15 and r16. Valid comparison requires:
- r17: single nano worker with `tokens_to_generate=131072` (fix applied)
- r18: nano + gpt workers with fix applied

## Next steps

- r17: re-run r15 config with the token budget fix → first valid single-worker result
- r18: re-run r16 config with the token budget fix → first valid two-worker result
- Compare r17 vs r18 to quantify the two-worker benefit
- Compare r17 vs r12 to isolate FLASH_ATTN effect (if any) on multi-agent accuracy
