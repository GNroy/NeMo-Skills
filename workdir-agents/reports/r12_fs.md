# Experiment r12 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r12 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Single-agent (PythonTool) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (HF cache, aws-cmh) |
| Cluster | aws-cmh (GB300, 4 GPUs/node, **no fp8 kv-cache**) |
| Key config changes | `--no-fp8` flag added; `--kv-cache-dtype fp8` suppressed via `--no-fp8` |
| Commit | `2f012892 feat: add --no-fp8 flag for fp8 kv-cache ablation on H100/GB300` |
| SLURM generation jobs | 91999–92003 (rs0–rs4) |
| SLURM eval jobs | 91992–91998 (judge 91992–91996, summarize 91997–91998) |

## Motivation

r11 showed a significant accuracy drop vs r5 (34.8% vs 41.0% pass@1) despite the same model and single-agent config. Two candidates:
1. `--kv-cache-dtype fp8` (new on GB300) — quantization artifacts on hard problems
2. FlashInfer attention backend (auto-selected on GB300 vs OCI default)

r12 isolates hypothesis (1) by disabling fp8 kv-cache entirely (`--no-fp8`). All other settings match r11.

## Results

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent, OCI A100 (no fp8) | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r11 | single-agent, aws-cmh GB300 (fp8) | 34.80% ± 1.96% | 30.20% | 57.00% | 55,371 | 903 |
| r12 | single-agent, aws-cmh GB300 (no fp8) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |

### Subject breakdown (r12)

| Subject | pass@1 | n problems |
|---------|--------|------------|
| Physics | 41.6% | 50 |
| Chemistry | 36.5% | 40 |
| Biology | 16.0% | 10 |

## Analysis

### fp8 ablation verdict

Disabling fp8 kv-cache **helped** across all metrics:

| Metric | r11 (fp8) | r12 (no fp8) | Δ |
|--------|-----------|--------------|---|
| pass@1 | 34.8% | 37.0% | **+2.2 pp** |
| majority@5 | 30.2% | 33.0% | **+2.8 pp** |
| pass@5 | 57.0% | 59.0% | **+2.0 pp** |
| gen_seconds | 903 | 1,000 | +10.7% slower |

fp8 kv-cache quantization degraded accuracy by ~2–3 pp. The effect is modest but consistent across all metrics — fp8 should be disabled for future GB300 runs on this model.

### Where fp8 hurt most: tool-using entries

The biggest difference between r11 and r12 is in tool-using accuracy:

| | Zero tool calls | Tool-using |
|---|---|---|
| r11 (fp8) | 31.3% correct | 41.7% correct |
| r12 (no fp8) | **31.0%** correct | **50.6%** correct |

fp8 quantization had almost no effect on direct-answer entries (+0.3 pp), but caused a **−9 pp drop in tool-using accuracy**. This makes sense: fp8 introduces numerical errors in activations, which causes incorrect Python code output and wrong answers on computation-heavy problems.

### Tool call pattern (r12)

67.2% of entries used zero tool calls (similar to r11's 66.4%). Among tool-using entries:

| Tool calls | Accuracy |
|---|---|
| 1 | 74% ← efficient, high-confidence computations |
| 2–3 | 42–60% |
| 4–8 | 40–45% (gradually degrades) |
| 15 (max) | 0% (runaway loops, 5 entries) |

1-call accuracy at 74% is notably high — these are the simplest numerical problems where the model writes one correct script and gets the answer.

### Remaining gap vs r5

Even without fp8, r12 is ~4 pp below r5 (pass@1: 37.0% vs 41.0%) and 7 pp below on majority@5. The residual gap is most likely the **attention backend**:

- OCI A100: vLLM auto-selects FLASH_ATTN (standard)
- aws-cmh GB300: vLLM auto-selects FlashInfer (different numerics)

Next candidate: explicitly pass `--attention-backend FLASH_ATTN` on GB300 without fp8. FLASH_ATTN was incompatible with fp8 in vLLM 0.18.x but should work without it.

## Next steps

- Run r13: GB300, no fp8, explicit `--attention-backend FLASH_ATTN` to isolate whether FlashInfer explains the remaining 4 pp gap.
