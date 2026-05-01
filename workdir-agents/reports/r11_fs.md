# Experiment r11 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r11 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Single-agent (PythonTool) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (loaded from HF cache copied from OCI) |
| Cluster | aws-cmh (GB300, 4 GPUs/node, fp8 kv-cache, vLLM auto-backend) |
| Key config changes | Per-cluster model_path + reasoning_parser_path in CLUSTER_PARAMS; worker enable_thinking=False; max_tool_calls=15 |
| Commit | `fdc862e1 feat: per-cluster model path + reasoning parser path in CLUSTER_PARAMS` |
| SLURM generation jobs | 91655 (rs0), 91656 (rs3), 91657 (rs2), 91658 (rs1), 91659 (rs4) |
| SLURM eval jobs | 91715–91720 (judge 91715–91719, summarize 91720) |
| Previous failed attempts | 90541–90545 (model path), 91199–91203 (FLASH_ATTN+fp8) |

## Results

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent (thinking), OCI A100 | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r6  | multi-agent (worker crashed), OCI A100 | 29.40% ± 5.59% | 23.70% | 53.00% | 66,893 | 4,843 |
| r7  | multi-agent (killed — timeout) | — | — | — | — | — |
| r8  | multi-agent (killed — tokenizer crash + KeyError) | — | — | — | — | — |
| r9  | multi-agent (Tier 1 compaction; bad KeyError msg) | TBD | TBD | TBD | TBD | TBD |
| r10 | multi-agent (fixed KeyError tool listing) | TBD | TBD | TBD | TBD | TBD |
| r11 | single-agent, aws-cmh GB300, fixes applied | 34.80% ± 3.05% | 30.20% | 57.00% | 55,371 | 903 |

### Subject breakdown (r11)

| Subject | pass@1 | majority@5 | pass@5 | n problems |
|---------|--------|------------|--------|------------|
| Physics | 40.00% | 35.20% | 60.00% | 50 |
| Chemistry | 33.00% | 27.50% | 60.00% | 40 |
| Biology | 16.00% | 16.00% | 30.00% | 10 |

## Analysis

### Tool usage pattern

66.4% of entries (332/500) used **zero tool calls** — the model answered directly without running any Python. Tool-using entries outperformed non-tool entries (41.7% vs 31.3% correct), confirming the tool adds value when used. However, the model's reluctance to use tools is the main drag on accuracy.

12 entries hit `max_tool_calls=15` and scored 0% — runaway loops where the model kept calling the tool without converging. These entries were likely ill-posed or numerical problems that triggered repeated failed execution attempts.

134 of 500 generations were empty (blank output); these are always scored incorrect.

### Comparison to r5 (direct generation on OCI)

r11 is significantly worse than r5 on all accuracy metrics despite sharing the same model and single-agent configuration:

| Metric | r5 (OCI, thinking) | r11 (aws-cmh, thinking) | Delta |
|--------|---------------------|--------------------------|-------|
| pass@1 | 41.00% | 34.80% | **−6.2 pp** |
| majority@5 | 40.10% | 30.20% | **−9.9 pp** |
| pass@5 | 63.00% | 57.00% | **−6.0 pp** |
| avg_tokens | 50,296 | 55,371 | +5,075 |
| gen_seconds | 1,891 | 903 | **2.1× faster** |

The accuracy drop may be caused by:
1. **fp8 kv-cache quantization** (new in r11): GB300 enables `--kv-cache-dtype fp8` which OCI A100 cannot. Quantization artifacts could degrade quality on complex multi-step problems.
2. **vLLM backend difference**: OCI used the default backend; GB300 auto-selects FlashInfer. Different attention kernel, slightly different numerics.
3. **HF cache model load path**: Hub ID triggers an extra cache-lookup indirection vs a direct local path. Unlikely to affect weights but worth noting.
4. **Higher avg_tokens with lower accuracy**: The model is generating longer outputs on GB300 but arriving at worse answers — consistent with more "thinking" that goes off-track.

### Reasoning parser

All 500 entries have non-empty `reasoning_content` — the `nano_v3` parser worked correctly. Avg reasoning length was 634 chars, with an interesting inversion: **incorrect entries had longer reasoning** (732 chars) than correct ones (450 chars), suggesting the model overthinks on hard problems it ultimately fails.

## Notes

- First run on aws-cmh (GB300). Model loaded via HF cache (`HF_HOME=/alaptev/hf-cache`) with Hub ID `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- `--attention-backend FLASH_ATTN` was removed from `_H100_SERVER_ARGS` (incompatible with fp8 kv-cache in vLLM 0.18.x); vLLM auto-selected FlashInfer.
- `nano_v3_reasoning_parser.py` copied from OCI (`igitman/llm/hf_models/`) to `/alaptev/reasoning_parsers/` on aws-cmh.
- `o200k_base.tiktoken` downloaded to `/alaptev/tiktoken-encodings/` (required by `openai_harmony` for gpt-oss-120b judge).
- GB300 is 2.1× faster than A100 (903s vs 1,891s generation time for same 5×100 seeds).

## Next steps

- Investigate fp8 kv-cache impact: run a comparison without `--kv-cache-dtype fp8` on GB300 to isolate the accuracy drop.
- Investigate why 66.4% of entries skip tools entirely — consider a system-prompt nudge or tool-choice forcing.
- Try multi-agent on GB300 (now that the cluster is fully set up).
