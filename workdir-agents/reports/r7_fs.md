# Experiment r7 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r7 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + solver worker |
| Model | `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (nano-agent) |
| Key config changes | (1) `agent_server.py`: pop `endpoint_type` from `inference_params` before `generate_async()` call — fixes worker crash; (2) orchestrator `extra_args`: add `++chat_template_kwargs.enable_thinking=False` — prevents 131k-token exhaustion in first thinking block |
| Commit | `b8a3e150 fix worker crash + orchestrator token exhaustion in multi-agent setup` |
| SLURM jobs | 9504791–9504795 (generation), eval scheduled after |

## Results

*Pending — jobs submitted 2026-04-30.*

## Expected

With both bugs fixed:
- Worker calls should succeed (0 `endpoint_type` crashes)
- Orchestrator should emit tool calls on turn 1 (thinking disabled)
- pass@1 target: recover toward or above r5 baseline (41%)
- avg_tokens and gen_seconds should stay elevated (delegation overhead)

## Notes

- Orchestrator has `tool_choice=required` (first turn forced) + `enable_thinking=False` + `orchestrator.yaml` mandate — all three levers active simultaneously
- Worker retains `enable_thinking=True` (inherits from `InferenceConfig` default); the `code_agent` system prompt guides it to use Python tools

## Next steps

After r7 results arrive:
1. Check tool call success rate (should be near 100%)
2. Check async vs blocking delegation ratio
3. Compare pass@1 against r5 baseline
4. If accuracy is lower than r5: investigate whether orchestrator's lack of thinking hurts synthesis quality; consider `enable_thinking=True` with a thinking budget cap
5. If accuracy matches or exceeds r5: extend to physics and hle-physics benchmarks
