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

**KILLED** — two new bugs found in early logs; killed before completion.

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | single-agent (thinking) | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r6  | multi-agent (worker crashed) | 29.40% ± 5.59% | 23.70% | 53.00% | 66,893 | 4,843 |
| r7  | multi-agent (killed — timeout) | — | — | — | — | — |
| r8  | multi-agent (killed — tokenizer crash + KeyError) | — | — | — | — | — |

## Bug analysis

**Bug 1: Worker tokenizer crash**

`agent_server.py` never passed `require_tokenizer` to `get_tool_calling_model()`.
With `++inference.tokens_to_generate=32768` (added in r8), `_count_tool_response_tokens()`
raised `RuntimeError("Cannot count tool response tokens: no tokenizer available")` on every
worker code-execution call, crashing the worker response.

Fix: `require_tokenizer=cfg.inference.tokens_to_generate is not None` added to the
`get_tool_calling_model()` call in `agent_server.py`.

**Bug 2: `KeyError: 'solver'` in orchestrator**

The orchestrator (with `enable_thinking=False`) sometimes called a tool named `"solver"`
instead of `"call_solver"` — likely confused by the `"agent": "solver"` field in
`collect_results` JSON responses. `_execute_tool_call` fell through to `except Exception`
which returned the unhelpful `"Tool execution failed."` and logged a noisy traceback.

Fix: added `except KeyError` before the catch-all in `_execute_tool_call` that returns an
actionable error message listing available tool names, so the model can self-correct.

Both fixes committed as `d6479da2` (r9).

## Notes

r8 also introduced Tier 1 context compaction (see r9 setup) but was killed before any
compaction effects could be observed.
