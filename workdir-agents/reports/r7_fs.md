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

**Killed early by user — no accuracy data.**

## Analysis

Two new bugs surfaced after the r6 fixes landed:

**Bug 1 — Worker HTTP timeout (100% of problems)**
The orchestrator processes 100 problems with `max_concurrent_requests=512`,
firing `call_solver_async` for all of them within seconds.  The worker server
semaphore is 64, so 36 requests queue immediately.  Each worker call had
`tokens_to_generate=None` + `enable_thinking=True`, meaning up to 131k tokens
(~5 min) per call on the shared vLLM.  Queue wait + generation time routinely
exceeded the 600s HTTP timeout.  Log evidence: 543 timeout errors in rs0 alone
within the first 10 minutes; tqdm showed `2/100 [09:36<8:42:49, 320s/it]`.

**Bug 2 — `finish_reason` serialization**
`ToolCallingWrapper.generate_async()` accumulates `finish_reason` per-turn as a
list (e.g. `['tool_calls', 'tool_calls', 'stop']`) but never flattens it.
`AgentTaskResponse.finish_reason: str | None` caused a pydantic validation error
on every completed worker call that had run more than one turn.

## Next steps → r8

Fixes committed as `a38bf993`:
1. `tool_call.py`: flatten `finish_reason` to last element
2. `WORKER_EXTRA_ARGS`: `++inference.tokens_to_generate=32768` (caps per-call latency)
3. Orchestrator: `++tool_overrides.CallAgentTool.timeout_s=1200`
