# Experiment r6 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r6 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + solver worker |
| Model | `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (nano-agent) |
| Key config changes | (1) `tool_choice=required` in orchestrator `extra_body` — forces first-turn tool call; (2) `orchestrator.yaml` rewritten with "Mandatory workflow" section requiring `call_*_async` as first action |
| Commit | `c332aed5 orchestrator: force first-turn tool call + mandate async delegation` |
| SLURM jobs | 9499141–9499158 |

## Results

### frontierscience-olympiad

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r5  | multi-agent (no delegation) | 41.00% ± 3.16% | 40.10% | 63.00% | 50,296 | 1,891 |
| r6  | multi-agent (forced delegation) | **29.40% ± 5.59%** | 23.70% | 53.00% | 66,893 | 4,843 |

**Δ pass@1: −11.6 pp**

## Analysis

Accuracy dropped significantly despite the orchestrator now being forced to
call workers (avg_tokens +33%, gen_seconds +156% confirm actual delegation).

Two root causes found by inspecting rs0–rs4:

| Failure mode | Frequency (per seed) |
|---|---|
| Worker crash (`endpoint_type` duplicate arg) | 55–65% of problems |
| Context exhausted in thinking (`finish_reason=length`) | 22–38% of problems |
| Successful worker call + synthesis | 0% of problems |

**Bug 1 — Worker crash**: `asdict(InferenceConfig)` includes the `endpoint_type`
field (a dataclass member).  `agent_server._handle_task()` passed it both
explicitly (`endpoint_type=EndpointType.chat`) and again via `**inference_params`,
causing `TypeError: got multiple values for keyword argument 'endpoint_type'` on
every worker invocation.  This bug was dormant in r5 because the orchestrator
never called workers.

**Bug 2 — Token exhaustion**: `enable_thinking=True` + `tokens_to_generate=131072`
allows the model to spend the entire budget in the `<think>` block.  With
`tool_choice=required` the vLLM server *would* force a tool call — but only if
the model reaches the response body before running out of tokens.  When
`finish_reason=length` fires mid-think, no tool call is ever emitted.

## Notes

- All five seeds show the same distribution — this is a deterministic infrastructure
  failure, not variance.
- The `tool_choice` first-turn reset logic in `tool_call.py` worked correctly; the
  issue is upstream (worker crash + thinking exhaustion).

## Next steps (r7)

- Fix `endpoint_type` duplicate: pop from `inference_params` in `agent_server.py`
- Disable orchestrator thinking: add `++chat_template_kwargs.enable_thinking=False`
  to orchestrator `extra_args` — the worker handles heavy reasoning; the
  orchestrator only needs to delegate and synthesize
- Commit: `b8a3e150 fix worker crash + orchestrator token exhaustion`
