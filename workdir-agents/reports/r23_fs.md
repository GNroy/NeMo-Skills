# Experiment r23 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r23 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + nano worker (co-located) + gpt-oss-120b worker (own server + own sandbox) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` orchestrator + nano worker; `/hf_models/gpt-oss-120b` gpt worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, fp8, FlashInfer backend) |
| Key config changes | `expose_blocking_calls=False` (hide blocking call_{name} from schema); async description updated to "Call EXACTLY ONCE — pass full problem" |
| Commit | `70844588 fix: hide blocking call_{name} variants from tool list; r21/r22 results + r23 stub` |
| SLURM generation jobs | 134541, 134543, 134545, 134547, 134549 (rs0–rs4, het-jobs +0/+1) |
| SLURM eval jobs | 134551–134555 (judge rs0–rs4), 134556 (summarize) |

## Motivation

r21 and r22 both regressed from r20 (19.2% and 15.2% vs 28.8%), caused by a new
failure mode introduced when collect_results was hidden:

1. **Blocking tools exposed** — Both `call_{name}` (blocking) and `call_{name}_async`
   appeared in the tool list. Despite an explicit prompt ban, the model made 923
   blocking calls/500 records. Text-level bans are ineffective for Nano-30B.

2. **Async description invited sub-task decomposition** — `call_{name}_async` was
   described as "Use for independent sub-tasks." The model interpreted this as:
   decompose the problem into steps and call GPT for each. Avg 6.5–7.8 GPT async
   calls per session; 46–54% of sessions hit max_tool_calls=15.

r23 applies the schema-removal fix: adds `expose_blocking_calls=False` to
`CallAgentTool` default config, and updates the async description to
"Call this tool EXACTLY ONCE — pass the full, original problem statement and let
the agent handle the entire solution."

## Results

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| rec1 | non-agentic, GB300 fp8, t=1.0 | 48.80% ± 3.03% | 49.90% | 67.00% | 40,952 | 728 |
| r20 | nano + gpt workers, t=0.6 | 28.80% ± 2.28% | 29.20% | 58.00% | 3,232 | 4,098 |
| r21 | fp8 + no collect_results, t=0.6 | 19.20% ± 5.93% | 13.30% | 46.00% | 4,530 | 4,638 |
| r22 | fp8 + no collect_results, t=1.0 | 15.20% ± 3.27% | 7.60% | 37.00% | 5,773 | 3,435 |
| r23 | fp8 + no blocking + ONCE description, t=0.6 | **16.20% ± 2.05%** | **13.30%** | **39.00%** | **7,157** | **1,390** |

### Subject breakdown

| Subject | pass@1 | majority@5 | pass@5 | avg_tokens | n problems |
|---------|--------|------------|--------|------------|------------|
| Physics | 11.20% ± 2.68% | 4.80% | 34.00% | 10,283 | 50 |
| Chemistry | 21.00% ± 5.48% | 20.75% | 45.00% | 3,519 | 40 |
| Biology | 22.00% ± 8.37% | 26.00% | 40.00% | 6,074 | 10 |

### Worker behavior (diagnostic)

| Metric | r21 | r23 | Change |
|--------|-----|-----|--------|
| Records analyzed | 500 | 500 | — |
| Empty generations | 241 / 500 (48.2%) | 259 / 500 (51.8%) | +3.6 pp |
| Hit max_tool_calls=15 | 231 / 500 (46.2%) | 245 / 500 (49.0%) | +2.8 pp |
| Called both workers | 361 / 500 (72.2%) | 488 / 500 (97.6%) | **+25.4 pp** |
| collect_results calls | 15 | 9 | −6 |
| Blocking tool calls | 923 | **0** | **−923** |
| Avg GPT async calls/record | 6.49 | 10.04 | +3.55 |
| Avg nano async calls/record | 2.38 | 6.81 | +4.43 |
| Hallucinated tool names | ~0 | 52 | new |
| gen_time median | 165s | 149s | −16s |
| gen_time max | 4,633s | 1,246s | **−3,387s** |

Tool call distribution: {0:8, 1:4, 2:47, 3:12, 4:19, 5:24, 6:26, 7:25, 8:25, 9:11, 10:16, 11:12, 12:9, 13:5, 14:20, 15:237}

## Analysis

### Blocking tools fix confirmed: 0 blocking calls, gen_time max −3387s

Hiding blocking tools from the schema completely eliminated blocking calls (923 → 0).
The maximum session time dropped from 4,633s to 1,246s. This is the biggest win in r23 —
the 1,200s nano blocking timeouts are gone. gen_seconds dropped from 4,638 to 1,390.

### Both workers now called in 97.6% of sessions (up from 72.2%)

With only async tools visible and both having identical descriptions, the model calls
both workers nearly every session. This is correct protocol.

### Core failure mode unchanged: sub-task decomposition still exhausts budget

Despite hiding blocking tools and updating the description to "EXACTLY ONCE," the model
still calls each worker 6–10 times per session. Average: 10.04 GPT async + 6.81 nano
async = 16.85 async calls per session, immediately exhausting max_tool_calls=15 in 49%
of sessions (r21: 46.2%). The "EXACTLY ONCE" language had no behavioral effect.

The decomposition pattern in bad sessions now involves BOTH workers called in parallel
per sub-step (multi-tool-call turns), not just GPT. A typical bad trace:

```
Turn 2: [nano_async(sub-problem-1), gpt_async(sub-problem-1)] → 2 pending tickets
Turn 5: [nano_async(sub-problem-2), gpt_async(sub-problem-2)] → 2 more pending tickets
Turn 8: auto-inject: nano result = "" (timeout)
Turn 9–19: [gpt_async(full problem)] × 6 → retries after empty nano
→ tool_call_limit_reached
```

### tc=2 "good" sessions are mostly the model answering from its own knowledge

47 sessions (9.4%) completed with tc=2, apparently following the intended protocol.
But of these, 39/47 (83%) answered BEFORE any auto-inject arrived — the model fires
both async calls and immediately writes a final answer from its own knowledge without
waiting for worker results. Only 6/47 (13%) of tc=2 sessions actually used worker output.

This means the multi-agent setup has near-zero impact on accuracy: the model either
answers independently (ignoring workers) or exhausts its tool budget in retry loops.

### Hallucinated tool names: 52 calls to non-existent agents

The model invented 52 calls to agents that don't exist in the schema:
`call_eco_domains_researcher_async` (14), `call_bench_helper_async` (5),
`call_chemistry_expert_async` (4), `call_umap_solver_async` (2), etc. These are
nonsense names (the model is inventing domain specialists). They fail silently but
consume tool call budget.

### Root cause: model has no "stop delegating, collect results" operation

The collect_results removal was the wrong fix. In r20, collect_results served as an
explicit "I'm done delegating, give me all results now" signal. The model understood
this 3-step protocol: async → async → collect. Without it, the model has no completion
signal and loops indefinitely, calling workers for sub-tasks hoping to get results.

r20 with collect_results: 3,232 avg tokens, 28.8% pass@1, 15.8% max_tc.
r23 without collect_results: 7,157 avg tokens, 16.2% pass@1, 49.0% max_tc.

The collect_results overhead in r20 (1.84 extra calls/session) is far less harmful
than the unbounded retry loop it prevented.

## Next steps

### Re-enable collect_results (r24)

The fix: add `++tool_overrides.CallAgentTool.expose_collect_results=true` to the
orchestrator's extra_args in `run_nano_physics_agent.py`. Keep `expose_blocking_calls=False`.

Tool list for r24:
- `call_nano_solver_0_async` (fire nano)
- `call_gpt_solver_async` (fire gpt)
- `collect_results` (explicit wait + retrieve all results)

Expected behavior:
- Model calls async workers (2 calls) → collect_results([nano_ticket, gpt_ticket]) → synthesize
- TC: ~3-4 per session (matching r20's pattern)
- max_tc rate: ~15% (matching r20)
- No blocking timeouts (blocking tools still hidden)
- pass@1: potentially ≥ 28.8% (r20) — same protocol, faster nano execution

Commit: `<pending r24 commit>`
