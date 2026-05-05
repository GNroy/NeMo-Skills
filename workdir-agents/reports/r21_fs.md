# Experiment r21 — fs

## Setup

| Field | Value |
|-------|-------|
| Run ID | r21 |
| Benchmark | frontierscience-olympiad (fs) |
| Mode | Multi-agent: orchestrator + nano worker (co-located) + gpt-oss-120b worker (own server + own sandbox) |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` orchestrator + nano worker; `/hf_models/gpt-oss-120b` gpt worker |
| Cluster | aws-cmh (GB300, 4 GPUs/node, **fp8, FlashInfer backend**) |
| Key config changes | fp8 + FlashInfer (matching rec1); `collect_results` hidden from tool list (auto-injection only) |
| Commit | `03fd8b37 fix: hide collect_results from LLM tool list by default; r20 results + r21 stub` |
| SLURM generation jobs | 133475, 133469, 133471, 133473, 133477 (rs0–rs4, het-jobs +0/+1) |
| SLURM eval jobs | 133479–133483 (judge rs0–rs4), 133484 (summarize) |

## Motivation

r20 analysis identified two root causes:

1. **Nano worker times out 39.4% of the time** — r18/r19/r20 ran with
   `--no-fp8 --attention-backend FLASH_ATTN`, making the nano worker ~37%
   slower than its rec1 baseline (FLASH_ATTN is incompatible with fp8 kv-cache
   in vLLM 0.18.x; the correct fix — remove FLASH_ATTN, use fp8 + FlashInfer —
   was applied to the non-agentic script in commit `aa53c760` but not to the
   agentic run invocations).  With a ~1,000 s no-fp8 baseline and a 1,200 s
   timeout, ~40% of problems in the upper token tail time out.

2. **`collect_results` called in every session despite the prompt ban** — the
   model averages 1.84 explicit `collect_results` calls per session, inflating
   tool call counts from 2 (expected) to 4 (mode in r20) and making the
   auto-injection mechanism irrelevant.  Text-level bans do not work; removing
   the tool from the schema does.

r21 fixes both:
- Submit without `--no-fp8` and `--attention-backend FLASH_ATTN` → fp8 +
  FlashInfer matches rec1's ~728 s per problem, giving a 472 s margin before
  the 1,200 s timeout.
- `CallAgentTool.expose_collect_results=False` (new default) — `collect_results`
  is omitted from `list_tools()`, so the LLM cannot see or call it.  Results
  arrive only via auto-injection.

Expected outcome: tool calls settle at exactly 2 per session (call_nano_async
+ call_gpt_async), nano timeouts drop from 39.4% to near zero, and both workers
produce usable results in most sessions.  If these mechanical fixes hold,
accuracy should improve over r20 (28.8%).

## Results

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r12 | single-agent, GB300 (no fp8) | 37.00% ± 2.12% | 33.00% | 59.00% | 56,174 | 1,000 |
| rec1 | non-agentic, GB300 fp8, t=1.0 | 48.80% ± 3.03% | 49.90% | 67.00% | 40,952 | 728 |
| r19 | nano + gpt workers, delegate-once prompt | 28.20% ± 2.05% | 30.20% | 56.00% | 3,701 | 3,571 |
| r20 | nano + gpt workers, enforce-all + no-followup prompt | 28.80% ± 2.28% | 29.20% | 58.00% | 3,232 | 4,098 |
| r21 | nano + gpt workers, fp8 + no collect_results | **19.20% ± 5.93%** | **13.30%** | **46.00%** | **4,530** | **4,638** |

### Subject breakdown

| Subject | pass@1 | majority@5 | pass@5 | avg_tokens | n problems |
|---------|--------|------------|--------|------------|------------|
| Physics | 15.60% ± 8.65% | 8.60% | 44.00% | 4,528 | 50 |
| Chemistry | 21.50% ± 8.02% | 16.00% | 45.00% | 4,809 | 40 |
| Biology | 28.00% ± 4.47% | 26.00% | 60.00% | 3,430 | 10 |

### Worker behavior (diagnostic)

| Metric | r20 | r21 | Change |
|--------|-----|-----|--------|
| Records analyzed | 500 | 500 | — |
| Empty generations | 86 / 500 (17.2%) | 241 / 500 (48.2%) | **+31.0 pp** |
| Hit max_tool_calls=15 | 79 / 500 (15.8%) | 231 / 500 (46.2%) | **+30.4 pp** |
| Called both workers | 335 / 500 (67.0%) | 361 / 500 (72.2%) | +5.2 pp |
| collect_results calls (total) | 921 | 15 | **−906** |
| Blocking tool calls (total) | ~842 | 923 (626 GPT + 216 nano + 81 other) | — |
| Avg GPT async calls/record | ~1.0 | 6.49 | **+5.5** |
| Tool names (all) | gpt_async + nano_async + collect_results | call_gpt_solver_async (3,223), call_nano_solver_0_async (1,190), call_gpt_solver (626), call_nano_solver_0 (216), call_gpt_solver_0 (78), stateful_python_code_exec (22), collect_results (15) | — |

Tool call distribution: {0:9, 1:4, 2:7, 3:14, 4:63, 5:39, 6:33, 7:16, 8:22, 9:15, 10:11, 11:8, 12:11, 13:8, 14:7, 15:233}

## Analysis

### Expected fix worked partially: collect_results reduced from 921 → 15

Hiding collect_results from the schema reduced explicit polling from 921 calls to 15 (near zero). The schema removal is effective. However, removing collect_results exposed a new failure mode that dominates the results.

### New failure mode: model decomposes problems into GPT sub-tasks

Without collect_results available, the orchestrator adopted a completely different strategy: instead of calling each worker once and polling, it decomposes each problem into sequential sub-steps and calls `call_gpt_solver_async` for each sub-step (avg 6.49 calls per record vs ~1.0 in r20). A typical bad session:

```
Turn 2: call_gpt_solver_async("Solve the electrostatic boundary-value problem...")
Turn 4: call_gpt_solver_async("Derive the potential outside a grounded cylinder...")
Turn 6: call_gpt_solver_async("Compute the electric field E(r,θ)...")
Turn 8: call_gpt_solver_async("Provide the final expression...")
Turn 10: call_gpt_solver_async("Derive the potential...") [repeat]
... → tool_call_limit_reached at turn 15
```

This burns the entire 15-call budget on GPT sub-tasks before any synthesis can occur. 46.2% of sessions hit the limit (up from 15.8% in r20), and 48.2% of sessions produce an empty generation (up from 17.2%).

### Root cause: blocking tools in schema + async description invites sub-tasking

Two factors drive the multi-call behavior:

1. **Blocking tools exposed** — `call_gpt_solver` and `call_nano_solver_0` are listed alongside their async variants (4 tools total). The orchestrator prompt explicitly bans blocking calls, but the model ignores text-level prohibitions (same as collect_results in r20). Result: 923 blocking calls per 500 records (626 GPT + 216 nano blocking), many of which hit the 1,200s timeout.

2. **Async description invites sub-tasking** — the async tool description said "Use for independent sub-tasks that can run while you continue reasoning." The model interprets "independent sub-tasks" as an instruction to decompose the problem and call GPT for each piece.

The orchestrator system prompt says "Total tool calls: exactly N (one per agent)" and "blocking call_<name> are forbidden" — but text bans are ineffective for Nano-30B, as established in r20 with collect_results.

### Accuracy regressed: 19.2% vs 28.8% in r20

pass@1 dropped 9.6 pp from r20. The regression is entirely explained by the 46.2% max_tc hit rate: those sessions return empty generations and contribute 0% accuracy. The sessions that do complete (53.8%) likely perform similarly to r20, but the average is pulled down by the empty half.

### Temperature experiment (r22): made things worse

r22 ran r21 conditions + temperature=1.0. Results: 15.2% pass@1, 53.6% max_tc hit rate. Higher temperature increased the tendency to decompose and retry, making the failure mode worse. The temperature hypothesis from rec1's 48.8% baseline does not transfer to the agentic orchestrator.

## Next steps

### Root cause fix: hide blocking tool variants from schema (r23)

The same pattern applies: text-level bans are ignored; schema removal is reliable. Add `expose_blocking_calls=False` to `CallAgentTool` default config, gating `call_{name}` variants behind this flag — same mechanism as `expose_collect_results=False`.

Additionally, update the async tool description from "Use for independent sub-tasks" to "Call this tool EXACTLY ONCE — pass the full, original problem statement and let the agent handle the entire solution." This eliminates the language that invites sub-task decomposition.

Expected effect: orchestrator sees exactly 2 tools (call_nano_solver_0_async, call_gpt_solver_async), with descriptions that discourage multiple calls. Combined with the existing system prompt ("exactly N tool calls"), tool calls should settle at 2 per session.

Commit: `<pending r23 commit>`
