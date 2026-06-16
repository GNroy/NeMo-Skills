# NeMo-Skills MCP servers

This directory hosts the MCP servers NeMo-Skills ships out of the box.
They fall into two groups, distinguished by whether the file itself is
runnable as an MCP server or whether it only implements the in-process
`Tool` interface (`nemo_skills.mcp.tool_manager.Tool`).

## Group A: native FastMCP entrypoints

These modules already register their own `FastMCP` server and expose a
`main()` plus `if __name__ == "__main__": main()` block.  Run them
directly via `python -m`:

| Module | Run command |
|---|---|
| `nemo_skills.mcp.servers.python_tool` | `python -m nemo_skills.mcp.servers.python_tool` |
| `nemo_skills.mcp.servers.tavily_search_tool` | `python -m nemo_skills.mcp.servers.tavily_search_tool` |
| `nemo_skills.mcp.servers.exa_tool` | `python -m nemo_skills.mcp.servers.exa_tool` |

Hermes consumes these via the standard `mcp_servers:` config:

```yaml
# ~/.hermes/config.yaml
mcp_servers:
  ns_python:
    command: "python"
    args: ["-m", "nemo_skills.mcp.servers.python_tool"]
  ns_tavily:
    command: "python"
    args: ["-m", "nemo_skills.mcp.servers.tavily_search_tool"]
    env:
      TAVILY_API_KEY: "..."
```

## Group B: in-process `Tool` classes (wrapped by `ns-mcp-serve`)

These modules expose a `Tool` subclass but do not host their own MCP
server.  Use the generic stdio wrapper `nemo_skills.mcp.stdio_serve`
(console script `ns-mcp-serve`) to spawn each as an MCP stdio server:

| Module:Class | Description |
|---|---|
| `nemo_skills.mcp.servers.chemistry.periodictable_tool:PeriodictableTool` | Element / isotope lookups (no external deps). |
| `nemo_skills.mcp.servers.physics.coolprop_tool:CoolPropTool` | Thermo-physical properties via CoolProp. |
| `nemo_skills.mcp.servers.physics.particle_tool:ParticleTool` | Particle Data Group lookups. |
| `nemo_skills.mcp.servers.physics.radioactivedecay_tool:RadioactivedecayTool` | Decay chain calculations. |
| `nemo_skills.mcp.servers.web.arxiv_tool:ArxivSearchTool` | arXiv search + paper metadata. |
| `nemo_skills.mcp.servers.web.wikipedia_tool:WikipediaSearchTool` | Wikipedia search + article fetch. |
| `nemo_skills.mcp.servers.agentic.worklog_tool:WorklogTool` | Clock-in/clock-off work tracking; writes a markdown report per task ([agentic loop](#agentic-loop-tools-sci-548)). |
| `nemo_skills.mcp.servers.agentic.benchmark_tool:BenchmarkTool` | `load_benchmark` (ids manifest) / `plan_batch` (staged work-list handle, large-batch) / `get_problem` (one problem, no answers) — the answer-hiding trust boundary ([batch_solve](#batch_solve--the-benchmark-trust-boundary-p2)). |

Hermes consumes them uniformly:

```yaml
# ~/.hermes/config.yaml
mcp_servers:
  ns_periodictable:
    command: "ns-mcp-serve"
    args: ["nemo_skills.mcp.servers.chemistry.periodictable_tool:PeriodictableTool"]

  ns_arxiv:
    command: "ns-mcp-serve"
    args: ["nemo_skills.mcp.servers.web.arxiv_tool:ArxivSearchTool"]
```

Tool names emerge on the Hermes side as `<server_name>.<raw_tool>` (the
server name is the YAML key — `ns_periodictable`, `ns_arxiv`, …) so they
never collide with Hermes built-ins like `execute_code` or with each
other.

### Passing config overrides

`ns-mcp-serve` forwards a JSON object to `Tool.configure(...)`:

```yaml
mcp_servers:
  ns_arxiv:
    command: "ns-mcp-serve"
    args:
      - nemo_skills.mcp.servers.web.arxiv_tool:ArxivSearchTool
      - --overrides
      - '{"max_results": 5}'
```

### Custom server name (debug only)

By default the MCP server name advertised over stdio matches the `Tool`
class name.  Pass `--server-name foo` to override; rarely needed because
Hermes uses its own YAML key for prefixing.

### Agentic-loop tools (SCI-548)

`nemo_skills.mcp.servers.agentic.*` hosts the tools for the self-improving
Hermes swarm. **P1 ships `worklog`** — a clock-in / clock-off "punch clock":
the agent calls `clock_in(task_id, task_description)` before a unit of work and
`clock_off(task_id, status, report)` after, and the server writes one markdown
report per task. `reflect` (P3) reads those files off disk, so they are
persisted, not just returned inline.

Wire it with run-scoped `--overrides` (or the matching env vars), so the
launcher controls where reports land and how the run is labelled:

```yaml
mcp_servers:
  worklog:
    command: ns-mcp-serve
    args:
      - nemo_skills.mcp.servers.agentic.worklog_tool:WorklogTool
      - --overrides
      - '{"worklog_dir": "/path/to/run_artifacts", "run_id": "20260608T1200Z_run1", "agent_id": "orchestrator", "subdir": ""}'
```

The agent then sees the tools as `mcp_worklog_clock_in` / `mcp_worklog_clock_off`
— Hermes sanitises and prefixes each MCP tool as `mcp_<server>_<tool>`. They live
in toolset `mcp-worklog` (with `worklog` registered as an alias), so add
`mcp-worklog` (or `worklog`) to the manifest's `enabled_toolsets` whitelist or the
tools stay hidden. Reports are written to:

```
<worklog_dir>/<run_id>/<subdir>/<task_id>.md   # subdir defaults to "" (flat)
```

Each file is YAML frontmatter (`run_id, agent_id, task_id, status, closed_by,
started_at, ended_at, elapsed_s`) followed by the agent's markdown report.
`clock_off` also accepts an optional structured `tools_used`
(`[{name, helped, note}]`) self-report; when given it lands in the frontmatter
and `worklog_enrich` (below) diffs it against the trace's observed tool calls.

| Override / env | Default | Purpose |
|---|---|---|
| `worklog_dir` / `NS_WORKLOG_DIR` | `worklogs` | Root for report files. |
| `run_id` / `NS_WORKLOG_RUN_ID` | `run-<UTC ts>` | Run label = report subdir. **Set explicitly when several worklog servers must share one run dir.** |
| `agent_id` / `NS_WORKLOG_AGENT_ID` | `agent` | Who is logging (P2 workers pass their own per `clock_in`). |
| `subdir` / `NS_WORKLOG_SUBDIR` | `""` | Relative dir under the run (P2: `workers/chunk_<k>`). |

**Guaranteed close** (defence in depth — a `clock_in` never dangles):

1. **Eager stub (survives `SIGKILL`).** `clock_in` writes the report file
   immediately as `status: in_progress`, `closed_by: pending`. If the worker is
   killed abruptly before `clock_off` — e.g. the MCP SDK spawns the stdio server
   with `setsid()`, so an orphaned subprocess is `SIGKILL`ed without a catchable
   signal when the container is torn down — the stub still records that the task
   started. This is the only layer that survives `SIGKILL`.
2. **Shutdown sweep.** On a *graceful* exit the server flushes still-open timers
   to `status: error`, `closed_by: shutdown_sweep` — via `atexit` (normal/EOF
   exit) and a `SIGTERM`/`SIGHUP` handler (walltime kills).
3. (P2's `batch_solve` adds an orchestrator-side per-worker timeout backfill.)

A record progresses on disk: `in_progress` → `completed`/… (clock_off) or
`error`/`shutdown_sweep` (sweep); each step rewrites the file atomically.

#### Worklog enrichment from traces (`worklog_enrich`)

Worklogs are *self-reported*; the JSONL trajectory traces NeMo-Gym writes
(`<output_dir>/<trace_dir_subpath>/<agent>/<session_id>.jsonl`) are *ground
truth*. `nemo_skills.mcp.servers.agentic.worklog_enrich` joins them: for each
session it derives the `task_id` from that trace's own `mcp_worklog_clock_in`
event, reconstructs a factual tools-used table (`tool_start`/`tool_complete`),
and writes a `## Tools used (observed)` block onto the matching `<task_id>.md`
report plus a `<task_id>.tools.json` sidecar with a claim-vs-observed diff.

The join needs **per-agent trace files** — orchestrator *and* delegate
children. NeMo-Gym exposes a per-session `_trace_callback_factory` on the agent
and hermes-agent's `delegate_tool` binds each child its own session-scoped trace
callbacks, so every worker writes a standalone `<session_id>.jsonl` (no
emit-time session juggling). Run it standalone over any run, or via
`ns hermes_agent_rollouts --enrich_worklogs` (a gym-sentinel-gated post-step):

```
python -m nemo_skills.mcp.servers.agentic.worklog_enrich \
    --trace-dir <output_dir>/traces --worklog-dir <output_dir>/worklogs
```

### `batch_solve` & the benchmark trust boundary (P2)

The self-improving swarm's `batch_solve` is **not new code** — it is Hermes'
native `delegate_task` (toolset `delegation`): the orchestrator fans out one
worker subagent per problem, and the returned per-child results *are* the
status roster. The new NS code P2 adds is the **`benchmark` trust-boundary MCP
server** — the boundary between raw benchmark data (which carries answers and
grading config) and what any agent may see. Two tools, one server:

- `load_benchmark(benchmark, limit, shuffle, seed, shard)` → **orchestrator
  side**. Returns an ids-only manifest `{count, total, ids:[...]}` — no problem
  text, no answers. Keeps the orchestrator's context `O(N ids)` so it scales
  past a 128k window (workers fetch their own problems).
- `plan_batch(benchmark, goal_template, limit, shuffle, seed, shard)` →
  **orchestrator side, large-batch path (the one to use at HLE scale)**. Expands
  the (sharded/sampled) benchmark into a staged JSONL *work-list file* — one
  `delegate_task` goal per id, built from `goal_template` by substituting `{id}`
  (and optionally `{problem}`/`{modality}`, inlined server-side from the same
  whitelist as `get_problem`, for tool-less workers) — and returns a small
  `{handle, count, ...}` receipt: **not** the ids, never answers. The
  orchestrator hands `handle` straight to
  `delegate_task(tasks_source=handle, max_in_flight=N)` and dispatches the WHOLE
  benchmark in **one** call, so its context is `O(1)` in the batch size — it
  never generates (or holds) thousands of goal strings. This is what lets a
  single orchestrator session cover all 2158 HLE problems without manual
  sharding. Reuses `load_benchmark`'s id-derivation + sharding so plan/grade agree.
- `get_problem(id)` → **worker side**. Returns exactly `{id, prompt, modality}`
  and nothing else — never `expected_answer` / `reference_solution` /
  `verifier_metadata`. The return is *constructed* by a whitelist, so a new
  leaky column can't slip through; a tripwire re-checks against a denylist.
  (When `plan_batch` inlined `{problem}` into the goal, tool-less workers don't
  even need this — see `force_child_toolsets` below.)

Benchmark rows across Gym/NS are heterogeneous, so `id` and `prompt` are
auto-detected (`id`/`uuid`/`hash_id`/… → fallback `row-<index>`; direct
`problem`/`question` key → else the user turns of
`responses_create_params.input`). `modality` is `"text"` unless an image/audio
content part is present.

Wire it as an `mcp_servers:` entry alongside `worklog`. The orchestrator
manifest enables `delegation` + `mcp-benchmark` + `mcp-worklog`; delegate
children inherit the MCP toolsets automatically (`delegation.inherit_mcp_toolsets`
default True) and have `delegation`/`memory` stripped
(`DELEGATE_BLOCKED_TOOLS`) — so workers structurally can't delegate or write
memory, only fetch + solve + worklog. Example (see
`NeMo-Skills/workdir-agents/validation/batch_solve_qwen_manifest.yaml` for the
full rollout manifest + driving prompt):

```yaml
mcp_servers:
  benchmark:
    command: /usr/bin/python3       # or ns-mcp-serve
    args: [-m, nemo_skills.mcp.stdio_serve, "nemo_skills.mcp.servers.agentic.benchmark_tool:BenchmarkTool"]
    env:
      NS_BENCHMARK_PATH: /path/to/prepared_benchmark.jsonl   # lazy-load fallback for workers
```

| Override / env | Default | Purpose |
|---|---|---|
| `benchmark_root` / `NS_BENCHMARK_ROOT` | `None` | Dir to resolve a bare benchmark *name* → `<root>/<name>.jsonl`. |
| `benchmark_path` / `NS_BENCHMARK_PATH` | `None` | A single staged file so `get_problem` resolves without a prior `load_benchmark`. |
| `id_keys` | `["id","uuid","hash_id","_id","problem_id","qid"]` | Row keys tried, in order, for the join id. |
| `prompt_keys` | `["problem","question","prompt","text"]` | Row keys tried, in order, for the prompt. |

**Offline grading joins by `id` (no new grader).** The answer fields stay in the
staged JSONL, read only by the offline grader *after* the session ends. The
join key is `benchmark_tool.derive_problem_id(row, index)` — exported and used
by both the tool and any grader, so the answer↔response join always agrees.
Existing Gym `verify` / `judge_rollouts.py` grade as today; pass-rate
measurement across warm/cold iterations is P4.

#### Running batch_solve end-to-end (validated recipe)

Full HLE-2158 in **one** orchestrator job (no manual shards), validated
2026-06-16: **20.76% (448/2158)** — matches the prior 22-shard baseline (20.7%),
so the single-run path is a faithful, cheaper replacement.

1. **Stage the benchmark** in Gym format with answers in `verifier_metadata`
   (the trust-boundary tool hides them; the offline grader reads them). The same
   file feeds both the workers (no-answer view) and grading.
2. **Orchestrator prompt** (the rollout `input_file`) instructs, each tool EXACTLY
   once: `clock_in` → `plan_batch(benchmark, goal_template=<…{id}…>)` →
   `delegate_task(tasks_source=<handle>, max_in_flight=56)` → `clock_off`. Keep
   the goal_template literal with `{id}` (and `{problem}` for tool-less workers).
3. **Launch** (`NeMo-Skills/workdir-agents/validation/batch_solve_hle_newdeleg_full_k8_manifest.yaml`
   is the K=8 reference):

   ```bash
   export NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1
   export NEMO_SKILLS_SANDBOX_HOST='${SLURM_MASTER_NODE_HET_GROUP_0:-localhost}'
   ns hermes_agent_rollouts --cluster aws-cmh --config_dir cluster_configs \
     --agent_manifest <manifest>.yaml --input_file <orch_prompt>.jsonl \
     --output_dir <out> --expname <name> \
     --server_container .../sglang-v0.5.11.sqsh \         # FULL /lustre path (pyxis can't resolve /alaptev for an IMAGE)
     --gym_container .../nemo-skills-dc43f3e.sqsh --sandbox_container .../nemo-skills-dc43f3e.sqsh \
     --gym_path /alaptev/NeMo-Gym --hermes_agent_path /alaptev/hermes-agent \
     --no-merge_back --no-with_sandbox
   ```
   The orchestrator is one het-group (1 node); the worker pool is a second
   het-group (`server_type: sglang_router`, K nodes) that `delegate_task` reaches
   via the injected `DELEGATION_BASE_URL`. `delegate_task` returns a compact
   `batch_tally` (`status`, `completed`, `failed`, `failure_breakdown`,
   `next_step`); per-worker answers land in `worklogs/`.
4. **Grade** (deterministic join → LLM judge → tally; no answers ever touch an
   agent):
   ```bash
   python fs_grade.py build --benchmark <staged>.jsonl --worklog-dir <out>/worklogs/<run_id> --out judge_input.jsonl
   # judge/hle templates {problem}; fs_grade emits {question} -> add problem=question to each row, then:
   ns generate --cluster aws-cmh --server_type vllm --model /hf_models/gpt-oss-120b --server_gpus 4 \
     --server_container .../nemo-skills-vllm-dc43f3e.sqsh \
     --input_file judge_input.jsonl --output_dir <out>/judge \
     ++prompt_config=judge/hle ++generation_key=judgement ++add_generation_stats=False
   python fs_grade.py tally --judged <out>/judge/output.jsonl
   ```

#### hermes-agent delegation dependency

`batch_solve` runs on Hermes' native `delegate_task`, which lives in the
**hermes-agent** repo (branch `sandbox-hermes-tools`), deployed to
`/alaptev/hermes-agent` and put on the orchestrator's `PYTHONPATH` via
`--hermes_agent_path` (NOT the gym's bundled `.venv` copy, which is older). The
SCI-548 work added/fixed there, all required for the large-batch path:

- **`tasks_source` + `max_in_flight` + bounded queue + `batch_tally`** — the
  `plan_batch` handle consumer. The total count is unbounded (queued); at most
  `max_in_flight` workers run at once.
- **`_dispatch_delegate_task` forwards `tasks_source`/`max_in_flight`** — the
  single live call site (`run_agent.py`) maps the model's args → `delegate_task`;
  it must forward every schema field or the arg is silently dropped (this caused
  a `tasks_source`-ignored → blind-retry meltdown; a regression test now asserts
  every schema property is forwarded).
- **Lazy, width-bounded child construction** — build only `~max_in_flight`
  children before the first dispatch, then refill as slots free, instead of
  building all N up front. Building thousands of `AIAgent`s serially before any
  pool request left GPUs 100% idle for minutes → tripped the cluster idle-job
  reaper. Now the pool saturates immediately.
- **`delegation.force_child_toolsets`** (config, default unset) — pins every
  child's toolset exactly; `[]` makes workers **tool-less** (pure reasoning over
  an inlined `{problem}`), the no-tool-worker ablation. Honors an explicit empty
  list as "no tools" (not "inherit").
- **Structured terminal errors** — `delegate_task` errors carry
  `error_code` / `terminal` / `retryable` / `hint` / `received` (echo of the args
  the tool actually got) so the orchestrator self-corrects instead of
  blind-retrying, and a failure is reconstructable from the trace.

#### Gotchas (read before a fresh run)

- **Idle-job reaper**, not preemption: a het-job whose worker pool sits ≤1% GPU
  for 30 min is auto-`scancel`ed. Keep the pool saturated (lazy build above);
  the "other agent" runs under the **same `alaptev` uid**, so never
  `scancel`-by-`--me` — filter by jobid/expname.
- **Normal teardown looks like a cancel**: when the orchestrator finishes, nemo-run
  `scancel`s the persistent worker-pool het-group → `CANCELLED by <your-uid>`. That
  is success-then-cleanup, not a failure. Confirm via `worklogs` + the trace's
  final `batch_tally` + `clock_off`.
- **Always inspect the orchestrator trace** (`<out>/traces/scientist/sess_*.jsonl`,
  `tool_start`/`tool_complete`) to confirm exactly one `plan_batch` + one
  `delegate_task(tasks_source)` + `clock_off` — surface signals (job state,
  worklog counts) are necessary but not sufficient.
- **`.ng.jsonl` carries `verifier_metadata.expected_answer`** — the same staged
  file is both the workers' no-answer source (tool hides it) and the grader's
  answer source.

### Adding a new wrapped tool

1. Implement a `Tool` subclass in `nemo_skills/mcp/servers/<category>/<your>_tool.py`.
2. Make sure `default_config()`, `configure(...)`, `list_tools()`, and
   `execute(...)` all work without needing a `ToolManager` around it.
3. Add an entry to the table above.  No code changes anywhere else;
   `ns-mcp-serve` discovers the class via dotted-path lookup.

The Phase 2 design doc and Phase 0/1 context for this wrapper live in
`NeMo-Skills/workdir-agents/hermes_integration_plan.md`.
