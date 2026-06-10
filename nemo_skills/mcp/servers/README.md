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
| `nemo_skills.mcp.servers.agentic.benchmark_tool:BenchmarkTool` | `load_benchmark` (ids manifest) / `get_problem` (one problem, no answers) — the answer-hiding trust boundary ([batch_solve](#batch_solve--the-benchmark-trust-boundary-p2)). |

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
- `get_problem(id)` → **worker side**. Returns exactly `{id, prompt, modality}`
  and nothing else — never `expected_answer` / `reference_solution` /
  `verifier_metadata`. The return is *constructed* by a whitelist, so a new
  leaky column can't slip through; a tripwire re-checks against a denylist.

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

### Adding a new wrapped tool

1. Implement a `Tool` subclass in `nemo_skills/mcp/servers/<category>/<your>_tool.py`.
2. Make sure `default_config()`, `configure(...)`, `list_tools()`, and
   `execute(...)` all work without needing a `ToolManager` around it.
3. Add an entry to the table above.  No code changes anywhere else;
   `ns-mcp-serve` discovers the class via dotted-path lookup.

The Phase 2 design doc and Phase 0/1 context for this wrapper live in
`NeMo-Skills/workdir-agents/hermes_integration_plan.md`.
