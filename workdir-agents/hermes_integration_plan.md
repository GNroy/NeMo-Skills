# Hermes-Agent → NeMo-Skills Integration Plan

> Goal: turn `responses_api_agents/hermes_agent` (NeMo-Gym) into the runtime
> for multi-agent scientific-research systems launched from NeMo-Skills,
> with per-agent persistent memory/skills, MCP interop with NeMo-Skills tools,
> a JSONL trace pipeline, and a clear path to RL training.

## Decisions captured

| # | Question | Answer |
|---|---|---|
| 1 | Multi-agent topology | Per-agent SLURM het-group + `CallAgentTool`-style HTTP delegation. Multiple agents may share one LLM-server het-group (existing `ns agent` dedup). Kanban deferred to a later phase, but entry points reserved. |
| 2 | HERMES_HOME lifecycle | Per-run snapshot: copy template `hermes_home/` → job-local dir at job start, merge writable subset back at job end. |
| 3 | Tool stack | Hermes built-in toolsets (`terminal`, `file`, `code_execution`, `web`, `skills`, `memory`, `delegate`, etc.) **plus** NeMo-Skills MCP servers (`python_tool`, `tavily`, `exa`, `physics`, `chemistry`) exposed to Hermes via `mcp_servers:` in `~/.hermes/config.yaml`. |
| 4 | Observability | JSONL trace files at `<output_dir>/traces/<agent>/<session_id>.jsonl`. Hermes Dashboard deferred. |
| 5 | Template `hermes_home` source | Generate a minimal template committed to the NeMo-Gym repo (config.yaml + empty MEMORY.md + 1–2 seeded skills). |
| 6 | Terminal backend in cluster | Run Hermes terminal calls inside the existing NeMo-Skills sandbox container (`TERMINAL_ENV=docker`, with the sandbox container exposed to the agent process). |
| 7 | Merge-back policy | Merge only `MEMORY.md`, `USER.md`, and `skills/user/` from each agent's job-local home back to the template. **Excluded**: `sessions.db`, `kanban.db`, `logs/`, `.env`, and any non-user-authored skills. |
| 8 | NS MCP wrappers | stdio MCP servers (one process per tool per agent). HTTP/SSE deferred. |

---

## Architecture sketch

```
┌──────────────────── SLURM job (heterogeneous) ────────────────────┐
│                                                                    │
│  het-group 0:                                                      │
│    • vLLM server (orchestrator model)                              │
│    • NeMo-Gym head server hosting hermes_agent "orchestrator"      │
│         HERMES_HOME=<job_dir>/agents/orchestrator/hermes_home/     │
│         mcp_servers: ns_python_tool, ns_tavily, ...   (stdio)      │
│    • (optionally) co-located workers sharing this LLM server       │
│                                                                    │
│  het-group 1..K (one per unique worker model):                     │
│    • vLLM server                                                   │
│    • NeMo-Gym head server(s) for hermes_agent "<worker_name>"      │
│         each: HERMES_HOME=<job_dir>/agents/<name>/hermes_home/     │
│                                                                    │
│  shared:                                                           │
│    • Sandbox container (Docker) — Hermes TERMINAL_ENV=docker       │
│    • Trace dir <output_dir>/traces/<agent>/<session>.jsonl         │
│                                                                    │
│  flow:                                                             │
│    ns hermes_agent_rollouts → orchestrator /run → orchestrator's   │
│    Hermes calls call_<worker>_async (MCP tool) → worker NeMo-Gym   │
│    /task adapter → hermes_agent.responses → answer streams back    │
│                                                                    │
│  post-job:                                                         │
│    curator step rsyncs MEMORY.md, USER.md, skills/user/* from each │
│    agent home back to the template (with audit log).               │
└────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0 — Gap-closing in NeMo-Gym `hermes_agent`

**Repo:** `NeMo-Gym`
**Touch:**
- `responses_api_agents/hermes_agent/app.py`
- `responses_api_agents/hermes_agent/configs/hermes_agent.yaml`
- `responses_api_agents/hermes_agent/requirements.txt`
- new: `responses_api_agents/hermes_agent/trace_writer.py`

### Changes
1. **Configurable persistence flags.** Add `persist_memory`, `persist_skills`, `persist_session`, `save_trajectories`, `enable_compression`, all defaulting `False` (preserves current benchmark behavior). When `True`, pass through to `AIAgent(...)` instead of the current hard-disabling.
2. **HERMES_HOME isolation.** Add `hermes_home: str | None = None` to `HermesAgentConfig`. In `model_post_init`:
   - if set, `os.environ["HERMES_HOME"] = path` AND `hermes_constants.set_hermes_home_override(path)` (ContextVar, defense-in-depth).
   - Document supported topology: one agent per Python process. Multiple-agents-in-one-process is unsupported and will share global env (`TERMINAL_ENV`, `TERMINAL_TIMEOUT`).
3. **`/task` adapter endpoint.** Add `async def task(request, body: TaskRequest) -> dict` on `HermesAgent`:
   - Input shape: `{"task_id": str, "messages": [{"role": "user", "content": str}]}` (matches `CallAgentTool._http_call`).
   - Translate `messages` → `NeMoGymResponseCreateParamsNonStreaming` and call the agent's existing `responses` path.
   - Return `{"generation": <last_assistant_text>, "error": "" | <str>}`.
   - This lets `nemo_skills.mcp.servers.agent_tool.CallAgentTool` call into a NeMo-Gym hermes_agent worker unmodified.
4. **JSONL trace sink (`trace_writer.py`).** New small module exposing `build_callbacks(trace_dir, agent_name) -> dict[str, Callable]`. Returns thread-safe callbacks for `tool_progress_callback`, `tool_start_callback`, `tool_complete_callback`, `step_callback`, `interim_assistant_callback`, `thinking_callback`. Each appends one JSON line `{"ts", "agent", "session_id", "kind", "payload"}` to `<trace_dir>/<session_id>.jsonl`. Backed by a per-session `asyncio.Lock` + flushed open file handle.
5. **Plumb callbacks into `AIAgent`.** In `responses()`, if `config.trace_dir` is set, build callbacks and pass them in `AIAgent(...)`.
6. **`requirements.txt`.** Add the `mcp` Python SDK (already optional inside hermes-agent; pin to a tested SHA). Bump `hermes-agent` pin once the merge-back hooks land — see Phase 1 step 5.

### Tests (Phase 0)

Create:
- `responses_api_agents/hermes_agent/tests/test_task_adapter.py`
- `responses_api_agents/hermes_agent/tests/test_trace_writer.py`
- extend `responses_api_agents/hermes_agent/tests/test_config.py`

Cases:
- **T0.1** `test_config_persistence_flags_default_false` — `HermesAgentConfig()` keeps all five new flags `False`; current benchmark behavior unchanged.
- **T0.2** `test_hermes_home_override_isolation` — instantiate two `HermesAgent`s with different `hermes_home` paths in separate threads; assert `get_hermes_home()` resolves correctly per-thread (via ContextVar override).
- **T0.3** `test_task_endpoint_translates_payload` — POST `{"task_id":"t1","messages":[{"role":"user","content":"hello"}]}` to `/task` with the model server mocked to return `"hi"`; assert response is `{"generation":"hi","error":""}`.
- **T0.4** `test_task_endpoint_propagates_session_seed` — POST `/task` with `task_id` and verify the upstream resources-server `seed_session` was called with that ID (this is the contract that makes `verify()` work in multi-agent setups).
- **T0.5** `test_trace_writer_appends_one_event_per_line` — drive `build_callbacks(...)` events synthetically; assert the file is valid line-delimited JSON, ordered by `ts`, with required fields `(ts, agent, session_id, kind)`.
- **T0.6** `test_trace_writer_handles_concurrent_writes` — 50 fake tool_start events from `asyncio.gather`; assert no truncated lines and the total count matches.
- **T0.7** `test_responses_does_not_persist_when_flags_false` — run `responses()` with `persist_memory=False` and verify no writes happened under `hermes_home` (matches today's behavior).
- **T0.8** `test_responses_persists_when_flags_true` — same with flags True against a tmpdir `hermes_home`; assert at least one of `MEMORY.md`, `sessions.db`, or `skills/` saw a write.

Run: `ng_dev_test +entrypoint=responses_api_agents/hermes_agent`.

**Exit criteria for Phase 0:** all T0.* tests green; existing benchmark rollout (`README.md` example math run) reproduces the documented metrics unchanged.

---

## Phase 0.5 — overlay JSON merging + bundled `peer_agents` Hermes plugin

> Added after Phase 0 shipped, before the Phase 1 PR is opened.  Lands in
> NeMo-Gym as additional commits on the same `sandbox-hermes` branch.

**Why it exists.**  Phase 1 writes a per-agent `hermes_agent_overlay.json`
into each agent's `HERMES_HOME`, but Phase 0's `HermesAgent` doesn't read
that file — so the orchestrator's `peer_agents` map, `agent_name`, and
trace_dir overrides have no runtime effect.  Phase 0.5 closes that gap
with two minimal pieces:

### Changes in NeMo-Gym

1. `HermesAgentConfig.peer_agents: Optional[Dict[str, Dict[str, Any]]]`
   — new field, default `None`.  Mirrored in `configs/hermes_agent.yaml`.
2. `HermesAgent.model_post_init` now calls `_apply_overlay_file(<hermes_home>/hermes_agent_overlay.json)`,
   which JSON-loads the file and field-overrides `self.config` via
   `object.__setattr__` (pydantic-friendly).  Unknown keys log a warning
   and are ignored — the contract is additive so a forward-rolled pipeline
   never breaks an older agent.  Malformed JSON / non-object payloads are
   warned not raised.
3. Bundled Hermes plugin `hermes_home_template/plugins/peer_agents/`
   (registered via the template's `plugins.enabled: [peer_agents]`).
   At Hermes startup the plugin's `register(ctx)`:
   - Reads `<HERMES_HOME>/hermes_agent_overlay.json` (or no-ops if absent).
   - Iterates `peer_agents:` and registers a `call_<name>` tool per peer
     in the `peer_agents` toolset.
   - Each handler resolves the peer's host lazily from
     `SLURM_MASTER_NODE_HET_GROUP_<het_group>` (falling back to
     `localhost`) and POSTs to `http://<host>:<port>/task` with the
     `CallAgentTool` wire format.  HTTP via stdlib `urllib.request` only —
     no new dependency.

### Tests added (20 new; total 57)

- `tests/test_overlay.py` (6): known-field overlay merge, absent-overlay
  back-compat, unknown-key tolerance, malformed JSON tolerance, no
  overlay lookup when `hermes_home is None`, non-dict overlay tolerance.
- `tests/test_peer_agents_plugin.py` (13): overlay → peer_agents
  normalization, missing/malformed/empty overlay no-op, lazy host
  resolution (env var / fallback / explicit override), registration
  shape and toolset wiring, handler rejects missing/empty task,
  end-to-end POST against a real `http.server` peer, peer error
  passthrough, unreachable-peer error wrapping.
- `tests/test_config.py` extended: `peer_agents` defaults to `None` in
  the YAML; the template now ships the `plugins/peer_agents/` plugin.

### Phase 1 implications

The Phase 1 pipeline command already writes `peer_agents` into the
orchestrator's overlay JSON; Phase 0.5 makes that wiring *load-bearing*
on the runtime side.  No Phase 1 code changes are needed.  The
"Known gap" called out at the end of Phase 1's summary is now closed.

---

## Phase 1 — `ns hermes_agent_rollouts` pipeline command

**Repo:** `NeMo-Skills`
**Touch:**
- new: `nemo_skills/pipeline/hermes_agent_rollouts.py`
- new: `nemo_skills/pipeline/utils/scripts/hermes_agent.py` (new `HermesAgentScript`, `HermesHomeBootstrapScript`, `HermesHomeMergebackScript`)
- new: `nemo_skills/pipeline/utils/hermes_manifest.py` (manifest parser)
- new: `nemo_skills/pipeline/scripts/merge_hermes_home.py` (curator step CLI)
- register the command in `nemo_skills/pipeline/cli.py` / `app.py`

### Manifest schema

`agents.yaml`:
```yaml
template_hermes_home: /lustre/.../hermes_template      # optional; uses bundled template if omitted
trace_dir_subpath: traces                              # under output_dir
sandbox:
  container: ${cluster_config.containers.sandbox}
  mounts: []
agents:
  orchestrator:
    model: /hf_models/Nemotron-3-Nano-30B
    server_type: vllm
    server_gpus: 8
    server_args: "--max-model-len 131072"
    hermes:
      enabled_toolsets: [terminal, file, code_execution, web, skills, memory, delegate]
      disabled_toolsets: [send_message, cronjob, kanban]
      max_turns: 60
      persist_memory: true
      persist_skills: true
      save_trajectories: true
      terminal_backend: docker
      mcp_servers:
        ns_python_tool:
          command: ns-mcp-serve
          args: [nemo_skills.mcp.servers.python_tool:DirectPythonTool]
        ns_tavily:
          command: ns-mcp-serve
          args: [nemo_skills.mcp.servers.tavily_search_tool:TavilySearchTool]
    workers: [analyst, coder]
  analyst:
    model_share: orchestrator           # co-locate on orchestrator's LLM server
    hermes: { max_turns: 40, enabled_toolsets: [terminal, file, web] }
  coder:
    model: /hf_models/Qwen2.5-Coder-32B
    server_type: vllm
    server_gpus: 4
    hermes: { enabled_toolsets: [terminal, file, code_execution] }
```

### Pipeline assembly

For one rollout job:

1. **Parse manifest** → het-group plan: dedup `(model, server_type, server_args)` triples (re-use `ns agent`'s dedup logic at [pipeline/agent.py:15-30](../nemo_skills/pipeline/agent.py)).
2. For each agent, **BootstrapScript** runs first:
   - `rsync -a <template>/ <job_dir>/agents/<name>/hermes_home/`
   - Render `<...>/config.yaml` with per-agent overrides (model URL, mcp_servers config, terminal backend, trace_dir).
   - Touch `<job_dir>/traces/<name>/.keep`.
3. **ServerScript** per unique LLM het-group (existing infrastructure).
4. **SandboxScript** (shared, existing).
5. **HermesAgentScript** per agent — launches `ng_run` on a NeMo-Gym head server whose config contains only that agent's `hermes_agent`. Each on its own port. The orchestrator gets `++tool_overrides.CallAgentTool.agents.<worker_name>.{host,port,het_group}` for every worker.
6. **NemoGymRolloutsScript** runs only against the orchestrator (input JSONL flows there). Existing `ng_collect_rollouts` works unchanged.
7. **MergebackScript** runs after the orchestrator completes:
   - For each agent home: rsync `MEMORY.md`, `USER.md`, `skills/user/**` → `<template>`.
   - Emit `<output_dir>/merge_audit.json`: per-file before/after hash + size.

Re-use `ServerScript` / `SandboxScript` / `Command` / `CommandGroup` / `Pipeline` from `nemo_skills/pipeline/utils/scripts/`. Mirror the seed-sharded layout from `nemo_gym_rollouts.py`.

### Bundled minimal template (Phase 1 sub-deliverable)

Path: `NeMo-Gym/responses_api_agents/hermes_agent/template_hermes_home/`
Contents:
```
config.yaml                # empty providers; will be filled in per-agent
.env                       # empty
MEMORY.md                  # empty stub, header only
USER.md                    # empty stub, header only
skills/
  user/                    # empty — agent-curated skills land here
  seed/
    sci-research-notes.skill.md     # one seeded skill: take rigorous notes
    sci-research-cite.skill.md      # one seeded skill: cite sources when claiming facts
logs/.keep
```

### Tests (Phase 1)

Create `NeMo-Skills/tests/pipeline/test_hermes_agent_rollouts.py` and `NeMo-Skills/tests/unit/test_hermes_manifest.py`.

Cases:
- **T1.1** `test_manifest_dedupes_models` — manifest with three agents on two unique models produces exactly 2 LLM-server het-groups.
- **T1.2** `test_manifest_co_located_workers_share_model` — `model_share: orchestrator` puts the worker in het-group 0 with no extra GPU allocation.
- **T1.3** `test_bootstrap_copies_template` — given a tmp template and tmp job dir, `HermesHomeBootstrapScript.build_command()` produces an `rsync -a ...` that, when executed, populates `<job_dir>/agents/<name>/hermes_home/` and renders a valid `config.yaml`.
- **T1.4** `test_bootstrap_sets_trace_dir_in_config` — the rendered config has `trace_dir` pointing at `<output_dir>/traces/<name>/`.
- **T1.5** `test_orchestrator_gets_callagent_overrides` — assemble a manifest with two workers, run the pipeline in `dry_run=True`, capture the orchestrator command; assert `++tool_overrides.CallAgentTool.agents.analyst.{host,port,het_group}` are present.
- **T1.6** `test_mergeback_only_copies_allowlisted_paths` — populate a tmp agent home with `MEMORY.md`, `USER.md`, `skills/user/a.md`, `skills/seed/b.md`, `sessions.db`, `logs/agent.log`; run `merge_hermes_home.py`; assert template now has only the first three updated.
- **T1.7** `test_mergeback_emits_audit_json` — same setup as T1.6; verify `merge_audit.json` lists every changed file with `(before_sha256, after_sha256, size_delta)`.
- **T1.8** `test_pipeline_dry_run_validates_full_manifest` — end-to-end `dry_run=True` against a small (orchestrator + 1 worker) manifest; assert no exceptions, exactly N+1 components in the job, and HardwareConfig matches expected GPU counts.
- **T1.9** `test_existing_nemo_gym_rollouts_unchanged` — smoke-call the existing `ns nemo_gym_rollouts` dry-run; assert behavior unchanged.

Integration smoke (local cluster):
- **T1.S1** Run the pipeline with `--cluster local --dry_run=False` against the bundled `data/example.jsonl` (5 entries), orchestrator only (no workers), tiny model. Assert: `rollouts.jsonl` has 5 lines, `traces/orchestrator/*.jsonl` exists, `merge_audit.json` exists, template `MEMORY.md` was updated.

**Exit criteria for Phase 1:** T1.* unit tests green; T1.S1 succeeds locally; manual run on `oci` cluster with the orchestrator + analyst manifest from the example produces rollouts indistinguishable in reward from a single-agent run on the same problems (regression-free baseline).

---

## Phase 2 — NeMo-Skills MCP servers ↔ Hermes (stdio)

**Status:** delivered.  20 unit tests pass.

**Repo:** `NeMo-Skills`
**Files added / changed:**
- new: `nemo_skills/mcp/stdio_serve.py` (CLI entrypoint `ns-mcp-serve <module>:<Class>`)
- new: `nemo_skills/mcp/servers/README.md` (per-tool consumption guide)
- new: `tests/test_ns_mcp_serve.py`
- changed: `pyproject.toml` — `ns-mcp-serve` console script registered

### How the landscape actually looks

Phase 0 reconnaissance assumed NS MCP servers were all `Tool` classes that
needed wrapping.  In practice the directory has two groups (documented in
`nemo_skills/mcp/servers/README.md`):

* **Group A — native FastMCP entrypoints.**  Files already register a
  `FastMCP` server and ship a `def main()` plus `if __name__ == "__main__"`.
  Hermes consumes via `python -m nemo_skills.mcp.servers.<name>`:
    - `nemo_skills.mcp.servers.python_tool`
    - `nemo_skills.mcp.servers.tavily_search_tool`
    - `nemo_skills.mcp.servers.exa_tool`
  No wrapping needed; they were already MCP servers.
* **Group B — in-process `Tool` classes** with no FastMCP entry.  The
  Phase 2 wrapper exposes these as MCP stdio servers:
    - `nemo_skills.mcp.servers.chemistry.periodictable_tool:PeriodictableTool`
    - `nemo_skills.mcp.servers.physics.coolprop_tool:CoolPropTool`
    - `nemo_skills.mcp.servers.physics.particle_tool:ParticleTool`
    - `nemo_skills.mcp.servers.physics.radioactivedecay_tool:RadioactivedecayTool`
    - `nemo_skills.mcp.servers.web.arxiv_tool:ArxivSearchTool`
    - `nemo_skills.mcp.servers.web.wikipedia_tool:WikipediaSearchTool`

### Wrapper (`ns-mcp-serve`)

A generic stdio MCP wrapper for any `nemo_skills.mcp.tool_manager.Tool`:

```text
ns-mcp-serve <module>:<Class> [--overrides JSON] [--server-name NAME]
```

It uses the low-level `mcp.server.Server` (not FastMCP, which expects
pydantic-typed signatures) so the wrapper can faithfully relay arbitrary
JSON-Schema `input_schema` entries declared by the `Tool`.  Pipeline:

1. Import `module:Class`, instantiate the `Tool`.
2. `tool.configure(overrides, {})` then `tool.post_configure()` (matches
   `ToolManager` semantics).
3. `@server.list_tools()` walks `tool.list_tools()` and emits
   `MCPTool(name, description, inputSchema=...)` per entry.
4. `@server.call_tool()` routes to `tool.execute(name, args, {})`.
5. Exceptions in `execute()` are caught and returned as
   `TextContent(type="text", text="Error: ...")` so the orchestrator sees
   a tool error rather than a transport failure.
6. Non-string returns from `execute()` are JSON-encoded (with a
   `default=str` fallback) before being wrapped in `TextContent`.

Hermes consumes them uniformly:

```yaml
mcp_servers:
  ns_periodictable:
    command: "ns-mcp-serve"
    args: ["nemo_skills.mcp.servers.chemistry.periodictable_tool:PeriodictableTool"]
```

Tool names emerge on the Hermes side as `<server_name>.<raw_tool>` —
Hermes prefixes by the YAML key, so no collisions with built-ins.

### Tests (`tests/test_ns_mcp_serve.py`, 20 cases, real subprocess)

The plan's T2.* lineup mapped onto a concrete test layout:

- **T2.1** `test_t21_list_tools_round_trip` — spawn the wrapper as a real
  subprocess via `mcp.client.stdio.stdio_client`, initialize a
  `ClientSession`, and assert the stub tool's three tools come back with
  correct names and `inputSchema` shape.
- **T2.2** `test_t22_call_tool_happy_path` — round-trip
  `call_tool("echo", {"msg": "hi"})` and assert the wrapper returns
  `TextContent(text="echo: hi")`.
- **T2.3** `test_t23_call_tool_error_passthrough` — a tool that raises
  yields a `TextContent` whose `text` starts with `"Error: RuntimeError"`.
- **T2.4** `test_t24_wrapped_tool_schemas_are_well_formed` —
  parametrised over the 6 Group-B classes; each must instantiate, list
  ≥1 tool, and have `input_schema` of `type=object` with typed
  properties.  Cheaper than pulling in `tools/schema_sanitizer.py` from
  hermes-agent (which would add a hard test dep).
- **T2.5** `test_t25_subprocess_exits_after_session_close` — closing the
  client session lets the wrapper EOF and exit promptly (no orphan PIDs).
- **T2.6** `test_t26_main_rejects_*` (3 cases) — `main()` exits with a
  clear message on bad spec, non-JSON overrides, or non-object overrides.
- **T2.7** `test_t27_coerce_result_*` (3 cases) — direct unit on the
  result coercion path: strings pass through, dicts/lists JSON-encode,
  exotic objects fall back to `str()`.
- Plus 4 supporting cases: `_resolve_spec` error paths,
  `_build_server` wires the Tool, structured payload serialization
  end-to-end.

The original T2.5 ("Hermes can list NS tools") and T2.6 ("no tool name
collision") from the original draft are implicitly verified by the
Phase 0.5 Hermes-side plumbing (Hermes prefixes tool names by the
server's YAML key in `mcp_servers:`) and a future Phase-1 cluster smoke;
both would require pulling in hermes-agent at test time, which we
deferred.  This is documented in the README for follow-up.

### Exit criteria

- [x] All 20 Phase 2 tests pass locally:
      `pytest tests/test_ns_mcp_serve.py` → 20 passed in ~5s.
- [x] Each of the 6 Group-B Tool classes round-trips through the wrapper
      (T2.4 parametrised).
- [x] README documents the Group A / Group B split + Hermes config recipe.
- [ ] **Pending cluster smoke (carried into Phase 3):** in a Phase 1
      multi-agent run with `ns_periodictable` configured in the
      orchestrator's overlay, the first tool call's trace event names
      `ns_periodictable.element_info` (or equivalent).

---

## Phase 3 — Trace pipeline + multi-agent attribution

**Status:** delivered.  7 unit tests pass (T3.1–T3.5 plus two CallAgentTool
emit cases); 49 Phase 1+2 tests + 57 NeMo-Gym tests still green.

**Repos touched:** `NeMo-Skills` and `NeMo-Gym` (Phase 0 trace writer +
peer_agents plugin).

**Files added / changed:**
- new: `NeMo-Skills/nemo_skills/scripts/render_hermes_fleet_trace.py`
- new: `NeMo-Skills/tests/observability/test_fleet_trace.py`
- changed: `NeMo-Skills/nemo_skills/mcp/servers/agent_tool.py` —
  `delegation_trace_dir` + `delegation_agent_name` config keys; emits
  `delegation` events on blocking dispatch, async dispatch, and on
  injection of completed async results
- changed: `NeMo-Gym/responses_api_agents/hermes_agent/trace_writer.py` —
  `step_callback` now forwards every kwarg (incl. `prompt_token_ids`,
  `generation_token_ids`, `generation_log_probs`) for RL training
- changed: `NeMo-Gym/.../hermes_home_template/plugins/peer_agents/__init__.py`
  — emits `delegation` events around each peer call and reuses the
  caller-supplied ticket_id as the wire `task_id`

### Event schema (`<trace_dir>/<agent>/<session_id>.jsonl`)

```json
{
  "ts": 1.736e9,                      // wall-clock float seconds
  "agent": "orchestrator",            // who emitted it
  "session_id": "20260518_abc123",    // Hermes session_id (= wire task_id on workers)
  "kind": "tool_start|tool_progress|tool_complete|step|assistant|thinking|delegation|session_start|session_end|status|reasoning|assistant_interim",
  "payload": {                        // kind-specific
    // tool_start:    {tc_id, tool_name, args}
    // tool_complete: {tc_id, tool_name, args, result}
    // delegation:    {mode: blocking|async|result, worker, ticket_id, ok?, task_preview?, result_preview?}
    // step:          {api_call_count, prev_tools, prompt_token_ids?, generation_token_ids?, generation_log_probs?}
    // thinking:      {msg}
    // session_start: {model, max_turns, history_len, user_message_preview}
  }
}
```

Both the orchestrator-side delegation event and the worker's
`session_start` carry the same id — the orchestrator writes it as
`payload.ticket_id`, the worker as `session_id` (because the Phase 0
`/task` adapter seeds `request.state.task_id` and
`HermesAgent._resolve_session_id` uses it).  That equality is the
renderer's join key.

### Delegation event sources

Two code paths can emit delegation events, both opt-in via overlay /
config:

1. **NeMo-Gym `peer_agents` plugin** (Hermes-driven runs — the default
   Phase 0.5 path).  Reads `trace_dir` + `agent_name` from the
   orchestrator's overlay JSON.  Each `call_<peer>` invocation emits
   `mode=blocking` on dispatch and `mode=result` on return.  Writes to
   `<trace_dir>/<agent_name>/delegations.jsonl` because Hermes does not
   surface a per-call session id to plugin handlers.
2. **NS `CallAgentTool`** (legacy / non-Hermes orchestrators).  New
   config keys `delegation_trace_dir` and `delegation_agent_name`.
   Emits `mode=blocking` on `_call_blocking`, `mode=async` on
   `_fire_async`, and `mode=result` on `get_pending_injections`.

### Fleet trace viewer

`python -m nemo_skills.scripts.render_hermes_fleet_trace <output_dir>`
walks `traces/**/*.jsonl`, merges by `ts`, and prints a colored
chronological log.  Flags:

- `--format text|json` — text (default) or one JSON object per line
- `--no-color` — disable ANSI colors even on a TTY
- `--filter-agents agent1,agent2` — limit to specific emitters
- `--filter-kinds tool_start,delegation` — limit to specific event kinds
- `--ticket TICKET_ID` — restrict output to one delegation and its
  matching worker session (joined on `ticket_id` ↔ `session_id`)

### Tests (`tests/observability/test_fleet_trace.py`, 7 cases)

- **T3.1** `test_t31_event_schema_fields_required` — every event from
  `build_callbacks(...)` plus `emit_event(...)` has the required
  `{ts, agent, session_id, kind, payload}` keys.
- **T3.2** `test_t32_delegation_events_link_orchestrator_and_worker` —
  given a synthesized delegation pair on the orchestrator and a worker
  session keyed by the same ticket, `_events_linked_to_ticket(...)`
  returns exactly the matching events; unrelated tickets are excluded.
- **T3.3** `test_t33_fleet_renderer_orders_by_ts` — out-of-order writes
  across two agents merge into a monotonic ts stream via `render_main`.
- **T3.4** `test_t34_token_ids_preserved_in_step_events` — Phase 0
  `step_callback` now forwards `prompt_token_ids`,
  `generation_token_ids`, and `generation_log_probs` (RL prep).
- **T3.5** `test_t35_trace_files_truncate_safely` — a deliberately
  truncated trailing line is skipped with a stderr warning; the renderer
  yields the well-formed events and exits 0.
- **CallAgentTool**: two extra cases verify the `_emit_delegation` path
  writes a valid two-event delegation file when `delegation_trace_dir`
  is set and is a silent no-op when it is not.

T3.1 and T3.4 import the Phase-0 trace writer; the test file probes
`NEMO_GYM_PATH` env var and a couple of standard local paths, falling
through to `pytest.skip` if NeMo-Gym is not on `sys.path`.

**Exit criteria for Phase 3:**
- [x] Trace files from a multi-agent rollout can be joined into a
      coherent fleet timeline (renderer + T3.2 + T3.3).
- [x] Token IDs survive end-to-end on at least the orchestrator (T3.4).
- [x] All 7 Phase 3 unit tests green; Phases 0+0.5 (57 tests) and
      Phases 1+2 (49 tests) regression-free.
- [ ] Cluster smoke: multi-agent run produces a `delegations.jsonl`
      next to per-session trace files and the renderer joins them by
      ticket — deferred until cluster access (rolled into the same
      Phase-1/2 cluster-smoke ticket).

---

## Phase 4 — Kanban entry points

**Status:** delivered.  ~10 new test cases pass (see below); Phases 0–3
regression-free (57 NeMo-Gym + 60 NeMo-Skills hermes-related cases).

**Repo:** `NeMo-Skills`
**Files changed:**
- `nemo_skills/pipeline/utils/hermes_manifest.py` — `HermesAgentSpec.kind`
  (``"agent" | "dispatcher"``), `dispatcher_cpus`, `dispatcher_mem_gb`;
  `HermesFleetManifest.shared_kanban_db`; new `ServerGroup.is_dispatcher`
  property; `build_server_groups` allocates a fresh het-group per
  dispatcher (no model dedup); strict validation rejects
  ``kind=dispatcher`` entries that also set ``model``, ``role:
  orchestrator``, or ``workers:``.
- `nemo_skills/pipeline/utils/scripts/hermes_agent.py` — bootstrap script
  symlinks `<agent_home>/kanban.db` → `shared_kanban_db` when set
  (creating the parent dir and touching the target file so the link is
  immediately valid); new ``HermesKanbanDispatcherScript`` runs
  ``hermes plugins enable kanban && exec hermes kanban dispatcher run``.
- `nemo_skills/pipeline/hermes_agent_rollouts.py` — branches on
  ``group.is_dispatcher``: no LLM server, no sandbox, no head, no
  rollouts, HardwareConfig with ``num_gpus=0`` and ``num_tasks=1``;
  ``peer_agents`` map excludes dispatcher entries; merge-back skips
  dispatcher HERMES_HOMEs.
- `nemo_skills/pipeline/utils/scripts/__init__.py` — re-exports the new
  dispatcher script.

### Manifest extensions

```yaml
shared_kanban_db: /lustre/.../kanban.db
agents:
  orch:
    model: /m/orch
    server_gpus: 8
  worker1:
    model_share: orch
    hermes:
      enabled_toolsets: [terminal, file, kanban_*]
  kdisp:
    kind: dispatcher
    dispatcher_cpus: 4
    dispatcher_mem_gb: 16
```

### Tests
- **T4.1** `test_t41_manifest_dispatcher_entry` — dispatcher manifest
  produces a het-group plan with no LLM server allocation
  (``group.server_gpus == 0`` for the dispatcher group).
- **T4.2** `test_t42_bootstrap_symlinks_shared_kanban_db` — when
  ``shared_kanban_db`` is set, the bootstrap script materialises
  ``<agent_home>/kanban.db`` as a symlink to the shared path.  Companion
  case verifies the symlink is absent when the field is unset.
- Plus pipeline-level cases: dispatcher excluded from orchestrator's
  peer map, mergeback skips dispatcher HERMES_HOMEs, dispatcher script
  emits the expected hermes-CLI two-liner, and parser rejects bad
  dispatcher specs (model set, orchestrator role, workers list, all-
  dispatcher manifest, unknown kind).

**Exit criteria:** all Phase 4 cases pass; Phase 1 + 2 + 3 tests
regression-free.  Cluster smoke (kanban DB shared between dispatcher and
orchestrator, dispatcher routes tasks back to a worker via its kanban
tools) is deferred until cluster access — rolled into the existing
Phase-1/2/3 cluster-smoke ticket.

---

## Phase 5 — RL preparation

**Status:** delivered (minimal surface, leaving per-worker judge wiring
for later).  4 new tests pass.

**Repo:** `NeMo-Skills`
**Files added:**
- `nemo_skills/scripts/export_hermes_trajectories.py` — walks
  ``<output_dir>/traces/**/*.jsonl``, buckets events by
  ``(agent, session_id)``, builds a ShareGPT-style ``conversations``
  list per session, lifts ``prompt_token_ids`` /
  ``generation_token_ids`` / ``generation_log_probs`` out of ``step``
  events into a ``token_ids`` sub-object, and propagates the
  orchestrator's terminal reward (from ``rollouts.jsonl``) to its
  workers via the ``ticket_id ↔ session_id`` join key Phase 3 set up.
  Per-worker judges are intentionally deferred — easy to swap in later
  by changing the reward-resolution step without touching the schema.
- `tests/observability/test_trajectory_export.py` — 4 cases.

### Reward attribution model

The exporter implements design choice **(a)** from the original plan —
the orchestrator's reward propagates to all its workers:

* Orchestrator sessions are ordered by their first event's timestamp;
  reward[k] = rollouts[task_index=k].
* Worker session_id equals the dispatch ticket_id.  Each delegation
  event in ``delegations.jsonl`` carries the ticket_id and a timestamp;
  we attach the delegation to the temporally-nearest orchestrator
  session, then forward that session's reward to the worker.

This is the simplest faithful join given Phase 3's data; replacing it
with per-worker reward judges is a localized change to
``_load_rollouts`` / ``export_trajectories`` and does not touch the
trace schema or the trajectory format.

### ShareGPT export shape

One JSON line per session:

```json
{
  "agent": "orchestrator",
  "session_id": "task_42",
  "model": "Nemotron-3-Nano-30B",
  "conversations": [
    {"from": "human", "value": "solve task X"},
    {"from": "gpt",   "value": "I'll delegate to analyst."},
    {"from": "tool",  "value": "{\"tool_name\": \"calculator\", \"result\": \"42\"}"},
    {"from": "gpt",   "value": "result: 42"}
  ],
  "reward": 1.0,
  "token_ids": {
    "prompt_token_ids":      [[101, 102, 103]],
    "generation_token_ids":  [[201, 202]],
    "generation_log_probs":  [[-0.1, -0.2]]
  }
}
```

``token_ids`` is omitted entirely when no step event carried RL tensors,
so non-RL runs don't pay the storage cost.

### Tests
- **T5.1** `test_t51_per_agent_trajectory_export` — synthesizes a
  two-agent run (orchestrator + analyst with one delegation), runs the
  exporter, verifies each agent's trajectory file has the ShareGPT
  shape, the orchestrator's reward propagated to the analyst, and the
  ``token_ids`` sub-object survived end-to-end.
- **T5.1.b** `test_t51_no_rollouts_means_reward_none` — without a
  rollouts file, trajectories still export with ``reward: null``.
- **T5.1.c** `test_build_trajectory_skips_unknown_kinds` — unknown
  event kinds are dropped from ``conversations`` (forward-compat with
  future Hermes callbacks).
- **T5.2** `test_t52_trajectory_compressor_handles_export` — feeds an
  exported trajectory's ``conversations`` list into
  ``trajectory_compressor.TrajectoryCompressor._find_protected_indices``
  (the schema-validation entry point), confirms the compressor
  recognises our role tags ({system, human, gpt, tool}) and returns a
  non-empty ``protected`` set.  We exercise the pure code path rather
  than instantiating ``TrajectoryCompressor`` because the constructor
  reaches out to OpenRouter / HF — out of scope for unit tests.

**Exit criteria:**
- [x] T5.* pass under the project's miniconda env.
- [x] Existing Phase 3 step events carry token IDs without exporter
      involvement (verified by T3.4 + T5.1 together).
- [ ] A real multi-agent rollout produces an exporter output the
      trajectory_compressor instance can fully process (i.e. with
      provider creds + tokenizer available) — deferred to cluster
      smoke alongside Phases 1/2/3/4.

---

## Deliverables checklist

- [x] Phase 0 — NeMo-Gym `hermes_agent`:
  - [x] `HermesAgentConfig` persistence flags + `hermes_home` field
  - [x] `/task` endpoint
  - [x] `trace_writer.py` with thread-safe JSONL sink
  - [x] `tests/test_task_adapter.py`, `tests/test_trace_writer.py`, updated `tests/test_config.py`
  - [x] `hermes_home_template/` bundled with two seeded skills
        (note: directory renamed from `template_hermes_home/` to dodge the
        repo's `temp*` `.gitignore` rule)
- [x] Phase 0.5 — overlay merge + bundled peer_agents plugin (NeMo-Gym):
  - [x] `HermesAgentConfig.peer_agents` field
  - [x] `HermesAgent._apply_overlay_file` invoked in `model_post_init`
  - [x] `hermes_home_template/plugins/peer_agents/` (plugin.yaml +
        `__init__.py`) enabled via template `plugins.enabled:`
  - [x] `tests/test_overlay.py` + `tests/test_peer_agents_plugin.py`
- [x] Phase 1 — NeMo-Skills pipeline:
  - [x] `nemo_skills/pipeline/hermes_agent_rollouts.py`
  - [x] `nemo_skills/pipeline/utils/scripts/hermes_agent.py` (3 scripts)
  - [x] `nemo_skills/pipeline/utils/hermes_manifest.py`
  - [x] `nemo_skills/scripts/merge_hermes_home.py`
  - [x] `tests/test_hermes_agent_rollouts.py`, `tests/test_hermes_manifest.py`
  - [x] Local dry-run smoke validated (T1.S1 cluster smoke still pending)
- [x] Phase 2 — MCP wrappers:
  - [x] `nemo_skills/mcp/stdio_serve.py` + `ns-mcp-serve` console script
  - [x] 6 wrapped tools verified (periodictable, coolprop, particle,
        radioactivedecay, arxiv, wikipedia) plus README documenting the
        Group-A FastMCP entrypoints (python_tool / tavily / exa)
  - [x] `tests/test_ns_mcp_serve.py` (20 cases including a real
        subprocess round-trip; Hermes-side integration deferred to a
        Phase-1 cluster smoke)
- [x] Phase 3 — Trace + viewer:
  - [x] `nemo_skills/scripts/render_hermes_fleet_trace.py`
  - [x] `CallAgentTool` delegation events (NS-side)
  - [x] NeMo-Gym `peer_agents` plugin delegation events (Hermes-side)
  - [x] `trace_writer.step_callback` forwards token-id kwargs
  - [x] `tests/observability/test_fleet_trace.py` (7 cases incl. T3.1–T3.5)
- [x] Phase 4 — Kanban hooks:
  - [x] manifest schema: `kind: dispatcher`, `shared_kanban_db`,
        `dispatcher_cpus/mem_gb`
  - [x] bootstrap symlink to shared kanban DB
  - [x] `HermesKanbanDispatcherScript` and `_build_jobs` dispatcher branch
  - [x] pipeline-level dispatcher tests (peer-exclusion, mergeback skip,
        hardware shape)
- [x] Phase 5 — RL surface:
  - [x] `nemo_skills/scripts/export_hermes_trajectories.py`
  - [x] `tests/observability/test_trajectory_export.py`
  - [ ] cluster-smoke: full trajectory_compressor run (deferred)
- [x] Phase 6 — Validation experiment (scaffolding, ready to submit):
  - [x] NeMo-Gym `resources_servers/passthrough/` (no-op verifier so
        offline judges grade the generation later)
  - [x] NeMo-Gym `HermesAgentConfig.chat_template_kwargs` knob (Kimi
        uses `thinking=false`, not the Qwen `enable_thinking`)
  - [x] NS bootstrap merges `mcp_servers` into `<HERMES_HOME>/config.yaml`
        so Hermes picks the manifest-declared MCPs up at startup
  - [x] `workdir-agents/validation/frontierscience_manifest.yaml` —
        single Hermes agent on Kimi-K2.6 with every built-in toolset
        + all NS MCP servers
  - [x] `workdir-agents/validation/scripts/`: convert_ns_to_ng,
        convert_ng_to_ns, run_validation (3-run driver),
        submit_judges (gpt-oss-120b), compare_runs (delta table)
  - [x] `tests/test_hermes_agent_rollouts.py` extends with bootstrap
        ↔ mcp_servers merge regression cases
  - [ ] Cluster run on aws-cmh — pending user mirroring Oleksii's
        `vllm-glm51-cu130-ray.sqsh` container + the model checkpoint
        availability at `/hf_models/Kimi-K2.6`

## Phase 6 — Validation experiment (frontierscience-olympiad)

**Status:** scaffolding delivered locally, ready for cluster submit.

**Goal:** answer the headline question "does the agent's memory/skill
update (Phase 1 merge-back) actually improve accuracy on a hard
scientific-research benchmark when the agent has every tool/MCP
available?"  Three-run A/B/C with Kimi-K2.6 as the orchestrator on
`aws-cmh`, judged offline by `gpt-oss-120b` (same pattern as
`inference_scripts/agentic/run_nano_physics_agent.py`).

Full layout, smoke + full-100 recipes, and the comparator output are
documented in [`workdir-agents/validation/README.md`](validation/README.md).

The Phase 6 NeMo-Gym additions (passthrough verifier +
`chat_template_kwargs` knob) are the smallest NeMo-Gym surface that
keeps `hermes_agent_rollouts` runnable against benchmarks whose judge
lives entirely in NeMo-Skills.

## How to verify progressively

Each phase has a smoke command we should run before moving on.

| Phase | Smoke command | What it confirms |
|---|---|---|
| 0 | `ng_dev_test +entrypoint=responses_api_agents/hermes_agent` + the existing math rollout from the README | persistence flags + /task + trace writer + no regression in existing benchmark |
| 1 | `ns hermes_agent_rollouts --cluster local --agent_manifest .../orchestrator-only.yaml --input_file data/example.jsonl --output_dir /tmp/h1 --dry_run=False` | bootstrap → server → orchestrator → merge-back loop works end-to-end on a single agent |
| 2 | same as Phase 1 + manifest with `ns_python_tool` enabled; inspect trace for `ns_python_tool.*` tool calls | MCP wrapping & schema sanity |
| 3 | Phase 1 manifest with orchestrator + 1 worker; run `render_hermes_fleet_trace.py /tmp/h3` | per-agent traces join cleanly on ticket_id |
| 4 | manifest with `shared_kanban_db` + 1 dispatcher + 2 workers; verify `kanban list` shows tasks created by orchestrator | kanban hooks active without breaking Phase 1/2/3 |
| 6 | `python workdir-agents/validation/scripts/run_validation.py --limit 5 --dry-run ...` then submit + `compare_runs.py` | 3-run A/B/C wiring works end-to-end, frontier-science answer survives NS↔NG conversion both ways |

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| `TERMINAL_ENV` / `TERMINAL_TIMEOUT` are process-global → two agents in one process collide | One agent per process, enforced by pipeline (one head-server per agent). Document loudly. |
| Hermes built-in tools and NS MCP tools overlap (e.g. `execute_code`) | MCP servers are prefixed by server name in Hermes; disable Hermes's built-in equivalent in `disabled_toolsets` when an NS replacement is preferred. |
| Skill curator may write low-quality skills back to template on every run | Merge-back is allowlisted to `skills/user/` only; add a `--dry-run-merge` flag for the curator step; add an audit JSON the user reviews before promoting changes |
| Hermes git pin lags behind upstream | Re-pin to a tested commit per phase; CI smoke against the pin |
| `mcp` SDK breakage between versions | Pin the SDK in `requirements.txt` with a `<next-major>` ceiling, matching Hermes's dependency policy |

## Open follow-ups (non-blocking)

- Decide per-worker reward attribution before training (Phase 5).
- Decide whether to make `ns hermes_agent_rollouts` a thin wrapper over `ns nemo_gym_rollouts` once the manifest support is generic enough to upstream into the latter.
- Discuss with Hermes maintainers whether the `/task` adapter and trace callbacks belong in hermes-agent itself rather than in the NeMo-Gym wrapper.
