# NeMo-Skills Agent Integration Plan

## Answers to Open Questions

| # | Question | Answer |
|---|---|---|
| 1 | Tool scope | Arbitrary MCP tools (not just code execution) |
| 2 | Trajectory format | Final answer required + optional `conversation`, `num_tool_calls` in JSONL |
| 3 | Planning phase | Skip for now — extend via custom agent loops later if needed |
| 4 | Multi-agent scope | **Multi-agent from the start** — peer-to-peer async communication with hierarchy via tool-list restrictions |
| 5 | Loop engine | NeMo-Skills' native `ToolCallingWrapper` |

---

## Executive Summary

NeMo-Skills already contains most of the infrastructure for agentic loops.
`ToolCallingWrapper` (`nemo_skills/inference/model/tool_call.py`) is already
a correct ReAct-style agent loop. The missing pieces are:

1. **Trajectory saving** — preserve `conversation` / `num_tool_calls` in JSONL output
2. **Worker server mode** — each agent can act as an HTTP server for peer calls
3. **`CallAgentTool`** — MCP tool that delegates tasks to other agent workers
4. **`ns agent` pipeline command** — multi-agent SLURM het-group job setup

NAT is **not** used as the execution engine (it requires a persistent local service,
incompatible with fire-and-forget SLURM batch jobs). NAT patterns inform the design
but the implementation is native to NeMo-Skills.

---

## 1. Architecture

### 1.1 Multi-Agent SLURM Job Structure

```
SLURM het-job
├── Het-group 0  (orchestrator)
│   ├── LLM server A   (vLLM / SGLang)
│   ├── Sandbox        (optional, for code execution)
│   └── Orchestrator   → nemo_skills.inference.agent_generate  (batch mode)
│                         reads input JSONL, writes output JSONL
│                         tools: [python_exec, call_analyst, call_coder, ...]
│
├── Het-group 1  (worker: analyst)
│   ├── LLM server B
│   └── Agent worker   → nemo_skills.inference.agent_server  (server mode)
│                         listens on port 7777, processes task requests
│                         tools: [python_exec, ...]
│
└── Het-group 2  (worker: coder)
    ├── LLM server C
    └── Agent worker   → nemo_skills.inference.agent_server  (server mode)
                          listens on port 7777
                          tools: [python_exec, ...]
```

**Inter-agent communication:**
- Orchestrator calls workers via HTTP POST to `http://{SLURM_MASTER_NODE_HET_GROUP_N}:{port}/task`
- The `CallAgentTool` makes async HTTP requests (non-blocking from orchestrator's perspective)
- Hierarchy is enforced by which agent names appear in each agent's `tool_modules`

**Address resolution:**
- `SLURM_MASTER_NODE_HET_GROUP_N` env vars are exported by NeMo-Run before Python starts
- `CallAgentTool` reads them at `configure()` time via `os.environ`
- `ns agent` injects `het_group` indices into `tool_overrides.CallAgentTool.agents`

### 1.2 New Files

```
NeMo-Skills/
├── nemo_skills/
│   ├── inference/
│   │   ├── agent_generate.py          ← NEW: AgentTask (batch orchestrator)
│   │   └── agent_server.py            ← NEW: HTTP server for worker agents
│   │
│   ├── mcp/servers/
│   │   └── agent_tool.py              ← NEW: CallAgentTool MCP tool
│   │
│   └── pipeline/
│       ├── agent.py                   ← NEW: `ns agent` pipeline command
│       └── utils/scripts/
│           └── agent.py               ← NEW: AgentWorkerScript
│
└── nemo_skills/prompt/config/agents/  ← NEW: agent prompt templates
    ├── default_agent.yaml
    ├── code_agent.yaml
    └── orchestrator.yaml
```

**Modified files:**
- `nemo_skills/inference/factory.py` — add `agent_generate` type
- `nemo_skills/pipeline/cli.py` — register `ns agent`
- `nemo_skills/pipeline/utils/scripts/__init__.py` — export `AgentWorkerScript`

### 1.3 Data Flow per Request

```
Input JSONL sample
      │
      ▼
AgentTask.process_single_datapoint()
      │  fills prompt → ToolCallingWrapper.generate_async()
      │
      ▼ (ToolCallingWrapper loop)
  LLM generates ──────────────────────────────────┐
      │                                             │
      ├─ tool_call: python_exec  ──► DirectPythonTool.execute()
      │                                             │
      ├─ tool_call: call_analyst ──► CallAgentTool.execute("analyst", task)
      │                                  │
      │                                  ▼ async HTTP POST /task
      │                              AgentServer (het-group 1)
      │                                  │ ToolCallingWrapper loop
      │                                  │ (worker's own tools)
      │                                  ▼
      │                              generation / tool results
      │                                  │ HTTP response
      │                              ◄───┘
      │
      └─ no more tool calls → break
      │
      ▼
AgentTask.postprocess_single_output()
  → output["generation"]          (final answer)
  → output["num_tool_calls"]      (total across all rounds)
  → output["conversation"]        (full turn history, if save_trajectory=True)
      │
      ▼
Write to output.jsonl
```

---

## 2. Component Details

### 2.1 `nemo_skills/inference/agent_generate.py`

Extends `GenerationTask` (mirrors `generate.py`):

```python
@nested_dataclass(kw_only=True)
class AgentTaskConfig(GenerationTaskConfig):
    prompt_format: str = "openai"   # agents use chat messages format
    save_trajectory: bool = True    # save conversation in output JSONL

class AgentTask(GenerationTask):
    async def postprocess_single_output(self, output, original_data_point):
        await super().postprocess_single_output(output, original_data_point)
        if not self.cfg.save_trajectory:
            output.pop("conversation", None)
        output.pop("tools", None)      # schema not useful in output

GENERATION_TASK_CLASS = AgentTask
```

Key behaviours:
- Forces `tool_modules` to be non-empty (logs a warning if not set)
- `save_trajectory=True` preserves full `conversation` turns in JSONL
- `finish_reason` always saved (explains why agent stopped)

### 2.2 `nemo_skills/inference/agent_server.py`

Standalone FastAPI server for worker agents:

```python
@nested_dataclass(kw_only=True)
class AgentServerConfig:
    server: dict = field(default_factory=dict)
    agent_port: int = 7777
    agent_name: str = "agent_worker"
    tool_modules: list[str] | None = None
    tool_overrides: dict | None = field(default_factory=dict)
    schema_overrides: dict | None = field(default_factory=dict)
    max_tool_calls: int = -1
    sandbox: dict = field(default_factory=dict)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
```

HTTP endpoints:
- `POST /task` — accepts `{task_id, messages}`, runs agent loop, returns result
- `GET /health` — returns `{"status": "ok", "name": agent_name}`

### 2.3 `nemo_skills/mcp/servers/agent_tool.py`

`CallAgentTool` handles all configured workers:

```python
# Configuration via tool_overrides:
tool_overrides = {
    "CallAgentTool": {
        "agents": {
            "analyst": {"het_group": 1, "port": 7777},
            "coder":   {"het_group": 2, "port": 7777},
        }
    }
}
```

At `configure()` time, resolves addresses:
```python
host = os.environ.get(f"SLURM_MASTER_NODE_HET_GROUP_{het_group}", "localhost")
url  = f"http://{host}:{port}"
```

Exposes one tool per agent:
```python
# list_tools() returns:
[
    {"name": "call_analyst", "description": "Delegate to analyst agent", ...},
    {"name": "call_coder",   "description": "Delegate to coder agent",   ...},
]
```

### 2.4 `nemo_skills/pipeline/utils/scripts/agent.py`

`AgentWorkerScript` extends `BaseJobScript`:
- Starts `nemo_skills.inference.agent_server` in the het-group
- Resolves LLM server address lazily via `server_script.hostname_ref()`
- Exposes `port` for the orchestrator's `CallAgentTool` configuration

### 2.5 `nemo_skills/pipeline/agent.py`

`ns agent` pipeline command:

```bash
# Single agent
ns agent --cluster my_cluster --model /path/to/model --server-type vllm \
         --server-gpus 8 --input-file /data/problems.jsonl \
         --output-dir /results ++tool_modules=[...]

# Multi-agent
ns agent --cluster my_cluster --model /path/to/orchestrator --server-type vllm \
         --server-gpus 8 --input-file /data/problems.jsonl \
         --output-dir /results \
         --worker-model /path/to/model-B /path/to/model-C \
         --worker-server-type vllm --worker-server-gpus 4 \
         --worker-names analyst coder \
         ++tool_modules=["nemo_skills.mcp.servers.agent_tool::CallAgentTool",\
                         "nemo_skills.mcp.servers.python_tool::DirectPythonTool"]
```

Internally:
1. Creates Het-group 0: LLM-A + orchestrator script
2. Creates Het-group N: LLM-N + worker script for each worker
3. Auto-injects `tool_overrides.CallAgentTool.agents.{name}.het_group={N}` into orchestrator's args
4. Auto-injects `tool_overrides.CallAgentTool.agents.{name}.port={port}` for each worker

---

## 3. Implementation Sequence (Phase 1)

1. `nemo_skills/inference/agent_generate.py` — `AgentTask` + `AgentTaskConfig`
2. `nemo_skills/inference/agent_server.py` — `AgentServer` + `AgentServerConfig` + FastAPI app
3. `nemo_skills/mcp/servers/agent_tool.py` — `CallAgentTool`
4. `nemo_skills/pipeline/utils/scripts/agent.py` — `AgentWorkerScript`
5. `nemo_skills/pipeline/agent.py` — `ns agent` command
6. `nemo_skills/inference/factory.py` — register `agent_generate`
7. `nemo_skills/pipeline/cli.py` — register `ns agent`
8. `nemo_skills/pipeline/utils/scripts/__init__.py` — export `AgentWorkerScript`
9. Prompt templates in `nemo_skills/prompt/config/agents/`

---

## 4. Phase 2: NAT Integration (Future)

- Run NAT A2A server **inside** a SLURM het-group for complex multi-agent orchestration
- Wire NAT's observability exporters into `AgentTask`
- Use NAT's Test-Time Compute (TTC) as a trajectory selection strategy

---

## 5. Phase 1 Implementation Summary

### Delivered files

| File | Purpose |
|---|---|
| `nemo_skills/inference/agent_generate.py` | `AgentTask` — batch orchestrator; extends `GenerationTask` with trajectory saving (`conversation`, `num_tool_calls`, `finish_reason` preserved in JSONL) |
| `nemo_skills/inference/agent_server.py` | FastAPI HTTP server for worker agents; runs inside a SLURM het-group; accepts `/task` POST requests from peer agents |
| `nemo_skills/mcp/servers/agent_tool.py` | `CallAgentTool` MCP tool; exposes one `call_{name}` tool per configured worker; resolves addresses from `SLURM_MASTER_NODE_HET_GROUP_N` env vars at startup |
| `nemo_skills/pipeline/utils/scripts/agent.py` | `AgentWorkerScript` — lazy-evaluated script component that starts `agent_server.py` in a het-group alongside its LLM |
| `nemo_skills/pipeline/agent.py` | `ns agent` CLI command; auto-wires `CallAgentTool` het-group indices into orchestrator args |
| `nemo_skills/prompt/config/agents/default_agent.yaml` | General-purpose agent system prompt |
| `nemo_skills/prompt/config/agents/code_agent.yaml` | Code-execution agent system prompt |
| `nemo_skills/prompt/config/agents/orchestrator.yaml` | Multi-agent orchestrator system prompt |

### Modified files

| File | Change |
|---|---|
| `nemo_skills/inference/factory.py` | Added `agent_generate` to `GenerationType` enum and `GENERATION_MODULE_MAP` |
| `nemo_skills/pipeline/cli.py` | Registered `ns agent` command |
| `nemo_skills/pipeline/utils/scripts/__init__.py` | Exported `AgentWorkerScript` |
| `nemo_skills/pipeline/utils/__init__.py` | Exported `AgentWorkerScript` from top-level utils |

### Key design properties

- **Hierarchy** is enforced purely through `tool_modules` config — workers that should not call each other simply do not have `CallAgentTool` in their tool list.
- **Address wiring** is zero-config: `ns agent` auto-injects `het_group` indices; `CallAgentTool.configure()` resolves hostnames from SLURM env vars at job startup.
- **Single-agent** is a degenerate case — omit `--worker-model` and it is just `ns generate` with trajectory saving and arbitrary MCP tools.
- **All execution is remote** — every component runs inside the SLURM job; nothing requires a local persistent process.

---

## 6. Phase 1 Extension: Manager-Worker Pattern

### Motivation

The basic multi-agent setup gives the orchestrator all tools (CallAgentTool + PythonTool).
A more structured pattern separates roles:
- **Orchestrator** (manager): receives problem, delegates computation, validates result,
  produces final answer.  Has only `CallAgentTool` — cannot execute code directly.
- **Solver** (worker): receives delegated task, runs Python to solve it, returns result.
  Has `PythonTool`, no `CallAgentTool`.

### Changes to support worker system messages

Workers receive tasks as bare `[{"role": "user", "content": task}]` messages from
`CallAgentTool.execute()`.  Without a system prompt the worker has no instruction to
use Python tools.

**`agent_server.py`** (`AgentServerConfig`):
- Added `system_message: str = ""` — prepended to every incoming task request.
- Added `system_message_yaml: str = ""` — if set (e.g. `"agents/code_agent"`), loads
  the system message from `nemo_skills/prompt/config/{yaml}.yaml` at server startup.

**`agent_generate.py`** (`AgentTaskConfig`):
- Added `system_message_yaml: str = ""` — if set, loads system message from YAML and
  overrides the `system_message` default before generation starts.
- Resolved in `AgentTask.__init__` before `super().__init__(cfg)` so `get_prompt`
  picks up the correct value.

### Usage in `run_nano_physics_agent.py`

```
Single-agent:
  orchestrator extra_args = base_args + PYTHON_TOOL
  (system_message = default_agent.yaml system, prompt_config = benchmark default)

Multi-agent:
  orchestrator extra_args = base_args + ORCHESTRATOR_TOOL + ++system_message_yaml=agents/orchestrator
  worker_extra_args       = PYTHON_TOOL + ++system_message_yaml=agents/code_agent
```

`WORKER_EXTRA_ARGS` is broadcast to all workers (normalize_parameter handles single→N).

### Running the manager-worker evaluation

```bash
# Dry run (verify args)
python inference_scripts/agentic/run_nano_physics_agent.py \
    --run-id 1 --benchmarks fs \
    --worker-model /hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --worker-names solver \
    --dry-run

# Full run
python inference_scripts/agentic/run_nano_physics_agent.py \
    --run-id 1 --benchmarks fs \
    --worker-model /hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --worker-names solver
```

`--worker-names solver` gives the tool `call_solver` — matching the orchestrator
system prompt's phrasing.  Omit for default `call_worker_0`.

---

## 7. Phase 1 Extension: Non-Blocking Async Agent Delegation

### Motivation

In the blocking `call_{name}` pattern the orchestrator's loop is idle for the
entire duration of the worker's run.  This is wasteful when:
- multiple independent sub-tasks can be delegated concurrently, or
- the orchestrator can continue reasoning (e.g. verify constraints, formulate
  follow-up tasks) while a slow solver is computing.

### Design

`CallAgentTool` now exposes **three tools per worker** instead of one:

| Tool | Behaviour |
|---|---|
| `call_{name}(task)` | Existing blocking call — awaits the full response |
| `call_{name}_async(task)` | Fires HTTP request as background `asyncio.Task`; returns `{"ticket_id":"...","status":"pending"}` immediately |
| `collect_results(ticket_ids)` | Explicit blocking wait for specific tickets; already-injected tickets returned from cache |

Completed background tasks are **automatically injected** into the conversation
as new messages before each LLM turn, so the orchestrator sees results without
having to call any polling tool.

### Scheduling diagram

```
Orchestrator loop (single problem)

Turn N:   <tool_call: call_solver_async("compute X")>   ← fires bg task T1
          <tool_call: call_solver_async("compute Y")>   ← fires bg task T2
          tool results: {"ticket_id":"t1","status":"pending"}
                        {"ticket_id":"t2","status":"pending"}

Turn N+1: [ToolCallingWrapper checks get_pending_injections — nothing done yet]
          LLM continues reasoning, calls python_exec, etc.

Turn N+2: [ToolCallingWrapper checks — T1 done]
          → injects: [Async result for ticket t1 — call_solver_async "compute X"]: ...
          LLM sees injected result; T2 still running

Turn N+3: [ToolCallingWrapper checks — T2 done]
          → injects: [Async result for ticket t2 — call_solver_async "compute Y"]: ...
          LLM synthesises final answer
```

Multiple problems proceed concurrently at the `asyncio` level — each has its own
`request_id` and independent `_pending_tasks` dict, so background tasks for
different problems never interfere.

### Implementation

**`nemo_skills/mcp/tool_manager.py`**
- `Tool` base: added `async get_pending_injections(request_id) → List[Dict]` no-op hook
- `ToolManager`: added `iter_tools()` to expose live tool instances to callers

**`nemo_skills/mcp/servers/agent_tool.py`** — extended `CallAgentTool`:
- `_config["injection_role"]` (default `"user"`) — role of the injected message;
  configurable via `++tool_overrides.CallAgentTool.injection_role=system`
- New state dicts scoped by `request_id`:
  - `_pending_tasks`: `{request_id → {ticket_id → asyncio.Task[str]}}`
  - `_request_tickets`: `{request_id → [ticket_ids]}` for cleanup
  - `_ticket_meta`: `{ticket_id → {agent_name, task_snippet}}`
  - `_ticket_results`: `{ticket_id → str}` — cache after task drains
- `_fire_async`: calls `asyncio.create_task(_http_call(...))`, returns ticket JSON
- `_collect_results`: `asyncio.gather` on requested pending tasks
- `get_pending_injections`: drains `.done()` tasks, formats injection messages,
  caches results so `collect_results` can still serve already-injected tickets
- `cleanup_request`: cancels and awaits any still-running tasks; clears all per-request state

**`nemo_skills/inference/model/tool_call.py`** — `ToolCallingWrapper.generate_async`
and `_stream_single`: before each `model.generate_async(...)` call, iterate
`tool_manager.iter_tools()` and extend conversation with any pending injections.

**`nemo_skills/prompt/config/agents/orchestrator.yaml`** — updated to document
all three delegation tools, when to use each, and the auto-injection format.

### Key properties

- **Zero extra round-trips**: async results arrive as injected messages; the
  orchestrator never needs to poll.
- **Explicit fallback**: `collect_results` lets the orchestrator block on a
  specific ticket if it cannot proceed without it.
- **Per-request isolation**: all state is keyed by `request_id`; concurrent
  problems running on the same orchestrator process are fully independent.
- **Injection role is configurable**: use `"user"` (default, safe for all
  models) or `"system"` (semantically accurate but model-dependent).
- **Graceful cleanup**: `cleanup_request` (called in the `finally` block of
  `ToolCallingWrapper.generate_async`) cancels any unfinished background tasks,
  preventing resource leaks across problems.
