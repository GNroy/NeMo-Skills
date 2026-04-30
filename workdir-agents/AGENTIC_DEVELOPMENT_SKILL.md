# Agentic Development Skill

Protocol for iterative development and evaluation of NeMo-Skills multi-agent
pipelines.  Follow this skill whenever working on `inference_scripts/agentic/`
or the supporting infrastructure (`agent_generate.py`, `agent_server.py`,
`tool_call.py`, `agent_tool.py`, prompt configs, `ns agent` pipeline).

---

## 0. Environment setup (every session)

```bash
# Activate conda — required before running any ns/inference_scripts command
source ~/miniconda3/bin/activate

# Verify git state — NeMo-Run packages the repo; tracked files must be clean
cd /home/alaptev/Projects/NeMo-Skills
git status
```

Commit any tracked-file changes before submitting:
```bash
git -c commit.gpgsign=false commit -m "..."
```

(GPG signing is unavailable in this environment; never use `--no-verify`.)

---

## 1. Iteration loop

```
 ┌─► make code changes
 │   commit (clean working tree)
 │   submit experiment  ──► new --run-id
 │   wait for SLURM jobs to finish
 │   analyze results
 │   write experiment report
 │   propose next change
 └───────────────────────────────┘
```

Each iteration gets its own `--run-id` (monotonically increasing integer).
Never reuse a run-id; resubmitting over existing output silently reuses
partial results (`skip_filled=True` by default).

---

## 2. Submitting experiments

### Main script

```bash
source ~/miniconda3/bin/activate   # if not already done

# Single-agent
python inference_scripts/agentic/run_nano_physics_agent.py \
    --run-id N --benchmarks physics,fs

# Multi-agent (orchestrator + solver worker)
python inference_scripts/agentic/run_nano_physics_agent.py \
    --run-id N --benchmarks fs \
    --worker-model /hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --worker-names solver

# Single benchmark, dry run (verify config without submitting)
python inference_scripts/agentic/run_nano_physics_agent.py \
    --run-id N --benchmarks physics --dry-run
```

### Cluster

Cluster: `oci` (default).  SSH for manual inspection:
```bash
ssh -i /home/alaptev/.ssh/clusters/oci/id_ecdsa \
    alaptev@draco-oci-login-01.draco-oci-iad.nvidia.com
```

### Output layout (on cluster)

```
/lustre/fsw/portfolios/llmservice/users/alaptev/exp/eval/nano_physics_agent/
└── agent_{bench}-{model_short}-r{N}/
    ├── tmp-eval-results/{benchmark}/     ← raw output JSONL from agent runs
    │   ├── output-rs0.jsonl              ← one file per random seed
    │   └── output-rs0.jsonl.done         ← sentinel; skip_filled reads these
    ├── eval-results/{benchmark}/
    │   ├── summarized-results/
    │   │   └── main_*_srun.log           ← accuracy table (primary metric)
    │   └── metrics.json
    └── agent-logs/
        ├── main_agent_*_srun.log         ← orchestrator stdout
        └── agent_worker_*_srun.log       ← worker stdout
```

---

## 3. Analyzing results

### Summary accuracy table

```bash
ssh -i ~/.ssh/clusters/oci/id_ecdsa \
    alaptev@draco-oci-login-01.draco-oci-iad.nvidia.com \
    "cat /lustre/.../agent_{bench}-{model_short}-r{N}/eval-results/{benchmark}/summarized-results/main_*_srun.log"
```

Key metrics from the table: `pass@1[avg-of-5]` (`judge_correct`),
`majority@5`, `pass@5`.  Compare against the baseline run.

### Tool usage & failure modes

Run a Python snippet over the output JSONL via SSH:

```bash
ssh ... "python3 -c \"
import json

with open('.../tmp-eval-results/{benchmark}/output-rs0.jsonl') as f:
    lines = f.readlines()

# Categorise per problem
errors, length_hits, tool_ok = 0, 0, 0
for line in lines:
    rec = json.loads(line)
    conv = rec.get('conversation', [])
    has_tool = any(t['role'] == 'tool' for t in conv)
    if not has_tool:
        if 'length' in str(rec.get('finish_reason', '')):
            length_hits += 1
        else:
            tool_ok += 1
    else:
        has_err = any('Error from agent' in str(t.get('content',''))
                      for t in conv if t['role'] == 'tool')
        if has_err: errors += 1
        else: tool_ok += 1

print(f'errors={errors}  length={length_hits}  ok={tool_ok}  total={len(lines)}')
\""
```

Key things to check:

| Signal | Meaning |
|--------|---------|
| `Error from agent '...'` in tool result | Worker crash — check worker log |
| `finish_reason: length` with no tool call | Context exhausted in thinking before first tool call |
| `num_tool_calls: 0` across all seeds | Orchestrator self-solving (check `enable_thinking`) |
| `call_*_async` in tool names | Non-blocking delegation working |
| `avg_tokens` increased vs baseline | Orchestrator actually calling workers |
| `gen_seconds` increased vs baseline | Multi-turn overhead (expected with delegation) |

### Reading a specific conversation

```bash
ssh ... "python3 -c \"
import json
with open('.../output-rs0.jsonl') as f:
    rec = json.loads(f.readline())
for i, t in enumerate(rec['conversation']):
    tc = t.get('tool_calls', [])
    if tc:
        print(f'Turn {i} [{t[\"role\"]}]: tool_calls={[x[\"function\"][\"name\"] for x in tc]}')
    else:
        print(f'Turn {i} [{t[\"role\"]}]: {str(t.get(\"content\",\"\"))[:200]}')
\""
```

### Agent logs

```bash
# Errors, exceptions
ssh ... "grep -i 'error\|exception\|traceback' .../agent-logs/main_*_srun.log | head -20"
ssh ... "grep -i 'error\|exception\|traceback' .../agent-logs/agent_worker_*_srun.log | head -20"

# Inference config (verify parameters went through correctly)
ssh ... "grep 'Agent.*config\|InferenceConfig' .../agent-logs/agent_worker_*_srun.log | head -5"
```

---

## 4. Experiment report

Write one report per submitted run (or per meaningful result).  Save it in
`workdir-agents/reports/` as `r{N}_{benchmark}.md`.

### Report template

```markdown
# Experiment r{N} — {benchmark}

## Setup

| Field | Value |
|-------|-------|
| Run ID | r{N} |
| Benchmark | {benchmark} |
| Mode | single-agent / multi-agent (orchestrator + {worker_names}) |
| Model | {model_short} |
| Key config changes | {what changed from the previous run} |
| Commit | {git log --oneline -1} |
| SLURM jobs | {job IDs from submission output} |

## Results

| Run | Mode | pass@1 (judge) | majority@5 | pass@5 | avg_tokens | gen_seconds |
|-----|------|----------------|------------|--------|------------|-------------|
| r{baseline} | {mode} | {%} ± {%} | {%} | {%} | {N} | {N} |
| r{N}        | {mode} | {%} ± {%} | {%} | {%} | {N} | {N} |

## Analysis

{2–5 sentences explaining *why* the numbers look the way they do.  Include:
- tool call statistics (% async, % blocking, % no-call, % error)
- failure modes observed
- token budget behaviour}

## Notes

{Optional: edge cases, surprises, things to watch in the next run}

## Next steps

- {concrete change to try in r{N+1}}
```

---

## 5. Known gotchas

See `workdir-agents/NOTES.md` for full context on each item.

| # | Symptom | Fix |
|---|---------|-----|
| 1 | `rsync: host key verification failed` | `ssh-keygen -R <stale-ip>` |
| 2 | `ConfigKeyError: Key '...' is not in struct` | `nested_dataclass` MRO bug — walk full MRO in `check_class.__annotations__` |
| 3 | `KeyError: 'messages'` on benchmark | `prompt_format='openai'` without `prompt_config` — use `++prompt_config=generic/default` |
| 4 | `KeyError: 'problem'` at import time | `{word}` in a `nested_dataclass` comment — use prose instead |
| 5 | `ImportError: pipeline requires full package` | `source ~/miniconda3/bin/activate` |
| 6 | `gpg failed to sign the data` | `git -c commit.gpgsign=false commit ...` |
| 7 | Partial results silently reused | Change `--run-id` or wipe output dir before resubmitting |
| 8 | Worker crash: `got multiple values for keyword argument 'endpoint_type'` | `asdict(InferenceConfig)` includes `endpoint_type` — pop it before `generate_async()` call in `agent_server.py` |
| 9 | Orchestrator never calls worker (0% tool calls) | `enable_thinking=True` fills 131k token budget in `<think>` before tool call — disable thinking for orchestrator in multi-agent mode |
| 10 | `tool_choice=required` has no effect | Only works if model doesn't exhaust token budget first; combine with `enable_thinking=False` for orchestrator |
| 11 | `KeyError: '"ticket_id"'` | `{...}` in YAML system prompt interpreted by `str.format()` — escape as `{{...}}` |

---

## 6. Important file locations

| File | Purpose |
|------|---------|
| `inference_scripts/agentic/run_nano_physics_agent.py` | Main experiment script — model config, benchmark config, submission logic |
| `nemo_skills/inference/agent_generate.py` | Orchestrator batch task loop |
| `nemo_skills/inference/agent_server.py` | Worker HTTP server |
| `nemo_skills/inference/model/tool_call.py` | `ToolCallingWrapper` — ReAct loop, first-turn `tool_choice` enforcement |
| `nemo_skills/mcp/servers/agent_tool.py` | `CallAgentTool` — async delegation, injection, cleanup |
| `nemo_skills/prompt/config/agents/orchestrator.yaml` | Orchestrator system prompt |
| `nemo_skills/prompt/config/agents/code_agent.yaml` | Worker system prompt |
| `cluster_configs/oci.yaml` | Cluster config (SSH, mounts, account, timeouts) |
| `workdir-agents/NOTES.md` | Failure postmortems and non-obvious details |
| `workdir-agents/agent_integration_plan.md` | Architecture reference |
| `workdir-agents/reports/` | Per-run experiment reports |
