# Lazy-Allocation Agentic Swarm — System Handoff (SCI-548)

**Purpose.** Run the agentic Hermes swarm (orchestrator + `delegate_task` workers) over a full
benchmark (e.g. HLE, 2158 problems) across multiple seeds on a large model, **without** waiting for a
monolithic multi-node GPU reservation. GPU model-servers are submitted as small independent jobs that
**register dynamically** into a standalone router; the agentic CPU work is decoupled from the GPU block
and survives preemption / walltime caps via an automated supervisor + resume.

This doc describes every component, how they interact, the gotchas, and a copy-paste runbook so other
agents can extend the work in parallel.

---

## 1. Architecture at a glance

```
                         ┌─────────────────────────────────────────────┐
   CPU partition         │  co-located SERVICES job (1 cpu-long node)    │
   (cpu-long, 7-day)     │   • sgl-model-gateway ROUTER  :20000          │
                         │   • shared SANDBOX (jupyter)  :6000           │
                         │   publishes router_url + sandbox_addr → pool  │
                         └───────────────▲───────────────▲──────────────┘
                                         │ POST /workers  │ python exec
   GPU partition (batch, 4h cap)         │ (register)     │ (HTTP)
   ┌──────────────┐  ┌──────────────┐    │                │
   │ WORKER job 1 │  │ WORKER job N │  ──┘                │
   │ 2-node TP=8  │  │ 2-node TP=8  │  (sglang serves BF16, registers self,
   │ sglang BF16  │  │ sglang BF16  │   self-deregisters on exit)
   └──────────────┘  └──────────────┘
                                         ▲ /v1/chat/completions (round-robin)
   CPU partition (per-seed DRIVERS)      │
   ┌───────────────────────────────────────────────────────────────────┐
   │ DRIVER job, seed k  (cpu-short / cpu-normal, 4h)                    │
   │   ns hermes_agent_rollouts  (prehosted_url = router)               │
   │   ├─ orchestrator LLM ── routes to router ──┐                      │
   │   ├─ delegate_task children (process-mode) ─┘                      │
   │   └─ 3 local HTTP-MCP SIDECARS on 127.0.0.1: worklog / benchmark / │
   │      python  (python → shared sandbox; worklog/benchmark → disk)   │
   └───────────────────────────────────────────────────────────────────┘
                                         ▲
   Spark (laptop/devbox)                 │ submits + manages (ssh)
   ┌───────────────────────────────────────────────────────────────────┐
   │ SUPERVISOR (background): top worker pool to N; resume-restart any   │
   │ dead+incomplete driver; stop on all-done or stall. Runs `ns` here.  │
   └───────────────────────────────────────────────────────────────────┘
```

**One run = one shared services node + N GPU worker jobs + 5 driver jobs + 1 supervisor.** Workers serve
the model only (tool-agnostic), so the SAME pool serves the tool and no-tool arms.

---

## 2. Components

### 2.1 Router — `sgl-model-gateway` (smg)
- Cluster's "sglang-router 0.3.2" is actually **sgl-model-gateway**. Dynamic REST API on `:20000`:
  - `GET /workers` → `{workers:[...], total}`
  - `POST /workers` body `{"url":"http://host:port"}` → 202, async **health-gated**, returns `worker_id`
  - `DELETE /workers/<worker_id>` → removes (query-string DELETE 405s; must be path form)
  - OpenAI routes proxied at `/v1/...`; `policy=round_robin`.
- Launched by `router_pool/start_router.sh` (standalone) or as a step in `shared_services_bf16.sh`.
- **The request `model` must match the workers' served-model-name** (= the model path). Mismatch → 503.

### 2.2 Workers
- **BF16 (current):** `router_pool/bf16_worker.sh` — a **2-node** job (`#SBATCH --nodes=2
  --gres=gpu:4 --ntasks-per-node=1`). Runs `serve_sglang --num_nodes 2 --port 5656
  --dist_init_addr $SLURM_MASTER_NODE --node_rank $SLURM_PROCID --dtype bfloat16
  --expert-parallel-size 8 ...EAGLE` → TP=8 across both nodes (1.1T BF16 won't fit TP=4). The batch
  script runs on the head node (node_rank 0); a background registrar waits for `/health` then
  `POST /workers` the head URL; an EXIT trap `DELETE`s it.
- **NVFP4 (older, 1-node):** `router_pool/gpu_worker.sh` — `--nodes=1 --gres=gpu:4`, `--tp-size 4
  --quantization modelopt_fp4`. Same register/deregister pattern.
- Pool sizing: width ≈ workers × max-running-requests(8). Worker count affects **throughput only, not
  accuracy** — fine to shrink under contention.
- `router_pool/topup_workers.sh N` — tops the live (RUNNING+PENDING) BF16 worker count up to N (pure
  sbatch; run on the cluster login node). Workers self-heal: die at 4h → topup resubmits → re-register.

### 2.3 Services (router + sandbox), co-located
- `router_pool/shared_services_bf16.sh` — ONE `cpu-long` (7-day) `--exclusive` job runs BOTH the router
  and a shared sandbox as two `srun --overlap` steps. **STAGGERED** (sandbox first, settle 30s, then
  router) to avoid a dual-step-creation race that 0-byte'd the router previously. Footprint = 1 CPU node;
  leaves cpu-normal free for co-tenants. Publishes `router_url` + `sandbox_addr` to the pool dir
  (`/alaptev/data/lazypool_bf16/`).
- Alternatives: `start_router.sh` + `shared_sandbox.sh` (two separate jobs) — proven but uses 2 nodes.
- Shared sandbox = one jupyter/uwsgi server (`nemo-skills-sandbox` image, `/start-with-nginx.sh`),
  `NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT=600` reaps idle kernels so the live-session set stays bounded.

### 2.4 Driver — `ns hermes_agent_rollouts` (prehosted)
- One per seed, on a CPU node. The launcher (`nemo_skills/pipeline/hermes_agent_rollouts.py`) is
  **HTTP-decoupled**: with a manifest `prehosted_url`, the group is CPU-only (no in-job GPU server) and
  the orchestrator LLM + delegate children both route to that URL.
- `--no-with_sandbox` (sandbox is the shared external one), `--no-merge_back`, `--qos <cpu-short|cpu-normal>`.
- Driver walltime = cluster_config `timeouts.cpu` = **4h** regardless of QOS → every driver dies at 4h →
  the supervisor's resume-restart is **essential**.
- Gated launcher knobs (default-off, additive): `NS_PREHOSTED_PARTITION` / `NS_PREHOSTED_GPUS` (env) place
  the driver on a different partition / co-locate a server. Not used by the cpu-partition recipe.
- Wrapper: `bf16_run/launch_bf16_driver.sh <seed_k> <qos> [tool|notool]`.

### 2.5 Manifest (`workdir-agents/validation/batch_solve_a0py_bf16_manifest.yaml`)
- `prehosted_url: "$(cat /alaptev/data/lazypool_bf16/router_url)/v1"` (runtime shell expr).
- `model:` must equal the worker served-model-name (`/hf_models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`).
- `enabled_toolsets`: `delegation, mcp-worklog, mcp-benchmark, mcp-python`. `mcp_servers` point at the
  local sidecars `http://127.0.0.1:9101|9102|9103/mcp`.
- **No-tool arm** = `..._bf16_notool_manifest.yaml`: drop `mcp-python` from `enabled_toolsets` AND the
  `python:` mcp_server entry. (Same everything else.)

### 2.6 Sidecars (`router_pool/gym_sidecars.sh`)
- Stage-1 design: ONE set of shared HTTP-MCP tool servers per driver instead of one stdio MCP server per
  child (killed the 387-proc / 28 GB per-child MCP layer). Launches 3 servers on 127.0.0.1:
  - **worklog** (9101) + **benchmark** (9102): `python -m nemo_skills.mcp.http_serve
    <ToolClass>` (generic StreamableHTTP wrapper, `nemo_skills/mcp/http_serve.py`).
  - **python** (9103): `python -m nemo_skills.mcp.servers.python_tool --transport streamable-http`,
    pointed at the shared sandbox via `NS_SIDECAR_SANDBOX_POOL`.
- Run via the gated `NS_GYM_SIDECAR_SCRIPT` hook in `nemo_gym.py` (bg script + `.ready` sentinel wait).
- Per-seed wrappers `gym_sidecars_bf16_s{k}.sh` (tool) / `gym_sidecars_bf16nt_s{k}.sh` (no-tool) set
  `NS_SIDECAR_OUT_DIR`, `NS_SIDECAR_RUN_ID`, `NS_SIDECAR_SANDBOX_POOL`, `NS_BATCH_RESUME=1`.
- Uses `/usr/bin/python3 + PYTHONPATH=/alaptev/NeMo-Skills-clean` (the proven prod interpreter; the gym uv
  venv may lack mcp/starlette).

### 2.7 Orchestrator input prompt (`/alaptev/data/batch_solve_a0py_lazy_full_input.jsonl`)
- A rigid 5-step protocol: `clock_in("batch")` → `plan_batch(limit=2158)` → `delegate_task(handle,
  max_in_flight=128)` → `clock_off` → "Final Answer: batch complete". `goal_template` (per-problem)
  tells each worker to clock_in/get_problem/solve/clock_off-with-"Final Answer:".
- **GOTCHA: the input message MUST be `role: user`.** A `role: system` message makes Hermes append an
  empty user turn → the gateway 500s ("message content cannot be empty") → every rollout dies in ~1 min.
  (`batch_solve_a0py_lazy_full_input.jsonl` is the correct role=user file; `*_full2158_input.jsonl` is the
  bad role=system one — do not use it.)
- No-tool input `..._bf16_notool_input.jsonl`: same protocol, goal_template says "reason ONLY, no tools".

### 2.8 Resume (`NS_BATCH_RESUME`, default-off)
- `benchmark_tool._completed_ids_to_skip(ids)` (deployed on the cluster at `/alaptev/NeMo-Skills-clean`,
  **not in this repo checkout**): when `NS_BATCH_RESUME` is truthy, `plan_batch` reads the worklog dir
  (`NS_WORKLOG_DIR`/`NS_WORKLOG_RUN_ID`) and drops ids whose frontmatter `status: completed`. So a restart
  re-plans only the remainder. The sidecar wrappers set `NS_BATCH_RESUME=1`; `gym_sidecars.sh` passes the
  worklog env to the benchmark sidecar. Validated: restart after 11 done → dispatched 2147, 0/11 re-solved.

### 2.9 Supervisor (`bf16_run/bf16_supervisor.sh`) — the automation
- Background loop on the Spark (needs the `nemo-skills` conda env to run `ns`). Each round:
  1. `topup_workers.sh N` (keep the pool full),
  2. for each seed: count `status: completed` worklogs; if `< TARGET` and the driver job is dead,
     resume-restart it via `launch_bf16_driver.sh`,
  3. stop when all seeds hit TARGET, or when **no global progress for STALL_ROUNDS WITH ≥1 running
     worker** (the stall-guard is gated on running workers so GPU starvation ≠ convergence).
- **flock singleton** (`.supervisor.<variant>.lock`) — two supervisors would double-submit drivers.
- `VARIANT=tool|notool` env selects job-name prefix / out_dir / run_id / launch arg (arms share the pool).
- QOS layout: 4 drivers `cpu-short` (MaxSubmit=4) + 1 `cpu-normal` (leaves the 2nd cpu-normal for co-tenants).

### 2.10 Grading (`grading/`)
- Trust-boundary: the benchmark tool keeps `expected_answer` server-side; workers emit "Final Answer:" in
  their worklog. `fs_grade.py build` joins them by id (mode=full = judge the whole worker report, which
  recovers the ~40% of workers that state the answer in prose without a literal "Final Answer:" line).
- `grade_bf16.sh`: per seed, build → bridge to `output-rs{k}.jsonl` **filtering `worker_status==completed`**
  → `rejudge_hle_gpt4o.py --seeds 5` (gpt-4o API judge, key `JUDGE_API_KEY`, per-id cached).
- `bf16_final_tally.py`: full-2158 pass@1 (incomplete=wrong, standard HLE metric) AND completed-only,
  per-seed + mean±95%CI.

---

## 3. SLURM / QOS facts (aws-cmh) — drives the whole allocation strategy
- All partitions `OverSubscribe=NO` (a job = a whole node). `batch` (GPU) rejects 0-GPU jobs; MaxTime 4h.
  `batch_long` (7-day GPU) is **off-limits for our team**.
- CPU QOS: `cpu-long` node=1 / 7-day; `cpu-normal` node=2 / 1-day; `cpu-short` node=20 / 4h / MaxSubmit=4.
  ⇒ only **one** long-lived (>1-day) CPU node → services MUST co-locate there.
- Strategy: services → 1 cpu-long; 5 drivers → 4 cpu-short + 1 cpu-normal (leaves 1 cpu-normal for the
  co-tenant — never take both, it starves their `alaptev`-uid jobs); workers → `batch` (`normal` QOS,
  node=1000). Can't `scontrol update qos=` a RUNNING job.

---

## 4. Gotchas (hard-won — read before changing anything)
1. **Input role MUST be `user`** (§2.7) — else silent 1-min death of every rollout.
2. **Model name must match served-model-name** or the router 503s.
3. **Co-located services: stagger the two srun steps** (sandbox → settle → router); simultaneous launch
   0-byte's the router. Cold container import is minutes → health waits must be generous (600s).
4. **Synchronized worker expiry:** if all workers are submitted in one window they all hit the 4h cap
   together → full-pool outage. (Acceptable with the supervisor, but stagger submits to smooth it.)
5. **Saturated cluster:** replacement workers can sit PENDING (Priority) for hours; 16×2-node=32 nodes is
   a hard ask. Shrink the pool (8) — the remaining tail is latency-bound, not throughput-bound.
6. **Driver launched with 0 workers FAILS exit-1** (orchestrator's own LLM call 503s). Harmless — the
   supervisor restarts it; resume re-delegates the remainder.
7. **Grade cache-poisoning:** in-progress worklogs have NON-EMPTY partial bodies, so "skip empty
   generation" does NOT isolate completed problems — filter on frontmatter `status==completed`. A
   preliminary grade over partials poisons the per-id rejudge cache → delete cache + re-grade fresh, or
   only ever grade completed.
8. **Asymptotic tail:** ~5-6% of problems never reach `completed` (hit the 1800s child cap, retried each
   restart). Call convergence when rate < ~1.5/min rather than grind for tenths of a percent.
9. **Single-node consolidation is the WRONG architecture** for N≥3 drivers (concurrent stdlib imports off
   a shared FS on one node → spurious ModuleNotFoundError). 1 driver per node is the proven remedy. Do not
   revive the consolidated launcher (`cons5n_launch.sh`).
10. The `/root/.deno/env` "Permission denied" line in every ssh command is harmless `.bashrc` noise.

---

## 5. Runbook — launch a full 5-seed run

```bash
# All paths: cluster /alaptev = /lustre/fsw/portfolios/nemotron/users/alaptev ; ssh key in ~/.ssh/clusters/aws-cmh/
A=/lustre/fsw/portfolios/nemotron/users/alaptev ; RP=$A/router_pool

# 1. Services (router+sandbox) on cpu-long. Clear the pool dir first.
ssh …cmh "rm -f $A/data/lazypool_bf16/{router_url,sandbox_addr}; \
  POOL_DIR_HOST=$A/data/lazypool_bf16 sbatch --qos=cpu-long --time=72:00:00 \
  -J lazyalloc_bf16_services --export=ALL,POOL_DIR_HOST=$A/data/lazypool_bf16 $RP/shared_services_bf16.sh"
# wait until $A/data/lazypool_bf16/router_url + sandbox_addr exist and /workers responds.

# 2. Workers (8 here; 16 if the cluster is free). Each is a 2-node BF16 job.
ssh …cmh "bash $RP/topup_workers.sh 8"        # wait ~10-15 min for model load + POST /workers

# 3. Drivers + supervisor (on the Spark, nemo-skills conda env). VARIANT=tool or notool.
#    The supervisor launches all 5 drivers itself and keeps everything alive.
cd ~/Projects/bf16_run
VARIANT=tool WORKER_TARGET=8 STALL_ROUNDS=20 nohup ./bf16_supervisor.sh >/dev/null 2>&1 &
#    (it loops until 5×2158 completed or a real stall; logs to supervisor.<variant>.log)

# 4. Grade (after convergence; reads worklogs from disk + gpt-4o API — no compute needed)
ssh …cmh "cd $A/grading && bash grade_bf16.sh && python3 bf16_final_tally.py"
#    For the no-tool arm, point grade_bf16.sh at a0py_bf16nt_s{k}/nhwbf16nts{k} (or parameterize).

# 5. Tear down (free GPUs) — ONLY your jobs, never the co-tenant's mcp_ablation_*:
pkill -9 -f bf16_supervisor.sh
ssh …cmh "scancel \$(squeue --me -h -o '%i|%j' | grep '|lazyalloc_bf16' | cut -d'|' -f1)"
```

Monitoring helpers (Spark): `monitor_once.sh` (2h progress watcher), `recovery_long.sh` (waits for worker
backfill). All are read-only ssh loops.

---

## 6. File locations
- **Spark** `~/Projects/NeMo-Skills` (editable `nemo-skills` conda env — launcher edits are LIVE here):
  `nemo_skills/pipeline/hermes_agent_rollouts.py`, `…/utils/hermes_manifest.py`,
  `…/utils/scripts/nemo_gym.py`, `nemo_skills/mcp/http_serve.py`, `nemo_skills/mcp/servers/python_tool.py`.
  Manifests + mirrored scripts under `workdir-agents/validation/{,router_pool/,grading/,bf16_run/}`.
- **Spark** `~/Projects/bf16_run/` — the orchestration scripts actually run from here (supervisor, launcher,
  topup, monitors). Mirrored into the repo at `workdir-agents/validation/bf16_run/`.
- **Cluster** `/alaptev/router_pool/` — the worker/service/sidecar scripts the SLURM jobs execute.
- **Cluster** `/alaptev/NeMo-Skills-clean` — the prod interpreter for sidecars AND where the **resume**
  change (`benchmark_tool.py`) is deployed (`.bak.resume` backups). NOT in this repo checkout.
- **Cluster** `/alaptev/grading/` — grading pipeline + per-run judge caches.
- **Cluster** `/alaptev/exp/a0py_bf16_s{k}/` (tool) / `a0py_bf16nt_s{k}/` (no-tool) — worklogs/batch_plans.

---

## 7. Results so far (HLE full-2158, gpt-4o judge, 5 seeds)
- **BF16 + python (tool), agentic swarm:** full-2158 **26.15% ± 0.70**; completed-only 27.73% ± 0.74
  (94.3% completion; std ≈0.8 across seeds). 2026-06-26.
- No-tool BF16 arm: **in progress** (this is the apples-to-apples comparison for the tool-effect delta).
- Context (precision study, separate direct-tool harness): BF16 python +5pp vs no-tool; NVFP4 python −3.5pp.

## 8. Open / next
- Finish the no-tool BF16 arm → clean tool-effect delta with error bars.
- Optional: stagger worker submit times; fold `benchmark_tool` resume into this repo checkout; debug
  `shared_services.sh` router step for a 2-service single job; parameterize `grade_bf16.sh` for variants.
