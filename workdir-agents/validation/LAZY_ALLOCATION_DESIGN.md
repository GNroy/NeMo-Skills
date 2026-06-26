# Lazy / dynamic allocation for the agentic Hermes swarm (SCI-548)

**Date:** 2026-06-24  **Cluster:** aws-cmh (GB300, 4 GPU/node, 4h batch invariant)

## Problem
The swarm runs as ONE het SLURM job: `g0-0` (1 GPU node = orchestrator's own sglang
server) + `g0-5` (16 GPU nodes = worker sglang_router pool, 1 node = 1 replica + EAGLE).
That 17-node monolith with cross-group co-scheduling sits behind ~69 competing 16-node
jobs; the decisive width-128 run (663522) was planned to start ~5h after submit. The
GPU block is the bottleneck, not the agentic work.

## Key facts that make decoupling possible
1. **The agentic driver is already CPU-bound + HTTP-decoupled.** In
   `hermes_agent_rollouts.py` the worker group is a `delegation_pool` => SERVER-ONLY
   (no gym head). The orchestrator loop + `delegate_task` children + ProcessPoolExecutor
   workers + MCP + sandbox all run on the orchestrator's node and reach the worker LLMs
   purely over HTTP via `DELEGATION_BASE_URL`. With process-mode, the workers are CPU
   subprocesses. **The GPU nodes only serve the LLM.**
2. **External-URL escape hatch already exists.** `NemoGymRolloutsScript` accepts either
   `server` (a ServerScript resolved via het-group `hostname_ref()`) OR a plain
   `server_address` string (nemo_gym.py:91/107/135). `DELEGATION_BASE_URL` is just an
   env var exported in the command body (nemo_gym.py:182-188).
3. **aws-cmh has a 7-day `cpu` partition** (66 nodes, 96 CPU, 245 GB RAM, no GPU; ~40
   idle, ~17 pending => lands in seconds). Perfect home for the agentic driver — config
   caps it at 4h (`timeouts.cpu`) but the partition allows 7 days.
4. **The cluster "sglang-router 0.3.2" is actually `sgl-model-gateway` (smg)** — a
   newer rewrite. The classic `/add_worker`/`/remove_worker` routes are GONE (404).
   Dynamic worker mgmt is a REST `/workers` API (CONFIRMED on cluster, job 664790):
   - `GET /workers` -> `{"workers":[...],"total":N,...}`
   - `POST /workers` body `{"url":"http://host:port"}` (Content-Type json) -> **202**,
     returns `worker_id` + `location:/workers/<id>`. ASYNC, health-gated (a URL that
     fails /health never enters the list).
   - `DELETE /workers/<worker_id>` removes it (query-string DELETE -> 405).
   - OpenAI-compatible: `/v1/chat/completions`, `/v1/models` routed to workers.
   NeMo-Skills only uses static `--worker-urls` today (serve_sglang_router.py:96-103);
   the standalone-router + dynamic-POST path is unused but fully supported.

## Target architecture (lazy/dynamic)
```
            CPU node (cpu partition, 7-day, instant alloc)
        ┌──────────────────────────────────────────────────┐
        │  standalone sglang_router  (CPU, :PORT)            │
        │     ^  /add_worker  /remove_worker                 │
        │  gym hermes driver (orchestrator agent loop)       │
        │     + delegate_task ProcessPoolExecutor workers    │
        │     + MCP (worklog/benchmark/python) + sandbox     │
        │  server_address = http://localhost:PORT/v1         │
        │  DELEGATION_BASE_URL = http://localhost:PORT/v1    │
        └──────────────────────────────────────────────────┘
                         ^            ^            ^
        register /add_worker on boot, /remove_worker on exit
        ┌──────────┐ ┌──────────┐ ┌──────────┐   ... chunked, INDEPENDENT
        │ GPU job1 │ │ GPU job2 │ │ GPU job3 │       1-node sglang servers
        │ 1 node   │ │ 1 node   │ │ 1 node   │       (batch partition, 4h)
        └──────────┘ └──────────┘ └──────────┘
```
- Orchestrator's OWN LLM + delegate children BOTH hit the standalone router => one URL,
  no dedicated orchestrator GPU node (folds idea #1 in: 17 GPU nodes -> N worker nodes).
- GPU servers submitted as **independent 1-node (or small-chunk) jobs**. Each backfills
  into scheduling holes a 16-node reservation can't use => lands far sooner; the run
  STARTS as soon as the first replica registers and scales as more land.
- Graceful degrade: a preempted/expired GPU job `/remove_worker`s itself; throughput =
  f(live replicas). 4h GPU walltime stops mattering — replicas churn under a 7-day driver.

## Build plan (increments)
- **L0 (verify):** confirm `/add_worker` + `/remove_worker` + `/list_workers` routes on
  the cluster router 0.3.2. [in progress]
- **L1 (router harness):** `router_pool/` scripts —
  - `start_router.sh`  : CPU job; launch_router with empty workers; write
    `router_url` to a shared lustre file; idle-wait.
  - `gpu_worker.sh`    : 1-node sglang server job; on READY POST /add_worker (url from
    file); trap EXIT -> /remove_worker.
  - `submit_chunk.sh N`: sbatch N independent gpu_worker jobs.
  - `pool_status.sh`   : curl /list_workers.
- **L2 (driver on CPU, external URLs):** launch the gym hermes driver server-less on the
  cpu partition with `server_address` + `DELEGATION_BASE_URL` = router_url. Either
  (a) minimal launcher patch: manifest agent `server_type: external` + `prehosted_url`
  => CPU-only group, pass server_address, export DELEGATION_BASE_URL; or
  (b) standalone `ng_run`/`ng_collect_rollouts` invocation outside the het launcher.
- **L3 (dynamic width):** orchestrator scales `delegate_task(max_in_flight)` to live
  replica count (poll /list_workers); start on first replica.
- **L4:** deploy + run; compare time-to-first-token and time-to-completion vs the
  monolithic het job.

## VALIDATION RESULTS (2026-06-24)
- **L0 + L1 PROVEN live on aws-cmh** (router 664799 CPU, worker 664800 GPU):
  - CPU router landed in **seconds** on the cpu partition; published router_url.
  - A **1-node GPU worker also reached RUNNING in seconds** (backfilled) — vs the
    16-node block's multi-hour wait. The chunking premise is confirmed.
  - register: model up in ~7 min, POST /workers (202) -> router health-gated -> added
    (`total:1, is_healthy:true`).
  - serve: 16 concurrent /v1/chat/completions load-balanced to the worker, **correct
    answers** (1*7=7 ... 15*7=105) in 2.8-3.7s.
  - deregister: scancel -> worker EXIT trap fires `DELETE /workers/<id>` -> `total:0`.
- **L2 launcher patch validated**: `prehosted_url` field (additive, gated, default None)
  in hermes_manifest.py + hermes_agent_rollouts.py + nemo_gym.py. Unit test: lazy
  manifest -> single group het=0 is_prehosted=True server_gpus=0. `ns
  hermes_agent_rollouts --dry_run` -> "Pipeline validated successfully" as a SINGLE
  CPU job (not het). Editable install -> live in the `nemo-skills` env, no deploy.
- Harness: `router_pool/{start_router,gpu_worker,submit_chunk,pool_status}.sh` +
  `loadtest.py`; lazy manifest `batch_solve_a0py_lazy_manifest.yaml`. All on cluster
  at `/alaptev/router_pool/` + `/alaptev/data/lazypool/`.

## End-to-end run gotcha (r1, FIXED)
First end-to-end attempt (driver 664883) launched while the 8 GPU workers were still
loading (cold container extraction on fresh GPU nodes ≈6 min BEFORE model load even
begins → ~12-15 min to healthy, not ~7). The orchestrator started against an EMPTY pool
and every call 500'd (`messages: Validation error: message content cannot be empty` —
the smg gateway's response when 0 healthy workers, NOT a request-format bug; the
monolithic procdemo proves the Hermes request format is fine). Root cause: the gym's
generic server wait (`get_server_wait_cmd` = `curl -X PUT`, passes on mere
reachability) treats "router up" as "ready", but a dynamic router is reachable with 0
models. FIX (in nemo_gym.py, both the orchestrator-server wait and the delegation wait
for the prehosted/literal case): `until curl -sf <url>/models | grep -q '"id"'; do
sleep 5; done` — wait for a REAL model entry. This also IS the "start when the first
replica lands" gate. Operationally also bring the pool to ≥1 healthy before/at launch.

## Risks / open
- Router host stability: CPU job host fixed for its lifetime; workers read host from the
  shared file. If the driver CPU job dies, the router dies — acceptable (it's the driver).
- add_worker readiness: register only after the server passes /health (else router marks
  it unhealthy). Use --worker-startup-timeout.
- EAGLE / server_args identical to current worker pool; sandbox is HTTP (shared).
- Trust boundary unchanged (benchmark tool never returns answers).
