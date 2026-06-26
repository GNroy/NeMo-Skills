# Lazy/dynamic allocation — plan for next session (SCI-548)

**Date written:** 2026-06-24  **Cluster:** aws-cmh  **Memory:** see [[project-lazy-allocation]], [[feedback-job-naming]]
**Companion docs (same dir):** `LAZY_ALLOCATION_DESIGN.md` (architecture + L0/L1 validation), this file (forward plan).

---

## UPDATE 2026-06-24 (cont) — STAGE 1 DONE + VALIDATED; s0 FD-wall found+fixed

**Stage 1 (shared HTTP-MCP sidecars) is BUILT, deployed, and VALIDATED LIVE** (job 667689,
limit-200 width-128, 16 workers). Per-child stdio MCP servers (387 procs / ~28 GB) → **3 shared
HTTP sidecars (387→0 stdio procs)**. Peak host mem **~156 GB vs ~168 baseline**; **oserr=0** (no FD
storm); throughput ~30+/min ≈ r4; **accuracy 33.0% pass@1 (gpt-4o, n=200) ≈ r4 31.0% → parity, no
regression**. All 3 tools share-safe per seed with ZERO tool-code change.
- New/changed (all additive, gated, default-off): `nemo_skills/mcp/http_serve.py` (generic
  StreamableHTTP wrapper, reuses `stdio_serve._build_server`); `python_tool.py:main()`
  `--transport streamable-http/--host/--port` (default stdio); `nemo_gym.py` `NS_GYM_SIDECAR_SCRIPT`
  hook (+ `ulimit -n` raise, see below); `router_pool/gym_sidecars.sh` (worklog:9101/benchmark:9102/
  python:9103 on 127.0.0.1 via `/usr/bin/python3`); `batch_solve_a0py_stage1_manifest.yaml` (url:).
  Deployed to cluster `/alaptev/NeMo-Skills-clean` (+ `.bak.httpserve`) + `/alaptev/router_pool/`.
- **Launch recipe:** `NS_GYM_SIDECAR_SCRIPT=/alaptev/router_pool/gym_sidecars.sh ns
  hermes_agent_rollouts ... --agent_manifest batch_solve_a0py_stage1_manifest.yaml` + a
  `lazypool_stage1` pool (router + `submit_chunk.sh 16 30001 stage1`).

**s0 full-HLE STALLED at 1176/2158 = the monolithic-driver FD WALL.** At full-2158 width-128 the
single driver node exhausts file descriptors/sockets (~1100 completions in): 2100+ `ClientOSError`
on `127.0.0.1:18333/run`, workers idle. SALVAGED → **25.19% pass@1 (gpt-4o, n=1175 completed)** =
partial full-HLE ref (front-loaded/harder than the 200-subset). **FIX:** `ulimit -n 1048576`
(best-effort) at the top of the gym driver script in `nemo_gym.py` (live in 667689). Stage 1 ALSO
relieves FD pressure (fewer per-child procs). Grading at cluster `/alaptev/grading/`.

**STAGE 2 (shared sandbox) — MEASURE gate PASSED + service BUILT 2026-06-24:** the sandbox is now
the dominant per-driver cost. **Measured** (job 668033, real container on a CPU node): idle sandbox =
**112.7 GB** (96 uwsgi workers preloading scipy/sympy — the multiplexable hog); per Jupyter
session/kernel ≈ **67-73 MB**; 400 concurrent sessions = +27 GB; **DELETE /sessions fully reclaims**
(after deleting 400, mem returned BELOW baseline). Cleanup lever = `NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT`
(env, **0=DISABLED by default** in the image → kernels accumulate the whole run = the stage1/s0 growth;
the container serves `local_sandbox_server.py` as /app/main.py so it honors this + `NEMO_SKILLS_SANDBOX_MEM_LIMIT`).
**Memory model, 5 seeds sharing ONE reaped sandbox:** ~112 GB preload + ~128 working kernels×73 MB ≈
**~121 GB shared** vs 5×120 ≈ 600 GB separate → **saves ~480 GB**; consolidated node ≈ 121 + 5×46 ≈
**351 GB → fits 370.** Thesis confirmed. **Built (no launcher change needed):** `router_pool/shared_sandbox.sh`
(standalone sandbox service, tunable UWSGI_PROCESSES + SESSION_TIMEOUT reaping, publishes `sandbox_addr`);
`gym_sidecars.sh` python sidecar points at an external sandbox via `NS_SIDECAR_SANDBOX_POOL`/`_ADDR` (else
in-job); `gym_sidecars_stage2.sh` wrapper; `batch_solve_a0py_stage2_manifest.yaml`. Driver uses
`--no-with_sandbox` (CONFIRMED drops the in-job sandbox srun: stage2 sbatch has 0 sandbox refs, only
bootstrap+gym groups). Measurement harness: `router_pool/measure_sandbox_sessions.sh`.

**Stage-2 PLUMBING VALIDATED LIVE (job 668154, limit-200, 8 workers, shared sandbox cpu-0097):**
driver launched `--no-with_sandbox` (no in-job sandbox srun) + `gym_sidecars_stage2.sh` →
`[sidecars] python -> SHARED sandbox cpu-0097:6000`; children executed code on the SHARED sandbox
(495+ `POST /mcp`), and **session reaping (SESSION_TIMEOUT=600) held active_sessions bounded at
~3-14** across the whole run (tracks the working set, NOT cumulative — the key correctness result),
oserr=0. The sandbox is fully DECOUPLED from the driver node. **Accuracy 32.54% pass@1 (gpt-4o,
n=169 completed, noans 0.6%) ≈ r4 31.0% ≈ stage1 33.0% → parity, no regression** (expected: Stage 2
only relocates the sandbox).

**MEMORY-METRIC CORRECTION (important):** `free used` is NOT a reliable per-job number on these
nodes (driver showed `used`=168 GB but process-RSS summed to only ~38 GB, `/dev/shm`=36K,
`available`=201 GB → the gap is container page-cache/accounting, not working set). Re-measure via
**process RSS**, not `free`. Real footprints: **driver node ~38 GB RSS** (hermes agents 28 + ray 6 +
sidecars 0.3 — sandbox-independent); **shared sandbox RSS is LOAD-DEPENDENT** ~33 GB (light) → 55 GB
(~10 concurrent sessions) → toward ~110 GB at full 96-worker load (the original PSS "110 GB" = the
fully-loaded figure). So the earlier `free`-based "112 GB idle / 156 vs 168 GB / saves 480 GB"
figures were CONFOUNDED; state Stage-2 savings as: sharing pays the sandbox's load-dependent
footprint ONCE for N seeds instead of N times, and reaping bounds sessions to the working set. Both
nodes had ample `available` (driver 201 GB, sandbox 331 GB) → not memory-constrained.

**MULTI-SEED VALIDATED LIVE 2026-06-24 (2 seeds mA+mB, jobs 668679/668683):** 2 seeds ran
CONCURRENTLY sharing ONE sandbox (cpu-0006) + ONE router/pool, each a SEPARATE driver job
(`--no-with_sandbox` + per-seed `gym_sidecars_<seed>.sh` wrapper → out_dir/run_id, pool
`lazypool_multi`). Both seeds' python sidecars → `SHARED sandbox cpu-0006` confirmed; sessions
reaped/bounded ~4-7 under 2-seed load; oserr=0. **Process-RSS: 1 shared sandbox ~22 GB + mA 38.0 +
mB 38.8 GB** → one sandbox serves both = the multiplexing win.
- **ARCHITECTURE DECISION:** chose **shared-infra + SEPARATE driver jobs** over the fragile
  single-mega-sbatch hand-templatizing (Part 3 "Consolidated launcher" below). Reuses Stage-2 verbatim,
  robust, fits ~3-4 seeds within CPU QOS (2 cpu-normal: sandbox+router; ≤4 cpu-short: drivers). The
  single-sbatch consolidation is only needed for **5+ seeds** (or cpu-long). Per-seed light sidecars
  bind 127.0.0.1 on their OWN node (no cross-seed port collision); only the heavy sandbox is shared.
- **LESSON (worker scaling):** 8 workers shared by 2 seeds = 256 children / 64 slots = too
  oversubscribed → very slow per-seed (no clean accuracy in reasonable time; mB n=53 early sample
  20.75% is biased-partial, NOT representative). Scale the shared pool **~8-16 workers/seed**.
- Artifacts: `batch_solve_a0py_multi_manifest.yaml`, `router_pool/gym_sidecars_{mA,mB}.sh`,
  `shared_sandbox.sh` (reused). The "Consolidated single-node launcher" design in Part 3 below is
  superseded for ≤4 seeds by this simpler approach; keep it for the 5+/single-job case.

**REVISED NEXT STEPS:** Stage 1 (Part 3 below) is DONE — skip to **Stage 2** (share sandbox+benchmark
per node; the 110 GB sandbox is now the dominant hog) + the **consolidated single-node launcher**,
then the **clean full-2158 multi-seed run** (now unblocked by the ulimit fix + Stage-1 FD relief) →
grade → mean±CI. Parts 3-6 below remain the reference; Part 6 step 1 (s0) and step 2 (Stage-1
prototype) are now COMPLETE.

---

## 0. TL;DR — where we are

Lazy/dynamic allocation is **built, validated end-to-end, and graded**. It runs the agentic
swarm with the CPU driver on the `cpu` partition + a standalone sgl-model-gateway router fed
by separately-submitted GPU workers that register dynamically — escaping the monolithic
17-node het-job wait (workers backfill in seconds–minutes vs hours).

The open thread is **scaling to parallel multi-seed runs** (e.g. full-HLE 5-seed) for fast
iteration. The blocker is NOT GPU — it's **CPU-job count (QOS caps)** and **per-driver memory
(~168 GB)**. The forward plan (Part 3) is a **shared-HTTP-tool-sidecar + consolidated
single-CPU-node** architecture that makes ~5 full-width seeds fit on one node = 1 SLURM job.

---

## 1. DONE & validated (do not redo)

- **L0/L1/L2 + e2e** (see LAZY_ALLOCATION_DESIGN.md): register→serve→deregister proven; the
  `prehosted_url` launcher patch is **live in the editable `nemo-skills` env** (no deploy).
- **r3** (8 workers, width-128, limit-200): 199/200 ✓. **r4** (16 workers, 1:1, limit-200):
  200/200 ✓, **graded 31.0% pass@1** (gpt-4o API judge, n=200, 4.5% no-answer).
- **Throughput:** ~27 problems/min steady-state at 16 workers (~2× the 8-worker rate; scales
  ~linearly with live replicas). Total wall-clock is **tail-bound** by 1-2 pathological
  reasoners hitting the 1800s child cap — pool-size-independent (lower `child_timeout` to fix,
  not more workers).
- **Two e2e bugs fixed (in `nemo_gym.py`, editable-live):** (a) prehosted server-wait now polls
  `/v1/models` for a real model, not mere router reachability; (b) orchestrator input must be
  `role: user` (gateway 400s the empty-user turn the agent appends after a system-only input;
  plain sglang tolerated it → why monolithic worked but lazy didn't).
- **Job naming:** `lazyalloc_` identifier; `submit_chunk.sh` takes a RUN_LABEL 3rd arg →
  `lazyalloc_worker_s{N}_p{port}`. ALWAYS pass per seed.
- **Grading pipeline works:** `fs_grade.py build` (default `mode=full` → feeds the judge each
  child's FULL report, so `Answer:`/`Final Answer:`/prose all extract) → bridge to rejudge
  format (`question`→`problem`, add `subset_for_metrics`) → `rejudge_hle_gpt4o.py
  --result-dir <dir>` (gpt-4o API judge; **NEW-judge section** is the real number, the OLD is
  empty if no prior `judgement` field). Endpoint `inference-api.nvidia.com/v1`, model
  `azure/openai/gpt-4o`, key env `JUDGE_API_KEY` (SECRET — ask user; was provided this session).
- **s0 full-HLE single-seed** (job 666927, 16 workers, width-128, full 2158) running at writing
  (~914/2158). When done: grade it (fs_grade build over its worklogs → rejudge) = the full-HLE
  single-seed reference number. **Tear down its pool (router 665483-era + 16 workers) after.**

---

## 2. Plumbing-shake findings (the constraints the plan must respect)

**A 5×16 parallel attempt this session surfaced hard limits — all CPU-side, not GPU:**

- **GPU scales fine:** `normal` QOS = node=1000/user. 80 workers OK (modulo GPU availability;
  16 backfilled in ~10 min during contention).
- **CPU-job count is capped by QOS** (the lazy design puts BOTH router and driver on CPU):
  | QOS | MaxJobs(running) | MaxSubmit | Wall |
  |---|---|---|---|
  | cpu-normal (default) | **2** | 200 | 1 day |
  | cpu-short | 20 | **4** | 4 h |
  | cpu-long | **1** | 100 | 7 day |
  Concurrent-CPU-job ceiling ≈ **6** (2+4 usable). 5 seeds need 2N=10 CPU jobs → **impossible
  as-architected.** Pass `--qos cpu-short` (CLI flag exists, flows to prehosted sbatch_kwargs;
  trim walltime ≤4h via `scontrol update ... TimeLimit=04:00:00`); `scontrol update JobId=X
  qos=cpu-short` flips a pending job.
- **Submit-burst drop:** 80 concurrent `sbatch &` dropped later seeds (assoc/rate limit under
  `set -e`). Submit per-seed sequentially with a `sleep`.
- **Per-driver MEMORY ≈ 168 GB PSS** on the 96-core/370 GB node → only ~2 full-width
  drivers/node. **CPU is NOT the constraint** (load ~6.6/96 — children are GPU-I/O-bound).
  Memory breakdown (PSS):
  | category | PSS | what |
  |---|---|---|
  | **sandbox uwsgi** | **110 GB** | 96 workers (=nproc) each preloading scipy/sympy ≈1 GB |
  | MCP stdio | 28 GB | per-child worklog/benchmark/python stdio servers (3/child, 387 procs) |
  | gym head + Ray | 27 GB | ng_run + 3 gym servers + Ray |
  | hermes children | ~0 | the agent loops are cheap |
- Sandbox knobs (Dockerfile.sandbox, `tiangolo/uwsgi-nginx-flask`): `UWSGI_PROCESSES` (worker
  cap), `UWSGI_CHEAPER` (lazy scale + idle reap). Capping risks a python-queue at peak IF
  concurrent-exec ≈ 96 (the 96 workers DID get spawned → peak demand was high) — so capping is
  not guaranteed free; cheaper-mode + a moderate cap is the safer combo.

---

## 3. THE PLAN — shared-sidecar + consolidated single-node (start here)

**Goal:** run an N-seed (e.g. 5) full-HLE experiment as **ONE CPU SLURM job** (all routers +
drivers on one node) so parallelism is gated only by "how many CPU nodes," picking the CPU QOS
by walltime estimate (cpu-short ≤4h / cpu-normal ≤1d / cpu-long ≤7d). This bypasses the 2N-CPU-job
ceiling AND, with the memory work below, fits ~5 full-width seeds on one 370 GB node.

**Verified enabler:** the hermes-agent MCP client speaks **HTTP/StreamableHTTP + SSE**
(`mcp_tool.py:5-6,51` — config by `url:` not `command:`). The python sandbox is ALREADY a shared
HTTP service (`python_tool.py` httpx → sandbox). Precedent: the keyless websearch sidecar
([[project-search-sidecar]]) — a thin FastAPI HTTP backend consumers hit by base-URL.

### Stage 1 — tools as shared HTTP sidecars (per seed). LOWER RISK, do first.
Serve worklog/benchmark/python-MCP as **HTTP MCP services** and point children at them by `url:`
instead of `command:` (stdio). Effects:
- Kills the **28 GB / 387-proc** per-child MCP overhead → ~3 servers.
- **Removes process-mode's per-process MCP registration** (delegate_tool.py increment-3) — children
  become uniform thin HTTP clients (like they already are for the LLM router + sandbox). Net
  simplification + lighter children.
- Keeps per-seed isolation (no shared-failure risk yet).
- **Measure** the memory delta vs the current s0 baseline before going further.

Build: wrap `nemo_skills.mcp.servers.{worklog_tool,benchmark_tool,python_tool}` as concurrent
HTTP-MCP servers (FastMCP/SSE). worklog/benchmark are stdio modules today; python is already an
httpx client to the sandbox — just needs HTTP-MCP serving. Manifest `mcp_servers` entries switch
from `command:`+`args:` to `url:` (+ `transport: sse` if needed).

### Stage 2 — share sidecars per NODE (across seeds). The big multiplexing win.
- **benchmark:** read-only → one shared sidecar for all seeds on the node.
- **sandbox:** session-isolated already → one shared sidecar; **statistical multiplexing** of
  bursty python-exec across seeds means ~96-128 workers serve all 5 (vs 5×96) → ~110 GB once,
  not 5×. Tune `UWSGI_PROCESSES`/`UWSGI_CHEAPER` here (scoped to the run, NOT shared cluster
  config — other agents unaffected).
- **worklog:** needs per-seed namespacing (run_id) → either one sidecar that takes run_id
  per-call, or one lightweight worklog sidecar per seed.
- **MEASURE FIRST:** per-Jupyter-session memory in the sandbox (5×128=640 sessions on one pool —
  could add up). And accept the shared-sandbox single-point-of-failure trade-off for density.

### Consolidated single-node launcher (the orchestration piece)
One `sbatch` CPU job (QOS by walltime estimate) whose body:
1. starts the shared sidecars (1 sandbox + 1 benchmark + N worklog) as background HTTP services
   on the node (distinct ports);
2. starts N routers (background, ports 20000+s) — OR keep routers separate (they're light);
3. starts N gym-driver processes (background), each = the gym rollout command for seed s pointed
   at its router + the shared sidecar URLs + output_dir_s. **Reuse the gym command** — read it
   from a live run's generated `nemo-run/scripts/nemo-run-2.sh` and templatize (POOL_DIR,
   output_dir, HERMES_HOME, ports, sidecar URLs); do NOT hand-reinvent the venv/ng_run wiring.
4. waits on all N; each driver self-gates on its router's `/v1/models`.
GPU workers stay as N×16 separate jobs (normal QOS) via `submit_chunk.sh N 30001 s{seed}`.

**Memory projection (5 seeds/node):** 1 shared tuned sandbox (~50-110 GB) + benchmark (~0.5) +
5 light worklog + 5 drivers of thin children (~10-30 GB each) ≈ **150-250 GB → fits 370 GB.**

---

## 4. Cheaper alternative (if the sidecar build isn't worth it now)

UWSGI worker-tuning ONLY + pack **2-3 full-width seeds/node across 2-3 CPU jobs** (fits QOS:
2 cpu-normal + 1 cpu-short). No refactor; ~2h iteration today; 5 seeds = 2-3 CPU jobs. Worse
than the sidecar end-state but immediate. Decide Stage-1-build vs this based on how often
multi-seed full-HLE iteration is needed (user signalled: often → sidecar worth it).

---

## 5. Operational reference

- **SSH:** `ssh -i ~/.ssh/clusters/aws-cmh/id_ecdsa alaptev@aws-cmh-slurm-1-login-01.nvidia.com`.
  `/alaptev` = container mount; on login node it's `/lustre/fsw/portfolios/nemotron/users/alaptev`.
- **Harness:** `workdir-agents/validation/router_pool/{start_router,gpu_worker,submit_chunk,pool_status,loadtest}.sh,_memprofile.sh`
  (deployed to cluster `/alaptev/router_pool/`). Pool dirs `/alaptev/data/lazypool[_s{N}]/` (router_url published there).
- **Manifests:** `batch_solve_a0py_lazy_manifest.yaml` (base, prehosted), `..._s{0-4}_manifest.yaml`
  (per-seed). **Inputs:** `batch_solve_a0py_lazy_input_v2.jsonl` (limit-200, role=user),
  `batch_solve_a0py_lazy_full_input.jsonl` (limit-2158).
- **Launch a driver:** `ns hermes_agent_rollouts --cluster aws-cmh --agent_manifest <m> --input_file <in>
  --output_dir /alaptev/exp/<x> --expname lazyalloc_<x> --no-merge_back --qos cpu-short
  --server_container .../sglang-v0.5.11.sqsh --gym_container .../nemo-skills-dc43f3e.sqsh
  --sandbox_container .../nemo-skills-sandbox-dc43f3e.sqsh --hermes_agent_path /alaptev/hermes-agent
  --gym_path /alaptev/NeMo-Gym`. Env: `NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1`,
  `NEMO_SKILLS_SANDBOX_HOST='${SLURM_MASTER_NODE_HET_GROUP_0:-localhost}'`. Run from `nemo-skills` conda env.
- **Patched launcher files** (editable `/home/alaptev/Projects/NeMo-Skills`, live in env):
  `nemo_skills/pipeline/utils/hermes_manifest.py` (prehosted_url field + ServerGroup.is_prehosted),
  `.../hermes_agent_rollouts.py` (prehosted branch: CPU group, server_address, delegation literal),
  `.../utils/scripts/nemo_gym.py` (server_address + delegation_base_url_literal + /v1/models waits).
- **GOTCHAS:** fresh output_dir per run (stale `.gym_done` first-exit-kills-all); never scancel by
  `squeue --me` alone (other agents share the `alaptev` uid — their jobs are `mcp_aworkflow_*`/
  `mcp_ablation_*`; cancel by precise jobid or `lazyalloc_*` name); cluster_configs gitignored
  (never `git add -f`); hermes-agent edits unsigned + NOT pushed to public NousResearch remote.

---

## 6. Immediate next steps (next session, in order)

1. **Finish s0:** confirm 666927 hit 2158/2158, grade it (full-HLE single-seed reference number),
   then tear down its pool (router + 16 workers).
2. **Stage 1 prototype:** stand up worklog/benchmark/python as HTTP-MCP sidecars for ONE seed;
   wire children by `url:`; run a width-128 seed; **measure the memory delta** (expect ~168 → ~140
   GB from killing the 28 GB MCP) + confirm throughput holds (~27/min).
3. **Sandbox tuning experiment:** UWSGI_PROCESSES/CHEAPER on the shared sandbox; measure the
   memory-vs-throughput curve (find the cap that holds ~27/min).
4. **Measure per-session sandbox cost** (gate for Stage-2 per-node sharing).
5. If Stage 1 + tuning land the projected ~90-110 GB/driver → **build the consolidated single-node
   launcher** and run a 2-seed-on-one-node validation, then scale to 5.
6. Then the real deliverable: **full-HLE 5-seed run** (parallel on 1-2 nodes) → grade all 5 with
   the gpt-4o judge → **mean ± CI**. (Seeds: unpinned worker RNG gives natural variance; pin
   `--random-seed` per seed for reproducibility if needed.)
