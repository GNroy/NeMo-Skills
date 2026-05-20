# Hermes integration — Phase 6 validation experiment

A three-run A/B/C study of the Hermes ↔ NeMo-Skills integration on the
`frontierscience-olympiad` benchmark (100 problems across chem / bio /
physics).  The question we want to answer:

> Does the agent's memory/skill update step (Phase 1 merge-back) cause
> a measurable accuracy improvement on a hard scientific-research
> benchmark when the agent has every tool/MCP available?

## Layout

```
workdir-agents/validation/
├── frontierscience_manifest.yaml       # single Hermes agent + Kimi-K2.6 + all tools
├── data/
│   ├── README.md                       # how to materialize the full 100-sample input
│   └── smoke_5.ns.jsonl                # 5 hand-written NS-shape problems for wiring smoke
└── scripts/
    ├── convert_ns_to_ng.py             # NS frontier-science → NeMo-Gym input
    ├── convert_ng_to_ns.py             # NeMo-Gym rollouts.jsonl → NS-judgeable
    ├── run_validation.py               # submits Run 0 / Run 1 / Run 2
    ├── submit_judges.py                # chains gpt-oss-120b judge after each run
    └── compare_runs.py                 # parses judgements + emits the delta table
```

## A/B/C design

| Run | Template at start | `persist_memory` | `merge_back` | Template at end |
|---|---|---|---|---|
| Run 0 (baseline)    | frozen seed              | **false** | **off** | unchanged       |
| Run 1 (learning)    | frozen seed              | true      | **on**  | **merged**      |
| Run 2 (evaluation)  | Run 1's merged template  | true      | **off** | unchanged       |

Key reads:

- `Run 2 - Run 0` — **net effect of the memory update** (cold vs warm).
- `Run 1 - Run 0` — intra-run learning happening *during* the warm
  pass; not the headline number because the agent's HERMES_HOME is
  itself drifting through Run 1.
- `Run 2 - Run 1` — stability between two warm runs.  Large means the
  curator step is mutating the template in non-converged ways.

## Cluster + model requirements

| Cluster | Model path | vLLM container | GPUs | Notes |
|---|---|---|---|---|
| `aws-cmh` | `/hf_models/Kimi-K2.6` | `vllm-glm51-cu130-ray.sqsh` (mirror Oleksii's) | 1 node × 4 GPUs (TP=4 + EP) | One GB300 (4 × 288 GB HBM = 1152 GB) is enough; halves the SLURM footprint vs. Oleksii's 2-node setup.  Kimi expects `thinking=false` not `enable_thinking`; the manifest sets this via the Phase-6 `chat_template_kwargs` knob. |

Before submitting on AWS-CMH, mirror the Kimi-compatible vLLM container
into your alaptev mount and update `cluster_configs/aws-cmh.yaml`:

```yaml
containers:
  # ...
  vllm_kimi: /alaptev/containers/vllm-glm51-cu130-ray.sqsh
```

Then pass `--server_container vllm_kimi` (or the literal path) to the
driver via the trailing `-- ...` slot.

## End-to-end: smoke (5 problems)

```bash
# 1. (One-time) Convert NS smoke data into NG format for sanity:
python workdir-agents/validation/scripts/convert_ns_to_ng.py \
    workdir-agents/validation/data/smoke_5.ns.jsonl /tmp/smoke_ng.jsonl

# 2. Dry-run the driver to check the SLURM command shape:
python workdir-agents/validation/scripts/run_validation.py \
    --manifest workdir-agents/validation/frontierscience_manifest.yaml \
    --input-ns workdir-agents/validation/data/smoke_5.ns.jsonl \
    --output-base /alaptev/exp/hermes_validation/smoke \
    --cluster aws-cmh \
    --limit 5 \
    --dry-run -- \
    --server_container /alaptev/containers/vllm-glm51-cu130-ray.sqsh

# 3. Submit Run 0 + Run 1 (Run 2 picks Run 1's merged template after Run 1 lands):
python workdir-agents/validation/scripts/run_validation.py \
    --manifest workdir-agents/validation/frontierscience_manifest.yaml \
    --input-ns workdir-agents/validation/data/smoke_5.ns.jsonl \
    --output-base /alaptev/exp/hermes_validation/smoke \
    --cluster aws-cmh \
    --limit 5 \
    --skip-run2 -- \
    --server_container /alaptev/containers/vllm-glm51-cu130-ray.sqsh

# 4. After Run 1 lands (check `<output-base>/run1_learning/merge_audit.json`):
python workdir-agents/validation/scripts/run_validation.py \
    --manifest workdir-agents/validation/frontierscience_manifest.yaml \
    --input-ns workdir-agents/validation/data/smoke_5.ns.jsonl \
    --output-base /alaptev/exp/hermes_validation/smoke \
    --cluster aws-cmh \
    --limit 5 \
    --skip-run0 --skip-run1 -- \
    --server_container /alaptev/containers/vllm-glm51-cu130-ray.sqsh

# 5. Submit judge jobs (gpt-oss-120b, mirrors run_nano_physics_agent.py):
python workdir-agents/validation/scripts/submit_judges.py \
    /alaptev/exp/hermes_validation/smoke

# 6. After judges finish, aggregate:
python workdir-agents/validation/scripts/compare_runs.py \
    /alaptev/exp/hermes_validation/smoke
```

## Scaling to the full 100 problems

```bash
# 1. Materialize the dataset via ns prepare_data:
ns prepare_data --benchmarks frontierscience-olympiad --data_dir /alaptev/data

# 2. Same driver, no --limit, fresh output base:
python workdir-agents/validation/scripts/run_validation.py \
    --manifest workdir-agents/validation/frontierscience_manifest.yaml \
    --input-ns /alaptev/data/frontierscience-olympiad/all.jsonl \
    --output-base /alaptev/exp/hermes_validation/full100 \
    --cluster aws-cmh \
    --skip-run2 -- \
    --server_container /alaptev/containers/vllm-glm51-cu130-ray.sqsh

# 3. — 4. — 5. As in the smoke flow, against the full output-base.
```

## What the comparator prints

```
=== run0_baseline ===
  dir:                  /alaptev/exp/hermes_validation/full100/run0_baseline
  rollouts:             100
  correct (overall):    12
  pass@1:               0.120
  judged vs em-fallback:100 / 0
    [  chemistry] pass@1 = 0.150  (5/33)
    ...

=== deltas ===
  Δ run1 - run0 (in-run)      = +0.040  (0.120 → 0.160)
  Δ run2 - run0 (net)         = +0.070  (0.120 → 0.190)
  Δ run2 - run1 (stability)   = +0.030  (0.160 → 0.190)
```

The headline number is **Δ run2 - run0** — that is the answer to the
question this experiment exists to answer.
