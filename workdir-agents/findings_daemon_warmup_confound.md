# Findings: daemon-state confound contaminates A/B/C pass comparisons

**Date observed:** 2026-05-29
**Status:** Open investigation — confirmation experiments not yet run.
**Filed from:** Phase 8 v5 control run (chunked architecture, no nudge,
no memory carryover possible).

Cross-references:
- Main plan: [`hermes_integration_plan.md`](hermes_integration_plan.md) — see Phase 7 + Phase 8 sections
- Linear: SCI-480 — *Self-Improving Science Agent*

---

## TL;DR

In every A/B/C run we have so far, **Pass 2 (warm-template eval) shares
the same vLLM daemon process with Pass 0 (cold baseline)**.  The daemon's
internal state (prefix cache, scheduling, paged-attention pool) drifts
across passes.  That drift alone produces score differences large enough
to be mistaken for a self-improvement effect — even when the merge-back
template is provably empty.

**Implication:** the headline question of SCI-480 — "does the
self-improvement loop help?" — cannot be answered from any A/B/C
comparison we have run to date.

---

## Observation

v5 control run (`/lustre/fsw/portfolios/nemotron/users/alaptev/exp/chunked_25_control/20260529T210330Z`):

| Pass | Config | Score |
|---|---|---|
| 0 (A) | cold baseline | 15/25 (60.0%) |
| 1 (B) | learning pass | 16/25 (64.0%) |
| 2 (C) | warm-template eval | 21/25 (84.0%) |

Pass 0 → Pass 2 lift: **+24 points** (24 percentage points, **~2.4σ for n=25**).

This was on a deliberately *clean* control: chunked architecture, no
nudge, same 25-problem slice as v4, no system-prompt manipulation.  The
big jump was unexpected and warranted a deep look.

## What ruled out the obvious explanations

### 1. The warm template is **not** warm

Mergeback audit (`pass1/merge_audit.json`):

```
total files in audit: 6
actions: {'unchanged': 6}
  unchanged  size_delta=    +0  MEMORY.md
  unchanged  size_delta=    +0  USER.md
  unchanged  size_delta=    +0  skills/user/.keep
  …
```

All 6 allowlisted files in the merge target had **size_delta=0** and
**SHA-256 unchanged**.  Confirmed by direct compare:

```bash
diff -rq /lustre/.../hermes_home/frontier_seed \
         /lustre/.../hermes_home/run2_template
# empty output → byte-identical
```

So Pass 2's HERMES_HOME template is **byte-identical** to Pass 0's.  The
agent didn't write to memory or skills during Pass 1 (the well-known
"K2.6 doesn't tool-call memory on self-contained problems" finding from
Phase 7).

This means: **whatever caused the +24pt lift, it cannot have come from
content carried over via the merge-back machinery.**

### 2. The questions are the same in each pass

Chunk slicing is deterministic (`row_index % NUM_CHUNKS`), so Pass 0 and
Pass 2 see the exact same 25 problems.  Verified by question hash — 24
of 25 problems present in all three passes (one row dropped from one
chunk for unrelated reasons).

### 3. It's not just sampling noise on a few easy problems

Of the 24 problems common to all three passes:
- Pass 0 wrong, Pass 2 wrong: 3
- Pass 0 right, Pass 2 right: 13
- **Pass 0 wrong, Pass 2 right: 7** ← the +24pt lift
- Pass 0 right, Pass 2 wrong: 1

7 problems flipped wrong→right between Pass 0 and Pass 2.  Only 1
flipped the other way.  Concentration on one side at this magnitude is
unusual for pure variance.

## What the trajectories show

Sampled three of the W→R flips:

| Problem | Pass 0 | Pass 2 |
|---|---|---|
| "Suppose the Moon is 3 degrees above the horizon…" | turns=2, answer 8,575 chars | turns=1, answer **123,412** chars |
| "Consider the following system in a rotating 2D…" | turns=2, answer 166 chars | turns=1, answer **27,061** chars |
| "Domain Wall 2" | turns=2, answer 82 chars | turns=1, answer **13,707** chars |

The pattern is uniform across all 7 W→R flips: **Pass 0 invokes a tool,
gets a short truncated/abandoned response (the agent
early-termination quirk we documented in Phase 8 follow-up); Pass 2
goes single-shot, produces a long detailed analysis ending in a
correct FINAL ANSWER.**

Average `turns_used` per pass (n=24):

| Pass | avg turns | min | max |
|---|---|---|---|
| 0 | 1.25 | 1 | 2 |
| 1 | 1.38 | 1 | 2 |
| 2 | **1.21** | 1 | 2 |

Pass 2 uses tools *less* than Pass 0.  The fewer tool calls → fewer
mid-thought terminations → longer single-shot reasoning → more correct
answers.

## Hypothesis

The vLLM daemon serving Kimi-K2.6 was alive for all three passes.  Over
that ~2-hour lifetime its internal state evolved:

- **Prefix-cache fill**: vLLM's automatic prefix caching has accumulated
  ~hundreds of long science-problem trajectories from Passes 0 and 1.
  When Pass 2 starts, more of the system-prompt-plus-user-turn prefix
  hits cache.  Time-to-first-token drops; the model's behaviour at the
  first sampling step may shift subtly (e.g. less inclined to issue a
  tool_call, because the first sampled distribution skews differently
  when prefill is instantaneous vs. when it takes a second of
  background compute).
- **Continuous batching pool**: vLLM's continuous batcher is in a
  different state by Pass 2 than Pass 0.  Different requests share
  GPU compute differently; per-token latency variance shifts.
- **Scheduler order effects**: paged-attention block allocation,
  warmup of CUDA kernels for sequence lengths we've already seen.

Any of these *could* produce small per-token sampling differences.
Cumulatively, across 25 long agentic rollouts, those differences would
amplify into different *high-level* decisions (call a tool vs. answer
directly) and then into different verdicts.

**Phase 7's monolithic A/B/C had the same confound** but we missed it
because:
- The model didn't tool-call on most problems anyway (n=100, low
  base-rate of the failure mode).
- Per-pass scores were 67 / 64 / 63 — within the noise band, no lift to
  explain away.

## Why this matters for SCI-480

The headline goal of SCI-480 is to validate that the self-improvement
loop (memory + skill writes + merge-back) actually helps.  The standard
test would be: A vs. C, holding everything else constant.

But "everything else" is currently *not* constant.  In a single A/B/C
session, the daemon state is monotonically more-warm by the time C
runs.  Any non-zero ΔAC could be either:

1. The self-improvement loop actually helping (the thing we want to
   measure), or
2. Daemon warm-up shifting sampling outcomes (a nuisance variable).

We can't tell the two apart from the current pipeline.  The +24pt lift
in v5, where (1) is provably impossible (template byte-identical),
demonstrates that (2) alone can produce a misleading "positive" signal.

## Confirmation / disambiguation experiments

Three experiments would isolate the confound.  Cheap → ranked:

### E1 — add a "Pass D" cold rerun after Pass C  (cheapest)

Architecture:  daemon stays up.  After Pass 2 finishes, run a *fourth*
pass with the cold template and persist_memory=false (i.e. exactly
the same config as Pass 0).  Compare scores:

- If Pass D ≈ Pass 0 (~60%): daemon state is *not* the cause; something
  is genuinely carrying across the pipeline.  Need a deeper hunt for
  hidden carryover (file system state, env vars, chunk-worker logs the
  next chunk somehow reads, etc.).
- If Pass D ≈ Pass 2 (~84%): the entire +24 was daemon warm-up.  All
  prior A/B/C comparisons are contaminated by this effect.
- Intermediate (~70–75%): warmup is part of it, but maybe there's a
  smaller real effect on top.

Cost: ~30–45 min of additional CPU + the shared GPU daemon.  Minimal
new code — extend `launch_chunked.sh` to do a 4th pass, or just add a
`--passes 4` flag and let pass 3 reuse pass 0's template config.

### E2 — kill+restart daemon between passes  (medium cost)

Architecture: scancel the Kimi daemon after each pass, sbatch a fresh
one, wait for endpoint, run the next pass.  Each pass starts against
an identically-cold daemon.

Cost: +15 min per pass for daemon boot × 3 passes = +45 min.  More
substantial launcher rewrite (per-pass daemon lifecycle).  Cleanest
control but most expensive.

If the new A/B/C now looks like A ≈ C and ΔAB and ΔBC are within
sampling noise, that confirms warmup was the whole story.  If ΔAC
remains positive even with fresh daemons per pass, *that* is the
real self-improvement signal.

### E3 — multiple random seeds per cell  (most expensive)

Run each (pass, daemon-config) combination 3–5 times with different
RNG seeds, average the scores.  Gives a proper variance estimate so we
can put error bars on any claim of "X moves the needle by Y points."

Cost: 3–5× current GPU time, but it's the only way to put credible
intervals on results we report externally (e.g. for HLE scaling).

## Suggested order of operations

1. **E1 first.**  One additional pass on the same daemon — tells us
   immediately whether the lift is daemon-state or hidden carryover.
2. If E1 says "all warmup," **E2 to validate the clean comparison
   protocol** for all future A/B/C runs.
3. **E3 as the gold-standard** before scaling to HLE — multi-seed
   averages on the controls + the agent variants we want to compare.

After E1, we'll know enough to decide whether the rest of the
investigation is worth it before scaling up.

## Data references

All files live on aws-cmh under
`/lustre/fsw/portfolios/nemotron/users/alaptev/`.

- v5 control run (the one with the +24pt lift):
  `exp/chunked_25_control/20260529T210330Z/`
  - `input.jsonl`  — the 25-problem prepared input
  - `pass{0,1,2}/chunk{0,1}/rollouts.jsonl`  — per-chunk per-pass rollouts
  - `pass1/merge_audit.json`  — audit showing 0 actual file copies
  - `logs/`  — per-job sbatch stdout/stderr
- Templates:
  - `hermes_home/frontier_seed`  — TEMPLATE_RUN0 (cold)
  - `hermes_home/run1_template`  — TEMPLATE_RUN1 (mutated by mergeback)
  - `hermes_home/run2_template`  — TEMPLATE_RUN2 (snapshot of run1 post-mergeback)
- Phase 7 100-problem A/B/C (different daemon lifecycle, same confound
  in principle but masked by variance):
  `exp/abc/20260527_full100_050731Z/run{0,1,2}/judged.jsonl`

Comparison scripts used in the diagnostic (not committed; recreate from
this file's content if needed):

- `/tmp/cmp_passes.py` — per-problem cross-pass comparison + turns_used
- `/tmp/v4_compare.py` — Phase 7 vs Phase 8 v1 vs Phase 8 v4 score
  comparison on a 25-problem slice
- `/tmp/inspect_response.py` — dumps `response.output` structure for a
  single rollout (used to identify the early-termination pattern)

## Open questions

- Does vLLM's prefix caching actually cause the cross-pass behaviour we
  see, or is there a different daemon-state mechanism?  Could profile
  TTFT and per-token latency in Pass 0 vs Pass 2.
- Is there any cluster-level state (file caches, lustre metadata, IB
  link warmup) that contributes in addition to the in-daemon state?
- For training (the eventual SCI-480 endpoint), is this confound a
  problem or a feature?  If the production loop also reuses the daemon,
  the warmup effect is part of the deployed system's behaviour.  The
  problem is only the *measurement* — we need a clean evaluation
  protocol that doesn't confuse "the system is better" with "the second
  run on the same daemon scores higher."
