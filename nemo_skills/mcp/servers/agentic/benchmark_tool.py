# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``benchmark`` — the answer-hiding trust-boundary tool (SCI-548 P2).

The orchestrator of the self-improving swarm sits *outside* NeMo-Gym's
per-row rollout loop: it loads a whole benchmark, fans one worker out per
problem (``delegate_task``), and grades offline. Loading is therefore a
**tool call**, and this tool is the **trust boundary** between the raw
benchmark data — which carries the answers and grading config — and what any
agent (orchestrator or worker) is ever allowed to see.

Three tools, split so the orchestrator never holds problem content (§10/§11 of
the design doc):

- ``load_benchmark(benchmark, limit, shuffle, seed, shard)`` — **orchestrator
  side**. Returns an ids-only *manifest* ``{count, total, ids: [...]}``. No
  problem text, no answers. The orchestrator builds one ``delegate_task`` task
  per id (*"solve <id>; call get_problem('<id>')"*) so its context stays
  ``O(N ids)`` regardless of problem size — the key to scaling past a 128k
  window (HLE-scale).
- ``plan_batch(benchmark, goal_template, limit, shuffle, seed, shard)`` —
  **orchestrator side, large-batch path**. Expands the (sharded/sampled)
  benchmark into a staged JSONL *work-list file* (one ``delegate_task`` goal per
  id, built from ``goal_template`` with ``{id}`` substituted) and returns a
  small ``{handle, count, ...}`` receipt — NOT the ids, never answers. The
  orchestrator hands the ``handle`` straight to
  ``delegate_task(tasks_source=handle, max_in_flight=N)`` and dispatches the
  WHOLE set in one call, so it never has to *generate* (or even hold) thousands
  of goal strings — context stays ``O(1)`` in the batch size. Reuses the same
  id derivation + sharding as ``load_benchmark`` so plan/load/grade agree.
- ``get_problem(id)`` — **worker side**. Returns one problem's *perceivable*
  content ``{id, prompt, modality}`` and nothing else. Inherited by
  ``delegate_task`` children as an ``mcp-*`` toolset, so each worker pulls its
  own problem into *its* context; the orchestrator never sees it.

**The boundary is a whitelist, not a blacklist.** ``get_problem`` *constructs*
its return value field-by-field (``id`` + ``prompt`` + ``modality``); it never
echoes the raw row. So a future leaky field (a new ``solution_hint`` column,
say) cannot slip through by default — it is simply never read. As belt-and-
braces, ``_assert_no_answer_leak`` re-checks the constructed dict against a
denylist (``expected_answer``, ``reference_solution``, ``verifier_metadata``,
``verifier_type``, ``answer``, ``solution``, …) and raises if any appear.

Grading data (``expected_answer`` etc.) stays in the staged JSONL, read **only
by the offline grader** (``grade_rollouts`` / Gym ``verify``), which joins
worker answers back to answers by ``id`` *after* the orchestrator session ends.
No agent ever sees it. ``derive_problem_id`` is the shared id-derivation used
by both this tool and the grader so the join key always agrees.

Benchmark rows in the Gym/NS ecosystem are **heterogeneous**: ``gpqa`` writes
``{problem, question, expected_answer, uuid}``; ``code_gen`` writes
``{responses_create_params: {input: [...]}, verifier_metadata, hash_id}``. So
id + prompt are auto-detected (configurable key lists), with a fallback to
``row-<index>`` for the id and to the user turns of
``responses_create_params.input`` for the prompt.

Run as a Group-B in-process Tool via the generic stdio wrapper::

    ns-mcp-serve nemo_skills.mcp.servers.agentic.benchmark_tool:BenchmarkTool \\
        --overrides '{"benchmark_root": "/path/to/staged"}'

Config (``--overrides`` JSON; each also falls back to an env var):

==================  ===========================  ==========================================
key                 env fallback                 default
==================  ===========================  ==========================================
``benchmark_root``  ``NS_BENCHMARK_ROOT``        ``None`` (then ``benchmark`` must be a path)
``benchmark_path``  ``NS_BENCHMARK_PATH``        ``None`` (a single staged file; lets
                                                 ``get_problem`` lazy-load without a prior
                                                 ``load_benchmark``)
``id_keys``         ``-``                        ``["id","uuid","hash_id","_id","problem_id","qid"]``
``prompt_keys``     ``-``                        ``["problem","question","prompt","text"]``
``plan_dir``        ``NS_BATCH_PLAN_DIR``        ``None`` (then parent of ``NS_WORKLOG_DIR``, i.e.
                                                 the run output_dir; ``plan_batch`` writes its
                                                 work-list files here — must be a mount the
                                                 orchestrator can also read)
==================  ===========================  ==========================================

One MCP server process serves the orchestrator *and* every delegate worker
(they share the parent's MCP client), so the in-memory id→row index built by
``load_benchmark`` is visible to all subsequent ``get_problem`` calls.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nemo_skills.mcp.tool_manager import Tool

logger = logging.getLogger(__name__)

DEFAULT_ID_KEYS = ("id", "uuid", "hash_id", "_id", "problem_id", "qid")
DEFAULT_PROMPT_KEYS = ("problem", "question", "prompt", "text")

# Default per-task goal for plan_batch when the orchestrator gives none. Mirrors
# the worker contract used by load_benchmark's manifest: fetch the problem by id,
# solve it, and commit a single 'Final Answer:' line the grader can join on.
_DEFAULT_GOAL_TEMPLATE = (
    "Solve benchmark problem '{id}'. Call get_problem('{id}') to retrieve the "
    "problem text, solve it, and end your report with a single line "
    "'Final Answer: <your answer>'. Do not use any other tools."
)

# Fields that carry the answer or grading config. NEVER returned to any agent.
# The whitelist construction already excludes them; this is the defence-in-depth
# tripwire (``_assert_no_answer_leak``) in case the return shape ever changes.
ANSWER_DENYLIST = frozenset(
    {
        "expected_answer",
        "reference_solution",
        "verifier_metadata",
        "verifier_type",
        "answer",
        "solution",
        "correct_answer",
        "subset_for_metrics",
        "unit_tests",
        "gold",
        "label",
    }
)


def derive_problem_id(row: Dict[str, Any], index: int, id_keys=DEFAULT_ID_KEYS) -> str:
    """Derive a stable join key for one benchmark row.

    Tries ``id_keys`` in order; falls back to ``row-<index>``. Shared between
    this tool and the offline grader so the answer↔response join always agrees.
    Must be deterministic in ``(row, index)``.
    """
    for key in id_keys:
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return f"row-{index}"


class BenchmarkTool(Tool):
    """In-process ``Tool`` exposing ``load_benchmark`` / ``get_problem``.

    State (the id→problem index for the active benchmark, plus a per-path
    cache) lives on the instance for the life of the MCP server process. Since
    one server process serves the orchestrator and all delegate workers, the
    index built by ``load_benchmark`` is what ``get_problem`` resolves against.
    """

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {
            "benchmark_root": None,
            "benchmark_path": None,
            "id_keys": None,
            "prompt_keys": None,
            # Directory the plan_batch work-list files are written to. Must be on
            # a filesystem the orchestrator (which reads the handle via
            # delegate_task) can also see — defaults to the worklog output dir.
            "plan_dir": None,
        }
        self._configured = False
        self._lock = threading.RLock()
        # path -> {"index": {id: problem_dict}, "order": [ids in file order]}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._active_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def default_config(self) -> Dict[str, Any]:
        return dict(self._config)

    def configure(self, overrides: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(self._config)
        if overrides:
            cfg.update(overrides)

        cfg["benchmark_root"] = cfg.get("benchmark_root") or os.environ.get("NS_BENCHMARK_ROOT")
        cfg["benchmark_path"] = cfg.get("benchmark_path") or os.environ.get("NS_BENCHMARK_PATH")
        cfg["id_keys"] = tuple(cfg.get("id_keys") or DEFAULT_ID_KEYS)
        cfg["prompt_keys"] = tuple(cfg.get("prompt_keys") or DEFAULT_PROMPT_KEYS)
        cfg["plan_dir"] = cfg.get("plan_dir") or os.environ.get("NS_BATCH_PLAN_DIR")

        self._config = cfg
        self._configured = True

    async def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "load_benchmark",
                "description": (
                    "ORCHESTRATOR tool. Load a prepared benchmark and return an ids-only "
                    "manifest {count, total, ids:[...]} — NO problem text, NO answers. Build "
                    "one worker task per id whose goal is to call get_problem('<id>') and solve "
                    "it; never load problem text into your own context. Use limit/shard to "
                    "process a subset, shuffle+seed for a reproducible sample."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "benchmark": {
                            "type": "string",
                            "description": (
                                "Benchmark name (resolved under the configured benchmark_root) "
                                "or a path to a prepared .jsonl file."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Optional cap on the number of ids returned (applied after shuffle/shard).",
                        },
                        "shuffle": {
                            "type": "boolean",
                            "description": "Shuffle ids before limit/shard (deterministic with seed). Default false.",
                        },
                        "seed": {
                            "type": "integer",
                            "description": "Seed for the shuffle/sample, for reproducibility.",
                        },
                        "shard": {
                            "type": "string",
                            "description": "Optional 'i/n' (1-based) to take shard i of n contiguous splits.",
                        },
                    },
                    "required": ["benchmark"],
                },
            },
            {
                "name": "plan_batch",
                "description": (
                    "ORCHESTRATOR tool. Expand a benchmark (optionally sharded/sampled) into a "
                    "staged work-list FILE and return {handle, count, benchmark, shard} — a small "
                    "constant-size receipt, NOT the ids or problem text. Each line of the file is "
                    "one delegate_task goal built from your goal_template with '{id}' substituted "
                    "(the worker then calls get_problem('<id>') and solves it). Hand the returned "
                    "'handle' to delegate_task(tasks_source=handle, max_in_flight=N) to dispatch "
                    "the whole set in ONE call WITHOUT enumerating thousands of goals in your "
                    "context. No answers ever touch the file. Use this instead of load_benchmark "
                    "+ a hand-built tasks array for anything but a tiny batch."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "benchmark": {
                            "type": "string",
                            "description": (
                                "Benchmark name (resolved under benchmark_root) or a path to a "
                                "prepared .jsonl file."
                            ),
                        },
                        "goal_template": {
                            "type": "string",
                            "description": (
                                "Per-task goal string with a literal '{id}' placeholder, e.g. "
                                "\"Solve problem '{id}': call get_problem('{id}'), solve it, and "
                                "end with a line 'Final Answer: <answer>'.\". '{id}' is replaced "
                                "with each problem id. If omitted, a sensible default is used."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Optional cap on number of tasks (applied after shuffle/shard).",
                        },
                        "shuffle": {
                            "type": "boolean",
                            "description": "Shuffle ids before limit/shard (deterministic with seed). Default false.",
                        },
                        "seed": {
                            "type": "integer",
                            "description": "Seed for the shuffle/sample, for reproducibility.",
                        },
                        "shard": {
                            "type": "string",
                            "description": "Optional 'i/n' (1-based) to take shard i of n contiguous splits.",
                        },
                    },
                    "required": ["benchmark"],
                },
            },
            {
                "name": "get_problem",
                "description": (
                    "WORKER tool. Fetch ONE problem's content by id: returns {id, prompt, "
                    "modality} and nothing else (never the answer or grading config). Call this "
                    "for the id you were assigned, then solve it."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "The problem id from the manifest / your assigned task.",
                        },
                    },
                    "required": ["id"],
                },
            },
        ]

    async def execute(
        self, tool_name: str, arguments: Dict[str, Any], extra_args: Optional[Dict[str, Any]] = None
    ) -> Any:
        args = dict(arguments or {})
        if tool_name == "load_benchmark":
            return self._load_benchmark(
                benchmark=args.get("benchmark"),
                limit=args.get("limit"),
                shuffle=bool(args.get("shuffle", False)),
                seed=args.get("seed"),
                shard=args.get("shard"),
            )
        if tool_name == "plan_batch":
            return self._plan_batch(
                benchmark=args.get("benchmark"),
                goal_template=args.get("goal_template"),
                limit=args.get("limit"),
                shuffle=bool(args.get("shuffle", False)),
                seed=args.get("seed"),
                shard=args.get("shard"),
            )
        if tool_name == "get_problem":
            return self._get_problem(id=args.get("id"))
        return f"Error: unknown tool '{tool_name}'"

    # ------------------------------------------------------------------
    # load_benchmark / get_problem
    # ------------------------------------------------------------------

    def _load_benchmark(
        self,
        benchmark: Any,
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        shard: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._ensure_configured()
        name = self._require(benchmark, "benchmark")
        path = self._resolve_path(name)

        index, order = self._load_indexed(path)
        with self._lock:
            self._active_path = str(path)

        ids = list(order)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(ids)
        if shard:
            ids = self._apply_shard(ids, shard)
        total_after_shard = len(ids)
        if limit is not None:
            try:
                ids = ids[: max(0, int(limit))]
            except (TypeError, ValueError):
                raise ValueError(f"limit must be an integer; got {limit!r}")

        return {
            "benchmark": name,
            "path": str(path),
            "total": len(order),
            "count": len(ids),
            "ids": ids,
            "selected_from": total_after_shard,
        }

    def _plan_batch(
        self,
        benchmark: Any,
        goal_template: Optional[str] = None,
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        shard: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Expand a (sharded/sampled) benchmark into a staged JSONL work-list and
        return a small handle receipt. Reuses ``_load_benchmark`` for id derivation
        and sharding so plan/load/grade all agree on ids. The file holds only
        templated goal strings + non-sensitive ids — never answers."""
        manifest = self._load_benchmark(
            benchmark=benchmark, limit=limit, shuffle=shuffle, seed=seed, shard=shard
        )
        ids = manifest["ids"]
        template = goal_template if (isinstance(goal_template, str) and goal_template.strip()) else _DEFAULT_GOAL_TEMPLATE
        if "{id}" not in template:
            raise ValueError("goal_template must contain the literal '{id}' placeholder.")

        out_path = self._resolve_plan_path(manifest["benchmark"], shard)
        n = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for pid in ids:
                # Only {id} is substituted; any other braces in the template are
                # left intact (str.replace, not str.format, so worker code
                # snippets with their own braces survive).
                goal = template.replace("{id}", str(pid))
                fh.write(json.dumps({"goal": goal}) + "\n")
                n += 1

        return {
            "handle": str(out_path),
            "count": n,
            "benchmark": manifest["benchmark"],
            "shard": shard,
            "selected_from": manifest["selected_from"],
            "total": manifest["total"],
        }

    def _resolve_plan_path(self, benchmark_name: str, shard: Optional[str]) -> Path:
        """Pick a directory both this MCP server and the orchestrator can read.

        Priority: configured plan_dir / NS_BATCH_PLAN_DIR -> the parent of
        NS_WORKLOG_DIR (the run's output_dir, already a shared mount) -> cwd."""
        plan_dir = self._config.get("plan_dir")
        if not plan_dir:
            worklog_dir = os.environ.get("NS_WORKLOG_DIR")
            if worklog_dir:
                # NS_WORKLOG_DIR is "{output_dir}/worklogs"; stage plans beside it.
                plan_dir = os.path.join(os.path.dirname(worklog_dir.rstrip("/")), "batch_plans")
            else:
                plan_dir = os.path.join(os.getcwd(), "batch_plans")
        Path(plan_dir).mkdir(parents=True, exist_ok=True)
        safe_bench = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(str(benchmark_name)))
        safe_shard = re.sub(r"[^A-Za-z0-9]", "-", str(shard)) if shard else "all"
        return Path(plan_dir) / f"plan_{safe_bench}_{safe_shard}.jsonl"

    def _get_problem(self, id: Any) -> Dict[str, Any]:
        self._ensure_configured()
        pid = self._require(id, "id")

        index = self._active_index()
        if index is None:
            return {
                "error": (
                    "no benchmark loaded; the orchestrator must call load_benchmark first "
                    "(or configure benchmark_path)"
                ),
                "id": pid,
            }
        problem = index.get(pid)
        if problem is None:
            return {"error": f"unknown problem id {pid!r}", "id": pid}

        # Defence in depth: the dict was built by the whitelist extractor, but
        # re-verify nothing answer-shaped rides along before it leaves the tool.
        _assert_no_answer_leak(problem)
        return dict(problem)

    # ------------------------------------------------------------------
    # Loading / indexing
    # ------------------------------------------------------------------

    def _load_indexed(self, path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        """Read the JSONL, build the id→{id,prompt,modality} index. Cached per path."""
        key = str(path)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached["index"], cached["order"]

        if not path.exists():
            raise FileNotFoundError(f"benchmark file not found: {path}")

        id_keys = self._config["id_keys"]
        prompt_keys = self._config["prompt_keys"]
        index: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        seen: Dict[str, int] = {}
        dropped = 0

        with path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    dropped += 1
                    continue
                if not isinstance(row, dict):
                    dropped += 1
                    continue
                raw_id = derive_problem_id(row, i, id_keys)
                # Disambiguate collisions deterministically so every row is
                # addressable (and the grader, using the same derivation +
                # de-dup, agrees).
                if raw_id in seen:
                    seen[raw_id] += 1
                    pid = f"{raw_id}#{seen[raw_id]}"
                else:
                    seen[raw_id] = 0
                    pid = raw_id
                prompt, modality = self._extract_prompt(row, prompt_keys)
                problem = {"id": pid, "prompt": prompt, "modality": modality}
                _assert_no_answer_leak(problem)
                index[pid] = problem
                order.append(pid)

        if dropped:
            logger.warning("benchmark %s: skipped %d malformed/non-dict rows", path, dropped)
        if not order:
            raise ValueError(f"benchmark file has no usable rows: {path}")

        with self._lock:
            self._cache[key] = {"index": index, "order": order}
        return index, order

    def _extract_prompt(self, row: Dict[str, Any], prompt_keys) -> Tuple[str, str]:
        """Whitelist-extract the perceivable prompt + modality from a row.

        Tries the direct ``prompt_keys`` first (gpqa/NS style), then the user
        turns of ``responses_create_params.input`` (Gym responses style). Never
        reads answer/grading fields. ``modality`` is ``"text"`` unless a
        non-text content part is present, then ``"multimodal"``.
        """
        for key in prompt_keys:
            val = row.get(key)
            if isinstance(val, str) and val.strip():
                return val, "text"

        rcp = row.get("responses_create_params")
        if isinstance(rcp, dict):
            text, modality = self._extract_from_messages(rcp.get("input"))
            if text:
                return text, modality

        # Last resort: a bare top-level ``input`` (some prepared shapes).
        text, modality = self._extract_from_messages(row.get("input"))
        if text:
            return text, modality

        logger.warning("benchmark row had no recognizable prompt field; returning empty prompt")
        return "", "text"

    @staticmethod
    def _extract_from_messages(messages: Any) -> Tuple[str, str]:
        """Pull user-turn text (and detect non-text parts) from a messages list."""
        if not isinstance(messages, list):
            return "", "text"
        chunks: List[str] = []
        modality = "text"
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") not in (None, "user"):
                # Skip system/developer/assistant turns — the problem is the user ask.
                continue
            content = msg.get("content")
            if isinstance(content, str):
                if content.strip():
                    chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        chunks.append(part)
                    elif isinstance(part, dict):
                        ptype = part.get("type", "")
                        text = part.get("text") or part.get("input_text")
                        if isinstance(text, str) and text.strip():
                            chunks.append(text)
                        if ("image" in ptype) or ("audio" in ptype) or ("file" in ptype):
                            modality = "multimodal"
        return "\n\n".join(c for c in chunks if c.strip()), modality

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, name: str) -> Path:
        """Resolve a benchmark name-or-path to a concrete .jsonl file."""
        # Explicit path forms: ends with .jsonl, or contains a separator, or exists.
        looks_like_path = name.endswith(".jsonl") or ("/" in name) or os.path.isabs(name)
        if looks_like_path:
            p = Path(name).expanduser()
            if p.exists():
                return p
            # fall through to root resolution if a bare relative path missed
        root = self._config.get("benchmark_root")
        if root:
            root_path = Path(root).expanduser()
            for candidate in (
                root_path / name,
                root_path / f"{name}.jsonl",
                root_path / name / "test.jsonl",
                root_path / name / "data.jsonl",
            ):
                if candidate.exists():
                    return candidate
        # Last try: the literal name as a path (so a clear error names it).
        p = Path(name).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(
            f"could not resolve benchmark {name!r}"
            + (f" under benchmark_root {root!r}" if root else " (no benchmark_root configured)")
        )

    @staticmethod
    def _apply_shard(ids: List[str], shard: str) -> List[str]:
        try:
            i_str, n_str = str(shard).split("/")
            i, n = int(i_str), int(n_str)
        except (ValueError, AttributeError):
            raise ValueError(f"shard must be 'i/n' (1-based); got {shard!r}")
        if n <= 0 or i < 1 or i > n:
            raise ValueError(f"shard 'i/n' out of range: {shard!r}")
        total = len(ids)
        per = (total + n - 1) // n  # ceil
        start = (i - 1) * per
        return ids[start : start + per]

    def _active_index(self) -> Optional[Dict[str, Dict[str, Any]]]:
        with self._lock:
            if self._active_path is not None:
                cached = self._cache.get(self._active_path)
                if cached is not None:
                    return cached["index"]
        # Lazy-load a configured single file so get_problem works even if the
        # worker process never saw a load_benchmark call.
        path = self._config.get("benchmark_path")
        if path:
            index, _ = self._load_indexed(Path(path).expanduser())
            with self._lock:
                self._active_path = str(Path(path).expanduser())
            return index
        return None

    def _ensure_configured(self) -> None:
        if not self._configured:
            self.configure(None, None)

    @staticmethod
    def _require(value: Any, name: str) -> str:
        s = str(value).strip() if value is not None else ""
        if not s:
            raise ValueError(f"{name} is required and must be a non-empty string")
        return s


def _assert_no_answer_leak(payload: Dict[str, Any]) -> None:
    """Tripwire: raise if an answer/grading field rode into an agent-bound dict."""
    leaked = ANSWER_DENYLIST.intersection(payload.keys())
    if leaked:
        raise AssertionError(f"trust-boundary violation: answer field(s) {sorted(leaked)} in tool output")
