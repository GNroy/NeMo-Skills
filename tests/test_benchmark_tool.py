# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""SCI-548 P2 — ``benchmark`` load_benchmark/get_problem trust-boundary tool.

Coverage:
  In-process (fast, direct Tool calls):
    B1  — load_benchmark returns ids-only manifest (count/total/ids), NO text/answers
    B2  — get_problem returns exactly {id, prompt, modality}
    B3  — id auto-detect across schemas (id / uuid / hash_id / row-<index> fallback)
    B4  — prompt auto-detect: direct problem/question key AND responses_create_params.input
    B5  — TRUST BOUNDARY: answer/grading fields never leak (gpqa + code_gen shaped rows)
    B6  — limit / shuffle+seed (deterministic) / shard select the right ids
    B7  — get_problem unknown id → error; before any load → error
    B8  — duplicate derived ids are disambiguated (#1, #2) so every row is addressable
    B9  — benchmark_path lets get_problem lazy-load without a prior load_benchmark
    B10 — multimodal content part → modality "multimodal" (text still extracted)
    B11 — list_tools advertises load_benchmark/get_problem with the right shape
    B12 — composes under ToolManager (qualified names)
    B13 — _assert_no_answer_leak tripwire raises on an answer-shaped payload
    B14 — derive_problem_id is deterministic + shared join-key derivation
    B15 — benchmark_root resolves a bare benchmark name to <root>/<name>.jsonl
  Over real stdio (ns-mcp-serve wrapper subprocess):
    B16 — list_tools round-trip
    B17 — load_benchmark + get_problem round-trip (manifest then fetch one problem)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from nemo_skills.mcp.servers.agentic.benchmark_tool import (
    ANSWER_DENYLIST,
    BenchmarkTool,
    _assert_no_answer_leak,
    derive_problem_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


# gpqa-style: direct prompt key + uuid + expected_answer (the answer must be hidden).
GPQA_ROWS = [
    {"problem": "What is 2+2?\nA: 3\nB: 4", "question": "What is 2+2?", "expected_answer": "B", "uuid": "u-aaa"},
    {"problem": "Capital of France?\nA: Paris\nB: Rome", "expected_answer": "A", "uuid": "u-bbb"},
]

# code_gen-style: prompt buried in responses_create_params.input + grading in verifier_metadata.
CODEGEN_ROWS = [
    {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Write a function that adds two ints."},
            ]
        },
        "verifier_metadata": {"unit_tests": {"inputs": ["1 2"], "outputs": ["3"]}},
        "hash_id": "h-111",
    },
]


def _make_tool(**overrides: Any) -> BenchmarkTool:
    tool = BenchmarkTool()
    tool.configure(dict(overrides), {})
    return tool


# ---------------------------------------------------------------------------
# B1 / B2 — manifest + single-problem fetch shapes
# ---------------------------------------------------------------------------


def test_b1_load_returns_ids_only_manifest(tmp_path: Path) -> None:
    f = _write_jsonl(tmp_path / "gpqa.jsonl", GPQA_ROWS)
    tool = _make_tool()
    man = _run(tool.execute("load_benchmark", {"benchmark": str(f)}))
    assert man["count"] == 2
    assert man["total"] == 2
    assert man["ids"] == ["u-aaa", "u-bbb"]
    # The manifest is ids-only — no prompt text, no answers anywhere in it.
    blob = json.dumps(man)
    assert "2+2" not in blob and "Paris" not in blob
    assert "expected_answer" not in blob and '"B"' not in blob


def test_b2_get_problem_whitelist_shape(tmp_path: Path) -> None:
    f = _write_jsonl(tmp_path / "gpqa.jsonl", GPQA_ROWS)
    tool = _make_tool()
    _run(tool.execute("load_benchmark", {"benchmark": str(f)}))
    prob = _run(tool.execute("get_problem", {"id": "u-aaa"}))
    assert set(prob.keys()) == {"id", "prompt", "modality"}
    assert prob["id"] == "u-aaa"
    assert prob["prompt"].startswith("What is 2+2?")
    assert prob["modality"] == "text"


# ---------------------------------------------------------------------------
# B3 — id auto-detect across heterogeneous schemas
# ---------------------------------------------------------------------------


def test_b3_id_autodetect_and_fallback(tmp_path: Path) -> None:
    rows = [
        {"id": "explicit-1", "question": "q1"},
        {"uuid": "uu-2", "question": "q2"},
        {"hash_id": "hh-3", "question": "q3"},
        {"question": "q4 with no id key"},  # → row-3 (0-based index)
    ]
    f = _write_jsonl(tmp_path / "mixed.jsonl", rows)
    tool = _make_tool()
    man = _run(tool.execute("load_benchmark", {"benchmark": str(f)}))
    assert man["ids"] == ["explicit-1", "uu-2", "hh-3", "row-3"]


# ---------------------------------------------------------------------------
# B4 — prompt auto-detect (direct key + responses_create_params.input)
# ---------------------------------------------------------------------------


def test_b4_prompt_from_direct_key_and_messages(tmp_path: Path) -> None:
    f = _write_jsonl(tmp_path / "codegen.jsonl", CODEGEN_ROWS)
    tool = _make_tool()
    _run(tool.execute("load_benchmark", {"benchmark": str(f)}))
    prob = _run(tool.execute("get_problem", {"id": "h-111"}))
    # Pulled from the USER turn of responses_create_params.input (system turn skipped).
    assert prob["prompt"] == "Write a function that adds two ints."
    assert "You are helpful" not in prob["prompt"]


# ---------------------------------------------------------------------------
# B5 — THE trust boundary: no answer/grading field ever leaves the tool
# ---------------------------------------------------------------------------


def test_b5_no_answer_leak_gpqa_and_codegen(tmp_path: Path) -> None:
    f1 = _write_jsonl(tmp_path / "gpqa.jsonl", GPQA_ROWS)
    f2 = _write_jsonl(tmp_path / "codegen.jsonl", CODEGEN_ROWS)
    tool = _make_tool()

    for f, pid, forbidden in [
        (f1, "u-aaa", ["expected_answer", "\"B\"", "\"A\""]),
        (f2, "h-111", ["verifier_metadata", "unit_tests", "outputs"]),
    ]:
        _run(tool.execute("load_benchmark", {"benchmark": str(f)}))
        prob = _run(tool.execute("get_problem", {"id": pid}))
        assert set(prob.keys()) == {"id", "prompt", "modality"}
        assert not ANSWER_DENYLIST.intersection(prob.keys())
        blob = json.dumps(prob)
        for token in forbidden:
            assert token not in blob, f"answer token {token!r} leaked for {pid}"


# ---------------------------------------------------------------------------
# B6 — selection: limit / shuffle+seed / shard
# ---------------------------------------------------------------------------


def test_b6_limit_shuffle_shard(tmp_path: Path) -> None:
    rows = [{"id": f"p{i}", "question": f"q{i}"} for i in range(10)]
    f = _write_jsonl(tmp_path / "big.jsonl", rows)
    tool = _make_tool()

    # limit
    man = _run(tool.execute("load_benchmark", {"benchmark": str(f), "limit": 3}))
    assert man["ids"] == ["p0", "p1", "p2"]
    assert man["total"] == 10 and man["count"] == 3

    # shuffle is deterministic with a seed (and still resolvable via get_problem)
    m1 = _run(tool.execute("load_benchmark", {"benchmark": str(f), "shuffle": True, "seed": 7}))
    m2 = _run(tool.execute("load_benchmark", {"benchmark": str(f), "shuffle": True, "seed": 7}))
    assert m1["ids"] == m2["ids"]
    assert sorted(m1["ids"]) == sorted(p["id"] for p in rows)
    assert _run(tool.execute("get_problem", {"id": m1["ids"][0]}))["modality"] == "text"

    # shard 1/2 and 2/2 partition the ids without overlap
    s1 = _run(tool.execute("load_benchmark", {"benchmark": str(f), "shard": "1/2"}))["ids"]
    s2 = _run(tool.execute("load_benchmark", {"benchmark": str(f), "shard": "2/2"}))["ids"]
    assert s1 == [f"p{i}" for i in range(5)]
    assert s2 == [f"p{i}" for i in range(5, 10)]
    assert set(s1).isdisjoint(s2)


# ---------------------------------------------------------------------------
# B7 — error paths
# ---------------------------------------------------------------------------


def test_b7_unknown_id_and_no_benchmark(tmp_path: Path) -> None:
    tool = _make_tool()
    # get_problem before any load → clear error, no crash
    res = _run(tool.execute("get_problem", {"id": "whatever"}))
    assert "no benchmark loaded" in res["error"]

    f = _write_jsonl(tmp_path / "b.jsonl", [{"id": "x1", "question": "q"}])
    _run(tool.execute("load_benchmark", {"benchmark": str(f)}))
    res = _run(tool.execute("get_problem", {"id": "nope"}))
    assert "unknown problem id" in res["error"]


# ---------------------------------------------------------------------------
# B8 — duplicate ids disambiguated so every row stays addressable
# ---------------------------------------------------------------------------


def test_b8_duplicate_ids_disambiguated(tmp_path: Path) -> None:
    rows = [{"id": "dup", "question": "first"}, {"id": "dup", "question": "second"}]
    f = _write_jsonl(tmp_path / "dup.jsonl", rows)
    tool = _make_tool()
    man = _run(tool.execute("load_benchmark", {"benchmark": str(f)}))
    assert man["ids"] == ["dup", "dup#1"]
    assert _run(tool.execute("get_problem", {"id": "dup"}))["prompt"] == "first"
    assert _run(tool.execute("get_problem", {"id": "dup#1"}))["prompt"] == "second"


# ---------------------------------------------------------------------------
# B9 — benchmark_path lazy-load (worker fetch without a prior load_benchmark)
# ---------------------------------------------------------------------------


def test_b9_benchmark_path_lazy_load(tmp_path: Path) -> None:
    f = _write_jsonl(tmp_path / "staged.jsonl", [{"id": "z1", "question": "lazy q"}])
    tool = _make_tool(benchmark_path=str(f))
    # No load_benchmark — get_problem still resolves via the configured path.
    prob = _run(tool.execute("get_problem", {"id": "z1"}))
    assert prob["prompt"] == "lazy q"


# ---------------------------------------------------------------------------
# B10 — multimodal detection
# ---------------------------------------------------------------------------


def test_b10_multimodal_modality(tmp_path: Path) -> None:
    rows = [
        {
            "id": "img-1",
            "responses_create_params": {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "What is in this image?"},
                            {"type": "input_image", "image_url": "data:..."},
                        ],
                    }
                ]
            },
        }
    ]
    f = _write_jsonl(tmp_path / "img.jsonl", rows)
    tool = _make_tool()
    _run(tool.execute("load_benchmark", {"benchmark": str(f)}))
    prob = _run(tool.execute("get_problem", {"id": "img-1"}))
    assert prob["modality"] == "multimodal"
    assert prob["prompt"] == "What is in this image?"  # text part still extracted


# ---------------------------------------------------------------------------
# B11 / B12 — schema + ToolManager composition
# ---------------------------------------------------------------------------


def test_b11_list_tools_shape() -> None:
    tool = _make_tool()
    entries = {e["name"]: e for e in _run(tool.list_tools())}
    assert set(entries) == {"load_benchmark", "get_problem"}
    assert entries["load_benchmark"]["input_schema"]["required"] == ["benchmark"]
    assert entries["get_problem"]["input_schema"]["required"] == ["id"]


def test_b12_composes_under_tool_manager(tmp_path: Path) -> None:
    from nemo_skills.mcp.tool_manager import ToolManager

    f = _write_jsonl(tmp_path / "tm.jsonl", [{"id": "p1", "question": "q1"}])
    mgr = ToolManager(
        ["nemo_skills.mcp.servers.agentic.benchmark_tool::BenchmarkTool"],
        overrides={"BenchmarkTool": {"benchmark_path": str(f)}},
    )
    listed = _run(mgr.list_all_tools())
    assert sorted(t["name"] for t in listed) == ["get_problem", "load_benchmark"]
    man = _run(mgr.execute_tool("load_benchmark", {"benchmark": str(f)}))
    assert man["ids"] == ["p1"]
    prob = _run(mgr.execute_tool("get_problem", {"id": "p1"}))
    assert prob["prompt"] == "q1"


# ---------------------------------------------------------------------------
# B13 / B14 — tripwire + shared id derivation
# ---------------------------------------------------------------------------


def test_b13_assert_no_answer_leak_tripwire() -> None:
    _assert_no_answer_leak({"id": "x", "prompt": "p", "modality": "text"})  # ok
    with pytest.raises(AssertionError, match="trust-boundary violation"):
        _assert_no_answer_leak({"id": "x", "prompt": "p", "expected_answer": "B"})


def test_b14_derive_problem_id_deterministic() -> None:
    assert derive_problem_id({"id": "a"}, 5) == "a"
    assert derive_problem_id({"uuid": "u"}, 5) == "u"
    assert derive_problem_id({"hash_id": "h"}, 5) == "h"
    assert derive_problem_id({"question": "q"}, 5) == "row-5"
    # explicit id wins over uuid/hash_id
    assert derive_problem_id({"id": "a", "uuid": "u"}, 5) == "a"


# ---------------------------------------------------------------------------
# B15 — benchmark_root name resolution
# ---------------------------------------------------------------------------


def test_b15_benchmark_root_name_resolution(tmp_path: Path) -> None:
    root = tmp_path / "staged"
    root.mkdir()
    _write_jsonl(root / "mybench.jsonl", [{"id": "p1", "question": "q1"}])
    tool = _make_tool(benchmark_root=str(root))
    man = _run(tool.execute("load_benchmark", {"benchmark": "mybench"}))
    assert man["ids"] == ["p1"]
    assert man["path"].endswith("mybench.jsonl")


# ---------------------------------------------------------------------------
# B16 / B17 — over real stdio (ns-mcp-serve wrapper subprocess)
# ---------------------------------------------------------------------------

_SPEC = "nemo_skills.mcp.servers.agentic.benchmark_tool:BenchmarkTool"


def _server_params(overrides: Dict[str, Any]) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "nemo_skills.mcp.stdio_serve", _SPEC, "--overrides", json.dumps(overrides)],
        env=os.environ.copy(),
    )


def _text_of(result) -> str:
    """The stdio wrapper serializes a non-string return as JSON in a TextContent block."""
    return result.content[0].text


async def _list_over_stdio(overrides: Dict[str, Any]):
    async with stdio_client(_server_params(overrides)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.list_tools()


def test_b16_list_tools_over_stdio(tmp_path: Path) -> None:
    f = _write_jsonl(tmp_path / "s.jsonl", [{"id": "p1", "question": "q1"}])
    listed = asyncio.run(_list_over_stdio({"benchmark_path": str(f)}))
    assert sorted(t.name for t in listed.tools) == ["get_problem", "load_benchmark"]


async def _roundtrip_over_stdio(overrides: Dict[str, Any], path: str):
    async with stdio_client(_server_params(overrides)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            man = await session.call_tool("load_benchmark", {"benchmark": path})
            prob = await session.call_tool("get_problem", {"id": "u-aaa"})
            return _text_of(man), _text_of(prob)


def test_b17_load_and_get_over_stdio(tmp_path: Path) -> None:
    f = _write_jsonl(tmp_path / "gpqa.jsonl", GPQA_ROWS)
    man_text, prob_text = asyncio.run(_roundtrip_over_stdio({}, str(f)))
    man = json.loads(man_text)
    assert man["ids"] == ["u-aaa", "u-bbb"]
    prob = json.loads(prob_text)
    assert set(prob.keys()) == {"id", "prompt", "modality"}
    assert prob["prompt"].startswith("What is 2+2?")
    # The answer never crossed the boundary, even over the wire.
    assert "expected_answer" not in man_text and "expected_answer" not in prob_text
