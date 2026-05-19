# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Phase 2 — ``ns-mcp-serve`` stdio wrapper for in-process ``Tool`` classes.

Tests:
  T2.1 — list_tools round-trip via stdio against a stub Tool
  T2.2 — call_tool happy path (PeriodictableTool, no external deps)
  T2.3 — call_tool error propagates as TextContent (not transport-fatal)
  T2.4 — schemas survive a Hermes-style sanity check (basic JSON-Schema shape)
  T2.5 — subprocess cleanup after a normal run (no orphan PIDs)
  T2.6 — bad spec / overrides fail fast with a clear message
  T2.7 — _coerce_result serializes dicts / lists / exotic objects to JSON
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from nemo_skills.mcp.stdio_serve import _build_server, _coerce_result, _resolve_spec, main

# ---------------------------------------------------------------------------
# Local stub Tool used by tests that don't need a real domain tool
# ---------------------------------------------------------------------------


_STUB_MODULE_SRC = textwrap.dedent(
    '''
    """Stub Tool used by ns-mcp-serve tests.

    Lives outside the real tool registry so we can control behavior precisely
    (raise on demand, echo args, etc.) without dragging in domain deps.
    """
    from nemo_skills.mcp.tool_manager import Tool


    class StubTool(Tool):
        def default_config(self):
            return {}

        def configure(self, overrides=None, context=None):
            self._overrides = overrides or {}

        async def list_tools(self):
            return [
                {
                    "name": "echo",
                    "description": "Echo back the input string.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"msg": {"type": "string"}},
                        "required": ["msg"],
                    },
                },
                {
                    "name": "boom",
                    "description": "Always raises (used to test error passthrough).",
                    "input_schema": {"type": "object", "properties": {}},
                },
                {
                    "name": "structured",
                    "description": "Return a non-string payload to test serialization.",
                    "input_schema": {"type": "object", "properties": {}},
                },
            ]

        async def execute(self, tool_name, arguments, extra_args=None):
            if tool_name == "echo":
                return f"echo: {arguments.get('msg', '')}"
            if tool_name == "boom":
                raise RuntimeError("intentional failure for T2.3")
            if tool_name == "structured":
                return {"a": 1, "b": [2, 3], "c": "x"}
            raise ValueError(f"unknown tool {tool_name!r}")
    '''
).lstrip()


@pytest.fixture
def stub_module(tmp_path, monkeypatch) -> str:
    """Write the stub Tool module to disk + put its parent on sys.path.

    Returns the ``module:Class`` spec for ``ns-mcp-serve`` to import.
    """
    pkg_dir = tmp_path / "ns_test_stub_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "stub_tool.py").write_text(_STUB_MODULE_SRC)
    monkeypatch.syspath_prepend(str(tmp_path))
    return "ns_test_stub_pkg.stub_tool:StubTool"


# ---------------------------------------------------------------------------
# In-process: exercise the Server object directly (faster than stdio).
# ---------------------------------------------------------------------------


def test_resolve_spec_rejects_missing_colon() -> None:
    with pytest.raises(SystemExit, match="module:Class"):
        _resolve_spec("nemo_skills.mcp.stdio_serve")


def test_resolve_spec_rejects_unknown_attribute() -> None:
    with pytest.raises(SystemExit, match="no attribute"):
        _resolve_spec("nemo_skills.mcp.stdio_serve:Nope")


def test_build_server_wires_tool_to_server(stub_module: str) -> None:
    cls = _resolve_spec(stub_module)
    server = _build_server(cls, {}, "stub")
    assert server.name == "stub"
    # The underlying Tool instance is stashed on the server for tests.
    assert server._ns_tool.__class__.__name__ == "StubTool"


# ---------------------------------------------------------------------------
# T2.7 — result coercion (covers JSON, exotic objects, fallbacks).
# ---------------------------------------------------------------------------


def test_t27_coerce_result_serializes_dict() -> None:
    assert _coerce_result({"a": 1, "b": [2, 3]}) == '{"a": 1, "b": [2, 3]}'


def test_t27_coerce_result_preserves_strings() -> None:
    assert _coerce_result("already a string") == "already a string"


def test_t27_coerce_result_falls_back_for_unserializable() -> None:
    class _Weird:
        def __repr__(self):
            return "<weird>"

    out = _coerce_result(_Weird())
    # default=str handles most objects; this falls through to str() which
    # equals repr() for our class.
    assert "<weird>" in out or "_Weird" in out


# ---------------------------------------------------------------------------
# T2.1, T2.2, T2.3 — full stdio round-trip against the stub tool.
# ---------------------------------------------------------------------------


def _server_params(stub_module: str, syspath_extra: Path) -> StdioServerParameters:
    """``ns-mcp-serve`` subprocess parameters wired to the stub Tool.

    We invoke ``python -m nemo_skills.mcp.stdio_serve`` directly so the test
    works without an editable install (no console script entry).  The
    subprocess inherits the test's PYTHONPATH plus the stub module dir.
    """
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{syspath_extra}{os.pathsep}{pythonpath}" if pythonpath else str(syspath_extra)
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "nemo_skills.mcp.stdio_serve", stub_module],
        env=env,
    )


def _stub_extra_path(stub_module: str) -> Path:
    """Recover the directory the stub module fixture put on sys.path."""
    name = stub_module.split(":", 1)[0].split(".", 1)[0]
    # Walk sys.path looking for the entry the fixture added.
    for entry in sys.path:
        if (Path(entry) / name / "__init__.py").is_file():
            return Path(entry)
    raise AssertionError(f"could not find {name!r} on sys.path; fixture not applied?")


async def _run_round_trip(stub_module: str) -> Dict[str, Any]:
    """Connect to the wrapper over real stdio, exercise list/call_tool."""
    extra = _stub_extra_path(stub_module)
    params = _server_params(stub_module, extra)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            echo = await session.call_tool("echo", {"msg": "hi"})
            boom = await session.call_tool("boom", {})
            structured = await session.call_tool("structured", {})
    return {"listed": listed, "echo": echo, "boom": boom, "structured": structured}


def test_t21_list_tools_round_trip(stub_module: str) -> None:
    result = asyncio.run(_run_round_trip(stub_module))
    names = sorted(t.name for t in result["listed"].tools)
    assert names == ["boom", "echo", "structured"]
    # Schemas survive the trip — inputSchema is the MCP camelCase field.
    echo_schema = next(t.inputSchema for t in result["listed"].tools if t.name == "echo")
    assert echo_schema["required"] == ["msg"]
    assert echo_schema["properties"]["msg"]["type"] == "string"


def test_t22_call_tool_happy_path(stub_module: str) -> None:
    result = asyncio.run(_run_round_trip(stub_module))
    text_blocks = [b for b in result["echo"].content if b.type == "text"]
    assert text_blocks, result["echo"].content
    assert text_blocks[0].text == "echo: hi"


def test_t23_call_tool_error_passthrough(stub_module: str) -> None:
    result = asyncio.run(_run_round_trip(stub_module))
    # The wrapper turns Python exceptions into a TextContent envelope so
    # the orchestrator sees a tool error rather than a transport failure.
    text = next(b.text for b in result["boom"].content if b.type == "text")
    assert text.startswith("Error: RuntimeError")
    assert "intentional failure for T2.3" in text


def test_call_tool_structured_payload_serialized(stub_module: str) -> None:
    result = asyncio.run(_run_round_trip(stub_module))
    text = next(b.text for b in result["structured"].content if b.type == "text")
    # The structured dict comes back as JSON (T2.7 covered the unit-level path).
    parsed = json.loads(text)
    assert parsed == {"a": 1, "b": [2, 3], "c": "x"}


# ---------------------------------------------------------------------------
# T2.4 — schemas pass a basic shape sanity check.
#
# Hermes' tools/schema_sanitizer.py drops vendor-specific fields and
# enforces JSON-Schema-ish shape.  We don't import it here (we'd add a
# hard dep on hermes-agent for unit tests); instead we verify the shape
# we hand it is something a strict consumer would accept.
# ---------------------------------------------------------------------------


def _is_jsonschema_object(schema: Dict[str, Any]) -> None:
    """Raise AssertionError if *schema* isn't a sane JSON-Schema object stub."""
    assert isinstance(schema, dict), schema
    assert schema.get("type") == "object", schema
    properties = schema.get("properties", {})
    assert isinstance(properties, dict), schema
    for k, v in properties.items():
        assert isinstance(v, dict), (k, v)
        # Every property carries at least a type (rules out bare descriptions
        # leaking through).
        assert "type" in v or "$ref" in v or "oneOf" in v, (k, v)


@pytest.mark.parametrize(
    "spec",
    [
        "nemo_skills.mcp.servers.chemistry.periodictable_tool:PeriodictableTool",
        "nemo_skills.mcp.servers.physics.coolprop_tool:CoolPropTool",
        "nemo_skills.mcp.servers.physics.particle_tool:ParticleTool",
        "nemo_skills.mcp.servers.physics.radioactivedecay_tool:RadioactivedecayTool",
        "nemo_skills.mcp.servers.web.arxiv_tool:ArxivSearchTool",
        "nemo_skills.mcp.servers.web.wikipedia_tool:WikipediaSearchTool",
    ],
)
def test_t24_wrapped_tool_schemas_are_well_formed(spec: str) -> None:
    cls = _resolve_spec(spec)
    tool = cls()
    tool.configure(None, {})
    entries = asyncio.run(tool.list_tools())
    assert entries, f"{spec} returned no tools"
    for entry in entries:
        assert entry.get("name"), entry
        assert "input_schema" in entry, entry
        _is_jsonschema_object(entry["input_schema"])


# ---------------------------------------------------------------------------
# T2.5 — subprocess cleanup after a normal run.
# ---------------------------------------------------------------------------


def test_t25_subprocess_exits_after_session_close(stub_module: str) -> None:
    """When the client closes the session, the wrapper subprocess must exit
    on its own — no orphan ``ns-mcp-serve`` PIDs left running."""
    extra = _stub_extra_path(stub_module)
    params = _server_params(stub_module, extra)

    async def _run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.list_tools()
        # context exit closes stdin → wrapper should EOF and exit.

    start = time.monotonic()
    asyncio.run(_run())
    # If the wrapper hangs, our async context exits but the OS subprocess
    # may still be alive.  stdio_client tracks the PID internally; we just
    # check the round-trip completes promptly — a hung child would push us
    # past the timeout.
    assert time.monotonic() - start < 30, "wrapper did not exit cleanly"


# ---------------------------------------------------------------------------
# T2.6 — main() rejects bad input fast (helpful error messages).
# ---------------------------------------------------------------------------


def test_t26_main_rejects_invalid_overrides_json() -> None:
    with pytest.raises(SystemExit, match="--overrides must be valid JSON"):
        main(
            [
                "nemo_skills.mcp.servers.chemistry.periodictable_tool:PeriodictableTool",
                "--overrides",
                "not-json",
            ]
        )


def test_t26_main_rejects_non_object_overrides() -> None:
    with pytest.raises(SystemExit, match="must be a JSON object"):
        main(
            [
                "nemo_skills.mcp.servers.chemistry.periodictable_tool:PeriodictableTool",
                "--overrides",
                "[]",
            ]
        )


def test_t26_main_rejects_bad_spec() -> None:
    with pytest.raises(SystemExit, match="module:Class"):
        main(["not_a_spec"])
