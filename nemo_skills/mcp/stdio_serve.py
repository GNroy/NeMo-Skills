# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Generic stdio MCP wrapper for ``nemo_skills.mcp.tool_manager.Tool``.

Usage::

    ns-mcp-serve <module>:<Class> [--overrides JSON] [--server-name NAME]

The wrapper imports the named ``Tool`` subclass, instantiates it, applies
``--overrides`` via ``tool.configure(...)``, then runs an MCP stdio server
that exposes every entry from ``tool.list_tools()`` as a callable MCP
tool.  Calls are routed to ``tool.execute(...)`` and the return value is
serialized to a single ``TextContent`` block (JSON-encoded when the
result is not already a string).

Why a generic wrapper instead of per-tool ``def main()`` blocks like
``python_tool.py`` / ``tavily_search_tool.py``?  Many domain tools
(``chemistry/periodictable_tool.py``, ``physics/*.py``, ``web/*.py``) only
implement the in-process ``Tool`` interface — they have no FastMCP entry
point.  This wrapper gives them one without duplicating boilerplate.

Hermes consumes the server via ``~/.hermes/config.yaml``::

    mcp_servers:
      ns_periodictable:
        command: "ns-mcp-serve"
        args: ["nemo_skills.mcp.servers.chemistry.periodictable_tool:PeriodictableTool"]

Tool names emerge on the Hermes side as ``ns_periodictable.element_info``
etc. — Hermes prefixes by server name (see ``tools/mcp_tool.py``).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import logging
import sys
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent
from mcp.types import Tool as MCPTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec resolution
# ---------------------------------------------------------------------------


def _resolve_spec(spec: str):
    """Parse ``module:Class`` and return the Tool subclass."""
    if ":" not in spec:
        raise SystemExit(
            f"--spec must be in 'module:Class' form, got {spec!r}.\n"
            f"Example: nemo_skills.mcp.servers.python_tool:DirectPythonTool"
        )
    module_name, _, class_name = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"could not import {module_name!r}: {exc}")
    if not hasattr(module, class_name):
        raise SystemExit(f"module {module_name!r} has no attribute {class_name!r}")
    cls = getattr(module, class_name)
    if not inspect.isclass(cls):
        raise SystemExit(f"{spec!r} resolves to a {type(cls).__name__}, not a class")
    return cls


def _coerce_result(value: Any) -> str:
    """Serialize a tool execute() return value to text for MCP TextContent.

    Strings pass through verbatim; everything else is JSON-encoded with a
    ``default=str`` fallback so exotic objects (dataclasses, numpy scalars,
    etc.) don't crash the round-trip.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        return repr(value)


def _build_server(tool_cls, overrides: Dict[str, Any], server_name: str) -> Server:
    """Construct the MCP Server wired against a fresh ``tool_cls()`` instance."""
    tool = tool_cls()
    tool.configure(overrides or None, {})
    # Many Tool subclasses use ``post_configure()`` to finalize state once
    # config + context are settled (e.g. opening an HTTP client).  Honor it
    # the same way ToolManager does.
    post_configure = getattr(tool, "post_configure", None)
    if callable(post_configure):
        post_configure()

    server = Server(name=server_name)

    @server.list_tools()
    async def _mcp_list_tools() -> List[MCPTool]:
        entries = await tool.list_tools() or []
        out: List[MCPTool] = []
        seen: set[str] = set()
        for entry in entries:
            name = entry.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            # NS tools use ``input_schema`` (camelCase MCP expects
            # ``inputSchema``).  Pass through verbatim.  Fall back to a
            # permissive empty-object schema when omitted.
            schema = entry.get("input_schema") or {"type": "object", "properties": {}}
            out.append(
                MCPTool(
                    name=name,
                    description=entry.get("description", "") or "",
                    inputSchema=schema,
                )
            )
        return out

    @server.call_tool()
    async def _mcp_call_tool(name: str, arguments: Dict[str, Any] | None):
        try:
            result = await tool.execute(name, arguments or {}, {})
        except Exception as exc:  # noqa: BLE001 - any failure becomes a tool error
            logger.exception("ns-mcp-serve: tool %r failed", name)
            text = f"Error: {type(exc).__name__}: {exc}"
            return [TextContent(type="text", text=text)]
        return [TextContent(type="text", text=_coerce_result(result))]

    # Expose for tests; ToolManager-style introspection if a caller wants it.
    server._ns_tool = tool  # type: ignore[attr-defined]
    return server


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ns-mcp-serve",
        description="Expose a nemo_skills.mcp.tool_manager.Tool as an MCP stdio server.",
    )
    p.add_argument(
        "spec",
        help="Fully-qualified module:Class reference, e.g. nemo_skills.mcp.servers.physics.coolprop_tool:CoolPropTool",
    )
    p.add_argument(
        "--overrides",
        default="{}",
        help="JSON object forwarded to Tool.configure() (default: {}).",
    )
    p.add_argument(
        "--server-name",
        default=None,
        help="Server name advertised to MCP clients (defaults to the Tool "
        "class name; Hermes prefixes tool names with the server name).",
    )
    return p


async def _serve(server: Server) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(
        # MCP servers log to stderr so stdout stays a pure JSON-RPC channel.
        stream=sys.stderr,
        level="INFO",
        format="%(asctime)s ns-mcp-serve %(levelname)s %(message)s",
    )
    args = _build_parser().parse_args(argv)

    try:
        overrides = json.loads(args.overrides)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--overrides must be valid JSON: {exc}")
    if not isinstance(overrides, dict):
        raise SystemExit("--overrides must be a JSON object")

    tool_cls = _resolve_spec(args.spec)
    server_name = args.server_name or tool_cls.__name__
    server = _build_server(tool_cls, overrides, server_name)

    logger.info("starting MCP stdio server %s for %s", server_name, args.spec)
    asyncio.run(_serve(server))
    return 0


if __name__ == "__main__":
    sys.exit(main())
