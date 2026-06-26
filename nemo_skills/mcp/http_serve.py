# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Generic **HTTP (StreamableHTTP) MCP** wrapper for ``nemo_skills.mcp.tool_manager.Tool``.

The stdio sibling (``stdio_serve.py``, a.k.a. ``ns-mcp-serve``) spawns ONE server
*subprocess per MCP client*.  In the agentic swarm that means 3 stdio servers
(worklog / benchmark / python) per delegate child × ~129 children = ~390 procs
holding ~28 GB just for the MCP layer (see LAZY_ALLOCATION_NEXT_SESSION_PLAN.md
Part 2).  This wrapper instead serves the SAME ``Tool`` as ONE long-lived HTTP
service that every child reaches by URL — collapsing N subprocesses to 1.

It reuses ``stdio_serve._build_server`` verbatim (same ``Tool.list_tools`` /
``Tool.execute`` wiring, same result coercion), so a tool behaves identically
whether served over stdio or HTTP.  Only the transport differs: a
``StreamableHTTPSessionManager`` mounted in a Starlette app under ``--path``
(default ``/mcp``), run by uvicorn.

Usage::

    ns-mcp-serve-http nemo_skills.mcp.servers.agentic.worklog_tool:WorklogTool \\
        --host 0.0.0.0 --port 9001 --path /mcp \\
        --overrides '{"run_id": "run1"}'

The hermes-agent MCP client then consumes it by URL instead of command::

    mcp_servers:
      worklog:
        url: "http://<node>:9001/mcp"          # StreamableHTTP (default)
        # transport: sse                        # only if served with --sse

**Statelessness.**  Default is ``--stateless`` (each JSON-RPC request is handled
independently, no server-side MCP session persisted).  This is the right mode
for a service fanned out to many short-lived clients: no per-client session
state accumulates on the server, and our agentic tools key their own state by
call arguments (worklog by ``task_id`` + per-call ``agent_id``; benchmark by the
shared ``benchmark_path`` + problem ``id``) rather than by MCP session.  Pass
``--stateful`` to keep classic per-session semantics if a tool ever needs them.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from typing import Any, Dict, List

import uvicorn
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

# Reuse the exact stdio wiring so HTTP and stdio behave identically.
from nemo_skills.mcp.stdio_serve import _build_server, _resolve_spec

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ns-mcp-serve-http",
        description="Expose a nemo_skills.mcp.tool_manager.Tool as a StreamableHTTP MCP server.",
    )
    p.add_argument(
        "spec",
        help="Fully-qualified module:Class reference, e.g. "
        "nemo_skills.mcp.servers.agentic.worklog_tool:WorklogTool",
    )
    p.add_argument("--overrides", default="{}", help="JSON object forwarded to Tool.configure() (default: {}).")
    p.add_argument(
        "--server-name",
        default=None,
        help="Server name advertised to MCP clients (defaults to the Tool class name).",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    p.add_argument("--port", type=int, default=9000, help="Bind port (default: 9000).")
    p.add_argument(
        "--path",
        default="/mcp",
        help="HTTP mount path the StreamableHTTP endpoint lives at (default: /mcp). "
        "The client's url must end with this path.",
    )
    p.add_argument(
        "--json-response",
        action="store_true",
        default=False,
        help="Return a single JSON response per request instead of an SSE stream. "
        "Slightly lighter for pure request/response tools; off by default.",
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--stateless",
        dest="stateless",
        action="store_true",
        default=True,
        help="Handle each request independently with no persisted MCP session (default). "
        "Best for a service fanned out to many short-lived clients.",
    )
    grp.add_argument(
        "--stateful",
        dest="stateless",
        action="store_false",
        help="Keep classic per-client MCP session state on the server.",
    )
    return p


def build_app(spec: str, overrides: Dict[str, Any], server_name: str, *, path: str, json_response: bool,
              stateless: bool) -> Starlette:
    """Build the Starlette ASGI app serving ``spec`` over StreamableHTTP at ``path``."""
    tool_cls = _resolve_spec(spec)
    server = _build_server(tool_cls, overrides, server_name)

    session_manager = StreamableHTTPSessionManager(
        app=server,
        json_response=json_response,
        stateless=stateless,
    )

    async def _handle(scope: Scope, receive: Receive, send: Send) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def _lifespan(app: Starlette):
        # The session manager's run() context owns the StreamableHTTP task group;
        # it must wrap the whole serving lifetime.
        async with session_manager.run():
            logger.info("ns-mcp-serve-http: %s ready at %s (stateless=%s)", server_name, path, stateless)
            yield

    norm = "/" + path.strip("/")
    return Starlette(routes=[Mount(norm, app=_handle)], lifespan=_lifespan)


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(
        stream=sys.stderr,
        level="INFO",
        format="%(asctime)s ns-mcp-serve-http %(levelname)s %(message)s",
    )
    args = _build_parser().parse_args(argv)

    try:
        overrides = json.loads(args.overrides)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--overrides must be valid JSON: {exc}")
    if not isinstance(overrides, dict):
        raise SystemExit("--overrides must be a JSON object")

    server_name = args.server_name or _resolve_spec(args.spec).__name__
    app = build_app(
        args.spec,
        overrides,
        server_name,
        path=args.path,
        json_response=args.json_response,
        stateless=args.stateless,
    )

    logger.info(
        "starting StreamableHTTP MCP server %s for %s on %s:%d%s",
        server_name, args.spec, args.host, args.port, "/" + args.path.strip("/"),
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
