# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MCP tool for delegating tasks to worker agents in a multi-agent SLURM job.

CallAgentTool wraps one or more agent workers behind a single MCP tool class.
For each configured worker it exposes three tools:

    call_{name}(task)
        Blocking call — awaits the full agent response before returning.

    call_{name}_async(task) → {"ticket_id": "...", "status": "pending"}
        Non-blocking call — fires the HTTP request as a background asyncio
        Task and returns immediately.  Results are injected into the
        conversation automatically (via get_pending_injections) before
        the next LLM turn.  collect_results() can be used for an explicit
        blocking wait.

    collect_results(ticket_ids)
        Block until all specified ticket_ids are complete; return their
        results.  Tickets already auto-injected are returned from cache.

Worker address resolution
-------------------------
Each worker runs in a separate SLURM het-group alongside its own LLM server.
NeMo-Run exports the master node hostname for each het-group as an environment
variable before starting Python:

    SLURM_MASTER_NODE_HET_GROUP_0   <- orchestrator's group
    SLURM_MASTER_NODE_HET_GROUP_1   <- first worker's group
    ...

CallAgentTool reads these at configure() time.  The `ns agent` pipeline
command automatically injects the correct het_group index for each worker via
tool_overrides.

Configuration example (via tool_overrides from ns agent):
    tool_overrides = {
        "CallAgentTool": {
            "agents": {
                "analyst": {"het_group": 1, "port": 7777},
                "coder":   {"het_group": 2, "port": 7777},
            },
            "injection_role": "user",   # role for auto-injected async results
        }
    }

injection_role
--------------
Controls the message role used when a completed async result is injected into
the conversation before the next LLM turn.  Accepted values:
    "user"   (default) — safest for cross-model compatibility
    "system" — semantically more accurate but not supported by all models
Set via ++tool_overrides.CallAgentTool.injection_role=system.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, List

import httpx

from nemo_skills.mcp.tool_manager import FatalToolError, Tool

LOG = logging.getLogger(__name__)

# Default timeout for a single agent-to-agent call (seconds).
# Keep generous — workers may take many LLM turns to respond.
_DEFAULT_TIMEOUT_S = 600.0


class CallAgentTool(Tool):
    """Delegate tasks to worker agents via HTTP.

    One instance of this class handles all configured workers.
    It exposes one blocking tool, one async tool, and a shared
    collect_results tool per worker set.
    """

    # Approximate chars-per-token ratio, matching ToolCallingWrapper.
    _CHARS_PER_TOKEN = 4

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {
            # agents: {name: {het_group?: int, host?: str, port: int}}
            "agents": {},
            # Per-call timeout in seconds
            "timeout_s": _DEFAULT_TIMEOUT_S,
            # Role used when auto-injecting completed async results
            "injection_role": "user",
            # If >= 0, truncate injected/collected worker results to this many
            # tokens (tail kept).  Prevents long worker outputs from blowing
            # up the orchestrator's context window.
            "max_injection_tokens": -1,
            # Whether to expose collect_results as a tool to the LLM.
            # False (default): results arrive only via auto-injection; the LLM
            # cannot poll explicitly.  True: restore the explicit polling tool.
            "expose_collect_results": False,
        }
        # Resolved at configure() time: {name: url}
        self._agent_urls: Dict[str, str] = {}

        # Async delegation state — all keyed by request_id to scope to one
        # orchestrator loop invocation (one problem being solved).
        #
        # _pending_tasks:   request_id → {ticket_id → asyncio.Task[str]}
        # _request_tickets: request_id → [ticket_id, ...]  (for cleanup)
        # _ticket_meta:     ticket_id  → {agent_name, task_snippet}
        # _ticket_results:  ticket_id  → str result  (after task completes)
        self._pending_tasks: Dict[str, Dict[str, asyncio.Task]] = {}
        self._request_tickets: Dict[str, List[str]] = {}
        self._ticket_meta: Dict[str, Dict[str, str]] = {}
        self._ticket_results: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def default_config(self) -> Dict[str, Any]:
        return dict(self._config)

    def configure(
        self,
        overrides: Dict[str, Any] | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        if overrides:
            for k, v in overrides.items():
                if k == "agents" and isinstance(v, dict) and isinstance(self._config.get("agents"), dict):
                    # Deep-merge agent entries so partial overrides work
                    for agent_name, agent_cfg in v.items():
                        self._config["agents"].setdefault(agent_name, {})
                        self._config["agents"][agent_name].update(agent_cfg)
                else:
                    self._config[k] = v

        self._agent_urls = {}
        for agent_name, agent_cfg in self._config.get("agents", {}).items():
            if "host" in agent_cfg:
                host = agent_cfg["host"]
            elif "het_group" in agent_cfg:
                het_idx = agent_cfg["het_group"]
                env_key = f"SLURM_MASTER_NODE_HET_GROUP_{het_idx}"
                host = os.environ.get(env_key, "localhost")
                if host == "localhost":
                    LOG.warning(
                        "Env var %s not found; defaulting to localhost for agent '%s'. "
                        "This is expected in local testing but should not happen in a SLURM job.",
                        env_key,
                        agent_name,
                    )
            else:
                raise ValueError(
                    f"Agent '{agent_name}' must specify either 'host' or 'het_group' in its config."
                )
            port = agent_cfg.get("port", 7777)
            self._agent_urls[agent_name] = f"http://{host}:{port}"
            LOG.info("CallAgentTool: agent '%s' → %s", agent_name, self._agent_urls[agent_name])

    async def list_tools(self) -> List[Dict[str, Any]]:
        tools = []
        for agent_name in self._agent_urls:
            # Blocking variant
            tools.append(
                {
                    "name": f"call_{agent_name}",
                    "description": (
                        f"Delegate a task to the '{agent_name}' specialist agent and wait for the result. "
                        f"Use when you need the result before proceeding."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": (
                                    "The complete task description for the agent. "
                                    "Be specific — include all context the agent needs."
                                ),
                            }
                        },
                        "required": ["task"],
                    },
                }
            )
            # Non-blocking (async) variant
            tools.append(
                {
                    "name": f"call_{agent_name}_async",
                    "description": (
                        f"Delegate a task to the '{agent_name}' specialist agent without blocking. "
                        f"Returns a ticket_id immediately; the result will be automatically injected "
                        f"into this conversation when the agent finishes. "
                        f"Use for independent sub-tasks that can run while you continue reasoning."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": (
                                    "The complete task description for the agent. "
                                    "Be specific — include all context the agent needs."
                                ),
                            }
                        },
                        "required": ["task"],
                    },
                }
            )

        # collect_results is only meaningful if there are agents to call,
        # and only exposed when explicitly enabled (default: False so the LLM
        # cannot poll and must rely on auto-injection).
        if self._agent_urls and self._config.get("expose_collect_results", False):
            tools.append(
                {
                    "name": "collect_results",
                    "description": (
                        "Block until all specified async tickets complete and return their results. "
                        "Use when you need results from previously-fired async calls before proceeding. "
                        "Tickets that have already been auto-injected are returned from cache."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "ticket_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of ticket_ids returned by call_{name}_async.",
                            }
                        },
                        "required": ["ticket_ids"],
                    },
                }
            )

        return tools

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        extra_args: Dict[str, Any] | None = None,
    ) -> str:
        request_id = (extra_args or {}).get("request_id", "")

        if tool_name == "collect_results":
            return await self._collect_results(arguments.get("ticket_ids", []), request_id)

        if tool_name.endswith("_async") and tool_name.startswith("call_"):
            agent_name = tool_name[len("call_"):-len("_async")]
            return await self._fire_async(agent_name, arguments.get("task", ""), request_id)

        if tool_name.startswith("call_"):
            agent_name = tool_name[len("call_"):]
            return await self._call_blocking(agent_name, arguments.get("task", ""), request_id)

        raise ValueError(f"CallAgentTool: unexpected tool name '{tool_name}'")

    # ------------------------------------------------------------------
    # Blocking call
    # ------------------------------------------------------------------

    async def _call_blocking(self, agent_name: str, task: str, request_id: str) -> str:
        url = self._get_url(agent_name)
        if not task:
            return "Error: no task provided."
        return await self._http_call(agent_name, url, task)

    # ------------------------------------------------------------------
    # Non-blocking (async) call
    # ------------------------------------------------------------------

    async def _fire_async(self, agent_name: str, task: str, request_id: str) -> str:
        """Fire the HTTP call as a background task; return ticket immediately."""
        url = self._get_url(agent_name)
        if not task:
            return json.dumps({"error": "no task provided"})

        ticket_id = str(uuid.uuid4())

        # Store metadata for the injection label (truncate for readability)
        snippet = task[:80].replace("\n", " ")
        self._ticket_meta[ticket_id] = {"agent_name": agent_name, "task_snippet": snippet}

        # Fire the HTTP call as a background asyncio Task
        bg_task = asyncio.create_task(self._http_call(agent_name, url, task))

        # Register under request_id for scoped lookup and cleanup
        self._pending_tasks.setdefault(request_id, {})[ticket_id] = bg_task
        self._request_tickets.setdefault(request_id, []).append(ticket_id)

        LOG.info(
            "CallAgentTool: dispatched async task ticket=%s agent=%s request_id=%s",
            ticket_id,
            agent_name,
            request_id,
        )
        return json.dumps({"ticket_id": ticket_id, "status": "pending"})

    async def _collect_results(self, ticket_ids: List[str], request_id: str) -> str:
        """Block until all specified tickets complete; return structured results."""
        if not ticket_ids:
            return json.dumps([])

        results = []
        tasks_to_await: Dict[str, asyncio.Task] = {}

        for tid in ticket_ids:
            # Already drained by auto-injection — result is in cache
            if tid in self._ticket_results:
                results.append(
                    {
                        "ticket_id": tid,
                        "status": "already_injected",
                        "result": self._ticket_results[tid],
                    }
                )
                continue

            task = self._pending_tasks.get(request_id, {}).get(tid)
            if task is None:
                results.append({"ticket_id": tid, "status": "not_found"})
                continue

            tasks_to_await[tid] = task

        if tasks_to_await:
            outcomes = await asyncio.gather(*tasks_to_await.values(), return_exceptions=True)
            for tid, outcome in zip(tasks_to_await.keys(), outcomes):
                result_str = f"Error: {outcome}" if isinstance(outcome, Exception) else str(outcome)

                # Drain from pending and cache
                self._pending_tasks.get(request_id, {}).pop(tid, None)
                self._ticket_results[tid] = result_str

                meta = self._ticket_meta.get(tid, {})
                results.append(
                    {
                        "ticket_id": tid,
                        "status": "completed",
                        "agent": meta.get("agent_name", ""),
                        "result": self._truncate_result(result_str),
                    }
                )

        return json.dumps(results)

    # ------------------------------------------------------------------
    # Async injection hook
    # ------------------------------------------------------------------

    async def get_pending_injections(self, request_id: str) -> List[Dict]:
        """Drain any completed background tasks and return them as messages.

        Called by ToolCallingWrapper before each LLM turn.  Completed tasks
        are removed from _pending_tasks, their results cached in
        _ticket_results, and formatted messages returned for injection into
        the conversation.
        """
        pending = self._pending_tasks.get(request_id, {})
        if not pending:
            return []

        completed_ids = [tid for tid, task in pending.items() if task.done()]
        if not completed_ids:
            return []

        injection_role = self._config.get("injection_role", "user")
        injections = []

        for tid in completed_ids:
            task = pending.pop(tid)
            try:
                result = task.result()
            except Exception as exc:
                result = f"Error: {exc}"

            # Cache the full result so collect_results can serve it later
            self._ticket_results[tid] = result

            meta = self._ticket_meta.get(tid, {})
            agent_name = meta.get("agent_name", "unknown")
            snippet = meta.get("task_snippet", "")
            label = f'[Async result for ticket {tid} — call_{agent_name}_async "{snippet}"]'
            content = f"{label}:\n{self._truncate_result(result)}"

            injections.append({"role": injection_role, "content": content})
            LOG.info(
                "CallAgentTool: injecting completed ticket=%s agent=%s",
                tid,
                agent_name,
            )

        return injections

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup_request(self, request_id: str) -> None:
        """Cancel any still-running background tasks for this request."""
        pending = self._pending_tasks.pop(request_id, {})
        for tid, task in pending.items():
            if not task.done():
                LOG.warning(
                    "CallAgentTool: cancelling pending async task ticket=%s at request end", tid
                )
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # Clean up metadata for all tickets belonging to this request
        for tid in self._request_tickets.pop(request_id, []):
            self._ticket_meta.pop(tid, None)
            self._ticket_results.pop(tid, None)

    async def shutdown(self) -> None:
        pass  # httpx clients are created per-request; nothing to clean up

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate_result(self, result: str) -> str:
        """Truncate a worker result to max_injection_tokens if configured."""
        max_tokens = self._config.get("max_injection_tokens", -1)
        if max_tokens < 0:
            return result
        max_chars = max_tokens * self._CHARS_PER_TOKEN
        if len(result) <= max_chars:
            return result
        return (
            f"[result truncated — {len(result)} chars total, "
            f"showing last ~{max_tokens} tokens]\n"
            + result[-max_chars:]
        )

    def _get_url(self, agent_name: str) -> str:
        url = self._agent_urls.get(agent_name)
        if url is None:
            available = list(self._agent_urls.keys())
            raise ValueError(
                f"CallAgentTool: unknown agent '{agent_name}'. "
                f"Available agents: {available}"
            )
        return url

    async def _http_call(self, agent_name: str, url: str, task: str) -> str:
        """Perform the HTTP POST to the agent server and return its generation."""
        task_id = str(uuid.uuid4())
        timeout = self._config.get("timeout_s", _DEFAULT_TIMEOUT_S)
        payload = {
            "task_id": task_id,
            "messages": [{"role": "user", "content": task}],
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{url}/task", json=payload)
                response.raise_for_status()
                result = response.json()
        except httpx.TimeoutException:
            LOG.error("Timeout calling agent '%s' (task_id=%s)", agent_name, task_id)
            return f"Error: agent '{agent_name}' timed out after {timeout}s."
        except httpx.HTTPError as exc:
            LOG.error("HTTP error calling agent '%s': %s", agent_name, exc)
            return f"Error: could not reach agent '{agent_name}': {exc}"

        if result.get("error"):
            LOG.warning("Agent '%s' returned error: %s", agent_name, result["error"])
            return f"Error from agent '{agent_name}': {result['error']}"

        return result.get("generation", "")
