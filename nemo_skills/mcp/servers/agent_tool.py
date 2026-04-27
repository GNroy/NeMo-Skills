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
For each configured worker it exposes a separate tool named `call_{agent_name}`
that the orchestrator (or any peer) can invoke.

Worker address resolution
-------------------------
Each worker runs in a separate SLURM het-group alongside its own LLM server.
NeMo-Run exports the master node hostname for each het-group as an environment
variable before starting Python:

    SLURM_MASTER_NODE_HET_GROUP_0   ← orchestrator's group
    SLURM_MASTER_NODE_HET_GROUP_1   ← first worker's group
    SLURM_MASTER_NODE_HET_GROUP_2   ← second worker's group
    …

CallAgentTool reads these at configure() time.  The `ns agent` pipeline
command automatically injects the correct het_group index for each worker via
tool_overrides.

Configuration example (via tool_overrides from ns agent):
    tool_overrides = {
        "CallAgentTool": {
            "agents": {
                "analyst": {"het_group": 1, "port": 7777},
                "coder":   {"het_group": 2, "port": 7777},
            }
        }
    }

For local testing (no SLURM), you can specify explicit addresses:
    tool_overrides = {
        "CallAgentTool": {
            "agents": {
                "analyst": {"host": "localhost", "port": 7778},
            }
        }
    }

The tool exposes:
    call_analyst(task: str) → str   (final generation from the analyst worker)
    call_coder(task: str)   → str

Peer-to-peer hierarchy
-----------------------
Hierarchy is enforced by configuring agents with different tool sets:
- Top-level orchestrator has CallAgentTool with all worker agents listed.
- Each worker agent has CallAgentTool with only the agents it is allowed to
  call (e.g., no agents at all → leaf node, or just a subset of peers).
This is configured purely through tool_overrides at job submission time.
"""

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
    It exposes one tool per worker: call_{agent_name}.
    """

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {
            # agents: {name: {het_group?: int, host?: str, port: int}}
            "agents": {},
            # Per-call timeout in seconds
            "timeout_s": _DEFAULT_TIMEOUT_S,
        }
        # Resolved at configure() time: {name: url}
        self._agent_urls: Dict[str, str] = {}

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
            # Merge overrides — replace top-level keys
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
            tools.append(
                {
                    "name": f"call_{agent_name}",
                    "description": (
                        f"Delegate a task to the '{agent_name}' specialist agent. "
                        f"The agent will use its own tools and reasoning to complete "
                        f"the task and return the result."
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
        return tools

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        extra_args: Dict[str, Any] | None = None,
    ) -> str:
        # tool_name is like "call_analyst" → agent_name is "analyst"
        if not tool_name.startswith("call_"):
            raise ValueError(f"CallAgentTool: unexpected tool name '{tool_name}'")
        agent_name = tool_name[len("call_"):]

        url = self._agent_urls.get(agent_name)
        if url is None:
            available = list(self._agent_urls.keys())
            raise ValueError(
                f"CallAgentTool: unknown agent '{agent_name}'. "
                f"Available agents: {available}"
            )

        task = arguments.get("task", "")
        if not task:
            return "Error: no task provided."

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

    async def shutdown(self) -> None:
        pass  # Nothing to clean up — httpx clients are created per-request
