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

"""HTTP server for worker agents in a multi-agent SLURM job.

Each worker agent runs this server inside its own SLURM het-group alongside
an LLM server.  The orchestrator agent calls workers via CallAgentTool, which
POSTs to this server's /task endpoint.

Communication protocol:
  POST /task   { task_id: str, messages: list[dict] }
               → { task_id, generation, conversation?, num_tool_calls, error? }
  GET  /health → { status: "ok", name: str }

Usage (via pipeline, started automatically by AgentWorkerScript):
    python -m nemo_skills.inference.agent_server \
        ++server.host=<LLM_HOST> ++server.port=<LLM_PORT> \
        ++server.server_type=vllm ++server.model=<MODEL_PATH> \
        ++agent_port=7777 ++agent_name=analyst \
        ++tool_modules=["nemo_skills.mcp.servers.python_tool::DirectPythonTool"]
"""

import asyncio
import logging
import sys
import uuid
from dataclasses import field

import hydra
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.generate import InferenceConfig
from nemo_skills.inference.model import get_tool_calling_model, get_model, server_params
from nemo_skills.inference.model.base import EndpointType
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    get_server_wait_cmd,
    nested_dataclass,
    setup_logging,
)
import subprocess

LOG = logging.getLogger(get_logger_name(__file__))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@nested_dataclass(kw_only=True)
class AgentServerConfig:
    """Configuration for a worker agent server."""

    # LLM server connection (same fields as GenerationTaskConfig.server)
    server: dict = field(default_factory=dict)

    # Port this agent server listens on for incoming task requests
    agent_port: int = 7777

    # Human-readable name for this worker (used in logs and /health)
    agent_name: str = "agent_worker"

    # Tool configuration (same semantics as GenerationTaskConfig)
    tool_modules: list = field(default_factory=list)
    tool_overrides: dict = field(default_factory=dict)
    schema_overrides: dict = field(default_factory=dict)
    max_tool_calls: int = -1

    # Sandbox configuration (for tools that need code execution)
    sandbox: dict = field(default_factory=dict)

    # LLM call parameters
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Concurrency: max simultaneous in-flight task requests
    max_concurrent_tasks: int = 64


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_agent_server_config", node=AgentServerConfig)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AgentTaskRequest(BaseModel):
    """Incoming task request from an orchestrator or peer agent."""

    task_id: str = ""
    # Full OpenAI-style conversation history, e.g.:
    #   [{"role": "user", "content": "Solve: ..."}, ...]
    messages: list


class AgentTaskResponse(BaseModel):
    """Result returned to the calling agent."""

    task_id: str
    generation: str
    conversation: list | None = None
    num_tool_calls: int = 0
    finish_reason: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class AgentServer:
    """Worker agent: FastAPI HTTP server backed by ToolCallingWrapper."""

    def __init__(self, cfg: AgentServerConfig):
        self.cfg = cfg
        self.semaphore = asyncio.Semaphore(cfg.max_concurrent_tasks)

        sandbox = get_sandbox(**cfg.sandbox) if cfg.sandbox else None

        if cfg.tool_modules:
            self.llm = get_tool_calling_model(
                **cfg.server,
                tool_modules=cfg.tool_modules,
                tool_overrides=cfg.tool_overrides,
                schema_overrides=cfg.schema_overrides,
                max_tool_calls=cfg.max_tool_calls,
                additional_config={"sandbox": cfg.sandbox} if cfg.sandbox else {},
            )
        else:
            LOG.warning("AgentServer %s has no tool_modules; using plain model.", cfg.agent_name)
            self.llm = get_model(**cfg.server)

        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI(title=f"Agent Worker: {self.cfg.agent_name}")

        @app.post("/task", response_model=AgentTaskResponse)
        async def process_task(request: AgentTaskRequest) -> AgentTaskResponse:
            return await self._handle_task(request)

        @app.get("/health")
        async def health():
            return {"status": "ok", "name": self.cfg.agent_name}

        return app

    async def _handle_task(self, request: AgentTaskRequest) -> AgentTaskResponse:
        task_id = request.task_id or str(uuid.uuid4())
        async with self.semaphore:
            try:
                from dataclasses import asdict, is_dataclass

                if is_dataclass(self.cfg.inference):
                    inference_params = asdict(self.cfg.inference)
                else:
                    inference_params = dict(self.cfg.inference)

                # Remove stream from params — server mode uses non-streaming for simplicity
                inference_params.pop("stream", None)

                result = await self.llm.generate_async(
                    prompt=request.messages,
                    endpoint_type=EndpointType.chat,
                    **inference_params,
                )

                return AgentTaskResponse(
                    task_id=task_id,
                    generation=result.get("generation", ""),
                    conversation=result.get("conversation"),
                    num_tool_calls=result.get("num_tool_calls", 0),
                    finish_reason=result.get("finish_reason"),
                )
            except Exception as exc:
                LOG.exception("Error processing task %s", task_id)
                return AgentTaskResponse(
                    task_id=task_id,
                    generation="",
                    num_tool_calls=0,
                    error=str(exc),
                )

    def wait_for_llm_server(self):
        server = self.cfg.server
        if not server.get("base_url") and not (server.get("host") and server.get("port")):
            LOG.info("Skipping LLM server wait — no address configured.")
            return
        address = server.get("base_url") or f"{server['host']}:{server['port']}"
        if address and address != "None":
            subprocess.run(get_server_wait_cmd(address), shell=True, check=True)

    def run(self):
        self.wait_for_llm_server()
        LOG.info(
            "Starting agent worker '%s' on port %d",
            self.cfg.agent_name,
            self.cfg.agent_port,
        )
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.cfg.agent_port,
            log_level="warning",
        )


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_name="base_agent_server_config")
def run_server(cfg: AgentServerConfig):
    cfg = AgentServerConfig(_init_nested=True, **cfg)
    LOG.info("Agent server config: %s", cfg)
    server = AgentServer(cfg)
    server.run()


HELP_MESSAGE = get_help_message(AgentServerConfig)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        run_server()
