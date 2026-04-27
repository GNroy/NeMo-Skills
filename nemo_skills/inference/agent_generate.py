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

"""Agent generation task.

Extends the standard generation task with:
- Trajectory saving (full conversation + tool call counts)
- Enforcement of tool_modules (agents need tools)

Worker/server mode lives in agent_server.py.  This module is the orchestrator
that reads an input JSONL, drives the ToolCallingWrapper loop for each sample,
and writes enriched JSONL results.

Usage (via pipeline):
    ns agent --model ... --server-type vllm ...

Usage (direct Hydra):
    python -m nemo_skills.inference.agent_generate \
        ++server.host=localhost ++server.port=5000 ++server.server_type=vllm \
        ++input_file=/data/problems.jsonl ++output_file=/results/output.jsonl \
        ++tool_modules=["nemo_skills.mcp.servers.python_tool::DirectPythonTool"]
"""

import logging
import sys

import hydra

from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class AgentTaskConfig(GenerationTaskConfig):
    """Agent generation config.

    Inherits all fields from GenerationTaskConfig and adds agent-specific options.
    Key differences from vanilla generation:
    - prompt_format defaults to "openai"   (agents use chat messages)
    - save_trajectory defaults to True     (full conversation saved in JSONL)
    """

    # Override parent default: agents use chat messages format
    prompt_format: str = "openai"

    # Save full agent trajectory (conversation turns) in output JSONL.
    # Set to False to keep output files smaller when only the final answer matters.
    save_trajectory: bool = True

    def _get_disallowed_params(self):
        # Agent tasks do not restrict any generation parameters.
        return []


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_agent_config", node=AgentTaskConfig)


class AgentTask(GenerationTask):
    """Generation task for agentic loops.

    Wraps GenerationTask with agent-specific output handling:
    - Preserves num_tool_calls and finish_reason from ToolCallingWrapper
    - Optionally preserves full conversation history (save_trajectory=True)
    - Strips the tools schema (not useful in output files)
    - Warns when tool_modules is not configured
    """

    def __init__(self, cfg: AgentTaskConfig):
        if not cfg.tool_modules:
            LOG.warning(
                "AgentTask initialized without tool_modules. "
                "Agents need at least one tool to be useful. "
                "Set ++tool_modules=[...] to configure tools."
            )
        super().__init__(cfg)

    async def postprocess_single_output(self, output, original_data_point):
        await super().postprocess_single_output(output, original_data_point)

        # ToolCallingWrapper returns these fields; preserve them in output.
        # num_tool_calls and finish_reason are always kept (small, informative).
        # conversation is kept only when save_trajectory=True.
        if not self.cfg.save_trajectory:
            output.pop("conversation", None)

        # Drop the tools schema — it's large and not needed downstream.
        output.pop("tools", None)


# Required sentinel for the pipeline to locate the task class.
GENERATION_TASK_CLASS = AgentTask


@hydra.main(version_base=None, config_name="base_agent_config")
def agent_generate(cfg: AgentTaskConfig):
    cfg = AgentTaskConfig(_init_nested=True, **cfg)
    LOG.info("Agent config: %s", cfg)
    task = AgentTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(AgentTaskConfig)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        agent_generate()
