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
from pathlib import Path

import hydra
import yaml

from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))

# Load the agent system message from the canonical YAML once at import time.
# This is injected via system_message so that benchmark-specific prompt_configs
# (e.g. generic/physics with its LaTeX/boxed instructions) are used as-is for
# the user turn while the agent tool-use instructions still appear in the system
# turn.  Avoids duplicating the system prompt text in Python source.
_AGENT_SYSTEM_YAML = Path(__file__).parents[1] / "prompt" / "config" / "agents" / "default_agent.yaml"
_AGENT_SYSTEM_MSG: str = yaml.safe_load(_AGENT_SYSTEM_YAML.read_text())["system"]


@nested_dataclass(kw_only=True)
class AgentTaskConfig(GenerationTaskConfig):
    """Agent generation config.

    Inherits all fields from GenerationTaskConfig and adds agent-specific options.
    Key differences from vanilla generation:
    - prompt_format defaults to "openai"         (agents use chat messages)
    - prompt_config falls back to agents/default_agent when no benchmark sets one
    - system_message always carries the agent tool-use instructions so that
      benchmark-specific prompt_configs keep their user-turn format
    - save_trajectory defaults to True           (full conversation saved in JSONL)
    """

    # Agents use the chat-completion (OpenAI) format.  prompt_config is the
    # fallback for benchmarks that do not set their own via GENERATION_ARGS.
    # When a benchmark does set ++prompt_config (e.g. generic/physics), that
    # override takes effect and the benchmark user template is used as-is.
    prompt_format: str = "openai"
    prompt_config: str = "agents/default_agent"

    # The agent system message is always injected on top of whatever prompt_config
    # is active (get_prompt merges system_message into the loaded config dict).
    # This ensures the model receives tool-use instructions even when the
    # benchmark prompt_config only defines a user template.
    system_message: str = _AGENT_SYSTEM_MSG

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
