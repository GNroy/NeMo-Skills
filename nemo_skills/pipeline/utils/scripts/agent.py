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

"""Script class for agent worker processes in a multi-agent SLURM het-job.

AgentWorkerScript launches `nemo_skills.inference.agent_server` inside a
het-group alongside its own LLM server.  The orchestrator (het-group 0)
communicates with workers via HTTP; worker addresses are resolved lazily
using SLURM_MASTER_NODE_HET_GROUP_N env vars.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from nemo_skills.pipeline.utils.scripts.base import BaseJobScript
from nemo_skills.pipeline.utils.scripts.server import ServerScript
from nemo_skills.pipeline.utils.server import get_free_port
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


@dataclass(kw_only=True)
class AgentWorkerScript(BaseJobScript):
    """Script that starts an agent worker HTTP server inside a SLURM het-group.

    The worker server (nemo_skills.inference.agent_server) binds to
    0.0.0.0:{agent_port} and waits for task requests from orchestrators or
    peer agents.

    Attributes:
        server_script:        ServerScript for the LLM in the same het-group.
                              Provides a lazily-resolved hostname reference.
        server_address_prehosted:
                              Use this instead of server_script when pointing
                              at a pre-hosted (externally running) LLM.
        extra_arguments:      Additional Hydra++ arguments forwarded verbatim
                              to nemo_skills.inference.agent_server.
        agent_port:           Port for the worker HTTP server (default: auto).
        allocate_port:        Allocate agent_port automatically if True.
        agent_name:           Logical name for this worker (used in logs).
        log_prefix:           Prefix for SLURM log files.
    """

    server_script: Optional[ServerScript] = None
    server_address_prehosted: Optional[str] = None
    extra_arguments: str = ""
    agent_port: Optional[int] = None
    allocate_port: bool = True
    agent_name: str = "agent_worker"

    log_prefix: str = field(default="agent_worker", init=False)

    def __post_init__(self):
        if self.agent_port is None and self.allocate_port:
            self.agent_port = get_free_port(strategy="random")
            LOG.debug("Allocated port %d for agent worker '%s'", self.agent_port, self.agent_name)
        if self.agent_port is None:
            raise ValueError("AgentWorkerScript requires agent_port when allocate_port=False")

        def build_cmd() -> Tuple[str, Dict]:
            # Resolve LLM server address lazily (at job startup, after nemo-run
            # exports SLURM_MASTER_NODE_HET_GROUP_N env vars).
            if self.server_script is not None:
                server_host = self.server_script.hostname_ref()
                server_port = self.server_script.port
                server_args = (
                    f"++server.host={server_host} "
                    f"++server.port={server_port} "
                    f"++server.server_type={self.server_script.server_type} "
                    f"++server.model={self.server_script.model_path} "
                )
            elif self.server_address_prehosted is not None:
                host, port = self.server_address_prehosted.rsplit(":", 1)
                server_args = f"++server.base_url=http://{self.server_address_prehosted}/v1 "
            else:
                server_args = ""

            cmd = (
                f"python -m nemo_skills.inference.agent_server "
                f"++agent_port={self.agent_port} "
                f"++agent_name={self.agent_name} "
                f"{server_args}"
                f"{self.extra_arguments}"
            )
            return cmd, {}

        self.set_inline(build_cmd)
        super().__post_init__()
