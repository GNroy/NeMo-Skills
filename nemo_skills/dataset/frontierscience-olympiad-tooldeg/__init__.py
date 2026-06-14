# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# TEMPORARY iteration benchmark: FrontierScience-Olympiad problems that degraded with the
# python tool (python_correct < notool_correct over 5 seeds). 21 problems. Used to iterate
# on fixing the python-tool accuracy regression (SCI-575). Identical config to the parent
# `frontierscience-olympiad` benchmark; only the split differs (test = the subset).
METRICS_TYPE = "frontierscience-olympiad"
GENERATION_ARGS = "++prompt_config=generic/default ++eval_type=math"
EVAL_SPLIT = "test"

JUDGE_PIPELINE_ARGS = {
    "model": "o3-mini-2025-01-31",
    "server_type": "openai",
    "server_address": "https://api.openai.com/v1",
}
JUDGE_ARGS = "++prompt_config=judge/frontierscience-olympiad ++generation_key=judgement ++add_generation_stats=False"
