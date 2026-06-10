# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agentic-loop MCP tools for the self-improving Hermes swarm (SCI-548).

Tools in this package support the orchestrator + worker loop:

- ``worklog_tool`` — clock-in / clock-off work tracking that writes a
  markdown report per task and *guarantees* a close even on crash/kill.
- ``benchmark_tool`` — the answer-hiding trust boundary: ``load_benchmark``
  (ids-only manifest, orchestrator side) + ``get_problem`` (one problem's
  ``{id, prompt, modality}``, worker side; never the answer). Powers
  ``batch_solve`` (Hermes-native ``delegate_task``) over a Gym benchmark.
"""
