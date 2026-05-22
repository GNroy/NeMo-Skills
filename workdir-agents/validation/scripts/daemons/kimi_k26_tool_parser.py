# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool-call parser for Kimi-K2.6 as served by vLLM 0.19.x.

Background:
    vLLM ships a stock `kimi_k2` tool-call parser whose regex expects
    the Kimi chat template's INPUT rendering format:

        <|tool_call_begin|>FUNC:N<|tool_call_argument_begin|>ARGS<|tool_call_end|>

    Direct probes (`curl /v1/chat/completions` on a Kimi-K2.6 daemon)
    show the model actually emits a different ORDER:

        <|tool_call_begin|><|tool_call_end|>FUNC:N{ARGS}<|tool_call_argument_begin|>

    With the stock regex, `tool_calls` comes back empty for every
    tool-using rollout, breaking any agentic experiment.  This parser
    matches the observed emit format and synthesises proper tool calls.

Usage:
    Pass to vllm serve as a plugin + select by name:

        vllm serve ... \\
            --tool-parser-plugin /path/to/kimi_k26_tool_parser.py \\
            --tool-call-parser kimi_k26 \\
            --enable-auto-tool-choice

    The plugin file is registered into vLLM's `ToolParserManager` via the
    decorator at import time, mirroring how
    `nano_v3_reasoning_parser.py` plugs into `ReasoningParserManager`.

Notes:
    * Streaming is intentionally not supported — set
      ``HermesAgentConfig.disable_streaming=True`` (or equivalent) so the
      client uses the non-streaming code path.
    * If Moonshot/vLLM ship a stock parser that handles K2.6 in a future
      release, prefer that over this shim.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence, Union

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)


_TOOL_CALL_REGEX = re.compile(
    r"<\|tool_call_begin\|>\s*<\|tool_call_end\|>\s*"
    r"(?P<tool_call_id>[\w\.]+:\d+)\s*"
    r"(?P<function_arguments>\{.*?\})\s*"
    r"<\|tool_call_argument_begin\|>",
    re.DOTALL,
)


@ToolParserManager.register_module("kimi_k26")
class KimiK26ToolParser(ToolParser):
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if "<|tool_call_begin|>" not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        matches = _TOOL_CALL_REGEX.findall(model_output)
        if not matches:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls = []
        for tcid, raw_args in matches:
            # `tcid` looks like ``functions.calculator:0`` — the segment
            # after the last dot and before the colon is the registered
            # function name.  Fall back to the whole id if the format
            # deviates so we never raise on slightly off rows.
            try:
                name = tcid.split(":", 1)[0].rsplit(".", 1)[-1]
            except Exception:  # noqa: BLE001
                name = tcid
            tool_calls.append(
                ToolCall(
                    id=tcid,
                    type="function",
                    function=FunctionCall(name=name, arguments=raw_args),
                )
            )

        cleaned = _TOOL_CALL_REGEX.sub("", model_output).strip()
        # Strip lingering bare markers the regex couldn't match (e.g. an
        # orphan section_begin/end if the model emits them outside the
        # observed pattern).  Keeps content readable for the agent.
        for stray in (
            "<|tool_calls_section_begin|>",
            "<|tool_calls_section_end|>",
            "<|tool_call_begin|>",
            "<|tool_call_end|>",
            "<|tool_call_argument_begin|>",
        ):
            cleaned = cleaned.replace(stray, "")
        cleaned = cleaned.strip()

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=cleaned if cleaned else None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None, type(None)]:
        # Streaming intentionally not supported — Hermes/Gym set
        # disable_streaming=True at the agent layer.
        return None
