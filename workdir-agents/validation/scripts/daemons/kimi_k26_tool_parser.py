# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool-call parser shim for Kimi-K2.6 on vLLM 0.19.x.

The stock ``kimi_k2`` parser's regex expects the chat template's INPUT
rendering of a tool call:

    <|tool_call_begin|>FUNC:N<|tool_call_argument_begin|>ARGS<|tool_call_end|>

Direct curl probes of a Kimi-K2.6 daemon (vllm-glm51-cu130-ray,
vllm 0.19.1.dev1) show the model actually EMITS:

    <|tool_call_begin|><|tool_call_end|>FUNC:N{ARGS}<|tool_call_argument_begin|>

The order/grouping is different, so the stock regex never matches and
``tool_calls`` is silently returned empty.  This plugin overrides the
regex + the start-of-tools sentinel; the inherited ``extract_tool_calls``
otherwise does the right thing.

Wiring (vllm serve):

    --tool-parser-plugin /alaptev/reasoning_parsers/kimi_k26_tool_parser.py
    --tool-call-parser kimi_k26
    --enable-auto-tool-choice

Streaming is intentionally a no-op (Hermes/Gym sets
``disable_streaming=True`` on the agent side); revisit if we ever
re-enable streaming.

If/when Moonshot or vLLM ship a stock parser that handles K2.6's emit
format upstream, prefer that over this shim.
"""

from __future__ import annotations

import regex as re

from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParserManager
from vllm.tool_parsers.kimi_k2_tool_parser import KimiK2ToolParser


@ToolParserManager.register_module(["kimi_k26"])
class KimiK26ToolParser(KimiK2ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        # K2.6 never emits the <|tool_calls_section_begin|> sentinel the
        # stock parser keys off; switch to the per-call begin marker.
        # ``content`` becomes everything before the first tool-call begin,
        # which is what we want (visible prelude / reasoning chatter).
        self.tool_calls_start_token = "<|tool_call_begin|>"
        # Match the observed emit token order.
        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*<\|tool_call_end\|>\s*"
            r"(?P<tool_call_id>[\w\.]+:\d+)\s*"
            r"(?P<function_arguments>\{.*?\})\s*"
            r"<\|tool_call_argument_begin\|>",
            re.DOTALL,
        )

    def extract_tool_calls_streaming(self, *args, **kwargs):
        # We don't stream; Hermes/Gym sets disable_streaming=True so this
        # path is never exercised.  Return None to be defensive.
        return None
