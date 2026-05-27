# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser shim for Kimi-K2.6 on vLLM 0.19.x.

The stock ``kimi_k2`` reasoning parser
(`vllm/reasoning/kimi_k2_reasoning_parser.py`) assumes the model emits
exactly one ``<think>`` block at the **start** of the output:

    extract_reasoning():
        start = model_output.find("<think>")
        # If start != 0, force start = 0  (silently lose the prefix text)
        end = model_output.find("</think>")
        return (model_output[start:end], model_output[end+8:])

For K2.6 with the Hermes agent's large system prompt + tool defs, the
model sometimes emits ``<think>`` blocks **mid-stream** (interleaved
with visible prose) or **multiple** blocks in one response.  The stock
parser's single-block / start-anchored logic then misses the tags
entirely or mis-extracts them, leaving the whole response in
``content`` with raw ``<think>…</think>`` tags inside.  Hermes' own
``_strip_think_blocks`` then reduces ``content`` to empty, trips the
"thinking-only response → prefill" + "empty response → retry" cascade,
and the agent burns ~5 retries per turn before bailing with
``content="(empty)"``.

This shim does an XML-style multi-block extraction:

    re.findall(r'<think>(.*?)</think>', model_output, re.DOTALL)

then returns

    reasoning  = "\n\n".join(inner_blocks)
    content    = model_output with all <think>…</think> blocks removed

…and also strips the K2.6 tool-section start token (the marker our
custom tool parser keys off) into the content side so the tool path
still works.

Wiring (vllm serve)::

    --reasoning-parser-plugin /alaptev/reasoning_parsers/kimi_k26_reasoning_parser.py
    --reasoning-parser kimi_k26

Streaming is best-effort: we delegate to the stock implementation since
Hermes/Gym sets ``disable_streaming=True`` and the streaming path is
never exercised in our pipeline.  If/when Moonshot or vLLM ship a
proper K2.6-aware reasoning parser upstream, prefer that over this
shim.
"""

from __future__ import annotations

import regex as re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerBase

from vllm.reasoning.abs_reasoning_parsers import ReasoningParser, ReasoningParserManager

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_TOOL_SECTION_START_TOKEN = "<|tool_calls_section_begin|>"


@ReasoningParserManager.register_module(["kimi_k26"])
class KimiK26ReasoningParser(ReasoningParser):
    """Regex-based multi-block ``<think>`` extractor for Kimi-K2.6."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        # Resolve special-token ids when present for ``is_reasoning_end``
        # heuristics; tolerate absence (K2.6 sometimes ships a tokenizer
        # without these as named tokens).
        vocab = getattr(self, "vocab", None) or {}
        self._end_token = "</think>"
        self._tool_section_start_token = _TOOL_SECTION_START_TOKEN
        self._end_token_id = vocab.get(self._end_token)
        self._tool_section_start_token_id = vocab.get(self._tool_section_start_token)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self._end_token_id is not None and self._end_token_id in input_ids:
            return True
        if (
            self._tool_section_start_token_id is not None
            and self._tool_section_start_token_id in input_ids
        ):
            return True
        return False

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        """Pull every ``<think>…</think>`` block into reasoning, leave the rest as content."""
        if not model_output:
            return (None, model_output or None)

        blocks: list[str] = []

        def _collect(match: "re.Match[str]") -> str:
            inner = match.group(1)
            if inner is not None and inner.strip():
                blocks.append(inner)
            return ""

        stripped = _THINK_BLOCK_RE.sub(_collect, model_output)

        # K2.6 tool-call section marker — when present, treat everything
        # from it on as content (the tool parser will consume it).  This
        # mirrors the stock parser's tool-section short-circuit.
        if not blocks:
            tool_idx = stripped.find(self._tool_section_start_token)
            if tool_idx != -1:
                return (
                    stripped[:tool_idx] or None,
                    stripped[tool_idx:] or None,
                )
            return (None, stripped or None)

        reasoning = "\n\n".join(blocks)
        content = stripped.strip("\n") or None
        return (reasoning, content)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> "DeltaMessage | None":
        # We don't stream; Hermes/Gym sets ``disable_streaming=True`` so this
        # path is never exercised.  Returning ``None`` is safe — the
        # streaming consumer treats a None delta as "no output this chunk".
        return None
