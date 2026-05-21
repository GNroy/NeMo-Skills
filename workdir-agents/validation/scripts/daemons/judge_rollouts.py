#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone judge for NeMo-Gym rollouts JSONL.

Re-implements `FrontierScienceJudgeServer.verify`'s scoring path (strip
thinking traces -> format the judge prompt -> POST to
``/v1/chat/completions`` -> parse ``Judgement: YES/NO``) so we can grade
a rollouts file produced by the passthrough verifier in a SEPARATE
phase, against a separate vLLM daemon running the judge model.

This split lets us hold Kimi GPUs and judge GPUs disjoint in time —
the cluster has tight GPU-utilisation rules and we don't want both
daemons co-resident for the full smoke duration.

Input rollouts JSONL (output of ng_collect_rollouts under passthrough)
must contain at minimum::

    {
      "response": {"output": [...], "output_text": "..."},
      "verifier_metadata": {"question": "...", "expected_answer": "..."}
    }

Output JSONL (per-row, in the same order):

    {
      ...original row...,
      "judge_reward": 1.0 | 0.0,
      "judge_verdict": "YES" | "NO" | null,
      "judge_output": "<full judge response>"
    }
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)
_JUDGEMENT_RE = re.compile(r"Judgement:\s*(YES|NO)", re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _THINKING_RE.sub("", text)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_assistant_text(response: Any) -> str:
    if not isinstance(response, dict):
        return ""
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return _strip_thinking(direct)
    last = ""
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        for chunk in item.get("content") or []:
            text = chunk.get("text") if isinstance(chunk, dict) else None
            if isinstance(text, str) and text.strip():
                last = text
    return _strip_thinking(last)


def _parse_verdict(judge_text: str) -> Optional[str]:
    if not judge_text:
        return None
    matches = list(_JUDGEMENT_RE.finditer(judge_text))
    if not matches:
        return None
    return matches[-1].group(1).upper()


def _post_chat_completions(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
    ).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"]


# Inline copy of NeMo-Gym's frontierscience_judge prompt.  The YAML on
# disk has a single ``user: |-`` block — the simplest robust thing is to
# carry the text here so we don't pull pyyaml into the judge container.
# Source: NeMo-Gym/resources_servers/frontierscience_judge/prompts/judge.yaml
_INLINE_PROMPT_TEMPLATE = (
    "You are grading an attempted answer to a science olympiad problem. You will be given the\n"
    "problem, attempted answer, and reference answer. Evaluate the solution against the provided\n"
    "reference solution, ensuring it is complete and matches the reference solution. Pay close\n"
    "attention to detail and grade it strictly, but fairly.\n"
    "The reference answer is either a single number or expression in latex formatting, a chemical\n"
    "formula, a compound name, or a phrase referring to a specific name, entity, or method.\n"
    "Mark the attempted answer as correct if it fully matches the reference answer or is otherwise\n"
    "equivalent (e.g., an equivalent algebraic expression, a numerical number within 1 decimal\n"
    "place rounding of the reference answer (e.g., 6.69 ≈ 6.7), an equivalent name for a\n"
    "compound/formula, equivalent when accounting for units, etc.). Mark it as incorrect if it is\n"
    "not equivalent to the reference answer.\n"
    "***\n"
    "The problem: {question}\n"
    "***\n"
    "The reference answer: {expected_answer}\n"
    "***\n"
    "The attempted answer: {generation}\n"
    "***\n"
    "First, think step-by-step about whether the attempted answer matches the reference answer.\n"
    "If the attempted answer is correct, write \"Judgement: YES\" in the last line of your\n"
    "response, with no other text or formatting. If it is incorrect, write \"Judgement: NO\".\n"
)


def _load_prompt_template(prompt_yaml: Optional[Path]) -> str:
    """Load the judge prompt template.

    If ``prompt_yaml`` is provided and readable, parse it (single ``user``
    key); otherwise return the inlined copy.  Reading the YAML lets you
    point judge_rollouts.py at a customised prompt without re-editing this
    file, but isn't required for the default path.
    """
    if prompt_yaml is None:
        return _INLINE_PROMPT_TEMPLATE
    try:
        text = prompt_yaml.read_text()
    except OSError:
        return _INLINE_PROMPT_TEMPLATE
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(text)
        return parsed["user"]
    except Exception:  # noqa: BLE001 — best-effort YAML; fall back to inline
        return _INLINE_PROMPT_TEMPLATE


def _extract_question(entry: dict) -> str:
    meta = entry.get("verifier_metadata") or {}
    q = meta.get("question")
    if isinstance(q, str) and q.strip():
        return q
    rcp = entry.get("responses_create_params") or {}
    for msg in reversed(rcp.get("input") or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
    return ""


def judge_file(
    input_path: Path,
    output_path: Path,
    base_url: str,
    api_key: str,
    model: str,
    prompt_template: str,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_yes = 0
    n_no = 0
    n_unparsed = 0
    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            n_total += 1
            question = _extract_question(entry)
            expected = (entry.get("verifier_metadata") or {}).get("expected_answer", "")
            generation = _extract_assistant_text(entry.get("response"))
            prompt = prompt_template.format(
                question=question, expected_answer=expected, generation=generation
            )
            judge_text = ""
            last_err = None
            for attempt in range(retries):
                try:
                    judge_text = _post_chat_completions(
                        base_url, api_key, model, prompt, max_tokens, timeout
                    )
                    break
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                    last_err = exc
                    time.sleep(2 ** attempt)
            else:
                print(f"[judge] giving up on row {n_total - 1}: {last_err}", file=sys.stderr)
            verdict = _parse_verdict(judge_text)
            if verdict == "YES":
                reward = 1.0
                n_yes += 1
            elif verdict == "NO":
                reward = 0.0
                n_no += 1
            else:
                reward = 0.0
                n_unparsed += 1
            entry["judge_reward"] = reward
            entry["judge_verdict"] = verdict
            entry["judge_output"] = judge_text
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {
        "total": n_total,
        "yes": n_yes,
        "no": n_no,
        "unparsed": n_unparsed,
        "pass_rate": (n_yes / n_total) if n_total else 0.0,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="judge_rollouts", description=__doc__)
    p.add_argument("input", type=Path, help="rollouts.jsonl (passthrough output)")
    p.add_argument("output", type=Path, help="where to write judged.jsonl")
    p.add_argument(
        "--judge-url",
        required=True,
        help="judge model base URL, e.g. http://hostname:port/v1",
    )
    p.add_argument("--judge-model", required=True, help="served-model-name on the judge daemon")
    p.add_argument("--judge-api-key", default="dummy")
    p.add_argument(
        "--prompt-yaml",
        type=Path,
        default=None,
        help=(
            "Optional path to a YAML file with a `user:` key holding the "
            "judge prompt.  Defaults to an inlined copy of Gym's "
            "frontierscience_judge prompt."
        ),
    )
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--retries", type=int, default=3)
    args = p.parse_args(argv)

    template = _load_prompt_template(args.prompt_yaml)
    stats = judge_file(
        args.input,
        args.output,
        base_url=args.judge_url,
        api_key=args.judge_api_key,
        model=args.judge_model,
        prompt_template=template,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
