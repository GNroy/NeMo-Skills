#!/usr/bin/env python3
"""FrontierScience judge-join grader for the agentic batch_solve path.

The benchmark trust-boundary tool keeps `expected_answer` server-side; workers
emit a free-form 'Final Answer: <x>' in their worklog report. This grader joins
the two by the SAME id derivation the benchmark tool uses (`derive_problem_id`:
top-level id_keys, else `row-<index>` — FrontierScience rows carry id+answer
nested in `verifier_metadata`, so they join as `row-0`..`row-N`), then scores
with the existing FrontierScience-Olympiad LLM judge.

Two steps (kept separate so the judge endpoint is only needed for step 2):

  build  --benchmark FS.jsonl --worklog-dir DIR --out judge_input.jsonl
         Join expected (from verifier_metadata) + predicted (from worklog
         'Final Answer:' line) into rows {id, question, expected_answer,
         generation, worker_status}. Missing/errored worker -> empty generation
         (will be judged NO). Fully deterministic; no model needed.

  Then grade with the stock NS judge (gpt-oss-120b), e.g.:
    ns generate --cluster aws-cmh --server_type vllm --model /hf_models/gpt-oss-120b \
      --server_gpus 4 --input_file judge_input.jsonl --output_dir JUDGE \
      ++prompt_config=judge/frontierscience-olympiad ++generation_key=judgement \
      ++add_generation_stats=False

  tally  --judged JUDGE/output.jsonl
         Parse 'Judgement: YES/NO' (last line) -> pass@1 + per-id table.

stdlib only.
"""
import argparse
import json
import os
import re
import sys

DEFAULT_ID_KEYS = ("id", "uuid", "hash_id", "_id", "problem_id", "qid")


def derive_problem_id(row, index, id_keys=DEFAULT_ID_KEYS):
    """Mirror benchmark_tool.derive_problem_id EXACTLY (top-level keys only)."""
    for key in id_keys:
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return f"row-{index}"


def extract_question(row):
    """Prompt the worker actually saw: user turns of responses_create_params.input,
    falling back to verifier_metadata.question."""
    rcp = row.get("responses_create_params") or {}
    inp = rcp.get("input") or []
    users = [m.get("content", "") for m in inp if isinstance(m, dict) and m.get("role") == "user"]
    if users:
        return "\n\n".join(users)
    vm = row.get("verifier_metadata") or {}
    return vm.get("question", "")


def extract_expected(row):
    vm = row.get("verifier_metadata") or {}
    return vm.get("expected_answer", row.get("expected_answer", ""))


# Stub/placeholder values that are NOT real answers (eager clock_in stub +
# shutdown_sweep stub write '## Final answer\n(not reported)' / '(unknown)').
_PLACEHOLDERS = {"(not reported)", "not reported", "(unknown)", "unknown",
                 "(none)", "none", "n/a", "na", "tbd", "-", ""}


def _clean_answer(s):
    """Strip markdown / LaTeX wrappers from an extracted answer."""
    s = s.strip()
    # \boxed{...} -> inner
    m = re.search(r"\\boxed\{(.+?)\}\s*\$*\s*$", s)
    if m:
        s = m.group(1).strip()
    # surrounding $$ ... $$ or $ ... $ or \[ ... \]
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"^\\\[|\\\]$", "", s).strip()
    # surrounding ** bold ** / backticks
    s = s.strip("*").strip("`").strip()
    return s


def extract_final_answer(text):
    """Extract the worker's final answer, robust to non-uniform formats:
      * colon form: 'Final Answer: <x>' (possibly with ** / # decoration)
      * header form: '## Final Answer' / '### Final Answer' / '**Final Answer**'
        with the answer on the FOLLOWING non-empty line(s) (often $$\\boxed{}$$).
    Returns the answer for the LAST 'final answer' occurrence (workers often
    restate it at the end)."""
    lines = text.splitlines()
    ans = ""
    for i, line in enumerate(lines):
        low = line.lower()
        idx = low.find("final answer")
        if idx == -1:
            continue
        after = line[idx + len("final answer"):]
        same = after.lstrip(" \t").lstrip(":").strip().strip("*").strip()
        if same:  # colon / inline form
            cand = _clean_answer(same)
            if cand.lower() not in _PLACEHOLDERS:
                ans = cand
            continue
        # header form -> collect following non-empty, non-header lines
        collected = []
        for j in range(i + 1, min(i + 6, len(lines))):
            l = lines[j].strip()
            if not l:
                if collected:
                    break
                continue
            if l.startswith("#"):
                break
            collected.append(l)
            if "\\boxed" in l or l.startswith("$") or len(collected) >= 2:
                break
        if collected:
            cand = _clean_answer(" ".join(collected))
            if cand.lower() not in _PLACEHOLDERS:
                ans = cand
    return ans


def strip_frontmatter(text):
    """Drop the leading YAML frontmatter block (---\\n...\\n---\\n) so only the
    worker's solution report body remains."""
    m = re.match(r"^---\n.*?\n---\n?", text, re.DOTALL)
    return text[m.end():].lstrip("\n") if m else text


def parse_worklog(path):
    """Return (final_answer, status, body). status from frontmatter; final_answer
    via the format-robust extractor; body = the full report (frontmatter stripped),
    which is the worker's complete solution = the 'generation' to judge (mirrors how
    the FrontierScience LLM judge reads a single model's full output)."""
    if not os.path.exists(path):
        return "", "missing", ""
    text = open(path, encoding="utf-8", errors="replace").read()
    status = ""
    m = re.search(r"^status:\s*(\S+)", text, re.MULTILINE)
    if m:
        status = m.group(1)
    body = strip_frontmatter(text)
    return extract_final_answer(text), (status or "unknown"), body


def cmd_build(args):
    rows = [json.loads(l) for l in open(args.benchmark) if l.strip()]
    out = []
    for i, row in enumerate(rows):
        pid = derive_problem_id(row, i)
        ans, status, body = parse_worklog(os.path.join(args.worklog_dir, f"{pid}.md"))
        # mode=full (default): feed the judge the worker's COMPLETE solution body
        # (frontmatter stripped) — identical to how the FrontierScience LLM judge
        # reads a single model's full generation. This avoids undercounting the
        # ~20% of workers that state the answer in prose without a literal
        # 'Final Answer:' line (extraction-gap, not a wrong/absent answer).
        # mode=answer: legacy — feed only the regex-extracted final-answer string.
        gen = body if args.mode == "full" else ans
        out.append({
            "id": pid,
            "question": extract_question(row),
            "expected_answer": extract_expected(row),
            "generation": gen,
            "extracted_answer": ans,
            "worker_status": status,
        })
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    nonempty = sum(1 for r in out if r["generation"].strip())
    has_ans = sum(1 for r in out if r["extracted_answer"])
    print(f"wrote {args.out} (mode={args.mode}): {len(out)} rows, "
          f"{nonempty} non-empty generation, {has_ans} with an extractable 'Final Answer:' line")
    for r in out:
        disp = r["extracted_answer"] or (r["generation"][:50].replace(chr(10), " ") if r["generation"].strip() else "<empty>")
        print(f"  {r['id']:>7}  status={r['worker_status']:<14} ans={disp}")


_JUDGE_RE = re.compile(r"judgement\s*:\s*(yes|no)", re.IGNORECASE)


def cmd_tally(args):
    rows = [json.loads(l) for l in open(args.judged) if l.strip()]
    yes = 0
    table = []
    for r in rows:
        j = r.get("judgement") or r.get("generation") or ""
        verdict = None
        for line in reversed(str(j).splitlines()):
            m = _JUDGE_RE.search(line)
            if m:
                verdict = m.group(1).lower()
                break
        ok = verdict == "yes"
        yes += int(ok)
        table.append((r.get("id", "?"), verdict or "??", ok))
    n = len(rows)
    print(f"pass@1 = {yes}/{n} = {100.0*yes/max(n,1):.1f}%")
    for pid, v, ok in table:
        print(f"  {pid:>7}  {'YES' if ok else 'no ':<3}  ({v})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--benchmark", required=True)
    b.add_argument("--worklog-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--mode", choices=["full", "answer"], default="full",
                   help="full (default): judge the whole worker solution body; "
                        "answer: judge only the extracted 'Final Answer:' string.")
    b.set_defaults(func=cmd_build)
    t = sub.add_parser("tally")
    t.add_argument("--judged", required=True)
    t.set_defaults(func=cmd_tally)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
