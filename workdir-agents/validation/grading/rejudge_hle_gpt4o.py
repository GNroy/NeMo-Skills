#!/usr/bin/env python
"""Re-judge existing HLE generations with a different judge (e.g. gpt-4o via inference-api.nvidia.com),
WITHOUT regenerating. Tests the judge confound in SCI-575 (python helps under Jiacheng's gpt-4o judge
vs hurts under our gpt-oss-120b dev judge).

Uses the EXACT NeMo-Skills HLE judge prompt (nemo_skills/prompt/config/judge/hle.yaml) so the only
changed variable is the judge model. Reads output-rs{N}.jsonl (fields: problem, generation,
expected_answer, subset_for_metrics, judgement[=old gpt-oss]). Writes <dir>/rejudge_<tag>/output-rs{N}.jsonl
with the new judgement cached, and prints judge_correct pass@1[avg-of-seeds] overall + by domain, for
BOTH the new judge and the cached old judge (apples-to-apples).

Key via env JUDGE_API_KEY. Zero non-stdlib deps (urllib + ThreadPoolExecutor).
"""
import argparse, glob, json, os, re, sys, time, urllib.request, urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {problem}

[response]: {generation}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {expected_answer}

Reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

Judgement: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

Confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""

JUDGE_RE = re.compile(r"judg(?:e?ment)?\**\s*:\s*\**\s*(yes|no)", re.IGNORECASE)
NOANS_RE = re.compile(r"extracted_final_answer\**\s*:\s*\**\s*['\"]?none['\"]?", re.IGNORECASE)


def is_correct(judgement):
    if not judgement:
        return None
    m = JUDGE_RE.search(judgement)
    return None if not m else (m.group(1).lower() == "yes")


def is_noans(judgement):
    return bool(judgement and NOANS_RE.search(judgement))


def call_judge(endpoint, model, key, prompt, temperature, retries=4):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 1024,
    }).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == retries - 1:
                return f"__ERROR__ {e}"
            time.sleep(2 * (attempt + 1))


def agg(records):
    """records: list of dicts with subset + correct(bool/None). Returns overall + per-domain accuracy."""
    by = defaultdict(lambda: [0, 0, 0])  # domain -> [correct, total, noans]
    tot = [0, 0, 0]
    for r in records:
        c = r["correct"]
        dom = r["subset"] or "Unknown"
        by[dom][1] += 1; tot[1] += 1
        if c:
            by[dom][0] += 1; tot[0] += 1
        if r["noans"]:
            by[dom][2] += 1; tot[2] += 1
    return tot, by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True, help="dir containing eval-results/hle/output-rs*.jsonl")
    ap.add_argument("--endpoint", default="https://inference-api.nvidia.com/v1")
    ap.add_argument("--model", default="azure/openai/gpt-4o")
    ap.add_argument("--tag", default="gpt4o")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-records-per-seed", type=int, default=0, help="0 = all (subset for quick test)")
    args = ap.parse_args()

    key = os.environ.get("JUDGE_API_KEY")
    assert key, "set JUDGE_API_KEY"
    base = os.path.join(args.result_dir, "eval-results", "hle")
    outdir = os.path.join(base, f"rejudge_{args.tag}")
    os.makedirs(outdir, exist_ok=True)

    new_records, old_records = [], []
    for seed in range(args.seeds):
        src = os.path.join(base, f"output-rs{seed}.jsonl")
        if not os.path.exists(src):
            continue
        rows = [json.loads(l) for l in open(src) if l.strip()]
        if args.max_records_per_seed:
            rows = rows[: args.max_records_per_seed]

        cache = os.path.join(outdir, f"output-rs{seed}.jsonl")
        cached = {}
        if os.path.exists(cache):
            for l in open(cache):
                if l.strip():
                    d = json.loads(l); cached[d["id"]] = d.get("judgement_new")

        todo = [r for r in rows if r.get("id") not in cached]
        print(f"seed {seed}: {len(rows)} rows, {len(cached)} cached, {len(todo)} to judge", flush=True)

        def work(r):
            p = JUDGE_PROMPT.format(problem=r["problem"], generation=r["generation"],
                                    expected_answer=r["expected_answer"])
            return r["id"], call_judge(args.endpoint, args.model, key, p, args.temperature)

        if todo:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                for i, (rid, jj) in enumerate(ex.map(work, todo)):
                    cached[rid] = jj
                    if i % 100 == 0:
                        print(f"  seed {seed}: {i}/{len(todo)}", flush=True)
            with open(cache, "w") as f:
                for r in rows:
                    f.write(json.dumps({"id": r["id"], "judgement_new": cached.get(r["id"])}) + "\n")

        for r in rows:
            jn = cached.get(r["id"])
            new_records.append({"subset": r.get("subset_for_metrics"),
                                "correct": is_correct(jn), "noans": is_noans(jn)})
            old_records.append({"subset": r.get("subset_for_metrics"),
                                "correct": is_correct(r.get("judgement")), "noans": is_noans(r.get("judgement"))})

    def report(name, recs):
        tot, by = agg(recs)
        c, t, na = tot
        print(f"\n=== {name}  (n={t}) ===")
        print(f"  OVERALL judge_correct: {100*c/t:.2f}%   no-answer: {100*na/t:.2f}%")
        for dom in sorted(by):
            dc, dt, dna = by[dom]
            print(f"    {dom:30s} {100*dc/dt:6.2f}%  (n={dt}, noans {100*dna/dt:.1f}%)")

    print(f"\n#### {args.result_dir}")
    report(f"NEW judge = {args.model}", new_records)
    report("OLD judge = gpt-oss-120b (cached)", old_records)


if __name__ == "__main__":
    main()
