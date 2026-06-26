#!/bin/bash
# Grade the BF16 5-seed agentic run with the gpt-4o API judge (avg-of-5).
# Per seed: worklogs -> fs_grade build (mode=full) -> bridge to output-rs{k}.jsonl
# (SKIP empty/not-yet-completed problems -> unbiased over completed) -> ONE result
# dir -> rejudge_hle_gpt4o.py --seeds 5 prints pooled pass@1 (= avg-of-5). The
# rejudge caches per-id, so a preliminary run now is reused by the final run.
set -euo pipefail
A=/lustre/fsw/portfolios/nemotron/users/alaptev
G=$A/grading; RD=$G/bf16_rejudge; ER=$RD/eval-results/hle
BENCH=$A/data/hle_full2158.ng.jsonl
mkdir -p "$ER"; cd "$G"
for k in 0 1 2 3 4; do
  W=$A/exp/a0py_bf16_s${k}/worklogs/nhwbf16s${k}
  echo "=== seed $k: build from $W ==="
  python3 fs_grade.py build --benchmark "$BENCH" --worklog-dir "$W" \
    --out "$G/bf16_s${k}_judge_input.jsonl" --mode full 2>&1 | grep -E "wrote|non-empty"
  python3 - "$k" <<PY
import json,sys
k=sys.argv[1]
inp=f"$G/bf16_s{k}_judge_input.jsonl"; out=f"$ER/output-rs{k}.jsonl"
n=0
with open(out,"w") as w:
    for l in open(inp):
        r=json.loads(l)
        if r.get("worker_status") != "completed": continue   # only completed count
        w.write(json.dumps({"id":r["id"],"problem":r["question"],"generation":r["generation"],"expected_answer":r["expected_answer"],"subset_for_metrics":"overall"})+"\n"); n+=1
print(f"  seed {k} graded rows (completed): {n}")
PY
done
echo "=== gpt-4o rejudge (pooled = avg-of-5 over completed) ==="
JUDGE_API_KEY="${JUDGE_API_KEY:?set JUDGE_API_KEY (NVIDIA inference API judge key)}" \
  python3 rejudge_hle_gpt4o.py --result-dir "$RD" --seeds 5 --tag gpt4o --concurrency 48
