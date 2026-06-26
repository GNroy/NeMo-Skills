#!/usr/bin/env python3
# Final BF16 5-seed tally from the gpt-4o rejudge cache + worker_status.
# Reports BOTH: full-2158 pass@1 (incomplete=wrong, standard HLE metric) and
# completed-only pass@1, each per-seed + mean/std/95%CI. No API calls.
import json, re, math
G="/lustre/fsw/portfolios/nemotron/users/alaptev/grading"
CACHE=G+"/bf16_rejudge/eval-results/hle/rejudge_gpt4o"
TOTAL=2158
JRE=re.compile(r"judg(?:e?ment)?\**\s*:\s*\**\s*(yes|no)", re.IGNORECASE)
def yes(j):
    m=JRE.search(j or ""); return bool(m and m.group(1).lower()=="yes")
def stats(xs):
    m=sum(xs)/len(xs); sd=(sum((x-m)**2 for x in xs)/(len(xs)-1))**0.5 if len(xs)>1 else 0
    return m, sd, sd/math.sqrt(len(xs))
full=[]; comp=[]; rows=[]
for k in range(5):
    status={json.loads(l)["id"]:json.loads(l).get("worker_status") for l in open(f"{G}/bf16_s{k}_judge_input.jsonl")}
    jc={json.loads(l)["id"]:json.loads(l).get("judgement_new") for l in open(f"{CACHE}/output-rs{k}.jsonl")}
    done=[i for i in status if status[i]=="completed"]
    y=sum(1 for i in done if yes(jc.get(i)))
    nd=len(done)
    full.append(100*y/TOTAL); comp.append(100*y/nd if nd else 0)
    rows.append((k,nd,y))
print(f"{seed:>4} {completed:>10} {correct:>8} {full-2158:>10} {compl-only:>11}")
for k,nd,y in rows:
    print(f"{k:>4} {nd:>10} {y:>8} {100*y/TOTAL:>9.2f}% {100*y/nd:>10.2f}%")
fm,fsd,fse=stats(full); cm,csd,cse=stats(comp)
print(f"\nFULL-2158 (incomplete=wrong): mean={fm:.2f}%  std={fsd:.2f}  95%CI +/- {1.96*fse:.2f}")
print(f"COMPLETED-ONLY              : mean={cm:.2f}%  std={csd:.2f}  95%CI +/- {1.96*cse:.2f}")
