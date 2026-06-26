#!/usr/bin/env python3
"""L1 live validation: hammer the standalone router with concurrent chat
completions and confirm the dynamically-registered worker pool serves them.
Proves register->serve works end-to-end without any gym/launcher involvement."""
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

MODEL = os.environ.get("MODEL", "/hf_models/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16

# Runs inside the sglang container (pool dir mounted at /alaptev) OR on the host
# (lustre path). Try the in-container path first, then the host path.
_CANDIDATES = [
    os.environ.get("POOL_DIR_HOST", ""),
    "/alaptev/data/lazypool",
    "/lustre/fsw/portfolios/nemotron/users/alaptev/data/lazypool",
]
base = None
for POOL in _CANDIDATES:
    if POOL and os.path.exists(os.path.join(POOL, "router_url")):
        with open(os.path.join(POOL, "router_url")) as f:
            base = f.read().strip()
        break
if base is None:
    raise SystemExit(f"router_url not found under any of {_CANDIDATES}")
print(f"router={base} model={MODEL} concurrency={N}", flush=True)


def workers():
    try:
        with urllib.request.urlopen(f"{base}/workers", timeout=10) as r:
            return r.read().decode()
    except Exception as e:
        return f"(err {e})"


print("workers:", workers(), flush=True)


def one(i):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Reply with just the number {i}*7."}],
        "max_tokens": 32,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
        ch = (d or {}).get("choices") or []
        if not ch:
            return (i, round(time.time() - t0, 1), "ERR", f"no choices: {str(d)[:60]}")
        txt = ch[0]["message"]["content"][:40].replace("\n", " ")
        return (i, round(time.time() - t0, 1), "ok", txt)
    except Exception as e:
        return (i, round(time.time() - t0, 1), "ERR", str(e)[:80])


t0 = time.time()
ok = 0
with cf.ThreadPoolExecutor(max_workers=N) as ex:
    for i, dt, st, txt in ex.map(one, range(N)):
        print(f"  [{i:2d}] {st} {dt:6.1f}s  {txt}", flush=True)
        ok += st == "ok"
print(f"DONE {ok}/{N} ok in {round(time.time()-t0,1)}s", flush=True)
print("workers_after:", workers(), flush=True)
