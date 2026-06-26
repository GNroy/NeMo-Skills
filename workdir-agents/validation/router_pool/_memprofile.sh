#!/bin/bash
# Memory breakdown of a running driver node, by process category, RSS + PSS.
echo "=== free ==="; free -g | head -2
echo "=== category RSS/PSS (GB) + counts ==="
python3 - <<'PY'
import os, glob, re
cats = {}
def cat(cmd):
    if 'uwsgi' in cmd: return 'sandbox_uwsgi'
    if re.search(r'raylet|gcs_server|ray/core|plasma|ray::', cmd): return 'ray'
    if re.search(r'ng_run|ng_collect|nemo_gym', cmd): return 'gym_head'
    if 'uvicorn' in cmd: return 'gym_uvicorn'
    if re.search(r'stdio_serve|python_tool|mcp', cmd): return 'mcp'
    if 'run_agent' in cmd or 'hermes' in cmd: return 'hermes_agent'
    if 'python' in cmd: return 'other_python'
    return 'non_python'
for p in glob.glob('/proc/[0-9]*'):
    pid = p.split('/')[-1]
    try:
        cmd = open(f'{p}/cmdline','rb').read().replace(b'\x00',b' ').decode(errors='ignore').strip()
        if not cmd: continue
        rss = pss = 0
        for l in open(f'{p}/smaps_rollup'):
            if l.startswith('Rss:'): rss = int(l.split()[1])
            elif l.startswith('Pss:'): pss = int(l.split()[1])
    except Exception: continue
    k = cat(cmd)
    d = cats.setdefault(k, [0,0,0])
    d[0]+=rss; d[1]+=pss; d[2]+=1
print(f"{'category':18} {'RSS_GB':>8} {'PSS_GB':>8} {'count':>6}")
tr=tp=0
for k,(r,p,n) in sorted(cats.items(), key=lambda x:-x[1][1]):
    print(f"{k:18} {r/1024/1024:8.1f} {p/1024/1024:8.1f} {n:6d}"); tr+=r; tp+=p
print(f"{'TOTAL':18} {tr/1024/1024:8.1f} {tp/1024/1024:8.1f}")
PY
echo "=== uwsgi worker count + ini files ==="
ps -eo cmd --no-headers | grep -oE 'worker[0-9]+_uwsgi' | sort -u | wc -l
echo "=== sandbox uwsgi config (workers/cheaper) ==="
cat /tmp/worker1_uwsgi.ini 2>/dev/null | grep -iE 'processes|workers|cheaper|threads' | head
