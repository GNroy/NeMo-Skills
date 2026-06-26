#!/bin/bash
# Re-register pre-warmed GPU workers into the CURRENT router. Works both in-container
# (/alaptev) and on the bare node (/lustre/.../alaptev). Run node-local via srun.
if [ -d /alaptev/data/lazypool_5seed ]; then POOL=/alaptev/data/lazypool_5seed
else POOL=/lustre/fsw/portfolios/nemotron/users/alaptev/data/lazypool_5seed; fi
RPORT="${ROUTER_PORT:-20000}"
echo "POOL=$POOL urls=$(wc -l < "$POOL/worker_urls.txt" 2>/dev/null)"
n=0
while read -r u; do
  [ -z "$u" ] && continue
  resp=$(curl -s -m 10 -X POST -H 'Content-Type: application/json' -d "{\"url\":\"$u\"}" "http://localhost:${RPORT}/workers")
  echo "  POST $u -> ${resp:0:80}"
  n=$((n+1))
done < "$POOL/worker_urls.txt"
echo "posted $n"
sleep 8
echo -n "registered now: "; curl -s -m 10 "http://localhost:${RPORT}/workers" | grep -oc '"url"'
