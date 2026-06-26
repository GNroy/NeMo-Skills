#!/bin/bash
# Probe the sgl-model-gateway (smg) /workers REST API for dynamic worker mgmt.
set -uo pipefail
PORT=21078
echo "LAUNCH:"
python3 -m sglang_router.launch_router --host 127.0.0.1 --port $PORT --policy round_robin >/tmp/r.log 2>&1 &
RP=$!
for i in $(seq 1 40); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done

show() { echo "== $1 =="; curl -s -w "\n[HTTP %{http_code}]\n" "${@:2}"; echo; }

show "GET /workers"            http://127.0.0.1:$PORT/workers
FAKE='http://127.0.0.1:39999'
# REST POST /workers with JSON body (several shapes the smg API might accept):
show "POST /workers {url}"     -X POST -H 'Content-Type: application/json' -d "{\"url\":\"$FAKE\"}"            http://127.0.0.1:$PORT/workers
show "POST /workers {worker_urls}" -X POST -H 'Content-Type: application/json' -d "{\"worker_urls\":[\"$FAKE\"]}" http://127.0.0.1:$PORT/workers
show "POST /workers?url"        -X POST "http://127.0.0.1:$PORT/workers?url=$FAKE"
show "GET /workers after add"   http://127.0.0.1:$PORT/workers
show "DELETE /workers?url"      -X DELETE "http://127.0.0.1:$PORT/workers?url=$FAKE"
show "GET /v1/models"           http://127.0.0.1:$PORT/v1/models
# Discover all routes if an openapi/route dump exists:
show "GET /openapi.json"        http://127.0.0.1:$PORT/openapi.json
echo "PY_HELP:"; python3 -m sglang_router.launch_router --help 2>&1 | tail -40
echo "ROUTER_LOG_TAIL:"; tail -8 /tmp/r.log
kill $RP 2>/dev/null || true
echo "PROBE_DONE"
