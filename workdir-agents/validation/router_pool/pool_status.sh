#!/bin/bash
# Show the live worker set the router currently routes to.
set -euo pipefail
POOL_DIR_HOST="${POOL_DIR_HOST:-/lustre/fsw/portfolios/nemotron/users/alaptev/data/lazypool}"
ROUTER_URL=$(cat "${POOL_DIR_HOST}/router_url" 2>/dev/null || true)
if [ -z "$ROUTER_URL" ]; then echo "no router_url yet"; exit 0; fi
echo "router: $ROUTER_URL"
echo "--- GET /workers ---"
curl -s "${ROUTER_URL}/workers" || echo "(no /workers route)"
echo
echo "--- GET /v1/models ---"
curl -s "${ROUTER_URL}/v1/models" || true
echo
