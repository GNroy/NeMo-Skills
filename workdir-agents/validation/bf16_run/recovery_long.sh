#!/bin/bash
# Long watcher for the GPU-blocked tail: poll every 15 min, up to ~4h.
# Notifies (exits) when workers start RUNNING (run resumes) or run completes.
set -uo pipefail
SSH="ssh -i /home/alaptev/.ssh/clusters/aws-cmh/id_ecdsa -o ConnectTimeout=20 alaptev@aws-cmh-slurm-1-login-01.nvidia.com"
A=/lustre/fsw/portfolios/nemotron/users/alaptev
RU=$($SSH "cat $A/data/lazypool_bf16/router_url" 2>/dev/null | tr -dc '[:print:]')
intify(){ echo "${1//[^0-9]/}" | grep -oE '^[0-9]+' || echo 0; }
for r in $(seq 1 16); do
  RWK=$(intify "$($SSH "squeue --me -h -o '%j|%T' 2>/dev/null | grep '^lazyalloc_bf16worker_' | grep -c RUNNING")")
  DRV=$(intify "$($SSH "squeue --me -h -o '%j' 2>/dev/null | grep -c 'bf16_s[0-4]_g0'")")
  T=0; for k in 0 1 2 3 4; do D=$(intify "$($SSH "grep -rl '^status: completed' $A/exp/a0py_bf16_s${k}/worklogs/nhwbf16s${k}/ 2>/dev/null | wc -l")"); T=$((T+D)); done
  SUP=$(pgrep -f bf16_supervisor.sh >/dev/null && echo UP || echo DOWN)
  echo "[$(date +%H:%M:%S)] sup=$SUP drivers=${DRV}/5 running_workers=${RWK} total=${T}/10790"
  [ "$T" -ge 10780 ] && { echo "RUN COMPLETE"; break; }
  if [ "$RWK" -ge 1 ] && [ "$T" -gt 9121 ]; then echo "RECOVERED: workers running + progress resumed"; break; fi
  [ "$SUP" = DOWN ] && { echo "WARN supervisor down"; break; }
  sleep 900
done
