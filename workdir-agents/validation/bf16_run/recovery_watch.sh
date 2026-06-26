#!/bin/bash
set -uo pipefail
SSH="ssh -i /home/alaptev/.ssh/clusters/aws-cmh/id_ecdsa -o ConnectTimeout=20 alaptev@aws-cmh-slurm-1-login-01.nvidia.com"
A=/lustre/fsw/portfolios/nemotron/users/alaptev
RU=$($SSH "cat $A/data/lazypool_bf16/router_url" 2>/dev/null)
for r in $(seq 1 8); do
  HW=$($SSH "curl -s --max-time 8 '$RU/workers' 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)[\"total\"])' 2>/dev/null" || echo "?")
  WST=$($SSH "squeue --me -h -o '%T' | grep -c '' ; squeue --me -h -t RUNNING -o '%j'|grep -c bf16worker" 2>/dev/null | tr '\n' ' ')
  DRV=$($SSH "squeue --me -h -o '%j %M' | grep 'bf16_s[0-4]_g0' | awk '{printf \"%s=%s \", substr(\$1,15,2), \$2}'" 2>/dev/null)
  T=0; for k in 0 1 2 3 4; do D=$($SSH "grep -rl '^status: completed' $A/exp/a0py_bf16_s${k}/worklogs/nhwbf16s${k}/ 2>/dev/null | wc -l"); T=$((T+D)); done
  echo "[$(date +%H:%M:%S)] healthy_workers=$HW running_workers=$(echo $WST|awk '{print $2}') total=$T/10790 | drivers: $DRV"
  [ "$T" -ge 10780 ] && { echo DONE; break; }
  sleep 180
done
