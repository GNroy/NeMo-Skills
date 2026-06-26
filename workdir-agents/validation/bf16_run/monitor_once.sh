#!/bin/bash
# Periodic watcher: ~2h of 10-min checks of the BF16 run, then exits + notifies.
# Reports per-seed completions, worker count, supervisor health; flags trouble.
set -uo pipefail
SSH="ssh -i /home/alaptev/.ssh/clusters/aws-cmh/id_ecdsa -o ConnectTimeout=20 alaptev@aws-cmh-slurm-1-login-01.nvidia.com"
A=/lustre/fsw/portfolios/nemotron/users/alaptev
for r in $(seq 1 12); do
  SUP=$(pgrep -f bf16_supervisor.sh >/dev/null && echo UP || echo DOWN)
  SQ=$($SSH "squeue --me -h -o '%j|%T'" 2>/dev/null)
  WK=$(echo "$SQ" | grep -c '^lazyalloc_bf16worker_' || true)
  DRV=$(echo "$SQ" | grep -c '^lazyalloc_bf16_s[0-4]_g0|' || true)
  LINE=""; TOTAL=0
  for k in 0 1 2 3 4; do
    D=$($SSH "grep -rl '^status: completed' $A/exp/a0py_bf16_s${k}/worklogs/nhwbf16s${k}/ 2>/dev/null | wc -l" 2>/dev/null)
    D=${D:-0}; TOTAL=$((TOTAL+D)); LINE="$LINE s${k}=$D"
  done
  echo "[$(date +%H:%M:%S)] sup=$SUP drivers=$DRV/5 workers=$WK total=$TOTAL/10790 |$LINE"
  [ "$TOTAL" -ge 10780 ] && { echo "RUN ESSENTIALLY COMPLETE"; break; }
  [ "$SUP" = "DOWN" ] && { echo "WARNING: supervisor DOWN"; break; }
  sleep 600
done
