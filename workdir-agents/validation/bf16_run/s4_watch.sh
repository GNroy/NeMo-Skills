#!/bin/bash
set -uo pipefail
SSH="ssh -i /home/alaptev/.ssh/clusters/aws-cmh/id_ecdsa -o ConnectTimeout=20 alaptev@aws-cmh-slurm-1-login-01.nvidia.com"
A=/lustre/fsw/portfolios/nemotron/users/alaptev
for r in $(seq 1 8); do
  C=$($SSH "grep -rl '^status: completed' $A/exp/a0py_bf16_s4/worklogs/nhwbf16s4/ 2>/dev/null | wc -l")
  ALIVE=$($SSH "squeue --me -h -o '%j %M' | grep bf16_s4_g0 || echo DEAD")
  echo "[$(date +%H:%M:%S)] s4 completed=$C  driver: $ALIVE"
  sleep 180
done
