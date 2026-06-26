#!/bin/bash
# BF16 5-seed supervisor (runs on Spark, background). Until all seeds hit TARGET
# completed problems:
#   1. top the worker pool up to WORKER_TARGET (workers self-deregister at 4h cap),
#   2. resume-restart any seed whose driver is dead and seed still incomplete
#      (resume skips already-completed ids -> safe + idempotent),
#   3. exit when all seeds done, OR when the WHOLE pool makes no progress for
#      STALL_ROUNDS consecutive checks (pathological-tail convergence guard).
set -uo pipefail
# VARIANT=tool|notool (default tool). Selects driver job-name / out_dir / run_id /
# launch-arg so the two arms can run independently (sharing the same worker pool,
# which is model-only and tool-agnostic). Set VARIANT=notool for the no-tool arm.
VARIANT="${VARIANT:-tool}"
if [ "$VARIANT" = "notool" ]; then PFX=lazyalloc_bf16nt; OUTP=a0py_bf16nt_s; RID=nhwbf16nts
else PFX=lazyalloc_bf16; OUTP=a0py_bf16_s; RID=nhwbf16s; fi
# Singleton guard: only ONE supervisor per variant may run (two would double-submit).
exec 9>/home/alaptev/Projects/bf16_run/.supervisor.${VARIANT}.lock
flock -n 9 || { echo "another $VARIANT supervisor holds the lock; exiting" >&2; exit 0; }
SSH="ssh -i /home/alaptev/.ssh/clusters/aws-cmh/id_ecdsa -o ConnectTimeout=20 alaptev@aws-cmh-slurm-1-login-01.nvidia.com"
A=/lustre/fsw/portfolios/nemotron/users/alaptev
SEEDS=(0 1 2 3 4)
# 4 cpu-short (MaxSubmit=4) + 1 cpu-normal (leaves the 2nd cpu-normal node for the co-tenant)
declare -A QOS=( [0]=cpu-short [1]=cpu-short [2]=cpu-short [3]=cpu-short [4]=cpu-normal )
TARGET=2158
WORKER_TARGET="${WORKER_TARGET:-16}"
STALL_ROUNDS="${STALL_ROUNDS:-10}"      # ~10*INTERVAL with no global progress -> stop
INTERVAL="${INTERVAL:-180}"
LAUNCH=/home/alaptev/Projects/bf16_run/launch_bf16_driver.sh
LOG=/home/alaptev/Projects/bf16_run/supervisor.${VARIANT}.log
declare -A PREV
TOTAL_PREV=-1; STALL=0; ROUND=0
echo "=== supervisor start $(date) variant=$VARIANT target=$TARGET workers=$WORKER_TARGET ===" >>"$LOG"
while true; do
  ROUND=$((ROUND+1)); TS=$(date +%H:%M:%S)
  $SSH "bash $A/router_pool/topup_workers.sh $WORKER_TARGET" >>"$LOG" 2>&1
  SQ=$($SSH "squeue --me -h -o '%j|%T'" 2>/dev/null)
  WK=$(echo "$SQ" | grep -c '^lazyalloc_bf16worker_' || true)
  RWK=$(echo "$SQ" | grep '^lazyalloc_bf16worker_' | grep -c 'RUNNING' || true)
  ALLDONE=1; TOTAL=0; LINE=""
  for k in "${SEEDS[@]}"; do
    DONE=$($SSH "grep -rl '^status: completed' $A/exp/${OUTP}${k}/worklogs/${RID}${k}/ 2>/dev/null | wc -l" 2>/dev/null)
    DONE=${DONE:-0}; TOTAL=$((TOTAL+DONE))
    ALIVE=$(echo "$SQ" | grep -c "^${PFX}_s${k}_g0|" || true)
    LINE="$LINE s${k}=${DONE}(drv${ALIVE})"
    if [ "$DONE" -lt "$TARGET" ]; then
      ALLDONE=0
      if [ "$ALIVE" -eq 0 ]; then
        echo "[$TS] seed $k incomplete ($DONE/$TARGET) + driver dead -> restart qos=${QOS[$k]}" >>"$LOG"
        $LAUNCH "$k" "${QOS[$k]}" "$VARIANT" >>"$LOG" 2>&1
      fi
    fi
    PREV[$k]=$DONE
  done
  echo "[$TS r$ROUND] total=$TOTAL/$((TARGET*5)) workers=$WK(run $RWK) |$LINE" >>"$LOG"
  if [ "$ALLDONE" -eq 1 ]; then echo "[$TS] ALL SEEDS DONE total=$TOTAL" >>"$LOG"; break; fi
  # Only count a stall round when serving capacity EXISTS (>=1 running worker).
  # GPU starvation (0 running workers, backfill-queued) is NOT convergence.
  if [ "$RWK" -ge 1 ] && [ "$TOTAL" -le "$TOTAL_PREV" ]; then STALL=$((STALL+1)); else STALL=0; fi
  TOTAL_PREV=$TOTAL
  if [ "$STALL" -ge "$STALL_ROUNDS" ]; then
    echo "[$TS] STALL: no progress for $STALL rounds WITH workers running; stopping (converged tail). total=$TOTAL" >>"$LOG"; break
  fi
  sleep "$INTERVAL"
done
echo "=== supervisor end $(date) ===" >>"$LOG"
