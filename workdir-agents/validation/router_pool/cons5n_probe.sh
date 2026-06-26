#!/bin/bash
# De-risk probe: node-level RSS/FD/socket pressure + worker/sandbox + driver progress.
JID="${1:-669413}"; POOL=/lustre/fsw/portfolios/nemotron/users/alaptev/data/lazypool_5seed
# run the node-local probe inside the job's allocation
srun --jobid="$JID" --overlap --nodes=1 --ntasks=1 --gpus-per-node=0 bash -c '
  M=$(free -g | awk "/^Mem:/{print \$3\"/\"\$2\"G used,\"\$7\"avail\"}")
  FN=$(awk "{print \$1}" /proc/sys/fs/file-nr)
  EST=$(ss -tn state established 2>/dev/null | tail -n +1 | wc -l)
  TW=$(ss -tan state time-wait 2>/dev/null | tail -n +1 | wc -l)
  CH=$(pgrep -c -f hermes 2>/dev/null || echo "?")
  W=$(curl -s localhost:20000/workers 2>/dev/null | grep -oE "\"url\"" | wc -l)
  SOK=$(curl -s -o /dev/null -w "%{http_code}" localhost:6000/health 2>/dev/null)
  echo "MEM=$M FD=$FN est=$EST timewait=$TW hermesProcs=$CH workers=$W sandboxHealth=$SOK"
' 2>/dev/null || echo "PROBE_SRUN_FAILED"
echo -n "progress: "; grep -hoE "RUNNING\[[0-9]+/[0-9]+\]|[0-9]+/2158" "$POOL"/cons2n_driver_s*.log 2>/dev/null | tail -2 | paste -sd' ' ; \
  grep -hcE "ClientOSError|MemoryError|Cannot allocate|Killed|Too many open" "$POOL"/cons2n_driver_s*.log "$POOL"/shared_sandbox.log 2>/dev/null | paste -sd'+' | bc 2>/dev/null | sed 's/^/errSignatures=/'
