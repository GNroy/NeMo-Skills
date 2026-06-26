#!/bin/bash
#SBATCH -p cpu
#SBATCH -A nemotron_reason_science
#SBATCH --qos=cpu-short
#SBATCH -t 00:08:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=0
A=/lustre/fsw/portfolios/nemotron/users/alaptev
GYM_IMG=/lustre/fsw/portfolios/nemotron/users/igitman/images/nemo-skills-dc43f3e.sqsh
export UV_PYTHON_INSTALL_DIR=/alaptev/uv_python PYTHONDONTWRITEBYTECODE=1
echo "[test] launching 5 concurrent containers, each heavy-importing from isolated venvs"
pids=()
for k in 0 1 2 3 4; do
  srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=0 --container-env=UV_PYTHON_INSTALL_DIR,PYTHONDONTWRITEBYTECODE \
    --container-image="$GYM_IMG" --container-mounts="$A:/alaptev" bash -lc '
      cd /alaptev/NeMo-Gym-5seed/resources_servers/passthrough && source .venv/bin/activate
      python -c "import sys,asyncio,json.decoder,zipfile,zipfile._path,ssl,ctypes,ray,markdown_it,wandb; print(\"S'$k' OK\", sys.version.split()[0], sys.executable)"
    ' > "$A/data/lazypool_5seed/concimport_s${k}.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[test] all done rc=$rc"
echo "=== results ==="
for k in 0 1 2 3 4; do echo "s$k: $(tail -1 $A/data/lazypool_5seed/concimport_s${k}.log)"; done
