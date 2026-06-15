# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serve a data-parallel pool of single-node SGLang workers behind one sglang_router endpoint.

Topology (Ray-free): the owning het-group gets ``num_nodes`` nodes, launched with
one task per node (ntasks-per-node=1). Each task starts one self-contained
``sglang.launch_server`` worker (TP=num_gpus, one replica per node) on an internal
port. Rank 0 additionally launches ``sglang_router`` on the public ``--port`` and
points it at every worker URL (enumerated from the step's SLURM nodelist). Clients
hit the het-group master node on ``--port`` and the router load-balances across the
K replicas. No cross-node tensor/data parallelism, so none of the multi-node Ray /
shm_broadcast instability that sinks ``vllm_dp_ray`` for this model.
"""

import argparse
import os
import subprocess
import sys
from shlex import join


def _worker_hostnames():
    """Hostnames of the nodes in THIS het-group, for the router's --worker-urls.

    Primary source is ``SLURM_GROUP_NODES`` (space-separated), which the pipeline
    expands HOST-SIDE via scontrol and injects into the container env -- scontrol
    is not installed in the serving images. Falls back to an in-container scontrol
    expansion, then to localhost (single-node).
    """
    env_hosts = os.environ.get("SLURM_GROUP_NODES", "").strip()
    if env_hosts:
        return [h for h in env_hosts.split() if h.strip()]
    nodelist = os.environ.get("SLURM_STEP_NODELIST") or os.environ.get("SLURM_NODELIST") or ""
    if nodelist:
        try:
            out = subprocess.check_output(["scontrol", "show", "hostnames", nodelist]).decode()
            hosts = [h for h in out.split() if h.strip()]
            if hosts:
                return hosts
        except Exception as e:
            print(f"[serve_sglang_router] scontrol fallback failed: {e}")
    return ["localhost"]


def main():
    parser = argparse.ArgumentParser(description="Serve an SGLang DP worker pool behind sglang_router")
    parser.add_argument("--model", help="Path to the model or a model name to pull from HF")
    parser.add_argument("--num_gpus", type=int, required=True, help="GPUs per worker replica (TP size)")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of worker replicas (one per node)")
    parser.add_argument("--port", type=int, default=20000, help="Public router port (clients hit this)")
    # router-control args (consumed here, NOT forwarded to the worker sglang.launch_server)
    parser.add_argument("--router_policy", default="round_robin",
                        help="sglang_router load-balancing policy (round_robin/cache_aware/...)")
    parser.add_argument("--router_worker_startup_timeout_secs", type=int, default=1800,
                        help="How long the router waits for workers to come up (model load is slow)")
    parser.add_argument("--router_request_timeout_secs", type=int, default=3600,
                        help="Per-request timeout at the router (long reasoning solves)")
    args, unknown = parser.parse_known_args()

    worker_port = args.port + 1  # internal; router lives on args.port
    proc_id = int(os.environ.get("SLURM_PROCID", "0"))
    node = os.environ.get("SLURMD_NODENAME", "localhost")
    extra_arguments = join(unknown)

    print(f"[serve_sglang_router] rank={proc_id} node={node} worker_port={worker_port} router_port={args.port}")
    sys.stdout.flush()

    # --- every node: one self-contained TP=num_gpus worker on the internal port ---
    worker_cmd = (
        f"python3 -m sglang.launch_server "
        f'    --model="{args.model}" '
        f'    --served-model-name="{args.model}" '
        f"    --trust-remote-code "
        f'    --host="0.0.0.0" '
        f"    --port={worker_port} "
        f"    --tensor-parallel-size={args.num_gpus} "
        f"    {extra_arguments} "
    )
    worker = subprocess.Popen(worker_cmd, shell=True)

    router = None
    if proc_id == 0:
        hosts = _worker_hostnames()
        worker_urls = " ".join(f"http://{h}:{worker_port}" for h in hosts)
        print(f"[serve_sglang_router] router worker-urls: {worker_urls}")
        sys.stdout.flush()
        router_cmd = (
            f"python3 -m sglang_router.launch_router "
            f'    --host="0.0.0.0" '
            f"    --port={args.port} "
            f"    --worker-urls {worker_urls} "
            f"    --policy {args.router_policy} "
            f"    --worker-startup-timeout-secs {args.router_worker_startup_timeout_secs} "
            f"    --request-timeout-secs {args.router_request_timeout_secs} "
        )
        router = subprocess.Popen(router_cmd, shell=True)

    # Block on the worker so the task stays alive; if the worker dies the task exits
    # non-zero and srun's --kill-on-bad-exit tears the pool down cleanly.
    rc = worker.wait()
    if router is not None:
        router.terminate()
    sys.exit(rc)


if __name__ == "__main__":
    main()
