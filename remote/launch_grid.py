#!/usr/bin/env python3
"""Launch sharded full-dataset sweep workers on all remote machines.

Each worker gets a deterministic shard of models and trains its assigned
configurations once on the full dataset.

Usage:
    python multi_remote/launch_sweep.py --threshold 0.0008
    python multi_remote/launch_sweep.py --workers h100a,h100b -- --max-minutes 60
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import load_workers, parse_worker_names, ssh

SWEEP_SCRIPT = "scripts/sweep_model_configs_genie.py"
TMUX_SESSION = "sweep"


def main():
    parser = argparse.ArgumentParser(
        description="Launch sharded sweep on all workers. "
                    "Pass extra sweep args after '--'.",
    )
    parser.add_argument("--workers", type=str, default=None,
                        help="Comma-separated worker names (default: all)")
    parser.add_argument("--sweep-dir", type=str,
                        default="checkpoints/model_config_sweep_genie",
                        help="Sweep output dir relative to project root")
    parser.add_argument("--session", type=str, default=TMUX_SESSION,
                        help="Tmux session name (default: sweep)")
    args, extra = parser.parse_known_args()

    # Strip leading '--' separator if present
    if extra and extra[0] == "--":
        extra = extra[1:]
    sweep_extra = " ".join(extra)

    workers = load_workers(parse_worker_names(args.workers))
    num_shards = len(workers)

    print(f"Launching sweep across {num_shards} worker(s): {[w.name for w in workers]}")
    if sweep_extra:
        print(f"Extra sweep args: {sweep_extra}")

    for shard_index, worker in enumerate(workers):
        sweep_cmd = " ".join([
            "python", SWEEP_SCRIPT,
            f"--shard-index {shard_index}",
            f"--num-shards {num_shards}",
            f"--output-dir {args.sweep_dir}",
            sweep_extra,
        ])

        launcher_script = "\n".join([
            "#!/bin/bash",
            ". /opt/miniforge3/etc/profile.d/conda.sh",
            "conda activate mario",
            "set -e",
            f"cd {worker.project_dir}",
            f"exec {sweep_cmd}",
            "",
        ])

        remote_script = f"{worker.project_dir}/run_sweep.sh"
        print(f"  [{worker.name}] shard {shard_index}/{num_shards}")

        # Write launcher script via stdin
        ssh(worker, f"cat > {remote_script} && chmod +x {remote_script}",
            input=launcher_script)

        # Kill existing session (if any) and start fresh
        ssh(worker, f"tmux kill-session -t {args.session} 2>/dev/null || true",
            check=False)
        ssh(worker, f"tmux new-session -d -s {args.session} 'bash {remote_script}'")
        print(f"  [{worker.name}] tmux session '{args.session}' started")

    print(f"\nAll {num_shards} worker(s) launched.")
    print(f"Monitor: python multi_remote/connect.py <name> --session {args.session}")


if __name__ == "__main__":
    main()
