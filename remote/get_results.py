#!/usr/bin/env python3
"""Retrieve training results from all remote machines."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import (
    PROJECT_ROOT,
    load_workers,
    parse_worker_names,
    report_results,
    rsync_from,
    run_on_all,
)


def main():
    parser = argparse.ArgumentParser(description="Get results from remotes")
    parser.add_argument("subdir", nargs="?", default=None,
                        help="Checkpoint subdirectory to fetch (default: all)")
    parser.add_argument("--workers", type=str, default=None,
                        help="Comma-separated worker names (default: all)")
    parser.add_argument("--images", action="store_true",
                        help="Only sync images and JSON (skip model weights)")
    args = parser.parse_args()

    workers = load_workers(parse_worker_names(args.workers))

    remote_suffix = "checkpoints/"
    local_base = str(PROJECT_ROOT / "results" / "model_config_sweep_genie")
    if args.subdir:
        remote_suffix += args.subdir + "/"
        local_base = os.path.join(local_base, args.subdir)

    rsync_extras = []
    if args.images:
        rsync_extras = [
            "--include=*/", "--include=*.png", "--include=*.json", "--exclude=*",
        ]

    def fetch(worker):
        local_dst = local_base + "/"
        os.makedirs(local_dst, exist_ok=True)
        rsync_from(
            worker,
            f"{worker.project_dir}/{remote_suffix}",
            local_dst,
            extra_args=["--progress"] + rsync_extras,
        )

    print(f"Fetching results from {len(workers)} worker(s): {[w.name for w in workers]}")
    results = run_on_all(workers, fetch, desc="get results")
    sys.exit(0 if report_results(results) else 1)


if __name__ == "__main__":
    main()
