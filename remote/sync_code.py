#!/usr/bin/env python3
"""Sync local code to all remote machines (without full setup/install)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import (
    PROJECT_ROOT,
    load_workers,
    parse_worker_names,
    report_results,
    rsync_to,
    run_on_all,
)


def sync_worker(worker):
    rsync_to(
        worker,
        [
            str(PROJECT_ROOT / "src"),
            str(PROJECT_ROOT / "scripts"),
            str(PROJECT_ROOT / "environment.yml"),
        ],
        f"{worker.project_dir}/",
        extra_args=[
            "--exclude=__pycache__",
            "--exclude=.git",
            "--exclude=data",
            "--exclude=checkpoints",
            "--exclude=results",
            "--exclude=*.pyc",
            "--quiet",
        ],
        capture=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Sync code to all remotes")
    parser.add_argument("--workers", type=str, default=None,
                        help="Comma-separated worker names (default: all)")
    args = parser.parse_args()

    workers = load_workers(parse_worker_names(args.workers))
    print(f"Syncing code to {len(workers)} worker(s): {[w.name for w in workers]}")
    results = run_on_all(workers, sync_worker, desc="sync code")
    sys.exit(0 if report_results(results) else 1)


if __name__ == "__main__":
    main()
