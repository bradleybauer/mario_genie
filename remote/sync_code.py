#!/usr/bin/env python3
"""Sync local code to remote machines (without full setup/install)."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from remote.helpers import (
    load_workers,
    report_results,
    rsync_to,
    run_on_all,
    show_workers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync code to remotes")
    parser.add_argument("workers", nargs="*", help="Worker names, or 'all' (omit to list available)")
    return parser.parse_args()


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
            "--exclude=*.pyc",
            "--quiet",
        ],
        capture=True,
    )


def main():
    args = parse_args()

    if not args.workers:
        show_workers()
        sys.exit(0)

    workers = load_workers(None if "all" in args.workers else args.workers)
    print(f"Syncing code to {len(workers)} worker(s): {[w.name for w in workers]}")
    results = run_on_all(workers, sync_worker, desc="sync code")
    sys.exit(0 if report_results(results) else 1)


if __name__ == "__main__":
    main()
