#!/usr/bin/env python3
"""Sync local data/ to all remote machines."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import (
    PROJECT_ROOT,
    load_workers,
    report_results,
    rsync_to,
    run_on_all,
    show_workers,
    ssh,
)


def send_data(worker):
    ssh(worker, f"mkdir -p {worker.project_dir}/data", capture=True)
    rsync_to(
        worker,
        str(PROJECT_ROOT / "data") + "/",
        f"{worker.project_dir}/data/",
        extra_args=["--progress"],
    )


def main():
    parser = argparse.ArgumentParser(description="Send data to all remotes")
    parser.add_argument("workers", nargs="*", help="Worker names (omit to list available)")
    args = parser.parse_args()

    if not args.workers:
        show_workers()
        sys.exit(0)

    workers = load_workers(args.workers)
    print(f"Sending data to {len(workers)} worker(s): {[w.name for w in workers]}")
    results = run_on_all(workers, send_data, desc="send data")
    sys.exit(0 if report_results(results) else 1)


if __name__ == "__main__":
    main()
