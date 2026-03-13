#!/usr/bin/env python3
"""Setup all remote machines: sync code and install dependencies."""

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
    ssh,
)


def setup_worker(worker):
    ssh(worker, f"mkdir -p {worker.project_dir}", capture=True)
    rsync_to(
        worker,
        [
            str(PROJECT_ROOT / "src"),
            str(PROJECT_ROOT / "scripts"),
            str(PROJECT_ROOT / "environment.yml"),
        ],
        f"{worker.project_dir}/",
        extra_args=["--exclude=__pycache__", "--quiet"],
        capture=True,
    )
    ssh(worker, (
        "apt-get install -y build-essential htop vim && "
        ". /opt/miniforge3/etc/profile.d/conda.sh && "
        f"cd {worker.project_dir} && "
        "(conda env create -f environment.yml 2>/dev/null || conda env update -f environment.yml)"
    ), capture=True)


def main():
    parser = argparse.ArgumentParser(description="Setup all remote machines")
    parser.add_argument("--workers", type=str, default=None,
                        help="Comma-separated worker names (default: all)")
    args = parser.parse_args()

    workers = load_workers(parse_worker_names(args.workers))
    print(f"Setting up {len(workers)} worker(s): {[w.name for w in workers]}")
    results = run_on_all(workers, setup_worker, desc="setup")
    sys.exit(0 if report_results(results) else 1)


if __name__ == "__main__":
    main()
