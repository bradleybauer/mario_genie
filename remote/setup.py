#!/usr/bin/env python3
"""Setup remote machines: sync code and install dependencies."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Setup remote machines")
    parser.add_argument("workers", nargs="*", help="Worker names, or 'all' (omit to list available)")
    return parser.parse_args()


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
        "(conda env create -f environment.yml 2>/dev/null || conda env update -f environment.yml) && "
        "grep -qxF '. /opt/miniforge3/etc/profile.d/conda.sh' ~/.bashrc || echo '. /opt/miniforge3/etc/profile.d/conda.sh' >> ~/.bashrc && "
        "grep -qxF 'conda activate mario' ~/.bashrc || echo 'conda activate mario' >> ~/.bashrc && "
        "grep -qxF 'alias py=python' ~/.bashrc || echo 'alias py=python' >> ~/.bashrc && "
        f"grep -qxF 'cd {worker.project_dir}' ~/.bashrc || echo 'cd {worker.project_dir}' >> ~/.bashrc"
    ), capture=True)


def main():
    args = parse_args()

    if not args.workers:
        show_workers()
        sys.exit(0)

    workers = load_workers(None if "all" in args.workers else args.workers)
    print(f"Setting up {len(workers)} worker(s): {[w.name for w in workers]}")
    results = run_on_all(workers, setup_worker, desc="setup")
    sys.exit(0 if report_results(results) else 1)


if __name__ == "__main__":
    main()
