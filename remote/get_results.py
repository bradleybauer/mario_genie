#!/usr/bin/env python3
"""Retrieve training results from all remote machines."""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from remote.helpers import (
    load_workers,
    report_results,
    rsync_from,
    run_on_all,
    show_workers,
    ssh,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get results from remotes")
    parser.add_argument("subdir", nargs="?", default=None,
                        help="Checkpoint subdirectory to fetch (default: all)")
    parser.add_argument("--workers", nargs="+",
                        help="Worker names, or 'all' (omit to list available)")
    parser.add_argument("--images", action="store_true",
                        help="Only sync images and JSON (skip model weights)")
    return parser.parse_args()


def _remote_result_roots(subdir: str | None) -> list[tuple[str, str]]:
    """Return candidate remote roots and matching local destinations."""
    candidates = []
    remote_suffix = "checkpoints/"
    local_base = str(PROJECT_ROOT / "checkpoints")
    if subdir:
        remote_suffix += subdir + "/"
        local_base = os.path.join(local_base, subdir)
    candidates.append((remote_suffix, local_base))
    return candidates


def _resolve_remote_source(worker, subdir: str | None) -> tuple[str, str]:
    """Find the first existing remote results directory for a worker."""
    checked = []
    for remote_suffix, local_base in _remote_result_roots(subdir):
        remote_path = f"{worker.project_dir}/{remote_suffix}"
        checked.append(remote_path)
        result = ssh(worker, f"test -d {remote_path} && echo OK || true", check=False, capture=True)
        if "OK" in (result.stdout or ""):
            return remote_suffix, local_base
    raise FileNotFoundError(f"No remote results directory found on {worker.name}: {checked}")


def main():
    args = parse_args()

    workers = load_workers(None if args.workers and "all" in args.workers else args.workers)

    rsync_extras = []
    if args.images:
        rsync_extras = [
            "--include=*/", "--include=*.png", "--include=*.json", "--exclude=*",
        ]

    def fetch(worker):
        remote_suffix, local_base = _resolve_remote_source(worker, args.subdir)
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
