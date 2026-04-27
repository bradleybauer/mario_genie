#!/usr/bin/env python3
"""Sync local data/ to remote machines."""

from __future__ import annotations

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
    ssh,
)


def local_data_sources(*, include_raw: bool = False, only_raw: bool = False) -> list[Path]:
    """Return existing local data directories to sync."""
    if only_raw:
        candidates = [PROJECT_ROOT / "data" / "raw"]
    else:
        candidates = [
            PROJECT_ROOT / "data" / "normalized",
            PROJECT_ROOT / "data" / "latents",
        ]
        if include_raw:
            candidates.append(PROJECT_ROOT / "data" / "raw")
    return [path for path in candidates if path.exists()]


def rsync_extra_args_for_source(source: Path) -> list[str]:
    """Return rsync flags tailored to a specific data directory."""
    extra_args = ["--progress"]
    if source.name == "raw":
        extra_args.append("--exclude=*.mss")
    return extra_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send data to remotes")
    raw_group = parser.add_mutually_exclusive_group()
    raw_group.add_argument(
        "--raw",
        action="store_true",
        help="Also sync data/raw in addition to normalized data and latents.",
    )
    raw_group.add_argument(
        "--only-raw",
        action="store_true",
        help="Sync only data/raw.",
    )
    parser.add_argument("workers", nargs="*", help="Worker names, or 'all' (omit to list available)")
    return parser.parse_args()


def send_data(worker, *, include_raw: bool = False, only_raw: bool = False):
    sources = local_data_sources(include_raw=include_raw, only_raw=only_raw)
    if not sources:
        if only_raw:
            selected = "raw"
        else:
            selected = "normalized, latents, raw" if include_raw else "normalized, latents"
        return f"skipped (no local data directories found: {selected})"

    ssh(worker, f"mkdir -p {worker.project_dir}/data", capture=True)
    for source in sources:
        rsync_to(
            worker,
            str(source),
            f"{worker.project_dir}/data/",
            extra_args=rsync_extra_args_for_source(source),
        )


def main():
    args = parse_args()

    if not args.workers:
        show_workers()
        sys.exit(0)

    workers = load_workers(None if "all" in args.workers else args.workers)
    if args.only_raw:
        raw_suffix = " (raw only)"
    elif args.raw:
        raw_suffix = " (+ raw)"
    else:
        raw_suffix = ""
    print(f"Sending data to {len(workers)} worker(s){raw_suffix}: {[w.name for w in workers]}")

    def send_selected_data(worker):
        return send_data(worker, include_raw=args.raw, only_raw=args.only_raw)

    results = run_on_all(workers, send_selected_data, desc="send data")
    sys.exit(0 if report_results(results) else 1)


if __name__ == "__main__":
    main()
