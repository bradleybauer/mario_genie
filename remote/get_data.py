#!/usr/bin/env python3
"""Fetch normalized and latent datasets from remote machines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from remote.helpers import load_workers, report_results, rsync_from, run_on_all, show_workers, ssh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch data from remotes")
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--only-normalized",
        action="store_true",
        help="Fetch only data/normalized.",
    )
    data_group.add_argument(
        "--only-latents",
        action="store_true",
        help="Fetch only data/latents.",
    )
    parser.add_argument("workers", nargs="*", help="Worker names, or 'all' (omit to list available)")
    return parser.parse_args()


def selected_data_names(*, only_normalized: bool = False, only_latents: bool = False) -> list[str]:
    if only_normalized:
        return ["normalized"]
    if only_latents:
        return ["latents"]
    return ["normalized", "latents"]


def remote_data_sources(worker, *, only_normalized: bool = False, only_latents: bool = False) -> list[str]:
    """Return remote data directories that exist for a worker."""
    sources: list[str] = []
    for name in selected_data_names(only_normalized=only_normalized, only_latents=only_latents):
        remote_path = f"{worker.project_dir}/data/{name}"
        result = ssh(worker, f"test -d {remote_path} && echo OK || true", check=False, capture=True)
        if "OK" in (result.stdout or ""):
            sources.append(name)
    return sources


def fetch_data(worker, *, only_normalized: bool = False, only_latents: bool = False):
    sources = remote_data_sources(
        worker,
        only_normalized=only_normalized,
        only_latents=only_latents,
    )
    if not sources:
        if only_normalized:
            selected = "normalized"
        elif only_latents:
            selected = "latents"
        else:
            selected = "normalized, latents"
        return f"skipped (no remote data directories found: {selected})"

    for name in sources:
        local_dst = PROJECT_ROOT / "data" / name
        local_dst.mkdir(parents=True, exist_ok=True)
        rsync_from(
            worker,
            f"{worker.project_dir}/data/{name}/",
            f"{local_dst}/",
            extra_args=["--progress"],
        )

    return ", ".join(sources)


def main() -> None:
    args = parse_args()

    if not args.workers:
        show_workers()
        sys.exit(0)

    workers = load_workers(None if "all" in args.workers else args.workers)
    if args.only_normalized:
        data_suffix = " (normalized only)"
    elif args.only_latents:
        data_suffix = " (latents only)"
    else:
        data_suffix = ""
    print(f"Fetching data from {len(workers)} worker(s){data_suffix}: {[w.name for w in workers]}")

    def fetch_selected_data(worker):
        return fetch_data(
            worker,
            only_normalized=args.only_normalized,
            only_latents=args.only_latents,
        )

    results = run_on_all(workers, fetch_selected_data, desc="get data")
    sys.exit(0 if report_results(results) else 1)


if __name__ == "__main__":
    main()