#!/usr/bin/env python3
"""Sync checkpoint files or run directories to remote machines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from remote.helpers import load_workers, report_results, rsync_to, run_on_all, ssh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send checkpoints to remotes")
    parser.add_argument(
        "source",
        help=(
            "Checkpoint run name, checkpoint directory, or checkpoint file. "
            "Examples: video_vae_20260411_232136 or checkpoints/video_vae_20260411_232136/best.pt"
        ),
    )
    parser.add_argument(
        "--workers",
        nargs="+",
        help="Worker names, or 'all' (omit for all configured workers)",
    )
    return parser.parse_args()


def _candidate_sources(raw: str) -> list[Path]:
    source = Path(raw)
    candidates: list[Path] = []
    if source.is_absolute():
        candidates.append(source)
    else:
        candidates.append((PROJECT_ROOT / source).resolve())
        candidates.append((PROJECT_ROOT / "checkpoints" / source).resolve())
    return candidates


def resolve_local_source(raw: str) -> Path:
    checked: list[Path] = []
    for candidate in _candidate_sources(raw):
        checked.append(candidate)
        if candidate.exists():
            if candidate.is_file() and candidate.suffix == ".pt":
                return candidate.parent.resolve()
            return candidate.resolve()
    tried = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(f"Checkpoint source not found: {raw}. Checked: {tried}")


def checkpoint_relative_path(source: Path) -> Path:
    checkpoints_root = (PROJECT_ROOT / "checkpoints").resolve()
    try:
        return source.resolve().relative_to(checkpoints_root)
    except ValueError:
        return Path(source.name)


def remote_destination_parent(relative_path: Path) -> str:
    parent = relative_path.parent.as_posix()
    if parent == ".":
        return "checkpoints"
    return f"checkpoints/{parent}"


def send_checkpoint(worker, *, local_source: Path, relative_path: Path) -> None:
    remote_parent = remote_destination_parent(relative_path)
    remote_parent_abs = f"{worker.project_dir}/{remote_parent}"
    ssh(worker, f"mkdir -p {remote_parent_abs}", capture=True)
    rsync_to(
        worker,
        str(local_source),
        f"{remote_parent_abs}/",
        extra_args=["--progress"],
    )


def main() -> None:
    args = parse_args()
    local_source = resolve_local_source(args.source)
    if not local_source.is_dir():
        raise NotADirectoryError(
            f"Resolved source must be a directory after checkpoint expansion: {local_source}"
        )

    relative_path = checkpoint_relative_path(local_source)
    workers = load_workers(None if args.workers and "all" in args.workers else args.workers)

    print(
        f"Sending checkpoint {relative_path.as_posix()} to {len(workers)} worker(s): {[w.name for w in workers]}"
    )

    def send(worker):
        send_checkpoint(worker, local_source=local_source, relative_path=relative_path)
        return relative_path.as_posix()

    results = run_on_all(workers, send, desc="send checkpoint")
    sys.exit(0 if report_results(results) else 1)


if __name__ == "__main__":
    main()