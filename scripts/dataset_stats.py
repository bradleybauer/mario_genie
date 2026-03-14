#!/usr/bin/env python3
"""
Quick dataset statistics summary.

Usage:
    python scripts/dataset_stats.py data/nes
    python scripts/dataset_stats.py data/rgb
    python scripts/dataset_stats.py data/nes data/rgb
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mario_world_model.config import SEQUENCE_LENGTH


def gather_stats(data_dir: Path) -> dict:
    """Scan all session meta.json + npz files and return aggregate stats."""
    meta_files = sorted(data_dir.rglob("session_*.meta.json"))
    npz_files = sorted(data_dir.rglob("session_*.npz"))

    total_sessions = 0
    total_frames = 0
    action_counts: Counter[int] = Counter()
    level_counts: Counter[str] = Counter()
    frame_shape = None
    dtype = None
    disk_bytes = 0

    total_sequences = 0

    # Fast path: use meta.json when available
    for mf in meta_files:
        with open(mf) as f:
            meta = json.load(f)
        total_sessions += 1
        nf = meta.get("num_frames", 0)
        total_frames += nf
        if nf >= SEQUENCE_LENGTH:
            total_sequences += (nf - SEQUENCE_LENGTH) // SEQUENCE_LENGTH + 1
        for act, cnt in meta.get("action_summary", {}).items():
            action_counts[int(act)] += cnt
        for key, cnt in meta.get("exact_progression_summary", {}).items():
            parts = key.split(":")
            if len(parts) >= 2:
                level_counts[f"{parts[0]}-{parts[1]}"] += cnt

    # Fall back to npz if no meta files found
    if not meta_files:
        for npz_path in npz_files:
            npz = np.load(npz_path, mmap_mode="r")
            frames = npz["frames"]
            actions = npz["actions"]
            total_sessions += 1
            total_frames += frames.shape[0]
            if frames.shape[0] >= SEQUENCE_LENGTH:
                total_sequences += (frames.shape[0] - SEQUENCE_LENGTH) // SEQUENCE_LENGTH + 1
            if frame_shape is None:
                frame_shape = list(frames.shape)
                dtype = str(frames.dtype)
            for act, cnt in zip(*np.unique(actions, return_counts=True)):
                action_counts[int(act)] += int(cnt)
            if "world" in npz and "stage" in npz:
                for w, s in zip(npz["world"].flat, npz["stage"].flat):
                    level_counts[f"{w}-{s}"] += 1

    for f in npz_files:
        disk_bytes += f.stat().st_size

    return {
        "data_dir": str(data_dir),
        "num_sessions": total_sessions,
        "total_frames": total_frames,
        "total_sequences": total_sequences,
        "sequence_length": SEQUENCE_LENGTH,
        "frame_shape": frame_shape,
        "dtype": dtype,
        "disk_size_mb": round(disk_bytes / 1e6, 1),
        "action_counts": dict(sorted(action_counts.items())),
        "num_unique_actions": len(action_counts),
        "level_counts": dict(sorted(level_counts.items())),
        "num_unique_levels": len(level_counts),
    }


def print_stats(stats: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Dataset: {stats['data_dir']}")
    print(f"{'='*60}")
    print(f"  Sessions:        {stats['num_sessions']}")
    print(f"  Total frames:    {stats['total_frames']:,}")
    print(f"  Video sequences: {stats['total_sequences']:,}  (length {stats['sequence_length']})")
    if stats["frame_shape"]:
        shape = stats["frame_shape"]
        print(f"  Frame shape:     {shape}  (N, C, H, W)")
    if stats["dtype"]:
        print(f"  Dtype:           {stats['dtype']}")
    print(f"  Disk size:       {stats['disk_size_mb']} MB")

    if stats["action_counts"]:
        print(f"\n  Actions ({stats['num_unique_actions']} unique):")
        total_act = sum(stats["action_counts"].values())
        for act in sorted(stats["action_counts"]):
            cnt = stats["action_counts"][act]
            pct = 100 * cnt / total_act
            bar = "#" * int(pct / 2)
            print(f"    {act:>3d}: {cnt:>8,}  ({pct:5.1f}%)  {bar}")

    if stats["level_counts"]:
        print(f"\n  Levels ({stats['num_unique_levels']} unique):")
        total_lvl = sum(stats["level_counts"].values())
        for lvl in sorted(stats["level_counts"]):
            cnt = stats["level_counts"][lvl]
            pct = 100 * cnt / total_lvl
            print(f"    World {lvl}: {cnt:>8,}  ({pct:5.1f}%)")

    print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Quick dataset statistics summary.")
    parser.add_argument(
        "dirs",
        nargs="*",
        default=["data/nes"],
        help="One or more data directories to scan (default: data/nes)",
    )
    args = parser.parse_args()

    for d in [Path(p) for p in args.dirs]:
        if not d.exists():
            print(f"Warning: {d} does not exist, skipping.", file=sys.stderr)
            continue
        stats = gather_stats(d)
        print_stats(stats)


if __name__ == "__main__":
    main()
