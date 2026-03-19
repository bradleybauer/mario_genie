#!/usr/bin/env python3
"""Count how often each NES palette colour appears across the entire dataset.

Usage:
    python scripts/palette_frequency.py data/nes/
    python scripts/palette_frequency.py data/ --sort
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

CHUNK = 1024  # frames per chunk to cap memory per worker


def _count_file(npz_path: str) -> tuple[np.ndarray, int]:
    """Return (counts_256, total_pixels) for one .npz session."""
    counts = np.zeros(256, dtype=np.int64)
    with np.load(npz_path) as data:
        frames = data["frames"]
        pixels = frames.size
        n = frames.shape[0]
        for start in range(0, n, CHUNK):
            chunk = frames[start : start + CHUNK].ravel()
            counts += np.bincount(chunk, minlength=256)
    return counts, pixels


def main() -> None:
    parser = argparse.ArgumentParser(description="Count palette colour frequencies across all sessions.")
    parser.add_argument("data_dir", type=Path, help="Directory containing session .npz files")
    parser.add_argument("--sort", action="store_true", help="Sort by frequency (descending)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes")
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()

    # Load palette for display
    palette = None
    palette_path = data_dir / "palette.json"
    if palette_path.exists():
        with palette_path.open("r") as f:
            palette = json.load(f)

    # Gather all session files (search recursively)
    npz_files = sorted(str(p) for p in data_dir.rglob("session_*.npz"))
    if not npz_files:
        print(f"No session .npz files found in {data_dir}")
        return

    n_workers = min(args.workers or 1, len(npz_files))
    print(f"Processing {len(npz_files)} sessions from {data_dir} with {n_workers} workers ...")

    total_counts = np.zeros(256, dtype=np.int64)
    total_pixels = 0
    done = 0

    with mp.Pool(n_workers) as pool:
        for counts, pixels in pool.imap_unordered(_count_file, npz_files, chunksize=8):
            total_counts += counts
            total_pixels += pixels
            done += 1
            if done % 100 == 0 or done == len(npz_files):
                print(f"  {done}/{len(npz_files)} sessions processed")

    # Trim trailing zeros
    last_nonzero = np.max(np.nonzero(total_counts)) + 1
    total_counts = total_counts[:last_nonzero]

    # Build results
    indices = np.arange(len(total_counts))
    if args.sort:
        order = np.argsort(-total_counts)
    else:
        order = indices

    print(f"\nTotal pixels: {total_pixels:,}")
    print(f"{'Idx':>4}  {'Count':>14}  {'%':>7}  RGB")
    print("-" * 50)
    for idx in order:
        count = total_counts[idx]
        if count == 0:
            continue
        pct = 100.0 * count / total_pixels
        rgb_str = f"{palette[idx]}" if palette and idx < len(palette) else "?"
        print(f"{idx:>4}  {count:>14,}  {pct:>6.2f}%  {rgb_str}")


if __name__ == "__main__":
    main()
