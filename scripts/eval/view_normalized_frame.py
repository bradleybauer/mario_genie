#!/usr/bin/env python3
"""Load a random normalized frame and visualize it.

By default this selects a random recording from data/normalized/, then a
random frame within that recording, reconstructs RGB pixels using
palette.json, and opens a matplotlib window.

Usage:
    python scripts/view_normalized_frame.py
    python scripts/view_normalized_frame.py --seed 0
    python scripts/view_normalized_frame.py --recording "some_recording.npz"
    python scripts/view_normalized_frame.py --frame-index 12
    python scripts/view_normalized_frame.py --save /tmp/frame.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
plt.style.use("dark_background")
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "normalized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing normalized .npz files and palette.json.",
    )
    parser.add_argument(
        "--recording",
        type=str,
        default=None,
        help="Specific .npz filename to use. Default: choose one at random.",
    )
    return parser.parse_args()


def load_palette(data_dir: Path) -> np.ndarray:
    palette_path = data_dir / "palette.json"
    with open(palette_path) as f:
        palette_info = json.load(f)
    colors = np.asarray(palette_info["colors_rgb"], dtype=np.uint8)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"Invalid palette RGB data in {palette_path}")
    return colors


def choose_recording(data_dir: Path, recording_name: str | None, rng: np.random.Generator) -> Path:
    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No normalized .npz files found in {data_dir}")
    if recording_name is None:
        return npz_files[int(rng.integers(len(npz_files)))]

    recording_path = data_dir / recording_name
    if recording_path.exists():
        return recording_path

    matches = [path for path in npz_files if path.name == recording_name]
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Recording not found: {recording_name}")


def choose_frame(num_frames: int, frame_index: int | None, rng: np.random.Generator) -> int:
    if num_frames <= 0:
        raise ValueError("Recording has no frames")
    if frame_index is None:
        return int(rng.integers(num_frames))
    if not 0 <= frame_index < num_frames:
        raise IndexError(f"frame_index must be in [0, {num_frames - 1}], got {frame_index}")
    return frame_index


def main() -> None:
    args = parse_args()

    matplotlib.use("TkAgg")

    rng = np.random.default_rng()
    data_dir = args.data_dir
    palette = load_palette(data_dir)
    recording_path = choose_recording(data_dir, args.recording, rng)

    with np.load(recording_path) as data:
        frames = data["frames"]
        actions = data["actions"] if "actions" in data else None
        frame_idx = choose_frame(len(frames), None, rng)
        frame = frames[frame_idx]
        rgb = palette[frame]
        action_value = int(actions[frame_idx]) if actions is not None else None

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb, interpolation="nearest")
    plt.axis("off")

    title = f"{recording_path.name} | frame {frame_idx}"
    if action_value is not None:
        title += f" | action {action_value}"
    plt.title(title)
    plt.tight_layout()

    print(f"Recording: {recording_path.name}")
    print(f"Frame index: {frame_idx}")
    if action_value is not None:
        print(f"Reduced action index: {action_value}")

    plt.show()


if __name__ == "__main__":
    main()