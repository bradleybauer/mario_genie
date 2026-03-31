#!/usr/bin/env python3
"""Load a normalized frame and overlay a patchify grid on top of it.

By default this selects a random recording from data/normalized/, then a
random frame within that recording, reconstructs RGB pixels using
palette.json, and draws the patchify grid over the image.

Usage:
    python scripts/view_patchify_grid.py
    python scripts/view_patchify_grid.py --seed 0 --patch-size 16
    python scripts/view_patchify_grid.py --recording "some_recording.npz" --frame-index 12
    python scripts/view_patchify_grid.py --patch-size 8 --save /tmp/frame_grid.png --no-show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Frame index to visualize. Default: choose one at random.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used when recording or frame index is not specified.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=4,
        help="Square patch size to overlay. The frame dimensions must be divisible by this value.",
    )
    parser.add_argument(
        "--grid-color",
        type=str,
        default="white",
        help="Matplotlib color for grid lines.",
    )
    parser.add_argument(
        "--grid-linewidth",
        type=float,
        default=0.8,
        help="Line width used for grid lines.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output path for the rendered image.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Render without opening an interactive window.",
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


def validate_patch_size(frame: np.ndarray, patch_size: int) -> tuple[int, int]:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    height, width = frame.shape[:2]
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Frame shape {height}x{width} is not divisible by patch_size={patch_size}"
        )
    return height // patch_size, width // patch_size


def draw_patch_grid(
    rgb: np.ndarray,
    patch_size: int,
    grid_color: str,
    grid_linewidth: float,
    title: str,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_axis_off()

    height, width = rgb.shape[:2]
    x_positions = np.arange(-0.5, width, patch_size)
    y_positions = np.arange(-0.5, height, patch_size)

    for x in x_positions:
        ax.axvline(x=x, color=grid_color, linewidth=grid_linewidth, alpha=0.9)
    ax.axvline(x=width - 0.5, color=grid_color, linewidth=grid_linewidth, alpha=0.9)

    for y in y_positions:
        ax.axhline(y=y, color=grid_color, linewidth=grid_linewidth, alpha=0.9)
    ax.axhline(y=height - 0.5, color=grid_color, linewidth=grid_linewidth, alpha=0.9)

    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def main() -> None:
    args = parse_args()

    if args.no_show:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")

    rng = np.random.default_rng(args.seed)
    data_dir = args.data_dir
    palette = load_palette(data_dir)
    recording_path = choose_recording(data_dir, args.recording, rng)

    with np.load(recording_path) as data:
        frames = data["frames"]
        actions = data["actions"] if "actions" in data else None
        frame_idx = choose_frame(len(frames), args.frame_index, rng)
        frame = frames[frame_idx]
        rgb = palette[frame]
        action_value = int(actions[frame_idx]) if actions is not None else None

    patch_rows, patch_cols = validate_patch_size(frame, args.patch_size)

    title = (
        f"{recording_path.name} | frame {frame_idx} | "
        f"patch {args.patch_size} ({patch_rows}x{patch_cols})"
    )
    if action_value is not None:
        title += f" | action {action_value}"

    fig, _ = draw_patch_grid(
        rgb=rgb,
        patch_size=args.patch_size,
        grid_color=args.grid_color,
        grid_linewidth=args.grid_linewidth,
        title=title,
    )

    print(f"Recording: {recording_path.name}")
    print(f"Frame index: {frame_idx}")
    print(f"Patch size: {args.patch_size}")
    print(f"Patch grid: {patch_rows} rows x {patch_cols} cols")
    if action_value is not None:
        print(f"Reduced action index: {action_value}")

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {args.save}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()