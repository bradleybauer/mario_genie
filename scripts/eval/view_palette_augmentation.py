#!/usr/bin/env python3
"""Render examples of palette-index augmentation on normalized clips.

The output image stacks a few sampled clips. Within each sample block, the rows
are:
1. clean clip
2. augmented clip

Usage examples
--------------
  python scripts/eval/view_palette_augmentation.py
  python scripts/eval/view_palette_augmentation.py --palette-aug-prob 0.05
    python scripts/eval/view_palette_augmentation.py --palette-aug-sample-prob 0.5 --palette-aug-prob 0.1
  python scripts/eval/view_palette_augmentation.py --recording "some_recording.npz" --num-samples 3
    python scripts/eval/view_palette_augmentation.py --output /tmp/palette_aug.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.data.normalized_dataset import load_palette_tensor
from src.training.palette_video_vae_training import apply_palette_index_augmentation


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "normalized"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing normalized .npz files and palette metadata.",
    )
    parser.add_argument(
        "--recording",
        type=str,
        default=None,
        help="Optional .npz filename to sample clips from. Default: choose recordings at random.",
    )
    parser.add_argument(
        "--palette-aug-sample-prob",
        type=float,
        default=1.0,
        help="Per-sample probability of applying any palette augmentation at all.",
    )
    parser.add_argument(
        "--palette-aug-prob",
        type=float,
        default=0.25,
        help="Per-pixel probability of replacing a palette index once a sample is selected for augmentation.",
    )
    parser.add_argument(
        "--palette-aug-file",
        type=str,
        default="palette_distribution.json",
        help="Distribution JSON filename inside --data-dir used to sample replacement indices.",
    )
    parser.add_argument(
        "--clip-frames",
        type=int,
        default=6,
        help="Number of frames to show per sampled clip.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of sampled clips to include in the output sheet.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for recording/window selection and augmentation sampling.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Nearest-neighbour upscale factor applied to the saved image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the rendered augmentation sheet.",
    )
    args = parser.parse_args(argv)
    if not (0.0 <= args.palette_aug_sample_prob <= 1.0):
        parser.error("--palette-aug-sample-prob must be in [0, 1]")
    if not (0.0 <= args.palette_aug_prob <= 1.0):
        parser.error("--palette-aug-prob must be in [0, 1]")
    if args.clip_frames <= 0:
        parser.error("--clip-frames must be > 0")
    if args.num_samples <= 0:
        parser.error("--num-samples must be > 0")
    if args.scale <= 0:
        parser.error("--scale must be > 0")
    return args


def load_palette_probabilities(data_dir: Path, filename: str) -> torch.Tensor:
    dist_path = data_dir / filename
    if not dist_path.is_file():
        raise FileNotFoundError(f"Palette augmentation distribution file not found: {dist_path}")

    with dist_path.open() as handle:
        dist = json.load(handle)

    probs_data = dist.get("probabilities")
    counts_data = dist.get("counts")
    if probs_data is not None:
        probs = torch.tensor(probs_data, dtype=torch.float32)
    elif counts_data is not None:
        probs = torch.tensor(counts_data, dtype=torch.float32)
    else:
        raise ValueError(f"{dist_path} must contain either 'counts' or 'probabilities'")

    if probs.ndim != 1 or probs.numel() == 0:
        raise ValueError(f"{dist_path} must contain a non-empty 1D distribution")

    return probs / probs.sum().clamp_min(1e-12)


def choose_recording(data_dir: Path, recording_name: str | None, rng: np.random.Generator) -> Path:
    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No normalized .npz files found in {data_dir}")

    if recording_name is None:
        return npz_files[int(rng.integers(len(npz_files)))]

    candidate = data_dir / recording_name
    if candidate.is_file():
        return candidate

    matches = [path for path in npz_files if path.name == recording_name]
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Recording not found: {recording_name}")


def load_random_clip(
    recording_path: Path,
    *,
    clip_frames: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, int, int]:
    with np.load(recording_path) as data:
        if "frames" not in data:
            raise KeyError(f"{recording_path.name} is missing the 'frames' array")
        frames = np.asarray(data["frames"])

    total_frames = int(frames.shape[0])
    if total_frames < clip_frames:
        raise ValueError(
            f"{recording_path.name} has only {total_frames} frames, fewer than clip length {clip_frames}"
        )

    max_start = total_frames - clip_frames
    start_idx = int(rng.integers(max_start + 1)) if max_start > 0 else 0
    clip = torch.from_numpy(np.asarray(frames[start_idx:start_idx + clip_frames]).copy()).long().unsqueeze(0)
    return clip, start_idx, total_frames


def frames_to_rgb(frames: torch.Tensor, palette_rgb: np.ndarray) -> np.ndarray:
    frames_np = frames.squeeze(0).detach().cpu().numpy()
    return palette_rgb[frames_np]


def concatenate_frames(frames_rgb: np.ndarray, *, gap: int, fill_value: int) -> np.ndarray:
    tiles = [frame for frame in frames_rgb]
    if len(tiles) == 1:
        return tiles[0]
    channel_count = tiles[0].shape[2]
    separator = np.full((tiles[0].shape[0], gap, channel_count), fill_value, dtype=np.uint8)
    row = tiles[0]
    for tile in tiles[1:]:
        row = np.concatenate([row, separator, tile], axis=1)
    return row


def build_sample_panel(
    clean_frames: torch.Tensor,
    augmented_frames: torch.Tensor,
    *,
    palette_rgb: np.ndarray,
    frame_gap: int = 2,
    row_gap: int = 2,
) -> np.ndarray:
    clean_rgb = frames_to_rgb(clean_frames, palette_rgb)
    augmented_rgb = frames_to_rgb(augmented_frames, palette_rgb)

    clean_row = concatenate_frames(clean_rgb, gap=frame_gap, fill_value=24)
    augmented_row = concatenate_frames(augmented_rgb, gap=frame_gap, fill_value=24)

    spacer = np.full((row_gap, clean_row.shape[1], 3), 24, dtype=np.uint8)
    return np.concatenate([clean_row, spacer, augmented_row], axis=0)


def stack_sample_panels(panels: list[np.ndarray], *, gap: int = 8) -> np.ndarray:
    if not panels:
        raise ValueError("panels must not be empty")
    width = max(panel.shape[1] for panel in panels)
    padded: list[np.ndarray] = []
    for panel in panels:
        if panel.shape[1] < width:
            pad = np.full((panel.shape[0], width - panel.shape[1], 3), 24, dtype=np.uint8)
            panel = np.concatenate([panel, pad], axis=1)
        padded.append(panel)

    if len(padded) == 1:
        return padded[0]

    separator = np.full((gap, width, 3), 255, dtype=np.uint8)
    canvas = padded[0]
    for panel in padded[1:]:
        canvas = np.concatenate([canvas, separator, panel], axis=0)
    return canvas


def maybe_show(image: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    palette_tensor = load_palette_tensor(args.data_dir)
    palette_rgb = (palette_tensor.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    replacement_probs = load_palette_probabilities(args.data_dir, args.palette_aug_file)

    panels: list[np.ndarray] = []
    print("Output row order per sample: clean, augmented, changed-mask")
    for sample_idx in range(args.num_samples):
        recording_path = choose_recording(args.data_dir, args.recording, rng)
        clean_frames, start_idx, total_frames = load_random_clip(
            recording_path,
            clip_frames=args.clip_frames,
            rng=rng,
        )
        augmented_frames = apply_palette_index_augmentation(
            clean_frames,
            sample_prob=args.palette_aug_sample_prob,
            replacement_prob=args.palette_aug_prob,
            replacement_probs=replacement_probs,
        )
        changed_fraction = float((clean_frames != augmented_frames).float().mean().item())
        panels.append(
            build_sample_panel(
                clean_frames,
                augmented_frames,
                palette_rgb=palette_rgb,
            )
        )
        print(
            f"sample={sample_idx} recording={recording_path.name} frames={start_idx}:{start_idx + args.clip_frames} "
            f"total_frames={total_frames} changed={changed_fraction * 100.0:.2f}%"
        )

    image = stack_sample_panels(panels)
    if args.scale != 1:
        image = np.asarray(
            Image.fromarray(image).resize(
                (image.shape[1] * args.scale, image.shape[0] * args.scale),
                resample=Image.Resampling.NEAREST,
            )
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(args.output)
        print(f"Saved augmentation sheet to {args.output}")

    maybe_show(image)


if __name__ == "__main__":
    main()