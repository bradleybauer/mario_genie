#!/usr/bin/env python3
"""Trim leading frames from GIF files.

Usage:
    python scripts/eval/trim_gifs.py previews_rollout_step_*.gif
    python scripts/eval/trim_gifs.py . --frames 200 --overwrite
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image, ImageSequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="GIF files, directories, or glob patterns to process.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of leading frames to remove from each GIF.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the original GIFs instead of writing *_trimmed.gif files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when an input is a directory or glob.",
    )
    return parser.parse_args()


def resolve_inputs(values: list[str], recursive: bool) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()

    for value in values:
        raw_path = Path(value)
        if any(char in value for char in "*?[]"):
            matches = [Path(match) for match in glob.glob(value, recursive=recursive)]
        elif raw_path.is_dir():
            pattern = "**/*.gif" if recursive else "*.gif"
            matches = sorted(raw_path.glob(pattern))
        else:
            matches = [raw_path]

        for match in matches:
            if not match.is_file() or match.suffix.lower() != ".gif":
                continue
            resolved = match.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)

    return sorted(paths)


def output_path_for(path: Path, overwrite: bool) -> Path:
    if overwrite:
        return path
    return path.with_name(f"{path.stem}_trimmed{path.suffix}")


def load_trimmed_frames(path: Path, frames_to_drop: int) -> tuple[list[Image.Image], list[int], int, int]:
    if frames_to_drop < 0:
        raise ValueError("--frames must be non-negative")

    with Image.open(path) as source:
        total_frames = getattr(source, "n_frames", 1)
        if frames_to_drop >= total_frames:
            raise ValueError(
                f"cannot drop {frames_to_drop} frames from a GIF with {total_frames} frames"
            )

        default_duration = int(source.info.get("duration", 100))
        loop = int(source.info.get("loop", 0))
        trimmed_frames: list[Image.Image] = []
        durations: list[int] = []

        for index, frame in enumerate(ImageSequence.Iterator(source)):
            if index < frames_to_drop:
                continue
            trimmed_frames.append(frame.convert("RGBA"))
            durations.append(int(frame.info.get("duration", default_duration)))

    return trimmed_frames, durations, total_frames, loop


def save_gif(path: Path, frames: list[Image.Image], durations: list[int], loop: int) -> None:
    first_frame, *remaining_frames = frames
    first_frame.save(
        path,
        save_all=True,
        append_images=remaining_frames,
        duration=durations,
        loop=loop,
        disposal=2,
        optimize=False,
    )


def trim_gif(path: Path, frames_to_drop: int, overwrite: bool) -> tuple[Path, int, int]:
    trimmed_frames, durations, total_frames, loop = load_trimmed_frames(path, frames_to_drop)
    destination = output_path_for(path, overwrite)

    if overwrite:
        with NamedTemporaryFile(prefix=path.stem + "_", suffix=path.suffix, dir=path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            save_gif(temp_path, trimmed_frames, durations, loop)
            temp_path.replace(path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        return path, total_frames, len(trimmed_frames)

    save_gif(destination, trimmed_frames, durations, loop)
    return destination, total_frames, len(trimmed_frames)


def main() -> int:
    args = parse_args()
    gif_paths = resolve_inputs(args.inputs, recursive=args.recursive)
    if not gif_paths:
        print("No GIF files matched the provided inputs.", file=sys.stderr)
        return 1

    exit_code = 0
    for gif_path in gif_paths:
        try:
            destination, original_count, trimmed_count = trim_gif(
                gif_path,
                frames_to_drop=args.frames,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            exit_code = 1
            print(f"Failed: {gif_path} ({exc})", file=sys.stderr)
            continue

        print(f"Trimmed {gif_path} -> {destination} ({original_count} -> {trimmed_count} frames)")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())