#!/usr/bin/env python3
"""Show the distribution of palette indices across all normalized frames.

Usage examples
--------------
  # Basic distribution with bar chart
  python scripts/palette_distribution.py

  # Use a different data directory
  python scripts/palette_distribution.py --data-dir data/normalized

  # Show matplotlib plot
  python scripts/palette_distribution.py --plot

  # Show top-10 only
  python scripts/palette_distribution.py --top 10

  # Show per-file breakdown
  python scripts/palette_distribution.py --per-file
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Show palette index distribution across normalized frame data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir", type=str, default="data/normalized",
        help="Directory containing normalized .npz files and palette.json.",
    )
    p.add_argument(
        "--top", type=int, default=0,
        help="Show only the top N most frequent indices (0 = show all).",
    )
    p.add_argument(
        "--plot", action="store_true",
        help="Show a matplotlib bar chart of the distribution.",
    )
    p.add_argument(
        "--per-file", action="store_true",
        help="Print a per-file breakdown of index usage.",
    )
    p.add_argument(
        "--log-scale", action="store_true",
        help="Use log scale on the y-axis for --plot.",
    )
    return p.parse_args()


def load_palette_info(data_dir: Path) -> dict:
    palette_path = data_dir / "palette.json"
    if not palette_path.is_file():
        print(f"Error: {palette_path} not found.", file=sys.stderr)
        sys.exit(1)
    with palette_path.open() as f:
        return json.load(f)


def count_indices_in_file(path: Path, num_colors: int, chunk_size: int = 256) -> np.ndarray:
    """Return a histogram of palette indices for one .npz file.

    Streams frames from the compressed NPZ member in chunks to keep memory bounded.
    """
    counts = np.zeros(num_colors, dtype=np.int64)

    def _read_npy_header(fp):
        version = np.lib.format.read_magic(fp)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(fp)
        if version == (2, 0):
            return np.lib.format.read_array_header_2_0(fp)
        if version == (3, 0) and hasattr(np.lib.format, "read_array_header_3_0"):
            return np.lib.format.read_array_header_3_0(fp)
        raise ValueError(f"Unsupported .npy header version: {version}")

    with zipfile.ZipFile(path, "r") as zf:
        if "frames.npy" not in zf.namelist():
            raise KeyError(f"{path.name} is missing frames.npy")

        with zf.open("frames.npy", "r") as frames_fp:
            shape, _fortran_order, dtype = _read_npy_header(frames_fp)
            dtype = np.dtype(dtype)

            if dtype.hasobject:
                raise ValueError(f"{path.name} frames.npy uses object dtype")
            if len(shape) < 2:
                raise ValueError(f"{path.name} frames.npy shape is invalid: {shape}")

            n = int(shape[0])
            frame_elems = int(np.prod(shape[1:], dtype=np.int64))
            bytes_per_elem = dtype.itemsize

            for start in range(0, n, chunk_size):
                take = min(chunk_size, n - start)
                elems = take * frame_elems
                expected_bytes = elems * bytes_per_elem
                raw = frames_fp.read(expected_bytes)
                if len(raw) != expected_bytes:
                    raise ValueError(
                        f"Unexpected EOF while reading {path.name}: "
                        f"wanted {expected_bytes} bytes, got {len(raw)}"
                    )

                chunk = np.frombuffer(raw, dtype=dtype, count=elems)
                if np.issubdtype(dtype, np.unsignedinteger):
                    counts += np.bincount(chunk, minlength=num_colors)[:num_colors]
                else:
                    valid = chunk[(chunk >= 0) & (chunk < num_colors)]
                    counts += np.bincount(valid.astype(np.int64, copy=False), minlength=num_colors)
    return counts


def print_distribution(counts: np.ndarray, palette_rgb: list[list[int]],
                       top: int, label: str = "Global") -> None:
    total = counts.sum()
    if total == 0:
        print(f"{label}: no pixels found.")
        return

    num_colors = len(counts)
    used = int(np.count_nonzero(counts))
    print(f"{label} palette index distribution:")
    print(f"  Total pixels:   {total:,}")
    print(f"  Palette size:   {num_colors}")
    print(f"  Indices used:   {used} / {num_colors}")
    if used < num_colors:
        unused = sorted(np.where(counts == 0)[0].tolist())
        if len(unused) <= 20:
            print(f"  Unused indices: {unused}")
        else:
            print(f"  Unused indices: {unused[:10]} ... ({len(unused)} total)")
    print()

    order = np.argsort(-counts)
    n = top if top > 0 else num_colors
    max_bar = 50

    print(f"  {'Idx':>4}  {'RGB':>13}  {'Count':>14}  {'Pct':>7}  Bar")
    print(f"  {'---':>4}  {'---':>13}  {'---':>14}  {'---':>7}  ---")

    for rank, idx in enumerate(order[:n]):
        c = int(counts[idx])
        if c == 0:
            break
        pct = 100.0 * c / total
        r, g, b = palette_rgb[idx]
        rgb_str = f"({r:3d},{g:3d},{b:3d})"
        bar_len = int(pct / 100.0 * max_bar) if total > 0 else 0
        bar = "#" * max(bar_len, 1) if c > 0 else ""
        print(f"  {idx:4d}  {rgb_str:>13}  {c:>14,}  {pct:6.2f}%  {bar}")

    remaining = num_colors - n
    if remaining > 0 and top > 0:
        remaining_used = int(np.count_nonzero(counts[order[n:]]))
        print(f"  ... {remaining} more indices ({remaining_used} with nonzero counts)")
    print()


def plot_distribution(counts: np.ndarray, palette_rgb: list[list[int]],
                      log_scale: bool = False) -> None:
    num_colors = len(counts)
    total = counts.sum()
    pcts = 100.0 * counts / max(total, 1)

    # Color each bar by its actual palette RGB
    bar_colors = [
        to_hex([r / 255.0, g / 255.0, b / 255.0])
        for r, g, b in palette_rgb
    ]

    fig, ax = plt.subplots(figsize=(max(num_colors * 0.15, 10), 5))
    x = np.arange(num_colors)
    ax.bar(x, pcts, width=0.9, color=bar_colors, edgecolor="black", linewidth=0.3)

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Frequency (%,  log scale)")
    else:
        ax.set_ylabel("Frequency (%)")

    ax.set_xlabel("Palette Index")
    ax.set_title(f"Palette Index Distribution ({total:,} pixels, {num_colors} colors)")
    ax.set_xlim(-0.5, num_colors - 0.5)

    # Only label every Nth tick to avoid clutter
    tick_step = max(1, num_colors // 20)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels(x[::tick_step], fontsize=7)

    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    palette_info = load_palette_info(data_dir)
    palette_rgb: list[list[int]] = palette_info["colors_rgb"]
    num_colors = len(palette_rgb)

    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    console.print(f"Scanning {len(npz_files)} files in {data_dir} ...")
    global_counts = np.zeros(num_colors, dtype=np.int64)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing files", total=len(npz_files))
        for path in npz_files:
            progress.update(task, description=f"[bold blue]{path.name}")
            file_counts = count_indices_in_file(path, num_colors)
            global_counts += file_counts

            if args.per_file:
                total = file_counts.sum()
                used = int(np.count_nonzero(file_counts))
                top_idx = int(np.argmax(file_counts))
                top_pct = 100.0 * file_counts[top_idx] / max(total, 1)
                progress.console.print(
                    f"  {path.name}: {total:,} px, "
                    f"{used}/{num_colors} used, "
                    f"top={top_idx} ({top_pct:.1f}%)"
                )
            progress.advance(task)

    if args.per_file:
        print()

    print_distribution(global_counts, palette_rgb, args.top)

    if args.plot:
        plot_distribution(global_counts, palette_rgb, log_scale=args.log_scale)


if __name__ == "__main__":
    main()
