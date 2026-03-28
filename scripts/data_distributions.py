#!/usr/bin/env python3
"""Explore data distributions over arbitrary columns in the recording database.

Usage examples
--------------
  # Histogram of SMB1 player state byte
  python scripts/data_distributions.py ram_000e

  # Joint distribution of two columns
  python scripts/data_distributions.py ram_0756 ram_000e

  # Distribution of action bitmask
  python scripts/data_distributions.py action

  # Filter to a specific recording and show top-20
  python scripts/data_distributions.py ram_075f --recording 0 --top 20

  # Show all unique values (no histogram, just value counts)
  python scripts/data_distributions.py ram_075c --counts

  # List available columns
  python scripts/data_distributions.py --list-columns

  # Database overview
  python scripts/data_distributions.py --info
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.npy_db import load_recordings, build_dataframe, SMB1_RAM_LABELS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Show data distributions from the recording database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "columns", nargs="*",
        help="Column name(s) to analyse (e.g. ram_000e, action, frame_number).",
    )
    p.add_argument(
        "--info", action="store_true",
        help="Print database summary and exit.",
    )
    p.add_argument(
        "--list-columns", action="store_true",
        help="List all available frame columns and exit.",
    )
    p.add_argument(
        "--recording", type=int, default=None,
        help="Filter to a specific recording_id.",
    )
    p.add_argument(
        "--top", type=int, default=30,
        help="Show top N most frequent values (default: 30).",
    )
    p.add_argument(
        "--counts", action="store_true",
        help="Print full value counts (no histogram bars).",
    )
    p.add_argument(
        "--plot", action="store_true",
        help="Show a matplotlib histogram.",
    )
    p.add_argument(
        "--bins", type=int, default=50,
        help="Number of histogram bins for --plot (default: 50).",
    )
    p.add_argument(
        "--data-dir", type=Path, default=None,
        help="Override Mesen2 data directory.",
    )
    p.add_argument(
        "--progression", action="store_true",
        help="Show progression coverage plot: frame density across "
             "(world, stage, x_position) colored by level.",
    )
    p.add_argument(
        "--sort", choices=["value", "freq"], default=None,
        help="Sort output by value (lexicographic) or frequency. "
             "Default: value for numeric columns, freq otherwise.",
    )
    p.add_argument(
        "--ram-only", nargs="*", type=str, default=None, metavar="ADDR",
        help="Only load specific RAM addresses (hex, e.g. 000e 0756). "
             "Speeds up loading when you know which columns you need.",
    )
    return p.parse_args()


def _sort_by_value(series) -> bool:
    """Return True if the series should default to value-sorted output."""
    return series.dtype.kind in ("i", "u", "f")


def print_value_counts(series, top: int, show_bars: bool = True, sort: str | None = None):
    """Print a frequency table for a pandas Series."""
    total = len(series)
    nunique = series.nunique()

    print(f"  Total rows:    {total:,}")
    print(f"  Unique values: {nunique}")
    print(f"  Min: {series.min()}  Max: {series.max()}")
    if hasattr(series, "mean"):
        try:
            print(f"  Mean: {series.mean():.2f}  Std: {series.std():.2f}")
        except TypeError:
            pass
    print()

    use_value = (sort == "value") if sort else _sort_by_value(series)
    if use_value:
        vc_freq = series.value_counts().sort_index().head(top)
    else:
        vc_freq = series.value_counts().head(top)
    max_val_width = max(len(str(v)) for v in vc_freq.index)
    max_cnt_width = max(len(f"{c:,}") for c in vc_freq.values)

    for val, cnt in vc_freq.items():
        pct = 100 * cnt / total
        if show_bars:
            bar_len = int(pct / 2)
            bar = "#" * bar_len
            print(f"  {str(val):>{max_val_width}}: {cnt:>{max_cnt_width},}  ({pct:5.1f}%)  {bar}")
        else:
            print(f"  {str(val):>{max_val_width}}: {cnt:>{max_cnt_width},}  ({pct:5.1f}%)")

    if nunique > top:
        print(f"  ... and {nunique - top} more values")


def plot_progression(df):
    """Plot frame density across game progression (world-stage-x), colored by level."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # Columns needed: ram_075f (world), ram_075c (stage), ram_006d (x_page)
    for col in ("ram_075f", "ram_075c", "ram_006d"):
        if col not in df.columns:
            print(f"Error: column '{col}' not found. "
                  "Progression plot needs world/stage/x_page RAM addresses.",
                  file=sys.stderr)
            sys.exit(1)

    world = df["ram_075f"].values
    stage = df["ram_075c"].values
    x_page = df["ram_006d"].values

    # Build (world, stage, x_page) tuples and count
    progression = list(zip(world.tolist(), stage.tolist(), x_page.tolist()))
    from collections import Counter
    counts = Counter(progression)

    # Sort bins by (world, stage, x_page)
    bins_sorted = sorted(counts.keys())
    bin_counts = [counts[b] for b in bins_sorted]
    total = sum(bin_counts)
    densities = [c / total for c in bin_counts]

    # Color by level (world, stage)
    levels_ordered = sorted({(w, s) for w, s, _ in bins_sorted})
    cmap = plt.get_cmap("tab20", max(len(levels_ordered), 1))
    level_to_colour = {lvl: cmap(i) for i, lvl in enumerate(levels_ordered)}
    colours = [level_to_colour[(w, s)] for w, s, _ in bins_sorted]

    # Uniform reference
    num_bins = len(bins_sorted)
    uniform = 1.0 / num_bins if num_bins else 0
    y_max = max(densities) * 1.15 if densities else 1

    # Split across two rows
    n = len(bins_sorted)
    mid = (n + 1) // 2
    row_slices = [slice(0, mid), slice(mid, n)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    for row_idx, sl in enumerate(row_slices):
        ax = axes[row_idx]
        row_bins = bins_sorted[sl]
        row_densities = densities[sl]
        row_colours = colours[sl]
        x = np.arange(len(row_bins))

        if len(row_bins) == 0:
            ax.set_visible(False)
            continue

        ax.bar(x, row_densities, width=1.0, color=row_colours,
               edgecolor="none", linewidth=0)
        ax.axhline(uniform, color="red", linewidth=1.2, linestyle="--",
                   label=f"uniform = {uniform:.5f}" if row_idx == 0 else None)

        # Vertical separators between levels
        prev_level = None
        for i, (w, s, _) in enumerate(row_bins):
            lvl = (w, s)
            if prev_level is not None and lvl != prev_level:
                ax.axvline(i - 0.5, color="grey", linewidth=0.4, alpha=0.5)
            prev_level = lvl

        # X-axis: show world-stage labels at midpoint of each level's span
        level_spans: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, (w, s, _) in enumerate(row_bins):
            level_spans[(w, s)].append(i)
        tick_positions = []
        tick_labels = []
        for lvl in levels_ordered:
            if lvl not in level_spans:
                continue
            indices = level_spans[lvl]
            tick_positions.append(indices[len(indices) // 2])
            tick_labels.append(f"{lvl[0]+1}-{lvl[1]+1}")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")
        ax.set_xlim(-0.5, len(row_bins) - 0.5)
        ax.set_ylim(0, y_max)
        ax.set_ylabel("Density")

    axes[0].set_title(
        f"Progression Coverage — {total:,} frames across "
        f"{num_bins} (world, stage, x_page) bins"
    )
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("World-Stage")

    # Text summary
    print(f"Progression coverage: {total:,} frames, {num_bins} bins, "
          f"{len(levels_ordered)} unique levels")
    for lvl in levels_ordered:
        lvl_frames = sum(c for (w, s, _), c in counts.items() if (w, s) == lvl)
        pct = 100 * lvl_frames / total
        print(f"  World {lvl[0]+1}-{lvl[1]+1}: {lvl_frames:>8,} frames ({pct:5.1f}%)")

    plt.tight_layout()
    plt.show()


def print_joint_distribution(df, col_a: str, col_b: str, top: int, sort: str | None = None):
    """Print a cross-tabulation of two columns."""
    total = len(df)
    grouped = df.groupby([col_a, col_b]).size().reset_index(name="count")

    both_numeric = df[col_a].dtype.kind in ("i", "u", "f") and df[col_b].dtype.kind in ("i", "u", "f")
    use_value = (sort == "value") if sort else both_numeric
    if use_value:
        grouped = grouped.sort_values([col_a, col_b]).head(top)
    else:
        grouped = grouped.sort_values("count", ascending=False).head(top)

    print(f"  Total rows:  {total:,}")
    print(f"  Unique pairs: {df.groupby([col_a, col_b]).ngroups}")
    print()

    max_a = max(len(str(v)) for v in grouped[col_a])
    max_b = max(len(str(v)) for v in grouped[col_b])
    max_c = max(len(f"{c:,}") for c in grouped["count"])
    max_a = max(max_a, len(col_a))
    max_b = max(max_b, len(col_b))

    print(f"  {col_a:>{max_a}}  {col_b:>{max_b}}  {'count':>{max_c}}   pct")
    print(f"  {'-'*max_a}  {'-'*max_b}  {'-'*max_c}  -----")

    for _, row in grouped.iterrows():
        pct = 100 * row["count"] / total
        bar = "#" * int(pct / 2)
        print(
            f"  {str(row[col_a]):>{max_a}}  {str(row[col_b]):>{max_b}}  "
            f"{row['count']:>{max_c},}  ({pct:5.1f}%)  {bar}"
        )


def main():
    args = parse_args()

    # --- Load recordings (numpy) ---
    recordings = load_recordings(data_dir=args.data_dir)
    if not recordings:
        d = args.data_dir or "data/"
        print(f"No recordings (*.ram.npy) in {d}", file=sys.stderr)
        sys.exit(1)

    # --- --info: summary from raw arrays, no DataFrame needed ---
    if args.info:
        total_frames = sum(len(r.frames) for r in recordings)
        total_states = sum(len(r.save_states) for r in recordings)
        ram_size = recordings[0].ram.shape[1]
        wram_size = recordings[0].wram.shape[1] if recordings[0].wram is not None else 0
        print(f"Recordings: {len(recordings)}")
        for i, rec in enumerate(recordings):
            print(f"  [{i}] {rec.name}  ({len(rec.frames):,} frames)")
        print(f"Total frames: {total_frames:,}")
        print(f"Save states:  {total_states}")
        print(f"RAM size:     {ram_size}")
        print(f"WRAM size:    {wram_size}")
        return

    # --- --list-columns: show available column names ---
    if args.list_columns:
        ram_size = recordings[0].ram.shape[1]
        wram_size = recordings[0].wram.shape[1] if recordings[0].wram is not None else 0

        print("Frame columns (non-memory):")
        for c in ("recording_id", "frame_number", "action"):
            print(f"  {c}")

        if ram_size:
            print(f"\nRAM columns: {ram_size} (ram_0000 … ram_{ram_size-1:04x})")
            print("\nSemantically labeled RAM addresses:")
            print(f"  {'Column':<14} {'Label':<22} Description")
            print(f"  {'------':<14} {'-----':<22} -----------")
            for addr in sorted(SMB1_RAM_LABELS.keys()):
                if addr < ram_size:
                    label, desc = SMB1_RAM_LABELS[addr]
                    print(f"  ram_{addr:04x}       {label:<22} {desc}")

        if wram_size:
            print(f"\nWRAM columns: {wram_size} (wram_0000 … wram_{wram_size-1:04x})")
        else:
            print("\nWRAM columns: 0 (no WRAM in these recordings)")
        return

    # --- Determine which RAM columns to expand into the DataFrame ---
    ram_columns = None
    _PROGRESSION_ADDRS = [0x075F, 0x075C, 0x006D]

    if args.ram_only is not None:
        ram_columns = [int(a, 16) for a in args.ram_only]
    elif args.progression:
        ram_columns = list(_PROGRESSION_ADDRS)
    elif args.columns:
        needed = []
        any_ram_requested = False
        for c in args.columns:
            if c.startswith("ram_") or c.startswith("wram_"):
                any_ram_requested = True
            if c.startswith("ram_"):
                try:
                    needed.append(int(c[4:], 16))
                except ValueError:
                    pass
        if all(c.startswith(("ram_", "wram_")) or c in ("action", "frame_number", "recording_id") for c in args.columns):
            ram_columns = needed if any_ram_requested else []

    df = build_dataframe(recordings, ram_columns=ram_columns)

    n_frames = len(df)
    ram_cols = sum(1 for c in df.columns if c.startswith("ram_"))
    wram_cols = sum(1 for c in df.columns if c.startswith("wram_"))
    mb = df.memory_usage(deep=True).sum() / 1e6
    print(
        f"Built DataFrame: {len(recordings)} recordings, {n_frames:,} frames "
        f"({ram_cols} RAM + {wram_cols} WRAM cols, {mb:.0f} MB)",
        file=sys.stderr,
    )

    if args.progression:
        if args.recording is not None:
            df = df[df["recording_id"] == args.recording].copy()
            print(f"Filtered to recording_id={args.recording}: {len(df):,} frames\n")
        plot_progression(df)
        return

    if not args.columns:
        print("No columns specified. Use --info, --list-columns, or pass column names.")
        print("Examples:  ram_000e  action  frame_number")
        sys.exit(1)

    if args.recording is not None:
        df = df[df["recording_id"] == args.recording].copy()
        print(f"Filtered to recording_id={args.recording}: {len(df):,} frames\n")

    for col in args.columns:
        if col not in df.columns:
            print(f"Error: column '{col}' not found. Use --list-columns to see available.", file=sys.stderr)
            sys.exit(1)

    # Print semantic labels for any RAM/WRAM columns being queried
    labeled = []
    for col in args.columns:
        if col.startswith("ram_"):
            try:
                addr = int(col[4:], 16)
            except ValueError:
                continue
            if addr in SMB1_RAM_LABELS:
                label, desc = SMB1_RAM_LABELS[addr]
                labeled.append((col, label, desc))
    if labeled:
        for col, label, desc in labeled:
            print(f"  {col}  →  {label}: {desc}")
        print()

    if len(args.columns) == 1:
        col = args.columns[0]
        print(f"Distribution of '{col}':")
        print_value_counts(df[col], top=args.top, show_bars=not args.counts, sort=args.sort)

    elif len(args.columns) == 2:
        col_a, col_b = args.columns
        print(f"Joint distribution of '{col_a}' × '{col_b}':")
        print_joint_distribution(df, col_a, col_b, top=args.top, sort=args.sort)

    else:
        # 3+ columns: show each individually
        for col in args.columns:
            print(f"\n{'='*50}")
            print(f"Distribution of '{col}':")
            print_value_counts(df[col], top=args.top, show_bars=not args.counts, sort=args.sort)

    # Optional plot
    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(args.columns), figsize=(6 * len(args.columns), 4))
        if len(args.columns) == 1:
            axes = [axes]

        for ax, col in zip(axes, args.columns):
            data = df[col].dropna()
            if data.nunique() <= args.bins:
                vc = data.value_counts().sort_index()
                ax.bar(vc.index.astype(str), vc.values)
            else:
                ax.hist(data, bins=args.bins, edgecolor="black", linewidth=0.5)
            ax.set_title(col)
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
