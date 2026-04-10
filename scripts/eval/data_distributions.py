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
from collections import Counter
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
plt.style.use("dark_background")
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.data.npy_db import load_recordings, build_dataframe
from src.data.smb1_memory_map import SMB1_RAM_LABELS


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
        "--x-bins", action="store_true",
        help="Show distribution of (world, stage, x_bin) tuples "
             "matching targeted_collect.py binning.",
    )
    p.add_argument(
        "--sample", type=str, default=None, metavar="VALUE",
        help="Show random frames matching this value in the current mode. "
             "Combine with --x-bins or a column name (e.g. --x-bins --sample 6016).",
    )
    p.add_argument(
        "--sample-count", type=int, default=9,
        help="Number of sample frames to display (default: 9).",
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
    p.add_argument(
        "--bin-width", type=int, default=128,
        help="X-position bin width in pixels for --progression (default: 128).",
    )
    return p.parse_args()


_NES_BUTTONS = [
    (0, "Up"), (1, "Down"), (2, "Left"), (3, "Right"),
    (4, "Start"), (5, "Select"), (6, "B"), (7, "A"),
]


def action_label(byte_val: int) -> str:
    """Convert a NES controller bitmask to a human-readable button string."""
    parts = [name for bit, name in _NES_BUTTONS if byte_val & (1 << bit)]
    return "+".join(parts) if parts else "None"


def _sort_by_value(series) -> bool:
    """Return True if the series should default to value-sorted output."""
    return series.dtype.kind in ("i", "u", "f")


def print_value_counts(series, top: int, show_bars: bool = True,
                       sort: str | None = None, label_fn=None):
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

    if nunique <= 255:
        top = nunique

    use_value = (sort == "value") if sort else _sort_by_value(series)
    if use_value:
        vc_freq = series.value_counts().sort_index().head(top)
    else:
        vc_freq = series.value_counts().head(top)

    def _fmt(v):
        s = str(v)
        if label_fn is not None:
            return f"{s} ({label_fn(v)})"
        return s

    labels = [_fmt(v) for v in vc_freq.index]
    max_val_width = max(len(l) for l in labels)
    max_cnt_width = max(len(f"{c:,}") for c in vc_freq.values)

    for label, (val, cnt) in zip(labels, vc_freq.items()):
        pct = 100 * cnt / total
        if show_bars:
            bar_len = int(pct / 2)
            bar = "#" * bar_len
            print(f"  {label:>{max_val_width}}: {cnt:>{max_cnt_width},}  ({pct:5.1f}%)  {bar}")
        else:
            print(f"  {label:>{max_val_width}}: {cnt:>{max_cnt_width},}  ({pct:5.1f}%)")

    if nunique > top:
        print(f"  ... and {nunique - top} more values")


def plot_progression(df, bin_width: int = 128):
    """Plot frame density across game progression (world-stage-x_bin), colored by level."""
    # Columns needed: ram_075f (world), ram_075c (stage), ram_006d (x_page), ram_0086 (x_screen)
    for col in ("ram_075f", "ram_075c", "ram_006d", "ram_0086"):
        if col not in df.columns:
            print(f"Error: column '{col}' not found. "
                  "Progression plot needs world/stage/x_page/x_screen RAM addresses.",
                  file=sys.stderr)
            sys.exit(1)

    world = df["ram_075f"].values
    stage = df["ram_075c"].values
    x_abs = df["ram_006d"].values.astype(int) * 256 + df["ram_0086"].values.astype(int)
    x_bin = (x_abs // bin_width) * bin_width

    # Build (world, stage, x_bin) tuples and count
    progression = list(zip(world.tolist(), stage.tolist(), x_bin.tolist()))
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
        f"{num_bins} (world, stage, x_bin[{bin_width}px]) bins"
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

    # --- Per-powerup figures: same two-row layout, one figure each ---
    _POWERUP_NAMES = {0: "Small", 1: "Big", 2: "Fire"}
    _POWERUP_COLOURS = {0: "#4dabf7", 1: "#51cf66", 2: "#ff6b6b"}

    if "ram_0756" not in df.columns:
        print("Warning: ram_0756 (powerup_state) not available, "
              "skipping powerup breakdown.", file=sys.stderr)
        plt.show()
        return

    powerup = df["ram_0756"].values
    powerup_ids = sorted({int(v) for v in np.unique(powerup)})

    # Count per (world, stage, x_bin, powerup)
    prog_pw = list(zip(world.tolist(), stage.tolist(),
                       x_bin.tolist(), powerup.tolist()))
    counts_pw = Counter(prog_pw)

    for pw_id in powerup_ids:
        pw_name = _POWERUP_NAMES.get(pw_id, f"pw={pw_id}")
        pw_colour = _POWERUP_COLOURS.get(pw_id, cmap(0))
        pw_total = int(np.sum(powerup == pw_id))

        fig_pw, axes_pw = plt.subplots(2, 1, figsize=(14, 7))

        for row_idx, sl in enumerate(row_slices):
            ax = axes_pw[row_idx]
            row_bins = bins_sorted[sl]
            x = np.arange(len(row_bins))

            if len(row_bins) == 0:
                ax.set_visible(False)
                continue

            heights = np.array(
                [counts_pw.get((w, s, xp, pw_id), 0) / total
                 for w, s, xp in row_bins]
            )
            ax.bar(x, heights, width=1.0, color=pw_colour,
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

            # X-axis labels
            level_spans_pw: dict[tuple[int, int], list[int]] = defaultdict(list)
            for i, (w, s, _) in enumerate(row_bins):
                level_spans_pw[(w, s)].append(i)
            tick_positions = []
            tick_labels = []
            for lvl in levels_ordered:
                if lvl not in level_spans_pw:
                    continue
                indices = level_spans_pw[lvl]
                tick_positions.append(indices[len(indices) // 2])
                tick_labels.append(f"{lvl[0]+1}-{lvl[1]+1}")
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")
            ax.set_xlim(-0.5, len(row_bins) - 0.5)
            ax.set_ylim(0, y_max)
            ax.set_ylabel("Density")

        pct = 100 * pw_total / total
        axes_pw[0].set_title(
            f"{pw_name} — {pw_total:,} frames ({pct:.1f}%) across "
            f"{num_bins} (world, stage, x_bin[{bin_width}px]) bins"
        )
        axes_pw[0].legend(fontsize=8)
        axes_pw[1].set_xlabel("World-Stage")
        plt.tight_layout()

    # Text summary for powerup breakdown
    print(f"\nPowerup breakdown:")
    for pw_id in powerup_ids:
        pw_frames = int(np.sum(powerup == pw_id))
        pct = 100 * pw_frames / total
        name = _POWERUP_NAMES.get(pw_id, f"pw={pw_id}")
        print(f"  {name:>5}: {pw_frames:>8,} frames ({pct:5.1f}%)")

    plt.show()


def show_x_bins(df, bin_width: int = 128, top: int = 30,
                show_bars: bool = True, sort: str | None = None):
    """Print the distribution of x_bin values (horizontal level position buckets)."""
    for col in ("ram_006d", "ram_0086"):
        if col not in df.columns:
            print(f"Error: column '{col}' not found.", file=sys.stderr)
            sys.exit(1)

    x_abs = df["ram_006d"].values.astype(int) * 256 + df["ram_0086"].values.astype(int)
    x_bin = (x_abs // bin_width) * bin_width

    counts: Counter = Counter(x_bin.tolist())
    total = sum(counts.values())
    nunique = len(counts)
    print(f"Distribution of x_bin (bin_width={bin_width}px):")
    print(f"  Total frames:  {total:,}")
    print(f"  Unique bins:   {nunique}")
    print()

    use_value = (sort == "value") if sort else True
    if use_value:
        items = sorted(counts.items(), key=lambda kv: kv[0])
    else:
        items = sorted(counts.items(), key=lambda kv: -kv[1])

    items = items[:top]

    labels = [str(xb) for xb, _ in items]
    max_lbl = max(len(l) for l in labels)
    max_cnt = max(len(f"{c:,}") for _, c in items)

    for label, (xb, cnt) in zip(labels, items):
        pct = 100 * cnt / total
        if show_bars:
            bar = "#" * int(pct / 2)
            print(f"  {label:>{max_lbl}}: {cnt:>{max_cnt},}  ({pct:5.1f}%)  {bar}")
        else:
            print(f"  {label:>{max_lbl}}: {cnt:>{max_cnt},}  ({pct:5.1f}%)")

    if nunique > top:
        print(f"  ... and {nunique - top} more bins")


def sample_frames(recordings, df, mask, title: str, count: int = 9):
    """Show random frames from recordings matching a boolean mask."""
    matching = df.index[mask].values

    if len(matching) == 0:
        print(f"No frames found for {title}")
        sys.exit(1)

    count = min(count, len(matching))
    chosen = np.random.choice(matching, size=count, replace=False)
    chosen.sort()

    print(f"Sampling {count} frames for {title} "
          f"({len(matching):,} matching frames)")

    # Build cumulative frame offsets per recording
    cum_offsets = []
    offset = 0
    for rec in recordings:
        cum_offsets.append(offset)
        offset += len(rec.frames)

    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if count == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, global_idx in enumerate(chosen):
        # Find which recording this frame belongs to
        rec_id = int(df.iloc[global_idx]["recording_id"])
        rec = recordings[rec_id]
        local_idx = global_idx - cum_offsets[rec_id]

        # Read frame from AVI
        if rec.avi_path is None or not rec.avi_path.exists():
            axes[i].text(0.5, 0.5, "No AVI", ha="center", va="center",
                         transform=axes[i].transAxes)
            axes[i].set_title(f"rec={rec_id} frame={int(rec.frames[local_idx])}")
            continue

        cap = cv2.VideoCapture(str(rec.avi_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            axes[i].text(0.5, 0.5, "Read failed", ha="center", va="center",
                         transform=axes[i].transAxes)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[i].imshow(frame_rgb, interpolation="nearest")

        world = int(rec.ram[local_idx][0x075F]) + 1
        stage = int(rec.ram[local_idx][0x075C]) + 1
        x_pos = int(rec.ram[local_idx][0x006D]) * 256 + int(rec.ram[local_idx][0x0086])
        axes[i].set_title(f"{world}-{stage} x={x_pos}", fontsize=9)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13)
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
    _PROGRESSION_ADDRS = [0x075F, 0x075C, 0x006D, 0x0086, 0x0756]

    if args.ram_only is not None:
        ram_columns = [int(a, 16) for a in args.ram_only]
    elif args.progression or args.x_bins or args.sample is not None:
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
        plot_progression(df, bin_width=args.bin_width)
        return

    if args.x_bins:
        if args.recording is not None:
            df = df[df["recording_id"] == args.recording].copy()
            print(f"Filtered to recording_id={args.recording}: {len(df):,} frames\n")
        if args.sample is not None:
            target = int(args.sample)
            x_abs = df["ram_006d"].values.astype(int) * 256 + df["ram_0086"].values.astype(int)
            x_bin = (x_abs // args.bin_width) * args.bin_width
            sample_frames(recordings, df, x_bin == target,
                          f"x_bin={target} (bin_width={args.bin_width})",
                          args.sample_count)
        else:
            show_x_bins(df, bin_width=args.bin_width, top=args.top,
                        show_bars=not args.counts, sort=args.sort)
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

    def _label_fn(col):
        return action_label if col == "action" else None

    if len(args.columns) == 1:
        col = args.columns[0]
        if args.sample is not None:
            target = type(df[col].iloc[0])(args.sample)
            sample_frames(recordings, df, df[col].values == target,
                          f"{col}={args.sample}", args.sample_count)
            return
        print(f"Distribution of '{col}':")
        print_value_counts(df[col], top=args.top, show_bars=not args.counts,
                           sort=args.sort, label_fn=_label_fn(col))

    elif len(args.columns) == 2:
        col_a, col_b = args.columns
        print(f"Joint distribution of '{col_a}' × '{col_b}':")
        print_joint_distribution(df, col_a, col_b, top=args.top, sort=args.sort)

    else:
        # 3+ columns: show each individually
        for col in args.columns:
            print(f"\n{'='*50}")
            print(f"Distribution of '{col}':")
            print_value_counts(df[col], top=args.top, show_bars=not args.counts,
                               sort=args.sort, label_fn=_label_fn(col))

    # Optional plot
    if args.plot:
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
