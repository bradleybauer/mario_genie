#!/usr/bin/env python3
"""
scripts/balance_report.py

Dataset statistics and progression/action balance reports for collected data.

Usage:
    python scripts/balance_report.py --data-dir data/nes --stats
    python scripts/balance_report.py --data-dir data/nes --text
    python scripts/balance_report.py --data-dir data/nes --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.config import SEQUENCE_LENGTH
from mario_world_model.coverage import (
    PROGRESSION_BIN_SIZE,
    compute_progression_balance,
    print_progression_report,
    ProgressionBalanceReport,
    scan_progression_coverage,
)
from mario_world_model.envs import default_level_pool


# ---------------------------------------------------------------------------
# Dataset statistics (merged from dataset_stats.py)
# ---------------------------------------------------------------------------

def gather_stats(data_dir: Path) -> dict:
    """Scan all session meta.json + npz files and return aggregate stats."""
    meta_files = sorted(data_dir.rglob("session_*.meta.json"))
    npz_files = sorted(data_dir.rglob("session_*.npz"))

    total_sessions = 0
    total_frames = 0
    action_counts: Counter[int] = Counter()
    level_counts: Counter[str] = Counter()
    disk_bytes = 0
    total_sequences = 0

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

    if not meta_files:
        print("No meta.json files found")

    for f in npz_files:
        disk_bytes += f.stat().st_size

    return {
        "data_dir": str(data_dir),
        "num_sessions": total_sessions,
        "total_frames": total_frames,
        "total_sequences": total_sequences,
        "sequence_length": SEQUENCE_LENGTH,
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset statistics and balance reports for collected Mario data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/nes"),
        help="Directory containing session_*.npz files",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics summary (sessions, frames, actions, levels)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a live matplotlib chart of progression distribution",
    )
    parser.add_argument(
        "--progression-bin-size",
        type=int,
        default=PROGRESSION_BIN_SIZE,
        help="Progression coverage bin width in pixels; default is {}".format(PROGRESSION_BIN_SIZE),
    )
    parser.add_argument(
        "--watch-interval",
        type=float,
        default=1.0,
        help="Seconds between on-disk refresh checks while plotting (default: 1.0)",
    )
    return parser.parse_args()


def build_progression_report(data_dir: Path, progression_bin_size: int) -> ProgressionBalanceReport:
    prog_cov = scan_progression_coverage(data_dir, bin_size=progression_bin_size)
    observed_bins = set(prog_cov.keys())
    return compute_progression_balance(
        prog_cov,
        observed_bins,
        all_levels=default_level_pool(),
        bin_size=progression_bin_size,
    )


def _data_fingerprint(data_dir: Path) -> tuple[tuple[str, int, int], ...]:
    paths = sorted(data_dir.rglob("session_*.meta.json"))
    rollout_path = data_dir / "rollouts.jsonl"
    if rollout_path.exists():
        paths.append(rollout_path)
    fingerprint: list[tuple[str, int, int]] = []
    for path in paths:
        stat = path.stat()
        fingerprint.append((path.name, int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(fingerprint)


def _draw_progression_axes(axes, report: ProgressionBalanceReport, message: Optional[str] = None) -> None:
    """Draw progression data as a probability distribution (bar chart).

    x-axis: every (world, stage, bin) tuple in sorted order.
    y-axis: density (frame_count / total_frames).
    A horizontal line marks the support-uniform reference density.
    Bars are coloured by level, with vertical separators between levels.
    """
    import matplotlib.pyplot as plt

    bins_sorted = sorted(report.bins, key=lambda b: (b.world, b.stage, b.x_bin))
    for ax in axes:
        ax.clear()

    if not bins_sorted:
        axes[0].text(0.5, 0.5, "No progression data", ha="center", va="center", transform=axes[0].transAxes)
        axes[1].axis("off")
        return

    axes[1].set_visible(True)

    total = report.total_frames or 1
    densities = [b.frame_count / total for b in bins_sorted]
    x = np.arange(len(bins_sorted))

    # Colour each bar by level
    levels_ordered = sorted({(b.world, b.stage) for b in bins_sorted})
    cmap = plt.get_cmap("tab20", len(levels_ordered))
    level_to_colour = {lvl: cmap(i) for i, lvl in enumerate(levels_ordered)}
    colours = [level_to_colour[(b.world, b.stage)] for b in bins_sorted]

    # Split bins across two rows so the plot isn't extremely wide
    n = len(bins_sorted)
    mid = (n + 1) // 2  # first row gets the extra bin if odd
    row_slices = [slice(0, mid), slice(mid, n)]

    uniform = 1.0 / report.num_bins if report.num_bins else 0
    y_max = max(densities) * 1.15 if densities else 1

    # Cap figure width so fine-grained bins do not request enormous X11 pixmaps.
    for row_idx, sl in enumerate(row_slices):
        ax = axes[row_idx]
        row_x = x[sl] - x[sl].min()  # re-zero the x positions for this row
        row_densities = densities[sl.start : sl.stop]
        row_colours = colours[sl.start : sl.stop]
        row_bins = bins_sorted[sl.start : sl.stop]

        ax.bar(row_x, row_densities, width=1.0, color=row_colours, edgecolor="none", linewidth=0)
        ax.axhline(uniform, color="red", linewidth=1.2, linestyle="--",
                    label=f"support-uniform = {uniform:.4f}" if row_idx == 0 else None)

        # Vertical separators between levels
        prev_level = None
        for i, b in enumerate(row_bins):
            lvl = (b.world, b.stage)
            if prev_level is not None and lvl != prev_level:
                ax.axvline(i - 0.5, color="grey", linewidth=0.4, alpha=0.5)
            prev_level = lvl

        # x-axis: show level labels at the midpoint of each level's span
        row_level_spans: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, b in enumerate(row_bins):
            row_level_spans[(b.world, b.stage)].append(i)
        tick_positions = []
        tick_labels_row = []
        for lvl in levels_ordered:
            if lvl not in row_level_spans:
                continue
            indices = row_level_spans[lvl]
            mid_pos = indices[len(indices) // 2]
            tick_positions.append(mid_pos)
            tick_labels_row.append(f"{lvl[0]}-{lvl[1]}")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_row, fontsize=6, rotation=45, ha="right")

        ax.set_xlim(-0.5, len(row_bins) - 0.5)
        ax.set_ylim(0, y_max)
        ax.set_ylabel("Density")

    axes[0].set_title(
        f"Progression Distribution — {report.total_frames:,} frames across "
        f"{report.num_bins} bins ({report.bin_size}px/bin)"
    )
    if message:
        axes[0].text(0.995, 0.98, message, ha="right", va="top", fontsize=8, transform=axes[0].transAxes)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("World-Stage")


def plot_progression_live(args: argparse.Namespace, initial_report: ProgressionBalanceReport) -> None:
    import matplotlib.pyplot as plt

    figure_width = min(32.0, max(14.0, len(default_level_pool()) * 0.8))
    fig, axes = plt.subplots(2, 1, figsize=(figure_width, 8))
    last_fingerprint = _data_fingerprint(args.data_dir)
    _draw_progression_axes(axes, initial_report, message="watching")
    fig.tight_layout()

    def refresh_if_needed() -> None:
        nonlocal last_fingerprint
        current_fingerprint = _data_fingerprint(args.data_dir)
        if current_fingerprint == last_fingerprint:
            return
        last_fingerprint = current_fingerprint
        refreshed_report = build_progression_report(args.data_dir, args.progression_bin_size)
        print("\n[watch] Data updated on disk. Refreshing progression plot...")
        _draw_progression_axes(axes, refreshed_report, message="updated")
        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=max(100, int(args.watch_interval * 1000)))
    timer.add_callback(refresh_if_needed)
    timer.start()

    def _restart_timer(_event=None) -> None:
        timer.stop()
        timer.start()

    fig.canvas.mpl_connect("draw_event", _restart_timer)
    plt.show()


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        print(f"Error: directory {args.data_dir} does not exist.")
        sys.exit(1)

    print(f"Scanning {args.data_dir} ...")

    stats = gather_stats(args.data_dir)
    if args.stats:
        print_stats(stats)

    prog_report = build_progression_report(args.data_dir, args.progression_bin_size)
    print_progression_report(prog_report, total_sequences=stats["total_sequences"])
    if args.plot:
        print(f"[watch] Plotting live updates every {args.watch_interval:.1f}s while files change on disk.")
        plot_progression_live(args, prog_report)


if __name__ == "__main__":
    main()
