#!/usr/bin/env python3
"""
scripts/balance_report.py

Print (and optionally save) a world-stage balance report for collected data.

Usage:
    python scripts/balance_report.py --data-dir data/human_play
    python scripts/balance_report.py --data-dir data/human_play --output balance.json
    python scripts/balance_report.py --data-dir data/human_play --progression --plot
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.coverage import (
    compute_action_balance,
    compute_balance_report,
    compute_progression_balance,
    default_level_pool,
    print_action_report,
    print_progression_report,
    print_report,
    ProgressionBalanceReport,
    save_report,
    scan_action_coverage,
    scan_coverage,
    scan_progression_coverage,
)
from mario_world_model.rollouts import RolloutIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a world-stage balance report for collected Mario data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/human_play"),
        help="Directory containing chunk_*.npz files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write a machine-readable JSON report",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Only show the N most-deficient levels (0 = show all)",
    )
    parser.add_argument(
        "--progression",
        action="store_true",
        default=True,
        help="Show per-(level, screen-bin) x-position progression coverage",
    )
    parser.add_argument(
        "--actions",
        action="store_true",
        help="Show action distribution balance",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a matplotlib heatmap of progression distribution (requires --progression)",
    )
    return parser.parse_args()


def plot_progression(report: ProgressionBalanceReport) -> None:
    """Show progression data as a probability distribution (bar chart).

    x-axis: every (world, stage, bin) tuple in sorted order.
    y-axis: density (frame_count / total_frames).
    A horizontal line marks the uniform target density.
    Bars are coloured by level, with vertical separators between levels.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    bins_sorted = sorted(report.bins, key=lambda b: (b.world, b.stage, b.x_bin))
    if not bins_sorted:
        print("No progression data to plot.")
        return

    total = report.total_frames or 1
    densities = [b.frame_count / total for b in bins_sorted]
    labels = [f"{b.world}-{b.stage}:{b.x_bin}" for b in bins_sorted]
    x = np.arange(len(bins_sorted))

    # Colour each bar by level
    levels_ordered = sorted({(b.world, b.stage) for b in bins_sorted})
    cmap = plt.cm.get_cmap("tab20", len(levels_ordered))
    level_to_colour = {lvl: cmap(i) for i, lvl in enumerate(levels_ordered)}
    colours = [level_to_colour[(b.world, b.stage)] for b in bins_sorted]

    # Split bins across two rows so the plot isn't extremely wide
    n = len(bins_sorted)
    mid = (n + 1) // 2  # first row gets the extra bin if odd
    row_slices = [slice(0, mid), slice(mid, n)]

    uniform = 1.0 / report.num_bins if report.num_bins else 0
    y_max = max(densities) * 1.15 if densities else 1

    fig, axes = plt.subplots(2, 1, figsize=(max(14, mid * 0.18), 8))

    for row_idx, sl in enumerate(row_slices):
        ax = axes[row_idx]
        row_x = x[sl] - x[sl].min()  # re-zero the x positions for this row
        row_densities = densities[sl.start : sl.stop]
        row_colours = colours[sl.start : sl.stop]
        row_bins = bins_sorted[sl.start : sl.stop]

        ax.bar(row_x, row_densities, width=1.0, color=row_colours, edgecolor="none", linewidth=0)
        ax.axhline(uniform, color="red", linewidth=1.2, linestyle="--",
                    label=f"uniform = {uniform:.4f}" if row_idx == 0 else None)

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
        f"{report.num_bins} bins"
    )
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("World-Stage")
    fig.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        print(f"Error: directory {args.data_dir} does not exist.")
        sys.exit(1)

    pool = default_level_pool()

    # --- Level balance (always shown) ---
    print(f"Scanning {args.data_dir} ...")
    coverage = scan_coverage(args.data_dir)
    report = compute_balance_report(coverage, level_pool=pool)
    print_report(report, top_n=args.top)

    # --- Progression balance ---
    if args.progression:
        prog_cov = scan_progression_coverage(args.data_dir)
        # Load rollout data to determine which bins are actually reachable
        ri = RolloutIndex(args.data_dir)
        reachable = ri.reachable_bins()
        prog_report = compute_progression_balance(prog_cov, reachable)
        print_progression_report(prog_report, top_n=args.top)
        if args.plot:
            plot_progression(prog_report)

    # --- Action balance ---
    if args.actions:
        # Try to determine num_actions from the project
        try:
            from mario_world_model.actions import get_num_actions, get_action_meanings
            num_actions = get_num_actions()
            action_meanings = get_action_meanings()
        except ImportError:
            num_actions = 26  # fallback
            action_meanings = None
        act_cov = scan_action_coverage(args.data_dir)
        act_report = compute_action_balance(act_cov, num_actions)
        print_action_report(act_report, action_meanings=action_meanings, top_n=args.top)

    if args.output:
        save_report(report, args.output)
        print(f"\nJSON report saved to {args.output}")


if __name__ == "__main__":
    main()
