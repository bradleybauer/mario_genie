#!/usr/bin/env python3
"""Compare sweep results across multiple training runs.

Scans a results directory recursively for folders containing config.json +
metrics.json, then produces:
  1. A summary table (printed and optionally saved as CSV).
  2. A multi-panel comparison plot (recon loss + codebook usage).

Usage:
    python scripts/compare_sweeps.py --results-dir checkpoints/magvit2
    python scripts/compare_sweeps.py --results-dir checkpoints/magvit2 results/capacity_sweep
    python scripts/compare_sweeps.py --results-dir checkpoints/magvit2 -o sweep_comparison.png
    python scripts/compare_sweeps.py --results-dir checkpoints/magvit2 --x-axis step
"""

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth(values: list[float], sigma: float) -> np.ndarray:
    """Gaussian smoothing with the given sigma (in data points)."""
    if sigma <= 0 or len(values) <= 1:
        return np.asarray(values)
    return gaussian_filter1d(values, sigma=sigma)


LAYER_ABBREVIATIONS = {
    "compress_space": "cs",
    "compress_time": "ct",
    "attend_space": "as",
    "attend_time": "at",
    "consecutive_residual": "r",
    "residual": "r",
}

DISTINCT_COLORS = [
    tuple(color) for color in plt.get_cmap("tab10")(np.linspace(0, 1, 10))
] + [
    tuple(color) for color in plt.get_cmap("Dark2")(np.linspace(0, 1, 8))
]
LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 2, 1, 2, 1, 2))]
LINE_WIDTHS = [1.8, 1.2, 2.4]


def get_recon_values(metrics: list[dict]) -> tuple[str | None, list[float]]:
    """Return the preferred reconstruction-loss values for summaries and sorting."""
    for key in ("smoothed_recon_loss", "recon_loss"):
        values = [m[key] for m in metrics if key in m]
        if values:
            return key, values
    return None, []


def get_x_axis_metadata(x_axis: str) -> tuple[str, float, str]:
    """Return the metric key, scale factor, and label for the chosen x-axis."""
    if x_axis == "time":
        return "elapsed_s", 1.0 / 3600.0, "Elapsed Time (hours)"
    return "step", 1.0, "Step"


def get_recon_series(metrics: list[dict], x_axis: str) -> tuple[str | None, list[float], list[float]]:
    """Return the preferred reconstruction-loss series for plotting."""
    key, _ = get_recon_values(metrics)
    if key is not None:
        axis_key, axis_scale, _ = get_x_axis_metadata(x_axis)
        points = [m for m in metrics if key in m and axis_key in m]
        if points:
            return key, [m[axis_key] * axis_scale for m in points], [m[key] for m in points]
    return None, [], []


def extract_dataset_size(run_name: str) -> int | None:
    """Extract the dataset size from a run name ending in _n<k>."""
    match = re.search(r"_n(\d+)$", run_name)
    if not match:
        return None
    return int(match.group(1))


def build_run_styles(runs: list[dict]) -> dict[str, dict[str, object]]:
    """Assign distinct plot styles across the compared subset.

    Ordering still follows dataset-size suffixes when present so adjacent sizes read naturally.
    """
    if not runs:
        return {}

    ordered_runs = sorted(
        runs,
        key=lambda run: (
            extract_dataset_size(run["name"]) is None,
            extract_dataset_size(run["name"]) or 0,
            run["name"],
        ),
    )

    n_colors = len(DISTINCT_COLORS)
    n_styles = len(LINE_STYLES)
    n_widths = len(LINE_WIDTHS)

    return {
        run["name"]: {
            "color": DISTINCT_COLORS[index % n_colors],
            "linestyle": LINE_STYLES[(index * 3) % n_styles],
            "linewidth": LINE_WIDTHS[(index * 2) % n_widths],
        }
        for index, run in enumerate(ordered_runs)
    }


FACET_PROPERTIES = ("init_dim", "codebook_size", "model", "attention")


def extract_run_property(run: dict, prop: str) -> str:
    """Extract a categorical property from a run for grouping/faceting."""
    cfg = run["config"]
    name = run["name"]
    if prop == "init_dim":
        return str(cfg.get("init_dim", "?"))
    if prop == "codebook_size":
        return str(cfg.get("codebook_size", "?"))
    if prop == "model":
        for token in ("small", "base", "large"):
            if token in name:
                return token
        return "other"
    if prop == "attention":
        return "With Attention" if "attn" in name else "Without Attention"
    return str(cfg.get(prop, "?"))


def abbreviate_layer_component(component: str) -> str:
    """Shorten verbose layer names for compact summary tables."""
    name, sep, value = component.partition(":")
    abbreviated_name = LAYER_ABBREVIATIONS.get(name, name)
    if not sep:
        return abbreviated_name
    return f"{abbreviated_name}{sep}{value}"


def format_layers(layers: object) -> str:
    """Render layer definitions with abbreviated component names."""
    if not isinstance(layers, list):
        return str(layers)

    formatted_layers = []
    for layer in layers:
        if isinstance(layer, list) and len(layer) == 2:
            formatted_layers.append(
                abbreviate_layer_component(f"{layer[0]}:{layer[1]}")
            )
        else:
            formatted_layers.append(abbreviate_layer_component(str(layer)))
    return ",".join(formatted_layers)


def get_codebook_usage_ylim(runs: list[dict]) -> tuple[float, float] | None:
    """Fit codebook-usage axis limits to observed values only."""
    usages = [
        m["codebook_usage"]
        for run in runs
        for m in run["metrics"]
        if "codebook_usage" in m
    ]
    if not usages:
        return None

    min_usage = min(usages)
    max_usage = max(usages)
    if min_usage == max_usage:
        padding = max(1.0, max_usage * 0.05)
    else:
        padding = max(1.0, (max_usage - min_usage) * 0.05)

    lower = max(0.0, min_usage - padding)
    upper = max_usage + padding
    if lower == upper:
        upper = lower + 1.0
    return lower, upper


def discover_runs(results_dirs: list[str]) -> list[dict]:
    """Find all directories under one or more results trees with config.json and metrics.json."""
    runs = []
    bases = [Path(results_dir) for results_dir in results_dirs]
    include_base_prefix = len(bases) > 1

    for base in bases:
        candidates = [base]
        if base.is_dir():
            candidates.extend(sorted(path for path in base.rglob("*") if path.is_dir()))
        base_label = base.as_posix()

        for d in candidates:
            config_path = d / "config.json"
            metrics_path = d / "metrics.json"
            if config_path.exists() and metrics_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                with open(metrics_path) as f:
                    metrics = json.load(f)
                if metrics:  # skip empty
                    if d == base:
                        run_name = base_label if include_base_prefix else d.name
                    else:
                        relative_name = d.relative_to(base).as_posix()
                        run_name = f"{base_label}/{relative_name}" if include_base_prefix else relative_name
                    runs.append({
                        "name": run_name,
                        "dir": str(d),
                        "config": config,
                        "metrics": metrics,
                    })
    return runs


def summarise(runs: list[dict]) -> list[OrderedDict]:
    """Extract key stats from each run into a flat row."""
    rows = []
    for run in runs:
        cfg = run["config"]
        metrics = run["metrics"]

        smoothed_recon_losses = [
            m["smoothed_recon_loss"] for m in metrics if "smoothed_recon_loss" in m
        ]
        min_smoothed_recon = (
            min(smoothed_recon_losses) if smoothed_recon_losses else float("nan")
        )

        # Final codebook usage
        cb_entries = [m for m in metrics if "codebook_usage" in m]
        final_cb = cb_entries[-1]["codebook_usage"] if cb_entries else None
        max_cb = max(m["codebook_usage"] for m in cb_entries) if cb_entries else None

        # Last step / elapsed
        last = metrics[-1]
        total_steps = last.get("step", 0)
        elapsed_s = last.get("elapsed_s", 0)

        steps_per_s = total_steps / elapsed_s if elapsed_s > 0 else float("nan")

        rows.append(OrderedDict([
            ("name", run["name"]),
            ("num_params", f"{cfg.get('num_parameters', 0):,}"),
            ("min_smoothed_recon", f"{min_smoothed_recon:.4f}"),
            ("final_cb_usage", final_cb),
            ("max_cb_usage", max_cb),
            ("total_steps", total_steps),
            ("elapsed_s", f"{elapsed_s:.0f}"),
            ("steps/s", f"{steps_per_s:.2f}"),
        ]))

    # Sort by min_smoothed_recon ascending
    rows.sort(key=lambda r: float(r["min_smoothed_recon"]))
    return rows


def print_table(rows: list[OrderedDict]) -> None:
    if not rows:
        print("No runs found.")
        return
    headers = list(rows[0].keys())
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row.get(h, ""))))

    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    sep = "-+-".join("-" * col_widths[h] for h in headers)
    print(header_line)
    print(sep)
    for row in rows:
        print(" | ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers))


def save_csv(rows: list[OrderedDict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV to {path}")


def plot_comparison(runs: list[dict], output_path: str | None = None, x_axis: str = "step", smooth_window: int = 1, show_legend: bool = True) -> None:
    if not runs:
        print("Nothing to plot.")
        return

    has_cb = any("codebook_usage" in m for run in runs for m in run["metrics"])
    has_smoothed_recon = any("smoothed_recon_loss" in m for run in runs for m in run["metrics"])
    axis_key, axis_scale, axis_label = get_x_axis_metadata(x_axis)
    run_styles = build_run_styles(runs)
    nrows = 2 if has_cb else 1
    height_ratios = [2, 1] if nrows == 2 else [1]
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 5 * nrows), sharex=True,
                             gridspec_kw={"height_ratios": height_ratios})
    if nrows == 1:
        axes = [axes]

    # Sort runs by the best available reconstruction curve so the legend is ordered.
    sorted_runs = sorted(
        runs,
        key=lambda r: min(get_recon_series(r["metrics"], x_axis)[2], default=float("inf")),
    )

    # --- Recon loss ---
    ax = axes[0]
    for run in sorted_runs:
        _, x_values, recon = get_recon_series(run["metrics"], x_axis)
        if x_values:
            style = run_styles[run["name"]]
            label = run["name"]
            params = run["config"].get("num_parameters", 0)
            if params:
                label += f" ({params / 1e6:.1f}M)"
            ax.plot(
                x_values,
                smooth(recon, smooth_window),
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                alpha=0.9,
                linewidth=style["linewidth"],
            )
    ylabel = "Smoothed Reconstruction Loss" if has_smoothed_recon else "Reconstruction Loss"
    title = "Smoothed Reconstruction Loss Comparison" if has_smoothed_recon else "Reconstruction Loss Comparison"
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_title(title)
    if show_legend:
        ax.legend(fontsize=7, ncol=2, frameon=True, handlelength=3.0)
    ax.grid(True, alpha=0.3)

    # --- Codebook usage ---
    if has_cb:
        ax = axes[1]
        for run in sorted_runs:
            metrics = run["metrics"]
            style = run_styles[run["name"]]
            cb_points = [m for m in metrics if "codebook_usage" in m and axis_key in m]
            cb_steps = [m[axis_key] * axis_scale for m in cb_points]
            cb_usage = [m["codebook_usage"] for m in cb_points]
            if cb_steps:
                cb_size = run["config"].get("codebook_size", None)
                label = run["name"]
                line = ax.plot(
                    cb_steps,
                    smooth(cb_usage, smooth_window),
                    marker=".",
                    markersize=2,
                    label=label,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    alpha=0.85,
                    linewidth=style["linewidth"],
                )[0]
                if cb_size is not None:
                    ax.axhline(cb_size, color=line.get_color(), linestyle=":", alpha=0.25, linewidth=0.6)
        usage_ylim = get_codebook_usage_ylim(sorted_runs)
        if usage_ylim is not None:
            ax.set_ylim(*usage_ylim)
        ax.set_ylabel("Unique codes used")
        ax.set_title("Codebook Usage")
        if show_legend:
            ax.legend(fontsize=7, ncol=2, frameon=True, handlelength=3.0)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(axis_label)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def plot_faceted(runs: list[dict], facet_by: str, output_path: str | None = None, x_axis: str = "step", smooth_window: int = 1, show_legend: bool = True) -> None:
    """Small-multiple plot: one subplot per facet value, shared y-axis."""
    if not runs:
        print("Nothing to plot.")
        return

    groups: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        key = extract_run_property(run, facet_by)
        groups[key].append(run)

    def sort_key(k: str) -> tuple:
        try:
            return (0, int(k), k)
        except ValueError:
            return (1, 0, k)

    sorted_keys = sorted(groups.keys(), key=sort_key)
    ncols = len(sorted_keys)
    run_styles = build_run_styles(runs)
    has_smoothed = any("smoothed_recon_loss" in m for run in runs for m in run["metrics"])
    axis_key, axis_scale, axis_label = get_x_axis_metadata(x_axis)

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, key in zip(axes, sorted_keys):
        group_runs = sorted(
            groups[key],
            key=lambda r: min(get_recon_series(r["metrics"], x_axis)[2], default=float("inf")),
        )
        for run in group_runs:
            _, x_values, recon = get_recon_series(run["metrics"], x_axis)
            if x_values:
                style = run_styles[run["name"]]
                label = run["name"].rsplit("/", 1)[-1]
                params = run["config"].get("num_parameters", 0)
                if params:
                    label += f" ({params / 1e6:.1f}M)"
                ax.plot(x_values, smooth(recon, smooth_window), label=label,
                        color=style["color"], linestyle=style["linestyle"],
                        alpha=0.9, linewidth=style["linewidth"])
        ax.set_title(key)
        ax.set_yscale("log")
        ax.set_xlabel(axis_label)
        if show_legend:
            ax.legend(fontsize=7, frameon=True)
        ax.grid(True, alpha=0.3)

    ylabel = "Smoothed Recon Loss" if has_smoothed else "Recon Loss"
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"Loss by {facet_by}", fontsize=14, y=1.02)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def plot_bar(runs: list[dict], output_path: str | None = None, color_by: str | None = None, show_legend: bool = True) -> None:
    """Horizontal bar chart of best reconstruction loss per run."""
    if not runs:
        print("Nothing to plot.")
        return

    entries = []
    for run in runs:
        _, recon_losses = get_recon_values(run["metrics"])
        best = min(recon_losses) if recon_losses else float("inf")
        params = run["config"].get("num_parameters", 0)
        label = run["name"].rsplit("/", 1)[-1]
        if params:
            label += f" ({params / 1e6:.1f}M)"
        prop = extract_run_property(run, color_by) if color_by else None
        entries.append((label, best, prop))

    entries.sort(key=lambda e: e[1])  # best first

    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    props = [e[2] for e in entries]

    # Color by property if requested
    if color_by:
        unique_props = sorted(set(props))
        prop_colors = {p: DISTINCT_COLORS[i % len(DISTINCT_COLORS)] for i, p in enumerate(unique_props)}
        colors = [prop_colors[p] for p in props]
    else:
        colors = [DISTINCT_COLORS[0]] * len(entries)

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.45)))
    y_pos = range(len(labels))
    ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()  # best (smallest) at top
    ax.set_xlabel("Best Reconstruction Loss")
    ax.set_title("Best Reconstruction Loss by Run")
    ax.grid(True, axis="x", alpha=0.3)

    if color_by and show_legend:
        from matplotlib.patches import Patch
        legend_handles = [Patch(facecolor=prop_colors[p], label=f"{color_by}={p}") for p in unique_props]
        ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=str, nargs="+", default=["checkpoints/magvit2"],
                        help="One or more directories containing run folders anywhere under them")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save comparison plot to file (e.g. sweep_comparison.png)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save summary table as CSV")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only include runs whose name contains this substring")
    parser.add_argument("--x-axis", choices=("time", "step"), default="step",
                        help="X-axis for plots: elapsed time in hours or training step")
    parser.add_argument("--facet", choices=FACET_PROPERTIES, default=None,
                        help="Small-multiple subplots split by this property")
    parser.add_argument("--bar", action="store_true",
                        help="Horizontal bar chart of best reconstruction loss")
    parser.add_argument("--color-by", choices=FACET_PROPERTIES, default=None,
                        help="Color bar chart bars by this property")
    parser.add_argument("--smooth", type=float, default=0.0,
                        help="Gaussian sigma for plot smoothing (0 = no smoothing)")
    parser.add_argument("--no-legend", action="store_true",
                        help="Hide the legend on plots")
    args = parser.parse_args()

    runs = discover_runs(args.results_dir)
    if args.filter:
        runs = [r for r in runs if args.filter in r["name"]]

    if not runs:
        searched_dirs = ", ".join(args.results_dir)
        print(f"No runs found in {searched_dirs}", file=sys.stderr)
        sys.exit(1)

    searched_dirs = ", ".join(args.results_dir)
    print(f"Found {len(runs)} run(s) in {searched_dirs}\n")

    rows = summarise(runs)
    print_table(rows)
    print()

    if args.csv:
        save_csv(rows, args.csv)

    show_legend = not args.no_legend
    if args.bar:
        plot_bar(runs, output_path=args.output, color_by=args.color_by, show_legend=show_legend)
    elif args.facet:
        plot_faceted(runs, args.facet, output_path=args.output, x_axis=args.x_axis, smooth_window=args.smooth, show_legend=show_legend)
    else:
        plot_comparison(runs, output_path=args.output, x_axis=args.x_axis, smooth_window=args.smooth, show_legend=show_legend)


if __name__ == "__main__":
    main()
