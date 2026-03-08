#!/usr/bin/env python3
"""Compare sweep results across multiple training runs.

Scans a results directory recursively for folders containing config.json +
metrics.json, then produces:
  1. A summary table (printed and optionally saved as CSV).
  2. A multi-panel comparison plot (recon loss + codebook usage).

Usage:
    python scripts/compare_sweeps.py --results-dir checkpoints/magvit2
    python scripts/compare_sweeps.py --results-dir checkpoints/magvit2 -o sweep_comparison.png
"""

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LAYER_ABBREVIATIONS = {
    "compress_space": "cs",
    "compress_time": "ct",
    "attend_space": "as",
    "attend_time": "at",
    "consecutive_residual": "r",
    "residual": "r",
}

DISTINCT_COLORS = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Dark2").colors)
LINE_STYLES = ["-", "--", "-.", ":"]


def get_recon_values(metrics: list[dict]) -> tuple[str | None, list[float]]:
    """Return the preferred reconstruction-loss values for summaries and sorting."""
    for key in ("smoothed_recon_loss", "recon_loss"):
        values = [m[key] for m in metrics if key in m]
        if values:
            return key, values
    return None, []


def get_recon_series(metrics: list[dict]) -> tuple[str | None, list[float], list[float]]:
    """Return the preferred reconstruction-loss series for plotting."""
    key, _ = get_recon_values(metrics)
    if key is not None:
        points = [m for m in metrics if key in m and "step" in m]
        if points:
            return key, [m["step"] for m in points], [m[key] for m in points]
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

    return {
        run["name"]: {
            "color": DISTINCT_COLORS[index % len(DISTINCT_COLORS)],
            "linestyle": LINE_STYLES[(index // len(DISTINCT_COLORS)) % len(LINE_STYLES)],
        }
        for index, run in enumerate(ordered_runs)
    }


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


def discover_runs(results_dir: str) -> list[dict]:
    """Find all directories under the results tree with config.json and metrics.json."""
    runs = []
    base = Path(results_dir)
    candidates = [base]
    if base.is_dir():
        candidates.extend(sorted(path for path in base.rglob("*") if path.is_dir()))
    for d in candidates:
        config_path = d / "config.json"
        metrics_path = d / "metrics.json"
        if config_path.exists() and metrics_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            with open(metrics_path) as f:
                metrics = json.load(f)
            if metrics:  # skip empty
                run_name = d.relative_to(base).as_posix() if d != base else d.name
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

        # Best (minimum) reconstruction loss
        _, recon_losses = get_recon_values(metrics)
        best_recon = min(recon_losses) if recon_losses else float("nan")
        final_recon = recon_losses[-1] if recon_losses else float("nan")

        # Final codebook usage
        cb_entries = [m for m in metrics if "codebook_usage" in m]
        final_cb = cb_entries[-1]["codebook_usage"] if cb_entries else None
        max_cb = max(m["codebook_usage"] for m in cb_entries) if cb_entries else None

        # Last step / elapsed
        last = metrics[-1]
        total_steps = last.get("step", 0)
        elapsed_s = last.get("elapsed_s", 0)

        layers_str = format_layers(cfg.get("layers", "?"))

        rows.append(OrderedDict([
            ("name", run["name"]),
            ("init_dim", cfg.get("init_dim", "?")),
            ("codebook_size", cfg.get("codebook_size", "?")),
            ("layers", layers_str),
            ("num_params", f"{cfg.get('num_parameters', 0):,}"),
            ("best_recon", f"{best_recon:.4f}"),
            ("final_recon", f"{final_recon:.4f}"),
            ("final_cb_usage", final_cb),
            ("max_cb_usage", max_cb),
            ("total_steps", total_steps),
            ("elapsed_s", f"{elapsed_s:.0f}"),
        ]))

    # Sort by best_recon ascending
    rows.sort(key=lambda r: float(r["best_recon"]))
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


def plot_comparison(runs: list[dict], output_path: str | None = None) -> None:
    if not runs:
        print("Nothing to plot.")
        return

    has_cb = any("codebook_usage" in m for run in runs for m in run["metrics"])
    has_smoothed_recon = any("smoothed_recon_loss" in m for run in runs for m in run["metrics"])
    run_styles = build_run_styles(runs)
    nrows = 2 if has_cb else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 5 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    # Sort runs by the best available reconstruction curve so the legend is ordered.
    sorted_runs = sorted(
        runs,
        key=lambda r: min(get_recon_series(r["metrics"])[2], default=float("inf")),
    )

    # --- Recon loss ---
    ax = axes[0]
    for run in sorted_runs:
        _, steps, recon = get_recon_series(run["metrics"])
        if steps:
            style = run_styles[run["name"]]
            label = run["name"]
            params = run["config"].get("num_parameters", 0)
            if params:
                label += f" ({params / 1e6:.1f}M)"
            ax.plot(
                steps,
                recon,
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                alpha=0.9,
                linewidth=1.6,
            )
    ylabel = "Smoothed Reconstruction Loss" if has_smoothed_recon else "Reconstruction Loss"
    title = "Smoothed Reconstruction Loss Comparison" if has_smoothed_recon else "Reconstruction Loss Comparison"
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, frameon=True, handlelength=3.0)
    ax.grid(True, alpha=0.3)

    # --- Codebook usage ---
    if has_cb:
        ax = axes[1]
        for run in sorted_runs:
            metrics = run["metrics"]
            style = run_styles[run["name"]]
            cb_steps = [m["step"] for m in metrics if "codebook_usage" in m]
            cb_usage = [m["codebook_usage"] for m in metrics if "codebook_usage" in m]
            if cb_steps:
                cb_size = run["config"].get("codebook_size", None)
                label = run["name"]
                line = ax.plot(
                    cb_steps,
                    cb_usage,
                    marker=".",
                    markersize=2,
                    label=label,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    alpha=0.85,
                    linewidth=1.2,
                )[0]
                if cb_size is not None:
                    ax.axhline(cb_size, color=line.get_color(), linestyle=":", alpha=0.25, linewidth=0.6)
        usage_ylim = get_codebook_usage_ylim(sorted_runs)
        if usage_ylim is not None:
            ax.set_ylim(*usage_ylim)
        ax.set_ylabel("Unique codes used")
        ax.set_title("Codebook Usage")
        ax.legend(fontsize=7, ncol=2, frameon=True, handlelength=3.0)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=str, default="checkpoints/magvit2",
                        help="Directory containing run folders anywhere under it")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save comparison plot to file (e.g. sweep_comparison.png)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save summary table as CSV")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only include runs whose name contains this substring")
    args = parser.parse_args()

    runs = discover_runs(args.results_dir)
    if args.filter:
        runs = [r for r in runs if args.filter in r["name"]]

    if not runs:
        print(f"No runs found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(runs)} run(s) in {args.results_dir}\n")

    rows = summarise(runs)
    print_table(rows)
    print()

    if args.csv:
        save_csv(rows, args.csv)

    plot_comparison(runs, output_path=args.output)


if __name__ == "__main__":
    main()
