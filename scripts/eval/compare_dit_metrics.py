#!/usr/bin/env python3
"""Compare metrics across multiple video-latent DiT training runs.

Scans one or more results directories recursively for folders containing
config.json + metrics.json, then produces:
    1. A summary table (printed and optionally saved as CSV).
    2. A comparison plot for flow loss and related metrics.

Usage:
        python scripts/compare_dit_metrics.py --results-dir checkpoints/
        python scripts/compare_dit_metrics.py --results-dir checkpoints/dit_runs checkpoints/ablations
        python scripts/compare_dit_metrics.py --results-dir checkpoints/ -o dit_comparison.png
        python scripts/compare_dit_metrics.py --results-dir checkpoints/ --x-axis step
"""

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use("dark_background")
import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth(values: list[float], sigma: float) -> np.ndarray:
    if sigma <= 0 or len(values) <= 1:
        return np.asarray(values)
    return gaussian_filter1d(values, sigma=sigma)


def _format_elapsed(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


DISTINCT_COLORS = [
    tuple(color) for color in plt.get_cmap("tab10")(np.linspace(0, 1, 10))
] + [
    tuple(color) for color in plt.get_cmap("Dark2")(np.linspace(0, 1, 8))
]
LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 2, 1, 2, 1, 2))]
LINE_WIDTHS = [1.8, 1.2, 2.4]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FONT_SCALE = 1.6
DEFAULT_FIGURE_SCALE = 1.25
DEFAULT_SAVE_DPI = 180
BASE_PLOT_RC = {
    "font.size": 11.0,
    "axes.titlesize": 15.0,
    "axes.labelsize": 13.0,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 11.0,
    "legend.title_fontsize": 12.0,
    "figure.titlesize": 18.0,
}


def build_plot_rc(font_scale: float) -> dict[str, float]:
    scale = max(font_scale, 0.5)
    return {key: value * scale for key, value in BASE_PLOT_RC.items()}


def scale_figsize(width: float, height: float, figure_scale: float) -> tuple[float, float]:
    scale = max(figure_scale, 0.5)
    return width * scale, height * scale


def legend_columns(num_series: int) -> int:
    if num_series <= 4:
        return 1
    if num_series <= 10:
        return 2
    return 3


def get_x_axis_metadata(x_axis: str) -> tuple[str, float, str]:
    if x_axis == "time":
        return "elapsed_s", 1.0 / 3600.0, "Elapsed Time (hours)"
    return "step", 1.0, "Step"


def get_metric_series(
    metrics: list[dict], key: str, x_axis: str
) -> tuple[list[float], list[float]]:
    axis_key, axis_scale, _ = get_x_axis_metadata(x_axis)
    points = [m for m in metrics if key in m and axis_key in m]
    if not points:
        return [], []
    return [m[axis_key] * axis_scale for m in points], [m[key] for m in points]


def build_run_styles(runs: list[dict]) -> dict[str, dict]:
    if not runs:
        return {}
    ordered = sorted(runs, key=lambda r: r["name"])
    n_colors = len(DISTINCT_COLORS)
    n_styles = len(LINE_STYLES)
    n_widths = len(LINE_WIDTHS)
    return {
        run["name"]: {
            "color": DISTINCT_COLORS[i % n_colors],
            "linestyle": LINE_STYLES[(i * 3) % n_styles],
            "linewidth": LINE_WIDTHS[(i * 2) % n_widths],
        }
        for i, run in enumerate(ordered)
    }


def is_dit_run(config: dict) -> bool:
    """Return True if config looks like a video-latent DiT run."""
    return config.get("model_name") == "video_latent_dit" or "d_model" in config


def discover_runs(results_dirs: list[str]) -> list[dict]:
    runs = []
    bases = [Path(d) for d in results_dirs]
    include_base_prefix = len(bases) > 1

    for base in bases:
        candidates = [base]
        if base.is_dir():
            candidates.extend(sorted(p for p in base.rglob("*") if p.is_dir()))
        base_label = base.as_posix()

        for d in candidates:
            config_path = d / "config.json"
            metrics_path = d / "metrics.json"
            if not (config_path.exists() and metrics_path.exists()):
                continue
            with open(config_path) as f:
                config = json.load(f)
            if not is_dit_run(config):
                continue
            with open(metrics_path) as f:
                metrics = json.load(f)
            if not metrics:
                continue
            if d == base:
                run_name = base_label if include_base_prefix else d.name
            else:
                rel = d.relative_to(base).as_posix()
                run_name = f"{base_label}/{rel}" if include_base_prefix else rel
            runs.append({"name": run_name, "dir": str(d), "config": config, "metrics": metrics})
    return runs


def _best_flow(metrics: list[dict]) -> float:
    vals = [m["flow_loss"] for m in metrics if "flow_loss" in m]
    return min(vals) if vals else float("nan")


def summarise(runs: list[dict]) -> list[OrderedDict]:
    rows = []
    for run in runs:
        cfg = run["config"]
        metrics = run["metrics"]
        last = metrics[-1]

        best_flow = _best_flow(metrics)
        last_flow = last.get("flow_loss")
        last_x0_mse = last.get("x0_mse")
        last_lr = last.get("lr")
        total_steps = int(last.get("step", 0))
        elapsed_s = last.get("elapsed_s", 0)

        num_params = cfg.get("num_parameters", 0)
        d_model = cfg.get("d_model", "?")
        num_layers = cfg.get("num_layers", "?")
        num_heads = cfg.get("num_heads", "?")
        context_frames = cfg.get("context_frames", "?")
        clip_frames = cfg.get("clip_frames", "?")
        flow_loss_type = cfg.get("flow_loss", "?")

        rows.append(OrderedDict([
            ("name", run["name"]),
            ("d_model", d_model),
            ("layers", num_layers),
            ("heads", num_heads),
            ("ctx/clip", f"{context_frames}/{clip_frames}"),
            ("loss_type", flow_loss_type),
            ("params", f"{num_params / 1e6:.1f}M" if num_params else "?"),
            ("batch", cfg.get("batch_size", "?")),
            ("best_flow", f"{best_flow:.6f}" if not np.isnan(best_flow) else ""),
            ("last_flow", f"{last_flow:.6f}" if last_flow is not None else ""),
            ("last_x0_mse", f"{last_x0_mse:.6f}" if last_x0_mse is not None else ""),
            ("lr", f"{last_lr:.2e}" if last_lr is not None else ""),
            ("steps", total_steps),
            ("elapsed", _format_elapsed(elapsed_s) if elapsed_s else "?"),
        ]))

    rows.sort(key=lambda r: float(r["best_flow"]) if r["best_flow"] else float("inf"))
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


def plot_comparison(
    runs: list[dict],
    output_path: str | None = None,
    x_axis: str = "step",
    smooth_window: float = 0.0,
    show_legend: bool = True,
    font_scale: float = DEFAULT_FONT_SCALE,
    figure_scale: float = DEFAULT_FIGURE_SCALE,
) -> None:
    if not runs:
        print("Nothing to plot.")
        return

    has_x0_mse = any("x0_mse" in m for run in runs for m in run["metrics"])
    has_lr = any("lr" in m for run in runs for m in run["metrics"])
    axis_key, axis_scale, axis_label = get_x_axis_metadata(x_axis)
    run_styles = build_run_styles(runs)

    panels = ["flow"]
    if has_x0_mse:
        panels.append("x0_mse")
    if has_lr:
        panels.append("lr")
    nrows = len(panels)
    height_ratios = [2] + [1] * (nrows - 1)

    # Sort runs best-first for legend ordering
    sorted_runs = sorted(runs, key=lambda r: _best_flow(r["metrics"]))

    with plt.rc_context(build_plot_rc(font_scale)):
        fig, axes = plt.subplots(
            nrows, 1,
            figsize=scale_figsize(12, 4 * nrows, figure_scale),
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        if nrows == 1:
            axes = [axes]

        marker_medium = max(4.0, 2.4 * font_scale)
        marker_large = max(6.0, 3.8 * font_scale)
        legend_ncols = legend_columns(len(runs))

        # --- Flow loss ---
        ax = axes[panels.index("flow")]
        for run in sorted_runs:
            x_vals, y_vals = get_metric_series(run["metrics"], "flow_loss", x_axis)
            if not x_vals:
                continue
            style = run_styles[run["name"]]
            params = run["config"].get("num_parameters", 0)
            label = run["name"]
            if params:
                label += f" ({params / 1e6:.1f}M)"
            ax.plot(
                x_vals, smooth(y_vals, smooth_window),
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker="o", markersize=marker_medium,
                alpha=0.9,
            )
        ax.set_ylabel("Flow Loss")
        ax.set_yscale("log")
        ax.set_title("Eval Flow Loss by Run")
        if show_legend:
            ax.legend(ncol=legend_ncols, frameon=True, handlelength=3.0)
        ax.grid(True, alpha=0.3)

        # --- x0 MSE ---
        if "x0_mse" in panels:
            ax = axes[panels.index("x0_mse")]
            for run in sorted_runs:
                x_vals, y_vals = get_metric_series(run["metrics"], "x0_mse", x_axis)
                if not x_vals:
                    continue
                style = run_styles[run["name"]]
                ax.plot(
                    x_vals, smooth(y_vals, smooth_window),
                    label=run["name"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    marker="s", markersize=marker_medium,
                    alpha=0.9,
                )
            ax.set_ylabel("x0 MSE")
            ax.set_yscale("log")
            ax.set_title("Eval x0 MSE (Clean Reconstruction) by Run")
            if show_legend:
                ax.legend(ncol=legend_ncols, frameon=True, handlelength=3.0)
            ax.grid(True, alpha=0.3)

        # --- Learning rate ---
        if "lr" in panels:
            ax = axes[panels.index("lr")]
            marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]
            lr_series = []
            for run in sorted_runs:
                x_vals, lr_vals = get_metric_series(run["metrics"], "lr", x_axis)
                if x_vals:
                    lr_series.append((run, x_vals, lr_vals))

            if lr_series:
                x_min = min(min(x) for _, x, _ in lr_series)
                x_max = max(max(x) for _, x, _ in lr_series)
                x_span = max(x_max - x_min, 1.0)
                jitter = x_span * 0.0015 if len(lr_series) > 1 else 0.0
                offsets = np.linspace(-0.5, 0.5, len(lr_series)) if len(lr_series) > 1 else np.array([0.0])
                for idx, (run, x_vals, lr_vals) in enumerate(lr_series):
                    style = run_styles[run["name"]]
                    x_plot = [x + offsets[idx] * jitter for x in x_vals] if jitter > 0 else x_vals
                    marker = marker_cycle[idx % len(marker_cycle)]
                    markevery = max(1, len(x_plot) // 10)
                    ax.plot(
                        x_plot, lr_vals,
                        label=run["name"],
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        marker=marker, markersize=marker_large,
                        markerfacecolor="none", markeredgewidth=1.4,
                        markevery=markevery, alpha=0.9,
                    )
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule by Run")
            ax.ticklabel_format(axis="y", style="scientific", scilimits=(-4, -4))
            if show_legend:
                ax.legend(ncol=legend_ncols, frameon=True, handlelength=3.0)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel(axis_label)
        fig.tight_layout(pad=max(1.0, 1.0 * font_scale))

        if output_path:
            fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
            print(f"Saved plot to {output_path}")
        else:
            plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=str, nargs="+", default=["checkpoints/"],
                        help="One or more directories containing run folders")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save comparison plot to file (e.g. dit_comparison.png)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save summary table as CSV")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only include runs whose name contains this substring")
    parser.add_argument("--x-axis", choices=("time", "step"), default="step",
                        help="X-axis for plots: elapsed time in hours or training step")
    parser.add_argument("--smooth", type=float, default=0.0,
                        help="Gaussian sigma for plot smoothing (0 = no smoothing)")
    parser.add_argument("--no-legend", action="store_true",
                        help="Hide the legend on plots")
    parser.add_argument("--font-scale", type=float, default=DEFAULT_FONT_SCALE,
                        help="Scale all plot text")
    parser.add_argument("--figure-scale", type=float, default=DEFAULT_FIGURE_SCALE,
                        help="Scale the base figure size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runs = discover_runs(args.results_dir)
    if args.filter:
        runs = [r for r in runs if args.filter in r["name"]]

    if not runs:
        searched = ", ".join(args.results_dir)
        print(f"No DiT runs found in {searched}", file=sys.stderr)
        sys.exit(1)

    searched = ", ".join(args.results_dir)
    print(f"Found {len(runs)} DiT run(s) in {searched}\n")

    rows = summarise(runs)
    print_table(rows)
    print()

    if args.csv:
        save_csv(rows, args.csv)

    plot_comparison(
        runs,
        output_path=args.output,
        x_axis=args.x_axis,
        smooth_window=args.smooth,
        show_legend=not args.no_legend,
        font_scale=args.font_scale,
        figure_scale=args.figure_scale,
    )


if __name__ == "__main__":
    main()
