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
import numpy as np
from scipy.ndimage import gaussian_filter1d

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.plot_style import apply_plot_style
apply_plot_style()


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
DEFAULT_TRAIN_PLOTS = [
    {
        "name": "loss",
        "keys": ("loss_smooth", "loss"),
        "metric_type": "train",
        "ylabel": "Loss",
        "title": "Training Loss by Run",
        "log_scale": True,
        "marker": "o",
    },
    {
        "name": "flow",
        "keys": ("flow_loss",),
        "metric_type": "train",
        "ylabel": "Flow Loss",
        "title": "Training Flow Loss by Run",
        "log_scale": True,
        "marker": "o",
    },
    {
        "name": "mse",
        "keys": ("x0_mse",),
        "metric_type": "train",
        "ylabel": "x0 MSE",
        "title": "Training x0 MSE by Run",
        "log_scale": True,
        "marker": "s",
    },
    {
        "name": "grad_norm",
        "keys": ("grad_norm",),
        "metric_type": "train",
        "ylabel": "Grad Norm",
        "title": "Training Grad Norm by Run",
        "log_scale": True,
        "marker": "^",
    },
]
OPTIONAL_EVAL_PLOTS = [
    {
        "name": "eval_flow",
        "keys": ("flow_loss",),
        "metric_type": "eval",
        "ylabel": "Flow Loss",
        "title": "Eval Flow Loss by Run",
        "log_scale": True,
        "marker": "o",
    },
    {
        "name": "eval_mse",
        "keys": ("x0_mse",),
        "metric_type": "eval",
        "ylabel": "x0 MSE",
        "title": "Eval x0 MSE (Clean Reconstruction) by Run",
        "log_scale": True,
        "marker": "s",
    },
]


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
    metrics: list[dict], key: str, x_axis: str, metric_type: str | None = None
) -> tuple[list[float], list[float]]:
    axis_key, axis_scale, _ = get_x_axis_metadata(x_axis)
    points = [
        m for m in metrics
        if key in m and axis_key in m and (metric_type is None or m.get("type") == metric_type)
    ]
    if not points:
        return [], []
    return [m[axis_key] * axis_scale for m in points], [m[key] for m in points]


def get_preferred_metric_series(
    metrics: list[dict], keys: tuple[str, ...], x_axis: str, metric_type: str | None = None
) -> tuple[str | None, list[float], list[float]]:
    for key in keys:
        x_vals, y_vals = get_metric_series(metrics, key, x_axis, metric_type=metric_type)
        if x_vals:
            return key, x_vals, y_vals
    return None, [], []


def get_latest_metric(metrics: list[dict], *keys: str, metric_type: str | None = None) -> float | None:
    for metric in reversed(metrics):
        if metric_type is not None and metric.get("type") != metric_type:
            continue
        for key in keys:
            value = metric.get(key)
            if value is not None:
                return value
    return None


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
    vals = [m["flow_loss"] for m in metrics if m.get("type") == "eval" and "flow_loss" in m]
    return min(vals) if vals else float("nan")


def _best_train_loss(metrics: list[dict]) -> float:
    _, _, vals = get_preferred_metric_series(metrics, ("loss_smooth", "loss"), "step", metric_type="train")
    return min(vals) if vals else float("nan")


def _has_metric_series(runs: list[dict], keys: tuple[str, ...], x_axis: str, metric_type: str | None) -> bool:
    for run in runs:
        _, x_vals, _ = get_preferred_metric_series(run["metrics"], keys, x_axis, metric_type=metric_type)
        if x_vals:
            return True
    return False


def _resolve_plot_specs(
    runs: list[dict],
    x_axis: str,
    show_eval: bool,
    show_lr: bool,
) -> tuple[list[dict], list[str]]:
    specs: list[dict] = []
    missing: list[str] = []

    for spec in DEFAULT_TRAIN_PLOTS:
        if _has_metric_series(runs, spec["keys"], x_axis, spec["metric_type"]):
            specs.append(spec)
        else:
            missing.append(spec["name"])

    if show_eval:
        for spec in OPTIONAL_EVAL_PLOTS:
            if _has_metric_series(runs, spec["keys"], x_axis, spec["metric_type"]):
                specs.append(spec)

    if show_lr and _has_metric_series(runs, ("lr",), x_axis, "train"):
        specs.append({
            "name": "lr",
            "keys": ("lr",),
            "metric_type": "train",
            "ylabel": "Learning Rate",
            "title": "Learning Rate Schedule by Run",
            "log_scale": False,
            "marker": None,
        })

    return specs, missing


def summarise(runs: list[dict]) -> list[OrderedDict]:
    rows = []
    for run in runs:
        cfg = run["config"]
        metrics = run["metrics"]
        best_flow = _best_flow(metrics)
        best_train = _best_train_loss(metrics)
        last_flow = get_latest_metric(metrics, "flow_loss", metric_type="eval")
        last_x0_mse = get_latest_metric(metrics, "x0_mse", metric_type="eval")
        last_train = get_latest_metric(metrics, "loss_smooth", "loss", metric_type="train")
        last_lr = get_latest_metric(metrics, "lr")
        total_steps = int(get_latest_metric(metrics, "step") or 0)
        elapsed_s = get_latest_metric(metrics, "elapsed_s") or 0

        num_params = cfg.get("num_parameters", 0)
        d_model = cfg.get("d_model", "?")
        num_layers = cfg.get("num_layers", "?")
        num_heads = cfg.get("num_heads", "?")
        context_latents = cfg.get("context_latents", "?")
        clip_latents = cfg.get("clip_latents", "?")
        flow_loss_type = cfg.get("flow_loss", "?")

        rows.append(OrderedDict([
            ("name", run["name"]),
            ("d_model", d_model),
            ("layers", num_layers),
            ("heads", num_heads),
            ("ctx/clip", f"{context_latents}/{clip_latents}"),
            ("loss_type", flow_loss_type),
            ("params", f"{num_params / 1e6:.1f}M" if num_params else "?"),
            ("batch", cfg.get("batch_size", "?")),
            ("best_train", f"{best_train:.6f}" if not np.isnan(best_train) else ""),
            ("last_train", f"{last_train:.6f}" if last_train is not None else ""),
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
    show_eval: bool = False,
    show_lr: bool = False,
) -> None:
    if not runs:
        print("Nothing to plot.")
        return

    _, _, axis_label = get_x_axis_metadata(x_axis)
    run_styles = build_run_styles(runs)

    plot_specs, missing_defaults = _resolve_plot_specs(runs, x_axis, show_eval=show_eval, show_lr=show_lr)
    if not plot_specs:
        print(f"No requested metrics available for x-axis '{x_axis}'.")
        return
    if missing_defaults:
        print("Skipping unavailable default train metrics: " + ", ".join(missing_defaults))

    nrows = len(plot_specs)
    height_ratios = [2 if spec["name"] in {"loss", "flow", "eval_flow"} else 1 for spec in plot_specs]

    # Sort runs best-first for legend ordering
    sorted_runs = sorted(
        runs,
        key=lambda r: (_best_train_loss(r["metrics"]), _best_flow(r["metrics"])),
    )

    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(12, 4 * nrows),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if nrows == 1:
        axes = [axes]

    marker_medium = 4
    marker_large = 6
    legend_ncols = legend_columns(len(runs))

    for spec_index, spec in enumerate(plot_specs):
        ax = axes[spec_index]

        if spec["name"] == "lr":
            marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]
            lr_series = []
            for run in sorted_runs:
                x_vals, lr_vals = get_metric_series(run["metrics"], "lr", x_axis, metric_type="train")
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
            ax.ticklabel_format(axis="y", style="scientific", scilimits=(-4, -4))
        else:
            for run in sorted_runs:
                _, x_vals, y_vals = get_preferred_metric_series(
                    run["metrics"], spec["keys"], x_axis, metric_type=spec["metric_type"]
                )
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
                    marker=spec["marker"], markersize=marker_medium,
                    alpha=0.9,
                )

        ax.set_ylabel(spec["ylabel"])
        ax.set_title(spec["title"])
        if spec["log_scale"]:
            ax.set_yscale("log")
        if show_legend:
            ax.legend(ncol=legend_ncols, frameon=True, handlelength=3.0)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(axis_label)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
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
    parser.add_argument("--show-eval", action="store_true",
                        help="Include eval flow/MSE panels when available")
    parser.add_argument("--show-lr", action="store_true",
                        help="Include the learning-rate schedule")
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
        show_eval=args.show_eval,
        show_lr=args.show_lr,
    )


if __name__ == "__main__":
    main()
