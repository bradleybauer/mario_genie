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
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth(values: list[float], sigma: float) -> np.ndarray:
    """Gaussian smoothing with the given sigma (in data points)."""
    if sigma <= 0 or len(values) <= 1:
        return np.asarray(values)
    return gaussian_filter1d(values, sigma=sigma)


def _format_elapsed(seconds: float) -> str:
    """Format seconds as a compact human-readable duration."""
    seconds = max(int(seconds), 0)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


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
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_metric_series(metrics: list[dict], key: str, x_axis: str) -> tuple[list[float], list[float]]:
    """Return (x_values, y_values) for a given metric key."""
    axis_key, axis_scale, _ = get_x_axis_metadata(x_axis)
    points = [m for m in metrics if key in m and axis_key in m]
    if not points:
        return [], []
    return [m[axis_key] * axis_scale for m in points], [m[key] for m in points]


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


def _parse_model_dim_cb(cfg: dict) -> tuple[str, str]:
    """Extract init_dim and codebook_size from config, falling back to model name."""
    dim = cfg.get("init_dim")
    cb = cfg.get("codebook_size")
    if dim is None or cb is None:
        model_name = cfg.get("model", "")
        m = re.match(r"dim(\d+)_cb(\d+)", model_name)
        if m:
            dim = dim or int(m.group(1))
            cb = cb or int(m.group(2))
    return str(dim) if dim is not None else "?", str(cb) if cb is not None else "?"


def extract_run_property(run: dict, prop: str) -> str:
    """Extract a categorical property from a run for grouping/faceting."""
    cfg = run["config"]
    name = run["name"]
    if prop == "init_dim":
        return _parse_model_dim_cb(cfg)[0]
    if prop == "codebook_size":
        return _parse_model_dim_cb(cfg)[1]
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


def resolve_data_dir(run: dict) -> Path | None:
    """Resolve a run's configured data directory to an existing path."""
    data_dir = run["config"].get("data_dir")
    if not data_dir:
        return None

    configured_path = Path(data_dir)
    candidates = []
    if configured_path.is_absolute():
        candidates.append(configured_path)
    else:
        candidates.extend([
            Path.cwd() / configured_path,
            Path(run["dir"]) / configured_path,
            PROJECT_ROOT / configured_path,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


@lru_cache(maxsize=None)
def count_dataset_sequences(data_dir: str, sequence_length: int) -> int | None:
    """Approximate dataset size from saved session metadata."""
    root = Path(data_dir)
    meta_files = sorted(root.rglob("session_*.meta.json"))
    if not meta_files:
        return None

    total_sequences = 0
    for meta_file in meta_files:
        with open(meta_file) as f:
            meta = json.load(f)
        num_frames = meta.get("num_frames", 0)
        if num_frames >= sequence_length:
            total_sequences += (num_frames - sequence_length) // sequence_length + 1
    return total_sequences


def estimate_train_sample_count(run: dict) -> int | None:
    """Estimate the effective training-set size used by a run."""
    cfg = run["config"]
    data_dir = resolve_data_dir(run)
    if data_dir is None:
        return None

    from mario_world_model.config import SEQUENCE_LENGTH

    sequence_length = int(cfg.get("sequence_length", SEQUENCE_LENGTH) or SEQUENCE_LENGTH)
    total_sequences = count_dataset_sequences(str(data_dir), sequence_length)
    if total_sequences is None:
        return None

    overfit_n = int(cfg.get("overfit_n", 0) or 0)
    if overfit_n > 0:
        return min(overfit_n, total_sequences)

    eval_samples = int(cfg.get("eval_samples", 0) or 0)
    if eval_samples > 0 and total_sequences > eval_samples:
        return total_sequences - eval_samples
    return total_sequences


def get_total_samples(metrics: list[dict]) -> int | None:
    """Return the latest recorded total sample count."""
    for metric in reversed(metrics):
        total_samples = metric.get("total_samples")
        if total_samples is not None:
            return int(total_samples)
    return None


def get_total_steps(metrics: list[dict]) -> int:
    """Return the latest recorded optimizer step count."""
    last = metrics[-1]
    if "total_steps" in last:
        return int(last["total_steps"])
    return int(last.get("step", 0))


def format_dataset_seen_pct(run: dict) -> str:
    """Render approximate dataset coverage as effective epochs in percent."""
    train_sample_count = estimate_train_sample_count(run)
    total_samples = get_total_samples(run["metrics"])
    if not train_sample_count or total_samples is None:
        return "?"
    return f"{(100.0 * total_samples / train_sample_count):.1f}%"


def format_dataset_seen_per_step(run: dict) -> str:
    """Render approximate dataset coverage normalized by steps taken."""
    train_sample_count = estimate_train_sample_count(run)
    total_samples = get_total_samples(run["metrics"])
    total_steps = get_total_steps(run["metrics"])
    if not train_sample_count or total_samples is None or total_steps <= 0:
        return "?"
    seen_pct = 100.0 * total_samples / train_sample_count
    return f"{(seen_pct / total_steps):.4f}%"


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

        _, recon_values = get_recon_values(metrics)
        min_recon = min(recon_values) if recon_values else float("nan")

        # Final codebook usage
        cb_entries = [m for m in metrics if "codebook_usage" in m]
        final_cb = cb_entries[-1]["codebook_usage"] if cb_entries else None
        max_cb = max(m["codebook_usage"] for m in cb_entries) if cb_entries else None

        # Eval metrics (from final entry if present)
        last = metrics[-1]
        eval_recon = last.get("eval_recon_loss")
        eval_pixel_acc = last.get("eval_pixel_accuracy")
        eval_cb_util = last.get("eval_codebook_utilization")
        eval_entropy = last.get("eval_code_entropy_bits")
        eval_max_entropy = last.get("eval_max_entropy_bits")

        # Last step / elapsed
        total_steps = get_total_steps(metrics)
        elapsed_s = last.get("elapsed_s", 0)

        steps_per_s = total_steps / elapsed_s if elapsed_s > 0 else float("nan")

        # LFQ loss breakdown from final entry
        final_lfq_aux = last.get("lfq_aux_loss")
        final_lfq_commitment = last.get("lfq_commitment")
        final_lfq_batch_entropy = last.get("lfq_batch_entropy")
        final_lfq_per_sample_entropy = last.get("lfq_per_sample_entropy")
        final_lr = last.get("lr")
        samples_per_sec = last.get("samples_per_sec")

        dim, cb_size = _parse_model_dim_cb(cfg)

        rows.append(OrderedDict([
            ("name", run["name"]),
            ("dim", dim),
            ("cb_size", cb_size),
            ("num_params", f"{cfg.get('num_parameters', 0):,}"),
            ("batch_size", cfg.get("batch_size", "?")),
            ("min_recon", f"{min_recon:.4f}"),
            ("eval_recon", f"{eval_recon:.4f}" if eval_recon is not None else ""),
            ("eval_px_acc", f"{eval_pixel_acc:.4f}" if eval_pixel_acc is not None else ""),
            ("final_cb", final_cb),
            ("cb_util", f"{eval_cb_util:.1%}" if eval_cb_util is not None else ""),
            ("entropy", f"{eval_entropy:.2f}/{eval_max_entropy:.2f}" if eval_entropy is not None and eval_max_entropy is not None else ""),
            ("lfq_aux", f"{final_lfq_aux:.4f}" if final_lfq_aux is not None else ""),
            ("batch_H", f"{final_lfq_batch_entropy:.2f}" if final_lfq_batch_entropy is not None else ""),
            ("commit", f"{final_lfq_commitment:.4f}" if final_lfq_commitment is not None else ""),
            ("lr", f"{final_lr:.2e}" if final_lr is not None else ""),
            ("steps", total_steps),
            ("elapsed", _format_elapsed(elapsed_s)),
            ("samp/s", f"{samples_per_sec:.1f}" if samples_per_sec is not None else ""),
        ]))

    # Sort by eval_recon (if available), then min_smoothed_recon as fallback
    def _sort_key(r):
        eval_val = float(r["eval_recon"]) if r["eval_recon"] else float("inf")
        recon_val = float(r["min_recon"])
        return (eval_val, recon_val)
    rows.sort(key=_sort_key)
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
    has_lfq = any("lfq_batch_entropy" in m for run in runs for m in run["metrics"])
    has_lr = any("lr" in m for run in runs for m in run["metrics"])
    has_eval = any("eval_recon_loss" in m for run in runs for m in run["metrics"])
    axis_key, axis_scale, axis_label = get_x_axis_metadata(x_axis)
    run_styles = build_run_styles(runs)

    # Build panel list: recon loss always, then optional panels
    panels = ["recon"]
    if has_eval:
        panels.append("eval")
    if has_cb:
        panels.append("codebook")
    if has_lfq:
        panels.append("lfq")
    if has_lr:
        panels.append("lr")
    nrows = len(panels)
    height_ratios = [2] + [1] * (nrows - 1)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 4 * nrows), sharex=True,
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
    if "codebook" in panels:
        ax = axes[panels.index("codebook")]
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

    # --- Eval recon loss + pixel accuracy ---
    if "eval" in panels:
        ax = axes[panels.index("eval")]
        ax2 = ax.twinx()
        for run in sorted_runs:
            style = run_styles[run["name"]]
            x_vals, eval_recon = get_metric_series(run["metrics"], "eval_recon_loss", x_axis)
            if x_vals:
                ax.plot(x_vals, eval_recon, label=run["name"],
                        color=style["color"], linestyle=style["linestyle"],
                        marker="o", markersize=3,
                        alpha=0.9, linewidth=style["linewidth"])
            x_vals, px_acc = get_metric_series(run["metrics"], "eval_pixel_accuracy", x_axis)
            if x_vals:
                ax2.plot(x_vals, px_acc,
                         color=style["color"], linestyle=":",
                         marker="s", markersize=2,
                         alpha=0.5, linewidth=max(0.8, style["linewidth"] * 0.6))
        ax.set_ylabel("Eval Recon Loss")
        ax.set_yscale("log")
        ax2.set_ylabel("Pixel Accuracy", alpha=0.5)
        ax.set_title("Eval Recon Loss (solid) + Pixel Accuracy (dotted)")
        if show_legend:
            ax.legend(fontsize=7, ncol=2, frameon=True, handlelength=3.0)
        ax.grid(True, alpha=0.3)

    # --- LFQ loss breakdown ---
    if "lfq" in panels:
        ax = axes[panels.index("lfq")]
        ax2 = ax.twinx()
        for run in sorted_runs:
            style = run_styles[run["name"]]
            x_vals, batch_h = get_metric_series(run["metrics"], "lfq_batch_entropy", x_axis)
            if x_vals:
                ax.plot(x_vals, smooth(batch_h, smooth_window),
                        label=f"{run['name']} (batch H)",
                        color=style["color"], linestyle=style["linestyle"],
                        alpha=0.9, linewidth=style["linewidth"])
            x_vals, commit = get_metric_series(run["metrics"], "lfq_commitment", x_axis)
            if x_vals:
                ax2.plot(x_vals, smooth(commit, smooth_window),
                         label=f"{run['name']} (commit)",
                         color=style["color"], linestyle=":",
                         alpha=0.5, linewidth=max(0.8, style["linewidth"] * 0.6))
        ax.set_ylabel("Batch Entropy")
        ax2.set_ylabel("Commitment", alpha=0.5)
        ax.set_title("Quantizer Health: Batch Entropy (solid) + Commitment (dotted)")
        if show_legend:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, ncol=2, frameon=True, handlelength=3.0)
        ax.grid(True, alpha=0.3)

    # --- Learning rate schedule ---
    if "lr" in panels:
        ax = axes[panels.index("lr")]
        marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"]
        lr_series = []
        for run in sorted_runs:
            x_vals, lr_vals = get_metric_series(run["metrics"], "lr", x_axis)
            if x_vals:
                lr_series.append((run, x_vals, lr_vals))

        if lr_series:
            x_min = min(min(x_vals) for _, x_vals, _ in lr_series)
            x_max = max(max(x_vals) for _, x_vals, _ in lr_series)
            x_span = max(x_max - x_min, 1.0)
            jitter = x_span * 0.0015 if len(lr_series) > 1 else 0.0
            offsets = np.linspace(-0.5, 0.5, len(lr_series)) if len(lr_series) > 1 else np.array([0.0])

            for idx, (run, x_vals, lr_vals) in enumerate(lr_series):
                style = run_styles[run["name"]]
                x_plot = [x + offsets[idx] * jitter for x in x_vals] if jitter > 0 else x_vals
                marker = marker_cycle[idx % len(marker_cycle)]
                markevery = max(1, len(x_plot) // 10)
                ax.plot(
                    x_plot,
                    lr_vals,
                    label=run["name"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    alpha=0.9,
                    linewidth=style["linewidth"],
                    marker=marker,
                    markersize=6,
                    markerfacecolor="none",
                    markeredgewidth=1.4,
                    markevery=markevery,
                )
        ax.set_ylabel("Learning Rate")
        ax.set_title("LR Schedule (tiny x-offsets for overlap visibility)")
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(-3, -3))
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

    unique_props: list[str] = []
    prop_colors: dict[str | None, tuple[float, float, float, float]] = {}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=str, nargs="+", default=["results/"],
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
    return parser.parse_args()


def main():
    args = parse_args()

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
