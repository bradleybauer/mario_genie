#!/usr/bin/env python3
"""Compare metrics across multiple training runs.

Scans one or more results directories recursively for folders containing
config.json + metrics.json, then produces:
    1. A summary table (printed and optionally saved as CSV).
    2. A comparison plot for reconstruction loss and related metrics.

Usage:
        python scripts/compare_sweeps.py --results-dir checkpoints/magvit2
        python scripts/compare_sweeps.py --results-dir checkpoints/magvit2 checkpoints/capacity_runs
        python scripts/compare_sweeps.py --results-dir checkpoints/magvit2 -o run_comparison.png
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
plt.style.use("dark_background")
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.config import SEQUENCE_LENGTH


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
    """Build matplotlib rcParams for readable high-DPI plots."""
    scale = max(font_scale, 0.5)
    return {key: value * scale for key, value in BASE_PLOT_RC.items()}


def scale_figsize(width: float, height: float, figure_scale: float) -> tuple[float, float]:
    """Scale a base figure size without repeating the math at each call site."""
    scale = max(figure_scale, 0.5)
    return width * scale, height * scale


def legend_columns(num_series: int) -> int:
    """Choose a compact legend layout based on the number of compared runs."""
    if num_series <= 3:
        return 1
    if num_series <= 6:
        return 2
    return 3


def get_metric_series(metrics: list[dict], key: str, x_axis: str) -> tuple[list[float], list[float]]:
    """Return (x_values, y_values) for a given metric key."""
    axis_key, axis_scale, _ = get_x_axis_metadata(x_axis)
    points = [m for m in metrics if key in m and axis_key in m]
    if not points:
        return [], []
    return [m[axis_key] * axis_scale for m in points], [m[key] for m in points]


def get_recon_values(metrics: list[dict]) -> tuple[str | None, list[float]]:
    """Return the preferred reconstruction-loss values for summaries and sorting."""
    for key in ("smoothed_recon_loss", "recon_loss", "video_recon_loss"):
        values = [m[key] for m in metrics if key in m]
        if values:
            return key, values
    return None, []


def get_latest_eval_metric(metrics: list[dict], *keys: str) -> float | None:
    """Return the latest eval metric value for the first matching key."""
    eval_row = next((metric for metric in reversed(metrics) if metric.get("type") == "eval"), None)
    if eval_row is None:
        return None

    for key in keys:
        value = eval_row.get(key)
        if value is not None:
            return value
    return None


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
RUN_TIMESTAMP_PATTERN = re.compile(r"^(?P<stem>.+?)_(?P<date>20\d{6})_(?P<time>\d{6})$")


def _cfg_get(cfg: dict, key: str, default=None):
    """Read a config value from either the legacy flat schema or nested sections."""
    if key in cfg:
        return cfg.get(key, default)

    for section_name in ("training", "model", "data", "runtime"):
        section = cfg.get(section_name)
        if isinstance(section, dict) and key in section:
            return section[key]

    return default


def _cfg_model_name(cfg: dict) -> str:
    """Return a string model identifier when one exists."""
    model_value = cfg.get("model")
    if isinstance(model_value, str):
        return model_value

    model_name = cfg.get("model_name")
    if isinstance(model_name, str):
        return model_name

    return ""


def _truncate_middle(text: str, max_length: int = 30) -> str:
    """Keep labels compact without losing both the prefix and suffix."""
    if len(text) <= max_length:
        return text

    keep = max_length - 3
    left = keep // 2
    right = keep - left
    return f"{text[:left]}...{text[-right:]}"


def _simplify_run_label(name: str) -> tuple[str, str | None]:
    """Turn a run directory name into a compact plot label."""
    base_name = name.rsplit("/", 1)[-1]
    time_suffix = None

    match = RUN_TIMESTAMP_PATTERN.match(base_name)
    if match:
        base_name = match.group("stem")
        time_suffix = match.group("time")

    label = base_name.replace("_", " ")
    label = re.sub(r"\bdim(\d+)\b", r"d\1", label)
    label = re.sub(r"\bvideo latent dit\b", "latent dit", label)
    label = re.sub(r"\bspatiotemporal\b", "spatiotemp", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label, time_suffix


def build_run_labels(runs: list[dict], max_length: int = 30) -> dict[str, str]:
    """Build readable, stable, and mostly unique labels for plots."""
    grouped: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for run in runs:
        label, time_suffix = _simplify_run_label(run["name"])
        grouped[label].append((run["name"], time_suffix))

    labels: dict[str, str] = {}
    for label, entries in grouped.items():
        for index, (run_name, time_suffix) in enumerate(entries, start=1):
            unique_label = label
            if len(entries) > 1:
                if time_suffix:
                    unique_label = f"{label} {time_suffix}"
                else:
                    unique_label = f"{label} #{index}"
            labels[run_name] = _truncate_middle(unique_label, max_length=max_length)

    return labels


def _parse_model_dim_cb(cfg: dict) -> tuple[str, str]:
    """Extract init_dim and codebook_size from config, falling back to model name."""
    dim = _cfg_get(cfg, "init_dim")
    cb = _cfg_get(cfg, "codebook_size")

    if dim is None:
        for dim_key in ("latent_dim", "latent_channels", "hidden_dim", "base_channels"):
            dim = _cfg_get(cfg, dim_key)
            if dim is not None:
                break

    if dim is None or cb is None:
        model_name = _cfg_model_name(cfg)
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
    return str(_cfg_get(cfg, prop, "?"))


def abbreviate_layer_component(component: str) -> str:
    """Shorten verbose layer names for compact summary tables."""
    name, sep, value = component.partition(":")
    abbreviated_name = LAYER_ABBREVIATIONS.get(name, name)
    if not sep:
        return abbreviated_name
    return f"{abbreviated_name}{sep}{value}"


def resolve_data_dir(run: dict) -> Path | None:
    """Resolve a run's configured data directory to an existing path."""
    data_dir = _cfg_get(run["config"], "data_dir")
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

    sequence_length = int(_cfg_get(cfg, "sequence_length", SEQUENCE_LENGTH) or SEQUENCE_LENGTH)
    total_sequences = count_dataset_sequences(str(data_dir), sequence_length)
    if total_sequences is None:
        return None

    overfit_n = int(_cfg_get(cfg, "overfit_n", 0) or 0)
    if overfit_n > 0:
        return min(overfit_n, total_sequences)

    eval_samples = int(_cfg_get(cfg, "eval_samples", 0) or 0)
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


def filter_runs(
    runs: list[dict],
    include_filter: str | None = None,
    exclude_filter: str | None = None,
) -> list[dict]:
    """Filter runs by optional include/exclude name substrings."""
    filtered_runs = runs
    if include_filter:
        filtered_runs = [run for run in filtered_runs if include_filter in run["name"]]
    if exclude_filter:
        filtered_runs = [run for run in filtered_runs if exclude_filter not in run["name"]]
    return filtered_runs


def summarise(runs: list[dict]) -> list[OrderedDict]:
    """Extract key stats from each run into a flat row."""
    rows = []
    for run in runs:
        cfg = run["config"]
        metrics = run["metrics"]

        _, recon_values = get_recon_values(metrics)
        min_recon = min(recon_values) if recon_values else float("nan")

        # Eval metrics (from the latest eval entry if present)
        last = metrics[-1]
        eval_recon = get_latest_eval_metric(metrics, "eval_recon_loss", "recon_loss", "video_recon_loss")
        eval_pixel_acc = get_latest_eval_metric(metrics, "eval_pixel_accuracy")

        # Last step / elapsed
        total_steps = get_total_steps(metrics)
        elapsed_s = last.get("elapsed_s", 0)

        final_lr = last.get("lr")
        samples_per_sec = last.get("samples_per_sec")

        dim, _ = _parse_model_dim_cb(cfg)

        rows.append(OrderedDict([
            ("name", run["name"]),
            ("dim", dim),
            ("num_params", f"{_cfg_get(cfg, 'num_parameters', 0):,}"),
            ("batch_size", _cfg_get(cfg, "batch_size", "?")),
            ("min_recon", f"{min_recon:.4f}"),
            ("eval_recon", f"{eval_recon:.4f}" if eval_recon is not None else ""),
            ("eval_px_acc", f"{eval_pixel_acc:.4f}" if eval_pixel_acc is not None else ""),
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


def plot_comparison(
    runs: list[dict],
    output_path: str | None = None,
    x_axis: str = "step",
    smooth_window: int = 1,
    show_legend: bool = True,
    font_scale: float = DEFAULT_FONT_SCALE,
    figure_scale: float = DEFAULT_FIGURE_SCALE,
) -> None:
    if not runs:
        print("Nothing to plot.")
        return

    has_smoothed_recon = any("smoothed_recon_loss" in m for run in runs for m in run["metrics"])
    has_lr = any("lr" in m for run in runs for m in run["metrics"])
    _, _, axis_label = get_x_axis_metadata(x_axis)
    run_styles = build_run_styles(runs)
    run_labels = build_run_labels(runs)

    # Build panel list: recon loss always, then optional panels that vary over the x-axis.
    panels = ["recon"]
    if has_lr:
        panels.append("lr")
    nrows = len(panels)
    height_ratios = [2] + [1] * (nrows - 1)

    with plt.rc_context(build_plot_rc(font_scale)):
        fig, axes = plt.subplots(
            nrows,
            1,
            figsize=scale_figsize(12, 4 * nrows, figure_scale),
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        if nrows == 1:
            axes = [axes]

        marker_large = max(6.0, 3.8 * font_scale)
        legend_ncols = legend_columns(len(runs))

        # Sort runs by the best available reconstruction curve so the legend is ordered.
        sorted_runs = sorted(
            runs,
            key=lambda r: min(get_recon_series(r["metrics"], x_axis)[2], default=float("inf")),
        )
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=run_styles[run["name"]]["color"],
                linestyle=run_styles[run["name"]]["linestyle"],
                linewidth=run_styles[run["name"]]["linewidth"],
            )
            for run in sorted_runs
        ]
        legend_labels = [run_labels[run["name"]] for run in sorted_runs]

        # --- Recon loss ---
        ax = axes[0]
        for run in sorted_runs:
            _, x_values, recon = get_recon_series(run["metrics"], x_axis)
            if x_values:
                style = run_styles[run["name"]]
                label = run_labels[run["name"]]
                line = ax.plot(
                    x_values,
                    smooth(recon, smooth_window),
                    label=label,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    alpha=0.9,
                    linewidth=style["linewidth"],
                )[0]
        ylabel = "Smoothed Reconstruction Loss" if has_smoothed_recon else "Reconstruction Loss"
        title = "Smoothed Reconstruction Loss by Run" if has_smoothed_recon else "Reconstruction Loss by Run"
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_title(title)
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
                        label=run_labels[run["name"]],
                        color=style["color"],
                        linestyle=style["linestyle"],
                        alpha=0.9,
                        linewidth=style["linewidth"],
                        marker=marker,
                        markersize=marker_large,
                        markerfacecolor="none",
                        markeredgewidth=1.4,
                        markevery=markevery,
                    )
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule by Run")
            ax.ticklabel_format(axis="y", style="scientific", scilimits=(-3, -3))
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel(axis_label)
        if show_legend and legend_handles:
            legend_rows = (len(legend_handles) + legend_ncols - 1) // legend_ncols
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncol=legend_ncols,
                frameon=True,
                handlelength=2.6,
                columnspacing=1.2,
            )
            top_margin = max(0.82, 0.98 - 0.03 * legend_rows)
            fig.tight_layout(rect=(0, 0, 1, top_margin), pad=max(1.0, 1.0 * font_scale))
        else:
            fig.tight_layout(pad=max(1.0, 1.0 * font_scale))

        if output_path:
            fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
            print(f"Saved plot to {output_path}")
        else:
            plt.show()


def plot_faceted(
    runs: list[dict],
    facet_by: str,
    output_path: str | None = None,
    x_axis: str = "step",
    smooth_window: int = 1,
    show_legend: bool = True,
    font_scale: float = DEFAULT_FONT_SCALE,
    figure_scale: float = DEFAULT_FIGURE_SCALE,
) -> None:
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
    run_labels = build_run_labels(runs)
    has_smoothed = any("smoothed_recon_loss" in m for run in runs for m in run["metrics"])
    axis_key, axis_scale, axis_label = get_x_axis_metadata(x_axis)

    with plt.rc_context(build_plot_rc(font_scale)):
        fig, axes = plt.subplots(1, ncols, figsize=scale_figsize(6 * ncols, 5, figure_scale), sharey=True)
        if ncols == 1:
            axes = [axes]
        sorted_runs = sorted(
            runs,
            key=lambda r: min(get_recon_series(r["metrics"], x_axis)[2], default=float("inf")),
        )
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=run_styles[run["name"]]["color"],
                linestyle=run_styles[run["name"]]["linestyle"],
                linewidth=run_styles[run["name"]]["linewidth"],
            )
            for run in sorted_runs
        ]
        legend_labels = [run_labels[run["name"]] for run in sorted_runs]

        for ax, key in zip(axes, sorted_keys):
            group_runs = sorted(
                groups[key],
                key=lambda r: min(get_recon_series(r["metrics"], x_axis)[2], default=float("inf")),
            )
            for run in group_runs:
                _, x_values, recon = get_recon_series(run["metrics"], x_axis)
                if x_values:
                    style = run_styles[run["name"]]
                    label = run_labels[run["name"]]
                    line = ax.plot(
                        x_values,
                        smooth(recon, smooth_window),
                        label=label,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        alpha=0.9,
                        linewidth=style["linewidth"],
                    )[0]
            ax.set_title(str(key))
            ax.set_yscale("log")
            ax.set_xlabel(axis_label)
            ax.grid(True, alpha=0.3)

        ylabel = "Smoothed Recon Loss" if has_smoothed else "Recon Loss"
        axes[0].set_ylabel(ylabel)
        fig.suptitle(f"Reconstruction Loss by {facet_by}", y=1.02)
        if show_legend and legend_handles:
            legend_ncols = legend_columns(len(legend_handles))
            legend_rows = (len(legend_handles) + legend_ncols - 1) // legend_ncols
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncol=legend_ncols,
                frameon=True,
                handlelength=2.6,
                columnspacing=1.2,
            )
            top_margin = max(0.82, 0.96 - 0.03 * legend_rows)
            fig.tight_layout(rect=(0, 0, 1, top_margin), pad=max(1.0, 1.0 * font_scale))
        else:
            fig.tight_layout(pad=max(1.0, 1.0 * font_scale))

        if output_path:
            fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
            print(f"Saved plot to {output_path}")
        else:
            plt.show()


def plot_bar(
    runs: list[dict],
    output_path: str | None = None,
    color_by: str | None = None,
    show_legend: bool = True,
    font_scale: float = DEFAULT_FONT_SCALE,
    figure_scale: float = DEFAULT_FIGURE_SCALE,
) -> None:
    """Horizontal bar chart of best reconstruction loss per run."""
    if not runs:
        print("Nothing to plot.")
        return

    run_labels = build_run_labels(runs)
    entries = []
    for run in runs:
        _, recon_losses = get_recon_values(run["metrics"])
        best = min(recon_losses) if recon_losses else float("inf")
        label = run_labels[run["name"]]
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

    with plt.rc_context(build_plot_rc(font_scale)):
        fig, ax = plt.subplots(figsize=scale_figsize(10, max(4, len(labels) * 0.45), figure_scale))
        y_pos = range(len(labels))
        ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # best (smallest) at top
        ax.set_xlabel("Best Reconstruction Loss")
        ax.set_title("Best Reconstruction Loss by Run")
        ax.grid(True, axis="x", alpha=0.3)

        if color_by and show_legend:
            legend_handles = [Patch(facecolor=prop_colors[p], label=f"{color_by}={p}") for p in unique_props]
            ax.legend(handles=legend_handles, loc="lower right")

        fig.tight_layout(pad=max(1.0, 1.0 * font_scale))

        if output_path:
            fig.savefig(output_path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
            print(f"Saved plot to {output_path}")
        else:
            plt.show()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=str, nargs="+", default=["checkpoints/"],
                        help="One or more directories containing run folders anywhere under them")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save comparison plot to file (e.g. run_comparison.png)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save summary table as CSV")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only include runs whose name contains this substring")
    parser.add_argument("--exclude-filter", type=str, default=None,
                        help="Exclude runs whose name contains this substring")
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
    parser.add_argument("--font-scale", type=float, default=DEFAULT_FONT_SCALE,
                        help="Scale all plot text for higher-DPI displays")
    parser.add_argument("--figure-scale", type=float, default=DEFAULT_FIGURE_SCALE,
                        help="Scale the base figure size used for plots")
    return parser.parse_args(argv)


def main():
    args = parse_args()

    runs = filter_runs(
        discover_runs(args.results_dir),
        include_filter=args.filter,
        exclude_filter=args.exclude_filter,
    )

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
        plot_bar(
            runs,
            output_path=args.output,
            color_by=args.color_by,
            show_legend=show_legend,
            font_scale=args.font_scale,
            figure_scale=args.figure_scale,
        )
    elif args.facet:
        plot_faceted(
            runs,
            args.facet,
            output_path=args.output,
            x_axis=args.x_axis,
            smooth_window=args.smooth,
            show_legend=show_legend,
            font_scale=args.font_scale,
            figure_scale=args.figure_scale,
        )
    else:
        plot_comparison(
            runs,
            output_path=args.output,
            x_axis=args.x_axis,
            smooth_window=args.smooth,
            show_legend=show_legend,
            font_scale=args.font_scale,
            figure_scale=args.figure_scale,
        )


if __name__ == "__main__":
    main()
