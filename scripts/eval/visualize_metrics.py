#!/usr/bin/env python3
"""Visualize training metrics from a checkpoint."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.plot_style import apply_plot_style
apply_plot_style()


def load_metrics(checkpoint_dir: str, strict: bool = True) -> tuple[list[dict], dict] | None:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        # Try as a name under checkpoints/
        checkpoint_dir = Path("checkpoints") / checkpoint_dir
    if not checkpoint_dir.exists():
        msg = f"checkpoint directory not found: {checkpoint_dir}"
        if strict:
            print(f"Error: {msg}")
            sys.exit(1)
        print(f"Warning: skipping {msg}")
        return None

    ckpt_path = checkpoint_dir / "latest.pt"
    if not ckpt_path.exists():
        msg = f"no latest.pt found in {checkpoint_dir}"
        if strict:
            print(f"Error: {msg}")
            sys.exit(1)
        print(f"Warning: skipping {msg}")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics", [])

    config_path = checkpoint_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    return metrics, config


HW_METRICS = {"gpu_mem_pct", "samples_per_sec", "steps_per_sec", "throughput_mb_per_sec"}
ALWAYS_SKIP = {"type", "step", "frame_size", "elapsed_s", "error_buffer_size"}
LOG_SCALE_PATTERNS = {"loss", "kl", "grad_norm"}
MODEL_COLOR_PALETTE = list(plt.get_cmap("tab20").colors)


def _is_numeric_metric_value(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_lr_metric(name: str) -> bool:
    lower = name.lower()
    return lower == "lr" or lower.startswith("lr_") or lower.endswith("_lr") or "learning_rate" in lower


def _use_log_scale(name: str) -> bool:
    lower = name.lower()
    return any(p in lower for p in LOG_SCALE_PATTERNS)


def _make_label(name: str, config: dict) -> str:
    """Short label from run name or checkpoint dir name."""
    parts = [name]
    training_cfg = config.get("training", {})
    if "hidden_dim" in training_cfg:
        parts.append(f"h={training_cfg['hidden_dim']}")
    if "latent_dim" in training_cfg:
        parts.append(f"z={training_cfg['latent_dim']}")
    if "batch_size" in training_cfg:
        parts.append(f"bs={training_cfg['batch_size']}")
    return " | ".join(parts)


def _model_color_key(name: str, config: dict) -> str:
    """Stable key used to pick a deterministic color per run/model pair."""
    model_name = str(config.get("model_name", ""))
    run_name = str(name)
    if model_name and model_name != run_name:
        return f"{model_name}:{run_name}"
    return run_name


def _color_for_model(name: str, config: dict):
    key = _model_color_key(name, config)
    # Use a stable hash so colors don't change when run ordering changes.
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(MODEL_COLOR_PALETTE)
    return MODEL_COLOR_PALETTE[idx]


def _discover_metrics(
    all_entries: list[list[dict]],
    metric_names: list[str] | None,
    show_hw: bool,
    show_lr: bool,
) -> list[str]:
    skip_keys = set(ALWAYS_SKIP)
    if not show_hw:
        skip_keys |= HW_METRICS
    if metric_names is not None:
        return metric_names
    all_keys: set[str] = set()
    for entries in all_entries:
        for m in entries:
            all_keys.update(m.keys())
    candidates = sorted(all_keys - skip_keys)
    # Keep only numeric
    numeric = []
    for name in candidates:
        if not show_lr and _is_lr_metric(name):
            continue
        for entries in all_entries:
            sample = next((m[name] for m in entries if name in m and _is_numeric_metric_value(m[name])), None)
            if _is_numeric_metric_value(sample):
                numeric.append(name)
                break
    return numeric


def _numeric_series(entries: list[dict], name: str) -> tuple[list[int], list[float]]:
    points = [(m["step"], m[name]) for m in entries if name in m and _is_numeric_metric_value(m[name])]
    return [step for step, _ in points], [value for _, value in points]


def _smooth(vals: list[float], window: int, passes: int = 1) -> list[float]:
    out = vals
    for _ in range(max(1, passes)):
        smoothed = []
        for i in range(len(out)):
            start = max(0, i - window + 1)
            smoothed.append(sum(out[start:i + 1]) / (i - start + 1))
        out = smoothed
    return out


def plot_metrics(
    metrics: list[dict],
    config: dict,
    metric_names: list[str] | None = None,
    smooth: int = 10,
    smooth_passes: int = 1,
    show_hw: bool = False,
    show_eval: bool = False,
    show_lr: bool = False,
):
    train = [m for m in metrics if m.get("type") == "train"]
    evals = [m for m in metrics if m.get("type") == "eval"] if show_eval else []

    if not train and not evals:
        print("No metrics found in checkpoint.")
        sys.exit(1)

    numeric_metrics = _discover_metrics([train, evals], metric_names, show_hw, show_lr)
    if not numeric_metrics:
        print("No numeric metrics to plot.")
        sys.exit(1)

    n = len(numeric_metrics)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    title = _make_label(config.get("model_name", "unknown"), config)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    fig.suptitle(title, fontweight="bold")
    run_color = _color_for_model(config.get("model_name", "unknown"), config)

    for idx, name in enumerate(numeric_metrics):
        ax = axes[idx // cols][idx % cols]

        train_steps, train_vals = _numeric_series(train, name)

        if train_steps:
            if smooth > 1 and len(train_vals) >= smooth:
                ax.plot(train_steps, train_vals, alpha=0.3, color=run_color, linewidth=0.8)
                ax.plot(
                    train_steps,
                    _smooth(train_vals, smooth, smooth_passes),
                    color=run_color,
                    linewidth=1.5,
                    label="train",
                )
            else:
                ax.plot(train_steps, train_vals, color=run_color, linewidth=1.5, label="train")

        eval_steps, eval_vals = _numeric_series(evals, name)
        if eval_steps:
            ax.plot(eval_steps, eval_vals, color=run_color, linewidth=2, marker="o", markersize=4, linestyle="--", label="eval")

        ax.set_xlabel("step")
        ax.set_ylabel(name)
        ax.set_title(name)
        if _use_log_scale(name):
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_compare(
    runs: list[tuple[str, list[dict], dict]],
    metric_names: list[str] | None = None,
    smooth: int = 10,
    smooth_passes: int = 1,
    show_hw: bool = False,
    show_eval: bool = False,
    show_lr: bool = False,
):
    all_train = []
    all_evals = []
    for _, metrics, _ in runs:
        all_train.append([m for m in metrics if m.get("type") == "train"])
        all_evals.append([m for m in metrics if m.get("type") == "eval"] if show_eval else [])

    numeric_metrics = _discover_metrics(all_train + all_evals, metric_names, show_hw, show_lr)
    if not numeric_metrics:
        print("No numeric metrics to plot.")
        sys.exit(1)

    n = len(numeric_metrics)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    fig.suptitle(f"Comparing {len(runs)} runs", fontweight="bold")

    for idx, name in enumerate(numeric_metrics):
        ax = axes[idx // cols][idx % cols]

        for run_idx, (run_name, _, config) in enumerate(runs):
            color = _color_for_model(run_name, config)
            label = _make_label(run_name, config)
            train = all_train[run_idx]
            evals = all_evals[run_idx]

            train_steps, train_vals = _numeric_series(train, name)

            if train_steps:
                ax.plot(train_steps, train_vals, alpha=0.15, color=color, linewidth=0.6)
                if smooth > 1 and len(train_vals) >= smooth:
                    ax.plot(
                        train_steps,
                        _smooth(train_vals, smooth, smooth_passes),
                        color=color,
                        linewidth=1.5,
                        label=label,
                    )
                else:
                    ax.plot(train_steps, train_vals, color=color, linewidth=1.5, label=label)

            eval_steps, eval_vals = _numeric_series(evals, name)
            if eval_steps:
                ax.plot(eval_steps, eval_vals, color=color, linewidth=2, marker="o", markersize=4, linestyle="--", label=f"{label} (eval)")

        ax.set_xlabel("step")
        ax.set_ylabel(name)
        ax.set_title(name)
        if _use_log_scale(name):
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics from one or more checkpoints")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint directories or run names (pass multiple to compare)")
    parser.add_argument("--metrics", nargs="+", help="Specific metrics to plot (default: all numeric metrics)")
    parser.add_argument("--smooth", type=int, default=10, help="Smoothing window size (default: 10)")
    parser.add_argument("--smooth-passes", type=int, default=2, help="How many times to apply smoothing (default: 2)")
    parser.add_argument("--hw", action="store_true", help="Include hardware perf metrics (gpu_mem_pct, samples/steps per sec)")
    parser.add_argument("--eval", action="store_true", help="Include eval metrics")
    parser.add_argument("--show-lr", action="store_true", help="Include learning-rate metrics in auto-discovery")
    args = parser.parse_args()

    if args.smooth_passes < 1:
        parser.error("--smooth-passes must be >= 1")

    if len(args.checkpoints) == 1:
        loaded = load_metrics(args.checkpoints[0], strict=True)
        if loaded is None:
            # strict=True exits, but keep type-checkers happy.
            sys.exit(1)
        metrics, config = loaded
        plot_metrics(
            metrics,
            config,
            metric_names=args.metrics,
            smooth=args.smooth,
            smooth_passes=args.smooth_passes,
            show_hw=args.hw,
            show_eval=getattr(args, "eval"),
            show_lr=args.show_lr,
        )
    else:
        runs = []
        for ckpt in args.checkpoints:
            loaded = load_metrics(ckpt, strict=False)
            if loaded is None:
                continue
            metrics, config = loaded
            run_name = Path(ckpt).name
            runs.append((run_name, metrics, config))

        if not runs:
            print("Error: no valid checkpoints to compare.")
            sys.exit(1)

        plot_compare(
            runs,
            metric_names=args.metrics,
            smooth=args.smooth,
            smooth_passes=args.smooth_passes,
            show_hw=args.hw,
            show_eval=getattr(args, "eval"),
            show_lr=args.show_lr,
        )


if __name__ == "__main__":
    main()
