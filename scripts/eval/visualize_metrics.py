#!/usr/bin/env python3
"""Visualize training metrics from a checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use("dark_background")
import torch


def load_metrics(checkpoint_dir: str) -> tuple[list[dict], dict]:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        # Try as a name under checkpoints/
        checkpoint_dir = Path("checkpoints") / checkpoint_dir
    if not checkpoint_dir.exists():
        print(f"Error: checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    ckpt_path = checkpoint_dir / "latest.pt"
    if not ckpt_path.exists():
        print(f"Error: no latest.pt found in {checkpoint_dir}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics", [])

    config_path = checkpoint_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    return metrics, config


HW_METRICS = {"gpu_mem_pct", "samples_per_sec", "steps_per_sec", "throughput_mb_per_sec"}
ALWAYS_SKIP = {"type", "step", "frame_size"}
LOG_SCALE_PATTERNS = {"loss", "kl", "grad_norm"}


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
            sample = next((m[name] for m in entries if name in m), None)
            if isinstance(sample, (int, float)):
                numeric.append(name)
                break
    return numeric


def _smooth(vals: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        out.append(sum(vals[start:i + 1]) / (i - start + 1))
    return out


def plot_metrics(
    metrics: list[dict],
    config: dict,
    metric_names: list[str] | None = None,
    smooth: int = 10,
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
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for idx, name in enumerate(numeric_metrics):
        ax = axes[idx // cols][idx % cols]

        train_steps = [m["step"] for m in train if name in m]
        train_vals = [m[name] for m in train if name in m]

        if train_steps:
            ax.plot(train_steps, train_vals, alpha=0.3, color="C0", linewidth=0.8, label="train")
            if smooth > 1 and len(train_vals) >= smooth:
                ax.plot(train_steps, _smooth(train_vals, smooth), color="C0", linewidth=1.5, label=f"train (smooth={smooth})")

        eval_steps = [m["step"] for m in evals if name in m]
        eval_vals = [m[name] for m in evals if name in m]
        if eval_steps:
            ax.plot(eval_steps, eval_vals, color="C1", linewidth=2, marker="o", markersize=4, label="eval")

        ax.set_xlabel("step")
        ax.set_ylabel(name)
        ax.set_title(name)
        if _use_log_scale(name):
            ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_compare(
    runs: list[tuple[str, list[dict], dict]],
    metric_names: list[str] | None = None,
    smooth: int = 10,
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
    fig.suptitle(f"Comparing {len(runs)} runs", fontsize=14, fontweight="bold")

    for idx, name in enumerate(numeric_metrics):
        ax = axes[idx // cols][idx % cols]

        for run_idx, (run_name, _, config) in enumerate(runs):
            color = f"C{run_idx}"
            label = _make_label(run_name, config)
            train = all_train[run_idx]
            evals = all_evals[run_idx]

            train_steps = [m["step"] for m in train if name in m]
            train_vals = [m[name] for m in train if name in m]

            if train_steps:
                ax.plot(train_steps, train_vals, alpha=0.15, color=color, linewidth=0.6)
                if smooth > 1 and len(train_vals) >= smooth:
                    ax.plot(train_steps, _smooth(train_vals, smooth), color=color, linewidth=1.5, label=label)
                else:
                    ax.plot(train_steps, train_vals, color=color, linewidth=1.5, label=label)

            eval_steps = [m["step"] for m in evals if name in m]
            eval_vals = [m[name] for m in evals if name in m]
            if eval_steps:
                ax.plot(eval_steps, eval_vals, color=color, linewidth=2, marker="o", markersize=4, linestyle="--", label=f"{label} (eval)")

        ax.set_xlabel("step")
        ax.set_ylabel(name)
        ax.set_title(name)
        if _use_log_scale(name):
            ax.set_yscale("log")
        ax.legend(fontsize=7)
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
    parser.add_argument("--hw", action="store_true", help="Include hardware perf metrics (gpu_mem_pct, samples/steps per sec)")
    parser.add_argument("--eval", action="store_true", help="Include eval metrics")
    parser.add_argument("--show-lr", action="store_true", help="Include learning-rate metrics in auto-discovery")
    args = parser.parse_args()

    if len(args.checkpoints) == 1:
        metrics, config = load_metrics(args.checkpoints[0])
        plot_metrics(
            metrics,
            config,
            metric_names=args.metrics,
            smooth=args.smooth,
            show_hw=args.hw,
            show_eval=getattr(args, "eval"),
            show_lr=args.show_lr,
        )
    else:
        runs = []
        for ckpt in args.checkpoints:
            metrics, config = load_metrics(ckpt)
            run_name = Path(ckpt).name
            runs.append((run_name, metrics, config))
        plot_compare(
            runs,
            metric_names=args.metrics,
            smooth=args.smooth,
            show_hw=args.hw,
            show_eval=getattr(args, "eval"),
            show_lr=args.show_lr,
        )


if __name__ == "__main__":
    main()
