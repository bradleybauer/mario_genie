#!/usr/bin/env python
"""Plot training metrics from a metrics.json file produced by train_magvit.py."""

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def plot_metrics(metrics, output_path=None):
    steps = [m["step"] for m in metrics]
    loss = [m["loss"] for m in metrics]
    recon = [m["recon_loss"] for m in metrics]

    # Codebook usage is only logged at validation steps
    cb_steps = [m["step"] for m in metrics if "codebook_usage" in m]
    cb_usage = [m["codebook_usage"] for m in metrics if "codebook_usage" in m]

    has_cb = len(cb_steps) > 0
    nrows = 3 if has_cb else 2

    fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows), sharex=True)

    # --- Loss ---
    ax = axes[0]
    ax.plot(steps, loss, label="Total loss", alpha=0.7, linewidth=0.8)
    ax.plot(steps, recon, label="Recon loss", alpha=0.7, linewidth=0.8)
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    # --- Loss delta (rate of improvement) ---
    ax = axes[1]
    recon_arr = np.array(recon)
    window = max(1, len(recon_arr) // 50)
    if len(recon_arr) > window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(recon_arr, kernel, mode="valid")
        delta = -np.diff(smoothed)  # positive = improving
        delta_steps = steps[window:window + len(delta)]
        ax.plot(delta_steps, delta, linewidth=0.8, alpha=0.7, color="tab:orange")
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Δ Recon loss (smoothed)")
        ax.set_title("Rate of Improvement (positive = improving)")
        ax.grid(True, alpha=0.3)

    # --- Codebook usage ---
    if has_cb:
        ax = axes[2]
        ax.plot(cb_steps, cb_usage, marker="o", markersize=3, color="tab:purple")
        ax.set_ylabel("Unique codes used")
        ax.set_title("Codebook Usage")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")

    # Mark epoch boundaries
    epoch_starts = {}
    for m in metrics:
        e = m["epoch"]
        if e not in epoch_starts:
            epoch_starts[e] = m["step"]
    for ax in axes:
        for e, s in epoch_starts.items():
            if e > 0:
                ax.axvline(s, color="grey", linestyle="--", alpha=0.3, linewidth=0.7)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", nargs="?", default="checkpoints/magvit2/metrics.json",
                        help="Path to metrics.json (default: checkpoints/magvit2/metrics.json)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save plot to file instead of showing (e.g. metrics.png)")
    args = parser.parse_args()

    metrics = load_metrics(args.metrics_json)
    if not metrics:
        print("No metrics found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(metrics)} entries, steps {metrics[0]['step']}–{metrics[-1]['step']}")
    plot_metrics(metrics, output_path=args.output)


if __name__ == "__main__":
    main()
