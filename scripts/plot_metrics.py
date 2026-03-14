#!/usr/bin/env python
"""Plot training metrics from a metrics.json file produced by train_magvit.py."""

import argparse
import json
import os
import re
import sys

import matplotlib.pyplot as plt


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def plot_metrics(all_metrics, labels, max_codes, output_path=None):
    has_cb = any("codebook_usage" in m for metrics in all_metrics for m in metrics)
    nrows = 2 if has_cb else 1

    fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    # --- Recon Loss (log) ---
    ax_loss_log = axes[0]
    for metrics, label in zip(all_metrics, labels):
        steps = [m["step"] for m in metrics]
        recon = [m["recon_loss"] for m in metrics]
        ax_loss_log.plot(steps, recon, label=label, alpha=0.7, linewidth=0.8)

    ax_loss_log.set_ylabel("Loss")
    ax_loss_log.set_yscale("log")
    ax_loss_log.legend()
    ax_loss_log.set_title("Reconstruction Loss (Log)")
    ax_loss_log.grid(True, alpha=0.3)

    # --- Codebook usage ---
    if has_cb:
        ax_cb = axes[1]
        for metrics, label, max_c in zip(all_metrics, labels, max_codes):
            cb_steps = [m["step"] for m in metrics if "codebook_usage" in m]
            cb_usage = [m["codebook_usage"] for m in metrics if "codebook_usage" in m]
            if cb_steps:
                line = ax_cb.plot(cb_steps, cb_usage, marker="o", markersize=3, label=label)[0]
                if max_c is not None:
                    ax_cb.axhline(max_c, color=line.get_color(), linestyle="--", alpha=0.5, label=f"Max ({max_c})")
        ax_cb.set_ylabel("Unique codes used")
        ax_cb.set_title("Codebook Usage")
        ax_cb.grid(True, alpha=0.3)
        ax_cb.legend()

    axes[-1].set_xlabel("Step")

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", nargs="*", default=["checkpoints/magvit2/metrics.json"],
                        help="Paths to metrics.json (default: checkpoints/magvit2/metrics.json)")
    parser.add_argument("-l", "--labels", nargs="*", default=None,
                        help="Labels for each metrics file")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save plot to file instead of showing (e.g. metrics.png)")
    args = parser.parse_args()

    default_labels = []
    max_codes = []
    for path in args.metrics_json:
        parent_dir = os.path.basename(os.path.dirname(os.path.abspath(path)))
        default_labels.append(parent_dir)
        match = re.search(r'\d+', parent_dir)
        max_codes.append(int(match.group()) if match else None)

    all_metrics = []
    labels = args.labels if args.labels and len(args.labels) == len(args.metrics_json) else default_labels

    for path in args.metrics_json:
        metrics = load_metrics(path)
        all_metrics.append(metrics)

    if not all_metrics or not any(all_metrics):
        print("No metrics found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(all_metrics)} metrics files.")
    plot_metrics(all_metrics, labels, max_codes, output_path=args.output)


if __name__ == "__main__":
    main()
