#!/usr/bin/env python3
"""
Single-Pass Sweep for Open-Genie-Style MAGVIT Tokenizers
========================================================

Trains each Open-Genie-inspired MAGVIT tokenizer configuration exactly once
on the full dataset instead of binary-searching for the largest memorized
subset.

The script still counts the available samples for logging and summary output,
but it does not pass --overfit-n to the trainer.

This script intentionally preserves most of the sweep CLI surface from the
binary-search version so existing launch commands remain usable. Binary-search-
or subset-specific arguments are accepted for compatibility but are ignored.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.dataset_paths import find_session_files
from mario_world_model.model_configs import MODEL_CONFIGS, ModelConfig


@dataclass
class ModelRunResult:
    name: str
    total_samples: int
    best_recon: float
    passed: bool
    run_name: str
    elapsed_s: float = 0.0


def _count_file(filepath: str, seq_len: int) -> int:
    """Count fixed-length samples in a session file."""
    try:
        npz = np.load(filepath, mmap_mode="r")
        frames = npz["frames"]
        if frames.ndim != 4:
            return 0
        total_t = frames.shape[0]
        if total_t < seq_len:
            return 0
        return (total_t - seq_len) // seq_len + 1
    except Exception:
        return 0


def get_total_samples(data_dir: str, seq_len: int = 16) -> int:
    """Count samples the same way MarioVideoDataset does."""
    import concurrent.futures

    session_files = find_session_files(data_dir)
    total = 0
    with concurrent.futures.ThreadPoolExecutor() as pool:
        futures = [pool.submit(_count_file, f, seq_len) for f in session_files]
        for fut in concurrent.futures.as_completed(futures):
            total += fut.result()
    return total


def read_best_recon(metrics_path: str) -> float:
    """Read the minimum recon metric from a training run's metrics.json."""
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
        recon_values = [m["eval_recon_loss"] for m in metrics if "eval_recon_loss" in m]
        if not recon_values:
            recon_values = [m["smoothed_recon_loss"] for m in metrics if "smoothed_recon_loss" in m]
        if not recon_values:
            recon_values = [m["recon_loss"] for m in metrics if "recon_loss" in m]
        if recon_values:
            return min(recon_values)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return float("inf")


def build_train_cmd(
    model: ModelConfig,
    args: argparse.Namespace,
    extra_train_args: list[str],
) -> list[str]:
    return [
        sys.executable, "scripts/train_magvit.py",
        "--run-name", model.name,
        "--model", model.name,
        *extra_train_args,
    ]


def run_model(
    model: ModelConfig,
    total_samples: int,
    args: argparse.Namespace,
    extra_train_args: list[str],
) -> ModelRunResult:
    """Train one model configuration on the full dataset."""
    rname = model.name
    metrics_path = os.path.join(args.output_dir, rname, "metrics.json")

    if args.resume and os.path.exists(metrics_path):
        best_recon = read_best_recon(metrics_path)
        print(f"  [resume] {rname}: reusing existing metrics (best_recon={best_recon:.6f})")
        return ModelRunResult(
            name=model.name,
            total_samples=total_samples,
            best_recon=best_recon,
            passed=best_recon < args.threshold,
            run_name=rname,
        )

    cmd = build_train_cmd(model, args, extra_train_args)

    if args.dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return ModelRunResult(
            name=model.name,
            total_samples=total_samples,
            best_recon=float("nan"),
            passed=False,
            run_name=rname,
        )

    print(f"  Training {rname} on full dataset ({total_samples} samples) ...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [WARN] {rname} exited with code {result.returncode}")

    best_recon = read_best_recon(metrics_path)
    passed = best_recon < args.threshold
    status = "PASS" if passed else "FAIL"
    print(f"  {rname}: best_recon={best_recon:.6f}  [{status}]  ({elapsed:.0f}s)")

    return ModelRunResult(
        name=model.name,
        total_samples=total_samples,
        best_recon=best_recon,
        passed=passed,
        run_name=rname,
        elapsed_s=elapsed,
    )


def load_completed_results(
    summary_path: str,
    *,
    threshold: float,
    total_samples: int,
) -> dict[str, ModelRunResult]:
    """Load completed model runs when the saved summary matches this sweep."""
    if not os.path.exists(summary_path):
        return {}

    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[resume] Ignoring unreadable summary {summary_path}: {exc}")
        return {}

    saved_threshold = summary.get("threshold")
    saved_total_samples = summary.get("total_samples")
    if saved_threshold != threshold or saved_total_samples != total_samples:
        print(
            f"[resume] Ignoring summary {summary_path}: "
            f"expected threshold={threshold}, total_samples={total_samples}; "
            f"found threshold={saved_threshold}, total_samples={saved_total_samples}"
        )
        return {}

    results = {}
    for item in summary.get("results", []):
        name = item.get("name")
        if not name:
            continue
        results[name] = ModelRunResult(
            name=name,
            total_samples=item.get("total_samples", total_samples),
            best_recon=item.get("best_recon", float("inf")),
            passed=item.get("passed", False),
            run_name=item.get("run_name", name),
            elapsed_s=item.get("elapsed_s", 0.0),
        )
    return results


def write_summary(
    summary_path: str,
    threshold: float,
    total_samples: int,
    total_models: int,
    results: list[ModelRunResult],
) -> None:
    """Persist sweep progress so resume can skip completed models."""
    summary = {
        "threshold": threshold,
        "total_samples": total_samples,
        "total_models": total_models,
        "results": [asdict(r) for r in results],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-pass sweep: train each model config once on the full dataset."
    )
    parser.add_argument("--data-dir", default="data/nes")
    parser.add_argument("--output-dir", default="checkpoints/model_config_sweep_genie")
    parser.add_argument("--threshold", type=float, default=0.0008,
                        help="Recon loss threshold for pass/fail (default: 0.0008)")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only run models whose name contains this substring")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--resume", action="store_true",
                        help="Skip trials whose metrics.json already exists")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="This worker's shard index for multi-machine sweeps (0-based)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of workers sharing the sweep")
    # All other args (--lr, --max-minutes, --max-patience, etc.) are
    # forwarded directly to train_magvit.py.
    args, extra_train_args = parser.parse_known_args()

    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error(f"--shard-index must be in [0, {args.num_shards})")

    if args.num_shards > 1:
        configs = sorted(MODEL_CONFIGS, key=lambda c: c.name)
    else:
        configs = MODEL_CONFIGS.copy()
        random.shuffle(configs)

    if args.filter:
        configs = [c for c in configs if args.filter in c.name]
        if not configs:
            print(f"No models match filter '{args.filter}'.")
            sys.exit(1)

    if args.num_shards > 1:
        configs = [c for i, c in enumerate(configs) if i % args.num_shards == args.shard_index]
        if not configs:
            print(f"No models assigned to shard {args.shard_index}/{args.num_shards}.")
            sys.exit(0)

    if args.dry_run:
        total_samples = 18000
        print(f"Dataset size: ~{total_samples} samples (estimate, dry-run)")
    else:
        print("Counting samples in dataset (skipping windows with >=2 scene cuts)...")
        total_samples = get_total_samples(args.data_dir)
        print(f"Dataset size: {total_samples} samples")

    if total_samples < 1:
        print("Dataset too small.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "model_config_sweep_results.json")
    completed_results = load_completed_results(
        summary_path,
        threshold=args.threshold,
        total_samples=total_samples,
    ) if args.resume else {}

    print(f"\nSweeping {len(configs)} model(s) on full dataset, threshold={args.threshold}")
    all_results: list[ModelRunResult] = []

    for model in configs:
        if model.name in completed_results:
            result = completed_results[model.name]
            all_results.append(result)
            print(
                f"\n[resume] Skipping completed model {model.name} "
                f"(best_recon={result.best_recon:.6f})"
            )
            continue

        print(f"\n{'=' * 64}")
        print(f"Model: {model.name}  (dim={model.init_dim}  cb={model.codebook_size})")
        print(f"{'=' * 64}")
        result = run_model(model, total_samples, args, extra_train_args)
        all_results.append(result)
        write_summary(summary_path, args.threshold, total_samples, len(configs), all_results)

    ranked = sorted(all_results, key=lambda r: (r.passed, -r.best_recon), reverse=True)
    print(f"\n{'=' * 64}")
    print("LEADERBOARD — Fixed Dataset Sweep (recon < {:.4f})".format(args.threshold))
    print(f"{'=' * 64}")
    print(f"{'Rank':<6}{'Model':<30}{'Passed':>8}{'Best Recon':>12}{'Time':>10}")
    print("-" * 66)
    for i, r in enumerate(ranked, 1):
        recon_str = f"{r.best_recon:.6f}" if np.isfinite(r.best_recon) else "N/A"
        time_str = f"{r.elapsed_s:.0f}s" if r.elapsed_s > 0 else "-"
        passed_str = "yes" if r.passed else "no"
        print(f"{i:<6}{r.name:<30}{passed_str:>8}{recon_str:>12}{time_str:>10}")

    write_summary(summary_path, args.threshold, total_samples, len(configs), all_results)
    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()