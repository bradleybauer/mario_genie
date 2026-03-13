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

from mario_world_model.dataset_paths import find_chunk_files
from mario_world_model.model_configs import MODEL_CONFIGS, ModelConfig


@dataclass
class ModelRunResult:
    name: str
    total_samples: int
    best_recon: float
    passed: bool
    run_name: str
    elapsed_s: float = 0.0


def _count_scene_cuts(npz, seq_idx: int, t_start: int, seq_len: int) -> int:
    if "world" in npz.files and "stage" in npz.files:
        world_window = np.asarray(npz["world"][seq_idx, t_start : t_start + seq_len])
        stage_window = np.asarray(npz["stage"][seq_idx, t_start : t_start + seq_len])
        if len(world_window) <= 1:
            return 0
        transitions = (world_window[1:] != world_window[:-1]) | (stage_window[1:] != stage_window[:-1])
        return int(np.count_nonzero(transitions))

    if "dones" in npz.files:
        done_window = np.asarray(npz["dones"][seq_idx, t_start : t_start + seq_len], dtype=bool)
        return int(np.count_nonzero(done_window[:-1]))

    return 0


def _count_chunk(filepath: str, seq_len: int) -> int:
    """Count fixed-length samples with fewer than two scene cuts."""
    try:
        meta_path = filepath.replace(".npz", ".meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as mf:
                meta = json.load(mf)
            num_seqs = meta["num_sequences"]
            total_t = meta["sequence_length"]
        else:
            npz = np.load(filepath, mmap_mode="r")
            num_seqs, total_t = npz["frames"].shape[0], npz["frames"].shape[1]

        if total_t < seq_len:
            return 0

        npz = np.load(filepath, mmap_mode="r")
        count = 0
        for i in range(num_seqs):
            for t in range(0, total_t - seq_len + 1, seq_len):
                if _count_scene_cuts(npz, i, t, seq_len) >= 2:
                    continue
                count += 1
        return count
    except Exception:
        return 0


def get_total_samples(data_dir: str, seq_len: int = 16) -> int:
    """Count samples the same way MarioVideoDataset does."""
    import concurrent.futures

    chunk_files = find_chunk_files(data_dir)
    total = 0
    with concurrent.futures.ThreadPoolExecutor() as pool:
        futures = [pool.submit(_count_chunk, f, seq_len) for f in chunk_files]
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
    *,
    data_dir: str,
    output_dir: str,
    max_minutes: float,
    lr: float,
    warmup_steps: int,
    patience: float,
    val_interval: int,
    max_batch_size: int,
    seed: int,
    num_workers: int,
    threshold: float = 0.0,
) -> list[str]:
    rname = model.name
    cmd = [
        sys.executable, "scripts/train_magvit.py",
        "--data-dir", data_dir,
        "--run-name", rname,
        "--init-dim", str(model.init_dim),
        "--codebook-size", str(model.codebook_size),
        "--layers", model.layers,
        "--max-minutes", str(max_minutes),
        "--lr", str(lr),
        "--warmup-steps", str(warmup_steps),
        "--threshold", str(threshold),
        "--max-patience", str(patience),
        "--val-interval", str(val_interval),
        "--seed", str(seed),
        "--num-workers", str(num_workers),
        "--output-dir", output_dir,
        "--auto-batch-size",
        "--max-batch-size", str(max_batch_size),
        "--no-preload",
    ]
    return cmd


def run_model(
    model: ModelConfig,
    total_samples: int,
    *,
    dry_run: bool,
    resume: bool,
    **train_kwargs,
) -> ModelRunResult:
    """Train one model configuration on the full dataset."""
    rname = model.name
    metrics_path = os.path.join(train_kwargs["output_dir"], rname, "metrics.json")

    if resume and os.path.exists(metrics_path):
        best_recon = read_best_recon(metrics_path)
        print(f"  [resume] {rname}: reusing existing metrics (best_recon={best_recon:.6f})")
        return ModelRunResult(
            name=model.name,
            total_samples=total_samples,
            best_recon=best_recon,
            passed=best_recon < train_kwargs.get("threshold", 0.0005),
            run_name=rname,
        )

    cmd = build_train_cmd(model, **train_kwargs)

    if dry_run:
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
    threshold = train_kwargs.get("threshold", 0.0005)
    passed = best_recon < threshold
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
    parser.add_argument("--max-patience", type=float, default=60*4,
                        help="Minutes without improvement before early stop (default: 150)")
    parser.add_argument("--max-minutes", type=float, default=60*4,
                        help="Wall-clock minute budget per trial (default: 150)")
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Linear warmup steps passed to train_magvit (default: 1000)")
    parser.add_argument("--val-interval", type=int, default=100)
    parser.add_argument("--max-batch-size", type=int, default=8,
                        help="Cap auto-detected batch size (default: 8)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Accepted for CLI compatibility; ignored by this script")
    parser.add_argument("--initial-global-best", type=int, default=0,
                        help="Accepted for CLI compatibility; ignored by this script")
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
    args = parser.parse_args()

    if args.initial_global_best < 0:
        parser.error("--initial-global-best must be non-negative")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error(f"--shard-index must be in [0, {args.num_shards})")

    if args.initial_global_best > 0:
        print("[init] --initial-global-best is ignored in single-pass mode")
    if args.max_samples > 0:
        print("[init] --max-samples is ignored in single-pass mode")

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

    train_kwargs = dict(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_minutes=args.max_minutes,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        patience=args.max_patience * 60,
        val_interval=args.val_interval,
        max_batch_size=args.max_batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        threshold=args.threshold,
    )

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
        result = run_model(
            model,
            total_samples,
            dry_run=args.dry_run,
            resume=args.resume,
            **train_kwargs,
        )
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