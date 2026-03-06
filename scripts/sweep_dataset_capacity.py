#!/usr/bin/env python3
"""
Max-Dataset-Size Sweep for MAGVIT-2 Video Tokenizer
====================================================

For each model configuration (mirrored from sweep_arch.sh), uses binary search
over dataset sizes (via --overfit-n) to find the largest N-sample subset the
model can fit to recon_loss < threshold (default 0.0005).

Algorithm per model:
  1. Train on dataset_size=2 to convergence (patience-based plateau).
  2. If best recon >= threshold → model fails entirely → max_size=0.
  3. Else binary search: low=2, high=total_samples.
     - mid = (low+high)//2, train on mid samples.
     - If best recon < threshold → low=mid; else → high=mid.
     - Stop when high - low <= 1.
  4. Record max_dataset_size = low.

Cross-model early exit: if a model cannot beat the threshold on a dataset
size <= the current best from a prior model, skip remaining search.

Usage:
  python scripts/sweep_dataset_capacity.py                        # full sweep
  python scripts/sweep_dataset_capacity.py --dry-run              # print plan
  python scripts/sweep_dataset_capacity.py --filter dim32         # subset
  python scripts/sweep_dataset_capacity.py --resume               # skip done
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Model configurations — mirrored from sweep_arch.sh
# ---------------------------------------------------------------------------

BASELINE_LAYERS = "residual,compress_space,residual,compress_space,residual,compress_space"
DEEP = "residual,residual,compress_space,residual,residual,compress_space,residual,residual,compress_space"
ATTN_SPACE = "residual,compress_space,attend_space,residual,compress_space,attend_space,residual,compress_space,attend_space"
LIN_ATTN = "residual,compress_space,residual,compress_space,linear_attend_space,residual,compress_space,linear_attend_space"
ATTN_TIME = "residual,compress_space,attend_time,residual,compress_space,attend_time,residual,compress_space,attend_time"
FULL_ATTN = "residual,compress_space,residual,compress_space,attend_space,attend_time,residual,compress_space,attend_space,attend_time"


@dataclass
class ModelConfig:
    name: str
    init_dim: int
    codebook_size: int
    layers: str
    layer_type: str = "baseline"


DIMS = [32, 64]
CODEBOOK_SIZES = [8192, 16384, 32768, 65536]
LAYER_CONFIGS = {
    "baseline":   BASELINE_LAYERS,
    "deep":       DEEP,
    "attn_space": ATTN_SPACE,
    "lin_attn":   LIN_ATTN,
    "attn_time":  ATTN_TIME,
    "full_attn":  FULL_ATTN,
}

# Complexity rank: lower = simpler (used for sorting sweep order)
LAYER_COMPLEXITY = {
    "baseline":   0,
    "deep":       1,
    "lin_attn":   2,
    "attn_space": 3,
    "attn_time":  4,
    "full_attn":  5,
}

MODEL_CONFIGS = [
    ModelConfig(
        name=f"dim{dim}_cb{cb}{'_' + lname if lname != 'baseline' else ''}",
        init_dim=dim,
        codebook_size=cb,
        layers=layers,
        layer_type=lname,
    )
    for dim in DIMS
    for cb in CODEBOOK_SIZES
    for lname, layers in LAYER_CONFIGS.items()
]


# ---------------------------------------------------------------------------
# Per-model result tracking
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    dataset_size: int
    best_recon: float
    passed: bool
    run_name: str
    elapsed_s: float = 0.0


@dataclass
class ModelResult:
    name: str
    max_dataset_size: int = 0
    best_recon_at_max: float = float("inf")
    trials: list = field(default_factory=list)
    total_time_s: float = 0.0
    early_exited: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_chunk(filepath: str, seq_len: int) -> int:
    """Count valid samples in a single chunk file."""
    import numpy as np
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
        dones = np.array(npz["dones"])
        count = 0
        for i in range(num_seqs):
            for t in range(0, total_t - seq_len + 1, seq_len):
                if not np.any(dones[i, t : t + seq_len - 1]):
                    count += 1
        return count
    except Exception:
        return 0


def get_total_samples(data_dir: str, seq_len: int = 16) -> int:
    """Count valid samples the same way MarioVideoDataset does."""
    import concurrent.futures
    import glob

    chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    total = 0
    with concurrent.futures.ThreadPoolExecutor() as pool:
        futures = [pool.submit(_count_chunk, f, seq_len) for f in chunk_files]
        for fut in concurrent.futures.as_completed(futures):
            total += fut.result()
    return total


def read_best_recon(metrics_path: str) -> float:
    """Read the minimum eval recon_loss from a training run's metrics.json."""
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
        # Prefer eval_recon_loss (computed in eval mode at val steps)
        recon_values = [m["eval_recon_loss"] for m in metrics if "eval_recon_loss" in m]
        if not recon_values:
            recon_values = [m["recon_loss"] for m in metrics if "recon_loss" in m]
        if recon_values:
            return min(recon_values)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return float("inf")


def run_name_for(model_name: str, dataset_size: int) -> str:
    return f"{model_name}_n{dataset_size}"


def build_train_cmd(
    model: ModelConfig,
    dataset_size: int,
    *,
    data_dir: str,
    output_dir: str,
    max_steps: int,
    lr: float,
    patience: int,
    val_interval: int,
    batch_size: int,
    auto_batch: bool,
    seed: int,
    num_workers: int,
    threshold: float = 0.0,
) -> list[str]:
    rname = run_name_for(model.name, dataset_size)
    cmd = [
        sys.executable, "scripts/train_magvit.py",
        "--data-dir", data_dir,
        "--run-name", rname,
        "--overfit-n", str(dataset_size),
        "--init-dim", str(model.init_dim),
        "--codebook-size", str(model.codebook_size),
        "--layers", model.layers,
        "--epochs", "9999",
        "--max-steps", str(max_steps),
        "--lr", str(lr),
        "--threshold", str(threshold),
        "--max-patience", str(patience),
        "--val-interval", str(val_interval),
        "--seed", str(seed),
        "--num-workers", str(num_workers),
        "--output-dir", output_dir,
    ]
    if auto_batch:
        cmd.append("--auto-batch-size")
    else:
        cmd.extend(["--batch-size", str(batch_size)])
    if dataset_size <= batch_size:
        cmd.append("--no-shuffle")
    return cmd


def run_trial(
    model: ModelConfig,
    dataset_size: int,
    *,
    dry_run: bool,
    resume: bool,
    **train_kwargs,
) -> TrialResult:
    """Train a single (model, dataset_size) trial and return the result."""
    rname = run_name_for(model.name, dataset_size)
    metrics_path = os.path.join(
        train_kwargs["output_dir"], rname, "metrics.json"
    )

    # Resume: if metrics already exist, reuse them
    if resume and os.path.exists(metrics_path):
        best_recon = read_best_recon(metrics_path)
        print(f"  [resume] {rname}: reusing existing metrics (best_recon={best_recon:.6f})")
        return TrialResult(
            dataset_size=dataset_size,
            best_recon=best_recon,
            passed=best_recon < train_kwargs.get("threshold", 0.0005),
            run_name=rname,
        )

    cmd = build_train_cmd(model, dataset_size, **train_kwargs)

    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return TrialResult(
            dataset_size=dataset_size,
            best_recon=float("nan"),
            passed=False,
            run_name=rname,
        )

    print(f"  Training {rname} (n={dataset_size}) ...")
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

    return TrialResult(
        dataset_size=dataset_size,
        best_recon=best_recon,
        passed=passed,
        run_name=rname,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Binary search
# ---------------------------------------------------------------------------

def binary_search_max_size(
    model: ModelConfig,
    total_samples: int,
    *,
    threshold: float,
    global_best_max: int,
    dry_run: bool,
    resume: bool,
    **train_kwargs,
) -> ModelResult:
    """Find the maximum dataset size this model can fit below threshold."""
    result = ModelResult(name=model.name)
    t0 = time.time()
    train_kwargs["threshold"] = threshold

    print(f"\n{'='*64}")
    print(f"Model: {model.name}  (dim={model.init_dim}  cb={model.codebook_size})")
    print(f"{'='*64}")

    # ── Step 1: Can the model fit the smallest dataset (size=2)? ─────────
    trial = run_trial(model, 2, dry_run=dry_run, resume=resume, **train_kwargs)
    result.trials.append(asdict(trial))

    if dry_run:
        # In dry-run mode, simulate the full binary search to show all commands
        low, high = 2, total_samples
        while high - low > 1:
            mid = (low + high) // 2
            trial = run_trial(model, mid, dry_run=True, resume=resume, **train_kwargs)
            result.trials.append(asdict(trial))
            high = mid  # arbitrary direction for dry-run illustration
        result.total_time_s = 0.0
        return result

    if not trial.passed:
        print(f"  {model.name}: FAILED on size=2 — skipping entirely.")
        result.total_time_s = time.time() - t0
        return result

    # ── Step 2: Early exit — can the model beat the global champion? ─────
    low = 2       # last known passing size
    high = total_samples  # upper bound (exclusive — fails or untested)

    if global_best_max > 2:
        trial_best = run_trial(
            model, global_best_max, dry_run=False, resume=resume, **train_kwargs
        )
        result.trials.append(asdict(trial_best))
        if trial_best.passed:
            low = global_best_max
        else:
            high = global_best_max
            print(f"  {model.name}: cannot fit size={global_best_max} "
                  f"(current best). Searching in [2, {global_best_max}).")

    # ── Step 3: Binary search ────────────────────────────────────────────

    while high - low > 1:
        mid = (low + high) // 2
        trial = run_trial(model, mid, dry_run=False, resume=resume, **train_kwargs)
        result.trials.append(asdict(trial))
        if trial.passed:
            low = mid
        else:
            high = mid

    result.max_dataset_size = low
    # Find the best recon at the max passing size
    for t in result.trials:
        if t["dataset_size"] == low and t["passed"]:
            result.best_recon_at_max = t["best_recon"]
            break
    result.total_time_s = time.time() - t0

    print(f"\n  >> {model.name}: max_dataset_size = {result.max_dataset_size} "
          f"(best_recon={result.best_recon_at_max:.6f}, "
          f"{len(result.trials)} trials, {result.total_time_s:.0f}s)")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Binary-search sweep: find max dataset size each model can memorize."
    )
    parser.add_argument("--data-dir", default="data/human_play")
    parser.add_argument("--output-dir", default="checkpoints/capacity_sweep")
    parser.add_argument("--threshold", type=float, default=0.0008,
                        help="Recon loss threshold for pass/fail (default: 0.0008)")
    parser.add_argument("--max-patience", type=int, default=5,
                        help="Patience for plateau-based convergence (default: 5)")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Gradient step budget per trial (default: 5000)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--auto-batch", action="store_true",
                        help="Use auto batch size detection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=48)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Override upper bound for dataset size "
                             "(0 = auto-detect from data)")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only run models whose name contains this substring")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--resume", action="store_true",
                        help="Skip trials whose metrics.json already exists")
    args = parser.parse_args()

    # ── Select model configs ─────────────────────────────────────────────
    # Sort from simplest to most complex: layer complexity, then dim, then codebook
    configs = sorted(
        MODEL_CONFIGS,
        key=lambda c: (LAYER_COMPLEXITY[c.layer_type], c.init_dim, c.codebook_size),
    )
    if args.filter:
        configs = [c for c in configs if args.filter in c.name]
        if not configs:
            print(f"No models match filter '{args.filter}'.")
            sys.exit(1)

    # ── Determine total dataset size ─────────────────────────────────────
    if args.max_samples > 0:
        total_samples = args.max_samples
        print(f"Upper bound: {total_samples} (user override)")
    elif args.dry_run:
        total_samples = 18000  # placeholder for dry-run
        print(f"Upper bound: ~{total_samples} (estimate, dry-run)")
    else:
        print("Counting valid samples in dataset...")
        total_samples = get_total_samples(args.data_dir)
        print(f"Upper bound: {total_samples} valid samples")

    if total_samples < 2:
        print("Dataset too small.")
        sys.exit(1)

    # ── Shared training kwargs ───────────────────────────────────────────
    train_kwargs = dict(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        lr=args.lr,
        patience=args.max_patience,
        val_interval=args.val_interval,
        batch_size=args.batch_size,
        auto_batch=args.auto_batch,
        seed=args.seed,
        num_workers=args.num_workers,
        threshold=args.threshold,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "capacity_sweep_results.json")

    # ── Run sweep ────────────────────────────────────────────────────────
    print(f"\nSweeping {len(configs)} model(s), threshold={args.threshold}")
    all_results: list[ModelResult] = []
    global_best_max = 0

    for model in configs:
        mr = binary_search_max_size(
            model,
            total_samples,
            global_best_max=global_best_max,
            dry_run=args.dry_run,
            resume=args.resume,
            **train_kwargs,
        )
        all_results.append(mr)

        if mr.max_dataset_size > global_best_max:
            global_best_max = mr.max_dataset_size
            print(f"  ** New global best: {model.name} with max_size={global_best_max}")

    # ── Save summary ─────────────────────────────────────────────────────
    summary = {
        "threshold": args.threshold,
        "total_samples": total_samples,
        "results": [asdict(r) for r in all_results],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print leaderboard ────────────────────────────────────────────────
    ranked = sorted(all_results, key=lambda r: r.max_dataset_size, reverse=True)
    print(f"\n{'='*64}")
    print("LEADERBOARD — Max Dataset Size (recon < {:.4f})".format(args.threshold))
    print(f"{'='*64}")
    print(f"{'Rank':<6}{'Model':<30}{'Max Size':>10}{'Best Recon':>12}{'Trials':>8}{'Time':>10}")
    print("-" * 76)
    for i, r in enumerate(ranked, 1):
        recon_str = f"{r.best_recon_at_max:.6f}" if r.max_dataset_size > 0 else "N/A"
        time_str = f"{r.total_time_s:.0f}s" if r.total_time_s > 0 else "-"
        print(f"{i:<6}{r.name:<30}{r.max_dataset_size:>10}{recon_str:>12}"
              f"{len(r.trials):>8}{time_str:>10}")

    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
