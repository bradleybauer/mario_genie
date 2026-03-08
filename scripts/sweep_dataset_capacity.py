#!/usr/bin/env python3
"""
Max-Dataset-Size Sweep for MAGVIT-2 Video Tokenizer
====================================================

For each model configuration (mirrored from sweep_arch.sh), uses binary search
over dataset sizes (via --overfit-n) to find the largest N-sample subset the
model can fit to recon_loss < threshold (default 0.0005).

Algorithm per model:
    1. Run a binary search over dataset sizes to find the largest passing size.
    2. If a current best model capacity exists, use that size as the first probe.
    3. If that probe fails, stop early for this model.
    4. If the probe passes, continue searching upward.
    5. Record max_dataset_size.

Usage:
  python scripts/sweep_dataset_capacity.py                        # full sweep
  python scripts/sweep_dataset_capacity.py --dry-run              # print plan
  python scripts/sweep_dataset_capacity.py --filter dim32         # subset
  python scripts/sweep_dataset_capacity.py --resume               # skip done
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
# ---------------------------------------------------------------------------
# Model configurations — mirrored from sweep_arch.sh
# ---------------------------------------------------------------------------

BASELINE_LAYERS = "residual,compress_space,residual,compress_space,residual,compress_space"
DEEP = "residual,residual,compress_space,residual,residual,compress_space,residual,residual,compress_space"
ATTN_SPACE = "residual,compress_space,attend_space,residual,compress_space,attend_space,residual,compress_space,attend_space"
LIN_ATTN = "residual,compress_space,residual,compress_space,linear_attend_space,residual,compress_space,linear_attend_space"
ATTN_TIME = "residual,compress_space,attend_time,residual,compress_space,attend_time,residual,compress_space,attend_time"
FULL_ATTN = "residual,compress_space,residual,compress_space,attend_space,attend_time,residual,compress_space,attend_space,attend_time"
GENIE_SMALL = "consecutive_residual:2,compress_space:64,consecutive_residual:2,compress_time:96,compress_space:128,consecutive_residual:2,compress_time:128,compress_space:128,consecutive_residual:2"


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
    "genie_small": GENIE_SMALL,
    "attn_space": ATTN_SPACE,
    "lin_attn":   LIN_ATTN,
    "attn_time":  ATTN_TIME,
    "full_attn":  FULL_ATTN,
}

# Complexity rank: lower = simpler (used for sorting sweep order)
LAYER_COMPLEXITY = {
    "genie_small": -1,
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
    import glob

    chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    total = 0
    with concurrent.futures.ThreadPoolExecutor() as pool:
        futures = [pool.submit(_count_chunk, f, seq_len) for f in chunk_files]
        for fut in concurrent.futures.as_completed(futures):
            total += fut.result()
    return total


def read_best_recon(metrics_path: str) -> float:
    """Read the minimum smoothed recon_loss from a training run's metrics.json."""
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
        # Prefer eval_recon_loss (computed in eval mode at val steps)
        recon_values = [m["eval_recon_loss"] for m in metrics if "eval_recon_loss" in m]
        if not recon_values:
            # Use smoothed_recon_loss to match the threshold the trainer uses to stop
            recon_values = [m["smoothed_recon_loss"] for m in metrics if "smoothed_recon_loss" in m]
        if not recon_values:
            recon_values = [m["recon_loss"] for m in metrics if "recon_loss" in m]
        if recon_values:
            return min(recon_values)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return float("inf")


def run_name_for(model_name: str, dataset_size: int) -> str:
    return f"{model_name}_n{dataset_size}"


def collect_existing_trials(
    model_name: str,
    *,
    output_dir: str,
    threshold: float,
) -> dict[int, TrialResult]:
    """Load completed trials for a model from existing run directories."""
    trials: dict[int, TrialResult] = {}
    prefix = f"{model_name}_n"

    if not os.path.isdir(output_dir):
        return trials

    for entry in os.scandir(output_dir):
        if not entry.is_dir() or not entry.name.startswith(prefix):
            continue

        suffix = entry.name[len(prefix):]
        if not suffix.isdigit():
            continue

        metrics_path = os.path.join(entry.path, "metrics.json")
        if not os.path.exists(metrics_path):
            continue

        best_recon = read_best_recon(metrics_path)
        if not math.isfinite(best_recon):
            continue

        dataset_size = int(suffix)
        trials[dataset_size] = TrialResult(
            dataset_size=dataset_size,
            best_recon=best_recon,
            passed=best_recon < threshold,
            run_name=entry.name,
        )

    return dict(sorted(trials.items()))


def model_result_from_dict(payload: dict) -> ModelResult:
    """Convert a saved JSON payload back into a ModelResult."""
    return ModelResult(
        name=payload["name"],
        max_dataset_size=payload.get("max_dataset_size", 0),
        best_recon_at_max=payload.get("best_recon_at_max", float("inf")),
        trials=payload.get("trials", []),
        total_time_s=payload.get("total_time_s", 0.0),
        early_exited=payload.get("early_exited", False),
    )


def load_completed_results(
    summary_path: str,
    *,
    threshold: float,
    total_samples: int,
) -> dict[str, ModelResult]:
    """Load completed model results when the saved summary matches this sweep."""
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
        results[name] = model_result_from_dict(item)
    return results


def write_summary(summary_path: str, threshold: float, total_samples: int, results: list[ModelResult]) -> None:
    """Persist sweep progress so resume can skip completed models."""
    summary = {
        "threshold": threshold,
        "total_samples": total_samples,
        "results": [asdict(r) for r in results],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def build_train_cmd(
    model: ModelConfig,
    dataset_size: int,
    *,
    data_dir: str,
    output_dir: str,
    max_minutes: float,
    lr: float,
    patience: float,
    val_interval: int,
    max_batch_size: int,
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
        "--max-minutes", str(max_minutes),
        "--lr", str(lr),
        "--threshold", str(threshold),
        "--max-patience", str(patience),
        "--val-interval", str(val_interval),
        "--seed", str(seed),
        "--num-workers", str(num_workers),
        "--output-dir", output_dir,
        "--auto-batch-size",
        "--max-batch-size", str(max_batch_size),
    ]
    if dataset_size <= max_batch_size:
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
    trials_by_size = collect_existing_trials(
        model.name,
        output_dir=train_kwargs["output_dir"],
        threshold=threshold,
    ) if resume else {}

    def size_to_index(dataset_size: int) -> int:
        if dataset_size <= 0:
            return 0
        return dataset_size - 1

    def index_to_size(index: int) -> int:
        if index <= 0:
            return 0
        return index + 1

    print(f"\n{'='*64}")
    print(f"Model: {model.name}  (dim={model.init_dim}  cb={model.codebook_size})")
    print(f"{'='*64}")

    low_idx = 0
    high_idx = total_samples
    if trials_by_size:
        print(f"  [resume] restored {len(trials_by_size)} completed trial(s) from disk")
        for trial in trials_by_size.values():
            trial_idx = size_to_index(trial.dataset_size)
            if trial.passed:
                low_idx = max(low_idx, trial_idx)
            else:
                high_idx = min(high_idx, trial_idx)

    if global_best_max > 0 and high_idx < total_samples and size_to_index(global_best_max) >= high_idx:
        result.early_exited = True
        result.max_dataset_size = index_to_size(low_idx)
        result.trials = [asdict(t) for _, t in sorted(trials_by_size.items())]
        max_trial = trials_by_size.get(result.max_dataset_size)
        if max_trial and max_trial.passed:
            result.best_recon_at_max = max_trial.best_recon
        result.total_time_s = time.time() - t0
        print(
            f"  [resume] existing failure at size={index_to_size(high_idx)} "
            f"already rules out current best size={global_best_max}. Early exit."
        )
        return result

    if high_idx - low_idx <= 1:
        print("  [resume] binary-search bounds already resolved")
        probe_idx = None
    elif global_best_max > 0 and global_best_max not in trials_by_size:
        probe_idx = max(1, min(total_samples - 1, size_to_index(global_best_max)))
    else:
        probe_idx = max(1, (low_idx + high_idx) // 2)

    while probe_idx is not None:
        dataset_size = index_to_size(probe_idx)
        trial = trials_by_size.get(dataset_size)
        if trial is not None:
            print(
                f"  [resume] {trial.run_name}: using existing result "
                f"(best_recon={trial.best_recon:.6f})"
            )
        else:
            trial = run_trial(
                model,
                dataset_size,
                dry_run=dry_run,
                resume=resume,
                **train_kwargs,
            )
            trials_by_size[dataset_size] = trial

        if trial.passed:
            low_idx = probe_idx
        else:
            if global_best_max > 0 and dataset_size == global_best_max:
                result.early_exited = True
                result.max_dataset_size = index_to_size(low_idx)
                result.total_time_s = time.time() - t0
                result.trials = [asdict(t) for _, t in sorted(trials_by_size.items())]
                max_trial = trials_by_size.get(result.max_dataset_size)
                if max_trial and max_trial.passed:
                    result.best_recon_at_max = max_trial.best_recon
                print(
                    f"  {model.name}: cannot fit size={global_best_max} "
                    f"(current best). Early exit."
                )
                return result
            high_idx = probe_idx

        if high_idx - low_idx <= 1:
            break

        probe_idx = (low_idx + high_idx) // 2

    result.max_dataset_size = index_to_size(low_idx)
    result.trials = [asdict(t) for _, t in sorted(trials_by_size.items())]
    # Find the best recon at the max passing size
    max_trial = trials_by_size.get(result.max_dataset_size)
    if max_trial and max_trial.passed:
        result.best_recon_at_max = max_trial.best_recon
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
    parser.add_argument("--max-patience", type=float, default=12,
                        help="Minutes without improvement before early stop (default: 12)")
    parser.add_argument("--max-minutes", type=float, default=120,
                        help="Wall-clock minute budget per trial (default: 120)")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--val-interval", type=int, default=100)
    parser.add_argument("--max-batch-size", type=int, default=16,
                        help="Cap auto-detected batch size (default: 16)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=64)
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
        print("Counting samples in dataset (skipping windows with >=2 scene cuts)...")
        total_samples = get_total_samples(args.data_dir)
        print(f"Upper bound: {total_samples} samples")

    if total_samples < 2:
        print("Dataset too small.")
        sys.exit(1)

    # ── Shared training kwargs ───────────────────────────────────────────
    train_kwargs = dict(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_minutes=args.max_minutes,
        lr=args.lr,
        patience=args.max_patience * 60,
        val_interval=args.val_interval,
        max_batch_size=args.max_batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        threshold=args.threshold,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "capacity_sweep_results.json")
    completed_results = load_completed_results(
        summary_path,
        threshold=args.threshold,
        total_samples=total_samples,
    ) if args.resume else {}

    # ── Run sweep ────────────────────────────────────────────────────────
    print(f"\nSweeping {len(configs)} model(s), threshold={args.threshold}")
    all_results: list[ModelResult] = []
    global_best_max = max(
        (r.max_dataset_size for r in completed_results.values()),
        default=0,
    )

    for model in configs:
        if model.name in completed_results:
            mr = completed_results[model.name]
            all_results.append(mr)
            print(
                f"\n[resume] Skipping completed model {model.name} "
                f"(max_size={mr.max_dataset_size})"
            )
            continue

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

        write_summary(summary_path, args.threshold, total_samples, all_results)

    # ── Save summary ─────────────────────────────────────────────────────
    write_summary(summary_path, args.threshold, total_samples, all_results)

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
