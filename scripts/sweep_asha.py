#!/usr/bin/env python3
"""
ASHA Sweep for MAGVIT Tokenizers (single machine)
===================================================

Asynchronous Successive Halving Algorithm (ASHA) sweep over model
configurations and batch sizes.

ASHA trains all trials for a small step budget (rung), eliminates the
bottom performers, and continues the survivors to progressively longer
rungs.  This dramatically reduces compute compared to training every
configuration to completion.

This script runs on a single machine.  For multi-machine sweeps with
global ranking, use ``remote/launch_asha.py`` instead.

Typical usage:

    python scripts/sweep_asha.py \
        --data-dir data/nes \
        --output-dir checkpoints/asha_sweep \
        --batch-sizes 4,8,16 \
        --rungs 500,1500,4500,13500 \
        --reduction-factor 3 \
        --resume \
        --tf32 --lr 1e-3 --warmup-steps 100

Arguments not consumed by this script (--tf32, --lr, --warmup-steps, etc.)
are forwarded directly to train_magvit.py.
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.dataset_paths import find_session_files
from mario_world_model.model_configs import MODEL_CONFIGS, MODEL_CONFIGS_BY_NAME


# ── Data classes ──────────────────────────────────────────────────


@dataclass
class Trial:
    """One (model_config, batch_size) combination."""
    model_name: str
    batch_size: int
    run_name: str
    best_recon: float = float("inf")
    completed_rung: int = -1    # index of last completed rung (-1 = not started)
    eliminated_at_rung: int = -1  # rung where eliminated (-1 = still alive)
    elapsed_s: float = 0.0


# ── Helpers ───────────────────────────────────────────────────────


def read_best_recon(metrics_path: str) -> float:
    """Read best smoothed recon loss from a training run's metrics.json."""
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
        vals = [m["smoothed_recon_loss"] for m in metrics if "smoothed_recon_loss" in m]
        if not vals:
            vals = [m["recon_loss"] for m in metrics if "recon_loss" in m]
        return min(vals) if vals else float("inf")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return float("inf")


def read_max_step(metrics_path: str) -> int:
    """Read the highest recorded step from metrics.json."""
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
        steps = [m["step"] for m in metrics if "step" in m]
        return max(steps) if steps else 0
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 0


def strip_train_args(extra_args: list[str], names: set[str]) -> list[str]:
    """Remove named arguments and their values from an arg list.

    Handles both ``--name value`` and ``--name=value`` forms.
    """
    result = []
    skip_next = False
    for arg in extra_args:
        if skip_next:
            skip_next = False
            continue
        if arg in names:
            skip_next = True
            continue
        if any(arg.startswith(f"{n}=") for n in names):
            continue
        result.append(arg)
    return result


# ── Sweep state persistence ──────────────────────────────────────


def write_sweep_state(
    path: str,
    *,
    rungs: list[int],
    eta: int,
    batch_sizes: list[int],
    trials: list[Trial],
    current_rung: int,
) -> None:
    state = {
        "rungs": rungs,
        "reduction_factor": eta,
        "batch_sizes": batch_sizes,
        "current_rung": current_rung,
        "trials": [asdict(t) for t in trials],
    }
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_sweep_state(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ── Trial execution ──────────────────────────────────────────────


def run_trial_rung(
    trial: Trial,
    rung_steps: int,
    max_rung_steps: int,
    args: argparse.Namespace,
    extra_train_args: list[str],
) -> None:
    """Train a trial to ``rung_steps`` total steps, resuming if possible."""
    run_dir = os.path.join(args.output_dir, trial.run_name)
    metrics_path = os.path.join(run_dir, "metrics.json")
    state_path = os.path.join(run_dir, "training_state.pt")

    # Check if already at or past this rung
    if os.path.exists(metrics_path):
        existing_step = read_max_step(metrics_path)
        if existing_step >= rung_steps:
            trial.best_recon = read_best_recon(metrics_path)
            print(f"    Already at step {existing_step} >= {rung_steps}, skipping")
            return

    # Build command — sweep controls these args; strip them from extra
    managed = {
        "--model", "--batch-size", "--max-steps", "--total-steps",
        "--resume-from", "--run-name", "--output-dir", "--data-dir",
    }
    clean_extra = strip_train_args(extra_train_args, managed)

    cmd = [
        sys.executable, "scripts/train_magvit.py",
        "--data-dir", args.data_dir,
        "--output-dir", args.output_dir,
        "--run-name", trial.run_name,
        "--model", trial.model_name,
        "--batch-size", str(trial.batch_size),
        "--max-steps", str(rung_steps),
        "--total-steps", str(max_rung_steps),
        *clean_extra,
    ]

    # Resume from previous rung's checkpoint
    if trial.completed_rung >= 0 and os.path.exists(state_path):
        cmd.extend(["--resume-from", state_path])

    if args.dry_run:
        print(f"    [dry-run] {' '.join(cmd)}")
        return

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    trial.elapsed_s += elapsed

    if result.returncode != 0:
        print(f"    [WARN] {trial.run_name} exited with code {result.returncode}")

    trial.best_recon = read_best_recon(metrics_path)


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ASHA sweep: successive halving over model configs × batch sizes. "
            "Unrecognised args are forwarded to train_magvit.py."
        ),
    )
    parser.add_argument("--data-dir", default="data/nes")
    parser.add_argument("--output-dir", default="checkpoints/asha_sweep")
    parser.add_argument(
        "--rungs", type=str, default="500,1500,4500,13500",
        help="Comma-separated step budgets for successive halving (default: 500,1500,4500,13500)",
    )
    parser.add_argument(
        "--reduction-factor", type=int, default=3,
        help="Keep top 1/η trials after each rung (default: 3)",
    )
    parser.add_argument(
        "--batch-sizes", type=str, default="4,8,16",
        help="Comma-separated batch sizes to sweep (default: 4,8,16)",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only run models whose name contains this substring",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--resume", action="store_true", help="Resume from existing sweep state")
    args, extra_train_args = parser.parse_known_args()

    rungs = sorted(int(x) for x in args.rungs.split(","))
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    eta = args.reduction_factor
    max_rung_steps = rungs[-1]

    # ── Build trial list ─────────────────────────────────────────────
    configs = MODEL_CONFIGS.copy()

    if args.filter:
        configs = [c for c in configs if args.filter in c.name]
        if not configs:
            print(f"No models match filter '{args.filter}'.")
            sys.exit(1)

    trials: list[Trial] = []
    for mc in configs:
        for bs in batch_sizes:
            trials.append(Trial(
                model_name=mc.name,
                batch_size=bs,
                run_name=f"{mc.name}_bs{bs}",
            ))

    random.shuffle(trials)

    # ── Resume: restore trial state from previous run ────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    state_path = os.path.join(args.output_dir, "asha_sweep_state.json")
    start_rung_idx = 0

    if args.resume:
        saved = load_sweep_state(state_path)
        if saved:
            saved_by_name = {t["run_name"]: t for t in saved.get("trials", [])}
            for trial in trials:
                if trial.run_name in saved_by_name:
                    s = saved_by_name[trial.run_name]
                    trial.best_recon = s.get("best_recon", float("inf"))
                    trial.completed_rung = s.get("completed_rung", -1)
                    trial.eliminated_at_rung = s.get("eliminated_at_rung", -1)
                    trial.elapsed_s = s.get("elapsed_s", 0.0)
            start_rung_idx = saved.get("current_rung", 0)
            print(f"[resume] Loaded state: starting from rung {start_rung_idx}")

    # ── Print plan ───────────────────────────────────────────────────
    n_trials = len(trials)
    plan_budget = 0
    n_alive = n_trials
    for i, r in enumerate(rungs):
        prev = rungs[i - 1] if i > 0 else 0
        incremental = r - prev
        plan_budget += n_alive * incremental
        n_alive = max(1, math.ceil(n_alive / eta))

    print(f"\nASHA Sweep")
    print(f"  Trials:        {n_trials} ({len(configs)} models × {len(batch_sizes)} batch sizes)")
    print(f"  Rungs:         {rungs}")
    print(f"  Reduction (η): {eta}")
    print(f"  Batch sizes:   {batch_sizes}")
    print(f"  Est. budget:   {plan_budget:,} total steps  "
          f"(vs {n_trials * max_rung_steps:,} exhaustive)")
    print(f"  Saving to:     {args.output_dir}")
    if extra_train_args:
        print(f"  Extra args:    {' '.join(extra_train_args)}")
    print()

    # ── Run ASHA ─────────────────────────────────────────────────────
    for rung_idx, rung_steps in enumerate(rungs):
        # Determine which trials are still alive for this rung
        active = [t for t in trials if t.eliminated_at_rung < 0]

        # Skip rungs that were fully completed in a previous run
        if rung_idx < start_rung_idx:
            # Re-apply elimination for this rung (from saved state)
            already_done = all(t.completed_rung >= rung_idx for t in active)
            if already_done:
                active.sort(key=lambda t: t.best_recon)
                n_keep = max(1, math.ceil(len(active) / eta))
                if rung_idx < len(rungs) - 1:
                    for t in active[n_keep:]:
                        t.eliminated_at_rung = rung_idx
                continue

        print(f"{'=' * 64}")
        print(f"RUNG {rung_idx + 1}/{len(rungs)}: train to {rung_steps} steps "
              f"— {len(active)} trial(s)")
        print(f"{'=' * 64}")

        for i, trial in enumerate(active):
            # Skip trials already past this rung (from resume)
            if trial.completed_rung >= rung_idx:
                print(f"  [{i+1}/{len(active)}] {trial.run_name} "
                      f"— already done (best_recon={trial.best_recon:.6f})")
                continue

            print(f"  [{i+1}/{len(active)}] {trial.run_name} "
                  f"(model={trial.model_name}, bs={trial.batch_size})")
            run_trial_rung(trial, rung_steps, max_rung_steps, args, extra_train_args)
            trial.completed_rung = rung_idx

            # Save after each trial for fine-grained resume
            write_sweep_state(
                state_path,
                rungs=rungs, eta=eta, batch_sizes=batch_sizes,
                trials=trials, current_rung=rung_idx,
            )

        # ── Rank and eliminate ───────────────────────────────────────
        active.sort(key=lambda t: t.best_recon)
        n_keep = max(1, math.ceil(len(active) / eta))

        print(f"\n--- Rung {rung_idx + 1} results "
              f"(keeping top {n_keep}/{len(active)}) ---")
        for rank, t in enumerate(active, 1):
            marker = "  KEEP " if rank <= n_keep else "  ELIM "
            recon_str = f"{t.best_recon:.6f}" if np.isfinite(t.best_recon) else "N/A"
            print(f"  {marker} {rank:>3}. {t.run_name:<45} best_recon={recon_str}")

        # Don't eliminate on the last rung
        if rung_idx < len(rungs) - 1:
            for t in active[n_keep:]:
                t.eliminated_at_rung = rung_idx

        write_sweep_state(
            state_path,
            rungs=rungs, eta=eta, batch_sizes=batch_sizes,
            trials=trials, current_rung=rung_idx + 1,
        )

    # ── Final leaderboard ────────────────────────────────────────────
    ranked = sorted(trials, key=lambda t: t.best_recon)
    print(f"\n{'=' * 80}")
    print("FINAL LEADERBOARD — ASHA Sweep")
    print(f"{'=' * 80}")
    print(f"{'Rank':<6}{'Trial':<45}{'BS':>4}{'Best Recon':>12}{'Rung':>6}{'Time':>10}")
    print("-" * 83)
    for i, t in enumerate(ranked, 1):
        recon_str = f"{t.best_recon:.6f}" if np.isfinite(t.best_recon) else "N/A"
        time_str = f"{t.elapsed_s:.0f}s" if t.elapsed_s > 0 else "-"
        rung_str = f"{t.completed_rung + 1}/{len(rungs)}"
        print(f"{i:<6}{t.run_name:<45}{t.batch_size:>4}{recon_str:>12}{rung_str:>6}{time_str:>10}")

    total_elapsed = sum(t.elapsed_s for t in trials)
    print(f"\nTotal training time: {total_elapsed:.0f}s")
    print(f"Results saved to: {state_path}")


if __name__ == "__main__":
    main()
