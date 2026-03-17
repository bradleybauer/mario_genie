#!/usr/bin/env python3
"""
Asynchronous ASHA Sweep Orchestrator
======================================

True asynchronous successive halving across multiple remote machines.
Unlike synchronous SHA (launch_asha.py), workers never wait for each other:

  1. Each worker trains exactly one trial at a time
  2. When a worker finishes, the coordinator immediately:
     a. Pulls metrics and records the result
     b. Checks if the trial is globally promotable (top 1/η at its rung)
     c. Assigns the best available job — promotions before new trials
  3. Workers are never idle while work remains

Promotion rule (Li et al. 2020):
  A completed trial at rung r is promotable if the number of trials already
  promoted past rung r is less than ⌊completed_at_r / η⌋.  Higher-rung
  promotions get priority over starting new configurations.

Sticky worker assignment:
  Once a trial starts on a worker, it stays there (checkpoint locality).
  Trials are assigned lazily when workers first become free.

Usage:
    python remote/launch_async_asha.py \\
        --rungs 6000,18000,54000,162000 \\
        --reduction-factor 3 \\
        -- --lr 8e-4 --warmup-steps 100
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import (
    PROJECT_ROOT,
    Worker,
    load_workers,
    parse_worker_names,
    rsync_from,
    run_on_all,
    ssh,
)

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.model_configs import MODEL_CONFIGS


# ── Cost estimation ───────────────────────────────────────────────


def estimate_param_counts(configs: list) -> dict[str, int]:
    """Instantiate each model config and count parameters."""
    from magvit2_pytorch import VideoTokenizer

    counts = {}
    for mc in configs:
        layers = tuple(
            (name, int(val)) if ":" in tok else tok
            for tok in mc.layers.split(",")
            for name, _, val in [tok.partition(":")]
        )
        model = VideoTokenizer(
            image_size=256,
            init_dim=mc.init_dim,
            channels=23,
            codebook_size=mc.codebook_size,
            layers=layers,
            use_gan=False,
            perceptual_loss_weight=0.0,
        )
        counts[mc.name] = sum(p.numel() for p in model.parameters())
        del model
    return counts


# ── Data classes ──────────────────────────────────────────────────


@dataclass
class Trial:
    model_name: str
    batch_size: int
    run_name: str
    max_completed_rung: int = -1  # -1 = pending
    rung_metrics: dict = field(default_factory=dict)  # {rung_idx: metric}
    assigned_worker: str = ""  # sticky, set on first launch
    is_running: bool = False
    target_rung_idx: int = -1
    elapsed_s: float = 0.0
    _launch_time: float = 0.0  # transient, not persisted


# ── State persistence ────────────────────────────────────────────

STATE_FILENAME = "async_asha_state.json"


def save_state(path: str, *, rungs, eta, batch_size, trials, extra_args):
    state = {
        "rungs": rungs,
        "reduction_factor": eta,
        "batch_size": batch_size,
        "extra_args": extra_args,
        "trials": [
            {
                "model_name": t.model_name,
                "batch_size": t.batch_size,
                "run_name": t.run_name,
                "max_completed_rung": t.max_completed_rung,
                "rung_metrics": {str(k): v for k, v in t.rung_metrics.items()},
                "assigned_worker": t.assigned_worker,
                "elapsed_s": t.elapsed_s,
            }
            for t in trials
        ],
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def load_state(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ── Metric reading ───────────────────────────────────────────────


def read_best_recon(metrics_path: str) -> float:
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
        if metrics and "eval_recon_loss" in metrics[-1]:
            return metrics[-1]["eval_recon_loss"]
        vals = [m["smoothed_recon_loss"] for m in metrics if "smoothed_recon_loss" in m]
        if not vals:
            vals = [m["recon_loss"] for m in metrics if "recon_loss" in m]
        return min(vals) if vals else float("inf")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return float("inf")


# ── Worker interaction ────────────────────────────────────────────

TMUX_SESSION_PREFIX = "asha"


def tmux_session_name(worker: Worker) -> str:
    return f"{TMUX_SESSION_PREFIX}_{worker.name}"


def is_tmux_running(worker: Worker) -> bool:
    session = tmux_session_name(worker)
    result = ssh(
        worker,
        f"tmux has-session -t {session} 2>/dev/null && echo RUNNING || echo DONE",
        check=False,
        capture=True,
    )
    return "RUNNING" in (result.stdout or "")


def build_trial_script(
    worker: Worker,
    trial: Trial,
    rung_steps: int,
    max_rung_steps: int,
    sweep_dir: str,
    data_dir: str,
    extra_args: str,
    no_compile: bool = False,
    eval_samples: int = 3000,
) -> str:
    """Build a bash script that trains exactly one trial to one rung."""
    run_dir = f"{sweep_dir}/{trial.run_name}"
    state_file = f"{run_dir}/training_state.pt"

    cmd_parts = [
        "python scripts/train_magvit.py",
        f"--data-dir {data_dir}",
        f"--output-dir {sweep_dir}",
        f"--run-name {trial.run_name}",
        f"--model {trial.model_name}",
        f"--batch-size {trial.batch_size}",
        "--auto-batch-size",
        f"--max-batch-size {trial.batch_size}",
        f"--max-steps {rung_steps}",
        f"--total-steps {max_rung_steps}",
        f"--eval-samples {eval_samples}",
    ]

    if no_compile:
        cmd_parts.append("--no-compile")

    cmd_parts.append(
        f'$( [ -f {state_file} ] && echo "--resume-from {state_file}" )'
    )

    if extra_args:
        cmd_parts.append(extra_args)

    return "\n".join([
        "#!/bin/bash",
        ". /opt/miniforge3/etc/profile.d/conda.sh",
        "conda activate mario",
        f"cd {worker.project_dir}",
        "",
        f"echo '>>> Training {trial.run_name} to step {rung_steps}'",
        " \\\n    ".join(cmd_parts),
        f"echo '>>> Done: {trial.run_name}'",
        "",
    ])


def launch_trial_on_worker(worker: Worker, trial: Trial, script: str) -> None:
    """Upload script and launch via tmux."""
    session = tmux_session_name(worker)
    remote_script = f"{worker.project_dir}/run_asha_trial.sh"
    ssh(worker, f"cat > {remote_script} && chmod +x {remote_script}", input=script)
    ssh(worker, f"tmux kill-session -t {session} 2>/dev/null || true", check=False)
    ssh(worker, f"tmux new-session -d -s {session} 'bash {remote_script}'")


def pull_trial_metrics(
    worker: Worker,
    trial: Trial,
    sweep_dir: str,
    local_results_dir: str,
) -> None:
    """Rsync metrics JSON files for one trial from a worker."""
    local_trial_dir = os.path.join(local_results_dir, trial.run_name)
    os.makedirs(local_trial_dir, exist_ok=True)
    rsync_from(
        worker,
        f"{worker.project_dir}/{sweep_dir}/{trial.run_name}/",
        local_trial_dir + "/",
        extra_args=["--include=*.json", "--exclude=*"],
        capture=True,
    )


# ── ASHA promotion logic ─────────────────────────────────────────


def get_next_job_for_worker(
    worker_name: str,
    trials: list[Trial],
    rungs: list[int],
    eta: int,
) -> tuple[Trial | None, int]:
    """ASHA async scheduling: find the best job for a specific worker.

    Scans rungs from highest to lowest looking for globally-promotable
    trials assigned to this worker.  Falls back to claiming an unassigned
    pending trial.  This implements the get_job() routine from Li et al.
    2020 ("A System for Massively Parallel Hyperparameter Tuning").

    Returns (trial, target_rung_idx) or (None, -1) if no work available.
    """
    for rung_idx in reversed(range(len(rungs) - 1)):
        # Global: all trials that have ever completed this rung
        completed = [t for t in trials if t.max_completed_rung >= rung_idx]
        if not completed:
            continue

        # Global: how many are already promoted past this rung?
        # (completed a higher rung, or currently training toward one)
        promoted_past = sum(
            1 for t in trials
            if t.max_completed_rung > rung_idx
            or (t.is_running and t.target_rung_idx > rung_idx)
        )

        n_promotable = max(1, math.floor(len(completed) / eta))
        if promoted_past >= n_promotable:
            continue

        # Which trials are globally in the top n_promotable at this rung?
        completed.sort(
            key=lambda t: t.rung_metrics.get(rung_idx, float("inf"))
        )
        top_names = {t.run_name for t in completed[:n_promotable]}

        # Find a candidate on THIS worker: completed exactly this rung,
        # idle, and globally promotable
        for t in completed:
            if (
                t.max_completed_rung == rung_idx
                and not t.is_running
                and t.assigned_worker == worker_name
                and t.run_name in top_names
            ):
                return t, rung_idx + 1

    # No promotions available — claim an unassigned pending trial
    for t in trials:
        if t.max_completed_rung < 0 and not t.is_running and t.assigned_worker == "":
            return t, 0

    return None, -1


# ── Status display ────────────────────────────────────────────────


def print_status(
    trials: list[Trial],
    rungs: list[int],
    worker_assignments: dict[str, Trial | None],
) -> None:
    n_pending = sum(
        1 for t in trials if t.max_completed_rung < 0 and not t.is_running
    )
    n_running = sum(1 for t in trials if t.is_running)
    rung_completed = [
        sum(1 for t in trials if t.max_completed_rung >= r)
        for r in range(len(rungs))
    ]
    print(
        f"\n  Status: {n_pending} pending | {n_running} running | "
        f"rung completions {rung_completed}"
    )
    for wname, trial in worker_assignments.items():
        if trial:
            print(
                f"    [{wname}] {trial.run_name} "
                f"-> rung {trial.target_rung_idx + 1}/{len(rungs)}"
            )
        else:
            print(f"    [{wname}] idle")
    print()


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Async ASHA sweep: true asynchronous successive halving across "
            "multiple remote machines.  Extra args after '--' are forwarded "
            "to train_magvit.py."
        ),
    )
    parser.add_argument(
        "--data-dir", default="data/",
        help="Data dir on remote machines",
    )
    parser.add_argument(
        "--sweep-dir", default="checkpoints/asha_sweep",
        help="Sweep output dir on remote machines",
    )
    parser.add_argument(
        "--local-results", default=None,
        help="Local dir for metrics (default: results/asha_sweep)",
    )
    parser.add_argument(
        "--rungs", type=str, default="6000,18000,54000,162000",
        help="Comma-separated step budgets (default: 6000,18000,54000,162000)",
    )
    parser.add_argument(
        "--reduction-factor", type=int, default=3,
        help="Keep top 1/eta after each rung (default: 3)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Max batch size per trial (auto-sized down if needed, default: 8)",
    )
    parser.add_argument(
        "--eval-samples", type=int, default=3000,
        help="Held-out eval samples per trial (default: 3000)",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only models whose name contains this substring",
    )
    parser.add_argument(
        "--workers", type=str, default=None,
        help="Comma-separated worker names (default: all)",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=30,
        help="Seconds between completion polls (default: 30)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from saved sweep state",
    )
    parser.add_argument(
        "--no-compile-rungs", type=int, default=2,
        help="Skip torch.compile for the first N rungs (default: 2)",
    )

    args, extra_train_args = parser.parse_known_args()
    if extra_train_args and extra_train_args[0] == "--":
        extra_train_args = extra_train_args[1:]
    extra_args_str = " ".join(extra_train_args)

    rungs = sorted(int(x) for x in args.rungs.split(","))
    eta = args.reduction_factor
    max_rung_steps = rungs[-1]

    local_results = args.local_results or str(
        PROJECT_ROOT / "results" / "asha_sweep"
    )
    state_path = os.path.join(local_results, STATE_FILENAME)

    workers = load_workers(parse_worker_names(args.workers))
    worker_names = {w.name for w in workers}

    # ── Build trial list ──────────────────────────────────────────

    configs = MODEL_CONFIGS.copy()
    if args.filter:
        configs = [c for c in configs if args.filter in c.name]
        if not configs:
            print(f"No models match filter '{args.filter}'.")
            sys.exit(1)

    print("Estimating model param counts ...")
    param_counts = estimate_param_counts(configs)

    # Cheapest first: faster initial throughput → more rung-0 data early
    configs.sort(key=lambda mc: param_counts.get(mc.name, 0))

    trials: list[Trial] = [
        Trial(
            model_name=mc.name,
            batch_size=args.batch_size,
            run_name=mc.name,
        )
        for mc in configs
    ]

    # ── Resume or clean start ─────────────────────────────────────

    os.makedirs(local_results, exist_ok=True)

    if args.resume:
        saved = load_state(state_path)
        if saved:
            saved_by_name = {t["run_name"]: t for t in saved.get("trials", [])}
            for t in trials:
                s = saved_by_name.get(t.run_name)
                if s:
                    t.max_completed_rung = s.get("max_completed_rung", -1)
                    t.rung_metrics = {
                        int(k): v for k, v in s.get("rung_metrics", {}).items()
                    }
                    t.assigned_worker = s.get("assigned_worker", "")
                    t.elapsed_s = s.get("elapsed_s", 0.0)

            # Drop stale worker assignments if worker no longer available
            for t in trials:
                if t.assigned_worker and t.assigned_worker not in worker_names:
                    t.assigned_worker = ""

            print(f"[resume] Loaded state from {state_path}")

        # Kill stale tmux sessions — will re-launch from checkpoints
        def kill_session(w):
            ssh(
                w,
                f"tmux kill-session -t {tmux_session_name(w)} 2>/dev/null || true",
                check=False,
            )

        run_on_all(workers, kill_session, desc="kill stale sessions")
    else:
        print("[clean] Wiping remote sweep dirs and local results ...")

        def clean_worker(w):
            ssh(
                w,
                f"tmux kill-session -t {tmux_session_name(w)} 2>/dev/null || true",
                check=False,
            )
            ssh(w, f"rm -rf {w.project_dir}/{args.sweep_dir}")

        run_on_all(workers, clean_worker, desc="clean remote")

        if os.path.isdir(local_results):
            shutil.rmtree(local_results)
        os.makedirs(local_results, exist_ok=True)
        print("[clean] Done")

    # ── Print plan ────────────────────────────────────────────────

    n_trials = len(trials)
    n_alive, plan_budget = n_trials, 0
    for i, r in enumerate(rungs):
        prev = rungs[i - 1] if i > 0 else 0
        plan_budget += n_alive * (r - prev)
        n_alive = max(1, math.ceil(n_alive / eta))

    print(f"\nAsync ASHA Sweep")
    print(f"  Workers:       {[w.name for w in workers]}")
    print(f"  Trials:        {n_trials} models, bs<={args.batch_size}")
    print(f"  Rungs:         {rungs}")
    print(f"  Reduction (n): {eta}")
    print(f"  Est. budget:   {plan_budget:,} steps "
          f"(vs {n_trials * max_rung_steps:,} exhaustive)")
    if extra_args_str:
        print(f"  Extra args:    {extra_args_str}")
    print()

    # ── Async ASHA main loop ──────────────────────────────────────

    worker_assignments: dict[str, Trial | None] = {w.name: None for w in workers}
    worker_by_name = {w.name: w for w in workers}
    last_status_time = 0.0
    sweep_start = time.time()

    while True:
        any_change = False

        # ── Detect completed workers ──────────────────────────────

        for worker in workers:
            trial = worker_assignments[worker.name]
            if trial is None:
                continue
            if is_tmux_running(worker):
                continue

            # Worker finished its trial
            pull_trial_metrics(worker, trial, args.sweep_dir, local_results)

            local_metrics = os.path.join(
                local_results, trial.run_name, "metrics.json"
            )
            metric = read_best_recon(local_metrics)
            rung_idx = trial.target_rung_idx

            trial.rung_metrics[rung_idx] = metric
            trial.max_completed_rung = rung_idx
            trial.is_running = False
            if trial._launch_time > 0:
                trial.elapsed_s += time.time() - trial._launch_time
                trial._launch_time = 0.0

            worker_assignments[worker.name] = None
            any_change = True

            recon_str = f"{metric:.6f}" if np.isfinite(metric) else "N/A"
            elapsed_min = (time.time() - sweep_start) / 60
            print(
                f"  [{worker.name}] done {trial.run_name} "
                f"rung {rung_idx + 1}/{len(rungs)} "
                f"recon={recon_str}  [{elapsed_min:.0f}m elapsed]"
            )

        # ── Assign work to idle workers ───────────────────────────

        for worker in workers:
            if worker_assignments[worker.name] is not None:
                continue

            trial, target_rung_idx = get_next_job_for_worker(
                worker.name, trials, rungs, eta,
            )
            if trial is None:
                continue

            # Claim unassigned trial
            if trial.assigned_worker == "":
                trial.assigned_worker = worker.name

            rung_steps = rungs[target_rung_idx]
            no_compile = target_rung_idx < args.no_compile_rungs

            script = build_trial_script(
                worker, trial, rung_steps, max_rung_steps,
                args.sweep_dir, args.data_dir, extra_args_str,
                no_compile=no_compile,
                eval_samples=args.eval_samples,
            )
            launch_trial_on_worker(worker, trial, script)

            trial.is_running = True
            trial.target_rung_idx = target_rung_idx
            trial._launch_time = time.time()
            worker_assignments[worker.name] = trial
            any_change = True

            action = "promote" if target_rung_idx > 0 else "start"
            print(
                f"  [{worker.name}] {action} {trial.run_name} "
                f"-> rung {target_rung_idx + 1}/{len(rungs)} "
                f"(step {rung_steps})"
            )

        # ── Termination ───────────────────────────────────────────

        if all(a is None for a in worker_assignments.values()):
            break

        # ── Save state & periodic status ──────────────────────────

        if any_change:
            save_state(
                state_path, rungs=rungs, eta=eta,
                batch_size=args.batch_size, trials=trials,
                extra_args=extra_args_str,
            )

        now = time.time()
        if any_change or now - last_status_time >= 300:
            print_status(trials, rungs, worker_assignments)
            last_status_time = now

        time.sleep(args.poll_interval)

    # ── Final state save ──────────────────────────────────────────

    save_state(
        state_path, rungs=rungs, eta=eta,
        batch_size=args.batch_size, trials=trials,
        extra_args=extra_args_str,
    )

    # ── Leaderboard ───────────────────────────────────────────────

    ranked = sorted(
        trials,
        key=lambda t: (
            -t.max_completed_rung,
            t.rung_metrics.get(t.max_completed_rung, float("inf")),
        ),
    )

    print(f"\n{'=' * 80}")
    print("FINAL LEADERBOARD — Async ASHA Sweep")
    print(f"{'=' * 80}")
    print(f"{'Rank':<6}{'Trial':<40}{'Best Recon':>12}{'Rung':>10}{'Worker':>8}")
    print("-" * 76)
    for i, t in enumerate(ranked, 1):
        best = t.rung_metrics.get(t.max_completed_rung, float("inf"))
        recon_str = f"{best:.6f}" if np.isfinite(best) else "N/A"
        rung_str = (
            f"{t.max_completed_rung + 1}/{len(rungs)}"
            if t.max_completed_rung >= 0
            else "-"
        )
        print(
            f"{i:<6}{t.run_name:<40}{recon_str:>12}"
            f"{rung_str:>10}{t.assigned_worker:>8}"
        )

    total_elapsed = time.time() - sweep_start
    print(f"\nCompleted in {total_elapsed / 3600:.1f}h")
    print(f"State saved to: {state_path}")
    print(
        "Pull full results: "
        "python remote/get_results.py --subdir asha_sweep --images"
    )


if __name__ == "__main__":
    main()
