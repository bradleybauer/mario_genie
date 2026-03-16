#!/usr/bin/env python3
"""
Distributed ASHA Sweep Orchestrator
====================================

Coordinates an ASHA (Asynchronous Successive Halving) sweep across multiple
remote machines.  The local machine acts as the coordinator:

  1. Build all (model, batch_size) trials
  2. For each rung assign alive trials to workers (sticky — a trial always
     runs on the same worker so checkpoints don't need to move)
  3. Generate a bash runner script per worker and launch via tmux
  4. Poll tmux sessions until every worker finishes its rung
  5. Rsync metrics back, rank globally, eliminate bottom trials
  6. Repeat for the next rung

Usage:
    python remote/launch_asha.py \\
        --rungs 500,1500,4500,13500 \\
        --batch-sizes 4,8,16 \\
        --reduction-factor 3 \\
        -- --lr 8e-4 --warmup-steps 100

Everything after '--' is forwarded to train_magvit.py on each worker.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
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

# Also add src/ so we can import model configs locally
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.model_configs import MODEL_CONFIGS


# ── Data classes ──────────────────────────────────────────────────


@dataclass
class Trial:
    model_name: str
    batch_size: int
    run_name: str
    worker_name: str  # sticky worker assignment
    best_recon: float = float("inf")
    completed_rung: int = -1
    eliminated_at_rung: int = -1
    elapsed_s: float = 0.0


# ── State persistence (local) ────────────────────────────────────

STATE_FILENAME = "asha_distributed_state.json"


def save_state(path: str, *, rungs, eta, batch_sizes, trials, current_rung):
    state = {
        "rungs": rungs,
        "reduction_factor": eta,
        "batch_sizes": batch_sizes,
        "current_rung": current_rung,
        "trials": [asdict(t) for t in trials],
    }
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_state(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ── Metric reading (from local rsync'd copies) ───────────────────


def read_best_recon(metrics_path: str) -> float:
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
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
        steps = [m["step"] for m in metrics if "step" in m]
        return max(steps) if steps else 0
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 0


# ── Worker interaction ────────────────────────────────────────────

TMUX_SESSION = "asha"


def is_tmux_running(worker: Worker, session: str = TMUX_SESSION) -> bool:
    """Check if a tmux session is still active on a worker."""
    result = ssh(worker, f"tmux has-session -t {session} 2>/dev/null && echo RUNNING || echo DONE",
                 check=False, capture=True)
    return "RUNNING" in (result.stdout or "")


def build_worker_script(
    worker: Worker,
    trials: list[Trial],
    rung_steps: int,
    max_rung_steps: int,
    sweep_dir: str,
    extra_args: str,
) -> str:
    """Build a bash script that trains all assigned trials for one rung."""
    lines = [
        "#!/bin/bash",
        ". /opt/miniforge3/etc/profile.d/conda.sh",
        "conda activate mario",
        "set -e",
        f"cd {worker.project_dir}",
        "",
    ]

    for trial in trials:
        run_dir = f"{sweep_dir}/{trial.run_name}"
        state_file = f"{run_dir}/training_state.pt"

        cmd_parts = [
            "python scripts/train_magvit.py",
            f"--data-dir data/",
            f"--output-dir {sweep_dir}",
            f"--run-name {trial.run_name}",
            f"--model {trial.model_name}",
            f"--auto-batch-size",
            f"--max-batch-size {trial.batch_size}",
            f"--max-steps {rung_steps}",
            f"--total-steps {max_rung_steps}",
        ]

        # Resume from previous rung's checkpoint if it exists
        if trial.completed_rung >= 0:
            cmd_parts.append(
                f'$( [ -f {state_file} ] && echo "--resume-from {state_file}" )'
            )

        if extra_args:
            cmd_parts.append(extra_args)

        lines.append(f"echo '>>> Training {trial.run_name} to step {rung_steps}'")
        lines.append(" \\\n    ".join(cmd_parts))
        lines.append("")

    lines.append("echo '>>> Worker done'")
    lines.append("")
    return "\n".join(lines)


def launch_rung_on_workers(
    workers: list[Worker],
    trials_by_worker: dict[str, list[Trial]],
    rung_steps: int,
    max_rung_steps: int,
    sweep_dir: str,
    extra_args: str,
    session: str = TMUX_SESSION,
) -> None:
    """Upload scripts and launch training on all workers for one rung."""
    for worker in workers:
        worker_trials = trials_by_worker.get(worker.name, [])
        if not worker_trials:
            continue

        script = build_worker_script(
            worker, worker_trials, rung_steps, max_rung_steps,
            sweep_dir, extra_args,
        )

        remote_script = f"{worker.project_dir}/run_asha_rung.sh"
        ssh(worker, f"cat > {remote_script} && chmod +x {remote_script}",
            input=script)

        # Kill existing session and start fresh
        ssh(worker, f"tmux kill-session -t {session} 2>/dev/null || true", check=False)
        ssh(worker, f"tmux new-session -d -s {session} 'bash {remote_script}'")
        print(f"    [{worker.name}] launched {len(worker_trials)} trial(s)")


def wait_for_workers(
    workers: list[Worker],
    trials_by_worker: dict[str, list[Trial]],
    poll_interval: float = 30,
    session: str = TMUX_SESSION,
) -> None:
    """Block until all workers' tmux sessions finish."""
    active_workers = [
        w for w in workers if trials_by_worker.get(w.name)
    ]
    if not active_workers:
        return

    print(f"    Waiting for {len(active_workers)} worker(s) ...")
    last_status_print = 0.0
    while active_workers:
        time.sleep(poll_interval)
        still_running = []
        for w in active_workers:
            if is_tmux_running(w, session):
                still_running.append(w)
            else:
                print(f"    [{w.name}] done")
        active_workers = still_running
        if active_workers and time.time() - last_status_print >= 900:
            names = ", ".join(w.name for w in active_workers)
            print(f"    Still running: {names}")
            last_status_print = time.time()


def pull_metrics(
    workers: list[Worker],
    sweep_dir: str,
    local_results_dir: str,
) -> None:
    """Rsync metrics JSON files from all workers to local results dir."""
    os.makedirs(local_results_dir, exist_ok=True)

    def fetch(worker):
        rsync_from(
            worker,
            f"{worker.project_dir}/{sweep_dir}/",
            local_results_dir + "/",
            extra_args=[
                "--include=*/",
                "--include=*.json",
                "--exclude=*",
            ],
            capture=True,
        )

    run_on_all(workers, fetch, desc="pull metrics")


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Distributed ASHA sweep: orchestrate successive halving across "
            "multiple remote machines. Extra args after '--' are forwarded "
            "to train_magvit.py."
        ),
    )
    parser.add_argument("--data-dir", default="data/",
                        help="Data dir on remote machines (default: data/)")
    parser.add_argument("--sweep-dir", default="checkpoints/asha_sweep",
                        help="Sweep output dir on remote machines")
    parser.add_argument("--local-results", default=None,
                        help="Local dir for metrics (default: results/asha_sweep)")
    parser.add_argument(
        "--rungs", type=str, default="6000,18000,54000,162000",
        help="Comma-separated step budgets (default: 6000,18000,54000,162000)",
    )
    parser.add_argument(
        "--reduction-factor", type=int, default=3,
        help="Keep top 1/η trials after each rung (default: 3)",
    )
    parser.add_argument(
        "--batch-sizes", type=str, default="4,8,16",
        help="Comma-separated batch sizes (default: 4,8,16)",
    )
    parser.add_argument("--filter", type=str, default=None,
                        help="Only run models whose name contains this substring")
    parser.add_argument("--workers", type=str, default=None,
                        help="Comma-separated worker names (default: all)")
    parser.add_argument("--poll-interval", type=float, default=30,
                        help="Seconds between completion polls (default: 30)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved sweep state")
    parser.add_argument("--session", type=str, default=TMUX_SESSION,
                        help="Tmux session name on workers (default: asha)")
    args, extra_train_args = parser.parse_known_args()

    if extra_train_args and extra_train_args[0] == "--":
        extra_train_args = extra_train_args[1:]
    extra_args_str = " ".join(extra_train_args)

    rungs = sorted(int(x) for x in args.rungs.split(","))
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    eta = args.reduction_factor
    max_rung_steps = rungs[-1]

    local_results = args.local_results or str(PROJECT_ROOT / "results" / "asha_sweep")
    state_path = os.path.join(local_results, STATE_FILENAME)

    workers = load_workers(parse_worker_names(args.workers))
    worker_names = [w.name for w in workers]

    # ── Build trial list with sticky worker assignment ───────────
    configs = MODEL_CONFIGS.copy()
    if args.filter:
        configs = [c for c in configs if args.filter in c.name]
        if not configs:
            print(f"No models match filter '{args.filter}'.")
            sys.exit(1)

    trials: list[Trial] = []
    for i, mc in enumerate(configs):
        for j, bs in enumerate(batch_sizes):
            # Round-robin across workers
            worker_idx = (i * len(batch_sizes) + j) % len(workers)
            trials.append(Trial(
                model_name=mc.name,
                batch_size=bs,
                run_name=f"{mc.name}_bs{bs}",
                worker_name=worker_names[worker_idx],
            ))

    # ── Resume ───────────────────────────────────────────────────
    os.makedirs(local_results, exist_ok=True)
    start_rung_idx = 0

    if args.resume:
        saved = load_state(state_path)
        if saved:
            saved_by_name = {t["run_name"]: t for t in saved.get("trials", [])}
            for trial in trials:
                if trial.run_name in saved_by_name:
                    s = saved_by_name[trial.run_name]
                    trial.best_recon = s.get("best_recon", float("inf"))
                    trial.completed_rung = s.get("completed_rung", -1)
                    trial.eliminated_at_rung = s.get("eliminated_at_rung", -1)
                    trial.elapsed_s = s.get("elapsed_s", 0.0)
                    # Preserve sticky assignment from saved state
                    saved_worker = s.get("worker_name")
                    if saved_worker and saved_worker in worker_names:
                        trial.worker_name = saved_worker
            start_rung_idx = saved.get("current_rung", 0)
            print(f"[resume] Loaded state from {state_path}, starting at rung {start_rung_idx}")
    else:
        # ── Clean start: wipe remote sweep dirs, kill stale tmux, clear local results ──
        print("[clean] Killing stale tmux sessions and wiping remote sweep dirs ...")

        def clean_worker(worker):
            ssh(worker, f"tmux kill-session -t {args.session} 2>/dev/null || true", check=False)
            ssh(worker, f"rm -rf {worker.project_dir}/{args.sweep_dir}")

        run_on_all(workers, clean_worker, desc="clean remote")

        # Clear local results
        import shutil
        if os.path.isdir(local_results):
            shutil.rmtree(local_results)
        os.makedirs(local_results, exist_ok=True)
        print("[clean] Done")

    # ── Print plan ───────────────────────────────────────────────
    n_trials = len(trials)
    plan_budget = 0
    n_alive = n_trials
    for i, r in enumerate(rungs):
        prev = rungs[i - 1] if i > 0 else 0
        plan_budget += n_alive * (r - prev)
        n_alive = max(1, math.ceil(n_alive / eta))

    worker_counts = {}
    for t in trials:
        worker_counts[t.worker_name] = worker_counts.get(t.worker_name, 0) + 1

    print(f"\nDistributed ASHA Sweep")
    print(f"  Workers:       {worker_names}")
    print(f"  Trials:        {n_trials} ({len(configs)} models × {len(batch_sizes)} batch sizes)")
    print(f"  Assignment:    {worker_counts}")
    print(f"  Rungs:         {rungs}")
    print(f"  Reduction (η): {eta}")
    print(f"  Batch sizes:   {batch_sizes}")
    print(f"  Est. budget:   {plan_budget:,} total steps  "
          f"(vs {n_trials * max_rung_steps:,} exhaustive)")
    if extra_args_str:
        print(f"  Extra args:    {extra_args_str}")
    print()

    # ── Run ASHA rung by rung ────────────────────────────────────
    for rung_idx, rung_steps in enumerate(rungs):
        active = [t for t in trials if t.eliminated_at_rung < 0]

        # Skip completed rungs on resume
        if rung_idx < start_rung_idx:
            already_done = all(t.completed_rung >= rung_idx for t in active)
            if already_done:
                active.sort(key=lambda t: t.best_recon)
                n_keep = max(1, math.ceil(len(active) / eta))
                if rung_idx < len(rungs) - 1:
                    for t in active[n_keep:]:
                        t.eliminated_at_rung = rung_idx
                continue

        # Trials needing training this rung
        to_train = [t for t in active if t.completed_rung < rung_idx]

        print(f"{'=' * 64}")
        print(f"RUNG {rung_idx + 1}/{len(rungs)}: train to {rung_steps} steps "
              f"— {len(active)} alive, {len(to_train)} to train")
        print(f"{'=' * 64}")

        if to_train:
            # Group by worker
            trials_by_worker: dict[str, list[Trial]] = {}
            for t in to_train:
                trials_by_worker.setdefault(t.worker_name, []).append(t)

            for wn, wt in trials_by_worker.items():
                print(f"    [{wn}] {len(wt)} trial(s)")

            # Launch on all workers
            launch_rung_on_workers(
                workers, trials_by_worker,
                rung_steps, max_rung_steps,
                args.sweep_dir, extra_args_str,
                session=args.session,
            )

            # Wait for all workers to finish
            wait_for_workers(
                workers, trials_by_worker,
                poll_interval=args.poll_interval,
                session=args.session,
            )

            # Pull metrics
            print(f"\n  Pulling metrics ...")
            pull_metrics(workers, args.sweep_dir, local_results)

        # Read metrics and update trials
        for t in active:
            local_metrics = os.path.join(local_results, t.run_name, "metrics.json")
            t.best_recon = read_best_recon(local_metrics)
            t.completed_rung = rung_idx

        # Rank and eliminate
        active.sort(key=lambda t: t.best_recon)
        n_keep = max(1, math.ceil(len(active) / eta))

        print(f"\n--- Rung {rung_idx + 1} results "
              f"(keeping top {n_keep}/{len(active)}) ---")
        for rank, t in enumerate(active, 1):
            marker = "  KEEP " if rank <= n_keep else "  ELIM "
            recon_str = f"{t.best_recon:.6f}" if np.isfinite(t.best_recon) else "N/A"
            print(f"  {marker} {rank:>3}. {t.run_name:<45} "
                  f"best_recon={recon_str}  [{t.worker_name}]")

        if rung_idx < len(rungs) - 1:
            for t in active[n_keep:]:
                t.eliminated_at_rung = rung_idx

        save_state(
            state_path,
            rungs=rungs, eta=eta, batch_sizes=batch_sizes,
            trials=trials, current_rung=rung_idx + 1,
        )

    # ── Final leaderboard ────────────────────────────────────────
    ranked = sorted(trials, key=lambda t: t.best_recon)
    print(f"\n{'=' * 80}")
    print("FINAL LEADERBOARD — Distributed ASHA Sweep")
    print(f"{'=' * 80}")
    print(f"{'Rank':<6}{'Trial':<45}{'BS':>4}{'Best Recon':>12}{'Rung':>6}{'Worker':>8}")
    print("-" * 81)
    for i, t in enumerate(ranked, 1):
        recon_str = f"{t.best_recon:.6f}" if np.isfinite(t.best_recon) else "N/A"
        rung_str = f"{t.completed_rung + 1}/{len(rungs)}"
        print(f"{i:<6}{t.run_name:<45}{t.batch_size:>4}{recon_str:>12}"
              f"{rung_str:>6}{t.worker_name:>8}")

    print(f"\nState saved to: {state_path}")
    print("Pull full results (with images): "
          f"python remote/get_results.py --subdir asha_sweep --images")


if __name__ == "__main__":
    main()
