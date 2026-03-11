#!/usr/bin/env python3
"""Pick a random model config and train it once on the full dataset."""

import argparse
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.model_configs import MODEL_CONFIGS


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a random model config.")
    parser.add_argument("--data-dir", default="data/nes")
    parser.add_argument("--output-dir", default="checkpoints/model_config_sweep_genie")
    parser.add_argument("--threshold", type=float, default=0.0008)
    parser.add_argument("--max-patience", type=float, default=60 * 34)
    parser.add_argument("--max-minutes", type=float, default=60 * 34)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--val-interval", type=int, default=100)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (also passed to trainer). Random if omitted.")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--filter", type=str, default=None,
                        help="Only consider models whose name contains this substring")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configs = MODEL_CONFIGS.copy()
    if args.filter:
        configs = [c for c in configs if args.filter in c.name]
        if not configs:
            print(f"No models match filter '{args.filter}'.")
            sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)
    model = random.choice(configs)

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    run_name = f"{model.name}_full"

    cmd = [
        sys.executable, "scripts/train_magvit.py",
        "--data-dir", args.data_dir,
        "--run-name", run_name,
        "--init-dim", str(model.init_dim),
        "--codebook-size", str(model.codebook_size),
        "--layers", model.layers,
        "--max-minutes", str(args.max_minutes),
        "--lr", str(args.lr),
        "--warmup-steps", str(args.warmup_steps),
        "--threshold", str(args.threshold),
        "--max-patience", str(args.max_patience * 60),
        "--val-interval", str(args.val_interval),
        "--seed", str(seed),
        "--num-workers", str(args.num_workers),
        "--output-dir", args.output_dir,
        "--auto-batch-size",
        "--max-batch-size", str(args.max_batch_size),
        "--no-preload",
    ]

    print(f"Selected model: {model.name}")
    print(f"Run name:       {run_name}")
    print(f"Command:        {' '.join(cmd)}")

    if args.dry_run:
        return

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
