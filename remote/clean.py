#!/usr/bin/env python3
"""Remove the checkpoints directory on all remote workers."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import load_workers, report_results, run_on_all, show_workers, ssh


def main():
    parser = argparse.ArgumentParser(description="Clean checkpoints on all remote workers")
    parser.add_argument("workers", nargs="*", help="Worker names (omit to list available)")
    parser.add_argument("--subdir", type=str, default=None,
                        help="Only remove this subdirectory under checkpoints/")
    parser.add_argument("--yes", action="store_true",
                        help="Skip confirmation prompt")
    args = parser.parse_args()

    if not args.workers:
        show_workers()
        sys.exit(0)

    workers = load_workers(args.workers)

    if args.subdir:
        target = f"checkpoints/{args.subdir}"
    else:
        target = "checkpoints"

    if not args.yes:
        names = [w.name for w in workers]
        answer = input(f"Remove {target}/ on {names}? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    print(f"Cleaning '{target}/' on {len(workers)} worker(s): {[w.name for w in workers]}")

    def clean(worker):
        ssh(worker, f"rm -rf {worker.project_dir}/{target}")

    results = run_on_all(workers, clean, desc="clean")
    report_results(results)


if __name__ == "__main__":
    main()
