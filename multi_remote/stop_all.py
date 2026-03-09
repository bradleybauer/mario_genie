#!/usr/bin/env python3
"""Kill the sweep tmux session on all remote workers."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import load_workers, parse_worker_names, report_results, run_on_all, ssh


def main():
    parser = argparse.ArgumentParser(description="Stop sweep on all remote workers")
    parser.add_argument("--workers", type=str, default=None,
                        help="Comma-separated worker names (default: all)")
    parser.add_argument("--session", type=str, default="sweep",
                        help="Tmux session name to kill (default: sweep)")
    args = parser.parse_args()

    workers = load_workers(parse_worker_names(args.workers))
    print(f"Stopping '{args.session}' on {len(workers)} worker(s): {[w.name for w in workers]}")

    def stop(worker):
        ssh(worker, f"tmux kill-session -t {args.session} 2>/dev/null || true", check=False)

    results = run_on_all(workers, stop, desc="kill sweep")
    report_results(results)


if __name__ == "__main__":
    main()
