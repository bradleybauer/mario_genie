#!/usr/bin/env python3
"""Connect to a specific remote worker (interactive SSH).

Run with no args to list available workers.
"""

import argparse
import os
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import load_workers, show_workers, ssh_base_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSH to a remote worker")
    parser.add_argument("worker", nargs="?", default=None, help="Worker name from config")
    parser.add_argument("--port-forward", type=int, default=8080,
                        help="Local port to forward (default: 8080)")
    parser.add_argument("--session", default="sweep",
                        help="Tmux session to attach after connecting (default: sweep)")
    parser.add_argument("--no-tmux", action="store_true",
                        help="Open a plain interactive SSH session without tmux attach")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Pass -vvv to SSH for debug output")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.worker:
        show_workers()
        sys.exit(0)

    workers = load_workers([args.worker])
    w = workers[0]
    print(f"Connecting to {w.name} ({w.host})...")

    ssh_cmd = [
        *ssh_base_args(w),
        "-L", f"{args.port_forward}:localhost:{args.port_forward}",
    ]
    if args.verbose:
        ssh_cmd.append("-vvv")
    if args.no_tmux:
        ssh_cmd.append(w.remote)
    else:
        session = shlex.quote(args.session)
        ssh_cmd.extend([
            "-t",
            w.remote,
            f"tmux attach -t {session} || exec \"${{SHELL:-/bin/bash}}\" -l",
        ])
    os.execvp("ssh", ssh_cmd)


if __name__ == "__main__":
    main()
