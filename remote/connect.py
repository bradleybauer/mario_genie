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


def build_remote_attach_command(multiplexer: str, session: str) -> str:
    # Source cargo env so ~/.cargo/bin (zellij) is on PATH in non-login SSH shells
    cargo_env = '[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"; '
    shell_fallback = 'exec "${SHELL:-/bin/bash}" -l'
    tmux_cmd = f"tmux attach -t {session} || tmux new -s {session}"
    zellij_cmd = f"zellij attach {session} || zellij attach -c {session}"

    if multiplexer == "tmux":
        return f"{tmux_cmd} || {shell_fallback}"
    if multiplexer == "zellij":
        return f"{cargo_env}{zellij_cmd} || {shell_fallback}"

    return (
        f"{cargo_env}"
        "if command -v zellij >/dev/null 2>&1 && zellij list-sessions 2>/dev/null | grep -Fxq "
        f"{session}; then zellij attach {session}; "
        "elif command -v tmux >/dev/null 2>&1 && tmux has-session -t "
        f"{session} 2>/dev/null; then tmux attach -t {session}; "
        "elif command -v zellij >/dev/null 2>&1; then "
        f"{zellij_cmd}; "
        "elif command -v tmux >/dev/null 2>&1; then "
        f"{tmux_cmd}; "
        "else false; "
        f"fi || {shell_fallback}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSH to a remote worker")
    parser.add_argument("worker", nargs="?", default=None, help="Worker name from config")
    parser.add_argument("--port-forward", type=int, default=8080,
                        help="Local port to forward (default: 8080)")
    parser.add_argument("--session", default="sweep",
                        help="Remote multiplexer session to attach or create (default: sweep)")
    parser.add_argument("--multiplexer", choices=("auto", "tmux", "zellij", "none"),
                        default="zellij",
                        help="Remote multiplexer to use after connecting (default: auto)")
    parser.add_argument("--no-tmux", action="store_true",
                        help="Open a plain interactive SSH session without attaching to tmux or zellij")
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
    multiplexer = "none" if args.no_tmux else args.multiplexer
    if multiplexer == "none":
        ssh_cmd.append(w.remote)
    else:
        session = shlex.quote(args.session)
        ssh_cmd.extend([
            "-t",
            w.remote,
            build_remote_attach_command(multiplexer, session),
        ])
    os.execvp("ssh", ssh_cmd)


if __name__ == "__main__":
    main()
