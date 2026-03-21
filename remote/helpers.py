"""Shared helpers for multi-machine remote operations."""

import importlib.util
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Worker:
    name: str
    host: str
    port: int
    user: str
    project_name: str
    instance_id: int | None = None

    @property
    def remote(self) -> str:
        return f"{self.user}@{self.host}"

    @property
    def home(self) -> str:
        return "/root" if self.user == "root" else f"/home/{self.user}"

    @property
    def project_dir(self) -> str:
        return f"{self.home}/{self.project_name}"


def load_config():
    """Load config.py from the multi_remote directory."""
    config_path = Path(__file__).parent / "config.py"
    spec = importlib.util.spec_from_file_location("_multi_remote_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config from {config_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_workers(names: list[str] | None = None) -> list[Worker]:
    """Load workers from config, optionally filtering by name."""
    cfg = load_config()
    project_name = getattr(cfg, "PROJECT_NAME", "mario")
    workers = [
        Worker(
            name=w["name"],
            host=w["host"],
            port=w["port"],
            user=w["user"],
            project_name=project_name,
            instance_id=w.get("instance_id"),
        )
        for w in cfg.WORKERS
    ]
    if names:
        name_set = set(names)
        workers = [w for w in workers if w.name in name_set]
        missing = name_set - {w.name for w in workers}
        if missing:
            print(f"Warning: workers not found in config: {missing}", file=sys.stderr)
    if not workers:
        print("No workers configured. Edit multi_remote/config.py", file=sys.stderr)
        sys.exit(1)
    return workers


def ssh_base_args(worker: Worker) -> list[str]:
    """Return the common SSH args for connecting to a worker."""
    return ["ssh", *_ssh_transport_args(worker)]


def _ssh_transport_args(worker: Worker) -> list[str]:
    """Return transport-only SSH args shared by ssh and rsync."""
    return [
        "-p", str(worker.port),
        "-o", "StrictHostKeyChecking=accept-new",
    ]


def rsync_ssh_command(worker: Worker) -> str:
    """Return the SSH command string rsync should use for this worker."""
    return shlex.join(["ssh", *_ssh_transport_args(worker)])


def ssh(
    worker: Worker,
    command: str,
    *,
    check: bool = True,
    capture: bool = False,
    input: str | None = None,
) -> subprocess.CompletedProcess:
    """Run a command on a remote worker via SSH."""
    cmd = [*ssh_base_args(worker), worker.remote, command]
    return subprocess.run(cmd, check=check, capture_output=capture, text=True, input=input)


def rsync_to(
    worker: Worker,
    local_srcs: str | list[str],
    remote_dst: str,
    *,
    extra_args: list[str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Rsync local files/dirs to a remote worker."""
    if isinstance(local_srcs, str):
        local_srcs = [local_srcs]
    cmd = ["rsync", "-avz", "-e", rsync_ssh_command(worker)]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(local_srcs)
    cmd.append(f"{worker.remote}:{remote_dst}")
    return subprocess.run(cmd, check=True, capture_output=capture, text=capture)


def rsync_from(
    worker: Worker,
    remote_src: str,
    local_dst: str,
    *,
    extra_args: list[str] | None = None,    capture: bool = False,) -> subprocess.CompletedProcess:
    """Rsync files from a remote worker to local."""
    cmd = ["rsync", "-avz", "-e", rsync_ssh_command(worker)]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(f"{worker.remote}:{remote_src}")
    cmd.append(local_dst)
    return subprocess.run(cmd, check=True, capture_output=capture, text=capture)


def run_on_all(
    workers: list[Worker], fn, *, desc: str = "operation",
) -> dict[str, tuple[bool, object]]:
    """Run fn(worker) in parallel across all workers.

    Returns {worker_name: (success, result_or_error)}.
    """
    results: dict[str, tuple[bool, object]] = {}
    total = len(workers)
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = {pool.submit(fn, w): w for w in workers}
        for future in as_completed(futures):
            w = futures[future]
            try:
                result = future.result()
                results[w.name] = (True, result)
                detail = f": {result}" if isinstance(result, str) and result else ": OK"
                remaining = total - len(results)
                suffix = f"  ({remaining} remaining)" if remaining > 0 else ""
                print(f"  [{w.name}] {desc}{detail}{suffix}")
            except Exception as e:
                results[w.name] = (False, e)
                remaining = total - len(results)
                suffix = f"  ({remaining} remaining)" if remaining > 0 else ""
                print(f"  [{w.name}] {desc}: FAILED — {e}{suffix}")
    return results


def show_workers():
    """Run 'provision.py ls' to show available workers."""
    provision = Path(__file__).resolve().parent / "provision.py"
    subprocess.run([sys.executable, str(provision), "ls"])


def parse_worker_names(raw: str | None) -> list[str] | None:
    """Parse comma-separated worker names, returning None for 'all'."""
    if not raw:
        return None
    return [n.strip() for n in raw.split(",") if n.strip()]


def report_results(results: dict[str, tuple[bool, object]]) -> bool:
    """Print summary and return True if all succeeded."""
    failed = [name for name, (ok, _) in results.items() if not ok]
    if failed:
        print(f"\nFailed: {failed}")
        return False
    print(f"\nAll {len(results)} worker(s) OK.")
    return True
