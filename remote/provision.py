#!/usr/bin/env python3
"""Provision Vast.ai instances and auto-generate config.py.

Usage:
    python remote/provision.py up                  # search & provision
    python remote/provision.py up --sort dlperf    # sort by DL perf
    python remote/provision.py up --gpu RTX_5090   # filter by GPU type
    python remote/provision.py ls                  # show running instances
    python remote/provision.py down                # tear down everything
    python remote/provision.py update-config       # regenerate config.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing 'requests' — pip install requests")

API = "https://console.vast.ai/api/v0"
CONFIG_PATH = Path(__file__).resolve().parent / "config.py"

IMAGE = "vastai/pytorch:cuda-13.0.2-auto"

SORT_OPTIONS = {
    "price": ["dph_total", "asc"],
    "dlperf": ["dlperf", "desc"],
    "value": ["dlperf_per_dphtotal", "desc"],
    "vram": ["gpu_ram", "desc"],
    "bw": ["gpu_mem_bw", "desc"],
}

FILTERS = {
    "reliability": {"gte": 0.95},
    "inet_down": {"gte": 1000},
    "inet_up": {"gte": 1000},
    "cpu_cores": {"gte": 32},
    "cpu_ram": {"gte": 99000},
    "gpu_ram": {"gt": 19000},
    "cuda_max_good": {"gte": 13.0},
    "num_gpus": {"eq": 1},
    "direct_port_count": {"gte": 1},
    "rentable": {"eq": True},
    "rented": {"eq": False},
    "verified": {"eq": True},
}

# ── API helpers ──────────────────────────────────────────────────


def _api_key() -> str:
    key = os.environ.get("VAST_API_KEY")
    if key:
        return key.strip()
    for p in [Path.home() / ".config/vastai/vast_api_key", Path.home() / ".vast_api_key"]:
        if p.is_file():
            key = p.read_text().strip()
            if key:
                return key
    sys.exit("No API key. Set VAST_API_KEY or run: vastai set api-key <KEY>")


def _req(method: str, path: str, api_key: str, **kwargs) -> dict | list:
    resp = requests.request(
        method, f"{API}{path}",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=30, **kwargs,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_sort(sort_str: str) -> list[list[str]]:
    keys = [k.strip() for k in sort_str.split(",")]
    order = []
    for k in keys:
        if k not in SORT_OPTIONS:
            sys.exit(f"Unknown sort key: {k}  (valid: {', '.join(SORT_OPTIONS)})")
        order.append(SORT_OPTIONS[k])
    return order


def _search(api_key: str, disk: int, sort: str, gpu: str | None = None) -> list[dict]:
    body = {
        "limit": 10000,
        "type": "on-demand",
        "order": _parse_sort(sort),
        "allocated_storage": disk,
        **FILTERS,
    }
    if gpu:
        body["gpu_name"] = {"eq": gpu.replace("_", " ")}
    data = _req("POST", "/bundles/", api_key, json=body)
    offers = data.get("offers", data) if isinstance(data, dict) else data
    return offers if isinstance(offers, list) else [offers]


def _create(api_key: str, offer_id: int, disk: int, image: str) -> dict:
    return _req("PUT", f"/asks/{offer_id}/", api_key,
                json={"image": image, "disk": disk, "runtype": "ssh_direct"})


def _instances(api_key: str) -> list[dict]:
    data = _req("GET", "/instances/", api_key)
    return data.get("instances", []) if isinstance(data, dict) else data


def _destroy(api_key: str, iid: int) -> dict:
    return _req("DELETE", f"/instances/{iid}/", api_key)


def _ssh_info(inst: dict) -> tuple[str, int] | None:
    ports = inst.get("ports") or {}
    ssh = ports.get("22/tcp")
    if ssh:
        ip, port = inst.get("public_ipaddr"), ssh[0].get("HostPort")
        if ip and port:
            return ip, int(port)
    host, port = inst.get("ssh_host"), inst.get("ssh_port")
    if host and port:
        return host, int(port)
    return None


# ── Table printer ────────────────────────────────────────────────


def _print_table(headers: tuple, rows: list[tuple]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print("  " + fmt.format(*headers))
    print("  " + "  ".join("─" * w for w in widths))
    for row in rows:
        print("  " + fmt.format(*row))


# ── Commands ─────────────────────────────────────────────────────


def cmd_up(api_key: str, args: argparse.Namespace) -> None:
    gpu_msg = f", gpu: {args.gpu}" if args.gpu else ""
    cores = FILTERS.get("cpu_cores", {}).get("gte", "?")
    ram_gb = int(FILTERS.get("cpu_ram", {}).get("gte", 0) / 1000)
    vram_gb = int(FILTERS.get("gpu_ram", {}).get("gt", FILTERS.get("gpu_ram", {}).get("gte", 0)) / 1000)
    inet = FILTERS.get("inet_down", {}).get("gte", "?")
    print(f"Searching for offers (sort: {args.sort}{gpu_msg}) ...")
    print(f"  Filters: ≥{cores} cores, ≥{ram_gb} GB RAM, >{vram_gb} GB VRAM, ≥{inet} Mbps")
    offers = _search(api_key, args.disk, args.sort, gpu=args.gpu)
    if not offers:
        sys.exit("No matching offers found.")

    print(f"  {len(offers)} offer(s), disk: {args.disk} GB\n")

    offer_by_row = {}
    row_by_id = {}
    rows = []
    for i, o in enumerate(offers, 1):
        offer_by_row[i] = o
        gpu_dph = o.get("dph_base", o.get("dph_total", 0))
        disk_dph = o.get("storage_cost", 0.15) * args.disk / 730
        total = gpu_dph + disk_dph
        dlp = o.get("dlperf", 0) or 0
        dlp_dollar = dlp / total if total > 0 else 0
        rows.append((
            f"{i})",
            f"{o.get('num_gpus', '?')}×{o.get('gpu_name', '?')}",
            f"{o.get('gpu_ram', 0) / 1000:.0f} GB",
            f"{o.get('gpu_mem_bw', 0):.0f}",
            f"{o.get('cpu_cores', '?')}",
            f"{o.get('cpu_ram', 0) / 1000:.0f} GB",
            f"{dlp:.1f}",
            f"{dlp_dollar:.1f}",
            f"${total:.3f}/hr",
            o.get("geolocation", "?"),
        ))
        row_by_id[o["id"]] = rows[-1]

    _print_table(("", "GPU", "VRAM", "BW", "Cores", "RAM", "DLPerf", "DLP/$", "Total", "Location"), rows)

    print(f"\nPick row(s) to provision (e.g. '1' or '1,3,5'):")
    choice = input("> ").strip()
    if not choice:
        sys.exit("No selection.")
    try:
        selected = [offer_by_row[int(r.strip())] for r in choice.split(",") if int(r.strip()) in offer_by_row]
    except (ValueError, KeyError):
        sys.exit("Invalid input.")
    if not selected:
        sys.exit("No valid selections.")

    # Cost summary
    total_cost = 0.0
    for o in selected:
        gpu_dph = o.get("dph_base", o.get("dph_total", 0))
        total_cost += gpu_dph + o.get("storage_cost", 0.15) * args.disk / 730

    headers = ("", "GPU", "VRAM", "BW", "Cores", "RAM", "DLPerf", "DLP/$", "Total", "Location")
    sel_rows = [row_by_id[o["id"]] for o in selected if o["id"] in row_by_id]
    print()
    _print_table(headers, sel_rows)
    print(f"\n  {len(selected)} instance(s): ${total_cost:.3f}/hr (${total_cost * 24:.2f}/day)")

    if input("  Proceed? [y/N] ").strip().lower() != "y":
        sys.exit("Aborted.")

    # Create & wait
    ids = []
    for o in selected:
        print(f"  Creating from offer #{o['id']} ...")
        r = _create(api_key, o["id"], args.disk, args.image)
        if r.get("success"):
            ids.append(r["new_contract"])
            print(f"    → instance {r['new_contract']}")
        else:
            print(f"    Failed: {r}")

    if not ids:
        sys.exit("No instances created.")

    print(f"\nWaiting for {len(ids)} instance(s) ...")
    ready = _wait(api_key, ids, args.timeout)
    if ready:
        # Write config for ALL running instances, not just new ones
        all_running = [i for i in _instances(api_key) if i.get("actual_status") == "running"]
        _write_config(all_running)

        # Show ls-style table for the new instances
        existing = _read_existing_config()
        id_to_name = {iid: w["name"] for iid, w in existing.items()}
        new_set = set(ids)
        new_running = [i for i in all_running if i["id"] in new_set]
        ls_rows = []
        for inst in new_running:
            gpu_dph = inst.get("dph_total", 0)
            dlp = inst.get("dlperf", 0) or 0
            dlp_dollar = dlp / gpu_dph if gpu_dph > 0 else 0
            mem_used = inst.get("mem_usage", 0) or 0
            mem_limit = inst.get("mem_limit", 0) or 0
            ls_rows.append((
                id_to_name.get(inst["id"], "?"),
                str(inst["id"]),
                inst.get("actual_status", "?"),
                inst.get("gpu_name", "?"),
                f"{inst.get('gpu_ram', 0) / 1000:.0f} GB",
                f"{inst.get('gpu_mem_bw', 0):.0f}",
                f"{inst.get('cpu_cores', '?')}",
                f"{dlp:.1f}",
                f"{dlp_dollar:.1f}",
                f"${gpu_dph:.3f}/hr",
                inst.get("geolocation", "?"),
                "   ",
                f"{inst.get('gpu_util', 0) or 0:.0f}%",
                f"{inst.get('gpu_temp', 0) or 0:.0f}°C",
                f"{inst.get('cpu_util', 0) or 0:.0f}%",
                f"{mem_used:.0f}/{mem_limit:.0f} GB",
            ))
        print()
        _print_table(("Name", "ID", "Status", "GPU", "VRAM", "BW", "Cores", "DLPerf", "DLP/$", "Total", "Location", "   ", "GPU%", "Temp", "CPU%", "RAM"), ls_rows)

        print(f"\nDone! Next: python remote/setup_all.py")
    else:
        sys.exit("No instances became ready.")


def _wait(api_key: str, ids: list[int], timeout: int) -> list[dict]:
    start = time.time()
    ready = {}
    while len(ready) < len(ids):
        if time.time() - start > timeout:
            print(f"  Timeout ({timeout}s) — {len(ready)}/{len(ids)} ready")
            break
        by_id = {i["id"]: i for i in _instances(api_key)}
        for iid in ids:
            if iid in ready:
                continue
            inst = by_id.get(iid)
            if inst and inst.get("actual_status") == "running" and _ssh_info(inst):
                ready[iid] = inst
                conn = _ssh_info(inst)
                print(f"    {iid} ready @ {conn[0]}:{conn[1]}")
        if len(ready) < len(ids):
            print(f"    {len(ids) - len(ready)} pending ... ({int(time.time() - start)}s)")
            time.sleep(15)
    return list(ready.values())


def cmd_ls(api_key: str, _args: argparse.Namespace) -> None:
    instances = _instances(api_key)
    if not instances:
        return print("No instances.")
    existing = _read_existing_config()
    id_to_name = {iid: w["name"] for iid, w in existing.items()}
    rows = []
    for inst in instances:
        conn = _ssh_info(inst)
        gpu_dph = inst.get("dph_total", 0)
        dlp = inst.get("dlperf", 0) or 0
        dlp_dollar = dlp / gpu_dph if gpu_dph > 0 else 0
        mem_used = inst.get("mem_usage", 0) or 0
        mem_limit = inst.get("mem_limit", 0) or 0
        rows.append((
            id_to_name.get(inst["id"], "?"),
            str(inst["id"]),
            inst.get("actual_status", "?"),
            inst.get("gpu_name", "?"),
            f"{inst.get('gpu_ram', 0) / 1000:.0f} GB",
            f"{inst.get('gpu_mem_bw', 0):.0f}",
            f"{inst.get('cpu_cores', '?')}",
            f"{dlp:.1f}",
            f"{dlp_dollar:.1f}",
            f"${gpu_dph:.3f}/hr",
            inst.get("geolocation", "?"),
            "   ",
            f"{inst.get('gpu_util', 0) or 0:.0f}%",
            f"{inst.get('gpu_temp', 0) or 0:.0f}°C",
            f"{inst.get('cpu_util', 0) or 0:.0f}%",
            f"{mem_used:.0f}/{mem_limit:.0f} GB",
        ))
    _print_table(("Name", "ID", "Status", "GPU", "VRAM", "BW", "Cores", "DLPerf", "DLP/$", "Total", "Location", "   ", "GPU%", "Temp", "CPU%", "RAM"), rows)


def cmd_down(api_key: str, args: argparse.Namespace) -> None:
    all_instances = _instances(api_key)
    if not all_instances:
        return print("No instances.")

    # Map config name → instance via instance_id
    existing = _read_existing_config()
    id_to_name = {iid: w["name"] for iid, w in existing.items()}

    def _inst_rows(insts: list[dict]) -> list[tuple]:
        rows = []
        for i in insts:
            gpu_dph = i.get("dph_total", 0)
            dlp = i.get("dlperf", 0) or 0
            dlp_dollar = dlp / gpu_dph if gpu_dph > 0 else 0
            mem_used = i.get("mem_usage", 0) or 0
            mem_limit = i.get("mem_limit", 0) or 0
            rows.append((
                id_to_name.get(i["id"], "?"),
                str(i["id"]),
                i.get("actual_status", "?"),
                i.get("gpu_name", "?"),
                f"{i.get('gpu_ram', 0) / 1000:.0f} GB",
                f"{i.get('gpu_mem_bw', 0):.0f}",
                f"{i.get('cpu_cores', '?')}",
                f"{dlp:.1f}",
                f"{dlp_dollar:.1f}",
                f"${gpu_dph:.3f}/hr",
                i.get("geolocation", "?"),
                "   ",
                f"{i.get('gpu_util', 0) or 0:.0f}%",
                f"{i.get('gpu_temp', 0) or 0:.0f}°C",
                f"{i.get('cpu_util', 0) or 0:.0f}%",
                f"{mem_used:.0f}/{mem_limit:.0f} GB",
            ))
        return rows

    headers = ("Name", "ID", "Status", "GPU", "VRAM", "BW", "Cores", "DLPerf", "DLP/$", "Total", "Location", "   ", "GPU%", "Temp", "CPU%", "RAM")

    if not args.name:
        _print_table(headers, _inst_rows(all_instances))
        sys.exit("\nUsage: provision.py down <id ...> | all")

    if "all" in args.name:
        targets = all_instances
    else:
        id_to_inst = {str(inst["id"]): inst for inst in all_instances}
        targets = []
        for n in args.name:
            if n in id_to_inst:
                targets.append(id_to_inst[n])
            else:
                print(f"  Unknown instance: {n}")
        if not targets:
            sys.exit("No matching instances.")

    _print_table(headers, _inst_rows(targets))
    if input(f"\nDestroy {len(targets)} instance(s)? [y/N] ").strip().lower() != "y":
        return print("Aborted.")
    for i in targets:
        try:
            _destroy(api_key, i["id"])
            name = id_to_name.get(i["id"])
            label = f"{name} ({i['id']})" if name else str(i["id"])
            print(f"  Destroyed {label}")
        except Exception as e:
            print(f"  Failed {i['id']}: {e}")

    # Update config to reflect remaining instances
    remaining = [i for i in _instances(api_key) if i.get("actual_status") == "running"]
    if remaining:
        _write_config(remaining)
    elif CONFIG_PATH.is_file():
        _write_config([])
        print("  Config cleared (no running instances)")


def cmd_update_config(api_key: str, _args: argparse.Namespace) -> None:
    running = [i for i in _instances(api_key) if i.get("actual_status") == "running"]
    if not running:
        sys.exit("No running instances.")
    _write_config(running)


def _read_existing_config() -> dict[int, dict]:
    """Parse existing config.py to get instance_id → worker dict mappings."""
    mapping: dict[int, dict] = {}
    if not CONFIG_PATH.is_file():
        return mapping
    try:
        ns: dict = {}
        exec(CONFIG_PATH.read_text(), ns)
        seen_names: set[str] = set()
        for w in ns.get("WORKERS", []):
            name = w["name"]
            if name in seen_names:
                sys.exit(f"Duplicate worker name '{name}' in {CONFIG_PATH}")
            seen_names.add(name)
            iid = w.get("instance_id")
            if iid is not None:
                mapping[int(iid)] = w
    except SystemExit:
        raise
    except Exception:
        pass
    return mapping


def _write_config(instances: list[dict]) -> None:
    existing = _read_existing_config()
    used_names = {w["name"] for w in existing.values()}

    # Assign names: preserve existing by instance_id, allocate next free letter for new
    entries = []
    for inst in instances:
        conn = _ssh_info(inst)
        if not conn:
            print(f"  Warning: {inst.get('id')} has no SSH info, skipping")
            continue
        host, port = conn
        iid = inst["id"]
        prev = existing.get(iid)
        name = prev["name"] if prev else None
        if not name:
            for c in (chr(ord("a") + j) for j in range(26)):
                if c not in used_names:
                    name = c
                    break
            else:
                name = f"w{len(entries)}"
        used_names.add(name)
        entries.append((name, host, port, iid, inst.get("gpu_name", "?")))

    # Sort by name for stable output
    entries.sort(key=lambda e: e[0])

    workers = []
    for name, host, port, iid, gpu in entries:
        workers.append(
            f'    {{"name": "{name}", "host": "{host}", "port": {port:<6}, "user": "root", "instance_id": {iid}}},  # {gpu}'
        )
    CONFIG_PATH.write_text(
        '"""\nMulti-machine worker inventory.\n\n'
        'Auto-generated by remote/provision.py — safe to edit manually.\n"""\n\n'
        '# fmt: off\nWORKERS = [\n' + "\n".join(workers) + '\n]\n# fmt: on\n\n'
        'PROJECT_NAME = "mario"\n'
    )
    print(f"\nWrote {CONFIG_PATH} with {len(workers)} worker(s)")


# ── CLI ──────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Vast.ai instance manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  provision.py up                       Search & provision instances
  provision.py up --gpu RTX_5090        Only show RTX 5090 offers
  provision.py up --sort dlperf         Sort by DL performance
  provision.py ls                       List all running instances with stats
  provision.py down 32920182            Destroy instance by ID
  provision.py down 32920182 32927114   Destroy multiple instances
  provision.py down all                 Destroy all instances
  provision.py update-config            Regenerate config.py from running instances

sort keys (comma-separated):
  price    Total $/hr ascending (default)
  dlperf   DL performance descending
  value    DL perf per dollar descending
  vram     GPU memory descending
  bw       GPU memory bandwidth descending
""",
    )
    sub = p.add_subparsers(dest="cmd")

    up = sub.add_parser("up", help="Search & provision instances",
                        description="Search Vast.ai for GPU offers matching filters, display results, and provision selected instances.")
    up.add_argument("--disk", type=int, default=50, help="Disk space in GB (default: 50)")
    up.add_argument("--image", default=IMAGE, help=f"Docker image (default: {IMAGE})")
    up.add_argument("--timeout", type=int, default=600, help="Seconds to wait for instances to become ready (default: 600)")
    up.add_argument("--sort", default="price",
                    help="Sort keys, comma-separated: price,dlperf,value,vram,bw (default: price)")
    up.add_argument("--gpu", default=None, help="Filter by GPU type, e.g. RTX_5090, RTX_4090")

    sub.add_parser("ls", help="List instances",
                   description="Show all Vast.ai instances with hardware specs and live usage stats.")

    down = sub.add_parser("down", help="Destroy instances",
                          description="Destroy one or more instances by ID. Run with no args to see available instances.")
    down.add_argument("name", nargs="*", help="Instance ID(s) to destroy, or 'all'")

    sub.add_parser("update-config", help="Regenerate config.py",
                   description="Regenerate remote/config.py from all currently running instances.")

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        sys.exit(1)

    key = _api_key()
    {"up": cmd_up, "ls": cmd_ls, "down": cmd_down, "update-config": cmd_update_config}[args.cmd](key, args)


if __name__ == "__main__":
    main()
