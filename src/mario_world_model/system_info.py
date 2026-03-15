"""Collect and display comprehensive system and GPU hardware info."""

import os
import platform
import shutil
import subprocess

import torch


def collect_system_info():
    """Collect comprehensive system and GPU hardware info."""
    info = {}

    # ── CPU ──
    info["cpu_model"] = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        info["cpu_cores_physical"] = os.cpu_count()
        info["cpu_cores_available"] = len(os.sched_getaffinity(0))
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            freqs = [float(line.split(":")[1]) for line in f if line.startswith("cpu MHz")]
        if freqs:
            info["cpu_mhz_current"] = round(max(freqs), 0)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["lscpu"], text=True, stderr=subprocess.DEVNULL, timeout=5
        )
        for line in out.splitlines():
            k, _, v = line.partition(":")
            v = v.strip()
            if "max MHz" in k.lower():
                info["cpu_mhz_max"] = round(float(v), 0)
            elif "min MHz" in k.lower():
                info["cpu_mhz_min"] = round(float(v), 0)
            elif k.strip().startswith("L1d cache"):
                info["cpu_cache_l1d"] = v
            elif k.strip().startswith("L1i cache"):
                info["cpu_cache_l1i"] = v
            elif k.strip().startswith("L2 cache"):
                info["cpu_cache_l2"] = v
            elif k.strip().startswith("L3 cache"):
                info["cpu_cache_l3"] = v
    except Exception:
        pass

    # ── System RAM ──
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["ram_total_gb"] = round(int(line.split()[1]) / 2**20, 1)
                elif line.startswith("MemAvailable:"):
                    info["ram_available_gb"] = round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        pass

    # Memory speed/type via dmidecode (best-effort, works if user has access)
    try:
        out = subprocess.check_output(
            ["dmidecode", "-t", "memory"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Type:") and "ram_type" not in info:
                val = line.split(":", 1)[1].strip()
                if val not in ("Unknown", "Other"):
                    info["ram_type"] = val
            elif line.startswith("Speed:") and "ram_speed_mhz" not in info:
                val = line.split(":", 1)[1].strip()
                if "MT/s" in val or "MHz" in val:
                    info["ram_speed_mhz"] = int(val.split()[0])
    except Exception:
        pass

    # ── Disk ──
    try:
        usage = shutil.disk_usage("/")
        info["disk_total_gb"] = round(usage.total / 2**30, 1)
        info["disk_used_gb"] = round(usage.used / 2**30, 1)
        info["disk_free_gb"] = round(usage.free / 2**30, 1)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["lsblk", "-dno", "NAME,ROTA,SIZE,MODEL"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        for line in out.splitlines():
            parts = line.split(None, 3)
            if len(parts) >= 2:
                name, rota = parts[0], parts[1]
                # Skip loop, ram, rom devices
                if any(name.startswith(p) for p in ("loop", "ram", "sr")):
                    continue
                info["disk_device"] = f"/dev/{name}"
                info["disk_type"] = "HDD" if rota == "1" else "SSD"
                if len(parts) >= 3:
                    info["disk_size"] = parts[2]
                if len(parts) >= 4:
                    info["disk_model"] = parts[3].strip()
                break  # first real disk
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["df", "-T", "/"], text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        lines = out.strip().splitlines()
        if len(lines) >= 2:
            info["disk_filesystem"] = lines[1].split()[1]
    except Exception:
        pass
    # Bus type and theoretical max bandwidth
    if "disk_device" in info:
        dev_name = info["disk_device"].split("/")[-1]  # e.g. "sda" or "nvme0n1"
        try:
            # Query all devices (avoids "not a block device" in containers)
            out = subprocess.check_output(
                ["lsblk", "-dno", "NAME,TRAN"],
                text=True, stderr=subprocess.DEVNULL, timeout=5,
            )
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] == dev_name:
                    info["disk_bus"] = parts[1]
                    break
        except Exception:
            pass
        bus = info.get("disk_bus", "")
        bus_max_mbps = {"sata": 550, "ata": 550, "usb": 625, "nvme": 3500}
        if bus == "nvme":
            # Check NVMe link speed/width from sysfs for accurate estimate
            try:
                ctrl = dev_name.rstrip("0123456789").rstrip("np")  # nvme0n1 -> nvme0
                with open(f"/sys/class/nvme/{ctrl}/device/current_link_speed") as f:
                    speed_str = f.read().strip()  # e.g. "32.0 GT/s PCIe"
                with open(f"/sys/class/nvme/{ctrl}/device/current_link_width") as f:
                    width = int(f.read().strip())
                gt_s = float(speed_str.split()[0])
                encoding = 128 / 130 if gt_s >= 8.0 else 8 / 10
                info["disk_max_mbps"] = round(gt_s * width * encoding / 8 * 1000)
            except Exception:
                info["disk_max_mbps"] = bus_max_mbps["nvme"]
        elif bus in bus_max_mbps:
            info["disk_max_mbps"] = bus_max_mbps[bus]

    # ── GPU ──
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_total_mem_gb"] = round(props.total_memory / 2**30, 1)
        info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        info["gpu_streaming_multiprocessors"] = props.multi_processor_count

        # nvidia-smi queries for detailed GPU info
        _nvsmi_fields = {
            "gpu_driver_version": "driver_version",
            "gpu_pcie_gen": "pcie.link.gen.current",
            "gpu_pcie_gen_max": "pcie.link.gen.max",
            "gpu_pcie_width": "pcie.link.width.current",
            "gpu_pcie_width_max": "pcie.link.width.max",
            "gpu_clock_graphics_mhz": "clocks.current.graphics",
            "gpu_clock_mem_mhz": "clocks.current.memory",
            "gpu_clock_graphics_max_mhz": "clocks.max.graphics",
            "gpu_clock_mem_max_mhz": "clocks.max.memory",
            "gpu_power_draw_w": "power.draw",
            "gpu_power_limit_w": "power.limit",
        }
        query = ",".join(_nvsmi_fields.values())
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=5,
            ).strip()
            values = [v.strip() for v in out.split(",")]
            for (key, _), val in zip(_nvsmi_fields.items(), values):
                if val and val not in ("[Not Supported]", "N/A", "[N/A]"):
                    try:
                        info[key] = int(val) if val.isdigit() else float(val)
                    except ValueError:
                        info[key] = val
        except Exception:
            pass

        # Compute PCIe bandwidth from gen + width
        pcie_gen_speeds = {1: 2.5, 2: 5.0, 3: 8.0, 4: 16.0, 5: 32.0, 6: 64.0}
        gen = info.get("gpu_pcie_gen")
        width = info.get("gpu_pcie_width")
        if gen and width:
            try:
                lane_gbps = pcie_gen_speeds.get(int(gen), 0)
                # Effective throughput: raw * encoding efficiency (128b/130b for gen3+)
                encoding = 128 / 130 if int(gen) >= 3 else 8 / 10
                info["gpu_pcie_bandwidth_gbps"] = round(lane_gbps * int(width) * encoding / 8, 1)
            except (ValueError, TypeError):
                pass

    # ── PyTorch / CUDA ──
    info["torch_version"] = torch.__version__
    info["cuda_version"] = torch.version.cuda or "N/A"
    info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
    info["python_version"] = platform.python_version()
    info["platform"] = platform.platform()

    return info


def print_system_info(info):
    """Pretty-print system info dict."""
    print("\n" + "=" * 60)
    print("  SYSTEM INFO")
    print("=" * 60)

    sections = [
        ("CPU", ["cpu_model", "cpu_cores_physical", "cpu_cores_available",
                 "cpu_mhz_current", "cpu_mhz_max",
                 "cpu_cache_l1d", "cpu_cache_l1i", "cpu_cache_l2", "cpu_cache_l3"]),
        ("RAM", ["ram_total_gb", "ram_available_gb", "ram_type", "ram_speed_mhz"]),
        ("Disk", ["disk_device", "disk_model", "disk_type", "disk_bus",
                  "disk_size", "disk_filesystem",
                  "disk_total_gb", "disk_used_gb", "disk_free_gb", "disk_max_mbps"]),
        ("GPU", ["gpu_name", "gpu_total_mem_gb", "gpu_compute_capability",
                 "gpu_streaming_multiprocessors", "gpu_clock_graphics_mhz", "gpu_clock_graphics_max_mhz",
                 "gpu_clock_mem_mhz", "gpu_clock_mem_max_mhz",
                 "gpu_power_draw_w", "gpu_power_limit_w"]),
        ("PCIe", ["gpu_pcie_gen", "gpu_pcie_gen_max",
                  "gpu_pcie_width", "gpu_pcie_width_max",
                  "gpu_pcie_bandwidth_gbps"]),
        ("Software", ["torch_version", "cuda_version", "cudnn_version",
                      "python_version", "gpu_driver_version", "platform"]),
    ]

    units = {
        "ram_total_gb": "GB", "ram_available_gb": "GB",
        "ram_speed_mhz": "MT/s",
        "disk_total_gb": "GB", "disk_used_gb": "GB", "disk_free_gb": "GB",
        "disk_max_mbps": "MB/s",
        "gpu_total_mem_gb": "GB",
        "gpu_pcie_bandwidth_gbps": "GB/s",
        "gpu_clock_graphics_mhz": "MHz", "gpu_clock_mem_mhz": "MHz",
        "gpu_clock_graphics_max_mhz": "MHz", "gpu_clock_mem_max_mhz": "MHz",
        "cpu_mhz_current": "MHz", "cpu_mhz_max": "MHz",
        "gpu_power_draw_w": "W", "gpu_power_limit_w": "W",
    }

    for section, keys in sections:
        present = [(k, info[k]) for k in keys if k in info]
        if not present:
            continue
        print(f"\n  [{section}]")
        for k, v in present:
            label = k.replace("gpu_", "").replace("cpu_", "").replace("ram_", "").replace("disk_", "").replace("_", " ").title()
            unit = units.get(k, "")
            print(f"    {label:.<35s} {v} {unit}")

    print("=" * 60 + "\n")
