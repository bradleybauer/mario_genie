from __future__ import annotations

import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def format_bytes(num_bytes: int) -> str:
    num_bytes = max(int(num_bytes), 0)
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda_runtime(
    *,
    enable_tf32: bool = True,
    cudnn_benchmark: bool = True,
    matmul_precision: str | None = None,
) -> None:
    if not torch.cuda.is_available():
        return
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32
    torch.backends.cudnn.benchmark = cudnn_benchmark


def make_output_dir(
    *,
    project_root: Path,
    output_dir: str | Path | None,
    resume_from: str | Path | None,
    run_name: str | None,
    default_prefix: str,
    resume_parent: bool = True,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if resume_from is not None:
        resume_path = Path(resume_from).resolve()
        return resume_path.parent if resume_parent else resume_path
    resolved_run_name = run_name or datetime.now().strftime(f"{default_prefix}_%Y%m%d_%H%M%S")
    return project_root / "checkpoints" / resolved_run_name


def build_trainer_config(
    *,
    model_name: str,
    args: Any,
    device: torch.device,
    mixed_precision: str,
    num_processes: int,
    data: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_payload: dict[str, Any] = {
        "device": str(device),
        "mixed_precision": mixed_precision,
        "num_processes": int(num_processes),
        "timestamp": datetime.now().isoformat(),
    }
    if runtime is not None:
        runtime_payload.update(runtime)
    return {
        "model_name": model_name,
        "training": vars(args).copy(),
        "data": data or {},
        "model": model or {},
        "runtime": runtime_payload,
    }


def is_periodic_event_due(step: int, *, interval: int, max_steps: int) -> bool:
    if max_steps <= 0:
        return False
    if step == max_steps - 1:
        return True
    return interval > 0 and (step + 1) % interval == 0


def should_log_step(step: int, *, start_step: int, log_interval: int, max_steps: int) -> bool:
    if max_steps <= 0:
        return False
    if step == start_step or step == max_steps - 1:
        return True
    return log_interval > 0 and (step + 1) % log_interval == 0


def preview_path(output_dir: Path, *, split: str, step: int, suffix: str = ".png") -> Path:
    return output_dir / "previews" / f"{split}_step_{step:06d}{suffix}"


def warmup_cosine_lr_lambda(
    *,
    max_steps: int,
    warmup_steps: int,
    min_lr_scale: float,
):
    warmup_steps = max(int(warmup_steps), 0)
    min_lr_scale = float(min_lr_scale)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step + 1, 1) / float(warmup_steps)
        if max_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / float(max(max_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return lr_lambda


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    max_steps: int,
    warmup_steps: int,
    min_lr_scale: float = 0.1,
) -> LambdaLR:
    return LambdaLR(
        optimizer,
        lr_lambda=warmup_cosine_lr_lambda(
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            min_lr_scale=min_lr_scale,
        ),
    )


def gpu_stats(device: torch.device) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    index = device.index or 0
    stats: dict[str, float] = {}
    try:
        mem_used = torch.cuda.memory_allocated(index) / 2**30
        mem_total = torch.cuda.get_device_properties(index).total_memory / 2**30
        stats["gpu_mem_pct"] = round(100.0 * mem_used / max(mem_total, 1e-9), 1)
    except Exception:
        pass
    try:
        stats["gpu_util_pct"] = float(torch.cuda.utilization(index))
    except Exception:
        pass
    try:
        stats["gpu_temp_c"] = float(torch.cuda.temperature(index))
    except Exception:
        pass
    return stats