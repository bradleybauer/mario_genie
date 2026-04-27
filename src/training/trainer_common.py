from __future__ import annotations

import argparse
import math
import random
import shlex
import sys
from dataclasses import dataclass
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
    argv = [str(part) for part in sys.argv]
    runtime_payload: dict[str, Any] = {
        "device": str(device),
        "mixed_precision": mixed_precision,
        "num_processes": int(num_processes),
        "timestamp": datetime.now().isoformat(),
        "argv": argv,
        "command": " ".join(shlex.quote(part) for part in argv),
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


def build_constant_scheduler(optimizer: Optimizer) -> LambdaLR:
    return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


def build_scheduler_from_metadata(
    optimizer: Optimizer,
    metadata: dict[str, Any],
) -> LambdaLR:
    scheduler_name = str(metadata.get("name", "warmup_cosine"))
    if scheduler_name == "warmup_cosine":
        return build_warmup_cosine_scheduler(
            optimizer,
            max_steps=int(metadata["max_steps"]),
            warmup_steps=int(metadata.get("warmup_steps", 0)),
            min_lr_scale=float(metadata.get("min_lr_scale", 0.1)),
        )
    if scheduler_name == "constant":
        return build_constant_scheduler(optimizer)
    raise ValueError(f"Unsupported scheduler metadata name: {scheduler_name}")


def set_optimizer_learning_rate(optimizer: Optimizer, lr: float) -> None:
    lr = float(lr)
    for group in optimizer.param_groups:
        group["lr"] = lr
        group["initial_lr"] = lr


def resolve_resume_max_steps(*, start_step: int, max_steps: int, resume_extra_steps: int) -> int:
    if resume_extra_steps > 0:
        return start_step + resume_extra_steps
    return max_steps


def warmup_cosine_scheduler_metadata(
    *,
    max_steps: int,
    warmup_steps: int,
    min_lr_scale: float,
) -> dict[str, int | float | str]:
    return {
        "name": "warmup_cosine",
        "max_steps": int(max_steps),
        "warmup_steps": int(warmup_steps),
        "min_lr_scale": float(min_lr_scale),
    }


def infer_resume_scheduler_metadata(
    checkpoint: dict[str, Any],
    *,
    start_step: int,
    fallback_max_steps: int,
    warmup_steps: int,
    min_lr_scale: float,
) -> dict[str, int | float | str]:
    metadata = checkpoint.get("scheduler_metadata")
    if isinstance(metadata, dict) and metadata.get("name"):
        return metadata

    inferred_max_steps = max(int(fallback_max_steps), 1)
    if start_step >= inferred_max_steps:
        inferred_max_steps = max(start_step, 1)

    return warmup_cosine_scheduler_metadata(
        max_steps=inferred_max_steps,
        warmup_steps=warmup_steps,
        min_lr_scale=min_lr_scale,
    )


def checkpoint_last_lr(checkpoint: dict[str, Any]) -> float | None:
    scheduler_state = checkpoint.get("scheduler")
    if not isinstance(scheduler_state, dict):
        return None

    last_lr = scheduler_state.get("_last_lr")
    if isinstance(last_lr, list) and last_lr:
        try:
            return float(last_lr[0])
        except (TypeError, ValueError):
            return None
    return None


def add_resume_scheduler_args(
    parser: argparse.ArgumentParser,
    *,
    default_tail_final_lr_scale: float = 0.25,
) -> None:
    parser.add_argument(
        "--resume-lr-mode",
        type=str,
        default="state",
        choices=("state", "constant", "tail", "restart"),
        help="How to handle generator LR when resuming: restore scheduler state, hold constant, start a cosine tail from the resumed LR, or restart warmup+cosine from --lr.",
    )
    parser.add_argument(
        "--resume-extra-steps",
        type=int,
        default=0,
        help="When resuming, train this many additional steps beyond the checkpoint step and use that as the effective max step budget.",
    )
    parser.add_argument(
        "--resume-tail-final-lr-scale",
        type=float,
        default=default_tail_final_lr_scale,
        help="For --resume-lr-mode tail, decay from the resumed LR to this multiplicative scale of that LR.",
    )


def add_mlflow_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable MLflow tracking for this training run.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="mario",
        help="MLflow experiment name to log into when --mlflow is enabled.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="Optional MLflow tracking URI. Defaults to the current MLflow environment configuration.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="Optional MLflow run name override. Defaults to --run-name or the output directory name.",
    )
    parser.add_argument(
        "--mlflow-tag",
        action="append",
        default=None,
        help="Additional MLflow tag in key=value format. Pass multiple times to add multiple tags.",
    )
    parser.add_argument(
        "--mlflow-log-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload the full output directory to MLflow when training finishes.",
    )


def validate_resume_scheduler_args(
    parser: argparse.ArgumentParser,
    args: Any,
) -> None:
    if args.resume_extra_steps < 0:
        parser.error("--resume-extra-steps must be >= 0")
    if args.resume_tail_final_lr_scale <= 0.0:
        parser.error("--resume-tail-final-lr-scale must be > 0")
    if args.resume_extra_steps > 0 and args.resume_from is None:
        parser.error("--resume-extra-steps requires --resume-from")


@dataclass
class SchedulerSetup:
    scheduler: LambdaLR
    scheduler_metadata: dict[str, int | float | str]
    max_steps: int
    start_step: int
    log_messages: list[str]


def configure_resume_scheduler(
    optimizer: Optimizer,
    *,
    max_steps: int,
    warmup_steps: int,
    min_lr_scale: float,
    checkpoint: dict[str, Any] | None = None,
    resume_lr_mode: str = "state",
    resume_extra_steps: int = 0,
    resume_tail_final_lr_scale: float = 0.25,
    restart_base_lr: float | None = None,
) -> SchedulerSetup:
    max_steps = int(max_steps)
    warmup_steps = max(int(warmup_steps), 0)
    log_messages: list[str] = []

    if checkpoint is None:
        scheduler_metadata = warmup_cosine_scheduler_metadata(
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            min_lr_scale=min_lr_scale,
        )
        scheduler = build_scheduler_from_metadata(optimizer, scheduler_metadata)
        if warmup_steps > 0:
            floor_pct = int(round(min_lr_scale * 100))
            log_messages.append(f"[lr] Warmup: {warmup_steps} steps -> cosine decay to {floor_pct}% of peak")
        else:
            log_messages.append("[lr] Cosine decay (no warmup)")
        return SchedulerSetup(
            scheduler=scheduler,
            scheduler_metadata=scheduler_metadata,
            max_steps=max_steps,
            start_step=0,
            log_messages=log_messages,
        )

    start_step = int(checkpoint["step"]) + 1
    max_steps = resolve_resume_max_steps(
        start_step=start_step,
        max_steps=max_steps,
        resume_extra_steps=resume_extra_steps,
    )
    if resume_extra_steps > 0:
        log_messages.append(
            f"[resume] Extending training by {resume_extra_steps} steps to max_steps={max_steps}."
        )

    remaining_steps = max(max_steps - start_step, 0)
    resumed_lr = float(optimizer.param_groups[0]["lr"])

    if resume_lr_mode == "state":
        fallback_schedule_max_steps = max_steps
        if resume_extra_steps > 0 and "scheduler_metadata" not in checkpoint:
            fallback_schedule_max_steps = max(start_step, 1)
        scheduler_metadata = infer_resume_scheduler_metadata(
            checkpoint,
            start_step=start_step,
            fallback_max_steps=fallback_schedule_max_steps,
            warmup_steps=warmup_steps,
            min_lr_scale=min_lr_scale,
        )
        scheduler = build_scheduler_from_metadata(optimizer, scheduler_metadata)
        scheduler.load_state_dict(checkpoint["scheduler"])
        log_messages.append("[resume-lr] Restored scheduler state from checkpoint.")
    elif resume_lr_mode == "constant":
        checkpoint_lr = checkpoint_last_lr(checkpoint)
        if checkpoint_lr is not None and not math.isclose(
            resumed_lr,
            checkpoint_lr,
            rel_tol=1e-6,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "Constant LR resume mismatch: resumed optimizer LR "
                f"{resumed_lr:.2e} does not match checkpoint scheduler LR {checkpoint_lr:.2e}."
            )
        set_optimizer_learning_rate(optimizer, resumed_lr)
        scheduler_metadata = {"name": "constant"}
        scheduler = build_constant_scheduler(optimizer)
        constant_scheduler_lr = float(scheduler.get_last_lr()[0])
        if not math.isclose(
            resumed_lr,
            constant_scheduler_lr,
            rel_tol=1e-6,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "Constant LR resume mismatch: constant scheduler initialized at "
                f"{constant_scheduler_lr:.2e} instead of resumed LR {resumed_lr:.2e}."
            )
        log_messages.append(f"[resume-lr] Holding LR constant at {resumed_lr:.2e}.")
        log_messages.append(
            f"[resume-lr][sanity] Constant mode confirmed at {constant_scheduler_lr:.2e}."
        )
    elif resume_lr_mode == "tail":
        set_optimizer_learning_rate(optimizer, resumed_lr)
        scheduler_metadata = warmup_cosine_scheduler_metadata(
            max_steps=max(remaining_steps, 1),
            warmup_steps=0,
            min_lr_scale=resume_tail_final_lr_scale,
        )
        scheduler = build_scheduler_from_metadata(optimizer, scheduler_metadata)
        log_messages.append(
            "[resume-lr] Cosine tail from "
            f"{resumed_lr:.2e} to {resumed_lr * resume_tail_final_lr_scale:.2e} "
            f"over {remaining_steps} step(s)."
        )
    elif resume_lr_mode == "restart":
        if restart_base_lr is None:
            raise ValueError("restart_base_lr is required when resume_lr_mode='restart'")
        set_optimizer_learning_rate(optimizer, restart_base_lr)
        scheduler_metadata = warmup_cosine_scheduler_metadata(
            max_steps=max(remaining_steps, 1),
            warmup_steps=warmup_steps,
            min_lr_scale=min_lr_scale,
        )
        scheduler = build_scheduler_from_metadata(optimizer, scheduler_metadata)
        log_messages.append(
            "[resume-lr] Restarted warmup+cosine schedule at "
            f"{restart_base_lr:.2e} for {remaining_steps} remaining step(s)."
        )
    else:
        raise ValueError(f"Unsupported resume_lr_mode: {resume_lr_mode}")

    return SchedulerSetup(
        scheduler=scheduler,
        scheduler_metadata=scheduler_metadata,
        max_steps=max_steps,
        start_step=start_step,
        log_messages=log_messages,
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