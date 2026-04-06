from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset


@dataclass(frozen=True)
class AcceleratorRuntime:
    accelerator: Accelerator
    device: torch.device
    is_main_process: bool


def create_accelerator_runtime(*, output_dir: Path, mixed_precision: str) -> AcceleratorRuntime:
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        project_config=ProjectConfiguration(project_dir=str(output_dir)),
    )
    return AcceleratorRuntime(
        accelerator=accelerator,
        device=accelerator.device,
        is_main_process=accelerator.is_main_process,
    )


def split_train_eval_dataset(
    dataset: Dataset,
    *,
    eval_samples: int,
    seed: int,
) -> tuple[Dataset, Dataset | None]:
    if eval_samples <= 0 or len(dataset) <= eval_samples:
        return dataset, None

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(dataset), generator=generator)
    eval_indices = permutation[:eval_samples].tolist()
    train_indices = permutation[eval_samples:].tolist()
    return Subset(dataset, train_indices), Subset(dataset, eval_indices)


def build_replacement_train_loader(
    train_dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    sampler_num_samples: int = 10**7,
) -> DataLoader:
    effective_workers = max(num_workers, 0)
    sampler = RandomSampler(train_dataset, replacement=True, num_samples=sampler_num_samples)
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=pin_memory,
        persistent_workers=effective_workers > 0,
        prefetch_factor=2 if effective_workers > 0 else None,
    )


def build_eval_loader(
    eval_dataset: Dataset | None,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader | None:
    if eval_dataset is None:
        return None
    return DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def infinite_batches(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def build_progress(*, use_live: bool, refresh_per_second: int = 2) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}"),
        disable=not use_live,
        refresh_per_second=refresh_per_second,
    )


def advance_progress(
    progress: Progress,
    task_id: TaskID,
    *,
    advance: float = 1,
    status: str | None = None,
) -> None:
    update_kwargs: dict[str, Any] = {"advance": advance}
    if status is not None:
        task = progress.tasks[task_id]
        previous_status = task.fields.get("status", "")
        update_kwargs["status"] = status or previous_status
    progress.update(task_id, **update_kwargs)


@dataclass
class ThroughputTracker:
    window_steps: int = 20
    _window_start: float = field(default_factory=time.time)
    _window_steps: int = 0
    _window_samples: int = 0

    def update(self, *, samples: int) -> tuple[float, float]:
        self._window_steps += 1
        self._window_samples += int(samples)

        now = time.time()
        elapsed = max(now - self._window_start, 1e-9)
        samples_per_second = self._window_samples / elapsed
        steps_per_second = self._window_steps / elapsed

        if self._window_steps >= self.window_steps:
            self._window_start = now
            self._window_steps = 0
            self._window_samples = 0

        return float(samples_per_second), float(steps_per_second)


def save_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=indent)


def save_metrics_json(path: Path, metrics: list[dict[str, Any]]) -> None:
    save_json(path, metrics, indent=2)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    unwrapped = model
    while True:
        changed = False
        module = getattr(unwrapped, "module", None)
        if module is not None:
            unwrapped = module
            changed = True
        orig_mod = getattr(unwrapped, "_orig_mod", None)
        if orig_mod is not None:
            unwrapped = orig_mod
            changed = True
        if not changed:
            break
    return unwrapped


def get_model_state_dict(
    model: torch.nn.Module,
    *,
    accelerator: Accelerator | None = None,
) -> dict[str, Any]:
    if accelerator is not None:
        return accelerator.get_state_dict(model)
    return unwrap_model(model).state_dict()
