from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.training.trainer_common import (
    build_constant_scheduler,
    build_scheduler_from_metadata,
    build_warmup_cosine_scheduler,
    configure_resume_scheduler,
    resolve_resume_max_steps,
    set_optimizer_learning_rate,
)


def _make_optimizer(lr: float = 2e-4) -> AdamW:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    return AdamW([parameter], lr=lr)


def test_build_constant_scheduler_keeps_learning_rate_fixed() -> None:
    optimizer = _make_optimizer(lr=3e-4)
    scheduler = build_constant_scheduler(optimizer)

    learning_rates = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates == [pytest.approx(3e-4)] * 4


def test_build_scheduler_from_metadata_restores_floor_after_original_horizon() -> None:
    optimizer = _make_optimizer(lr=2e-4)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        max_steps=5,
        warmup_steps=0,
        min_lr_scale=0.25,
    )
    for _ in range(5):
        optimizer.step()
        scheduler.step()

    floor_lr = optimizer.param_groups[0]["lr"]
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()

    resumed_optimizer = _make_optimizer(lr=2e-4)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler = build_scheduler_from_metadata(
        resumed_optimizer,
        {
            "name": "warmup_cosine",
            "max_steps": 5,
            "warmup_steps": 0,
            "min_lr_scale": 0.25,
        },
    )
    resumed_scheduler.load_state_dict(scheduler_state)

    resumed_lrs = []
    for _ in range(3):
        resumed_optimizer.step()
        resumed_scheduler.step()
        resumed_lrs.append(resumed_optimizer.param_groups[0]["lr"])

    assert resumed_lrs == [pytest.approx(floor_lr)] * 3


def test_tail_scheduler_metadata_decays_from_resumed_lr_without_jump() -> None:
    optimizer = _make_optimizer(lr=2e-4)
    resumed_lr = 5e-5
    set_optimizer_learning_rate(optimizer, resumed_lr)
    scheduler = build_scheduler_from_metadata(
        optimizer,
        {
            "name": "warmup_cosine",
            "max_steps": 4,
            "warmup_steps": 0,
            "min_lr_scale": 0.5,
        },
    )

    learning_rates = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[0] <= resumed_lr
    assert learning_rates[-1] == pytest.approx(2.5e-5)
    assert all(learning_rates[index] >= learning_rates[index + 1] for index in range(len(learning_rates) - 1))


def test_resolve_resume_max_steps_uses_extra_budget_from_checkpoint() -> None:
    assert resolve_resume_max_steps(start_step=1_000, max_steps=2_000, resume_extra_steps=250) == 1_250
    assert resolve_resume_max_steps(start_step=1_000, max_steps=2_000, resume_extra_steps=0) == 2_000


def test_configure_resume_scheduler_constant_mode_extends_from_checkpoint() -> None:
    optimizer = _make_optimizer(lr=2e-4)
    warmup_scheduler = build_warmup_cosine_scheduler(
        optimizer,
        max_steps=10,
        warmup_steps=0,
        min_lr_scale=0.25,
    )
    for _ in range(10):
        optimizer.step()
        warmup_scheduler.step()

    checkpoint = {
        "step": 9,
        "scheduler": warmup_scheduler.state_dict(),
        "scheduler_metadata": {
            "name": "warmup_cosine",
            "max_steps": 10,
            "warmup_steps": 0,
            "min_lr_scale": 0.25,
        },
    }

    setup = configure_resume_scheduler(
        optimizer,
        max_steps=10,
        warmup_steps=0,
        min_lr_scale=0.25,
        checkpoint=checkpoint,
        resume_lr_mode="constant",
        resume_extra_steps=5,
    )

    assert setup.start_step == 10
    assert setup.max_steps == 15
    assert setup.scheduler_metadata == {"name": "constant"}
    assert any("Extending training by 5 steps" in message for message in setup.log_messages)
    assert any("Holding LR constant" in message for message in setup.log_messages)
    assert any("Constant mode confirmed" in message for message in setup.log_messages)
    assert setup.scheduler.get_last_lr()[0] == pytest.approx(optimizer.param_groups[0]["lr"])


def test_configure_resume_scheduler_constant_mode_rejects_mismatched_checkpoint_lr() -> None:
    optimizer = _make_optimizer(lr=2e-4)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        max_steps=10,
        warmup_steps=0,
        min_lr_scale=0.25,
    )
    for _ in range(10):
        optimizer.step()
        scheduler.step()

    mismatched_scheduler_state = scheduler.state_dict()
    mismatched_scheduler_state["_last_lr"] = [9e-5]
    checkpoint = {
        "step": 9,
        "scheduler": mismatched_scheduler_state,
        "scheduler_metadata": {
            "name": "warmup_cosine",
            "max_steps": 10,
            "warmup_steps": 0,
            "min_lr_scale": 0.25,
        },
    }

    with pytest.raises(RuntimeError, match="Constant LR resume mismatch"):
        configure_resume_scheduler(
            optimizer,
            max_steps=10,
            warmup_steps=0,
            min_lr_scale=0.25,
            checkpoint=checkpoint,
            resume_lr_mode="constant",
            resume_extra_steps=5,
        )


def test_configure_resume_scheduler_state_mode_preserves_floor_for_legacy_checkpoint() -> None:
    optimizer = _make_optimizer(lr=2e-4)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        max_steps=5,
        warmup_steps=0,
        min_lr_scale=0.25,
    )
    for _ in range(5):
        optimizer.step()
        scheduler.step()

    floor_lr = optimizer.param_groups[0]["lr"]
    checkpoint = {
        "step": 4,
        "scheduler": scheduler.state_dict(),
    }

    setup = configure_resume_scheduler(
        optimizer,
        max_steps=20,
        warmup_steps=0,
        min_lr_scale=0.25,
        checkpoint=checkpoint,
        resume_lr_mode="state",
        resume_extra_steps=5,
    )

    optimizer.step()
    setup.scheduler.step()

    assert setup.start_step == 5
    assert setup.max_steps == 10
    assert optimizer.param_groups[0]["lr"] == pytest.approx(floor_lr)


def test_configure_resume_scheduler_tail_mode_starts_from_resumed_lr() -> None:
    optimizer = _make_optimizer(lr=2e-4)
    warmup_scheduler = build_warmup_cosine_scheduler(
        optimizer,
        max_steps=10,
        warmup_steps=0,
        min_lr_scale=0.25,
    )
    for _ in range(10):
        optimizer.step()
        warmup_scheduler.step()

    resumed_lr = optimizer.param_groups[0]["lr"]
    checkpoint = {
        "step": 9,
        "scheduler": warmup_scheduler.state_dict(),
        "scheduler_metadata": {
            "name": "warmup_cosine",
            "max_steps": 10,
            "warmup_steps": 0,
            "min_lr_scale": 0.25,
        },
    }

    setup = configure_resume_scheduler(
        optimizer,
        max_steps=10,
        warmup_steps=0,
        min_lr_scale=0.25,
        checkpoint=checkpoint,
        resume_lr_mode="tail",
        resume_extra_steps=4,
        resume_tail_final_lr_scale=0.5,
    )

    assert setup.scheduler.get_last_lr()[0] == pytest.approx(resumed_lr)
    optimizer.step()
    setup.scheduler.step()
    assert optimizer.param_groups[0]["lr"] <= resumed_lr
