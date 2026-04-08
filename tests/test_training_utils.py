from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.training.training_utils import load_model_state_dict, normalize_state_dict_keys


def test_normalize_state_dict_keys_strips_compile_prefix() -> None:
    state_dict = {
        "_orig_mod.weight": torch.tensor([1.0]),
        "_orig_mod.bias": torch.tensor([2.0]),
    }

    normalized = normalize_state_dict_keys(state_dict)

    assert set(normalized.keys()) == {"weight", "bias"}


def test_load_model_state_dict_accepts_compiled_prefix_keys() -> None:
    model = torch.nn.Linear(3, 2)
    reference_model = torch.nn.Linear(3, 2)
    prefixed_state_dict = {
        f"_orig_mod.{key}": value.clone()
        for key, value in reference_model.state_dict().items()
    }

    load_model_state_dict(model, prefixed_state_dict)

    for key, value in reference_model.state_dict().items():
        assert torch.equal(model.state_dict()[key], value)


def test_load_model_state_dict_rejects_real_shape_mismatch() -> None:
    model = torch.nn.Linear(3, 2)
    bad_state_dict = {
        "_orig_mod.weight": torch.zeros(5, 5),
        "_orig_mod.bias": torch.zeros(2),
    }

    with pytest.raises(RuntimeError):
        load_model_state_dict(model, bad_state_dict)
