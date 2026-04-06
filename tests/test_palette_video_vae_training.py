from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.training.palette_video_vae_training import apply_palette_index_augmentation


def test_palette_index_augmentation_can_be_disabled() -> None:
    frames = torch.tensor([[[[0, 1], [2, 3]]]], dtype=torch.long)
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)

    augmented = apply_palette_index_augmentation(
        frames,
        sample_prob=1.0,
        replacement_prob=0.0,
        replacement_probs=probs,
    )

    assert augmented is frames


def test_palette_index_augmentation_can_be_disabled_per_sample() -> None:
    frames = torch.tensor([[[[0, 1], [2, 3]]]], dtype=torch.long)
    probs = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

    augmented = apply_palette_index_augmentation(
        frames,
        sample_prob=0.0,
        replacement_prob=1.0,
        replacement_probs=probs,
    )

    assert augmented is frames


def test_palette_index_augmentation_samples_from_distribution() -> None:
    torch.manual_seed(0)
    frames = torch.tensor(
        [
            [
                [[0, 1], [1, 0]],
                [[1, 0], [0, 1]],
            ]
        ],
        dtype=torch.long,
    )
    probs = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    augmented = apply_palette_index_augmentation(
        frames,
        sample_prob=1.0,
        replacement_prob=1.0,
        replacement_probs=probs,
    )

    assert augmented.shape == frames.shape
    assert augmented.dtype == frames.dtype
    assert torch.equal(augmented, torch.full_like(frames, 2))