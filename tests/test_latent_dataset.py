from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(PROJECT_ROOT / "src")
sys.path.insert(0, SRC_DIR)

from src.data.latent_dataset import LatentSequenceDataset


def test_latent_dataset_returns_temporal_windows(tmp_path: Path) -> None:
    latents = np.arange(4 * 8 * 2 * 3, dtype=np.float16).reshape(4, 8, 2, 3)
    actions = np.arange(8, dtype=np.uint8)
    np.savez_compressed(tmp_path / "sample.npz", latents=latents, actions=actions)

    dataset = LatentSequenceDataset(tmp_path, clip_frames=4, include_actions=True)
    sample = dataset[2]

    assert len(dataset) == 5
    assert dataset.latent_channels == 4
    assert dataset.latent_height == 2
    assert dataset.latent_width == 3
    assert sample["latents"].shape == (4, 4, 2, 3)
    assert sample["actions"].tolist() == [2, 3, 4, 5]


def test_latent_dataset_rejects_action_length_mismatch(tmp_path: Path) -> None:
    latents = np.zeros((4, 8, 2, 3), dtype=np.float16)
    actions = np.zeros((7,), dtype=np.uint8)
    np.savez_compressed(tmp_path / "bad.npz", latents=latents, actions=actions)

    with pytest.raises(ValueError, match="latent steps"):
        LatentSequenceDataset(tmp_path, clip_frames=4, include_actions=True)