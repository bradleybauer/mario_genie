from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.data.latent_dataset import LatentSequenceDataset


def test_latent_dataset_returns_temporal_windows(tmp_path: Path) -> None:
    latents = np.arange(4 * 8 * 2 * 3, dtype=np.float16).reshape(4, 8, 2, 3)
    actions = np.arange(16, dtype=np.uint8).reshape(8, 2)
    np.savez_compressed(tmp_path / "sample.npz", latents=latents, actions=actions)

    dataset = LatentSequenceDataset(tmp_path, clip_latents=4, include_actions=True)
    sample = dataset[2]

    assert len(dataset) == 5
    assert dataset.latent_channels == 4
    assert dataset.latent_height == 2
    assert dataset.latent_width == 3
    assert sample["latents"].shape == (4, 4, 2, 3)
    assert dataset.action_frame_count == 2
    assert sample["actions"].tolist() == [[4, 5], [6, 7], [8, 9], [10, 11]]


def test_latent_dataset_rejects_action_length_mismatch(tmp_path: Path) -> None:
    latents = np.zeros((4, 8, 2, 3), dtype=np.float16)
    actions = np.zeros((7, 2), dtype=np.uint8)
    np.savez_compressed(tmp_path / "bad.npz", latents=latents, actions=actions)

    with pytest.raises(ValueError, match="latent steps"):
        LatentSequenceDataset(tmp_path, clip_latents=4, include_actions=True)


def test_latent_dataset_rejects_scalar_action_arrays(tmp_path: Path) -> None:
    latents = np.zeros((4, 8, 2, 3), dtype=np.float16)
    actions = np.zeros((8,), dtype=np.uint8)
    np.savez_compressed(tmp_path / "bad.npz", latents=latents, actions=actions)

    with pytest.raises(ValueError, match="actions shape"):
        LatentSequenceDataset(tmp_path, clip_latents=4, include_actions=True)
