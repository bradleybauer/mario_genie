from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.data.normalized_dataset import NormalizedSequenceDataset


def test_audio_mode_requires_audio_arrays(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "sample.npz",
        frames=np.zeros((8, 224, 224), dtype=np.uint8),
        actions=np.zeros((8,), dtype=np.uint8),
        ram=np.zeros((8, 4), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="missing audio arrays"):
        NormalizedSequenceDataset(tmp_path, clip_frames=4, include_audio=True)


def test_audio_mode_accepts_audio_arrays(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "sample.npz",
        frames=np.zeros((8, 224, 224), dtype=np.uint8),
        actions=np.zeros((8,), dtype=np.uint8),
        ram=np.zeros((8, 4), dtype=np.uint8),
        audio=np.zeros((8, 400), dtype=np.int16),
        audio_lengths=np.full((8,), 400, dtype=np.uint16),
        audio_sample_rate=np.int32(24000),
    )

    dataset = NormalizedSequenceDataset(tmp_path, clip_frames=4, include_audio=True)
    sample = dataset[0]

    assert sample["frames"].shape == (4, 224, 224)
    assert sample["audio"].shape == (4, 400)
    assert sample["audio_lengths"].shape == (4,)


def test_ram_mode_can_index_without_frames_array(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "sample.npz",
        ram=np.zeros((8, 4), dtype=np.uint8),
    )

    dataset = NormalizedSequenceDataset(
        tmp_path,
        clip_frames=4,
        include_frames=False,
        include_ram=True,
    )
    sample = dataset[0]

    assert len(dataset) == 5
    assert sample["ram"].shape == (4, 4)


def test_audio_mode_can_index_without_frames_array(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "sample.npz",
        audio=np.zeros((8, 400), dtype=np.int16),
        audio_lengths=np.full((8,), 400, dtype=np.uint16),
        audio_sample_rate=np.int32(24000),
    )

    dataset = NormalizedSequenceDataset(
        tmp_path,
        clip_frames=4,
        include_frames=False,
        include_audio=True,
    )
    sample = dataset[0]

    assert len(dataset) == 5
    assert "frames" not in sample
    assert sample["audio"].shape == (4, 400)
    assert sample["audio_lengths"].shape == (4,)


def test_actions_mode_can_index_without_frames_array(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "sample.npz",
        actions=np.zeros((8,), dtype=np.uint8),
    )

    dataset = NormalizedSequenceDataset(
        tmp_path,
        clip_frames=4,
        include_frames=False,
        include_actions=True,
    )
    sample = dataset[0]

    assert len(dataset) == 5
    assert sample["actions"].shape == (4,)