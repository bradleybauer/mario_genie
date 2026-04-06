from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.normalized_dataset import NormalizedSequenceDataset


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