from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


def _load_view_palette_augmentation_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "view_palette_augmentation.py"
    spec = importlib.util.spec_from_file_location("view_palette_augmentation", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_sample_panel_renders_changed_mask_as_rgb() -> None:
    module = _load_view_palette_augmentation_module()
    palette_rgb = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
        ],
        dtype=np.uint8,
    )
    clean_frames = torch.tensor(
        [
            [
                [[0, 1], [1, 0]],
                [[1, 0], [0, 1]],
            ]
        ],
        dtype=torch.long,
    )
    augmented_frames = clean_frames.clone()
    augmented_frames[0, 1, 0, 1] = 2

    panel = module.build_sample_panel(
        clean_frames,
        augmented_frames,
        palette_rgb=palette_rgb,
        frame_gap=1,
        row_gap=1,
    )

    assert panel.ndim == 3
    assert panel.shape[2] == 3
    assert np.any(np.all(panel == 255, axis=2))