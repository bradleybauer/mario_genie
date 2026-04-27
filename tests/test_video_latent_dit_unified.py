from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("diffusers")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.models.video_latent_dit_unified import VideoLatentDiTUnified


def _make_model(**kwargs) -> VideoLatentDiTUnified:
    defaults = dict(
        latent_channels=8,
        num_actions=16,
        action_frame_count=2,
        d_model=32,
        num_layers=2,
        num_heads=4,
        max_latents=32,
    )
    defaults.update(kwargs)
    return VideoLatentDiTUnified(**defaults)


def test_unified_forward_output_shape() -> None:
    model = _make_model()
    batch, channels, context_latents, height, width = 2, 8, 4, 4, 4
    latents = torch.randn(batch, channels, context_latents + 1, height, width)
    actions = torch.randint(0, 16, (batch, context_latents + 1, 2), dtype=torch.long)
    timesteps = torch.rand(batch)

    velocity = model(latents, actions, timesteps, context_latents=context_latents)

    assert velocity.shape == (batch, channels, 1, height, width)


def test_unified_rejects_scalar_action_timesteps() -> None:
    model = _make_model()
    context_latents = 4
    latents = torch.randn(1, 8, context_latents + 1, 4, 4)
    actions = torch.randint(0, 16, (1, context_latents + 1), dtype=torch.long)

    with pytest.raises(ValueError, match="action_frame_count"):
        model(latents, actions, torch.tensor([0.5]), context_latents=context_latents)
