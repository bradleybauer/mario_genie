from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.video_latent_dit import VideoLatentDiT


def test_video_latent_dit_output_shape_matches_input() -> None:
    model = VideoLatentDiT(
        latent_channels=8,
        num_actions=16,
        d_model=32,
        num_layers=2,
        num_heads=4,
        max_frames=16,
    )
    noisy_latents = torch.randn(2, 8, 6, 4, 4)
    actions = torch.randint(0, 16, (2, 6), dtype=torch.long)
    timesteps = torch.rand(2)

    velocity = model(noisy_latents, actions, timesteps)

    assert velocity.shape == noisy_latents.shape


def test_video_latent_dit_single_timestep_broadcasts() -> None:
    model = VideoLatentDiT(
        latent_channels=4,
        num_actions=8,
        d_model=24,
        num_layers=2,
        num_heads=4,
        max_frames=16,
    )
    noisy_latents = torch.randn(3, 4, 5, 3, 3)
    actions = torch.randint(0, 8, (3, 5), dtype=torch.long)

    velocity = model(noisy_latents, actions, torch.tensor([0.5]))

    assert velocity.shape == noisy_latents.shape


def test_video_latent_dit_rejects_steps_exceeding_max_frames() -> None:
    model = VideoLatentDiT(
        latent_channels=6,
        num_actions=10,
        d_model=32,
        num_layers=1,
        num_heads=4,
        max_frames=4,
    )
    noisy_latents = torch.randn(1, 6, 5, 2, 2)
    actions = torch.randint(0, 10, (1, 5), dtype=torch.long)

    with pytest.raises(ValueError, match="exceeds max_frames"):
        model(noisy_latents, actions, torch.tensor([0.3]))
