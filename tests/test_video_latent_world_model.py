from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.video_latent_world_model import VideoLatentWorldModel


def test_video_latent_world_model_output_shape_matches_input() -> None:
    model = VideoLatentWorldModel(
        latent_channels=8,
        num_actions=16,
        d_model=32,
        num_layers=2,
        num_heads=4,
        max_frames=16,
    )
    latents = torch.randn(2, 8, 6, 4, 4)
    actions = torch.randint(0, 16, (2, 6), dtype=torch.long)

    predicted = model(latents, actions)

    assert predicted.shape == latents.shape


def test_video_latent_world_model_is_causal_over_time() -> None:
    torch.manual_seed(7)
    model = VideoLatentWorldModel(
        latent_channels=4,
        num_actions=8,
        d_model=24,
        num_layers=2,
        num_heads=4,
        max_frames=16,
    )
    model.eval()

    latents = torch.randn(1, 4, 8, 3, 3)
    actions = torch.randint(0, 8, (1, 8), dtype=torch.long)

    baseline = model(latents, actions)

    perturbed_latents = latents.clone()
    perturbed_actions = actions.clone()
    perturbed_latents[:, :, 5:] = torch.randn_like(perturbed_latents[:, :, 5:])
    perturbed_actions[:, 5:] = torch.randint(0, 8, (1, 3), dtype=torch.long)

    perturbed = model(perturbed_latents, perturbed_actions)

    assert torch.allclose(baseline[:, :, :5], perturbed[:, :, :5], atol=1e-6, rtol=1e-5)


def test_shifted_training_contract_has_matching_shapes() -> None:
    model = VideoLatentWorldModel(
        latent_channels=6,
        num_actions=10,
        d_model=32,
        num_layers=1,
        num_heads=4,
        max_frames=16,
    )

    latents = torch.randn(3, 6, 7, 2, 2)
    actions = torch.randint(0, 10, (3, 7), dtype=torch.long)

    predicted_next = model(latents[:, :, :-1], actions[:, :-1])
    target_next = latents[:, :, 1:]

    assert predicted_next.shape == target_next.shape
