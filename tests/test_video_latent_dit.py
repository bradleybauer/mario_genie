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

from src.models.video_latent_dit_diffusers import VideoLatentDiTDiffusers


def _make_model(**kwargs) -> VideoLatentDiTDiffusers:
    defaults = dict(
        latent_channels=8,
        num_actions=16,
        d_model=32,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        max_latents=32,
    )
    defaults.update(kwargs)
    return VideoLatentDiTDiffusers(**defaults)


def test_forward_output_shape() -> None:
    model = _make_model()
    B, C, ctx, fut, H, W = 2, 8, 4, 2, 4, 4
    noisy_latents = torch.randn(B, C, ctx + fut, H, W)
    actions = torch.randint(0, 16, (B, ctx + fut), dtype=torch.long)
    timesteps = torch.rand(B)

    velocity = model(noisy_latents, actions, timesteps, context_latents=ctx)

    assert velocity.shape == (B, C, fut, H, W)


def test_forward_single_timestep_broadcasts() -> None:
    model = _make_model(latent_channels=4, num_actions=8, d_model=24)
    B, ctx, fut = 3, 3, 2
    noisy_latents = torch.randn(B, 4, ctx + fut, 3, 3)
    actions = torch.randint(0, 8, (B, ctx + fut), dtype=torch.long)

    velocity = model(noisy_latents, actions, torch.tensor([0.5]), context_latents=ctx)

    assert velocity.shape == (B, 4, fut, 3, 3)


def test_encode_decode_matches_forward() -> None:
    """encode_history + decode_future must produce the same result as forward()."""
    model = _make_model()
    model.eval()
    B, C, ctx, fut, H, W = 2, 8, 4, 2, 4, 4
    noisy_latents = torch.randn(B, C, ctx + fut, H, W)
    actions = torch.randint(0, 16, (B, ctx + fut), dtype=torch.long)
    timesteps = torch.rand(B)

    with torch.no_grad():
        vel_forward = model(noisy_latents, actions, timesteps, context_latents=ctx)

        encoded = model.encode_history(noisy_latents[:, :, :ctx], actions[:, :ctx])
        vel_split = model.decode_future(
            noisy_latents[:, :, ctx:], actions, timesteps, encoded, ctx
        )

    assert torch.allclose(vel_forward, vel_split, atol=1e-5)


def test_encode_history_cached_across_steps() -> None:
    """Calling decode_future multiple times with the same encoded_history is consistent."""
    model = _make_model()
    model.eval()
    B, C, ctx, fut, H, W = 1, 8, 4, 2, 3, 3
    history = torch.randn(B, C, ctx, H, W)
    history_actions = torch.randint(0, 16, (B, ctx), dtype=torch.long)
    future = torch.randn(B, C, fut, H, W)
    actions = torch.randint(0, 16, (B, ctx + fut), dtype=torch.long)

    with torch.no_grad():
        encoded = model.encode_history(history, history_actions)
        t = torch.tensor([0.5])
        v1 = model.decode_future(future, actions, t, encoded, ctx)
        v2 = model.decode_future(future, actions, t, encoded, ctx)

    assert torch.allclose(v1, v2)


def test_dropped_action_conditioning_is_invariant_to_action_ids() -> None:
    model = _make_model()
    model.eval()
    B, C, ctx, fut, H, W = 2, 8, 4, 2, 4, 4
    noisy_latents = torch.randn(B, C, ctx + fut, H, W)
    actions_a = torch.randint(0, 16, (B, ctx + fut), dtype=torch.long)
    actions_b = torch.randint(0, 16, (B, ctx + fut), dtype=torch.long)
    timesteps = torch.rand(B)
    dropped = torch.zeros(B)

    with torch.no_grad():
        vel_a = model(
            noisy_latents,
            actions_a,
            timesteps,
            context_latents=ctx,
            action_cond_scale=dropped,
        )
        vel_b = model(
            noisy_latents,
            actions_b,
            timesteps,
            context_latents=ctx,
            action_cond_scale=dropped,
        )

    assert torch.allclose(vel_a, vel_b, atol=1e-5)


def test_encode_decode_matches_forward_when_actions_dropped() -> None:
    model = _make_model()
    model.eval()
    B, C, ctx, fut, H, W = 2, 8, 4, 2, 4, 4
    noisy_latents = torch.randn(B, C, ctx + fut, H, W)
    actions = torch.randint(0, 16, (B, ctx + fut), dtype=torch.long)
    timesteps = torch.rand(B)
    dropped = torch.zeros(B)

    with torch.no_grad():
        vel_forward = model(
            noisy_latents,
            actions,
            timesteps,
            context_latents=ctx,
            action_cond_scale=dropped,
        )

        encoded = model.encode_history(
            noisy_latents[:, :, :ctx],
            actions[:, :ctx],
            action_cond_scale=dropped,
        )
        vel_split = model.decode_future(
            noisy_latents[:, :, ctx:],
            actions,
            timesteps,
            encoded,
            ctx,
            action_cond_scale=dropped,
        )

    assert torch.allclose(vel_forward, vel_split, atol=1e-5)


def test_rejects_total_frames_exceeding_max_latents() -> None:
    model = _make_model(max_latents=4)
    noisy_latents = torch.randn(1, 8, 5, 2, 2)
    actions = torch.randint(0, 16, (1, 5), dtype=torch.long)

    with pytest.raises(ValueError, match="exceeds max_latents"):
        model(noisy_latents, actions, torch.tensor([0.3]), context_latents=3)


def test_rejects_invalid_context_latents() -> None:
    model = _make_model()
    noisy_latents = torch.randn(1, 8, 6, 4, 4)
    actions = torch.randint(0, 16, (1, 6), dtype=torch.long)

    with pytest.raises(ValueError, match="context_latents"):
        model(noisy_latents, actions, torch.tensor([0.5]), context_latents=6)
