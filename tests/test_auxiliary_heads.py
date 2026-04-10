"""Tests for auxiliary heads (next-frame predictor, temporal smoothness, RAM alignment)."""
from __future__ import annotations

import torch
import pytest

from src.models.auxiliary_heads import (
    NextFramePredictor,
    RAMAlignmentHead,
    align_actions_to_latent_time,
    align_ram_to_latent_time,
    temporal_smoothness_loss,
)


# ---------------------------------------------------------------------------
# Shape constants matching a typical video VAE setup
# ---------------------------------------------------------------------------
B = 2
C = 16            # latent_channels
T_LAT = 10        # latent temporal frames
H, W = 7, 7       # spatial latent size
T_INPUT = 19       # input frames (clip_frames=16 + context_frames=3)
NUM_ACTIONS = 42
RAM_DIM = 32


@pytest.fixture
def posterior_mean() -> torch.Tensor:
    return torch.randn(B, C, T_LAT, H, W, requires_grad=True)


@pytest.fixture
def actions() -> torch.Tensor:
    return torch.randint(0, NUM_ACTIONS, (B, T_INPUT))


@pytest.fixture
def ram_mean() -> torch.Tensor:
    return torch.randn(B, T_INPUT, RAM_DIM)


# ---------------------------------------------------------------------------
# Temporal alignment helpers
# ---------------------------------------------------------------------------

class TestAlignActionsToLatentTime:
    def test_output_shape(self, actions: torch.Tensor) -> None:
        aligned = align_actions_to_latent_time(actions, T_LAT)
        assert aligned.shape == (B, T_LAT)

    def test_selects_even_indices(self, actions: torch.Tensor) -> None:
        aligned = align_actions_to_latent_time(actions, T_LAT)
        for k in range(T_LAT):
            assert (aligned[:, k] == actions[:, 2 * k]).all()


class TestAlignRamToLatentTime:
    def test_output_shape(self, ram_mean: torch.Tensor) -> None:
        aligned = align_ram_to_latent_time(ram_mean, T_LAT)
        assert aligned.shape == (B, T_LAT, RAM_DIM)

    def test_pair_averaging(self) -> None:
        # Deterministic test: ram_mean has frame values 0..7
        T = 8
        ram = torch.arange(T, dtype=torch.float32).view(1, T, 1).expand(1, T, 2)
        aligned = align_ram_to_latent_time(ram, t_lat=4)
        # Pair (0,1) -> 0.5, (2,3) -> 2.5, (4,5) -> 4.5, (6,7) -> 6.5
        expected = torch.tensor([0.5, 2.5, 4.5, 6.5]).view(1, 4, 1).expand(1, 4, 2)
        assert torch.allclose(aligned, expected)

    def test_odd_length_padding(self) -> None:
        T = 5
        ram = torch.ones(1, T, RAM_DIM)
        # Should handle odd T by padding last frame
        aligned = align_ram_to_latent_time(ram, t_lat=3)
        assert aligned.shape == (1, 3, RAM_DIM)


# ---------------------------------------------------------------------------
# NextFramePredictor
# ---------------------------------------------------------------------------

class TestNextFramePredictor:
    def test_output_shapes(self, posterior_mean: torch.Tensor, actions: torch.Tensor) -> None:
        predictor = NextFramePredictor(latent_dim=C, num_actions=NUM_ACTIONS)
        preds, targets = predictor(posterior_mean, actions)
        assert preds.shape == (B, T_LAT - 1, C)
        assert targets.shape == (B, T_LAT - 1, C)

    def test_gradient_flows_to_encoder(self, posterior_mean: torch.Tensor, actions: torch.Tensor) -> None:
        predictor = NextFramePredictor(latent_dim=C, num_actions=NUM_ACTIONS)
        preds, targets = predictor(posterior_mean, actions)
        loss = torch.nn.functional.mse_loss(preds, targets)
        loss.backward()
        assert posterior_mean.grad is not None
        assert posterior_mean.grad.abs().sum() > 0

    def test_loss_is_scalar(self, posterior_mean: torch.Tensor, actions: torch.Tensor) -> None:
        predictor = NextFramePredictor(latent_dim=C, num_actions=NUM_ACTIONS)
        preds, targets = predictor(posterior_mean, actions)
        loss = torch.nn.functional.mse_loss(preds, targets)
        assert loss.ndim == 0
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# Temporal Smoothness
# ---------------------------------------------------------------------------

class TestTemporalSmoothness:
    def test_returns_scalar(self, posterior_mean: torch.Tensor) -> None:
        loss = temporal_smoothness_loss(posterior_mean)
        assert loss.ndim == 0

    def test_range(self, posterior_mean: torch.Tensor) -> None:
        loss = temporal_smoothness_loss(posterior_mean)
        # 1 - cosine_sim: range [0, 2] but typically [0, 1] for non-adversarial inputs
        assert 0.0 <= loss.item() <= 2.0

    def test_identical_frames_near_zero(self) -> None:
        # If all temporal frames are identical, temporal smoothness should be ~0
        z = torch.randn(1, C, 1, H, W).expand(1, C, T_LAT, H, W).clone()
        z.requires_grad_(True)
        loss = temporal_smoothness_loss(z)
        assert loss.item() < 1e-5

    def test_gradient_flows(self, posterior_mean: torch.Tensor) -> None:
        loss = temporal_smoothness_loss(posterior_mean)
        loss.backward()
        assert posterior_mean.grad is not None


# ---------------------------------------------------------------------------
# RAM Alignment
# ---------------------------------------------------------------------------

class TestRAMAlignmentHead:
    def test_output_shape(self, posterior_mean: torch.Tensor) -> None:
        head = RAMAlignmentHead(video_latent_dim=C, ram_latent_dim=RAM_DIM)
        pooled = posterior_mean.mean(dim=(-2, -1)).permute(0, 2, 1)  # (B, T_lat, C)
        projected = head(pooled)
        assert projected.shape == (B, T_LAT, RAM_DIM)

    def test_loss_scalar(self, posterior_mean: torch.Tensor, ram_mean: torch.Tensor) -> None:
        head = RAMAlignmentHead(video_latent_dim=C, ram_latent_dim=RAM_DIM)
        loss = head.loss(posterior_mean, ram_mean)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_gradient_flows_to_video_not_ram(
        self, posterior_mean: torch.Tensor, ram_mean: torch.Tensor
    ) -> None:
        ram_mean = ram_mean.clone().requires_grad_(True)
        head = RAMAlignmentHead(video_latent_dim=C, ram_latent_dim=RAM_DIM)
        loss = head.loss(posterior_mean, ram_mean)
        loss.backward()
        # Gradients should flow to video encoder (posterior_mean)
        assert posterior_mean.grad is not None
        assert posterior_mean.grad.abs().sum() > 0
        # RAM side should be detached (frozen)
        assert ram_mean.grad is None or ram_mean.grad.abs().sum() == 0

    def test_loss_decreases_with_training(self, posterior_mean: torch.Tensor, ram_mean: torch.Tensor) -> None:
        head = RAMAlignmentHead(video_latent_dim=C, ram_latent_dim=RAM_DIM)
        optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
        pm = posterior_mean.detach()  # Fix encoder output, just train head
        initial_loss = head.loss(pm, ram_mean).item()
        for _ in range(50):
            optimizer.zero_grad()
            loss = head.loss(pm, ram_mean)
            loss.backward()
            optimizer.step()
        final_loss = head.loss(pm, ram_mean).item()
        assert final_loss < initial_loss
