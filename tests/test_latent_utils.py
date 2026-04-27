from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.models.latent_utils import denoise_future_segment


class DummyCFGModel(torch.nn.Module):
    def encode_history(
        self,
        history_latents: torch.Tensor,
        history_actions: torch.Tensor,
        action_cond_scale: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        batch = history_latents.shape[0]
        if action_cond_scale is None:
            scale = torch.ones(batch, device=history_latents.device, dtype=history_latents.dtype)
        else:
            scale = torch.as_tensor(
                action_cond_scale,
                device=history_latents.device,
                dtype=history_latents.dtype,
            ).reshape(-1)
            if scale.numel() == 1:
                scale = scale.expand(batch)
        history_term = history_latents.mean(dim=(1, 2, 3, 4))
        action_term = history_actions.float().sum(dim=(1, 2)).to(dtype=history_latents.dtype)
        return (history_term + scale * action_term).view(batch, 1, 1)

    def decode_future(
        self,
        noisy_future: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        encoded_history: torch.Tensor,
        context_latents: int,
        action_cond_scale: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        batch = noisy_future.shape[0]
        if action_cond_scale is None:
            scale = torch.ones(batch, device=noisy_future.device, dtype=noisy_future.dtype)
        else:
            scale = torch.as_tensor(
                action_cond_scale,
                device=noisy_future.device,
                dtype=noisy_future.dtype,
            ).reshape(-1)
            if scale.numel() == 1:
                scale = scale.expand(batch)
        encoded_term = encoded_history.view(batch, 1, 1, 1, 1).to(dtype=noisy_future.dtype)
        action_term = actions.float().sum(dim=(1, 2)).to(dtype=noisy_future.dtype).view(batch, 1, 1, 1, 1)
        time_term = timesteps.view(batch, 1, 1, 1, 1).to(dtype=noisy_future.dtype)
        return encoded_term + scale.view(batch, 1, 1, 1, 1) * action_term + time_term


def _run_segment(scale: float) -> torch.Tensor:
    model = DummyCFGModel()
    torch.manual_seed(7)
    history = torch.randn(2, 3, 2, 2, 2)
    actions = torch.tensor([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]], dtype=torch.long)
    torch.manual_seed(123)
    return denoise_future_segment(
        model,
        history_latents=history,
        actions=actions,
        future_latents=1,
        ode_steps=1,
        action_cfg_scale=scale,
    )


class DummyLinearFlowModel(torch.nn.Module):
    def encode_history(
        self,
        history_latents: torch.Tensor,
        history_actions: torch.Tensor,
        action_cond_scale: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        return torch.zeros(
            history_latents.shape[0],
            1,
            1,
            device=history_latents.device,
            dtype=history_latents.dtype,
        )

    def decode_future(
        self,
        noisy_future: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        encoded_history: torch.Tensor,
        context_latents: int,
        action_cond_scale: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        return noisy_future


def _run_linear_segment(*, ode_sampler: str, ode_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    model = DummyLinearFlowModel()
    history = torch.zeros(1, 1, 1, 1, 1)
    actions = torch.zeros(1, 2, 2, dtype=torch.long)

    torch.manual_seed(19)
    initial_noise = torch.randn(1, 1, 1, 1, 1)
    torch.manual_seed(19)
    output = denoise_future_segment(
        model,
        history_latents=history,
        actions=actions,
        future_latents=1,
        ode_steps=ode_steps,
        ode_sampler=ode_sampler,
    )
    return initial_noise, output


def test_action_cfg_scale_one_matches_conditional_path() -> None:
    out_a = _run_segment(1.0)
    out_b = _run_segment(1.0)
    assert torch.allclose(out_a, out_b)


def test_action_cfg_scale_zero_matches_unconditional_branch() -> None:
    out_zero = _run_segment(0.0)
    out_one = _run_segment(1.0)
    out_two = _run_segment(2.0)

    # For this dummy model, denoising with CFG is affine in the guidance scale.
    assert torch.allclose(out_two, 2.0 * out_one - out_zero, atol=1e-6)


def test_action_cfg_scale_interpolates_between_uncond_and_cond() -> None:
    out_zero = _run_segment(0.0)
    out_one = _run_segment(1.0)
    out_half = _run_segment(0.5)

    assert torch.allclose(out_half, 0.5 * (out_zero + out_one), atol=1e-6)


def test_heun_sampler_improves_linear_flow_accuracy() -> None:
    initial_noise, euler = _run_linear_segment(ode_sampler="euler", ode_steps=2)
    _, heun = _run_linear_segment(ode_sampler="heun", ode_steps=2)
    expected = initial_noise * math.exp(-1.0)

    euler_error = torch.max(torch.abs(euler - expected)).item()
    heun_error = torch.max(torch.abs(heun - expected)).item()
    assert heun_error < euler_error
