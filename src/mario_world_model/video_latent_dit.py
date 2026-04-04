from __future__ import annotations

import math

import torch
from einops import rearrange, repeat
from torch import Tensor, nn


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """Create sinusoidal timestep embeddings.

    timesteps: shape (B,) float in [0, 1] or any real-valued scale.
    returns: shape (B, dim)
    """
    if timesteps.ndim != 1:
        raise ValueError(f"Expected timesteps with shape (B,), got {tuple(timesteps.shape)}")

    half = dim // 2
    if half == 0:
        return timesteps[:, None]

    device = timesteps.device
    exponent = -math.log(max_period) * torch.arange(0, half, device=device, dtype=torch.float32) / half
    freqs = torch.exp(exponent)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat((torch.cos(args), torch.sin(args)), dim=1)
    if dim % 2 == 1:
        emb = torch.cat((emb, torch.zeros_like(emb[:, :1])), dim=1)
    return emb


class AdaLayerNorm(nn.Module):
    """LayerNorm with FiLM-style modulation from a conditioning vector."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mod = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        if cond.ndim != 2:
            raise ValueError(f"Expected cond with shape (B, D), got {tuple(cond.shape)}")
        shift_scale = self.mod(cond).unsqueeze(1)
        shift, scale = shift_scale.chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


class LatentDiTBlock(nn.Module):
    """Bidirectional DiT block with action cross-attention."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        hidden_dim = int(d_model * mlp_ratio)
        self.norm1 = AdaLayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm3 = AdaLayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, *, action_tokens: Tensor, cond: Tensor) -> Tensor:
        h = self.norm1(tokens, cond)
        self_attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        tokens = tokens + self_attn_out

        q = self.norm2(tokens)
        cross_out, _ = self.cross_attn(q, action_tokens, action_tokens, need_weights=False)
        tokens = tokens + cross_out

        tokens = tokens + self.mlp(self.norm3(tokens, cond))
        return tokens


class VideoLatentDiT(nn.Module):
    """Video-latent DiT for flow-matching future latent prediction.

    Inputs:
      noisy_latents: (B, C, T, H, W) with clean history + noisy future segment
      actions: (B, T) reduced action IDs aligned to latent time steps
      timesteps: (B,) scalar diffusion/flow timestep for the noised future segment

    Output:
      velocity prediction v_theta over full sequence, shape (B, C, T, H, W)
    """

    def __init__(
        self,
        *,
        latent_channels: int,
        num_actions: int,
        d_model: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_frames: int = 64,
    ) -> None:
        super().__init__()
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")

        self.latent_channels = latent_channels
        self.max_frames = max_frames
        self.d_model = d_model

        self.input_proj = nn.Linear(latent_channels, d_model)
        self.output_proj = nn.Linear(d_model, latent_channels)

        self.action_embed = nn.Embedding(num_actions, d_model)
        self.spatial_coord_proj = nn.Linear(2, d_model)
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_frames, d_model))

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.action_cond_proj = nn.Linear(d_model, d_model)

        self.blocks = nn.ModuleList(
            [
                LatentDiTBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.final_mod = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.final_mod.weight)
        nn.init.zeros_(self.final_mod.bias)

    @staticmethod
    def _spatial_coords(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).reshape(height * width, 2)

    def forward(self, noisy_latents: Tensor, actions: Tensor, timesteps: Tensor) -> Tensor:
        if noisy_latents.ndim != 5:
            raise ValueError(
                f"Expected noisy_latents with shape (B, C, T, H, W), got {tuple(noisy_latents.shape)}"
            )
        if actions.ndim != 2:
            raise ValueError(f"Expected actions with shape (B, T), got {tuple(actions.shape)}")

        batch, channels, steps, height, width = noisy_latents.shape
        if channels != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, got {channels}")
        if actions.shape[0] != batch or actions.shape[1] != steps:
            raise ValueError(
                f"Actions must match latent batch/time dims: latents=({batch}, {steps}), actions={tuple(actions.shape)}"
            )
        if steps > self.max_frames:
            raise ValueError(f"steps ({steps}) exceeds max_frames ({self.max_frames})")

        if actions.dtype != torch.long:
            actions = actions.long()

        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.ndim != 1:
            raise ValueError(f"Expected timesteps with shape (B,), got {tuple(timesteps.shape)}")
        if timesteps.shape[0] == 1 and batch > 1:
            timesteps = timesteps.expand(batch)
        if timesteps.shape[0] != batch:
            raise ValueError(f"Expected {batch} timesteps, got {timesteps.shape[0]}")

        tokens = rearrange(noisy_latents, "b c t h w -> (b h w) t c")
        tokens = self.input_proj(tokens)

        action_tokens = self.action_embed(actions)
        action_tokens_rep = repeat(action_tokens, "b t d -> (b n) t d", n=height * width)

        temporal = self.temporal_pos[:, :steps].to(dtype=tokens.dtype)
        spatial_coords = self._spatial_coords(height, width, device=noisy_latents.device, dtype=tokens.dtype)
        spatial = self.spatial_coord_proj(spatial_coords)
        spatial = repeat(spatial, "n d -> (b n) 1 d", b=batch)

        tokens = tokens + temporal + spatial

        t_embed = timestep_embedding(timesteps, self.d_model).to(dtype=tokens.dtype)
        t_embed = self.time_mlp(t_embed)
        action_cond = self.action_cond_proj(action_tokens.mean(dim=1))
        cond = t_embed + action_cond
        cond = repeat(cond, "b d -> (b n) d", n=height * width)

        for block in self.blocks:
            tokens = block(tokens, action_tokens=action_tokens_rep, cond=cond)

        tokens = self.final_norm(tokens)
        shift_scale = self.final_mod(cond).unsqueeze(1)
        shift, scale = shift_scale.chunk(2, dim=-1)
        tokens = tokens * (1.0 + scale) + shift

        pred = self.output_proj(tokens)
        return rearrange(pred, "(b h w) t c -> b c t h w", b=batch, h=height, w=width)
