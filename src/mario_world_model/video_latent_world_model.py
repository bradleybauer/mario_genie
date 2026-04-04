from __future__ import annotations

import torch
from einops import rearrange, repeat
from torch import Tensor, nn


class LatentTemporalBlock(nn.Module):
    """Temporal transformer block used on per-cell latent token sequences."""

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
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, *, causal_mask: Tensor) -> Tensor:
        attn_in = self.norm1(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, attn_mask=causal_mask, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class VideoLatentWorldModel(nn.Module):
    """Action-conditioned next-latent predictor for video VAE latents.

    Inputs:
      latents: (B, C, T, H, W)
      actions: (B, T)

    Outputs:
      predicted next latents aligned to each input step: (B, C, T, H, W)

    Training uses teacher forcing with shifted targets:
      pred = model(latents[:, :, :-1], actions[:, :-1])
      target = latents[:, :, 1:]
    """

    def __init__(
        self,
        *,
        latent_channels: int,
        num_actions: int,
        d_model: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_frames: int = 64,
    ) -> None:
        super().__init__()
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")

        self.latent_channels = latent_channels
        self.max_frames = max_frames

        self.input_proj = nn.Linear(latent_channels, d_model)
        self.output_proj = nn.Linear(d_model, latent_channels)
        self.action_embed = nn.Embedding(num_actions, d_model)
        self.spatial_coord_proj = nn.Linear(2, d_model)
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_frames, d_model))

        self.blocks = nn.ModuleList(
            [
                LatentTemporalBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        self.register_buffer("_causal_mask_cache", torch.empty(0, 0, dtype=torch.bool), persistent=False)

    def _causal_mask(self, length: int, device: torch.device) -> Tensor:
        cache = self._causal_mask_cache
        if cache.device != device or cache.shape[0] < length:
            cache = torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)
            self._causal_mask_cache = cache
        return cache[:length, :length]

    @staticmethod
    def _spatial_coords(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).reshape(height * width, 2)

    def forward(self, latents: Tensor, actions: Tensor) -> Tensor:
        if latents.ndim != 5:
            raise ValueError(f"Expected latents with shape (B, C, T, H, W), got {tuple(latents.shape)}")
        if actions.ndim != 2:
            raise ValueError(f"Expected actions with shape (B, T), got {tuple(actions.shape)}")

        batch, channels, steps, height, width = latents.shape
        if channels != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} latent channels, got {channels}"
            )
        if actions.shape[0] != batch or actions.shape[1] != steps:
            raise ValueError(
                f"Actions must match latent batch/time dims: latents=({batch}, {steps}), "
                f"actions={tuple(actions.shape)}"
            )
        if steps > self.max_frames:
            raise ValueError(f"steps ({steps}) exceeds max_frames ({self.max_frames})")

        if actions.dtype != torch.long:
            actions = actions.long()

        tokens = rearrange(latents, "b c t h w -> (b h w) t c")
        tokens = self.input_proj(tokens)

        action_tokens = self.action_embed(actions)
        action_tokens = repeat(action_tokens, "b t d -> (b n) t d", n=height * width)

        temporal = self.temporal_pos[:, :steps].to(dtype=tokens.dtype)

        spatial_coords = self._spatial_coords(height, width, device=latents.device, dtype=tokens.dtype)
        spatial = self.spatial_coord_proj(spatial_coords)
        spatial = repeat(spatial, "n d -> (b n) 1 d", b=batch)

        tokens = tokens + action_tokens + temporal + spatial

        mask = self._causal_mask(steps, latents.device)
        for block in self.blocks:
            tokens = block(tokens, causal_mask=mask)

        tokens = self.final_norm(tokens)
        predicted = self.output_proj(tokens)
        return rearrange(predicted, "(b h w) t c -> b c t h w", b=batch, h=height, w=width)
