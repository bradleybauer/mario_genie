"""Unified bidirectional video-latent DiT for flow-matching future prediction.

Matrix-Game 3.0 style: history and noisy future tokens are concatenated
into a single sequence and processed jointly via self-attention.  This
removes the encoder-decoder information bottleneck at the cost of
reprocessing history at every ODE step during inference.
"""

from __future__ import annotations

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from torch import Tensor, nn


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


class UnifiedDiTBlock(nn.Module):
    """Bidirectional block: all tokens (history + noisy future) share self-attention."""

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

        head_dim = d_model // num_heads
        hidden_dim = int(d_model * mlp_ratio)

        self.norm1 = AdaLayerNorm(d_model)
        self.self_attn = Attention(
            query_dim=d_model,
            heads=num_heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=True,
            out_bias=True,
            qk_norm="rms_norm",
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.action_cross_attn = Attention(
            query_dim=d_model,
            cross_attention_dim=d_model,
            heads=num_heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=True,
            out_bias=True,
            qk_norm="rms_norm",
        )

        self.norm3 = AdaLayerNorm(d_model)
        self.mlp = FeedForward(d_model, inner_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        tokens: Tensor,
        *,
        action_tokens: Tensor,
        cond: Tensor,
        action_cond_scale: Tensor | None = None,
    ) -> Tensor:
        h = self.norm1(tokens, cond)
        tokens = tokens + self.self_attn(h)

        q = self.norm2(tokens)
        action_update = self.action_cross_attn(
            q,
            encoder_hidden_states=action_tokens,
        )
        if action_cond_scale is not None:
            action_update = action_update * action_cond_scale
        tokens = tokens + action_update

        tokens = tokens + self.mlp(self.norm3(tokens, cond))
        return tokens


class VideoLatentDiTUnified(ModelMixin, ConfigMixin):
    """Unified bidirectional video-latent DiT for flow-matching future prediction.

    History and noisy future tokens are concatenated and processed jointly
    through a single stack of self-attention blocks.  The loss target covers
    only the future portion of the output.
    """

    @register_to_config
    def __init__(
        self,
        *,
        latent_channels: int,
        num_actions: int,
        action_frame_count: int,
        action_dim: int = 32,
        action_values: list[int] | None = None,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_latents: int = 64,
    ) -> None:
        super().__init__()
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive")
        if action_frame_count <= 0:
            raise ValueError("action_frame_count must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if max_latents <= 0:
            raise ValueError("max_latents must be positive")

        self.latent_channels = latent_channels
        self.num_actions = num_actions
        self.action_frame_count = action_frame_count
        self.max_latents = max_latents
        self.d_model = d_model
        self.num_heads = num_heads

        self.input_proj = nn.Linear(latent_channels, d_model)
        self.output_proj = nn.Linear(d_model, latent_channels)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        if action_values is None:
            action_values = list(range(num_actions))
        if len(action_values) != num_actions:
            raise ValueError(
                f"action_values must contain exactly num_actions={num_actions} entries, "
                f"got {len(action_values)}"
            )

        action_bit_lut = torch.empty((num_actions, 8), dtype=torch.float32)
        for idx, value in enumerate(action_values):
            packed = int(value) & 0xFF
            for bit in range(8):
                action_bit_lut[idx, bit] = float((packed >> bit) & 1)
        self.register_buffer("action_bit_lut", action_bit_lut, persistent=False)

        self.action_mlp = nn.Sequential(
            nn.Linear(8, action_dim),
            nn.SiLU(),
            nn.Linear(action_dim, action_dim),
            nn.SiLU(),
        )
        self.action_frame_pos = nn.Parameter(torch.zeros(1, 1, action_frame_count, action_dim))
        self.action_to_model = nn.Linear(action_dim * action_frame_count, d_model)

        self.spatial_coord_proj = nn.Linear(2, d_model)
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_latents, d_model))

        self.time_proj = Timesteps(
            num_channels=d_model,
            flip_sin_to_cos=False,
            downscale_freq_shift=0.0,
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=d_model,
            time_embed_dim=d_model,
            act_fn="silu",
        )

        self.blocks = nn.ModuleList(
            [
                UnifiedDiTBlock(
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
    def _spatial_coords(
        height: int, width: int, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).reshape(height * width, 2)

    def _positional_encoding(
        self,
        height: int,
        width: int,
        *,
        start_latent: int,
        num_latents: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        temporal = self.temporal_pos[:, start_latent : start_latent + num_latents].to(dtype=dtype)
        spatial_coords = self._spatial_coords(height, width, device=device, dtype=dtype)
        spatial = self.spatial_coord_proj(spatial_coords)
        positional = spatial.unsqueeze(0).unsqueeze(2) + temporal.unsqueeze(1)
        return rearrange(positional, "b n t d -> b (n t) d")

    def _encode_actions(self, actions: Tensor, *, dtype: torch.dtype) -> Tensor:
        if actions.dtype != torch.long:
            actions = actions.long()
        bits = self.action_bit_lut[actions].to(dtype=dtype)
        action_emb = self.action_mlp(bits)
        action_emb = action_emb + self.action_frame_pos.to(dtype=dtype)
        action_emb = rearrange(action_emb, "b t f d -> b t (f d)")
        return self.action_to_model(action_emb)

    @staticmethod
    def _prepare_action_cond_scale(
        action_cond_scale: Tensor | float | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if action_cond_scale is None:
            return None
        scale = torch.as_tensor(action_cond_scale, device=device, dtype=dtype)
        if scale.ndim == 0:
            scale = scale.expand(batch_size)
        elif scale.ndim == 2 and scale.shape[1] == 1:
            scale = scale[:, 0]
        elif scale.ndim != 1:
            raise ValueError(
                f"Expected action_cond_scale scalar, (B,), or (B, 1); got {tuple(scale.shape)}"
            )
        if scale.shape[0] != batch_size:
            raise ValueError(f"Expected action_cond_scale batch size {batch_size}, got {scale.shape[0]}")
        return scale.view(batch_size, 1, 1)

    def _validate_actions(self, actions: Tensor) -> None:
        if torch.any(actions < 0) or torch.any(actions >= self.num_actions):
            raise ValueError(
                f"Action IDs must be in [0, {self.num_actions - 1}] "
                f"but got min={int(actions.min())}, max={int(actions.max())}"
            )

    def forward(
        self,
        latents: Tensor,
        actions: Tensor,
        timesteps: Tensor,
        context_latents: int,
        action_cond_scale: Tensor | float | None = None,
    ) -> Tensor:
        """Forward pass through the unified bidirectional DiT.

        Parameters
        ----------
        latents:
            ``(B, C, T, H, W)`` — concatenation of clean history and noisy
            future latents along the temporal dimension.
        actions:
            ``(B, T, action_frame_count)`` — grouped frame action indices for all ``T`` latent timesteps.
        timesteps:
            ``(B,)`` — diffusion timestep for the future portion.
        context_latents:
            Number of leading temporal positions that are clean history.

        Returns
        -------
        Tensor of shape ``(B, C, 1, H, W)`` — velocity prediction for the
        single future latent.
        """
        if latents.ndim != 5:
            raise ValueError(f"Expected latents (B, C, T, H, W), got {tuple(latents.shape)}")
        if actions.ndim != 3:
            raise ValueError(f"Expected actions (B, T, action_frame_count), got {tuple(actions.shape)}")

        batch, channels, total_latents, height, width = latents.shape
        expected_total_latents = context_latents + 1

        if channels != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, got {channels}")
        if context_latents < 1:
            raise ValueError(f"context_latents must be >= 1, got {context_latents}")
        if total_latents != expected_total_latents:
            raise ValueError(
                "Unified DiT expects exactly one future latent per forward pass: "
                f"got total_latents={total_latents}, context_latents={context_latents}"
            )
        if actions.shape != (batch, total_latents, self.action_frame_count):
            raise ValueError(
                f"actions must be (B={batch}, T={total_latents}, "
                f"action_frame_count={self.action_frame_count}), got {tuple(actions.shape)}"
            )
        if total_latents > self.max_latents:
            raise ValueError(
                f"total_latents ({total_latents}) exceeds max_latents ({self.max_latents})"
            )

        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.ndim != 1:
            raise ValueError(f"Expected timesteps (B,), got {tuple(timesteps.shape)}")
        if timesteps.shape[0] == 1 and batch > 1:
            timesteps = timesteps.expand(batch)
        if timesteps.shape[0] != batch:
            raise ValueError(f"Expected {batch} timesteps, got {timesteps.shape[0]}")

        self._validate_actions(actions)
        cond_scale = self._prepare_action_cond_scale(
            action_cond_scale,
            batch_size=batch,
            device=latents.device,
            dtype=latents.dtype,
        )

        # Project all latents (history + noisy future) into token space.
        tokens = rearrange(latents, "b c t h w -> b (h w t) c")
        tokens = self.input_proj(tokens)
        tokens = tokens + self._positional_encoding(
            height,
            width,
            start_latent=0,
            num_latents=total_latents,
            device=latents.device,
            dtype=tokens.dtype,
        )

        # Action tokens for all timesteps.
        action_tokens = self._encode_actions(actions, dtype=tokens.dtype)

        # Timestep conditioning.
        t_proj = self.time_proj(timesteps.float()).to(
            device=latents.device, dtype=tokens.dtype
        )
        cond = self.time_embedding(t_proj).to(dtype=tokens.dtype)

        # Process all tokens jointly through the unified stack.
        for block in self.blocks:
            tokens = block(
                tokens,
                action_tokens=action_tokens,
                cond=cond,
                action_cond_scale=cond_scale,
            )

        # Final modulated norm + output projection.
        tokens = self.final_norm(tokens)
        shift_scale = self.final_mod(cond).unsqueeze(1)
        shift, scale = shift_scale.chunk(2, dim=-1)
        tokens = tokens * (1.0 + scale) + shift
        pred = self.output_proj(tokens)

        # Reshape back and return only the future portion.
        pred = rearrange(pred, "b (h w t) c -> b c t h w", h=height, w=width, t=total_latents)
        return pred[:, :, context_latents:context_latents + 1]
