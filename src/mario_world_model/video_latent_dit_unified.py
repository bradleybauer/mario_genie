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


def _init_zero_attention_out(attn: Attention) -> None:
    nn.init.zeros_(attn.to_out[0].weight)
    if attn.to_out[0].bias is not None:
        nn.init.zeros_(attn.to_out[0].bias)


def _init_zero_ff_out(ff: FeedForward) -> None:
    final_linear = ff.net[-1]
    if isinstance(final_linear, nn.Linear):
        nn.init.zeros_(final_linear.weight)
        if final_linear.bias is not None:
            nn.init.zeros_(final_linear.bias)


class UnifiedDiffusersBlock(nn.Module):
    """Single-stack DiT block over full clip tokens.

    The block performs bidirectional self-attention across all spatiotemporal
    tokens, then causal action cross-attention, followed by a timestep-
    conditioned MLP.
    """

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

        _init_zero_attention_out(self.self_attn)
        _init_zero_attention_out(self.action_cross_attn)
        _init_zero_ff_out(self.mlp)

    def forward(
        self,
        tokens: Tensor,
        *,
        action_tokens: Tensor,
        cond: Tensor,
        action_attn_bias: Tensor | None,
    ) -> Tensor:
        h = self.norm1(tokens, cond)
        tokens = tokens + self.self_attn(h)

        q = self.norm2(tokens)
        tokens = tokens + self.action_cross_attn(
            q,
            encoder_hidden_states=action_tokens,
            attention_mask=action_attn_bias,
        )

        tokens = tokens + self.mlp(self.norm3(tokens, cond))
        return tokens


class VideoLatentDiTUnified(ModelMixin, ConfigMixin):
    """Unsplit video-latent DiT for flow-matching future prediction.

    Unlike the encoder/decoder variant, this model processes clean history and
    noisy future tokens together in one bidirectional transformer stack.

    forward() interface:
        latents        : (B, C, T, H, W)  clean history + noisy future
        actions        : (B, T)
        timesteps      : (B,)
        context_frames : int
        returns        : (B, C, future_frames, H, W)
    """

    @register_to_config
    def __init__(
        self,
        *,
        latent_channels: int,
        num_actions: int,
        action_dim: int = 32,
        action_values: list[int] | None = None,
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
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")

        self.latent_channels = latent_channels
        self.num_actions = num_actions
        self.max_frames = max_frames
        self.d_model = d_model
        self.action_dim = action_dim
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
        self.action_to_model = nn.Linear(action_dim, d_model)
        self.spatial_coord_proj = nn.Linear(2, d_model)
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_frames, d_model))

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
                UnifiedDiffusersBlock(
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

    def _positional_encoding(
        self,
        height: int,
        width: int,
        *,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        temporal = self.temporal_pos[:, :num_frames].to(dtype=dtype)
        spatial_coords = self._spatial_coords(height, width, device=device, dtype=dtype)
        spatial = self.spatial_coord_proj(spatial_coords)
        positional = spatial.unsqueeze(0).unsqueeze(2) + temporal.unsqueeze(1)
        return rearrange(positional, "b n t d -> b (n t) d")

    def _encode_actions(self, actions: Tensor, *, dtype: torch.dtype) -> Tensor:
        if actions.dtype != torch.long:
            actions = actions.long()
        bits = self.action_bit_lut[actions].to(dtype=dtype)
        return self.action_to_model(self.action_mlp(bits))

    @staticmethod
    def _causal_action_mask(
        *,
        query_absolute_steps: Tensor,
        num_action_steps: int,
        device: torch.device,
    ) -> Tensor:
        key_steps = torch.arange(num_action_steps, device=device)
        return key_steps.unsqueeze(0) > query_absolute_steps.unsqueeze(1)

    @staticmethod
    def _to_attention_bias(
        mask: Tensor,
        *,
        batch_size: int,
        num_heads: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        bias = torch.zeros(mask.shape, device=device, dtype=dtype)
        bias = bias.masked_fill(mask, torch.tensor(-10_000.0, device=device, dtype=dtype))
        return bias.unsqueeze(0).repeat(batch_size * num_heads, 1, 1)

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
        context_frames: int,
    ) -> Tensor:
        if latents.ndim != 5:
            raise ValueError(f"Expected latents (B, C, T, H, W), got {tuple(latents.shape)}")
        if actions.ndim != 2:
            raise ValueError(f"Expected actions (B, T), got {tuple(actions.shape)}")

        batch, channels, total_frames, height, width = latents.shape
        if channels != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, got {channels}")
        if actions.shape != (batch, total_frames):
            raise ValueError(
                f"actions must be (B={batch}, T={total_frames}), got {tuple(actions.shape)}"
            )
        if context_frames < 1 or context_frames >= total_frames:
            raise ValueError(
                f"context_frames must be in [1, T-1], got {context_frames} with T={total_frames}"
            )
        if total_frames > self.max_frames:
            raise ValueError(f"total_frames ({total_frames}) exceeds max_frames ({self.max_frames})")

        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.ndim != 1:
            raise ValueError(f"Expected timesteps (B,), got {tuple(timesteps.shape)}")
        if timesteps.shape[0] == 1 and batch > 1:
            timesteps = timesteps.expand(batch)
        if timesteps.shape[0] != batch:
            raise ValueError(f"Expected {batch} timesteps, got {timesteps.shape[0]}")

        self._validate_actions(actions)

        tokens = rearrange(latents, "b c t h w -> b (h w t) c")
        tokens = self.input_proj(tokens)
        tokens = tokens + self._positional_encoding(
            height,
            width,
            num_frames=total_frames,
            device=latents.device,
            dtype=tokens.dtype,
        )

        action_tokens = self._encode_actions(actions, dtype=tokens.dtype)
        spatial_tokens = height * width
        query_steps_abs = torch.arange(total_frames, device=latents.device).repeat(spatial_tokens)
        action_mask = self._causal_action_mask(
            query_absolute_steps=query_steps_abs,
            num_action_steps=total_frames,
            device=latents.device,
        )
        action_bias = self._to_attention_bias(
            action_mask,
            batch_size=batch,
            num_heads=self.num_heads,
            device=latents.device,
            dtype=tokens.dtype,
        )

        t_proj = self.time_proj(timesteps.float()).to(device=latents.device, dtype=tokens.dtype)
        cond = self.time_embedding(t_proj).to(dtype=tokens.dtype)

        for block in self.blocks:
            tokens = block(
                tokens,
                action_tokens=action_tokens,
                cond=cond,
                action_attn_bias=action_bias,
            )

        tokens = self.final_norm(tokens)
        shift_scale = self.final_mod(cond).unsqueeze(1)
        shift, scale = shift_scale.chunk(2, dim=-1)
        tokens = tokens * (1.0 + scale) + shift

        pred = self.output_proj(tokens)
        pred = rearrange(pred, "b (h w t) c -> b c t h w", h=height, w=width, t=total_frames)
        return pred[:, :, context_frames:]