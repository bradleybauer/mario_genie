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


class DiffusersEncoderBlock(nn.Module):
    """History encoder block built from Diffusers attention/MLP layers."""

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

        self.norm1 = nn.LayerNorm(d_model)
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

        self.norm3 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, inner_dim=hidden_dim, dropout=dropout)

        _init_zero_attention_out(self.self_attn)
        _init_zero_attention_out(self.action_cross_attn)
        _init_zero_ff_out(self.mlp)

    def forward(
        self,
        tokens: Tensor,
        *,
        action_tokens: Tensor,
        action_attn_mask: Tensor | None,
        action_cond_scale: Tensor | None = None,
    ) -> Tensor:
        h = self.norm1(tokens)
        tokens = tokens + self.self_attn(h)

        q = self.norm2(tokens)
        action_update = self.action_cross_attn(
            q,
            encoder_hidden_states=action_tokens,
            attention_mask=action_attn_mask,
        )
        if action_cond_scale is not None:
            action_update = action_update * action_cond_scale
        tokens = tokens + action_update

        tokens = tokens + self.mlp(self.norm3(tokens))
        return tokens


class DiffusersDecoderBlock(nn.Module):
    """Future decoder block built from Diffusers attention/MLP layers."""

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
        self.history_cross_attn = Attention(
            query_dim=d_model,
            cross_attention_dim=d_model,
            heads=num_heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=True,
            out_bias=True,
            qk_norm="rms_norm",
        )

        self.norm3 = nn.LayerNorm(d_model)
        self.action_cross_attn = Attention(
            query_dim=d_model,
            cross_attention_dim=d_model,
            heads=num_heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=True,
            out_bias=True,
        )

        self.norm4 = AdaLayerNorm(d_model)
        self.mlp = FeedForward(d_model, inner_dim=hidden_dim, dropout=dropout)

        _init_zero_attention_out(self.self_attn)
        _init_zero_attention_out(self.history_cross_attn)
        _init_zero_attention_out(self.action_cross_attn)
        _init_zero_ff_out(self.mlp)

    def forward(
        self,
        tokens: Tensor,
        *,
        encoded_history: Tensor,
        action_tokens: Tensor,
        cond: Tensor,
        action_attn_mask: Tensor | None,
        action_cond_scale: Tensor | None = None,
    ) -> Tensor:
        h = self.norm1(tokens, cond)
        tokens = tokens + self.self_attn(h)

        q = self.norm2(tokens)
        tokens = tokens + self.history_cross_attn(
            q,
            encoder_hidden_states=encoded_history,
        )

        q = self.norm3(tokens)
        action_update = self.action_cross_attn(
            q,
            encoder_hidden_states=action_tokens,
            attention_mask=action_attn_mask,
        )
        if action_cond_scale is not None:
            action_update = action_update * action_cond_scale
        tokens = tokens + action_update

        tokens = tokens + self.mlp(self.norm4(tokens, cond))
        return tokens


class VideoLatentDiTDiffusers(ModelMixin, ConfigMixin):
    """Diffusers-native video-latent DiT for flow-matching future prediction."""

    @register_to_config
    def __init__(
        self,
        *,
        latent_channels: int,
        num_actions: int,
        action_dim: int = 32,
        action_values: list[int] | None = None,
        d_model: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
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
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if max_latents <= 0:
            raise ValueError("max_latents must be positive")

        self.latent_channels = latent_channels
        self.num_actions = num_actions
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
        self.action_to_model = nn.Linear(action_dim, d_model)

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

        self.encoder_blocks = nn.ModuleList(
            [
                DiffusersEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                DiffusersDecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
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
        return self.action_to_model(self.action_mlp(bits))

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
        # Diffusers Attention expects additive attention bias with shape (B*H, Nq, Nk).
        bias = torch.zeros(mask.shape, device=device, dtype=dtype)
        bias = bias.masked_fill(mask, torch.tensor(-10_000.0, device=device, dtype=dtype))
        return bias.unsqueeze(0).repeat(batch_size * num_heads, 1, 1)

    def _validate_actions(self, actions: Tensor) -> None:
        if torch.any(actions < 0) or torch.any(actions >= self.num_actions):
            raise ValueError(
                f"Action IDs must be in [0, {self.num_actions - 1}] "
                f"but got min={int(actions.min())}, max={int(actions.max())}"
            )

    def encode_history(
        self,
        history_latents: Tensor,
        history_actions: Tensor,
        action_cond_scale: Tensor | float | None = None,
    ) -> Tensor:
        if history_latents.ndim != 5:
            raise ValueError(
                f"Expected history_latents (B, C, T, H, W), got {tuple(history_latents.shape)}"
            )
        batch, channels, context_latents, height, width = history_latents.shape
        if channels != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, got {channels}")
        if history_actions.shape != (batch, context_latents):
            raise ValueError(
                f"history_actions must be (B={batch}, context_latents={context_latents}), "
                f"got {tuple(history_actions.shape)}"
            )
        if context_latents > self.max_latents:
            raise ValueError(f"context_latents ({context_latents}) exceeds max_latents ({self.max_latents})")

        self._validate_actions(history_actions)
        cond_scale = self._prepare_action_cond_scale(
            action_cond_scale,
            batch_size=batch,
            device=history_latents.device,
            dtype=history_latents.dtype,
        )

        tokens = rearrange(history_latents, "b c t h w -> b (h w t) c")
        tokens = self.input_proj(tokens)
        tokens = tokens + self._positional_encoding(
            height,
            width,
            start_latent=0,
            num_latents=context_latents,
            device=history_latents.device,
            dtype=tokens.dtype,
        )

        action_tokens = self._encode_actions(history_actions, dtype=tokens.dtype)

        for block in self.encoder_blocks:
            tokens = block(
                tokens,
                action_tokens=action_tokens,
                action_attn_mask=None,
                action_cond_scale=cond_scale,
            )

        return tokens

    def decode_future(
        self,
        noisy_future: Tensor,
        actions: Tensor,
        timesteps: Tensor,
        encoded_history: Tensor,
        context_latents: int,
        action_cond_scale: Tensor | float | None = None,
    ) -> Tensor:
        if noisy_future.ndim != 5:
            raise ValueError(
                f"Expected noisy_future (B, C, future_latents, H, W), got {tuple(noisy_future.shape)}"
            )
        batch, channels, future_latents, height, width = noisy_future.shape
        total_latents = context_latents + future_latents

        if channels != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, got {channels}")
        if actions.shape != (batch, total_latents):
            raise ValueError(
                f"actions must be (B={batch}, context+future={total_latents}), "
                f"got {tuple(actions.shape)}"
            )
        if total_latents > self.max_latents:
            raise ValueError(f"total_latents ({total_latents}) exceeds max_latents ({self.max_latents})")

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
            device=noisy_future.device,
            dtype=noisy_future.dtype,
        )

        tokens = rearrange(noisy_future, "b c t h w -> b (h w t) c")
        tokens = self.input_proj(tokens)
        tokens = tokens + self._positional_encoding(
            height,
            width,
            start_latent=context_latents,
            num_latents=future_latents,
            device=noisy_future.device,
            dtype=tokens.dtype,
        )

        action_tokens = self._encode_actions(actions, dtype=tokens.dtype)
        spatial_tokens = height * width
        query_steps_abs = (
            torch.arange(future_latents, device=noisy_future.device).repeat(spatial_tokens)
            + context_latents
        )
        action_mask = self._causal_action_mask(
            query_absolute_steps=query_steps_abs,
            num_action_steps=total_latents,
            device=noisy_future.device,
        )
        action_bias = self._to_attention_bias(
            action_mask,
            batch_size=batch,
            num_heads=self.num_heads,
            device=noisy_future.device,
            dtype=tokens.dtype,
        )

        t_proj = self.time_proj(timesteps.float()).to(device=noisy_future.device, dtype=tokens.dtype)
        cond = self.time_embedding(t_proj).to(dtype=tokens.dtype)

        for block in self.decoder_blocks:
            tokens = block(
                tokens,
                encoded_history=encoded_history,
                action_tokens=action_tokens,
                cond=cond,
                action_attn_mask=action_bias,
                action_cond_scale=cond_scale,
            )

        tokens = self.final_norm(tokens)
        shift_scale = self.final_mod(cond).unsqueeze(1)
        shift, scale = shift_scale.chunk(2, dim=-1)
        tokens = tokens * (1.0 + scale) + shift

        pred = self.output_proj(tokens)
        return rearrange(pred, "b (h w t) c -> b c t h w", h=height, w=width, t=future_latents)

    def forward(
        self,
        latents: Tensor,
        actions: Tensor,
        timesteps: Tensor,
        context_latents: int,
        action_cond_scale: Tensor | float | None = None,
    ) -> Tensor:
        if latents.ndim != 5:
            raise ValueError(f"Expected latents (B, C, T, H, W), got {tuple(latents.shape)}")
        if actions.ndim != 2:
            raise ValueError(f"Expected actions (B, T), got {tuple(actions.shape)}")
        _, _, total_latents, _, _ = latents.shape
        if context_latents < 1 or context_latents >= total_latents:
            raise ValueError(
                f"context_latents must be in [1, T-1], got {context_latents} with T={total_latents}"
            )

        clean_history = latents[:, :, :context_latents]
        noisy_future = latents[:, :, context_latents:]

        encoded_history = self.encode_history(
            clean_history,
            actions[:, :context_latents],
            action_cond_scale=action_cond_scale,
        )
        return self.decode_future(
            noisy_future,
            actions,
            timesteps,
            encoded_history,
            context_latents,
            action_cond_scale=action_cond_scale,
        )
