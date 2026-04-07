"""Joint RAM/video VAE with a query-based video renderer.

The encoder operates on RAM byte sequences and produces a per-frame latent.
RAM reconstruction stays close to that shared state, while the video branch
builds causal RAM address-group memory tokens and lets a coarse spatial grid
refine itself by attending over those memory tokens before decoding the
resulting low-resolution video latent grid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.ram_vae import ResidualFC, TemporalResBlock
from src.models.video_vae import CausalConv3d, ResidualBlock3D, SpatialUpsample3D, _num_groups


@dataclass
class RAMVideoVAEv2Output:
    video_logits: Tensor
    ram_reconstruction: Tensor
    posterior_mean: Tensor
    posterior_logvar: Tensor
    latents: Tensor
    video_latents: Tensor


class SpatialCrossAttentionBlock(nn.Module):
    """Cross-attend spatial queries over temporal RAM-derived video memory."""

    def __init__(self, dim: int, *, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        hidden_dim = dim * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        *,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        # queries: (B, T * H_lat * W_lat, D), memory: (B, T, D)
        memory_norm = self.memory_norm(memory)
        attn_out, _ = self.attn(
            self.query_norm(queries),
            memory_norm,
            memory_norm,
            attn_mask=attn_mask,
            need_weights=False,
        )
        queries = queries + attn_out
        return queries + self.ffn(self.ffn_norm(queries))


class RAMVideoVAEv2(nn.Module):
    """Encode RAM into shared state, then render video with RAM-group attention."""

    spatial_downsample_factor = 32

    def __init__(
        self,
        *,
        n_bytes: int,
        num_colors: int,
        frame_height: int,
        frame_width: int,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        n_fc_blocks: int = 2,
        n_temporal_blocks: int = 2,
        temporal_kernel_size: int = 3,
        video_base_channels: int = 24,
        video_latent_channels: int = 16,
        temporal_downsample: int = 0,
        video_adapter_dim: int = 256,
        video_adapter_heads: int = 8,
        n_ram_groups: int = 128,
        n_video_temporal_blocks: int = 2,
        n_video_renderer_blocks: int = 2,
    ) -> None:
        super().__init__()
        if n_bytes <= 0:
            raise ValueError("n_bytes must be positive")
        if num_colors <= 0:
            raise ValueError("num_colors must be positive")
        if frame_height <= 0 or frame_width <= 0:
            raise ValueError("frame_height and frame_width must be positive")
        if temporal_downsample not in (0, 1):
            raise ValueError("temporal_downsample must be 0 or 1")
        if video_adapter_dim <= 0:
            raise ValueError("video_adapter_dim must be positive")
        if video_adapter_heads <= 0:
            raise ValueError("video_adapter_heads must be positive")
        if video_adapter_dim % video_adapter_heads != 0:
            raise ValueError("video_adapter_dim must be divisible by video_adapter_heads")
        if n_ram_groups <= 0:
            raise ValueError("n_ram_groups must be positive")

        self.n_bytes = n_bytes
        self.num_colors = num_colors
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.video_latent_channels = video_latent_channels
        self.temporal_downsample = temporal_downsample
        self.video_adapter_dim = video_adapter_dim
        self.latent_height = max(1, math.ceil(frame_height / self.spatial_downsample_factor))
        self.latent_width = max(1, math.ceil(frame_width / self.spatial_downsample_factor))
        self.num_spatial_queries = self.latent_height * self.latent_width
        self.video_projection_in_dim = latent_dim * (2 if temporal_downsample == 1 else 1)
        self.ram_packed_width = n_bytes * (2 if temporal_downsample == 1 else 1)
        self.n_ram_groups = min(n_ram_groups, self.ram_packed_width)
        self.ram_group_width = math.ceil(self.ram_packed_width / self.n_ram_groups)
        self.ram_packed_width_padded = self.ram_group_width * self.n_ram_groups

        self.encoder_in = nn.Linear(n_bytes, hidden_dim)
        self.encoder_blocks = nn.ModuleList(
            [ResidualFC(hidden_dim) for _ in range(n_fc_blocks)]
        )
        self.encoder_temporal = nn.ModuleList(
            [TemporalResBlock(hidden_dim, temporal_kernel_size) for _ in range(n_temporal_blocks)]
        )
        self.encoder_out = nn.Conv1d(hidden_dim, latent_dim * 2, kernel_size=1)

        self.ram_decoder_in = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)
        self.ram_decoder_temporal = nn.ModuleList(
            [TemporalResBlock(hidden_dim, temporal_kernel_size) for _ in range(n_temporal_blocks)]
        )
        self.ram_decoder_blocks = nn.ModuleList(
            [ResidualFC(hidden_dim) for _ in range(n_fc_blocks)]
        )
        self.ram_decoder_out = nn.Linear(hidden_dim, n_bytes)

        self.video_query_init = nn.Linear(
            self.video_projection_in_dim,
            self.num_spatial_queries * video_adapter_dim,
        )
        self.video_query_from_state = nn.Linear(self.video_projection_in_dim, video_adapter_dim)
        self.ram_group_proj = nn.Linear(self.ram_group_width, video_adapter_dim)
        self.ram_group_embeddings = nn.Parameter(
            torch.randn(self.n_ram_groups, video_adapter_dim) * 0.02
        )
        self.video_memory_temporal = nn.ModuleList(
            [TemporalResBlock(video_adapter_dim, temporal_kernel_size) for _ in range(n_video_temporal_blocks)]
        )
        self.video_spatial_queries = nn.Parameter(
            torch.randn(self.num_spatial_queries, video_adapter_dim) * 0.02
        )
        self.video_renderer_blocks = nn.ModuleList(
            [
                SpatialCrossAttentionBlock(
                    video_adapter_dim,
                    num_heads=video_adapter_heads,
                )
                for _ in range(n_video_renderer_blocks)
            ]
        )
        self.video_latent_norm = nn.LayerNorm(video_adapter_dim)
        self.video_latent_out = nn.Linear(video_adapter_dim, video_latent_channels)

        hidden_2 = video_base_channels * 2
        hidden_4 = video_base_channels * 4
        hidden_8 = video_base_channels * 8
        self.video_decoder_in = nn.Conv3d(video_latent_channels, hidden_8, kernel_size=1)
        self.video_decoder_mid = ResidualBlock3D(hidden_8)
        self.video_decoder_block6 = ResidualBlock3D(hidden_8)

        self.video_decoder_up1 = SpatialUpsample3D(hidden_8, hidden_8, upsample_time=temporal_downsample == 1)
        self.video_decoder_block5 = ResidualBlock3D(hidden_8)

        self.video_decoder_up2 = SpatialUpsample3D(hidden_8, hidden_4)
        self.video_decoder_block4 = ResidualBlock3D(hidden_4)

        self.video_decoder_up3 = SpatialUpsample3D(hidden_4, hidden_2)
        self.video_decoder_block3 = ResidualBlock3D(hidden_2)

        self.video_decoder_up4 = SpatialUpsample3D(hidden_2, video_base_channels)
        self.video_decoder_block2 = ResidualBlock3D(video_base_channels)

        self.video_decoder_up5 = SpatialUpsample3D(video_base_channels, video_base_channels)
        self.video_decoder_block1 = ResidualBlock3D(video_base_channels)

        self.video_decoder_norm = nn.GroupNorm(_num_groups(video_base_channels), video_base_channels)
        self.video_decoder_out = CausalConv3d(video_base_channels, num_colors, kernel_size=3)

    @staticmethod
    def kl_loss(mean: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()

    def encode(self, ram: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, time_steps, n_bytes = ram.shape
        # Flatten frames so the per-frame MLP sees (B * T, N_bytes).
        x = ram.reshape(batch_size * time_steps, n_bytes)
        x = F.silu(self.encoder_in(x))
        for block in self.encoder_blocks:
            x = block(x)

        # Temporal blocks expect channel-first layout: (B, hidden_dim, T).
        x = x.reshape(batch_size, time_steps, self.hidden_dim).permute(0, 2, 1)
        for temporal_block in self.encoder_temporal:
            x = temporal_block(x)

        # Split posterior stats back into per-frame latents: (B, T, latent_dim).
        x = self.encoder_out(x).permute(0, 2, 1)
        mean, logvar = x.chunk(2, dim=-1)
        return mean, torch.clamp(logvar, min=-30.0, max=10.0)

    def reparameterize(
        self,
        mean: Tensor,
        logvar: Tensor,
        sample_posterior: bool = True,
    ) -> Tensor:
        if not sample_posterior:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode_ram(self, latents: Tensor) -> Tensor:
        batch_size, time_steps, _ = latents.shape
        # RAM decoder mirrors the encoder: temporal convs in (B, latent_dim, T).
        x = latents.permute(0, 2, 1)
        x = self.ram_decoder_in(x)
        for temporal_block in self.ram_decoder_temporal:
            x = temporal_block(x)

        # Return to per-frame byte prediction space: (B * T, hidden_dim) -> (B, T, N_bytes).
        x = x.permute(0, 2, 1).reshape(batch_size * time_steps, self.hidden_dim)
        for block in self.ram_decoder_blocks:
            x = block(x)

        x = torch.sigmoid(self.ram_decoder_out(x))
        return x.reshape(batch_size, time_steps, self.n_bytes)

    def prepare_video_tokens(self, latents: Tensor) -> Tensor:
        video_tokens = latents
        if self.temporal_downsample == 1:
            if video_tokens.shape[1] % 2 != 0:
                video_tokens = torch.cat((video_tokens, video_tokens[:, -1:]), dim=1)
            # Pack adjacent frames so video tokens become (B, ceil(T / 2), 2 * latent_dim).
            video_tokens = rearrange(video_tokens, "b (t pair) d -> b t (d pair)", pair=2)
        return video_tokens

    def prepare_ram_group_tokens(self, ram: Tensor) -> Tensor:
        ram_tokens = ram
        if self.temporal_downsample == 1:
            if ram_tokens.shape[1] % 2 != 0:
                ram_tokens = torch.cat((ram_tokens, ram_tokens[:, -1:]), dim=1)
            # Pack adjacent RAM frames so grouped memory matches the video latent time axis.
            ram_tokens = rearrange(ram_tokens, "b (t pair) n -> b t (pair n)", pair=2)
        if self.ram_packed_width_padded > self.ram_packed_width:
            ram_tokens = F.pad(ram_tokens, (0, self.ram_packed_width_padded - self.ram_packed_width))
        return ram_tokens.reshape(ram_tokens.shape[0], ram_tokens.shape[1], self.n_ram_groups, self.ram_group_width)

    def build_video_attention_mask(self, time_steps: int, *, device: torch.device) -> Tensor:
        # Each spatial query at time t may only attend to RAM-group keys at times <= t.
        query_times = torch.arange(time_steps, device=device).repeat_interleave(self.num_spatial_queries)
        key_times = torch.arange(time_steps, device=device).repeat_interleave(self.n_ram_groups)
        return key_times.unsqueeze(0) > query_times.unsqueeze(1)

    def to_video_latents(self, ram: Tensor, latents: Tensor) -> Tensor:
        video_tokens = self.prepare_video_tokens(latents)
        batch_size, time_steps, _ = video_tokens.shape
        ram_group_tokens = self.prepare_ram_group_tokens(ram)

        # Project contiguous RAM address groups into memory tokens: (B, T', G_ram, D_video).
        memory = F.silu(self.ram_group_proj(ram_group_tokens))
        memory = memory + self.ram_group_embeddings.view(1, 1, self.n_ram_groups, self.video_adapter_dim)

        # Temporal refinement runs independently for each RAM group over time.
        memory_temporal = memory.permute(0, 2, 3, 1).reshape(
            batch_size * self.n_ram_groups,
            self.video_adapter_dim,
            time_steps,
        )
        for temporal_block in self.video_memory_temporal:
            memory_temporal = temporal_block(memory_temporal)
        memory = memory_temporal.reshape(
            batch_size,
            self.n_ram_groups,
            self.video_adapter_dim,
            time_steps,
        ).permute(0, 3, 1, 2)
        memory = memory.reshape(batch_size, time_steps * self.n_ram_groups, self.video_adapter_dim)

        # Start from a coarse spatial grid derived from the shared state, then refine it with attention.
        spatial_queries = self.video_spatial_queries.view(1, 1, self.num_spatial_queries, self.video_adapter_dim)
        coarse_queries = self.video_query_init(video_tokens).reshape(
            batch_size,
            time_steps,
            self.num_spatial_queries,
            self.video_adapter_dim,
        )
        frame_queries = self.video_query_from_state(video_tokens).unsqueeze(2)
        queries = coarse_queries + spatial_queries + frame_queries
        queries = queries.reshape(batch_size, time_steps * self.num_spatial_queries, self.video_adapter_dim)

        attn_mask = self.build_video_attention_mask(time_steps, device=memory.device)
        for block in self.video_renderer_blocks:
            queries = block(queries, memory, attn_mask=attn_mask)

        # Map attended query slots back into a low-res video grid: (B, C_lat, T', H_lat, W_lat).
        video_tokens = self.video_latent_out(self.video_latent_norm(queries))
        video_tokens = video_tokens.reshape(
            batch_size,
            time_steps,
            self.num_spatial_queries,
            self.video_latent_channels,
        )
        return rearrange(
            video_tokens,
            "b t (h w) c -> b c t h w",
            h=self.latent_height,
            w=self.latent_width,
        )

    def decode_video(
        self,
        video_latents: Tensor,
        *,
        output_shape: tuple[int, int, int] | None = None,
    ) -> Tensor:
        x = self.video_decoder_in(video_latents)
        x = self.video_decoder_mid(x)
        x = self.video_decoder_block6(x)
        x = self.video_decoder_up1(x)
        x = self.video_decoder_block5(x)
        x = self.video_decoder_up2(x)
        x = self.video_decoder_block4(x)
        x = self.video_decoder_up3(x)
        x = self.video_decoder_block3(x)
        x = self.video_decoder_up4(x)
        x = self.video_decoder_block2(x)
        x = self.video_decoder_up5(x)
        x = self.video_decoder_block1(x)
        x = self.video_decoder_out(F.silu(self.video_decoder_norm(x)))
        if output_shape is not None:
            output_frames, output_height, output_width = output_shape
            if (
                x.shape[2] < output_frames
                or x.shape[3] < output_height
                or x.shape[4] < output_width
            ):
                raise ValueError(
                    f"Decoded tensor shape {tuple(x.shape)} is smaller than requested output shape {output_shape}"
                )
            x = x[:, :, :output_frames, :output_height, :output_width]
        return x

    def forward(
        self,
        ram: Tensor,
        *,
        output_video_shape: tuple[int, int, int] | None = None,
        sample_posterior: bool = True,
    ) -> RAMVideoVAEv2Output:
        if ram.dtype == torch.uint8:
            ram = ram.float() / 255.0
        if ram.ndim != 3:
            raise ValueError(
                f"Expected RAM tensor with shape (B, T, N_bytes), got {tuple(ram.shape)}"
            )
        if ram.shape[-1] != self.n_bytes:
            raise ValueError(f"Expected {self.n_bytes} RAM bytes, got {ram.shape[-1]}")

        mean, logvar = self.encode(ram)
        latents = self.reparameterize(mean, logvar, sample_posterior=sample_posterior)
        video_latents = self.to_video_latents(ram, latents)
        if output_video_shape is None:
            output_video_shape = (ram.shape[1], self.frame_height, self.frame_width)
        video_logits = self.decode_video(video_latents, output_shape=output_video_shape)
        ram_reconstruction = self.decode_ram(latents)
        return RAMVideoVAEv2Output(
            video_logits=video_logits,
            ram_reconstruction=ram_reconstruction,
            posterior_mean=mean,
            posterior_logvar=logvar,
            latents=latents,
            video_latents=video_latents,
        )