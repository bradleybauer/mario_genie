"""RAM VAE: temporal autoencoder for NES RAM byte sequences.

Game-agnostic design — works for any NES game given its set of non-constant
RAM addresses.  Processes T-frame clips (matching the video VAE clip length)
with causal temporal convolutions so frame t depends only on frames <= t.

Architecture (Option 1 — Flat Temporal MLP):
  Encoder: per-frame MLP -> causal temporal conv -> mean/logvar
  Decoder: causal temporal conv -> per-frame MLP -> sigmoid
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class RAMVAEOutput:
    reconstruction: Tensor      # (B, T, N_bytes)
    posterior_mean: Tensor      # (B, T, latent_dim)
    posterior_logvar: Tensor    # (B, T, latent_dim)
    latents: Tensor             # (B, T, latent_dim)


class CausalConv1d(nn.Module):
    """1D convolution with causal (left-only) replicate padding on the time axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0), mode="replicate")
        return self.conv(x)


class ResidualFC(nn.Module):
    """Residual fully-connected block: LayerNorm -> SiLU -> Linear -> residual."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fc(F.silu(self.norm(x)))


class TemporalResBlock(nn.Module):
    """CausalConv1d temporal residual block: GroupNorm -> SiLU -> CausalConv1d."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.conv = CausalConv1d(channels, channels, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T)
        return x + self.conv(F.silu(self.norm(x)))


class RAMVAE(nn.Module):
    """Temporal RAM VAE using per-frame MLP + causal temporal convolutions.

    Input:  (B, T, N_bytes) uint8 or float normalized to [0, 1]
    Output: RAMVAEOutput with reconstruction, mean, logvar, latents
    """

    def __init__(
        self,
        *,
        n_bytes: int,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        n_fc_blocks: int = 2,
        n_temporal_blocks: int = 2,
        temporal_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.n_bytes = n_bytes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # --- Encoder: per-frame byte processing ---
        self.encoder_in = nn.Linear(n_bytes, hidden_dim)
        self.encoder_blocks = nn.ModuleList(
            [ResidualFC(hidden_dim) for _ in range(n_fc_blocks)]
        )

        # --- Encoder: causal temporal mixing ---
        self.encoder_temporal = nn.ModuleList(
            [TemporalResBlock(hidden_dim, temporal_kernel_size)
             for _ in range(n_temporal_blocks)]
        )

        # --- Encoder: bottleneck ---
        self.encoder_out = nn.Conv1d(hidden_dim, latent_dim * 2, kernel_size=1)

        # --- Decoder: temporal processing ---
        self.decoder_in = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)
        self.decoder_temporal = nn.ModuleList(
            [TemporalResBlock(hidden_dim, temporal_kernel_size)
             for _ in range(n_temporal_blocks)]
        )

        # --- Decoder: per-frame byte reconstruction ---
        self.decoder_blocks = nn.ModuleList(
            [ResidualFC(hidden_dim) for _ in range(n_fc_blocks)]
        )
        self.decoder_out = nn.Linear(hidden_dim, n_bytes)

    @staticmethod
    def kl_loss(mean: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()

    def encode(self, ram: Tensor) -> tuple[Tensor, Tensor]:
        """Encode (B, T, N_bytes) -> (mean, logvar) each (B, T, latent_dim)."""
        B, T, N = ram.shape

        # Per-frame MLP
        x = ram.reshape(B * T, N)
        x = F.silu(self.encoder_in(x))
        for block in self.encoder_blocks:
            x = block(x)

        # Reshape for temporal conv: (B, hidden, T)
        x = x.reshape(B, T, self.hidden_dim).permute(0, 2, 1)
        for temporal_block in self.encoder_temporal:
            x = temporal_block(x)

        # Bottleneck: (B, latent_dim * 2, T)
        x = self.encoder_out(x)

        # -> (B, T, latent_dim * 2) then split
        x = x.permute(0, 2, 1)
        mean, logvar = x.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, min=-30.0, max=10.0)
        return mean, logvar

    def reparameterize(
        self, mean: Tensor, logvar: Tensor, sample_posterior: bool = True,
    ) -> Tensor:
        if not sample_posterior:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, latents: Tensor) -> Tensor:
        """Decode (B, T, latent_dim) -> (B, T, N_bytes) in [0, 1]."""
        B, T, _ = latents.shape

        # Temporal processing: (B, latent_dim, T)
        x = latents.permute(0, 2, 1)
        x = self.decoder_in(x)
        for temporal_block in self.decoder_temporal:
            x = temporal_block(x)

        # Per-frame MLP: (B*T, hidden)
        x = x.permute(0, 2, 1).reshape(B * T, self.hidden_dim)
        for block in self.decoder_blocks:
            x = block(x)

        # Output: (B, T, N_bytes)
        x = torch.sigmoid(self.decoder_out(x))
        return x.reshape(B, T, self.n_bytes)

    def forward(
        self, ram: Tensor, *, sample_posterior: bool = True,
    ) -> RAMVAEOutput:
        """Full forward pass.

        Args:
            ram: (B, T, N_bytes) float tensor normalized to [0, 1],
                 or uint8 which will be converted automatically.
        """
        if ram.dtype == torch.uint8:
            ram = ram.float() / 255.0

        if ram.ndim != 3:
            raise ValueError(
                f"Expected RAM tensor with shape (B, T, N_bytes), got {tuple(ram.shape)}"
            )
        if ram.shape[-1] != self.n_bytes:
            raise ValueError(
                f"Expected {self.n_bytes} RAM bytes, got {ram.shape[-1]}"
            )

        mean, logvar = self.encode(ram)
        latents = self.reparameterize(mean, logvar, sample_posterior=sample_posterior)
        reconstruction = self.decode(latents)

        return RAMVAEOutput(
            reconstruction=reconstruction,
            posterior_mean=mean,
            posterior_logvar=logvar,
            latents=latents,
        )
