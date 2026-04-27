"""RAM VAE: temporal autoencoder for NES RAM byte sequences.

Game-agnostic design — works for any NES game given its set of non-constant
RAM addresses.  Processes T-frame clips (matching the video VAE clip length)
with causal temporal convolutions so frame t depends only on frames <= t.

Categorical I/O: each RAM address is treated as a categorical variable with
a per-address vocabulary of observed byte values.  The encoder embeds each
address value and projects the concatenation to hidden_dim.  The decoder
outputs logits over the per-address vocabularies (concatenated into a flat
total_classes dimension) for cross-entropy training.

Architecture:
  Encoder: per-address embedding -> concat -> linear -> per-frame MLP ->
           causal temporal conv -> [optional 2× temporal pack] -> mean/logvar
  Decoder: [optional 2× temporal unpack] -> causal temporal conv ->
           per-frame MLP -> logits (total_classes)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class RAMVAEOutput:
    logits: Tensor              # (B, T, total_classes) raw logits
    reconstruction: Tensor      # (B, T, N_addr) predicted byte values
    posterior_mean: Tensor      # (B, T_lat, latent_dim)
    posterior_logvar: Tensor    # (B, T_lat, latent_dim)
    latents: Tensor             # (B, T_lat, latent_dim)


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

    def __init__(self, dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fc(self.dropout(F.silu(self.norm(x))))


class TemporalResBlock(nn.Module):
    """CausalConv1d temporal residual block: GroupNorm -> SiLU -> CausalConv1d."""

    def __init__(self, channels: int, kernel_size: int = 3, *, dropout: float = 0.0) -> None:
        super().__init__()
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.dropout = nn.Dropout(dropout)
        self.conv = CausalConv1d(channels, channels, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T)
        return x + self.conv(self.dropout(F.silu(self.norm(x))))


class RAMVAE(nn.Module):
    """Temporal RAM VAE with categorical input/output.

    Each RAM address is treated as a categorical variable.  The encoder embeds
    per-address byte values via a shared embedding table (with per-address
    offsets), concatenates the embeddings, and projects to hidden_dim.  The
    decoder outputs flat logits over the concatenated per-address vocabularies.

    Input:  (B, T, N_addr) uint8 or long raw byte values
    Output: RAMVAEOutput with logits, reconstruction, mean, logvar, latents
    """

    def __init__(
        self,
        *,
        values_per_address: list[list[int]],
        embed_dim: int = 8,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        n_fc_blocks: int = 2,
        n_temporal_blocks: int = 2,
        temporal_kernel_size: int = 3,
        temporal_downsample: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if temporal_downsample not in (0, 1):
            raise ValueError("temporal_downsample must be 0 or 1")

        n_addresses = len(values_per_address)
        cardinalities = [len(v) for v in values_per_address]
        total_classes = sum(cardinalities)

        self.n_addresses = n_addresses
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.total_classes = total_classes
        self.temporal_downsample = temporal_downsample

        # --- Lookup tables (buffers, not parameters) ---
        value_to_index = torch.zeros(n_addresses, 256, dtype=torch.long)
        index_to_value = torch.zeros(total_classes, dtype=torch.long)
        offsets = []
        offset = 0
        for i, vals in enumerate(values_per_address):
            for cls_idx, byte_val in enumerate(vals):
                value_to_index[i, byte_val] = cls_idx
                index_to_value[offset + cls_idx] = byte_val
            offsets.append(offset)
            offset += len(vals)

        self.register_buffer("value_to_index", value_to_index)
        self.register_buffer("index_to_value", index_to_value)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.register_buffer("cardinalities", torch.tensor(cardinalities, dtype=torch.long))
        # Pre-built address-of mapping for vectorized CE: address_of[j] = address index
        address_of = torch.repeat_interleave(
            torch.arange(n_addresses, dtype=torch.long),
            torch.tensor(cardinalities, dtype=torch.long),
        )
        self.register_buffer("address_of", address_of)

        # Pre-built gather indices for vectorized logits_to_values (compile-friendly)
        max_card = max(cardinalities)
        self.max_card = max_card
        gather_indices = torch.zeros(n_addresses, max_card, dtype=torch.long)
        gather_mask = torch.zeros(n_addresses, max_card, dtype=torch.bool)
        for i in range(n_addresses):
            c = cardinalities[i]
            gather_indices[i, :c] = torch.arange(offsets[i], offsets[i] + c)
            gather_mask[i, :c] = True
        self.register_buffer("gather_indices", gather_indices)  # (N_addr, max_card)
        self.register_buffer("gather_mask", gather_mask)        # (N_addr, max_card)

        # --- Encoder: categorical embedding + projection ---
        self.value_embedding = nn.Embedding(total_classes, embed_dim)
        self.encoder_in = nn.Linear(n_addresses * embed_dim, hidden_dim)
        self.encoder_blocks = nn.ModuleList(
            [ResidualFC(hidden_dim, dropout=dropout) for _ in range(n_fc_blocks)]
        )

        # --- Encoder: causal temporal mixing ---
        self.encoder_temporal = nn.ModuleList(
            [TemporalResBlock(hidden_dim, temporal_kernel_size, dropout=dropout)
             for _ in range(n_temporal_blocks)]
        )

        # --- Encoder: bottleneck (with optional 2× temporal packing) ---
        bottleneck_in = hidden_dim * (2 if temporal_downsample == 1 else 1)
        self.encoder_out = nn.Conv1d(bottleneck_in, latent_dim * 2, kernel_size=1)

        # --- Decoder: temporal processing (with optional 2× unpack) ---
        decoder_hidden_in = hidden_dim * (2 if temporal_downsample == 1 else 1)
        self.decoder_in = nn.Conv1d(latent_dim, decoder_hidden_in, kernel_size=1)
        self.decoder_temporal = nn.ModuleList(
            [TemporalResBlock(hidden_dim, temporal_kernel_size, dropout=dropout)
             for _ in range(n_temporal_blocks)]
        )

        # --- Decoder: per-frame MLP -> logits ---
        self.decoder_blocks = nn.ModuleList(
            [ResidualFC(hidden_dim, dropout=dropout) for _ in range(n_fc_blocks)]
        )
        self.decoder_out = nn.Linear(hidden_dim, total_classes)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def raw_to_indices(self, ram: Tensor) -> Tensor:
        """Convert raw byte values (B, T, N_addr) -> per-address class indices."""
        B, T, N = ram.shape
        ram_long = ram.long()
        addr_idx = torch.arange(N, device=ram.device).expand(B, T, N)
        return self.value_to_index[addr_idx, ram_long]

    def logits_to_values(self, logits: Tensor) -> Tensor:
        """Convert (B, T, total_classes) logits -> (B, T, N_addr) byte values."""
        B, T, _ = logits.shape
        flat = logits.reshape(B * T, self.total_classes)

        # Gather into (BT, N_addr, max_card), mask invalid positions, argmax
        gathered = flat[:, self.gather_indices.reshape(-1)].reshape(
            B * T, self.n_addresses, self.max_card,
        )
        gathered = gathered.masked_fill(~self.gather_mask.unsqueeze(0), -1e30)
        local_argmax = gathered.argmax(dim=-1)  # (BT, N_addr)

        global_idx = local_argmax + self.offsets.unsqueeze(0)
        return self.index_to_value[global_idx].reshape(B, T, self.n_addresses)

    def logits_to_expected_values(self, logits: Tensor) -> Tensor:
        """Convert logits to differentiable per-address expected byte values."""
        B, T, _ = logits.shape
        flat = logits.reshape(B * T, self.total_classes)

        gathered = flat[:, self.gather_indices.reshape(-1)].reshape(
            B * T, self.n_addresses, self.max_card,
        )
        gathered = gathered.masked_fill(~self.gather_mask.unsqueeze(0), -1e30)

        probs = gathered.softmax(dim=-1)
        gather_values = self.index_to_value[self.gather_indices].to(dtype=logits.dtype)
        expected = (probs * gather_values.unsqueeze(0)).sum(dim=-1)
        return expected.reshape(B, T, self.n_addresses)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def kl_loss(mean: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()

    def categorical_loss(self, logits: Tensor, ram: Tensor, *, gamma: float = 0.0) -> Tensor:
        """Vectorized per-address cross-entropy with optional focal weighting.

        When ``gamma > 0`` the loss becomes focal:
            loss_i = -(1 - p_t)^gamma * log(p_t)
        which down-weights easy (high-confidence) predictions.
        Equivalent to standard cross-entropy when ``gamma = 0``.

        Args:
            logits: (B, T, total_classes) raw decoder logits.
            ram:    (B, T, N_addr) raw byte values (uint8 or long).
            gamma:  focal loss focusing parameter (0 = standard CE).

        Returns:
            Scalar mean loss averaged over addresses and batch elements.
        """
        B, T, _ = ram.shape
        BT = B * T
        logits_flat = logits.reshape(BT, self.total_classes)
        indices = self.raw_to_indices(ram)
        global_targets = indices.reshape(BT, self.n_addresses) + self.offsets.unsqueeze(0)

        # Numerically-stable logsumexp per address via scatter
        addr = self.address_of.unsqueeze(0).expand(BT, -1)  # (BT, total_classes)

        # Per-address max for stability
        max_per_addr = torch.full(
            (BT, self.n_addresses), -1e30,
            device=logits.device, dtype=logits.dtype,
        )
        max_per_addr.scatter_reduce_(1, addr, logits_flat, reduce="amax", include_self=False)

        # exp(logit - max), then sum per address
        stabilized = (logits_flat - max_per_addr.gather(1, addr)).exp()
        sum_exp = torch.zeros(
            BT, self.n_addresses, device=logits.device, dtype=logits.dtype,
        )
        sum_exp.scatter_add_(1, addr, stabilized)

        log_partition = max_per_addr + sum_exp.log()
        target_logits = logits_flat.gather(1, global_targets)
        log_p_t = target_logits - log_partition  # log probability of target

        if gamma > 0.0:
            p_t = log_p_t.exp()
            nll = -((1.0 - p_t) ** gamma) * log_p_t
        else:
            nll = -log_p_t

        return nll.mean()

    # ------------------------------------------------------------------
    # Encode / Decode / Forward
    # ------------------------------------------------------------------

    def latent_time(self, input_time: int) -> int:
        """Return the latent temporal length for a given input T."""
        if self.temporal_downsample == 1:
            return math.ceil(input_time / 2)
        return input_time

    def encode(self, ram: Tensor) -> tuple[Tensor, Tensor]:
        """Encode (B, T, N_addr) raw byte values -> (mean, logvar)."""
        B, T, N = ram.shape

        # Per-address class indices -> global embedding indices
        indices = self.raw_to_indices(ram)                     # (B, T, N)
        global_indices = indices + self.offsets.unsqueeze(0)   # broadcast (N,)
        embeds = self.value_embedding(global_indices)          # (B, T, N, E)
        x = embeds.reshape(B * T, N * self.embed_dim)

        # Per-frame MLP
        x = F.silu(self.encoder_in(x))
        for block in self.encoder_blocks:
            x = block(x)

        # Reshape for temporal conv: (B, hidden, T)
        x = x.reshape(B, T, self.hidden_dim).permute(0, 2, 1)
        for temporal_block in self.encoder_temporal:
            x = temporal_block(x)

        # Optional 2× temporal packing: (B, hidden, T) -> (B, hidden*2, T//2)
        if self.temporal_downsample == 1:
            if x.shape[2] % 2 != 0:
                x = torch.cat((x, x[:, :, -1:]), dim=2)
            x = rearrange(x, "b c (t pair) -> b (c pair) t", pair=2)

        # Bottleneck: (B, latent_dim * 2, T_lat) -> (B, T_lat, latent_dim * 2)
        x = self.encoder_out(x).permute(0, 2, 1)
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

    def decode(self, latents: Tensor, *, output_time: int | None = None) -> Tensor:
        """Decode (B, T_lat, latent_dim) -> (B, T, total_classes) raw logits."""
        B, T_lat, _ = latents.shape

        # (B, latent_dim, T_lat) -> (B, decoder_hidden_in, T_lat)
        x = latents.permute(0, 2, 1)
        x = self.decoder_in(x)

        # Optional 2× temporal unpacking: (B, hidden*2, T_lat) -> (B, hidden, T_lat*2)
        if self.temporal_downsample == 1:
            x = rearrange(x, "b (c pair) t -> b c (t pair)", pair=2)
            # Trim to the requested output length (undo pad-to-even from encoder)
            if output_time is not None:
                x = x[:, :, :output_time]

        T_out = x.shape[2]
        for temporal_block in self.decoder_temporal:
            x = temporal_block(x)

        # Per-frame MLP: (B*T_out, hidden)
        x = x.permute(0, 2, 1).reshape(B * T_out, self.hidden_dim)
        for block in self.decoder_blocks:
            x = block(x)

        # Output: (B, T_out, total_classes) raw logits
        return self.decoder_out(x).reshape(B, T_out, self.total_classes)

    def forward(
        self, ram: Tensor, *, sample_posterior: bool = True,
    ) -> RAMVAEOutput:
        """Full forward pass.

        Args:
            ram: (B, T, N_addr) raw byte values — uint8 or long.
        """
        if ram.dtype == torch.uint8:
            ram = ram.long()
        if ram.dtype != torch.long:
            ram = ram.long()

        if ram.ndim != 3:
            raise ValueError(
                f"Expected RAM tensor with shape (B, T, N_addr), got {tuple(ram.shape)}"
            )
        if ram.shape[-1] != self.n_addresses:
            raise ValueError(
                f"Expected {self.n_addresses} RAM addresses, got {ram.shape[-1]}"
            )

        T = ram.shape[1]
        mean, logvar = self.encode(ram)
        latents = self.reparameterize(mean, logvar, sample_posterior=sample_posterior)
        logits = self.decode(latents, output_time=T)
        reconstruction = self.logits_to_values(logits)

        return RAMVAEOutput(
            logits=logits,
            reconstruction=reconstruction,
            posterior_mean=mean,
            posterior_logvar=logvar,
            latents=latents,
        )
