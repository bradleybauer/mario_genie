"""Lightweight auxiliary heads for shaping video VAE latents.

These modules attach to the *encoder output* (posterior mean) of an already-
trained or in-training VideoVAE.  They add small predictive losses that
encourage the latent space to be temporally smooth, action-predictable, and
aligned with a frozen RAM VAE.

None of these modules modify the core VAE architecture.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.ram_vae import RAMVAE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spatial_pool(z: Tensor) -> Tensor:
    """Average-pool spatial dims of a video latent.

    Args:
        z: ``(B, C, T, H, W)`` video VAE posterior mean.

    Returns:
        ``(B, C, T)`` spatially pooled latent.
    """
    return z.mean(dim=(-2, -1))


def align_actions_to_latent_time(actions: Tensor, t_lat: int) -> Tensor:
    """Select actions aligned to latent temporal grid.

    The video VAE pair-packs frames (0,1), (2,3), … so latent frame *k*
    corresponds to input frames *2k* and *2k+1*.  We take the action at the
    first frame of each pair (even indices).

    Args:
        actions: ``(B, T_input)`` per-frame action indices.
        t_lat:   number of latent temporal frames.

    Returns:
        ``(B, t_lat)`` action indices aligned to latent frames.
    """
    return actions[:, : 2 * t_lat : 2]


def align_ram_to_latent_time(ram_mean: Tensor, t_lat: int) -> Tensor:
    """Pool consecutive RAM latent pairs to match video latent temporal grid.

    Args:
        ram_mean: ``(B, T, D)`` per-frame RAM VAE posterior mean.
        t_lat:    number of video latent temporal frames.

    Returns:
        ``(B, t_lat, D)`` temporally aligned RAM means.
    """
    T = ram_mean.shape[1]
    # Ensure even length for pair packing
    if T % 2 != 0:
        ram_mean = torch.cat([ram_mean, ram_mean[:, -1:]], dim=1)
    # Average consecutive pairs: (0,1), (2,3), ...
    paired = ram_mean.unfold(1, 2, 2).mean(dim=-1)  # (B, T//2, D)
    return paired[:, :t_lat]


# ---------------------------------------------------------------------------
# Next-Frame Predictor
# ---------------------------------------------------------------------------

class NextFramePredictor(nn.Module):
    """Predict next spatially-pooled latent frame from current frame + action.

    Intentionally weak (2-layer MLP) so the encoder must produce predictable
    latents rather than the head compensating for bad structure.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        num_actions: int = 42,
        action_embed_dim: int = 16,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, action_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + action_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        posterior_mean: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict next-frame pooled latent for each consecutive pair.

        Args:
            posterior_mean: ``(B, C, T_lat, H, W)`` encoder posterior mean.
            actions: ``(B, T_input)`` per-frame action indices.

        Returns:
            (predictions, targets) each ``(B, T_lat - 1, C)``.
            Targets are **not** detached — gradients flow to encoder.
        """
        # Upcast to float32 — auxiliary heads are tiny; bf16 MSE can overflow
        # and bf16 spatial pooling can lose precision.
        z = _spatial_pool(posterior_mean).float()  # (B, C, T_lat)
        z = z.permute(0, 2, 1)  # (B, T_lat, C)
        t_lat = z.shape[1]

        aligned_actions = align_actions_to_latent_time(actions, t_lat).long()  # (B, T_lat)
        a_embed = self.action_embed(aligned_actions)  # (B, T_lat, action_embed_dim)

        # Predict z_{t+1} from (z_t, a_t) for t in [0, T_lat - 1)
        z_input = z[:, :-1]  # (B, T_lat-1, C)
        a_input = a_embed[:, :-1]  # (B, T_lat-1, action_embed_dim)
        x = torch.cat([z_input, a_input], dim=-1)  # (B, T_lat-1, C + action_embed_dim)
        predictions = self.mlp(x)  # (B, T_lat-1, C)
        targets = z[:, 1:]  # (B, T_lat-1, C)

        return predictions, targets


# ---------------------------------------------------------------------------
# Temporal Smoothness
# ---------------------------------------------------------------------------

def temporal_smoothness_loss(posterior_mean: Tensor) -> Tensor:
    """Cosine-similarity temporal smoothness on encoder posterior mean.

    Encourages consecutive latent frames to be similar (scale-invariant,
    does not fight with KL which controls magnitude).

    Args:
        posterior_mean: ``(B, C, T_lat, H, W)`` encoder posterior mean.

    Returns:
        Scalar loss: ``1 - mean(cosine_sim(z_t, z_{t+1}))``.
    """
    B, C, T, H, W = posterior_mean.shape
    # Upcast to float32 — cosine similarity norms underflow in bf16 for
    # 784-dim vectors, producing NaN.
    z = posterior_mean.float().permute(0, 2, 1, 3, 4).reshape(B, T, C * H * W)

    z_curr = z[:, :-1]  # (B, T-1, D)
    z_next = z[:, 1:]   # (B, T-1, D)

    cos_sim = F.cosine_similarity(z_curr, z_next, dim=-1)  # (B, T-1)
    return 1.0 - cos_sim.mean()


# ---------------------------------------------------------------------------
# RAM Alignment
# ---------------------------------------------------------------------------

class RAMAlignmentHead(nn.Module):
    """Project spatially-pooled video latent to frozen RAM latent space.

    Single linear layer — intentionally minimal so the video encoder
    must do the heavy lifting to match RAM structure.
    """

    def __init__(self, video_latent_dim: int = 16, ram_latent_dim: int = 32) -> None:
        super().__init__()
        self.proj = nn.Linear(video_latent_dim, ram_latent_dim)

    def forward(self, pooled_video: Tensor) -> Tensor:
        """Project pooled video latent to RAM space.

        Args:
            pooled_video: ``(B, T_lat, video_latent_dim)``

        Returns:
            ``(B, T_lat, ram_latent_dim)`` projected latent.
        """
        return self.proj(pooled_video)

    def loss(
        self,
        posterior_mean: Tensor,
        ram_mean: Tensor,
    ) -> Tensor:
        """MSE between projected video latent and frozen RAM mean.

        Args:
            posterior_mean: ``(B, C, T_lat, H, W)`` video encoder posterior mean.
            ram_mean: ``(B, T, ram_latent_dim)`` frozen RAM VAE posterior mean
                (per-frame, before temporal alignment).

        Returns:
            Scalar MSE loss.  RAM side is detached (frozen); gradients flow
            to video encoder through posterior_mean.
        """
        # Upcast to float32 for stable MSE computation.
        z = _spatial_pool(posterior_mean).float()  # (B, C, T_lat)
        z = z.permute(0, 2, 1)  # (B, T_lat, C)
        t_lat = z.shape[1]

        # Align RAM temporal resolution to video latent temporal resolution
        ram_aligned = align_ram_to_latent_time(ram_mean.detach(), t_lat)  # (B, T_lat, D_ram)

        projected = self.proj(z)  # (B, T_lat, D_ram)
        return F.mse_loss(projected, ram_aligned.float())


# ---------------------------------------------------------------------------
# Frozen RAM VAE loader
# ---------------------------------------------------------------------------

def load_frozen_ram_vae(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> RAMVAE:
    """Load a pre-trained RAM VAE in frozen eval mode.

    Args:
        checkpoint_path: path to the ``best.pt`` checkpoint file.
        data_dir: path to ``data/normalized`` containing ``ram_addresses.json``.
        device: target device.

    Returns:
        Frozen :class:`RAMVAE` with all gradients disabled.
    """
    checkpoint_path = Path(checkpoint_path)
    data_dir = Path(data_dir)

    # Load config to get architecture hyperparams
    config_path = checkpoint_path.parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Load values_per_address from data dir
    ram_json_path = data_dir / "ram_addresses.json"
    with open(ram_json_path) as f:
        ram_addr_info = json.load(f)
    values_per_address: list[list[int]] = ram_addr_info["values_per_address"]

    model_cfg = config.get("model", config.get("training", {}))

    # Infer embed_dim from checkpoint: if encoder_in.weight exists, its
    # in_features / n_addresses gives the per-address embedding dim.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint.get("model", checkpoint)
    n_addresses = len(values_per_address)
    embed_dim = model_cfg.get("embed_dim", 8)
    if "encoder_in.weight" in model_state:
        ckpt_in_features = model_state["encoder_in.weight"].shape[1]
        embed_dim = ckpt_in_features // n_addresses

    model = RAMVAE(
        values_per_address=values_per_address,
        embed_dim=embed_dim,
        hidden_dim=model_cfg.get("hidden_dim", 256),
        latent_dim=model_cfg.get("latent_dim", 32),
        n_fc_blocks=model_cfg.get("n_fc_blocks", 2),
        n_temporal_blocks=model_cfg.get("n_temporal_blocks", 2),
        temporal_kernel_size=model_cfg.get("temporal_kernel_size", 3),
    )

    # Only load params/buffers whose shapes match (older checkpoints may
    # have a different decoder layout; we only need the encoder path).
    current_state = model.state_dict()
    filtered = {
        k: v for k, v in model_state.items()
        if k in current_state and current_state[k].shape == v.shape
    }
    model.load_state_dict(filtered, strict=False)

    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    return model
