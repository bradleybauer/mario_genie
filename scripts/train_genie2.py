"""
Genie 2 – Autoregressive Latent Diffusion World Model (small version)

Architecture:
  1. Autoencoder (VAE): compresses 224×224 palette-indexed frames into
     continuous latent grids (latent_dim, 14, 14).
  2. Dynamics Transformer: causal transformer over a sequence of latent
     frames + action embeddings, producing context vectors.
  3. Latent Diffusion: DDPM-style denoiser that predicts the next frame's
     latent conditioned on the transformer's output.

Training is two-phase:
  Phase 1 – Train the autoencoder on reconstruction (cross-entropy over
            palette indices + KL regularisation).
  Phase 2 – Freeze the autoencoder, encode the dataset into latents, and
            train the dynamics transformer + diffusion model to predict
            the next frame's latent given previous frames and actions.

Usage:
  # Phase 1: train autoencoder
  python scripts/train_genie2.py --phase autoencoder --epochs 50

  # Phase 2: train dynamics
  python scripts/train_genie2.py --phase dynamics --ae-checkpoint checkpoints/genie2/ae_best.pt --epochs 100
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.system_info import get_available_memory, get_effective_cpu_count

# ---------------------------------------------------------------------------
# Hyperparameters (small model)
# ---------------------------------------------------------------------------
LATENT_CHANNELS = 4       # autoencoder latent depth
LATENT_SIZE = 14           # 224 / 16 = 14
KL_WEIGHT = 1e-5           # VAE KL penalty (small to avoid posterior collapse on categorical data)

TRANSFORMER_DIM = 256
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 6
TRANSFORMER_MLP_RATIO = 4

DIFFUSION_STEPS = 200      # DDPM T
BETA_START = 1e-4
BETA_END = 0.02
NUM_ACTIONS = 42

CROP_224_SIZE = 224


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

def _index_file(file_idx: int, filepath: str, seq_len: int) -> tuple[int, int, list[tuple[int, int]]]:
    """Index a normalized .npz file.

    Supports both the current `(N, H, W)` frame layout and the legacy
    `(N, 1, H, W)` layout.
    """
    try:
        with np.load(filepath, mmap_mode="r") as npz:
            frames = npz["frames"]
            if frames.ndim == 3:
                total_t = frames.shape[0]
            elif frames.ndim == 4 and frames.shape[1] == 1:
                total_t = frames.shape[0]
            else:
                print(
                    f"Skipping {filepath}: expected (N, H, W) or (N, 1, H, W), "
                    f"got {tuple(frames.shape)}"
                )
                return file_idx, 0, []

        if total_t < seq_len:
            return file_idx, 0, []
        samples = [(file_idx, t) for t in range(total_t - seq_len + 1)]
        return file_idx, total_t, samples
    except Exception as e:
        print(f"Skipping {filepath} due to error: {e}")
        return file_idx, 0, []


def find_session_files(data_dir: str) -> list[str]:
    return sorted(str(p) for p in Path(data_dir).glob("*.npz"))


def load_palette_info(data_dir: str) -> tuple[int, np.ndarray]:
    palette_path = os.path.join(data_dir, "palette.json")
    if not os.path.isfile(palette_path):
        raise FileNotFoundError(f"No palette.json found in {data_dir}")
    with open(palette_path) as f:
        palette_info = json.load(f)
    return palette_info["num_colors"], np.array(palette_info["colors_rgb"], dtype=np.uint8)


def load_num_actions(data_dir: str) -> int:
    actions_path = os.path.join(data_dir, "actions.json")
    if not os.path.isfile(actions_path):
        return NUM_ACTIONS
    with open(actions_path) as f:
        actions_info = json.load(f)
    return int(actions_info.get("num_actions", NUM_ACTIONS))


def _normalize_frames_chunk(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 3:
        return frames
    if frames.ndim == 4 and frames.shape[1] == 1:
        return frames[:, 0]
    raise ValueError(f"Unexpected frame chunk shape {tuple(frames.shape)}")

class Genie2Dataset(Dataset):
    """Loads sequences of (frames, actions) for Genie 2 training.

    Each sample is a window of ``seq_len`` consecutive frames together with
    the corresponding actions.  Frames are palette indices (uint8), and
    actions are uint8 indices into COMPLEX_MOVEMENT.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 16,
        crop_size: int = CROP_224_SIZE,
        subset_n: int = 0,
        seed: int = 42,
        num_workers: int | None = None,
    ):
        super().__init__()
        if num_workers is None:
            num_workers = get_effective_cpu_count()
        self.seq_len = seq_len
        self.crop_size = crop_size
        self.data_files = find_session_files(data_dir)
        self.num_files = len(self.data_files)
        self.dataset_bytes = sum(Path(path).stat().st_size for path in self.data_files)
        assert self.data_files, f"No session files found in {data_dir}"

        self.samples: list[tuple[int, int]] = []
        frame_counts = [0] * self.num_files

        print(f"Indexing {self.num_files} data files (stride=1)...")
        index_workers = max(int(num_workers), 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=index_workers) as pool:
            futures = {
                pool.submit(_index_file, idx, path, seq_len): idx
                for idx, path in enumerate(self.data_files)
            }
            for future in concurrent.futures.as_completed(futures):
                file_idx, total_t, samples = future.result()
                frame_counts[file_idx] = total_t
                self.samples.extend(samples)

        valid = [i for i in range(self.num_files) if frame_counts[i] > 0]
        if len(valid) < self.num_files:
            old_to_new = {old: new for new, old in enumerate(valid)}
            skipped = self.num_files - len(valid)
            print(f"Dropped {skipped} file(s) with no usable frames")
            self.data_files = [self.data_files[i] for i in valid]
            frame_counts = [frame_counts[i] for i in valid]
            self.samples = [(old_to_new[fi], t) for fi, t in self.samples]
            self.num_files = len(self.data_files)

        self.samples.sort()

        if subset_n > 0 and subset_n < len(self.samples):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(self.samples), size=subset_n, replace=False)
            self.samples = [self.samples[i] for i in indices]
            print(f"Subset: kept {subset_n} samples")

        self.total_frames = sum(frame_counts)
        self.frames_by_file: list[np.ndarray] | None = None
        self.actions_by_file: list[np.ndarray] | None = None

        if self.total_frames > 0:
            available = get_available_memory()
            with np.load(self.data_files[0], mmap_mode="r") as npz:
                probe_frames = _normalize_frames_chunk(npz["frames"][:1])
                probe_actions = np.asarray(npz["actions"][:1])
            bytes_per_frame = probe_frames[0].nbytes
            bytes_per_action = probe_actions[0].nbytes
            total_bytes = self.total_frames * (bytes_per_frame + bytes_per_action)
            headroom = 2 * 2**30

            if available > 0 and total_bytes < available - headroom:
                print(
                    f"Loading frames/actions into RAM ({total_bytes / 2**30:.1f} GB, "
                    f"{available / 2**30:.1f} GB available)..."
                )
                self.frames_by_file = [None] * self.num_files
                self.actions_by_file = [None] * self.num_files

                def _load_file(idx: int) -> tuple[int, np.ndarray, np.ndarray]:
                    with np.load(self.data_files[idx]) as npz:
                        frames = _normalize_frames_chunk(npz["frames"])
                        actions = np.asarray(npz["actions"])
                    return idx, frames, actions

                with concurrent.futures.ThreadPoolExecutor(max_workers=index_workers) as pool:
                    futures = [pool.submit(_load_file, i) for i in range(self.num_files)]
                    for future in concurrent.futures.as_completed(futures):
                        idx, frames, actions = future.result()
                        self.frames_by_file[idx] = frames
                        self.actions_by_file[idx] = actions
            else:
                print(
                    f"Dataset too large for RAM ({total_bytes / 2**30:.1f} GB, "
                    f"{available / 2**30:.1f} GB available). Using mmap."
                )

        print(f"[Genie2Dataset] {len(self.samples)} windows from {len(self.data_files)} files")

    def __len__(self) -> int:
        return len(self.samples)

    def _get_frames_mmap(self, file_idx: int) -> np.ndarray:
        if not hasattr(self, "_mmap_frames_cache"):
            self._mmap_frames_cache = {}
        if file_idx not in self._mmap_frames_cache:
            npz = np.load(self.data_files[file_idx], mmap_mode="r")
            self._mmap_frames_cache[file_idx] = _normalize_frames_chunk(npz["frames"])
        return self._mmap_frames_cache[file_idx]

    def _get_actions_mmap(self, file_idx: int) -> np.ndarray:
        if not hasattr(self, "_mmap_actions_cache"):
            self._mmap_actions_cache = {}
        if file_idx not in self._mmap_actions_cache:
            npz = np.load(self.data_files[file_idx], mmap_mode="r")
            self._mmap_actions_cache[file_idx] = np.asarray(npz["actions"])
        return self._mmap_actions_cache[file_idx]

    def __getitem__(self, idx: int):
        fi, t0 = self.samples[idx]
        if self.frames_by_file is not None and self.actions_by_file is not None:
            frames = self.frames_by_file[fi][t0 : t0 + self.seq_len]
            actions = self.actions_by_file[fi][t0 : t0 + self.seq_len]
        else:
            frames = self._get_frames_mmap(fi)[t0 : t0 + self.seq_len]
            actions = self._get_actions_mmap(fi)[t0 : t0 + self.seq_len]

        # Centre-crop 256 → 224 if needed
        h, w = frames.shape[-2:]
        if (h, w) != (self.crop_size, self.crop_size):
            border = (h - self.crop_size) // 2
            frames = frames[..., border : h - border, border : w - border]

        return torch.from_numpy(frames.copy()).long(), torch.from_numpy(np.asarray(actions).reshape(-1).copy()).long()


# ═══════════════════════════════════════════════════════════════════════════
# Autoencoder (VAE)
# ═══════════════════════════════════════════════════════════════════════════

def _down_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 4, stride=2, padding=1),
        nn.GroupNorm(8, out_c),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.GroupNorm(8, out_c),
        nn.SiLU(inplace=True),
    )


def _up_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1),
        nn.GroupNorm(8, out_c),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.GroupNorm(8, out_c),
        nn.SiLU(inplace=True),
    )


class FrameVAE(nn.Module):
    """Small VAE that compresses a palette-indexed frame to a continuous latent.

    Encoder: 224×224 → 112 → 56 → 28 → 14  (four 2× downsamples)
    Decoder: 14 → 28 → 56 → 112 → 224      (four 2× upsamples)
    """

    def __init__(self, num_palette_colors: int, latent_channels: int = LATENT_CHANNELS):
        super().__init__()
        self.num_palette_colors = num_palette_colors
        self.latent_channels = latent_channels
        ch = 32  # base channel width

        # Encoder: one-hot input (K channels) → latent
        self.encoder = nn.Sequential(
            _down_block(num_palette_colors, ch),      # 224→112
            _down_block(ch, ch * 2),                  # 112→56
            _down_block(ch * 2, ch * 4),              # 56→28
            _down_block(ch * 4, ch * 4),              # 28→14
        )
        self.enc_mu = nn.Conv2d(ch * 4, latent_channels, 1)
        self.enc_logvar = nn.Conv2d(ch * 4, latent_channels, 1)

        # Decoder: latent → logits over palette
        self.decoder = nn.Sequential(
            _up_block(latent_channels, ch * 4),       # 14→28
            _up_block(ch * 4, ch * 4),                # 28→56
            _up_block(ch * 4, ch * 2),                # 56→112
            _up_block(ch * 2, ch),                    # 112→224
        )
        self.dec_out = nn.Conv2d(ch, num_palette_colors, 1)

    def encode(self, x_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one-hot frames → (z, mu, logvar).  z is reparametrised."""
        h = self.encoder(x_onehot)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent → palette logits (B, K, H, W)."""
        return self.dec_out(self.decoder(z))

    def forward(self, frame_indices: torch.Tensor):
        """
        Args:
            frame_indices: (B, H, W) long – palette indices.
        Returns:
            logits: (B, K, H, W)
            mu, logvar: for KL
        """
        x_oh = F.one_hot(frame_indices, self.num_palette_colors).float()  # (B,H,W,K)
        x_oh = x_oh.permute(0, 3, 1, 2)                                  # (B,K,H,W)
        z, mu, logvar = self.encode(x_oh)
        logits = self.decode(z)
        return logits, mu, logvar


# ═══════════════════════════════════════════════════════════════════════════
# Diffusion helpers (DDPM)
# ═══════════════════════════════════════════════════════════════════════════

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from 'Improved DDPM'."""
    t = torch.linspace(0, timesteps, timesteps + 1)
    f = torch.cos((t / timesteps + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(max=0.999)


class DiffusionSchedule:
    """Pre-computes and stores all DDPM schedule quantities."""

    def __init__(self, timesteps: int = DIFFUSION_STEPS, device: torch.device | str = "cpu"):
        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.timesteps = timesteps
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()
        self.sqrt_recip_alphas = (1.0 / alphas).sqrt()
        self.posterior_variance = betas * (1.0 - torch.cat([alphas_cumprod[:1], alphas_cumprod[:-1]])) / (1.0 - alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t]
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t]
        # Broadcast to spatial dimensions
        while sqrt_ac.dim() < x0.dim():
            sqrt_ac = sqrt_ac.unsqueeze(-1)
            sqrt_omac = sqrt_omac.unsqueeze(-1)
        return sqrt_ac * x0 + sqrt_omac * noise, noise

    @torch.no_grad()
    def ddpm_sample(self, denoiser: nn.Module, context: torch.Tensor,
                    shape: tuple, device: torch.device) -> torch.Tensor:
        """Full DDPM reverse process: sample z_0 from pure noise."""
        z = torch.randn(shape, device=device)
        for t_idx in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            eps_pred = denoiser(z, t_batch, context)
            beta = self.betas[t_idx]
            sqrt_recip_alpha = self.sqrt_recip_alphas[t_idx]
            sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t_idx]
            z = sqrt_recip_alpha * (z - beta / sqrt_omac * eps_pred)
            if t_idx > 0:
                z += self.posterior_variance[t_idx].sqrt() * torch.randn_like(z)
        return z

    def to(self, device: torch.device):
        for attr in ["betas", "alphas", "alphas_cumprod", "sqrt_alphas_cumprod",
                      "sqrt_one_minus_alphas_cumprod", "sqrt_recip_alphas", "posterior_variance"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


# ═══════════════════════════════════════════════════════════════════════════
# Denoiser network (small U-Net on 14×14 latent)
# ═══════════════════════════════════════════════════════════════════════════

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10_000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class LatentDenoiser(nn.Module):
    """Predicts noise ε given noisy latent z_t, diffusion timestep, and
    conditioning context from the dynamics transformer.

    Operates on 14×14 spatial latents.  Tiny U-Net: down 14→7, up 7→14.
    """

    def __init__(
        self,
        latent_channels: int = LATENT_CHANNELS,
        cond_dim: int = TRANSFORMER_DIM,
        base_ch: int = 64,
    ):
        super().__init__()
        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, base_ch * 2),
            nn.SiLU(),
            nn.Linear(base_ch * 2, base_ch * 2),
        )
        # Context projection (transformer output → spatial feature map)
        self.cond_proj = nn.Linear(cond_dim, base_ch * 2)

        ch = base_ch
        # Encoder
        self.conv_in = nn.Conv2d(latent_channels, ch, 3, padding=1)
        self.down1 = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),    # 14→7
            nn.GroupNorm(8, ch * 2), nn.SiLU(),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
            nn.GroupNorm(8, ch * 2), nn.SiLU(),
        )
        # Bottleneck
        self.mid = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
            nn.GroupNorm(8, ch * 2), nn.SiLU(),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
            nn.GroupNorm(8, ch * 2), nn.SiLU(),
        )
        # Decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),  # 7→14
            nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.SiLU(),
        )
        self.conv_out = nn.Conv2d(ch, latent_channels, 1)

    def forward(
        self,
        z_noisy: torch.Tensor,        # (B, C, 14, 14)
        t: torch.Tensor,              # (B,) int timesteps
        context: torch.Tensor,        # (B, cond_dim)
    ) -> torch.Tensor:
        t_emb = self.time_mlp(t)                          # (B, ch*2)
        c_emb = self.cond_proj(context)                   # (B, ch*2)
        cond = (t_emb + c_emb).unsqueeze(-1).unsqueeze(-1)  # (B, ch*2, 1, 1)

        h = self.conv_in(z_noisy)         # (B, ch, 14, 14)
        h1 = self.down1(h)                # (B, ch*2, 7, 7)
        h1 = h1 + cond                    # broadcast conditioning
        h1 = self.mid(h1)
        h = self.up1(h1) + h              # skip connection
        return self.conv_out(h)


# ═══════════════════════════════════════════════════════════════════════════
# Dynamics Transformer
# ═══════════════════════════════════════════════════════════════════════════

class DynamicsTransformer(nn.Module):
    """Causal transformer over a sequence of latent frames + actions.

    Each frame's latent grid (C, 14, 14) is patchified into spatial tokens,
    interleaved with action embeddings, and processed with causal masking.
    The output for the last frame position is used to condition the diffusion
    denoiser for predicting the next frame.
    """

    def __init__(
        self,
        latent_channels: int = LATENT_CHANNELS,
        latent_size: int = LATENT_SIZE,
        dim: int = TRANSFORMER_DIM,
        num_heads: int = TRANSFORMER_HEADS,
        num_layers: int = TRANSFORMER_LAYERS,
        mlp_ratio: int = TRANSFORMER_MLP_RATIO,
        num_actions: int = NUM_ACTIONS,
        max_frames: int = 16,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.dim = dim
        num_spatial = latent_size * latent_size          # 196 tokens per frame
        self.num_spatial = num_spatial

        # Project flattened latent patch → token
        self.latent_proj = nn.Linear(latent_channels, dim)
        self.action_embed = nn.Embedding(num_actions, dim)

        # Positional embeddings: frame-level + spatial
        self.frame_pos = nn.Embedding(max_frames, dim)
        self.spatial_pos = nn.Embedding(num_spatial, dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

        # Output: pool spatial tokens of the last frame into a single context vector
        self.context_pool = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        latents: torch.Tensor,          # (B, T, C, H, W) continuous latents
        actions: torch.Tensor,           # (B, T) action indices
    ) -> torch.Tensor:
        """Returns context vector (B, dim) summarising the sequence for
        conditioning the diffusion model on the next frame."""
        B, T, C, H, W = latents.shape
        device = latents.device
        S = self.num_spatial  # H * W = 196

        # Flatten each frame's latent grid into spatial tokens
        # (B, T, C, H, W) → (B, T, H*W, C) → project → (B, T, S, dim)
        tokens = latents.reshape(B, T, C, S).permute(0, 1, 3, 2)  # (B,T,S,C)
        tokens = self.latent_proj(tokens)                          # (B,T,S,dim)

        # Add positional embeddings
        frame_ids = torch.arange(T, device=device)
        spatial_ids = torch.arange(S, device=device)
        tokens = tokens + self.frame_pos(frame_ids)[None, :, None, :]    # frame pos
        tokens = tokens + self.spatial_pos(spatial_ids)[None, None, :, :]  # spatial pos

        # Interleave action tokens: action_t is placed before frame_t's tokens
        # Sequence per frame: [action_t, spatial_0, ..., spatial_{S-1}]
        act_tokens = self.action_embed(actions)                     # (B, T, dim)
        act_tokens = act_tokens + self.frame_pos(frame_ids)[None, :, :]

        # Build full sequence: (B, T*(1+S), dim)
        frame_seqs = []
        for t_idx in range(T):
            frame_seqs.append(act_tokens[:, t_idx : t_idx + 1, :])  # (B, 1, dim)
            frame_seqs.append(tokens[:, t_idx, :, :])               # (B, S, dim)
        seq = torch.cat(frame_seqs, dim=1)  # (B, T*(1+S), dim)

        # Causal mask: frame t can attend to frames ≤ t (block-causal)
        tokens_per_frame = 1 + S  # action + spatial tokens
        total = T * tokens_per_frame
        mask = torch.ones(total, total, device=device, dtype=torch.bool)
        for t_idx in range(T):
            start = t_idx * tokens_per_frame
            end = (t_idx + 1) * tokens_per_frame
            # Frame t can attend to all tokens from frames 0..t
            mask[start:end, : end] = False
        # mask=True means "do not attend"

        seq = self.transformer(seq, mask=mask, is_causal=False)
        seq = self.norm(seq)

        # Extract the last frame's spatial tokens and pool → context
        last_start = (T - 1) * tokens_per_frame + 1  # skip last action token
        last_end = T * tokens_per_frame
        last_tokens = seq[:, last_start:last_end, :]  # (B, S, dim)
        context = self.context_pool(last_tokens.mean(dim=1))  # (B, dim)
        return context


# ═══════════════════════════════════════════════════════════════════════════
# Training – Phase 1: Autoencoder
# ═══════════════════════════════════════════════════════════════════════════

def train_autoencoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = Genie2Dataset(
        args.data_dir,
        seq_len=1,
        crop_size=CROP_224_SIZE,
        subset_n=args.overfit_n,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    if args.overfit_n > 0:
        print(f"[overfit] Using {len(dataset)} samples")
    n_eval = min(500, max(1, len(dataset) // 10))
    train_ds, eval_ds = random_split(dataset, [len(dataset) - n_eval, n_eval])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)

    num_colors, palette = load_palette_info(args.data_dir)
    print(f"Palette: {num_colors} colours")

    model = FrameVAE(num_palette_colors=num_colors, latent_channels=LATENT_CHANNELS).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"FrameVAE: {num_params:,} parameters")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    warmup = LinearLR(optimizer, start_factor=1e-6, total_iters=500)
    cosine = CosineAnnealingLR(optimizer, T_max=max(args.epochs * len(train_loader) - 500, 1),
                               eta_min=args.lr * 0.1)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[500])

    os.makedirs(args.output_dir, exist_ok=True)
    best_eval = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_recon = total_kl = 0.0
        n_batches = 0
        train_preview_frames = None
        pbar = tqdm(train_loader, desc=f"AE epoch {epoch}/{args.epochs}")
        step = 0
        for frames_batch, _ in pbar:
            if args.max_steps > 0 and step >= args.max_steps:
                break
            frames = frames_batch[:, 0].to(device)  # (B, H, W), seq_len=1 so squeeze
            if train_preview_frames is None:
                train_preview_frames = frames[: min(8, frames.shape[0])].detach().clone()
            logits, mu, logvar = model(frames)

            recon_loss = F.cross_entropy(logits, frames)
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / frames.shape[0]
            loss = recon_loss + KL_WEIGHT * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1
            step += 1
            pbar.set_postfix(recon=f"{recon_loss.item():.4f}", kl=f"{kl_loss.item():.2f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # Eval
        model.eval()
        eval_recon = eval_correct = eval_total = 0
        preview_frames = None
        with torch.no_grad():
            for frames_batch, _ in eval_loader:
                frames = frames_batch[:, 0].to(device)
                logits, mu, logvar = model(frames)
                if preview_frames is None:
                    preview_frames = frames[: min(8, frames.shape[0])].clone()
                eval_recon += F.cross_entropy(logits, frames, reduction="sum").item()
                eval_correct += (logits.argmax(1) == frames).sum().item()
                eval_total += frames.numel()
        eval_recon /= eval_total
        eval_acc = eval_correct / eval_total
        print(f"  eval recon={eval_recon:.6f}  pixel_acc={eval_acc:.4f}  "
              f"train_recon={total_recon / n_batches:.4f}  train_kl={total_kl / n_batches:.2f}")

        preview_source_frames = train_preview_frames if args.overfit_n > 0 else preview_frames
        if preview_source_frames is not None:
            save_ae_eval_image(model, preview_source_frames, palette, args.output_dir, epoch)

        if eval_recon < best_eval:
            best_eval = eval_recon
            torch.save(model.state_dict(), os.path.join(args.output_dir, "ae_best.pt"))
            print(f"  ✓ saved ae_best.pt (eval_recon={eval_recon:.6f})")
        torch.save(model.state_dict(), os.path.join(args.output_dir, "ae_latest.pt"))


# ═══════════════════════════════════════════════════════════════════════════
# Training – Phase 2: Dynamics (Transformer + Diffusion)
# ═══════════════════════════════════════════════════════════════════════════

def train_dynamics(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seq_len = 8  # context frames for dynamics training

    dataset = Genie2Dataset(
        args.data_dir,
        seq_len=seq_len + 1,
        crop_size=CROP_224_SIZE,
        subset_n=args.overfit_n,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    if args.overfit_n > 0:
        print(f"[overfit] Using {len(dataset)} samples")
    n_eval = min(500, max(1, len(dataset) // 10))
    train_ds, eval_ds = random_split(dataset, [len(dataset) - n_eval, n_eval])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)

    num_colors, palette = load_palette_info(args.data_dir)
    num_actions = load_num_actions(args.data_dir)

    # Load frozen autoencoder
    ae = FrameVAE(num_palette_colors=num_colors, latent_channels=LATENT_CHANNELS).to(device)
    ae.load_state_dict(torch.load(args.ae_checkpoint, map_location=device, weights_only=True))
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    print(f"Loaded frozen AE from {args.ae_checkpoint}")

    # Models
    dynamics = DynamicsTransformer(
        latent_channels=LATENT_CHANNELS,
        latent_size=LATENT_SIZE,
        dim=TRANSFORMER_DIM,
        num_heads=TRANSFORMER_HEADS,
        num_layers=TRANSFORMER_LAYERS,
        num_actions=num_actions,
        max_frames=seq_len + 1,
    ).to(device)
    denoiser = LatentDenoiser(
        latent_channels=LATENT_CHANNELS,
        cond_dim=TRANSFORMER_DIM,
    ).to(device)

    d_params = sum(p.numel() for p in dynamics.parameters())
    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"DynamicsTransformer: {d_params:,}  LatentDenoiser: {n_params:,}  "
          f"Total: {d_params + n_params:,}")

    schedule = DiffusionSchedule(DIFFUSION_STEPS, device)

    all_params = list(dynamics.parameters()) + list(denoiser.parameters())
    optimizer = AdamW(all_params, lr=args.lr)
    warmup = LinearLR(optimizer, start_factor=1e-6, total_iters=500)
    cosine = CosineAnnealingLR(optimizer, T_max=max(args.epochs * len(train_loader) - 500, 1),
                               eta_min=args.lr * 0.1)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[500])

    os.makedirs(args.output_dir, exist_ok=True)
    best_eval = float("inf")

    @torch.no_grad()
    def encode_frames(frames_seq: torch.Tensor) -> torch.Tensor:
        """Encode a batch of frame sequences through frozen AE.

        Args:
            frames_seq: (B, T, H, W) long palette indices
        Returns:
            latents: (B, T, C, Hl, Wl) float latent means (no sampling during dynamics training)
        """
        B, T, H, W = frames_seq.shape
        flat = frames_seq.reshape(B * T, H, W)
        x_oh = F.one_hot(flat, num_colors).float().permute(0, 3, 1, 2)
        h = ae.encoder(x_oh)
        mu = ae.enc_mu(h)  # deterministic: use mean
        return mu.reshape(B, T, *mu.shape[1:])

    for epoch in range(1, args.epochs + 1):
        dynamics.train()
        denoiser.train()
        total_loss = 0.0
        n_batches = 0
        train_preview_batch = None
        pbar = tqdm(train_loader, desc=f"Dyn epoch {epoch}/{args.epochs}")

        step = 0
        for frames_batch, actions_batch in pbar:
            if args.max_steps > 0 and step >= args.max_steps:
                break
            frames_batch = frames_batch.to(device)    # (B, T+1, H, W)
            actions_batch = actions_batch.to(device)   # (B, T+1)
            if train_preview_batch is None:
                train_preview_batch = (
                    frames_batch[: min(4, frames_batch.shape[0])].detach().clone(),
                    actions_batch[: min(4, actions_batch.shape[0])].detach().clone(),
                )

            # Encode all frames
            latents = encode_frames(frames_batch)      # (B, T+1, C, Hl, Wl)

            # Context: first T frames and their actions
            ctx_latents = latents[:, :-1]              # (B, T, C, Hl, Wl)
            ctx_actions = actions_batch[:, :-1]        # (B, T)
            target_latent = latents[:, -1]             # (B, C, Hl, Wl)  next frame

            # Dynamics transformer → context vector
            context = dynamics(ctx_latents, ctx_actions)  # (B, dim)

            # Diffusion: sample random timestep, add noise, predict noise
            t = torch.randint(0, DIFFUSION_STEPS, (target_latent.shape[0],), device=device)
            z_noisy, noise = schedule.q_sample(target_latent, t)
            noise_pred = denoiser(z_noisy, t, context)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # Eval
        dynamics.eval()
        denoiser.eval()
        eval_loss = 0.0
        eval_n = 0
        preview_batch = None
        with torch.no_grad():
            for frames_batch, actions_batch in eval_loader:
                frames_batch = frames_batch.to(device)
                actions_batch = actions_batch.to(device)
                if preview_batch is None:
                    preview_batch = (
                        frames_batch[: min(4, frames_batch.shape[0])].clone(),
                        actions_batch[: min(4, actions_batch.shape[0])].clone(),
                    )
                latents = encode_frames(frames_batch)
                ctx_latents = latents[:, :-1]
                ctx_actions = actions_batch[:, :-1]
                target_latent = latents[:, -1]
                context = dynamics(ctx_latents, ctx_actions)
                t = torch.randint(0, DIFFUSION_STEPS, (target_latent.shape[0],), device=device)
                z_noisy, noise = schedule.q_sample(target_latent, t)
                noise_pred = denoiser(z_noisy, t, context)
                eval_loss += F.mse_loss(noise_pred, noise, reduction="sum").item()
                eval_n += noise.numel()
        eval_loss /= max(eval_n, 1)
        print(f"  eval_loss={eval_loss:.6f}  train_loss={total_loss / n_batches:.6f}")

        preview_source_batch = train_preview_batch if args.overfit_n > 0 else preview_batch
        if preview_source_batch is not None:
            preview_frames, preview_actions = preview_source_batch
            save_dynamics_eval_image(
                ae,
                dynamics,
                denoiser,
                schedule,
                preview_frames,
                preview_actions,
                palette,
                args.output_dir,
                epoch,
                num_colors,
            )

        state = {
            "dynamics": dynamics.state_dict(),
            "denoiser": denoiser.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        if eval_loss < best_eval:
            best_eval = eval_loss
            torch.save(state, os.path.join(args.output_dir, "dynamics_best.pt"))
            print(f"  ✓ saved dynamics_best.pt (eval_loss={eval_loss:.6f})")
        torch.save(state, os.path.join(args.output_dir, "dynamics_latest.pt"))


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def palette_to_rgb(indices: torch.Tensor, palette: np.ndarray) -> np.ndarray:
    """Convert (H, W) palette indices to (H, W, 3) uint8 RGB."""
    return palette[indices.cpu().numpy()].astype(np.uint8)


def _save_eval_grid(rows: list[np.ndarray], output_dir: str, stem: str, epoch: int):
    from PIL import Image

    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    if not rows:
        return

    h_sep = np.full((2, rows[0].shape[1], 3), 255, dtype=np.uint8)
    parts = []
    for i, row in enumerate(rows):
        if i > 0:
            parts.append(h_sep)
        parts.append(row)
    grid = np.concatenate(parts, axis=0)

    epoch_path = os.path.join(eval_dir, f"{stem}_epoch_{epoch:03d}.png")
    latest_path = os.path.join(eval_dir, f"{stem}_latest.png")
    Image.fromarray(grid).save(epoch_path)
    Image.fromarray(grid).save(latest_path)
    print(f"  saved {epoch_path}")


@torch.no_grad()
def save_ae_eval_image(
    model: FrameVAE,
    frames: torch.Tensor,
    palette: np.ndarray,
    output_dir: str,
    epoch: int,
):
    logits, _, _ = model(frames)
    pred_indices = logits.argmax(1)

    rows = []
    for i in range(frames.shape[0]):
        gt_rgb = palette_to_rgb(frames[i], palette)
        pred_rgb = palette_to_rgb(pred_indices[i], palette)
        sep = np.full((gt_rgb.shape[0], 2, 3), 255, dtype=np.uint8)
        rows.append(np.concatenate([gt_rgb, sep, pred_rgb], axis=1))

    _save_eval_grid(rows, output_dir, "ae_eval", epoch)


@torch.no_grad()
def save_dynamics_eval_image(
    ae: FrameVAE,
    dynamics: DynamicsTransformer,
    denoiser: LatentDenoiser,
    schedule: DiffusionSchedule,
    frames_batch: torch.Tensor,
    actions_batch: torch.Tensor,
    palette: np.ndarray,
    output_dir: str,
    epoch: int,
    num_colors: int,
):
    ctx_frames = frames_batch[:, :-1]
    gt_next = frames_batch[:, -1]
    last_ctx = frames_batch[:, -2]

    bsz, steps, height, width = ctx_frames.shape
    flat = ctx_frames.reshape(bsz * steps, height, width)
    x_oh = F.one_hot(flat, num_colors).float().permute(0, 3, 1, 2)
    h = ae.encoder(x_oh)
    mu = ae.enc_mu(h)
    ctx_latents = mu.reshape(bsz, steps, *mu.shape[1:])
    ctx_actions = actions_batch[:, :-1]

    context = dynamics(ctx_latents, ctx_actions)
    latent_shape = (bsz, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    pred_latent = schedule.ddpm_sample(denoiser, context, latent_shape, frames_batch.device)
    pred_logits = ae.decode(pred_latent)
    pred_indices = pred_logits.argmax(1)

    rows = []
    for i in range(frames_batch.shape[0]):
        ctx_rgb = palette_to_rgb(last_ctx[i], palette)
        gt_rgb = palette_to_rgb(gt_next[i], palette)
        pred_rgb = palette_to_rgb(pred_indices[i], palette)
        sep = np.full((ctx_rgb.shape[0], 2, 3), 255, dtype=np.uint8)
        rows.append(np.concatenate([ctx_rgb, sep, gt_rgb, sep, pred_rgb], axis=1))

    _save_eval_grid(rows, output_dir, "dynamics_eval", epoch)


def visualize(args):
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_colors, palette = load_palette_info(args.data_dir)
    num_actions = load_num_actions(args.data_dir)

    # Load AE
    ae = FrameVAE(num_palette_colors=num_colors, latent_channels=LATENT_CHANNELS).to(device)
    ae_ckpt = args.ae_checkpoint or os.path.join(args.output_dir, "ae_best.pt")
    ae.load_state_dict(torch.load(ae_ckpt, map_location=device, weights_only=True))
    ae.eval()
    print(f"Loaded AE from {ae_ckpt}")

    # Load dataset
    has_dynamics = args.dyn_checkpoint is not None or os.path.exists(
        os.path.join(args.output_dir, "dynamics_best.pt"))
    seq_len = 9 if has_dynamics else 1
    dataset = Genie2Dataset(
        args.data_dir,
        seq_len=seq_len,
        crop_size=CROP_224_SIZE,
        subset_n=args.overfit_n,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    if args.overfit_n > 0:
        print(f"[overfit] Using {len(dataset)} samples")

    n_samples = min(args.vis_n, len(dataset))
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=False, num_workers=0)
    frames_batch, actions_batch = next(iter(loader))
    frames_batch = frames_batch.to(device)
    actions_batch = actions_batch.to(device)

    out_dir = os.path.join(args.output_dir, "vis")
    os.makedirs(out_dir, exist_ok=True)

    # --- AE reconstructions ---
    print("Generating AE reconstructions...")
    with torch.no_grad():
        single_frames = frames_batch[:, 0]  # (N, H, W)
        logits, _, _ = ae(single_frames)
        pred_indices = logits.argmax(1)  # (N, H, W)

    rows = []
    for i in range(n_samples):
        gt_rgb = palette_to_rgb(single_frames[i], palette)
        pred_rgb = palette_to_rgb(pred_indices[i], palette)
        # 2px white separator
        sep = np.full((gt_rgb.shape[0], 2, 3), 255, dtype=np.uint8)
        rows.append(np.concatenate([gt_rgb, sep, pred_rgb], axis=1))
    # Stack vertically with separators
    h_sep = np.full((2, rows[0].shape[1], 3), 255, dtype=np.uint8)
    parts = []
    for i, r in enumerate(rows):
        if i > 0:
            parts.append(h_sep)
        parts.append(r)
    ae_grid = np.concatenate(parts, axis=0)
    ae_path = os.path.join(out_dir, "ae_recon.png")
    Image.fromarray(ae_grid).save(ae_path)
    print(f"  Saved {ae_path}  (left=GT, right=recon)")

    # --- Dynamics predictions ---
    if not has_dynamics:
        print("No dynamics checkpoint found, skipping dynamics visualization.")
        return

    dyn_ckpt = args.dyn_checkpoint or os.path.join(args.output_dir, "dynamics_best.pt")
    dynamics = DynamicsTransformer(
        latent_channels=LATENT_CHANNELS, latent_size=LATENT_SIZE,
        dim=TRANSFORMER_DIM, num_heads=TRANSFORMER_HEADS,
        num_layers=TRANSFORMER_LAYERS, num_actions=num_actions,
        max_frames=seq_len,
    ).to(device)
    denoiser = LatentDenoiser(
        latent_channels=LATENT_CHANNELS, cond_dim=TRANSFORMER_DIM,
    ).to(device)
    state = torch.load(dyn_ckpt, map_location=device, weights_only=True)
    dynamics.load_state_dict(state["dynamics"])
    denoiser.load_state_dict(state["denoiser"])
    dynamics.eval()
    denoiser.eval()
    print(f"Loaded dynamics from {dyn_ckpt}")

    schedule = DiffusionSchedule(DIFFUSION_STEPS, device)

    print("Generating dynamics predictions (DDPM sampling)...")
    with torch.no_grad():
        # Encode context frames
        ctx_frames = frames_batch[:, :-1]  # (N, T, H, W)
        B, T, H, W = ctx_frames.shape
        flat = ctx_frames.reshape(B * T, H, W)
        x_oh = F.one_hot(flat, num_colors).float().permute(0, 3, 1, 2)
        h = ae.encoder(x_oh)
        mu = ae.enc_mu(h)
        ctx_latents = mu.reshape(B, T, *mu.shape[1:])
        ctx_actions = actions_batch[:, :-1]

        # Get dynamics context vector
        context = dynamics(ctx_latents, ctx_actions)

        # DDPM reverse sample
        latent_shape = (B, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
        pred_latent = schedule.ddpm_sample(denoiser, context, latent_shape, device)

        # Decode predicted latent → palette indices
        pred_logits = ae.decode(pred_latent)
        pred_indices = pred_logits.argmax(1)  # (N, H, W)

        # Ground truth next frame
        gt_next = frames_batch[:, -1]  # (N, H, W)

        # Also show last context frame for reference
        last_ctx = frames_batch[:, -2]  # (N, H, W)

    rows = []
    for i in range(n_samples):
        ctx_rgb = palette_to_rgb(last_ctx[i], palette)
        gt_rgb = palette_to_rgb(gt_next[i], palette)
        pred_rgb = palette_to_rgb(pred_indices[i], palette)
        sep = np.full((ctx_rgb.shape[0], 2, 3), 255, dtype=np.uint8)
        rows.append(np.concatenate([ctx_rgb, sep, gt_rgb, sep, pred_rgb], axis=1))
    h_sep = np.full((2, rows[0].shape[1], 3), 255, dtype=np.uint8)
    parts = []
    for i, r in enumerate(rows):
        if i > 0:
            parts.append(h_sep)
        parts.append(r)
    dyn_grid = np.concatenate(parts, axis=0)
    dyn_path = os.path.join(out_dir, "dynamics_pred.png")
    Image.fromarray(dyn_grid).save(dyn_path)
    print(f"  Saved {dyn_path}  (left=last_ctx, mid=GT_next, right=predicted)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genie 2 – small latent diffusion world model")
    parser.add_argument("--phase", choices=["autoencoder", "dynamics", "visualize"], required=True)
    parser.add_argument("--data-dir", type=str, default="data/normalized")
    parser.add_argument("--output-dir", type=str, default="checkpoints/genie2")
    parser.add_argument("--ae-checkpoint", type=str, default=None,
                        help="Path to trained AE checkpoint (required for dynamics phase)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=min(get_effective_cpu_count(), 32))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit-n", type=int, default=0,
                        help="Train on N random samples (overfit sanity check)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after this many gradient steps per epoch (0 = no limit)")
    parser.add_argument("--dyn-checkpoint", type=str, default=None,
                        help="Path to dynamics checkpoint (for visualize phase)")
    parser.add_argument("--vis-n", type=int, default=8,
                        help="Number of samples to visualize")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.phase == "autoencoder":
        train_autoencoder(args)
    elif args.phase == "dynamics":
        if not args.ae_checkpoint:
            raise SystemExit("--ae-checkpoint is required for dynamics phase")
        train_dynamics(args)
    elif args.phase == "visualize":
        visualize(args)


if __name__ == "__main__":
    main()
