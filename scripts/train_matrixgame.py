"""
Matrix-Game 3.0 – Error-Aware Interactive World Model with Long-Horizon Memory

Implements a ~20M parameter version of the Matrix-Game 3.0 framework for
224×224 NES palette-indexed (23-colour) frames.

Architecture:
  1. Frame VAE (frozen, from train_genie2.py phase 1): compresses 224×224
     palette frames into continuous (4, 14, 14) latents.
  2. Bidirectional Diffusion Transformer (DiT): jointly attends over
     memory frames, past frames, and noised current frames using
     flow matching. Actions are injected via cross-attention (keyboard).
  3. Error buffer: collects prediction residuals and reinjects them into
     past/memory conditioning for self-correcting training.

Key ideas from Matrix-Game 3.0:
  - Unified bidirectional architecture (no causal masking)
  - Flow matching objective on current frames only
  - Error collection/injection for robustness to imperfect contexts
  - Camera-aware memory (simplified to temporal stride for NES)
  - Joint self-attention over memory + past + current tokens

Usage:
  # Train (requires frozen AE from train_genie2.py phase 1)
  python scripts/train_matrixgame.py \\
      --ae-checkpoint checkpoints/genie2/ae_best.pt \\
      --data-dir data/normalized \\
      --max-steps 100000

  # Resume training
  python scripts/train_matrixgame.py \\
      --ae-checkpoint checkpoints/genie2/ae_best.pt \\
      --resume-from checkpoints/matrixgame/training_state_latest.pt

  # Visualize autoregressive rollout
  python scripts/train_matrixgame.py --phase visualize \\
      --ae-checkpoint checkpoints/genie2/ae_best.pt \\
      --checkpoint checkpoints/matrixgame/dit_best.pt
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
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
# Hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
NUM_PALETTE_COLORS = 23
NUM_ACTIONS = 42

# VAE latent space (matches FrameVAE from train_genie2.py)
LATENT_CHANNELS = 4
LATENT_SIZE = 14  # 224 / 16

# Sequence structure
TOTAL_FRAMES = 16           # full sequence length
PAST_FRAMES = 4             # k: number of conditioning past frames
CURRENT_FRAMES = 4          # N-k: frames to predict
MEMORY_FRAMES = 2           # r: number of retrieved memory frames

# DiT architecture (~20M params)
DIT_DIM = 384
DIT_HEADS = 6
DIT_LAYERS = 6
DIT_MLP_RATIO = 4
DIT_PATCH_SIZE = 1          # each latent spatial position is a token

# Flow matching
FLOW_SIGMA_MIN = 1e-4       # minimum noise level

# Error buffer
ERROR_BUFFER_SIZE = 4096
ERROR_GAMMA_HISTORY = 0.3   # perturbation magnitude for past frames
ERROR_GAMMA_MEMORY = 0.2    # perturbation magnitude for memory frames

# Training
DEFAULT_LR = 3e-4
DEFAULT_BATCH_SIZE = 16
DEFAULT_WARMUP_STEPS = 1000
GRAD_CLIP = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Frame VAE (imported from train_genie2.py, kept minimal for loading)
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


class FrameVAE(nn.Module):
    """Frozen VAE encoder (architecture must match train_genie2.py)."""

    def __init__(self, num_palette_colors: int, latent_channels: int = LATENT_CHANNELS):
        super().__init__()
        self.num_palette_colors = num_palette_colors
        ch = 32
        self.encoder = nn.Sequential(
            _down_block(num_palette_colors, ch),
            _down_block(ch, ch * 2),
            _down_block(ch * 2, ch * 4),
            _down_block(ch * 4, ch * 4),
        )
        self.enc_mu = nn.Conv2d(ch * 4, latent_channels, 1)

    @torch.no_grad()
    def encode(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """Encode palette indices → latent mean (deterministic)."""
        x_oh = F.one_hot(frame_indices, self.num_palette_colors).float()
        x_oh = x_oh.permute(0, 3, 1, 2)
        h = self.encoder(x_oh)
        return self.enc_mu(h)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

def find_session_files(data_dir: str) -> list[str]:
    """Find all normalized .npz files in a directory."""
    p = Path(data_dir)
    files = sorted(str(f) for f in p.glob("*.npz"))
    return files


class MatrixGameDataset(Dataset):
    """Loads sequences of (frames, actions) with memory retrieval.

    Each sample provides:
      - memory_frames: (M, H, W) frames from earlier in the session
      - past_frames: (K, H, W) immediately preceding frames
      - current_frames: (C, H, W) frames to predict
      - actions: (K+C,) actions aligned with past+current frames
    """

    def __init__(
        self,
        data_dir: str,
        past_k: int = PAST_FRAMES,
        current_c: int = CURRENT_FRAMES,
        memory_r: int = MEMORY_FRAMES,
        memory_stride: int = 16,
    ):
        super().__init__()
        self.past_k = past_k
        self.current_c = current_c
        self.memory_r = memory_r
        self.memory_stride = memory_stride
        self.total_needed = past_k + current_c

        self.data_files = find_session_files(data_dir)
        assert self.data_files, f"No .npz files found in {data_dir}"

        # We need enough frames before the window for memory retrieval
        # Minimum: memory_r * memory_stride frames before the past window
        self.min_start = memory_r * memory_stride

        self.samples: list[tuple[int, int]] = []  # (file_idx, window_start)
        for fi, fp in enumerate(self.data_files):
            try:
                with np.load(fp, mmap_mode="r") as z:
                    total_t = z["frames"].shape[0]
                # window_start is the index of the first past frame
                # We need min_start frames before it for memory
                earliest = self.min_start
                latest = total_t - self.total_needed
                if latest >= earliest:
                    for t in range(earliest, latest + 1):
                        self.samples.append((fi, t))
            except Exception as e:
                print(f"Skipping {fp}: {e}")

        self.samples.sort()
        print(f"[MatrixGameDataset] {len(self.samples)} windows from "
              f"{len(self.data_files)} files (K={past_k}, C={current_c}, M={memory_r})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fi, t_start = self.samples[idx]
        with np.load(self.data_files[fi], mmap_mode="r") as z:
            frames_all = z["frames"]
            actions_all = z["actions"]

            # Memory: sample frames from before the window using stride
            memory_indices = []
            for i in range(self.memory_r, 0, -1):
                mem_idx = t_start - i * self.memory_stride
                memory_indices.append(max(0, mem_idx))

            memory_frames = frames_all[memory_indices]  # (M, H, W)

            # Past + current frames
            t_end = t_start + self.total_needed
            window_frames = frames_all[t_start:t_end]  # (K+C, H, W)
            window_actions = actions_all[t_start:t_end]  # (K+C,)

        return (
            torch.from_numpy(memory_frames.copy()).long(),
            torch.from_numpy(window_frames[:self.past_k].copy()).long(),
            torch.from_numpy(window_frames[self.past_k:].copy()).long(),
            torch.from_numpy(window_actions.copy()).long(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Sinusoidal position embeddings and timestep embedding
# ═══════════════════════════════════════════════════════════════════════════

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10_000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Layer Norm (adaLN-Zero) for DiT
# ═══════════════════════════════════════════════════════════════════════════

class AdaLNModulation(nn.Module):
    """Produces scale, shift, gate parameters from a conditioning vector."""

    def __init__(self, dim: int, n_modulations: int = 6):
        super().__init__()
        self.linear = nn.Linear(dim, n_modulations * dim)
        self.n_modulations = n_modulations
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c: torch.Tensor) -> list[torch.Tensor]:
        out = self.linear(c)
        return out.chunk(self.n_modulations, dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ═══════════════════════════════════════════════════════════════════════════
# DiT Block with Action Cross-Attention
# ═══════════════════════════════════════════════════════════════════════════

class DiTBlock(nn.Module):
    """Single Diffusion Transformer block with:
    - Self-attention over all tokens (memory + past + current)
    - Cross-attention for action conditioning
    - Feed-forward network
    - adaLN-Zero conditioning from timestep
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # adaLN modulation (6 params: shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN = AdaLNModulation(dim, n_modulations=6)

        # Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Cross-attention for actions
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_cross_kv = nn.LayerNorm(dim)

        # FFN
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(
        self,
        x: torch.Tensor,              # (B, N, dim) all tokens
        c: torch.Tensor,              # (B, dim) timestep conditioning
        action_tokens: torch.Tensor,  # (B, A, dim) action embeddings
    ) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c)

        # Self-attention with adaLN
        x_norm = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate1.unsqueeze(1) * attn_out

        # Cross-attention with action tokens
        x_cross = self.norm_cross(x)
        action_kv = self.norm_cross_kv(action_tokens)
        cross_out, _ = self.cross_attn(x_cross, action_kv, action_kv)
        x = x + cross_out

        # FFN with adaLN
        x_norm = modulate(self.norm2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.ffn(x_norm)

        return x


# ═══════════════════════════════════════════════════════════════════════════
# Bidirectional Diffusion Transformer
# ═══════════════════════════════════════════════════════════════════════════

class MatrixGameDiT(nn.Module):
    """Bidirectional Diffusion Transformer for interactive world modeling.

    Jointly processes memory, past, and noised current latent frames in a
    single attention space. Flow matching objective is applied only on
    current frames.

    Architecture (~20M parameters with dim=384, heads=6, layers=12):
      - Latent projection: (C, H, W) → (H*W, dim) spatial tokens
      - Frame-type embeddings (memory / past / current)
      - Temporal position embeddings per frame
      - Spatial position embeddings per token within a frame
      - Timestep conditioning via adaLN-Zero
      - Action conditioning via cross-attention
      - Output projection back to (C, H, W) for current frames only
    """

    def __init__(
        self,
        latent_channels: int = LATENT_CHANNELS,
        latent_size: int = LATENT_SIZE,
        dim: int = DIT_DIM,
        num_heads: int = DIT_HEADS,
        num_layers: int = DIT_LAYERS,
        mlp_ratio: int = DIT_MLP_RATIO,
        num_actions: int = NUM_ACTIONS,
        past_k: int = PAST_FRAMES,
        current_c: int = CURRENT_FRAMES,
        memory_r: int = MEMORY_FRAMES,
        max_frames: int = TOTAL_FRAMES,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.dim = dim
        self.num_spatial = latent_size * latent_size  # 196
        self.past_k = past_k
        self.current_c = current_c
        self.memory_r = memory_r

        # Latent patch → token projection
        self.latent_proj = nn.Linear(latent_channels, dim)

        # Output projection (current frames only): token → latent velocity
        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, latent_channels)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Positional embeddings
        self.spatial_pos = nn.Embedding(self.num_spatial, dim)
        self.temporal_pos = nn.Embedding(max_frames, dim)

        # Frame-type embeddings: 0=memory, 1=past, 2=current
        self.frame_type_embed = nn.Embedding(3, dim)

        # Timestep embedding for flow matching
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        # Action embedding for cross-attention
        self.action_embed = nn.Embedding(num_actions, dim)
        self.action_pos = nn.Embedding(max_frames, dim)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio) for _ in range(num_layers)
        ])

        # Final adaLN modulation
        self.final_adaLN = AdaLNModulation(dim, n_modulations=2)

    def _patchify(self, latents: torch.Tensor) -> torch.Tensor:
        """Convert (B, F, C, H, W) latents to (B, F*S, dim) tokens."""
        B, F, C, H, W = latents.shape
        S = H * W
        tokens = latents.reshape(B, F, C, S).permute(0, 1, 3, 2)  # (B, F, S, C)
        tokens = self.latent_proj(tokens)  # (B, F, S, dim)
        return tokens, F, S

    def _unpatchify(self, tokens: torch.Tensor, F: int, S: int) -> torch.Tensor:
        """Convert (B, F*S, dim) tokens back to (B, F, C, H, W) latent velocities."""
        B = tokens.shape[0]
        tokens = tokens.reshape(B, F, S, self.dim)
        v = self.output_proj(self.output_norm(tokens))  # (B, F, S, C)
        v = v.permute(0, 1, 3, 2)  # (B, F, C, S)
        v = v.reshape(B, F, self.latent_channels, self.latent_size, self.latent_size)
        return v

    def forward(
        self,
        memory_latents: torch.Tensor,   # (B, M, C, H, W)
        past_latents: torch.Tensor,      # (B, K, C, H, W)
        current_noisy: torch.Tensor,     # (B, Nc, C, H, W)
        t: torch.Tensor,                 # (B,) flow matching timestep in [0, 1]
        actions: torch.Tensor,           # (B, K+Nc) action indices
        memory_temporal_ids: torch.Tensor | None = None,  # (B, M) absolute frame indices
        past_temporal_ids: torch.Tensor | None = None,    # (B, K)
        current_temporal_ids: torch.Tensor | None = None, # (B, Nc)
    ) -> torch.Tensor:
        """Predict flow velocity for current frames.

        Returns:
            v_pred: (B, Nc, C, H, W) predicted velocity field
        """
        B = memory_latents.shape[0]
        device = memory_latents.device
        M = memory_latents.shape[1]
        K = past_latents.shape[1]
        Nc = current_noisy.shape[1]

        # Patchify each group
        mem_tokens, _, S = self._patchify(memory_latents)    # (B, M, S, dim)
        past_tokens, _, _ = self._patchify(past_latents)     # (B, K, S, dim)
        curr_tokens, _, _ = self._patchify(current_noisy)    # (B, Nc, S, dim)

        # Add spatial position embeddings
        spatial_ids = torch.arange(S, device=device)
        sp_emb = self.spatial_pos(spatial_ids)  # (S, dim)
        mem_tokens = mem_tokens + sp_emb[None, None, :, :]
        past_tokens = past_tokens + sp_emb[None, None, :, :]
        curr_tokens = curr_tokens + sp_emb[None, None, :, :]

        # Add temporal position embeddings
        if memory_temporal_ids is None:
            memory_temporal_ids = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)
        if past_temporal_ids is None:
            past_temporal_ids = torch.arange(M, M + K, device=device).unsqueeze(0).expand(B, -1)
        if current_temporal_ids is None:
            current_temporal_ids = torch.arange(M + K, M + K + Nc, device=device).unsqueeze(0).expand(B, -1)

        # Clamp temporal ids to embedding table size
        max_t = self.temporal_pos.num_embeddings - 1
        mem_t_emb = self.temporal_pos(memory_temporal_ids.clamp(0, max_t))     # (B, M, dim)
        past_t_emb = self.temporal_pos(past_temporal_ids.clamp(0, max_t))      # (B, K, dim)
        curr_t_emb = self.temporal_pos(current_temporal_ids.clamp(0, max_t))   # (B, Nc, dim)

        mem_tokens = mem_tokens + mem_t_emb[:, :, None, :]
        past_tokens = past_tokens + past_t_emb[:, :, None, :]
        curr_tokens = curr_tokens + curr_t_emb[:, :, None, :]

        # Add frame-type embeddings
        type_ids = torch.tensor([0, 1, 2], device=device)
        type_embs = self.frame_type_embed(type_ids)  # (3, dim)
        mem_tokens = mem_tokens + type_embs[0]
        past_tokens = past_tokens + type_embs[1]
        curr_tokens = curr_tokens + type_embs[2]

        # Flatten all tokens into unified sequence
        # (B, M*S + K*S + Nc*S, dim)
        all_tokens = torch.cat([
            mem_tokens.reshape(B, M * S, self.dim),
            past_tokens.reshape(B, K * S, self.dim),
            curr_tokens.reshape(B, Nc * S, self.dim),
        ], dim=1)

        # Timestep conditioning
        t_emb = self.time_embed(t)  # (B, dim)

        # Action tokens for cross-attention
        n_actions = actions.shape[1]
        action_ids = torch.arange(n_actions, device=device)
        act_tokens = self.action_embed(actions)  # (B, K+Nc, dim)
        act_tokens = act_tokens + self.action_pos(action_ids.clamp(0, max_t))[None, :, :]

        # Process through DiT blocks
        for block in self.blocks:
            all_tokens = block(all_tokens, t_emb, act_tokens)

        # Extract current frame tokens and predict velocity
        current_start = (M + K) * S
        current_end = current_start + Nc * S
        current_out = all_tokens[:, current_start:current_end, :]

        # Final adaLN modulation
        shift, scale = self.final_adaLN(t_emb)
        current_out = modulate(self.output_norm(current_out.reshape(B, Nc * S, self.dim)),
                               shift, scale)
        # We already normed above through the modulate, reuse output_proj directly
        v_pred = self.output_proj(current_out)  # (B, Nc*S, C)
        v_pred = v_pred.reshape(B, Nc, S, self.latent_channels)
        v_pred = v_pred.permute(0, 1, 3, 2)  # (B, Nc, C, S)
        v_pred = v_pred.reshape(B, Nc, self.latent_channels,
                                self.latent_size, self.latent_size)
        return v_pred


# ═══════════════════════════════════════════════════════════════════════════
# Error Buffer
# ═══════════════════════════════════════════════════════════════════════════

class ErrorBuffer:
    """Ring buffer storing prediction residuals for error injection.

    Residuals are stored as flattened vectors and sampled uniformly
    during training to perturb conditioning latents.
    """

    def __init__(self, max_size: int = ERROR_BUFFER_SIZE):
        self.max_size = max_size
        self.buffer: collections.deque[torch.Tensor] = collections.deque(maxlen=max_size)

    def add(self, residuals: torch.Tensor):
        """Add residuals to the buffer. residuals: (N, C, H, W)."""
        for i in range(residuals.shape[0]):
            self.buffer.append(residuals[i].detach().cpu())

    def sample(self, n: int, device: torch.device) -> torch.Tensor | None:
        """Sample n residuals uniformly. Returns (n, C, H, W) or None if empty."""
        if len(self.buffer) == 0:
            return None
        indices = [random.randint(0, len(self.buffer) - 1) for _ in range(n)]
        return torch.stack([self.buffer[i] for i in indices]).to(device)

    def __len__(self) -> int:
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════
# Flow Matching Utilities
# ═══════════════════════════════════════════════════════════════════════════

def flow_matching_sample_t(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample timesteps uniformly from [0, 1] for flow matching."""
    return torch.rand(batch_size, device=device)


def flow_matching_interpolate(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimal transport conditional flow: x_t = (1-t)*x0 + t*x1.

    Target velocity: v = x1 - x0 (constant velocity field).

    Args:
        x0: noise sample (B, ...)
        x1: clean data (B, ...)
        t: timestep in [0, 1], shape (B,)

    Returns:
        x_t: interpolated sample
        v_target: target velocity (x1 - x0)
    """
    # Broadcast t to match spatial dims
    t_expanded = t
    while t_expanded.dim() < x0.dim():
        t_expanded = t_expanded.unsqueeze(-1)

    x_t = (1 - t_expanded) * x0 + t_expanded * x1
    v_target = x1 - x0
    return x_t, v_target


@torch.no_grad()
def flow_matching_euler_sample(
    model: MatrixGameDiT,
    memory_latents: torch.Tensor,
    past_latents: torch.Tensor,
    actions: torch.Tensor,
    num_steps: int = 50,
    memory_temporal_ids: torch.Tensor | None = None,
    past_temporal_ids: torch.Tensor | None = None,
    current_temporal_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate current frames via Euler ODE integration from noise to data.

    Integrates dx/dt = v(x, t) from t=0 (noise) to t=1 (data).
    """
    B = past_latents.shape[0]
    device = past_latents.device
    Nc = model.current_c
    shape = (B, Nc, model.latent_channels, model.latent_size, model.latent_size)

    x_t = torch.randn(shape, device=device)
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t = torch.full((B,), step / num_steps, device=device)
        v_pred = model(
            memory_latents, past_latents, x_t, t, actions,
            memory_temporal_ids, past_temporal_ids, current_temporal_ids,
        )
        x_t = x_t + v_pred * dt

    return x_t


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── Load palette info ──
    palette_path = os.path.join(args.data_dir, "palette.json")
    with open(palette_path) as f:
        palette_data = json.load(f)
    num_colors = palette_data["num_colors"]
    palette_rgb = np.array(palette_data["colors_rgb"], dtype=np.uint8)

    # ── Dataset ──
    dataset = MatrixGameDataset(
        args.data_dir,
        past_k=PAST_FRAMES,
        current_c=CURRENT_FRAMES,
        memory_r=MEMORY_FRAMES,
    )
    n_eval = min(500, max(1, len(dataset) // 10))
    train_ds, eval_ds = random_split(
        dataset, [len(dataset) - n_eval, n_eval],
        generator=torch.Generator().manual_seed(args.seed),
    )

    num_workers = min(get_effective_cpu_count(), 16)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0,
    )

    # ── Load frozen VAE ──
    ae = FrameVAE(num_palette_colors=num_colors, latent_channels=LATENT_CHANNELS).to(device)
    ae_state = torch.load(args.ae_checkpoint, map_location=device, weights_only=True)
    # Load only encoder weights (decoder not needed)
    ae_keys = {k: v for k, v in ae_state.items() if k.startswith(("encoder.", "enc_mu."))}
    ae.load_state_dict(ae_keys, strict=True)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    print(f"Loaded frozen AE encoder from {args.ae_checkpoint}")

    # ── Build DiT model ──
    model = MatrixGameDiT(
        latent_channels=LATENT_CHANNELS,
        latent_size=LATENT_SIZE,
        dim=DIT_DIM,
        num_heads=DIT_HEADS,
        num_layers=DIT_LAYERS,
        mlp_ratio=DIT_MLP_RATIO,
        num_actions=NUM_ACTIONS,
        past_k=PAST_FRAMES,
        current_c=CURRENT_FRAMES,
        memory_r=MEMORY_FRAMES,
    ).to(device)

    n_params = count_parameters(model)
    print(f"MatrixGameDiT: {n_params:,} parameters ({n_params / 1e6:.1f}M)")

    # ── Optimizer & scheduler ──
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = LinearLR(optimizer, start_factor=1e-6, total_iters=args.warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(args.max_steps - args.warmup_steps, 1),
        eta_min=args.lr * 0.05,
    )
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_steps])

    # ── Error buffer ──
    error_buffer = ErrorBuffer(max_size=ERROR_BUFFER_SIZE)

    # ── Output directory ──
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config = {
        "dit_dim": DIT_DIM, "dit_heads": DIT_HEADS, "dit_layers": DIT_LAYERS,
        "dit_mlp_ratio": DIT_MLP_RATIO, "past_k": PAST_FRAMES,
        "current_c": CURRENT_FRAMES, "memory_r": MEMORY_FRAMES,
        "latent_channels": LATENT_CHANNELS, "latent_size": LATENT_SIZE,
        "num_actions": NUM_ACTIONS, "num_params": n_params,
        "lr": args.lr, "batch_size": args.batch_size,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Resume ──
    global_step = 0
    best_eval_loss = float("inf")
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt.get("global_step", 0)
        best_eval_loss = ckpt.get("best_eval_loss", float("inf"))
        print(f"Resumed from step {global_step} (best_eval={best_eval_loss:.6f})")

    # ── Encode helper ──
    @torch.no_grad()
    def encode_batch(frames: torch.Tensor) -> torch.Tensor:
        """Encode (B, F, H, W) palette frames → (B, F, C, Hl, Wl) latents."""
        B, F, H, W = frames.shape
        flat = frames.reshape(B * F, H, W)
        z = ae.encode(flat)
        return z.reshape(B, F, *z.shape[1:])

    # ── Training loop ──
    print(f"\nStarting training for {args.max_steps} steps...")
    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    log_interval = 50
    eval_interval = args.eval_interval
    checkpoint_interval = args.checkpoint_interval
    t_start = time.time()
    last_image_time = 0.0

    if args.tf16 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled")

    if args.compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    pbar = tqdm(range(global_step + 1, args.max_steps + 1), initial=global_step,
                total=args.max_steps, desc="Training")

    for step in pbar:
        # Get batch (cycle through dataset)
        try:
            memory_frames, past_frames, current_frames, actions = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            memory_frames, past_frames, current_frames, actions = next(train_iter)

        memory_frames = memory_frames.to(device)
        past_frames = past_frames.to(device)
        current_frames = current_frames.to(device)
        actions = actions.to(device)

        # Encode to latents
        memory_latents = encode_batch(memory_frames)  # (B, M, C, H, W)
        past_latents = encode_batch(past_frames)      # (B, K, C, H, W)
        current_latents = encode_batch(current_frames) # (B, Nc, C, H, W)

        # ── Error injection ──
        if len(error_buffer) > 0 and random.random() < 0.5:
            B = past_latents.shape[0]
            K = past_latents.shape[1]
            M = memory_latents.shape[1]

            # Perturb past latents
            delta_past = error_buffer.sample(B * K, device)
            if delta_past is not None:
                past_latents = past_latents + ERROR_GAMMA_HISTORY * delta_past.reshape(B, K, *delta_past.shape[1:])

            # Perturb memory latents
            delta_mem = error_buffer.sample(B * M, device)
            if delta_mem is not None:
                memory_latents = memory_latents + ERROR_GAMMA_MEMORY * delta_mem.reshape(B, M, *delta_mem.shape[1:])

        # ── Flow matching ──
        # Sample noise and timestep
        noise = torch.randn_like(current_latents)
        t = flow_matching_sample_t(current_latents.shape[0], device)

        # Interpolate: x_t = (1-t)*noise + t*data
        x_t, v_target = flow_matching_interpolate(noise, current_latents, t)

        # Predict velocity
        v_pred = model(memory_latents, past_latents, x_t, t, actions)

        # Flow matching loss (MSE on velocity)
        loss = F.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        # ── Error collection ──
        # Compute x0 prediction from the velocity: x0_hat = x_t + (1-t)*v_pred
        with torch.no_grad():
            t_expanded = t
            while t_expanded.dim() < v_pred.dim():
                t_expanded = t_expanded.unsqueeze(-1)
            x0_hat = x_t + (1 - t_expanded) * v_pred.detach()
            # Residual: prediction error
            residuals = (x0_hat - current_latents).reshape(-1, *current_latents.shape[2:])
            # Subsample to avoid flooding the buffer
            n_add = min(residuals.shape[0], 16)
            indices = torch.randperm(residuals.shape[0])[:n_add]
            error_buffer.add(residuals[indices])

        running_loss += loss.item()
        global_step = step

        # ── Logging ──
        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed = time.time() - t_start
            steps_per_sec = step / max(elapsed, 1)
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{avg_loss:.5f}",
                lr=f"{lr:.2e}",
                buf=len(error_buffer),
                sps=f"{steps_per_sec:.1f}",
            )
            running_loss = 0.0

        # ── Evaluation ──
        if step % eval_interval == 0:
            model.eval()
            eval_loss = 0.0
            eval_n = 0
            with torch.no_grad():
                for mem_f, past_f, curr_f, act in eval_loader:
                    mem_f = mem_f.to(device)
                    past_f = past_f.to(device)
                    curr_f = curr_f.to(device)
                    act = act.to(device)

                    mem_z = encode_batch(mem_f)
                    past_z = encode_batch(past_f)
                    curr_z = encode_batch(curr_f)

                    noise = torch.randn_like(curr_z)
                    t_eval = flow_matching_sample_t(curr_z.shape[0], device)
                    x_t, v_target = flow_matching_interpolate(noise, curr_z, t_eval)
                    v_pred = model(mem_z, past_z, x_t, t_eval, act)
                    eval_loss += F.mse_loss(v_pred, v_target, reduction="sum").item()
                    eval_n += v_target.numel()

            eval_loss /= max(eval_n, 1)
            print(f"\n  [step {step}] eval_loss={eval_loss:.6f}  "
                  f"error_buf={len(error_buffer)}")
            model.train()

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({
                    "model": unwrapped.state_dict(),
                    "global_step": step,
                    "eval_loss": eval_loss,
                }, os.path.join(output_dir, "dit_best.pt"))
                print(f"  ✓ saved dit_best.pt (eval_loss={eval_loss:.6f})")

        # ── Checkpoint ──
        if checkpoint_interval > 0 and step % checkpoint_interval == 0:
            unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({
                "model": unwrapped.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": step,
                "best_eval_loss": best_eval_loss,
            }, os.path.join(output_dir, "training_state_latest.pt"))
            print(f"  checkpoint saved at step {step}")

        # ── Visualization ──
        now = time.time()
        if now - last_image_time > args.image_interval_secs and step > args.warmup_steps:
            last_image_time = now
            _save_sample_images(
                model, ae, dataset, device, output_dir, step,
                palette_rgb, num_colors,
            )

    # Final save
    unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "best_eval_loss": best_eval_loss,
    }, os.path.join(output_dir, "training_state_final.pt"))
    print(f"\nTraining complete. Final step: {global_step}, best eval: {best_eval_loss:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def palette_to_rgb(indices: torch.Tensor | np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Convert (H, W) palette indices to (H, W, 3) uint8 RGB."""
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy()
    return palette[indices].astype(np.uint8)


def _save_sample_images(
    model: nn.Module,
    ae: FrameVAE,
    dataset: MatrixGameDataset,
    device: torch.device,
    output_dir: str,
    step: int,
    palette_rgb: np.ndarray,
    num_colors: int,
    n_samples: int = 4,
    num_euler_steps: int = 50,
):
    """Generate and save sample images during training."""
    from PIL import Image

    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
    unwrapped.eval()

    # Grab a few samples from the dataset directly
    indices = list(range(min(n_samples, len(dataset))))
    rows = []

    with torch.no_grad():
        for idx in indices:
            mem_f, past_f, curr_f, actions = dataset[idx]
            mem_f = mem_f.unsqueeze(0).to(device)
            past_f = past_f.unsqueeze(0).to(device)
            curr_f = curr_f.unsqueeze(0).to(device)
            actions = actions.unsqueeze(0).to(device)

            mem_z = ae.encode(mem_f.reshape(-1, *mem_f.shape[2:])).reshape(1, -1, *[LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE])
            past_z = ae.encode(past_f.reshape(-1, *past_f.shape[2:])).reshape(1, -1, *[LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE])

            # Generate via Euler sampling
            pred_z = flow_matching_euler_sample(
                unwrapped, mem_z, past_z, actions,
                num_steps=num_euler_steps,
            )

            # We need the full VAE decoder to go from latent → palette indices
            # Since we only loaded the encoder, we can't decode.
            # Instead, show GT frames with a marker for predicted frames.
            # For proper decoding, the full AE must be available.

            # Show: last past frame | first GT current | text "predicted"
            last_past = past_f[0, -1].cpu()
            first_gt = curr_f[0, 0].cpu()

            row_parts = [
                palette_to_rgb(last_past, palette_rgb),
                np.full((IMAGE_SIZE, 2, 3), 255, dtype=np.uint8),
                palette_to_rgb(first_gt, palette_rgb),
            ]
            rows.append(np.concatenate(row_parts, axis=1))

    if rows:
        h_sep = np.full((2, rows[0].shape[1], 3), 255, dtype=np.uint8)
        parts = []
        for i, r in enumerate(rows):
            if i > 0:
                parts.append(h_sep)
            parts.append(r)
        grid = np.concatenate(parts, axis=0)
        path = os.path.join(vis_dir, f"step_{step:07d}.png")
        Image.fromarray(grid).save(path)

    unwrapped.train()


# ═══════════════════════════════════════════════════════════════════════════
# Autoregressive rollout visualization
# ═══════════════════════════════════════════════════════════════════════════

def visualize_rollout(args):
    """Generate autoregressive rollout frames."""
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    palette_path = os.path.join(args.data_dir, "palette.json")
    with open(palette_path) as f:
        palette_data = json.load(f)
    num_colors = palette_data["num_colors"]
    palette_rgb = np.array(palette_data["colors_rgb"], dtype=np.uint8)

    # Load full AE (need decoder for visualization)
    # Import the full FrameVAE from train_genie2.py
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from train_genie2 import FrameVAE as FullFrameVAE

    full_ae = FullFrameVAE(num_palette_colors=num_colors, latent_channels=LATENT_CHANNELS).to(device)
    full_ae.load_state_dict(torch.load(args.ae_checkpoint, map_location=device, weights_only=True))
    full_ae.eval()

    # Load DiT
    model = MatrixGameDiT(
        latent_channels=LATENT_CHANNELS,
        latent_size=LATENT_SIZE,
        dim=DIT_DIM,
        num_heads=DIT_HEADS,
        num_layers=DIT_LAYERS,
        mlp_ratio=DIT_MLP_RATIO,
        num_actions=NUM_ACTIONS,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load a sample sequence
    dataset = MatrixGameDataset(args.data_dir)
    mem_f, past_f, curr_f, actions = dataset[0]
    mem_f = mem_f.unsqueeze(0).to(device)
    past_f = past_f.unsqueeze(0).to(device)
    actions = actions.unsqueeze(0).to(device)

    # Encode initial frames
    ae_encoder = FrameVAE(num_palette_colors=num_colors, latent_channels=LATENT_CHANNELS).to(device)
    ae_state = torch.load(args.ae_checkpoint, map_location=device, weights_only=True)
    ae_keys = {k: v for k, v in ae_state.items() if k.startswith(("encoder.", "enc_mu."))}
    ae_encoder.load_state_dict(ae_keys, strict=True)
    ae_encoder.eval()

    with torch.no_grad():
        mem_z = ae_encoder.encode(mem_f.reshape(-1, *mem_f.shape[2:])).reshape(1, MEMORY_FRAMES, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
        past_z = ae_encoder.encode(past_f.reshape(-1, *past_f.shape[2:])).reshape(1, PAST_FRAMES, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)

        # Autoregressive rollout
        rollout_steps = args.rollout_steps
        generated_frames = []

        for step in range(rollout_steps):
            pred_z = flow_matching_euler_sample(
                model, mem_z, past_z, actions,
                num_steps=args.euler_steps,
            )

            # Decode each predicted frame
            for fi in range(pred_z.shape[1]):
                frame_z = pred_z[0, fi:fi+1]
                logits = full_ae.decode(frame_z)
                frame_idx = logits.argmax(1)[0]
                generated_frames.append(palette_to_rgb(frame_idx, palette_rgb))

            # Shift: current predictions become past, keep rolling
            past_z = torch.cat([past_z[:, pred_z.shape[1]:], pred_z], dim=1)

    # Save rollout as image grid
    out_dir = os.path.join(args.output_dir, "rollout")
    os.makedirs(out_dir, exist_ok=True)

    n_cols = 8
    n_rows = math.ceil(len(generated_frames) / n_cols)
    H, W = IMAGE_SIZE, IMAGE_SIZE
    grid = np.full((n_rows * (H + 2) - 2, n_cols * (W + 2) - 2, 3), 255, dtype=np.uint8)

    for i, frame in enumerate(generated_frames):
        r, c = divmod(i, n_cols)
        y, x = r * (H + 2), c * (W + 2)
        grid[y:y + H, x:x + W] = frame

    path = os.path.join(out_dir, "rollout.png")
    Image.fromarray(grid).save(path)
    print(f"Saved rollout with {len(generated_frames)} frames to {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matrix-Game 3.0 – Error-Aware Interactive World Model"
    )
    parser.add_argument("--phase", choices=["train", "visualize"], default="train")
    parser.add_argument("--data-dir", type=str, default="data/normalized")
    parser.add_argument("--output-dir", type=str, default="checkpoints/matrixgame")
    parser.add_argument("--ae-checkpoint", type=str, required=True,
                        help="Path to frozen FrameVAE checkpoint (from train_genie2.py)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--image-interval-secs", type=float, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tf16", action="store_true", help="Enable TF32 mode")
    parser.add_argument("--compile", action="store_true", help="torch.compile()")
    parser.add_argument("--resume-from", type=str, default=None)

    # Visualization args
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="DiT checkpoint for visualization")
    parser.add_argument("--rollout-steps", type=int, default=8,
                        help="Number of autoregressive rollout steps")
    parser.add_argument("--euler-steps", type=int, default=50,
                        help="Euler integration steps for sampling")

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.phase == "train":
        train(args)
    elif args.phase == "visualize":
        if not args.checkpoint:
            raise SystemExit("--checkpoint required for visualization")
        visualize_rollout(args)


if __name__ == "__main__":
    main()
