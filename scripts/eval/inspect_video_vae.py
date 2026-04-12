#!/usr/bin/env python3
"""Interactive diagnostic viewer for Video VAE reconstructions.

Loads a checkpoint, samples a random clip from the dataset, runs a forward
pass, and displays six diagnostic panels:

  Row 1: Target RGB | Reconstruction RGB | Error overlay
  Row 2: P(correct) confidence | Class weight map | Latent channel grid

Controls:
  Frame slider   – scrub through time
  Play / Pause   – animate playback
  Left / Right   – step one frame
  Any other key  – resample a new random clip

Usage:
    python scripts/eval/inspect_video_vae.py checkpoints/video_vae_20260410_134808
    python scripts/eval/inspect_video_vae.py checkpoints/video_vae_20260410_134808 --seed 42
    python scripts/eval/inspect_video_vae.py checkpoints/video_vae_20260410_134808 --best
    python scripts/eval/inspect_video_vae.py checkpoints/video_vae_20260410_134808 --predicted-frames 64
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plot_style import apply_plot_style, enable_slider_scroll, style_image_axes, style_widget, BG_COLOR
apply_plot_style()

from src.data.dataset_index import load_index
from src.data.normalized_dataset import load_palette_tensor
from src.data.video_frames import resize_palette_frames
from src.models.video_vae import VideoVAE
from src.training.losses import (
    focal_cross_entropy,
    softened_inverse_frequency_weights,
    spatial_weight_map,
    temporal_change_weight,
)
from src.training.palette_video_vae_training import frames_to_one_hot
from src.training.training_utils import load_model_state_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_config(checkpoint_dir: Path) -> dict:
    cfg_path = checkpoint_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {checkpoint_dir}")
    with cfg_path.open() as f:
        return json.load(f)


@dataclass
class _LatentNormalization:
    mean: torch.Tensor  # (Z, 1, H, W)
    std: torch.Tensor   # (Z, 1, H, W)
    stats_path: Path
    warned_shape_mismatch: bool = False


def _load_latent_normalization(latents_dir: Path) -> _LatentNormalization | None:
    stats_path = latents_dir / "latent_stats.json"
    if not stats_path.is_file():
        return None
    try:
        with stats_path.open() as f:
            stats = json.load(f)

        version = int(stats.get("latent_stats_version"))
        if version != 2:
            print(f"  latent normalization: skipping {stats_path} (unsupported latent_stats_version={version})")
            return None

        scheme = stats.get("normalization_scheme")
        if scheme != "component_chw_shared_time":
            print(
                "  latent normalization: "
                f"skipping {stats_path} (unsupported normalization_scheme={scheme!r})"
            )
            return None

        mean_v = stats.get("component_mean")
        std_v = stats.get("component_std_clamped")
        if mean_v is None or std_v is None:
            print(
                "  latent normalization: "
                f"skipping {stats_path} (missing component_mean/component_std_clamped)"
            )
            return None

        mean_np = np.asarray(mean_v, dtype=np.float32)
        std_np = np.asarray(std_v, dtype=np.float32)
        if mean_np.ndim != 3 or std_np.shape != mean_np.shape:
            print(
                "  latent normalization: "
                f"skipping {stats_path} (invalid shapes mean={mean_np.shape}, std={std_np.shape})"
            )
            return None

        eps = max(float(stats.get("std_epsilon", 1e-6)), 1e-6)
        std_np = np.nan_to_num(std_np, nan=eps, posinf=eps, neginf=eps)
        std_np = np.maximum(std_np, eps)

        return _LatentNormalization(
            mean=torch.from_numpy(mean_np).unsqueeze(1),
            std=torch.from_numpy(std_np).unsqueeze(1),
            stats_path=stats_path,
        )
    except Exception as exc:
        print(f"  latent normalization: failed to load {stats_path}: {exc}")
        return None


def _load_class_weights(data_dir: str, *, num_classes: int, soften: float) -> torch.Tensor | None:
    dist_path = Path(data_dir) / "palette_distribution.json"
    if not dist_path.is_file():
        return None
    with dist_path.open() as f:
        dist = json.load(f)
    counts = torch.tensor(dist.get("counts") or dist["probabilities"], dtype=torch.float32)
    if counts.numel() != num_classes:
        return None
    return softened_inverse_frequency_weights(counts, soften=soften)


def _build_model(config: dict, device: torch.device) -> VideoVAE:
    tcfg = config["training"]
    dcfg = config["data"]
    model = VideoVAE(
        num_colors=dcfg["num_colors"],
        base_channels=tcfg["base_channels"],
        latent_channels=tcfg["latent_channels"],
        temporal_downsample=tcfg.get("temporal_downsample", 0),
        dropout=0.0,
        onehot_conv=tcfg.get("onehot_conv", False),
        global_bottleneck_attn=tcfg.get("global_bottleneck_attn", False),
        global_bottleneck_attn_heads=tcfg.get("global_bottleneck_attn_heads", 8),
    )
    return model.to(device)


def _load_checkpoint(model: VideoVAE, ckpt_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    load_model_state_dict(model, ckpt["model"], strict=False)
    model.eval()
    return ckpt


def _forward_video_vae_chunked(
    model: VideoVAE,
    frames: torch.Tensor,
    config: dict,
    device: torch.device,
    *,
    chunk_frames: int,
    chunk_overlap: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run chunked temporal inference and stitch outputs on CPU.

    Returns:
        logits_cpu: (1, C, T, H, W)
        mean_cpu: (1, Z, T_lat, H_lat, W_lat)
        logvar_cpu: (1, Z, T_lat, H_lat, W_lat)
    """
    tcfg = config["training"]
    dcfg = config["data"]
    use_onehot_conv = tcfg.get("onehot_conv", False)
    onehot_dtype_name = tcfg.get("onehot_dtype", "float32")
    onehot_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(onehot_dtype_name, torch.float32)
    mixed_precision = tcfg.get("mixed_precision", "no")
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mixed_precision)

    total_frames = int(frames.shape[1])
    chunk_frames = min(max(int(chunk_frames), 1), total_frames)
    chunk_overlap = max(int(chunk_overlap), 0)
    if chunk_frames <= 2 * chunk_overlap:
        # Ensure each chunk contributes at least one new frame.
        chunk_overlap = max(0, (chunk_frames - 1) // 2)
    core_step = max(1, chunk_frames - 2 * chunk_overlap)

    logits_parts: list[torch.Tensor] = []
    mean_parts: list[torch.Tensor] = []
    logvar_parts: list[torch.Tensor] = []

    core_start = 0
    while core_start < total_frames:
        core_end = min(core_start + core_step, total_frames)
        in_start = max(0, core_start - chunk_overlap)
        in_end = min(total_frames, core_end + chunk_overlap)

        frames_chunk = frames[:, in_start:in_end]
        if use_onehot_conv:
            model_input = frames_chunk.byte()
        else:
            model_input = frames_to_one_hot(frames_chunk.long(), dcfg["num_colors"], dtype=onehot_dtype)

        amp_ctx = torch.autocast(device_type=device.type, dtype=autocast_dtype) if autocast_dtype and device.type == "cuda" else nullcontext()
        with amp_ctx:
            outputs = model(model_input, sample_posterior=False)

        logits_chunk = outputs.logits.float().cpu()
        mean_chunk = outputs.posterior_mean.float().cpu()
        logvar_chunk = outputs.posterior_logvar.float().cpu()

        keep_start = core_start - in_start
        keep_end = keep_start + (core_end - core_start)
        logits_parts.append(logits_chunk[:, :, keep_start:keep_end])

        t_in = int(in_end - in_start)
        t_lat = int(mean_chunk.shape[2])
        lat_start = int(round(keep_start * t_lat / max(t_in, 1)))
        lat_end = int(round(keep_end * t_lat / max(t_in, 1)))
        if keep_end > keep_start:
            lat_end = max(lat_end, lat_start + 1)
        lat_start = max(0, min(lat_start, t_lat))
        lat_end = max(lat_start, min(lat_end, t_lat))

        mean_parts.append(mean_chunk[:, :, lat_start:lat_end])
        logvar_parts.append(logvar_chunk[:, :, lat_start:lat_end])

        core_start = core_end

    return (
        torch.cat(logits_parts, dim=2),
        torch.cat(mean_parts, dim=2),
        torch.cat(logvar_parts, dim=2),
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _latent_channel_grid(latent: np.ndarray, sep: int = 1) -> np.ndarray:
    """Arrange (C, H, W) latent channels into a rectangular grid image.

    *sep* pixels of NaN are inserted between channels so that the colormap
    renders them as a visible separator line.
    """
    C, H, W = latent.shape
    cols = 8
    rows = (C + cols - 1) // cols
    pad = rows * cols - C
    if pad > 0:
        latent = np.concatenate([latent, np.full((pad, H, W), np.nan, dtype=latent.dtype)], axis=0)
    latent = latent.reshape(rows, cols, H, W)

    cell_h = H + sep
    cell_w = W + sep
    grid = np.full((rows * cell_h - sep, cols * cell_w - sep), np.nan, dtype=latent.dtype)
    for r in range(rows):
        for c in range(cols):
            y = r * cell_h
            x = c * cell_w
            grid[y:y + H, x:x + W] = latent[r, c]
    return grid


def _entropy(logits_t: torch.Tensor) -> np.ndarray:
    """Per-pixel entropy from logits (C, H, W) → (H, W) numpy."""
    probs = F.softmax(logits_t.float(), dim=0)
    log_probs = torch.log2(probs.clamp_min(1e-12))
    return -(probs * log_probs).sum(dim=0).cpu().numpy()


def _correct_prob(logits_t: torch.Tensor, target_t: torch.Tensor) -> np.ndarray:
    """P(correct class) per pixel from logits (C, H, W) and target (H, W)."""
    probs = F.softmax(logits_t.float(), dim=0)  # (C, H, W)
    gathered = probs.gather(0, target_t.unsqueeze(0).long()).squeeze(0)  # (H, W)
    return gathered.cpu().numpy()


def _build_effective_loss_weight_map(
    frames: torch.Tensor,
    class_weight: torch.Tensor | None,
    *,
    context_frames: int,
    class_weight_radius: float,
    class_weight_hardness: float,
    class_weight_temporal_ema: float,
    temporal_change_boost: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return (effective_display_weight, pixel_weight) matching training semantics.

    ``effective_display_weight`` is the full multiplicative factor applied per pixel
    on the predicted frames only, suitable for visualization. ``pixel_weight`` is
    the tensor passed to ``focal_cross_entropy`` alongside the optional per-class
    vector, matching the training call structure.
    """
    recon_targets = frames[:, context_frames:] if context_frames > 0 else frames
    effective_weight: torch.Tensor | None = None
    pixel_weight: torch.Tensor | None = None

    if class_weight is not None and class_weight_radius >= 0.5:
        spatial_pw = spatial_weight_map(
            frames,
            class_weight,
            radius=class_weight_radius,
            hardness=class_weight_hardness,
            temporal_ema=class_weight_temporal_ema,
        )
        effective_weight = spatial_pw[:, context_frames:] if context_frames > 0 else spatial_pw
        pixel_weight = effective_weight
    elif class_weight is not None:
        effective_weight = class_weight.to(frames.device, dtype=torch.float32)[recon_targets.long()]

    if temporal_change_boost > 0:
        change_pw = temporal_change_weight(
            frames,
            boost=temporal_change_boost,
            context_frames=context_frames,
        )
        pixel_weight = change_pw if pixel_weight is None else pixel_weight * change_pw
        effective_weight = change_pw if effective_weight is None else effective_weight * change_pw

    return effective_weight, pixel_weight


# ---------------------------------------------------------------------------
# Lightweight random clip sampler (mmap, no full dataset load)
# ---------------------------------------------------------------------------

class _ClipSampler:
    """Indexes .npz files on disk; loads individual clips via mmap on demand."""

    def __init__(self, data_dir: str | Path, clip_frames: int, frame_size: int) -> None:
        self.data_dir = Path(data_dir)
        self.clip_frames = clip_frames
        self.frame_size = frame_size
        self.npz_files = sorted(self.data_dir.glob("*.npz"))
        if not self.npz_files:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")
        # Build a lightweight index: frame count per file.
        # Fast path: use pre-built JSON index if available.
        cached = load_index(self.data_dir)
        if cached is not None:
            idx_names = {e["filename"] for e in cached.get("files", [])}
            disk_names = {p.name for p in self.npz_files}
            if idx_names == disk_names:
                length_map = {e["filename"]: e["frame_count"] for e in cached["files"]}
                self._file_lengths = [length_map[p.name] for p in self.npz_files]
            else:
                cached = None  # stale – fall through to slow path
        if cached is None:
            self._file_lengths = []
            for path in self.npz_files:
                npz = np.load(path, mmap_mode="r")
                n = npz["frames"].shape[0]
                self._file_lengths.append(n)
                del npz
        self._valid_files = [
            i for i, n in enumerate(self._file_lengths) if n >= clip_frames
        ]
        if not self._valid_files:
            raise RuntimeError(
                f"No files have >= {clip_frames} frames in {self.data_dir}"
            )

    def sample(self, rng: np.random.Generator) -> tuple[torch.Tensor, str]:
        """Return (frames_tensor, info_string).  frames is (T, H, W) uint8."""
        fidx = self._valid_files[int(rng.integers(0, len(self._valid_files)))]
        path = self.npz_files[fidx]
        npz = np.load(path, mmap_mode="r")
        n = self._file_lengths[fidx]
        start = int(rng.integers(0, n - self.clip_frames + 1))
        frames = np.array(npz["frames"][start:start + self.clip_frames], dtype=np.uint8)
        del npz
        frames = resize_palette_frames(
            frames,
            target_height=self.frame_size,
            target_width=self.frame_size,
        )
        info = f"{path.stem} [{start}:{start + self.clip_frames}]"
        return torch.from_numpy(frames), info


# ---------------------------------------------------------------------------
# Core inspection
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_inspection(
    model: VideoVAE,
    sampler: _ClipSampler,
    palette_u8: np.ndarray,
    class_weight: torch.Tensor | None,
    latent_norm: _LatentNormalization | None,
    config: dict,
    device: torch.device,
    rng: np.random.Generator,
    *,
    chunk_frames: int,
    chunk_overlap: int,
) -> dict:
    """Sample a random clip, run the model, and compute all diagnostic arrays."""
    tcfg = config["training"]
    context_frames = tcfg.get("context_frames", 0)
    focal_gamma = tcfg.get("focal_gamma", 0.0)
    cw_radius = tcfg.get("class_weight_radius", 0.0)
    cw_hardness = tcfg.get("class_weight_hardness", 5.0)
    cw_temporal_ema = tcfg.get("class_weight_temporal_ema", 0.0)
    temporal_change_boost = tcfg.get("temporal_change_boost", 0.0)

    clip_frames, clip_info = sampler.sample(rng)
    frames = clip_frames.unsqueeze(0).to(device)  # (1, T, H, W) uint8

    logits_cpu, posterior_mean_cpu, posterior_logvar_cpu = _forward_video_vae_chunked(
        model,
        frames,
        config,
        device,
        chunk_frames=chunk_frames,
        chunk_overlap=chunk_overlap,
    )
    recon_indices = logits_cpu.argmax(dim=1)  # (1, T, H, W)
    recon_logits = logits_cpu[:, :, context_frames:] if context_frames > 0 else logits_cpu
    recon_targets = frames.long()[:, context_frames:] if context_frames > 0 else frames.long()

    T = frames.shape[1]
    frames_cpu = frames[0].cpu()  # (T, H, W)
    logits_cpu = logits_cpu[0].float()  # (C, T, H, W)
    recon_cpu = recon_indices[0].cpu()  # (T, H, W)
    latent_mean = posterior_mean_cpu[0].float()  # (Z, T_lat, H_lat, W_lat)
    latent_logvar = posterior_logvar_cpu[0].float()
    latent_standardized = False

    if latent_norm is not None:
        latent_shape = (latent_mean.shape[0], latent_mean.shape[2], latent_mean.shape[3])
        stats_shape = (latent_norm.mean.shape[0], latent_norm.mean.shape[2], latent_norm.mean.shape[3])
        if latent_shape == stats_shape:
            latent_mean = (latent_mean - latent_norm.mean) / latent_norm.std
            latent_standardized = True
        elif not latent_norm.warned_shape_mismatch:
            print(
                "  latent normalization: shape mismatch; "
                f"model latent (Z,H,W)={latent_shape} vs stats={stats_shape} from {latent_norm.stats_path}. "
                "Using raw latent means."
            )
            latent_norm.warned_shape_mismatch = True

    # RGB arrays
    target_rgb = palette_u8[frames_cpu.numpy()]  # (T, H, W, 3)
    recon_rgb = palette_u8[recon_cpu.numpy()]  # (T, H, W, 3)

    # Error overlay: dim target + red on wrong pixels
    error_mask = (recon_cpu != frames_cpu).numpy()  # (T, H, W) bool
    error_rgb = (target_rgb.astype(np.float32) * 0.3).astype(np.uint8)
    for t in range(T):
        error_rgb[t][error_mask[t]] = [255, 50, 50]

    # Per-frame accuracy
    H, W = frames_cpu.shape[1], frames_cpu.shape[2]
    per_frame_acc = np.array([
        1.0 - error_mask[t].sum() / (H * W) for t in range(T)
    ])

    # P(correct class) heatmap per frame
    confidence = np.stack([
        _correct_prob(logits_cpu[:, t], frames_cpu[t]) for t in range(T)
    ])  # (T, H, W)

    # Entropy per frame
    entropy = np.stack([
        _entropy(logits_cpu[:, t]) for t in range(T)
    ])  # (T, H, W)

    effective_weight, pixel_weight = _build_effective_loss_weight_map(
        frames_cpu.unsqueeze(0),
        class_weight.cpu() if class_weight is not None else None,
        context_frames=context_frames,
        class_weight_radius=cw_radius,
        class_weight_hardness=cw_hardness,
        class_weight_temporal_ema=cw_temporal_ema,
        temporal_change_boost=temporal_change_boost,
    )
    weight_map = None
    if effective_weight is not None:
        weight_map = np.zeros((T, H, W), dtype=np.float32)
        weight_map[context_frames:] = effective_weight[0].numpy()

    # Latent channel grid per time step
    latent_np = latent_mean.numpy()  # (Z, T_lat, H_lat, W_lat)
    T_lat = latent_np.shape[1]

    # KL divergence per spatial position (mean over channels)
    kl_per_pos = 0.5 * (
        latent_mean.square() + latent_logvar.exp() - 1.0 - latent_logvar
    )  # (Z, T_lat, H_lat, W_lat)
    kl_spatial = kl_per_pos.mean(dim=0).numpy()  # (T_lat, H_lat, W_lat)

    # Source file info already captured from sampler

    # Per-frame loss using the same focal/class/pixel weighting as training.
    per_frame_loss = np.full(T, np.nan, dtype=np.float32)
    class_weight_for_loss = class_weight.cpu() if (class_weight is not None and cw_radius < 0.5) else None
    for t in range(recon_targets.shape[1]):
        pixel_weight_t = None if pixel_weight is None else pixel_weight[:, t:t + 1]
        loss_t = focal_cross_entropy(
            recon_logits[:, :, t:t + 1].float().cpu(),
            recon_targets[:, t:t + 1].cpu(),
            gamma=focal_gamma,
            class_weight=class_weight_for_loss,
            pixel_weight=pixel_weight_t,
        ).item()
        per_frame_loss[context_frames + t] = loss_t

    return {
        "target_rgb": target_rgb,
        "recon_rgb": recon_rgb,
        "error_rgb": error_rgb,
        "error_mask": error_mask,
        "confidence": confidence,
        "entropy": entropy,
        "weight_map": weight_map,
        "latent": latent_np,
        "kl_spatial": kl_spatial,
        "T": T,
        "T_lat": T_lat,
        "per_frame_acc": per_frame_acc,
        "per_frame_loss": per_frame_loss,
        "latent_standardized": latent_standardized,
        "context_frames": context_frames,
        "clip_info": clip_info,
    }


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------

def _show_viewer(
    model: VideoVAE,
    sampler: _ClipSampler,
    palette_u8: np.ndarray,
    class_weight: torch.Tensor | None,
    latent_norm: _LatentNormalization | None,
    config: dict,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    rng = np.random.default_rng(args.seed)
    data = _run_inspection(
        model,
        sampler,
        palette_u8,
        class_weight,
        latent_norm,
        config,
        device,
        rng,
        chunk_frames=args.chunk_frames,
        chunk_overlap=args.chunk_overlap,
    )

    T = data["T"]
    ctx = data["context_frames"]

    # --- Layout: 2 rows × 3 cols ---
    scale = args.scale
    H, W = data["target_rgb"].shape[1], data["target_rgb"].shape[2]
    panel_w = W * scale / 100
    panel_h = H * scale / 100
    fig_w = panel_w * 3 + 1.5
    fig_h = panel_h * 2 + 2.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        2, 3,
        wspace=0.08, hspace=0.15,
        left=0.02, right=0.98, top=0.88, bottom=0.18,
    )

    ax_tgt = fig.add_subplot(gs[0, 0])
    ax_rec = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[0, 2])
    ax_conf = fig.add_subplot(gs[1, 0])
    ax_cw = fig.add_subplot(gs[1, 1])
    ax_lat = fig.add_subplot(gs[1, 2])

    for ax in [ax_tgt, ax_rec, ax_err, ax_conf, ax_cw, ax_lat]:
        ax.set_axis_off()
        style_image_axes(ax)

    im_tgt = ax_tgt.imshow(data["target_rgb"][0], interpolation="nearest")
    im_rec = ax_rec.imshow(data["recon_rgb"][0], interpolation="nearest")
    im_err = ax_err.imshow(data["error_rgb"][0], interpolation="nearest")
    im_conf = ax_conf.imshow(data["confidence"][0], interpolation="nearest",
                             cmap="RdYlGn", vmin=0, vmax=1)

    if data["weight_map"] is not None:
        cw_vmin = data["weight_map"].min()
        cw_vmax = data["weight_map"].max()
        im_cw = ax_cw.imshow(data["weight_map"][0], interpolation="nearest",
                             cmap="inferno", vmin=cw_vmin, vmax=cw_vmax)
    else:
        im_cw = ax_cw.imshow(np.zeros((H, W)), interpolation="nearest", cmap="gray")
        ax_cw.set_title("(no loss weighting)", fontsize=9)

    # Latent grid for frame 0
    lat_t0 = _latent_channel_grid(data["latent"][:, 0])
    lat_cmap = plt.get_cmap("coolwarm").copy()
    lat_cmap.set_bad(color=BG_COLOR)
    im_lat = ax_lat.imshow(lat_t0, interpolation="nearest", cmap=lat_cmap,
                           vmin=-3, vmax=3)

    ax_tgt.set_title("Target", fontsize=9)
    ax_rec.set_title("Recon", fontsize=9)
    ax_err.set_title("Error", fontsize=9)
    ax_conf.set_title("Correct Probability", fontsize=9)
    ax_cw.set_title("Loss Weight", fontsize=9)
    def _latent_panel_title(d: dict) -> str:
        if d.get("latent_standardized", False):
            return "Standardized Latent μ (z-score)"
        return "Raw Latent μ"

    ax_lat.set_title(_latent_panel_title(data), fontsize=9)

    state = {"playing": False, "data": data, "frame": 0}

    def _fmt_title():
        d = state["data"]
        t = state["frame"]
        acc = d["per_frame_acc"][t]
        loss = d["per_frame_loss"][t]
        mean_acc = d["per_frame_acc"][ctx:].mean() if ctx < d["T"] else d["per_frame_acc"].mean()
        ctx_str = "  [CTX]" if t < ctx else ""
        loss_str = "n/a" if np.isnan(loss) else f"{loss:.3f}"
        return (
            f"{d['clip_info']}  |  frame {t+1}/{d['T']}{ctx_str}  |  "
            f"acc={acc:.3f}  loss={loss_str}  |  mean_acc={mean_acc:.3f}"
        )

    title_text = fig.suptitle(_fmt_title(), fontsize=10, fontweight="bold")

    # Slider
    ax_slider = plt.axes([0.15, 0.08, 0.55, 0.03])
    slider = Slider(ax_slider, "Frame", 1, T, valinit=1, valstep=1, valfmt="%d")
    style_widget(slider)
    enable_slider_scroll(slider)

    # Buttons
    ax_play = plt.axes([0.75, 0.08, 0.08, 0.04])
    btn_play = Button(ax_play, "Play")
    style_widget(btn_play)

    ax_resample = plt.axes([0.85, 0.08, 0.12, 0.04])
    btn_resample = Button(ax_resample, "Resample")
    style_widget(btn_resample)

    def _map_lat_frame(t: int) -> int:
        """Map display frame index to latent time index."""
        d = state["data"]
        T_lat = d["T_lat"]
        T_vid = d["T"]
        if T_lat == T_vid:
            return t
        return min(int(t * T_lat / T_vid), T_lat - 1)

    def _update_frame(t: int) -> None:
        d = state["data"]
        state["frame"] = t
        im_tgt.set_data(d["target_rgb"][t])
        im_rec.set_data(d["recon_rgb"][t])
        im_err.set_data(d["error_rgb"][t])
        im_conf.set_data(d["confidence"][t])
        if d["weight_map"] is not None:
            im_cw.set_data(d["weight_map"][t])
        lt = _map_lat_frame(t)
        lat_grid = _latent_channel_grid(d["latent"][:, lt])
        im_lat.set_data(lat_grid)
        title_text.set_text(_fmt_title())
        fig.canvas.draw_idle()

    def on_slider(val):
        _update_frame(int(val) - 1)

    slider.on_changed(on_slider)

    def _resample():
        d = _run_inspection(
            model,
            sampler,
            palette_u8,
            class_weight,
            latent_norm,
            config,
            device,
            rng,
            chunk_frames=args.chunk_frames,
            chunk_overlap=args.chunk_overlap,
        )
        state["data"] = d
        ax_lat.set_title(_latent_panel_title(d), fontsize=9)
        # Update slider range if clip length changed (unlikely but safe)
        slider.valmax = d["T"]
        slider.set_val(1)
        # Update class weight colorbar range
        if d["weight_map"] is not None:
            im_cw.set_clim(d["weight_map"].min(), d["weight_map"].max())
        _update_frame(0)

    def on_play(event):
        if state["playing"]:
            anim.event_source.stop()
            btn_play.label.set_text("Play")
        else:
            anim.event_source.start()
            btn_play.label.set_text("Pause")
        state["playing"] = not state["playing"]

    btn_play.on_clicked(on_play)
    btn_resample.on_clicked(lambda _: _resample())

    def animate(_i):
        d = state["data"]
        next_frame = (state["frame"] + 1) % d["T"]
        _update_frame(next_frame)
        # Update slider without triggering on_slider (avoid double-draw)
        slider.eventson = False
        slider.set_val(next_frame + 1)
        slider.eventson = True
        return []

    def on_key(event):
        d = state["data"]
        if event.key in ("q", "escape"):
            return
        if event.key == "left":
            slider.set_val(max(1, int(slider.val) - 1))
        elif event.key == "right":
            slider.set_val(min(d["T"], int(slider.val) + 1))
        else:
            _resample()

    fig.canvas.mpl_connect("key_press_event", on_key)

    fps = args.fps
    anim = FuncAnimation(fig, animate, interval=1000 / fps, blit=False, cache_frame_data=False)

    # Start paused
    def _stop_after_init(_event):
        anim.event_source.stop()
        fig.canvas.mpl_disconnect(state["_stop_cid"])

    state["_stop_cid"] = fig.canvas.mpl_connect("draw_event", _stop_after_init)

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive diagnostic viewer for Video VAE reconstructions.",
    )
    p.add_argument("checkpoint", help="Checkpoint directory (must contain config.json + latest.pt or best.pt)")
    p.add_argument("--best", action="store_true", help="Load best.pt instead of latest.pt")
    p.add_argument("--data-dir", default="data/normalized", help="Dataset directory (default: data/normalized)")
    p.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    p.add_argument(
        "--predicted-frames",
        type=int,
        default=None,
        help=(
            "Number of predicted frames to inspect (default: checkpoint config value). "
            "Total sampled frames are context_frames + predicted_frames."
        ),
    )
    p.add_argument(
        "--chunk-frames",
        type=int,
        default=64,
        help="Temporal chunk size used for VAE forward pass (default: 64)",
    )
    p.add_argument(
        "--chunk-overlap",
        type=int,
        default=8,
        help="Overlap between temporal chunks in frames (default: 8)",
    )
    p.add_argument("--fps", type=int, default=8, help="Playback FPS (default: 8)")
    p.add_argument("--scale", type=float, default=3.0, help="Display scale multiplier (default: 3.0)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device (default: auto)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.checkpoint)
    if not ckpt_dir.is_dir():
        raise SystemExit(f"Not a directory: {ckpt_dir}")

    config = _load_config(ckpt_dir)
    tcfg = config["training"]
    dcfg = config["data"]
    device = _resolve_device(args.device)

    ckpt_name = "best.pt" if args.best else "latest.pt"
    ckpt_path = ckpt_dir / ckpt_name
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading {ckpt_path} …")
    model = _build_model(config, device)
    ckpt = _load_checkpoint(model, ckpt_path, device)
    step = ckpt.get("step", "?")
    print(f"  step={step}  params={sum(p.numel() for p in model.parameters()):,}")

    # Palette
    palette = load_palette_tensor(args.data_dir)
    palette_u8 = (palette.clamp(0, 1).numpy() * 255).astype(np.uint8)
    num_colors = dcfg["num_colors"]

    # Class weights
    class_weight = None
    if tcfg.get("use_class_weights", False):
        soften = tcfg.get("class_weight_soften", 0.3)
        class_weight = _load_class_weights(args.data_dir, num_classes=num_colors, soften=soften)
        if class_weight is not None:
            print(f"  class weights: soften={soften}, range=[{class_weight.min():.3f}, {class_weight.max():.3f}]")

    # Lightweight clip sampler (mmap, no full dataset load)
    context_frames = tcfg.get("context_frames", 0)
    predicted_frames = int(args.predicted_frames) if args.predicted_frames is not None else int(dcfg["clip_frames"])
    if predicted_frames <= 0:
        raise SystemExit("--predicted-frames must be > 0")
    if args.chunk_frames <= 0:
        raise SystemExit("--chunk-frames must be > 0")
    if args.chunk_overlap < 0:
        raise SystemExit("--chunk-overlap must be >= 0")
    if args.chunk_frames <= 2 * args.chunk_overlap:
        raise SystemExit("--chunk-frames must be greater than 2 * --chunk-overlap")
    clip_frames = predicted_frames + context_frames
    frame_size = dcfg.get("frame_height", dcfg.get("frame_size", 224))
    print(
        "  Indexing clips: "
        f"{args.data_dir}, clip_frames={clip_frames} "
        f"({context_frames} context + {predicted_frames} predicted), frame_size={frame_size}"
    )
    sampler = _ClipSampler(args.data_dir, clip_frames=clip_frames, frame_size=frame_size)
    print(f"  {len(sampler._valid_files)} files indexed ({len(sampler.npz_files)} total)")

    latent_norm = _load_latent_normalization(PROJECT_ROOT / "data" / "latents")
    if latent_norm is not None:
        print(f"  latent normalization: loaded {latent_norm.stats_path}")

    _show_viewer(model, sampler, palette_u8, class_weight, latent_norm, config, device, args)


if __name__ == "__main__":
    main()
