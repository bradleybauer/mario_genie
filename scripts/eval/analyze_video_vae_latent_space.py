#!/usr/bin/env python3
"""Comprehensive latent-space analysis for a Video VAE checkpoint.

Loads a trained Video VAE, encodes many clips from the dataset, and produces
a thorough set of diagnostic plots and statistics covering:

  1. **Global statistics** — per-channel mean, std, min, max, kurtosis, skewness
  2. **KL divergence** — per-channel KL, spatial KL heatmap, temporal KL profile
  3. **Posterior analysis** — mean/logvar distributions, posterior collapse detection
  4. **Channel activity & correlation** — variance-based activity ranking,
     inter-channel correlation matrix, dead/redundant channel detection
  5. **Spatial structure** — per-position energy heatmap, spatial autocorrelation
  6. **Temporal dynamics** — frame-to-frame latent velocity, temporal
     autocorrelation per channel, spectral analysis
  7. **Reconstruction quality** — accuracy vs. latent norm scatter, per-clip
     loss vs. latent statistics
  8. **Dimensionality** — PCA explained variance, effective dimensionality,
     t-SNE / UMAP 2D embedding colored by game / recording
  9. **Traversals** — single-channel interpolation reconstructions for the
     top-k most active channels
 10. **Summary report** — JSON file with all scalar metrics

Usage:
    python scripts/eval/analyze_video_vae_latent_space.py checkpoints/video_vae_20260410_134808
    python scripts/eval/analyze_video_vae_latent_space.py checkpoints/video_vae_20260410_134808 \\
        --num-clips 512 --output-dir results/latent_analysis --device cuda
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plot_style import apply_plot_style

apply_plot_style()

from src.data.normalized_dataset import NormalizedSequenceDataset, load_palette_tensor
from src.data.video_frames import resize_palette_frames
from src.models.video_vae import VideoVAE
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


def _save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    return obj


# ---------------------------------------------------------------------------
# Data collection: encode clips and gather statistics
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_latent_stats(
    model: VideoVAE,
    dataset: NormalizedSequenceDataset,
    config: dict,
    device: torch.device,
    num_clips: int,
    batch_size: int,
    seed: int,
    use_amp: bool = False,
    auto_batch_shrink: bool = True,
    min_batch_size: int = 1,
) -> dict:
    """Encode clips and collect comprehensive latent statistics.

    Returns a dict with numpy arrays of aggregated statistics.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(num_clips, len(dataset)), replace=False)
    indices = np.sort(indices)

    tcfg = config["training"]
    context_frames = tcfg.get("context_frames", 0)

    # Accumulators
    all_means: list[np.ndarray] = []       # (Z, T_lat, H_lat, W_lat)
    all_logvars: list[np.ndarray] = []
    all_latents: list[np.ndarray] = []     # sampled latents
    all_kl: list[np.ndarray] = []          # per-element KL
    all_accuracies: list[float] = []
    all_losses: list[float] = []
    all_latent_norms: list[float] = []
    all_file_idxs: list[int] = []
    # Pooled latent vectors for dimensionality analysis
    all_pooled: list[np.ndarray] = []      # (Z,)

    n_processed = 0
    t_start = time.time()

    if min_batch_size < 1:
        raise ValueError("min_batch_size must be >= 1")

    # Process in mini-batches with optional OOM recovery by shrinking batch size.
    effective_batch_size = max(batch_size, min_batch_size)
    batch_start = 0
    while batch_start < len(indices):
        current_batch_size = min(effective_batch_size, len(indices) - batch_start)
        batch_indices = indices[batch_start:batch_start + current_batch_size]
        batch_frames = []
        batch_file_idxs = []

        for idx in batch_indices:
            sample = dataset[int(idx)]
            batch_frames.append(sample["frames"])
            batch_file_idxs.append(sample.get("file_idx", -1))

        frames = None
        outputs = None
        logits = None
        mean = None
        logvar = None
        latents = None
        kl = None
        target = None
        pred_logits = None
        pred_classes = None

        try:
            # Stack into batch tensor
            frames = torch.stack(batch_frames, dim=0).to(device)  # (B, T, H, W)
            B = frames.shape[0]

            # Forward pass (AMP can significantly reduce VRAM usage on small GPUs)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp and device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                use_onehot_conv = tcfg.get("onehot_conv", False)
                if use_onehot_conv:
                    model_input = frames.byte()
                else:
                    dcfg = config["data"]
                    onehot_dtype_name = tcfg.get("onehot_dtype", "float32")
                    oh_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(onehot_dtype_name, torch.float32)
                    model_input = frames_to_one_hot(frames.long(), dcfg["num_colors"], dtype=oh_dtype)
                outputs = model(model_input, sample_posterior=True)
                logits = outputs.logits             # (B, C, T, H, W)
                mean = outputs.posterior_mean       # (B, Z, T_lat, H_lat, W_lat)
                logvar = outputs.posterior_logvar
                latents = outputs.latents

                # Per-element KL: 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
                kl = 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar)

            # Reconstruction accuracy & loss (skip context frames)
            target = frames[:, context_frames:].long()
            pred_logits = logits[:, :, context_frames:]
            pred_classes = pred_logits.argmax(dim=1)
            for b in range(B):
                acc = (pred_classes[b] == target[b]).float().mean().item()
                loss = F.cross_entropy(
                    pred_logits[b].unsqueeze(0), target[b].unsqueeze(0)
                ).item()
                lat_norm = latents[b].norm().item()

                all_accuracies.append(acc)
                all_losses.append(loss)
                all_latent_norms.append(lat_norm)
                all_file_idxs.append(batch_file_idxs[b])

            # Store per-clip latent stats (move to CPU/numpy)
            mean_np = mean.cpu().float().numpy()
            logvar_np = logvar.cpu().float().numpy()
            latents_np = latents.cpu().float().numpy()
            kl_np = kl.cpu().float().numpy()

            for b in range(B):
                all_means.append(mean_np[b])
                all_logvars.append(logvar_np[b])
                all_latents.append(latents_np[b])
                all_kl.append(kl_np[b])
                # Global average pooled latent
                all_pooled.append(latents_np[b].mean(axis=(1, 2, 3)))  # (Z,)

            n_processed += B
            batch_start += B
            elapsed = time.time() - t_start
            rate = n_processed / elapsed if elapsed > 0 else 0
            print(f"\r  Encoded {n_processed}/{len(indices)} clips ({rate:.1f} clips/s)", end="", flush=True)
        except torch.OutOfMemoryError as exc:
            if device.type != "cuda" or not auto_batch_shrink:
                raise

            if current_batch_size <= min_batch_size:
                raise RuntimeError(
                    f"CUDA OOM even at minimum batch size ({min_batch_size}). "
                    "Try --device cpu, reducing --num-clips, or enabling --amp."
                ) from exc

            next_batch_size = max(min_batch_size, current_batch_size // 2)
            if next_batch_size >= current_batch_size:
                next_batch_size = current_batch_size - 1

            print(
                f"\n  CUDA OOM at batch_size={current_batch_size}; "
                f"retrying with batch_size={next_batch_size}"
            )
            effective_batch_size = next_batch_size
            torch.cuda.empty_cache()
        finally:
            del frames, outputs, logits, mean, logvar, latents, kl, target, pred_logits, pred_classes

    print()

    # Stack all collected arrays
    all_means_arr = np.stack(all_means)       # (N, Z, T, H, W)
    all_logvars_arr = np.stack(all_logvars)
    all_latents_arr = np.stack(all_latents)
    all_kl_arr = np.stack(all_kl)
    all_pooled_arr = np.stack(all_pooled)     # (N, Z)

    return {
        "means": all_means_arr,
        "logvars": all_logvars_arr,
        "latents": all_latents_arr,
        "kl": all_kl_arr,
        "pooled": all_pooled_arr,
        "accuracies": np.array(all_accuracies),
        "losses": np.array(all_losses),
        "latent_norms": np.array(all_latent_norms),
        "file_idxs": np.array(all_file_idxs),
        "num_clips": n_processed,
        "latent_shape": all_means_arr.shape[1:],  # (Z, T, H, W)
    }


# ---------------------------------------------------------------------------
# Analysis 1: Global per-channel statistics
# ---------------------------------------------------------------------------

def analyze_global_stats(stats: dict) -> dict:
    """Compute per-channel global statistics from collected latents."""
    latents = stats["latents"]  # (N, Z, T, H, W)
    means = stats["means"]
    logvars = stats["logvars"]
    N, Z = latents.shape[0], latents.shape[1]

    # Flatten spatial+temporal for per-channel stats
    flat = latents.reshape(N, Z, -1)  # (N, Z, THW)
    all_flat = flat.reshape(N * flat.shape[2], Z).T  # (Z, N*THW)

    # For means & logvars too
    mean_flat = means.reshape(N, Z, -1)
    logvar_flat = logvars.reshape(N, Z, -1)

    channel_stats = {}
    for ch in range(Z):
        vals = all_flat[ch]
        mu_vals = mean_flat[:, ch].ravel()
        lv_vals = logvar_flat[:, ch].ravel()
        std_val = vals.std()
        channel_stats[ch] = {
            "mean": float(vals.mean()),
            "std": float(std_val),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "abs_mean": float(np.abs(vals).mean()),
            "kurtosis": float(_kurtosis(vals)),
            "skewness": float(_skewness(vals)),
            "posterior_mean_avg": float(mu_vals.mean()),
            "posterior_mean_std": float(mu_vals.std()),
            "posterior_logvar_avg": float(lv_vals.mean()),
            "posterior_std_avg": float(np.exp(0.5 * lv_vals).mean()),
        }

    return {"channel_stats": channel_stats, "num_channels": Z}


def _kurtosis(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    if s < 1e-12:
        return 0.0
    return float((((x - m) / s) ** 4).mean()) - 3.0  # excess kurtosis


def _skewness(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    if s < 1e-12:
        return 0.0
    return float((((x - m) / s) ** 3).mean())


def plot_global_stats(analysis: dict, output_dir: Path | None) -> None:
    Z = analysis["num_channels"]
    cs = analysis["channel_stats"]
    channels = list(range(Z))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Per-Channel Latent Statistics", fontweight="bold")

    # Mean & std
    ax = axes[0, 0]
    means = [cs[c]["mean"] for c in channels]
    stds = [cs[c]["std"] for c in channels]
    ax.bar(channels, means, alpha=0.7, label="Mean")
    ax.errorbar(channels, means, yerr=stds, fmt="none", color="white", alpha=0.5, capsize=2)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Value")
    ax.set_title("Latent Mean ± Std")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Posterior mean std (how much the posterior mean varies across data)
    ax = axes[0, 1]
    pm_stds = [cs[c]["posterior_mean_std"] for c in channels]
    ax.bar(channels, pm_stds, alpha=0.7, color="C1")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Std of posterior mean")
    ax.set_title("Posterior Mean Variation\n(higher = more informative)")
    ax.grid(True, alpha=0.3)

    # Average posterior std (how uncertain the encoder is)
    ax = axes[0, 2]
    p_stds = [cs[c]["posterior_std_avg"] for c in channels]
    ax.bar(channels, p_stds, alpha=0.7, color="C2")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="Prior std=1")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Avg posterior std")
    ax.set_title("Posterior Uncertainty\n(≈1 → collapsed to prior)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Kurtosis
    ax = axes[1, 0]
    kurts = [cs[c]["kurtosis"] for c in channels]
    colors = ["C3" if abs(k) > 3 else "C0" for k in kurts]
    ax.bar(channels, kurts, alpha=0.7, color=colors)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Excess Kurtosis")
    ax.set_title("Kurtosis (0 = Gaussian)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Skewness
    ax = axes[1, 1]
    skews = [cs[c]["skewness"] for c in channels]
    colors = ["C3" if abs(s) > 1 else "C0" for s in skews]
    ax.bar(channels, skews, alpha=0.7, color=colors)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Skewness")
    ax.set_title("Skewness (0 = symmetric)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Range (min/max)
    ax = axes[1, 2]
    mins = [cs[c]["min"] for c in channels]
    maxs = [cs[c]["max"] for c in channels]
    ax.bar(channels, maxs, alpha=0.5, label="Max", color="C1")
    ax.bar(channels, mins, alpha=0.5, label="Min", color="C4")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Value")
    ax.set_title("Latent Value Range")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        _save_fig(fig, output_dir, "01_global_channel_stats")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Analysis 2: KL divergence analysis
# ---------------------------------------------------------------------------

def analyze_kl(stats: dict) -> dict:
    kl = stats["kl"]  # (N, Z, T, H, W)
    N, Z, T, H, W = kl.shape

    # Per-channel total KL (averaged over spatial/temporal, summed across clips)
    kl_per_channel = kl.mean(axis=(0, 2, 3, 4))  # (Z,)

    # Spatial KL heatmap (averaged over channels and time)
    kl_spatial = kl.mean(axis=(0, 1, 2))  # (H, W)

    # Temporal KL profile (averaged over channels and spatial)
    kl_temporal = kl.mean(axis=(0, 1, 3, 4))  # (T,)

    # Total KL per clip
    kl_per_clip = kl.sum(axis=(1, 2, 3, 4)) / (T * H * W)  # (N,)

    # Fraction of total KL per channel
    total_kl = kl_per_channel.sum()
    kl_fraction = kl_per_channel / (total_kl + 1e-12)

    return {
        "kl_per_channel": kl_per_channel,
        "kl_spatial": kl_spatial,
        "kl_temporal": kl_temporal,
        "kl_per_clip": kl_per_clip,
        "kl_fraction": kl_fraction,
        "total_kl_mean": float(total_kl),
    }


def plot_kl(analysis: dict, output_dir: Path | None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("KL Divergence Analysis", fontweight="bold")

    # Per-channel KL
    ax = axes[0, 0]
    kl_ch = analysis["kl_per_channel"]
    Z = len(kl_ch)
    sorted_idx = np.argsort(kl_ch)[::-1]
    ax.bar(range(Z), kl_ch[sorted_idx], alpha=0.8)
    ax.set_xticks(range(Z))
    ax.set_xticklabels([str(i) for i in sorted_idx], fontsize=7)
    ax.set_xlabel("Channel (sorted by KL)")
    ax.set_ylabel("Mean KL")
    ax.set_title(f"Per-Channel KL (total={analysis['total_kl_mean']:.4f})")
    ax.grid(True, alpha=0.3)

    # KL fraction (cumulative)
    ax = axes[0, 1]
    kl_frac_sorted = analysis["kl_fraction"][sorted_idx]
    cumulative = np.cumsum(kl_frac_sorted)
    ax.bar(range(Z), kl_frac_sorted, alpha=0.6, label="Fraction")
    ax.plot(range(Z), cumulative, color="C1", linewidth=2, marker="o", markersize=3, label="Cumulative")
    ax.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="90%")
    ax.set_xticks(range(Z))
    ax.set_xticklabels([str(i) for i in sorted_idx], fontsize=7)
    ax.set_xlabel("Channel (sorted by KL)")
    ax.set_ylabel("Fraction of total KL")
    ax.set_title("KL Concentration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Spatial KL heatmap
    ax = axes[1, 0]
    im = ax.imshow(analysis["kl_spatial"], cmap="hot", interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spatial KL Heatmap (avg over channels & time)")

    # Temporal KL profile
    ax = axes[1, 1]
    kl_t = analysis["kl_temporal"]
    ax.plot(range(len(kl_t)), kl_t, marker="o", markersize=4)
    ax.set_xlabel("Latent time step")
    ax.set_ylabel("Mean KL")
    ax.set_title("Temporal KL Profile")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        _save_fig(fig, output_dir, "02_kl_divergence")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Analysis 3: Posterior collapse detection
# ---------------------------------------------------------------------------

def analyze_posterior_collapse(stats: dict) -> dict:
    means = stats["means"]      # (N, Z, T, H, W)
    logvars = stats["logvars"]  # (N, Z, T, H, W)
    N, Z = means.shape[0], means.shape[1]

    # A channel is "collapsed" if its posterior ≈ prior N(0,1):
    # - posterior mean ≈ 0 across all data points
    # - posterior logvar ≈ 0 (std ≈ 1)
    # We measure AU (Active Units): channels whose posterior mean has
    # variance > threshold across data points.
    posterior_mean_per_clip = means.mean(axis=(2, 3, 4))  # (N, Z)
    au_variance = posterior_mean_per_clip.var(axis=0)      # (Z,)

    au_threshold = 0.01
    active_units = au_variance > au_threshold

    # Average posterior std per channel
    avg_std = np.exp(0.5 * logvars).mean(axis=(0, 2, 3, 4))  # (Z,)

    # Mutual information lower bound: KL[q(z|x) || p(z)] - KL[q(z) || p(z)]
    # Approximate via variance of posterior mean
    mi_lower_bound = 0.5 * np.log(1 + au_variance)  # per channel

    return {
        "au_variance": au_variance,
        "active_units": active_units,
        "num_active": int(active_units.sum()),
        "num_total": Z,
        "au_threshold": au_threshold,
        "avg_posterior_std": avg_std,
        "mi_lower_bound": mi_lower_bound,
    }


def plot_posterior_collapse(analysis: dict, output_dir: Path | None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Posterior Collapse Detection — {analysis['num_active']}/{analysis['num_total']} active units",
        fontweight="bold",
    )

    Z = analysis["num_total"]

    # AU variance
    ax = axes[0]
    au_var = analysis["au_variance"]
    colors = ["C0" if a else "C3" for a in analysis["active_units"]]
    ax.bar(range(Z), au_var, color=colors, alpha=0.8)
    ax.axhline(analysis["au_threshold"], color="red", linestyle="--", label=f"Threshold={analysis['au_threshold']}")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Var[E[z|x]]")
    ax.set_title("Active Unit Variance\n(red = collapsed)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Average posterior std
    ax = axes[1]
    ax.bar(range(Z), analysis["avg_posterior_std"], alpha=0.8, color="C2")
    ax.axhline(1.0, color="red", linestyle="--", label="Prior std=1")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Avg posterior std")
    ax.set_title("Posterior Std per Channel")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MI lower bound
    ax = axes[2]
    mi = analysis["mi_lower_bound"]
    ax.bar(range(Z), mi, alpha=0.8, color="C4")
    ax.set_xlabel("Channel")
    ax.set_ylabel("MI lower bound (nats)")
    ax.set_title("Mutual Information I(x; z_c)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        _save_fig(fig, output_dir, "03_posterior_collapse")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Analysis 4: Channel activity & correlation
# ---------------------------------------------------------------------------

def analyze_channel_correlation(stats: dict) -> dict:
    pooled = stats["pooled"]  # (N, Z)
    N, Z = pooled.shape

    # Per-channel variance (activity measure)
    channel_var = pooled.var(axis=0)  # (Z,)

    # Correlation matrix
    # Center the data
    centered = pooled - pooled.mean(axis=0, keepdims=True)
    stds = centered.std(axis=0, keepdims=True)
    stds = np.where(stds < 1e-12, 1.0, stds)
    normalized = centered / stds
    corr = (normalized.T @ normalized) / N  # (Z, Z)

    # Off-diagonal average absolute correlation
    mask = ~np.eye(Z, dtype=bool)
    avg_abs_corr = np.abs(corr[mask]).mean()

    # Find highly correlated pairs
    redundant_pairs = []
    for i in range(Z):
        for j in range(i + 1, Z):
            if abs(corr[i, j]) > 0.8:
                redundant_pairs.append((i, j, float(corr[i, j])))

    return {
        "channel_variance": channel_var,
        "correlation_matrix": corr,
        "avg_abs_correlation": float(avg_abs_corr),
        "redundant_pairs": redundant_pairs,
    }


def plot_channel_correlation(analysis: dict, output_dir: Path | None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Channel Activity & Correlation", fontweight="bold")

    Z = len(analysis["channel_variance"])

    # Channel variance (activity)
    ax = axes[0]
    var = analysis["channel_variance"]
    sorted_idx = np.argsort(var)[::-1]
    ax.bar(range(Z), var[sorted_idx], alpha=0.8)
    ax.set_xticks(range(Z))
    ax.set_xticklabels([str(i) for i in sorted_idx], fontsize=7)
    ax.set_xlabel("Channel (sorted by variance)")
    ax.set_ylabel("Variance")
    ax.set_title("Channel Activity (pooled latent variance)")
    ax.grid(True, alpha=0.3)

    # Correlation matrix
    ax = axes[1]
    im = ax.imshow(analysis["correlation_matrix"], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    ax.set_title(f"Correlation Matrix (avg |r|={analysis['avg_abs_correlation']:.3f})")

    # Correlation distribution
    ax = axes[2]
    corr = analysis["correlation_matrix"]
    mask = ~np.eye(Z, dtype=bool)
    off_diag = corr[mask]
    ax.hist(off_diag, bins=50, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Correlation coefficient")
    ax.set_ylabel("Count")
    ax.set_title("Off-Diagonal Correlation Distribution")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        _save_fig(fig, output_dir, "04_channel_correlation")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Analysis 5: Spatial structure
# ---------------------------------------------------------------------------

def analyze_spatial(stats: dict) -> dict:
    latents = stats["latents"]  # (N, Z, T, H, W)
    N, Z, T, H, W = latents.shape

    # Per-position energy (variance across clips, channels, time)
    energy = latents.var(axis=(0, 1, 2))  # (H, W)

    # Per-channel spatial energy
    channel_spatial_energy = latents.var(axis=(0, 2))  # (Z, H, W)

    return {
        "spatial_energy": energy,
        "channel_spatial_energy": channel_spatial_energy,
        "H": H,
        "W": W,
    }


def plot_spatial(analysis: dict, output_dir: Path | None) -> None:
    Z_channels = analysis["channel_spatial_energy"].shape[0]
    cols = min(8, Z_channels)
    rows = (Z_channels + cols - 1) // cols

    fig = plt.figure(figsize=(2.5 * cols + 2, 2.5 * (rows + 1) + 1))
    gs = GridSpec(rows + 1, cols + 1, width_ratios=[1] * cols + [0.05], figure=fig)
    fig.suptitle("Spatial Latent Structure", fontweight="bold")

    # Overall spatial energy
    ax_main = fig.add_subplot(gs[0, :cols])
    im = ax_main.imshow(analysis["spatial_energy"], cmap="viridis", interpolation="nearest")
    ax_main.set_title("Overall Spatial Energy (variance across clips/channels/time)")
    fig.colorbar(im, cax=fig.add_subplot(gs[0, -1]))

    # Per-channel spatial energy
    ch_energy = analysis["channel_spatial_energy"]
    vmin, vmax = ch_energy.min(), ch_energy.max()
    for ch in range(Z_channels):
        row = 1 + ch // cols
        col = ch % cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(ch_energy[ch], cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Ch {ch}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for idx in range(Z_channels, rows * cols):
        row = 1 + idx // cols
        col = idx % cols
        fig.add_subplot(gs[row, col]).set_visible(False)

    plt.tight_layout()
    if output_dir:
        _save_fig(fig, output_dir, "05_spatial_structure")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Analysis 6: Temporal dynamics
# ---------------------------------------------------------------------------

def analyze_temporal(stats: dict) -> dict:
    latents = stats["latents"]  # (N, Z, T, H, W)
    N, Z, T, H, W = latents.shape

    # Frame-to-frame velocity (L2 norm of difference in latent space)
    if T > 1:
        diffs = latents[:, :, 1:] - latents[:, :, :-1]  # (N, Z, T-1, H, W)
        velocity = np.sqrt((diffs ** 2).sum(axis=1)).mean(axis=(2, 3))  # (N, T-1)
        mean_velocity = velocity.mean(axis=0)  # (T-1,)
        velocity_per_clip = velocity.mean(axis=1)  # (N,)
    else:
        mean_velocity = np.zeros(0)
        velocity_per_clip = np.zeros(N)

    # Temporal autocorrelation per channel (using pooled spatial)
    # Pool spatial dims first
    pooled_t = latents.mean(axis=(3, 4))  # (N, Z, T)

    max_lag = min(T - 1, 8)
    autocorr = np.zeros((Z, max_lag))
    for ch in range(Z):
        for lag in range(1, max_lag + 1):
            x = pooled_t[:, ch, :-lag].ravel()
            y = pooled_t[:, ch, lag:].ravel()
            if x.std() < 1e-12 or y.std() < 1e-12:
                autocorr[ch, lag - 1] = 0
            else:
                autocorr[ch, lag - 1] = np.corrcoef(x, y)[0, 1]

    return {
        "mean_velocity": mean_velocity,
        "velocity_per_clip": velocity_per_clip,
        "autocorrelation": autocorr,
        "max_lag": max_lag,
    }


def plot_temporal(analysis: dict, output_dir: Path | None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Temporal Dynamics", fontweight="bold")

    # Mean velocity over time
    ax = axes[0]
    vel = analysis["mean_velocity"]
    if len(vel) > 0:
        ax.plot(range(1, len(vel) + 1), vel, marker="o", markersize=4)
        ax.set_xlabel("Time step transition")
        ax.set_ylabel("Mean L2 velocity")
        ax.set_title("Frame-to-Frame Latent Velocity")
    else:
        ax.text(0.5, 0.5, "T=1, no temporal dynamics", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    # Velocity distribution across clips
    ax = axes[1]
    vpc = analysis["velocity_per_clip"]
    if len(vpc) > 0:
        ax.hist(vpc, bins=40, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axvline(vpc.mean(), color="C1", linestyle="--", label=f"Mean={vpc.mean():.3f}")
        ax.set_xlabel("Mean latent velocity")
        ax.set_ylabel("Count")
        ax.set_title("Velocity Distribution Across Clips")
        ax.legend()
    ax.grid(True, alpha=0.3)

    # Temporal autocorrelation heatmap
    ax = axes[2]
    ac = analysis["autocorrelation"]
    if ac.shape[1] > 0:
        im = ax.imshow(ac, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Channel")
        ax.set_xticks(range(ac.shape[1]))
        ax.set_xticklabels(range(1, ac.shape[1] + 1))
        ax.set_title("Temporal Autocorrelation per Channel")
        fig.colorbar(im, ax=ax, shrink=0.8)
    else:
        ax.text(0.5, 0.5, "T=1", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    if output_dir:
        _save_fig(fig, output_dir, "06_temporal_dynamics")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Analysis 7: Reconstruction quality vs. latent properties
# ---------------------------------------------------------------------------

def plot_reconstruction_vs_latent(stats: dict, output_dir: Path | None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Reconstruction Quality vs. Latent Properties", fontweight="bold")

    accs = stats["accuracies"]
    losses = stats["losses"]
    norms = stats["latent_norms"]

    # Accuracy vs latent norm
    ax = axes[0]
    ax.scatter(norms, accs, alpha=0.3, s=8, edgecolors="none")
    ax.set_xlabel("Latent L2 norm")
    ax.set_ylabel("Reconstruction accuracy")
    ax.set_title("Accuracy vs. Latent Norm")
    ax.grid(True, alpha=0.3)

    # Loss vs latent norm
    ax = axes[1]
    ax.scatter(norms, losses, alpha=0.3, s=8, edgecolors="none")
    ax.set_xlabel("Latent L2 norm")
    ax.set_ylabel("CE Loss")
    ax.set_title("Loss vs. Latent Norm")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Accuracy histogram
    ax = axes[2]
    ax.hist(accs, bins=50, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(accs.mean(), color="C1", linestyle="--", label=f"Mean={accs.mean():.4f}")
    ax.axvline(np.median(accs), color="C2", linestyle="--", label=f"Median={np.median(accs):.4f}")
    ax.set_xlabel("Reconstruction accuracy")
    ax.set_ylabel("Count")
    ax.set_title("Accuracy Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        _save_fig(fig, output_dir, "07_recon_vs_latent")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Analysis 8: Dimensionality (PCA)
# ---------------------------------------------------------------------------

def analyze_dimensionality(stats: dict) -> dict:
    pooled = stats["pooled"]  # (N, Z)
    N, Z = pooled.shape

    # PCA via SVD
    centered = pooled - pooled.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained_var = S ** 2 / (N - 1)
    total_var = explained_var.sum()
    explained_ratio = explained_var / (total_var + 1e-12)
    cumulative = np.cumsum(explained_ratio)

    # Effective dimensionality: number of PCs to reach 95% variance
    dim_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    dim_99 = int(np.searchsorted(cumulative, 0.99) + 1)

    # Participation ratio (another measure of effective dim)
    participation_ratio = (explained_var.sum() ** 2) / ((explained_var ** 2).sum() + 1e-12)

    # 2D PCA projection for plotting
    pca_2d = (centered @ Vt[:2].T)  # (N, 2)

    return {
        "explained_ratio": explained_ratio,
        "cumulative_ratio": cumulative,
        "singular_values": S,
        "dim_95": dim_95,
        "dim_99": dim_99,
        "participation_ratio": float(participation_ratio),
        "pca_2d": pca_2d,
        "pca_components": Vt,
    }


def plot_dimensionality(analysis: dict, stats: dict, output_dir: Path | None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Dimensionality — Eff. dim (95%): {analysis['dim_95']}, "
        f"Participation ratio: {analysis['participation_ratio']:.1f}",
        fontweight="bold",
    )

    Z = len(analysis["explained_ratio"])

    # Explained variance
    ax = axes[0]
    ax.bar(range(Z), analysis["explained_ratio"], alpha=0.7, label="Individual")
    ax.plot(range(Z), analysis["cumulative_ratio"], color="C1", linewidth=2, label="Cumulative")
    ax.axhline(0.95, color="red", linestyle="--", alpha=0.5, label="95%")
    ax.axhline(0.99, color="orange", linestyle="--", alpha=0.5, label="99%")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Explained Variance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Singular value spectrum
    ax = axes[1]
    ax.plot(range(Z), analysis["singular_values"], marker="o", markersize=4)
    ax.set_xlabel("Component")
    ax.set_ylabel("Singular Value")
    ax.set_title("Singular Value Spectrum")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # 2D PCA scatter colored by file index
    ax = axes[2]
    pca_2d = analysis["pca_2d"]
    file_idxs = stats["file_idxs"]
    unique_files = np.unique(file_idxs)
    if len(unique_files) <= 20:
        for fi in unique_files:
            mask = file_idxs == fi
            ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1], s=8, alpha=0.5, label=f"File {fi}")
        if len(unique_files) <= 10:
            ax.legend(fontsize=6, markerscale=2)
    else:
        scatter = ax.scatter(pca_2d[:, 0], pca_2d[:, 1], c=file_idxs, s=8, alpha=0.5, cmap="tab20")
        fig.colorbar(scatter, ax=ax, shrink=0.8, label="File idx")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("2D PCA of Pooled Latents")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        _save_fig(fig, output_dir, "08_dimensionality")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Analysis 9: Latent distribution shape
# ---------------------------------------------------------------------------

def plot_latent_distributions(stats: dict, top_k: int, output_dir: Path | None) -> None:
    """Plot histograms of latent values for the most active channels."""
    pooled = stats["pooled"]  # (N, Z)
    Z = pooled.shape[1]
    channel_var = pooled.var(axis=0)
    top_channels = np.argsort(channel_var)[::-1][:top_k]

    latents = stats["latents"]  # (N, Z, T, H, W)
    cols = min(5, top_k)
    rows = (top_k + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    fig.suptitle(f"Latent Value Distributions — Top {top_k} Active Channels", fontweight="bold")

    for idx, ch in enumerate(top_channels):
        ax = axes[idx // cols][idx % cols]
        vals = latents[:, ch].ravel()
        # Subsample if too many values
        if len(vals) > 500_000:
            vals = np.random.default_rng(42).choice(vals, 500_000, replace=False)
        ax.hist(vals, bins=80, alpha=0.7, density=True, edgecolor="white", linewidth=0.3)
        # Overlay standard normal for reference
        x = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(x, np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi), color="red", alpha=0.6,
                linestyle="--", label="N(0,1)")
        ax.set_title(f"Channel {ch} (var={channel_var[ch]:.3f})", fontsize=9)
        ax.set_xlabel("z")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(top_k, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    if output_dir:
        _save_fig(fig, output_dir, "09_latent_distributions")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Analysis 10: Latent traversals
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_traversals(
    model: VideoVAE,
    stats: dict,
    config: dict,
    palette_u8: np.ndarray,
    device: torch.device,
    top_k: int,
    output_dir: Path | None,
) -> None:
    """Traverse individual latent channels and show reconstructions."""
    pooled = stats["pooled"]
    Z = pooled.shape[1]
    channel_var = pooled.var(axis=0)
    top_channels = np.argsort(channel_var)[::-1][:top_k]

    # Use the mean latent as the base point
    mean_latent = stats["means"].mean(axis=0)  # (Z, T, H, W)
    base = torch.from_numpy(mean_latent).unsqueeze(0).to(device)  # (1, Z, T, H, W)

    tcfg = config["training"]
    context_frames = tcfg.get("context_frames", 0)

    n_steps = 7
    alphas = np.linspace(-3, 3, n_steps)
    t_show = mean_latent.shape[1] // 2  # Show middle time step

    fig, axes = plt.subplots(top_k, n_steps, figsize=(2.5 * n_steps, 2.5 * top_k), squeeze=False)
    fig.suptitle("Latent Channel Traversals (middle frame)", fontweight="bold")

    for row, ch in enumerate(top_channels):
        ch_std = float(np.sqrt(channel_var[ch]))
        for col, alpha in enumerate(alphas):
            z = base.clone()
            z[0, ch] = z[0, ch] + alpha * ch_std
            logits = model.decode(z.float())
            pred = logits[0, :, t_show].argmax(dim=0).cpu().numpy()
            rgb = palette_u8[pred]
            axes[row][col].imshow(rgb, interpolation="nearest")
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
            if row == 0:
                axes[row][col].set_title(f"α={alpha:.1f}", fontsize=8)
        axes[row][0].set_ylabel(f"Ch {ch}", fontsize=9)

    plt.tight_layout()
    if output_dir:
        _save_fig(fig, output_dir, "10_latent_traversals")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def build_summary(
    config: dict,
    stats: dict,
    global_stats: dict,
    kl_analysis: dict,
    collapse_analysis: dict,
    corr_analysis: dict,
    temporal_analysis: dict,
    dim_analysis: dict,
    elapsed: float,
) -> dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "model_name": config.get("model_name", "unknown"),
        "latent_shape": list(stats["latent_shape"]),
        "num_clips_analyzed": stats["num_clips"],
        "elapsed_seconds": round(elapsed, 1),
        "reconstruction": {
            "mean_accuracy": float(stats["accuracies"].mean()),
            "median_accuracy": float(np.median(stats["accuracies"])),
            "std_accuracy": float(stats["accuracies"].std()),
            "mean_loss": float(stats["losses"].mean()),
        },
        "kl": {
            "total_kl_mean": kl_analysis["total_kl_mean"],
            "kl_per_channel": kl_analysis["kl_per_channel"].tolist(),
        },
        "posterior_collapse": {
            "active_units": collapse_analysis["num_active"],
            "total_units": collapse_analysis["num_total"],
            "au_threshold": collapse_analysis["au_threshold"],
            "participation_ratio": dim_analysis["participation_ratio"],
        },
        "dimensionality": {
            "effective_dim_95": dim_analysis["dim_95"],
            "effective_dim_99": dim_analysis["dim_99"],
            "participation_ratio": dim_analysis["participation_ratio"],
        },
        "correlation": {
            "avg_abs_correlation": corr_analysis["avg_abs_correlation"],
            "redundant_pairs": corr_analysis["redundant_pairs"],
        },
        "temporal": {
            "mean_velocity": float(temporal_analysis["velocity_per_clip"].mean()) if len(temporal_analysis["velocity_per_clip"]) > 0 else 0,
        },
        "channel_stats": global_stats["channel_stats"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Comprehensive latent-space analysis for a Video VAE checkpoint.",
    )
    p.add_argument("checkpoint", help="Checkpoint directory (must contain config.json + latest.pt or best.pt)")
    p.add_argument("--best", action="store_true", help="Load best.pt instead of latest.pt")
    p.add_argument("--data-dir", default=None, help="Dataset directory (default: from config)")
    p.add_argument("--num-clips", type=int, default=256, help="Number of clips to encode (default: 256)")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for encoding (default: 16)")
    p.add_argument("--min-batch-size", type=int, default=1,
                   help="Minimum batch size when auto-shrinking after CUDA OOM (default: 1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--top-k", type=int, default=8, help="Top-K channels for distributions & traversals (default: 8)")
    p.add_argument("--output-dir", type=str, default=None, help="Save plots & summary to this directory (default: show interactively)")
    p.add_argument("--no-traversals", action="store_true", help="Skip latent traversals (faster)")
    p.add_argument("--amp", action="store_true",
                   help="Use CUDA AMP (float16) for model forward pass during encoding")
    p.add_argument("--no-auto-batch-shrink", action="store_true",
                   help="Disable automatic batch-size reduction on CUDA OOM")
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

    # --- Load model ---
    print(f"Loading {ckpt_path} …")
    model = _build_model(config, device)
    ckpt = _load_checkpoint(model, ckpt_path, device)
    step = ckpt.get("step", "?")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  step={step}  params={n_params:,}")

    # --- Load palette ---
    data_dir = args.data_dir or tcfg.get("data_dir", "data/normalized")
    palette = load_palette_tensor(data_dir)
    palette_u8 = (palette.clamp(0, 1).numpy() * 255).astype(np.uint8)

    # --- Build dataset ---
    clip_frames = dcfg["clip_frames"]
    frame_size = dcfg.get("frame_height", dcfg.get("frame_size", 224))
    context_frames = tcfg.get("context_frames", 0)

    print(f"  Loading dataset from {data_dir} (clip_frames={clip_frames}, frame_size={frame_size})")
    dataset = NormalizedSequenceDataset(
        data_dir=data_dir,
        clip_frames=clip_frames,
        frame_size=frame_size,
        stride=clip_frames,  # Non-overlapping for diverse samples
    )
    print(f"  Dataset: {len(dataset)} clips from {len(dataset.data_files)} files")

    # --- Output directory ---
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output directory: {output_dir}")

    # --- Collect latent statistics ---
    print(f"\nEncoding {args.num_clips} clips …")
    if device.type == "cuda":
        print(
            "  CUDA encode settings: "
            f"batch_size={args.batch_size}, min_batch_size={args.min_batch_size}, "
            f"amp={args.amp}, auto_batch_shrink={not args.no_auto_batch_shrink}"
        )
    t0 = time.time()
    stats = collect_latent_stats(
        model, dataset, config, device,
        num_clips=args.num_clips,
        batch_size=args.batch_size,
        seed=args.seed,
        use_amp=args.amp,
        auto_batch_shrink=not args.no_auto_batch_shrink,
        min_batch_size=args.min_batch_size,
    )
    encode_time = time.time() - t0
    Z, T_lat, H_lat, W_lat = stats["latent_shape"]
    print(f"  Latent shape: ({Z}, {T_lat}, {H_lat}, {W_lat})")
    print(f"  Encoding took {encode_time:.1f}s")

    # --- Run analyses ---
    print("\nAnalyzing …")

    print("  [1/9] Global per-channel statistics")
    global_analysis = analyze_global_stats(stats)
    plot_global_stats(global_analysis, output_dir)

    print("  [2/9] KL divergence")
    kl_analysis = analyze_kl(stats)
    plot_kl(kl_analysis, output_dir)

    print("  [3/9] Posterior collapse detection")
    collapse_analysis = analyze_posterior_collapse(stats)
    plot_posterior_collapse(collapse_analysis, output_dir)

    print("  [4/9] Channel activity & correlation")
    corr_analysis = analyze_channel_correlation(stats)
    plot_channel_correlation(corr_analysis, output_dir)

    print("  [5/9] Spatial structure")
    spatial_analysis = analyze_spatial(stats)
    plot_spatial(spatial_analysis, output_dir)

    print("  [6/9] Temporal dynamics")
    temporal_analysis = analyze_temporal(stats)
    plot_temporal(temporal_analysis, output_dir)

    print("  [7/9] Reconstruction vs. latent properties")
    plot_reconstruction_vs_latent(stats, output_dir)

    print("  [8/9] Dimensionality (PCA)")
    dim_analysis = analyze_dimensionality(stats)
    plot_dimensionality(dim_analysis, stats, output_dir)

    print("  [9/9] Latent value distributions")
    plot_latent_distributions(stats, top_k=args.top_k, output_dir=output_dir)

    # --- Traversals (optional, requires decode) ---
    if not args.no_traversals:
        print("\n  [Bonus] Latent traversals")
        plot_traversals(model, stats, config, palette_u8, device, top_k=min(args.top_k, 6), output_dir=output_dir)

    # --- Summary ---
    total_elapsed = time.time() - t0
    summary = build_summary(
        config, stats, global_analysis, kl_analysis,
        collapse_analysis, corr_analysis, temporal_analysis,
        dim_analysis, total_elapsed,
    )

    print(f"\n{'='*60}")
    print(f"  Clips analyzed:       {stats['num_clips']}")
    print(f"  Latent shape:         ({Z}, {T_lat}, {H_lat}, {W_lat})")
    print(f"  Mean accuracy:        {stats['accuracies'].mean():.4f}")
    print(f"  Mean loss:            {stats['losses'].mean():.4f}")
    print(f"  Total KL (mean):      {kl_analysis['total_kl_mean']:.4f}")
    print(f"  Active units:         {collapse_analysis['num_active']}/{collapse_analysis['num_total']}")
    print(f"  Eff. dim (95%):       {dim_analysis['dim_95']}")
    print(f"  Participation ratio:  {dim_analysis['participation_ratio']:.1f}")
    print(f"  Avg |correlation|:    {corr_analysis['avg_abs_correlation']:.4f}")
    if corr_analysis["redundant_pairs"]:
        print(f"  Redundant pairs:      {len(corr_analysis['redundant_pairs'])} (|r| > 0.8)")
    print(f"  Total time:           {total_elapsed:.1f}s")
    print(f"{'='*60}")

    if output_dir:
        summary_path = output_dir / "summary.json"
        with summary_path.open("w") as f:
            json.dump(_to_serializable(summary), f, indent=2)
        print(f"\n  Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
