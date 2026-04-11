#!/usr/bin/env python3
"""Visualize per-pixel class weights as a heatmap for a random clip.

Loads one random clip from the normalized dataset, looks up the class weight
for every pixel's palette index, and displays the original frames alongside
the weight heatmaps.

Usage:
    python scripts/eval/view_class_weight_heatmap.py
    python scripts/eval/view_class_weight_heatmap.py --data-dir data/normalized --soften 0.3
    python scripts/eval/view_class_weight_heatmap.py --clip-frames 16 --frame-size 224
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plot_style import apply_plot_style, enable_slider_scroll, style_image_axes, style_widget
apply_plot_style()

from src.data.normalized_dataset import load_palette_tensor
from src.data.video_frames import resize_palette_frames
from src.training.losses import softened_inverse_frequency_weights


def soft_max_pool(weight_maps: np.ndarray, radius: float, hardness: float = 5.0,
                  temporal_alpha: float = 0.0) -> np.ndarray:
    """LogSumExp pooling: smooth approximation of local max over a circular region.

    For each pixel, computes (1/β)·log(mean(exp(β·x))) over the circular
    neighbourhood, where β = *hardness*.  As hardness → 0 this is the local
    mean; as hardness → ∞ it approaches the hard local max.  The result is
    always smooth and >= the local mean (typically close to the local max for
    moderate hardness).

    Temporal smoothing is applied causally: each frame is an exponential
    moving average (in LogSumExp space) over past frames, so no future
    information leaks backward.
    """
    if radius < 0.5:
        return weight_maps
    r = int(np.ceil(radius))
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    footprint = (x * x + y * y) <= radius * radius
    kernel = footprint.astype(np.float32)
    kernel /= kernel.sum()

    # Batch spatial LogSumExp: operate on full (T, H, W) at once.
    # maximum_filter with a 2D-only footprint (no temporal axis).
    fp3d = footprint[np.newaxis, :, :]  # (1, k, k)
    local_max = maximum_filter(weight_maps, footprint=fp3d)
    shifted = hardness * (weight_maps - local_max)
    exp_shifted = np.exp(shifted)
    # 3D fftconvolve with a (1, k, k) kernel applies spatial conv per frame.
    kernel3d = kernel[np.newaxis, :, :]
    avg_exp = fftconvolve(exp_shifted, kernel3d, mode="same")
    np.clip(avg_exp, 1e-30, None, out=avg_exp)
    out = local_max + np.log(avg_exp) / hardness

    # Optional causal temporal persistence.  At each pixel, keep the max of
    # the current value and the decayed accumulator — so the current frame is
    # never diminished, but past high values linger and fade out.
    if temporal_alpha > 0:
        for t in range(1, out.shape[0]):
            np.maximum(out[t], temporal_alpha * out[t - 1], out=out[t])

    return out


def load_class_weights(data_dir: str, *, num_classes: int, soften: float) -> torch.Tensor:
    import json
    dist_path = Path(data_dir) / "palette_distribution.json"
    if not dist_path.is_file():
        raise FileNotFoundError(f"Not found: {dist_path}")
    with dist_path.open() as f:
        dist = json.load(f)
    counts = torch.tensor(dist.get("counts") or dist["probabilities"], dtype=torch.float32)
    if counts.numel() != num_classes:
        raise ValueError(f"Expected {num_classes} classes, got {counts.numel()}")
    return softened_inverse_frequency_weights(counts, soften=soften)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize class weights as a heatmap on a random clip.")
    p.add_argument("--data-dir", type=str, default="data/normalized")
    p.add_argument("--clip-frames", type=int, default=16)
    p.add_argument("--frame-size", type=int, default=224)
    p.add_argument("--soften", type=float, default=0.3,
                   help="Softening exponent for inverse-frequency weights (0=uniform, 1=full).")
    p.add_argument("--seed", type=int, default=None, help="Random seed (default: random).")
    p.add_argument("--fps", type=int, default=8, help="Playback speed in frames per second.")
    p.add_argument("--scale", type=float, default=3.0, help="Display scale multiplier.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    palette = load_palette_tensor(args.data_dir)  # (C, 3) float in [0,1]
    num_colors = palette.shape[0]

    weights = load_class_weights(args.data_dir, num_classes=num_colors, soften=args.soften)
    print(f"Class weights (soften={args.soften}): min={weights.min():.3f}, max={weights.max():.3f}")
    for i in range(num_colors):
        rgb = (palette[i] * 255).to(torch.uint8).tolist()
        print(f"  [{i:2d}] weight={weights[i]:.4f}  rgb={rgb}")

    # Load a single random clip directly via mmap to avoid reading all data into RAM.
    data_dir = Path(args.data_dir)
    palette_np = (palette * 255).to(torch.uint8).numpy()  # (C, 3)
    weights_np = weights.numpy()  # (C,)
    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No .npz files found in {data_dir}")

    rng = np.random.default_rng(args.seed)

    def _load_random_clip():
        """Sample a random clip, returning (rgb_frames, weight_maps, palette_frames, info_str)."""
        fidx = int(rng.integers(0, len(npz_files)))
        npz = np.load(npz_files[fidx], mmap_mode="r")
        af = npz["frames"]
        n = af.shape[0]
        if n < args.clip_frames:
            raise RuntimeError(f"{npz_files[fidx].name} has only {n} frames")
        s = int(rng.integers(0, n - args.clip_frames + 1))
        fr = np.array(af[s:s + args.clip_frames], dtype=np.uint8)
        fr = resize_palette_frames(fr, target_height=args.frame_size, target_width=args.frame_size)
        rgb = palette_np[fr]
        wm = weights_np[fr]
        info = f"{npz_files[fidx].name} [{s}:{s+args.clip_frames}]"
        return rgb, wm, fr, info

    rgb_frames, weight_maps, pal_frames, clip_info = _load_random_clip()
    T, H, W = weight_maps.shape
    print(f"Clip: {clip_info}, {T}x{H}x{W}")

    vmin, vmax = weights_np.min(), weights_np.max()

    # Pre-compute pooled maps; updated when radius/hardness/ema/boost sliders change.
    state = {"playing": False, "radius": 0.0, "hardness": 5.0, "ema": 0.0,
             "change_boost": 0.0,
             "rgb": rgb_frames, "raw": weight_maps, "pal": pal_frames,
             "pooled": weight_maps.copy()}

    # Animated side-by-side viewer (RGB | heatmap) – equal-sized panels
    fh, fw = H, W
    panel_w = fw * args.scale / 100
    panel_h = fh * args.scale / 100
    fig_w = panel_w * 2 + 1.4  # two equal panels + colorbar column
    fig_h = panel_h + 1.6      # extra room for sliders
    fig, (ax_rgb, ax_hm, ax_cb) = plt.subplots(
        1, 3, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [1, 1, 0.05], "wspace": 0.08},
    )
    plt.subplots_adjust(bottom=0.22)

    im_rgb = ax_rgb.imshow(rgb_frames[0], interpolation="nearest")
    ax_rgb.set_axis_off()
    style_image_axes(ax_rgb)
    ax_rgb.set_title("RGB", fontsize=10)

    im_hm = ax_hm.imshow(weight_maps[0], interpolation="nearest",
                          cmap="inferno", vmin=vmin, vmax=vmax)
    ax_hm.set_axis_off()
    style_image_axes(ax_hm)
    ax_hm.set_title("Weight", fontsize=10)

    fig.colorbar(im_hm, cax=ax_cb, label="weight")

    fig.suptitle(
        f"Class-weight heatmap  (soften={args.soften}, "
        f"range [{vmin:.2f}, {vmax:.2f}])",
        fontsize=11,
    )

    ax_slider = plt.axes([0.25, 0.10, 0.45, 0.03])
    slider = Slider(ax_slider, "Frame", 1, T, valinit=1, valstep=1, valfmt="%d")
    style_widget(slider)
    enable_slider_scroll(slider)

    ax_radius = plt.axes([0.25, 0.055, 0.45, 0.03])
    slider_radius = Slider(ax_radius, "Radius", 0, 20, valinit=0, valstep=0.5, valfmt="%.1f")
    style_widget(slider_radius)
    enable_slider_scroll(slider_radius)

    ax_hard = plt.axes([0.25, 0.015, 0.45, 0.03])
    slider_hard = Slider(ax_hard, "Hardness", 0.5, 30, valinit=5.0, valstep=0.5, valfmt="%.1f")
    style_widget(slider_hard)
    enable_slider_scroll(slider_hard)

    ax_ema = plt.axes([0.75, 0.015, 0.2, 0.03])
    slider_ema = Slider(ax_ema, "EMA", 0.0, 0.99, valinit=0.0, valstep=0.01, valfmt="%.2f")
    style_widget(slider_ema)
    enable_slider_scroll(slider_ema)

    ax_boost = plt.axes([0.75, 0.055, 0.2, 0.03])
    slider_boost = Slider(ax_boost, "Δ boost", 0.0, 10.0, valinit=0.0, valstep=0.25, valfmt="%.2f")
    style_widget(slider_boost)
    enable_slider_scroll(slider_boost)

    ax_pause = plt.axes([0.75, 0.09, 0.1, 0.05])
    btn_pause = Button(ax_pause, "Play")
    style_widget(btn_pause)

    def _recompute():
        pooled = soft_max_pool(
            state["raw"], state["radius"], state["hardness"],
            temporal_alpha=state["ema"],
        )
        # Apply temporal-change boost: pixels that differ from previous frame
        # get weight *= (1 + boost).
        boost = state["change_boost"]
        if boost > 0:
            pal = state["pal"]
            changed = np.zeros_like(pal[:1], dtype=np.float32)  # frame 0: no change
            changed = np.concatenate([changed, (pal[1:] != pal[:-1]).astype(np.float32)], axis=0)
            pooled = pooled * (1.0 + boost * changed)
        state["pooled"] = pooled

    def _update_frame(idx):
        im_rgb.set_data(state["rgb"][idx])
        im_hm.set_data(state["pooled"][idx])
        fig.canvas.draw_idle()

    def on_slider(val):
        _update_frame(int(val) - 1)

    def on_radius(val):
        state["radius"] = float(val)
        _recompute()
        _update_frame(int(slider.val) - 1)

    def on_hard(val):
        state["hardness"] = float(val)
        _recompute()
        _update_frame(int(slider.val) - 1)

    def on_ema(val):
        state["ema"] = float(val)
        _recompute()
        _update_frame(int(slider.val) - 1)

    def on_boost(val):
        state["change_boost"] = float(val)
        _recompute()
        _update_frame(int(slider.val) - 1)

    slider_radius.on_changed(on_radius)
    slider_hard.on_changed(on_hard)
    slider_ema.on_changed(on_ema)
    slider_boost.on_changed(on_boost)

    slider.on_changed(on_slider)

    def on_pause(event):
        if state["playing"]:
            anim.event_source.stop()
            btn_pause.label.set_text("Play")
        else:
            anim.event_source.start()
            btn_pause.label.set_text("Pause")
        state["playing"] = not state["playing"]

    btn_pause.on_clicked(on_pause)

    def animate(_i):
        next_val = int(slider.val) % T + 1
        slider.set_val(next_val)
        return [im_rgb, im_hm]

    def _resample():
        rgb_new, wm_new, pal_new, info = _load_random_clip()
        state["rgb"] = rgb_new
        state["raw"] = wm_new
        state["pal"] = pal_new
        _recompute()
        slider.set_val(1)
        print(f"Clip: {info}")

    def on_key(event):
        if event.key in ("q", "escape"):
            return  # let matplotlib handle quit
        if event.key == "left":
            slider.set_val(max(1, int(slider.val) - 1))
        elif event.key == "right":
            slider.set_val(min(T, int(slider.val) + 1))
        else:
            _resample()

    fig.canvas.mpl_connect("key_press_event", on_key)

    interval = 1000 / args.fps
    anim = FuncAnimation(fig, animate, interval=interval, blit=False, cache_frame_data=False)

    def _stop_after_init(_event):
        anim.event_source.stop()
        fig.canvas.mpl_disconnect(state["_stop_cid"])

    state["_stop_cid"] = fig.canvas.mpl_connect("draw_event", _stop_after_init)

    plt.show()


if __name__ == "__main__":
    main()
