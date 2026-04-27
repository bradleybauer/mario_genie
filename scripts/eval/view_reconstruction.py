#!/usr/bin/env python3
"""Load reconstruction PNGs and play them back as looping video.

If a directory with subfolders containing PNGs is given, shows a clickable
directory picker first.  Click a run to view its reconstructions, then use
the Back button (or Escape) to return to the picker.

Usage:
    python scripts/view_reconstruction.py results/sweep/
    python scripts/view_reconstruction.py results/sweep/some_run/ --fps 8
    python scripts/view_reconstruction.py checkpoints/magvit2/step_000200.png
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.plot_style import (
    apply_plot_style, enable_slider_scroll, style_image_axes, style_widget,
    WIDGET_BG, WIDGET_HOVER, WIDGET_ACTIVE, WIDGET_TEXT,
)
apply_plot_style()


DEFAULT_PREDICTED_FRAMES = None


def visible_frames(frames, include_context=False, predicted_frames=DEFAULT_PREDICTED_FRAMES):
    """Return the frame sequence shown in the viewer.

    By default shows all frames. If *predicted_frames* is set, keeps only the
    last *predicted_frames* frames, which are typically the model's
    predictions (skipping however many context frames the training code
    happened to include in the PNG).
    """
    if include_context or predicted_frames is None or predicted_frames <= 0:
        return frames
    if len(frames) <= predicted_frames:
        return frames
    return frames[-predicted_frames:]


def _load_stacked_frames(path):
    """Split a vertically-stacked PNG into individual frames."""
    img = Image.open(path)
    w, h = img.size
    target = w // 2
    if target == 0 or h == 0:
        return [np.array(img)] if w and h else []
    if h % target == 0:
        frame_h = target
    else:
        # Find the divisor of h closest to the target frame height.
        divisors = set()
        for i in range(1, int(h**0.5) + 1):
            if h % i == 0:
                divisors.add(i)
                divisors.add(h // i)
        frame_h = min(divisors, key=lambda d: abs(d - target))
    n_frames = h // frame_h
    arr = np.array(img)
    return [arr[i * frame_h:(i + 1) * frame_h] for i in range(n_frames)]


def _load_rollout_frames(path):
    """Decode rollout preview into per-timestep side-by-side frames (GT left, prediction right)."""
    return _load_stacked_frames(path)


def load_frames(path, *, frame_format="auto"):
    """Load frames from either stacked previews or rollout preview grids."""
    base = os.path.basename(path).lower()
    if frame_format == "rollout" or (frame_format == "auto" and base.startswith("rollout_step_")):
        return _load_rollout_frames(path)
    return _load_stacked_frames(path)


def _should_keep_full_frames(path, frame_format):
    if frame_format == "rollout":
        return True
    if frame_format == "auto":
        return os.path.basename(path).lower().startswith("rollout_step_")
    return False


def collect_images(path):
    """Return a sorted list of reconstruction PNG paths from a file or directory."""
    if os.path.isfile(path):
        return [path]
    pngs = sorted(glob.glob(os.path.join(path, "*.png")))
    pngs = [p for p in pngs if "histogram" not in os.path.basename(p)]
    basenames = [os.path.basename(p) for p in pngs]
    has_ram_previews = any("_ram_" in name for name in basenames)
    has_video_previews = any("_video_" in name for name in basenames)
    if has_ram_previews and has_video_previews:
        pngs = [p for p in pngs if "_video_" in os.path.basename(p)]
    if not pngs:
        raise FileNotFoundError(f"No PNG files found in {path}")
    return pngs


def find_subfolders_with_pngs(root):
    """Return sorted subfolder names that contain at least one PNG."""
    subs = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.png")):
            subs.append(name)
    return subs


def show_viewer(
    folder_path,
    fps,
    scale,
    include_context=False,
    predicted_frames=DEFAULT_PREDICTED_FRAMES,
    frame_format="auto",
):
    """Show the reconstruction viewer for a single folder (no sidebar)."""
    image_paths = collect_images(folder_path)
    loaded = [
        (
            p,
            load_frames(p, frame_format=frame_format),
        )
        for p in image_paths
    ]
    loaded = [
        (
            p,
            frames
            if _should_keep_full_frames(p, frame_format)
            else visible_frames(
                frames,
                include_context=include_context,
                predicted_frames=predicted_frames,
            ),
        )
        for p, frames in loaded
    ]
    loaded = [(p, f) for p, f in loaded if f]
    if not loaded:
        print(f"No valid reconstruction frames in {folder_path}")
        return
    image_paths, all_frames = zip(*loaded)
    image_paths, all_frames = list(image_paths), list(all_frames)

    state = {"img_idx": 0, "playing": False}

    frames = all_frames[0]
    n = len(frames)
    fh, fw = frames[0].shape[:2]

    fig_w = fw * scale / 100
    fig_h = fh * scale / 100 + 0.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plt.subplots_adjust(bottom=0.18)

    im = ax.imshow(frames[0], interpolation="nearest")
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    style_image_axes(ax)
    title = ax.set_title("", fontsize=10)

    folder_label = os.path.basename(os.path.normpath(folder_path))

    def _title_text():
        img_name = os.path.basename(image_paths[state["img_idx"]])
        return f"{folder_label} / {img_name}  ({state['img_idx']+1}/{len(image_paths)})"

    title.set_text(_title_text())

    ax_slider = plt.axes([0.25, 0.06, 0.45, 0.04])
    slider = Slider(ax_slider, "Frame", 1, n, valinit=1, valstep=1, valfmt="%d")
    style_widget(slider)
    enable_slider_scroll(slider)

    ax_prev = plt.axes([0.05, 0.05, 0.08, 0.05])
    btn_prev = Button(ax_prev, "◀ Prev")
    style_widget(btn_prev)
    ax_next = plt.axes([0.87, 0.05, 0.08, 0.05])
    btn_next = Button(ax_next, "Next ▶")
    style_widget(btn_next)
    ax_pause = plt.axes([0.75, 0.05, 0.1, 0.05])
    btn_pause = Button(ax_pause, "Play")
    style_widget(btn_pause)
    ax_save = plt.axes([0.14, 0.05, 0.1, 0.05])
    btn_save = Button(ax_save, "Save GIF")
    style_widget(btn_save)

    def _switch_image(new_idx):
        new_idx = new_idx % len(image_paths)
        state["img_idx"] = new_idx
        cur = all_frames[new_idx]
        new_n = len(cur)
        slider.valmin = 1
        slider.valmax = new_n
        slider.ax.set_xlim(1, new_n)
        slider.set_val(1)
        im.set_data(cur[0])
        title.set_text(_title_text())
        fig.canvas.draw_idle()

    def on_prev(event):
        _switch_image(state["img_idx"] - 1)

    def on_next(event):
        _switch_image(state["img_idx"] + 1)

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    def on_slider(val):
        cur = all_frames[state["img_idx"]]
        idx = int(val) - 1
        idx = idx % len(cur)
        im.set_data(cur[idx])
        fig.canvas.draw_idle()

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

    def on_save(_event):
        cur = all_frames[state["img_idx"]]
        img_name = os.path.splitext(os.path.basename(image_paths[state["img_idx"]]))[0]
        out_path = f"{folder_label}_{img_name}.gif"
        pil_frames = [Image.fromarray(f) for f in cur]
        pil_frames[0].save(
            out_path, save_all=True, append_images=pil_frames[1:],
            duration=int(1000 / fps), loop=0,
        )
        print(f"Saved {out_path}")
        btn_save.label.set_text("Saved!")
        fig.canvas.draw_idle()

    btn_save.on_clicked(on_save)

    def animate(_i):
        cur = all_frames[state["img_idx"]]
        n_cur = len(cur)
        next_val = int(slider.val) % n_cur + 1
        slider.set_val(next_val)
        return [im]

    def on_key(event):
        if event.key == "left":
            _switch_image(state["img_idx"] - 1)
        elif event.key == "right":
            _switch_image(state["img_idx"] + 1)

    fig.canvas.mpl_connect("key_press_event", on_key)

    interval = 1000 / fps
    anim = FuncAnimation(fig, animate, interval=interval, blit=False, cache_frame_data=False)

    def _stop_after_init(_event):
        anim.event_source.stop()
        fig.canvas.mpl_disconnect(state["_stop_cid"])

    state["_stop_cid"] = fig.canvas.mpl_connect("draw_event", _stop_after_init)

    fig.canvas.manager.set_window_title(folder_label)
    plt.show()


class _LazyRun:
    """Stores paths eagerly, loads/validates frames per-image on demand."""
    __slots__ = (
        "paths",
        "include_context",
        "predicted_frames",
        "frame_format",
        "_frames",
    )

    def __init__(
        self,
        paths,
        include_context=False,
        predicted_frames=DEFAULT_PREDICTED_FRAMES,
        frame_format="auto",
    ):
        self.paths = paths
        self.include_context = include_context
        self.predicted_frames = predicted_frames
        self.frame_format = frame_format
        self._frames = [None] * len(paths)

    def _ensure_loaded(self, idx):
        if self._frames[idx] is None:
            frames = load_frames(
                self.paths[idx],
                frame_format=self.frame_format,
            ) or []
            if _should_keep_full_frames(self.paths[idx], self.frame_format):
                self._frames[idx] = frames
            else:
                self._frames[idx] = visible_frames(
                    frames,
                    include_context=self.include_context,
                    predicted_frames=self.predicted_frames,
                )

    def frames(self, idx):
        self._ensure_loaded(idx)
        return self._frames[idx]

    def first_valid_frames(self):
        """Load images one by one until we find one with frames."""
        for i in range(len(self.paths)):
            f = self.frames(i)
            if f:
                return f
        return None


def _discover_run(
    root,
    name,
    include_context=False,
    predicted_frames=DEFAULT_PREDICTED_FRAMES,
    frame_format="auto",
):
    """Return a _LazyRun for a subfolder, or None if no valid PNGs."""
    folder = os.path.join(root, name)
    try:
        image_paths = collect_images(folder)
    except FileNotFoundError:
        return None
    if not image_paths:
        return None
    return _LazyRun(
        image_paths,
        include_context=include_context,
        predicted_frames=predicted_frames,
        frame_format=frame_format,
    )


def show_combined(
    root,
    subfolders,
    fps,
    scale,
    include_context=False,
    predicted_frames=DEFAULT_PREDICTED_FRAMES,
    frame_format="auto",
):
    """Single-window UI with run selector on the left and viewer on the right."""
    # Discover which subfolders have PNGs (cheap: just glob, no image loading)
    available = []
    for name in subfolders:
        folder = os.path.join(root, name)
        if glob.glob(os.path.join(folder, "*.png")):
            available.append(name)

    if not available:
        print(f"No valid reconstruction frames found under {root}")
        return

    # Lazy cache — runs are loaded on first access
    run_cache = {}

    def _get_run(name):
        if name not in run_cache:
            run_cache[name] = _discover_run(
                root,
                name,
                include_context=include_context,
                predicted_frames=predicted_frames,
                frame_format=frame_format,
            )
        return run_cache[name]

    state = {
        "run_idx": 0,
        "img_idx": 0,
        "playing": False,
        "anim": None,
    }

    # Sidebar layout
    sidebar_frac = 0.28

    image_left = sidebar_frac + 0.02
    image_bottom = 0.18
    image_width_frac = 1.0 - sidebar_frac - 0.04
    image_height_frac = 0.78

    n_runs = len(available)
    btn_h_pts = 22
    btn_gap_pts = 4
    sidebar_top_margin_pts = 30
    sidebar_needed_pts = sidebar_top_margin_pts + n_runs * (btn_h_pts + btn_gap_pts)

    # Viewer sizing from first run's first frame
    first_run = _get_run(available[0])
    first_frames = first_run.first_valid_frames()
    if not first_frames:
        print(f"No valid reconstruction frames found under {root}")
        return
    first_frame = first_frames[0]
    fh, fw = first_frame.shape[:2]
    viewer_w_pts = fw * scale
    viewer_h_pts = fh * scale

    fig_w_in = viewer_w_pts / (image_width_frac * 100.0)
    fig_h_in = max(sidebar_needed_pts / 100.0, viewer_h_pts / (image_height_frac * 100.0))
    fig_h_in = max(fig_h_in, 4.0)

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    fig.canvas.manager.set_window_title(f"Reconstructions — {root}")

    # Image axes (right side)
    ax_img = fig.add_axes([image_left, image_bottom, image_width_frac, image_height_frac])
    ax_img.set_axis_off()
    style_image_axes(ax_img)

    im = ax_img.imshow(first_frame, interpolation="nearest")
    ax_img.set_aspect("equal", adjustable="box")
    title = ax_img.set_title("", fontsize=10)

    # Controls along the bottom-right
    ctrl_left = image_left
    ctrl_w = image_width_frac

    ax_slider = fig.add_axes([ctrl_left + 0.15, 0.06, ctrl_w - 0.45, 0.04])
    slider = Slider(ax_slider, "Frame", 1, len(first_frames), valinit=1, valstep=1, valfmt="%d")
    style_widget(slider)
    enable_slider_scroll(slider)

    ax_prev = fig.add_axes([ctrl_left, 0.05, 0.07, 0.05])
    btn_prev = Button(ax_prev, "◀ Prev")
    style_widget(btn_prev)
    ax_next = fig.add_axes([ctrl_left + ctrl_w - 0.07, 0.05, 0.07, 0.05])
    btn_next = Button(ax_next, "Next ▶")
    style_widget(btn_next)
    ax_pause = fig.add_axes([ctrl_left + ctrl_w - 0.18, 0.05, 0.09, 0.05])
    btn_pause = Button(ax_pause, "Play")
    style_widget(btn_pause)
    ax_save = fig.add_axes([ctrl_left, 0.00, 0.09, 0.05])
    btn_save = Button(ax_save, "Save GIF")
    style_widget(btn_save)

    # Run selector buttons (left sidebar)
    run_buttons = []
    for i, name in enumerate(available):
        y_frac = 1.0 - (sidebar_top_margin_pts + i * (btn_h_pts + btn_gap_pts)) / (fig_h_in * 100.0)
        h_frac = btn_h_pts / (fig_h_in * 100.0)
        ax_btn = fig.add_axes([0.01, y_frac - h_frac, sidebar_frac - 0.02, h_frac])
        btn = Button(ax_btn, name, color=WIDGET_BG, hovercolor=WIDGET_HOVER)
        btn.label.set_fontfamily("monospace")
        btn.label.set_fontsize(8)
        btn.label.set_color(WIDGET_TEXT)

        def on_run_click(_event, idx=i):
            _select_run(idx)

        btn.on_clicked(on_run_click)
        run_buttons.append(btn)

    def _cur_run():
        return _get_run(available[state["run_idx"]])

    def _title_text():
        run = _cur_run()
        img_name = os.path.basename(run.paths[state["img_idx"]])
        return f"{available[state['run_idx']]} / {img_name}  ({state['img_idx']+1}/{len(run.paths)})"

    def _highlight_button(idx):
        for i, btn in enumerate(run_buttons):
            btn.color = WIDGET_ACTIVE if i == idx else WIDGET_BG
            btn.ax.set_facecolor(btn.color)

    def _select_run(idx):
        state["run_idx"] = idx
        state["img_idx"] = 0
        run = _cur_run()
        cur_frames = run.first_valid_frames()
        if not cur_frames:
            title.set_text(f"{available[idx]} — no valid frames")
            _highlight_button(idx)
            fig.canvas.draw_idle()
            return
        n = len(cur_frames)
        slider.valmin = 1
        slider.valmax = n
        slider.ax.set_xlim(1, n)
        slider.set_val(1)
        im.set_data(cur_frames[0])
        im.set_clim(cur_frames[0].min(), cur_frames[0].max())
        title.set_text(_title_text())
        _highlight_button(idx)
        fig.canvas.draw_idle()

    def _switch_image(new_idx):
        run = _cur_run()
        new_idx = new_idx % len(run.paths)
        state["img_idx"] = new_idx
        cur_frames = run.frames(new_idx)
        if not cur_frames:
            return
        n = len(cur_frames)
        slider.valmin = 1
        slider.valmax = n
        slider.ax.set_xlim(1, n)
        slider.set_val(1)
        im.set_data(cur_frames[0])
        title.set_text(_title_text())
        fig.canvas.draw_idle()

    def on_prev(_event):
        _switch_image(state["img_idx"] - 1)

    def on_next(_event):
        _switch_image(state["img_idx"] + 1)

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    def on_slider(val):
        run = _cur_run()
        cur_frames = run.frames(state["img_idx"])
        if not cur_frames:
            return
        idx = (int(val) - 1) % len(cur_frames)
        im.set_data(cur_frames[idx])
        fig.canvas.draw_idle()

    slider.on_changed(on_slider)

    def on_pause(_event):
        if state["playing"]:
            state["anim"].event_source.stop()
            btn_pause.label.set_text("Play")
        else:
            state["anim"].event_source.start()
            btn_pause.label.set_text("Pause")
        state["playing"] = not state["playing"]

    btn_pause.on_clicked(on_pause)

    def on_save(_event):
        run = _cur_run()
        cur_frames = run.frames(state["img_idx"])
        run_name = available[state["run_idx"]]
        img_name = os.path.splitext(os.path.basename(run.paths[state["img_idx"]]))[0]
        out_path = f"{run_name}_{img_name}.gif"
        pil_frames = [Image.fromarray(f) for f in cur_frames]
        pil_frames[0].save(
            out_path, save_all=True, append_images=pil_frames[1:],
            duration=int(1000 / fps), loop=0,
        )
        print(f"Saved {out_path}")
        btn_save.label.set_text("Saved!")
        fig.canvas.draw_idle()

    btn_save.on_clicked(on_save)

    def animate(_i):
        run = _cur_run()
        cur_frames = run.frames(state["img_idx"])
        n_cur = len(cur_frames)
        if n_cur == 0:
            return [im]
        next_val = int(slider.val) % n_cur + 1
        slider.set_val(next_val)
        return [im]

    def on_key(event):
        if event.key == "left":
            _switch_image(state["img_idx"] - 1)
        elif event.key == "right":
            _switch_image(state["img_idx"] + 1)
        elif event.key == "up":
            _select_run((state["run_idx"] - 1) % len(available))
        elif event.key == "down":
            _select_run((state["run_idx"] + 1) % len(available))

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Initial highlight
    _highlight_button(0)
    title.set_text(_title_text())

    interval = 1000 / fps
    anim = FuncAnimation(fig, animate, interval=interval, blit=False, cache_frame_data=False)
    state["anim"] = anim

    def _stop_after_init(_event):
        anim.event_source.stop()
        fig.canvas.mpl_disconnect(state["_stop_cid"])

    state["_stop_cid"] = fig.canvas.mpl_connect("draw_event", _stop_after_init)

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View reconstruction PNG(s) as video")
    parser.add_argument("path", help="Path to a reconstruction PNG or a directory of PNGs")
    parser.add_argument("--fps", type=int, default=16, help="Playback frames per second (default: 4)")
    parser.add_argument("--scale", type=float, default=4.0, help="Display scale factor (default: 4)")
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Show all frames including context (by default only predicted frames are shown)",
    )
    parser.add_argument(
        "--predicted-frames",
        type=int,
        default=DEFAULT_PREDICTED_FRAMES,
        help="Number of trailing predicted frames to show; default shows all frames",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["auto", "stacked", "rollout"],
        default="auto",
        help=(
            "Image layout format. 'auto' detects rollout_step_*.png as rollout grids; "
            "otherwise uses stacked format."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # If it's a file or a directory that directly has PNGs, go straight to viewer
    if os.path.isfile(args.path) or glob.glob(os.path.join(args.path, "*.png")):
        show_viewer(
            args.path,
            args.fps,
            args.scale,
            include_context=args.include_context,
            predicted_frames=args.predicted_frames,
            frame_format=args.format,
        )
        return

    # Otherwise look for subfolders containing PNGs
    subfolders = find_subfolders_with_pngs(args.path)
    if not subfolders:
        print(f"No PNGs found in {args.path} or its immediate subfolders.")
        return

    show_combined(
        args.path,
        subfolders,
        args.fps,
        args.scale,
        include_context=args.include_context,
        predicted_frames=args.predicted_frames,
        frame_format=args.format,
    )


if __name__ == "__main__":
    main()
