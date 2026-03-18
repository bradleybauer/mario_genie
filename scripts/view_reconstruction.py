#!/usr/bin/env python3
"""Load reconstruction PNGs and play them back as looping video.

If a directory with subfolders containing PNGs is given, shows a clickable
directory picker first.  Click a run to view its reconstructions, then use
the Back button (or Escape) to return to the picker.

Usage:
    python scripts/view_reconstruction.py results/asha_sweep/
    python scripts/view_reconstruction.py results/asha_sweep/some_run/ --fps 8
    python scripts/view_reconstruction.py checkpoints/magvit2/step_000200.png
"""

import argparse
import glob
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation


def load_frames(path):
    """Split a vertically-stacked grid PNG into individual frames."""
    img = Image.open(path)
    w, h = img.size
    frame_h = w // 2
    if h % frame_h != 0:
        for div in range(1, h + 1):
            if h % div == 0 and abs(div - w // 2) < abs(frame_h - w // 2):
                frame_h = div
    n_frames = h // frame_h
    arr = np.array(img)
    frames = [arr[i * frame_h:(i + 1) * frame_h] for i in range(n_frames)]
    return frames


def collect_images(path):
    """Return a sorted list of PNG paths from a file or directory."""
    if os.path.isfile(path):
        return [path]
    pngs = sorted(glob.glob(os.path.join(path, "*.png")))
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


def show_viewer(folder_path, fps, scale):
    """Show the reconstruction viewer for a single folder (no sidebar)."""
    image_paths = collect_images(folder_path)
    loaded = [(p, load_frames(p)) for p in image_paths]
    loaded = [(p, f) for p, f in loaded if f]
    if not loaded:
        print(f"No valid reconstruction frames in {folder_path}")
        return
    image_paths, all_frames = zip(*loaded)
    image_paths, all_frames = list(image_paths), list(all_frames)

    state = {"img_idx": 0, "playing": True}

    frames = all_frames[0]
    n = len(frames)
    fh, fw = frames[0].shape[:2]

    fig_w = fw * scale / 100
    fig_h = fh * scale / 100 + 0.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plt.subplots_adjust(bottom=0.18)

    im = ax.imshow(frames[0], interpolation="nearest")
    ax.set_axis_off()
    title = ax.set_title("", fontsize=10)

    folder_label = os.path.basename(os.path.normpath(folder_path))

    def _title_text():
        img_name = os.path.basename(image_paths[state["img_idx"]])
        return f"{folder_label} / {img_name}  ({state['img_idx']+1}/{len(image_paths)})"

    title.set_text(_title_text())

    ax_slider = plt.axes([0.25, 0.06, 0.45, 0.04])
    slider = Slider(ax_slider, "Frame", 1, n, valinit=1, valstep=1, valfmt="%d")

    ax_prev = plt.axes([0.05, 0.05, 0.08, 0.05])
    btn_prev = Button(ax_prev, "\u25c0 Prev")
    ax_next = plt.axes([0.87, 0.05, 0.08, 0.05])
    btn_next = Button(ax_next, "Next \u25b6")
    ax_pause = plt.axes([0.75, 0.05, 0.1, 0.05])
    btn_pause = Button(ax_pause, "Pause")

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
            anim.pause()
            btn_pause.label.set_text("Play")
        else:
            anim.resume()
            btn_pause.label.set_text("Pause")
        state["playing"] = not state["playing"]

    btn_pause.on_clicked(on_pause)

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

    fig.canvas.manager.set_window_title(folder_label)
    plt.show()


def show_combined(root, subfolders, fps, scale):
    """Single-window UI with run selector on the left and viewer on the right."""
    # Pre-load all runs
    run_data = {}
    for name in subfolders:
        folder = os.path.join(root, name)
        try:
            image_paths = collect_images(folder)
        except FileNotFoundError:
            continue
        loaded = [(p, load_frames(p)) for p in image_paths]
        loaded = [(p, f) for p, f in loaded if f]
        if loaded:
            paths, frames = zip(*loaded)
            run_data[name] = {"paths": list(paths), "frames": list(frames)}

    if not run_data:
        print(f"No valid reconstruction frames found under {root}")
        return

    available = [name for name in subfolders if name in run_data]

    state = {
        "run_idx": 0,
        "img_idx": 0,
        "playing": True,
        "anim": None,
    }

    # Sidebar layout
    sidebar_frac = 0.28
    n_runs = len(available)
    btn_h_pts = 22
    btn_gap_pts = 4
    sidebar_top_margin_pts = 30
    sidebar_needed_pts = sidebar_top_margin_pts + n_runs * (btn_h_pts + btn_gap_pts)

    # Viewer sizing from first run's first frame
    first_run = run_data[available[0]]
    fh, fw = first_run["frames"][0][0].shape[:2]
    viewer_w_pts = fw * scale
    viewer_h_pts = fh * scale + 80  # extra for controls

    fig_w_in = (viewer_w_pts / (1.0 - sidebar_frac)) / 100.0
    fig_h_in = max(sidebar_needed_pts, viewer_h_pts) / 100.0
    fig_h_in = max(fig_h_in, 4.0)

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    fig.canvas.manager.set_window_title(f"Reconstructions — {root}")

    # Image axes (right side)
    ax_img = fig.add_axes([sidebar_frac + 0.02, 0.18, 1.0 - sidebar_frac - 0.04, 0.78])
    ax_img.set_axis_off()

    first_frames = first_run["frames"][0]
    im = ax_img.imshow(first_frames[0], interpolation="nearest")
    title = ax_img.set_title("", fontsize=10)

    # Controls along the bottom-right
    ctrl_left = sidebar_frac + 0.02
    ctrl_w = 1.0 - sidebar_frac - 0.04

    ax_slider = fig.add_axes([ctrl_left + 0.15, 0.06, ctrl_w - 0.45, 0.04])
    slider = Slider(ax_slider, "Frame", 1, len(first_frames), valinit=1, valstep=1, valfmt="%d")

    ax_prev = fig.add_axes([ctrl_left, 0.05, 0.07, 0.05])
    btn_prev = Button(ax_prev, "\u25c0 Prev")
    ax_next = fig.add_axes([ctrl_left + ctrl_w - 0.07, 0.05, 0.07, 0.05])
    btn_next = Button(ax_next, "Next \u25b6")
    ax_pause = fig.add_axes([ctrl_left + ctrl_w - 0.18, 0.05, 0.09, 0.05])
    btn_pause = Button(ax_pause, "Pause")

    # Run selector buttons (left sidebar)
    run_buttons = []
    for i, name in enumerate(available):
        y_frac = 1.0 - (sidebar_top_margin_pts + i * (btn_h_pts + btn_gap_pts)) / (fig_h_in * 100.0)
        h_frac = btn_h_pts / (fig_h_in * 100.0)
        ax_btn = fig.add_axes([0.01, y_frac - h_frac, sidebar_frac - 0.02, h_frac])
        btn = Button(ax_btn, name, color="0.92", hovercolor="0.78")
        btn.label.set_fontfamily("monospace")
        btn.label.set_fontsize(8)

        def on_run_click(_event, idx=i):
            _select_run(idx)

        btn.on_clicked(on_run_click)
        run_buttons.append(btn)

    def _cur_run():
        return run_data[available[state["run_idx"]]]

    def _title_text():
        run = _cur_run()
        img_name = os.path.basename(run["paths"][state["img_idx"]])
        return f"{available[state['run_idx']]} / {img_name}  ({state['img_idx']+1}/{len(run['paths'])})"

    def _highlight_button(idx):
        for i, btn in enumerate(run_buttons):
            btn.color = "0.72" if i == idx else "0.92"
            btn.ax.set_facecolor(btn.color)

    def _select_run(idx):
        state["run_idx"] = idx
        state["img_idx"] = 0
        run = _cur_run()
        cur_frames = run["frames"][0]
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
        new_idx = new_idx % len(run["paths"])
        state["img_idx"] = new_idx
        cur_frames = run["frames"][new_idx]
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
        cur_frames = run["frames"][state["img_idx"]]
        idx = (int(val) - 1) % len(cur_frames)
        im.set_data(cur_frames[idx])
        fig.canvas.draw_idle()

    slider.on_changed(on_slider)

    def on_pause(_event):
        if state["playing"]:
            state["anim"].pause()
            btn_pause.label.set_text("Play")
        else:
            state["anim"].resume()
            btn_pause.label.set_text("Pause")
        state["playing"] = not state["playing"]

    btn_pause.on_clicked(on_pause)

    def animate(_i):
        run = _cur_run()
        cur_frames = run["frames"][state["img_idx"]]
        n_cur = len(cur_frames)
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

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="View reconstruction PNG(s) as video")
    parser.add_argument("path", help="Path to a reconstruction PNG or a directory of PNGs")
    parser.add_argument("--fps", type=int, default=4, help="Playback frames per second (default: 4)")
    parser.add_argument("--scale", type=float, default=4.0, help="Display scale factor (default: 4)")
    args = parser.parse_args()

    # If it's a file or a directory that directly has PNGs, go straight to viewer
    if os.path.isfile(args.path) or glob.glob(os.path.join(args.path, "*.png")):
        show_viewer(args.path, args.fps, args.scale)
        return

    # Otherwise look for subfolders containing PNGs
    subfolders = find_subfolders_with_pngs(args.path)
    if not subfolders:
        print(f"No PNGs found in {args.path} or its immediate subfolders.")
        return

    show_combined(args.path, subfolders, args.fps, args.scale)


if __name__ == "__main__":
    main()