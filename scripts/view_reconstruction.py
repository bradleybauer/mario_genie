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


def show_picker(root, subfolders):
    """Show a clickable table of runs split by '_'. Returns chosen subfolder name or None."""
    chosen = [None]
    # Split names into columns
    rows = [name.split("_") for name in subfolders]
    n_cols = max(len(r) for r in rows)
    # Pad columns to find max width per column
    rows = [r + [""] * (n_cols - len(r)) for r in rows]
    col_widths = [max(len(rows[i][j]) for i in range(len(rows))) for j in range(n_cols)]

    # Build formatted labels with fixed-width columns
    labels = []
    for row in rows:
        parts = []
        for j, cell in enumerate(row):
            parts.append(cell.ljust(col_widths[j]))
        labels.append("  ".join(parts))

    n_rows = len(subfolders)
    btn_h = 0.06
    gap = 0.015
    top_margin = 0.08
    total = n_rows * btn_h + (n_rows - 1) * gap + top_margin + 0.04
    fig_h = max(3.0, total * 5)
    fig, ax = plt.subplots(figsize=(max(7, n_cols * 1.4), fig_h))
    ax.set_axis_off()
    ax.set_title(f"Runs in {root}", fontsize=12, fontweight="bold")
    fig.canvas.manager.set_window_title(f"Pick a run — {root}")

    buttons = []
    for i, (name, label) in enumerate(zip(subfolders, labels)):
        y = 1.0 - top_margin - i * (btn_h + gap)
        ax_btn = fig.add_axes([0.04, y - btn_h, 0.92, btn_h])
        btn = Button(ax_btn, label, color="0.92", hovercolor="0.78")
        btn.label.set_fontfamily("monospace")
        btn.label.set_fontsize(9)

        def on_click(_event, _name=name):
            chosen[0] = _name
            plt.close(fig)

        btn.on_clicked(on_click)
        buttons.append(btn)

    plt.show()
    return chosen[0]


def show_viewer(folder_path, fps, scale):
    """Show the reconstruction viewer for a single folder. Returns True to go back."""
    go_back = [False]

    image_paths = collect_images(folder_path)
    all_frames = [load_frames(p) for p in image_paths]

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

    # --- Slider ---
    ax_slider = plt.axes([0.25, 0.06, 0.45, 0.04])
    slider = Slider(ax_slider, "Frame", 1, n, valinit=1, valstep=1, valfmt="%d")

    # --- Buttons ---
    ax_back = plt.axes([0.02, 0.05, 0.08, 0.05])
    btn_back = Button(ax_back, "\u25c0 Back")
    ax_prev = plt.axes([0.12, 0.05, 0.08, 0.05])
    btn_prev = Button(ax_prev, "\u25c0 Prev")
    ax_next = plt.axes([0.90, 0.05, 0.08, 0.05])
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

    def on_back(_event):
        go_back[0] = True
        plt.close(fig)

    def on_prev(event):
        _switch_image(state["img_idx"] - 1)

    def on_next(event):
        _switch_image(state["img_idx"] + 1)

    btn_back.on_clicked(on_back)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    def _update_frame(idx):
        cur = all_frames[state["img_idx"]]
        idx = int(idx) % len(cur)
        im.set_data(cur[idx])
        fig.canvas.draw_idle()

    def on_slider(val):
        _update_frame(int(val) - 1)

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
        elif event.key == "escape":
            on_back(None)

    fig.canvas.mpl_connect("key_press_event", on_key)

    interval = 1000 / fps
    anim = FuncAnimation(fig, animate, interval=interval, blit=False, cache_frame_data=False)

    fig.canvas.manager.set_window_title(folder_label)
    plt.show()
    return go_back[0]


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

    # Picker loop: pick a run, view it, come back
    while True:
        name = show_picker(args.path, subfolders)
        if name is None:
            break
        came_back = show_viewer(os.path.join(args.path, name), args.fps, args.scale)
        if not came_back:
            break


if __name__ == "__main__":
    main()