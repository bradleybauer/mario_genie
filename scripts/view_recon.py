#!/usr/bin/env python3
"""Load reconstruction PNGs and play them back as looping video.

Accepts a single PNG or a directory of PNGs.  Use the Prev / Next buttons
(or left/right arrow keys) to switch between images.

Usage:
    python scripts/view_recon.py checkpoints/magvit2/step_000200.png
    python scripts/view_recon.py checkpoints/magvit2/ --fps 8
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


def main():
    parser = argparse.ArgumentParser(description="View reconstruction PNG(s) as video")
    parser.add_argument("path", help="Path to a reconstruction PNG or a directory of PNGs")
    parser.add_argument("--fps", type=int, default=4, help="Playback frames per second (default: 4)")
    parser.add_argument("--scale", type=float, default=4.0, help="Display scale factor (default: 4)")
    args = parser.parse_args()

    image_paths = collect_images(args.path)
    all_frames = [load_frames(p) for p in image_paths]

    state = {"img_idx": 0, "playing": True}

    frames = all_frames[0]
    n = len(frames)
    fh, fw = frames[0].shape[:2]

    fig_w = fw * args.scale / 100
    fig_h = fh * args.scale / 100 + 0.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plt.subplots_adjust(bottom=0.18)

    im = ax.imshow(frames[0], interpolation="nearest")
    ax.set_axis_off()
    title = ax.set_title("", fontsize=10)

    def _title_text():
        img_name = os.path.basename(image_paths[state["img_idx"]])
        return f"{img_name}  ({state['img_idx']+1}/{len(image_paths)})"

    title.set_text(_title_text())

    # --- Slider ---
    ax_slider = plt.axes([0.25, 0.06, 0.45, 0.04])
    slider = Slider(ax_slider, "Frame", 1, n, valinit=1, valstep=1, valfmt="%d")

    # --- Buttons ---
    ax_prev = plt.axes([0.02, 0.05, 0.08, 0.05])
    btn_prev = Button(ax_prev, "\u25C0 Prev")
    ax_next = plt.axes([0.90, 0.05, 0.08, 0.05])
    btn_next = Button(ax_next, "Next \u25B6")
    ax_pause = plt.axes([0.75, 0.05, 0.1, 0.05])
    btn_pause = Button(ax_pause, "Pause")

    def _switch_image(new_idx):
        new_idx = new_idx % len(image_paths)
        state["img_idx"] = new_idx
        cur = all_frames[new_idx]
        new_n = len(cur)
        # Update slider range
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
        # Slider drives the loop: advance by 1, wrap around
        next_val = int(slider.val) % n_cur + 1
        slider.set_val(next_val)
        return [im]

    def on_key(event):
        if event.key == "left":
            _switch_image(state["img_idx"] - 1)
        elif event.key == "right":
            _switch_image(state["img_idx"] + 1)

    fig.canvas.mpl_connect("key_press_event", on_key)

    interval = 1000 / args.fps
    anim = FuncAnimation(fig, animate, interval=interval, blit=False, cache_frame_data=False)

    fig.canvas.manager.set_window_title(
        args.path if os.path.isdir(args.path) else os.path.basename(args.path)
    )
    plt.show()


if __name__ == "__main__":
    main()
