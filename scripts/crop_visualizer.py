#!/usr/bin/env python3
"""Interactive crop visualizer for NES frames.

Shows a frame with adjustable symmetric crop overlay.
The cropped region is highlighted; the removed area is dimmed.
Displays the resulting dimensions and latent grid size.

Controls:
    Up/Down           — adjust vertical crop by ±8px
    Left/Right        — adjust vertical crop by ±1px
    Shift+Up/Down     — adjust horizontal crop by ±8px
    Shift+Left/Right  — adjust horizontal crop by ±1px
    G                 — snap both to nearest multiple of 16
    D                 — toggle 2× downsample
    F                 — step through frames
    R / N             — load a random sample
    Escape / Q        — quit

Usage:
    python scripts/crop_visualizer.py data/nes/
    python scripts/crop_visualizer.py data/nes/session_000000.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize crop amounts on NES frames")
    parser.add_argument("path", type=Path, help="Session .npz file or data directory")
    parser.add_argument("--scale", type=int, default=3, help="Display scale factor (default: 3)")
    return parser.parse_args()


def load_palette(data_dir: Path) -> np.ndarray | None:
    current = data_dir.resolve()
    while True:
        palette_path = current / "palette.json"
        if palette_path.exists():
            with palette_path.open("r") as f:
                pal = json.load(f)
            return np.array(pal, dtype=np.uint8)
        parent = current.parent
        if parent == current:
            return None
        current = parent


def frame_to_rgb(frame: np.ndarray, palette: np.ndarray | None) -> np.ndarray:
    """Convert a single (C, H, W) frame to (H, W, 3) RGB uint8."""
    c, h, w = frame.shape
    if c == 1 and palette is not None:
        return palette[frame[0]]
    elif c == 3:
        return np.transpose(frame, (1, 2, 0))
    elif c == 1:
        gray = frame[0]
        return np.stack([gray, gray, gray], axis=-1)
    else:
        raise ValueError(f"Unexpected channel count: {c}")


def resolve_session(path: Path) -> Path:
    if path.is_file() and path.suffix == ".npz":
        return path
    if path.is_dir():
        sessions = sorted(path.glob("session_*.npz"))
        if sessions:
            return sessions[8]
        # Recurse one level
        sessions = sorted(path.glob("*/session_*.npz"))
        if sessions:
            return sessions[8]
    raise FileNotFoundError(f"No session .npz found at {path}")


def list_sessions(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".npz":
        return [path]
    if path.is_dir():
        sessions = sorted(path.glob("session_*.npz"))
        if sessions:
            return sessions
        sessions = sorted(path.glob("*/session_*.npz"))
        if sessions:
            return sessions
    raise FileNotFoundError(f"No session .npz found at {path}")


def main():
    args = parse_args()

    session_paths = list_sessions(args.path)
    session_path = resolve_session(args.path)
    session_idx = session_paths.index(session_path)
    rng = np.random.default_rng()

    data = None
    frames = None
    n_frames = c = h = w = 0

    palette = load_palette(args.path if args.path.is_dir() else args.path.parent)

    scale = args.scale
    crop_v = 0  # symmetric vertical crop (pixels removed from top and bottom)
    crop_h = 0  # symmetric horizontal crop (pixels removed from left and right)
    downsample = False  # toggle 2× downsample
    frame_idx = 0

    pygame.init()
    screen_w, screen_h = 0, 0
    info_bar_h = 80
    screen = None
    pygame.display.set_caption("Crop Visualizer")
    font = pygame.font.SysFont("monospace", 18)
    clock = pygame.time.Clock()

    def load_sample(new_session_idx: int, new_frame_idx: int | None = None) -> None:
        nonlocal data, frames, n_frames, c, h, w, session_idx, frame_idx, screen_w, screen_h, screen

        if data is not None:
            data.close()

        session_idx = new_session_idx
        session_path = session_paths[session_idx]
        print(f"Loading {session_path} ...")
        data = np.load(session_path, mmap_mode="r")
        frames = data["frames"]
        n_frames, c, h, w = frames.shape
        print(f"  {n_frames} frames, shape ({c}, {h}, {w})")

        if new_frame_idx is None:
            frame_idx = int(rng.integers(n_frames))
        else:
            frame_idx = max(0, min(new_frame_idx, n_frames - 1))

        new_screen_w, new_screen_h = w * scale, h * scale
        if screen is None or new_screen_w != screen_w or new_screen_h != screen_h:
            screen_w, screen_h = new_screen_w, new_screen_h
            screen = pygame.display.set_mode((screen_w, screen_h + info_bar_h))

    def load_random_sample() -> None:
        if len(session_paths) == 1:
            next_session_idx = session_idx
        else:
            choices = len(session_paths) - 1
            offset = int(rng.integers(choices))
            next_session_idx = (session_idx + 1 + offset) % len(session_paths)
        load_sample(next_session_idx)

    load_sample(session_idx, 0)
    assert frames is not None
    assert screen is not None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_UP:
                    if shift:
                        crop_h = min(crop_h + 8, w // 2 - 8)
                    else:
                        crop_v = min(crop_v + 8, h // 2 - 8)
                elif event.key == pygame.K_DOWN:
                    if shift:
                        crop_h = max(crop_h - 8, 0)
                    else:
                        crop_v = max(crop_v - 8, 0)
                elif event.key == pygame.K_RIGHT:
                    if shift:
                        crop_h = min(crop_h + 1, w // 2 - 8)
                    else:
                        crop_v = min(crop_v + 1, h // 2 - 8)
                elif event.key == pygame.K_LEFT:
                    if shift:
                        crop_h = max(crop_h - 1, 0)
                    else:
                        crop_v = max(crop_v - 1, 0)
                elif event.key == pygame.K_g:
                    # Snap both so cropped sizes are multiples of 16
                    cropped_h = h - 2 * crop_v
                    snapped_h = (cropped_h // 16) * 16
                    crop_v = (h - snapped_h) // 2
                    cropped_w_val = w - 2 * crop_h
                    snapped_w = (cropped_w_val // 16) * 16
                    crop_h = (w - snapped_w) // 2
                elif event.key == pygame.K_d:
                    downsample = not downsample
                elif event.key == pygame.K_f:
                    frame_idx = (frame_idx + n_frames // 10) % n_frames
                elif event.key in (pygame.K_r, pygame.K_n):
                    load_random_sample()

            assert frames is not None
            assert screen is not None

        # Render frame
        rgb = frame_to_rgb(frames[frame_idx], palette)  # (H, W, 3)

        # Apply crop and optional downsample for preview
        cropped = rgb[crop_v:h - crop_v if crop_v else h,
                      crop_h:w - crop_h if crop_h else w]
        if downsample:
            # Nearest-neighbor 2× downsample then upscale to show pixelation
            small = cropped[::2, ::2]
            # Repeat each pixel 2× in both dims to show the blocky result
            preview = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
            # Trim if odd crop caused size mismatch
            preview = preview[:cropped.shape[0], :cropped.shape[1]]
        else:
            preview = cropped

        # Build full display: dim original with crop overlay
        display = rgb.copy()
        if crop_v > 0:
            display[:crop_v] = (display[:crop_v].astype(np.uint16) * 77) >> 8
            display[-crop_v:] = (display[-crop_v:].astype(np.uint16) * 77) >> 8
        if crop_h > 0:
            display[:, :crop_h] = (display[:, :crop_h].astype(np.uint16) * 77) >> 8
            display[:, -crop_h:] = (display[:, -crop_h:].astype(np.uint16) * 77) >> 8

        # Replace the crop region with the (possibly downsampled) preview
        display[crop_v:h - crop_v if crop_v else h,
                crop_h:w - crop_h if crop_h else w] = preview

        # Scale up
        surf = pygame.surfarray.make_surface(np.transpose(display, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (screen_w, screen_h))

        screen.fill((30, 30, 30))
        screen.blit(surf, (0, 0))

        # Draw crop boundary lines
        if crop_v > 0:
            top_y = crop_v * scale
            bot_y = (h - crop_v) * scale
            pygame.draw.line(screen, (255, 80, 80), (0, top_y), (screen_w, top_y), 2)
            pygame.draw.line(screen, (255, 80, 80), (0, bot_y), (screen_w, bot_y), 2)
        if crop_h > 0:
            left_x = crop_h * scale
            right_x = (w - crop_h) * scale
            pygame.draw.line(screen, (80, 180, 255), (left_x, 0), (left_x, screen_h), 2)
            pygame.draw.line(screen, (80, 180, 255), (right_x, 0), (right_x, screen_h), 2)

        # Info text
        cropped_h = h - 2 * crop_v
        cropped_w = w - 2 * crop_h
        ds = 2 if downsample else 1
        final_h = cropped_h // ds
        final_w = cropped_w // ds
        valid_h = final_h % 16 == 0
        valid_w = final_w % 16 == 0
        valid = valid_h and valid_w
        latent_h = final_h // 16 if valid_h else final_h / 16
        latent_w = final_w // 16 if valid_w else final_w / 16
        tokens = int(latent_h * latent_w) if valid else 0
        pct = (final_h * final_w) / (h * w) * 100

        ds_tag = "  [2× DOWN]" if downsample else ""
        lines = [
            f"V-crop: {crop_v}px  H-crop: {crop_h}px  |  "
            f"{final_w}x{final_h}{ds_tag}  |  "
            f"latent: {latent_w}x{latent_h}  |  "
            f"tokens: {tokens}  |  "
            f"{pct:.0f}% of original"
            + ("" if valid else "  [NOT div by 16 — press G]"),
            f"Sample {session_idx + 1}/{len(session_paths)}  |  Frame {frame_idx}/{n_frames}  |  "
            f"Up/Down: ±V  Shift+Up/Down: ±H  G: snap  D: 2× down  F: next frame  R/N: random sample  Q: quit",
        ]

        y = screen_h + 8
        for line in lines:
            color = (255, 255, 255) if valid else (255, 180, 80)
            text_surf = font.render(line, True, color)
            screen.blit(text_surf, (10, y))
            y += 28

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    if data is not None:
        data.close()


if __name__ == "__main__":
    main()
