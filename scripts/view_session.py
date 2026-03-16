#!/usr/bin/env python3
"""Play back a recorded session (.npz) with an interactive pygame viewer.

Controls:
    Space       — pause / resume
    Left/Right  — step backward / forward (while paused)
    Up/Down     — increase / decrease playback speed
    Escape / Q  — quit

Usage:
    python scripts/view_session.py data/nes_sessions/session_000000.npz
    python scripts/view_session.py data/nes_sessions/  # pick latest session
    python scripts/view_session.py data/nes_sessions/ --session 3
    python scripts/view_session.py data/nes_sessions/session_000000.npz --fps 30
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

from mario_world_model.actions import get_action_meanings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_palette(data_dir: Path) -> np.ndarray | None:
    """Load palette.json → (K, 3) uint8 array, searching up from data_dir."""
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


def frames_to_rgb(frames: np.ndarray, palette: np.ndarray | None) -> np.ndarray:
    """Convert (N, C, H, W) frames to (N, H, W, 3) RGB uint8.

    Handles both palette-indexed (C=1) and raw RGB (C=3) sessions.
    """
    n, c, h, w = frames.shape
    if c == 1 and palette is not None:
        # Palette-indexed: (N, 1, H, W) → look up RGB
        indices = frames[:, 0, :, :]  # (N, H, W)
        return palette[indices]       # (N, H, W, 3)
    elif c == 3:
        return np.transpose(frames, (0, 2, 3, 1))  # (N, H, W, 3)
    elif c == 1:
        # Grayscale fallback
        gray = frames[:, 0, :, :]
        return np.stack([gray, gray, gray], axis=-1)
    else:
        raise ValueError(f"Unexpected channel count: {c}")


def resolve_session_path(path: Path, session_idx: int | None) -> Path:
    """Resolve a directory or file path to a single session .npz."""
    if path.is_file() and path.suffix == ".npz":
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Not a file or directory: {path}")
    sessions = sorted(path.glob("session_*.npz"))
    if not sessions:
        raise FileNotFoundError(f"No session_*.npz files found in {path}")
    if session_idx is not None:
        target = path / f"session_{session_idx:06d}.npz"
        if not target.exists():
            raise FileNotFoundError(f"Session not found: {target}")
        return target
    return sessions[-1]  # latest


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

def run_viewer(
    npz_path: Path,
    fps: int,
    scale: int,
):
    data = np.load(npz_path)
    frames_raw = data["frames"]
    actions = data["actions"]
    dones = data["dones"]

    # Metadata arrays (optional)
    meta_keys = ["world", "stage", "x_pos", "y_pos", "score", "coins", "life", "time", "status"]
    meta = {k: data[k] for k in meta_keys if k in data}

    # Load palette from the same directory
    palette = load_palette(npz_path.parent)
    frames_rgb = frames_to_rgb(frames_raw, palette)
    n_frames = len(frames_rgb)

    # Action names for display
    action_meanings = get_action_meanings()

    # Meta JSON (if present)
    meta_json_path = npz_path.with_suffix("").with_suffix(".meta.json")
    session_meta = {}
    if meta_json_path.exists():
        with meta_json_path.open("r") as f:
            session_meta = json.load(f)

    # --- Pygame setup ---
    pygame.init()
    fh, fw = frames_rgb.shape[1], frames_rgb.shape[2]
    win_w, win_h = fw * scale, fh * scale
    hud_height = 48
    screen = pygame.display.set_mode((win_w, win_h + hud_height))
    pygame.display.set_caption(f"Session Viewer — {npz_path.name}")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("monospace", 16)
    small_font = pygame.font.SysFont("monospace", 13)

    frame_idx = 0
    paused = False
    speed = 1.0
    running = True

    session_label = session_meta.get("mode", "?")
    sw = session_meta.get("start_world", "?")
    ss = session_meta.get("start_stage", "?")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT and paused:
                    frame_idx = min(frame_idx + 1, n_frames - 1)
                elif event.key == pygame.K_LEFT and paused:
                    frame_idx = max(frame_idx - 1, 0)
                elif event.key == pygame.K_UP:
                    speed = min(speed * 2, 16.0)
                elif event.key == pygame.K_DOWN:
                    speed = max(speed / 2, 0.25)

        # Draw frame
        frame_hwc = frames_rgb[frame_idx]
        surface = pygame.surfarray.make_surface(np.transpose(frame_hwc, (1, 0, 2)))
        if surface.get_size() != (win_w, win_h):
            surface = pygame.transform.scale(surface, (win_w, win_h))
        screen.fill((30, 30, 30))
        screen.blit(surface, (0, 0))

        # HUD
        action_idx = int(actions[frame_idx])
        action_name = "+".join(action_meanings[action_idx]) if action_idx < len(action_meanings) else str(action_idx)

        # Build info line
        parts = [f"Frame {frame_idx+1}/{n_frames}"]
        parts.append(f"Act: {action_name}")
        if "world" in meta and "stage" in meta:
            w, s = int(meta["world"][frame_idx]), int(meta["stage"][frame_idx])
            parts.append(f"W{w}-{s}")
        if "x_pos" in meta:
            parts.append(f"x={int(meta['x_pos'][frame_idx])}")
        if "life" in meta:
            parts.append(f"lives={int(meta['life'][frame_idx])}")
        if "coins" in meta:
            parts.append(f"coins={int(meta['coins'][frame_idx])}")
        if "score" in meta:
            parts.append(f"score={int(meta['score'][frame_idx])}")
        if dones[frame_idx]:
            parts.append("DONE")

        info_text = font.render("  ".join(parts), True, (220, 220, 220))
        screen.blit(info_text, (6, win_h + 4))

        # Status line
        status_parts = [f"mode={session_label}  start=W{sw}-{ss}"]
        status_parts.append(f"speed={speed:.2g}x")
        if paused:
            status_parts.append("PAUSED (arrows to step)")
        status_text = small_font.render("  ".join(status_parts), True, (160, 160, 160))
        screen.blit(status_text, (6, win_h + 26))

        pygame.display.flip()

        # Advance
        if not paused:
            frame_idx += 1
            if frame_idx >= n_frames:
                frame_idx = 0  # loop

        clock.tick(max(1, int(fps * speed)))

    pygame.quit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a recorded session (.npz)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path", type=Path,
        help="Path to a session .npz file or a directory containing sessions",
    )
    parser.add_argument("--session", type=int, default=None, help="Session index (when path is a directory)")
    parser.add_argument("--fps", type=int, default=60, help="Playback FPS (default: 60)")
    parser.add_argument("--scale", type=int, default=3, help="Window scale multiplier (default: 3)")
    args = parser.parse_args()

    npz_path = resolve_session_path(args.path, args.session)
    print(f"Loading {npz_path} ...")
    data = np.load(npz_path)
    print(f"  {data['frames'].shape[0]} frames, shape={data['frames'].shape[1:]}")

    run_viewer(npz_path, fps=args.fps, scale=args.scale)


if __name__ == "__main__":
    main()
