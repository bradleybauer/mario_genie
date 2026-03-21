#!/usr/bin/env python3
"""Rank sessions by the largest within-level frame-to-frame x-position jump.

Scans all session .npz files, computes |x_pos[t+1] - x_pos[t]| for consecutive
frames that share the same (world, stage) and are not separated by a done flag,
then prints a ranked list.

Usage:
    python scripts/rank_x_jumps.py --data-dir data/nes
    python scripts/rank_x_jumps.py --data-dir data/nes --top 20
    python scripts/rank_x_jumps.py --data-dir data/nes --min-jump 50
    python scripts/rank_x_jumps.py --data-dir data/nes --view

Viewer controls (--view):
    N             — next jump
    P             — previous jump
    Left / Right  — step frame backward / forward (while paused)
    Space         — play/pause around current jump
    Up / Down     — increase/decrease playback speed
    [ / ]         — shrink/expand context window (frames around jump)
    Escape / Q    — quit
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class JumpRecord:
    session: Path
    max_jump: int
    frame_idx: int
    x_before: int
    x_after: int
    world: int
    stage: int


def find_max_x_jump(npz_path: Path) -> JumpRecord | None:
    """Return the largest in-level x jump in a session, or None if < 2 frames."""
    data = np.load(npz_path)
    x_pos = data["x_pos"]
    world = data["world"]
    stage = data["stage"]
    dones = data["dones"]

    if len(x_pos) < 2:
        return None

    dx = np.abs(np.diff(x_pos))
    same_level = (world[:-1] == world[1:]) & (stage[:-1] == stage[1:])
    not_done = ~dones[:-1]
    valid = same_level & not_done

    if not valid.any():
        return None

    dx[~valid] = 0
    idx = int(np.argmax(dx))
    max_jump = int(dx[idx])
    if max_jump == 0:
        return None

    return JumpRecord(
        session=npz_path,
        max_jump=max_jump,
        frame_idx=idx,
        x_before=int(x_pos[idx]),
        x_after=int(x_pos[idx + 1]),
        world=int(world[idx]),
        stage=int(stage[idx]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank sessions by largest in-level x jump")
    parser.add_argument("--data-dir", type=Path, default='data', help="Directory with session npz files")
    parser.add_argument("--top", type=int, default=50, help="Number of results to show (default: 50)")
    parser.add_argument("--min-jump", type=int, default=0, help="Only show jumps >= this value")
    parser.add_argument("--view", action="store_true", help="Launch interactive viewer to browse jumps")
    parser.add_argument("--start", type=int, default=1, help="Start viewer at this rank (1-indexed, default: 1)")
    parser.add_argument("--context", type=int, default=120, help="Frames of context around each jump in viewer (default: 120)")
    parser.add_argument("--scale", type=int, default=3, help="Viewer window scale (default: 3)")
    args = parser.parse_args()

    npz_files = sorted(args.data_dir.rglob("session_*.npz"))
    if not npz_files:
        print(f"No session npz files found in {args.data_dir}")
        return

    records: list[JumpRecord] = []
    for npz_path in npz_files:
        rec = find_max_x_jump(npz_path)
        if rec is not None and rec.max_jump >= args.min_jump:
            records.append(rec)

    records.sort(key=lambda r: r.max_jump, reverse=True)
    records = records[: args.top]

    if not records:
        print("No x jumps found matching criteria.")
        return

    print(f"\n{'Rank':>4}  {'Max Jump':>9}  {'Frame':>6}  {'Level':>6}  {'X Before':>8} → {'X After':<8}  Session")
    print("-" * 90)
    for rank, rec in enumerate(records, 1):
        level = f"{rec.world}-{rec.stage}"
        session_name = rec.session.relative_to(args.data_dir)
        print(
            f"{rank:>4}  {rec.max_jump:>9}  {rec.frame_idx:>6}  {level:>6}  {rec.x_before:>8} → {rec.x_after:<8}  {session_name}"
        )

    if args.view:
        start_idx = max(0, args.start - 1)
        view_jumps(records, args.data_dir, context=args.context, scale=args.scale, start=start_idx)


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

def _load_palette(data_dir: Path) -> np.ndarray | None:
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


def _frames_to_rgb(frames: np.ndarray, palette: np.ndarray | None) -> np.ndarray:
    n, c, h, w = frames.shape
    if c == 1 and palette is not None:
        return palette[frames[:, 0, :, :]]
    elif c == 3:
        return np.transpose(frames, (0, 2, 3, 1))
    elif c == 1:
        gray = frames[:, 0, :, :]
        return np.stack([gray, gray, gray], axis=-1)
    else:
        raise ValueError(f"Unexpected channel count: {c}")


def _load_jump_clip(
    rec: JumpRecord, palette: np.ndarray | None, context: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load frames around a jump. Returns (rgb_frames, x_pos, dones, jump_offset)."""
    data = np.load(rec.session)
    n = len(data["x_pos"])
    start = max(0, rec.frame_idx - context)
    end = min(n, rec.frame_idx + context + 2)
    frames_rgb = _frames_to_rgb(data["frames"][start:end], palette)
    x_pos = data["x_pos"][start:end]
    dones = data["dones"][start:end]
    jump_offset = rec.frame_idx - start
    return frames_rgb, x_pos, dones, jump_offset


def view_jumps(
    records: list[JumpRecord],
    data_dir: Path,
    context: int = 120,
    scale: int = 3,
    start: int = 0,
) -> None:
    import pygame

    palette = _load_palette(data_dir)

    # Pre-load clip at requested start rank
    rec_idx = min(start, len(records) - 1)
    clip_frames, clip_x, clip_dones, jump_offset = _load_jump_clip(records[rec_idx], palette, context)
    n_clip = len(clip_frames)

    pygame.init()
    fh, fw = clip_frames.shape[1], clip_frames.shape[2]
    win_w, win_h = fw * scale, fh * scale
    hud_height = 64
    screen = pygame.display.set_mode((win_w, win_h + hud_height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)
    small_font = pygame.font.SysFont("monospace", 13)

    frame_idx = jump_offset
    paused = False
    speed = 1.0
    fps = 60
    running = True

    def reload_clip():
        nonlocal clip_frames, clip_x, clip_dones, jump_offset, n_clip, frame_idx
        clip_frames, clip_x, clip_dones, jump_offset, = _load_jump_clip(records[rec_idx], palette, context)
        n_clip = len(clip_frames)
        frame_idx = jump_offset

    def update_caption():
        rec = records[rec_idx]
        pygame.display.set_caption(
            f"Jump Viewer — #{rec_idx+1}/{len(records)}  Δx={rec.max_jump}  {rec.session.name}"
        )

    update_caption()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_n:
                    if rec_idx < len(records) - 1:
                        rec_idx += 1
                        reload_clip()
                        update_caption()
                        paused = True
                elif event.key == pygame.K_p:
                    if rec_idx > 0:
                        rec_idx -= 1
                        reload_clip()
                        update_caption()
                        paused = True
                elif event.key == pygame.K_RIGHT and paused:
                    frame_idx = min(frame_idx + 1, n_clip - 1)
                elif event.key == pygame.K_LEFT and paused:
                    frame_idx = max(frame_idx - 1, 0)
                elif event.key == pygame.K_UP:
                    speed = min(speed * 2, 16.0)
                elif event.key == pygame.K_DOWN:
                    speed = max(speed / 2, 0.25)
                elif event.key == pygame.K_RIGHTBRACKET:
                    context = min(context + 30, 600)
                    reload_clip()
                    update_caption()
                elif event.key == pygame.K_LEFTBRACKET:
                    context = max(context - 30, 10)
                    reload_clip()
                    update_caption()

        # Draw frame
        frame_hwc = clip_frames[frame_idx]
        surface = pygame.surfarray.make_surface(np.transpose(frame_hwc, (1, 0, 2)))
        if surface.get_size() != (win_w, win_h):
            surface = pygame.transform.scale(surface, (win_w, win_h))
        screen.fill((30, 30, 30))
        screen.blit(surface, (0, 0))

        # HUD line 1: jump info
        rec = records[rec_idx]
        at_jump = frame_idx == jump_offset
        x_val = int(clip_x[frame_idx])
        parts = [
            f"#{rec_idx+1}/{len(records)}",
            f"Δx={rec.max_jump}",
            f"W{rec.world}-{rec.stage}",
            f"x={x_val}",
            f"clip {frame_idx+1}/{n_clip}",
        ]
        if at_jump:
            parts.append("◄ JUMP")
        if clip_dones[frame_idx]:
            parts.append("DONE")
        color = (255, 100, 100) if at_jump else (220, 220, 220)
        screen.blit(font.render("  ".join(parts), True, color), (6, win_h + 4))

        # HUD line 2: controls
        status_parts = [f"speed={speed:.2g}x", f"ctx=±{context}"]
        if paused:
            status_parts.append("PAUSED  N/P=jump  ←/→=step  [/]=ctx  Space=play")
        else:
            status_parts.append("PLAYING  Space=pause  ↑↓=speed  N/P=jump")
        screen.blit(small_font.render("  ".join(status_parts), True, (140, 140, 140)), (6, win_h + 24))

        # HUD line 3: session
        screen.blit(small_font.render(str(rec.session.relative_to(data_dir)), True, (100, 100, 100)), (6, win_h + 44))

        pygame.display.flip()

        if not paused:
            frame_idx += 1
            if frame_idx >= n_clip:
                frame_idx = 0

        clock.tick(max(1, int(fps * speed)))

    pygame.quit()


if __name__ == "__main__":
    main()
