#!/usr/bin/env python3
"""Play back a Mesen research recording (.avi + .mdat) with RAM visualization.

Plays the lossless AVI video with audio while displaying synchronized
RAM/WRAM data from the companion .mdat file.

Features:
    - Video + audio playback from the .avi (OpenCV + pygame.mixer)
    - RAM value grid and activity heat-map
    - Decoded SMB1 game variables (auto-detected from RAM)
    - OAM sprite position mini-map
    - NES controller input display (all connected controllers)
    - Pause, step, speed controls

Controls:
    Space       - pause / resume
    Escape / Q  - quit

Usage:
    python scripts/play_recording.py path/to/recording.mdat
    python scripts/play_recording.py path/to/recording.avi
"""
from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.ram_viz import (
    RAM_CELL, RAM_COLS, RAM_ROWS,
    RAMGridRenderer, render_oam_minimap, draw_ram_region_labels,
)
from mario_world_model.game_decoders import decode_smb1, draw_decoded_sections


# ---------------------------------------------------------------------------
# MDAT parsing
# ---------------------------------------------------------------------------

MDAT_MAGIC = b"MSDC"
HEADER_SIZE = 32
HEADER_FMT = "<4sHHIIHHIII"


def parse_mdat(path: Path):
    """Parse an .mdat file.

    Returns (header_dict, frame_numbers, ram, wram_or_None, inputs).
    """
    data = path.read_bytes()
    if len(data) < HEADER_SIZE:
        raise ValueError(f"File too small: {len(data)} bytes")

    fields = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    (magic, version, console_type, ram_size, wram_size,
     input_bytes, num_controllers, start_frame, fps_x1000, _) = fields

    if magic != MDAT_MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}")

    header = {
        "console_type": console_type,
        "ram_size": ram_size,
        "wram_size": wram_size,
        "input_bytes_per_frame": input_bytes,
        "num_controllers": num_controllers,
        "start_frame": start_frame,
        "fps": fps_x1000 / 1000.0,
        "record_size": 4 + ram_size + wram_size + input_bytes,
    }

    record_size = header["record_size"]
    payload = data[HEADER_SIZE:]
    num_frames = len(payload) // record_size
    if len(payload) % record_size != 0:
        payload = payload[:num_frames * record_size]

    # Parse all records into contiguous arrays
    all_records = np.frombuffer(payload, dtype=np.uint8).reshape(num_frames, record_size)
    frame_numbers = np.frombuffer(
        all_records[:, :4].tobytes(), dtype=np.uint32
    ).copy()
    ram = all_records[:, 4:4 + ram_size].copy()
    wram = (all_records[:, 4 + ram_size:4 + ram_size + wram_size].copy()
            if wram_size > 0 else None)
    inputs = all_records[:, 4 + ram_size + wram_size:].copy()

    return header, frame_numbers, ram, wram, inputs


# ---------------------------------------------------------------------------
# NES controller display
# ---------------------------------------------------------------------------

# Bit layout in the NES controller byte (DataCollector order):
#   bit 0=Up, 1=Down, 2=Left, 3=Right, 4=Start, 5=Select, 6=B, 7=A

_BTN_SIZE = 20
_BTN_GAP = 2
_BTN_STEP = _BTN_SIZE + _BTN_GAP


def _draw_rect_btn(surface, font, label, x, y, pressed, w=_BTN_SIZE, h=_BTN_SIZE):
    bg = (255, 255, 100) if pressed else (45, 45, 50)
    fg = (0, 0, 0) if pressed else (90, 90, 90)
    pygame.draw.rect(surface, bg, (x, y, w, h), border_radius=3)
    txt = font.render(label, True, fg)
    surface.blit(txt, (x + (w - txt.get_width()) // 2, y + (h - txt.get_height()) // 2))


def _draw_circle_btn(surface, font, label, cx, cy, pressed):
    r = _BTN_SIZE // 2
    bg = (230, 60, 60) if pressed else (45, 45, 50)
    fg = (255, 255, 255) if pressed else (90, 90, 90)
    pygame.draw.circle(surface, bg, (cx, cy), r)
    txt = font.render(label, True, fg)
    surface.blit(txt, (cx - txt.get_width() // 2, cy - txt.get_height() // 2))


def draw_controller(
    surface: pygame.Surface,
    font: pygame.font.Font,
    input_bytes: np.ndarray,
    x: int, y: int,
    num_controllers: int,
) -> int:
    """Draw NES controller state for each controller. Returns y after drawing."""
    for ctrl in range(min(num_controllers, len(input_bytes))):
        bval = int(input_bytes[ctrl])
        lbl = font.render(f"P{ctrl + 1}", True, (200, 200, 200))
        surface.blit(lbl, (x, y))
        cy = y + 16

        # D-pad  (3x3 grid, centre empty)
        dx = x
        _draw_rect_btn(surface, font, "U", dx + _BTN_STEP, cy,
                        bool(bval & (1 << 0)))
        _draw_rect_btn(surface, font, "L", dx, cy + _BTN_STEP,
                        bool(bval & (1 << 2)))
        _draw_rect_btn(surface, font, "R", dx + 2 * _BTN_STEP, cy + _BTN_STEP,
                        bool(bval & (1 << 3)))
        _draw_rect_btn(surface, font, "D", dx + _BTN_STEP, cy + 2 * _BTN_STEP,
                        bool(bval & (1 << 1)))

        # Select / Start (pill-shaped)
        mx = dx + 3 * _BTN_STEP + 14
        _draw_rect_btn(surface, font, "Sel", mx, cy + _BTN_STEP,
                        bool(bval & (1 << 5)), w=30, h=_BTN_SIZE - 4)
        _draw_rect_btn(surface, font, "Str", mx + 34, cy + _BTN_STEP,
                        bool(bval & (1 << 4)), w=30, h=_BTN_SIZE - 4)

        # B / A (circles)
        bx = mx + 74
        _draw_circle_btn(surface, font, "B", bx, cy + _BTN_STEP + _BTN_SIZE // 2,
                          bool(bval & (1 << 6)))
        _draw_circle_btn(surface, font, "A", bx + _BTN_SIZE + 6,
                          cy + _BTN_STEP + _BTN_SIZE // 2,
                          bool(bval & (1 << 7)))

        y = cy + 3 * _BTN_STEP + 6

    return y


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def extract_audio_to_wav(avi_path: Path) -> Path | None:
    """Extract audio track from AVI to a temporary WAV file using ffmpeg."""
    if not shutil.which("ffmpeg"):
        print("Warning: ffmpeg not found, audio disabled", file=sys.stderr)
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(avi_path),
         "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
         tmp.name],
        capture_output=True,
    )
    if result.returncode != 0:
        print("Warning: audio extraction failed, audio disabled", file=sys.stderr)
        Path(tmp.name).unlink(missing_ok=True)
        return None
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------

def run(recording_path: Path, scale: int,
        ram_scale: int, audio_offset: float = 0.0):
    # Resolve companion files from either .mdat or .avi path.
    # Use stem manipulation instead of with_suffix() because filenames like
    # "Super Mario Bros. (World) 3_26_2026 10:51:16 PM.avi" contain dots
    # that with_suffix("") would incorrectly split on.
    name = recording_path.name
    for ext in (".mdat", ".avi"):
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    parent = recording_path.parent
    mdat_path = parent / (name + ".mdat")
    avi_path = parent / (name + ".avi")

    if not mdat_path.exists():
        raise FileNotFoundError(f"MDAT not found: {mdat_path}")
    if not avi_path.exists():
        raise FileNotFoundError(f"AVI not found: {avi_path}")

    # --- Load MDAT ---
    print(f"Loading {mdat_path.name} ...")
    header, frame_numbers, ram_data, wram_data, input_data = parse_mdat(mdat_path)
    n_mdat = len(frame_numbers)
    fps = round(header["fps"])
    print(f"  {n_mdat} frames, {header['ram_size']}B RAM, "
          f"{header['wram_size']}B WRAM, "
          f"{header['num_controllers']} controller(s), "
          f"{header['fps']:.1f} FPS")

    # --- Open AVI ---
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {avi_path}")
    avi_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = min(n_mdat, avi_total)
    print(f"  AVI: {avi_total} frames, {vid_w}x{vid_h}")
    print(f"  Synchronized frames: {n_frames}")

    # --- Extract audio ---
    wav_path = extract_audio_to_wav(avi_path)

    # --- Layout ---
    game_w = vid_w * scale
    game_h = vid_h * scale

    ram_cell = RAM_CELL * ram_scale
    ram_grid_w = RAM_COLS * ram_cell
    ram_grid_h = RAM_ROWS * ram_cell
    ram_label_margin = 80
    grid_gap = 14
    section_gap = 8
    top_margin = 14

    oam_map_w, oam_map_h = 192, 180
    oam_section_h = 18 + oam_map_h + 6
    ctrl_section_h = 16 + (3 * _BTN_STEP + 22) * min(header["num_controllers"], 4)

    panel_w = ram_label_margin + ram_grid_w + 16
    panel_h = (top_margin + (ram_grid_h + 16) * 2 + grid_gap
               + 200 + section_gap)

    hud_h = 36
    left_col_h = game_h + oam_section_h + ctrl_section_h
    win_w = game_w + panel_w
    win_h = max(left_col_h, panel_h) + hud_h

    # --- Pygame init ---
    pygame.init()
    pygame.mixer.init(frequency=44100, channels=2)
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"Recording — {name}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 13)
    hud_font = pygame.font.SysFont("monospace", 14)

    # Load audio but don't play yet — defer until first frame renders
    has_audio = False
    if wav_path:
        try:
            pygame.mixer.music.load(str(wav_path))
            has_audio = True
        except Exception as e:
            print(f"Warning: audio load failed: {e}", file=sys.stderr)

    ram_renderer = RAMGridRenderer()
    frame_idx = 0
    cap_next = 0          # next frame index the VideoCapture will read
    current_frame_rgb: np.ndarray | None = None
    paused = False
    running = True

    # --- Clock state ---
    # When audio is active at 1x speed, audio position is the master clock.
    # Otherwise wall-clock time drives playback so speed control still works.
    clock_origin: float = 0.0   # monotonic time when playback (re)started
    frame_origin: int = 0       # frame index at that origin
    audio_playing = False       # True while audio is actively driving sync

    def _start_audio(from_frame: int) -> None:
        """Start (or restart) audio from a given video frame."""
        nonlocal audio_playing
        if not has_audio:
            return
        pos_sec = from_frame / fps + audio_offset
        try:
            pygame.mixer.music.play(start=max(0.0, pos_sec))
            audio_playing = True
        except Exception:
            audio_playing = False

    def _stop_audio() -> None:
        nonlocal audio_playing
        if audio_playing:
            pygame.mixer.music.stop()
            audio_playing = False

    def _pause_audio() -> None:
        if audio_playing:
            pygame.mixer.music.pause()

    def _reset_wall_clock(from_frame: int) -> None:
        """Reset the wall-clock reference for non-audio playback."""
        nonlocal clock_origin, frame_origin
        clock_origin = time.monotonic()
        frame_origin = from_frame

    def _current_target_frame() -> int:
        """Derive which frame should be displayed right now."""
        if audio_playing:
            # Audio is master clock — get_pos() returns ms since play().
            # audio_offset is already baked into the play(start=...) call,
            # so we just need elapsed audio time + the frame we started from.
            audio_ms = pygame.mixer.music.get_pos()
            if audio_ms >= 0:
                return max(0, min(
                    frame_origin + int(audio_ms / 1000.0 * fps),
                    n_frames - 1))
        # Wall-clock fallback (no audio)
        elapsed = time.monotonic() - clock_origin
        return max(0, min(frame_origin + int(elapsed * fps), n_frames - 1))

    def read_video_frame(idx: int) -> np.ndarray | None:
        """Read a video frame, seeking only when necessary."""
        nonlocal cap_next, current_frame_rgb
        if idx != cap_next:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            cap_next = idx
        ret, bgr = cap.read()
        if ret:
            current_frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cap_next = idx + 1
        return current_frame_rgb

    # Kick off playback
    _reset_wall_clock(0)
    _start_audio(0)

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        if paused:
                            _pause_audio()
                        else:
                            # Resume — unpause keeps the exact sample position
                            # so get_pos() stays consistent with no startup gap.
                            _reset_wall_clock(frame_idx)
                            if audio_playing:
                                pygame.mixer.music.unpause()
                            else:
                                _start_audio(frame_idx)

            if not running:
                break

            # --- Derive current frame from clock ---
            if not paused:
                frame_idx = _current_target_frame()
                if frame_idx >= n_frames - 1:
                    # Loop
                    frame_idx = 0
                    _reset_wall_clock(0)
                    _start_audio(0)

            # --- Read data for current frame ---
            rgb = read_video_frame(frame_idx)
            if rgb is None:
                frame_idx = 0
                _reset_wall_clock(0)
                _start_audio(0)
                continue

            ram = ram_data[frame_idx]
            ram_renderer.update(ram)

            # Concatenate RAM+WRAM for the game decoder (stable-retro layout)
            if wram_data is not None:
                full_ram = np.concatenate([ram, wram_data[frame_idx]])
            else:
                full_ram = ram

            inp = input_data[frame_idx]

            # --- Draw ---
            screen.fill((10, 10, 15))

            # Video frame (top-left)
            game_surf = pygame.surfarray.make_surface(
                np.transpose(rgb, (1, 0, 2)))
            if game_surf.get_size() != (game_w, game_h):
                game_surf = pygame.transform.scale(game_surf, (game_w, game_h))
            screen.blit(game_surf, (0, 0))

            # OAM mini-map (below video)
            oy = game_h
            screen.blit(
                font.render("-- OAM SPRITES --", True, (200, 200, 100)),
                (4, oy))
            oy += 18
            screen.blit(render_oam_minimap(ram), (4, oy))
            oy += oam_map_h + 6

            # Controller input (below OAM)
            screen.blit(
                font.render("-- INPUT --", True, (200, 200, 100)),
                (4, oy))
            draw_controller(screen, font, inp, 4, oy + 16,
                            header["num_controllers"])

            # --- Right panel ---
            gx = game_w + ram_label_margin
            cy = top_margin

            # RAM value grid
            screen.blit(
                font.render("RAM VALUES", True, (200, 200, 200)), (gx, cy))
            cy += 16
            screen.blit(
                ram_renderer.render(ram, activity_mode=False, cell_size=ram_cell),
                (gx, cy))
            draw_ram_region_labels(screen, font, gx, cy, cell_size=ram_cell)
            cy += ram_grid_h + grid_gap

            # RAM activity grid
            screen.blit(
                font.render("RAM ACTIVITY", True, (200, 200, 200)), (gx, cy))
            cy += 16
            screen.blit(
                ram_renderer.render(ram, activity_mode=True, cell_size=ram_cell),
                (gx, cy))
            draw_ram_region_labels(screen, font, gx, cy, cell_size=ram_cell)
            cy += ram_grid_h + section_gap

            # Decoded game variables
            sections = decode_smb1(full_ram)
            draw_decoded_sections(screen, font, sections, game_w + 8, cy)

            # --- HUD bar ---
            hud_y = win_h - hud_h
            pygame.draw.rect(screen, (20, 20, 28), (0, hud_y, win_w, hud_h))

            parts = [f"Frame {frame_idx + 1}/{n_frames}"]
            parts.append(f"#{int(frame_numbers[frame_idx])}")
            elapsed = frame_idx / fps
            parts.append(f"{elapsed:.1f}s")
            if paused:
                parts.append("PAUSED")
            info = hud_font.render("  ".join(parts), True, (190, 190, 190))
            screen.blit(info, (8, hud_y + (hud_h - info.get_height()) // 2))

            pygame.display.flip()

            # Cap render rate to avoid busy-spinning; actual frame timing
            # is driven by the audio/wall clock, not by this tick.
            clock.tick(max(fps, 120))

    except KeyboardInterrupt:
        pass
    finally:
        _stop_audio()
        pygame.quit()
        cap.release()
        if wav_path:
            wav_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Play a Mesen recording (.avi + .mdat) with RAM visualization",
    )
    parser.add_argument(
        "path", type=Path,
        help="Path to the .mdat or .avi file (companion file auto-detected)",
    )
    parser.add_argument("--scale", type=int, default=2,
                        help="Video scale factor (default: 2)")
    parser.add_argument("--ram-scale", type=int, default=1,
                        help="RAM grid scale multiplier (default: 1)")
    parser.add_argument("--audio-offset", type=float, default=0.0,
                        help="Seconds to skip into audio track to fix A/V sync "
                             "(positive = audio was ahead, default: 0.0)")
    args = parser.parse_args()

    run(args.path, scale=args.scale,
        ram_scale=args.ram_scale, audio_offset=args.audio_offset)


if __name__ == "__main__":
    main()
