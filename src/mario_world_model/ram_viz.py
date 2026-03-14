"""NES RAM visualization utilities shared by play_nes and play_ram_viz.

Provides:
    - RAMGridRenderer: renders a 64x32 colour grid with region tints, fade-out
      change highlights, and an activity heat-map mode for internal 2KB RAM.
    - WRAMGridRenderer: renders an 8KB WRAM grid (128x64).
    - render_oam_minimap(): 256x240 mini-map showing OAM sprite positions.
    - draw_ram_region_labels(): draw colour-coded region labels beside the grid.
    - Constants, colormaps, and region definitions for the NES memory layout.
"""
from __future__ import annotations

import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAM_SIZE = 2048
RAM_COLS = 64
RAM_ROWS = 32   # 64 * 32 = 2048
RAM_CELL = 6    # pixels per cell

# WRAM (cartridge RAM, NES $6000-$7FFF, 8 KB)
WRAM_SIZE = 8192
WRAM_COLS = 128
WRAM_ROWS = 64   # 128 * 64 = 8192
WRAM_CELL = 3    # smaller cells — 8KB is 4× the data

# NES memory regions (byte ranges)
REGION_ZERO_PAGE = (0x0000, 0x00FF)
REGION_STACK     = (0x0100, 0x01FF)
REGION_OAM       = (0x0200, 0x02FF)
REGION_GAME_DATA = (0x0300, 0x07FF)

# Region base tints (R, G, B)
REGION_TINTS = {
    "zero_page": np.array([80, 110, 220], dtype=np.int16),
    "stack":     np.array([90, 90, 90],   dtype=np.int16),
    "oam":       np.array([210, 80, 210],  dtype=np.int16),
    "game_data": np.array([70, 200, 90],   dtype=np.int16),
}

# Region boundary rows in the 32-row grid
REGION_ROWS = {
    "zero_page": (0, 3),
    "stack":     (4, 7),
    "oam":       (8, 11),
    "game_data": (12, 31),
}

# Bright separator colours matching region tints
REGION_SEP_COLOURS = {
    "zero_page": (100, 130, 255),
    "stack":     (130, 130, 130),
    "oam":       (240, 110, 240),
    "game_data": (100, 230, 120),
}

# Change highlight fade duration in frames
FADE_FRAMES = 15

# Activity history window (frames)
ACTIVITY_WINDOW = 30

# NES native resolution
NES_W, NES_H = 256, 240
OAM_MAP_SCALE = 1


# ---------------------------------------------------------------------------
# Colourmap builders
# ---------------------------------------------------------------------------

def _build_value_colormap() -> np.ndarray:
    """256x3 uint8 -- blue->cyan->green->yellow->red for nonzero, dark for 0."""
    cmap = np.zeros((256, 3), dtype=np.uint8)
    cmap[0] = (20, 20, 20)
    for v in range(1, 256):
        t = v / 255.0
        if t < 0.25:
            s = t / 0.25
            r, g, b = 0, int(255 * s), 255
        elif t < 0.5:
            s = (t - 0.25) / 0.25
            r, g, b = 0, 255, int(255 * (1 - s))
        elif t < 0.75:
            s = (t - 0.5) / 0.25
            r, g, b = int(255 * s), 255, 0
        else:
            s = (t - 0.75) / 0.25
            r, g, b = 255, int(255 * (1 - s)), 0
        cmap[v] = (r, g, b)
    return cmap


def _build_activity_colormap() -> np.ndarray:
    """256x3 uint8 -- black (no activity) -> white (max activity)."""
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for v in range(256):
        t = v / 255.0
        if t < 0.33:
            s = t / 0.33
            cmap[v] = (0, 0, int(180 * s))
        elif t < 0.66:
            s = (t - 0.33) / 0.33
            cmap[v] = (int(255 * s), 80, 180)
        else:
            s = (t - 0.66) / 0.34
            cmap[v] = (255, int(80 + 175 * s), int(180 + 75 * s))
    return cmap


def _build_region_tint_array() -> np.ndarray:
    """(2048, 3) int16 array of per-byte region tints."""
    tints = np.zeros((RAM_SIZE, 3), dtype=np.int16)
    for name, (s, e) in [("zero_page", REGION_ZERO_PAGE), ("stack", REGION_STACK),
                          ("oam", REGION_OAM), ("game_data", REGION_GAME_DATA)]:
        tints[s:e + 1] = REGION_TINTS[name]
    return tints


VALUE_CMAP = _build_value_colormap()
ACTIVITY_CMAP = _build_activity_colormap()
REGION_TINT_ARRAY = _build_region_tint_array()


# ---------------------------------------------------------------------------
# RAMGridRenderer
# ---------------------------------------------------------------------------

class RAMGridRenderer:
    """Renders the 2KB RAM as a 64x32 colour grid with region tints and
    fade-out change highlights."""

    def __init__(self):
        self._fade = np.zeros(RAM_SIZE, dtype=np.float32)
        self._activity_ring = np.zeros((ACTIVITY_WINDOW, RAM_SIZE), dtype=np.uint8)
        self._ring_idx = 0
        self._prev_ram: np.ndarray | None = None

    def update(self, ram: np.ndarray) -> None:
        """Call once per frame with the current RAM snapshot."""
        if self._prev_ram is not None:
            changed = ram != self._prev_ram
            self._fade[changed] = 1.0
            self._fade[~changed] = np.maximum(self._fade[~changed] - 1.0 / FADE_FRAMES, 0.0)
            self._activity_ring[self._ring_idx] = changed.astype(np.uint8)
        self._ring_idx = (self._ring_idx + 1) % ACTIVITY_WINDOW
        self._prev_ram = ram.copy()

    def activity_map(self) -> np.ndarray:
        """Return (2048,) float32 in [0, 1] -- fraction of recent frames where each byte changed."""
        return self._activity_ring.mean(axis=0).astype(np.float32)

    def render(self, ram: np.ndarray, *, activity_mode: bool = False,
               cell_size: int = RAM_CELL) -> pygame.Surface:
        """Render a (RAM_COLS*cell, RAM_ROWS*cell) surface."""
        cell = cell_size
        w = RAM_COLS * cell
        h = RAM_ROWS * cell

        if activity_mode:
            act = self.activity_map()
            act_u8 = np.clip(act * 255, 0, 255).astype(np.uint8)
            colours = ACTIVITY_CMAP[act_u8].reshape(RAM_ROWS, RAM_COLS, 3).astype(np.int16)
        else:
            base = VALUE_CMAP[ram].astype(np.int16)
            region = REGION_TINT_ARRAY
            is_zero = (ram == 0)
            blended = np.where(
                is_zero[:, np.newaxis],
                region // 3,
                (base + region) // 2,
            )
            colours = blended.reshape(RAM_ROWS, RAM_COLS, 3)

        fade_2d = self._fade.reshape(RAM_ROWS, RAM_COLS)
        flash = (fade_2d[..., np.newaxis] * 180).astype(np.int16)
        colours = np.clip(colours + flash, 0, 255).astype(np.uint8)

        expanded = np.repeat(np.repeat(colours, cell, axis=0), cell, axis=1)
        expanded[::cell, :] = (30, 30, 30)
        expanded[:, ::cell] = (30, 30, 30)

        for rname, (r0, r1) in REGION_ROWS.items():
            sep_colour = REGION_SEP_COLOURS[rname]
            y_top = r0 * cell
            y_bot = min((r1 + 1) * cell, h - 1)
            if y_top > 0:
                expanded[max(0, y_top - 1):y_top + 1, :] = sep_colour
            if y_bot < h - 1:
                expanded[y_bot:min(y_bot + 2, h), :] = sep_colour

        return pygame.surfarray.make_surface(expanded.transpose(1, 0, 2))


# ---------------------------------------------------------------------------
# WRAMGridRenderer — 8KB cartridge RAM grid (128×64)
# ---------------------------------------------------------------------------

class WRAMGridRenderer:
    """Renders cartridge WRAM (8 KB) as a 128×64 colour grid."""

    def __init__(self):
        self._fade = np.zeros(WRAM_SIZE, dtype=np.float32)
        self._activity_ring = np.zeros((ACTIVITY_WINDOW, WRAM_SIZE), dtype=np.uint8)
        self._ring_idx = 0
        self._prev: np.ndarray | None = None

    def update(self, wram: np.ndarray) -> None:
        if self._prev is not None:
            changed = wram != self._prev
            self._fade[changed] = 1.0
            self._fade[~changed] = np.maximum(self._fade[~changed] - 1.0 / FADE_FRAMES, 0.0)
            self._activity_ring[self._ring_idx] = changed.astype(np.uint8)
        self._ring_idx = (self._ring_idx + 1) % ACTIVITY_WINDOW
        self._prev = wram.copy()

    def activity_map(self) -> np.ndarray:
        return self._activity_ring.mean(axis=0).astype(np.float32)

    def render(self, wram: np.ndarray, *, activity_mode: bool = False,
               cell_size: int = WRAM_CELL) -> pygame.Surface:
        cell = cell_size
        w = WRAM_COLS * cell
        h = WRAM_ROWS * cell

        if activity_mode:
            act = self.activity_map()
            act_u8 = np.clip(act * 255, 0, 255).astype(np.uint8)
            colours = ACTIVITY_CMAP[act_u8].reshape(WRAM_ROWS, WRAM_COLS, 3).astype(np.int16)
        else:
            colours = VALUE_CMAP[wram].reshape(WRAM_ROWS, WRAM_COLS, 3).astype(np.int16)

        fade_2d = self._fade.reshape(WRAM_ROWS, WRAM_COLS)
        flash = (fade_2d[..., np.newaxis] * 180).astype(np.int16)
        colours = np.clip(colours + flash, 0, 255).astype(np.uint8)

        expanded = np.repeat(np.repeat(colours, cell, axis=0), cell, axis=1)
        # Grid lines only if cells are large enough
        if cell >= 3:
            expanded[::cell, :] = (30, 30, 30)
            expanded[:, ::cell] = (30, 30, 30)

        return pygame.surfarray.make_surface(expanded.transpose(1, 0, 2))


# ---------------------------------------------------------------------------
# OAM sprite mini-map
# ---------------------------------------------------------------------------

def render_oam_minimap(ram: np.ndarray, *, map_w: int = 192, map_h: int = 180) -> pygame.Surface:
    """Render a compact mini-map showing OAM sprite positions."""
    w, h = map_w, map_h
    sx_scale = w / NES_W
    sy_scale = h / NES_H
    pixels = np.zeros((w, h, 3), dtype=np.uint8)
    pixels[:, :] = (15, 15, 25)

    # Grid lines at every 8px NES boundary in both directions
    grid_colour = np.array([30, 30, 30], dtype=np.uint8)
    for gx in range(0, NES_W + 1, 8):
        px = round(gx * sx_scale)
        if 0 <= px < w:
            pixels[px, :] = grid_colour
    for gy in range(0, NES_H + 1, 8):
        py = round(gy * sy_scale)
        if 0 <= py < h:
            pixels[:, py] = grid_colour

    for i in range(64):
        base = 0x200 + i * 4
        sprite_y = ram[base]
        attr = ram[base + 2]
        sprite_x = ram[base + 3]

        if sprite_y >= 0xEF or sprite_y == 0:
            continue

        # Map both corners: sprites are 8px wide, 16px tall (8x16 sprite mode)
        x0 = round(sprite_x * sx_scale)
        y0 = round(sprite_y * sy_scale)
        x1 = round((sprite_x + 8) * sx_scale)
        y1 = round((sprite_y + 16) * sy_scale)

        palette = attr & 0x03
        pal_colours = [
            (255, 100, 100),
            (100, 255, 100),
            (100, 100, 255),
            (255, 255, 100),
        ]
        colour = pal_colours[palette]

        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(1, min(x1, w))
        y1 = max(1, min(y1, h))
        pixels[x0:x1, y0:y1] = colour

    return pygame.surfarray.make_surface(pixels)


# ---------------------------------------------------------------------------
# Region label drawing
# ---------------------------------------------------------------------------

def draw_ram_region_labels(surface: pygame.Surface, font: pygame.font.Font,
                           x_offset: int, y_offset: int, *,
                           cell_size: int = RAM_CELL,
                           x_min: int = 0) -> None:
    """Draw region labels with matching colours next to the grid rows."""
    cell = cell_size
    for name, colour, (r0, r1) in [
        ("Zero Page",  REGION_SEP_COLOURS["zero_page"], REGION_ROWS["zero_page"]),
        ("Stack",      REGION_SEP_COLOURS["stack"],     REGION_ROWS["stack"]),
        ("OAM",        REGION_SEP_COLOURS["oam"],       REGION_ROWS["oam"]),
        ("Game Data",  REGION_SEP_COLOURS["game_data"],  REGION_ROWS["game_data"]),
    ]:
        mid_row = (r0 + r1) / 2
        ly = y_offset + int(mid_row * cell) - 6
        label = font.render(name, True, colour)
        lx = max(x_min, x_offset - label.get_width() - 6)
        surface.blit(label, (lx, ly))
