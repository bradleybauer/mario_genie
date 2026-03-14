#!/usr/bin/env python3
"""Play Mario with a live NES 2KB RAM visualizer.

Uses the same shimmed environment and heuristic/human policies as collect.py,
but renders the full 2048-byte NES RAM as a structured visualization alongside
the game frame in a single pygame window.

Features:
    - Region-coloured RAM grid (zero page / stack / OAM / game data)
    - Activity heat-map mode (toggle with TAB) showing change frequency
    - Fade-out change highlights (~15 frame decay)
    - Decoded SMB1 game variables panel
    - OAM sprite position mini-map (256x240 NES resolution)

Usage:
    # Heuristic bot with RAM visualization
    python scripts/play_ram_viz.py --mode heuristic --world 1 --stage 1

    # Human play
    python scripts/play_ram_viz.py --mode human --world 1 --stage 1

    # Random agent, natural progression
    python scripts/play_ram_viz.py --mode random

Controls (human mode):
    Arrow keys / WASD  - D-Pad
    O                  - A (jump)
    P                  - B (run)
    TAB                - Toggle value / activity heat-map
    Escape             - Quit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.actions import get_action_meanings
from mario_world_model.envs import make_shimmed_env
from mario_world_model.gamepad import GamepadState

# Reuse collect.py bot policies (not HumanActionPolicy — it creates its own window)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from collect import ActionPolicy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAM_SIZE = 2048
RAM_COLS = 64
RAM_ROWS = 32  # 64 * 32 = 2048
RAM_CELL = 6   # pixels per cell

# NES memory regions (byte ranges)
REGION_ZERO_PAGE = (0x0000, 0x00FF)  # Fast-access game variables
REGION_STACK     = (0x0100, 0x01FF)  # 6502 CPU stack
REGION_OAM       = (0x0200, 0x02FF)  # Sprite OAM DMA buffer
REGION_GAME_DATA = (0x0300, 0x07FF)  # Enemy arrays, level data, scores

# Region base tints (R, G, B) — blended into the value colour
REGION_TINTS = {
    "zero_page": np.array([80, 110, 220], dtype=np.int16),   # blue
    "stack":     np.array([90, 90, 90],   dtype=np.int16),    # grey (noise)
    "oam":       np.array([210, 80, 210],  dtype=np.int16),   # magenta
    "game_data": np.array([70, 200, 90],   dtype=np.int16),   # green
}

# Region boundary rows (used to draw separator lines)
# Each region start as a row index in the 32-row grid
REGION_ROWS = {
    "zero_page": (0, 3),    # rows 0-3   = $0000-$00FF
    "stack":     (4, 7),    # rows 4-7   = $0100-$01FF
    "oam":       (8, 11),   # rows 8-11  = $0200-$02FF
    "game_data": (12, 31),  # rows 12-31 = $0300-$07FF
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

# ---------------------------------------------------------------------------
# SMB1 RAM address map (verified against gym-super-mario-bros source)
# ---------------------------------------------------------------------------

# Player state machine ($000E)
_PLAYER_STATE_NAMES = {
    0x00: "Leftmost",
    0x01: "Vine",
    0x02: "Pipe (rev-L)",
    0x03: "Pipe (down)",
    0x04: "Auto-walk",
    0x05: "Auto-walk",
    0x06: "Dead",
    0x07: "Entering",
    0x08: "Normal",
    0x09: "Frozen",
    0x0B: "Dying",
    0x0C: "Palette cyc",
}

_PLAYER_STATUS_NAMES = {0: "Small", 1: "Big", 2: "Fire"}

# Enemy type names (partial — most common in SMB1)
_ENEMY_TYPE_NAMES = {
    0x00: "-",
    0x06: "Goomba",
    0x07: "Blooper",
    0x08: "BulletBill",
    0x09: "GreenKoopa",
    0x0A: "RedKoopa",
    0x0E: "RedKoopa(fly)",
    0x11: "Lakitu",
    0x12: "Spiny",
    0x14: "FlyFish",
    0x15: "Podoboo",
    0x1B: "FireBar",
    0x1F: "HammerBro",
    0x24: "Firework",
    0x2D: "Bowser",
    0x2E: "Fireball(B)",
    0x31: "Flagpole",
    0x35: "Toad",
}


def _decode_smb1_vars(ram: np.ndarray) -> dict:
    """Decode key game variables from SMB1 RAM."""
    # Player
    player_state_byte = ram[0x000E]
    player_state = _PLAYER_STATE_NAMES.get(player_state_byte, f"0x{player_state_byte:02X}")
    player_status = _PLAYER_STATUS_NAMES.get(ram[0x0756], f"0x{ram[0x0756]:02X}")
    x_pos = ram[0x006D] * 256 + ram[0x0086]
    y_pos = ram[0x00CE]
    y_viewport = ram[0x00B5]
    x_speed = np.int8(ram[0x0057])  # signed
    y_speed = np.int8(ram[0x001D])  # signed
    moving_dir = "R" if ram[0x0045] == 1 else ("L" if ram[0x0045] == 2 else "?")

    # Game state
    world = ram[0x075F] + 1
    stage = ram[0x075C] + 1
    lives = ram[0x075A]
    # time: 3 BCD digits at $07F8-$07FA
    time_val = ram[0x07F8] * 100 + ram[0x07F9] * 10 + ram[0x07FA]
    # coins: 2 BCD digits at $07ED-$07EE
    coins = ram[0x07ED] * 10 + ram[0x07EE]
    # score: 6 BCD digits at $07DE-$07E3
    score = 0
    for i in range(6):
        score = score * 10 + ram[0x07DE + i]
    score *= 10  # SMB1 scores are always multiples of 10

    # Gameplay mode
    gameplay_mode = ram[0x0770]
    gameplay_str = {0: "Demo", 1: "Playing", 2: "World End"}.get(gameplay_mode, f"0x{gameplay_mode:02X}")

    # Enemies (5 slots: type at $0016-$001A)
    enemies = []
    for i in range(5):
        etype = ram[0x0016 + i]
        if etype != 0:
            ex = ram[0x0087 + i]
            ey = ram[0x00CF + i]
            ename = _ENEMY_TYPE_NAMES.get(etype, f"0x{etype:02X}")
            enemies.append((ename, ex, ey))

    return {
        "player_state": player_state,
        "player_status": player_status,
        "x_pos": x_pos,
        "y_pos": y_pos,
        "y_viewport": y_viewport,
        "x_speed": x_speed,
        "y_speed": y_speed,
        "moving_dir": moving_dir,
        "world": world,
        "stage": stage,
        "lives": lives,
        "time": time_val,
        "coins": coins,
        "score": score,
        "gameplay": gameplay_str,
        "enemies": enemies,
    }


# ---------------------------------------------------------------------------
# RAM access
# ---------------------------------------------------------------------------

def _get_ram(env: gym.Env) -> np.ndarray:
    """Walk the wrapper chain and return the NES 2KB RAM as a (2048,) uint8 array."""
    e = env
    while hasattr(e, "env"):
        e = e.env
    if hasattr(e, "ram"):
        return np.array(e.ram, dtype=np.uint8)
    return np.zeros(RAM_SIZE, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Colourmap + region tint lookup (precomputed)
# ---------------------------------------------------------------------------

def _build_value_colormap() -> np.ndarray:
    """256×3 uint8 — blue→cyan→green→yellow→red for nonzero, dark for 0."""
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


def _build_region_tint_array() -> np.ndarray:
    """(2048, 3) int16 array of per-byte region tints."""
    tints = np.zeros((RAM_SIZE, 3), dtype=np.int16)
    s, e = REGION_ZERO_PAGE
    tints[s:e + 1] = REGION_TINTS["zero_page"]
    s, e = REGION_STACK
    tints[s:e + 1] = REGION_TINTS["stack"]
    s, e = REGION_OAM
    tints[s:e + 1] = REGION_TINTS["oam"]
    s, e = REGION_GAME_DATA
    tints[s:e + 1] = REGION_TINTS["game_data"]
    return tints


def _build_activity_colormap() -> np.ndarray:
    """256×3 uint8 — black (no activity) → white (max activity)."""
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for v in range(256):
        t = v / 255.0
        if t < 0.33:
            s = t / 0.33
            cmap[v] = (0, 0, int(180 * s))          # black → dark blue
        elif t < 0.66:
            s = (t - 0.33) / 0.33
            cmap[v] = (int(255 * s), 80, 180)        # dark blue → magenta
        else:
            s = (t - 0.66) / 0.34
            r = 255
            g = int(80 + 175 * s)
            b = int(180 + 75 * s)
            cmap[v] = (r, g, b)                       # magenta → white-ish
    return cmap


VALUE_CMAP = _build_value_colormap()
ACTIVITY_CMAP = _build_activity_colormap()
REGION_TINT_ARRAY = _build_region_tint_array()  # (2048, 3) int16


# ---------------------------------------------------------------------------
# RAM grid renderer
# ---------------------------------------------------------------------------

class RAMGridRenderer:
    """Renders the 2KB RAM as a 64×32 colour grid with region tints and
    fade-out change highlights."""

    def __init__(self):
        self._fade = np.zeros(RAM_SIZE, dtype=np.float32)  # 0..1 per byte
        self._activity_ring = np.zeros((ACTIVITY_WINDOW, RAM_SIZE), dtype=np.uint8)
        self._ring_idx = 0
        self._prev_ram: np.ndarray | None = None

    def update(self, ram: np.ndarray) -> None:
        """Call once per frame with the current RAM snapshot."""
        if self._prev_ram is not None:
            changed = ram != self._prev_ram
            # Fade: set to 1.0 where changed, decay elsewhere
            self._fade[changed] = 1.0
            self._fade[~changed] = np.maximum(self._fade[~changed] - 1.0 / FADE_FRAMES, 0.0)
            # Activity ring buffer
            self._activity_ring[self._ring_idx] = changed.astype(np.uint8)
        self._ring_idx = (self._ring_idx + 1) % ACTIVITY_WINDOW
        self._prev_ram = ram.copy()

    def activity_map(self) -> np.ndarray:
        """Return (2048,) float32 in [0, 1] — fraction of recent frames where each byte changed."""
        return self._activity_ring.mean(axis=0).astype(np.float32)

    def render(self, ram: np.ndarray, *, activity_mode: bool = False) -> pygame.Surface:
        """Render a (RAM_COLS*cell, RAM_ROWS*cell) surface."""
        cell = RAM_CELL
        w = RAM_COLS * cell
        h = RAM_ROWS * cell

        if activity_mode:
            # Activity heat-map: map frequency [0,1] → [0,255] → colourmap
            act = self.activity_map()
            act_u8 = np.clip(act * 255, 0, 255).astype(np.uint8)
            colours = ACTIVITY_CMAP[act_u8].reshape(RAM_ROWS, RAM_COLS, 3).astype(np.int16)
        else:
            # Value colour + region tint blend
            base = VALUE_CMAP[ram].astype(np.int16)  # (2048, 3)
            region = REGION_TINT_ARRAY                 # (2048, 3) int16
            # Where value is 0, show a visible region tint; otherwise blend 50/50
            is_zero = (ram == 0)
            blended = np.where(
                is_zero[:, np.newaxis],
                region // 3,  # visible tint for zero bytes
                (base + region) // 2,  # 50/50 blend
            )
            colours = blended.reshape(RAM_ROWS, RAM_COLS, 3)

        # Fade highlight (white flash overlay)
        fade_2d = self._fade.reshape(RAM_ROWS, RAM_COLS)
        flash = (fade_2d[..., np.newaxis] * 180).astype(np.int16)
        colours = np.clip(colours + flash, 0, 255).astype(np.uint8)

        # Expand cells — vectorised
        # colours is (ROWS, COLS, 3).  Repeat each pixel into cell×cell blocks.
        expanded = np.repeat(np.repeat(colours, cell, axis=0), cell, axis=1)  # (h, w, 3)

        # Grid lines (subtle)
        expanded[::cell, :] = (30, 30, 30)
        expanded[:, ::cell] = (30, 30, 30)

        # Region separator lines (bright, 2px)
        for rname, (r0, r1) in REGION_ROWS.items():
            sep_colour = REGION_SEP_COLOURS[rname]
            y_top = r0 * cell
            y_bot = min((r1 + 1) * cell, h - 1)
            if y_top > 0:
                expanded[max(0, y_top - 1):y_top + 1, :] = sep_colour
            if y_bot < h - 1:
                expanded[y_bot:min(y_bot + 2, h), :] = sep_colour

        # pygame wants (w, h, 3) with x as first axis
        return pygame.surfarray.make_surface(expanded.transpose(1, 0, 2))


# ---------------------------------------------------------------------------
# OAM sprite mini-map
# ---------------------------------------------------------------------------

NES_W, NES_H = 256, 240
OAM_MAP_SCALE = 1  # 1:1 NES pixels


def _render_oam_minimap(ram: np.ndarray) -> pygame.Surface:
    """Render a 256×240 mini-map showing OAM sprite positions."""
    w, h = NES_W * OAM_MAP_SCALE, NES_H * OAM_MAP_SCALE
    pixels = np.zeros((w, h, 3), dtype=np.uint8)
    pixels[:, :] = (15, 15, 25)  # dark background

    # Draw scanline guides every 16px
    for sy in range(0, NES_H, 16):
        pixels[:, sy * OAM_MAP_SCALE] = (25, 25, 35)

    # Parse OAM: 64 sprites × 4 bytes at $0200
    for i in range(64):
        base = 0x200 + i * 4
        sprite_y = ram[base]
        tile = ram[base + 1]
        attr = ram[base + 2]
        sprite_x = ram[base + 3]

        # Off-screen sprites have Y >= 0xEF
        if sprite_y >= 0xEF or sprite_y == 0:
            continue

        # Sprite is 8×8 or 8×16.  Draw an 8×8 dot.
        sx = sprite_x * OAM_MAP_SCALE
        sy = sprite_y * OAM_MAP_SCALE
        sz = max(6 * OAM_MAP_SCALE, 4)

        # Colour by palette (attr bits 0-1)
        palette = attr & 0x03
        colours = [
            (255, 100, 100),  # palette 0 — red (usually Mario)
            (100, 255, 100),  # palette 1 — green
            (100, 100, 255),  # palette 2 — blue
            (255, 255, 100),  # palette 3 — yellow
        ]
        colour = colours[palette]

        x0 = max(0, min(sx, w - sz))
        y0 = max(0, min(sy, h - sz))
        x1 = min(x0 + sz, w)
        y1 = min(y0 + sz, h)
        pixels[x0:x1, y0:y1] = colour

    # Border
    pixels[0, :] = pixels[-1, :] = (60, 60, 60)
    pixels[:, 0] = pixels[:, -1] = (60, 60, 60)

    return pygame.surfarray.make_surface(pixels)


# ---------------------------------------------------------------------------
# Text rendering helpers
# ---------------------------------------------------------------------------

def _draw_decoded_vars(surface: pygame.Surface, font: pygame.font.Font,
                       ram: np.ndarray, x: int, y: int) -> int:
    """Draw decoded SMB1 variables.  Returns y after last line."""
    v = _decode_smb1_vars(ram)
    sections = [
        ("-- PLAYER --", [
            f"State:  {v['player_state']}",
            f"Status: {v['player_status']}  Dir: {v['moving_dir']}",
            f"Pos:    ({v['x_pos']}, {v['y_pos']})  YVP: {v['y_viewport']}",
            f"Speed:  X={v['x_speed']:+4d}  Y={v['y_speed']:+4d}",
        ]),
        ("-- GAME --", [
            f"World {v['world']}-{v['stage']}  Mode: {v['gameplay']}",
            f"Lives: {v['lives']}  Time: {v['time']}",
            f"Score: {v['score']}  Coins: {v['coins']}",
        ]),
    ]
    if v["enemies"]:
        enemy_lines = [f"  {name} ({ex},{ey})" for name, ex, ey in v["enemies"]]
        sections.append((f"-- ENEMIES ({len(v['enemies'])}) --", enemy_lines))
    else:
        sections.append(("-- ENEMIES --", ["  (none)"]))

    line_h = 15
    cy = y
    for header, lines in sections:
        text = font.render(header, True, (200, 200, 100))
        surface.blit(text, (x, cy))
        cy += line_h
        for line in lines:
            text = font.render(line, True, (180, 180, 180))
            surface.blit(text, (x + 4, cy))
            cy += line_h
        cy += 4  # gap between sections

    return cy


def _draw_ram_region_labels(surface: pygame.Surface, font: pygame.font.Font,
                            x_offset: int, y_offset: int) -> None:
    """Draw region labels with matching colours next to the grid rows."""
    cell = RAM_CELL
    region_labels = [
        ("Zero Page",  REGION_SEP_COLOURS["zero_page"], REGION_ROWS["zero_page"]),
        ("Stack",      REGION_SEP_COLOURS["stack"],     REGION_ROWS["stack"]),
        ("OAM",        REGION_SEP_COLOURS["oam"],       REGION_ROWS["oam"]),
        ("Game Data",  REGION_SEP_COLOURS["game_data"],  REGION_ROWS["game_data"]),
    ]
    for name, colour, (r0, r1) in region_labels:
        mid_row = (r0 + r1) / 2
        ly = y_offset + int(mid_row * cell) - 6
        label = font.render(name, True, colour)
        lw = label.get_width()
        surface.blit(label, (x_offset - lw - 6, ly))


# ---------------------------------------------------------------------------
# Human input (inline — avoids HumanActionPolicy creating a second window)
# ---------------------------------------------------------------------------

_KEY_MAP = {
    pygame.K_RIGHT: "RIGHT", pygame.K_d: "RIGHT",
    pygame.K_LEFT: "LEFT",   pygame.K_a: "LEFT",
    pygame.K_UP: "UP",       pygame.K_w: "UP",
    pygame.K_DOWN: "DOWN",   pygame.K_s: "DOWN",
    pygame.K_o: "A",
    pygame.K_p: "B",
}


class _HumanInput:
    """Reads keyboard + gamepad and returns an action index for the shimmed env."""

    def __init__(self, action_meanings: list[list[str]]):
        self._action_to_index = {
            frozenset(t.lower() for t in a): i for i, a in enumerate(action_meanings)
        }
        self._noop = self._action_to_index.get(frozenset({"noop"}), 0)
        self._gamepad = GamepadState()
        self._held: set[str] = set()

    def process_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            btn = _KEY_MAP.get(event.key)
            if btn:
                self._held.add(btn)
        elif event.type == pygame.KEYUP:
            btn = _KEY_MAP.get(event.key)
            if btn:
                self._held.discard(btn)

    def sample(self) -> int:
        self._gamepad.poll()
        gp = self._gamepad.read_buttons()
        right = "RIGHT" in self._held or gp.right
        left = "LEFT" in self._held or gp.left
        up = "UP" in self._held or gp.up
        down = "DOWN" in self._held or gp.down
        jump = "A" in self._held or gp.a_btn
        sprint = "B" in self._held or gp.b_btn

        if right and left:
            right = left = False
        if up and not down:
            return self._lookup({"up"})
        buttons: set[str] = set()
        if down:
            buttons.add("down")
        if right:
            buttons.add("right")
        elif left:
            buttons.add("left")
        if jump:
            buttons.add("a")
        if sprint:
            buttons.add("b")
        return self._lookup(buttons) if buttons else self._noop

    def _lookup(self, buttons: set[str]) -> int:
        return self._action_to_index.get(frozenset(buttons), self._noop)


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def run(
    mode: str,
    world: Optional[int],
    stage: Optional[int],
    seed: int,
    fps: int,
    scale: int,
):
    env = make_shimmed_env(world=world, stage=stage, seed=seed, lock_level=False)
    action_meanings = get_action_meanings()

    obs, info = env.reset(seed=seed)
    frame_h, frame_w = obs.shape[:2]

    # --- Layout ---
    game_w = frame_w * scale
    game_h = frame_h * scale

    ram_grid_w = RAM_COLS * RAM_CELL
    ram_grid_h = RAM_ROWS * RAM_CELL
    ram_label_margin = 80   # space for region labels on left
    grid_gap = 20           # vertical gap between the two grids
    section_gap = 10
    top_margin = 20

    oam_map_w = NES_W * OAM_MAP_SCALE
    oam_map_h = NES_H * OAM_MAP_SCALE

    # Right panel: value grid, activity grid, decoded vars
    two_grids_h = ram_grid_h * 2 + grid_gap + 16 * 2  # 16px for each title
    decoded_h = 220
    panel_content_w = ram_label_margin + ram_grid_w
    panel_w = panel_content_w + 16
    panel_h = top_margin + two_grids_h + section_gap + decoded_h + 20

    # OAM minimap goes below the game frame
    oam_section_h = 16 + oam_map_h + 8  # title + map + pad
    left_col_h = game_h + oam_section_h

    win_w = game_w + panel_w
    win_h = max(left_col_h, panel_h)

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Mario + RAM Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 13)

    if mode == "human":
        human_input = _HumanInput(action_meanings=action_meanings)
    else:
        human_input = None
        policy = ActionPolicy(mode=mode, action_meanings=action_meanings, seed=seed)

    ram_renderer = RAMGridRenderer()
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                if human_input is not None:
                    human_input.process_event(event)

            if not running:
                break

            # Sample action
            if human_input is not None:
                action = human_input.sample()
            else:
                action = policy.sample(info)

            # Step
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()

            # Read RAM
            ram = _get_ram(env)
            ram_renderer.update(ram)

            # --- Draw ---
            screen.fill((10, 10, 15))

            # Game frame (left side)
            game_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            game_surface = pygame.transform.scale(game_surface, (game_w, game_h))
            screen.blit(game_surface, (0, 0))

            # --- Right panel ---
            gx = game_w + ram_label_margin  # grid X position
            cy = top_margin                 # cursor Y

            # Value grid
            val_title = font.render("RAM VALUES", True, (200, 200, 200))
            screen.blit(val_title, (gx, cy))
            cy += 16
            val_surface = ram_renderer.render(ram, activity_mode=False)
            screen.blit(val_surface, (gx, cy))
            _draw_ram_region_labels(screen, font, gx, cy)
            cy += ram_grid_h + grid_gap

            # Activity grid
            act_title = font.render("RAM ACTIVITY", True, (200, 200, 200))
            screen.blit(act_title, (gx, cy))
            cy += 16
            act_surface = ram_renderer.render(ram, activity_mode=True)
            screen.blit(act_surface, (gx, cy))
            _draw_ram_region_labels(screen, font, gx, cy)
            cy += ram_grid_h + section_gap

            # Decoded variables
            _draw_decoded_vars(screen, font, ram, game_w + 8, cy)

            # OAM sprite mini-map (below game frame)
            oam_y = game_h
            oam_label = font.render("-- OAM SPRITES --", True, (200, 200, 100))
            screen.blit(oam_label, (4, oam_y))
            oam_y += 16
            oam_surface = _render_oam_minimap(ram)
            screen.blit(oam_surface, (4, oam_y))

            pygame.display.flip()
            clock.tick(fps)

    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
        env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Play Mario with live NES RAM visualization",
    )
    parser.add_argument(
        "--mode", type=str, default="heuristic",
        choices=["random", "heuristic", "human"],
    )
    parser.add_argument("--world", type=int, default=None)
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--scale", type=int, default=2, help="Game frame scale factor")
    args = parser.parse_args()
    run(
        mode=args.mode,
        world=args.world,
        stage=args.stage,
        seed=args.seed,
        fps=args.fps,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
