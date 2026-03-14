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

from mario_world_model.ram_viz import (
    RAM_SIZE, RAM_COLS, RAM_ROWS, RAM_CELL,
    REGION_SEP_COLOURS, REGION_ROWS,
    NES_W, NES_H, OAM_MAP_SCALE,
    RAMGridRenderer, render_oam_minimap, draw_ram_region_labels,
)
from mario_world_model.game_decoders import decode_smb1, draw_decoded_sections


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
    ram_scale: int = 1,
):
    env = make_shimmed_env(world=world, stage=stage, seed=seed, lock_level=False)
    action_meanings = get_action_meanings()

    obs, info = env.reset(seed=seed)
    frame_h, frame_w = obs.shape[:2]

    # --- Layout ---
    game_w = frame_w * scale
    game_h = frame_h * scale

    ram_cell = RAM_CELL * ram_scale
    ram_grid_w = RAM_COLS * ram_cell
    ram_grid_h = RAM_ROWS * ram_cell
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
            val_surface = ram_renderer.render(ram, activity_mode=False, cell_size=ram_cell)
            screen.blit(val_surface, (gx, cy))
            draw_ram_region_labels(screen, font, gx, cy, cell_size=ram_cell)
            cy += ram_grid_h + grid_gap

            # Activity grid
            act_title = font.render("RAM ACTIVITY", True, (200, 200, 200))
            screen.blit(act_title, (gx, cy))
            cy += 16
            act_surface = ram_renderer.render(ram, activity_mode=True, cell_size=ram_cell)
            screen.blit(act_surface, (gx, cy))
            draw_ram_region_labels(screen, font, gx, cy, cell_size=ram_cell)
            cy += ram_grid_h + section_gap

            # Decoded variables
            sections = decode_smb1(ram)
            draw_decoded_sections(screen, font, sections, game_w + 8, cy)

            # OAM sprite mini-map (below game frame)
            oam_y = game_h
            oam_label = font.render("-- OAM SPRITES --", True, (200, 200, 100))
            screen.blit(oam_label, (4, oam_y))
            oam_y += 16
            oam_surface = render_oam_minimap(ram)
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
    parser.add_argument("--ram-scale", type=int, default=1,
                        help="RAM grid scale multiplier (default: 1)")
    args = parser.parse_args()
    run(
        mode=args.mode,
        world=args.world,
        stage=args.stage,
        seed=args.seed,
        fps=args.fps,
        scale=args.scale,
        ram_scale=args.ram_scale,
    )


if __name__ == "__main__":
    main()
