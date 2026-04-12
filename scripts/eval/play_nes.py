"""Play NES ROMs using nes_py or stable-retro.

Usage:
    python scripts/play_nes.py              # interactive menu
    python scripts/play_nes.py --rom mario  # partial name match
    python scripts/play_nes.py --rom 3      # pick by number
    python scripts/play_nes.py --ram --rom mario  # with RAM visualizer
    python scripts/play_nes.py --scale 4     # 4x game frame
    python scripts/play_nes.py --ram --ram-scale 2 --rom mario  # 2x RAM grid size

nes_py (LaiNES core) is used for mappers 0, 1, 2, 3.
stable-retro (FCEUmm core) is used as a fallback for all other mappers.

Controls:
    Arrow keys / WASD  - D-Pad
    X / O              - A (jump / confirm)
    Z / P              - B (run / back)
    Enter / Space      - Start
    Right Shift        - Select
    Escape / Q         - Quit
    Gamepad also supported via evdev.
"""
import argparse
from pathlib import Path
import sys
import os
import glob
import hashlib
import json
import shutil
import time

import numpy as np
import pygame
import stable_retro as retro
from nes_py._rom import ROM
from nes_py.nes_env import NESEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.data.gamepad import GamepadState

# --- Configuration ---
ROM_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'roms')
DEFAULT_SCALE = 3
FPS = 60
# Mappers supported by nes_py's LaiNES core
SUPPORTED_MAPPERS = {0, 1, 2, 3}

# NES controller bit layout used by nes_py (one byte, 8 bits)
#   bit 0 = A,  1 = B,  2 = Select, 3 = Start,
#   bit 4 = Up, 5 = Down, 6 = Left,  7 = Right
BUTTON_BITS = {
    'A': 0, 'B': 1, 'SELECT': 2, 'START': 3,
    'UP': 4, 'DOWN': 5, 'LEFT': 6, 'RIGHT': 7,
}

from src.data.ram_viz import (
    RAM_SIZE, RAM_COLS, RAM_ROWS, RAM_CELL,
    WRAM_SIZE, WRAM_COLS, WRAM_ROWS, WRAM_CELL,
    LEVEL_COLS_PER_SCREEN,
    RAMGridRenderer, WRAMGridRenderer,
    render_oam_minimap, draw_ram_region_labels,
    render_level_tilemap,
)
from src.data.game_decoders import (
    get_decoder, draw_decoded_sections, WRAM_OFFSET,
)

# stable-retro MultiBinary(9) button layout:
# index: 0=B, 1=None, 2=SELECT, 3=START, 4=UP, 5=DOWN, 6=LEFT, 7=RIGHT, 8=A
RETRO_BUTTON_INDEX = {
    'B': 0, 'SELECT': 2, 'START': 3,
    'UP': 4, 'DOWN': 5, 'LEFT': 6, 'RIGHT': 7, 'A': 8,
}


def nes_byte_to_retro_action(byte: int) -> np.ndarray:
    """Convert an NES controller byte to a stable-retro MultiBinary(9) action."""
    action = np.zeros(9, dtype=np.int8)
    for name, bit in BUTTON_BITS.items():
        if byte & (1 << bit):
            action[RETRO_BUTTON_INDEX[name]] = 1
    return action


def ensure_retro_game(rom_path: str) -> str:
    """Register a ROM with stable-retro if needed and return the game id (with -v0 suffix removed).

    Creates a minimal game directory under retro's data/stable/ so that
    retro.make() can find it.  If the game is already registered and the
    ROM hash matches, this is a no-op.
    """
    rom_basename = os.path.basename(rom_path)
    # Sanitise name: strip extension, replace non-alphanumeric with hyphens
    game_name = os.path.splitext(rom_basename)[0]
    safe_name = ''.join(c if c.isalnum() else '-' for c in game_name)
    # Collapse runs of hyphens and strip leading/trailing
    while '--' in safe_name:
        safe_name = safe_name.replace('--', '-')
    safe_name = safe_name.strip('-')
    retro_id = f"{safe_name}-Nes"

    # Compute ROM SHA1
    with open(rom_path, 'rb') as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()

    # Check if already registered with matching hash
    try:
        existing_path = retro.data.path(retro_id)
        sha_file = os.path.join(existing_path, 'rom.sha')
        if os.path.exists(sha_file):
            with open(sha_file) as f:
                if sha1 in f.read():
                    return retro_id
    except FileNotFoundError:
        pass

    # Create minimal game directory
    data_dir = retro.data.path()
    game_dir = os.path.join(data_dir, 'stable', retro_id)
    os.makedirs(game_dir, exist_ok=True)

    shutil.copy2(rom_path, os.path.join(game_dir, 'rom.nes'))

    with open(os.path.join(game_dir, 'rom.sha'), 'w') as f:
        f.write(sha1 + '\n')

    with open(os.path.join(game_dir, 'metadata.json'), 'w') as f:
        json.dump({'default_player_state': '', 'platform': 'Nes'}, f)

    with open(os.path.join(game_dir, 'data.json'), 'w') as f:
        json.dump({'info': {}}, f)

    with open(os.path.join(game_dir, 'scenario.json'), 'w') as f:
        json.dump({'done': {'condition': 'never'}, 'reward': {'variables': {}}}, f)

    return retro_id


def discover_roms():
    """Scan ROM_DIR for .nes files and return [(display_name, path, mapper, backend)].

    backend is 'nes_py' for mappers 0-3, 'retro' if stable-retro is available,
    or None if neither can handle the ROM.
    """
    roms = []
    for path in sorted(glob.glob(os.path.join(ROM_DIR, '*.nes'))):
        basename = os.path.basename(path)
        name = os.path.splitext(basename)[0]
        try:
            r = ROM(path)
            if r.mapper in SUPPORTED_MAPPERS:
                backend = 'nes_py'
            else:
                backend = 'retro'
            roms.append((name, path, r.mapper, backend))
        except Exception:
            roms.append((name, path, -1, 'retro'))
    return roms


def choose_rom(roms, query=None):
    """Pick a ROM by query string or interactive menu."""
    if query is not None:
        # Try as a number first
        if query.isdigit():
            idx = int(query) - 1
            if 0 <= idx < len(roms):
                return roms[idx]
        # Partial name match
        matches = [r for r in roms if query.lower() in r[0].lower()]
        if len(matches) == 1:
            return matches[0]
        if matches:
            print(f"Ambiguous match for '{query}':")
            for m in matches:
                print(f"  - {m[0]}")
        else:
            print(f"No ROM matches '{query}'.")
        print()

    print("Available ROMs:")
    for i, (name, _, mapper, backend) in enumerate(roms, 1):
        if backend == 'nes_py':
            tag = ""
        elif backend == 'retro':
            tag = "  [retro]"
        else:
            tag = f"  [mapper {mapper} - no backend available]"
        print(f"  {i:2d}. {name}{tag}")
    while True:
        choice = input(f"Pick a ROM (1-{len(roms)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(roms):
            return roms[int(choice) - 1]
        print("Invalid choice, try again.")


# Keyboard keys tracked via KEYDOWN/KEYUP events for reliable detection.
# Maps pygame key constant -> NES button name.
KEY_MAP = {
    pygame.K_RIGHT: 'RIGHT', pygame.K_d: 'RIGHT',
    pygame.K_LEFT: 'LEFT',   pygame.K_a: 'LEFT',
    pygame.K_UP: 'UP',       pygame.K_w: 'UP',
    pygame.K_DOWN: 'DOWN',   pygame.K_s: 'DOWN',
    pygame.K_x: 'A',         pygame.K_o: 'A',
    pygame.K_z: 'B',         pygame.K_p: 'B',
    pygame.K_RETURN: 'START', pygame.K_KP_ENTER: 'START', pygame.K_SPACE: 'START',
    pygame.K_RSHIFT: 'SELECT',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play NES ROMs using nes_py or stable-retro.")
    parser.add_argument("--rom", default=None, help="ROM number or partial name match.")
    parser.add_argument("--ram", action="store_true", help="Enable the RAM visualizer panel.")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE,
                        help=f"Game frame scale factor (default: {DEFAULT_SCALE})")
    parser.add_argument("--ram-scale", type=int, default=1,
                        help="RAM grid scale multiplier (default: 1)")
    return parser.parse_args()


class GamepadController:
    """Reads keyboard + gamepad and returns an NES controller byte."""
    def __init__(self):
        self._gamepad = GamepadState(extended_button_codes=True)
        self._held: set[str] = set()  # NES button names currently held

    def process_event(self, event: pygame.event.Event) -> None:
        """Track key presses/releases via events (more reliable than get_pressed)."""
        if event.type == pygame.KEYDOWN:
            btn = KEY_MAP.get(event.key)
            if btn:
                self._held.add(btn)
        elif event.type == pygame.KEYUP:
            btn = KEY_MAP.get(event.key)
            if btn:
                self._held.discard(btn)

    def get_action(self) -> int:
        """Return an NES controller byte (0-255) from keyboard + gamepad state."""
        self._gamepad.poll()

        right = 'RIGHT' in self._held
        left = 'LEFT' in self._held
        up = 'UP' in self._held
        down = 'DOWN' in self._held
        a_btn = 'A' in self._held
        b_btn = 'B' in self._held
        start = 'START' in self._held
        sel = 'SELECT' in self._held

        gp = self._gamepad.read_buttons()
        right = right or gp.right
        left = left or gp.left
        up = up or gp.up
        down = down or gp.down
        a_btn = a_btn or gp.a_btn
        b_btn = b_btn or gp.b_btn
        start = start or gp.start
        sel = sel or gp.select

        if right and left:
            right = left = False
        if up and down:
            up = down = False

        byte = 0
        for name, pressed in [('A', a_btn), ('B', b_btn), ('SELECT', sel),
                               ('START', start), ('UP', up), ('DOWN', down),
                               ('LEFT', left), ('RIGHT', right)]:
            if pressed:
                byte |= (1 << BUTTON_BITS[name])
        return byte


def main():
    args = parse_args()
    show_ram = args.ram
    rom_query = args.rom
    scale = args.scale
    ram_scale = args.ram_scale

    if scale < 1:
        raise SystemExit("--scale must be at least 1")
    if ram_scale < 1:
        raise SystemExit("--ram-scale must be at least 1")

    ram_cell = RAM_CELL * ram_scale

    roms = discover_roms()
    if not roms:
        print(f"No .nes files found in {ROM_DIR}")
        return

    name, path, mapper, backend = choose_rom(roms, rom_query)
    if backend is None:
        print(f"\n'{name}' uses mapper {mapper} which nes_py doesn't support.")
        print("Install stable-retro (`pip install stable-retro`) for broader mapper support.")
        return

    print(f"Loading {name} (backend: {backend}) ...")
    print("Controls: Arrows/WASD=D-Pad  X/O=A  Z/P=B  Enter/Space=Start  RShift=Select  Esc=Quit")

    decoder = get_decoder(name) if show_ram else None
    if decoder:
        print(f"Game decoder: {decoder.__name__}")

    if backend == 'retro':
        if show_ram:
            _run_retro_pygame(name, path, scale=scale, ram_cell=ram_cell, decoder=decoder)
        else:
            _run_retro(name, path, scale=scale)
    else:
        _run_nes_py(name, path, show_ram=show_ram, scale=scale, ram_cell=ram_cell, decoder=decoder)


def _run_nes_py(name, path, show_ram=False, scale=DEFAULT_SCALE, ram_cell=RAM_CELL, decoder=None):
    """Game loop using nes_py + pygame."""
    env = NESEnv(path)
    obs = env.reset()
    h, w, _ = obs.shape
    game_w = w * scale
    game_h = h * scale
    print(f'Resolution: {game_w}x{game_h} (native {w}x{h}, scale {scale}x)')

    FPS_BAR_H = 18  # height of status bar when no RAM panel
    if show_ram:
        win_w, win_h, layout = _ram_layout(game_w, game_h, ram_cell, has_decoder=decoder is not None)
    else:
        win_w, win_h = game_w, game_h + FPS_BAR_H

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"{name} (nes_py)")
    clock = pygame.time.Clock()
    controller = GamepadController()
    font = pygame.font.SysFont('monospace', 13) if show_ram else None
    ram_renderer = RAMGridRenderer() if show_ram else None

    fps_font = pygame.font.SysFont('monospace', 14, bold=True)
    fps_surface = fps_font.render('-- FPS', True, (0, 255, 0))
    fps_frame = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            controller.process_event(event)

        action = controller.get_action()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
        print(f'\robs.shape={obs.shape}', end='', flush=True)

        screen.fill((10, 10, 15))
        surface = pygame.surfarray.make_surface(np.swapaxes(obs, 0, 1))
        screen.blit(pygame.transform.scale(surface, (game_w, game_h)), (0, 0))

        if show_ram:
            full_ram = np.array(env.ram, dtype=np.uint8)
            ram = full_ram[:RAM_SIZE]
            _draw_ram_panel(screen, font, ram, ram_renderer, layout, game_w, game_h, ram_cell, decoder, full_ram=full_ram, rom_name=name)

        # Discrete FPS indicator (updated every 30 frames)
        fps_frame += 1
        if fps_frame >= 30:
            fps_val = clock.get_fps()
            fps_surface = fps_font.render(f'{fps_val:.0f} FPS', True, (0, 255, 0))
            fps_frame = 0
        if show_ram:
            screen.blit(fps_surface, (game_w + 4, win_h - fps_surface.get_height() - 4))
        else:
            screen.blit(fps_surface, (4, game_h + 2))

        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()


def _run_retro_pygame(name, path, scale=DEFAULT_SCALE, ram_cell=RAM_CELL, decoder=None):
    """Game loop using stable-retro rendered via pygame (for RAM viz)."""
    game_id = ensure_retro_game(path)
    env = retro.make(game_id, state=retro.State.NONE, render_mode=None,
                     use_restricted_actions=retro.Actions.ALL)
    obs, _info = env.reset()
    h, w, _ = obs.shape
    game_w = w * scale
    game_h = h * scale
    print(f'Resolution: {game_w}x{game_h} (native {w}x{h}, scale {scale}x)')

    # Check for WRAM support
    show_wram = len(env.get_ram()) > WRAM_OFFSET + WRAM_SIZE // 2
    wram_renderer = WRAMGridRenderer() if show_wram else None

    win_w, win_h, layout = _ram_layout(game_w, game_h, ram_cell,
                                       has_decoder=decoder is not None,
                                       has_wram=show_wram)

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h), pygame.DOUBLEBUF)
    pygame.display.set_caption(f"{name} (retro + RAM)")
    clock = pygame.time.Clock()
    controller = GamepadController()
    font = pygame.font.SysFont('monospace', 13)
    ram_renderer = RAMGridRenderer()

    fps_font = pygame.font.SysFont('monospace', 14, bold=True)
    fps_surface = fps_font.render('-- FPS', True, (0, 255, 0))
    fps_frame = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            controller.process_event(event)

        action_byte = controller.get_action()
        obs, reward, terminated, truncated, info = env.step(
            nes_byte_to_retro_action(action_byte)
        )
        if terminated or truncated:
            obs, _info = env.reset()
        print(f'\robs.shape={obs.shape}', end='', flush=True)

        screen.fill((10, 10, 15))
        surface = pygame.surfarray.make_surface(np.swapaxes(obs, 0, 1))
        screen.blit(pygame.transform.scale(surface, (game_w, game_h)), (0, 0))

        full_ram = np.array(env.get_ram(), dtype=np.uint8)
        ram = full_ram[:RAM_SIZE]
        _draw_ram_panel(screen, font, ram, ram_renderer, layout, game_w, game_h,
                        ram_cell, decoder, full_ram=full_ram,
                        wram_renderer=wram_renderer, rom_name=name)

        # Discrete FPS indicator (updated every 30 frames)
        fps_frame += 1
        if fps_frame >= 30:
            fps_val = clock.get_fps()
            fps_surface = fps_font.render(f'{fps_val:.0f} FPS', True, (0, 255, 0))
            fps_frame = 0
        screen.blit(fps_surface, (game_w + 4, win_h - fps_surface.get_height() - 4))

        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()


def _ram_layout(game_w, game_h, cell_size=RAM_CELL, has_decoder=False,
                wram_cell=WRAM_CELL, has_wram=False):
    """Compute layout metrics for the RAM panel.  Returns (win_w, win_h, layout_dict).

    Layout (right panel):
        Row 1:  [labels 80px]  RAM VALUES  [gap]  RAM ACTIVITY
        Row 2:  [labels 80px]  WRAM VALUES [gap]  WRAM ACTIVITY   (if has_wram)
        Below:  [decoded col1] [decoded col2] [OAM minimap]
    Below game frame: level tilemap (if has_wram)
    """
    ram_grid_w = RAM_COLS * cell_size
    ram_grid_h = RAM_ROWS * cell_size
    wram_grid_w = WRAM_COLS * wram_cell
    wram_grid_h = WRAM_ROWS * wram_cell

    label_margin = 88
    grid_gap = 20
    top_margin = 20
    row_gap = 24          # vertical gap between grid rows

    # Right panel: two grids side-by-side per row
    pair_w = max(ram_grid_w, wram_grid_w if has_wram else 0)
    panel_w = label_margin + pair_w + grid_gap + pair_w + 16

    # Right panel height: grids + decoded/OAM below
    ram_row_h = 16 + ram_grid_h        # title + grid
    wram_row_h = (16 + wram_grid_h) if has_wram else 0
    decoded_h = 250 if has_decoder else 0
    panel_h = top_margin + ram_row_h + (row_gap + wram_row_h if has_wram else 0) + 20 + decoded_h

    # Level/overworld tilemap below game frame (SMB3 with WRAM)
    # Overworld auto-scales up, so allocate extra vertical space
    level_map_h = (16 + 300 + 8) if has_wram else 0  # title + grid + pad

    win_w = game_w + panel_w
    win_h = max(game_h + level_map_h, panel_h)

    layout = dict(
        label_margin=label_margin,
        grid_gap=grid_gap,
        top_margin=top_margin,
        row_gap=row_gap,
        ram_grid_w=ram_grid_w,
        ram_grid_h=ram_grid_h,
        wram_grid_w=wram_grid_w,
        wram_grid_h=wram_grid_h,
        wram_cell=wram_cell,
        pair_w=pair_w,
        level_map_h=level_map_h,
    )
    return win_w, win_h, layout


def _draw_ram_panel(screen, font, ram, ram_renderer, layout, game_w, game_h,
                    cell_size=RAM_CELL, decoder=None, full_ram=None,
                    wram_renderer=None, rom_name=''):
    """Draw RAM grids, WRAM grids, OAM minimap, and decoded vars."""
    ram_renderer.update(ram)
    gx = game_w + layout['label_margin']
    cy = layout['top_margin']
    gap = layout['grid_gap']
    pair_w = layout['pair_w']

    # --- Row 1: RAM VALUES | RAM ACTIVITY (side by side) ---
    val_title = font.render('RAM VALUES', True, (200, 200, 200))
    act_title = font.render('RAM ACTIVITY', True, (200, 200, 200))
    screen.blit(val_title, (gx, cy))
    screen.blit(act_title, (gx + pair_w + gap, cy))
    cy += 16

    screen.blit(ram_renderer.render(ram, activity_mode=False, cell_size=cell_size), (gx, cy))
    draw_ram_region_labels(screen, font, gx, cy, cell_size=cell_size, x_min=game_w + 4)

    ax = gx + pair_w + gap
    screen.blit(ram_renderer.render(ram, activity_mode=True, cell_size=cell_size), (ax, cy))

    cy += layout['ram_grid_h']

    # --- Row 2: WRAM VALUES | WRAM ACTIVITY (side by side, if available) ---
    if wram_renderer is not None and full_ram is not None and len(full_ram) > WRAM_OFFSET:
        wram = full_ram[WRAM_OFFSET:WRAM_OFFSET + WRAM_SIZE]
        if len(wram) == WRAM_SIZE:
            wram_renderer.update(wram)
            wc = layout['wram_cell']
            cy += layout['row_gap']

            wt = font.render('WRAM VALUES', True, (200, 200, 200))
            wat = font.render('WRAM ACTIVITY', True, (200, 200, 200))
            screen.blit(wt, (gx, cy))
            screen.blit(wat, (gx + pair_w + gap, cy))
            cy += 16

            screen.blit(wram_renderer.render(wram, activity_mode=False, cell_size=wc), (gx, cy))

            screen.blit(wram_renderer.render(wram, activity_mode=True, cell_size=wc), (gx + pair_w + gap, cy))
            cy += layout['wram_grid_h']

    # --- Decoded vars (col1 + col2) + OAM minimap (col3) below the grids ---
    if decoder is not None:
        cy += 12
        sections = decoder(full_ram if full_ram is not None else ram)
        # Split: PLAYER+STATUS in col1, GAME+WRAM in col2
        oam_surface = render_oam_minimap(ram)
        oam_w = oam_surface.get_width()
        col_w = (pair_w * 2 + gap - oam_w - 12) // 2
        col2_x = gx + col_w + 8
        col2_start = min(2, len(sections))  # first 2 sections in col1
        draw_decoded_sections(screen, font, sections, gx, cy,
                              col2_x=col2_x, col2_after=col2_start)
    else:
        oam_surface = render_oam_minimap(ram)
        oam_w = oam_surface.get_width()

    # OAM minimap as third column
    oam_x = gx + pair_w * 2 + gap - oam_w
    oam_label = font.render('OAM SPRITES', True, (200, 200, 100))
    screen.blit(oam_label, (oam_x, cy if decoder else cy + 12))
    screen.blit(oam_surface, (oam_x, (cy if decoder else cy + 12) + 14))

    # --- Level / overworld tilemap below the game frame (SMB3 only) ---
    _is_smb3 = 'super mario bros. 3' in rom_name.lower() or 'super mario bros 3' in rom_name.lower()
    if _is_smb3 and full_ram is not None and len(full_ram) > WRAM_OFFSET + WRAM_SIZE // 2:
        # Detect overworld vs in-level
        time_val = (int(full_ram[0x05EE]) * 100
                    + int(full_ram[0x05EF]) * 10
                    + int(full_ram[0x05F0]))
        objset = int(full_ram[0x070A])
        is_overworld = not (time_val > 0 and objset > 0)

        if is_overworld:
            # Overworld cursor position: $0079 (X px), $0075 (Y px), $0077 (page)
            # Positions are in NES screen pixels, multiples of 0x20 (32 decimal).
            # The tile buffer row = y_px/16 + 13 (scroll offset to map area).
            map_page = int(full_ram[0x0077])
            map_x_px = int(full_ram[0x0079])
            map_y_px = int(full_ram[0x0075])
            OVERWORLD_ROW_OFFSET = 15
            px = map_page * LEVEL_COLS_PER_SCREEN + map_x_px // 16
            py = map_y_px // 16 + OVERWORLD_ROW_OFFSET
            level_surf = render_level_tilemap(full_ram, player_x=px,
                                              player_y=py, overworld=True)
            label_text = 'OVERWORLD MAP'
        else:
            player_page = int(full_ram[0x0075]) // 2
            px = int(full_ram[0x0075]) * 256 + int(full_ram[0x0090])
            py = int(full_ram[0x0087]) * 256 + int(full_ram[0x00A2])
            enemy_pos = []
            for i in range(5):
                ex = int(full_ram[0x0076 + i]) * 256 + int(full_ram[0x0091 + i])
                ey = int(full_ram[0x0088 + i]) * 256 + int(full_ram[0x00A3 + i])
                if ex > 0 or ey > 0:
                    enemy_pos.append((ex, ey))
            level_surf = render_level_tilemap(full_ram, player_page=player_page,
                                              player_x=px, player_y=py,
                                              enemies=enemy_pos)
            label_text = 'LEVEL TILEMAP'

        if level_surf is not None:
            ly = game_h + 4
            level_label = font.render(label_text, True, (200, 200, 100))
            screen.blit(level_label, (4, ly))
            screen.blit(level_surf, (4, ly + 14))


def _run_retro(name, path, scale=DEFAULT_SCALE):
    """Game loop using stable-retro's native viewer (pyglet) at given scale."""
    try:
        import pyglet.window.key as pkey
    except ImportError as exc:
        raise SystemExit(
            "Retro viewer requires pyglet OpenGL dependencies (missing GLU). "
            "Install system package libglu1-mesa (Ubuntu/Debian) and retry."
        ) from exc

    # Pyglet key -> NES button name (same bindings as pygame KEY_MAP)
    PYGLET_KEY_MAP = {
        pkey.RIGHT: 'RIGHT', pkey.D: 'RIGHT',
        pkey.LEFT: 'LEFT',   pkey.A: 'LEFT',
        pkey.UP: 'UP',       pkey.W: 'UP',
        pkey.DOWN: 'DOWN',   pkey.S: 'DOWN',
        pkey.X: 'A',         pkey.O: 'A',
        pkey.Z: 'B',         pkey.P: 'B',
        pkey.RETURN: 'START', pkey.SPACE: 'START',
        pkey.RSHIFT: 'SELECT',
    }

    game_id = ensure_retro_game(path)
    env = retro.make(game_id, state=retro.State.NONE, render_mode='human',
                     use_restricted_actions=retro.Actions.ALL)
    obs, _info = env.reset()
    h, w, _ = obs.shape
    print(f'Resolution: {w * scale}x{h * scale} (native {w}x{h}, scale {scale}x)')

    # First step triggers viewer creation
    obs, *_ = env.step(np.zeros(9, dtype=np.int8))

    # Resize the viewer window (GL stretches the texture to fill)
    env.viewer.window.set_size(w * scale, h * scale)
    env.viewer.window.set_caption(f"{name} (retro)")

    # Hook keyboard events on the pyglet viewer window
    held: set[str] = set()
    quit_requested = False

    @env.viewer.window.event
    def on_key_press(symbol, modifiers):
        nonlocal quit_requested
        if symbol in (pkey.ESCAPE, pkey.Q):
            quit_requested = True
            return
        btn = PYGLET_KEY_MAP.get(symbol)
        if btn:
            held.add(btn)

    @env.viewer.window.event
    def on_key_release(symbol, modifiers):
        btn = PYGLET_KEY_MAP.get(symbol)
        if btn:
            held.discard(btn)

    # Gamepad via evdev / USB (does not need pygame)
    gamepad = GamepadState(extended_button_codes=True)
    target_dt = 1.0 / FPS

    while env.viewer.isopen and not quit_requested:
        t0 = time.monotonic()
        gamepad.poll()
        gp = gamepad.read_buttons()

        right = 'RIGHT' in held or gp.right
        left = 'LEFT' in held or gp.left
        up = 'UP' in held or gp.up
        down = 'DOWN' in held or gp.down
        a_btn = 'A' in held or gp.a_btn
        b_btn = 'B' in held or gp.b_btn
        start = 'START' in held or gp.start
        sel = 'SELECT' in held or gp.select

        if right and left:
            right = left = False
        if up and down:
            up = down = False

        byte = 0
        for btn_name, pressed in [('A', a_btn), ('B', b_btn), ('SELECT', sel),
                                   ('START', start), ('UP', up), ('DOWN', down),
                                   ('LEFT', left), ('RIGHT', right)]:
            if pressed:
                byte |= (1 << BUTTON_BITS[btn_name])

        obs, reward, terminated, truncated, info = env.step(
            nes_byte_to_retro_action(byte)
        )
        if terminated or truncated:
            obs, _info = env.reset()
        print(f'\robs.shape={obs.shape}', end='', flush=True)

        elapsed = time.monotonic() - t0
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)

    env.close()


if __name__ == "__main__":
    main()
