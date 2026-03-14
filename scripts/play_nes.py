"""Play NES ROMs using nes_py or stable-retro.

Usage:
    python scripts/play_nes.py              # interactive menu
    python scripts/play_nes.py pacman       # partial name match
    python scripts/play_nes.py 3            # pick by number
    python scripts/play_nes.py --ram mario  # with RAM visualizer

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
import sys
import os
import glob
import hashlib
import json
import shutil

import numpy as np
import pygame

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mario_world_model.gamepad import GamepadState

try:
    import stable_retro as retro
except ImportError:
    try:
        import retro
    except ImportError:
        retro = None

# --- Configuration ---
ROM_DIR = os.path.join(os.path.dirname(__file__), '..', 'nes')
SCALE_FACTOR = 3
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

# ---------------------------------------------------------------------------
# RAM visualization
# ---------------------------------------------------------------------------

RAM_SIZE = 2048
RAM_COLS = 64
RAM_ROWS = 32   # 64 * 32 = 2048
RAM_CELL = 6    # pixels per cell

REGION_ZERO_PAGE = (0x0000, 0x00FF)
REGION_STACK     = (0x0100, 0x01FF)
REGION_OAM       = (0x0200, 0x02FF)
REGION_GAME_DATA = (0x0300, 0x07FF)

REGION_TINTS = {
    "zero_page": np.array([80, 110, 220], dtype=np.int16),
    "stack":     np.array([90, 90, 90],   dtype=np.int16),
    "oam":       np.array([210, 80, 210],  dtype=np.int16),
    "game_data": np.array([70, 200, 90],   dtype=np.int16),
}

REGION_ROWS = {
    "zero_page": (0, 3),
    "stack":     (4, 7),
    "oam":       (8, 11),
    "game_data": (12, 31),
}

REGION_SEP_COLOURS = {
    "zero_page": (100, 130, 255),
    "stack":     (130, 130, 130),
    "oam":       (240, 110, 240),
    "game_data": (100, 230, 120),
}

FADE_FRAMES = 15
ACTIVITY_WINDOW = 30
NES_W, NES_H = 256, 240
OAM_MAP_SCALE = 1


def _build_value_colormap():
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


def _build_activity_colormap():
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


def _build_region_tint_array():
    tints = np.zeros((RAM_SIZE, 3), dtype=np.int16)
    for name, (s, e) in [("zero_page", REGION_ZERO_PAGE), ("stack", REGION_STACK),
                          ("oam", REGION_OAM), ("game_data", REGION_GAME_DATA)]:
        tints[s:e + 1] = REGION_TINTS[name]
    return tints


VALUE_CMAP = _build_value_colormap()
ACTIVITY_CMAP = _build_activity_colormap()
REGION_TINT_ARRAY = _build_region_tint_array()


class RAMGridRenderer:
    def __init__(self):
        self._fade = np.zeros(RAM_SIZE, dtype=np.float32)
        self._activity_ring = np.zeros((ACTIVITY_WINDOW, RAM_SIZE), dtype=np.uint8)
        self._ring_idx = 0
        self._prev_ram = None

    def update(self, ram):
        if self._prev_ram is not None:
            changed = ram != self._prev_ram
            self._fade[changed] = 1.0
            self._fade[~changed] = np.maximum(self._fade[~changed] - 1.0 / FADE_FRAMES, 0.0)
            self._activity_ring[self._ring_idx] = changed.astype(np.uint8)
        self._ring_idx = (self._ring_idx + 1) % ACTIVITY_WINDOW
        self._prev_ram = ram.copy()

    def activity_map(self):
        return self._activity_ring.mean(axis=0).astype(np.float32)

    def render(self, ram, *, activity_mode=False):
        cell = RAM_CELL
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


def _render_oam_minimap(ram):
    w, h = NES_W * OAM_MAP_SCALE, NES_H * OAM_MAP_SCALE
    pixels = np.zeros((w, h, 3), dtype=np.uint8)
    pixels[:, :] = (15, 15, 25)
    for sy in range(0, NES_H, 16):
        pixels[:, sy * OAM_MAP_SCALE] = (25, 25, 35)
    for i in range(64):
        base = 0x200 + i * 4
        sprite_y = ram[base]
        attr = ram[base + 2]
        sprite_x = ram[base + 3]
        if sprite_y >= 0xEF or sprite_y == 0:
            continue
        sx = sprite_x * OAM_MAP_SCALE
        sy = sprite_y * OAM_MAP_SCALE
        sz = max(6 * OAM_MAP_SCALE, 4)
        palette = attr & 0x03
        pal_colours = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]
        colour = pal_colours[palette]
        x0 = max(0, min(sx, w - sz))
        y0 = max(0, min(sy, h - sz))
        pixels[x0:min(x0 + sz, w), y0:min(y0 + sz, h)] = colour
    pixels[0, :] = pixels[-1, :] = (60, 60, 60)
    pixels[:, 0] = pixels[:, -1] = (60, 60, 60)
    return pygame.surfarray.make_surface(pixels)


def _draw_ram_region_labels(surface, font, x_offset, y_offset):
    cell = RAM_CELL
    for name, colour, (r0, r1) in [
        ("Zero Page",  REGION_SEP_COLOURS["zero_page"], REGION_ROWS["zero_page"]),
        ("Stack",      REGION_SEP_COLOURS["stack"],     REGION_ROWS["stack"]),
        ("OAM",        REGION_SEP_COLOURS["oam"],       REGION_ROWS["oam"]),
        ("Game Data",  REGION_SEP_COLOURS["game_data"],  REGION_ROWS["game_data"]),
    ]:
        mid_row = (r0 + r1) / 2
        ly = y_offset + int(mid_row * cell) - 6
        label = font.render(name, True, colour)
        surface.blit(label, (x_offset - label.get_width() - 6, ly))

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
    from nes_py._rom import ROM
    roms = []
    for path in sorted(glob.glob(os.path.join(ROM_DIR, '*.nes'))):
        basename = os.path.basename(path)
        name = os.path.splitext(basename)[0]
        try:
            r = ROM(path)
            if r.mapper in SUPPORTED_MAPPERS:
                backend = 'nes_py'
            elif retro is not None:
                backend = 'retro'
            else:
                backend = None
            roms.append((name, path, r.mapper, backend))
        except Exception:
            if retro is not None:
                roms.append((name, path, -1, 'retro'))
            else:
                roms.append((name, path, -1, None))
    return roms


def choose_rom(roms):
    """Pick a ROM from CLI arg or interactive menu."""
    if len(sys.argv) > 1:
        query = sys.argv[1]
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
    # Check for --ram flag before ROM selection
    show_ram = '--ram' in sys.argv
    if show_ram:
        sys.argv.remove('--ram')

    roms = discover_roms()
    if not roms:
        print(f"No .nes files found in {ROM_DIR}")
        return

    name, path, mapper, backend = choose_rom(roms)
    if backend is None:
        print(f"\n'{name}' uses mapper {mapper} which nes_py doesn't support.")
        print("Install stable-retro (`pip install stable-retro`) for broader mapper support.")
        return

    print(f"Loading {name} (backend: {backend}) ...")
    print("Controls: Arrows/WASD=D-Pad  X/O=A  Z/P=B  Enter/Space=Start  RShift=Select  Esc=Quit")

    if backend == 'retro':
        if show_ram:
            _run_retro_pygame(name, path)
        else:
            _run_retro(name, path)
    else:
        _run_nes_py(name, path, show_ram=show_ram)


def _run_nes_py(name, path, show_ram=False):
    """Game loop using nes_py + pygame."""
    from nes_py.nes_env import NESEnv
    env = NESEnv(path)
    obs = env.reset()
    h, w, _ = obs.shape
    game_w = w * SCALE_FACTOR
    game_h = h * SCALE_FACTOR

    if show_ram:
        win_w, win_h, layout = _ram_layout(game_w, game_h)
    else:
        win_w, win_h = game_w, game_h

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"{name} (nes_py)")
    clock = pygame.time.Clock()
    controller = GamepadController()
    font = pygame.font.SysFont('monospace', 13) if show_ram else None
    ram_renderer = RAMGridRenderer() if show_ram else None

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

        screen.fill((10, 10, 15))
        surface = pygame.surfarray.make_surface(np.swapaxes(obs, 0, 1))
        screen.blit(pygame.transform.scale(surface, (game_w, game_h)), (0, 0))

        if show_ram:
            ram = np.array(env.ram, dtype=np.uint8)[:RAM_SIZE]
            _draw_ram_panel(screen, font, ram, ram_renderer, layout, game_w, game_h)

        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()


def _run_retro_pygame(name, path):
    """Game loop using stable-retro rendered via pygame (for RAM viz)."""
    game_id = ensure_retro_game(path)
    env = retro.make(game_id, state=retro.State.NONE, render_mode=None,
                     use_restricted_actions=retro.Actions.ALL)
    obs, _info = env.reset()
    h, w, _ = obs.shape
    game_w = w * SCALE_FACTOR
    game_h = h * SCALE_FACTOR

    win_w, win_h, layout = _ram_layout(game_w, game_h)

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"{name} (retro + RAM)")
    clock = pygame.time.Clock()
    controller = GamepadController()
    font = pygame.font.SysFont('monospace', 13)
    ram_renderer = RAMGridRenderer()

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

        screen.fill((10, 10, 15))
        surface = pygame.surfarray.make_surface(np.swapaxes(obs, 0, 1))
        screen.blit(pygame.transform.scale(surface, (game_w, game_h)), (0, 0))

        full_ram = env.get_ram()
        ram = np.array(full_ram, dtype=np.uint8)[:RAM_SIZE]
        _draw_ram_panel(screen, font, ram, ram_renderer, layout, game_w, game_h)

        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()


def _ram_layout(game_w, game_h):
    """Compute layout metrics for the RAM panel.  Returns (win_w, win_h, layout_dict)."""
    ram_grid_w = RAM_COLS * RAM_CELL
    ram_grid_h = RAM_ROWS * RAM_CELL
    label_margin = 80
    grid_gap = 20
    top_margin = 20

    two_grids_h = ram_grid_h * 2 + grid_gap + 16 * 2
    panel_w = label_margin + ram_grid_w + 16
    panel_h = top_margin + two_grids_h + 20

    oam_section_h = 16 + NES_H * OAM_MAP_SCALE + 8
    left_col_h = game_h + oam_section_h

    win_w = game_w + panel_w
    win_h = max(left_col_h, panel_h)

    layout = dict(
        label_margin=label_margin,
        grid_gap=grid_gap,
        top_margin=top_margin,
        ram_grid_h=ram_grid_h,
    )
    return win_w, win_h, layout


def _draw_ram_panel(screen, font, ram, ram_renderer, layout, game_w, game_h):
    """Draw value grid, activity grid, region labels, and OAM minimap."""
    ram_renderer.update(ram)
    gx = game_w + layout['label_margin']
    cy = layout['top_margin']

    # Value grid
    val_title = font.render('RAM VALUES', True, (200, 200, 200))
    screen.blit(val_title, (gx, cy))
    cy += 16
    screen.blit(ram_renderer.render(ram, activity_mode=False), (gx, cy))
    _draw_ram_region_labels(screen, font, gx, cy)
    cy += layout['ram_grid_h'] + layout['grid_gap']

    # Activity grid
    act_title = font.render('RAM ACTIVITY', True, (200, 200, 200))
    screen.blit(act_title, (gx, cy))
    cy += 16
    screen.blit(ram_renderer.render(ram, activity_mode=True), (gx, cy))
    _draw_ram_region_labels(screen, font, gx, cy)

    # OAM minimap below game frame
    oam_y = game_h
    oam_label = font.render('-- OAM SPRITES --', True, (200, 200, 100))
    screen.blit(oam_label, (4, oam_y))
    oam_y += 16
    screen.blit(_render_oam_minimap(ram), (4, oam_y))


def _run_retro(name, path):
    """Game loop using stable-retro's native viewer (pyglet) at SCALE_FACTOR."""
    import pyglet.window.key as pkey

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

    # First step triggers viewer creation
    obs, *_ = env.step(np.zeros(9, dtype=np.int8))

    # Resize the viewer window (GL stretches the texture to fill)
    env.viewer.window.set_size(w * SCALE_FACTOR, h * SCALE_FACTOR)
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

    import time
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

        elapsed = time.monotonic() - t0
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)

    env.close()


if __name__ == "__main__":
    main()
