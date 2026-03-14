"""Play NES ROMs using the nes_py (LaiNES) emulator.

Usage:
    python scripts/playNES_nepy.py              # interactive menu
    python scripts/playNES_nepy.py pacman       # partial name match
    python scripts/playNES_nepy.py 3            # pick by number

nes_py supports NES mappers 0, 1, 2, 3 only.
Incompatible ROMs (mapper 4, 9, 66, etc.) are listed but marked.

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

import numpy as np
import pygame

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mario_world_model.gamepad import GamepadState

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


def discover_roms():
    """Scan ROM_DIR for .nes files and return [(display_name, path, compatible, mapper)]."""
    from nes_py._rom import ROM
    roms = []
    for path in sorted(glob.glob(os.path.join(ROM_DIR, '*.nes'))):
        basename = os.path.basename(path)
        name = os.path.splitext(basename)[0]
        try:
            r = ROM(path)
            compat = r.mapper in SUPPORTED_MAPPERS
            roms.append((name, path, compat, r.mapper))
        except Exception:
            roms.append((name, path, False, -1))
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
    for i, (name, _, compat, mapper) in enumerate(roms, 1):
        tag = "" if compat else f"  [mapper {mapper} - unsupported]"
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
    roms = discover_roms()
    if not roms:
        print(f"No .nes files found in {ROM_DIR}")
        return

    name, path, compat, mapper = choose_rom(roms)
    if not compat:
        print(f"\n'{name}' uses mapper {mapper} which nes_py doesn't support.")
        print("Only mappers 0, 1, 2, 3 are supported. Try the stable_retro script instead.")
        return

    print(f"Loading {name} ...")
    print("Controls: Arrows/WASD=D-Pad  X/O=A  Z/P=B  Enter/Space=Start  RShift=Select  Esc=Quit")
    from nes_py.nes_env import NESEnv
    env = NESEnv(path)
    obs = env.reset()
    h, w, _ = obs.shape

    pygame.init()
    screen = pygame.display.set_mode((w * SCALE_FACTOR, h * SCALE_FACTOR))
    pygame.display.set_caption(f"{name} (nes_py)")
    clock = pygame.time.Clock()
    controller = GamepadController()

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

        surface = pygame.surfarray.make_surface(np.swapaxes(obs, 0, 1))
        screen.blit(
            pygame.transform.scale(surface, (w * SCALE_FACTOR, h * SCALE_FACTOR)),
            (0, 0),
        )
        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
