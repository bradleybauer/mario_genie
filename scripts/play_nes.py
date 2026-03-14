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
    Enter              - Start
    Right Shift        - Select
    Escape / Q         - Quit
    Gamepad also supported via evdev.
"""
import sys
import os
import glob
import select

import numpy as np
import pygame

try:
    import evdev
except ImportError:
    evdev = None

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


class GamepadController:
    """Reads keyboard + evdev gamepad and returns an NES controller byte."""
    def __init__(self):
        self._evdev_device = None
        self._evdev_axes = {"ABS_X": 128, "ABS_Y": 128}
        self._evdev_buttons = set()
        self._evdev_axis_ranges = {}
        self._init_evdev()

    def _init_evdev(self):
        if evdev is None:
            return
        try:
            for path in evdev.list_devices():
                try:
                    dev = evdev.InputDevice(path)
                except PermissionError:
                    continue
                caps = dev.capabilities(verbose=False)
                if evdev.ecodes.EV_ABS in caps or evdev.ecodes.EV_KEY in caps:
                    self._evdev_device = dev
                    for abs_code, abs_info in caps.get(evdev.ecodes.EV_ABS, []):
                        name = evdev.ecodes.ABS.get(abs_code, f"ABS_{abs_code}")
                        if isinstance(name, list):
                            name = name[0]
                        mid = (abs_info.min + abs_info.max) // 2
                        self._evdev_axes[name] = mid
                        self._evdev_axis_ranges[name] = (abs_info.min, abs_info.max)
                    print(f"Gamepad: {dev.name} ({dev.path})")
                    return
        except Exception as exc:
            print(f"Gamepad probe failed: {exc}")

    def _poll_evdev(self):
        dev = self._evdev_device
        if dev is None:
            return
        try:
            while select.select([dev.fd], [], [], 0)[0]:
                for event in dev.read():
                    if event.type == evdev.ecodes.EV_ABS:
                        name = evdev.ecodes.ABS.get(event.code, f"ABS_{event.code}")
                        if isinstance(name, list):
                            name = name[0]
                        # print(f"  [axis] {name}={event.value}")
                        self._evdev_axes[name] = event.value
                    elif event.type == evdev.ecodes.EV_KEY:
                        btn_name = evdev.ecodes.BTN.get(event.code) or evdev.ecodes.KEY.get(event.code) or '???'
                        if isinstance(btn_name, list):
                            btn_name = btn_name[0]
                        state = 'PRESSED' if event.value >= 1 else 'released'
                        # print(f"  [btn] code={event.code}  name={btn_name}  {state}")
                        if event.value >= 1:
                            self._evdev_buttons.add(event.code)
                        else:
                            self._evdev_buttons.discard(event.code)
        except (OSError, IOError):
            print("Gamepad disconnected!")
            self._evdev_device = None

    def get_action(self, keys) -> int:
        """Return an NES controller byte (0-255) from keyboard + gamepad state."""
        self._poll_evdev()

        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        up = keys[pygame.K_UP] or keys[pygame.K_w]
        down = keys[pygame.K_DOWN] or keys[pygame.K_s]
        a_btn = keys[pygame.K_x] or keys[pygame.K_o]
        b_btn = keys[pygame.K_z] or keys[pygame.K_p]
        start = keys[pygame.K_RETURN]
        sel = keys[pygame.K_RSHIFT]

        if self._evdev_device is not None:
            def _norm(axis):
                lo, hi = self._evdev_axis_ranges.get(axis, (0, 255))
                mid = (lo + hi) / 2.0
                half = max((hi - lo) / 2.0, 1)
                return (self._evdev_axes.get(axis, mid) - mid) / half

            x, y = _norm("ABS_X"), _norm("ABS_Y")
            right = right or x > 0.5
            left = left or x < -0.5
            up = up or y < -0.5
            down = down or y > 0.5

            hx, hy = _norm("ABS_HAT0X"), _norm("ABS_HAT0Y")
            right = right or hx > 0.5
            left = left or hx < -0.5
            up = up or hy < -0.5
            down = down or hy > 0.5

            btns = self._evdev_buttons
            a_btn = a_btn or any(b in btns for b in [
                getattr(evdev.ecodes, 'BTN_SOUTH', 304),
                getattr(evdev.ecodes, 'BTN_THUMB', 289),
            ])
            b_btn = b_btn or any(b in btns for b in [
                getattr(evdev.ecodes, 'BTN_WEST', 308),
                getattr(evdev.ecodes, 'BTN_EAST', 305),
                getattr(evdev.ecodes, 'BTN_TRIGGER', 288),
            ])
            start = start or any(b in btns for b in [
                getattr(evdev.ecodes, 'BTN_START', 315),
                297,  # BTN_BASE4 on generic USB gamepads
            ])
            sel = sel or any(b in btns for b in [
                getattr(evdev.ecodes, 'BTN_SELECT', 314),
                296,  # BTN_BASE3 on generic USB gamepads
            ])

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
        if byte:
            active = [n for n, p in [('A', a_btn), ('B', b_btn), ('SEL', sel),
                      ('START', start), ('UP', up), ('DOWN', down),
                      ('LEFT', left), ('RIGHT', right)] if p]
            # print(f"  -> action={byte} ({'+'.join(active)})")
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

        keys = pygame.key.get_pressed()
        action = controller.get_action(keys)

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
