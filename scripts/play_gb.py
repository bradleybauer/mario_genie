"""Play Game Boy ROMs with PyBoy.

Usage:
    python scripts/play_gb.py
    python scripts/play_gb.py --rom mario
    python scripts/play_gb.py --mario-land
    python scripts/play_gb.py --path /path/to/Super Mario Land.gb
    python scripts/play_gb.py --scale 4

The script defaults to original Game Boy (DMG) mode so DMG titles such as
Super Mario Land run with the expected hardware profile. Use --cgb to allow
Game Boy Color mode for color-capable ROMs.

Controls:
    Arrow keys / WASD  - D-Pad
    X / O              - A (jump / confirm)
    Z / P              - B (run / back)
    Enter / Space      - Start
    Right Shift        - Select
    Escape / Q         - Quit
    Gamepad also supported via evdev.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import zipfile

# Ensure SDL uses PulseAudio for sound (needed in WSL2 / WSLg)
if "SDL_AUDIODRIVER" not in os.environ:
    os.environ["SDL_AUDIODRIVER"] = "pulseaudio"

import numpy as np
import pygame

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from pyboy import PyBoy
except ImportError:
    PyBoy = None

from mario_world_model.gamepad import GamepadState


DEFAULT_SCALE = 4
FPS = 60
FPS_BAR_H = 18
SOUND_SAMPLE_RATE = 48000
DMG_PALETTE = (0xE0F8D0, 0x88C070, 0x346856, 0x081820)
ROM_SEARCH_DIRS = (
    "gb1",
    "gb",
    "gameboy",
    "roms/gb",
    "roms/gameboy",
)
ROM_PATTERNS = ("*.gb", "*.gbc", "*.zip")
ROM_EXTENSIONS = {".gb", ".gbc"}

KEY_MAP = {
    pygame.K_RIGHT: "right", pygame.K_d: "right",
    pygame.K_LEFT: "left",   pygame.K_a: "left",
    pygame.K_UP: "up",       pygame.K_w: "up",
    pygame.K_DOWN: "down",   pygame.K_s: "down",
    pygame.K_x: "a",         pygame.K_o: "a",
    pygame.K_z: "b",         pygame.K_p: "b",
    pygame.K_RETURN: "start", pygame.K_KP_ENTER: "start", pygame.K_SPACE: "start",
    pygame.K_RSHIFT: "select",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play Game Boy ROMs with PyBoy.")
    parser.add_argument("--rom", help="ROM number or partial name match.")
    parser.add_argument("--path", help="Direct path to a .gb or .gbc ROM.")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE, help="Window scale factor.")
    parser.add_argument("--cgb", action="store_true", help="Allow Game Boy Color mode instead of forcing DMG mode.")
    parser.add_argument("--mario-land", action="store_true", help="Filter ROM selection to Super Mario Land titles.")
    parser.add_argument("--list", action="store_true", help="List discovered ROMs and exit.")
    parser.add_argument("--save", action="store_true", help="Persist cartridge save RAM on exit.")
    parser.add_argument("--sound", action="store_true", help="Enable sound (off by default due to WSL2 choppiness).")
    return parser


def _normalize_name(name: str) -> str:
    return " ".join(name.lower().replace("-", " ").replace("_", " ").split())


def discover_roms() -> list[tuple[str, str]]:
    roms: list[tuple[str, str]] = []
    seen: set[str] = set()

    roots: list[str] = []
    env_root = os.environ.get("GB_ROM_DIR")
    if env_root:
        roots.append(env_root)
    roots.extend(os.path.join(PROJECT_ROOT, rel_path) for rel_path in ROM_SEARCH_DIRS)

    for root in roots:
        if not os.path.isdir(root):
            continue
        for pattern in ROM_PATTERNS:
            for path in sorted(glob.glob(os.path.join(root, pattern))):
                abs_path = os.path.abspath(path)
                if abs_path in seen:
                    continue
                seen.add(abs_path)
                name = display_name_for_path(abs_path)
                roms.append((name, abs_path))

    roms.sort(key=lambda item: item[0].lower())
    return roms


def filter_mario_land_roms(roms: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [rom for rom in roms if "super mario land" in _normalize_name(rom[0])]


def display_name_for_path(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    if path.lower().endswith(".zip"):
        try:
            member_name = zip_member_name(path)
        except ValueError:
            return name
        return os.path.splitext(os.path.basename(member_name))[0]
    return name


def zip_member_name(path: str) -> str:
    with zipfile.ZipFile(path) as archive:
        members = [
            name for name in archive.namelist()
            if os.path.splitext(name)[1].lower() in ROM_EXTENSIONS and not name.endswith("/")
        ]
    if not members:
        raise ValueError(f"Archive contains no .gb or .gbc ROM: {path}")
    if len(members) > 1:
        raise ValueError(f"Archive contains multiple ROM files, use --path with an extracted ROM instead: {path}")
    return members[0]


def materialize_rom(path: str) -> tuple[str, str, tempfile.TemporaryDirectory[str] | None]:
    abs_path = os.path.abspath(path)
    if abs_path.lower().endswith(".zip"):
        member_name = zip_member_name(abs_path)
        temp_dir = tempfile.TemporaryDirectory(prefix="play_gb_")
        with zipfile.ZipFile(abs_path) as archive:
            data = archive.read(member_name)
        rom_name = os.path.basename(member_name)
        extracted_path = os.path.join(temp_dir.name, rom_name)
        with open(extracted_path, "wb") as handle:
            handle.write(data)
        return os.path.splitext(rom_name)[0], extracted_path, temp_dir
    return display_name_for_path(abs_path), abs_path, None


def choose_rom(roms: list[tuple[str, str]], query: str | None = None) -> tuple[str, str]:
    if query is not None:
        if query.isdigit():
            idx = int(query) - 1
            if 0 <= idx < len(roms):
                return roms[idx]

        matches = [rom for rom in roms if query.lower() in rom[0].lower()]
        if len(matches) == 1:
            return matches[0]
        if matches:
            print(f"Ambiguous match for '{query}':")
            for match in matches:
                print(f"  - {match[0]}")
        else:
            print(f"No ROM matches '{query}'.")
        print()

    print("Available Game Boy ROMs:")
    for index, (name, _path) in enumerate(roms, 1):
        print(f"  {index:2d}. {name}")

    while True:
        choice = input(f"Pick a ROM (1-{len(roms)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(roms):
            return roms[int(choice) - 1]
        print("Invalid choice, try again.")


class GBController:
    def __init__(self) -> None:
        self._gamepad = GamepadState(extended_button_codes=True)
        self._held: set[str] = set()

    def process_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            button = KEY_MAP.get(event.key)
            if button:
                self._held.add(button)
        elif event.type == pygame.KEYUP:
            button = KEY_MAP.get(event.key)
            if button:
                self._held.discard(button)

    def get_pressed(self) -> set[str]:
        self._gamepad.poll()
        gp = self._gamepad.read_buttons()

        right = "right" in self._held or gp.right
        left = "left" in self._held or gp.left
        up = "up" in self._held or gp.up
        down = "down" in self._held or gp.down
        a_btn = "a" in self._held or gp.a_btn
        b_btn = "b" in self._held or gp.b_btn
        start = "start" in self._held or gp.start
        select = "select" in self._held or gp.select

        if right and left:
            right = left = False
        if up and down:
            up = down = False

        pressed: set[str] = set()
        if right:
            pressed.add("right")
        if left:
            pressed.add("left")
        if up:
            pressed.add("up")
        if down:
            pressed.add("down")
        if a_btn:
            pressed.add("a")
        if b_btn:
            pressed.add("b")
        if start:
            pressed.add("start")
        if select:
            pressed.add("select")
        return pressed


def sync_buttons(pyboy: PyBoy, current: set[str], previous: set[str]) -> None:
    for button in sorted(previous - current):
        pyboy.button_release(button)
    for button in sorted(current - previous):
        pyboy.button_press(button)


def frame_to_surface(frame: np.ndarray) -> pygame.Surface:
    rgb = np.ascontiguousarray(frame[..., :3])
    return pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))


def run_game(name: str, path: str, *, scale: int, force_dmg: bool, save: bool, sound: bool = False) -> None:
    _source_name, rom_path, temp_dir = materialize_rom(path)
    pyboy = None
    try:
        sound_enabled = sound
        kwargs = dict(
            window="SDL2",
            scale=1,
            cgb=False if force_dmg else None,
            title_status=False,
            log_level="ERROR",
            sound_emulated=sound_enabled,
            sound_volume=100 if sound_enabled else 0,
            sound_sample_rate=SOUND_SAMPLE_RATE,
            no_input=True,
        )
        if force_dmg:
            kwargs["color_palette"] = DMG_PALETTE
        pyboy = PyBoy(rom_path, **kwargs)
        pyboy.set_emulation_speed(0)
        pyboy.tick(1, True)

        frame = np.asarray(pyboy.screen.ndarray)
        height, width, _channels = frame.shape
        game_w = width * scale
        game_h = height * scale

        pygame.init()
        screen = pygame.display.set_mode((game_w, game_h + FPS_BAR_H), pygame.DOUBLEBUF)
        mode_name = "DMG" if force_dmg else "auto/CGB"
        pygame.display.set_caption(f"{name} ({mode_name})")
        clock = pygame.time.Clock()
        controller = GBController()
        fps_font = pygame.font.SysFont("monospace", 14, bold=True)
        fps_surface = fps_font.render("-- FPS", True, (0, 255, 0))
        fps_frame = 0
        previous_pressed: set[str] = set()

        print(f"Loading {name} ({mode_name}) ...")
        print("Controls: Arrows/WASD=D-Pad  X/O=A  Z/P=B  Enter/Space=Start  RShift=Select  Esc=Quit")
        print(f"Resolution: {game_w}x{game_h} (native {width}x{height}, scale {scale}x)")
        print(f"Obs shape: {frame.shape}")

        try:
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    controller.process_event(event)

                current_pressed = controller.get_pressed()
                sync_buttons(pyboy, current_pressed, previous_pressed)
                previous_pressed = current_pressed

                if not pyboy.tick(1, True, sound_enabled):
                    running = False
                    continue

                frame = np.asarray(pyboy.screen.ndarray)
                surface = frame_to_surface(frame)

                screen.fill((10, 10, 15))
                screen.blit(pygame.transform.scale(surface, (game_w, game_h)), (0, 0))

                fps_frame += 1
                if fps_frame >= 30:
                    fps_surface = fps_font.render(f"{clock.get_fps():.0f} FPS", True, (0, 255, 0))
                    fps_frame = 0
                screen.blit(fps_surface, (4, game_h + 2))

                pygame.display.flip()
                clock.tick(FPS)
        finally:
            for button in sorted(previous_pressed):
                pyboy.button_release(button)
            pygame.quit()
    finally:
        if pyboy is not None:
            pyboy.stop(save)
        if temp_dir is not None:
            temp_dir.cleanup()


def resolve_path(path_arg: str) -> tuple[str, str]:
    path = os.path.abspath(path_arg)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext not in ROM_EXTENSIONS | {".zip"}:
        raise ValueError(f"Unsupported ROM extension: {ext}")

    return display_name_for_path(path), path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if PyBoy is None:
        print("PyBoy is not installed. Install it with: pip install pyboy")
        raise SystemExit(1)

    if args.scale < 1:
        print("--scale must be at least 1")
        raise SystemExit(2)

    if args.path:
        try:
            name, path = resolve_path(args.path)
        except (FileNotFoundError, ValueError) as exc:
            print(exc)
            raise SystemExit(2) from exc
        run_game(name, path, scale=args.scale, force_dmg=not args.cgb, save=args.save, sound=args.sound)
        return

    all_roms = discover_roms()
    roms = filter_mario_land_roms(all_roms) if args.mario_land else all_roms

    if args.list:
        if not roms:
            if args.mario_land and all_roms:
                print("No Super Mario Land ROMs were found in the configured search paths.")
            else:
                print("No Game Boy ROMs found.")
        else:
            for index, (name, path) in enumerate(roms, 1):
                print(f"{index:2d}. {name} -> {path}")
        return

    if not roms:
        if args.mario_land and all_roms:
            print("No Super Mario Land ROMs were found in the configured search paths.")
            raise SystemExit(1)
        print("No Game Boy ROMs found.")
        print("Searched directories:")
        for rel_path in ROM_SEARCH_DIRS:
            print(f"  - {os.path.join(PROJECT_ROOT, rel_path)}")
        env_root = os.environ.get("GB_ROM_DIR")
        if env_root:
            print(f"  - {env_root} (from GB_ROM_DIR)")
        print("Use --path /path/to/game.gb to launch a ROM directly.")
        raise SystemExit(1)

    query = args.rom
    if query is None and args.mario_land:
        query = "super mario land"

    name, path = choose_rom(roms, query=query)
    run_game(name, path, scale=args.scale, force_dmg=not args.cgb, save=args.save, sound=args.sound)


if __name__ == "__main__":
    main()