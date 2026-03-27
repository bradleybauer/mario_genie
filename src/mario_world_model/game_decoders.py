"""Game-specific NES RAM decoders for the RAM visualizer.

Each decoder takes the full emulator RAM array and returns a list of
(header, [line, ...]) sections for display.  A registry maps ROM name
substrings to decoders so the visualizer can auto-detect the game.

stable-retro (FCEUmm core) returns 10 240 bytes:
  - bytes 0x0000-0x07FF  →  NES internal RAM  (2 KB)
  - bytes 0x0800-0x27FF  →  cartridge WRAM    (8 KB, NES $6000-$7FFF)

nes_py only exposes the 2 KB internal RAM.  Decoders must handle both
sizes gracefully.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pygame

# Type alias for a decoder function
Decoder = Callable[[np.ndarray], list[tuple[str, list[str]]]]

# Memory layout constants (stable-retro FCEUmm core)
INTERNAL_RAM_SIZE = 0x0800   # 2 KB
WRAM_OFFSET       = 0x0800   # array index where WRAM starts
WRAM_SIZE         = 0x2000   # 8 KB


def _wram(ram: np.ndarray, nes_addr: int) -> int:
    """Read a byte at a WRAM NES address ($6000-$7FFF). Returns 0 if unavailable."""
    idx = WRAM_OFFSET + (nes_addr - 0x6000)
    if idx < len(ram):
        return int(ram[idx])
    return 0

# ---------------------------------------------------------------------------
# SMB1 decoder
# ---------------------------------------------------------------------------

_SMB1_PLAYER_STATES = {
    0x00: "Leftmost", 0x01: "Vine", 0x02: "Pipe (rev-L)",
    0x03: "Pipe (down)", 0x04: "Auto-walk", 0x05: "Auto-walk",
    0x06: "Dead", 0x07: "Entering", 0x08: "Normal",
    0x09: "Frozen", 0x0B: "Dying", 0x0C: "Palette cyc",
}

_SMB1_PLAYER_STATUS = {0: "Small", 1: "Big", 2: "Fire"}

_SMB1_ENEMY_TYPES = {
    0x00: "-", 0x06: "Goomba", 0x07: "Blooper", 0x08: "BulletBill",
    0x09: "GreenKoopa", 0x0A: "RedKoopa", 0x0E: "RedKoopa(fly)",
    0x11: "Lakitu", 0x12: "Spiny", 0x14: "FlyFish", 0x15: "Podoboo",
    0x1B: "FireBar", 0x1F: "HammerBro", 0x24: "Firework",
    0x2D: "Bowser", 0x2E: "Fireball(B)", 0x31: "Flagpole", 0x35: "Toad",
}


def decode_smb1(ram: np.ndarray) -> list[tuple[str, list[str]]]:
    """Decode Super Mario Bros. 1 RAM."""
    # int() casts avoid uint8 overflow when the array comes from mdat parsing.
    r = lambda addr: int(ram[addr])
    ps = r(0x000E)
    player_state = _SMB1_PLAYER_STATES.get(ps, f"0x{ps:02X}")
    player_status = _SMB1_PLAYER_STATUS.get(r(0x0756), f"0x{r(0x0756):02X}")
    x_pos = r(0x006D) * 256 + r(0x0086)
    y_pos = r(0x00CE)
    y_viewport = r(0x00B5)
    x_speed = np.int8(ram[0x0057])
    y_speed = np.int8(ram[0x001D])
    moving_dir = "R" if r(0x0045) == 1 else ("L" if r(0x0045) == 2 else "?")

    world = r(0x075F) + 1
    stage = r(0x075C) + 1
    lives = r(0x075A)
    time_val = r(0x07F8) * 100 + r(0x07F9) * 10 + r(0x07FA)
    coins = r(0x07ED) * 10 + r(0x07EE)
    score = 0
    for i in range(6):
        score = score * 10 + r(0x07DE + i)

    gameplay_mode = r(0x0770)
    gameplay = {0: "Demo", 1: "Playing", 2: "World End"}.get(
        gameplay_mode, f"0x{gameplay_mode:02X}")

    enemies = []
    for i in range(5):
        etype = r(0x0016 + i)
        if etype != 0:
            ex, ey = r(0x0087 + i), r(0x00CF + i)
            enemies.append((_SMB1_ENEMY_TYPES.get(etype, f"0x{etype:02X}"), ex, ey))

    sections: list[tuple[str, list[str]]] = [
        ("-- PLAYER --", [
            f"State:  {player_state}",
            f"Status: {player_status}  Dir: {moving_dir}",
            f"Pos:    ({x_pos}, {y_pos})  YVP: {y_viewport}",
            f"Speed:  X={x_speed:+4d}  Y={y_speed:+4d}",
        ]),
        ("-- GAME --", [
            f"World {world}-{stage}  Mode: {gameplay}",
            f"Lives: {lives}  Time: {time_val}",
            f"Score: {score}  Coins: {coins}",
        ]),
    ]
    if enemies:
        sections.append((f"-- ENEMIES ({len(enemies)}) --",
                         [f"  {n} ({x},{y})" for n, x, y in enemies]))
    else:
        sections.append(("-- ENEMIES --", ["  (none)"]))
    return sections


# ---------------------------------------------------------------------------
# SMB3 decoder
# ---------------------------------------------------------------------------

_SMB3_POWER_STATES = {
    0x00: "Small", 0x01: "Big", 0x02: "Fire", 0x03: "Raccoon",
    0x04: "Frog", 0x05: "Tanooki", 0x06: "Hammer",
}

_SMB3_PLAYER_ACTIONS = {
    0x00: "Standing", 0x01: "Walking", 0x02: "Running",
    0x03: "Skidding", 0x04: "Jumping", 0x05: "Falling",
    0x06: "Climbing", 0x07: "Swimming", 0x08: "Ducking",
    0x09: "Sliding", 0x0A: "Dying", 0x0B: "Pipe (enter)",
    0x0C: "Pipe (exit)",
}


def decode_smb3(ram: np.ndarray) -> list[tuple[str, list[str]]]:
    """Decode Super Mario Bros. 3 RAM."""
    # Power-up state at $00ED
    power_byte = ram[0x00ED]
    power = _SMB3_POWER_STATES.get(power_byte, f"0x{power_byte:02X}")

    # Player position and motion
    # $0075 is a context-dependent coarse/high byte field used for level X and map Y.
    # In-level horizontal position still tracks with $0075/$0090.
    x_pos = ram[0x0075] * 256 + ram[0x0090]
    y_pos = ram[0x0087] * 256 + ram[0x00A2]
    x_speed = np.int8(ram[0x00BD])
    y_speed = np.int8(ram[0x00CF])

    # Facing direction at $00EF (bit 6 set = right, clear = left)
    direction = "R" if (ram[0x00EF] & 0x40) else "L"

    # In-air flag at $00D8 (0 = ground, else airborne)
    airborne = "Air" if ram[0x00D8] != 0 else "Ground"

    # World (no documented stage counter — SMB3 overworld is non-linear)
    world = ram[0x0727] + 1

    # Lives at $0736
    lives = ram[0x0736]

    # Timer: 3 decimal digits at $05EE-$05F0
    time_val = ram[0x05EE] * 100 + ram[0x05EF] * 10 + ram[0x05F0]

    # Score: big-endian 3 bytes at $0715-$0717, multiply by 10
    score = (int(ram[0x0715]) << 16 | int(ram[0x0716]) << 8 | int(ram[0x0717])) * 10

    # Coins are stored in WRAM at $7DA2 when available.
    coins = _wram(ram, 0x7DA2) if len(ram) > WRAM_OFFSET else None

    # Tileset / mode detection
    _OBJSET_NAMES = {
        0x01: "Plains", 0x02: "Dungeon", 0x03: "Hilly", 0x04: "Sky",
        0x05: "Piranha", 0x06: "Water", 0x08: "Pipe", 0x09: "Desert",
        0x0A: "Ship", 0x0B: "Giant", 0x0C: "Ice", 0x0D: "Cloudy",
        0x0E: "Underground",
    }
    objset = ram[0x070A]
    in_level = time_val > 0 and objset > 0
    mode = f"In Level ({_OBJSET_NAMES.get(objset, '?')})" if in_level else "Overworld"

    # P-meter at $03DD: bits 0-5 = arrows, bit 6 = P
    pmeter = ram[0x03DD]
    p_arrows = sum((pmeter >> i) & 1 for i in range(6))
    p_full = "P!" if (pmeter & 0x40) else ""
    p_str = f"{p_arrows}/6 {p_full}".strip()

    # Flight / duck / swim / boot
    flying = ram[0x057B] != 0
    ducking = ram[0x056F] != 0
    swimming = ram[0x0575] != 0
    boot = ram[0x0577] != 0

    # Star and invincibility timers
    star_timer = ram[0x0553]
    hit_inv = ram[0x0552]

    # Flight timer at $056E (countdown while raccoon/tanooki P-fly)
    flight_timer = ram[0x056E]
    # P-switch timer at $0567 (countdown after stomping P-switch)
    pswitch_timer = ram[0x0567]
    # Stomp counter at $05F4 (consecutive stomps for 1-up chain)
    stomp_count = ram[0x05F4]

    # Progress (WRAM)
    mario_cleared = None
    if len(ram) > WRAM_OFFSET:
        mario_cleared = sum(1 for a in range(0x7D00, 0x7D40) if _wram(ram, a))

    sections: list[tuple[str, list[str]]] = [
        ("-- PLAYER --", [
            f"Form:   {power}  Dir: {direction}  {airborne}",
            f"Pos:    ({x_pos}, {y_pos})",
            f"Speed:  X={x_speed:+4d}  Y={y_speed:+4d}",
            f"P-Meter: {p_str}" + ("  Flying" if flying else ""),
        ]),
        ("-- STATUS --", [
            l for l in [
                f"Duck: {'Y' if ducking else 'N'}  Swim: {'Y' if swimming else 'N'}  Boot: {'Y' if boot else 'N'}",
                f"Star: {star_timer}" if star_timer else None,
                f"Hit inv: {hit_inv}" if hit_inv else None,
                f"Flight: {flight_timer}" if flight_timer else None,
                f"P-Switch: {pswitch_timer}" if pswitch_timer else None,
                f"Stomps: {stomp_count}" if stomp_count else None,
            ] if l is not None
        ]),
        ("-- GAME --", [
            f"World {world}  {mode}",
            f"Lives: {lives}  Time: {time_val}",
            f"Score: {score}" + (f"  Coins: {coins}" if coins is not None else ""),
        ] + ([f"Cleared: {mario_cleared}/64"] if mario_cleared is not None else [])),
    ]

    # --- WRAM sections (stable-retro only) ---
    if len(ram) > WRAM_OFFSET:
        sections.extend(_decode_smb3_wram(ram))

    return sections


_SMB3_ITEMS = {
    0x00: "-", 0x01: "Mush", 0x02: "Flower", 0x03: "Leaf",
    0x04: "Frog", 0x05: "Tanooki", 0x06: "Hammer Suit", 0x07: "Cloud",
    0x08: "P-Wing", 0x09: "Star", 0x0A: "Anchor", 0x0B: "Hammer",
    0x0C: "Whistle", 0x0D: "Music Box",
}

_SMB3_CARDS = {0x00: "-", 0x01: "Mush", 0x02: "Flower", 0x03: "Star"}


def _decode_smb3_wram(ram: np.ndarray) -> list[tuple[str, list[str]]]:
    """Decode SMB3 WRAM ($6000-$7FFF) into display sections."""
    sections: list[tuple[str, list[str]]] = []

    # Overworld map position ($7976-$797A)
    mario_y = _wram(ram, 0x7976)
    mario_x = (_wram(ram, 0x7978) << 8) | _wram(ram, 0x797A)
    sections.append(("-- MAP POS --", [
        f"Map: X={mario_x} Y={mario_y}",
    ]))

    # Mario inventory ($7D80-$7D9B, 28 slots)
    inv = [_wram(ram, 0x7D80 + i) for i in range(28)]
    items = [_SMB3_ITEMS.get(v, f"?{v:02X}") for v in inv if v != 0]
    inv_str = " ".join(items) if items else "(empty)"

    # End-of-level cards ($7D9C-$7D9E)
    cards = [_SMB3_CARDS.get(_wram(ram, 0x7D9C + i), "?") for i in range(3)]

    sections.append(("-- INVENTORY --", [
        f"Items: {inv_str}",
        f"Cards: {' '.join(cards)}",
    ]))

    return sections


# ---------------------------------------------------------------------------
# Decoder registry
# ---------------------------------------------------------------------------

# Maps lowercase substring of ROM name -> decoder function
_DECODER_REGISTRY: list[tuple[str, Decoder]] = [
    ("super mario bros. 3", decode_smb3),
    ("super mario bros", decode_smb1),   # matches SMB1 and SMB1+Duck Hunt
]


def get_decoder(rom_name: str) -> Optional[Decoder]:
    """Return a decoder for the given ROM name, or None if no match."""
    lower = rom_name.lower()
    for pattern, decoder in _DECODER_REGISTRY:
        if pattern in lower:
            return decoder
    return None


# ---------------------------------------------------------------------------
# Pygame drawing helper
# ---------------------------------------------------------------------------

def draw_decoded_sections(
    surface: pygame.Surface,
    font: pygame.font.Font,
    sections: list[tuple[str, list[str]]],
    x: int,
    y: int,
    *,
    col2_x: int = 0,
    col2_after: int = 0,
) -> int:
    """Draw decoded variable sections.  Returns y after last line.

    If *col2_x* > 0, sections from index *col2_after* onward are drawn
    in a second column starting at *col2_x*.
    """
    line_h = 15
    cy = y
    cy2 = y
    for i, (header, lines) in enumerate(sections):
        if col2_x > 0 and i >= col2_after:
            cx = col2_x
            text = font.render(header, True, (200, 200, 100))
            surface.blit(text, (cx, cy2))
            cy2 += line_h
            for line in lines:
                text = font.render(line, True, (180, 180, 180))
                surface.blit(text, (cx + 4, cy2))
                cy2 += line_h
            cy2 += 4
        else:
            text = font.render(header, True, (200, 200, 100))
            surface.blit(text, (x, cy))
            cy += line_h
            for line in lines:
                text = font.render(line, True, (180, 180, 180))
                surface.blit(text, (x + 4, cy))
                cy += line_h
            cy += 4
    return max(cy, cy2)
