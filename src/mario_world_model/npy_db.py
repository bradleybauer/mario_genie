"""Mesen2 .npy recording loader.

Scans a data directory for recordings produced by Mesen2's data-recording
feature and provides two access patterns:

- **Numpy** — ``load_recordings()`` returns a list of ``Recording``
  named tuples with raw numpy arrays (zero copy, minimal memory).
- **DataFrame** — ``build_dataframe()`` expands selected RAM/WRAM bytes
  into a pandas DataFrame for tabular analysis.

Usage
-----
>>> from mario_world_model.npy_db import load_recordings, build_dataframe
>>> recs = load_recordings()              # list[Recording]
>>> recs[0].ram[:, 0x0E]                  # player-state as numpy slice
>>> df = build_dataframe(recs, ram_columns=[0x0E, 0x075F])
>>> df["ram_000e"].value_counts()
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

DEFAULT_DATA_DIR = Path.cwd() / "data"

_STATE_FRAME_RE = re.compile(r"frame_(\d+)\.mss$")


# ---------------------------------------------------------------------------
# SMB1 RAM semantic map  (sources: game_decoders.py + datacrystal wiki)
# ---------------------------------------------------------------------------
# Maps RAM byte index → (short_label, description).
# Only "interesting" addresses — the ones a human would actually query.

SMB1_RAM_LABELS: dict[int, tuple[str, str]] = {
    # -- Core player state --
    0x0001: ("player_anim", "Player animation frame"),
    0x0003: ("player_dir_raw", "Player direction (1=R, 2=L)"),
    0x0009: ("frame_counter", "Frame counter (wraps 0-255)"),
    0x000A: ("btn_ab", "Button state A/B flags (0x40=A, 0x80=B)"),
    0x000B: ("btn_ud", "Vertical direction input (0x40=Down, 0x80=Up)"),
    0x000E: ("player_state", "Player state (0x08=Normal, 0x06=Dead, 0x0B=Dying, …)"),
    0x0014: ("powerup_drawn", "Powerup sprite active on screen"),
    0x001B: ("powerup_onscreen", "Powerup on screen (0x00=No, 0x2E=Yes)"),
    0x001D: ("player_float", "Player float state (0=ground, 1=jump, 2=walk-off, 3=flagpole)"),
    0x0033: ("player_facing", "Player facing direction (0=off-screen, 1=R, 2=L)"),
    0x0039: ("powerup_type", "Powerup type when on screen (0=Mush, 1=Flower, 2=Star, 3=1up)"),
    0x0045: ("player_move_dir", "Player moving direction (1=R, 2=L)"),

    # -- Positions --
    0x0057: ("player_x_speed", "Player horizontal speed (signed)"),
    0x006D: ("player_x_page", "Player horizontal position in level (page)"),
    0x0086: ("player_x_screen", "Player X position on screen"),
    0x009F: ("player_y_vel", "Player vertical velocity (signed, whole pixels)"),
    0x00B5: ("player_y_viewport", "Player vertical screen position viewport"),
    0x00CE: ("player_y_screen", "Player Y position on screen"),

    # -- Enemies --
    0x0016: ("enemy0_type", "Enemy slot 0 type"),
    0x0017: ("enemy1_type", "Enemy slot 1 type"),
    0x0018: ("enemy2_type", "Enemy slot 2 type"),
    0x0019: ("enemy3_type", "Enemy slot 3 type"),
    0x001A: ("enemy4_type", "Enemy slot 4 type"),
    0x006E: ("enemy0_x_page", "Enemy 0 horizontal position in level"),
    0x006F: ("enemy1_x_page", "Enemy 1 horizontal position in level"),
    0x0070: ("enemy2_x_page", "Enemy 2 horizontal position in level"),
    0x0071: ("enemy3_x_page", "Enemy 3 horizontal position in level"),
    0x0072: ("enemy4_x_page", "Enemy 4 horizontal position in level"),
    0x0087: ("enemy0_x_screen", "Enemy 0 X on screen"),
    0x0088: ("enemy1_x_screen", "Enemy 1 X on screen"),
    0x0089: ("enemy2_x_screen", "Enemy 2 X on screen"),
    0x008A: ("enemy3_x_screen", "Enemy 3 X on screen"),
    0x008B: ("enemy4_x_screen", "Enemy 4 X on screen"),
    0x00CF: ("enemy0_y_screen", "Enemy 0 Y on screen"),
    0x00D0: ("enemy1_y_screen", "Enemy 1 Y on screen"),
    0x00D1: ("enemy2_y_screen", "Enemy 2 Y on screen"),
    0x00D2: ("enemy3_y_screen", "Enemy 3 Y on screen"),
    0x00D3: ("enemy4_y_screen", "Enemy 4 Y on screen"),

    # -- Physics --
    0x0400: ("player_x_moveforce", "Player X move force"),
    0x0433: ("player_y_frac_vel", "Player vertical fractional velocity"),
    0x0450: ("player_max_vel_L", "Player max velocity left"),
    0x0456: ("player_max_vel_R", "Player max velocity right"),
    0x0490: ("player_collision", "Player collision bits (0xFF=none, 0xFE=hit)"),
    0x0491: ("enemy_collision", "Enemy collision bits (0x01=player hit enemy)"),

    # -- Hitboxes --
    0x03AD: ("player_x_offset", "Player X pos within current screen offset"),
    0x03B8: ("player_y_offset", "Player Y pos within current screen"),

    # -- Audio --
    0x00FA: ("pause_effect", "Pause effect register"),
    0x00FB: ("area_music", "Area music register"),
    0x00FC: ("event_music", "Event music register"),
    0x00FD: ("sfx_reg1", "Sound effect register 1"),
    0x00FE: ("sfx_reg2", "Sound effect register 2"),
    0x00FF: ("sfx_reg3", "Sound effect register 3"),

    # -- Level / game state --
    0x0700: ("player_x_speed_abs", "Player X speed absolute (0-0x28)"),
    0x0701: ("friction_adder", "Friction adder high (1=braking)"),
    0x0704: ("swimming_flag", "Swimming flag (1=swim)"),
    0x0709: ("gravity", "Current gravity applied to player"),
    0x070A: ("fall_gravity", "Current fall gravity"),
    0x0714: ("ducking", "Ducking flag (0x04 when ducking as big)"),
    0x071A: ("current_screen", "Current screen in level"),
    0x071B: ("next_screen", "Next screen in level"),
    0x0722: ("hit_detect_flag", "Player hit-detect flag (0xFF=pass-through walls)"),
    0x0723: ("scroll_lock", "Scroll lock (1=prevent rightward scroll)"),
    0x0733: ("tree_mushroom_ctrl", "Tree/mushroom platform replacement control"),
    0x0743: ("cloud_tiles", "When true, ground/block tiles become clouds"),
    0x0744: ("bg_palette", "Background palette control"),
    0x074A: ("p1_buttons", "Player 1 buttons pressed (accurate, incl. paused)"),
    0x074B: ("p2_buttons", "Player 2 buttons pressed (accurate)"),
    0x0750: ("area_offset", "Area offset"),
    0x0752: ("level_entry_ctrl", "Level entry control (vine, pipe, fall)"),
    0x0754: ("player_size_state", "Player size state (0=Big, 1=Small)"),
    0x0756: ("powerup_state", "Powerup state (0=Small, 1=Big, 2=Fire)"),

    # -- World / stage / score --
    0x075A: ("lives", "Lives remaining"),
    0x075C: ("stage", "Stage number (0-indexed)"),
    0x075F: ("world", "World number (0-indexed)"),
    0x0760: ("level", "Level number"),
    0x0770: ("game_mode", "Game mode (0=Demo, 1=Playing, 2=World End)"),
    0x0772: ("level_load", "Level loading setting"),
    0x0773: ("level_palette", "Level palette (0=Normal, 1=Water, 2=Night, 3=Under, 4=Castle)"),
    0x0776: ("pause_status", "Game pause status"),
    0x077A: ("num_players", "Number of players"),
    0x077F: ("interval_timer", "Interval timer control (21-frame rule)"),

    # -- Timers --
    0x0781: ("timer_anim", "Player animation timer"),
    0x0782: ("timer_jumpswim", "Jump/swim timer"),
    0x0783: ("timer_running", "Running timer"),
    0x0784: ("timer_blockbounce", "Block bounce timer"),
    0x0786: ("timer_jumpspring", "Jumpspring timer"),
    0x0787: ("timer_gametime_ctrl", "Game timer control (0x02=freeze time)"),
    0x078F: ("timer_frenzy_enemy", "Frenzy enemy timer"),
    0x0791: ("timer_stomp", "Stomp timer"),
    0x0795: ("timer_falling_pit", "Timer: falling down a pit"),
    0x079D: ("timer_multicoin", "Timer: multi-coin block"),
    0x079E: ("timer_invincible", "Timer: invincible after enemy collision"),
    0x079F: ("timer_star", "Timer: star power"),
    0x07A0: ("timer_prelevel", "Timer: pre-level screen"),

    # -- Score / coins / timer display --
    0x07DD: ("score_100k", "Mario score digit: 100000s (BCD)"),
    0x07DE: ("score_10k", "Mario score digit: 10000s (BCD)"),
    0x07DF: ("score_1k", "Mario score digit: 1000s (BCD)"),
    0x07E0: ("score_100", "Mario score digit: 100s (BCD)"),
    0x07E1: ("score_10", "Mario score digit: 10s (BCD)"),
    0x07E2: ("score_1", "Mario score digit: 1s (BCD)"),
    0x07ED: ("coins_tens", "Coins tens digit"),
    0x07EE: ("coins_ones", "Coins ones digit"),
    0x07F8: ("timer_100", "Game timer: 100s digit"),
    0x07F9: ("timer_10", "Game timer: 10s digit"),
    0x07FA: ("timer_1", "Game timer: 1s digit"),
    0x07FC: ("difficulty", "Game difficulty (set after beating the game)"),
}


def get_ram_labels() -> dict[int, tuple[str, str]]:
    """Return the RAM semantic label map.  {byte_index: (short_name, description)}."""
    return dict(SMB1_RAM_LABELS)


# ---------------------------------------------------------------------------
# Recording data structure
# ---------------------------------------------------------------------------

class Recording(NamedTuple):
    """A single recording session loaded from .npy sidecar files."""
    name: str
    base: Path
    frames: np.ndarray                    # (N,) uint32 frame numbers
    ram: np.ndarray                       # (N, R) uint8
    wram: Optional[np.ndarray]            # (N, W) uint8 or None
    input: np.ndarray                     # (N,) uint8
    avi_path: Optional[Path]
    mss_path: Optional[Path]
    save_states: list[tuple[int, Path]]   # [(frame_number, path), ...]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def scan_recordings(data_dir: Optional[Path] = None) -> list[Path]:
    """Return sorted base paths for all recordings in *data_dir*."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    ram_files = sorted(data_dir.glob("*.ram.npy"))
    return [Path(str(f).removesuffix(".ram.npy")) for f in ram_files]


def load_recording(base: Path) -> Recording:
    """Load a single recording from its .npy sidecar files."""
    frame_nums = np.load(str(base) + ".frames.npy")
    ram = np.load(str(base) + ".ram.npy")
    inp = np.load(str(base) + ".input.npy")

    wram_path = Path(str(base) + ".wram.npy")
    wram = np.load(str(wram_path)) if wram_path.exists() else None

    avi = Path(str(base) + ".avi")
    mss = Path(str(base) + ".mss")

    states_dir = base.parent / (base.name + "_states")
    save_states: list[tuple[int, Path]] = []
    if states_dir.is_dir():
        for sf in sorted(states_dir.glob("*.mss")):
            m = _STATE_FRAME_RE.match(sf.name)
            if m:
                save_states.append((int(m.group(1)), sf))

    return Recording(
        name=base.name,
        base=base,
        frames=frame_nums,
        ram=ram,
        wram=wram,
        input=inp,
        avi_path=avi if avi.exists() else None,
        mss_path=mss if mss.exists() else None,
        save_states=save_states,
    )


def load_recordings(
    data_dir: Optional[Path] = None,
    verbose: bool = True,
) -> list[Recording]:
    """Scan *data_dir* and load all recordings.

    Returns an empty list (not an error) if no recordings are found.
    """
    bases = scan_recordings(data_dir)
    recordings = []
    for i, base in enumerate(bases):
        rec = load_recording(base)
        if verbose:
            print(f"[{i+1}/{len(bases)}] {rec.name} ({len(rec.frames):,} frames)",
                  file=sys.stderr)
        recordings.append(rec)
    return recordings


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def build_dataframe(
    recordings: list[Recording],
    *,
    ram_columns: Optional[list[int]] = None,
) -> "pd.DataFrame":
    """Build a pandas DataFrame from loaded recordings.

    Parameters
    ----------
    recordings : list[Recording]
        Output of ``load_recordings()``.
    ram_columns : list[int] | None
        RAM byte indices to expand into columns.
        ``None`` = all bytes, ``[]`` = no RAM columns.
    """
    import pandas as pd

    chunks = []
    for rec_id, rec in enumerate(recordings):
        n = len(rec.frames)
        chunk: dict[str, np.ndarray] = {
            "recording_id": np.full(n, rec_id, dtype=np.int32),
            "frame_number": rec.frames,
            "action": rec.input,
        }

        indices = ram_columns if ram_columns is not None else range(rec.ram.shape[1])
        for i in indices:
            chunk[f"ram_{i:04x}"] = rec.ram[:, i]

        if rec.wram is not None:
            w_indices = ram_columns if ram_columns is not None else range(rec.wram.shape[1])
            for i in w_indices:
                if i < rec.wram.shape[1]:
                    chunk[f"wram_{i:04x}"] = rec.wram[:, i]

        chunks.append(pd.DataFrame(chunk))

    return pd.concat(chunks, ignore_index=True)
