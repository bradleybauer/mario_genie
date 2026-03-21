#!/usr/bin/env python3
"""Convert Mesen binary recordings to session .npz files.

The Lua script (mesen_collect.lua) writes a .bin recording to disk.
This script reads it offline and produces the same session format used
by the rest of the training pipeline.

Usage:
    python scripts/collect_mesen.py --input recording_0000.bin --output data/smb3
    python scripts/collect_mesen.py --input /path/to/*.bin --output data/smb3
    python scripts/collect_mesen.py --input /path/to/dir/ --output data/smb3
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mario_world_model.palette_mapper import PaletteMapper
from mario_world_model.preprocess import preprocess_frame

# ---------------------------------------------------------------------------
# Action mapping (NES input bitmask → COMPLEX_MOVEMENT index)
# ---------------------------------------------------------------------------

def _build_action_map() -> tuple[dict[frozenset[str], int], int, int]:
    """Return (action_map, num_actions, noop_index)."""
    from mario_world_model.actions import ACTION_MEANINGS

    action_map: dict[frozenset[str], int] = {}
    for idx, buttons in enumerate(ACTION_MEANINGS):
        key = frozenset(b.lower() for b in buttons)
        action_map[key] = idx
    noop_idx = action_map.get(frozenset({"noop"}), 0)
    return action_map, len(ACTION_MEANINGS), noop_idx


def input_byte_to_action(
    byte: int,
    action_map: dict[frozenset[str], int],
    noop_idx: int,
) -> int:
    """Convert an 8-bit NES controller bitmask to a COMPLEX_MOVEMENT index.

    Bit layout: 0=A 1=B 2=Select 3=Start 4=Up 5=Down 6=Left 7=Right
    """
    a     = bool(byte & 0x01)
    b     = bool(byte & 0x02)
    up    = bool(byte & 0x10)
    down  = bool(byte & 0x20)
    left  = bool(byte & 0x40)
    right = bool(byte & 0x80)

    if left and right:
        left = False

    if up and not (a or b or left or right or down):
        return action_map.get(frozenset({"up"}), noop_idx)

    buttons: set[str] = set()
    if a:     buttons.add("a")
    if b:     buttons.add("b")
    if left:  buttons.add("left")
    if right: buttons.add("right")
    if down:  buttons.add("down")

    if not buttons:
        return noop_idx
    return action_map.get(frozenset(buttons), noop_idx)


# ---------------------------------------------------------------------------
# SMB3 RAM decoder
# ---------------------------------------------------------------------------

def decode_smb3_state(ram: np.ndarray, wram: np.ndarray) -> dict[str, int]:
    """Extract game-state metadata from SMB3 RAM."""
    world    = int(ram[0x0727]) + 1
    objset   = int(ram[0x070A])
    x_pos    = int(ram[0x0075]) * 256 + int(ram[0x0090])
    y_pos    = int(ram[0x0087]) * 256 + int(ram[0x00A2])
    life     = int(ram[0x0736])
    time_val = int(ram[0x05EE]) * 100 + int(ram[0x05EF]) * 10 + int(ram[0x05F0])
    score    = (int(ram[0x0715]) << 16 | int(ram[0x0716]) << 8 | int(ram[0x0717])) * 10
    power    = int(ram[0x00ED])

    coins_offset = 0x7DA2 - 0x6000
    coins = int(wram[coins_offset]) if coins_offset < len(wram) else 0

    return {
        "world":    world,
        "stage":    objset,
        "x_pos":    x_pos,
        "y_pos":    y_pos,
        "life":     life,
        "time":     time_val,
        "score":    score,
        "coins":    coins,
        "status":   power,
        "flag_get": 0,
    }


# ---------------------------------------------------------------------------
# Binary file reader
# ---------------------------------------------------------------------------

MAGIC        = b"MESD"
VERSION      = 1
RAM_SIZE     = 2048
WRAM_SIZE    = 8192
PALETTE_SIZE = 32
# Per-frame fixed portion: frame(4) + input(1) + ram + wram + palette + png_len(4)
FRAME_FIXED  = 4 + 1 + RAM_SIZE + WRAM_SIZE + PALETTE_SIZE + 4


def iter_frames(path: Path):
    """Yield frame dicts from a Mesen .bin recording."""
    with open(path, "rb") as f:
        header = f.read(5)
        if len(header) < 5:
            raise ValueError(f"{path}: too short for header")
        if header[:4] != MAGIC:
            raise ValueError(f"{path}: bad magic {header[:4]!r}")
        if header[4] != VERSION:
            raise ValueError(f"{path}: unsupported version {header[4]}")

        while True:
            fixed = f.read(FRAME_FIXED)
            if not fixed:
                break
            if len(fixed) < FRAME_FIXED:
                print(f"  Warning: truncated frame at end of file, skipping")
                break

            frame_num  = struct.unpack_from("<I", fixed, 0)[0]
            input_byte = fixed[4]

            off = 5
            ram = np.frombuffer(fixed, dtype=np.uint8, count=RAM_SIZE, offset=off).copy()
            off += RAM_SIZE
            wram = np.frombuffer(fixed, dtype=np.uint8, count=WRAM_SIZE, offset=off).copy()
            off += WRAM_SIZE
            palette = np.frombuffer(fixed, dtype=np.uint8, count=PALETTE_SIZE, offset=off).copy()
            off += PALETTE_SIZE
            png_len = struct.unpack_from("<I", fixed, off)[0]

            png_data = f.read(png_len)
            if len(png_data) < png_len:
                print(f"  Warning: truncated PNG at frame {frame_num}, skipping")
                break

            yield {
                "frame_num": frame_num,
                "input":     input_byte,
                "ram":       ram,
                "wram":      wram,
                "palette":   palette,
                "png":       png_data,
            }


# ---------------------------------------------------------------------------
# Session writer
# ---------------------------------------------------------------------------

META_KEYS = ("world", "stage", "x_pos", "y_pos",
             "score", "coins", "life", "time", "flag_get", "status")


class MesenSessionWriter:
    """Accumulates frames and writes session .npz + .meta.json."""

    _INIT_CAP = 4096

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(output_dir.glob("session_*.npz"))
        idx = max((int(p.stem.split("_")[1]) for p in existing), default=-1) + 1
        self.session_id = f"{idx:06d}"

        cap = self._INIT_CAP
        self._cap   = cap
        self._size  = 0
        self._frames: np.ndarray | None = None
        self._actions = np.empty(cap, dtype=np.uint8)
        self._dones   = np.empty(cap, dtype=bool)
        self._ram     = np.empty((cap, RAM_SIZE), dtype=np.uint8)
        self._wram    = np.empty((cap, WRAM_SIZE), dtype=np.uint8)
        self._meta    = {k: np.empty(cap, dtype=np.int32) for k in META_KEYS}

    def _grow(self) -> None:
        new = self._cap * 2
        for name in ("_actions", "_dones"):
            old = getattr(self, name)
            arr = np.empty(new, dtype=old.dtype)
            arr[: self._size] = old[: self._size]
            setattr(self, name, arr)
        for name in ("_ram", "_wram"):
            old = getattr(self, name)
            arr = np.empty((new, old.shape[1]), dtype=np.uint8)
            arr[: self._size] = old[: self._size]
            setattr(self, name, arr)
        if self._frames is not None:
            arr = np.empty((new, *self._frames.shape[1:]), dtype=np.uint8)
            arr[: self._size] = self._frames[: self._size]
            self._frames = arr
        for k in META_KEYS:
            old = self._meta[k]
            arr = np.empty(new, dtype=np.int32)
            arr[: self._size] = old[: self._size]
            self._meta[k] = arr
        self._cap = new

    def append(
        self,
        frame_chw: np.ndarray,
        action: int,
        ram: np.ndarray,
        wram: np.ndarray,
        state: dict[str, int],
    ) -> None:
        if self._size >= self._cap:
            self._grow()
        i = self._size
        if self._frames is None:
            self._frames = np.empty((self._cap, *frame_chw.shape), dtype=np.uint8)
        self._frames[i] = frame_chw
        self._actions[i] = action
        self._dones[i] = False
        self._ram[i] = ram
        self._wram[i] = wram
        for k in META_KEYS:
            self._meta[k][i] = state.get(k, 0)
        self._size += 1

    @property
    def num_frames(self) -> int:
        return self._size

    def write(self) -> Path | None:
        if self._size == 0 or self._frames is None:
            return None
        n = self._size
        arrays: dict[str, np.ndarray] = {
            "frames":  self._frames[:n],
            "actions": self._actions[:n],
            "dones":   self._dones[:n],
            "ram":     self._ram[:n],
            "wram":    self._wram[:n],
        }
        for k in META_KEYS:
            arrays[k] = self._meta[k][:n]

        npz_path = self.output_dir / f"session_{self.session_id}.npz"
        np.savez_compressed(npz_path, **arrays)

        meta_path = self.output_dir / f"session_{self.session_id}.meta.json"
        with meta_path.open("w") as f:
            json.dump({
                "session_id":      self.session_id,
                "source":          "mesen",
                "num_frames":      n,
            }, f, indent=2)
        return npz_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_inputs(raw: list[str]) -> list[Path]:
    """Expand globs, directories, and plain paths into .bin files."""
    result: list[Path] = []
    for item in raw:
        p = Path(item)
        if p.is_dir():
            result.extend(sorted(p.glob("recording_*.bin")))
        elif "*" in item or "?" in item:
            result.extend(Path(m) for m in sorted(glob.glob(item)))
        else:
            result.append(p)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Mesen .bin recordings to session .npz files"
    )
    parser.add_argument("--input", nargs="+", required=True,
                        help=".bin file(s), glob pattern, or directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory to write session .npz files")
    parser.add_argument("--max-session", type=int, default=0,
                        help="Split sessions after this many frames (0 = one session per file)")
    args = parser.parse_args()

    action_map, _, noop_idx = _build_action_map()

    # Ensure palette.json exists in output dir
    palette_path = args.output / "palette.json"
    if not palette_path.exists():
        src = PROJECT_ROOT / "data" / "palette.json"
        if src.exists():
            args.output.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, palette_path)
    palette_mapper = PaletteMapper(palette_path)

    # Write action meanings
    from mario_world_model.actions import ACTION_MEANINGS
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "action_meanings.json").open("w") as f:
        json.dump({i: m for i, m in enumerate(ACTION_MEANINGS)}, f, indent=2)

    bin_files = resolve_inputs(args.input)
    if not bin_files:
        print("No .bin files found")
        sys.exit(1)

    total_frames = 0
    total_sessions = 0

    for bin_path in bin_files:
        print(f"\nProcessing {bin_path}")
        writer = MesenSessionWriter(args.output)

        for pkt in iter_frames(bin_path):
            # PNG → RGB → pad 256×256 → palette index → (1, H, W)
            rgb = np.array(
                Image.open(io.BytesIO(pkt["png"])).convert("RGB"),
                dtype=np.uint8,
            )
            padded = preprocess_frame(rgb)
            idx_hw = palette_mapper.map_frame(padded)
            frame_chw = idx_hw[np.newaxis, :, :]

            action = input_byte_to_action(pkt["input"], action_map, noop_idx)
            state  = decode_smb3_state(pkt["ram"], pkt["wram"])

            writer.append(frame_chw, action, pkt["ram"], pkt["wram"], state)

            if pkt["frame_num"] % 1000 == 0:
                print(
                    f"  frame {pkt['frame_num']:>6d}  "
                    f"W{state['world']} x={state['x_pos']}"
                )

            # Session rotation
            if 0 < args.max_session <= writer.num_frames:
                path = writer.write()
                print(f"  Saved {path} ({writer.num_frames} frames)")
                total_frames += writer.num_frames
                total_sessions += 1
                writer = MesenSessionWriter(args.output)

        # Flush remaining
        if writer.num_frames > 0:
            path = writer.write()
            print(f"  Saved {path} ({writer.num_frames} frames)")
            total_frames += writer.num_frames
            total_sessions += 1

    print(f"\nDone: {total_sessions} session(s), {total_frames} total frames")


if __name__ == "__main__":
    main()
