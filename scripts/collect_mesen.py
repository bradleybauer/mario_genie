#!/usr/bin/env python3
"""Receive streaming data from Mesen's Lua collector and save session .npz files.

Each session contains palette-indexed frames, actions, full RAM/WRAM dumps,
and decoded game-state metadata.

Usage:
    python scripts/collect_mesen.py --output data/smb3
    python scripts/collect_mesen.py --output data/smb3 --max-session 18000

Then load mesen_collect.lua in Mesen's script window and play the game.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import socket
import struct
import sys
import time
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

    # Conflicting directions — prefer right
    if left and right:
        left = False

    # "up" is only a standalone action in COMPLEX_MOVEMENT
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
    """Extract game-state metadata from SMB3 RAM.

    Uses the same addresses as game_decoders.decode_smb3.
    """
    world    = int(ram[0x0727]) + 1
    objset   = int(ram[0x070A])
    x_pos    = int(ram[0x0075]) * 256 + int(ram[0x0090])
    y_pos    = int(ram[0x0087]) * 256 + int(ram[0x00A2])
    life     = int(ram[0x0736])
    time_val = int(ram[0x05EE]) * 100 + int(ram[0x05EF]) * 10 + int(ram[0x05F0])
    score    = (int(ram[0x0715]) << 16 | int(ram[0x0716]) << 8 | int(ram[0x0717])) * 10
    power    = int(ram[0x00ED])

    # Coins from WRAM $7DA2
    coins_offset = 0x7DA2 - 0x6000
    coins = int(wram[coins_offset]) if coins_offset < len(wram) else 0

    return {
        "world":    world,
        "stage":    objset,     # tileset acts as stage identifier in SMB3
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
# Binary protocol
# ---------------------------------------------------------------------------

MAGIC       = b"MESF"
RAM_SIZE    = 2048
WRAM_SIZE   = 8192
PALETTE_SIZE = 32
# magic(4) + frame(4) + input(1) + ram + wram + palette + png_len(4)
FIXED_SIZE  = 4 + 4 + 1 + RAM_SIZE + WRAM_SIZE + PALETTE_SIZE + 4


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            raise ConnectionError("disconnected")
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock: socket.socket) -> dict:
    """Read one frame packet from the Mesen Lua script."""
    fixed = _recv_exact(sock, FIXED_SIZE)

    if fixed[:4] != MAGIC:
        raise ValueError(f"bad magic: {fixed[:4]!r}")

    frame_num  = struct.unpack_from("<I", fixed, 4)[0]
    input_byte = fixed[8]

    off = 9
    ram = np.frombuffer(fixed, dtype=np.uint8, count=RAM_SIZE, offset=off).copy()
    off += RAM_SIZE
    wram = np.frombuffer(fixed, dtype=np.uint8, count=WRAM_SIZE, offset=off).copy()
    off += WRAM_SIZE
    palette = np.frombuffer(fixed, dtype=np.uint8, count=PALETTE_SIZE, offset=off).copy()
    off += PALETTE_SIZE
    png_len = struct.unpack_from("<I", fixed, off)[0]

    png_data = _recv_exact(sock, png_len)

    return {
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
    """Accumulates Mesen frames and writes session .npz + .meta.json."""

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

    def write(self, elapsed: float) -> Path | None:
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
                "elapsed_seconds": round(elapsed, 2),
            }, f, indent=2)
        return npz_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Mesen data collection receiver")
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory to write session .npz files")
    parser.add_argument("--port", type=int, default=7275)
    parser.add_argument("--max-session", type=int, default=0,
                        help="Split sessions after this many frames (0 = unlimited)")
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

    # Write action meanings for downstream compatibility
    from mario_world_model.actions import ACTION_MEANINGS
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "action_meanings.json").open("w") as f:
        json.dump({i: m for i, m in enumerate(ACTION_MEANINGS)}, f, indent=2)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print(f"Listening on port {args.port} — load mesen_collect.lua in Mesen")

    try:
        while True:
            conn, addr = srv.accept()
            print(f"Connected: {addr}")

            writer = MesenSessionWriter(args.output)
            t0 = time.time()

            try:
                while True:
                    pkt = recv_frame(conn)

                    # PNG → RGB → pad 256×256 → palette index → (1, H, W)
                    rgb = np.array(
                        Image.open(io.BytesIO(pkt["png"])).convert("RGB"),
                        dtype=np.uint8,
                    )
                    padded = preprocess_frame(rgb)
                    idx_hw = palette_mapper.map_frame(padded)
                    frame_chw = idx_hw[np.newaxis, :, :]  # (1, 256, 256)

                    action = input_byte_to_action(pkt["input"], action_map, noop_idx)
                    state  = decode_smb3_state(pkt["ram"], pkt["wram"])

                    writer.append(frame_chw, action, pkt["ram"], pkt["wram"], state)

                    if pkt["frame_num"] % 300 == 0:
                        elapsed = time.time() - t0
                        fps = writer.num_frames / max(elapsed, 0.001)
                        print(
                            f"  frame {pkt['frame_num']:>6d}  "
                            f"n={writer.num_frames}  "
                            f"{fps:.0f} fps  "
                            f"W{state['world']} x={state['x_pos']}"
                        )

                    # Session rotation
                    if 0 < args.max_session <= writer.num_frames:
                        elapsed = time.time() - t0
                        path = writer.write(elapsed)
                        print(f"Saved {path} ({writer.num_frames} frames, {elapsed:.1f}s)")
                        writer = MesenSessionWriter(args.output)
                        t0 = time.time()

            except (ConnectionError, ConnectionResetError, BrokenPipeError):
                print("Disconnected")
            except ValueError as exc:
                print(f"Protocol error: {exc}")
            finally:
                conn.close()
                if writer.num_frames > 0:
                    elapsed = time.time() - t0
                    path = writer.write(elapsed)
                    print(f"Saved {path} ({writer.num_frames} frames, {elapsed:.1f}s)")
                print("Waiting for next connection...\n")

    except KeyboardInterrupt:
        print("\nShutdown")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
