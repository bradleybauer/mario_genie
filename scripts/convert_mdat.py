#!/usr/bin/env python3
"""Convert .mdat research recording files to .npz format.

Usage:
    python convert_mdat.py recording.mdat [-o output.npz] [--extract-frames] [--delta-encode]

The .mdat file is a binary format written by Mesen2's DataCollector:
  - 32-byte header with memory layout metadata
  - Fixed-size per-frame records containing RAM, WRAM, and input state

Companion files (same basename):
  - .avi: lossless video+audio recorded by Mesen's AVI recorder
  - .mss: initial save state at recording start
  - _states/: periodic save states (if configured)
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


MDAT_MAGIC = b"MSDC"
MDAT_VERSION = 1
HEADER_SIZE = 32

# Header struct: magic(4) version(2) console_type(2) ram_size(4) wram_size(4)
#                input_bytes(2) num_controllers(2) start_frame(4) fps_x1000(4) reserved(4)
HEADER_FMT = "<4sHHIIHHIII"


def parse_header(data: bytes) -> dict:
    if len(data) < HEADER_SIZE:
        raise ValueError(f"File too small for header: {len(data)} bytes")

    fields = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    magic, version, console_type, ram_size, wram_size, input_bytes, num_controllers, start_frame, fps_x1000, _ = fields

    if magic != MDAT_MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}, expected {MDAT_MAGIC!r}")
    if version != MDAT_VERSION:
        raise ValueError(f"Unsupported version: {version}, expected {MDAT_VERSION}")

    return {
        "console_type": console_type,
        "ram_size": ram_size,
        "wram_size": wram_size,
        "input_bytes_per_frame": input_bytes,
        "num_controllers": num_controllers,
        "start_frame": start_frame,
        "fps": fps_x1000 / 1000.0,
        "record_size": 4 + ram_size + wram_size + input_bytes,
    }


def read_mdat(path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = path.read_bytes()
    header = parse_header(data)

    record_size = header["record_size"]
    payload = data[HEADER_SIZE:]
    num_frames = len(payload) // record_size

    if len(payload) % record_size != 0:
        tail = len(payload) % record_size
        print(f"Warning: {tail} trailing bytes (truncated frame), ignoring", file=sys.stderr)
        payload = payload[: num_frames * record_size]

    if num_frames == 0:
        raise ValueError("No frames in file")

    print(f"Frames: {num_frames}")
    print(f"RAM: {header['ram_size']} bytes/frame")
    print(f"WRAM: {header['wram_size']} bytes/frame")
    print(f"Input: {header['input_bytes_per_frame']} bytes/frame ({header['num_controllers']} controllers)")
    print(f"FPS: {header['fps']:.3f}")
    print(f"Record size: {record_size} bytes")

    frame_numbers = np.empty(num_frames, dtype=np.uint32)
    ram = np.empty((num_frames, header["ram_size"]), dtype=np.uint8)
    wram = np.empty((num_frames, header["wram_size"]), dtype=np.uint8) if header["wram_size"] > 0 else None
    actions = np.empty(num_frames, dtype=np.uint8)

    offset = 0
    for i in range(num_frames):
        rec = payload[offset : offset + record_size]

        # Frame number (uint32 LE)
        frame_numbers[i] = struct.unpack_from("<I", rec, 0)[0]

        # RAM
        ram_start = 4
        ram[i] = np.frombuffer(rec, dtype=np.uint8, count=header["ram_size"], offset=ram_start)

        # WRAM
        if wram is not None:
            wram_start = ram_start + header["ram_size"]
            wram[i] = np.frombuffer(rec, dtype=np.uint8, count=header["wram_size"], offset=wram_start)

        # Input: take first byte as action bitmask (controller 0 standard buttons)
        input_start = 4 + header["ram_size"] + header["wram_size"]
        if header["input_bytes_per_frame"] > 0:
            actions[i] = rec[input_start]
        else:
            actions[i] = 0

        offset += record_size

    return header, frame_numbers, ram, wram, actions


def delta_encode(arr: np.ndarray) -> np.ndarray:
    """XOR delta-encode: first frame stored as-is, subsequent frames XOR'd with previous."""
    result = np.empty_like(arr)
    result[0] = arr[0]
    result[1:] = np.bitwise_xor(arr[1:], arr[:-1])
    return result


def extract_frames_from_avi(avi_path: Path, num_frames: int) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        print("Warning: opencv-python not installed, skipping frame extraction", file=sys.stderr)
        return None

    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        print(f"Warning: could not open {avi_path}", file=sys.stderr)
        return None

    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if len(frames) != num_frames:
        print(
            f"Warning: AVI has {len(frames)} frames but mdat has {num_frames}",
            file=sys.stderr,
        )

    return np.array(frames, dtype=np.uint8) if frames else None


def main():
    parser = argparse.ArgumentParser(description="Convert .mdat research recording to .npz")
    parser.add_argument("input", type=Path, help="Input .mdat file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .npz file (default: same name)")
    parser.add_argument("--extract-frames", action="store_true", help="Extract frames from companion .avi")
    parser.add_argument("--delta-encode", action="store_true", help="XOR delta-encode RAM/WRAM for compression")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.input.with_suffix(".npz")

    header, frame_numbers, ram, wram, actions = read_mdat(args.input)

    arrays = {
        "frame_numbers": frame_numbers,
        "actions": actions,
    }

    if args.delta_encode:
        arrays["ram"] = delta_encode(ram)
        arrays["ram_delta_encoded"] = np.array(True)
        if wram is not None:
            arrays["wram"] = delta_encode(wram)
            arrays["wram_delta_encoded"] = np.array(True)
    else:
        arrays["ram"] = ram
        if wram is not None:
            arrays["wram"] = wram

    # Store header metadata
    arrays["fps"] = np.array(header["fps"], dtype=np.float64)
    arrays["console_type"] = np.array(header["console_type"], dtype=np.uint16)
    arrays["start_frame"] = np.array(header["start_frame"], dtype=np.uint32)

    if args.extract_frames:
        avi_path = args.input.with_suffix(".avi")
        if avi_path.exists():
            frames = extract_frames_from_avi(avi_path, len(frame_numbers))
            if frames is not None:
                arrays["frames"] = frames
                print(f"Extracted {len(frames)} video frames: {frames.shape}")
        else:
            print(f"Warning: {avi_path} not found, skipping frame extraction", file=sys.stderr)

    np.savez_compressed(output, **arrays)
    print(f"Saved {output} ({output.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  ram: {arrays['ram'].shape}")
    if "wram" in arrays:
        print(f"  wram: {arrays['wram'].shape}")
    print(f"  actions: {arrays['actions'].shape}")
    print(f"  frame_numbers: {arrays['frame_numbers'].shape}")


if __name__ == "__main__":
    main()
