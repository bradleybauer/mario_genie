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

DEFAULT_DATA_DIR = Path.cwd() / "data" / "raw"

_STATE_FRAME_RE = re.compile(r"frame_(\d+)\.mss$")

# Re-export from canonical location for backward compatibility
from mario_world_model.smb1_memory_map import SMB1_RAM_LABELS, get_ram_labels  # noqa: F401, E402


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
