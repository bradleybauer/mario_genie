from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Optional
import queue
import threading

import numpy as np


@dataclass
class ChunkMeta:
    chunk_index: int
    num_sequences: int
    sequence_length: int
    frame_shape_btchw: tuple[int, int, int, int, int]
    action_shape_bt: tuple[int, int]
    total_frames: int
    total_actions: int


class ChunkWriter:
    """
    Writes fixed-length sequence chunks to compressed .npz files.

    Output per chunk:
      - frames: uint8, shape (B, T, C, H, W)
      - actions: uint8/int16, shape (B, T)
      - dones: bool, shape (B, T)
      - info_json: json metadata sidecar for inspectability
    """

    def __init__(
        self,
        output_dir: str | Path,
        sequence_length: int,
        sequences_per_chunk: int,
        *,
        compress: bool = True,
        async_write: bool = False,
        max_pending_writes: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sequence_length = int(sequence_length)
        self.sequences_per_chunk = int(sequences_per_chunk)
        self.compress = bool(compress)
        self.async_write = bool(async_write)
        self.max_pending_writes = int(max_pending_writes)

        self._frames_buffer: list[np.ndarray] = []
        self._actions_buffer: list[np.ndarray] = []
        self._dones_buffer: list[np.ndarray] = []
        self._chunk_index = 0

        self._write_queue: Optional[queue.Queue] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_error: Optional[Exception] = None

        if self.async_write:
            self._write_queue = queue.Queue(maxsize=self.max_pending_writes)
            self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer_thread.start()

    def _check_writer_error(self) -> None:
        if self._writer_error is not None:
            err = self._writer_error
            self._writer_error = None
            raise RuntimeError("Background chunk writer failed") from err

    def _write_chunk(self, chunk_index: int, frames_btchw: np.ndarray, actions_bt: np.ndarray, dones_bt: np.ndarray) -> Path:
        out_path = self.output_dir / f"chunk_{chunk_index:06d}.npz"

        if self.compress:
            np.savez_compressed(
                out_path,
                frames=frames_btchw,
                actions=actions_bt,
                dones=dones_bt,
            )
        else:
            np.savez(
                out_path,
                frames=frames_btchw,
                actions=actions_bt,
                dones=dones_bt,
            )

        meta = ChunkMeta(
            chunk_index=chunk_index,
            num_sequences=int(frames_btchw.shape[0]),
            sequence_length=int(frames_btchw.shape[1]),
            frame_shape_btchw=tuple(int(x) for x in frames_btchw.shape),
            action_shape_bt=tuple(int(x) for x in actions_bt.shape),
            total_frames=int(frames_btchw.shape[0] * frames_btchw.shape[1]),
            total_actions=int(actions_bt.size),
        )

        with (self.output_dir / f"chunk_{chunk_index:06d}.meta.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)

        return out_path

    def _writer_loop(self) -> None:
        assert self._write_queue is not None
        while True:
            item = self._write_queue.get()
            if item is None:
                self._write_queue.task_done()
                break

            chunk_index, frames_btchw, actions_bt, dones_bt = item
            try:
                self._write_chunk(chunk_index, frames_btchw, actions_bt, dones_bt)
            except Exception as exc:
                self._writer_error = exc
            finally:
                self._write_queue.task_done()

    def add_sequence(self, frames_tchw: np.ndarray, actions_t: np.ndarray, dones_t: np.ndarray) -> Optional[Path]:
        self._check_writer_error()

        if frames_tchw.shape[0] != self.sequence_length:
            raise ValueError("frames sequence length mismatch")
        if actions_t.shape[0] != self.sequence_length:
            raise ValueError("actions sequence length mismatch")
        if dones_t.shape[0] != self.sequence_length:
            raise ValueError("dones sequence length mismatch")

        self._frames_buffer.append(frames_tchw)
        self._actions_buffer.append(actions_t)
        self._dones_buffer.append(dones_t)

        if len(self._frames_buffer) >= self.sequences_per_chunk:
            return self.flush()
        return None

    def flush(self) -> Optional[Path]:
        self._check_writer_error()

        if not self._frames_buffer:
            return None

        frames_btchw = np.stack(self._frames_buffer, axis=0).astype(np.uint8, copy=False)
        actions_bt = np.stack(self._actions_buffer, axis=0)
        dones_bt = np.stack(self._dones_buffer, axis=0)

        out_path = self.output_dir / f"chunk_{self._chunk_index:06d}.npz"

        chunk_index = self._chunk_index
        if self.async_write:
            assert self._write_queue is not None
            self._write_queue.put((chunk_index, frames_btchw, actions_bt, dones_bt))
        else:
            self._write_chunk(chunk_index, frames_btchw, actions_bt, dones_bt)

        self._chunk_index += 1
        self._frames_buffer.clear()
        self._actions_buffer.clear()
        self._dones_buffer.clear()

        return out_path

    def close(self) -> None:
        self._check_writer_error()
        self.flush()

        if self.async_write and self._write_queue is not None and self._writer_thread is not None:
            self._write_queue.put(None)
            self._writer_thread.join()

        self._check_writer_error()
