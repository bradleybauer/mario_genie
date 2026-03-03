from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Optional
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
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


def _writer_loop_proc(write_queue: mp.Queue, output_dir: Path, compress: bool) -> None:
    while True:
        item = write_queue.get()
        if item is None:
            break

        (
            chunk_index,
            f_name, f_shape, f_dtype,
            a_name, a_shape, a_dtype,
            d_name, d_shape, d_dtype,
            kwargs_meta,
        ) = item

        shm_f = SharedMemory(name=f_name)
        shm_a = SharedMemory(name=a_name)
        shm_d = SharedMemory(name=d_name)
        
        shm_kwargs = {}
        for k, k_name, k_shape, k_dtype in kwargs_meta:
            shm_kwargs[k] = SharedMemory(name=k_name)

        try:
            frames_btchw = np.ndarray(f_shape, dtype=f_dtype, buffer=shm_f.buf)
            actions_bt = np.ndarray(a_shape, dtype=a_dtype, buffer=shm_a.buf)
            dones_bt = np.ndarray(d_shape, dtype=d_dtype, buffer=shm_d.buf)
            
            kwargs_arrays = {}
            for k, (_, _, k_shape, k_dtype) in zip(shm_kwargs.keys(), kwargs_meta):
                kwargs_arrays[k] = np.ndarray(k_shape, dtype=k_dtype, buffer=shm_kwargs[k].buf)

            out_path = output_dir / f"chunk_{chunk_index:06d}.npz"

            if compress:
                np.savez_compressed(
                    out_path,
                    frames=frames_btchw,
                    actions=actions_bt,
                    dones=dones_bt,
                    **kwargs_arrays,
                )
            else:
                np.savez(
                    out_path,
                    frames=frames_btchw,
                    actions=actions_bt,
                    dones=dones_bt,
                    **kwargs_arrays,
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

            with (output_dir / f"chunk_{chunk_index:06d}.meta.json").open("w", encoding="utf-8") as f:
                json.dump(asdict(meta), f, indent=2)

        finally:
            shm_f.close()
            shm_f.unlink()
            shm_a.close()
            shm_a.unlink()
            shm_d.close()
            shm_d.unlink()
            for k_shm in shm_kwargs.values():
                k_shm.close()
                k_shm.unlink()


class ChunkWriter:
    """
    Writes fixed-length sequence chunks to compressed .npz files.
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

        # Autodetect next available chunk index to prevent overwriting
        existing_chunks = list(self.output_dir.glob("chunk_*.npz"))
        if existing_chunks:
            highest_chunk = max(int(p.stem.split("_")[1]) for p in existing_chunks)
            self._chunk_index = highest_chunk + 1
        else:
            self._chunk_index = 0

        self._current_seq_idx = 0

        self._shm_f: Optional[SharedMemory] = None
        self._shm_a: Optional[SharedMemory] = None
        self._shm_d: Optional[SharedMemory] = None
        self._frames_buf: Optional[np.ndarray] = None
        self._actions_buf: Optional[np.ndarray] = None
        self._dones_buf: Optional[np.ndarray] = None
        
        self._shm_kwargs: dict[str, SharedMemory] = {}
        self._kwargs_buf: dict[str, np.ndarray] = {}

        self._write_queue: Optional[mp.Queue] = None
        self._writer_process: Optional[mp.Process] = None

        if self.async_write:
            self._write_queue = mp.Queue(maxsize=self.max_pending_writes)
            self._writer_process = mp.Process(
                target=_writer_loop_proc,
                args=(self._write_queue, self.output_dir, self.compress),
                daemon=True,
            )
            self._writer_process.start()

    def _init_buffers(self, f_shape: tuple, f_dtype: np.dtype, a_shape: tuple, a_dtype: np.dtype, d_shape: tuple, d_dtype: np.dtype, kwargs_shapes_dtypes: dict) -> None:
        full_f_shape = (self.sequences_per_chunk,) + f_shape
        full_a_shape = (self.sequences_per_chunk,) + a_shape
        full_d_shape = (self.sequences_per_chunk,) + d_shape

        bytes_f = np.prod(full_f_shape) * np.dtype(f_dtype).itemsize
        bytes_a = np.prod(full_a_shape) * np.dtype(a_dtype).itemsize
        bytes_d = np.prod(full_d_shape) * np.dtype(d_dtype).itemsize

        self._shm_f = SharedMemory(create=True, size=int(bytes_f))
        self._shm_a = SharedMemory(create=True, size=int(bytes_a))
        self._shm_d = SharedMemory(create=True, size=int(bytes_d))

        self._frames_buf = np.ndarray(full_f_shape, dtype=f_dtype, buffer=self._shm_f.buf)
        self._actions_buf = np.ndarray(full_a_shape, dtype=a_dtype, buffer=self._shm_a.buf)
        self._dones_buf = np.ndarray(full_d_shape, dtype=d_dtype, buffer=self._shm_d.buf)
        
        self._shm_kwargs = {}
        self._kwargs_buf = {}
        for k, (k_shape, k_dtype) in kwargs_shapes_dtypes.items():
            full_k_shape = (self.sequences_per_chunk,) + k_shape
            bytes_k = np.prod(full_k_shape) * np.dtype(k_dtype).itemsize
            self._shm_kwargs[k] = SharedMemory(create=True, size=int(bytes_k))
            self._kwargs_buf[k] = np.ndarray(full_k_shape, dtype=k_dtype, buffer=self._shm_kwargs[k].buf)

    def add_sequence(self, frames_tchw: np.ndarray, actions_t: np.ndarray, dones_t: np.ndarray, **kwargs: np.ndarray) -> Optional[Path]:
        if frames_tchw.shape[0] != self.sequence_length:
            raise ValueError("frames sequence length mismatch")
        if actions_t.shape[0] != self.sequence_length:
            raise ValueError("actions sequence length mismatch")
        if dones_t.shape[0] != self.sequence_length:
            raise ValueError("dones sequence length mismatch")
        for k, v in kwargs.items():
            if v.shape[0] != self.sequence_length:
                raise ValueError(f"{k} sequence length mismatch")

        if self._frames_buf is None:
            kwargs_shapes_dtypes = {k: (v.shape, v.dtype) for k, v in kwargs.items()}
            self._init_buffers(
                frames_tchw.shape, frames_tchw.dtype,
                actions_t.shape, actions_t.dtype,
                dones_t.shape, dones_t.dtype,
                kwargs_shapes_dtypes
            )

        assert self._frames_buf is not None
        assert self._actions_buf is not None
        assert self._dones_buf is not None

        idx = self._current_seq_idx
        self._frames_buf[idx] = frames_tchw
        self._actions_buf[idx] = actions_t
        self._dones_buf[idx] = dones_t
        for k, v in kwargs.items():
            self._kwargs_buf[k][idx] = v

        self._current_seq_idx += 1

        if self._current_seq_idx >= self.sequences_per_chunk:
            return self.flush()
        return None

    def _write_chunk_sync(self, chunk_index: int, frames_btchw: np.ndarray, actions_bt: np.ndarray, dones_bt: np.ndarray, **kwargs_arrays: np.ndarray) -> Path:
        out_path = self.output_dir / f"chunk_{chunk_index:06d}.npz"
        if self.compress:
            np.savez_compressed(out_path, frames=frames_btchw, actions=actions_bt, dones=dones_bt, **kwargs_arrays)
        else:
            np.savez(out_path, frames=frames_btchw, actions=actions_bt, dones=dones_bt, **kwargs_arrays)

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

    def flush(self) -> Optional[Path]:
        if self._current_seq_idx == 0 or self._frames_buf is None:
            return None

        # Slice the used part
        frames_part = self._frames_buf[:self._current_seq_idx].astype(np.uint8, copy=False)
        actions_part = self._actions_buf[:self._current_seq_idx]
        dones_part = self._dones_buf[:self._current_seq_idx]
        kwargs_part = {k: v[:self._current_seq_idx] for k, v in self._kwargs_buf.items()}

        out_path = self.output_dir / f"chunk_{self._chunk_index:06d}.npz"
        chunk_index = self._chunk_index

        if self.async_write:
            assert self._write_queue is not None
            assert self._shm_f is not None and self._shm_a is not None and self._shm_d is not None
            
            if self._current_seq_idx == self.sequences_per_chunk:
                kwargs_meta = [(k, self._shm_kwargs[k].name, self._kwargs_buf[k].shape, self._kwargs_buf[k].dtype.str) for k in self._kwargs_buf]
                self._write_queue.put((
                    chunk_index,
                    self._shm_f.name, self._frames_buf.shape, self._frames_buf.dtype.str,
                    self._shm_a.name, self._actions_buf.shape, self._actions_buf.dtype.str,
                    self._shm_d.name, self._dones_buf.shape, self._dones_buf.dtype.str,
                    kwargs_meta,
                ))
                from multiprocessing.resource_tracker import unregister
                unregister(self._shm_f._name, 'shared_memory')
                unregister(self._shm_a._name, 'shared_memory')
                unregister(self._shm_d._name, 'shared_memory')
                for shm in self._shm_kwargs.values():
                    unregister(shm._name, 'shared_memory')
                
                self._shm_f.close()
                self._shm_a.close()
                self._shm_d.close()
                for shm in self._shm_kwargs.values():
                    shm.close()
                
                self._shm_f = self._shm_a = self._shm_d = None
                self._frames_buf = self._actions_buf = self._dones_buf = None
                self._shm_kwargs = {}
                self._kwargs_buf = {}
            else:
                kwargs_part_copy = {k: v.copy() for k, v in kwargs_part.items()}
                self._write_chunk_sync(chunk_index, frames_part.copy(), actions_part.copy(), dones_part.copy(), **kwargs_part_copy)
                self._shm_f.close()
                self._shm_f.unlink()
                self._shm_a.close()
                self._shm_a.unlink()
                self._shm_d.close()
                self._shm_d.unlink()
                for shm in self._shm_kwargs.values():
                    shm.close()
                    shm.unlink()
                
                self._shm_f = self._shm_a = self._shm_d = None
                self._frames_buf = self._actions_buf = self._dones_buf = None
                self._shm_kwargs = {}
                self._kwargs_buf = {}
        else:
            self._write_chunk_sync(chunk_index, frames_part, actions_part, dones_part, **kwargs_part)
            if self._shm_f is not None:
                self._shm_f.close()
                self._shm_f.unlink()
                self._shm_a.close()
                self._shm_a.unlink()
                self._shm_d.close()
                self._shm_d.unlink()
                for shm in self._shm_kwargs.values():
                    shm.close()
                    shm.unlink()
                self._shm_f = self._shm_a = self._shm_d = None
                self._frames_buf = self._actions_buf = self._dones_buf = None
                self._shm_kwargs = {}
                self._kwargs_buf = {}
        
        self._chunk_index += 1
        self._current_seq_idx = 0

        return out_path

    def close(self) -> None:
        self.flush()

        if self.async_write and self._write_queue is not None and self._writer_process is not None:
            self._write_queue.put(None)
            self._writer_process.join()

    def __del__(self):
        try:
            if getattr(self, '_shm_f', None) is not None:
                self._shm_f.close()
                self._shm_f.unlink()
            if getattr(self, '_shm_a', None) is not None:
                self._shm_a.close()
                self._shm_a.unlink()
            if getattr(self, '_shm_d', None) is not None:
                self._shm_d.close()
                self._shm_d.unlink()
            if getattr(self, '_shm_kwargs', None) is not None:
                for shm in self._shm_kwargs.values():
                    shm.close()
                    shm.unlink()
        except Exception:
            pass
