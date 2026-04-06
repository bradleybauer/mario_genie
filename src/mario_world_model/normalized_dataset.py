from __future__ import annotations

import concurrent.futures
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
except ImportError:
    Console = None
    Progress = None

from mario_world_model.config import AUDIO_SAMPLE_RATE
from mario_world_model.system_info import get_available_memory, get_effective_cpu_count


_DATASET_CONSOLE = Console() if Console is not None else None


def _log(message: str) -> None:
    if _DATASET_CONSOLE is not None:
        _DATASET_CONSOLE.print(message)
    else:
        print(message)


def load_palette_info(data_dir: str | Path) -> dict[str, Any]:
    palette_path = Path(data_dir) / "palette.json"
    if not palette_path.is_file():
        raise FileNotFoundError(f"Missing palette.json in {palette_path.parent}")
    with palette_path.open() as handle:
        return json.load(handle)


def load_palette_tensor(data_dir: str | Path) -> torch.Tensor:
    palette_info = load_palette_info(data_dir)
    return torch.tensor(palette_info["colors_rgb"], dtype=torch.float32) / 255.0


def _index_normalized_file(
    file_idx: int,
    path: Path,
    *,
    clip_frames: int,
    stride: int,
    include_audio: bool,
    include_actions: bool,
    include_ram: bool,
    load_arrays: bool,
) -> tuple[
    int,
    int,
    list[tuple[int, int]],
    int | None,
    int | None,
    float | None,
    str | None,
    str | None,
    tuple[int, ...] | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    try:
        mmap_mode = None if load_arrays else "r"
        with np.load(path, mmap_mode=mmap_mode) as npz:
            if "frames" not in npz.files:
                return file_idx, 0, [], None, None, None, None, f"{path.name} is missing the 'frames' array", None, None, None, None, None, None
            frames_arr = np.asarray(npz["frames"]) if load_arrays else npz["frames"]
            frame_count = int(frames_arr.shape[0])
            frame_shape = tuple(int(dim) for dim in frames_arr.shape[1:])
            loaded_frames = np.asarray(frames_arr) if load_arrays else None

            if include_actions and "actions" not in npz.files:
                return file_idx, 0, [], None, None, None, None, f"{path.name} is missing the 'actions' array", frame_shape, loaded_frames, None, None, None, None

            audio_width: int | None = None
            audio_sample_rate: int | None = None
            video_fps: float | None = None
            loaded_actions: np.ndarray | None = None
            loaded_audio: np.ndarray | None = None
            loaded_audio_lengths: np.ndarray | None = None
            loaded_ram: np.ndarray | None = None

            if include_actions and load_arrays:
                loaded_actions = np.asarray(npz["actions"])

            if include_ram and load_arrays and "ram" in npz.files:
                loaded_ram = np.asarray(npz["ram"])

            if include_audio:
                if "audio" not in npz.files:
                    return file_idx, 0, [], None, None, None, path.name, None, frame_shape, loaded_frames, loaded_actions, None, None, loaded_ram
                audio = np.asarray(npz["audio"]) if load_arrays else npz["audio"]
                if audio.shape[0] != frame_count:
                    return (
                        file_idx,
                        0,
                        [],
                        None,
                        None,
                        None,
                        None,
                        f"{path.name} has {frame_count} frames but {audio.shape[0]} audio entries",
                        frame_shape,
                        loaded_frames,
                        loaded_actions,
                        None,
                        None,
                        loaded_ram,
                    )
                audio_width = int(audio.shape[1])
                if "audio_sample_rate" in npz.files:
                    audio_sample_rate = int(np.asarray(npz["audio_sample_rate"]).item())
                if "video_fps" in npz.files:
                    video_fps = float(np.asarray(npz["video_fps"]).item())
                if load_arrays:
                    loaded_audio = np.asarray(audio)
                    if "audio_lengths" in npz.files:
                        loaded_audio_lengths = np.asarray(npz["audio_lengths"], dtype=np.int32)
                    else:
                        loaded_audio_lengths = np.full((frame_count,), audio.shape[1], dtype=np.int32)

            if frame_count < clip_frames:
                return (
                    file_idx,
                    0,
                    [],
                    audio_width,
                    audio_sample_rate,
                    video_fps,
                    None,
                    None,
                    frame_shape,
                    loaded_frames,
                    loaded_actions,
                    loaded_audio,
                    loaded_audio_lengths,
                    loaded_ram,
                )

            samples = [(file_idx, start_idx) for start_idx in range(0, frame_count - clip_frames + 1, stride)]
            return (
                file_idx,
                frame_count,
                samples,
                audio_width,
                audio_sample_rate,
                video_fps,
                None,
                None,
                frame_shape,
                loaded_frames,
                loaded_actions,
                loaded_audio,
                loaded_audio_lengths,
                loaded_ram,
            )
    except Exception as exc:
        return file_idx, 0, [], None, None, None, None, f"{path.name}: {exc}", None, None, None, None, None, None


class NormalizedSequenceDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        clip_frames: int,
        *,
        include_frames: bool = True,
        include_audio: bool = False,
        include_actions: bool = False,
        include_ram: bool = False,
        stride: int = 1,
        subset_n: int = 0,
        seed: int = 42,
        num_workers: int | None = None,
        system_info: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if clip_frames <= 0:
            raise ValueError("clip_frames must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        if num_workers is None:
            num_workers = get_effective_cpu_count(system_info)
        index_workers = max(int(num_workers), 1)

        self.data_dir = Path(data_dir)
        self.clip_frames = clip_frames
        self.include_frames = include_frames
        self.include_audio = include_audio
        self.include_actions = include_actions
        self.include_ram = include_ram
        self.stride = stride
        self.data_files = sorted(self.data_dir.glob("*.npz"))
        self.samples: list[tuple[int, int]] = []
        self._npz_cache: dict[int, np.lib.npyio.NpzFile] = {}
        self.frames_by_file: list[np.ndarray] | None = None
        self.actions_by_file: list[np.ndarray] | None = None
        self.audio_by_file: list[np.ndarray] | None = None
        self.audio_lengths_by_file: list[np.ndarray] | None = None
        self.ram_by_file: list[np.ndarray] | None = None
        self.ram_n_bytes: int | None = None
        self.audio_frame_size: int | None = None
        self.audio_sample_rate = AUDIO_SAMPLE_RATE
        self.audio_sample_rate_by_file: list[int | None] = []
        self.video_fps_by_file: list[float | None] = []
        self.num_files = len(self.data_files)
        self.total_frames = 0
        self.dataset_bytes = sum(path.stat().st_size for path in self.data_files)

        if not self.data_files:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")

        missing_audio: list[str] = []
        index_errors: list[str] = []
        frame_counts = [0] * self.num_files
        audio_widths: list[int | None] = [None] * self.num_files
        audio_sample_rates: list[int | None] = [None] * self.num_files
        video_fps_values: list[float | None] = [None] * self.num_files
        frame_shapes: list[tuple[int, ...] | None] = [None] * self.num_files

        available = get_available_memory(system_info)
        headroom = 2 * 2**30
        # Conservative gate: avoid a second pass only when disk size is comfortably below RAM budget.
        load_during_index = (
            available > 0
            and self.dataset_bytes < max(available - headroom, 0) * 0.5
        )
        if load_during_index:
            if self.include_frames:
                self.frames_by_file = [None] * self.num_files
            if self.include_actions:
                self.actions_by_file = [None] * self.num_files
            if self.include_audio:
                self.audio_by_file = [None] * self.num_files
                self.audio_lengths_by_file = [None] * self.num_files
            if self.include_ram:
                self.ram_by_file = [None] * self.num_files

        _log(f"[dataset] Indexing {self.num_files:,} data files (stride={self.stride})...")
        index_start = time.time()
        last_index_report = index_start
        completed_index = 0
        use_live_progress = Progress is not None and sys.stdout.isatty()
        with concurrent.futures.ThreadPoolExecutor(max_workers=index_workers) as pool:
            futures = {
                pool.submit(
                    _index_normalized_file,
                    file_idx,
                    path,
                    clip_frames=self.clip_frames,
                    stride=self.stride,
                    include_audio=self.include_audio,
                    include_actions=self.include_actions,
                    include_ram=self.include_ram,
                    load_arrays=load_during_index,
                ): file_idx
                for file_idx, path in enumerate(self.data_files)
            }
            if use_live_progress:
                assert _DATASET_CONSOLE is not None
                with Progress(
                    SpinnerColumn(style="cyan"),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=None),
                    TextColumn("[progress.percentage]{task.completed}/{task.total}"),
                    TextColumn("[cyan]{task.fields[rate]:>6.1f} files/s"),
                    TimeElapsedColumn(),
                    refresh_per_second=8,
                    console=_DATASET_CONSOLE,
                ) as progress:
                    index_task = progress.add_task("Indexing files", total=self.num_files, rate=0.0)
                    for future in concurrent.futures.as_completed(futures):
                        (
                            file_idx,
                            frame_count,
                            samples,
                            audio_width,
                            audio_sample_rate,
                            video_fps,
                            missing_audio_name,
                            error,
                            frame_shape,
                            loaded_frames,
                            loaded_actions,
                            loaded_audio,
                            loaded_audio_lengths,
                            loaded_ram,
                        ) = future.result()
                        if missing_audio_name is not None:
                            missing_audio.append(missing_audio_name)
                        elif error is not None:
                            index_errors.append(error)
                        else:
                            frame_counts[file_idx] = frame_count
                            self.samples.extend(samples)
                            audio_widths[file_idx] = audio_width
                            audio_sample_rates[file_idx] = audio_sample_rate
                            video_fps_values[file_idx] = video_fps
                            frame_shapes[file_idx] = frame_shape
                            if load_during_index:
                                if self.include_frames and self.frames_by_file is not None:
                                    self.frames_by_file[file_idx] = loaded_frames
                                if self.include_actions and self.actions_by_file is not None:
                                    self.actions_by_file[file_idx] = loaded_actions
                                if self.include_audio and self.audio_by_file is not None and self.audio_lengths_by_file is not None:
                                    self.audio_by_file[file_idx] = loaded_audio
                                    self.audio_lengths_by_file[file_idx] = loaded_audio_lengths
                                if self.include_ram and self.ram_by_file is not None:
                                    self.ram_by_file[file_idx] = loaded_ram

                        completed_index += 1
                        elapsed = max(time.time() - index_start, 1e-9)
                        progress.update(index_task, advance=1, rate=completed_index / elapsed)
            else:
                for future in concurrent.futures.as_completed(futures):
                    (
                        file_idx,
                        frame_count,
                        samples,
                        audio_width,
                        audio_sample_rate,
                        video_fps,
                        missing_audio_name,
                        error,
                        frame_shape,
                        loaded_frames,
                        loaded_actions,
                        loaded_audio,
                        loaded_audio_lengths,
                        loaded_ram,
                    ) = future.result()
                    if missing_audio_name is not None:
                        missing_audio.append(missing_audio_name)
                    elif error is not None:
                        index_errors.append(error)
                    else:
                        frame_counts[file_idx] = frame_count
                        self.samples.extend(samples)
                        audio_widths[file_idx] = audio_width
                        audio_sample_rates[file_idx] = audio_sample_rate
                        video_fps_values[file_idx] = video_fps
                        frame_shapes[file_idx] = frame_shape
                        if load_during_index:
                            if self.include_frames and self.frames_by_file is not None:
                                self.frames_by_file[file_idx] = loaded_frames
                            if self.include_actions and self.actions_by_file is not None:
                                self.actions_by_file[file_idx] = loaded_actions
                            if self.include_audio and self.audio_by_file is not None and self.audio_lengths_by_file is not None:
                                self.audio_by_file[file_idx] = loaded_audio
                                self.audio_lengths_by_file[file_idx] = loaded_audio_lengths
                            if self.include_ram and self.ram_by_file is not None:
                                self.ram_by_file[file_idx] = loaded_ram

                    completed_index += 1
                    now = time.time()
                    if completed_index == self.num_files or now - last_index_report >= 2.0:
                        elapsed = max(now - index_start, 1e-9)
                        rate = completed_index / elapsed
                        _log(
                            f"  Indexed {completed_index}/{self.num_files} files "
                            f"({rate:.1f} files/s)"
                        )
                        last_index_report = now

        if index_errors:
            preview = "; ".join(index_errors[:3])
            raise ValueError(f"Failed to index normalized dataset: {preview}")

        if missing_audio:
            preview = ", ".join(missing_audio[:3])
            raise ValueError(
                "Normalized dataset is missing audio arrays. Re-run scripts/normalize.py before training the audio VAE. "
                f"Examples: {preview}"
            )

        valid_files = [i for i, frame_count in enumerate(frame_counts) if frame_count > 0]
        if len(valid_files) < self.num_files:
            skipped = self.num_files - len(valid_files)
            _log(f"Dropped {skipped} file(s) with no usable frames")
            old_to_new = {old: new for new, old in enumerate(valid_files)}
            self.data_files = [self.data_files[i] for i in valid_files]
            frame_counts = [frame_counts[i] for i in valid_files]
            audio_widths = [audio_widths[i] for i in valid_files]
            audio_sample_rates = [audio_sample_rates[i] for i in valid_files]
            video_fps_values = [video_fps_values[i] for i in valid_files]
            frame_shapes = [frame_shapes[i] for i in valid_files]
            self.samples = [(old_to_new[file_idx], start_idx) for file_idx, start_idx in self.samples]
            if self.frames_by_file is not None:
                self.frames_by_file = [self.frames_by_file[i] for i in valid_files]
            if self.actions_by_file is not None:
                self.actions_by_file = [self.actions_by_file[i] for i in valid_files]
            if self.audio_by_file is not None:
                self.audio_by_file = [self.audio_by_file[i] for i in valid_files]
            if self.audio_lengths_by_file is not None:
                self.audio_lengths_by_file = [self.audio_lengths_by_file[i] for i in valid_files]
            if self.ram_by_file is not None:
                self.ram_by_file = [self.ram_by_file[i] for i in valid_files]
            self.num_files = len(self.data_files)

        _log(f"Indexing complete. Collected {len(self.samples):,} windows before filtering/sort.")
        _log("Sorting sample index...")
        sort_start = time.time()
        self.samples.sort()
        _log(f"Sorted {len(self.samples):,} windows in {time.time() - sort_start:.1f}s")

        if self.include_audio:
            widths = [width for width in audio_widths if width is not None]
            if widths:
                expected_width = widths[0]
                if any(width != expected_width for width in widths):
                    raise ValueError("Inconsistent audio frame width across normalized files")
                self.audio_frame_size = expected_width

            sample_rates = [sample_rate for sample_rate in audio_sample_rates if sample_rate is not None]
            if sample_rates:
                first_sample_rate = sample_rates[0]
                if any(sample_rate != first_sample_rate for sample_rate in sample_rates):
                    raise ValueError("Inconsistent audio sample rate across normalized files")
                self.audio_sample_rate = first_sample_rate

            self.audio_sample_rate_by_file = [
                sample_rate if sample_rate is not None else self.audio_sample_rate
                for sample_rate in audio_sample_rates
            ]
            self.video_fps_by_file = video_fps_values

        if subset_n > 0 and subset_n < len(self.samples):
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(self.samples), size=subset_n, replace=False)
            self.samples = [self.samples[int(idx)] for idx in indices]
            self.samples.sort()

        self.total_frames = sum(frame_counts)
        self.dataset_bytes = sum(path.stat().st_size for path in self.data_files)

        if self.total_frames > 0:
            total_bytes = 0
            if self.include_frames:
                shape_sample = next((shape for shape in frame_shapes if shape is not None), None)
                if shape_sample is None:
                    raise RuntimeError("Could not determine frame shape during indexing")
                # Normalized frames are palette indices stored as uint8.
                total_bytes += self.total_frames * int(math.prod(shape_sample))

            if self.include_actions:
                if self.actions_by_file is not None:
                    first_actions = next((arr for arr in self.actions_by_file if arr is not None), None)
                    if first_actions is not None:
                        total_bytes += self.total_frames * int(np.asarray(first_actions[0]).nbytes)
                else:
                    with np.load(self.data_files[0], mmap_mode="r") as probe_npz:
                        probe_actions = probe_npz["actions"]
                        total_bytes += self.total_frames * int(np.asarray(probe_actions[0]).nbytes)

            if self.include_audio:
                if self.audio_by_file is not None:
                    first_audio = next((arr for arr in self.audio_by_file if arr is not None), None)
                    if first_audio is not None:
                        total_bytes += self.total_frames * int(np.asarray(first_audio[0]).nbytes)
                else:
                    with np.load(self.data_files[0], mmap_mode="r") as probe_npz:
                        probe_audio = probe_npz["audio"]
                        bytes_per_audio_frame = int(math.prod(probe_audio.shape[1:]) * probe_audio.dtype.itemsize)
                        total_bytes += self.total_frames * bytes_per_audio_frame

                if self.audio_lengths_by_file is not None:
                    first_lengths = next((arr for arr in self.audio_lengths_by_file if arr is not None), None)
                    if first_lengths is not None:
                        total_bytes += self.total_frames * int(np.asarray(first_lengths[0]).nbytes)
                else:
                    with np.load(self.data_files[0], mmap_mode="r") as probe_npz:
                        if "audio_lengths" in probe_npz.files:
                            probe_lengths = probe_npz["audio_lengths"]
                            total_bytes += self.total_frames * int(np.asarray(probe_lengths[0]).nbytes)

            if self.include_ram:
                if self.ram_by_file is not None:
                    first_ram = next((arr for arr in self.ram_by_file if arr is not None), None)
                    if first_ram is not None:
                        self.ram_n_bytes = first_ram.shape[1]
                        total_bytes += self.total_frames * int(np.asarray(first_ram[0]).nbytes)
                else:
                    with np.load(self.data_files[0], mmap_mode="r") as probe_npz:
                        if "ram" in probe_npz.files:
                            probe_ram = probe_npz["ram"]
                            self.ram_n_bytes = probe_ram.shape[1]
                            total_bytes += self.total_frames * int(math.prod(probe_ram.shape[1:]) * probe_ram.dtype.itemsize)

            if available > 0 and total_bytes < available - headroom:
                if load_during_index and (
                    (not self.include_frames or self.frames_by_file is not None)
                    and (not self.include_actions or self.actions_by_file is not None)
                    and (not self.include_audio or (self.audio_by_file is not None and self.audio_lengths_by_file is not None))
                    and (not self.include_ram or self.ram_by_file is not None)
                ):
                    loaded_bytes = 0
                    if self.frames_by_file is not None:
                        loaded_bytes += int(sum(frames.nbytes for frames in self.frames_by_file if frames is not None))
                    if self.include_actions and self.actions_by_file is not None:
                        loaded_bytes += int(sum(actions.nbytes for actions in self.actions_by_file if actions is not None))
                    if self.include_audio and self.audio_by_file is not None and self.audio_lengths_by_file is not None:
                        loaded_bytes += int(sum(audio.nbytes for audio in self.audio_by_file if audio is not None))
                        loaded_bytes += int(sum(lengths.nbytes for lengths in self.audio_lengths_by_file if lengths is not None))
                    if self.include_ram and self.ram_by_file is not None:
                        loaded_bytes += int(sum(ram.nbytes for ram in self.ram_by_file if ram is not None))
                    _log(f"Loaded {loaded_bytes / 2**30:.1f} GB into RAM during indexing (single pass)")
                    return

                _log(
                    f"Loading dataset into RAM ({total_bytes / 2**30:.1f} GB, "
                    f"{available / 2**30:.1f} GB available)..."
                )

                if self.include_frames:
                    self.frames_by_file = [None] * self.num_files
                if self.include_actions:
                    self.actions_by_file = [None] * self.num_files
                if self.include_audio:
                    self.audio_by_file = [None] * self.num_files
                    self.audio_lengths_by_file = [None] * self.num_files
                if self.include_ram:
                    self.ram_by_file = [None] * self.num_files

                def _load_file(file_idx: int):
                    with np.load(self.data_files[file_idx]) as npz:
                        frames = np.asarray(npz["frames"]) if self.include_frames else None
                        actions = np.asarray(npz["actions"]) if self.include_actions else None
                        audio = np.asarray(npz["audio"]) if self.include_audio else None
                        ram = np.asarray(npz["ram"]) if self.include_ram and "ram" in npz.files else None
                        lengths = None
                        sample_rate = None
                        video_fps = None
                        if self.include_audio:
                            if "audio_lengths" in npz.files:
                                lengths = np.asarray(npz["audio_lengths"], dtype=np.int32)
                            else:
                                num_frames = audio.shape[0]
                                lengths = np.full((num_frames,), audio.shape[1], dtype=np.int32)
                            if "audio_sample_rate" in npz.files:
                                sample_rate = int(np.asarray(npz["audio_sample_rate"]).item())
                            if "video_fps" in npz.files:
                                video_fps = float(np.asarray(npz["video_fps"]).item())
                    return file_idx, frames, actions, audio, lengths, sample_rate, video_fps, ram

                load_start = time.time()
                last_load_report = load_start
                completed_load = 0
                with concurrent.futures.ThreadPoolExecutor(max_workers=index_workers) as pool:
                    futures = [pool.submit(_load_file, file_idx) for file_idx in range(self.num_files)]
                    if use_live_progress:
                        assert _DATASET_CONSOLE is not None
                        with Progress(
                            SpinnerColumn(style="green"),
                            TextColumn("[progress.description]{task.description}"),
                            BarColumn(bar_width=None),
                            TextColumn("[progress.percentage]{task.completed}/{task.total}"),
                            TextColumn("[green]{task.fields[rate]:>6.1f} files/s"),
                            TimeElapsedColumn(),
                            refresh_per_second=8,
                            console=_DATASET_CONSOLE,
                        ) as progress:
                            load_task = progress.add_task("Loading files into RAM", total=self.num_files, rate=0.0)
                            for future in concurrent.futures.as_completed(futures):
                                file_idx, frames, actions, audio, lengths, sample_rate, video_fps, ram = future.result()
                                if self.include_frames and self.frames_by_file is not None:
                                    self.frames_by_file[file_idx] = frames
                                if self.include_actions and self.actions_by_file is not None:
                                    self.actions_by_file[file_idx] = actions
                                if self.include_audio and self.audio_by_file is not None and self.audio_lengths_by_file is not None:
                                    self.audio_by_file[file_idx] = audio
                                    self.audio_lengths_by_file[file_idx] = lengths
                                    if sample_rate is not None:
                                        self.audio_sample_rate_by_file[file_idx] = sample_rate
                                    if video_fps is not None:
                                        self.video_fps_by_file[file_idx] = video_fps
                                if self.include_ram and self.ram_by_file is not None:
                                    self.ram_by_file[file_idx] = ram

                                completed_load += 1
                                elapsed = max(time.time() - load_start, 1e-9)
                                progress.update(load_task, advance=1, rate=completed_load / elapsed)
                    else:
                        for future in concurrent.futures.as_completed(futures):
                            file_idx, frames, actions, audio, lengths, sample_rate, video_fps = future.result()
                            if self.include_frames and self.frames_by_file is not None:
                                self.frames_by_file[file_idx] = frames
                            if self.include_actions and self.actions_by_file is not None:
                                self.actions_by_file[file_idx] = actions
                            if self.include_audio and self.audio_by_file is not None and self.audio_lengths_by_file is not None:
                                self.audio_by_file[file_idx] = audio
                                self.audio_lengths_by_file[file_idx] = lengths
                                if sample_rate is not None:
                                    self.audio_sample_rate_by_file[file_idx] = sample_rate
                                if video_fps is not None:
                                    self.video_fps_by_file[file_idx] = video_fps

                            completed_load += 1
                            now = time.time()
                            if completed_load == self.num_files or now - last_load_report >= 2.0:
                                elapsed = max(now - load_start, 1e-9)
                                rate = completed_load / elapsed
                                _log(
                                    f"  Loaded {completed_load}/{self.num_files} files into RAM "
                                    f"({rate:.1f} files/s)"
                                )
                                last_load_report = now

                loaded_bytes = 0
                if self.frames_by_file is not None:
                    loaded_bytes += int(sum(frames.nbytes for frames in self.frames_by_file if frames is not None))
                if self.include_actions and self.actions_by_file is not None:
                    loaded_bytes += int(sum(actions.nbytes for actions in self.actions_by_file if actions is not None))
                if self.include_audio and self.audio_by_file is not None and self.audio_lengths_by_file is not None:
                    loaded_bytes += int(sum(audio.nbytes for audio in self.audio_by_file if audio is not None))
                    loaded_bytes += int(sum(lengths.nbytes for lengths in self.audio_lengths_by_file if lengths is not None))
                if self.include_ram and self.ram_by_file is not None:
                    loaded_bytes += int(sum(ram.nbytes for ram in self.ram_by_file if ram is not None))
                _log(f"Loaded {loaded_bytes / 2**30:.1f} GB into RAM")
            else:
                if load_during_index:
                    self.frames_by_file = None
                    self.actions_by_file = None
                    self.audio_by_file = None
                    self.audio_lengths_by_file = None
                    self.ram_by_file = None
                _log(
                    f"Dataset too large for RAM ({total_bytes / 2**30:.1f} GB, "
                    f"{available / 2**30:.1f} GB available). Using mmap."
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __del__(self) -> None:
        for npz in getattr(self, "_npz_cache", {}).values():
            try:
                npz.close()
            except Exception:
                pass

    def _get_npz(self, file_idx: int) -> np.lib.npyio.NpzFile:
        if file_idx not in self._npz_cache:
            self._npz_cache[file_idx] = np.load(self.data_files[file_idx], mmap_mode="r")
        return self._npz_cache[file_idx]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_idx, start_idx = self.samples[idx]
        end_idx = start_idx + self.clip_frames

        npz = None
        sample: dict[str, Any] = {
            "file_idx": file_idx,
            "start_idx": start_idx,
        }

        if self.include_frames:
            if self.frames_by_file is not None:
                frames = self.frames_by_file[file_idx][start_idx:end_idx]
            else:
                npz = self._get_npz(file_idx)
                frames = np.asarray(npz["frames"][start_idx:end_idx])
            sample["frames"] = torch.from_numpy(np.asarray(frames).copy())

        if self.include_actions:
            if self.actions_by_file is not None:
                actions = self.actions_by_file[file_idx][start_idx:end_idx]
            else:
                if npz is None:
                    npz = self._get_npz(file_idx)
                actions = np.asarray(npz["actions"][start_idx:end_idx])
            sample["actions"] = torch.from_numpy(np.asarray(actions).copy())

        if self.include_audio:
            if self.audio_by_file is not None and self.audio_lengths_by_file is not None:
                audio = self.audio_by_file[file_idx][start_idx:end_idx]
                lengths = self.audio_lengths_by_file[file_idx][start_idx:end_idx]
                sample_rate = self.audio_sample_rate_by_file[file_idx]
                video_fps = self.video_fps_by_file[file_idx]
            else:
                if npz is None:
                    npz = self._get_npz(file_idx)
                audio = np.asarray(npz["audio"][start_idx:end_idx])
                if "audio_lengths" in npz.files:
                    lengths = np.asarray(npz["audio_lengths"][start_idx:end_idx], dtype=np.int32)
                else:
                    lengths = np.full((self.clip_frames,), audio.shape[1], dtype=np.int32)
                sample_rate = int(
                    np.asarray(npz["audio_sample_rate"]).item()
                    if "audio_sample_rate" in npz.files
                    else self.audio_sample_rate
                )
                video_fps = (
                    float(np.asarray(npz["video_fps"]).item())
                    if "video_fps" in npz.files
                    else None
                )

            sample["audio"] = torch.from_numpy(np.asarray(audio).copy())
            sample["audio_lengths"] = torch.from_numpy(np.asarray(lengths, dtype=np.int32).copy())
            sample["audio_sample_rate"] = int(sample_rate)
            if video_fps is not None:
                sample["video_fps"] = float(video_fps)

        if self.include_ram:
            if self.ram_by_file is not None:
                ram = self.ram_by_file[file_idx][start_idx:end_idx]
            else:
                if npz is None:
                    npz = self._get_npz(file_idx)
                ram = np.asarray(npz["ram"][start_idx:end_idx])
            sample["ram"] = torch.from_numpy(np.asarray(ram).copy())

        return sample