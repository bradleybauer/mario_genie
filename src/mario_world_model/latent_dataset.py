from __future__ import annotations

import concurrent.futures
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

from mario_world_model.system_info import get_available_memory, get_effective_cpu_count


_DATASET_CONSOLE = Console() if Console is not None else None


def _log(message: str) -> None:
    if _DATASET_CONSOLE is not None:
        _DATASET_CONSOLE.print(message)
    else:
        print(message)


def _index_latent_file(
    file_idx: int,
    path: Path,
    *,
    clip_frames: int,
    stride: int,
    include_actions: bool,
    load_arrays: bool,
) -> tuple[
    int,
    int,
    list[tuple[int, int]],
    tuple[int, int, int] | None,
    str | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    try:
        mmap_mode = None if load_arrays else "r"
        with np.load(path, mmap_mode=mmap_mode) as npz:
            if "latents" not in npz.files:
                return file_idx, 0, [], None, f"{path.name} is missing the 'latents' array", None, None

            latents_arr = np.asarray(npz["latents"]) if load_arrays else npz["latents"]
            if latents_arr.ndim != 4:
                return (
                    file_idx,
                    0,
                    [],
                    None,
                    f"{path.name} has invalid latents shape {tuple(latents_arr.shape)}; expected (C, T, H, W)",
                    None,
                    None,
                )

            frame_count = int(latents_arr.shape[1])
            latent_shape = tuple(int(dim) for dim in (latents_arr.shape[0], latents_arr.shape[2], latents_arr.shape[3]))
            loaded_latents = np.asarray(latents_arr) if load_arrays else None

            loaded_actions: np.ndarray | None = None
            if include_actions:
                if "actions" not in npz.files:
                    return file_idx, 0, [], latent_shape, f"{path.name} is missing the 'actions' array", loaded_latents, None

                actions_arr = np.asarray(npz["actions"]) if load_arrays else npz["actions"]
                if int(actions_arr.shape[0]) != frame_count:
                    return (
                        file_idx,
                        0,
                        [],
                        latent_shape,
                        f"{path.name} has {frame_count} latent steps but {actions_arr.shape[0]} actions",
                        loaded_latents,
                        None,
                    )
                if load_arrays:
                    loaded_actions = np.asarray(actions_arr)

            if frame_count < clip_frames:
                return file_idx, 0, [], latent_shape, None, loaded_latents, loaded_actions

            samples = [(file_idx, start_idx) for start_idx in range(0, frame_count - clip_frames + 1, stride)]
            return file_idx, frame_count, samples, latent_shape, None, loaded_latents, loaded_actions
    except Exception as exc:
        return file_idx, 0, [], None, f"{path.name}: {exc}", None, None


class LatentSequenceDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        clip_frames: int,
        *,
        include_actions: bool = False,
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
        self.include_actions = include_actions
        self.stride = stride
        self.data_files = sorted(self.data_dir.glob("*.npz"))
        self.samples: list[tuple[int, int]] = []
        self._npz_cache: dict[int, np.lib.npyio.NpzFile] = {}
        self.latents_by_file: list[np.ndarray] | None = None
        self.actions_by_file: list[np.ndarray] | None = None
        self.latent_channels = 0
        self.latent_height = 0
        self.latent_width = 0
        self.num_files = len(self.data_files)
        self.total_frames = 0
        self.dataset_bytes = sum(path.stat().st_size for path in self.data_files)

        if not self.data_files:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")

        index_errors: list[str] = []
        frame_counts = [0] * self.num_files
        latent_shapes: list[tuple[int, int, int] | None] = [None] * self.num_files

        available = get_available_memory(system_info)
        headroom = 2 * 2**30
        load_during_index = (
            available > 0
            and self.dataset_bytes < max(available - headroom, 0) * 0.5
        )
        if load_during_index:
            self.latents_by_file = [None] * self.num_files
            if self.include_actions:
                self.actions_by_file = [None] * self.num_files

        _log(f"[dataset] Indexing {self.num_files:,} latent files (stride={self.stride})...")
        index_start = time.time()
        last_index_report = index_start
        completed_index = 0
        use_live_progress = Progress is not None and sys.stdout.isatty()
        with concurrent.futures.ThreadPoolExecutor(max_workers=index_workers) as pool:
            futures = {
                pool.submit(
                    _index_latent_file,
                    file_idx,
                    path,
                    clip_frames=self.clip_frames,
                    stride=self.stride,
                    include_actions=self.include_actions,
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
                    index_task = progress.add_task("Indexing latent files", total=self.num_files, rate=0.0)
                    for future in concurrent.futures.as_completed(futures):
                        file_idx, frame_count, samples, latent_shape, error, loaded_latents, loaded_actions = future.result()
                        if error is not None:
                            index_errors.append(error)
                        else:
                            frame_counts[file_idx] = frame_count
                            self.samples.extend(samples)
                            latent_shapes[file_idx] = latent_shape
                            if load_during_index and self.latents_by_file is not None:
                                self.latents_by_file[file_idx] = loaded_latents
                            if load_during_index and self.include_actions and self.actions_by_file is not None:
                                self.actions_by_file[file_idx] = loaded_actions

                        completed_index += 1
                        elapsed = max(time.time() - index_start, 1e-9)
                        progress.update(index_task, advance=1, rate=completed_index / elapsed)
            else:
                for future in concurrent.futures.as_completed(futures):
                    file_idx, frame_count, samples, latent_shape, error, loaded_latents, loaded_actions = future.result()
                    if error is not None:
                        index_errors.append(error)
                    else:
                        frame_counts[file_idx] = frame_count
                        self.samples.extend(samples)
                        latent_shapes[file_idx] = latent_shape
                        if load_during_index and self.latents_by_file is not None:
                            self.latents_by_file[file_idx] = loaded_latents
                        if load_during_index and self.include_actions and self.actions_by_file is not None:
                            self.actions_by_file[file_idx] = loaded_actions

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
            raise ValueError(f"Failed to index latent dataset: {preview}")

        valid_files = [i for i, frame_count in enumerate(frame_counts) if frame_count > 0]
        if len(valid_files) < self.num_files:
            skipped = self.num_files - len(valid_files)
            _log(f"Dropped {skipped} latent file(s) with no usable frames")
            old_to_new = {old: new for new, old in enumerate(valid_files)}
            self.data_files = [self.data_files[i] for i in valid_files]
            frame_counts = [frame_counts[i] for i in valid_files]
            latent_shapes = [latent_shapes[i] for i in valid_files]
            self.samples = [(old_to_new[file_idx], start_idx) for file_idx, start_idx in self.samples]
            if self.latents_by_file is not None:
                self.latents_by_file = [self.latents_by_file[i] for i in valid_files]
            if self.actions_by_file is not None:
                self.actions_by_file = [self.actions_by_file[i] for i in valid_files]
            self.num_files = len(self.data_files)

        _log(f"Indexing complete. Collected {len(self.samples):,} latent windows before filtering/sort.")
        _log("Sorting sample index...")
        sort_start = time.time()
        self.samples.sort()
        _log(f"Sorted {len(self.samples):,} windows in {time.time() - sort_start:.1f}s")

        valid_shapes = [shape for shape in latent_shapes if shape is not None]
        if not valid_shapes:
            raise RuntimeError("Could not determine latent shape during indexing")
        first_shape = valid_shapes[0]
        if any(shape != first_shape for shape in valid_shapes):
            raise ValueError("Inconsistent latent channel/spatial shape across latent files")
        self.latent_channels, self.latent_height, self.latent_width = first_shape

        if subset_n > 0 and subset_n < len(self.samples):
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(self.samples), size=subset_n, replace=False)
            self.samples = [self.samples[int(idx)] for idx in indices]
            self.samples.sort()

        self.total_frames = sum(frame_counts)
        self.dataset_bytes = sum(path.stat().st_size for path in self.data_files)

        if self.total_frames > 0:
            total_bytes = self.total_frames * self.latent_channels * self.latent_height * self.latent_width * np.dtype(np.float16).itemsize
            if self.include_actions:
                if self.actions_by_file is not None:
                    first_actions = next((arr for arr in self.actions_by_file if arr is not None), None)
                    if first_actions is not None:
                        total_bytes += self.total_frames * int(np.asarray(first_actions[0]).nbytes)
                else:
                    with np.load(self.data_files[0], mmap_mode="r") as probe_npz:
                        probe_actions = probe_npz["actions"]
                        total_bytes += self.total_frames * int(np.asarray(probe_actions[0]).nbytes)

            if available > 0 and total_bytes < available - headroom:
                if load_during_index and self.latents_by_file is not None and (not self.include_actions or self.actions_by_file is not None):
                    loaded_bytes = int(sum(latents.nbytes for latents in self.latents_by_file if latents is not None))
                    if self.include_actions and self.actions_by_file is not None:
                        loaded_bytes += int(sum(actions.nbytes for actions in self.actions_by_file if actions is not None))
                    _log(f"Loaded {loaded_bytes / 2**30:.1f} GB into RAM during indexing (single pass)")
                    return

                _log(
                    f"Loading latent dataset into RAM ({total_bytes / 2**30:.1f} GB, "
                    f"{available / 2**30:.1f} GB available)..."
                )

                self.latents_by_file = [None] * self.num_files
                if self.include_actions:
                    self.actions_by_file = [None] * self.num_files

                def _load_file(file_idx: int):
                    with np.load(self.data_files[file_idx]) as npz:
                        latents = np.asarray(npz["latents"])
                        actions = np.asarray(npz["actions"]) if self.include_actions else None
                    return file_idx, latents, actions

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
                            load_task = progress.add_task("Loading latent files into RAM", total=self.num_files, rate=0.0)
                            for future in concurrent.futures.as_completed(futures):
                                file_idx, latents, actions = future.result()
                                assert self.latents_by_file is not None
                                self.latents_by_file[file_idx] = latents
                                if self.include_actions and self.actions_by_file is not None:
                                    self.actions_by_file[file_idx] = actions

                                completed_load += 1
                                elapsed = max(time.time() - load_start, 1e-9)
                                progress.update(load_task, advance=1, rate=completed_load / elapsed)
                    else:
                        for future in concurrent.futures.as_completed(futures):
                            file_idx, latents, actions = future.result()
                            assert self.latents_by_file is not None
                            self.latents_by_file[file_idx] = latents
                            if self.include_actions and self.actions_by_file is not None:
                                self.actions_by_file[file_idx] = actions

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

                loaded_bytes = int(sum(latents.nbytes for latents in self.latents_by_file if latents is not None))
                if self.include_actions and self.actions_by_file is not None:
                    loaded_bytes += int(sum(actions.nbytes for actions in self.actions_by_file if actions is not None))
                _log(f"Loaded {loaded_bytes / 2**30:.1f} GB into RAM")
            else:
                if load_during_index:
                    self.latents_by_file = None
                    self.actions_by_file = None
                _log(
                    f"Latent dataset too large for RAM ({total_bytes / 2**30:.1f} GB, "
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

        sample: dict[str, Any] = {
            "file_idx": file_idx,
            "start_idx": start_idx,
        }

        npz = None
        if self.latents_by_file is not None:
            latents = self.latents_by_file[file_idx][:, start_idx:end_idx]
        else:
            npz = self._get_npz(file_idx)
            latents = np.asarray(npz["latents"][:, start_idx:end_idx])
        sample["latents"] = torch.from_numpy(np.asarray(latents).copy())

        if self.include_actions:
            if self.actions_by_file is not None:
                actions = self.actions_by_file[file_idx][start_idx:end_idx]
            else:
                if npz is None:
                    npz = self._get_npz(file_idx)
                actions = np.asarray(npz["actions"][start_idx:end_idx])
            sample["actions"] = torch.from_numpy(np.asarray(actions).copy())

        return sample