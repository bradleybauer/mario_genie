#!/usr/bin/env python3
"""Interactive 2D visualization of precomputed VideoVAE latents.

The script reads existing latent `.npz` files, embeds sampled latent timesteps
to two dimensions, and shows the closest sample when hovering over the scatter.
The preview panel maps the latent back to the matching normalized recording and,
when a VideoVAE checkpoint is available, decodes the corresponding latent window.

Usage:
    python scripts/eval/visualize_latent_embedding.py --latent-dir data/latents
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.data.normalized_dataset import load_palette_tensor
from src.data.video_frames import resize_palette_frames
from src.path_utils import resolve_workspace_path
from src.plot_style import apply_plot_style
from src.system_info import get_available_memory

apply_plot_style()


@dataclass(frozen=True)
class LatentPoint:
    file_idx: int
    latent_idx: int


@dataclass(frozen=True)
class PreviewImage:
    rgb: np.ndarray
    title: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed existing latent dataset timesteps and inspect nearest samples on hover."
    )
    parser.add_argument("--latent-dir", type=Path, default=Path("data/latents"))
    parser.add_argument("--normalized-dir", type=Path, default=None,
                        help="Directory containing normalized .npz files. Defaults to latent_config.json source_data_dir.")
    parser.add_argument("--max-points", type=int, default=8000,
                        help="Maximum latent timesteps to embed.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-method", choices=["pca", "umap"], default="pca")
    parser.add_argument("--pca-dims", type=int, default=32,
                        help="Intermediate PCA dimensions before UMAP.")
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True,
                        help="Standardize latent components with latent_stats.json for embedding only.")
    parser.add_argument("--point-size", type=float, default=6.0)
    parser.add_argument("--hover-pixels", type=float, default=18.0,
                        help="Refresh preview only when the nearest point is within this many screen pixels.")
    parser.add_argument("--ram-budget-gb", type=float, default=10.0,
                        help="Approximate total RAM budget for this process; used to decide whether preview images are preloaded.")
    parser.add_argument("--preview-cache-fraction", type=float, default=0.5,
                        help="Maximum fraction of the RAM budget reserved for cached preview images.")
    parser.add_argument("--max-preload-files", type=int, default=0,
                        help="Only preload when sampled points touch at most this many files (0 = no file-count limit).")
    parser.add_argument("--hover-min-ms", type=float, default=16.0,
                        help="Minimum milliseconds between hover updates.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def resolve_optional_path(value: str | Path | None, *, config_dir: Path | None = None) -> Path | None:
    if value is None:
        return None
    return resolve_workspace_path(str(value), project_root=PROJECT_ROOT, config_dir=config_dir)


def load_latent_config(latent_dir: Path) -> dict[str, Any]:
    path = latent_dir / "latent_config.json"
    if not path.is_file():
        return {}
    return load_json(path)


def latent_npz_files(latent_dir: Path) -> list[Path]:
    files = sorted(path for path in latent_dir.glob("*.npz") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No latent .npz files found in {latent_dir}")
    return files


def validate_matching_normalized_files(latent_files: list[Path], normalized_dir: Path) -> list[Path]:
    normalized_files = [normalized_dir / path.name for path in latent_files]
    missing = [path.name for path in normalized_files if not path.is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} normalized file(s) in {normalized_dir}: {preview}{suffix}"
        )
    return normalized_files


def load_latent_stats(latent_dir: Path, latent_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    stats_path = resolve_optional_path(latent_config.get("latent_stats_path"), config_dir=latent_dir)
    if stats_path is None or not stats_path.is_file():
        fallback = latent_dir / str(latent_config.get("latent_stats_file", "latent_stats.json"))
        stats_path = fallback if fallback.is_file() else None
    if stats_path is None:
        return None

    stats = load_json(stats_path)
    if stats.get("latent_stats_version") != 2:
        return None
    mean = np.asarray(stats["component_mean"], dtype=np.float32)
    std = np.asarray(stats["component_std_clamped"], dtype=np.float32)
    return mean, std


def latent_counts_and_shape(latent_files: list[Path]) -> tuple[np.ndarray, tuple[int, int, int]]:
    counts: list[int] = []
    latent_shape: tuple[int, int, int] | None = None
    for path in latent_files:
        with np.load(path, mmap_mode="r") as data:
            if "latents" not in data.files:
                raise ValueError(f"{path.name} is missing the 'latents' array")
            latents = data["latents"]
            if latents.ndim != 4:
                raise ValueError(f"{path.name} has invalid latents shape {tuple(latents.shape)}")
            current_shape = (int(latents.shape[0]), int(latents.shape[2]), int(latents.shape[3]))
            if latent_shape is None:
                latent_shape = current_shape
            elif current_shape != latent_shape:
                raise ValueError(f"{path.name} has latent shape {current_shape}; expected {latent_shape}")
            counts.append(int(latents.shape[1]))
    if latent_shape is None:
        raise RuntimeError("Could not determine latent shape")
    return np.asarray(counts, dtype=np.int64), latent_shape


def sample_latent_points(counts: np.ndarray, max_points: int, seed: int) -> list[LatentPoint]:
    if max_points <= 0:
        raise ValueError("--max-points must be positive")
    total = int(counts.sum())
    if total <= 0:
        raise RuntimeError("Latent dataset contains no timesteps")

    size = min(int(max_points), total)
    rng = np.random.default_rng(seed)
    flat_indices = np.sort(rng.choice(total, size=size, replace=False))
    cumulative = np.cumsum(counts)
    file_indices = np.searchsorted(cumulative, flat_indices, side="right")
    file_starts = np.concatenate([np.asarray([0], dtype=np.int64), cumulative[:-1]])

    return [
        LatentPoint(file_idx=int(file_idx), latent_idx=int(flat_idx - file_starts[file_idx]))
        for flat_idx, file_idx in zip(flat_indices, file_indices)
    ]


def load_feature_matrix(
    latent_files: list[Path],
    points: list[LatentPoint],
    stats: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    by_file: dict[int, list[tuple[int, int]]] = {}
    for output_idx, point in enumerate(points):
        by_file.setdefault(point.file_idx, []).append((output_idx, point.latent_idx))

    first = True
    features: np.ndarray | None = None
    latent_norms = np.empty((len(points),), dtype=np.float32)
    component_mean: np.ndarray | None = None
    component_std: np.ndarray | None = None
    if stats is not None:
        component_mean, component_std = stats

    for file_idx, entries in by_file.items():
        entries.sort(key=lambda item: item[1])
        output_indices = [item[0] for item in entries]
        latent_indices = [item[1] for item in entries]
        with np.load(latent_files[file_idx], mmap_mode="r") as data:
            chunk = np.asarray(data["latents"][:, latent_indices], dtype=np.float32)
        chunk = np.moveaxis(chunk, 1, 0)
        raw_chunk_features = chunk.reshape(chunk.shape[0], -1)
        latent_norms[output_indices] = np.linalg.norm(raw_chunk_features, axis=1)
        if component_mean is not None and component_std is not None:
            chunk = (chunk - component_mean[None, :, :, :]) / component_std[None, :, :, :]
        chunk_features = chunk.reshape(chunk.shape[0], -1)
        if first:
            features = np.empty((len(points), chunk_features.shape[1]), dtype=np.float32)
            first = False
        assert features is not None
        features[output_indices] = chunk_features

    if features is None:
        raise RuntimeError("No features were loaded")
    return features, latent_norms


def pca_project(features: np.ndarray, dims: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dims <= 0:
        raise ValueError("PCA dims must be positive")
    centered = features.astype(np.float32, copy=True)
    mean = centered.mean(axis=0, keepdims=True)
    centered -= mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    effective_dims = min(dims, vt.shape[0])
    projected = centered @ vt[:effective_dims].T
    return projected.astype(np.float32, copy=False), mean.squeeze(0), vt[:effective_dims]


def embed_features(features: np.ndarray, method: str, pca_dims: int, seed: int) -> np.ndarray:
    if len(features) < 2:
        raise ValueError("Need at least two points to build a 2D embedding")
    if method == "pca":
        coords, _, _ = pca_project(features, 2)
        return coords

    pre_dims = min(max(2, pca_dims), features.shape[0] - 1, features.shape[1])
    reduced, _, _ = pca_project(features, pre_dims)
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "UMAP is not installed in this environment. Use --embedding-method pca "
            "or install umap-learn."
        ) from exc
    reducer = umap.UMAP(n_components=2, random_state=seed, metric="euclidean")
    return np.asarray(reducer.fit_transform(reduced), dtype=np.float32)


def palette_to_rgb(frames: np.ndarray, palette_u8: np.ndarray) -> np.ndarray:
    return palette_u8[np.asarray(frames, dtype=np.int64)]


def format_bytes(byte_count: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(byte_count)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


class Previewer:
    def __init__(
        self,
        *,
        normalized_files: list[Path],
        points: list[LatentPoint],
        coords: np.ndarray,
        palette_u8: np.ndarray,
        latent_stride: int,
        latent_counts: np.ndarray,
        preview_height: int,
        preview_width: int,
        hover_pixels: float,
        ram_budget_gb: float,
        preview_cache_fraction: float,
        max_preload_files: int,
    ) -> None:
        self.normalized_files = normalized_files
        self.points = points
        self.coords = coords
        self.palette_u8 = palette_u8
        self.latent_stride = latent_stride
        self.latent_counts = latent_counts
        self.preview_height = int(preview_height)
        self.preview_width = int(preview_width)
        self.hover_pixels = max(0.0, float(hover_pixels))
        self._normalized_cache: dict[int, np.lib.npyio.NpzFile] = {}
        self._preview_cache: OrderedDict[tuple[int, int], PreviewImage] = OrderedDict()
        self._preview_cache_bytes = 0
        self._preview_bytes_per_image = self.preview_height * self.preview_width * 3
        self._max_preload_files = max(0, int(max_preload_files))
        self._display_coords: np.ndarray | None = None
        available_memory = int(get_available_memory() or 0)
        requested_budget = max(0, int(float(ram_budget_gb) * (1024 ** 3)))
        fraction = min(max(float(preview_cache_fraction), 0.0), 1.0)
        budget_from_request = int(requested_budget * fraction)
        budget_from_available = int(available_memory * 0.5) if available_memory > 0 else budget_from_request
        candidates = [value for value in (budget_from_request, budget_from_available) if value > 0]
        self._preview_cache_limit_bytes = min(candidates) if candidates else 0

        estimated_total_bytes = len(self.points) * self._preview_bytes_per_image
        unique_file_count = len({point.file_idx for point in self.points})
        print(
            f"Preview images: {len(self.points):,} x {self.preview_height}x{self.preview_width} RGB "
            f"(~{format_bytes(estimated_total_bytes)})"
        )
        print(f"Preview source files touched: {unique_file_count}")
        if self._preview_cache_limit_bytes > 0:
            print(f"Preview cache budget: {format_bytes(self._preview_cache_limit_bytes)}")
        if (
            self._preview_cache_limit_bytes > 0
            and estimated_total_bytes <= self._preview_cache_limit_bytes
            and (self._max_preload_files <= 0 or unique_file_count <= self._max_preload_files)
        ):
            self._preload_all_previews()
        else:
            reasons = []
            if self._preview_cache_limit_bytes <= 0:
                reasons.append("no cache budget")
            elif estimated_total_bytes > self._preview_cache_limit_bytes:
                reasons.append("cache estimate exceeds budget")
            if self._max_preload_files > 0 and unique_file_count > self._max_preload_files:
                reasons.append(f"touches too many files ({unique_file_count} > {self._max_preload_files})")
            reason_text = ", ".join(reasons) if reasons else "heuristic disabled"
            print(f"Preview preload disabled; {reason_text}. Images will load on demand within the cache budget.")

    def close(self) -> None:
        for npz in self._normalized_cache.values():
            npz.close()

    def normalized_npz(self, file_idx: int) -> np.lib.npyio.NpzFile:
        if file_idx not in self._normalized_cache:
            self._normalized_cache[file_idx] = np.load(self.normalized_files[file_idx], mmap_mode="r")
        return self._normalized_cache[file_idx]

    def _frame_index_for_point(self, point: LatentPoint) -> int:
        frame_idx = int(point.latent_idx) * self.latent_stride
        max_frame_idx = max(0, int(self.latent_counts[point.file_idx]) * self.latent_stride - 1)
        return min(frame_idx, max_frame_idx)

    def _load_preview_image(self, point: LatentPoint) -> PreviewImage:
        frame_idx = self._frame_index_for_point(point)
        npz = self.normalized_npz(point.file_idx)
        frame = np.asarray(npz["frames"][frame_idx], dtype=np.uint8)
        frame = resize_palette_frames(
            frame,
            target_height=self.preview_height,
            target_width=self.preview_width,
        )
        rgb = palette_to_rgb(frame, self.palette_u8)
        title = (
            f"{self.normalized_files[point.file_idx].name} | latent {point.latent_idx} | frame {frame_idx}"
        )
        return PreviewImage(rgb=rgb, title=title)

    def cache_key(self, point_idx: int) -> tuple[int, int]:
        point = self.points[point_idx]
        return point.file_idx, point.latent_idx

    def _insert_preview(self, cache_key: tuple[int, int], preview: PreviewImage) -> None:
        existing = self._preview_cache.pop(cache_key, None)
        if existing is not None:
            self._preview_cache_bytes -= int(existing.rgb.nbytes)
        self._preview_cache[cache_key] = preview
        self._preview_cache_bytes += int(preview.rgb.nbytes)
        while self._preview_cache_limit_bytes > 0 and self._preview_cache_bytes > self._preview_cache_limit_bytes:
            _, evicted = self._preview_cache.popitem(last=False)
            self._preview_cache_bytes -= int(evicted.rgb.nbytes)

    def _preload_all_previews(self) -> None:
        print("Preloading preview images...")
        by_file: dict[int, list[tuple[int, int]]] = {}
        for point_idx, point in enumerate(self.points):
            by_file.setdefault(point.file_idx, []).append((point_idx, self._frame_index_for_point(point)))

        for completed, (file_idx, entries) in enumerate(by_file.items(), start=1):
            entries.sort(key=lambda item: item[1])
            point_indices = [item[0] for item in entries]
            frame_indices = [item[1] for item in entries]
            npz = self.normalized_npz(file_idx)
            frames_array = np.asarray(npz["frames"], dtype=np.uint8)
            safe_indices = np.minimum(np.asarray(frame_indices, dtype=np.int64), frames_array.shape[0] - 1)
            unique_indices, inverse = np.unique(safe_indices, return_inverse=True)
            frames = frames_array[unique_indices]
            frames = resize_palette_frames(
                frames,
                target_height=self.preview_height,
                target_width=self.preview_width,
            )
            rgbs = palette_to_rgb(frames, self.palette_u8)
            for output_idx, frame_idx, unique_offset in zip(point_indices, safe_indices, inverse, strict=True):
                cache_key = self.cache_key(output_idx)
                point = self.points[output_idx]
                preview = PreviewImage(
                    rgb=np.ascontiguousarray(rgbs[int(unique_offset)]),
                    title=(
                        f"{self.normalized_files[point.file_idx].name} | latent {point.latent_idx} | "
                        f"frame {int(frame_idx)}"
                    ),
                )
                self._insert_preview(cache_key, preview)
            if completed == len(by_file) or completed % 10 == 0:
                print(f"  Preloaded previews from {completed}/{len(by_file)} files")
        print(f"Preloaded {len(self._preview_cache):,} preview images.")

    def preview(self, point_idx: int) -> PreviewImage:
        cache_key = self.cache_key(point_idx)
        preview = self._preview_cache.get(cache_key)
        if preview is not None:
            self._preview_cache.move_to_end(cache_key)
            return preview

        point = self.points[point_idx]
        preview = self._load_preview_image(point)
        self._insert_preview(cache_key, preview)
        return preview

    def set_display_coords(self, display_coords: np.ndarray) -> None:
        self._display_coords = np.asarray(display_coords, dtype=np.float32)

    def closest_point(self, event: Any) -> int | None:
        if self._display_coords is None or event.x is None or event.y is None:
            return None
        dx = self._display_coords[:, 0] - np.float32(event.x)
        dy = self._display_coords[:, 1] - np.float32(event.y)
        d2 = dx * dx + dy * dy
        idx = int(np.argmin(d2))
        if self.hover_pixels > 0 and float(d2[idx]) > self.hover_pixels * self.hover_pixels:
            return None
        return idx


def run_interactive_plot(
    coords: np.ndarray,
    color_values: np.ndarray,
    previewer: Previewer,
    point_size: float,
    hover_min_ms: float,
) -> None:
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=(3.6, 2.0))
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_preview = fig.add_subplot(gs[0, 1])

    scatter = ax_scatter.scatter(
        coords[:, 0], coords[:, 1], c=color_values, s=point_size, alpha=0.7, cmap="viridis", linewidths=0
    )
    selected = ax_scatter.scatter([], [], s=80, facecolors="none", edgecolors="red", linewidths=1.5)
    fig.colorbar(scatter, ax=ax_scatter, shrink=0.8, label="latent L2 norm")
    ax_scatter.set_title("Latent Embedding")
    ax_scatter.set_xlabel("dim 1")
    ax_scatter.set_ylabel("dim 2")
    ax_scatter.grid(True, alpha=0.25)

    ax_preview.set_xticks([])
    ax_preview.set_yticks([])
    ax_preview.set_title("Preview")

    state = {"idx": None}
    last_hover_ms = 0.0

    def refresh_display_coords() -> None:
        previewer.set_display_coords(ax_scatter.transData.transform(coords))

    def update_preview(idx: int) -> None:
        if state["idx"] == idx:
            return
        state["idx"] = idx
        preview = previewer.preview(idx)
        selected.set_offsets(coords[idx:idx + 1])

        ax_preview.clear()
        ax_preview.imshow(preview.rgb, interpolation="nearest")
        ax_preview.set_title(preview.title, fontsize=9)
        ax_preview.set_xticks([])
        ax_preview.set_yticks([])
        fig.canvas.draw_idle()

    def on_motion(event: Any) -> None:
        nonlocal last_hover_ms
        if event.inaxes != ax_scatter:
            return
        now_ms = event.guiEvent.time if getattr(event, "guiEvent", None) is not None and hasattr(event.guiEvent, "time") else None
        if now_ms is None:
            import time
            now_ms = time.monotonic() * 1000.0
        if now_ms - last_hover_ms < hover_min_ms:
            return
        last_hover_ms = float(now_ms)
        idx = previewer.closest_point(event)
        if idx is not None:
            update_preview(idx)

    fig.canvas.draw()
    refresh_display_coords()
    update_preview(0)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("draw_event", lambda _event: refresh_display_coords())
    ax_scatter.callbacks.connect("xlim_changed", lambda _axis: refresh_display_coords())
    ax_scatter.callbacks.connect("ylim_changed", lambda _axis: refresh_display_coords())
    fig.suptitle("Hover the embedding to inspect the nearest latent timestep", fontweight="bold")
    fig.tight_layout()
    try:
        plt.show()
    finally:
        previewer.close()


def main() -> None:
    args = parse_args()
    latent_dir = args.latent_dir.resolve()
    latent_config = load_latent_config(latent_dir)

    normalized_dir = args.normalized_dir
    if normalized_dir is None:
        normalized_dir = resolve_optional_path(latent_config.get("source_data_dir"), config_dir=latent_dir)
    if normalized_dir is None:
        normalized_dir = PROJECT_ROOT / "data" / "normalized"
    normalized_dir = normalized_dir.resolve()

    latent_files = latent_npz_files(latent_dir)
    normalized_files = validate_matching_normalized_files(latent_files, normalized_dir)
    counts, latent_shape = latent_counts_and_shape(latent_files)
    points = sample_latent_points(counts, max_points=args.max_points, seed=args.seed)
    print(
        f"Loaded index for {len(latent_files)} files, {int(counts.sum()):,} latent timesteps; "
        f"embedding {len(points):,} sampled points with latent shape {latent_shape}."
    )

    stats = load_latent_stats(latent_dir, latent_config) if args.standardize else None
    features, latent_norms = load_feature_matrix(latent_files, points, stats)
    coords = embed_features(features, args.embedding_method, args.pca_dims, args.seed)
    del features

    palette = load_palette_tensor(latent_dir if (latent_dir / "palette.json").is_file() else normalized_dir)
    palette_u8 = (palette.clamp(0, 1).numpy() * 255).astype(np.uint8)

    latent_stride = int(latent_config.get("latent_temporal_stride", 2 ** int(latent_config.get("temporal_downsample", 0))))
    frame_height = int(latent_config.get("frame_height", latent_config.get("source_frame_height", 224)))
    frame_width = int(latent_config.get("frame_width", latent_config.get("source_frame_width", frame_height)))
    preview_height = max(1, frame_height // 2)
    preview_width = max(1, frame_width // 2)
    print(f"Preview resolution: {preview_height}x{preview_width}")

    previewer = Previewer(
        normalized_files=normalized_files,
        points=points,
        coords=coords,
        palette_u8=palette_u8,
        latent_stride=latent_stride,
        latent_counts=counts,
        preview_height=preview_height,
        preview_width=preview_width,
        hover_pixels=args.hover_pixels,
        ram_budget_gb=args.ram_budget_gb,
        preview_cache_fraction=args.preview_cache_fraction,
        max_preload_files=args.max_preload_files,
    )
    run_interactive_plot(coords, latent_norms, previewer, point_size=args.point_size, hover_min_ms=args.hover_min_ms)


if __name__ == "__main__":
    main()
