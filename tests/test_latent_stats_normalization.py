from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_compute_latent_component_stats_schema(tmp_path: Path) -> None:
    encode_latents = _load_module(
        "encode_latents_script", PROJECT_ROOT / "scripts" / "collect" / "encode_latents.py"
    )

    latents_a = np.arange(2 * 3 * 2 * 2, dtype=np.float16).reshape(2, 3, 2, 2)
    latents_b = (latents_a + 1).astype(np.float16)
    actions = np.zeros((3,), dtype=np.uint8)

    np.savez_compressed(tmp_path / "a.npz", latents=latents_a, actions=actions)
    np.savez_compressed(tmp_path / "b.npz", latents=latents_b, actions=actions)

    stats = encode_latents.compute_latent_component_stats(sorted(tmp_path.glob("*.npz")))

    assert stats["latent_stats_version"] == 2
    assert stats["normalization_scheme"] == "component_chw_shared_time"
    assert stats["latent_shape"] == {"channels": 2, "height": 2, "width": 2}
    assert stats["count_per_component"] == 6

    component_mean = np.asarray(stats["component_mean"], dtype=np.float32)
    component_std = np.asarray(stats["component_std"], dtype=np.float32)
    component_std_clamped = np.asarray(stats["component_std_clamped"], dtype=np.float32)

    assert component_mean.shape == (2, 2, 2)
    assert component_std.shape == (2, 2, 2)
    assert component_std_clamped.shape == (2, 2, 2)
    assert np.isfinite(component_mean).all()
    assert np.isfinite(component_std).all()
    assert np.isfinite(component_std_clamped).all()
    assert float(component_std_clamped.min()) >= float(stats["std_epsilon"])

    # Old channel-only keys should not be present in the new strict schema.
    assert "channel_mean" not in stats
    assert "channel_std" not in stats
    assert "channel_std_clamped" not in stats


def test_load_latent_normalization_requires_component_schema(tmp_path: Path) -> None:
    pytest.importorskip("diffusers")
    pytest.importorskip("accelerate")

    train_dit = _load_module(
        "train_video_latent_dit_script",
        PROJECT_ROOT / "scripts" / "train" / "train_video_latent_dit.py",
    )

    old_schema = {
        "channel_mean": [0.0, 0.0],
        "channel_std": [1.0, 1.0],
        "std_epsilon": 1e-6,
    }
    old_stats_path = tmp_path / "old_stats.json"
    with old_stats_path.open("w") as f:
        json.dump(old_schema, f)

    with pytest.raises(ValueError, match="Channel-only latent stats are no longer supported"):
        train_dit.load_latent_normalization(
            stats_path=old_stats_path,
            latent_channels=2,
            latent_height=2,
            latent_width=2,
            device=torch.device("cpu"),
        )

    new_schema = {
        "latent_stats_version": 2,
        "normalization_scheme": "component_chw_shared_time",
        "std_epsilon": 1e-6,
        "component_mean": [[[0.1, 0.2], [0.3, 0.4]], [[-0.1, -0.2], [-0.3, -0.4]]],
        "component_std": [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]],
        "component_std_clamped": [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]],
    }
    new_stats_path = tmp_path / "new_stats.json"
    with new_stats_path.open("w") as f:
        json.dump(new_schema, f)

    norm = train_dit.load_latent_normalization(
        stats_path=new_stats_path,
        latent_channels=2,
        latent_height=2,
        latent_width=2,
        device=torch.device("cpu"),
    )

    assert norm.mean.shape == (1, 2, 1, 2, 2)
    assert norm.std.shape == (1, 2, 1, 2, 2)
    assert norm.version == 2
    assert norm.scheme == "component_chw_shared_time"
