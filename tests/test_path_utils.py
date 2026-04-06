from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(PROJECT_ROOT / "src")
sys.path.insert(0, SRC_DIR)

from path_utils import resolve_workspace_path, serialize_project_path


def test_serialize_project_path_prefers_project_relative(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    checkpoint = project_root / "checkpoints" / "run" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("x")

    serialized = serialize_project_path(checkpoint, project_root=project_root)

    assert serialized == "checkpoints/run/model.pt"


def test_resolve_workspace_path_remaps_foreign_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    checkpoint = project_root / "checkpoints" / "run" / "model.pt"
    stats_path = project_root / "data" / "latents" / "latent_stats.json"
    checkpoint.parent.mkdir(parents=True)
    stats_path.parent.mkdir(parents=True)
    checkpoint.write_text("x")
    stats_path.write_text("{}")

    resolved_checkpoint = resolve_workspace_path(
        "/home/bradley/mario/checkpoints/run/model.pt",
        project_root=project_root,
    )
    resolved_stats = resolve_workspace_path(
        "/root/mario/data/latents/latent_stats.json",
        project_root=project_root,
    )

    assert resolved_checkpoint == checkpoint.resolve()
    assert resolved_stats == stats_path.resolve()