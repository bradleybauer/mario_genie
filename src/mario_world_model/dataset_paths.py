from __future__ import annotations

from pathlib import Path


def find_chunk_files(data_dir: str | Path) -> list[str]:
    """Return .npz chunk files under a data directory, including nested folders."""
    data_path = Path(data_dir)

    if data_path.is_file():
        return [str(data_path)] if data_path.suffix == ".npz" else []

    if not data_path.exists():
        return []

    return [str(path) for path in sorted(data_path.rglob("*.npz")) if path.is_file()]