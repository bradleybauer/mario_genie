from __future__ import annotations

from pathlib import Path


_PROJECT_PATH_ANCHORS = (
    "checkpoints",
    "data",
    "results",
    "pictures",
    "scripts",
    "src",
    "tests",
)


def serialize_project_path(path: str | Path, *, project_root: Path) -> str:
    resolved = Path(path).resolve()
    project_root = project_root.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return str(resolved)


def _iter_remapped_absolute_candidates(raw: Path, *, project_root: Path):
    if not raw.is_absolute():
        return

    seen: set[Path] = set()
    for index, part in enumerate(raw.parts):
        if part not in _PROJECT_PATH_ANCHORS:
            continue

        rel = Path(*raw.parts[index:])
        candidate = project_root / rel
        if candidate not in seen:
            seen.add(candidate)
            yield candidate

        if part == "checkpoints" and len(rel.parts) > 1:
            fallback = project_root / "results" / Path(*rel.parts[1:])
            if fallback not in seen:
                seen.add(fallback)
                yield fallback


def resolve_workspace_path(
    value: str | None,
    *,
    project_root: Path,
    config_dir: Path | None = None,
) -> Path | None:
    if value is None:
        return None

    raw = Path(value)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
        candidates.extend(_iter_remapped_absolute_candidates(raw, project_root=project_root))
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(project_root / raw)
        if config_dir is not None:
            candidates.append(config_dir / raw)
        if len(raw.parts) > 1 and raw.parts[0] == "checkpoints":
            candidates.append(project_root / "results" / Path(*raw.parts[1:]))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            if candidate.exists():
                return candidate.resolve()
        except PermissionError:
            continue
        except OSError:
            continue

    return (Path.cwd() / raw).resolve() if not raw.is_absolute() else raw