from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _flatten_mapping(
    payload: dict[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        key_str = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_mapping(value, prefix=key_str))
        else:
            flattened[key_str] = value
    return flattened


def _coerce_param_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, str)):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)


def _coerce_metric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        return None
    return None


def parse_mlflow_tags(raw_tags: list[str] | None) -> dict[str, str]:
    tags: dict[str, str] = {}
    for raw_tag in raw_tags or []:
        if "=" not in raw_tag:
            raise ValueError(f"Invalid MLflow tag {raw_tag!r}; expected key=value")
        key, value = raw_tag.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid MLflow tag {raw_tag!r}; key cannot be empty")
        tags[key] = value.strip()
    return tags


@dataclass
class MLflowRun:
    enabled: bool
    _module: Any = None
    _run: Any = None

    @classmethod
    def disabled(cls) -> "MLflowRun":
        return cls(enabled=False)

    @classmethod
    def create(
        cls,
        *,
        enabled: bool,
        experiment_name: str,
        run_name: str | None,
        tracking_uri: str | None,
        tags: dict[str, str] | None = None,
    ) -> "MLflowRun":
        if not enabled:
            return cls.disabled()

        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow logging was requested, but the mlflow package is not installed. "
                "Install the project environment again after updating dependencies."
            ) from exc

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)
        return cls(enabled=True, _module=mlflow, _run=run)

    @property
    def run_id(self) -> str | None:
        if not self.enabled or self._run is None:
            return None
        info = getattr(self._run, "info", None)
        return getattr(info, "run_id", None)

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled:
            return
        flattened = _flatten_mapping(params)
        sanitized: dict[str, str] = {}
        for key, value in flattened.items():
            coerced = _coerce_param_value(value)
            if coerced is None:
                continue
            sanitized[str(key)] = coerced
        for key in sorted(sanitized):
            self._module.log_param(key, sanitized[key])

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return
        step = metrics.get("step")
        metric_prefix = metrics.get("type")
        sanitized: dict[str, float] = {}
        for key, value in metrics.items():
            if key in {"type", "step"}:
                continue
            coerced = _coerce_metric_value(value)
            if coerced is None:
                continue
            metric_name = f"{metric_prefix}/{key}" if metric_prefix else str(key)
            sanitized[metric_name] = coerced
        if sanitized:
            self._module.log_metrics(sanitized, step=int(step) if isinstance(step, int) else None)

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        if not self.enabled:
            return
        self._module.log_dict(payload, artifact_file)

    def log_artifact(self, path: Path | str) -> None:
        if not self.enabled:
            return
        artifact_path = Path(path)
        if artifact_path.exists():
            self._module.log_artifact(str(artifact_path))

    def log_artifacts(self, path: Path | str) -> None:
        if not self.enabled:
            return
        artifact_dir = Path(path)
        if artifact_dir.is_dir():
            self._module.log_artifacts(str(artifact_dir))

    def finish(self, *, status: str = "FINISHED") -> None:
        if not self.enabled:
            return
        self._module.end_run(status=status)
