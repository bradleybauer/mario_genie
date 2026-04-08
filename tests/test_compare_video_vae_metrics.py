from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_compare_video_vae_metrics_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "compare_video_vae_metrics.py"
    spec = importlib.util.spec_from_file_location("compare_video_vae_metrics", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_filter_runs_supports_include_and_exclude_substrings() -> None:
    module = _load_compare_video_vae_metrics_module()
    runs = [
        {"name": "ram_video_vae_v2_run_a"},
        {"name": "ram_video_vae_v2_debug_run"},
        {"name": "other_model_run"},
    ]

    filtered = module.filter_runs(
        runs,
        include_filter="ram_video_vae_v2",
        exclude_filter="debug",
    )

    assert [run["name"] for run in filtered] == ["ram_video_vae_v2_run_a"]


def test_parse_args_accepts_exclude_filter() -> None:
    module = _load_compare_video_vae_metrics_module()

    args = module.parse_args(["--exclude-filter", "debug"])

    assert args.exclude_filter == "debug"
