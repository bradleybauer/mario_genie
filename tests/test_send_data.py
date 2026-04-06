from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from remote import send_data


def _make_project_root(tmp_path: Path, *subdirs: str) -> Path:
    project_root = tmp_path / "project"
    data_root = project_root / "data"
    for subdir in subdirs:
        (data_root / subdir).mkdir(parents=True)
    return project_root


def test_parse_args_raw_flag() -> None:
    with mock.patch("sys.argv", ["send_data.py", "--raw", "a", "b"]):
        args = send_data.parse_args()
    assert args.raw is True
    assert args.workers == ["a", "b"]


def test_send_data_default_syncs_normalized_and_latents_only(tmp_path: Path) -> None:
    worker = mock.Mock()
    worker.project_dir = "/root/mario"
    project_root = _make_project_root(tmp_path, "normalized", "latents")

    with (
        mock.patch.object(send_data, "PROJECT_ROOT", project_root),
        mock.patch.object(send_data, "ssh") as mock_ssh,
        mock.patch.object(send_data, "rsync_to") as mock_rsync,
    ):
        send_data.send_data(worker)

    mock_ssh.assert_called_once_with(worker, "mkdir -p /root/mario/data", capture=True)
    assert mock_rsync.call_count == 2
    synced_sources = [call.args[1] for call in mock_rsync.call_args_list]
    assert str(project_root / "data" / "normalized") in synced_sources
    assert str(project_root / "data" / "latents") in synced_sources
    assert str(project_root / "data" / "raw") not in synced_sources


def test_send_data_with_raw_syncs_raw_too(tmp_path: Path) -> None:
    worker = mock.Mock()
    worker.project_dir = "/root/mario"
    project_root = _make_project_root(tmp_path, "normalized", "latents", "raw")

    with (
        mock.patch.object(send_data, "PROJECT_ROOT", project_root),
        mock.patch.object(send_data, "ssh") as mock_ssh,
        mock.patch.object(send_data, "rsync_to") as mock_rsync,
    ):
        send_data.send_data(worker, include_raw=True)

    mock_ssh.assert_called_once_with(worker, "mkdir -p /root/mario/data", capture=True)
    assert mock_rsync.call_count == 3
    synced_sources = [call.args[1] for call in mock_rsync.call_args_list]
    assert str(project_root / "data" / "raw") in synced_sources
    raw_call = next(call for call in mock_rsync.call_args_list if call.args[1] == str(project_root / "data" / "raw"))
    assert "--exclude=*.mss" in raw_call.kwargs["extra_args"]


def test_send_data_non_raw_sources_do_not_exclude_mss(tmp_path: Path) -> None:
    worker = mock.Mock()
    worker.project_dir = "/root/mario"
    project_root = _make_project_root(tmp_path, "normalized", "latents")

    with (
        mock.patch.object(send_data, "PROJECT_ROOT", project_root),
        mock.patch.object(send_data, "ssh") as mock_ssh,
        mock.patch.object(send_data, "rsync_to") as mock_rsync,
    ):
        send_data.send_data(worker)

    mock_ssh.assert_called_once_with(worker, "mkdir -p /root/mario/data", capture=True)
    for call in mock_rsync.call_args_list:
        assert "--exclude=*.mss" not in call.kwargs["extra_args"]


def test_send_data_with_raw_skips_missing_normalized_dir(tmp_path: Path) -> None:
    worker = mock.Mock()
    worker.project_dir = "/root/mario"

    project_root = _make_project_root(tmp_path, "latents", "raw")

    with (
        mock.patch.object(send_data, "PROJECT_ROOT", project_root),
        mock.patch.object(send_data, "ssh") as mock_ssh,
        mock.patch.object(send_data, "rsync_to") as mock_rsync,
    ):
        send_data.send_data(worker, include_raw=True)

    mock_ssh.assert_called_once_with(worker, "mkdir -p /root/mario/data", capture=True)
    synced_sources = [call.args[1] for call in mock_rsync.call_args_list]
    assert str(project_root / "data" / "latents") in synced_sources
    assert str(project_root / "data" / "raw") in synced_sources
    assert str(project_root / "data" / "normalized") not in synced_sources


def test_send_data_raises_when_no_sources_exist(tmp_path: Path) -> None:
    worker = mock.Mock()
    worker.project_dir = "/root/mario"
    project_root = tmp_path / "project"

    with (
        mock.patch.object(send_data, "PROJECT_ROOT", project_root),
        mock.patch.object(send_data, "ssh") as mock_ssh,
        mock.patch.object(send_data, "rsync_to") as mock_rsync,
    ):
        try:
            send_data.send_data(worker, include_raw=True)
        except FileNotFoundError as exc:
            assert "No local data directories found to sync" in str(exc)
        else:
            raise AssertionError("Expected FileNotFoundError")

    mock_ssh.assert_not_called()
    mock_rsync.assert_not_called()