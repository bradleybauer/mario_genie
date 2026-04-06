"""Tests for remote/provision.py — all API calls mocked."""

import argparse
import os
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest

REMOTE_DIR = Path(__file__).resolve().parents[1] / "remote"
if str(REMOTE_DIR) not in sys.path:
    sys.path.insert(0, str(REMOTE_DIR))

import provision


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def api_key():
    return "test-api-key-12345"


@pytest.fixture
def tmp_config(tmp_path):
    """Temporarily point CONFIG_PATH at a temp file."""
    cfg = tmp_path / "config.py"
    orig = provision.CONFIG_PATH
    provision.CONFIG_PATH = cfg
    yield cfg
    provision.CONFIG_PATH = orig


def _make_instance(
    iid=100,
    gpu="RTX 5090",
    status="running",
    ssh_host="1.2.3.4",
    ssh_port=22222,
    gpu_ram=32000,
    gpu_mem_bw=1800,
    cpu_cores=64,
    cpu_ram=128000,
    dph_total=0.5,
    dlperf=50.0,
    geolocation="US",
    gpu_util=80,
    gpu_temp=65,
    cpu_util=40,
    mem_usage=8,
    mem_limit=64,
    vmem_usage=16000,
    ports=None,
    public_ipaddr=None,
):
    return {
        "id": iid,
        "gpu_name": gpu,
        "actual_status": status,
        "ssh_host": ssh_host,
        "ssh_port": ssh_port,
        "gpu_ram": gpu_ram,
        "gpu_mem_bw": gpu_mem_bw,
        "cpu_cores": cpu_cores,
        "cpu_ram": cpu_ram,
        "dph_total": dph_total,
        "dph_base": dph_total,
        "dlperf": dlperf,
        "storage_cost": 0.15,
        "geolocation": geolocation,
        "num_gpus": 1,
        "gpu_util": gpu_util,
        "gpu_temp": gpu_temp,
        "cpu_util": cpu_util,
        "mem_usage": mem_usage,
        "mem_limit": mem_limit,
        "vmem_usage": vmem_usage,
        "ports": ports,
        "public_ipaddr": public_ipaddr,
    }


SAMPLE_OFFER = {
    "id": 999,
    "gpu_name": "RTX 5090",
    "gpu_ram": 32000,
    "gpu_mem_bw": 1800,
    "cpu_cores": 64,
    "cpu_ram": 128000,
    "dph_total": 0.4,
    "dph_base": 0.4,
    "dlperf": 60.0,
    "storage_cost": 0.15,
    "geolocation": "US",
    "num_gpus": 1,
}


# ── _api_key ─────────────────────────────────────────────────────


class TestApiKey:
    def test_env_var_takes_priority(self, tmp_path):
        key_file = tmp_path / ".vast_api_key"
        key_file.write_text("file-key")
        with mock.patch.dict(os.environ, {"VAST_API_KEY": "  env-key  "}):
            assert provision._api_key() == "env-key"

    def test_config_file_fallback(self, tmp_path):
        key_file = tmp_path / ".config" / "vastai" / "vast_api_key"
        key_file.parent.mkdir(parents=True)
        key_file.write_text("file-key\n")
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("VAST_API_KEY", None)
            with mock.patch.object(Path, "home", return_value=tmp_path):
                assert provision._api_key() == "file-key"

    def test_home_vast_api_key(self, tmp_path):
        key_file = tmp_path / ".vast_api_key"
        key_file.write_text("  home-key  ")
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("VAST_API_KEY", None)
            with mock.patch.object(Path, "home", return_value=tmp_path):
                assert provision._api_key() == "home-key"

    def test_no_key_exits(self, tmp_path):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("VAST_API_KEY", None)
            with mock.patch.object(Path, "home", return_value=tmp_path):
                with pytest.raises(SystemExit):
                    provision._api_key()


# ── _ssh_info ────────────────────────────────────────────────────


class TestSshInfo:
    def test_ports_dict(self):
        inst = _make_instance(ports={"22/tcp": [{"HostPort": "12345"}]}, public_ipaddr="5.6.7.8")
        assert provision._ssh_info(inst) == ("5.6.7.8", 12345)

    def test_ssh_host_fallback(self):
        inst = _make_instance(ssh_host="10.0.0.1", ssh_port=9999, ports=None)
        assert provision._ssh_info(inst) == ("10.0.0.1", 9999)

    def test_ports_preferred_over_ssh_host(self):
        inst = _make_instance(
            ports={"22/tcp": [{"HostPort": "11111"}]},
            public_ipaddr="8.8.8.8",
            ssh_host="10.0.0.1",
            ssh_port=9999,
        )
        assert provision._ssh_info(inst) == ("8.8.8.8", 11111)

    def test_no_ssh_returns_none(self):
        inst = {"ports": {}, "ssh_host": None, "ssh_port": None}
        assert provision._ssh_info(inst) is None

    def test_empty_ports_uses_ssh_host(self):
        inst = _make_instance(ports={}, ssh_host="1.1.1.1", ssh_port=8080)
        assert provision._ssh_info(inst) == ("1.1.1.1", 8080)


# ── _print_table ─────────────────────────────────────────────────


class TestPrintTable:
    def test_basic_output(self, capsys):
        provision._print_table(("A", "BB"), [("x", "yy"), ("zz", "w")])
        out = capsys.readouterr().out
        lines = out.strip().split("\n")
        assert len(lines) == 4  # header + separator + 2 data rows
        assert "A" in lines[0]
        assert "─" in lines[1]
        assert "x" in lines[2]

    def test_empty_rows(self, capsys):
        provision._print_table(("Col1",), [])
        out = capsys.readouterr().out
        lines = out.strip().split("\n")
        assert len(lines) == 2  # header + separator only

    def test_column_widths_auto_expand(self, capsys):
        provision._print_table(("H",), [("longer_value",)])
        out = capsys.readouterr().out
        # Separator should match longest value width
        sep_line = out.strip().split("\n")[1]
        assert len(sep_line.strip()) >= len("longer_value")


# ── _read_existing_config / _write_config ────────────────────────


class TestConfigRoundTrip:
    def test_write_and_read_back(self, tmp_config):
        instances = [
            _make_instance(iid=100, gpu="RTX 5090", ssh_host="1.2.3.4", ssh_port=22222),
            _make_instance(iid=200, gpu="RTX 3060", ssh_host="5.6.7.8", ssh_port=33333),
        ]
        provision._write_config(instances)

        mapping = provision._read_existing_config()
        assert 100 in mapping
        assert 200 in mapping
        assert mapping[100]["name"] == "a"
        assert mapping[200]["name"] == "b"
        assert mapping[100]["instance_id"] == 100

    def test_preserves_existing_names(self, tmp_config):
        # Write initial config
        inst1 = [_make_instance(iid=100, ssh_host="1.2.3.4", ssh_port=22222)]
        provision._write_config(inst1)
        assert provision._read_existing_config()[100]["name"] == "a"

        # Add a second instance — first should keep name "a"
        inst2 = [
            _make_instance(iid=100, ssh_host="1.2.3.4", ssh_port=22222),
            _make_instance(iid=200, ssh_host="5.6.7.8", ssh_port=33333),
        ]
        provision._write_config(inst2)
        mapping = provision._read_existing_config()
        assert mapping[100]["name"] == "a"
        assert mapping[200]["name"] == "b"

    def test_name_not_recycled_on_remove(self, tmp_config):
        # Write two instances
        provision._write_config([
            _make_instance(iid=100, ssh_host="1.1.1.1", ssh_port=1111),
            _make_instance(iid=200, ssh_host="2.2.2.2", ssh_port=2222),
        ])
        # Remove first instance, add third — old name "a" is still reserved
        provision._write_config([
            _make_instance(iid=200, ssh_host="2.2.2.2", ssh_port=2222),
            _make_instance(iid=300, ssh_host="3.3.3.3", ssh_port=3333),
        ])
        mapping = provision._read_existing_config()
        assert mapping[200]["name"] == "b"
        # New instance gets "c" because "a" is still reserved from old config
        assert mapping[300]["name"] == "c"

    def test_empty_instances_clears_config(self, tmp_config):
        provision._write_config([_make_instance(iid=100, ssh_host="1.1.1.1", ssh_port=1111)])
        provision._write_config([])
        mapping = provision._read_existing_config()
        assert mapping == {}

    def test_read_missing_config_returns_empty(self, tmp_config):
        assert provision._read_existing_config() == {}

    def test_duplicate_name_in_config_exits(self, tmp_config):
        tmp_config.write_text(textwrap.dedent("""\
            WORKERS = [
                {"name": "a", "host": "1.1.1.1", "port": 1111, "user": "root", "instance_id": 100},
                {"name": "a", "host": "2.2.2.2", "port": 2222, "user": "root", "instance_id": 200},
            ]
        """))
        with pytest.raises(SystemExit, match="Duplicate worker name"):
            provision._read_existing_config()

    def test_skip_instance_without_ssh(self, tmp_config, capsys):
        inst = _make_instance(iid=999, ssh_host=None, ssh_port=None, ports=None)
        provision._write_config([inst])
        out = capsys.readouterr().out
        assert "no SSH info" in out
        assert provision._read_existing_config() == {}

    def test_config_file_content_is_valid_python(self, tmp_config):
        provision._write_config([_make_instance(iid=100, ssh_host="1.1.1.1", ssh_port=1111)])
        ns = {}
        exec(tmp_config.read_text(), ns)
        assert "WORKERS" in ns
        assert ns["PROJECT_NAME"] == "mario"


# ── _search ──────────────────────────────────────────────────────


class TestSearch:
    @mock.patch.object(provision, "_req")
    def test_returns_offers_list(self, mock_req, api_key):
        mock_req.return_value = {"offers": [SAMPLE_OFFER]}
        offers = provision._search(api_key, disk=50, sort="price")
        assert len(offers) == 1
        assert offers[0]["id"] == 999

    @mock.patch.object(provision, "_req")
    def test_gpu_filter_normalizes_underscores(self, mock_req, api_key):
        mock_req.return_value = {"offers": []}
        provision._search(api_key, disk=50, sort="price", gpu="RTX_5090")
        body = mock_req.call_args[1]["json"]
        assert body["gpu_name"] == {"eq": "RTX 5090"}

    @mock.patch.object(provision, "_req")
    def test_sort_order_passed_correctly(self, mock_req, api_key):
        mock_req.return_value = {"offers": []}
        provision._search(api_key, disk=50, sort="dlperf")
        body = mock_req.call_args[1]["json"]
        assert body["order"] == [["dlperf", "desc"]]

    @mock.patch.object(provision, "_req")
    def test_default_sort_is_price(self, mock_req, api_key):
        mock_req.return_value = {"offers": []}
        provision._search(api_key, disk=50, sort="price")
        body = mock_req.call_args[1]["json"]
        assert body["order"] == [["dph_total", "asc"]]

    @mock.patch.object(provision, "_req")
    def test_includes_all_filters(self, mock_req, api_key):
        mock_req.return_value = {"offers": []}
        provision._search(api_key, disk=50, sort="price")
        body = mock_req.call_args[1]["json"]
        assert body["reliability"] == {"gte": 0.95}
        assert body["num_gpus"] == {"eq": 1}
        assert body["rentable"] == {"eq": True}
        assert body["rented"] == {"eq": False}

    @mock.patch.object(provision, "_req")
    def test_no_gpu_filter_when_none(self, mock_req, api_key):
        mock_req.return_value = {"offers": []}
        provision._search(api_key, disk=50, sort="price", gpu=None)
        body = mock_req.call_args[1]["json"]
        assert "gpu_name" not in body

    @mock.patch.object(provision, "_req")
    def test_handles_flat_list_response(self, mock_req, api_key):
        """API sometimes returns a flat list instead of {"offers": [...]}."""
        mock_req.return_value = [SAMPLE_OFFER, SAMPLE_OFFER]
        offers = provision._search(api_key, disk=50, sort="price")
        assert len(offers) == 2


# ── _req ─────────────────────────────────────────────────────────


class TestReq:
    @mock.patch("requests.request")
    def test_sets_auth_header(self, mock_request, api_key):
        mock_resp = mock.Mock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status = mock.Mock()
        mock_request.return_value = mock_resp

        provision._req("GET", "/instances/", api_key)
        call_kwargs = mock_request.call_args
        assert call_kwargs[1]["headers"]["Authorization"] == f"Bearer {api_key}"

    @mock.patch("requests.request")
    def test_raises_on_http_error(self, mock_request, api_key):
        mock_resp = mock.Mock()
        mock_resp.raise_for_status.side_effect = provision.requests.exceptions.HTTPError("404")
        mock_request.return_value = mock_resp

        with pytest.raises(provision.requests.exceptions.HTTPError):
            provision._req("GET", "/bad/", api_key)


# ── cmd_ls ───────────────────────────────────────────────────────


class TestCmdLs:
    @mock.patch.object(provision, "_instances")
    def test_no_instances(self, mock_inst, api_key, capsys):
        mock_inst.return_value = []
        args = argparse.Namespace()
        provision.cmd_ls(api_key, args)
        assert "No instances" in capsys.readouterr().out

    @mock.patch.object(provision, "_read_existing_config")
    @mock.patch.object(provision, "_instances")
    def test_shows_instance_details(self, mock_inst, mock_config, api_key, capsys):
        inst = _make_instance(iid=100, gpu="RTX 5090", status="running")
        mock_inst.return_value = [inst]
        mock_config.return_value = {100: {"name": "a", "instance_id": 100}}
        provision.cmd_ls(api_key, argparse.Namespace())
        out = capsys.readouterr().out
        assert "RTX 5090" in out
        assert "running" in out
        assert "a" in out


# ── cmd_down ─────────────────────────────────────────────────────


class TestCmdDown:
    @mock.patch.object(provision, "_instances")
    def test_no_instances(self, mock_inst, api_key, capsys):
        mock_inst.return_value = []
        args = argparse.Namespace(name=[], yes=False)
        provision.cmd_down(api_key, args)
        assert "No instances" in capsys.readouterr().out

    @mock.patch.object(provision, "_instances")
    @mock.patch.object(provision, "_read_existing_config")
    def test_no_args_shows_usage(self, mock_config, mock_inst, api_key, tmp_config):
        inst = _make_instance(iid=100, status="running")
        mock_inst.return_value = [inst]
        mock_config.return_value = {100: {"name": "a", "instance_id": 100}}
        args = argparse.Namespace(name=[])
        with pytest.raises(SystemExit, match="Usage"):
            provision.cmd_down(api_key, args)

    @mock.patch.object(provision, "_write_config")
    @mock.patch.object(provision, "_destroy")
    @mock.patch.object(provision, "_instances")
    @mock.patch.object(provision, "_read_existing_config")
    def test_destroy_by_id(self, mock_config, mock_inst, mock_destroy, mock_write, api_key, tmp_config, capsys):
        inst = _make_instance(iid=100, status="running")
        mock_inst.side_effect = [
            [inst],    # initial list
            [],        # after destroy
        ]
        mock_config.return_value = {100: {"name": "a", "instance_id": 100}}
        mock_destroy.return_value = {"success": True}
        args = argparse.Namespace(name=["100"])
        with mock.patch("builtins.input", return_value="y"):
            provision.cmd_down(api_key, args)
        mock_destroy.assert_called_once_with(api_key, 100)

    @mock.patch.object(provision, "_write_config")
    @mock.patch.object(provision, "_destroy")
    @mock.patch.object(provision, "_instances")
    @mock.patch.object(provision, "_read_existing_config")
    def test_destroy_all(self, mock_config, mock_inst, mock_destroy, mock_write, api_key, tmp_config, capsys):
        insts = [
            _make_instance(iid=100, ssh_host="1.1.1.1", ssh_port=1111),
            _make_instance(iid=200, ssh_host="2.2.2.2", ssh_port=2222),
        ]
        mock_inst.side_effect = [
            insts,  # initial list
            [],     # after destroy
        ]
        mock_config.return_value = {100: {"name": "a"}, 200: {"name": "b"}}
        mock_destroy.return_value = {"success": True}
        args = argparse.Namespace(name=["all"])
        with mock.patch("builtins.input", return_value="y"):
            provision.cmd_down(api_key, args)
        assert mock_destroy.call_count == 2

    @mock.patch.object(provision, "_instances")
    @mock.patch.object(provision, "_read_existing_config")
    def test_unknown_instance_id(self, mock_config, mock_inst, api_key, capsys):
        mock_inst.return_value = [_make_instance(iid=100)]
        mock_config.return_value = {100: {"name": "a", "instance_id": 100}}
        args = argparse.Namespace(name=["999"])
        with pytest.raises(SystemExit, match="No matching"):
            provision.cmd_down(api_key, args)
        assert "Unknown instance: 999" in capsys.readouterr().out


# ── cmd_up ───────────────────────────────────────────────────────


class TestCmdUp:
    @mock.patch.object(provision, "_write_config")
    @mock.patch.object(provision, "_wait")
    @mock.patch.object(provision, "_create")
    @mock.patch.object(provision, "_search")
    @mock.patch.object(provision, "_instances")
    def test_full_provision_flow(self, mock_inst, mock_search, mock_create, mock_wait, mock_write, api_key, tmp_config, capsys):
        mock_search.return_value = [SAMPLE_OFFER]
        mock_create.return_value = {"success": True, "new_contract": 500}
        inst = _make_instance(iid=500, ssh_host="9.9.9.9", ssh_port=22)
        mock_wait.return_value = [inst]
        mock_inst.return_value = [inst]

        args = argparse.Namespace(
            disk=50, image="pytorch/pytorch", timeout=600,
            sort="price", gpu=None,
        )
        with mock.patch("builtins.input", side_effect=["1", "y"]):
            provision.cmd_up(api_key, args)
        mock_create.assert_called_once()
        mock_write.assert_called_once()

    @mock.patch.object(provision, "_search")
    def test_no_offers_exits(self, mock_search, api_key, capsys):
        mock_search.return_value = []
        args = argparse.Namespace(
            disk=50, image="pytorch/pytorch", timeout=600,
            sort="price", gpu=None,
        )
        with pytest.raises(SystemExit, match="No matching"):
            provision.cmd_up(api_key, args)

    @mock.patch.object(provision, "_create")
    @mock.patch.object(provision, "_search")
    def test_failed_create_exits(self, mock_search, mock_create, api_key, capsys):
        mock_search.return_value = [SAMPLE_OFFER]
        mock_create.return_value = {"success": False, "error": "out of stock"}

        args = argparse.Namespace(
            disk=50, image="pytorch/pytorch", timeout=600,
            sort="price", gpu=None,
        )
        with mock.patch("builtins.input", side_effect=["1", "y"]):
            with pytest.raises(SystemExit, match="No instances created"):
                provision.cmd_up(api_key, args)


# ── _wait ────────────────────────────────────────────────────────


class TestWait:
    @mock.patch.object(provision, "_instances")
    def test_returns_when_ready(self, mock_inst, api_key):
        inst = _make_instance(iid=100, status="running", ssh_host="1.1.1.1", ssh_port=22)
        mock_inst.return_value = [inst]
        result = provision._wait(api_key, [100], timeout=60)
        assert len(result) == 1
        assert result[0]["id"] == 100

    @mock.patch.object(provision, "_instances")
    @mock.patch("time.sleep")
    def test_timeout(self, mock_sleep, mock_inst, api_key, capsys):
        inst = _make_instance(iid=100, status="loading")
        mock_inst.return_value = [inst]
        call_count = 0

        def advancing_time():
            nonlocal call_count
            call_count += 1
            # First few calls return 0, then jump past timeout
            return 0 if call_count < 6 else 1000

        with mock.patch("time.time", side_effect=advancing_time):
            result = provision._wait(api_key, [100], timeout=5)
        assert result == []


# ── cmd_update_config ────────────────────────────────────────────


class TestCmdUpdateConfig:
    @mock.patch.object(provision, "_write_config")
    @mock.patch.object(provision, "_instances")
    def test_with_running_instances(self, mock_inst, mock_write, api_key, tmp_config):
        inst = _make_instance(iid=100, status="running")
        mock_inst.return_value = [inst]
        provision.cmd_update_config(api_key, argparse.Namespace())
        mock_write.assert_called_once()

    @mock.patch.object(provision, "_instances")
    def test_no_running_exits(self, mock_inst, api_key):
        mock_inst.return_value = [_make_instance(iid=100, status="exited")]
        with pytest.raises(SystemExit, match="No running"):
            provision.cmd_update_config(api_key, argparse.Namespace())


# ── main CLI parsing ─────────────────────────────────────────────


class TestMain:
    def test_no_command_exits(self):
        with mock.patch("sys.argv", ["provision.py"]):
            with pytest.raises(SystemExit):
                provision.main()

    @mock.patch.object(provision, "_api_key", return_value="k")
    @mock.patch.object(provision, "cmd_ls")
    def test_ls_dispatch(self, mock_ls, mock_key):
        with mock.patch("sys.argv", ["provision.py", "ls"]):
            provision.main()
        mock_ls.assert_called_once()

    @mock.patch.object(provision, "_api_key", return_value="k")
    @mock.patch.object(provision, "cmd_up")
    def test_up_args(self, mock_up, mock_key):
        with mock.patch("sys.argv", ["provision.py", "up", "--sort", "dlperf", "--gpu", "RTX_5090"]):
            provision.main()
        args = mock_up.call_args[0][1]
        assert args.sort == "dlperf"
        assert args.gpu == "RTX_5090"

    @mock.patch.object(provision, "_api_key", return_value="k")
    @mock.patch.object(provision, "cmd_down")
    def test_down_args(self, mock_down, mock_key):
        with mock.patch("sys.argv", ["provision.py", "down", "a", "b"]):
            provision.main()
        args = mock_down.call_args[0][1]
        assert args.name == ["a", "b"]
