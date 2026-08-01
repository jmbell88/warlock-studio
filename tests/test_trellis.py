from __future__ import annotations

import subprocess
import sys

from animancer3d.pipelines.trellis import TrellisServer


def test_argv_defaults_webp_off(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    assert srv._argv()[-2:] == ["--webp", "off"]


def test_argv_webp_on_when_configured(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971, webp=True)
    assert srv._argv()[-2:] == ["--webp", "on"]


def test_argv_includes_models_host_port(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    argv = srv._argv()
    assert argv[0] == str(tmp_path / "exe")
    assert "--models" in argv and str(tmp_path / "models") in argv
    assert "--port" in argv and "17971" in argv


def test_reap_if_dead_clears_an_already_exited_process(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    srv._proc = dead
    srv._reap_if_dead()
    assert srv._proc is None


def test_reap_if_dead_leaves_a_running_process_alone(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    running = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"]
    )
    try:
        srv._proc = running
        srv._reap_if_dead()
        assert srv._proc is running
    finally:
        running.kill()
        running.wait()
