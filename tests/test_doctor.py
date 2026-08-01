from __future__ import annotations

import socket

from animancer3d.config import Config
from animancer3d.doctor import run_checks


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _config(tmp_path, **overrides) -> Config:
    (tmp_path / "assets").mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        trellis_port=_free_port(),
    )
    kwargs.update(overrides)
    return Config(**kwargs)


def test_exe_check_reports_missing_exe_as_fatal(tmp_path):
    checks = {c.name: c for c in run_checks(_config(tmp_path))}
    assert checks["trellis-server.exe"].ok is False
    assert checks["trellis-server.exe"].fatal is True


def test_exe_check_passes_when_exe_exists(tmp_path):
    exe = tmp_path / "trellis-server.exe"
    exe.write_bytes(b"")
    checks = {c.name: c for c in run_checks(_config(tmp_path, trellis_server_exe=exe))}
    assert checks["trellis-server.exe"].ok is True


def test_gguf_check_finds_weight_files(tmp_path):
    models = tmp_path / "models"
    models.mkdir(parents=True)
    (models / "trellis.gguf").write_bytes(b"")
    checks = {c.name: c for c in run_checks(_config(tmp_path, trellis_models_dir=models))}
    assert checks["TRELLIS GGUF weights"].ok is True


def test_gguf_check_reports_missing_dir_as_fatal(tmp_path):
    checks = {c.name: c for c in run_checks(_config(tmp_path))}
    assert checks["TRELLIS GGUF weights"].ok is False
    assert checks["TRELLIS GGUF weights"].fatal is True


def test_birefnet_check_is_not_fatal_when_missing(tmp_path):
    checks = {c.name: c for c in run_checks(_config(tmp_path))}
    assert checks["birefnet.gguf (background removal)"].ok is False
    assert checks["birefnet.gguf (background removal)"].fatal is False


def test_port_check_reports_a_free_port_as_ok(tmp_path):
    checks = {c.name: c for c in run_checks(_config(tmp_path))}
    assert checks["trellis port"].ok is True


def test_port_check_reports_a_bound_port_as_not_ok(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        checks = {c.name: c for c in run_checks(_config(tmp_path, trellis_port=port))}
        assert checks["trellis port"].ok is False


def test_run_checks_returns_seven_checks(tmp_path):
    assert len(run_checks(_config(tmp_path))) == 7
