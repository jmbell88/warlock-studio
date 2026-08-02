from __future__ import annotations

import socket

from warlock import models as model_registry
from warlock.config import Config
from warlock.doctor import run_checks


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


def test_run_checks_returns_every_check(tmp_path):
    # Eight fixed checks plus one row per registry entry -- derived rather than
    # hardcoded so adding a model doesn't fail an unrelated assertion.
    expected = 8 + len(model_registry.BASE_MODELS) + len(model_registry.STYLE_LORAS)
    assert len(run_checks(_config(tmp_path))) == expected


def test_cuda_check_is_not_fatal_when_torch_missing_or_unavailable(tmp_path):
    checks = {c.name: c for c in run_checks(_config(tmp_path))}
    assert checks["CUDA"].fatal is False


def test_disk_check_is_not_fatal(tmp_path):
    checks = {c.name: c for c in run_checks(_config(tmp_path))}
    assert checks["free disk space"].fatal is False
    assert isinstance(checks["free disk space"].ok, bool)


def _t2i_names() -> list[str]:
    return [f"image model: {m.label}" for m in model_registry.BASE_MODELS.values()] + [
        f"style LoRA: {lora.label}" for lora in model_registry.STYLE_LORAS.values()
    ]


def test_every_image_model_and_lora_gets_its_own_non_fatal_row(tmp_path):
    # One row per registry entry, so the report names *which* optional download
    # is missing rather than collapsing five of them into one line.
    checks = {c.name: c for c in run_checks(_config(tmp_path, t2i_model_root=tmp_path / "m"))}
    for name in _t2i_names():
        assert checks[name].ok is False
        assert checks[name].fatal is False
        assert "hf download" in checks[name].detail


def test_base_model_check_passes_with_local_weights(tmp_path):
    root = tmp_path / "m"
    spec = model_registry.BASE_MODELS["turbo"]
    (root / spec.dir_name / "unet").mkdir(parents=True)
    (root / spec.dir_name / "model_index.json").write_text("{}")
    (root / spec.dir_name / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"")
    checks = {c.name: c for c in run_checks(_config(tmp_path, t2i_model_root=root))}
    assert checks[f"image model: {spec.label}"].ok is True


def test_style_lora_check_passes_when_file_present(tmp_path):
    root = tmp_path / "m"
    lora = model_registry.STYLE_LORAS["render3d"]
    (root / "loras").mkdir(parents=True)
    (root / "loras" / lora.filename).write_bytes(b"")
    checks = {c.name: c for c in run_checks(_config(tmp_path, t2i_model_root=root))}
    assert checks[f"style LoRA: {lora.label}"].ok is True


def test_turbo_dir_override_is_still_honoured(tmp_path):
    # WARLOCK_T2I_DIR predates the registry; existing setups point it at an
    # arbitrary diffusers dir and must keep working.
    override = tmp_path / "elsewhere"
    (override / "unet").mkdir(parents=True)
    (override / "model_index.json").write_text("{}")
    (override / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"")
    config = _config(tmp_path, t2i_model_root=tmp_path / "m", t2i_turbo_dir=override)
    checks = {c.name: c for c in run_checks(config)}
    assert checks[f"image model: {model_registry.BASE_MODELS['turbo'].label}"].ok is True


def test_gltfpack_check_is_non_fatal_when_missing(tmp_path, monkeypatch):
    from warlock import doctor
    from warlock.config import Config

    monkeypatch.setenv("WARLOCK_GLTFPACK", str(tmp_path / "nope.exe"))
    monkeypatch.setenv("WARLOCK_DATA_DIR", str(tmp_path))
    check = doctor._gltfpack_check(Config())
    assert check.ok is False
    assert check.fatal is False
