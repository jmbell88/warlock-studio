from __future__ import annotations

import socket

from warlock import doctor
from warlock import models as model_registry
from warlock.config import Config
from warlock.doctor import run_checks
from warlock.pipelines import matting


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
    assert checks["trellis: birefnet.gguf (background removal)"].ok is False
    assert checks["trellis: birefnet.gguf (background removal)"].fatal is False


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
    # Eleven fixed checks plus one row per registry entry -- derived rather
    # than hardcoded so adding a model doesn't fail an unrelated assertion.
    expected = (
        11
        + len(model_registry.BASE_MODELS)
        + len(model_registry.STYLE_LORAS)
        + len(model_registry.IP_ADAPTERS)
        + len(model_registry.CONTROLNETS)
        + len(model_registry.METRIC_MODELS)
        + len(model_registry.MATTING_MODELS)
        + len(model_registry.POSE_MODELS)
    )
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


def test_a_probe_driven_row_checks_every_file_it_names(tmp_path):
    """flux_klein has no unet/, so the default formula would never go green;
    and its text encoder is half the download, so a probe satisfied by the
    transformer alone would call a half-fetched model present."""
    root = tmp_path / "m"
    spec = model_registry.BASE_MODELS["flux_klein"]
    assert spec.probe, "this test is about the probe path"
    base = root / spec.dir_name
    base.mkdir(parents=True)
    (base / "model_index.json").write_text("{}")

    def row():
        checks = {c.name: c for c in run_checks(_config(tmp_path, t2i_model_root=root))}
        return checks[f"image model: {spec.label}"]

    for rel in spec.probe[:-1]:
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    assert row().ok is False, "a partial download must not read as present"
    last = base / spec.probe[-1]
    last.parent.mkdir(parents=True, exist_ok=True)
    last.write_bytes(b"")
    assert row().ok is True


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


def test_ip_adapter_row_checks_the_vision_encoder_too(tmp_path):
    """Weights without the encoder load fine and fail at the first call, which
    is not a failure a user can read back to a missing download."""
    # An explicit model root: the developer machine running these tests may
    # well have the real weights downloaded already.
    config = _config(tmp_path, t2i_model_root=tmp_path / "t2i")
    adapter = model_registry.IP_ADAPTERS["plus"]
    root = config.t2i_model_root / adapter.dir_name
    weights = root / adapter.subfolder / adapter.weight_name
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"x")

    checks = {c.name: c for c in run_checks(config)}
    row = checks[f"IP-Adapter: {adapter.label}"]
    assert row.ok is False
    assert row.fatal is False
    assert "encoder" in row.detail
    assert "hf download" in row.detail

    (root / adapter.image_encoder_dir).mkdir(parents=True, exist_ok=True)
    (root / adapter.image_encoder_dir / "config.json").write_text("{}")
    checks = {c.name: c for c in run_checks(config)}
    assert checks[f"IP-Adapter: {adapter.label}"].ok is True


def test_controlnet_rows_are_non_fatal_and_name_their_download(tmp_path):
    config = _config(tmp_path, t2i_model_root=tmp_path / "t2i")
    checks = {c.name: c for c in run_checks(config)}
    for cn in model_registry.CONTROLNETS.values():
        row = checks[f"ControlNet: {cn.label}"]
        assert row.ok is False
        assert row.fatal is False
        assert "hf download" in row.detail


def test_a_missing_matting_model_is_non_fatal_and_says_what_happens_instead(tmp_path):
    config = _config(tmp_path, t2i_model_root=tmp_path / "t2i")
    checks = {c.name: c for c in run_checks(config)}
    for spec in model_registry.MATTING_MODELS.values():
        row = checks[f"host matting: {spec.label}"]
        assert row.ok is False
        assert row.fatal is False
        # The row exists to explain a quality difference, so it has to name
        # both the consequence and the one-time download that removes it.
        assert "fall back" in row.detail
        assert "hf download" in row.detail
        # And the download is not only weights: the repo's modelling code
        # imports packages no resolver can see from the checkpoint, so the row
        # has to name the extra that declares them. It used to say "you may
        # also need: uv pip install timm torchvision", which was both
        # optional-sounding and short of einops and kornia.
        assert "uv sync --extra text2image" in row.detail


def _matting_weights(tmp_path):
    root = tmp_path / "t2i"
    for spec in model_registry.MATTING_MODELS.values():
        (root / spec.dir_name).mkdir(parents=True)
        (root / spec.dir_name / "config.json").write_text("{}", encoding="utf-8")
    return root


def test_a_present_matting_model_claims_the_weights_and_not_readiness(tmp_path, monkeypatch):
    # All the row has done is stat a directory and ask whether four modules
    # resolve. Whether the model then loads depends on code Warlock does not
    # ship, so a green row that said "ready" would be a promise it never made.
    monkeypatch.setattr(doctor, "_missing_modules", lambda names: [])
    monkeypatch.setattr(matting, "last_error", lambda: None)
    root = _matting_weights(tmp_path)
    checks = {c.name: c for c in run_checks(_config(tmp_path, t2i_model_root=root))}
    for spec in model_registry.MATTING_MODELS.values():
        row = checks[f"host matting: {spec.label}"]
        assert row.ok is True
        assert "weights present" in row.detail
        assert "not checked" in row.detail
        # trust_remote_code is disclosed where the user can see it, in the
        # words it deserves: not "loads modelling code", which reads as
        # loading weights, but that other people's Python runs in this process.
        assert ("third-party Python" in row.detail) is spec.remote_code


def test_matting_names_the_modules_that_do_not_resolve(tmp_path, monkeypatch):
    # The whole failure mode this row exists for: every weight on disk, a green
    # row, and _load raising ModuleNotFoundError on the first export. Stating
    # the weights alone made the row agree with the filesystem and disagree
    # with the program, so the imports are probed and the missing ones named.
    monkeypatch.setattr(doctor, "_missing_modules", lambda names: ["einops", "kornia"])
    monkeypatch.setattr(matting, "last_error", lambda: None)
    root = _matting_weights(tmp_path)
    checks = {c.name: c for c in run_checks(_config(tmp_path, t2i_model_root=root))}
    for spec in model_registry.MATTING_MODELS.values():
        row = checks[f"host matting: {spec.label}"]
        assert row.ok is False
        assert row.fatal is False
        assert "einops" in row.detail and "kornia" in row.detail


def test_matting_reports_a_load_that_already_failed_this_session(tmp_path, monkeypatch):
    # A checkpoint can be complete, every import can resolve, and the load can
    # still fail -- half a download, a shape mismatch. matting.py already
    # refuses to retry it; without this the user's only evidence is a log line
    # that scrolled past and edges that got worse.
    monkeypatch.setattr(doctor, "_missing_modules", lambda names: [])
    monkeypatch.setattr(matting, "last_error", lambda: "RuntimeError: half a download")
    root = _matting_weights(tmp_path)
    checks = {c.name: c for c in run_checks(_config(tmp_path, t2i_model_root=root))}
    for spec in model_registry.MATTING_MODELS.values():
        row = checks[f"host matting: {spec.label}"]
        assert row.ok is False
        assert "last load failed: RuntimeError: half a download" in row.detail


def test_no_probe_can_raise_out_of_run_checks(tmp_path):
    # The probe runs at startup, before anything is on screen. A module whose
    # metadata is broken enough that find_spec raises must cost a red row and
    # not the app.
    assert doctor._missing_modules(["warlock", "einops.no.such.thing", "definitely_not_here"]) == [
        "einops.no.such.thing",
        "definitely_not_here",
    ]


def test_the_metric_row_says_it_has_not_checked_that_the_model_loads(tmp_path):
    # torchvision is a declared dependency now rather than something that
    # happened to be in the venv, which makes "the weights are there" and
    # "ranking is on" two different claims: queue._rank_candidate catches the
    # ImportError and scores on composition alone, silently. The row says which
    # of the two it checked.
    root = tmp_path / "t2i"
    for spec in model_registry.METRIC_MODELS.values():
        (root / spec.dir_name).mkdir(parents=True)
        (root / spec.dir_name / "config.json").write_text("{}", encoding="utf-8")
    checks = {c.name: c for c in run_checks(_config(tmp_path, t2i_model_root=root))}
    for spec in model_registry.METRIC_MODELS.values():
        row = checks[f"metric model: {spec.label}"]
        assert row.ok is True
        assert "not checked" in row.detail
        assert "torchvision" in row.detail


def test_the_two_birefnet_rows_are_told_apart(tmp_path):
    # One BiRefNet lives inside trellis-server as a GGUF, the other on the host
    # for 2D exports. They are different downloads, and a user looking at rough
    # edges has to be able to tell which row is about which.
    names = [c.name for c in run_checks(_config(tmp_path))]
    birefnet = [n for n in names if "irefnet" in n or "iRefNet" in n]
    assert len(birefnet) == 2
    assert len(set(birefnet)) == 2
    assert any(n.startswith("trellis:") for n in birefnet)
    assert any(n.startswith("host matting:") for n in birefnet)


def test_a_base_model_missing_its_distillation_lora_is_not_reported_ready(tmp_path):
    """A base LoRA raises at load where a style LoRA is skipped, so it is part
    of the checkpoint as far as this file is concerned. Reporting the model
    ready meant finding out at job time with the weights already in VRAM."""
    from warlock import doctor, models
    from warlock.config import Config

    spec = next(s for s in models.BASE_MODELS.values() if s.base_lora)
    config = Config(t2i_model_root=tmp_path)
    root = tmp_path / spec.dir_name
    (root / "unet").mkdir(parents=True)
    (root / "model_index.json").write_text("{}")
    variant = f".{spec.variant}" if spec.variant else ""
    (root / "unet" / f"diffusion_pytorch_model{variant}.safetensors").write_bytes(b"x")

    def row():
        return next(
            c for c in doctor._t2i_checks(config) if c.name.endswith(spec.label)
        )

    assert row().ok is False
    assert spec.base_lora in row().detail
    # And it is not fatal: every image model is optional, and only the job that
    # picks this one is affected.
    assert row().fatal is False

    (tmp_path / "loras").mkdir()
    (tmp_path / "loras" / spec.base_lora).write_bytes(b"x")
    assert row().ok is True


def test_pose_model_row_is_not_fatal_and_names_the_consequence(tmp_path):
    """Missing, a humanoid rig still happens -- on the bbox-proportional fit,
    which is what every rig used before landmarks existed. That is a quality
    difference with no other visible cause, which is exactly what a row is
    for."""
    spec = model_registry.POSE_MODELS[model_registry.DEFAULT_POSE_MODEL]
    # The model root pinned empty rather than left at PROJECT_ROOT/models,
    # which is the rule CLAUDE.md states for warlockc.dll and gltfpack: a test
    # about what happens when weights are *missing* must own that they are.
    # Downloading vitpose on 2026-08-07 duly turned this red.
    config = _config(tmp_path, t2i_model_root=tmp_path / "no-models")
    row = {c.name: c for c in run_checks(config)}[f"pose model: {spec.label}"]
    assert row.ok is False
    assert row.fatal is False
    assert "bbox" in row.detail
    assert spec.download in row.detail


def test_pose_model_row_goes_green_on_weights(tmp_path):
    spec = model_registry.POSE_MODELS[model_registry.DEFAULT_POSE_MODEL]
    root = tmp_path / "t2i" / spec.dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    checks = {
        c.name: c for c in run_checks(_config(tmp_path, t2i_model_root=tmp_path / "t2i"))
    }
    assert checks[f"pose model: {spec.label}"].ok is True
