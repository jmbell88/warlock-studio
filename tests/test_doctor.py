from __future__ import annotations

import socket

from warlock import doctor, fetch, instance
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


def test_exe_check_reports_a_directory_distinctly_from_a_missing_file(tmp_path):
    """L01. ``Path.exists()`` is true of a directory as well as a file, so a
    broken unpack that left a *folder* named ``trellis-server.exe`` used to
    read exactly like the exe was never staged at all -- same row, same
    sentence, no way to tell "download it" from "your install is damaged"
    apart."""
    exe = tmp_path / "trellis-server.exe"
    exe.mkdir()
    checks = {c.name: c for c in run_checks(_config(tmp_path, trellis_server_exe=exe))}
    row = checks["trellis-server.exe"]
    assert row.ok is False
    assert row.fatal is True
    assert "exists but is not a file" in row.detail
    assert "not found at" not in row.detail


def test_gguf_check_finds_weight_files(tmp_path):
    models = tmp_path / "models"
    models.mkdir(parents=True)
    for name in model_registry.TRELLIS_GGUF_FILES:
        (models / name).write_bytes(b"weights")
    checks = {c.name: c for c in run_checks(_config(tmp_path, trellis_models_dir=models))}
    assert checks["TRELLIS GGUF weights"].ok is True


def test_gguf_check_reports_missing_weights_as_a_pending_install(tmp_path):
    """Absent weights are a download not made, not a broken install.

    This asserted ``fatal is True`` until 2026-09-04. Fatal put a red banner on
    every fresh launch and made ``warlock doctor`` exit 1 on a machine with
    nothing wrong with it -- the weights are a first-run download, and
    Settings -> Models is the button that fixes it. The exe beside it stays
    fatal, because the installer ships that one.
    """
    checks = {c.name: c for c in run_checks(_config(tmp_path))}
    row = checks["TRELLIS GGUF weights"]
    assert row.ok is False
    assert row.fatal is False
    assert row.pending_install is True


def test_a_check_may_not_be_both_broken_and_merely_uninstalled(tmp_path):
    """The two claims are exclusive, and the constructor says so.

    They are read by different consumers -- the exit code and the startup
    banner key on ``fatal``, the health band and Home's row key on
    ``pending_install`` -- so a row claiming both would be reported twice and
    counted twice.
    """
    import pytest

    from warlock.doctor import Check

    with pytest.raises(ValueError):
        Check("both", False, "", fatal=True, pending_install=True)


def test_nothing_is_fatal_on_a_host_that_has_simply_downloaded_nothing(tmp_path):
    """The whole point of the change, asserted end to end.

    ``_config`` points every root at an empty directory, which is exactly the
    shape of a machine five minutes after the installer finishes. The only
    fatal row left is the vendored exe, and that one is present in a real
    install because the installer stages it.
    """
    checks = run_checks(_config(tmp_path))
    fatal = [c.name for c in checks if not c.ok and c.fatal]
    assert fatal == ["trellis-server.exe"], fatal
    assert any(c.pending_install for c in checks)


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
    # Fourteen fixed checks plus one row per registry entry -- derived rather
    # than hardcoded so adding a model doesn't fail an unrelated assertion.
    # The twelfth is "single instance", which RUN-01 added alongside the
    # startup lock: the refusal fires once, and two Warlocks over one job
    # database is exactly what somebody goes looking for in diagnostics
    # afterwards. The thirteenth is "environment", RUN-03's single surface for
    # every WARLOCK_* value that would not parse. The fourteenth is "host
    # memory" (2026-08-21): the queue refuses jobs on a percentage of
    # system-wide commit and that refusal used to arrive with no context, on a
    # machine with 24 GiB of RAM free. The fifteenth is "job database": the
    # store is the app's single point of total failure, and until it had a row
    # a malformed image was the generic startup box on every launch with
    # nothing anywhere naming the file. The sixteenth is "text model": the
    # Flourish prompt's directory probe, a row with no registry entry behind
    # it until the measurement that picks the model pins a revision. The
    # seventeenth is "Muse (dependencies)": the music weights rows answer for
    # the disk, and nothing answered for the ``music`` extra until a packaged
    # build shipped Muse with none of it installed.
    expected = (
        17
        + len(model_registry.BASE_MODELS)
        + len(model_registry.STYLE_LORAS)
        + len(model_registry.IP_ADAPTERS)
        + len(model_registry.CONTROLNETS)
        + len(model_registry.METRIC_MODELS)
        + len(model_registry.MATTING_MODELS)
        + len(model_registry.POSE_MODELS)
        + len(model_registry.MUSIC_MODELS)
        + len(model_registry.SEPARATION_MODELS)
    )
    assert len(run_checks(_config(tmp_path))) == expected


def test_instance_check_probes_the_shared_database_and_model_root(tmp_path, monkeypatch):
    shared_db = tmp_path / "shared" / "jobs.sqlite"
    shared_models = tmp_path / "shared-models"
    first_config = _config(
        tmp_path / "one",
        home=tmp_path / "home-one",
        db_path=shared_db,
        t2i_model_root=shared_models,
    )
    second_config = _config(
        tmp_path / "two",
        home=tmp_path / "home-two",
        db_path=shared_db,
        t2i_model_root=shared_models,
    )
    holder = instance.InstanceLocks(instance.lock_paths(first_config))
    assert holder.acquire()
    try:
        # Simulate ``warlock doctor`` in another process rather than the Studio
        # process that owns ``holder``.
        monkeypatch.setattr(instance, "held_by_us", lambda: False)
        check = doctor._instance_check(second_config)
        assert check.ok is False
        assert "warlock-db.lock" in check.detail
    finally:
        holder.release()


def test_cuda_check_is_not_fatal_when_torch_missing_or_unavailable(tmp_path):
    checks = {c.name: c for c in run_checks(_config(tmp_path))}
    assert checks["CUDA"].fatal is False


def test_disk_check_is_not_fatal(tmp_path):
    checks = {c.name: c for c in run_checks(_config(tmp_path))}
    assert checks["free disk space"].fatal is False
    assert isinstance(checks["free disk space"].ok, bool)


def test_store_check_handles_uri_special_characters_in_the_path(tmp_path):
    """A raw f-string ``file:{path}?mode=ro`` breaks on a path containing '#'
    (truncates to a URI fragment) or '%' (starts a percent-escape) -- and a
    user's home directory is exactly the kind of path that was never shaped
    with URIs in mind. ``Path.as_uri()`` percent-encodes it correctly.
    """
    import sqlite3

    odd = tmp_path / "weird#dir%name"
    odd.mkdir()
    db_path = odd / "jobs.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE jobs (id TEXT)")
    conn.commit()
    conn.close()

    config = _config(tmp_path, db_path=db_path)
    check = doctor._store_check(config)
    assert check.ok, check.detail


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
    (root / spec.dir_name / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
    checks = {c.name: c for c in run_checks(_config(tmp_path, t2i_model_root=root))}
    assert checks[f"image model: {spec.label}"].ok is True


def test_style_lora_check_passes_when_file_present(tmp_path):
    root = tmp_path / "m"
    lora = model_registry.STYLE_LORAS["render3d"]
    (root / "loras").mkdir(parents=True)
    # Non-empty: a zero-byte file is exactly what M04's suspect-files check
    # now catches, and this test is about a genuinely present file passing.
    (root / "loras" / lora.filename).write_bytes(b"x")
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
        path.write_bytes(b"x")
    assert row().ok is False, "a partial download must not read as present"
    last = base / spec.probe[-1]
    last.parent.mkdir(parents=True, exist_ok=True)
    last.write_bytes(b"x")
    assert row().ok is True


def test_turbo_dir_override_is_still_honoured(tmp_path):
    # WARLOCK_T2I_DIR predates the registry; existing setups point it at an
    # arbitrary diffusers dir and must keep working.
    override = tmp_path / "elsewhere"
    (override / "unet").mkdir(parents=True)
    (override / "model_index.json").write_text("{}")
    (override / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
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


def test_gltfpack_check_reports_a_directory_distinctly_from_a_missing_file(
    tmp_path, monkeypatch
):
    """L01, ``gltfpack``'s half."""
    from warlock import doctor
    from warlock.config import Config

    directory = tmp_path / "gltfpack.exe"
    directory.mkdir()
    monkeypatch.setenv("WARLOCK_GLTFPACK", str(directory))
    monkeypatch.setenv("WARLOCK_DATA_DIR", str(tmp_path))
    check = doctor._gltfpack_check(Config())
    assert check.ok is False
    assert "exists but is not a file" in check.detail


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



def _checkpoint(directory):
    """A directory ``fetch.present`` accepts: a config *and* its weights.

    ``present`` names more than one file wherever more than one matters, and
    for the metric, pose and matting tables that means ``config.json`` plus a
    ``*.safetensors``: a config-only directory is a fetch interrupted partway,
    and reading it as installed is precisely what that door exists to stop --
    a green row above a checkpoint that fails at load with the job already
    dispatched.

    These fixtures predate the rule. They wrote the config alone, so six rows
    here went on asserting green against a directory the app had started
    calling absent -- and the suite stayed red on master rather than the
    tightening being finished.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    # Existence is the whole test ``present`` applies (``any(rglob)``); the
    # byte is so nothing downstream is handed a zero-length file.
    (directory / "model.safetensors").write_bytes(b"x")
    return directory


def _matting_weights(tmp_path):
    root = tmp_path / "t2i"
    for spec in model_registry.MATTING_MODELS.values():
        _checkpoint(root / spec.dir_name)
    return root


def test_a_present_matting_model_claims_the_weights_and_not_readiness(tmp_path, monkeypatch):
    # The row states what it has established and no more. It used to stat a
    # directory and ask whether four modules resolve, and said "not checked:
    # whether the model loads" -- which was honest and was the wrong amount of
    # honesty, because a green row above a silent fall-back to the corner fill
    # is the one outcome the row exists to prevent. Since N112 it tries the
    # load; the claim is still exactly what was checked, it is simply a bigger
    # check. ``probe_slow=False`` is the startup path, where the slow probe has
    # deliberately not run yet -- so the row says it is still checking.
    monkeypatch.setattr(doctor, "_missing_modules", lambda names: [])
    monkeypatch.setattr(matting, "last_error", lambda: None)
    root = _matting_weights(tmp_path)
    checks = {
        c.name: c
        for c in run_checks(_config(tmp_path, t2i_model_root=root), probe_slow=False)
    }
    for spec in model_registry.MATTING_MODELS.values():
        row = checks[f"host matting: {spec.label}"]
        assert row.ok is True
        assert "weights present" in row.detail
        assert "still checking" in row.detail
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
        _checkpoint(root / spec.dir_name)
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
    # which is the rule docs/INVARIANTS.md states for warlockc.dll and gltfpack: a test
    # about what happens when weights are *missing* must own that they are.
    # Downloading vitpose on 2026-08-07 duly turned this red.
    config = _config(tmp_path, t2i_model_root=tmp_path / "no-models")
    row = {c.name: c for c in run_checks(config)}[f"pose model: {spec.label}"]
    assert row.ok is False
    assert row.fatal is False
    assert "bbox" in row.detail
    # The *resolved* command, not ``spec.download``. The registry renders the
    # documented default home because it cannot see a Config; doctor can, and a
    # remedy that names a different directory from the one the row above just
    # reported the model missing at is how gigabytes get stranded (DST-02).
    assert fetch.download_text(config, "pose", spec) in row.detail
    assert str(config.t2i_model_root / spec.dir_name) in row.detail


def test_pose_model_row_goes_green_on_weights(tmp_path):
    """``probe_slow=False``, which is the startup path: the weights decide the
    row and the load probe has deliberately not run. With the probe on, a
    ``config.json`` containing ``{}`` is not a checkpoint and the row is
    correctly red -- which is the whole of N112 and is pinned below."""
    spec = model_registry.POSE_MODELS[model_registry.DEFAULT_POSE_MODEL]
    _checkpoint(tmp_path / "t2i" / spec.dir_name)
    checks = {
        c.name: c
        for c in run_checks(
            _config(tmp_path, t2i_model_root=tmp_path / "t2i"), probe_slow=False
        )
    }
    assert checks[f"pose model: {spec.label}"].ok is True


def test_a_checkpoint_that_will_not_load_is_red_once_the_probe_runs(tmp_path):
    """N112. The failure mode both model rows exist for: every file in place,
    a green row, and the pipeline silently falling back on every job. Only an
    attempted load settles it, so the row attempts one -- off the startup path
    and once per process."""
    spec = model_registry.POSE_MODELS[model_registry.DEFAULT_POSE_MODEL]
    _checkpoint(tmp_path / "t2i" / spec.dir_name)
    row = {
        c.name: c
        for c in run_checks(_config(tmp_path, t2i_model_root=tmp_path / "t2i"))
    }[f"pose model: {spec.label}"]
    assert row.ok is False
    assert row.fatal is False  # a missing pose model costs joint placement, not a job


def test_the_load_probe_is_keyed_on_the_weights_directory(tmp_path, monkeypatch):
    """Not on the kind. The bpy answer can be a bare global because it is a
    fact about the interpreter; this is a fact about a path, and
    ``WARLOCK_T2I_ROOT`` moves it -- so a kind-keyed cache would answer the
    second config with the first one's result.

    The probe itself is stubbed. Letting the real one run spawned a child
    interpreter and a torch import to reach an answer this test never reads --
    five seconds to establish a dictionary lookup.

    The cache is warmed by *running the checks* against ``good`` rather than by
    writing an entry into ``_probes`` by hand, and that is the whole difference
    between this test and the one it replaces. Seeding the dict directly pins
    the key's current shape: mutate the production key to ``which`` alone and
    the hand-written tuple simply stops matching, so the lookup misses, the
    stub runs, and the old test passed the very mutation it was written to
    catch. Warming it through the real path means the cache is keyed however
    the code chooses -- and a kind-keyed one then answers ``bad`` with
    ``good``'s ``True``, which is exactly what the last assertion refuses.
    """
    spec = model_registry.POSE_MODELS[model_registry.DEFAULT_POSE_MODEL]
    good, bad = tmp_path / "good", tmp_path / "bad"
    for root in (good, bad):
        _checkpoint(root / spec.dir_name)

    # The global survives the test that filled it, so it is restored rather
    # than left holding this tmp_path's answers for whatever runs next.
    monkeypatch.setattr(doctor, "_probes", {})
    probed = []

    def stub(_which, path):
        probed.append(path)
        # ``good`` loads and ``bad`` does not: two different answers is what
        # makes reusing one for the other detectable at all.
        return (good in path.parents, "stub")

    monkeypatch.setattr(doctor, "_run_load_probe", stub)

    def row(root):
        checks = {c.name: c for c in run_checks(_config(tmp_path, t2i_model_root=root))}
        return checks[f"pose model: {spec.label}"]

    assert row(good).ok is True
    assert len(probed) == 1
    # Same kind, different directory. A cache keyed on the kind alone would
    # hand back ``good``'s answer here without asking.
    assert row(bad).ok is False
    assert len(probed) == 2


# --- N112: the load probe, and why it is a child process --------------------


def test_the_load_probe_child_reports_a_failure_as_a_sentence(tmp_path):
    """It must never raise out of the child: a probe that does turns a red row
    into a traceback in a log nobody is reading yet."""
    from warlock.pipelines import loadprobe

    ok, detail = loadprobe.probe("pose", tmp_path / "nothing-here")
    assert ok is False
    assert detail  # names the exception type as well as its message
    assert ":" in detail


def test_the_load_probe_child_refuses_an_unknown_kind(capsys):
    from warlock.pipelines import loadprobe

    assert loadprobe.main(["nonsense", "x"]) == 2
    assert capsys.readouterr().out.startswith("fail usage:")


def test_a_probe_that_prints_nothing_is_a_failure_naming_what_it_said(monkeypatch, tmp_path):
    """The child can die before it prints -- an OOM kill, a DLL that will not
    load. Its stderr is the only thing that knows why."""
    from types import SimpleNamespace

    weights = tmp_path / "weights"
    weights.mkdir()
    monkeypatch.setattr(
        doctor.winjob, "run",
        lambda *a, **k: SimpleNamespace(stdout="", stderr="ImportError: DLL load failed"),
    )
    ok, detail = doctor._run_load_probe("matting", weights)
    assert ok is False
    assert "DLL load failed" in detail


def test_the_probe_does_not_run_at_all_without_weights(monkeypatch, tmp_path):
    """No spawn, no seconds, no torch import -- the cheap answer first, which is
    the ordering every model path in the repo follows."""
    def boom(*a, **k):
        raise AssertionError("should not have spawned anything")

    monkeypatch.setattr(doctor.winjob, "run", boom)
    ok, detail = doctor._run_load_probe("matting", tmp_path / "absent")
    assert ok is False
    assert "not on disk" in detail


def test_the_gguf_remedy_downloads_into_the_configured_models_dir(tmp_path):
    """The command is pasted from whatever directory the user's shell happens
    to be in, so a relative ``--local-dir`` put 16 GB somewhere Warlock never
    inspects and left the fatal row standing (audit 2026-08-19)."""
    config = _config(tmp_path)
    checks = {c.name: c for c in run_checks(config)}
    assert (
        f"--local-dir {fetch.quote_for_shell(config.trellis_models_dir)}"
        in checks["TRELLIS GGUF weights"].detail
    )


def test_the_exe_remedy_names_the_exact_release_asset_and_its_digest(tmp_path):
    """The binary is the one unsigned third-party download in the setup, so
    the remedy pins the direct v0.6.0 asset URL and the SHA-256 GitHub
    publishes for it -- a bare releases page gave the user nothing to verify
    a download against."""
    checks = {c.name: c for c in run_checks(_config(tmp_path))}
    detail = checks["trellis-server.exe"].detail
    assert (
        "https://github.com/pwilkin/trellis.cpp/releases/download/v0.6.0/"
        "trellis-cuda-windows-x64.zip"
    ) in detail
    assert "4d08ab27e83094035fd8349aaf34d3460738df0466ef9c4991ddd958c0344bc2" in detail


# --- host commit -------------------------------------------------------------


def _sysmem(total: float, limit: float):
    from warlock import memlog

    return memlog.SystemMemory(commit_total=total, commit_limit=limit)


def test_commit_headroom_row_warns_when_the_pagefile_is_the_constraint(monkeypatch):
    """The shape of the machine that prompted this row, 2026-08-21.

    63.5 GiB of RAM, a 14.2 GiB pagefile, so a 77.7 GiB commit limit -- and the
    queue refusing every job at the 90% ceiling **with 24 GiB of RAM free**.
    Nothing in the app said why, and the honest answer is not "close some
    applications": it is that the commit limit is barely above physical RAM, so
    normal GPU-driver and allocator commit puts the fraction at the wall while
    memory itself is plentiful.
    """
    from warlock import doctor, memlog

    monkeypatch.setattr(memlog, "system_memory", lambda: _sysmem(75.1, 77.7))
    monkeypatch.setattr(doctor, "_physical_ram_gib", lambda: 63.5)
    row = doctor._commit_check()
    assert row.ok is False
    assert row.fatal is False, "an informative row, never a startup blocker"
    assert "pagefile" in row.detail.lower(), row.detail
    assert "97" in row.detail or "96" in row.detail, row.detail


def test_commit_headroom_row_is_quiet_on_a_healthy_machine(monkeypatch):
    """A roomy limit and a modest charge is the ordinary case and must not
    grow a warning row -- the doctor is read by eye and a row that is always
    amber is a row nobody reads."""
    from warlock import doctor, memlog

    monkeypatch.setattr(memlog, "system_memory", lambda: _sysmem(30.0, 128.0))
    monkeypatch.setattr(doctor, "_physical_ram_gib", lambda: 64.0)
    row = doctor._commit_check()
    assert row.ok is True
    assert row.fatal is False


def test_commit_headroom_row_does_not_blame_the_pagefile_when_it_is_generous(
    monkeypatch,
):
    """High commit with a large pagefile is a different diagnosis: something is
    genuinely using the memory, and telling the user to grow a pagefile that is
    already 2x their RAM would be advice that does nothing."""
    from warlock import doctor, memlog

    monkeypatch.setattr(memlog, "system_memory", lambda: _sysmem(180.0, 192.0))
    monkeypatch.setattr(doctor, "_physical_ram_gib", lambda: 64.0)
    row = doctor._commit_check()
    assert row.ok is False
    assert "pagefile" not in row.detail.lower(), row.detail


def test_commit_headroom_row_says_so_when_it_cannot_measure(monkeypatch):
    """Off Windows, or when the call fails. The row must not claim a healthy
    machine it did not read -- every other doctor row states its own
    unavailability rather than passing by default."""
    from warlock import doctor, memlog

    monkeypatch.setattr(memlog, "system_memory", lambda: None)
    row = doctor._commit_check()
    assert row.ok is True
    assert row.fatal is False
    assert "not measured" in row.detail.lower() or "unavailable" in row.detail.lower()


def test_the_commit_row_is_in_the_volatile_set(tmp_path):
    """Commit changes minute to minute -- it is exactly a volatile row, and a
    static one would report the figure at startup forever."""
    from warlock import doctor

    config = _config(tmp_path)
    names = [c.name for c in doctor.volatile_checks(config)]
    assert "host memory" in names
    static = [c.name for c in doctor.static_checks(config, probe_slow=False)]
    assert "host memory" not in static


# --- a card the driver can see, on a host with no torch (2026-09-04) ---------


def test_the_vram_row_is_not_fatal_on_a_carded_host_that_has_no_torch(tmp_path, monkeypatch):
    """The prerequisite for making ``text2image`` a downloadable pack.

    ``vram.probe()`` returned None the moment torch was absent, and
    ``_vram_check``'s ``probe=True`` path never consults ``device_memory``, so
    it took the "no CUDA device at all" branch and reported **fatal**: *"no
    CUDA device means 3D reconstruction cannot run at all"* -- on a working
    RTX machine. Nothing about the card had changed; only the Python package
    set had, and trellis-server does not use torch in the first place.

    Pinned to a fake reading rather than the real driver so the verdict does
    not move with the machine running the suite.
    """
    from warlock import vram

    reading = vram.DeviceMemory(total_gib=32.0, free_gib=30.0, name="NVML GPU")
    monkeypatch.setattr(vram, "live_memory", lambda: reading)
    monkeypatch.setattr(vram, "probe", lambda: vram.device_memory())
    monkeypatch.delitem(__import__("sys").modules, "torch", raising=False)

    row = doctor._vram_check(_config(tmp_path), probe=True)
    assert row.fatal is False
    assert row.ok is True
    assert "no CUDA device" not in row.detail


def test_the_vram_row_is_still_fatal_when_there_really_is_no_card(tmp_path, monkeypatch):
    """The other direction, so the fix above cannot quietly disarm the row.

    A host with no card genuinely cannot reconstruct -- there is no CPU
    fallback -- and an amber "admission control is off" row reads as good news.
    """
    from warlock import vram

    monkeypatch.setattr(vram, "live_memory", lambda: None)
    monkeypatch.setattr(vram, "probe", lambda: None)
    monkeypatch.setattr(vram, "device_memory", lambda: None)

    row = doctor._vram_check(_config(tmp_path), probe=True)
    assert row.fatal is True
    assert "no CUDA device" in row.detail


# --- M04: fetch.present is presence-only, and every registry row now runs the
# same suspect_files check present alone cannot make. Reproduced per kind
# rather than asserted once: ``_registry_row`` is the shared choke point, but
# before this fix each of these called ``fetch.present`` directly and skipped
# it, so a per-kind test is what proves the fix reaches every one of them
# rather than only the base-model row it was first written for.


def test_engine_row_treats_zero_byte_gguf_files_as_missing_not_healthy(tmp_path):
    """The reproduction named in the finding: all ten engine probe files
    present and empty, Doctor green regardless. ``_gguf_check`` calls
    ``fetch.present`` (an ``is_file()`` sweep, blind to size) and used to stop
    there."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    for name in model_registry.TRELLIS_GGUF_FILES:
        (models_dir / name).touch()  # exists, and is zero bytes
    checks = {c.name: c for c in run_checks(_config(tmp_path, trellis_models_dir=models_dir))}
    row = checks["TRELLIS GGUF weights"]
    assert row.ok is False
    assert "empty" in row.detail


def test_style_lora_row_treats_a_zero_byte_file_as_missing_not_healthy(tmp_path):
    root = tmp_path / "m"
    lora = model_registry.STYLE_LORAS["render3d"]
    (root / "loras").mkdir(parents=True)
    (root / "loras" / lora.filename).touch()
    checks = {c.name: c for c in run_checks(_config(tmp_path, t2i_model_root=root))}
    row = checks[f"style LoRA: {lora.label}"]
    assert row.ok is False
    assert "empty" in row.detail


def test_ip_adapter_row_treats_a_zero_byte_weight_as_missing_not_healthy(tmp_path):
    config = _config(tmp_path, t2i_model_root=tmp_path / "t2i")
    adapter = model_registry.IP_ADAPTERS["plus"]
    root = config.t2i_model_root / adapter.dir_name
    weights = root / adapter.subfolder / adapter.weight_name
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.touch()
    (root / adapter.image_encoder_dir).mkdir(parents=True, exist_ok=True)
    (root / adapter.image_encoder_dir / "config.json").write_text("{}")

    checks = {c.name: c for c in run_checks(config)}
    row = checks[f"IP-Adapter: {adapter.label}"]
    assert row.ok is False
    assert "empty" in row.detail


def test_controlnet_row_treats_a_zero_byte_weight_as_missing_not_healthy(tmp_path):
    config = _config(tmp_path, t2i_model_root=tmp_path / "t2i")
    cn = next(iter(model_registry.CONTROLNETS.values()))
    base = config.t2i_model_root / cn.dir_name
    base.mkdir(parents=True)
    (base / "config.json").write_text("{}")
    variant = f".{cn.variant}" if cn.variant else ""
    (base / f"diffusion_pytorch_model{variant}.safetensors").touch()

    checks = {c.name: c for c in run_checks(config)}
    row = checks[f"ControlNet: {cn.label}"]
    assert row.ok is False
    assert "empty" in row.detail


def test_metric_row_treats_a_zero_byte_checkpoint_as_missing_not_healthy(tmp_path):
    spec = next(iter(model_registry.METRIC_MODELS.values()))
    root = tmp_path / "t2i"
    directory = root / spec.dir_name
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    (directory / "model.safetensors").touch()

    checks = {c.name: c for c in run_checks(_config(tmp_path, t2i_model_root=root))}
    row = checks[f"metric model: {spec.label}"]
    assert row.ok is False
    assert "empty" in row.detail


def test_music_row_treats_a_zero_byte_weight_as_missing_not_healthy(tmp_path):
    """``present`` reads ACE-Step off its ``config.json`` probe list, but the
    row's usability check is the same ``rglob``-for-weights sweep every
    non-engine kind shares -- so the file that has to be zero-length to prove
    it is one of the real weights beside a probed config, not the config
    itself."""
    spec = model_registry.MUSIC_MODELS[model_registry.DEFAULT_MUSIC_MODEL]
    root = tmp_path / "t2i"
    base = root / spec.dir_name
    for name in spec.probe:
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (base / "ace_step_transformer" / "diffusion_pytorch_model.safetensors").touch()

    checks = {c.name: c for c in run_checks(_config(tmp_path, t2i_model_root=root))}
    row = checks[f"music model: {spec.label}"]
    assert row.ok is False
    assert "empty" in row.detail


def test_separation_row_treats_a_zero_byte_checkpoint_as_missing_not_healthy(tmp_path):
    spec = model_registry.SEPARATION_MODELS[model_registry.DEFAULT_SEPARATION]
    root = tmp_path / "t2i"
    for name in spec.probe:
        path = root / spec.dir_name / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    checks = {c.name: c for c in run_checks(_config(tmp_path, t2i_model_root=root))}
    row = checks[f"stem separation: {spec.label}"]
    assert row.ok is False
    assert "empty" in row.detail


def test_pose_row_treats_a_zero_byte_checkpoint_as_missing_not_healthy(tmp_path):
    spec = model_registry.POSE_MODELS[model_registry.DEFAULT_POSE_MODEL]
    directory = tmp_path / "t2i" / spec.dir_name
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").touch()

    checks = {
        c.name: c
        for c in run_checks(
            _config(tmp_path, t2i_model_root=tmp_path / "t2i"), probe_slow=False
        )
    }
    row = checks[f"pose model: {spec.label}"]
    assert row.ok is False
    assert "empty" in row.detail


def test_matting_row_treats_a_zero_byte_checkpoint_as_missing_not_healthy(tmp_path):
    spec = next(iter(model_registry.MATTING_MODELS.values()))
    directory = tmp_path / "t2i" / spec.dir_name
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").touch()

    checks = {
        c.name: c
        for c in run_checks(
            _config(tmp_path, t2i_model_root=tmp_path / "t2i"), probe_slow=False
        )
    }
    row = checks[f"host matting: {spec.label}"]
    assert row.ok is False
    assert "empty" in row.detail
