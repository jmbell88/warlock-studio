"""The F-section of NEXT_ROADMAP Phase 2: onboarding, doctor and remedies.

The theme is that a failure the app already knows about should be answerable
without leaving the app. Every test here is about a *remedy* being present and
reachable, not about a check being right -- ``tests/test_doctor.py`` owns that.
"""

from __future__ import annotations

import pytest

from warlock import doctor
from warlock.config import Config
from warlock.service.errors import Invalid

# --- F54: the two fatal rows now name a way forward -------------------------


def _config(tmp_path) -> Config:
    return Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "no-models",
        t2i_model_root=tmp_path / "t2i",
        bench_dir=tmp_path / "bench",
    )


def test_every_failing_check_names_a_remedy(tmp_path):
    """The rule the non-fatal model rows have always followed, extended to the
    two rows that actually stop the app (F54).

    A remedy is a command, an install instruction or a stated consequence --
    what it may not be is a path and nothing else, which is what the exe and
    GGUF rows said before.
    """
    config = _config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    checks = doctor.static_checks(config, probe_slow=False)
    fatal = [c for c in checks if c.fatal and not c.ok]
    assert fatal, "the fixture is meant to be missing both fatal prerequisites"
    for check in fatal:
        assert any(
            word in check.detail
            for word in ("download", "unpack", "install", "WARLOCK_")
        ), f"{check.name} names no remedy: {check.detail!r}"


def test_the_gguf_remedy_is_the_command_from_the_install_instructions(tmp_path):
    """Spelled once. The README and the manual carry the same line, and a row
    that invented its own would send a user to a different repo."""
    config = _config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    row = next(
        c for c in doctor.static_checks(config, probe_slow=False)
        if c.name == "TRELLIS GGUF weights"
    )
    assert doctor.TRELLIS_GGUF_HINT in row.detail
    assert "ilintar/trellis2-gguf" in doctor.TRELLIS_GGUF_HINT


# --- F55: missing weights are refused at the door ---------------------------


def test_a_text_job_whose_checkpoint_is_absent_is_refused_with_its_command(svc):
    """Refused before the queue, in the shape ``check_vram`` set: name the
    problem and a thing the user can do about it."""
    from warlock import fetch, models
    from warlock.service import jobs as svc_jobs

    spec = models.BASE_MODELS[models.DEFAULT_BASE_MODEL]
    (fetch.base_model_dir(svc.config, spec) / "model_index.json").unlink()

    with pytest.raises(Invalid) as caught:
        svc_jobs.create_job(svc, kind="text", prompt="a rock", output="reference")
    message = caught.value.message
    assert caught.value.field == "base_model"
    # The command resolved against *this* service's config, not the registry's
    # default-home rendering: a refusal that names one directory and a remedy
    # that names another is DST-02, and this door is one of the places a user
    # copies the line straight out of.
    assert fetch.download_text(svc.config, "base", spec) in message
    assert str(fetch.base_model_dir(svc.config, spec)) in message
    # And it leads with the route that does not need a terminal.
    assert "Settings" in message


def test_a_missing_style_lora_is_refused_rather_than_silently_skipped(svc):
    """The one that would otherwise fail *quietly*: ``_load_loras`` skips a
    missing style adapter, so the job would finish looking wrong while its
    params claimed a style that never ran -- and that row would then join the
    findings corpus as evidence about it."""
    from warlock import fetch, models
    from warlock.service import jobs as svc_jobs

    key, lora = next(iter(models.STYLE_LORAS.items()))
    (svc.config.t2i_model_root / "loras" / lora.filename).unlink()

    with pytest.raises(Invalid) as caught:
        svc_jobs.create_job(
            svc, kind="text", prompt="a rock", output="reference",
            guidance_fields={"style_lora": key},
        )
    assert caught.value.field == "style_lora"
    assert fetch.download_text(svc.config, "lora", lora) in caught.value.message
    assert str(svc.config.t2i_model_root / "loras") in caught.value.message


def test_an_image_job_is_not_asked_about_image_models(svc):
    """An image job never touches SDXL, so its checkpoint is irrelevant --
    and refusing one for a missing image model would be a refusal the user
    could not act on, since no control of theirs chose it."""
    from warlock import fetch, models
    from warlock.service import jobs as svc_jobs

    for spec in models.BASE_MODELS.values():
        index = fetch.base_model_dir(svc.config, spec) / "model_index.json"
        if index.exists():
            index.unlink()

    png = _tiny_png()
    job = svc_jobs.create_job(svc, kind="image", image=png)
    assert job["id"]


def test_a_refused_job_leaves_nothing_behind(svc):
    """The ordering rule ``create_job`` states: the guard sits with the other
    door checks, before ``input.png`` is written."""
    from warlock import fetch, models
    from warlock.service import jobs as svc_jobs

    spec = models.BASE_MODELS[models.DEFAULT_BASE_MODEL]
    (fetch.base_model_dir(svc.config, spec) / "model_index.json").unlink()
    before = set(svc.config.data_dir.glob("*"))

    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text", prompt="a rock", output="reference")

    assert set(svc.config.data_dir.glob("*")) == before
    assert svc.store.list(10) == []


def _tiny_png() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "grey").save(buf, "PNG")
    return buf.getvalue()


# --- F57: the three surfaces that lead to troubleshooting -------------------


def test_troubleshooting_is_named_once_and_resolves():
    from warlock.studio.manual import loader, targets

    chapter, anchor = targets.TROUBLESHOOTING
    assert anchor is None
    assert chapter in {c.key for c in loader.chapters()}


def test_troubleshooting_is_not_a_help_target():
    """It is the same shape and deliberately not in that dict: HELP_TARGETS is
    asserted against the pane (?) call sites in both directions, and none of
    the three surfaces that lead here is a pane with a (?)."""
    from warlock.studio.manual.targets import HELP_TARGETS

    assert "diagnostics" not in HELP_TARGETS


# --- F59: Dismiss puts it away, it does not forget it -----------------------


def test_dismissing_the_banner_keeps_the_text_reachable():
    from warlock.studio.state import AppState

    state = AppState()
    state.note_error("trellis-server.exe: not found at C:/x")
    state.note_error("The GPU worker stopped: boom. Restart Warlock.")
    state.dismiss_errors()

    assert state.errors == []
    assert len(state.dismissed_errors) == 2
    assert "trellis-server.exe" in state.dismissed_errors[0]


def test_dismissing_twice_does_not_duplicate():
    from warlock.studio.state import AppState

    state = AppState()
    state.note_error("boom")
    state.dismiss_errors()
    state.note_error("boom")
    state.dismiss_errors()
    assert state.dismissed_errors == ["boom"]
