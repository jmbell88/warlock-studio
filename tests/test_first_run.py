"""Installer first-run setup is a one-shot view over startup facts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock import doctor, fetch, models, vram
from warlock.service import downloads
from warlock.studio.panes import app_settings, first_run
from warlock.studio.state import AppState


def _ctx(svc, *, checks=(), rows=(), plan=None, rigging=False):
    svc.vram_plan = plan
    return SimpleNamespace(
        svc=svc,
        runtime=SimpleNamespace(
            checks=list(checks),
            device_memory=vram.DeviceMemory(24.0, 20.0, "Test GPU"),
        ),
        state=AppState(),
        model_rows=list(rows),
        model_picks=set(),
        rigging_available=rigging,
        first_run=True,
        first_run_info={},
        gpu_name="",
        toast=lambda *_args: None,
    )


def test_a_marker_hides_the_overlay_on_the_next_start(svc, monkeypatch):
    ctx = _ctx(svc)
    assert first_run.pending(svc.config)
    monkeypatch.setattr(first_run.imgui, "close_current_popup", lambda: None)
    assert first_run.dismiss(ctx)
    assert first_run.marker_path(svc.config).is_file()
    assert not first_run.pending(svc.config)
    assert ctx.first_run is False


def test_the_snapshot_uses_startup_hardware_and_a_deduped_download_plan(
    svc, monkeypatch
):
    resolved = vram.plan(exclusive=False, total_gib=24.0)
    checks = (
        doctor.Check("CUDA", True, "available", fatal=False),
        doctor.Check("VRAM budget", True, resolved.reason, fatal=False),
    )
    rows = downloads.rows(svc)
    monkeypatch.setattr(fetch, "disk_refusal", lambda _jobs: None)
    ctx = _ctx(svc, checks=checks, rows=rows, plan=resolved, rigging=True)
    info = first_run.snapshot(ctx)
    expected = fetch.total_gib(
        fetch.plan(svc.config, [fetch.find(key) for key in first_run.REQUIRED_ROWS])
    )
    assert info["gpu_name"] == "Test GPU"
    assert info["vram_total_gib"] == pytest.approx(24.0)
    assert info["three_d"]["ready"] is True
    assert info["images"]["ready"] is True
    assert info["rigging"]["ready"] is True
    assert info["total_gib"] == pytest.approx(expected)
    assert info["total_gib"] == pytest.approx(
        models.ENGINE_MODELS["trellis_gguf"].fetch[0].size_gib
        + models.BASE_MODELS[models.DEFAULT_BASE_MODEL].fetch[0].size_gib
    )


def test_image_and_reconstruction_verdicts_have_separate_requirements(svc):
    plan = vram.plan(exclusive=False)
    ctx = _ctx(svc, checks=(), rows=downloads.rows(svc), plan=plan)
    info = first_run.snapshot(ctx)
    assert info["three_d"]["ready"] is False
    assert info["images"]["ready"] is True


def test_download_handoff_unions_picks_and_opens_settings_models(svc, monkeypatch):
    ctx = _ctx(svc)
    ctx.model_picks.add("lora:pixelxl")
    monkeypatch.setattr(first_run, "dismiss", lambda _ctx: True)
    first_run.download_models(ctx)
    assert ctx.model_picks == {"lora:pixelxl", *first_run.REQUIRED_ROWS}
    assert ctx.state.mode == "settings"
    assert ctx.state.preview[app_settings.CATEGORY_SLOT] == "models"
