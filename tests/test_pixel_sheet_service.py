"""Queuing a pixel restyle of a rendered sprite sheet.

Every refusal here is checked at the door rather than in the worker, for the
reason create_sheet already states: an unrenderable request should cost the
request, not a place in the queue and a minute of GPU.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from warlock import rigging
from warlock.service import jobs as svc_jobs
from warlock.service import sheets as svc_sheets
from warlock.service.errors import Invalid, NotFound


def _sheet_on_disk(svc, *, frame_size=128, columns=8, rows=1):
    """A finished job with one rendered sheet beside its mesh."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a knight")["id"]
    svc.store.set_status(job_id, "done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"glb")

    sheet_id = rigging.new_id()
    png = rigging.sheet_png_path(job_dir, sheet_id)
    png.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (frame_size * columns, frame_size * rows), (0, 0, 0, 0)).save(png)
    meta = {
        "version": 1,
        "id": sheet_id,
        "name": "turnaround",
        "source_job": job_id,
        "created": 1.0,
        "image": png.name,
        "frame_size": frame_size,
        "columns": columns,
        "rows": rows,
        "width": frame_size * columns,
        "height": frame_size * rows,
        "yaws": [i * 45.0 for i in range(columns)],
        "poses": [{"id": None, "name": "rest"}],
        "cells": [],
    }
    rigging.sheet_path(job_dir, sheet_id).write_text(json.dumps(meta), encoding="utf-8")
    return job_id, sheet_id


def test_a_restyle_is_queued_as_its_own_job_against_the_render(svc):
    job_id, sheet_id = _sheet_on_disk(svc)

    out = svc_sheets.create_pixel_sheet(svc, job_id, sheet_id)

    row = svc.store.get(out["id"])
    assert row["kind"] == "pixel_sheet"
    assert row["params"]["source_job"] == job_id
    assert row["params"]["sheet_id"] == sheet_id
    # Every param is an input, so a rerun copies them verbatim -- which is why
    # none of them joins DERIVED_PARAMS. The restyle's own recipe goes in the
    # pixel sidecar instead.
    assert row["params"]["logical_size"] == 32
    assert row["params"]["seed"] is not None


def test_a_missing_sheet_is_a_404_not_a_queued_job(svc):
    job_id, _sheet_id = _sheet_on_disk(svc)
    with pytest.raises(NotFound):
        svc_sheets.create_pixel_sheet(svc, job_id, rigging.new_id())


def test_a_sheet_with_no_atlas_yet_is_not_restylable(svc):
    # The sidecar is the completion marker, but a half-cleaned directory can
    # carry one with no PNG.
    job_id, sheet_id = _sheet_on_disk(svc)
    rigging.sheet_png_path(svc.job_dir(job_id), sheet_id).unlink()
    with pytest.raises(NotFound):
        svc_sheets.create_pixel_sheet(svc, job_id, sheet_id)


def test_a_sheet_too_wide_for_one_generation_is_refused_with_a_remedy(svc):
    job_id, sheet_id = _sheet_on_disk(svc, frame_size=256)
    with pytest.raises(Invalid) as excinfo:
        svc_sheets.create_pixel_sheet(svc, job_id, sheet_id)
    assert "2048px" in excinfo.value.message
    assert "128px or smaller" in excinfo.value.message


def test_a_logical_size_that_does_not_divide_the_cell_is_refused(svc):
    job_id, sheet_id = _sheet_on_disk(svc, frame_size=100, columns=8)
    with pytest.raises(Invalid, match="divide"):
        svc_sheets.create_pixel_sheet(svc, job_id, sheet_id, logical_size=32)


def test_the_size_and_colour_choices_are_the_offered_ones(svc):
    job_id, sheet_id = _sheet_on_disk(svc)
    with pytest.raises(Invalid, match="logical_size"):
        svc_sheets.create_pixel_sheet(svc, job_id, sheet_id, logical_size=33)
    with pytest.raises(Invalid, match="colors"):
        svc_sheets.create_pixel_sheet(svc, job_id, sheet_id, colors=7)


def test_strength_is_bounded_at_both_ends(svc):
    from warlock import models

    job_id, sheet_id = _sheet_on_disk(svc)
    for value in (0.1, 0.9):
        with pytest.raises(Invalid, match="strength"):
            svc_sheets.create_pixel_sheet(svc, job_id, sheet_id, strength=value)
    out = svc_sheets.create_pixel_sheet(
        svc, job_id, sheet_id, strength=models.IMG2IMG_STRENGTH_MAX
    )
    assert svc.store.get(out["id"])["params"]["strength"] == models.IMG2IMG_STRENGTH_MAX


def test_the_pixel_pair_is_invisible_to_the_sheet_listing(svc):
    """`<id>.pixel` fails is_valid_id, so list_sheets skips it without needing
    to know this feature exists."""
    job_id, sheet_id = _sheet_on_disk(svc)
    job_dir = svc.job_dir(job_id)
    rigging.sheet_pixel_png_path(job_dir, sheet_id).write_bytes(b"png")
    rigging.sheet_pixel_path(job_dir, sheet_id).write_text("{}", encoding="utf-8")

    listed = svc_sheets.list_sheets(svc, job_id)["sheets"]
    assert [s["id"] for s in listed] == [sheet_id]


def test_the_png_is_not_served_until_the_sidecar_lands(svc):
    job_id, sheet_id = _sheet_on_disk(svc)
    job_dir = svc.job_dir(job_id)
    rigging.sheet_pixel_png_path(job_dir, sheet_id).write_bytes(b"png")
    with pytest.raises(NotFound):
        svc_sheets.sheet_pixel_png(svc, job_id, sheet_id)

    rigging.sheet_pixel_path(job_dir, sheet_id).write_text(
        json.dumps({"version": 1}), encoding="utf-8"
    )
    assert svc_sheets.sheet_pixel_png(svc, job_id, sheet_id).exists()
    assert svc_sheets.get_pixel_sheet(svc, job_id, sheet_id)["version"] == 1


def test_deleting_a_sheet_takes_its_restyle_with_it(svc):
    # The restyle depicts this render and nothing else; leaving it behind is a
    # sprite sheet of a sheet that is gone.
    job_id, sheet_id = _sheet_on_disk(svc)
    job_dir = svc.job_dir(job_id)
    rigging.sheet_pixel_png_path(job_dir, sheet_id).write_bytes(b"png")
    rigging.sheet_pixel_path(job_dir, sheet_id).write_text("{}", encoding="utf-8")

    svc_sheets.delete_sheet(svc, job_id, sheet_id)

    assert not rigging.sheet_pixel_path(job_dir, sheet_id).exists()
    assert not rigging.sheet_pixel_png_path(job_dir, sheet_id).exists()
    assert not rigging.sheet_path(job_dir, sheet_id).exists()


def test_a_malformed_sheet_id_never_reaches_the_filesystem(svc):
    job_id, _sheet_id = _sheet_on_disk(svc)
    for bad in ("../../etc/passwd", "NOTHEX"):
        with pytest.raises((NotFound, Invalid)):
            svc_sheets.create_pixel_sheet(svc, job_id, bad)


def test_a_restyle_whose_pixel_lora_is_missing_is_refused(svc, monkeypatch):
    """The door checked that the LoRA *fits* the base, never that it is here.

    Missing, the worker's own tolerance takes over: it logs, restyles bare, and
    writes a sidecar naming a LoRA that never loaded -- so the job finishes
    looking like a plain img2img pass rather than pixel art, which is the
    feature not happening rather than a plainer version of it.
    """
    from warlock import fetch

    job_id, sheet_id = _sheet_on_disk(svc)
    monkeypatch.setattr(fetch, "present", lambda *a, **k: False)
    with pytest.raises(Invalid) as exc:
        svc_sheets.create_pixel_sheet(svc, job_id, sheet_id)
    assert "hf download" in str(exc.value)


def test_a_restyle_whose_checkpoint_is_missing_is_refused(svc, monkeypatch):
    from warlock import fetch

    job_id, sheet_id = _sheet_on_disk(svc)
    monkeypatch.setattr(fetch, "base_model_state", lambda *a, **k: (False, None))
    with pytest.raises(Invalid) as exc:
        svc_sheets.create_pixel_sheet(svc, job_id, sheet_id)
    assert exc.value.field == "base_model"
