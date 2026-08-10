"""The pixel restyle's LoRA pairing: refused at the door, dropped by the
worker, and recorded only when it ran.

The restyle's identity is one SDXL-fitted adapter (models.PIXEL_SHEET_LORA).
create_pixel_sheet writes the one base it pairs with today, so the door
refusal guards a pair of constants -- deliberately, because params outlive the
service that wrote them: a stored row naming a FLUX.2 klein base is a *known*
key that queue._pixel_sheet resolves without falling back. Before the worker
grew its own drop, that row generated bare (``_load_loras`` filters a foreign
adapter off the pipe) while the sidecar still recorded the style as part of
the recipe -- a claim about pixels the adapter never touched.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time

import pytest
from PIL import Image

from warlock import models, rigging
from warlock.config import Config
from warlock.db import JobStore
from warlock.queue import Worker
from warlock.service import Invalid
from warlock.service import jobs as svc_jobs
from warlock.service import sheets as svc_sheets


def test_the_shipped_pair_actually_fits():
    """The refusal below must stay unreachable with the real registry: the
    base create_pixel_sheet writes is one the pixel LoRA loads onto. If this
    fails, every restyle in the app is being refused at the door."""
    base = models.BASE_MODELS["sdxl_cfg"]
    lora = models.STYLE_LORAS[models.PIXEL_SHEET_LORA]
    assert models.lora_fits(base, lora)


# --- the door ----------------------------------------------------------------


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


def test_a_base_the_pixel_lora_does_not_fit_is_refused_at_the_door(svc, monkeypatch):
    """Reached by re-familying the adapter, because the pair really is two
    constants today -- what is under test is that the guard exists and names
    the control, not that the shipped registry is broken."""
    monkeypatch.setitem(
        models.STYLE_LORAS,
        models.PIXEL_SHEET_LORA,
        dataclasses.replace(
            models.STYLE_LORAS[models.PIXEL_SHEET_LORA],
            family=models.FAMILY_FLUX2_KLEIN,
        ),
    )
    job_id, sheet_id = _sheet_on_disk(svc)
    with pytest.raises(Invalid) as exc:
        svc_sheets.create_pixel_sheet(svc, job_id, sheet_id)
    # The field names the control the sentence mentions, and the message
    # carries the remedy -- the bases the adapter does fit.
    assert exc.value.field == "base_model"
    assert "pick one of" in str(exc.value)


# --- the worker --------------------------------------------------------------


@pytest.fixture
def worker(tmp_path, fake_pipelines):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    yield w
    store.close()


def _rendered_sheet(worker, source, *, frame_size=128, columns=8, rows=1):
    """A finished render on disk, with a subject in each cell."""
    source_dir = worker.config.job_dir(source)
    sheet_id = rigging.new_id()
    png = rigging.sheet_png_path(source_dir, sheet_id)
    png.parent.mkdir(parents=True, exist_ok=True)
    atlas = Image.new("RGBA", (frame_size * columns, frame_size * rows), (0, 0, 0, 0))
    for row in range(rows):
        for column in range(columns):
            block = Image.new("RGBA", (frame_size // 2,) * 2, (200, 60, 60, 255))
            atlas.paste(
                block,
                (column * frame_size + frame_size // 4, row * frame_size + frame_size // 4),
            )
    atlas.save(png)
    meta = {
        "version": 1, "id": sheet_id, "name": "turnaround", "source_job": source,
        "created": 1.0, "image": png.name, "frame_size": frame_size,
        "columns": columns, "rows": rows,
        "width": frame_size * columns, "height": frame_size * rows,
        "elevation": 30.0, "lighting": "flat",
        "yaws": [i * 360.0 / columns for i in range(columns)],
        "poses": [{"id": None, "name": "rest"}],
        "cells": [],
    }
    rigging.sheet_path(source_dir, sheet_id).write_text(json.dumps(meta), encoding="utf-8")
    return sheet_id


def _source_job(worker) -> str:
    job_id = worker.store.create("text", "a knight", {"seed": 1})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    worker.store.set_status(job_id, "done")
    return job_id


async def _run(worker, job_id):
    worker.start()
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if worker.store.get(job_id)["status"] in ("done", "error", "cancelled"):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("job did not finish before timeout")
    finally:
        await worker.shutdown()
    return worker.store.get(job_id)


@pytest.mark.asyncio
async def test_a_stored_klein_base_restyles_bare_and_the_sidecar_says_so(worker):
    """The known-key path the door cannot reach: the worker drops the adapter
    the base cannot load, and the recipe carries no style_lora -- recording
    what ran, not what was asked for."""
    source = _source_job(worker)
    sheet_id = _rendered_sheet(worker, source)
    job_id = worker.store.create(
        "pixel_sheet", "a knight",
        {"source_job": source, "sheet_id": sheet_id, "logical_size": 32,
         "colors": 8, "seed": 3, "base_model": "flux_klein_distilled"},
    )

    row = await _run(worker, job_id)

    assert row["error"] is None and row["status"] == "done"
    pipe = worker._text2image
    assert {lora for lora, _weight in pipe.lora_calls} == {None}
    doc = rigging.read_sheet_pixel(worker.config.job_dir(source), sheet_id)
    assert doc["restyle"]["base_model"] == "flux_klein_distilled"
    assert "style_lora" not in doc["restyle"]


@pytest.mark.asyncio
async def test_the_default_base_still_records_the_style_that_ran(worker):
    """The other half of "record what ran": on the base the adapter fits, the
    sidecar keeps naming it, exactly as before."""
    source = _source_job(worker)
    sheet_id = _rendered_sheet(worker, source)
    job_id = worker.store.create(
        "pixel_sheet", "a knight",
        {"source_job": source, "sheet_id": sheet_id, "logical_size": 32,
         "colors": 8, "seed": 3},
    )

    row = await _run(worker, job_id)

    assert row["error"] is None and row["status"] == "done"
    pipe = worker._text2image
    assert {lora for lora, _weight in pipe.lora_calls} == {models.PIXEL_SHEET_LORA}
    doc = rigging.read_sheet_pixel(worker.config.job_dir(source), sheet_id)
    assert doc["restyle"]["style_lora"] == models.PIXEL_SHEET_LORA
