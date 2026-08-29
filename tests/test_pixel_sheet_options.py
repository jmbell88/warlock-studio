"""The restyle's authored palette, dither and outline -- and the path they
deliberately do **not** touch.

``_pixel_sheet`` grew a branch rather than a new implementation, and the branch
is the point. ``PIXEL_SHEET_VERSION`` is recorded in every pixel sheet already
on disk, so a request that names none of the three options has to keep
producing exactly the bytes those sheets claim -- which is why the two original
lines are untouched and the new code runs only when somebody actually asked for
something. ``asset2d.pixel`` set the precedent: "with ``opts`` left at its
default the result is byte-identical ... which is pinned by a test". This is
that test, plus the three options it guards.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from PIL import Image

from warlock import rigging
from warlock.config import Config
from warlock.db import JobStore
from warlock.pipelines import pixel, pixelize, pixelsheet
from warlock.queue import Worker
from warlock.service import Invalid
from warlock.service import jobs as svc_jobs
from warlock.service import sheets as svc_sheets

RAMP = ("#101020", "#5a2878", "#c85a3c", "#f5f0d2")
RAMP_RGB = tuple(
    (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) for h in RAMP
)


# --- the door -----------------------------------------------------------------


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
        "version": 1, "id": sheet_id, "name": "turnaround", "source_job": job_id,
        "created": 1.0, "image": png.name, "frame_size": frame_size,
        "columns": columns, "rows": rows,
        "width": frame_size * columns, "height": frame_size * rows,
        "yaws": [i * 45.0 for i in range(columns)],
        "poses": [{"id": None, "name": "rest"}],
        "cells": [],
    }
    rigging.sheet_path(job_dir, sheet_id).write_text(json.dumps(meta), encoding="utf-8")
    return job_id, sheet_id


def test_the_defaults_are_the_old_request_exactly(svc):
    """Nothing a caller who names none of them can tell apart from before: the
    three keys are written, and every one of them is the value that takes the
    untouched branch in the worker."""
    job_id, sheet_id = _sheet_on_disk(svc)

    out = svc_sheets.create_pixel_sheet(svc, job_id, sheet_id)

    params = svc.store.get(out["id"])["params"]
    assert params["palette"] == ""
    assert params["dither"] is False
    assert params["outline"] == "none"
    # ``none`` here and ``inner`` on the synthesis path, and the difference is
    # not taste: a restyled sheet already on disk was cut without an outline.
    assert svc_sheets.DEFAULT_PIXEL_SHEET_OUTLINE == "none"


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"outline": "glow"}, "outline"),
        ({"palette": "never-installed"}, "palette"),
        ({"logical_size": 7}, "logical_size"),
        ({"colors": 7}, "colors"),
    ],
)
def test_a_bad_option_is_refused_on_its_own_field(svc, kwargs, field):
    job_id, sheet_id = _sheet_on_disk(svc)
    with pytest.raises(Invalid) as excinfo:
        svc_sheets.create_pixel_sheet(svc, job_id, sheet_id, **kwargs)
    assert excinfo.value.field == field


def test_a_palette_is_read_at_the_door_and_only_its_name_is_stored(svc, tmp_path):
    directory = tmp_path / "palettes"
    directory.mkdir(exist_ok=True)
    svc.config.palette_dir = directory
    (directory / "ramp.hex").write_text("\n".join(RAMP) + "\n", encoding="utf-8")
    job_id, sheet_id = _sheet_on_disk(svc)

    out = svc_sheets.create_pixel_sheet(
        svc, job_id, sheet_id, palette=" ramp ", dither=True, outline="inner"
    )

    params = svc.store.get(out["id"])["params"]
    assert params["palette"] == "ramp"
    assert params["dither"] is True
    assert params["outline"] == "inner"


def test_the_offered_outlines_are_the_ones_the_pixeliser_draws():
    options = svc_sheets.pixel_sheet_options()
    assert options["outlines"] == list(pixelize.OUTLINE_MODES)
    assert options["defaults"]["outline"] == "none"


def test_this_path_offers_no_reduce_mode(svc):
    """Deliberately absent rather than validated and dropped. The reduction is
    ``pixelsheet.downscale``'s integer NEAREST stride, fixed by the requirement
    that a cell boundary stay on an output pixel boundary -- so a reduce mode
    here would be the dead field ``check_pixel_options`` refuses to create."""
    job_id, sheet_id = _sheet_on_disk(svc)
    out = svc_sheets.create_pixel_sheet(svc, job_id, sheet_id)
    assert "reduce_mode" not in svc.store.get(out["id"])["params"]


# --- the worker ---------------------------------------------------------------


@pytest.fixture
def worker(tmp_path, fake_pipelines):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
    )
    config.palette_dir = tmp_path / "palettes"
    config.palette_dir.mkdir(parents=True, exist_ok=True)
    (config.palette_dir / "ramp.hex").write_text(
        "\n".join(RAMP) + "\n", encoding="utf-8"
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    yield w
    store.close()


def _source_job(worker) -> str:
    job_id = worker.store.create("text", "a knight", {"seed": 1})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    worker.store.set_status(job_id, "done")
    return job_id


def _rendered_sheet(worker, source, *, frame_size=128, columns=8, rows=1):
    """A finished render with a subject in each cell, so the restyle has a
    silhouette to remask onto rather than a full-bleed rectangle."""
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
                (
                    column * frame_size + frame_size // 4,
                    row * frame_size + frame_size // 4,
                ),
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
    rigging.sheet_path(source_dir, sheet_id).write_text(
        json.dumps(meta), encoding="utf-8"
    )
    return sheet_id


def _queue(worker, source, sheet_id, **overrides) -> str:
    params = {
        "source_job": source, "sheet_id": sheet_id, "logical_size": 32,
        "colors": 8, "seed": 3, "strength": 0.5, "base_model": "sdxl_cfg",
    }
    params.update(overrides)
    return worker.store.create("pixel_sheet", "a knight", params)


async def _run(worker, job_id):
    worker.start()
    try:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if worker.store.get(job_id)["status"] in ("done", "error", "cancelled"):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("job did not finish before timeout")
    finally:
        await worker.shutdown()
    return worker.store.get(job_id)


@pytest.fixture
def spies(monkeypatch):
    """What each of the two quantisers was asked, and what it answered."""
    calls: dict[str, list] = {"quantize_shared": [], "pixelize_atlas": []}
    real_quantize = pixelsheet.quantize_shared
    real_pixelize = pixelize.pixelize_atlas

    def quantize(atlas, colors):
        out = real_quantize(atlas, colors)
        calls["quantize_shared"].append(out)
        return out

    def atlas_pass(*args, **kwargs):
        out = real_pixelize(*args, **kwargs)
        calls["pixelize_atlas"].append(out)
        return out

    monkeypatch.setattr(pixelsheet, "quantize_shared", quantize)
    monkeypatch.setattr(pixelize, "pixelize_atlas", atlas_pass)
    return calls


@pytest.mark.asyncio
async def test_the_default_request_publishes_exactly_what_quantize_shared_returned(
    worker, spies
):
    """The byte-identity pin, and it is a pin on the *pixels* rather than on
    which function ran: the published atlas is the image ``quantize_shared``
    handed back, unmodified. Nothing else touched it, which is what "the default
    path did not move" has to mean for every sheet already on disk.
    """
    source = _source_job(worker)
    sheet_id = _rendered_sheet(worker, source)
    job_id = _queue(worker, source, sheet_id)

    row = await _run(worker, job_id)
    assert row["error"] is None and row["status"] == "done", row["error"]

    assert len(spies["quantize_shared"]) == 1
    assert spies["pixelize_atlas"] == []
    expected, expected_palette = spies["quantize_shared"][0]
    with Image.open(
        rigging.sheet_pixel_png_path(worker.config.job_dir(source), sheet_id)
    ) as published:
        assert published.convert("RGBA").tobytes() == expected.convert("RGBA").tobytes()
    record = rigging.read_sheet_pixel(worker.config.job_dir(source), sheet_id)
    assert record["palette"] == expected_palette
    assert record["version"] == pixelsheet.PIXEL_SHEET_VERSION == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "option",
    [{"palette": "ramp"}, {"dither": True}, {"outline": "inner"}],
)
async def test_any_one_of_the_three_options_takes_the_new_path(worker, spies, option):
    """A branch, so each option has to be able to open it on its own -- and the
    old two lines have to stop deciding the pixels when one does."""
    source = _source_job(worker)
    sheet_id = _rendered_sheet(worker, source)
    job_id = _queue(worker, source, sheet_id, **option)

    row = await _run(worker, job_id)
    assert row["error"] is None and row["status"] == "done", row["error"]

    assert len(spies["pixelize_atlas"]) == 1
    published_from = spies["pixelize_atlas"][0][0]
    with Image.open(
        rigging.sheet_pixel_png_path(worker.config.job_dir(source), sheet_id)
    ) as published:
        assert (
            published.convert("RGBA").tobytes()
            == published_from.convert("RGBA").tobytes()
        )


@pytest.mark.asyncio
async def test_a_named_palette_is_the_only_colours_in_the_sheet(worker):
    source = _source_job(worker)
    sheet_id = _rendered_sheet(worker, source)
    job_id = _queue(worker, source, sheet_id, palette="ramp")

    row = await _run(worker, job_id)
    assert row["error"] is None, row["error"]

    source_dir = worker.config.job_dir(source)
    with Image.open(rigging.sheet_pixel_png_path(source_dir, sheet_id)) as png:
        colours = {
            f"#{r:02x}{g:02x}{b:02x}"
            for _count, (r, g, b, a) in png.convert("RGBA").getcolors(1 << 24)
            if a > 0
        }
    assert colours and colours <= set(RAMP)
    recipe = rigging.read_sheet_pixel(source_dir, sheet_id)["restyle"]
    assert recipe["palette"] == "ramp"
    assert recipe["palette_source"] == "designed"
    assert recipe["palette_hash"] == pixel.palette_digest(RAMP_RGB)
    assert recipe["dither"] is False and recipe["outline"] == "none"


@pytest.mark.asyncio
async def test_the_recipe_records_the_options_on_the_default_path_too(worker):
    """Always recorded, on both branches: leaving the keys out when nothing was
    asked for would make "no outline" and "an older Warlock" the same reading.
    Additive, and it does not bump ``PIXEL_SHEET_VERSION`` -- a new optional key
    readers may ignore is not a new format, and the default bytes did not move.
    """
    source = _source_job(worker)
    sheet_id = _rendered_sheet(worker, source)
    job_id = _queue(worker, source, sheet_id)

    row = await _run(worker, job_id)
    assert row["error"] is None, row["error"]

    recipe = rigging.read_sheet_pixel(worker.config.job_dir(source), sheet_id)["restyle"]
    assert recipe["palette"] == "" and recipe["palette_hash"] == ""
    assert recipe["palette_source"] == "derived"
    assert recipe["dither"] is False and recipe["outline"] == "none"


@pytest.mark.asyncio
async def test_a_palette_deleted_since_the_door_fails_before_the_gpu(worker):
    source = _source_job(worker)
    sheet_id = _rendered_sheet(worker, source)
    job_id = _queue(worker, source, sheet_id, palette="gone")

    row = await _run(worker, job_id)

    assert row["status"] == "error"
    assert "palette 'gone' is no longer installed" in (row["error"] or "")
    assert worker._text2image is None or worker._text2image.seeds == []
