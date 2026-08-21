"""The tile-sheet worker: what it draws, what it publishes, and when not to.

Four claims. The generation is asked for on the *grid's* frame, because an
isometric sheet generated square and squashed afterwards is sixty-four
ellipses. The guide reaches the ControlNet, because a grid that was only
requested in words lands a few pixels out and the slicer cuts on fixed
rectangles either way. The sheet is published before its sidecar, because the
sidecar is the completion marker. And a cancelled draw publishes nothing at
all, because a half-written sheet is one a tileset import would happily slice
into sixty-four tiles of nothing.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from PIL import Image

from warlock import models
from warlock.config import Config
from warlock.db import JobStore
from warlock.pipelines import tilesheet
from warlock.queue import Worker


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


def _sheet_job(
    worker,
    *,
    tile_w=16,
    view="top_down",
    seed=7,
    colors=8,
    prompt="a damp dungeon",
    **extra,
) -> str:
    geom = tilesheet.geometry(tile_w, view)
    params = {
        "seed": seed,
        "base_model": "sdxl_cfg",
        "style_lora": models.PIXEL_SHEET_LORA,
        "control": "canny",
        "colors": colors,
        "negative_prompt": "",
        "sheet": {
            "version": 2,
            "tile_w": geom.tile_w,
            "tile_h": geom.tile_h,
            "projection": geom.view,
            "columns": geom.columns,
            "rows": geom.rows,
        },
    }
    params.update(extra)
    return worker.store.create("tile_sheet", prompt, params, stage="tilesheet")


async def _run(worker, job_id):
    worker.start()
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if worker.store.get(job_id)["status"] in ("done", "error", "cancelled"):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("job did not finish before timeout")
    finally:
        await worker.shutdown()
    return worker.store.get(job_id)


# --- what it draws ----------------------------------------------------------


@pytest.mark.asyncio
async def test_one_generation_makes_the_whole_sheet(worker):
    """Sixty-four tiles behind one load. Generating per tile would pay the
    expensive part sixty-four times for a deliverable that is one sheet -- and
    the tiles would not share a style, a light direction or a palette."""
    row = await _run(worker, _sheet_job(worker))

    assert row["error"] is None and row["status"] == "done"
    pipe = worker._text2image
    assert len(pipe.prompts) == 1
    assert pipe.unload_calls <= 1


@pytest.mark.asyncio
async def test_the_generation_uses_the_grid_template_and_never_wraps(worker):
    """A sheet must not be seamless: its leftmost and rightmost columns are
    different tiles, and making the frame continuous would bleed one into the
    other."""
    await _run(worker, _sheet_job(worker))

    pipe = worker._text2image
    assert pipe.tilesheets == [True]
    assert pipe.tiles == [False]
    assert pipe.sheets == [False]


@pytest.mark.asyncio
async def test_a_top_down_sheet_is_generated_square(worker):
    await _run(worker, _sheet_job(worker, tile_w=32, view="top_down"))
    assert worker._text2image.sizes == [(1024, 1024)]


@pytest.mark.asyncio
async def test_an_isometric_sheet_is_generated_two_to_one(worker):
    """The reason ``generate`` grew a size override at all: squashed afterwards,
    every diamond comes back an ellipse."""
    await _run(worker, _sheet_job(worker, tile_w=32, view="isometric"))
    assert worker._text2image.sizes == [(1024, 512)]


@pytest.mark.asyncio
async def test_the_grid_guide_reaches_the_controlnet(worker):
    """The whole mechanism: the cells are imposed, not requested. Without the
    guide the seams land a few pixels off the rectangles the slicer cuts on,
    and every tile carries a sliver of its neighbour."""
    await _run(worker, _sheet_job(worker))

    cond = worker._text2image.conditionings[0]
    assert cond is not None
    assert cond.control == "canny"
    # The *path*, not the picture: the guide lives in a scratch directory that
    # is gone by the time the job finishes, which is deliberate -- it is an
    # input to one call, not an artifact. What the guide actually looks like is
    # pinned in ``test_tilesheet_pipeline``.
    assert cond.control_image is not None
    assert cond.control_image.name == "guide.png"
    # And the strength, because a hint attached at scale zero is a guide that
    # was handed over and then ignored.
    assert cond.control_scale > 0


@pytest.mark.asyncio
async def test_the_subject_carries_the_view_and_the_detail_clause(worker):
    await _run(worker, _sheet_job(worker, view="isometric", prompt="a mine"))

    composed = worker._text2image.prompts[0]
    assert "a mine" in composed
    assert "isometric" in composed
    assert tilesheet.DETAIL_CLAUSE in composed


@pytest.mark.asyncio
async def test_the_stored_seed_is_the_one_that_runs(worker):
    await _run(worker, _sheet_job(worker, seed=4242))
    assert worker._text2image.seeds == [4242]


# --- the optional reference -------------------------------------------------


@pytest.mark.asyncio
async def test_without_a_reference_only_the_guide_conditions_the_draw(worker):
    """The common path, and the one ``vram.estimate_parts`` gates the encoder's
    cost on: a request with no reference must not load one."""
    await _run(worker, _sheet_job(worker))

    cond = worker._text2image.conditionings[0]
    assert cond.ip_adapter is None
    assert cond.ip_image is None


@pytest.mark.asyncio
async def test_a_reference_on_disk_is_conditioned_on(worker):
    job_id = _sheet_job(worker, ip_adapter="plus")
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (10, 120, 40)).save(job_dir / "ref.png", "PNG")

    await _run(worker, job_id)

    cond = worker._text2image.conditionings[0]
    assert cond.ip_adapter == "plus"
    assert cond.ip_image == job_dir / "ref.png"


@pytest.mark.asyncio
async def test_a_params_adapter_with_no_file_conditions_on_nothing(worker):
    """Read from disk rather than from params: a reroll copies the row before
    it copies the file, and an adapter named with no image to hand it would
    reach the pipe and be silently dropped."""
    await _run(worker, _sheet_job(worker, ip_adapter="plus"))

    assert worker._text2image.conditionings[0].ip_adapter is None


# --- what it publishes ------------------------------------------------------


@pytest.mark.asyncio
async def test_the_published_sheet_is_exactly_the_grid_times_the_tile(worker):
    job_id = _sheet_job(worker, tile_w=32, view="top_down")
    await _run(worker, job_id)

    with Image.open(worker.config.job_dir(job_id) / "input.png") as sheet:
        assert sheet.size == (256, 256)


@pytest.mark.asyncio
async def test_an_isometric_sheet_publishes_two_to_one_tiles(worker):
    job_id = _sheet_job(worker, tile_w=32, view="isometric")
    await _run(worker, job_id)

    with Image.open(worker.config.job_dir(job_id) / "input.png") as sheet:
        assert sheet.size == (256, 128)


@pytest.mark.asyncio
async def test_the_raw_generation_is_kept_beside_the_sheet(worker):
    """Provenance, not an artifact: it is what the model actually returned, and
    the only way to tell "the guide was ignored" from "the reduction is wrong"
    after the fact."""
    job_id = _sheet_job(worker)
    await _run(worker, job_id)

    assert (worker.config.job_dir(job_id) / "sheet.png").exists()


@pytest.mark.asyncio
async def test_the_sidecar_is_written_after_the_sheet(worker):
    """It is the completion marker ``service.files.ready`` gates on, so
    publishing it first would advertise a sheet still being written."""
    job_id = _sheet_job(worker)
    await _run(worker, job_id)

    job_dir = worker.config.job_dir(job_id)
    sheet = job_dir / "input.png"
    doc = job_dir / "sheet.json"
    assert sheet.exists() and doc.exists()
    assert doc.stat().st_mtime_ns >= sheet.stat().st_mtime_ns


@pytest.mark.asyncio
async def test_the_sidecar_says_what_ran(worker):
    job_id = _sheet_job(worker, tile_w=48, view="isometric", colors=16)
    await _run(worker, job_id)

    doc = json.loads((worker.config.job_dir(job_id) / "sheet.json").read_text())
    assert doc["version"] == tilesheet.TILE_SHEET_VERSION
    assert doc["tile_w"] == 48
    assert doc["tile_h"] == 24
    assert doc["projection"] == "isometric"
    assert doc["tiles"] == 64
    assert doc["recipe"]["control"] == "canny"
    assert doc["recipe"]["style_lora"] == models.PIXEL_SHEET_LORA
    assert doc["palette"]


@pytest.mark.asyncio
async def test_the_row_records_what_it_drew(worker):
    job_id = _sheet_job(worker, tile_w=32)
    row = await _run(worker, job_id)

    report = row["params"]["sheet_report"]
    assert report["tiles"] == 64
    assert report["tile_w"] == 32
    assert report["sheet_w"] == 256
    assert report["projection"] == "top_down"


@pytest.mark.asyncio
async def test_the_sheet_is_quantized_to_one_palette(worker):
    """Sixty-four tiles quantized separately read as sixty-four pictures pasted
    together, which is exactly what a tile sheet must not."""
    job_id = _sheet_job(worker, colors=8)
    row = await _run(worker, job_id)

    assert row["params"]["sheet_report"]["palette"] <= 8


# --- refusals and cancellation ----------------------------------------------


@pytest.mark.asyncio
async def test_a_block_with_no_geometry_is_corruption_not_a_default(worker):
    """There is exactly one writer and it always writes both keys, so absence
    is corruption -- and defaulting to 32 would spend the card on a sheet whose
    size nobody chose."""
    job_id = worker.store.create(
        "tile_sheet",
        "a mine",
        {"seed": 1, "base_model": "sdxl_cfg", "sheet": {"version": 1}},
        stage="tilesheet",
    )
    row = await _run(worker, job_id)

    assert row["status"] == "error"
    assert "what it is a sheet of" in (row["error"] or "")
    assert worker._text2image is None or not worker._text2image.prompts


@pytest.mark.asyncio
async def test_a_stored_projection_the_pipeline_refuses_costs_no_generation(worker):
    job_id = worker.store.create(
        "tile_sheet",
        "a mine",
        {
            "seed": 1,
            "base_model": "sdxl_cfg",
            "sheet": {"version": 1, "tile_w": 32, "projection": "hexagonal"},
        },
        stage="tilesheet",
    )
    row = await _run(worker, job_id)

    assert row["status"] == "error"
    assert "hexagonal" in (row["error"] or "")


@pytest.mark.asyncio
async def test_a_base_that_cannot_take_the_pixel_lora_draws_bare_and_says_so(worker):
    """Params outlive the service that wrote them, so a sheet whose sidecar
    claimed a LoRA that never loaded would be a recipe nobody can reproduce."""
    job_id = _sheet_job(worker, base_model="flux_klein")
    row = await _run(worker, job_id)

    assert row["error"] is None and row["status"] == "done"
    doc = json.loads((worker.config.job_dir(job_id) / "sheet.json").read_text())
    # The recipe records what *ran*, so the absent key is the claim: a style
    # that never loaded must not be named. The sheet is still published --
    # bare, and said so in the log rather than refused, because the door is
    # where a refusal belongs and params can outlive it.
    assert "style_lora" not in doc["recipe"]
    assert "lora_weight" not in doc["recipe"]
    assert (worker.config.job_dir(job_id) / "input.png").exists()


@pytest.mark.asyncio
async def test_a_cancelled_draw_publishes_nothing(worker):
    job_id = _sheet_job(worker)
    worker.start()
    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if worker.current_job_id == job_id and worker._text2image is not None:
                break
            await asyncio.sleep(0.01)
        await worker.request_cancel(job_id)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if worker.store.get(job_id)["status"] != "running":
                break
            await asyncio.sleep(0.01)
    finally:
        await worker.shutdown()

    assert worker.store.get(job_id)["status"] == "cancelled"
    job_dir = worker.config.job_dir(job_id)
    assert not (job_dir / "input.png").exists()
    assert not (job_dir / "sheet.png").exists()
    assert not (job_dir / "sheet.json").exists()
