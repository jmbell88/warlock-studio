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
async def test_the_sidecar_records_the_lattice_the_model_drew_on(worker):
    """Measurement only: nothing here reduces on the number, and recording it
    does not bump ``TILE_SHEET_VERSION``. It is measured once on the whole
    generated frame -- the lattice is a property of the generation, not of any
    one tile -- and it is what a later calibration of
    ``pixel.GRID_RESIDUAL_MAX`` will be run against."""
    job_id = _sheet_job(worker)
    await _run(worker, job_id)

    doc = json.loads((worker.config.job_dir(job_id) / "sheet.json").read_text())
    assert set(doc["grid"]) == {"scale", "residual"}
    assert doc["grid"]["scale"] is None or isinstance(doc["grid"]["scale"], int)
    assert 0.0 <= doc["grid"]["residual"] <= 1.0
    assert doc["version"] == tilesheet.TILE_SHEET_VERSION


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


# --- the palette tail -------------------------------------------------------
#
# Three claims. A sheet that names no palette and asks for no dither is the
# file it has always been, byte for byte -- reconstructed here from the
# provenance copy the job keeps beside it, which pins the *reduction* in front
# of the quantiser at the same time. A named palette is one table over every
# cell, which is the assertion a per-cell quantisation would fail. And a palette
# that has gone missing since the door costs no generation at all.


@pytest.fixture
def paldir(worker, tmp_path):
    directory = tmp_path / "palettes"
    directory.mkdir(exist_ok=True)
    worker.config.palette_dir = directory
    return directory


_QUAD = ((26, 28, 44), (93, 39, 93), (239, 125, 87), (255, 205, 117))


def _published(worker, job_id):
    import numpy as np

    with Image.open(worker.config.job_dir(job_id) / "input.png") as sheet:
        sheet.load()
        return np.asarray(sheet.convert("RGBA"))


@pytest.mark.asyncio
async def test_a_default_sheet_is_the_median_cut_it_has_always_been(worker):
    """The compatibility claim, reconstructed from ``sheet.png`` -- which is the
    byte-for-byte generation -- through the two steps that are supposed to have
    run: the measured two-stage reduction, then ``quantize_shared`` itself. Not
    ``resolve_palette`` + ``map_palette``: those are a different assignment of
    the same table, and routing the default through them would silently
    re-colour every sheet already in the library."""
    import numpy as np

    from warlock.pipelines.pixelsheet import quantize_shared

    job_id = _sheet_job(worker, tile_w=32, colors=8)
    await _run(worker, job_id)

    geom = tilesheet.geometry(32, "top_down")
    with Image.open(worker.config.job_dir(job_id) / "sheet.png") as raw:
        raw.load()
        full = raw.convert("RGBA")
    want, _hexes = quantize_shared(
        Image.fromarray(tilesheet.reduce_sheet(np.asarray(full), geom), "RGBA"), 8
    )
    assert np.array_equal(_published(worker, job_id), np.asarray(want))


@pytest.mark.asyncio
async def test_the_measured_two_stage_reducer_still_runs_on_every_cell(
    worker, monkeypatch
):
    """``pixelize.reduce``'s single box mean is precisely what
    ``docs/measurements/2026-08-17-ground-reduction.md`` rejected -- it
    regressed every tile to its mean colour -- so this path keeps
    ``reduce_cell`` and only the quantisation moved."""
    real = tilesheet.reduce_cell
    calls: list[tuple[int, int]] = []

    def spy(pixels, out_w, out_h):
        calls.append((int(out_w), int(out_h)))
        return real(pixels, out_w, out_h)

    monkeypatch.setattr(tilesheet, "reduce_cell", spy)
    await _run(worker, _sheet_job(worker, tile_w=32))

    assert len(calls) == 64
    assert set(calls) == {(32, 32)}


@pytest.mark.asyncio
async def test_a_designed_palette_is_one_table_over_every_cell(worker, paldir):
    """The assertion that catches per-cell quantisation. Checked cell by cell
    rather than over the sheet: a per-cell median cut would put each cell's own
    colours in it, and a union over the whole sheet is exactly the check that
    would not notice."""
    import numpy as np

    (paldir / "quad.hex").write_text(
        "".join(f"#{r:02x}{g:02x}{b:02x}\n" for r, g, b in _QUAD)
    )
    job_id = _sheet_job(worker, tile_w=16, colors=8, palette="quad")
    await _run(worker, job_id)

    sheet = _published(worker, job_id)
    geom = tilesheet.geometry(16, "top_down")
    seen: set[tuple[int, int, int]] = set()
    for cell in geom.cells:
        top, left = cell.row * geom.tile_h, cell.col * geom.tile_w
        block = sheet[top : top + geom.tile_h, left : left + geom.tile_w, :3]
        used = {tuple(int(c) for c in p) for p in block.reshape(-1, 3)}
        assert used <= set(_QUAD), f"cell {cell.row},{cell.col} left the palette"
        seen |= used
    # And not vacuous: the sixty-four guide cells are distinguishable shades, so
    # a sheet that came back one colour would be a different bug passing this.
    assert len(seen) > 1
    assert np.asarray(sheet).shape[:2] == (128, 128)


@pytest.mark.asyncio
async def test_the_recipe_names_the_palette_file_and_a_digest_of_its_colours(
    worker, paldir
):
    from warlock.pipelines import pixel

    (paldir / "quad.hex").write_text(
        "".join(f"#{r:02x}{g:02x}{b:02x}\n" for r, g, b in _QUAD)
    )
    job_id = _sheet_job(worker, tile_w=16, palette="quad", dither=True)
    await _run(worker, job_id)

    recipe = json.loads(
        (worker.config.job_dir(job_id) / "sheet.json").read_text()
    )["recipe"]
    assert recipe["palette_source"] == "designed"
    assert recipe["palette_file"] == "quad"
    assert recipe["palette_hash"] == pixel.palette_digest(_QUAD)
    assert recipe["dither"] is True


@pytest.mark.asyncio
async def test_a_default_sheet_records_no_palette_keys_at_all(worker):
    """A reader that never heard of these keys sees exactly the file it saw
    before, which is the same promise the bytes make."""
    job_id = _sheet_job(worker)
    await _run(worker, job_id)

    doc = json.loads((worker.config.job_dir(job_id) / "sheet.json").read_text())
    assert not {
        "palette_source", "palette_file", "palette_hash", "dither"
    } & set(doc["recipe"])
    assert json.loads(json.dumps(doc)) == doc


@pytest.mark.asyncio
async def test_dither_alone_records_a_derived_source_and_no_file(worker):
    job_id = _sheet_job(worker, dither=True)
    await _run(worker, job_id)

    recipe = json.loads(
        (worker.config.job_dir(job_id) / "sheet.json").read_text()
    )["recipe"]
    assert recipe == {**recipe, "palette_source": "derived", "dither": True}
    assert "palette_file" not in recipe


@pytest.mark.asyncio
async def test_a_palette_deleted_after_the_door_costs_no_generation(worker, paldir):
    """Resolved before the card is spent, not at the quantize phase where it is
    used: params outlive the door that wrote them, and the alternative is a
    whole sheet generated and then thrown away."""
    job_id = _sheet_job(worker, palette="gone")
    row = await _run(worker, job_id)

    assert row["status"] == "error"
    assert "no longer installed" in (row["error"] or "")
    assert worker._text2image is None or not worker._text2image.prompts
