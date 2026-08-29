"""The seamless-material tileset worker: N generations, one atlas.

The opposite claim to ``test_tilesheet_worker``'s, and the two files are the
pair. There, sixty-four tiles are a *slicing* of one frame and the frame must
not wrap, because its leftmost and rightmost columns are different tiles. Here,
every tile is its own generation and every one of them must wrap, because a
material is a torus -- so ``tile=True`` is the whole mechanism and the grid
guide is not present at all.

Five things are worth a test that the pipeline modules cannot already state.
Every pass wraps and asks for the one frame ``reduce_material`` can partition.
All N passes sit inside **one** acquire/release bracket, because the expensive
part is the checkpoint and paying it N times is the whole reason this is one
job. The seeds that ran are the ones the row stored, so the set is
reproducible. Nothing at all is published for a cancelled draw. And a row
written before any of this existed still reaches the grid coroutine, because
``rerun_job`` copies params forward and a stricter reading would fail every one
of them.
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
from warlock.pipelines import tileatlas, tilemask, tilesheet
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


def _seed_colour(seed: int) -> tuple[int, int, int]:
    """What ``FakeText2Image``'s ``tile`` arm paints for a seed. Kept in step
    with it here rather than imported, so a change to either side shows up as a
    failing assertion about *which material landed where* rather than as two
    files agreeing about nothing."""
    return ((seed * 71) % 251, (seed * 131) % 241, (seed * 197) % 239)


def _materials_job(
    worker,
    *,
    tile_w=32,
    prompts=("moss", "gravel", "water"),
    seed=100,
    colors=64,
    style_lock=False,
    prompt="a damp dungeon",
    **extra,
) -> str:
    seeds = tileatlas.material_seeds(seed, len(prompts))
    geom = tileatlas.material_geometry(tile_w, "top_down", len(prompts))
    params = {
        "seed": seed,
        "base_model": "sdxl_cfg",
        "colors": colors,
        "negative_prompt": "",
        "sheet": {
            "version": 3,
            "mode": "materials",
            "tile_w": geom.tile_w,
            "tile_h": geom.tile_h,
            "projection": "top_down",
            "columns": geom.columns,
            "rows": geom.rows,
            "layout": "grid",
            "materials": [
                {"index": index, "prompt": line, "variant": 1, "seed": material_seed}
                for index, (line, material_seed) in enumerate(
                    zip(prompts, seeds, strict=True)
                )
            ],
            "variants": 1,
            "style_lock": style_lock,
        },
    }
    params.update(extra)
    return worker.store.create("tile_sheet", prompt, params, stage="tilesheet")


def _terrain_job(worker, *, tile_w=32, seed=200, colors=64, mask=None, **extra) -> str:
    seeds = tileatlas.material_seeds(seed, 2)
    geom = tileatlas.terrain_geometry(tile_w, "top_down")
    params = {
        "seed": seed,
        "base_model": "sdxl_cfg",
        "colors": colors,
        "negative_prompt": "",
        "sheet": {
            "version": 3,
            "mode": "terrain",
            "tile_w": geom.tile_w,
            "tile_h": geom.tile_h,
            "projection": "top_down",
            "columns": geom.columns,
            "rows": geom.rows,
            "layout": "blob47",
            "materials": [
                {"index": 0, "prompt": "grass", "variant": 1, "seed": seeds[0]},
                {"index": 1, "prompt": "still water", "variant": 1, "seed": seeds[1]},
            ],
            "terrains": [
                {
                    "name": "grass",
                    "fill": [40, 120, 40, 255],
                    "outline": [20, 60, 20, 255],
                }
            ],
            "mask": mask
            if mask is not None
            else {
                "version": tilemask.MASK_VERSION,
                "seed": 5,
                "inset": None,
                "amplitude": None,
                "feather": None,
            },
            "variants": 1,
            "style_lock": False,
        },
    }
    params.update(extra)
    return worker.store.create("tile_sheet", "a shoreline", params, stage="tilesheet")


def _legacy_grid_job(worker, *, tile_w=16) -> str:
    """A row written before the seamless path existed: a version-2 block with
    no ``mode`` key at all."""
    geom = tilesheet.geometry(tile_w, "top_down")
    params = {
        "seed": 7,
        "base_model": "sdxl_cfg",
        "control": "canny",
        "colors": 8,
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
    return worker.store.create("tile_sheet", "a mine", params, stage="tilesheet")


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


# --- what it draws ----------------------------------------------------------


@pytest.mark.asyncio
async def test_every_material_is_its_own_generation(worker):
    """The whole claim of this path. One generation on a grid gives sixty-four
    identical cells -- every cell of the guide is identical, so there is no
    per-cell signal for variety. Variety is a property of the request."""
    row = await _run(worker, _materials_job(worker))

    assert row["error"] is None and row["status"] == "done"
    assert len(worker._text2image.prompts) == 3


@pytest.mark.asyncio
async def test_every_pass_wraps_and_none_of_them_uses_the_grid_template(worker):
    """``tile=True`` selects the circular-padded UNet, VAE and ControlNet --
    the only thing that makes a material tile against itself. ``tilesheet=True``
    is the *other* path's flag and carries an explicit no-wrap rule."""
    await _run(worker, _materials_job(worker))

    pipe = worker._text2image
    assert pipe.tiles == [True, True, True]
    assert pipe.tilesheets == [False, False, False]
    assert pipe.sheets == [False, False, False]


@pytest.mark.asyncio
async def test_every_pass_asks_for_one_full_sdxl_frame(worker):
    """``reduce_material`` refuses any factor that is not exact, and 1024 is
    the numerator that makes the tile sizes divide -- a 512px material reduced
    to 32px would be fine and a 48px one would not, which is the trap."""
    await _run(worker, _materials_job(worker, tile_w=32))

    assert worker._text2image.sizes == [(1024, 1024)] * 3


@pytest.mark.asyncio
async def test_the_stored_seeds_are_the_ones_that_run(worker):
    """Derived at the door and stored, so material ``i`` is reproducible on its
    own from the pair ``(seed, i)``. A worker that re-derived them would be a
    second copy of a derivation that is free to change."""
    await _run(worker, _materials_job(worker, seed=4242))

    assert worker._text2image.seeds == list(tileatlas.material_seeds(4242, 3))


@pytest.mark.asyncio
async def test_the_subjects_carry_the_lines_and_the_style_clause(worker):
    await _run(worker, _materials_job(worker, prompts=("moss", "gravel")))

    composed = worker._text2image.prompts
    assert "moss" in composed[0] and "gravel" in composed[1]
    assert all(tileatlas.MATERIAL_STYLE_CLAUSE in text for text in composed)


@pytest.mark.asyncio
async def test_all_the_passes_sit_inside_one_acquire_release_bracket(worker):
    """The expensive part is loading the checkpoint and the LoRA, and this pays
    it once for the whole set. A bracket per pass would load and give back
    ~7 GiB N times."""
    await _run(worker, _materials_job(worker, prompts=("a", "b", "c", "d", "e")))

    pipe = worker._text2image
    assert len(pipe.prompts) == 5
    # One release for five passes. This is the assertion that fails if the
    # bracket is ever moved inside the loop.
    assert pipe.unload_calls == 1


@pytest.mark.asyncio
async def test_no_pass_is_conditioned_when_nothing_was_attached(worker):
    """``None``, not an empty ``Conditioning()``. It is what ``_needs_handoff``
    reads, so an empty object would stop a warm trellis that ``vram.estimate``
    has already priced as co-resident."""
    await _run(worker, _materials_job(worker))

    assert worker._text2image.conditionings == [None, None, None]


@pytest.mark.asyncio
async def test_style_lock_hands_the_first_material_to_every_pass_after_it(worker):
    """N independent samples otherwise come back as N pictures of the right
    things in N different hands. Pass one has nothing to lock onto yet, which
    is why it is the one pass whose conditioning differs."""
    await _run(worker, _materials_job(worker, style_lock=True))

    conds = worker._text2image.conditionings
    assert conds[0] is None
    for cond in conds[1:]:
        assert cond is not None
        assert cond.ip_adapter == "plus"
        # The *first* material, and the same file for both -- one reference for
        # the whole set is what "one style" means.
        assert cond.ip_image.name == "material-00.png"
    assert conds[1].ip_image == conds[2].ip_image


@pytest.mark.asyncio
async def test_without_style_lock_no_pass_carries_a_reference(worker):
    await _run(worker, _materials_job(worker, style_lock=False))
    assert all(cond is None for cond in worker._text2image.conditionings)


# --- what it publishes ------------------------------------------------------


@pytest.mark.asyncio
async def test_the_published_atlas_is_exactly_the_grid_times_the_tile(worker):
    job_id = _materials_job(worker, tile_w=32, prompts=("moss", "gravel", "water"))
    await _run(worker, job_id)

    with Image.open(worker.config.job_dir(job_id) / "input.png") as atlas:
        # Three cells at eight columns is one row of three, not a square packed
        # to look tidy: a materials sheet is the list the user typed.
        assert atlas.size == (96, 32)


@pytest.mark.asyncio
async def test_a_second_row_starts_after_eight_materials(worker):
    job_id = _materials_job(worker, tile_w=16, prompts=tuple("abcdefghij"))
    await _run(worker, job_id)

    with Image.open(worker.config.job_dir(job_id) / "input.png") as atlas:
        assert atlas.size == (16 * 8, 16 * 2)


@pytest.mark.asyncio
async def test_each_material_lands_in_its_own_cell_in_the_order_it_was_typed(worker):
    """The failure this rules out is the one N-pass paths invite: N correct
    generations assembled in the wrong order, which looks perfect and is a
    different sheet from the one the sidecar describes."""
    job_id = _materials_job(worker, tile_w=32, seed=100)
    await _run(worker, job_id)

    expected = [_seed_colour(s) for s in tileatlas.material_seeds(100, 3)]
    with Image.open(worker.config.job_dir(job_id) / "input.png") as atlas:
        rgb = atlas.convert("RGB")
        landed = [rgb.getpixel((index * 32 + 16, 16)) for index in range(3)]

    # Matched by nearest rather than by equality: the atlas has been through a
    # shared-palette median cut, so a channel may move by a unit or two. What is
    # under test is *which* material is in each cell, and the three are far
    # enough apart in all three channels for nearest to answer that exactly.
    def _nearest(colour):
        return min(
            range(3),
            key=lambda i: sum(
                (a - b) ** 2 for a, b in zip(colour, expected[i], strict=True)
            ),
        )

    assert [_nearest(colour) for colour in landed] == [0, 1, 2]


@pytest.mark.asyncio
async def test_the_raw_materials_are_kept_beside_the_atlas(worker):
    """Provenance, not artifacts: they are what the model actually returned, and
    the only way to tell "the material did not tile" from "the reduction is
    wrong" after the fact."""
    job_id = _materials_job(worker, seed=100)
    await _run(worker, job_id)

    materials = worker.config.job_dir(job_id) / "materials"
    assert sorted(p.name for p in materials.iterdir()) == [
        "00.png", "01.png", "02.png"
    ]
    seeds = tileatlas.material_seeds(100, 3)
    for index, material_seed in enumerate(seeds):
        with Image.open(materials / f"{index:02d}.png") as raw:
            # Un-reduced and un-quantized, so this one is exact.
            assert raw.size == (1024, 1024)
            assert raw.convert("RGB").getpixel((512, 512)) == _seed_colour(material_seed)


@pytest.mark.asyncio
async def test_the_sidecar_is_written_after_the_atlas(worker):
    """It is the completion marker, so publishing it first would advertise a
    set still being written."""
    job_id = _materials_job(worker)
    await _run(worker, job_id)

    job_dir = worker.config.job_dir(job_id)
    atlas = job_dir / "input.png"
    doc = job_dir / "sheet.json"
    assert atlas.exists() and doc.exists()
    assert doc.stat().st_mtime_ns >= atlas.stat().st_mtime_ns


@pytest.mark.asyncio
async def test_the_sidecar_says_what_it_is_rather_than_leaving_it_to_be_seen(worker):
    job_id = _materials_job(worker, tile_w=32, prompts=("moss", "gravel", "water"))
    await _run(worker, job_id)

    doc = json.loads((worker.config.job_dir(job_id) / "sheet.json").read_text())
    assert doc["version"] == tileatlas.TILE_ATLAS_VERSION
    assert doc["mode"] == "materials"
    assert doc["layout"] == "grid"
    assert doc["view"] == "top_down"
    assert (doc["tile_w"], doc["tile_h"]) == (32, 32)
    assert (doc["columns"], doc["rows"], doc["tiles"]) == (3, 1, 3)
    # The lines the user typed, not the compiled subjects: the record is what
    # was asked for, and the style clause is this module's own addition.
    assert [entry["prompt"] for entry in doc["materials"]] == ["moss", "gravel", "water"]
    assert [entry["seed"] for entry in doc["materials"]] == list(
        tileatlas.material_seeds(100, 3)
    )
    assert doc["terrains"] == [] and doc["mask"] is None
    assert doc["recipe"]["style_lora"] == models.PIXEL_SHEET_LORA


@pytest.mark.asyncio
async def test_the_sidecar_records_one_lattice_per_generated_material(worker):
    """Per material, because a material is one generation and a cell is a crop
    out of one. Measurement only -- nothing reduces on it, and recording it
    does not bump ``TILE_ATLAS_VERSION``."""
    job_id = _materials_job(worker, prompts=("moss", "gravel", "water"))
    await _run(worker, job_id)

    doc = json.loads((worker.config.job_dir(job_id) / "sheet.json").read_text())
    assert [entry["material"] for entry in doc["grids"]] == [0, 1, 2]
    for entry in doc["grids"]:
        assert set(entry) == {"material", "scale", "residual"}
        assert entry["scale"] is None or isinstance(entry["scale"], int)
    assert doc["version"] == tileatlas.TILE_ATLAS_VERSION


@pytest.mark.asyncio
async def test_a_terrain_set_records_the_two_generations_and_not_its_cells(worker):
    """Forty-seven cells composited from two frames: the count says which of
    the two things is being measured."""
    job_id = _terrain_job(worker, tile_w=32)
    await _run(worker, job_id)

    doc = json.loads((worker.config.job_dir(job_id) / "sheet.json").read_text())
    assert len(doc["grids"]) == 2
    assert len(doc["materials"]) == 47


@pytest.mark.asyncio
async def test_the_seam_measurement_is_recorded_and_never_a_rejection(worker):
    """``SEAM_MAX`` was measured on turbo at four steps and ``seam.py`` says
    outright to re-measure per checkpoint, so a verdict from it cannot be
    allowed to throw away a generation that is already on disk."""
    job_id = _materials_job(worker)
    row = await _run(worker, job_id)

    report = row["params"]["sheet_report"]
    assert [entry["index"] for entry in report["seams"]] == [0, 1, 2]
    assert report["seam_worst"] == max(e["worst"] for e in report["seams"])
    # Three passes and no redraw: a bad ratio must not cost a fourth.
    assert len(worker._text2image.prompts) == 3


@pytest.mark.asyncio
async def test_the_row_records_what_it_drew(worker):
    job_id = _materials_job(worker, tile_w=32)
    row = await _run(worker, job_id)

    report = row["params"]["sheet_report"]
    assert report["mode"] == "materials"
    assert (report["sheet_w"], report["sheet_h"]) == (96, 32)
    assert report["materials"] == 3
    assert report["palette"]


# --- terrain ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_terrain_set_is_two_generations_and_forty_seven_tiles(worker):
    """The model draws two surfaces and never sees an edge. Asking for the
    boundary is what would put a *drawn* edge inside a tile that the field then
    cuts across."""
    job_id = _terrain_job(worker, tile_w=32)
    row = await _run(worker, job_id)

    assert row["error"] is None and row["status"] == "done"
    assert len(worker._text2image.prompts) == 2
    assert worker._text2image.tiles == [True, True]
    with Image.open(worker.config.job_dir(job_id) / "input.png") as atlas:
        assert atlas.size == (tilemask.TILE_COUNT * 32, 32)


@pytest.mark.asyncio
async def test_a_terrain_sidecar_declares_the_layout_it_cannot_be_measured_for(worker):
    """``roles.infer_roles`` returns ``None`` on a perfectly formed generated
    set -- two opaque textures have neither transparency nor a dominant ring
    colour. What catches it is not a measurement but a record."""
    job_id = _terrain_job(worker, tile_w=32)
    await _run(worker, job_id)

    doc = json.loads((worker.config.job_dir(job_id) / "sheet.json").read_text())
    assert doc["mode"] == "terrain"
    assert doc["layout"] == "blob47"
    assert doc["columns"] == tilemask.TILE_COUNT and doc["rows"] == 1
    assert len(doc["materials"]) == tilemask.TILE_COUNT
    assert doc["terrains"] == [
        {"name": "grass", "fill": [40, 120, 40, 255], "outline": [20, 60, 20, 255]}
    ]
    # Stamped by the sidecar rather than copied from the stored block: it says
    # which field implementation drew these pixels.
    assert doc["mask"]["version"] == tilemask.MASK_VERSION
    assert doc["mask"]["seed"] == 5
    # Both halves by name, because a per-column record cannot carry two
    # materials across forty-seven columns.
    seeds = tileatlas.material_seeds(200, 2)
    assert doc["recipe"]["terrain"]["inner"] == {"prompt": "grass", "seed": seeds[0]}
    assert doc["recipe"]["terrain"]["outer"]["prompt"] == "still water"


@pytest.mark.asyncio
async def test_a_terrain_row_names_two_materials_and_a_third_is_refused(worker):
    """Which of the two is inner is not a convention -- ``blob_rects`` makes the
    centre cell a member always, so it is which of them the forty-seven pictures
    are of."""
    job_id = _terrain_job(worker)
    params = worker.store.get(job_id)["params"]
    params["sheet"]["materials"].append(
        {"index": 2, "prompt": "sand", "variant": 1, "seed": 9}
    )
    worker.store.set_params(job_id, params)

    row = await _run(worker, job_id)
    assert row["status"] == "error"
    assert "inner and an outer" in (row["error"] or "")


# --- dispatch and compatibility ---------------------------------------------


@pytest.mark.asyncio
async def test_a_stored_version_two_block_still_reaches_the_grid_coroutine(worker):
    """``rerun_job`` copies params forward, so every row written before this
    path existed carries a block with no ``mode`` at all. Absent reads as
    grid."""
    job_id = _legacy_grid_job(worker)
    row = await _run(worker, job_id)

    assert row["error"] is None and row["status"] == "done"
    pipe = worker._text2image
    # One generation on the grid template, and it must not wrap: the sheet's
    # leftmost and rightmost columns are different tiles.
    assert pipe.tilesheets == [True]
    assert pipe.tiles == [False]
    doc = json.loads((worker.config.job_dir(job_id) / "sheet.json").read_text())
    assert doc["version"] == tilesheet.TILE_SHEET_VERSION
    assert doc["tiles"] == 64
    assert not (worker.config.job_dir(job_id) / "materials").exists()


@pytest.mark.asyncio
async def test_a_mode_this_build_does_not_know_is_refused_not_drawn_as_a_grid(worker):
    """A row from a *newer* build. Falling through to the grid path would spend
    the card drawing something nobody asked for."""
    job_id = _materials_job(worker)
    params = worker.store.get(job_id)["params"]
    params["sheet"]["mode"] = "hexagonal"
    worker.store.set_params(job_id, params)

    row = await _run(worker, job_id)
    assert row["status"] == "error"
    assert "hexagonal" in (row["error"] or "")
    assert worker._text2image is None or not worker._text2image.prompts


# --- refusals and cancellation ----------------------------------------------


@pytest.mark.asyncio
async def test_a_block_that_names_no_materials_costs_no_generation(worker):
    job_id = _materials_job(worker)
    params = worker.store.get(job_id)["params"]
    params["sheet"]["materials"] = []
    worker.store.set_params(job_id, params)

    row = await _run(worker, job_id)
    assert row["status"] == "error"
    assert "names none" in (row["error"] or "")
    assert worker._text2image is None or not worker._text2image.prompts


@pytest.mark.asyncio
async def test_a_material_with_no_seed_is_corruption_not_a_default(worker):
    """The door derives the seeds and stores them, so a missing one means the
    row was written by something else -- and a guessed seed would publish a
    sidecar claiming a draw that cannot be re-run."""
    job_id = _materials_job(worker)
    params = worker.store.get(job_id)["params"]
    del params["sheet"]["materials"][1]["seed"]
    worker.store.set_params(job_id, params)

    row = await _run(worker, job_id)
    assert row["status"] == "error"
    assert "which seed" in (row["error"] or "")


@pytest.mark.asyncio
async def test_a_tile_size_that_cannot_partition_the_frame_costs_no_generation(worker):
    """48 is in ``tilesheet.TILE_SIZES`` and does not divide 1024. A material is
    a torus, so its first and last block are neighbours -- and a block one pixel
    wider than its opposite number puts a step at the wrap seam."""
    job_id = _materials_job(worker)
    params = worker.store.get(job_id)["params"]
    params["sheet"]["tile_w"] = 48
    worker.store.set_params(job_id, params)

    row = await _run(worker, job_id)
    assert row["status"] == "error"
    assert "48" in (row["error"] or "")
    assert worker._text2image is None or not worker._text2image.prompts


@pytest.mark.asyncio
async def test_a_view_that_cannot_tile_costs_no_generation(worker):
    job_id = _materials_job(worker)
    params = worker.store.get(job_id)["params"]
    params["sheet"]["projection"] = "isometric"
    worker.store.set_params(job_id, params)

    row = await _run(worker, job_id)
    assert row["status"] == "error"
    assert "2:1 diamond" in (row["error"] or "")


@pytest.mark.asyncio
async def test_a_cancel_before_the_third_pass_publishes_nothing(worker, monkeypatch):
    """The cancel check sits before each pass and not only after the last:
    every one is ~20 s of GPU a cancelled job should not spend."""
    import warlock.pipelines.t2i_client as t2i_client_mod
    import warlock.pipelines.text2image as text2image_mod

    # Read back off the module the ``fake_pipelines`` fixture already patched,
    # rather than imported from ``tests.conftest`` -- conftest is loaded as a
    # top-level module by pytest and there is no ``tests`` package to import it
    # through.
    class CancelAfterTwo(text2image_mod.Text2Image):
        """Sets the worker's own cancel event once two materials are drawn.

        Deliberately not a timing race: the point under test is *where* the
        check sits in the loop, and a wall-clock cancel against three passes of
        a fake that takes 15 ms each would land wherever the scheduler put it.
        """

        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            if len(self.prompts) == 2 and worker._cancel is not None:
                worker._cancel.event.set()
            return result

    monkeypatch.setattr(text2image_mod, "Text2Image", CancelAfterTwo)
    monkeypatch.setattr(t2i_client_mod, "Text2ImageClient", CancelAfterTwo)

    job_id = _materials_job(worker)
    row = await _run(worker, job_id)

    assert row["status"] == "cancelled"
    # The third pass never ran, and nothing at all was published.
    assert len(worker._text2image.prompts) == 2
    job_dir = worker.config.job_dir(job_id)
    assert not (job_dir / "input.png").exists()
    assert not (job_dir / "sheet.json").exists()
    assert not (job_dir / "materials").exists()


def test_a_cancelled_tileset_leaves_no_materials_behind(worker, tmp_path):
    """``_discard_artifacts``' tile-sheet branch reaches this job's own
    directory, and the un-reduced materials are as much this run's output as
    ``input.png`` is. A directory, so it needs the tree removal rather than the
    unlink loop."""
    job_id = _materials_job(worker)
    job_dir = worker.config.job_dir(job_id)
    materials = job_dir / "materials"
    materials.mkdir(parents=True, exist_ok=True)
    (materials / "00.png").write_bytes(b"not really a png")
    (job_dir / "input.png").write_bytes(b"nor this")
    # The user's own attached reference, which is deliberately *not* on the
    # list: the door wrote it before the row existed, so it is an input.
    (job_dir / "ref.png").write_bytes(b"the user's picture")

    worker._discard_artifacts(worker.store.get(job_id))

    assert not materials.exists()
    assert not (job_dir / "input.png").exists()
    assert (job_dir / "ref.png").exists()


# --- the palette tail -------------------------------------------------------
#
# The same three claims ``test_tilesheet_worker`` makes about the grid, made
# again about the two seamless modes -- because the three of them share one
# quantize tail (``tilesheet.quantize_tiles``) and a claim about a shared tail
# that is only tested through one of its callers is a claim about one caller.
# The reconstructions below go through ``reduce_material`` and then through
# ``assemble`` or ``blob_atlas``, so they pin the measured reduction in front of
# the quantiser at the same time.


@pytest.fixture
def paldir(worker, tmp_path):
    directory = tmp_path / "palettes"
    directory.mkdir(exist_ok=True)
    worker.config.palette_dir = directory
    return directory


_QUAD = ((26, 28, 44), (93, 39, 93), (239, 125, 87), (255, 205, 117))


def _write_quad(paldir):
    (paldir / "quad.hex").write_text(
        "".join(f"#{r:02x}{g:02x}{b:02x}\n" for r, g, b in _QUAD)
    )


def _published(worker, job_id):
    import numpy as np

    with Image.open(worker.config.job_dir(job_id) / "input.png") as sheet:
        sheet.load()
        return np.asarray(sheet.convert("RGBA"))


def _reduced_materials(worker, job_id, geom):
    """The provenance copies put back through the measured reducer.

    The materials directory is byte-for-byte what the model returned, which is
    the entire point of keeping it -- so a reconstruction from it is a
    reconstruction from the generation and not from anything this path did to
    it afterwards.
    """
    import numpy as np

    directory = worker.config.job_dir(job_id) / "materials"
    out = []
    for path in sorted(directory.glob("*.png")):
        with Image.open(path) as raw:
            raw.load()
            frame = raw.convert("RGBA")
        out.append(
            tileatlas.reduce_material(np.asarray(frame), geom.tile_w, geom.tile_h)
        )
    return out


@pytest.mark.asyncio
async def test_a_default_materials_atlas_is_the_median_cut_it_always_was(worker):
    import numpy as np

    from warlock.pipelines.pixelsheet import quantize_shared

    job_id = _materials_job(worker, tile_w=32, colors=16)
    await _run(worker, job_id)

    geom = tileatlas.material_geometry(32, "top_down", 3)
    atlas = tileatlas.assemble(_reduced_materials(worker, job_id, geom), geom)
    want, _hexes = quantize_shared(Image.fromarray(atlas, "RGBA"), 16)
    assert np.array_equal(_published(worker, job_id), np.asarray(want))


@pytest.mark.asyncio
async def test_a_default_terrain_set_is_the_median_cut_it_always_was(worker):
    import numpy as np

    from warlock.pipelines.pixelsheet import quantize_shared

    job_id = _terrain_job(worker, tile_w=32, colors=16)
    await _run(worker, job_id)

    geom = tileatlas.terrain_geometry(32, "top_down")
    tiles = _reduced_materials(worker, job_id, geom)
    atlas = tilemask.blob_atlas(
        tiles[0], tiles[1], geom.tile_w, seed=5, inset=None, amplitude=None, feather=None
    )
    want, _hexes = quantize_shared(Image.fromarray(atlas, "RGBA"), 16)
    assert np.array_equal(_published(worker, job_id), np.asarray(want))


@pytest.mark.asyncio
async def test_the_measured_reducer_still_runs_once_per_material(worker, monkeypatch):
    """``pixelize.reduce``'s single box mean is what
    ``docs/measurements/2026-08-17-ground-reduction.md`` rejected, so this path
    keeps ``reduce_material`` and only the quantisation moved. Once per
    *material*, never per cell: a terrain set is forty-seven composites of two
    generations."""
    real = tileatlas.reduce_material
    calls: list[tuple[int, int]] = []

    def spy(pixels, out_w, out_h):
        calls.append((int(out_w), int(out_h)))
        return real(pixels, out_w, out_h)

    monkeypatch.setattr(tileatlas, "reduce_material", spy)
    await _run(worker, _terrain_job(worker, tile_w=32))

    assert calls == [(32, 32), (32, 32)]


@pytest.mark.asyncio
async def test_a_designed_palette_is_one_table_over_every_material_cell(worker, paldir):
    """The assertion a per-cell quantisation fails. Each material comes back a
    flat colour of its own, so quantized separately every cell would keep its
    own three bytes rather than land on the table."""
    _write_quad(paldir)
    job_id = _materials_job(worker, tile_w=32, colors=16, palette="quad")
    await _run(worker, job_id)

    atlas = _published(worker, job_id)
    geom = tileatlas.material_geometry(32, "top_down", 3)
    for cell in geom.cells:
        top, left = cell.row * geom.tile_h, cell.col * geom.tile_w
        block = atlas[top : top + geom.tile_h, left : left + geom.tile_w, :3]
        used = {tuple(int(c) for c in p) for p in block.reshape(-1, 3)}
        assert used <= set(_QUAD), f"cell {cell.index} left the palette"


@pytest.mark.asyncio
async def test_a_designed_palette_covers_all_forty_seven_terrain_cells(worker, paldir):
    _write_quad(paldir)
    job_id = _terrain_job(worker, tile_w=32, colors=16, palette="quad")
    await _run(worker, job_id)

    atlas = _published(worker, job_id)
    geom = tileatlas.terrain_geometry(32, "top_down")
    assert geom.tiles == 47
    for cell in geom.cells:
        top, left = cell.row * geom.tile_h, cell.col * geom.tile_w
        block = atlas[top : top + geom.tile_h, left : left + geom.tile_w, :3]
        used = {tuple(int(c) for c in p) for p in block.reshape(-1, 3)}
        assert used <= set(_QUAD), f"cell {cell.index} left the palette"


@pytest.mark.asyncio
async def test_the_recipe_names_the_palette_file_and_a_digest_of_its_colours(
    worker, paldir
):
    from warlock.pipelines import pixel

    _write_quad(paldir)
    job_id = _materials_job(worker, palette="quad", dither=True)
    await _run(worker, job_id)

    recipe = json.loads(
        (worker.config.job_dir(job_id) / "sheet.json").read_text()
    )["recipe"]
    assert recipe["palette_source"] == "designed"
    assert recipe["palette_file"] == "quad"
    assert recipe["palette_hash"] == pixel.palette_digest(_QUAD)
    assert recipe["dither"] is True
    # The budget is still recorded -- it is what the row asked for and a reroll
    # copies it -- and ``palette_source`` is what says it was superseded.
    assert "colors" in recipe


@pytest.mark.asyncio
async def test_a_default_set_records_no_palette_keys_at_all(worker):
    job_id = _materials_job(worker)
    await _run(worker, job_id)

    doc = json.loads((worker.config.job_dir(job_id) / "sheet.json").read_text())
    assert not {
        "palette_source", "palette_file", "palette_hash", "dither"
    } & set(doc["recipe"])
    assert json.loads(json.dumps(doc)) == doc


@pytest.mark.asyncio
async def test_a_palette_deleted_after_the_door_costs_no_generations(worker):
    """N generations, not one -- which is why this path resolves the palette
    before the bracket rather than at the quantize phase where it is used."""
    row = await _run(worker, _materials_job(worker, palette="gone"))

    assert row["status"] == "error"
    assert "no longer installed" in (row["error"] or "")
    assert worker._text2image is None or not worker._text2image.prompts
