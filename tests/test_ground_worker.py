"""The ground-set worker: what it generates, what it publishes, and when not to.

Three claims. Every generation is a *tile* generation, because a texture drawn
without circular padding seams in a painted field and looks fine in a thumbnail.
The atlas is published before its sidecar, because the sidecar is the completion
marker. And a cancelled paint publishes nothing at all, because a half-written
47-column sheet is one ``use_as_tileset`` would happily slice into forty-seven
tiles of nothing.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import json
import time
from pathlib import Path

import pytest
from PIL import Image

from warlock import models
from warlock.config import Config
from warlock.db import JobStore
from warlock.pipelines import ground
from warlock.queue import Worker
from warlock.studio.plotter import blob


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


def _terrains(count: int = 2):
    names = ("stone", "water", "lava", "moss")
    return [
        {
            "name": names[i],
            "material": f"{names[i]} surface",
            "edge": "",
            "color": [40 + i * 20, 90, 140, 255],
        }
        for i in range(count)
    ]


def _ground_job(worker, *, terrains=2, tile_w=16, tile_h=16, projection="orthogonal",
                seed=7, border=0, colors=8) -> str:
    return worker.store.create(
        "ground_set",
        "a damp dungeon",
        {
            "seed": seed,
            "base_model": "sdxl_cfg",
            "style_lora": models.PIXEL_SHEET_LORA,
            "colors": colors,
            "negative_prompt": "",
            "ground": {
                "version": 1,
                "tile_w": tile_w,
                "tile_h": tile_h,
                "projection": projection,
                "border": border,
                "terrains": _terrains(terrains),
            },
        },
        stage="ground",
    )


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


# --- what it generates -------------------------------------------------------


@pytest.mark.asyncio
async def test_every_terrain_gets_two_seamless_generations(worker):
    job_id = _ground_job(worker, terrains=3)
    row = await _run(worker, job_id)

    assert row["error"] is None and row["status"] == "done"
    pipe = worker._text2image
    assert len(pipe.prompts) == 3 * ground.TEXTURES_PER_TERRAIN
    # Every one of them, not most: a single non-tiled texture is one terrain
    # that seams while its neighbours do not.
    assert set(pipe.tiles) == {True}
    assert pipe.sheets == [False] * len(pipe.prompts)


@pytest.mark.asyncio
async def test_the_seeds_fan_out_from_the_one_stored_seed(worker):
    job_id = _ground_job(worker, terrains=2, seed=7)
    await _run(worker, job_id)

    expected = [
        ground.texture_seed(7, index, role)
        for index in range(2)
        for role in ground.ROLES
    ]
    assert worker._text2image.seeds == expected


@pytest.mark.asyncio
async def test_the_checkpoint_is_loaded_once_for_the_whole_sheet(worker):
    """Sixteen generations behind one handoff. Loading per texture would pay the
    expensive part sixteen times for a deliverable that is one sheet."""
    job_id = _ground_job(worker, terrains=3)
    await _run(worker, job_id)

    pipe = worker._text2image
    assert len(pipe.prompts) == 6
    assert pipe.unload_calls <= 1


@pytest.mark.asyncio
async def test_each_terrains_own_material_reaches_its_prompt(worker):
    job_id = _ground_job(worker, terrains=2)
    await _run(worker, job_id)

    prompts = worker._text2image.prompts
    assert "stone surface" in prompts[0]
    assert "water surface" in prompts[2]
    # The border prompt is the rim of that terrain's own material, not a
    # generic edge -- a grey rim around a green field is a different terrain.
    assert ground.default_edge("stone surface") in prompts[1]


# --- what it publishes -------------------------------------------------------


@pytest.mark.asyncio
async def test_the_published_atlas_is_a_47_column_terrain_sheet(worker):
    job_id = _ground_job(worker, terrains=2, tile_w=16, tile_h=8)
    await _run(worker, job_id)

    png = worker.config.job_dir(job_id) / "input.png"
    with Image.open(png) as opened:
        assert opened.size == (blob.TILE_COUNT * 16, 2 * 8)


@pytest.mark.asyncio
async def test_the_generated_textures_are_kept_beside_the_atlas(worker):
    job_id = _ground_job(worker, terrains=2)
    await _run(worker, job_id)

    textures = worker.config.job_dir(job_id) / "textures"
    assert sorted(p.name for p in textures.glob("*.png")) == [
        "t0-border.png", "t0-fill.png", "t1-border.png", "t1-fill.png"
    ]


@pytest.mark.asyncio
async def test_the_sidecar_is_written_after_the_atlas(worker, monkeypatch):
    """It is the completion marker: publishing it first would advertise a sheet
    that is still being written."""
    import warlock.queue as queue_mod

    real = queue_mod._publish_text
    seen: list[bool] = []

    def spy(path, text):
        seen.append((Path(path).parent / "input.png").exists())
        return real(path, text)

    monkeypatch.setattr(queue_mod, "_publish_text", spy)
    job_id = _ground_job(worker)
    await _run(worker, job_id)

    assert seen == [True]
    doc = json.loads((worker.config.job_dir(job_id) / "ground.json").read_text())
    assert doc["version"] == ground.GROUND_VERSION
    assert doc["image"] == "input.png"
    assert doc["palette"]


@pytest.mark.asyncio
async def test_the_row_records_what_was_painted(worker):
    job_id = _ground_job(worker, terrains=2)
    row = await _run(worker, job_id)

    report = row["params"]["ground_report"]
    assert report["textures"] == 4
    assert report["recipe"]["style_lora"] == models.PIXEL_SHEET_LORA


@pytest.mark.asyncio
async def test_an_isometric_set_leaves_the_corners_transparent(worker):
    job_id = _ground_job(worker, terrains=1, tile_w=32, tile_h=16, projection="isometric")
    await _run(worker, job_id)

    png = worker.config.job_dir(job_id) / "input.png"
    with Image.open(png) as opened:
        atlas = opened.convert("RGBA")
    # The interior case, whose diamond is inscribed in its rectangle: the four
    # corners of that rectangle are outside the cell entirely.
    x0 = blob.FULL * 32
    assert atlas.getpixel((x0, 0))[3] == 0
    assert atlas.getpixel((x0 + 16, 8))[3] == 255


# --- what it does not publish ------------------------------------------------


@pytest.mark.asyncio
async def test_a_cancelled_paint_publishes_nothing(worker):
    job_id = _ground_job(worker, terrains=4)
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
    assert not (job_dir / "ground.json").exists()
    assert not (job_dir / "textures").exists()


@pytest.mark.asyncio
async def test_a_base_the_pixel_lora_does_not_fit_paints_bare_and_says_so(
    worker, monkeypatch
):
    """The known-key path the door cannot reach -- params outlive the service
    that wrote them -- so the recipe records what ran, not what was asked for."""
    monkeypatch.setitem(
        models.STYLE_LORAS,
        models.PIXEL_SHEET_LORA,
        dataclasses.replace(
            models.STYLE_LORAS[models.PIXEL_SHEET_LORA],
            family=models.FAMILY_FLUX2_KLEIN,
        ),
    )
    pipes: list = []
    real = worker._get_text2image

    async def capture(base_key):
        pipe = await real(base_key)
        pipes.append(pipe)
        return pipe

    monkeypatch.setattr(worker, "_get_text2image", capture)
    job_id = _ground_job(worker, terrains=1)
    row = await _run(worker, job_id)

    assert row["status"] == "done"
    assert {lora for lora, _weight in pipes[0].lora_calls} == {None}
    doc = json.loads((worker.config.job_dir(job_id) / "ground.json").read_text())
    assert "style_lora" not in doc["recipe"]


@pytest.mark.asyncio
async def test_a_set_with_no_terrains_errors_rather_than_publishing(worker):
    """Params outlive the door that validated them, so the block is re-derived
    rather than trusted."""
    job_id = worker.store.create(
        "ground_set", "a theme", {"seed": 1, "ground": {"terrains": []}}, stage="ground"
    )
    row = await _run(worker, job_id)
    assert row["status"] == "error"
    assert not (worker.config.job_dir(job_id) / "input.png").exists()


# --- the cross-layer import, pinned -----------------------------------------


def test_the_worker_reaches_into_the_plotter_for_exactly_one_module():
    """The one queue-to-studio edge in the tree. It exists because the
    compositor is the piece both halves need and a second copy is how the
    worker's forty-seven cases come to disagree with the editor's -- so a
    *second* such import is a decision, not a detail."""
    import warlock._q_ground as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            base = "warlock." + (node.module or "")
            found.update(
                f"{base}.{alias.name}" for alias in node.names if base == "warlock.studio.plotter"
            )
            if node.module and node.module.startswith("studio"):
                found.update(f"warlock.{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(n.name for n in node.names if n.name.startswith("warlock.studio"))
    assert found == {"warlock.studio.plotter.groundtex"}


# --- phase variants ------------------------------------------------------------


def _phased_job(worker, *, phases: int, terrains: int = 1, tile: int = 16) -> str:
    params = {
        "seed": 7,
        "base_model": "sdxl_cfg",
        "style_lora": models.PIXEL_SHEET_LORA,
        "colors": 8,
        "negative_prompt": "",
        "ground": {
            "version": 1,
            "tile_w": tile,
            "tile_h": tile,
            "projection": "orthogonal",
            "border": 0,
            "phases": phases,
            "terrains": _terrains(terrains),
        },
    }
    return worker.store.create("ground_set", "a damp dungeon", params, stage="ground")


@pytest.mark.asyncio
async def test_a_phased_set_publishes_phase_squared_subrows(worker):
    job_id = _phased_job(worker, phases=2, terrains=2)
    row = await _run(worker, job_id)
    assert row["error"] is None and row["status"] == "done"

    job_dir = worker.config.job_dir(job_id)
    with Image.open(job_dir / "input.png") as atlas:
        assert atlas.size == (blob.TILE_COUNT * 16, 2 * 4 * 16)
    doc = json.loads((job_dir / "ground.json").read_text())
    assert doc["phases"] == 2
    assert row["params"]["ground_report"]["phases"] == 2


@pytest.mark.asyncio
async def test_a_block_without_phases_paints_the_classic_sheet(worker):
    """Every job stored before phases existed."""
    job_id = _ground_job(worker, terrains=2, tile_w=16, tile_h=16)
    row = await _run(worker, job_id)
    assert row["error"] is None and row["status"] == "done"
    job_dir = worker.config.job_dir(job_id)
    with Image.open(job_dir / "input.png") as atlas:
        assert atlas.size == (blob.TILE_COUNT * 16, 2 * 16)
    assert json.loads((job_dir / "ground.json").read_text())["phases"] == 1


@pytest.mark.asyncio
async def test_a_phase_count_outside_the_table_errors_before_the_card(worker):
    job_id = _phased_job(worker, phases=3)
    row = await _run(worker, job_id)
    assert row["status"] == "error"
    assert "1, 2 or 4" in row["error"]
    assert worker._text2image is None or not worker._text2image.prompts
