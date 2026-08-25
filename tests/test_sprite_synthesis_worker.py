"""``Worker._sprite_synthesis``: two generations, one draft, published last.

Driven through the real dispatcher with ``fake_pipelines``, so what is under
test is the worker's control flow -- what it conditions on, what it publishes,
and what a cancel leaves behind -- rather than anything about SDXL.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time

import pytest
from PIL import Image

from warlock import models, rigging
from warlock.config import Config
from warlock.db import JobStore
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


def _reference(worker) -> str:
    """A finished reference with a dark subject on a light background."""
    job_id = worker.store.create("text", "a knight", {"seed": 1}, stage="reference")
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (256, 256), (220, 220, 220))
    Image.Image.paste(
        image, Image.new("RGB", (100, 160), (30, 40, 50)), (78, 60)
    )
    image.save(job_dir / "input.png")
    worker.store.set_status(job_id, "done")
    return job_id


def _queue(worker, source, **overrides) -> tuple[str, str]:
    draft_id = rigging.new_id()
    params = {
        "source_job": source,
        "sheet_type": "turnaround",
        "logical_size": 64,
        "colors": 16,
        "seed_a": 11,
        "seed_b": 22,
        "draft_id": draft_id,
        "base_model": "sdxl_cfg",
    }
    params.update(overrides)
    job_id = worker.store.create("sprite_synthesis", "a knight", params)
    return job_id, params["draft_id"]


async def _run(worker, job_id, *, on_start=None):
    worker.start()
    try:
        deadline = time.monotonic() + 90.0
        if on_start is not None:
            await on_start()
        while time.monotonic() < deadline:
            if worker.store.get(job_id)["status"] in ("done", "error", "cancelled"):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("job did not finish before timeout")
    finally:
        await worker.shutdown()
    return worker.store.get(job_id)


# --- the happy path ---------------------------------------------------------


@pytest.mark.asyncio
async def test_two_generations_run_with_the_two_seeds(worker):
    source = _reference(worker)
    job_id, _draft_id = _queue(worker, source)

    row = await _run(worker, job_id)

    assert row["error"] is None and row["status"] == "done"
    pipe = worker._text2image
    assert pipe.seeds == [11, 22]
    # ``sheet=True`` for both: the atlas is one square generation, which is
    # what that flag selects.
    assert pipe.sheets == [True, True]


@pytest.mark.asyncio
async def test_both_adapters_ride_every_pass_and_no_init_does(worker):
    """The pose guide and the identity are the whole feature, and an init
    image would fight the guides in three of the four cells."""
    source = _reference(worker)
    job_id, _ = _queue(worker, source)

    await _run(worker, job_id)

    for cond in worker._text2image.conditionings:
        assert cond.uses_ip and cond.uses_control
        assert cond.ip_adapter == "plus" and cond.control == "canny"
        assert cond.uses_init is False


@pytest.mark.asyncio
async def test_the_draft_is_published_as_a_trio_with_the_sidecar_last(worker):
    source = _reference(worker)
    job_id, draft_id = _queue(worker, source)

    await _run(worker, job_id)

    source_dir = worker.config.job_dir(source)
    for letter in rigging.SPRITE_CANDIDATES:
        assert rigging.sprite_draft_png_path(source_dir, draft_id, letter).exists()
    record = rigging.read_sprite_draft(source_dir, draft_id)
    assert record is not None
    assert [c["seed"] for c in record["candidates"]] == [11, 22]
    assert [c["image"] for c in record["candidates"]] == [
        f"{draft_id}.a.png",
        f"{draft_id}.b.png",
    ]
    assert rigging.list_sprite_drafts(source_dir) == [record]


@pytest.mark.asyncio
async def test_the_published_atlas_matches_the_sidecars_grid(worker):
    source = _reference(worker)
    job_id, draft_id = _queue(worker, source, sheet_type="walk", logical_size=48)

    await _run(worker, job_id)

    source_dir = worker.config.job_dir(source)
    record = rigging.read_sprite_draft(source_dir, draft_id)
    assert record["sheet_type"] == "walk"
    assert len(record["cells"]) == 16
    with Image.open(rigging.sprite_draft_png_path(source_dir, draft_id, "a")) as png:
        assert png.size == (record["columns"] * 48, record["rows"] * 48)


@pytest.mark.asyncio
async def test_the_recipe_records_the_guide_and_the_conditioning(worker):
    source = _reference(worker)
    job_id, draft_id = _queue(worker, source)

    await _run(worker, job_id)

    recipe = rigging.read_sprite_draft(worker.config.job_dir(source), draft_id)["recipe"]
    assert recipe["base_model"] == "sdxl_cfg"
    assert recipe["style_lora"] == models.PIXEL_SHEET_LORA
    assert recipe["guide_template"] == "turnaround"
    # Recorded so that "my pose guide did nothing" is an answerable question
    # rather than a suspicion.
    assert recipe["guide_edge_fraction"] > 0.0
    assert recipe["control"] == "canny" and recipe["ip_adapter"] == "plus"


@pytest.mark.asyncio
async def test_a_candidate_with_warnings_is_still_published(worker):
    """Never discard a completed candidate: the pair is the deliverable, and
    one draft with a note beats one draft and nothing to compare it to."""
    source = _reference(worker)
    job_id, draft_id = _queue(worker, source)

    await _run(worker, job_id)

    record = rigging.read_sprite_draft(worker.config.job_dir(source), draft_id)
    # The fake pipeline paints a flat colour, so every cell is "unmatted" and
    # runs off its own edges -- exactly the shape of a warned candidate.
    assert all("warnings" in c for c in record["candidates"])
    assert len(record["candidates"]) == 2


# --- the awkward paths ------------------------------------------------------


@pytest.mark.asyncio
async def test_a_base_the_pixel_lora_does_not_fit_generates_bare_and_says_so(
    worker, monkeypatch
):
    """The known-key path the door cannot reach, exactly as ``_pixel_sheet``
    has: params outlive the service that wrote them."""
    monkeypatch.setitem(
        models.STYLE_LORAS,
        models.PIXEL_SHEET_LORA,
        dataclasses.replace(
            models.STYLE_LORAS[models.PIXEL_SHEET_LORA],
            family=models.FAMILY_FLUX2_KLEIN,
        ),
    )
    source = _reference(worker)
    job_id, draft_id = _queue(worker, source)

    pipes: list = []
    real = worker._get_text2image

    async def capture(base_key):
        pipe = await real(base_key)
        pipes.append(pipe)
        return pipe

    monkeypatch.setattr(worker, "_get_text2image", capture)

    row = await _run(worker, job_id)

    assert row["error"] is None and row["status"] == "done"
    assert {lora for lora, _weight in pipes[0].lora_calls} == {None}
    recipe = rigging.read_sprite_draft(worker.config.job_dir(source), draft_id)["recipe"]
    assert "style_lora" not in recipe


@pytest.mark.asyncio
async def test_a_cancel_publishes_nothing(worker, monkeypatch):
    """Nothing is written until both candidates are assembled, so a cancel
    finds nothing half-published -- and ``_discard_artifacts`` is never left
    deleting files it had just been told to make."""
    source = _reference(worker)
    job_id, draft_id = _queue(worker, source)

    # Slowed at the pipe, the way the LoRA test captures it, so the cancel
    # lands *inside* the first generation rather than racing a 60 ms one.
    real = worker._get_text2image

    async def slow(base_key):
        pipe = await real(base_key)
        pipe.steps = 400
        return pipe

    monkeypatch.setattr(worker, "_get_text2image", slow)

    worker.start()
    try:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if worker.store.get(job_id)["status"] == "running":
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.5)
        await worker.request_cancel(job_id)
        while time.monotonic() < deadline:
            if worker.store.get(job_id)["status"] in ("done", "error", "cancelled"):
                break
            await asyncio.sleep(0.01)
    finally:
        await worker.shutdown()

    assert worker.store.get(job_id)["status"] == "cancelled"
    source_dir = worker.config.job_dir(source)
    assert rigging.read_sprite_draft(source_dir, draft_id) is None
    assert rigging.list_sprite_drafts(source_dir) == []


@pytest.mark.asyncio
async def test_a_discard_leaves_a_strangers_drafts_alone(worker):
    """``_discard_artifacts`` names this job's draft id. The directory holds
    every earlier draft of the same reference, each from a different and
    successful job."""
    source = _reference(worker)
    stranger = rigging.new_id()
    source_dir = worker.config.job_dir(source)
    for letter in rigging.SPRITE_CANDIDATES:
        path = rigging.sprite_draft_png_path(source_dir, stranger, letter)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
    rigging.sprite_draft_path(source_dir, stranger).write_text("{}", encoding="utf-8")

    job_id, draft_id = _queue(worker, source)
    worker._discard_artifacts(worker.store.get(job_id))

    assert rigging.sprite_draft_path(source_dir, stranger).exists()


def test_a_discard_with_no_draft_id_touches_nothing(worker):
    source = _reference(worker)
    job_id = worker.store.create(
        "sprite_synthesis", "a knight", {"source_job": source, "draft_id": "../oops"}
    )
    # Returns rather than raising, and writes nothing: a malformed id becomes a
    # path, which is the class of bug the guard removes.
    worker._discard_artifacts(worker.store.get(job_id))


@pytest.mark.asyncio
async def test_a_deleted_source_errors_cleanly(worker):
    source = _reference(worker)
    job_id, _ = _queue(worker, source)
    (worker.config.job_dir(source) / "input.png").unlink()

    row = await _run(worker, job_id)

    assert row["status"] == "error"
    assert "no longer has an image" in (row["error"] or "")


@pytest.mark.asyncio
async def test_an_unknown_sheet_type_errors_rather_than_defaulting(worker):
    source = _reference(worker)
    job_id, _ = _queue(worker, source, sheet_type="isometric")

    row = await _run(worker, job_id)

    assert row["status"] == "error"
    assert "unknown sprite sheet type" in (row["error"] or "")


@pytest.mark.asyncio
async def test_a_draft_png_is_staged_rather_than_saved_onto_its_served_name(
    worker, monkeypatch
):
    """The rule this file's own restyle sibling states over
    ``sheet_pixel_png_path``: a write onto a served path is staged.

    This was the one served writer in ``_q_sprite`` not following it. Nothing
    normally holds these names -- the draft id is minted at the door -- and
    that is precisely the argument the restyle rejected: "a bare save tears a
    file that is already being served" does not stop being true because today's
    callers happen never to reuse an id, and ``_discard_artifacts`` deletes by
    that same id on the assumption that they do not.
    """
    from pathlib import Path

    saved: list[Path] = []
    real_save = Image.Image.save

    def record(self, fp, *args, **kwargs):
        saved.append(Path(fp))
        return real_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", record)

    source = _reference(worker)
    job_id, draft_id = _queue(worker, source)
    row = await _run(worker, job_id)
    assert row["status"] == "done", row["error"]

    source_dir = worker.config.job_dir(source)
    served = {
        rigging.sprite_draft_png_path(source_dir, draft_id, letter)
        for letter in rigging.SPRITE_CANDIDATES
    }
    assert served.isdisjoint(saved), "a served draft PNG was written in place"
    assert all(path.exists() for path in served), "and the renames still landed"
    assert list(served.pop().parent.glob(".*.tmp")) == []


@pytest.mark.asyncio
async def test_a_torn_draft_sidecar_leaves_no_marker_and_no_strand(worker, monkeypatch):
    """``rigging.list_sprite_drafts`` treats the sidecar as the completion
    marker, so a half-written one advertises a draft whose record cannot be
    parsed -- and a re-synthesis of the same draft_id would be truncating a
    marker that is already saying ready."""
    from pathlib import Path

    source = _reference(worker)
    job_id, draft_id = _queue(worker, source)

    real = Path.write_text

    def tamper(self, text, *a, **k):
        if self.name == f".{draft_id}.json.tmp":
            real(self, text[: len(text) // 2], *a, **k)
            raise OSError("the disk filled up")
        return real(self, text, *a, **k)

    monkeypatch.setattr(Path, "write_text", tamper)
    row = await _run(worker, job_id)

    assert row["status"] == "error"
    source_dir = worker.config.job_dir(source)
    assert not rigging.sprite_draft_path(source_dir, draft_id).exists()
    assert list(rigging.sprite_draft_path(source_dir, draft_id).parent.glob("*.tmp")) == []
