"""Phase 4: prompt -> approve -> a finished character sheet.

The chain is reference -> gate -> mesh -> rig -> sheet, and every link but the
last is machinery the repo already had. What these tests hold is the joins:

* the T-pose guide is *drawn*, never detected -- a canny pass over a stick
  figure returns two lines where the guide means one, which is the "why does my
  character have four legs" failure ``spritesynth.render_guide`` documents;
* the guide is redrawn per run rather than copied, so a rerolled reference --
  a new job id and an empty directory -- does not fail on a file the door wrote
  into the row it rerolled;
* the request rides across the promote gate, because the gate is the whole
  point of the two-step shape;
* every refusal a character sheet has is taken at a door, never in the worker,
  where it would cost 256 rendered frames.
"""

from __future__ import annotations

import asyncio

import pytest

from warlock import clips
from warlock.config import Config
from warlock.db import JobStore
from warlock.pipelines import charsheet, spritesynth
from warlock.queue import Worker
from warlock.service import jobs as svc_jobs
from warlock.service import troupe as svc_troupe
from warlock.service.errors import Invalid


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


async def _wait_until(predicate, timeout: float = 20.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


# -- the guide ---------------------------------------------------------------


def test_both_variants_load_and_draw():
    for variant in svc_troupe.TROUPE_VARIANTS:
        image = spritesynth.render_tpose_guide(variant)
        assert image.size == (spritesynth.ATLAS_PX, spritesynth.ATLAS_PX)


def test_an_unknown_variant_raises_rather_than_defaulting():
    """Silently conditioning on the other sex's guide is a wrong character
    published under the caller's name -- ``geometry``'s rule."""
    with pytest.raises(ValueError):
        spritesynth.load_tpose_guide("enby-warrior")


def test_the_tpose_grid_is_not_a_sprite_sheet_kind():
    """``tests/test_sprite_geometry_agreement.py`` owns the claim that
    GEOMETRY, SHEET_TYPES and inker's SHEET_KINDS name one set. Registering the
    T-pose grid there to save a name would make that test say something
    weaker."""
    assert "tpose" not in spritesynth.GEOMETRY
    assert "tpose" not in spritesynth.SHEET_TYPES
    assert len(spritesynth.TPOSE_GEOMETRY.cells) == 1


def test_the_guide_is_line_art_rather_than_a_photograph():
    """It is handed to the ControlNet directly, so it has to already be white
    strokes on black: a few percent of lit pixels, and nothing between."""
    from warlock.pipelines import control

    fraction = control.edge_fraction(spritesynth.render_tpose_guide("male"))
    assert 0.0005 < fraction < 0.05


# -- the clip library --------------------------------------------------------


def test_the_shipped_clips_fill_the_frame_table():
    """The one thing that makes a character sheet renderable at all. A
    seven-frame walk laid into an eight-frame table renders one cell of some
    other animation, and the user goes looking at the rig."""
    records = clips.expand_clips("humanoid")
    assert {k: len(v) for k, v in records.items()} == {
        name: frames for name, frames, _loop, _ms in charsheet.ANIMATIONS
    }


def test_a_template_with_no_clips_is_a_keyerror_not_an_empty_sheet():
    with pytest.raises(KeyError):
        clips.expand_clips("fish")


# -- the door ----------------------------------------------------------------


def test_the_block_is_stored_on_the_reference_job(svc):
    made = svc_jobs.create_job(
        svc,
        kind="text",
        prompt="a hooded ranger",
        output="reference",
        troupe={"variant": "female", "logical_size": 48, "colors": 32},
    )
    block = svc.store.get(made["id"])["params"]["troupe"]
    assert block["variant"] == "female"
    assert block["logical_size"] == 48
    assert block["colors"] == 32


def test_the_reference_is_wired_to_draw_its_own_guide(svc):
    """``control`` without a ``ref.png`` is normally a refusal at this very
    door. Troupe's hint is not derived from a reference at all, so the keys go
    on after that check and ``_conditioning`` reads them."""
    made = svc_jobs.create_job(
        svc, kind="text", prompt="a ranger", output="reference", troupe={}
    )
    params = svc.store.get(made["id"])["params"]
    assert params["control"] == "canny"
    assert params["control_hint_source"] == "guide"
    assert params["guide_variant"] == svc_troupe.DEFAULT_TROUPE_VARIANT
    assert not (svc.job_dir(made["id"]) / "ref.png").exists()


def test_the_rig_is_not_optional_and_is_measured(svc):
    """Every cell of a character sheet is a posed frame, so an unrigged mesh
    would render 256 copies of one T-pose -- and the shipped humanoid template
    is an A-pose that mis-fits a T-pose mesh."""
    made = svc_jobs.create_job(
        svc, kind="text", prompt="a ranger", output="reference", troupe={}
    )
    params = svc.store.get(made["id"])["params"]
    assert params["rig"] is True
    assert params["rig_template"] == svc_troupe.TROUPE_TEMPLATE
    assert params["rig_joints"] == "measured"


def test_a_character_sheet_cannot_be_asked_for_alongside_a_mesh(svc):
    with pytest.raises(Invalid) as excinfo:
        svc_jobs.create_job(
            svc, kind="text", prompt="a ranger", output="model", troupe={}
        )
    assert excinfo.value.field == "output"


def test_a_batch_of_characters_is_refused(svc):
    with pytest.raises(Invalid) as excinfo:
        svc_jobs.create_job(
            svc, kind="text", prompt="a ranger", output="reference", count=3, troupe={}
        )
    assert excinfo.value.field == "count"


@pytest.mark.parametrize(
    "block,field",
    [
        ({"variant": "robot"}, "variant"),
        ({"logical_size": 37}, "logical_size"),
        ({"colors": 7}, "colors"),
        ({"outline": "glow"}, "outline"),
        ({"reduce_mode": "lanczos"}, "reduce_mode"),
        ({"palette": "nothing-installed"}, "palette"),
    ],
)
def test_a_bad_option_is_refused_at_the_references_door(svc, block, field):
    """At *this* door, because the follow-up row is minted by the worker, which
    can refuse nothing -- so a bad option found there would be a refusal an
    hour later on a row the user never submitted."""
    with pytest.raises(Invalid) as excinfo:
        svc_jobs.create_job(
            svc, kind="text", prompt="a ranger", output="reference", troupe=block
        )
    assert excinfo.value.field == field


def test_a_plain_reference_stores_no_block(svc):
    made = svc_jobs.create_job(svc, kind="text", prompt="a ranger", output="reference")
    params = svc.store.get(made["id"])["params"]
    assert "troupe" not in params
    assert "control_hint_source" not in params


# -- the gate ----------------------------------------------------------------


def _finished_reference(svc, **block):
    made = svc_jobs.create_job(
        svc, kind="text", prompt="a ranger", output="reference", troupe=block
    )
    job_dir = svc.job_dir(made["id"])
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    svc.store.set_status(made["id"], "done")
    return made["id"]


def test_the_request_survives_the_promote_gate(svc):
    """The gate is the point of the two-step shape: the user approves a
    picture, and only then is the reconstruction spent. The request has to
    cross it, or approving the character would lose the sheet."""
    reference = _finished_reference(svc, logical_size=64)
    promoted = svc_jobs.promote_to_model(svc, reference)
    params = svc.store.get(promoted["id"])["params"]
    assert params["troupe"]["logical_size"] == 64
    assert params["rig"] is True


def test_the_guide_does_not_survive_the_promote_gate(svc):
    """A promotion is an image job: SDXL never runs, so a carried-over guide
    would describe a run that cannot happen."""
    reference = _finished_reference(svc)
    promoted = svc_jobs.promote_to_model(svc, reference)
    params = svc.store.get(promoted["id"])["params"]
    assert "control_hint_source" not in params
    assert "guide_variant" not in params
    assert "control" not in params


# -- the follow-up -----------------------------------------------------------


def _model_job(worker, **params):
    base = {
        "seed": 5,
        "rig": True,
        "rig_template": "humanoid",
        "troupe": {"logical_size": 32, "colors": 16, "outline": "outer"},
    }
    base.update(params)
    job_id = worker.store.create("image", "a ranger", base, stage="model")
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    return job_id


async def test_a_promoted_character_queues_its_sheet_behind_its_rig(worker):
    """Both follow-ups, in that order, and the order is the design: the queue
    is serial and FIFO, so the rig row is finished before the sheet row is
    claimed."""
    job_id = _model_job(worker)
    job = worker.store.get(job_id)
    await worker._maybe_queue_rig(job)
    await worker._maybe_queue_charsheet(job)

    rows = [j for j in worker.store.list() if j["kind"] in ("rig", "charsheet")]
    assert [j["kind"] for j in sorted(rows, key=lambda j: j["created_at"])] == [
        "rig",
        "charsheet",
    ]
    sheet = next(j for j in rows if j["kind"] == "charsheet")
    assert sheet["params"]["source_job"] == job_id
    assert sheet["params"]["logical_size"] == 32
    assert sheet["params"]["sheet_id"]


async def test_the_rig_follow_up_asks_for_measured_joints(worker):
    job_id = _model_job(worker, rig_joints="measured")
    await worker._maybe_queue_rig(worker.store.get(job_id))
    rig = next(j for j in worker.store.list() if j["kind"] == "rig")
    assert rig["params"]["joints"] == "measured"


async def test_an_ordinary_rig_still_asks_for_nothing(worker):
    """The plumbing is keyed on the request, so an ordinary auto-rig is the
    byte-identical rig it always was."""
    job_id = _model_job(worker)
    del worker.store.get(job_id)["params"]
    await worker._maybe_queue_rig(worker.store.get(job_id))
    rig = next(j for j in worker.store.list() if j["kind"] == "rig")
    assert "joints" not in rig["params"]


async def test_the_reference_stage_queues_nothing(worker):
    """The block rides on the reference and is honoured on the *promoted* row:
    the reference has no mesh for a sheet to depict."""
    job_id = worker.store.create(
        "text", "a ranger", {"troupe": {"logical_size": 32}}, stage="reference"
    )
    await worker._maybe_queue_charsheet(worker.store.get(job_id))
    assert not [j for j in worker.store.list() if j["kind"] == "charsheet"]


# -- the direct door ---------------------------------------------------------


def _rigged_mesh(svc, template="humanoid"):
    import json

    job_id = svc.store.create("image", "a ranger", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.json").write_text(json.dumps({"template": template}), "utf-8")
    svc.store.set_status(job_id, "done")
    return job_id


def test_the_direct_door_queues_a_sheet_for_a_rigged_mesh(svc):
    job_id = _rigged_mesh(svc)
    made = svc_troupe.create_charsheet(svc, job_id, logical_size=64)
    row = svc.store.get(made["id"])
    assert row["kind"] == "charsheet"
    assert row["params"]["logical_size"] == 64
    assert row["params"]["source_job"] == job_id


def test_an_unrigged_mesh_is_refused_by_the_missing_step(svc):
    """Named as the step the user has to take, not as a layout failure."""
    job_id = _rigged_mesh(svc)
    (svc.job_dir(job_id) / "rig.glb").unlink()
    with pytest.raises(Invalid) as excinfo:
        svc_troupe.create_charsheet(svc, job_id)
    assert "rig" in str(excinfo.value)


def test_a_mesh_rigged_as_something_else_is_refused(svc):
    """The clips are a humanoid's. A walk cycle means nothing to a fish, and
    ``clip_library`` answers "no clips" rather than failing -- so without this
    the refusal would land in the worker as a frame-count mismatch."""
    job_id = _rigged_mesh(svc, template="fish")
    with pytest.raises(Invalid) as excinfo:
        svc_troupe.create_charsheet(svc, job_id)
    assert "fish" in str(excinfo.value)


# -- cleanup -----------------------------------------------------------------


def test_a_cancelled_sheet_takes_its_own_render_and_nothing_else(worker):
    """The staged render is the longest-lived leftover in the job -- the gap
    between the pack and the publish is the pixel-art pass -- and it is named
    off the sheet id this row minted, so a cancel cannot take an earlier
    sheet of the same character with it."""
    from warlock import rigging

    source = worker.store.create("image", "a ranger", {}, stage="model")
    job_dir = worker.config.job_dir(source)
    mine, theirs = rigging.new_id(), rigging.new_id()
    for sheet_id in (mine, theirs):
        png = rigging.sheet_png_path(job_dir, sheet_id)
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(b"atlas")
        rigging.sheet_path(job_dir, sheet_id).write_text("{}", "utf-8")
    staged = rigging.sheet_png_path(job_dir, mine)
    render = staged.with_name(f".{staged.name}.render")
    render.write_bytes(b"unquantised")

    worker._discard_artifacts(
        {
            "id": "whatever",
            "kind": "charsheet",
            "params": {"source_job": source, "sheet_id": mine},
        }
    )
    assert not render.exists()
    assert not rigging.sheet_png_path(job_dir, mine).exists()
    assert rigging.sheet_png_path(job_dir, theirs).exists()


# -- the render --------------------------------------------------------------


def _fake_render(monkeypatch):
    """A Blender that writes one flat frame per cell, at the size it was asked
    for. The size is the point: this path renders at ``RENDER_SIZE`` and packs
    at the logical size, which is the one thing about it with no precedent."""
    from pathlib import Path

    from PIL import Image

    from warlock import rigging

    calls: list[dict] = []

    def fake(spec, **kwargs):
        calls.append({"spec": spec, **kwargs})
        frames_dir = Path(spec["frames_dir"])
        for cell in spec["cells"]:
            shade = 40 + (cell["index"] % 4) * 50
            frame = Image.new("RGBA", (spec["frame_size"],) * 2, (0, 0, 0, 0))
            # A blob in the middle, so the matte is not empty and the outline
            # has an edge to find.
            inset = spec["frame_size"] // 4
            frame.paste(
                (shade, shade // 2, 200, 255),
                (inset, inset, spec["frame_size"] - inset, spec["frame_size"] - inset),
            )
            frame.save(frames_dir / f"{cell['index']:04d}.png")
        return {"ok": True, "pivot": [0.5, 0.9]}

    monkeypatch.setattr(rigging, "run_worker", fake)
    return calls


async def test_the_sheet_renders_big_and_packs_small(worker, monkeypatch):
    """The forced shape of this whole stage. 256 cells at the 512px the program
    supersamples from would pack to 4096x16384, which ``check_atlas_size``
    refuses at 8192 -- so the reduction is the only route to an atlas at all,
    not an optimisation applied to one."""
    import json

    from PIL import Image

    from warlock import rigging

    calls = _fake_render(monkeypatch)
    source = worker.store.create("image", "a ranger", {}, stage="model")
    source_dir = worker.config.job_dir(source)
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "model.glb").write_bytes(b"fake-glb")
    (source_dir / "rig.glb").write_bytes(b"fake-rig")
    (source_dir / "rig.json").write_text(json.dumps({"template": "humanoid"}), "utf-8")
    worker.store.set_status(source, "done")

    sheet_id = rigging.new_id()
    job_id = worker.store.create(
        "charsheet",
        "a ranger",
        {
            "source_job": source,
            "sheet_id": sheet_id,
            "template": "humanoid",
            "logical_size": 32,
            "colors": 16,
            "outline": "outer",
            "reduce_mode": "box",
            "palette": "",
            "lighting": "flat",
        },
    )
    worker.start()
    try:
        await _wait_until(
            lambda: worker.store.get(job_id)["status"] in ("done", "error"), 60.0
        )
    finally:
        await worker.shutdown()

    assert worker.store.get(job_id)["error"] is None
    assert calls[0]["spec"]["frame_size"] == charsheet.RENDER_SIZE
    assert len(calls[0]["spec"]["cells"]) == len(charsheet.frame_table())

    png = rigging.sheet_png_path(source_dir, sheet_id)
    with Image.open(png) as atlas:
        assert atlas.size == (charsheet.COLUMNS * 32, 32 * (256 // charsheet.COLUMNS))
    # The un-quantised render is not left beside the sheet it produced.
    assert not png.with_name(f".{png.name}.render").exists()


async def test_the_sidecar_carries_the_engine_side_animation(worker, monkeypatch):
    """The gap this closes: a *rendered* sheet used to reach an engine as frame
    indices with no fps and no loop tags -- and the fps was the one thing the
    renderer knew and the importer could not guess."""
    import json

    from warlock import rigging

    _fake_render(monkeypatch)
    source = worker.store.create("image", "a ranger", {}, stage="model")
    source_dir = worker.config.job_dir(source)
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "model.glb").write_bytes(b"fake-glb")
    (source_dir / "rig.glb").write_bytes(b"fake-rig")
    (source_dir / "rig.json").write_text(json.dumps({"template": "humanoid"}), "utf-8")
    worker.store.set_status(source, "done")

    sheet_id = rigging.new_id()
    job_id = worker.store.create(
        "charsheet",
        "a ranger",
        {"source_job": source, "sheet_id": sheet_id, "logical_size": 16, "colors": 8},
    )
    worker.start()
    try:
        await _wait_until(
            lambda: worker.store.get(job_id)["status"] in ("done", "error"), 60.0
        )
    finally:
        await worker.shutdown()

    assert worker.store.get(job_id)["error"] is None
    meta = rigging.read_sheet(source_dir, sheet_id)
    tags = {t["name"]: t for t in meta["animation"]["tags"]}
    assert len(tags) == len(charsheet.ANIMATIONS) * len(charsheet.DIRECTIONS)
    assert tags["walk_front"]["loop"] is True
    # A play-once tag is spelled the way Inker's exporter spells it, so
    # ``version: 1`` cannot come to mean two subtly different documents.
    assert tags["attack_front"]["repeat"] == 1
    assert "repeat" not in tags["walk_front"]
    report = worker.store.get(job_id)["params"]["pixel_report"]
    assert report["palette"] == "derived"
    assert report["colors"] <= 8


async def test_an_unrigged_source_fails_the_sheet_rather_than_rendering_it(
    worker, monkeypatch
):
    """256 copies of one T-pose is the alternative."""
    from warlock import rigging

    _fake_render(monkeypatch)
    source = worker.store.create("image", "a ranger", {}, stage="model")
    source_dir = worker.config.job_dir(source)
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "model.glb").write_bytes(b"fake-glb")
    worker.store.set_status(source, "done")

    job_id = worker.store.create(
        "charsheet",
        "a ranger",
        {"source_job": source, "sheet_id": rigging.new_id(), "logical_size": 32},
    )
    worker.start()
    try:
        await _wait_until(
            lambda: worker.store.get(job_id)["status"] in ("done", "error"), 30.0
        )
    finally:
        await worker.shutdown()
    assert worker.store.get(job_id)["status"] == "error"
