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
import math

import pytest

from warlock import clips, rigging
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


def test_every_expanded_frame_has_its_own_identity():
    """``(id, frame)`` is a key, and a key has to be unique across the library.

    ``_q_troupe._charsheet`` flattens all five animations into one dict keyed
    exactly this way, so two clips sharing an id do not collide loudly -- the
    later one silently wins every frame of the earlier one.

    That is what happened. ``_expand`` derived the id by joining the keys'
    ``id`` fields, and a clip library's key poses are built with ``name`` and
    ``bones`` and no ``id``, so every id collapsed to ``":" * (len(keys) - 1)``
    -- and ``walk`` and ``run`` both have four keys. Run overwrote walk, so all
    64 walk cells of every character sheet rendered the run cycle. The unit
    tests could not see it because their key fixtures fabricate an ``id``; only
    the real library shows it, which is why this assertion lives here.
    """
    records = clips.expand_clips("humanoid")
    keys = [(r["id"], r["frame"]) for rows in records.values() for r in rows]
    assert len(set(keys)) == len(keys)
    # And the id names the animation it came from, rather than being an opaque
    # hash of it: the sidecar records it, so a wrong one is a wrong label on a
    # block a user reads.
    assert {name: rows[0]["id"] for name, rows in records.items()} == {
        name: name for name in records
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


def test_the_rig_is_not_optional(svc):
    """Every cell of a character sheet is a posed frame, so an unrigged mesh
    would render 256 copies of one reference pose."""
    made = svc_jobs.create_job(
        svc, kind="text", prompt="a ranger", output="reference", troupe={}
    )
    params = svc.store.get(made["id"])["params"]
    assert params["rig"] is True
    assert params["rig_template"] == svc_troupe.TROUPE_TEMPLATE


def test_the_reference_pose_decides_how_the_joints_are_found(svc):
    """The shipped humanoid template is an A-pose, and that is the whole rule.

    Against a T-posed mesh it mis-fits badly enough to skin the arms to the
    chest, which is why those are measured off the mesh's own vertices. An
    A-posed mesh is the pose the template is already in, so it is fitted
    directly -- and measuring needs the ViTPose weights, which a bare install
    does not have, so the A-pose is also the path that works without them.
    """
    for pose, joints in (("tpose", "measured"), ("apose", "template")):
        made = svc_jobs.create_job(
            svc,
            kind="text",
            prompt="a ranger",
            output="reference",
            troupe={"pose": pose},
        )
        params = svc.store.get(made["id"])["params"]
        assert params["guide_pose"] == pose
        assert params["rig_joints"] == joints, pose


def test_a_new_character_is_a_posed_by_default(svc):
    """Chosen on request 2026-08-23, over the T-pose that shipped before it."""
    assert svc_troupe.DEFAULT_TROUPE_POSE == "apose"
    made = svc_jobs.create_job(
        svc, kind="text", prompt="a ranger", output="reference", troupe={}
    )
    params = svc.store.get(made["id"])["params"]
    assert params["guide_pose"] == "apose"
    assert params["rig_joints"] == "template"


def test_an_unknown_pose_is_refused_at_the_door(svc):
    from warlock.service.errors import Invalid

    with pytest.raises(Invalid) as caught:
        svc_jobs.create_job(
            svc,
            kind="text",
            prompt="a ranger",
            output="reference",
            troupe={"pose": "crouch"},
        )
    assert caught.value.field == "pose"


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


async def test_a_character_sheet_enqueue_failure_is_recorded_on_the_mesh(
    worker, monkeypatch
):
    job_id = _model_job(worker)
    real_create = worker.store.create

    def fail_sheet(kind, *args, **kwargs):
        if kind == "charsheet":
            raise RuntimeError("queue insert failed")
        return real_create(kind, *args, **kwargs)

    monkeypatch.setattr(worker.store, "create", fail_sheet)
    await worker._maybe_queue_charsheet(worker.store.get(job_id))

    job = worker.store.get(job_id)
    assert job["status"] == "queued"
    failure = job["params"]["followup_failures"]["charsheet"]
    assert failure["label"] == "Character sheet"
    assert failure["message"] == "queue insert failed"


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


async def test_a_troupe_reference_records_no_rig_failure(worker):
    """The regression that made a finished character look broken.

    Troupe's door sets ``rig`` on the *reference* row -- it describes the mesh
    the gate will promote it into -- so ``_maybe_queue_rig`` ran on every
    Troupe reference completion, found no ``model.glb`` (there cannot be one
    before the user approves the drawing) and, once follow-up failures became
    durable, wrote a permanent false "the generated mesh artifact is missing"
    onto every character anyone started.
    """
    job_id = worker.store.create(
        "text",
        "a ranger",
        {"troupe": {"logical_size": 32}, "rig": True, "rig_template": "humanoid"},
        stage="reference",
    )
    await worker._maybe_queue_rig(worker.store.get(job_id))

    assert not [j for j in worker.store.list() if j["kind"] == "rig"]
    assert "followup_failures" not in worker.store.get(job_id)["params"]


async def test_a_mesh_that_asked_for_a_rig_and_has_none_still_records_it(worker):
    """The guard must not swallow the case the record exists for."""
    job_id = _model_job(worker)
    worker.config.job_dir(job_id).joinpath("model.glb").unlink()
    await worker._maybe_queue_rig(worker.store.get(job_id))

    failure = worker.store.get(job_id)["params"]["followup_failures"]["rig"]
    assert failure["message"] == "The generated mesh artifact is missing."


def test_a_stale_reference_stage_rig_failure_is_not_shown():
    """The rows written before the guard landed keep the record; the reader
    drops it, because a record about a follow-up this row could never have had
    is a fingerprint of the missing guard rather than evidence."""
    from warlock import followups

    params = {
        "followup_failures": {
            "rig": followups.failure_record("rig", "The generated mesh artifact is missing.")
        }
    }
    assert followups.records(params, "reference") == []
    assert len(followups.records(params, "model")) == 1
    # Unfiltered without a stage: a caller that does not know is better served
    # by the whole list than by a silent empty one.
    assert len(followups.records(params)) == 1


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


def test_the_direct_door_snapshots_a_configurable_layout(svc):
    job_id = _rigged_mesh(svc)
    request = {
        "version": 2,
        "movements": [
            {"key": "idle", "frames": 3, "directions": 1},
            {"key": "walk", "frames": 6, "directions": 4},
        ],
    }
    made = svc_troupe.create_charsheet(svc, job_id, layout=request)
    snapshot = svc.store.get(made["id"])["params"]["layout"]
    assert snapshot["cell_count"] == 27
    assert len(snapshot["runs"]) == 5
    assert snapshot["movements"][1]["directions"][2]["key"] == "back"


async def test_the_approved_layout_survives_the_automatic_follow_up(worker):
    layout = charsheet.resolve_layout(
        {
            "version": 2,
            "movements": [{"key": "run", "frames": 10, "directions": 4}],
        }
    ).as_dict()
    job_id = _model_job(worker, troupe={"layout": layout, "logical_size": 32})
    await worker._maybe_queue_charsheet(worker.store.get(job_id))
    row = next(j for j in worker.store.list() if j["kind"] == "charsheet")
    assert row["params"]["layout"] == layout


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


# -- the library door: send an existing mesh to Troupe ------------------------


def _plain_mesh(svc):
    """A finished mesh with no rig -- the common case the door exists for."""
    job_id = svc.store.create("image", "a ranger", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    svc.store.set_status(job_id, "done")
    return job_id


def test_sending_a_rigged_mesh_is_the_direct_door_and_nothing_else(svc):
    """No second implementation: a rigged mesh delegates verbatim."""
    job_id = _rigged_mesh(svc)
    made = svc_troupe.send_to_troupe(svc, job_id, logical_size=64)
    row = svc.store.get(made["id"])
    assert row["kind"] == "charsheet"
    assert row["params"]["logical_size"] == 64
    assert row["params"]["source_job"] == job_id
    # One row, not two: there is nothing to rig.
    assert [j["kind"] for j in svc.store.list() if j["kind"] == "rig"] == []


def test_sending_an_unrigged_mesh_mints_a_rig_that_carries_the_sheet(svc):
    """One press, one row -- and the second is minted by the worker later.

    That is what keeps the worker four ordinary jobs rather than an
    orchestrator, and it is why both rows cancel independently.
    """
    job_id = _plain_mesh(svc)
    made = svc_troupe.send_to_troupe(svc, job_id, logical_size=64, name="ranger")
    row = svc.store.get(made["id"])
    assert row["kind"] == "rig"
    assert made["rigged"] is False
    # Pinned humanoid rather than the user's Rig-stage preference: the sheet is
    # animated from the humanoid clip library.
    assert row["params"]["template"] == svc_troupe.TROUPE_TEMPLATE
    # Measured, for the reason _jobs_create gives where it sets the same flag.
    assert row["params"]["joints"] == "measured"
    block = row["params"]["troupe_sheet"]
    assert block["logical_size"] == 64 and block["name"] == "ranger"
    # A snapshot, not a promise to re-resolve: the row the worker mints later
    # must describe what was on screen when the button was pressed.
    assert block["layout"]["cell_count"] > 0
    # The mesh row is untouched. Stamping it would work as a marker and would
    # change what a reroll means -- rerun/promote copy everything not derived,
    # so the next Remesh would silently spend a rig and 256 rendered cells.
    assert "troupe_sheet" not in svc.store.get(job_id)["params"]
    assert "troupe" not in svc.store.get(job_id)["params"]


def test_the_marker_is_its_own_key_and_is_nested(svc):
    """``troupe`` and ``troupe_sheet`` are two claims, not one spelled twice.

    ``troupe`` on a reference means "run the whole chain, human gate
    included"; this means "render this sheet once the rig lands". Nested, so
    ``VECTOR_PARAMS`` -- an allowlist of flat settings -- cannot pick it up.
    """
    from warlock import vectors

    job_id = _plain_mesh(svc)
    made = svc_troupe.send_to_troupe(svc, job_id)
    params = svc.store.get(made["id"])["params"]
    assert "troupe" not in params
    assert isinstance(params["troupe_sheet"], dict)
    assert "troupe_sheet" not in vectors.VECTOR_PARAMS


def test_an_unrenderable_request_costs_the_request_and_not_a_rig(svc):
    """Everything knowable is refused here, one link earlier than before.

    ``create_charsheet``'s own argument: an unrenderable request should cost
    the request, not a place in the queue plus a rig plus 256 EEVEE frames.
    """
    job_id = _plain_mesh(svc)
    with pytest.raises(Invalid):
        svc_troupe.send_to_troupe(svc, job_id, logical_size=7)
    with pytest.raises(Invalid):
        svc_troupe.send_to_troupe(svc, job_id, name="x" * 500)
    assert [j for j in svc.store.list() if j["kind"] == "rig"] == []


def test_an_unfinished_mesh_is_refused_in_the_direct_doors_words(svc):
    job_id = _plain_mesh(svc)
    (svc.job_dir(job_id) / "model.glb").unlink()
    with pytest.raises(Invalid) as excinfo:
        svc_troupe.send_to_troupe(svc, job_id)
    assert "no finished mesh" in str(excinfo.value)


def test_a_mesh_already_rigged_as_something_else_is_still_refused(svc):
    """The template lives in rig.json, which only the service reads -- so the
    predicate the pane draws with cannot know it, and this is where the answer
    comes from. Instant, and before anything is minted."""
    job_id = _rigged_mesh(svc, template="fish")
    with pytest.raises(Invalid) as excinfo:
        svc_troupe.send_to_troupe(svc, job_id)
    assert "fish" in str(excinfo.value)


async def test_the_worker_mints_the_sheet_when_the_rig_lands(worker):
    """The second half, and it re-checks the rig rather than assuming it."""
    source = worker.store.create("image", "a ranger", {}, stage="model")
    source_dir = worker.config.job_dir(source)
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "model.glb").write_bytes(b"fake-glb")
    block = {"logical_size": 32, "colors": 16, "layout": {"version": 1}}
    rig_id = worker.store.create(
        "rig", "a ranger", {"source_job": source, "troupe_sheet": block}
    )

    # The user cancelled the rig: no artifact, so no sheet, and the failure is
    # recorded against the mesh the user is looking at.
    await worker._maybe_queue_sheet_after_rig(worker.store.get(rig_id))
    assert [j for j in worker.store.list() if j["kind"] == "charsheet"] == []

    (source_dir / "rig.glb").write_bytes(b"fake-glb")
    await worker._maybe_queue_sheet_after_rig(worker.store.get(rig_id))
    row = next(j for j in worker.store.list() if j["kind"] == "charsheet")
    assert row["params"]["source_job"] == source
    assert row["params"]["logical_size"] == 32
    # Its own fresh id, so a cancel deletes this sheet's atlas and no earlier
    # one of the same character.
    assert row["params"]["sheet_id"]


async def test_the_two_follow_up_paths_cannot_fire_for_each_other(worker):
    """Two markers, two guards. A rig row carrying ``troupe_sheet`` must not
    reach ``_maybe_queue_charsheet``, and a mesh carrying ``troupe`` must not
    reach this one."""
    job_id = _model_job(worker)
    await worker._maybe_queue_sheet_after_rig(worker.store.get(job_id))
    assert [j for j in worker.store.list() if j["kind"] == "charsheet"] == []

    rig_id = worker.store.create(
        "rig", "a ranger", {"source_job": job_id, "troupe_sheet": {"logical_size": 32}}
    )
    await worker._maybe_queue_charsheet(worker.store.get(rig_id))
    assert [j for j in worker.store.list() if j["kind"] == "charsheet"] == []

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
        {
            "source_job": source,
            "sheet_id": sheet_id,
            "logical_size": 16,
            "colors": 8,
            "layout": {
                "version": 2,
                "movements": [
                    {"key": "idle", "frames": 3, "directions": 1},
                    {"key": "attack", "frames": 5, "directions": 4},
                ],
            },
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
    meta = rigging.read_sheet(source_dir, sheet_id)
    tags = {t["name"]: t for t in meta["animation"]["tags"]}
    assert len(tags) == 5
    assert tags["idle_front"]["loop"] is True
    # A play-once tag is spelled the way Inker's exporter spells it, so
    # ``version: 1`` cannot come to mean two subtly different documents.
    assert tags["attack_back"]["repeat"] == 1
    assert "repeat" not in tags["idle_front"]
    assert meta["troupe"]["cell_count"] == 23
    assert worker.store.get(job_id)["params"]["layout"] == meta["troupe"]
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


# -- the A-pose --------------------------------------------------------------


def test_every_reference_pose_loads_and_draws():
    for pose in spritesynth.REFERENCE_POSES:
        for variant in svc_troupe.TROUPE_VARIANTS:
            image = spritesynth.render_reference_guide(variant, pose)
            assert image.size == (spritesynth.ATLAS_PX, spritesynth.ATLAS_PX)


def test_an_unknown_pose_raises_rather_than_defaulting():
    """``load_tpose_guide``'s rule, for the other axis: conditioning on the
    pose the caller did not ask for is a wrong character under their name."""
    with pytest.raises(ValueError):
        spritesynth.load_reference_guide("male", "crouch")


def test_the_a_pose_is_the_t_pose_with_the_arms_rotated_down():
    """Two poses of one figure, not two figures.

    Everything that is not an arm has to match to the last decimal, or a
    character drawn from one guide and a character drawn from the other
    reconstruct to different proportions and the choice stops being a pose.
    """
    for variant in svc_troupe.TROUPE_VARIANTS:
        tpose = spritesynth.load_reference_guide(variant, "tpose").poses[0].points
        apose = spritesynth.load_reference_guide(variant, "apose").poses[0].points
        assert set(tpose) == set(apose)
        for joint, point in tpose.items():
            if joint.startswith(("elbow", "hand")):
                continue
            assert apose[joint] == point, joint
        for side in ("L", "R"):
            shoulder = apose[f"shoulder.{side}"]
            for joint in (f"elbow.{side}", f"hand.{side}"):
                assert apose[joint][1] > tpose[joint][1], f"{joint} did not come down"
            # Lengths preserved: it is a rotation, not a redrawing.
            def _span(points, a, b):
                return math.hypot(points[a][0] - points[b][0], points[a][1] - points[b][1])

            assert _span(apose, f"shoulder.{side}", f"elbow.{side}") == pytest.approx(
                _span(tpose, f"shoulder.{side}", f"elbow.{side}"), abs=1e-3
            )
            assert _span(apose, f"elbow.{side}", f"hand.{side}") == pytest.approx(
                _span(tpose, f"elbow.{side}", f"hand.{side}"), abs=1e-3
            )
            assert shoulder == tpose[f"shoulder.{side}"]


def test_the_a_pose_guide_is_line_art_too():
    from warlock.pipelines import control

    fraction = control.edge_fraction(spritesynth.render_reference_guide("male", "apose"))
    assert 0.0005 < fraction < 0.05


def test_the_worker_draws_the_pose_the_row_asked_for():
    """The door records ``guide_pose``; the guide is drawn in the worker.

    Source-scanned because the branch lives inside ``_conditioning``, which
    needs a queue and a job row to reach -- and the two facts worth pinning are
    both textual: that the pose reaches the renderer at all, and what it falls
    back to when it is absent.
    """
    from pathlib import Path

    import warlock._q_generate as q_generate

    source = Path(q_generate.__file__).read_text(encoding="utf-8")
    assert 'params.get("guide_pose")' in source
    assert "render_reference_guide(variant, pose)" in source


def test_a_row_written_before_the_pose_existed_rerolls_as_a_t_pose():
    """The back-compat rule, and it is not the door's default.

    A reference queued before the pose was a choice carries no ``guide_pose``
    and was conditioned on the T-pose. Falling back to today's default would
    redraw it against a figure it never saw, which is a different character out
    of the same row.
    """
    from pathlib import Path

    import warlock._q_generate as q_generate

    source = Path(q_generate.__file__).read_text(encoding="utf-8")
    assert 'str(params.get("guide_pose") or "tpose")' in source
    assert svc_troupe.DEFAULT_TROUPE_POSE != "tpose", (
        "this test is only meaningful while the door's default differs"
    )


def test_the_sheet_cap_counts_every_door_that_reserves_a_slot(svc, monkeypatch):
    """Three doors mint rows that end as sheets of one job -- ``create_sheet``,
    ``create_charsheet`` and the rig row ``send_to_troupe`` mints for an
    unrigged mesh -- and they share one ``MAX_SHEETS`` pool. Each used to count
    a different subset of the others' queued rows, so the pool could be
    reserved one past the cap through whichever door was not counting."""
    from warlock.service import sheets as svc_sheets
    from warlock.service.errors import Conflict

    monkeypatch.setattr(rigging, "MAX_SHEETS", 1)
    plain = _plain_mesh(svc)
    svc_troupe.send_to_troupe(svc, plain, logical_size=64)
    with pytest.raises(Conflict, match="at most 1 sheet"):
        svc_troupe.send_to_troupe(svc, plain, logical_size=64)

    rigged = _rigged_mesh(svc)
    svc_troupe.create_charsheet(svc, rigged, logical_size=64)
    with pytest.raises(Conflict, match="at most 1 sheet"):
        svc_sheets.create_sheet(svc, rigged, frame_size=64)


# -- the staged render is not left behind -------------------------------------


async def test_a_failed_character_sheet_leaves_no_orphaned_render(worker):
    """The un-quantised atlas is cleaned up when the sheet fails, not only
    when it succeeds.

    It is written to ``pack_target`` inside the **mesh** job's directory, not
    the sheet job's, so an orphan here survives deleting the failed sheet and
    is only reclaimed by deleting the source asset. ``_discard_artifacts`` does
    not cover it: the queue calls that on a cancel, not on an ordinary error.

    Driven through a real failure rather than a patched one -- an unreadable
    ``rig.glb`` is what the Blender worker actually refuses -- so the test
    cannot pass because a fake returned early.
    """
    source_id = "a" * 12
    sheet_id = "b" * 12
    source_dir = worker.config.job_dir(source_id)
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "rig.glb").write_bytes(b"not a glb")

    png = rigging.sheet_png_path(source_dir, sheet_id)
    atlas = png.with_name(f".{png.name}.render")
    atlas.parent.mkdir(parents=True, exist_ok=True)
    atlas.write_bytes(b"a partially written render")

    # Named, not blind: an unreadable ``rig.glb`` is refused by the Blender
    # worker, and asserting *which* failure keeps this test honest if the
    # fixture ever stops reaching the render at all.
    with pytest.raises(rigging.BlenderError):
        await worker._charsheet(
            {
                "id": "c" * 12,
                "params": {
                    "source_job": source_id,
                    "sheet_id": sheet_id,
                    "logical_size": 32,
                },
            }
        )

    assert not atlas.exists(), "the staged render outlived the job that made it"
