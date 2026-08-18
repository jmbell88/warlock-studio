"""The prompt-driven sprite sheet: one press, two rows.

``create_sprite_synthesis`` needs a finished reference and IP-adapters off its
``input.png`` -- it has no prompt of its own. Create's Sheet output has nothing
*but* a prompt, so the gap is closed the way the rig checkbox already closes the
same one for meshes: the reference job carries a follow-up request, and the
worker queues the second job once the picture it needs exists.

Two claims carry this. The options are validated at the *reference's* door,
because the follow-up never passes through the service door that would
otherwise check them -- so a bad layout or a missing ControlNet discovered when
the follow-up is minted would be a refusal on a row the user never submitted,
an hour after they paid for the one they did. And the two candidate seeds are
*derived* from the character's own seed, because this worker has no RNG and
should not grow one.
"""

from __future__ import annotations

import asyncio

import pytest

from warlock import models
from warlock.config import Config
from warlock.db import JobStore
from warlock.pipelines import spritesynth
from warlock.queue import Worker
from warlock.service import jobs as svc_jobs
from warlock.service.errors import Invalid
from warlock.service.validation import MAX_SEED


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


def _reference(worker, **params) -> str:
    """A finished-looking reference row with a picture on disk."""
    base = {"seed": 11, "reference_seed": 11, "base_model": "sdxl_cfg"}
    base.update(params)
    job_id = worker.store.create("text", "a hooded ranger", base, stage="reference")
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    return job_id


# -- the door ----------------------------------------------------------------


def test_the_block_is_stored_on_the_reference_job(svc):
    made = svc_jobs.create_job(
        svc,
        kind="text",
        prompt="a hooded ranger",
        output="reference",
        sprite_sheet={"sheet_type": "walk", "logical_size": 32, "colors": 16},
    )
    params = svc.store.get(made["id"])["params"]
    assert params["sprite_sheet"] == {
        "sheet_type": "walk",
        "logical_size": 32,
        "colors": 16,
    }


def test_an_empty_block_takes_the_sprite_defaults(svc):
    """The pane always sends all three, but the API may not -- and a follow-up
    minted from a half-filled block would be a sheet whose layout nobody
    chose."""
    from warlock.service import sprites as svc_sprites

    made = svc_jobs.create_job(
        svc, kind="text", prompt="a ranger", output="reference", sprite_sheet={}
    )
    block = svc.store.get(made["id"])["params"]["sprite_sheet"]
    assert block["sheet_type"] == svc_sprites.DEFAULT_SPRITE_SHEET_TYPE
    assert block["logical_size"] == svc_sprites.DEFAULT_SPRITE_LOGICAL_SIZE


def test_a_sheet_cannot_be_asked_for_alongside_a_mesh(svc):
    """The follow-up reads ``input.png`` at the reference stage; a model job's
    is the picture it was reconstructed *from*, and the row is never at that
    stage when it finishes."""
    with pytest.raises(Invalid) as excinfo:
        svc_jobs.create_job(
            svc, kind="text", prompt="a ranger", output="model", sprite_sheet={}
        )
    assert excinfo.value.field == "output"


def test_a_batch_of_characters_is_refused(svc):
    """N characters each spawning two more generations is 3N passes from one
    button."""
    with pytest.raises(Invalid) as excinfo:
        svc_jobs.create_job(
            svc,
            kind="text",
            prompt="a ranger",
            output="reference",
            count=4,
            sprite_sheet={},
        )
    assert excinfo.value.field == "count"


@pytest.mark.parametrize(
    "block,field",
    [
        ({"sheet_type": "cartwheel"}, "sheet_type"),
        ({"logical_size": 96}, "logical_size"),
        ({"colors": 7}, "colors"),
        ({"logical_size": "big"}, "sheet_type"),
    ],
)
def test_a_bad_option_is_refused_at_the_references_door(svc, block, field):
    """Here rather than when the follow-up is minted: the worker builds that row
    itself, so a refusal there would land an hour later on a job the user never
    submitted."""
    with pytest.raises(Invalid) as excinfo:
        svc_jobs.create_job(
            svc, kind="text", prompt="a ranger", output="reference", sprite_sheet=block
        )
    assert excinfo.value.field == field


def test_a_missing_controlnet_is_refused_before_the_character_is_drawn(svc):
    """Both adapters are mandatory for a synthesis, so a host missing one would
    draw the character, queue the sheet and fail it -- which reads as a bug
    rather than as a download the user has not done."""
    from warlock import fetch

    spec = models.CONTROLNETS["canny"]
    (svc.config.t2i_model_root / spec.dir_name / "config.json").unlink()
    assert not fetch.present(svc.config, "control", spec)

    with pytest.raises(Invalid) as excinfo:
        svc_jobs.create_job(
            svc, kind="text", prompt="a ranger", output="reference", sprite_sheet={}
        )
    assert excinfo.value.field == "control"
    assert excinfo.value.rows


def test_the_follow_ups_card_is_admitted_at_the_references_door(svc, monkeypatch):
    """The VRAM half of the same rule: the worker mints the follow-up row
    itself and can refuse nothing, so the sprite sum -- checkpoint plus both
    adapters -- has to be admitted here, exactly as the direct door
    (``sprites.create_sprite_synthesis``) admits it. Without it, a coexist
    card that fits the reference but not the sprite sum drew the character and
    failed the sheet at dispatch, with no remedy in sight."""
    from warlock.service import _jobs_create
    from warlock.service import sprites as svc_sprites

    seen: list[tuple] = []
    monkeypatch.setattr(
        _jobs_create,
        "check_vram",
        lambda svc, kind, stage, params: seen.append(
            (kind, stage, params.get("base_model"))
        ),
    )
    svc_jobs.create_job(
        svc, kind="text", prompt="a ranger", output="reference", sprite_sheet={}
    )
    assert ("sprite_synthesis", "model", svc_sprites.SPRITE_BASE_MODEL) in seen


def test_a_reroll_holds_the_same_door_the_first_submit_did(svc):
    """The other way in, and it needs the check for the reason the tile sheet's
    reroll does: a reroll can be days later, against a models directory the
    user has pruned since -- and the row that would load these weights is minted
    by the *worker*, which cannot refuse anything. Without this the character
    redraws, the sheet queues behind it, and the pair fails at dispatch."""
    from warlock import fetch

    made = svc_jobs.create_job(
        svc, kind="text", prompt="a ranger", output="reference", sprite_sheet={}
    )
    spec = models.CONTROLNETS["canny"]
    (svc.config.t2i_model_root / spec.dir_name / "config.json").unlink()
    assert not fetch.present(svc.config, "control", spec)

    with pytest.raises(Invalid) as excinfo:
        svc_jobs.rerun_job(svc, made["id"], mode="reroll")
    assert excinfo.value.field == "control"


def test_a_reroll_of_a_plain_reference_is_untouched_by_that_door(svc):
    """The guard is keyed on the block, not on the kind: a reference with no
    follow-up request must not start needing a synthesis' weights to reroll."""
    from warlock import fetch

    made = svc_jobs.create_job(svc, kind="text", prompt="a ranger", output="reference")
    spec = models.CONTROLNETS["canny"]
    (svc.config.t2i_model_root / spec.dir_name / "config.json").unlink()
    assert not fetch.present(svc.config, "control", spec)

    assert svc_jobs.rerun_job(svc, made["id"], mode="reroll")["id"]


def test_a_plain_reference_stores_no_block(svc):
    """The key's presence is what the worker branches on."""
    made = svc_jobs.create_job(
        svc, kind="text", prompt="a ranger", output="reference"
    )
    assert "sprite_sheet" not in svc.store.get(made["id"])["params"]


# -- the follow-up -----------------------------------------------------------


async def test_a_finished_character_queues_its_sheet(worker):
    job_id = _reference(
        worker,
        sprite_sheet={"sheet_type": "walk", "logical_size": 32, "colors": 16},
    )
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await _wait_until(
        lambda: any(j["kind"] == "sprite_synthesis" for j in worker.store.list())
    )

    sheet = next(j for j in worker.store.list() if j["kind"] == "sprite_synthesis")
    assert sheet["params"]["source_job"] == job_id
    assert sheet["params"]["sheet_type"] == "walk"
    assert sheet["params"]["logical_size"] == 32
    assert sheet["params"]["colors"] == 16
    assert sheet["params"]["draft_id"]
    await worker.shutdown()


async def test_the_candidate_seeds_are_derived_and_differ(worker):
    """Derived rather than drawn: the whole two-step chain is then reproducible
    from the one seed the user can see and lock. Differing because the
    deliverable is a *pair* to pick between."""
    job_id = _reference(
        worker,
        seed=99,
        reference_seed=99,
        sprite_sheet={"sheet_type": "turnaround", "logical_size": 64, "colors": 32},
    )
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await _wait_until(
        lambda: any(j["kind"] == "sprite_synthesis" for j in worker.store.list())
    )

    params = next(
        j for j in worker.store.list() if j["kind"] == "sprite_synthesis"
    )["params"]
    assert params["seed_a"] == spritesynth.candidate_seed(99, "a")
    assert params["seed_b"] == spritesynth.candidate_seed(99, "b")
    assert params["seed_a"] != params["seed_b"]
    await worker.shutdown()


async def test_the_follow_up_pins_the_sprite_base_rather_than_inheriting_one(worker):
    """A synthesis is a pixel-art restyle, so its base is decided by the LoRA
    it must be able to take -- which is why ``create_sprite_synthesis`` writes
    a constant instead of a choice.

    Carrying the character's own ``base_model`` across looks like keeping its
    style and is the opposite: a character drawn on klein would queue a sheet
    whose base cannot take the pixel-art LoRA, and the worker's tolerance would
    draw both candidates bare and say so only in the log. The door's admission
    would be wrong too -- it checks this constant's weights and prices this
    constant's VRAM.
    """
    from warlock.service import sprites as svc_sprites

    job_id = _reference(
        worker,
        base_model="flux_klein",
        sprite_sheet={"sheet_type": "turnaround", "logical_size": 64, "colors": 32},
    )
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await _wait_until(
        lambda: any(j["kind"] == "sprite_synthesis" for j in worker.store.list())
    )

    sheet = next(j for j in worker.store.list() if j["kind"] == "sprite_synthesis")
    assert sheet["params"]["base_model"] == svc_sprites.SPRITE_BASE_MODEL
    await worker.shutdown()


async def test_no_follow_up_without_the_block(worker):
    job_id = _reference(worker)
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await asyncio.sleep(0.2)

    assert not any(j["kind"] == "sprite_synthesis" for j in worker.store.list())
    await worker.shutdown()


async def test_the_character_survives_a_follow_up_that_cannot_be_queued(worker, monkeypatch):
    """The picture is finished and on disk. A failure to queue the optional
    follow-up must not retroactively fail the job that produced it -- the rig's
    rule, verbatim."""
    real_create = worker.store.create

    def explode(kind, *args, **kwargs):
        if kind == "sprite_synthesis":
            raise RuntimeError("no")
        return real_create(kind, *args, **kwargs)

    job_id = _reference(
        worker,
        sprite_sheet={"sheet_type": "turnaround", "logical_size": 64, "colors": 32},
    )
    monkeypatch.setattr(worker.store, "create", explode)
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await asyncio.sleep(0.2)

    assert worker.store.get(job_id)["status"] == "done"
    assert worker.store.get(job_id)["error"] is None
    await worker.shutdown()


# -- the derivation ----------------------------------------------------------


def test_the_two_candidates_are_unrelated_not_adjacent():
    """One apart in seed space comes back looking like the same picture
    twice."""
    a = spritesynth.candidate_seed(1000, "a")
    b = spritesynth.candidate_seed(1000, "b")
    assert abs(a - b) > 1_000_000


def test_a_derived_seed_is_always_legal():
    for base in (0, 1, 2**31 - 2, 123456789):
        for letter in spritesynth.CANDIDATES:
            assert 0 <= spritesynth.candidate_seed(base, letter) <= MAX_SEED


def test_an_unknown_candidate_letter_raises():
    with pytest.raises(ValueError, match="unknown sprite candidate"):
        spritesynth.candidate_seed(1, "c")


def test_the_pipelines_seed_ceiling_matches_the_service_door():
    """A pipeline may not import the service layer, so the ceiling is restated
    -- and pinned here, so the copy cannot drift."""
    assert spritesynth.MAX_SEED == MAX_SEED
