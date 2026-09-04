"""``create_music_job``: what it refuses, and what it leaves on disk.

The door is a sibling of ``create_job`` rather than a third kind threaded
through its gate, so its invariants have to be asserted for themselves rather
than inherited: **validate first, directory before row, and rmtree if the row
write raises.**

Every refusal carries a ``field=``, because that is what puts the ring on the
control the user has to change -- a refusal with no field arrives as a toast
with no subject.
"""

from __future__ import annotations

import pytest

from warlock.service import _jobs_music as door
from warlock.service.errors import Invalid


@pytest.fixture(autouse=True)
def _admitted(monkeypatch):
    """Weights present and the card roomy, unless a test says otherwise.

    Both are separately tested below; pinning them here keeps every other test
    in this file about the door's own rules rather than about the machine it
    happens to run on.
    """
    monkeypatch.setattr(door, "check_weights", lambda svc, kind, params: None)
    monkeypatch.setattr(door, "check_vram", lambda svc, kind, stage, params: None)


def _brief(**kw):
    base = {"prompt": "dark ambient, dungeon"}
    base.update(kw)
    return base


def _make(svc, **kw):
    return door.create_music_job(svc, **_brief(**kw))


# --- what it produces --------------------------------------------------------


def test_a_press_makes_one_row_and_one_directory_per_take(svc):
    out = _make(svc, count=3)
    assert len(out["ids"]) == 3
    assert out["id"] == out["ids"][0]
    for job_id in out["ids"]:
        assert svc.config.job_dir(job_id).is_dir()
        row = svc.store.get(job_id)
        assert row["kind"] == "music"
        assert row["stage"] == "music"
        assert row["prompt"] == "dark ambient, dungeon"


def test_the_stage_is_its_own_rather_than_model(svc):
    """``model`` is overloaded with mesh-verdict semantics in db.py's queries.

    A music row wearing it would be graded on a scale that has no meaning for
    audio, and would join the findings corpus as evidence about a mesh.
    """
    from warlock.service import library

    job_id = _make(svc)["id"]
    assert svc.store.get(job_id)["stage"] == "music"
    assert library.PRIMARY["music"] == "track.wav"


def test_every_take_gets_its_own_seed(svc):
    ids = _make(svc, count=3)["ids"]
    seeds = [svc.store.get(job_id)["params"]["seed"] for job_id in ids]
    assert len(set(seeds)) == 3


def test_a_pinned_seed_applies_to_the_first_take_and_the_rest_walk_from_it(svc):
    """Four takes at one seed must be four *different* pieces.

    Reproducibility is about the take you asked for; four copies of one file is
    not a batch, it is a bug that looks like a batch.
    """
    ids = _make(svc, count=3, seed=100)["ids"]
    seeds = [svc.store.get(job_id)["params"]["seed"] for job_id in ids]
    assert seeds == [100, 101, 102]


def test_the_recipe_is_stored_under_the_models_own_parameter_names(svc):
    job_id = _make(
        svc,
        lyrics="[verse]\nhello",
        duration=120.0,
        infer_step=40,
        guidance_scale=9.0,
        scheduler_type="heun",
        cfg_type="cfg",
        omega_scale=6.0,
    )["id"]
    params = svc.store.get(job_id)["params"]
    assert params["lyrics"] == "[verse]\nhello"
    assert params["duration"] == pytest.approx(120.0)
    assert params["infer_step"] == 40
    assert params["guidance_scale"] == pytest.approx(9.0)
    assert params["scheduler_type"] == "heun"
    assert params["cfg_type"] == "cfg"
    assert params["omega_scale"] == pytest.approx(6.0)
    assert params["music_model"] == "ace_step_v1"


# --- what it refuses ---------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,field",
    [
        ({"prompt": "   "}, "prompt"),
        ({"lyrics": "x" * (door.MAX_LYRICS + 1)}, "lyrics"),
        ({"duration": door.MAX_DURATION + 1}, "duration"),
        ({"duration": door.MIN_DURATION - 1}, "duration"),
        ({"count": 0}, "count"),
        ({"count": door.MAX_COUNT + 1}, "count"),
        ({"count": 1.5}, "count"),
        ({"scheduler_type": "dpm"}, "scheduler_type"),
        ({"cfg_type": "none"}, "cfg_type"),
        ({"infer_step": 0}, "infer_step"),
        ({"infer_step": 500}, "infer_step"),
        ({"guidance_scale": -1.0}, "guidance_scale"),
        ({"omega_scale": 1000.0}, "omega_scale"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"music_model": "ace_step_v2"}, "music_model"),
    ],
)
def test_every_refusal_names_the_control_it_is_about(svc, kwargs, field):
    with pytest.raises(Invalid) as caught:
        _make(svc, **kwargs)
    assert caught.value.field == field


def test_a_refused_brief_leaves_nothing_on_disk(svc):
    before = set(svc.config.job_dir("").iterdir())
    with pytest.raises(Invalid):
        _make(svc, duration=9999.0)
    assert set(svc.config.job_dir("").iterdir()) == before


def test_duration_is_bounded_because_its_cost_is_unbounded():
    """The bound is the reason admission can refuse this kind at all.

    Duration sets the length of what the sampler produces, so it drives both the
    generation time and the figure ``vram.estimate`` has to price -- and an
    unpriceable job is one the door cannot refuse before it OOMs.
    """
    assert door.MIN_DURATION > 0
    assert door.MAX_DURATION <= 600


def test_a_row_that_cannot_be_written_takes_its_directory_with_it(svc, monkeypatch):
    """The invariant ``create_job`` states and this one had to restate.

    A job directory with no row is storage nothing will ever clean.
    """
    before = set(svc.config.job_dir("").iterdir())

    def _boom(*a, **k):
        raise RuntimeError("the store is gone")

    monkeypatch.setattr(svc.store, "create", _boom)
    with pytest.raises(RuntimeError):
        _make(svc, count=2)
    assert set(svc.config.job_dir("").iterdir()) == before


# --- admission ---------------------------------------------------------------


def test_missing_weights_are_refused_at_the_door_with_the_download(svc, monkeypatch):
    """Muse has no fallback and is not supposed to have one.

    Unlike a missing pose or matting model, which degrade, this is a refusal --
    so it has to name the thing that fixes it or the user is simply stuck.
    """
    from warlock import fetch
    from warlock.service import validation

    # ``undo`` first, because the autouse fixture above stubbed out the very
    # check this test is about; the vram stub is then put back, since a machine
    # with no card must not fail this test for the wrong reason.
    monkeypatch.undo()
    monkeypatch.setattr(door, "check_vram", lambda svc, kind, stage, params: None)
    monkeypatch.setattr(door, "check_weights", validation.check_weights)
    monkeypatch.setattr(fetch, "present", lambda config, kind, spec: False)
    with pytest.raises(Invalid) as caught:
        _make(svc)
    assert caught.value.field == "music_model"
    assert "hf download" in str(caught.value) or "Settings" in str(caught.value)


def test_a_music_job_is_priced_rather_than_admitted_for_free():
    """The branch has to come *before* ``estimate_parts``' catch-all.

    Priced at zero, ``check_vram`` admits a job that OOMs at load, which is the
    exact failure the door exists to prevent.
    """
    from warlock import models, vram

    total, checkpoint = vram.estimate_parts("music", "music", {}, exclusive=True)
    spec = models.MUSIC_MODELS[models.DEFAULT_MUSIC_MODEL]
    assert total == pytest.approx(spec.vram_gib)
    # Returned as the checkpoint term too, so ``queue._check_resources`` credits
    # a resident pipe back rather than charging the same weights twice.
    assert checkpoint == pytest.approx(spec.vram_gib)


def test_coexisting_with_trellis_is_priced_as_such():
    from warlock import vram

    exclusive, _ = vram.estimate_parts("music", "music", {}, exclusive=True)
    coexist, _ = vram.estimate_parts("music", "music", {}, exclusive=False)
    assert coexist > exclusive


# --- what must not be carried forward ----------------------------------------


def test_what_the_worker_records_about_the_output_is_stripped_on_a_rerun():
    """``actual_duration`` is a worker observation, not a request echo.

    A reroll at a different duration inheriting this one's would put a length on
    the row that the file does not have. The request echoes -- duration, lyrics,
    the recipe knobs -- deliberately stay: they are the request normalised, and
    "run that again" means running that.
    """
    from warlock.service.validation import DERIVED_PARAMS

    assert "actual_duration" in DERIVED_PARAMS
    for echo in ("duration", "lyrics", "music_model", "scheduler_type"):
        assert echo not in DERIVED_PARAMS


def test_music_adds_nothing_to_the_vector_params_allowlist():
    """Deliberately nothing.

    Its only consumer is the mesh-verdict findings corpus, and an entry with no
    aggregator reading it is dead weight a future music-findings module would
    have to un-teach itself from.
    """
    from warlock.vectors import VECTOR_PARAMS

    assert not {p for p in VECTOR_PARAMS if "music" in p or "lyric" in p}
