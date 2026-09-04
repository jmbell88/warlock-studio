"""``derive_music_job``: the second door, and what it refuses.

Its own file rather than more of ``test_jobs_music.py`` for the reason the door
is its own function: half of what it validates is measured against a *parent*,
so every test here needs a finished take on disk first, and folding that
fixture into the file about the brief would make every test in it pay for one.

The same rule holds all the same: **every refusal carries a ``field=``**, and
the parametrised table below is what makes that structural rather than a habit.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from warlock.service import _jobs_music as door
from warlock.service.errors import Invalid


@pytest.fixture(autouse=True)
def _admitted(monkeypatch):
    monkeypatch.setattr(door, "check_weights", lambda svc, kind, params: None)
    monkeypatch.setattr(door, "check_vram", lambda svc, kind, stage, params: None)


def _wav(seconds: float, rate: int = 44100) -> bytes:
    """A silent 16-bit stereo take -- the format ``WARLOCK 5/5`` writes."""
    frames = np.zeros((int(seconds * rate), 2), dtype="<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.tobytes())
    return out.getvalue()


@pytest.fixture
def parent(svc):
    """A finished 60 s take with a track on disk. -> its job id."""
    made = door.create_music_job(svc, prompt="dark ambient, dungeon", duration=60.0)
    job_id = made["id"]
    (svc.config.job_dir(job_id) / "track.wav").write_bytes(_wav(60.0))
    svc.store.set_status(job_id, "done")
    return job_id


def _derive(svc, parent, **kw):
    kw.setdefault("task", "retake")
    return door.derive_music_job(svc, parent, **kw)


# --- what it produces --------------------------------------------------------


def test_a_derivation_is_a_row_with_a_parent_and_a_copied_source(svc, parent):
    out = _derive(svc, parent, task="repaint", repaint_start=10.0, repaint_end=20.0)
    row = svc.store.get(out["id"])
    assert row["kind"] == "music"
    assert row["stage"] == "music"
    # A column, never a params key: a params key would be inherited by every
    # later derivation and the lineage would flatten.
    assert row["parent_id"] == parent
    assert "parent_id" not in row["params"]
    # Copied, not pointed at: the queue is serial, and a parent trashed while
    # this job waits must not strand it.
    assert (svc.config.job_dir(out["id"]) / "source.wav").exists()


def test_the_parents_noise_draw_is_inherited_and_the_variation_is_fresh(svc, parent):
    """The distinction the whole task family turns on.

    ``seed`` is the take's own draw; a child that re-ran with a new one would
    be a different piece of music that happened to be filed under a parent.
    ``retake_seed`` is the variation's, and it is what this door walks.
    """
    was = svc.store.get(parent)["params"]["seed"]
    out = _derive(svc, parent, task="retake", count=3)
    seeds = {svc.store.get(i)["params"]["retake_seed"] for i in out["ids"]}
    assert len(seeds) == 3
    for job_id in out["ids"]:
        assert svc.store.get(job_id)["params"]["seed"] == was


def test_the_task_block_survives_a_reroll_but_not_a_second_derivation(svc, parent):
    """``TASK_PARAMS`` is deliberately outside ``DERIVED_PARAMS``.

    The block is *the request normalised*, so "run that again" must keep it --
    but a repaint *of* an extend must not inherit a pair of pads it has nothing
    to do with, which is what stripping it at this door alone achieves.
    """
    first = _derive(svc, parent, task="extend", extend_right=20.0)["id"]
    (svc.config.job_dir(first) / "track.wav").write_bytes(_wav(80.0))
    svc.store.set_status(first, "done")

    second = _derive(svc, first, task="repaint", repaint_start=5.0, repaint_end=15.0)
    params = svc.store.get(second["id"])["params"]
    assert params["task"] == "repaint"
    assert "extend_right" not in params
    assert "extend_left" not in params


def test_an_extend_lengthens_the_take_it_derives_from(svc, parent):
    out = _derive(svc, parent, task="extend", extend_left=5.0, extend_right=10.0)
    params = svc.store.get(out["id"])["params"]
    assert params["duration"] == pytest.approx(75.0)
    assert params["parent_duration"] == pytest.approx(60.0)


def test_a_loop_centres_its_window_and_records_the_roll(svc, parent):
    """A loop is Muse's own name for a repaint across a rolled joint.

    The window is derived rather than asked for: the user chooses how much of
    the joint to rewrite, and the door puts it in the middle, where the model
    can see the music on both sides of it.
    """
    out = _derive(svc, parent, task="loop", repaint_start=0.0, repaint_end=8.0)
    params = svc.store.get(out["id"])["params"]
    assert params["roll"] == pytest.approx(30.0)
    assert params["repaint_start"] == pytest.approx(26.0)
    assert params["repaint_end"] == pytest.approx(34.0)


def test_an_edit_takes_the_new_words_as_the_rows_prompt(svc, parent):
    out = _derive(svc, parent, task="edit", edit_prompt="bright strings, major")
    row = svc.store.get(out["id"])
    assert row["prompt"] == "bright strings, major"
    assert row["params"]["edit_prompt"] == "bright strings, major"


# --- what it refuses ---------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,field",
    [
        ({"task": "remix"}, "task"),
        ({"task": "retake", "count": 0}, "count"),
        ({"task": "retake", "count": door.MAX_COUNT + 1}, "count"),
        ({"task": "retake", "count": 1.5}, "count"),
        ({"task": "retake", "seed": -1}, "seed"),
        ({"task": "retake", "retake_variance": 1.5}, "retake_variance"),
        ({"task": "retake", "retake_variance": -0.1}, "retake_variance"),
        # An extension longer than the parent is silently zero-filled by the
        # sampler's shape patch -- silence, not music. A refusal, by name.
        ({"task": "extend", "extend_right": 90.0}, "extend_right"),
        ({"task": "extend", "extend_left": 90.0}, "extend_left"),
        ({"task": "extend", "extend_right": -1.0}, "extend_right"),
        ({"task": "extend"}, "extend_right"),
        ({"task": "extend", "extend_right": 0.1}, "extend_right"),
        ({"task": "repaint", "repaint_start": -1.0, "repaint_end": 10.0}, "repaint_start"),
        ({"task": "repaint", "repaint_start": 0.0, "repaint_end": 999.0}, "repaint_start"),
        ({"task": "repaint", "repaint_start": 5.0, "repaint_end": 5.2}, "repaint_end"),
        # Repainting the whole take is what a retake is, and the sampler agrees.
        ({"task": "repaint", "repaint_start": 0.0, "repaint_end": 60.0}, "repaint_start"),
        ({"task": "loop", "repaint_start": 0.0, "repaint_end": 0.2}, "repaint_end"),
        ({"task": "loop", "repaint_start": 0.0, "repaint_end": 45.0}, "repaint_end"),
        ({"task": "edit", "edit_prompt": "   "}, "edit_prompt"),
        # An edit that changes nothing is a retake wearing the wrong name.
        ({"task": "edit", "edit_prompt": "dark ambient, dungeon"}, "edit_prompt"),
        ({"task": "edit", "edit_lyrics": "x" * (door.MAX_LYRICS + 1)}, "edit_lyrics"),
        ({"task": "edit", "edit_prompt": "x", "edit_n_min": 0.9, "edit_n_max": 0.1}, "edit_n_max"),
        ({"task": "audio2audio", "ref_audio_strength": 1.5}, "ref_audio_strength"),
        # At this strength the sampler takes zero steps and hands back its own
        # input. Both controls are named, because either one fixes it.
        ({"task": "audio2audio", "ref_audio_strength": 1.0}, "ref_audio_strength"),
    ],
)
def test_every_refusal_names_the_control_it_is_about(svc, parent, kwargs, field):
    with pytest.raises(Invalid) as caught:
        door.derive_music_job(svc, parent, **kwargs)
    assert caught.value.field == field


def test_a_take_with_no_audio_on_disk_is_refused_by_the_parent(svc, parent):
    (svc.config.job_dir(parent) / "track.wav").unlink()
    with pytest.raises(Invalid) as caught:
        _derive(svc, parent)
    assert caught.value.field == "parent_id"


def test_an_unfinished_take_cannot_be_derived_from(svc):
    made = door.create_music_job(svc, prompt="dark ambient", duration=60.0)
    with pytest.raises(Invalid) as caught:
        _derive(svc, made["id"])
    assert caught.value.field == "parent_id"


def test_only_a_track_can_be_derived_from(svc):
    job_id = svc.store.create("image", "a goblin", {}, stage="model")
    svc.store.set_status(job_id, "done")
    with pytest.raises(Invalid) as caught:
        _derive(svc, job_id)
    assert caught.value.field == "parent_id"


def test_a_refused_derivation_leaves_nothing_on_disk(svc, parent):
    before = set(svc.config.job_dir("").iterdir())
    with pytest.raises(Invalid):
        _derive(svc, parent, task="extend", extend_right=900.0)
    assert set(svc.config.job_dir("").iterdir()) == before
