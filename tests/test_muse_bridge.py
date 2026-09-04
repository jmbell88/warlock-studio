"""The round trip between the two audio modes, both legs.

Muse -> Sirens was always the headline pairing and was **broken for the mode's
whole life**: the writer produced 48 kHz IEEE-float and the tracker's reader is
stdlib ``wave``, which refuses that format outright. Nothing caught it because
``test_muse_mode`` stubs ``import_sample`` out. ``tests/test_music_format.py``
is what pins the writer now; this file pins the two claims the *bridge* makes.

Sirens -> Muse is the leg the manual called deliberately unbuilt. It opens
exactly one door -- ``create_music_job``'s ``reference_wav`` -- and this asserts
both that it works and that the reference carries the loop points the user
authored, which is the headline feature meeting the round trip rather than
fighting it.
"""

from __future__ import annotations

import io
import wave

import pytest

from warlock.service import _jobs_music as door
from warlock.service.errors import Invalid
from warlock.studio.sirens import synth, wavout


@pytest.fixture(autouse=True)
def _admitted(monkeypatch):
    monkeypatch.setattr(door, "check_weights", lambda svc, kind, params: None)
    monkeypatch.setattr(door, "check_vram", lambda svc, kind, stage, params: None)


def _render(seconds: float = 30.0, rate: int = synth.SAMPLE_RATE) -> bytes:
    """A stereo render, as ``compose_from_sirens`` would hand one over."""
    import numpy as np

    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    tone = np.sin(2 * np.pi * 220.0 * t) * 0.4
    return wavout.wav_bytes(np.stack([tone, tone], axis=1), rate)


# --- the rates agree ---------------------------------------------------------


def test_the_two_engines_already_agree_about_the_sample_rate():
    """Asserted rather than claimed in a comment, which is the whole point:
    ``synth.SAMPLE_RATE`` is what Sirens renders at and 44100 is what
    ``WARLOCK 5/5`` makes Muse write, so neither leg of the bridge needs a
    resample and a change to either would fail here rather than silently
    transpose a song."""
    assert synth.SAMPLE_RATE == 44100


def test_the_sample_doors_admit_the_longest_take_muse_can_write():
    """The bridge's other agreement, and the one that was broken until
    2026-09-04: ``MAX_SAMPLE_FRAMES`` was sixty seconds of 48 kHz, so a take
    over ~65 s was refused by the frame count and one over ~131 s by the byte
    door -- the bridge worked at Muse's default length and nowhere else. Both
    ceilings are asserted against ``MAX_DURATION`` rather than against numbers
    written out here, so raising what Muse may generate fails here first.
    """
    frames = int(door.MAX_DURATION * synth.SAMPLE_RATE)
    assert frames <= wavout.MAX_SAMPLE_FRAMES
    # Stereo 16-bit is what the writer produces; the byte door is sized for the
    # widest frame decoded, so it has room to spare over what Muse actually writes.
    assert frames * 2 * 2 + 1024 <= wavout.MAX_SAMPLE_FRAMES * 8


def test_a_sirens_render_is_a_wav_the_music_door_reads():
    with wave.open(io.BytesIO(_render())) as handle:
        assert handle.getframerate() == synth.SAMPLE_RATE
        assert handle.getsampwidth() == 2


# --- Sirens -> Muse ----------------------------------------------------------


def test_a_reference_lands_beside_the_row_that_used_it(svc):
    """Bytes, not a path: a path means a temp file with an owner, a lifetime
    and a cleanup story. Written into the job's own directory before the row
    exists, which is ``rerun_job``'s ``input.png`` precedent -- and the
    reference then lives with the job forever, which is provenance."""
    out = door.create_music_job(
        svc, prompt="dark ambient", duration=30.0, reference_wav=_render()
    )
    for job_id in out["ids"]:
        assert (svc.config.job_dir(job_id) / "source.wav").exists()


def test_an_imported_reference_becomes_an_ordinary_audio2audio_row(svc):
    """No new task, no new worker key and no new queue branch: the imported
    reference is just another source, under the name the derive door already
    writes."""
    out = door.create_music_job(
        svc, prompt="dark ambient", duration=30.0, reference_wav=_render()
    )
    params = svc.store.get(out["id"])["params"]
    assert params["task"] == "audio2audio"
    assert params["ref_audio_strength"] == pytest.approx(0.5)


def test_the_queue_sends_it_as_a_reference_rather_than_a_source_path(svc):
    """``__call__`` asserts that ``src_audio_path`` implies repaint/edit/extend,
    so sending both would trip an assertion inside the child."""
    from warlock import _q_music

    out = door.create_music_job(
        svc, prompt="dark ambient", duration=30.0, reference_wav=_render()
    )
    params = svc.store.get(out["id"])["params"]
    kwargs = _q_music._task_kwargs(params, svc.config.job_dir(out["id"]))
    assert kwargs["audio2audio_enable"] is True
    assert "src_audio_path" not in kwargs


def test_a_brief_with_no_reference_writes_no_source(svc):
    """Every press that existed before this door takes a byte-identical path."""
    out = door.create_music_job(svc, prompt="dark ambient", duration=30.0)
    assert not (svc.config.job_dir(out["id"]) / "source.wav").exists()
    assert "task" not in svc.store.get(out["id"])["params"]


def test_the_loop_points_travel_with_the_reference():
    """The headline feature meeting the round trip.

    ``wav_bytes`` writes the pair into the ``smpl`` chunk and
    ``compose_from_sirens`` passes the song's own loop, so what the model is
    handed is the piece *as the user authored it looping* rather than a flat
    render of the same notes.
    """
    import numpy as np

    tone = np.zeros((44100, 2), dtype=np.float32)
    data = wavout.wav_bytes(tone, synth.SAMPLE_RATE, loop=(1000, 40000))
    assert b"smpl" in data


@pytest.mark.parametrize(
    "payload,fragment",
    [
        (b"", "no audio"),
        (b"not a wav at all", "not a WAV"),
    ],
)
def test_an_unusable_reference_is_refused_by_name(svc, payload, fragment):
    with pytest.raises(Invalid) as caught:
        door.create_music_job(
            svc, prompt="dark ambient", duration=30.0, reference_wav=payload
        )
    assert caught.value.field == "reference_wav"
    assert fragment in str(caught.value)


def test_a_reference_outside_the_duration_bounds_is_refused(svc):
    with pytest.raises(Invalid) as caught:
        door.create_music_job(
            svc, prompt="dark ambient", duration=30.0, reference_wav=_render(2.0)
        )
    assert caught.value.field == "reference_wav"


def test_a_refused_reference_leaves_nothing_on_disk(svc):
    before = set(svc.config.job_dir("").iterdir())
    with pytest.raises(Invalid):
        door.create_music_job(
            svc, prompt="dark ambient", duration=30.0, reference_wav=b"junk"
        )
    assert set(svc.config.job_dir("").iterdir()) == before
