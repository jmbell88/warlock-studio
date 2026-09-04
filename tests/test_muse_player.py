"""The player's verbs: the playhead, the region, seeking and the exports.

``test_muse_mode``'s context, because these are the same controller -- but its
own file because every test here needs a *decoded take* on the state, which the
brief's tests have no use for.

Nothing here touches a sound card. ``sirens_audio`` is stubbed to a recorder, so
what is asserted is what the mode asks the device for -- which is the whole
contract between the two: the mixer owns one channel and Muse owns the offset
that makes its clock absolute.
"""

from __future__ import annotations

import io
import wave
from typing import Any

import numpy as np
import pytest
from test_muse_mode import FakeCtx

from warlock.studio import muse_io, muse_mode, muse_state

RATE = 44100


class _Device:
    """``sirens_audio``, as a recorder. One channel, exactly as the real one."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.tag_value = ""
        self.busy = False
        self.pos = 0.0
        self.level = 1.0

    RATE = RATE

    def play(self, pcm, rate=RATE, *, tag="", loops=0) -> bool:
        self.calls.append({"frames": len(pcm), "rate": rate, "tag": tag, "loops": loops})
        self.tag_value = tag
        self.busy = True
        return True

    def stop(self) -> None:
        self.busy = False
        self.tag_value = ""

    def playing(self) -> bool:
        return self.busy

    def tag(self) -> str:
        return self.tag_value if self.busy else ""

    def position(self) -> float:
        return self.pos

    def volume(self) -> float:
        return self.level

    def set_volume(self, value) -> None:
        self.level = value

    def unavailable_reason(self) -> str:
        return ""


@pytest.fixture
def device(monkeypatch):
    one = _Device()
    monkeypatch.setattr(muse_mode, "sirens_audio", one)
    return one


@pytest.fixture
def ctx(tmp_path):
    one = FakeCtx(tmp_path)
    one.cache = type("_Cache", (), {"jobs": [{"id": "a"}]})()
    return one


def _loaded(ctx, seconds: float = 10.0, job: str = "a"):
    """Put a decoded take on the state, the way ``on_task_done`` would."""
    from warlock.studio.muse import waveform

    pcm = np.zeros((int(seconds * RATE), 2), dtype=np.int16)
    state = muse_mode.ensure(ctx)
    state.player = muse_state.Player(
        job=job, pcm=pcm, rate=RATE, env=waveform.peaks(pcm), duration=seconds
    )
    return state.player


# --- the playhead ------------------------------------------------------------


def test_there_is_no_player_before_the_first_audition():
    """``player`` never builds one -- ``active``'s rule, on the other field."""
    from pathlib import Path

    assert muse_mode.player(FakeCtx(Path("."))) is None


def test_the_position_is_the_offset_plus_the_mixers_clock(ctx, device):
    """The two halves of one number.

    ``sirens_audio.position`` answers where the playhead is in the *buffer*, and
    seeking is slice-and-replay -- so the offset the slice began at is what
    makes it a position in the take. The mixer deliberately does not track it:
    it does not own the caller's buffer.
    """
    one = _loaded(ctx)
    one.play_offset = 4.0
    device.busy, device.tag_value, device.pos = True, "a", 1.5
    assert muse_mode.position(ctx) == pytest.approx(5.5)


def test_a_stopped_player_reads_at_its_offset_rather_than_at_zero(ctx, device):
    one = _loaded(ctx)
    one.play_offset = 4.0
    assert muse_mode.position(ctx) == pytest.approx(4.0)


def test_the_position_never_runs_past_the_end(ctx, device):
    one = _loaded(ctx, seconds=10.0)
    one.play_offset = 9.0
    device.busy, device.tag_value, device.pos = True, "a", 5.0
    assert muse_mode.position(ctx) == pytest.approx(10.0)


# --- seeking -----------------------------------------------------------------


def test_seeking_while_silent_moves_the_playhead_without_making_a_noise(ctx, device):
    """A seek is not a play. Clicking the waveform to look at something must
    not start the mode making a sound."""
    _loaded(ctx)
    muse_mode.seek(ctx, 3.0)
    assert muse_mode.position(ctx) == pytest.approx(3.0)
    assert device.calls == []


def test_seeking_while_sounding_replays_the_remainder(ctx, device):
    """Slice-and-replay: there is no device-side seek in this mixer, so the
    buffer handed to it *is* the remainder."""
    one = _loaded(ctx, seconds=10.0)
    device.busy, device.tag_value = True, "a"
    muse_mode.seek(ctx, 4.0)
    assert device.calls[-1]["frames"] == pytest.approx(6 * RATE, rel=0.01)
    assert one.play_offset == pytest.approx(4.0)


def test_a_seek_is_clamped_to_the_take(ctx, device):
    _loaded(ctx, seconds=10.0)
    muse_mode.seek(ctx, -5.0)
    assert muse_mode.position(ctx) == 0.0
    muse_mode.seek(ctx, 999.0)
    assert muse_mode.position(ctx) == pytest.approx(10.0)


def test_playing_a_region_repeats_it_and_stops_at_its_end(ctx, device):
    """A seam is judged by hearing it come round again, so the region audition
    repeats -- and ``sirens_audio.position`` already wraps modulo the buffer
    when loops is non-zero, so the playhead falls out with no arithmetic."""
    _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 6.0)
    muse_mode.play_region(ctx)
    call = device.calls[-1]
    assert call["loops"] == -1
    assert call["frames"] == pytest.approx(4 * RATE, rel=0.01)


def test_playing_outside_a_region_does_not_repeat(ctx, device):
    _loaded(ctx)
    device.busy, device.tag_value = True, "a"
    muse_mode.seek(ctx, 1.0)
    assert device.calls[-1]["loops"] == 0


# --- the region --------------------------------------------------------------


def test_a_reversed_region_is_ordered_rather_than_refused(ctx, device):
    """One place clamps and orders, because four surfaces set these markers --
    the finder, the two grips and the two keys."""
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 8.0, 2.0)
    assert (one.loop_start, one.loop_end) == (2.0, 8.0)


def test_a_region_is_clamped_to_the_take(ctx, device):
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, -4.0, 99.0)
    assert (one.loop_start, one.loop_end) == (0.0, 10.0)


def test_clearing_takes_both_markers(ctx, device):
    one = _loaded(ctx)
    muse_mode.set_region(ctx, 1.0, 2.0)
    muse_mode.set_region(ctx, None, None)
    assert one.loop_start is None and one.loop_end is None


def test_choosing_a_candidate_adopts_it_as_the_region(ctx, device):
    one = _loaded(ctx, seconds=10.0)
    from warlock.studio.muse.loops import Candidate

    one.candidates = [Candidate(RATE, RATE * 5, 0.1), Candidate(0, RATE * 3, 0.2)]
    muse_mode.choose_candidate(ctx, 1)
    assert (one.loop_start, one.loop_end) == (0.0, 3.0)


def test_choosing_a_candidate_that_is_not_there_does_nothing(ctx, device):
    one = _loaded(ctx)
    muse_mode.choose_candidate(ctx, 7)
    assert one.loop_start is None


# --- the finder --------------------------------------------------------------


def test_the_finder_runs_on_a_task_and_its_answer_lands_in_on_task_done(ctx, device):
    one = _loaded(ctx, seconds=3.0)
    muse_mode.find_loops(ctx)
    assert ctx.submitted[-1] == f"{muse_io.FIND_PREFIX}a"
    assert one.finding is True

    from warlock.studio.muse.loops import Candidate

    done = type("_Done", (), {"key": f"{muse_io.FIND_PREFIX}a", "result": [
        Candidate(0, RATE * 2, 0.1)
    ]})()
    muse_mode.on_task_done(ctx, done)
    assert one.finding is False
    # The best one is adopted immediately: the finder's output is a ranking,
    # and a second press to hear the answer it already has is a step with no
    # decision in it.
    assert (one.loop_start, one.loop_end) == (0.0, 2.0)


def test_an_answer_for_a_different_take_is_ignored(ctx, device):
    """The player holds one take; a result that arrives after the user moved on
    describes samples that are no longer in memory."""
    one = _loaded(ctx, job="a")
    one.finding = True
    done = type("_Done", (), {"key": f"{muse_io.FIND_PREFIX}b", "result": []})()
    muse_mode.on_task_done(ctx, done)
    assert one.finding is True


def test_no_candidates_says_so_rather_than_leaving_a_spinner(ctx, device):
    one = _loaded(ctx)
    one.finding = True
    done = type("_Done", (), {"key": f"{muse_io.FIND_PREFIX}a", "result": []})()
    muse_mode.on_task_done(ctx, done)
    assert one.finding is False
    assert any("No loop points" in message for message, _ in ctx.toasts)


# --- lifecycle ---------------------------------------------------------------


def test_a_decoded_take_becomes_the_player(ctx, device):
    from warlock.studio.muse import waveform

    pcm = np.zeros((RATE, 2), dtype=np.int16)
    done = type("_Done", (), {
        "key": f"{muse_mode.LOAD_PREFIX}a",
        "result": {
            "pcm": pcm, "rate": RATE, "env": waveform.peaks(pcm), "duration": 1.0
        },
    })()
    muse_mode.on_task_done(ctx, done)
    one = muse_mode.player(ctx)
    assert one is not None and one.job == "a"
    assert one.duration == pytest.approx(1.0)
    assert device.calls[-1]["tag"] == "a"


def test_the_player_is_dropped_when_its_take_leaves_the_library(ctx, device):
    """~42 MB for four minutes. One take at a time, and this is the half that
    lets go of it."""
    _loaded(ctx, job="gone")
    muse_mode.sync(ctx)
    assert muse_mode.player(ctx) is None


def test_a_player_survives_a_frame_where_the_cache_is_empty(ctx, device):
    """An empty cache is "not loaded yet", not "every take was deleted" -- and
    dropping the buffer on a refresh frame would stop playback mid-audition."""
    _loaded(ctx, job="a")
    ctx.cache.jobs = []
    muse_mode.sync(ctx)
    assert muse_mode.player(ctx) is not None


# --- export ------------------------------------------------------------------


def _read(data: bytes) -> tuple[int, int, int]:
    with wave.open(io.BytesIO(data)) as handle:
        return handle.getframerate(), handle.getnchannels(), handle.getnframes()


def test_the_loop_export_is_the_crossfaded_body(ctx, device, monkeypatch, tmp_path):
    one = _loaded(ctx, seconds=10.0)
    one.pcm = (np.ones((10 * RATE, 2)) * 1000).astype(np.int16)
    muse_mode.set_region(ctx, 2.0, 6.0)
    out = tmp_path / "loop.wav"
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: out)
    muse_io.export_loop(ctx, one)
    rate, channels, frames = _read(out.read_bytes())
    assert (rate, channels) == (RATE, 2)
    assert frames == pytest.approx(4 * RATE, rel=0.01)


def test_the_loop_export_carries_its_points_over_the_whole_file(ctx, device, monkeypatch, tmp_path):
    """The whole file *is* the loop, which is what tells an engine to repeat it
    seamlessly rather than to find the seam itself."""
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 6.0)
    out = tmp_path / "loop.wav"
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: out)
    muse_io.export_loop(ctx, one)
    assert b"smpl" in out.read_bytes()


def test_loop_points_are_refused_over_a_crossfade(ctx, device, monkeypatch, tmp_path):
    """The exclusivity, held at the door as well as at the button.

    The samples at a crossfaded seam do not exist in the take, so an ``smpl``
    chunk pointing into the untouched file is a loop that clicks wearing a label
    saying it does not.
    """
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 6.0)
    one.xfade_ms = 40.0
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: tmp_path / "x.wav")
    muse_io.export_with_points(ctx, one)
    assert not (tmp_path / "x.wav").exists()
    assert any("loop points" in message for message, _ in ctx.toasts)


def test_loop_points_over_a_plain_seam_write_the_whole_take(ctx, device, monkeypatch, tmp_path):
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 6.0)
    one.xfade_ms = 0.0
    out = tmp_path / "track.wav"
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: out)
    muse_io.export_with_points(ctx, one)
    _, _, frames = _read(out.read_bytes())
    assert frames == 10 * RATE
    assert b"smpl" in out.read_bytes()


def test_an_export_with_no_region_says_so_rather_than_writing_nothing(ctx, device):
    one = _loaded(ctx)
    muse_io.export_loop(ctx, one)
    assert any("loop region" in message for message, _ in ctx.toasts)
    assert ctx.submitted == []


def test_a_cancelled_picker_writes_no_file(ctx, device, monkeypatch, tmp_path):
    one = _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 6.0)
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: None)
    muse_io.export_loop(ctx, one)
    assert list(tmp_path.glob("*.wav")) == []
