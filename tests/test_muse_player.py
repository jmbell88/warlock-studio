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
        self.calls.append(
            {"frames": len(pcm), "rate": rate, "tag": tag, "loops": loops, "pcm": np.array(pcm)}
        )
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


def test_seeking_inside_the_region_loops_the_whole_region_not_just_the_tail(ctx, device):
    """M10 repro: seeking to 5s inside a 2-8s region used to slice
    ``pcm[start:loop_end]`` and loop *that* -- a buffer of only three seconds,
    not the marked six-second region. Fails against the unfixed code, whose
    buffer length there is ``loop_end - seek`` rather than
    ``loop_end - loop_start``.
    """
    _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 8.0)
    muse_mode.play_region(ctx)  # start the loop, as "Play the loop" does
    muse_mode.seek(ctx, 5.0)
    call = device.calls[-1]
    assert call["loops"] == -1
    assert call["frames"] == pytest.approx(6 * RATE, rel=0.01)


def test_seeking_before_the_region_does_not_force_a_loop(ctx, device):
    """A region existing at all used to loop *any* seek, even one that lands
    outside it -- ``repeat`` depended only on whether a region was set, never
    on whether the seek landed inside it. Fails against the unfixed code,
    which reports ``loops == -1`` here.
    """
    _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 8.0)
    device.busy, device.tag_value = True, "a"
    muse_mode.seek(ctx, 0.5)
    assert device.calls[-1]["loops"] == 0


def test_seeking_after_the_region_does_not_force_a_loop(ctx, device):
    """The mirror of the above, and a worse instance of the same bug: seeking
    past the region's end used to loop the unbounded remainder to the end of
    the take, forever. Fails against the unfixed code (``loops == -1`` and a
    one-second buffer standing in for the whole remainder).
    """
    _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 8.0)
    device.busy, device.tag_value = True, "a"
    muse_mode.seek(ctx, 9.0)
    assert device.calls[-1]["loops"] == 0
    assert device.calls[-1]["frames"] == pytest.approx(1 * RATE, rel=0.01)


def test_shrinking_the_region_while_sounding_is_picked_up_on_the_next_seek(ctx, device):
    """Changing the markers mid-loop must be reflected the next time the take
    is actually replayed. Fails against the unfixed code, whose loop buffer is
    always "seek point to loop end" (1s here) rather than the full, narrowed
    region (2s).
    """
    _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 8.0)
    muse_mode.play_region(ctx)
    muse_mode.set_region(ctx, 3.0, 5.0)  # narrowed while sounding
    device.busy, device.tag_value = True, "a"
    muse_mode.seek(ctx, 4.0)
    assert device.calls[-1]["frames"] == pytest.approx(2 * RATE, rel=0.01)


def test_the_position_wraps_within_the_region_after_a_seek_inside_it(ctx, device):
    """Old code's ``play_offset`` was the seek point and its buffer began
    there too, so adding the mixer's raw clock straight to it was correct by
    accident, for a buffer that was the wrong length. With the loop rotated to
    start at the seek point, the wrap has to be computed against the
    *region's* length and re-based at ``loop_start`` -- this pins that
    arithmetic. Fails against the unfixed code, which returns 8.5 here
    (``play_offset`` 5.0 plus the raw clock 3.5, with no wrap at all).
    """
    _loaded(ctx, seconds=10.0)
    muse_mode.set_region(ctx, 2.0, 8.0)
    device.busy, device.tag_value = True, "a"
    muse_mode.seek(ctx, 5.0)
    device.pos = 3.5
    assert muse_mode.position(ctx) == pytest.approx(2.5)


# --- resuming a loaded take (M07) --------------------------------------------


def test_stop_captures_the_playhead_rather_than_losing_it(ctx, device):
    """M07: the playhead the mixer was actually at must survive Stop. Fails
    against the unfixed code, which never reads ``position(ctx)`` before
    stopping and leaves ``play_offset`` wherever the last ``_play_from`` call
    had started -- 3.0 here, not the 4.0 the take had actually reached.
    """
    one = _loaded(ctx, seconds=10.0)
    one.play_offset = 3.0
    device.busy, device.tag_value, device.pos = True, "a", 1.0
    muse_mode.stop(ctx)
    assert one.play_offset == pytest.approx(4.0)


def test_stop_then_play_resumes_the_same_take_without_losing_its_state(
    ctx, device, monkeypatch
):
    """M07: pressing Play on an already-loaded take must not rebuild the
    ``Player``. The old code always resubmitted the decode -- so a 2-8s
    region, a 200ms crossfade and a mid-take offset all reverted to nothing on
    the very next Play, since ``on_task_done`` builds a brand-new
    ``MusePlayer`` for every successful load. Fails against the unfixed code,
    which submits a fresh decode (``ctx.submitted`` is non-empty) for a take
    already sitting in memory with nothing to re-read.
    """
    from test_muse_mode import _finished

    _finished(ctx, "a")
    monkeypatch.setattr(muse_mode, "_read_track", lambda path: {"pcm": [], "rate": RATE})
    one = _loaded(ctx, seconds=10.0, job="a")
    muse_mode.set_region(ctx, 2.0, 8.0)
    one.xfade_ms = 200.0
    device.busy, device.tag_value, device.pos = True, "a", 1.0
    one.play_offset = 3.0
    muse_mode.stop(ctx)
    assert one.play_offset == pytest.approx(4.0)

    muse_mode.play(ctx, "a")
    assert ctx.submitted == [], "an already-decoded take must be resumed, not re-read"
    resumed = muse_mode.player(ctx)
    assert resumed is one, "the same Player, not a fresh one built by on_task_done"
    assert (resumed.loop_start, resumed.loop_end) == (2.0, 8.0)
    assert resumed.xfade_ms == pytest.approx(200.0)
    call = device.calls[-1]
    assert call["loops"] == -1  # 4.0 is inside the (2, 8) region
    assert call["frames"] == pytest.approx(6 * RATE, rel=0.01)


# --- one buffer for the audition and the export (M09) ------------------------


def test_playing_the_loop_hands_the_mixer_the_export_buffer_not_a_raw_slice(
    ctx, device, monkeypatch, tmp_path
):
    """M09: before this, ``play_region``/seeking inside the region handed the
    mixer a raw, uncrossfaded slice of ``pcm`` while ``export_loop``
    crossfaded independently on write -- so the crossfade slider could be
    dragged to any value with no audible difference through the advertised
    audition. Fails against the unfixed code, whose buffer here is the plain
    slice and does not match ``muse.loops.crossfade``'s output at all once the
    fade is non-zero.
    """
    from warlock.studio.muse import loops as loops_mod

    one = _loaded(ctx, seconds=10.0)
    rng = np.random.default_rng(0)
    one.pcm = (rng.standard_normal((10 * RATE, 2)) * 5000).astype(np.int16)
    muse_mode.set_region(ctx, 2.0, 8.0)
    one.xfade_ms = 200.0
    rate = one.rate
    expected = loops_mod.crossfade(
        one.pcm, int(2.0 * rate), int(8.0 * rate), int(200.0 * rate / 1000.0)
    )

    muse_mode.play_region(ctx)  # phase 0: starts at the region's own start
    assert np.array_equal(device.calls[-1]["pcm"], expected)

    out = tmp_path / "loop.wav"
    monkeypatch.setattr(muse_io.dialogs, "save_file", lambda *a, **k: out)
    muse_io.export_loop(ctx, one)
    with wave.open(str(out)) as handle:
        raw = handle.readframes(handle.getnframes())
    exported = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
    assert np.array_equal(exported, expected), "the export must write the same buffer"


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

    muse_mode.ensure(ctx).audition_job = "a"
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
