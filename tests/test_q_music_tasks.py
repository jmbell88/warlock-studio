"""``_q_music``'s two pure helpers: what the sampler is told, and the roll.

Both are module level and pure precisely so they can be asserted here -- with
no client, no card and no queue -- which is the reason ``_music_needs_handoff``
gives for being a function rather than an inlined expression.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from warlock import _q_music as q


def _dir() -> Path:
    return Path("C:/jobs/abc")


# --- _task_kwargs ------------------------------------------------------------


def test_a_row_with_no_task_sends_nothing_at_all():
    """The guarantee, not a convenience.

    Every music row minted before tasks existed must take a byte-identical path
    through ``client.generate``, and an empty dict is what makes that true by
    construction rather than by inspection of the call site.
    """
    assert q._task_kwargs({"duration": 60.0}, _dir()) == {}
    assert q._task_kwargs({"task": ""}, _dir()) == {}


def test_a_stored_row_naming_an_unknown_task_is_refused():
    """``_music``'s rule for an unknown model, on the other string it dispatches
    on: silently downgrading to text2music would record a recipe never run."""
    with pytest.raises(RuntimeError, match="unknown music task"):
        q._task_kwargs({"task": "remix"}, _dir())


def test_audio2audio_never_sends_a_src_audio_path():
    """``__call__`` asserts that path implies repaint/edit/extend.

    Sending both would trip an assertion *inside the child*, with the weights
    already resident -- which is the failure the door and this table exist to
    move forward. The reference travels as ``ref_audio_input`` instead.
    """
    out = q._task_kwargs(
        {"task": "audio2audio", "ref_audio_strength": 0.3}, _dir()
    )
    assert out["task"] == "text2music"
    assert out["audio2audio_enable"] is True
    assert out["ref_audio_input"].endswith("source.wav")
    assert out["ref_audio_strength"] == pytest.approx(0.3)
    assert "src_audio_path" not in out


def test_a_retake_sends_no_source_because_it_re_runs_from_the_seed():
    out = q._task_kwargs({"task": "retake", "retake_variance": 0.2}, _dir())
    assert out == {"task": "retake", "retake_variance": 0.2}


def test_an_extend_is_encoded_as_a_negative_repaint_window():
    """Upstream's spelling: the head pad runs from -left to 0 and the tail from
    the parent's duration to duration+right."""
    out = q._task_kwargs(
        {
            "task": "extend",
            "extend_left": 5.0,
            "extend_right": 10.0,
            "parent_duration": 60.0,
        },
        _dir(),
    )
    assert out["task"] == "extend"
    assert out["repaint_start"] == pytest.approx(-5.0)
    assert out["repaint_end"] == pytest.approx(70.0)
    assert out["src_audio_path"].endswith("source.wav")


def test_a_loop_is_sent_as_a_repaint_under_muses_own_name():
    out = q._task_kwargs(
        {"task": "loop", "repaint_start": 26.0, "repaint_end": 34.0}, _dir()
    )
    assert out["task"] == "repaint"
    assert out["repaint_start"] == pytest.approx(26.0)
    assert out["repaint_end"] == pytest.approx(34.0)


def test_an_edit_swaps_only_the_target_conditioning():
    out = q._task_kwargs(
        {"task": "edit", "edit_prompt": "bright strings", "edit_lyrics": "la"},
        _dir(),
    )
    assert out["edit_target_prompt"] == "bright strings"
    assert out["edit_target_lyrics"] == "la"
    assert out["src_audio_path"].endswith("source.wav")


# --- _roll_wav ---------------------------------------------------------------


def _wav(frames: np.ndarray, rate: int = 44100) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(frames.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.astype("<i2").tobytes())
    return out.getvalue()


def _frames(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data)) as handle:
        channels = handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype="<i2").reshape(-1, channels)


def test_a_roll_and_its_inverse_return_the_identical_bytes():
    """Lossless is the whole argument for the rolled repaint.

    If the roll cost anything, a loop job would degrade the take it was trying
    to join up -- and the joint the model wrote would sit inside a piece of
    music that was no longer the one the user chose.
    """
    frames = np.arange(44100 * 2, dtype="<i2").reshape(-1, 2)
    original = _wav(frames)
    rolled = q._roll_wav(original, 0.25)
    assert rolled != original
    assert q._roll_wav(rolled, -0.25) == original


def test_the_roll_moves_the_joint_to_the_middle():
    rate = 1000
    frames = np.arange(2000, dtype="<i2").reshape(-1, 2)  # 1000 frames, 1 s
    rolled = _frames(q._roll_wav(_wav(frames, rate), 0.5))
    # What was frame 500 is now frame 0, so what was the head/tail joint --
    # the wrap from frame 999 to frame 0 -- now sits at frame 500.
    assert rolled[0].tolist() == frames[500].tolist()
    assert rolled[500].tolist() == frames[0].tolist()


def test_a_file_this_build_did_not_write_is_refused_rather_than_mangled():
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)  # 8-bit, which WARLOCK 5/5 never writes
        handle.setframerate(44100)
        handle.writeframes(b"\x00" * 100)
    with pytest.raises(RuntimeError, match="16-bit PCM"):
        q._roll_wav(out.getvalue(), 0.5)


# --- _stage_rolled_wav ---------------------------------------------------------


def test_a_loop_takes_roll_back_is_staged_not_written_in_place(tmp_path, monkeypatch):
    """The 2026-09-05 audit (muse-04): a loop's roll-back wrote to
    ``track.wav`` -- the job's served artifact name -- by reading it and
    writing the rolled bytes straight back in place. A process killed between
    that read and that write left a truncated or corrupt ``track.wav`` on
    disk, exactly the failure every other writer onto a served name in this
    module (``_write_stems_sidecar``, ``separation_worker``'s per-stem
    ``.tmp``) stages against.

    The fault is injected at the boundary between the read and the on-disk
    swap -- ``Path.replace`` raises -- so the assertion distinguishes staged
    from in-place: an in-place write has already clobbered ``track.wav``
    before ``replace`` is ever called, so ``track.wav`` would come back
    rolled (silently corrupt-on-crash); a staged write leaves it byte-for-byte
    the original, because the rolled bytes only ever touched the temp
    sibling.
    """
    frames = np.arange(44100 * 2, dtype="<i2").reshape(-1, 2)
    original = _wav(frames)
    output = tmp_path / "track.wav"
    output.write_bytes(original)

    def _boom(self, target):
        raise OSError("simulated kill between read and replace")

    monkeypatch.setattr(Path, "replace", _boom)

    with pytest.raises(OSError, match="simulated kill"):
        q._stage_rolled_wav(output, 0.25)

    assert output.read_bytes() == original
