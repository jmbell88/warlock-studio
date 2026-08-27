"""The one module that touches ``pygame.mixer`` -- and, mostly, the machine
that has no mixer at all.

**The no-device path is the important one**, because it is the one CI runs and
because it is what the whole mode's testability rests on: the engine is
import-pinned against pygame so that a box with no sound hardware can still
open a song, edit it and export a WAV, and that promise is only worth something
if every function here *answers* rather than raises. Every test below that
stubs ``pygame.mixer.init`` into a failure is asserting the same thing from a
different door.

The device *is* exercised, through a fake mixer rather than through a card:
what these tests are about is this module's own bookkeeping -- the reserved
channel, the held reference, the derived playhead -- and a real card would make
them a question about the host instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from warlock.studio import sirens_audio


@pytest.fixture(autouse=True)
def _fresh():
    """The cached answer is module-level, so it has to go between tests."""
    sirens_audio._reset()
    yield
    sirens_audio._reset()


class _FakeChannel:
    def __init__(self) -> None:
        self.played: list[Any] = []
        self.stopped = 0
        self.busy = False

    def play(self, sound: Any) -> None:
        self.played.append(sound)
        self.busy = True

    def stop(self) -> None:
        self.stopped += 1
        self.busy = False

    def get_busy(self) -> bool:
        return self.busy


class _FakeMixer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.init_args: Any = None
        self.reserved = 0
        self.channel = _FakeChannel()

    def init(self, *args: Any) -> None:
        import pygame

        if self.fail:
            raise pygame.error("no available audio device")
        self.init_args = args

    def set_reserved(self, count: int) -> None:
        self.reserved = count

    def Channel(self, index: int) -> _FakeChannel:  # noqa: N802 -- pygame's spelling
        assert index == 0
        return self.channel


def _install(monkeypatch, *, fail: bool = False) -> _FakeMixer:
    import pygame

    mixer = _FakeMixer(fail=fail)
    monkeypatch.setattr(pygame, "mixer", mixer)
    monkeypatch.setattr(
        pygame, "sndarray", type("_S", (), {"make_sound": staticmethod(lambda a: a)})
    )
    return mixer


def _pcm(seconds: float = 0.5) -> np.ndarray:
    frames = int(sirens_audio.RATE * seconds)
    return np.zeros((frames, 2), dtype=np.int16)


# --- no device ----------------------------------------------------------------


def test_no_device_is_an_answer_rather_than_an_exception(monkeypatch):
    _install(monkeypatch, fail=True)
    assert sirens_audio.available() is False
    assert sirens_audio.play(_pcm()) is False
    assert sirens_audio.playing() is False
    assert sirens_audio.position() == 0.0
    # And a stop with nothing to stop is not an error either: the mode calls it
    # from a tab close, which happens whether or not anything ever played.
    sirens_audio.stop()


def test_the_answer_is_cached_so_a_dead_device_is_asked_once(monkeypatch):
    """A failing ``mixer.init`` is not free -- SDL enumerates drivers on every
    attempt -- and a pane asking every frame would ask thousands of times a
    minute."""
    mixer = _install(monkeypatch, fail=True)
    calls: list[int] = []
    real = mixer.init

    def counted(*args: Any) -> None:
        calls.append(1)
        real(*args)

    mixer.init = counted
    for _ in range(5):
        sirens_audio.available()
    assert len(calls) == 1


def test_an_import_that_fails_outright_is_still_just_unavailable(monkeypatch):
    """A build with no SDL raises on the import rather than on ``init``, and
    the answer to both is the same False."""
    import builtins

    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pygame":
            raise ImportError("no pygame here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert sirens_audio.available() is False


def test_the_unavailable_sentence_says_what_still_works():
    """A fault with no consequence in it is a dead end; this one has to say
    that writing and exporting are unaffected, because they are."""
    text = sirens_audio.unavailable_reason()
    assert "playback is unavailable" in text
    assert "export" in text.lower()


# --- with a device ------------------------------------------------------------


def test_the_rate_and_the_buffer_are_ours(monkeypatch):
    """``pygame.init()`` already opened a mixer at whatever SDL picked; a rate
    that is not the engine's resamples every playback, audibly."""
    mixer = _install(monkeypatch)
    assert sirens_audio.available() is True
    assert mixer.init_args == (
        sirens_audio.RATE,
        sirens_audio.SIZE,
        sirens_audio.CHANNELS,
        sirens_audio.BUFFER,
    )


def test_the_engine_and_the_device_agree_on_the_sample_rate():
    """Deliberately not a shared import: this module stays importable with the
    engine absent, so the mismatch is caught here rather than hidden."""
    from warlock.studio.sirens import synth

    assert sirens_audio.RATE == synth.SAMPLE_RATE


def test_one_channel_is_reserved_so_playback_never_loses_its_slot(monkeypatch):
    mixer = _install(monkeypatch)
    sirens_audio.available()
    assert mixer.reserved == 1


def test_playing_a_buffer_holds_a_reference_to_it(monkeypatch):
    """pygame keeps none for us, and a garbage-collected ``Sound`` is silence
    halfway through a bar."""
    _install(monkeypatch)
    assert sirens_audio.play(_pcm()) is True
    assert sirens_audio._sound is not None


def test_a_second_play_stops_the_first(monkeypatch):
    mixer = _install(monkeypatch)
    sirens_audio.play(_pcm())
    sirens_audio.play(_pcm())
    assert mixer.channel.stopped >= 1


def test_a_mono_buffer_is_widened_rather_than_refused(monkeypatch):
    """The mixer is open stereo and ``make_sound`` refuses the mismatch with an
    exception that names no array."""
    _install(monkeypatch)
    assert sirens_audio.play(np.zeros(1000, dtype=np.int16)) is True
    assert sirens_audio._sound.shape[1] == 2


def test_a_buffer_at_the_wrong_rate_is_refused_rather_than_transposed(monkeypatch):
    _install(monkeypatch)
    assert sirens_audio.play(_pcm(), rate=22050) is False


def test_an_empty_buffer_plays_nothing(monkeypatch):
    _install(monkeypatch)
    assert sirens_audio.play(np.zeros((0, 2), dtype=np.int16)) is False


def test_the_playhead_lags_by_the_buffer_and_never_runs_backwards(monkeypatch):
    """The mixer has accepted samples the speaker has not reached; a cursor
    ignoring that draws the playhead a row early at 150 BPM, and a negative one
    scrolls the grid backwards before it scrolls forwards."""
    _install(monkeypatch)
    sirens_audio.play(_pcm(seconds=2.0))
    assert sirens_audio.position() == 0.0
    # Two buffers' worth later, the estimate is one buffer behind the clock.
    sirens_audio._started -= 3 * sirens_audio.BUFFER / sirens_audio.RATE
    latency = sirens_audio.BUFFER / sirens_audio.RATE
    assert sirens_audio.position() == pytest.approx(2 * latency, abs=1e-3)


def test_the_playhead_never_runs_past_the_buffer(monkeypatch):
    _install(monkeypatch)
    sirens_audio.play(_pcm(seconds=0.25))
    sirens_audio._started -= 10.0
    assert sirens_audio.position() == pytest.approx(0.25)


def test_stopping_clears_the_clock(monkeypatch):
    _install(monkeypatch)
    sirens_audio.play(_pcm())
    sirens_audio.stop()
    assert sirens_audio.playing() is False
    assert sirens_audio.position() == 0.0


# --- the pin ------------------------------------------------------------------


def test_this_is_the_only_module_in_the_repo_that_touches_the_mixer():
    """The engine's headless pin is stated in ``tests/sirens``; this is the
    other half of it, over the app layer -- a second module reaching for the
    device is a second place "this machine has no card" has to be handled, and
    the mode's whole degradation story rests on there being one."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "warlock"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if path.name == "sirens_audio.py":
            continue
        # Parsed rather than grepped: the engine's own docstring *names* this
        # module and the rule it keeps, and a scan that counted prose would
        # make writing the invariant down the thing that breaks it.
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            reached = (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "pygame"
                and node.attr in ("mixer", "sndarray")
            )
            if reached:
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == []
