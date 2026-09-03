"""The only module in the repo that touches ``pygame.mixer``.

Kept apart from everything else in Sirens deliberately, and the split is the
whole reason the mode is testable. ``studio/sirens/`` is import-pinned against
pygame (``tests/sirens/test_sirens_imports.py``) so that a machine with no
sound hardware can still open a song, edit it, render it and export a WAV; that
promise is only worth anything if the *mode* degrades the same way, and it can
only degrade cleanly if there is exactly one door to fail at. This is it.

**No device is never an error.** Every function here answers rather than
raises: :func:`available` returns False, :func:`play` returns False, and
:func:`playing` returns False forever after. The mode reads that and says
playback is unavailable, the way rigging says bpy is missing rather than
refusing to draw -- and CI, which has no card at all, exercises the same code
path a user with a broken driver gets.

**The mixer is initialised here even though ``pygame.init()`` already tries.**
``pygame.init()`` initialises every subsystem with whatever defaults SDL picks,
which for the mixer is a rate and a buffer nobody chose. Sirens renders at
44100 stereo int16, so a mixer at any other rate resamples every playback --
audibly, on a chiptune square wave -- and a buffer picked by the platform is a
latency the playhead cannot estimate. Calling ``mixer.init`` with our numbers
after ``pygame.init()`` is a no-op when the defaults already matched and a
re-open when they did not, and it is the one place the failure is legible.

**The playhead is derived, not read.** ``Channel.get_pos()`` does not exist in
pygame and ``Sound`` has no cursor, so the elapsed time is ``perf_counter()``
at play minus the buffer's latency: the samples the mixer has accepted are
ahead of the ones the speaker has reached by roughly one buffer, and a cursor
that ignored that draws the playhead a row early at 150 BPM. It is an estimate
and it is named one; nothing in the mode depends on it being exact, because the
document is the truth and this only moves a highlight.
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)

#: What we ask the device for. The rate is the engine's -- see
#: ``sirens.synth.SAMPLE_RATE``, which this deliberately does not import: this
#: module must stay importable with the engine absent, and a mismatch is caught
#: by a test rather than by a shared constant that hides it.
RATE = 44100
#: 16-bit signed, which is what ``wavout.to_int16`` already produces.
SIZE = -16
CHANNELS = 2
#: One buffer is ~23 ms at 44.1 kHz. Small enough that the playhead estimate is
#: within a row at any sane tempo, large enough that a frame that took 30 ms to
#: draw does not underrun -- playback runs on the mixer's own thread, but the
#: GIL is still shared with a frame loop doing GL work.
BUFFER = 1024

#: ``None`` until the first :func:`available` call; True or False after. Cached
#: because a failing ``mixer.init`` is not free -- on a machine with no device
#: SDL enumerates drivers on every attempt -- and because a mode that asks
#: every frame would ask thousands of times a minute.
_available: bool | None = None

#: The reserved channel, and the sound currently on it. Held so :func:`stop`
#: has something to stop and so the sound is not collected mid-playback --
#: pygame does not keep a reference for us, and a garbage-collected ``Sound``
#: is silence halfway through a bar.
_channel: Any = None
_sound: Any = None

#: ``perf_counter()`` at the moment :func:`play` handed the buffer over, and
#: the length in seconds of what was handed over. Both zero when nothing is
#: playing.
_started: float = 0.0
_length: float = 0.0

#: What is on the channel: the caller's own name for the buffer it handed over
#: (a tab uid for a song, an effect's key for an audition), or "". There is one
#: channel, so an audition replaces the song -- and a playhead drawn against a
#: song's row map while a sound effect is sounding is a highlight moving
#: through rows nothing is playing.
_tag: str = ""

#: What was passed to ``Channel.play``: 0 for once, -1 for forever. Kept
#: because :func:`position` has to answer differently for the two -- a looping
#: buffer's playhead wraps, and clamping it at the end would park the highlight
#: on the last row for the rest of the session.
_loops: int = 0


def _reset() -> None:
    """Forget the device answer and drop the held sound. Tests only.

    Exposed rather than left to a test poking module globals because the three
    pieces of state have to go together: a cached ``True`` with a released
    channel is a module that thinks it can play and cannot.
    """
    global _available, _channel, _sound, _started, _length, _tag, _loops
    _available = None
    _channel = None
    _tag = ""
    _loops = 0
    _sound = None
    _started = 0.0
    _length = 0.0


def available() -> bool:
    """Whether this machine can play anything. Cached after the first call.

    The ``pygame.error`` clause is the point of the function: a headless box,
    a container with no ``/dev/snd``, a laptop whose driver is mid-update --
    all of them raise here and all of them are ordinary. ``Exception`` rather
    than ``pygame.error`` alone for the outer clause because importing pygame
    itself can fail on a build without SDL, and the answer to that is the same
    False.
    """
    global _available, _channel
    if _available is not None:
        return _available
    try:
        import pygame

        pygame.mixer.init(RATE, SIZE, CHANNELS, BUFFER)
        # One channel reserved, so ``Sound.play()`` from anywhere else -- a UI
        # click, anything added later -- can never take the slot the transport
        # is using mid-bar. Reserved channels are allocated from index 0, so
        # channel 0 is ours and ``find_channel`` will not hand it out.
        pygame.mixer.set_reserved(1)
        _channel = pygame.mixer.Channel(0)
        _available = True
    except Exception:
        log.info("sirens: no audio device; playback is unavailable", exc_info=True)
        _channel = None
        _available = False
    return _available


def play(pcm: Any, rate: int = RATE, *, tag: str = "", loops: int = 0) -> bool:
    """Play an ``int16`` buffer. -> whether it started.

    ``pcm`` is ``(n, 2)`` ``int16`` -- what ``wavout.to_int16`` produces from a
    render. A mono ``(n,)`` buffer is accepted and widened here rather than at
    the call site, because the mixer is opened stereo and ``make_sound``
    refuses the mismatch with an exception that says nothing about which array.

    The rate is an argument and is *checked* rather than resampled: the mixer
    is open at :data:`RATE` and a buffer at any other rate would play at the
    wrong pitch. A caller that has one is asking for something this module
    cannot do, and saying so is better than transposing their song.
    """
    global _sound, _started, _length, _tag, _loops
    if not available():
        return False
    if int(rate) != RATE:
        log.warning("sirens: refusing to play %s Hz audio through a %s Hz mixer", rate, RATE)
        return False
    import numpy as np
    import pygame

    buffer = np.ascontiguousarray(pcm, dtype=np.int16)
    if buffer.ndim == 1:
        buffer = np.repeat(buffer[:, None], CHANNELS, axis=1)
    if buffer.ndim != 2 or buffer.shape[1] != CHANNELS:
        log.warning("sirens: refusing to play a buffer shaped %s", (buffer.shape,))
        return False
    if buffer.shape[0] == 0:
        return False
    try:
        _sound = pygame.sndarray.make_sound(buffer)
        stop()
        _channel.play(_sound, loops=int(loops))
    except Exception:
        log.exception("sirens: the device refused a buffer")
        _sound = None
        return False
    _started = time.perf_counter()
    _length = buffer.shape[0] / float(RATE)
    _tag = str(tag)
    _loops = int(loops)
    return True


def stop() -> None:
    """Silence. Safe with no device, and safe when nothing is playing."""
    global _started, _length, _tag, _loops
    _started = 0.0
    _length = 0.0
    _tag = ""
    _loops = 0
    if _channel is not None:
        try:
            _channel.stop()
        except Exception:  # pragma: no cover -- a device lost mid-session
            log.exception("sirens: the device refused a stop")


def playing() -> bool:
    """Whether a buffer is still sounding.

    Asked of the channel rather than of the clock, because the clock does not
    know about a device that stalled -- but the clock is the fallback when the
    channel cannot answer, which is what a mixer torn down under us looks like.
    """
    if _channel is None or _started == 0.0:
        return False
    try:
        return bool(_channel.get_busy())
    except Exception:  # pragma: no cover -- a device lost mid-session
        return time.perf_counter() - _started < _length


def tag() -> str:
    """Whose buffer is on the channel, or "". See :data:`_tag`."""
    return _tag if playing() else ""


def position() -> float:
    """Seconds into the current buffer, or ``0.0`` when nothing is playing.

    The buffer-latency subtraction is the estimate the module docstring
    describes, clamped at zero: for the first ~23 ms the speaker has reached
    nothing at all, and a negative playhead would scroll the grid backwards
    before it scrolled forwards.
    """
    if _started == 0.0:
        return 0.0
    elapsed = time.perf_counter() - _started - (BUFFER / float(RATE))
    if elapsed <= 0.0:
        return 0.0
    if _loops != 0 and _length > 0.0:
        return elapsed % _length
    return min(elapsed, _length)


def unavailable_reason() -> str:
    """One sentence for a pane to draw when :func:`available` is False.

    Here rather than in the pane so that the mode and the transport strip say
    the same thing -- and so the sentence names the consequence (you can still
    write and export) rather than only the fault, which is what turns a dead
    button into a fact the user can act on.
    """
    return (
        "No audio device: playback is unavailable on this machine."
        " Writing, saving and exporting all still work."
    )
