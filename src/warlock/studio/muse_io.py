"""Everything that touches a take's file on disk.

The ``sirens_mode``/``sirens_io`` split, at the same seam: the mode is the
controller and this is the I/O, so a reader looking for "what does Export write"
has one file to open. ``muse_mode._read_track`` moves here for that reason, with
a re-export left behind -- it is not new code, it is code that was in the wrong
file.

**Two loop products, and they are mutually exclusive.** Export the loop writes
the crossfaded body; Export the track with loop points writes the whole take
with an ``smpl`` chunk. A crossfade cannot be expressed as an ``smpl`` pointer
into the original file, because the samples at the seam *do not exist in the
source* -- this module writes them. So a ``smpl`` chunk pointing at ``(s, e)``
of an untouched track is a loop that clicks wearing a label saying it does not,
and the second export is refused whenever a fade is asked for. That is the kind
of thing that gets "simplified" back into a bug, so it is also in
``docs/INVARIANTS.md``.

**Importing ``wavout`` from here is the existing pattern, not an exception.**
``tests/sirens/test_sirens_imports.py`` pins what files *inside* that package
import, and ``sirens_io.export_plan`` already does ``from .sirens import
wavout`` from a ``studio/`` module. It is a RIFF encoder misfiled under
``sirens/`` because Sirens was its first caller; if a third one ever appears it
should be promoted to ``studio/wavout.py``.
"""

from __future__ import annotations

import logging
from typing import Any

from . import dialogs
from .muse import loops as loops_mod

log = logging.getLogger(__name__)

#: Task keys. Prefixed for ``on_task_done``'s routing, exactly as
#: ``muse_mode.LOAD_PREFIX`` is: a result delivered under a bare key is a result
#: delivered nowhere.
EXPORT_PREFIX = "muse-export:"
FIND_PREFIX = "muse-loops:"


def read_track(path: Any) -> dict[str, Any]:
    """A take's WAV as ``int16`` frames, its rate, its envelope and its length.

    Blocking; task work. ``soundfile`` rather than ``sirens/wavout.read_wav``,
    which looks like the obvious reuse and is the wrong function: that one
    exists to feed a chip voice, so it mixes to **mono** unconditionally. This
    is a finished stereo track being played back as itself -- only the *rate*
    needs reconciling with the mixer, not the channel count.

    ``scipy.signal.resample_poly`` is used and is allowed to be: it retimes the
    in-memory buffer handed to the mixer, and nothing exported depends on it.
    That is precisely why this function lives out here rather than in
    ``studio/muse/``, which bans scipy because what *it* computes ends up in a
    file. ``WARLOCK 5/5`` now writes 44 100 Hz, so the resample is the
    already-written path for an older take rather than the normal one.

    The envelope is computed here, on the thread that already holds the samples,
    because it is a full pass over ~10 M frames and the frame loop must not
    make one.
    """
    import math

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    from . import sirens_audio
    from .muse import waveform

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    rate = int(rate)
    target = sirens_audio.RATE
    if rate != target:
        g = math.gcd(rate, target)
        data = resample_poly(data, target // g, rate // g, axis=0)
        rate = target
    pcm = np.clip(np.round(data * 32767.0), -32768, 32767).astype(np.int16)
    env = waveform.peaks(pcm)
    if pcm.shape[1] == 1:
        pcm = pcm[:, 0]
    return {
        "pcm": pcm,
        "rate": rate,
        "env": env,
        "duration": pcm.shape[0] / float(rate),
    }


def find_loops(pcm: Any, rate: int) -> list[Any]:
    """``muse.loops.find``, as a task. -> the candidates, best first.

    A wrapper of one line, and it exists so the *task* has a name in this module
    beside the export it feeds. Being slow here costs a spinner rather than a
    frozen window, which is the whole reason the finder is unbounded.
    """
    return loops_mod.find(pcm, rate)


# --- export ------------------------------------------------------------------


def _wav(pcm: Any, rate: int, loop: tuple[int, int] | None = None) -> bytes:
    import numpy as np

    from .sirens import wavout

    data = np.asarray(pcm)
    # ``wav_bytes`` takes floats and re-quantises. Divided by the same 32767
    # ``to_int16`` multiplies by, which is ``read_wav``'s own argument for that
    # constant: the pair is exact, so a take exported unchanged comes back
    # sample for sample.
    return wavout.wav_bytes(data.astype(np.float32) / 32767.0, rate, loop=loop)


def export_loop(ctx: Any, player: Any) -> None:
    """The crossfaded loop body, as its own file.

    The picker runs *inside* the task -- ``sirens_io``'s rule, for its reason: a
    native dialog is modal to the OS, and on the frame thread that is a frozen
    window. ``dialogs.save_file`` rather than ``select_folder`` because this
    writes one file under a name the user chooses, which is the case that
    picker is for.
    """
    if not _has_region(ctx, player):
        return
    rate = int(player.rate)
    start = int(player.loop_start * rate)
    end = int(player.loop_end * rate)
    fade = int(player.xfade_ms * rate / 1000.0)
    body = loops_mod.crossfade(player.pcm, start, end, fade)
    # ``loop=(0, len)``: the whole file *is* the loop, which is the difference
    # between this product and the other one.
    data = _wav(body, rate, loop=(0, int(body.shape[0])))
    _save(ctx, data, "loop.wav", "Export the loop")


def export_with_points(ctx: Any, player: Any) -> None:
    """The whole take, carrying its loop points in an ``smpl`` chunk.

    Refused whenever a crossfade is asked for, and the refusal is the point --
    see this module's docstring. The caller disables the control and states the
    same sentence; this is the door holding it as well, for the reason every
    door in this app is held twice.
    """
    if not _has_region(ctx, player):
        return
    if player.xfade_ms > 0.0:
        ctx.toast(
            "A crossfaded seam cannot be written as loop points: those samples"
            " are not in the take. Set the crossfade to zero, or export the"
            " loop itself.",
            "warn",
        )
        return
    rate = int(player.rate)
    loop = (int(player.loop_start * rate), int(player.loop_end * rate))
    _save(ctx, _wav(player.pcm, rate, loop=loop), "track.wav", "Export the track")


def _has_region(ctx: Any, player: Any) -> bool:
    if player is None or player.pcm is None:
        return False
    if player.loop_start is None or player.loop_end is None:
        ctx.toast("Set a loop region first -- press Find loop points.", "warn")
        return False
    if player.loop_end <= player.loop_start:
        ctx.toast("The loop ends before it starts.", "warn")
        return False
    return True


def _save(ctx: Any, data: bytes, default_name: str, title: str) -> None:
    from . import atomic

    def run() -> str | None:
        path = dialogs.save_file(title, default_name, dialogs.filters_for(".wav"))
        if path is None:
            return None
        # Staged, never written in place: the rule every other writer in this
        # app follows, and it matters most where the user has picked an
        # existing file to overwrite.
        atomic.write_bytes(path, data)
        return str(path)

    if not ctx.submit(f"{EXPORT_PREFIX}{default_name}", run):
        ctx.toast("Still exporting -- try again in a moment.")


__all__ = [
    "EXPORT_PREFIX",
    "FIND_PREFIX",
    "export_loop",
    "export_with_points",
    "find_loops",
    "read_track",
]
