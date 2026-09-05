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

#: ``muse_mode.precompute_loop``'s task key, one per job so a second region
#: tweak before the first blend lands is refused rather than queued -- the
#: newer call wins on the *next* settle, which is the loop-cache's existing
#: "not invalidated eagerly" policy applied to when it is *filled* too.
CACHE_PREFIX = "muse-loopcache:"


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


def loop_cache_key(player: Any) -> tuple[int, int, int] | None:
    """``(start, end, fade)`` in samples -- the three numbers that decide the
    seam, and so the cache key both :func:`loop_body` and
    ``muse_mode.precompute_loop`` key on. -> ``None`` with no usable region.

    Split out of ``loop_body`` so the precompute task can ask "is the cache
    already current" without paying for -- or risking -- the blend itself.
    """
    if player.loop_start is None or player.loop_end is None:
        return None
    rate = int(player.rate)
    start = int(player.loop_start * rate)
    end = int(player.loop_end * rate)
    if end <= start:
        return None
    fade = int(player.xfade_ms * rate / 1000.0)
    return (start, end, fade)


def _blend(player: Any, key: tuple[int, int, int]) -> Any:
    """The crossfade itself, for exactly ``key`` -- pure, and it never reads
    or writes ``player.loop_cache``/``loop_cache_key``.

    Split out of :func:`loop_body` by **incident-2026-09-05b**, a correction
    to muse-03: see that function's docstring for why nothing but the frame
    thread may touch the cache pair. Both :func:`compute_loop_cache` (the
    precompute task) and :func:`export_loop` (the export task) call this
    directly, on a task thread, precisely because it has nothing left to tear.
    """
    return loops_mod.crossfade(player.pcm, *key)


def compute_loop_cache(player: Any) -> tuple[tuple[int, int, int], Any] | None:
    """The ``(key, buffer)`` pair for the region as it stands right now.
    -> ``None`` with no usable region. Task work: pure, safe on any thread.

    **incident-2026-09-05b**, a correction to muse-03. The first pass at that
    finding had ``muse_mode.precompute_loop`` submit ``loop_body`` itself as
    the task -- which moved the *blend* off the frame thread, but the blend
    was never the whole story: ``loop_body`` also writes
    ``player.loop_cache``/``loop_cache_key``, in two separate statements,
    straight onto the shared ``Player``. Two different keys' writes could
    then interleave --

    1. task (key A): ``player.loop_cache = A``
    2. frame (key B): ``player.loop_cache = B``
    3. frame (key B): ``player.loop_cache_key = B``
    4. task (key A): ``player.loop_cache_key = A``

    -- leaving the player claiming key A while holding B's buffer, which is
    exactly the ordinary case of marking a region and pressing Play before a
    ~100 ms blend has finished. ``export_loop`` reads the same two fields, so
    the wrong bytes would land under the name the user chose -- M09's claim
    ("what you hear" and "what gets written" are the same buffer) broken by a
    race rather than a bug in the blend itself.

    The fix is not a lock; it is having only one thread ever write the pair.
    This function computes and returns it as a single value, touching nothing
    on ``player``; ``muse_mode.on_task_done`` is what installs the result,
    and it does that on the frame thread -- the same thread ``_play_from``
    calls :func:`loop_body` from, so the two writers can never overlap with
    each other or with themselves.
    """
    key = loop_cache_key(player)
    if key is None:
        return None
    return key, _blend(player, key)


def loop_body(player: Any) -> Any:
    """The crossfaded loop -- the *one* buffer both the audition and
    :func:`export_loop` play. -> ``None`` with no usable region.

    **M09.** Before this, ``muse_mode``'s ``_play_from`` handed the mixer a raw
    slice of ``pcm`` while this function crossfaded independently on export, so
    the crossfade slider could be dragged to any value with no audible
    difference through the button the strip advertises for judging it -- an
    audition reported byte-identical to the untreated take. Building the one
    buffer here and having both callers reach for it is what makes "what you
    hear" and "what gets written" the same claim rather than two
    implementations that happen to agree.

    **Cached by ``(start, end, fade)`` in samples**, on the player itself,
    because those three numbers are exactly what decides the seam -- a region
    or a crossfade the user has not touched since the last call is a repeated
    ``O(n)`` blend for nothing, which matters once the crossfade slider is
    something a drag calls every frame. The cache is not invalidated eagerly
    when a control changes; it simply no longer matches on the *next* call,
    which is the "refresh deliberately" half of M09 -- dragging the slider
    while a loop sounds does not itself restart the buffer on the channel
    (``muse_player``'s "re-played on release only" rule), but the next Play or
    seek picks up the new blend.

    **Frame-thread only (incident-2026-09-05b).** The two statements below
    are not atomic together, so this may be called from exactly one thread --
    the frame thread, via ``muse_mode._play_from``. Neither ``export_loop``
    nor the muse-03 precompute task calls this any more; they call
    :func:`_blend`/:func:`compute_loop_cache`, which touch nothing on
    ``player``, for that reason.
    """
    key = loop_cache_key(player)
    if key is None:
        return None
    if player.loop_cache_key != key:
        player.loop_cache = _blend(player, key)
        player.loop_cache_key = key
    return player.loop_cache


def export_loop(ctx: Any, player: Any) -> None:
    """The crossfaded loop body, as its own file.

    The picker runs *inside* the task -- ``sirens_io``'s rule, for its reason: a
    native dialog is modal to the OS, and on the frame thread that is a frozen
    window. ``dialogs.save_file`` rather than ``select_folder`` because this
    writes one file under a name the user chooses, which is the case that
    picker is for.

    **muse-02** (2026-09-05 audit): the blend and the whole WAV byte encode
    (``_wav``) used to run right here, before ``_save`` was ever reached -- on
    the frame thread, since nothing had submitted anything yet. A 240 s take
    made that a visible freeze on every press, and the cost was paid again
    every time because nothing cached the *encode*. Both now live inside the
    closure ``_save`` hands to ``ctx.submit``, so the frame thread sees only
    the request and, later, the result.

    **Calls ``_blend``, not ``loop_body`` (incident-2026-09-05b).** This runs
    on the export task's own thread, and ``loop_body`` writes the shared
    cache pair in two statements that are only safe from one thread at a
    time -- the frame thread, which this is not. ``_blend`` computes fresh
    every time instead, which costs a redundant O(n) blend on the rare
    press that follows one right after a Play, paid for on a task either way.
    """
    if not _has_region(ctx, player):
        return
    rate = int(player.rate)

    def make() -> bytes | None:
        key = loop_cache_key(player)
        if key is None:
            return None
        body = _blend(player, key)
        # ``loop=(0, len)``: the whole file *is* the loop, which is the
        # difference between this product and the other one.
        return _wav(body, rate, loop=(0, int(body.shape[0])))

    _save(ctx, make, "loop.wav", "Export the loop")


def export_with_points(ctx: Any, player: Any) -> None:
    """The whole take, carrying its loop points in an ``smpl`` chunk.

    Refused whenever a crossfade is asked for, and the refusal is the point --
    see this module's docstring. The caller disables the control and states the
    same sentence; this is the door holding it as well, for the reason every
    door in this app is held twice.

    **muse-02** (2026-09-05 audit): the refusal stays here, on the frame
    thread -- it is a field check, not a computation -- but the WAV encode of
    the whole take moves into ``_save``'s task for the same reason
    :func:`export_loop`'s does.
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
    pcm = player.pcm
    loop = (int(player.loop_start * rate), int(player.loop_end * rate))
    _save(ctx, lambda: _wav(pcm, rate, loop=loop), "track.wav", "Export the track")


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


def _save(ctx: Any, make: Any, default_name: str, title: str) -> None:
    """Ask where to write, then write ``make()``'s bytes there -- both on the
    task, never on the frame thread.

    The picker runs *inside* the task already, and for its own reason (a
    modal native dialog on the frame thread is a frozen window). ``make`` is
    called after the picker returns a path rather than before, so a cancelled
    export -- the common case for a press that was a change of mind -- costs
    no encode at all, not merely one hidden from the frame thread. See
    ``export_loop``/``export_with_points`` for what ``make`` does: the
    crossfade blend, the whole WAV byte encode, or both.
    """
    from . import atomic

    def run() -> str | None:
        path = dialogs.save_file(title, default_name, dialogs.filters_for(".wav"))
        if path is None:
            return None
        data = make()
        if data is None:
            return None
        # Staged, never written in place: the rule every other writer in this
        # app follows, and it matters most where the user has picked an
        # existing file to overwrite.
        atomic.write_bytes(path, data)
        return str(path)

    if not ctx.submit(f"{EXPORT_PREFIX}{default_name}", run):
        ctx.toast("Still exporting -- try again in a moment.")


__all__ = [
    "CACHE_PREFIX",
    "EXPORT_PREFIX",
    "FIND_PREFIX",
    "compute_loop_cache",
    "export_loop",
    "export_with_points",
    "find_loops",
    "loop_cache_key",
    "loop_body",
    "read_track",
]
