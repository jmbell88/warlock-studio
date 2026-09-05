"""What Muse remembers between frames.

Much smaller than ``sirens_state`` / ``inker_state`` / ``plotter_state``,
because **Muse holds no document.** The other document workspaces each own a file
format, a tab list, a dirty flag and an undo stack; a take is a job row that a
worker wrote, so the store owns it and there is nothing here to lose. What is
left is a form and a pointer at whatever is currently making a noise -- which is
Troupe's shape, the one other workspace whose subject is rows a worker
published.

``ensure`` and ``active`` live here rather than in ``muse_mode`` for the reason
they live in ``sirens_state``: they touch exactly one thing, ``ctx.state.muse``,
and neither knows that a task thread or a mixer exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The form's defaults, and the single place they are written down.
#:
#: They are deliberately *not* ``create_music_job``'s signature defaults
#: repeated: that function's defaults are what an API caller gets, this is what
#: a new user sees, and a test asserting the two agree would freeze a UI choice
#: to a door's choice. What must agree is that every value here is one the door
#: accepts, which is a bound and not a value.
DEFAULT_FORM: dict[str, Any] = {
    "prompt": "",
    "lyrics": "",
    "duration": 60.0,
    "count": 1,
    # The recipe half, drawn by the column rather than the bar. Their names are
    # the model's own, so what the user changes and what is asked for do not
    # need a translation between them.
    "infer_step": 60,
    "guidance_scale": 15.0,
    "scheduler_type": "euler",
    "cfg_type": "apg",
    "omega_scale": 10.0,
    # None means "a fresh one per take", which is what a generative mode
    # defaults to; a number makes the first take reproducible.
    "seed": None,
}


#: The derive popup's defaults. ``_jobs_music``'s bounds are the authority; as
#: with :data:`DEFAULT_FORM`, what must agree is that every value here is one
#: the door accepts -- a bound, not a value.
#:
#: ``repaint_start``/``repaint_end`` open on a window the door refuses for a
#: 60 s take only if it is the whole take, which it is not: 0-8 s is the first
#: phrase, which is the window a user most often wants and the one the loop
#: task reads as "eight seconds of joint".
DEFAULT_DERIVE: dict[str, Any] = {
    "task": "retake",
    "count": 1,
    "retake_variance": 0.5,
    "extend_left": 0.0,
    "extend_right": 30.0,
    "repaint_start": 0.0,
    "repaint_end": 8.0,
    "edit_prompt": "",
    "edit_lyrics": "",
    "ref_audio_strength": 0.5,
}


#: The default crossfade at a loop seam, in milliseconds.
#:
#: **A control, not a tuned constant**, which is why there is a default rather
#: than a figure: it trades two things the user can hear against each other. A
#: short fade keeps every transient at the seam and can click; a long one is
#: certainly smooth and audibly ducks the music through the join. 40 ms is
#: under a frame of most game audio and long enough to swallow a phase
#: mismatch the zero-crossing snap could not close.
DEFAULT_XFADE_MS = 40.0

#: The longest crossfade the player will offer. Half a second is already long
#: enough to hear as a dip rather than a join; past it the control is asking
#: for a different effect.
MAX_XFADE_MS = 500.0


@dataclass
class Player:
    """One decoded take, and where the player is in it.

    Held rather than re-read because the read is ~40 MB off disk -- which is
    why :func:`muse_mode.play` submits it to a task in the first place. **One
    take at a time**: four minutes of 44.1 kHz stereo int16 is ~42 MB, and a
    cache of every take the tray has ever shown would be a mode that grows
    without bound while the user auditions candidates. It is dropped in
    ``sync`` when its take leaves the Library.
    """

    #: Whose take this is. "" when nothing has been loaded yet.
    job: str = ""

    #: The decoded frames, ``int16``, at ``sirens_audio.RATE``. The mixer's
    #: buffer and the player's picture come from the one array.
    pcm: Any = None
    rate: int = 0

    #: ``muse.waveform.peaks`` of :attr:`pcm`, computed once on the same task
    #: thread that did the read. The pane windows it per frame.
    env: Any = None

    #: The take's length in seconds, from the frame count -- never from
    #: ``params["duration"]``, which is what the worker was *asked* for.
    duration: float = 0.0

    #: Where in the take the buffer currently on the channel began.
    #:
    #: ``SongTab.play_offset`` a second time, and for its reason: seeking is
    #: slice-and-replay (``sirens_play.play_from_caret``), so the mixer's own
    #: clock is relative to the slice and this is what makes it absolute again.
    #: ``sirens_audio`` deliberately gains no offset of its own -- it does not
    #: own the caller's buffer, and a second module tracking the same number is
    #: how the two come to disagree.
    play_offset: float = 0.0

    #: The loop region, in seconds. ``None`` until a region is set.
    loop_start: float | None = None
    loop_end: float | None = None

    #: The crossfade at the seam, in milliseconds. See :data:`DEFAULT_XFADE_MS`.
    xfade_ms: float = DEFAULT_XFADE_MS

    #: Where the buffer *currently on the channel* begins, in take-seconds,
    #: when that buffer is the rotated loop body rather than a plain remainder
    #: -- ``None`` otherwise. M10: seeking to a point inside the marked region
    #: no longer shrinks what loops to "seek point to loop end"; it rotates the
    #: whole region so playback starts exactly at the seek point and wraps
    #: through the rest of the region before repeating. ``position()`` needs
    #: this anchor to unwind that rotation -- without it, the mixer's own
    #: modulo-buffer-length clock describes a position in the *rotated*
    #: buffer, not in the take.
    loop_anchor: float | None = None

    #: The crossfaded loop body -- ``muse.loops.crossfade`` over
    #: ``(loop_start, loop_end, xfade_ms)`` -- cached here so pressing Play the
    #: loop twice does not redo an O(n) blend, and so the audition and
    #: ``muse_io.export_loop`` build from the exact same call rather than two
    #: that merely agree by construction (M09: before this, the strip
    #: auditioned a raw slice of ``pcm`` while the export crossfaded on write,
    #: so the crossfade control could not be judged through the button
    #: advertised for judging it). ``muse_io.loop_body`` owns the cache
    #: invalidation; see its docstring for the key.
    loop_cache: Any = None
    loop_cache_key: tuple[int, int, int] | None = None

    #: What ``muse.loops.find`` last offered, best first, and whether a search
    #: is in flight. A list rather than one answer because the finder is a
    #: heuristic over material nobody composed to loop -- see that module.
    candidates: list[Any] = field(default_factory=list)
    finding: bool = False


@dataclass
class MuseState:
    """The mode's whole memory."""

    #: The brief and the recipe, one flat dict, edited in place by the bar and
    #: the column. One dict rather than two so that a future "reuse this take's
    #: settings" is a copy rather than a merge of two halves.
    form: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_FORM))

    #: The job id currently being auditioned, or "". The *audio* is
    #: ``sirens_audio``'s -- it is tag-keyed and mode-agnostic, and Muse passes
    #: the job id as the tag -- so this is only what the results tray needs in
    #: order to draw one card's button as Stop rather than Play. Kept here
    #: rather than asked of the mixer every frame because a card has to know
    #: whether *it* is the one playing, which the mixer's tag answers and its
    #: "is anything playing" does not.
    playing_job: str = ""

    #: The take the tray has selected, or "". Space auditions this one, which
    #: is why it is state rather than a hover.
    selected_job: str = ""

    #: The job id of the most recent audition *request* -- set by ``play()``,
    #: cleared by ``stop()``. M11: a take's decode is a task, several can be in
    #: flight for different jobs at once, and they can land out of order.
    #: ``on_task_done`` adopts a completed decode only when it still matches
    #: this field, which is what stops an older, slower decode from landing on
    #: top of a newer one already sounding -- and stops one from starting
    #: playback at all after the user has pressed Stop.
    audition_job: str = ""

    #: The decoded take under the player strip, or ``None``. Built on the first
    #: audition and replaced whenever a different take is played.
    player: Player | None = None

    #: Which take the derive popup is open over, or "". One at a time by
    #: construction: a popup per card would be six sets of controls on screen
    #: at once, all but one of them about a take the user is not looking at.
    derive_job: str = ""

    #: The derive popup's own form. **Deliberately separate from ``form``**,
    #: which is the *brief* -- what the next Generate will ask for. A derivation
    #: is a statement about one finished take, so mixing the two would make
    #: opening a popup silently rewrite the brief, and closing it leave the
    #: brief wearing a repaint window. Reset from :data:`DEFAULT_DERIVE` every
    #: time the popup opens, for the same reason.
    derive_form: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_DERIVE))


def ensure(ctx: Any) -> MuseState:
    """The mode's state, built on first use."""
    state = ctx.state.muse
    if state is None:
        state = MuseState()
        ctx.state.muse = state
    return state


def active(ctx: Any) -> MuseState | None:
    """The state, or ``None``. Deliberately *not* through :func:`ensure`:
    asking what Muse holds must not create the state that says nothing."""
    return ctx.state.muse
