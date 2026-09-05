"""Sirens' renderer, its playhead and the four things that make a sound.

Split out of :mod:`.sirens_mode` on 2026-09-04 (T7 of the 2026-09-02 review).
The render task and its flag, the three audition keys -- a sound effect, one
pattern, one previewed note -- the transport, and the row map a playhead is
bisected out of.

The rule that shapes the whole file is one key per thing that can sound:
``TaskRunner.submit`` refuses a key already in flight, so anything sharing the
song's key would be silently dropped during a re-render, and anything sharing
its *arm* would land on ``SongTab.pcm`` and replace the song. Each prefix below
carries that argument where it is declared.

Reached from :mod:`.sirens_mode` and its siblings as ``sirens_mode.<name>``; see
the ``_MOVED`` table at the foot of that file.
"""

from __future__ import annotations

from typing import Any

from . import sirens_audio, sirens_mode
from .sirens_state import SongTab, Sounding, active, ensure  # noqa: F401

# --- rendering ----------------------------------------------------------------


def request_rerender(ctx: Any, tab: SongTab | None = None) -> None:
    """Mark the audio stale so the next pump renders it.

    Does not render here: rendering is a worker job the grid pane's pump owns,
    and a pane that started one would be doing a full synthesis pass on the
    frame thread.
    """
    tab = tab or active(ctx)
    if tab is None:
        return
    tab.render_dirty = True


def request_render(ctx: Any, tab: SongTab | None = None) -> None:
    """Ask for a render. Safe to call every frame -- that is the point."""
    tab = tab or active(ctx)
    if tab is None or not tab.render_dirty:
        return
    if not tab.doc.order:
        # Nothing to play, and nothing to render. Cleared rather than left
        # armed, so an empty song does not resubmit a no-op every frame.
        tab.pcm, tab.loop, tab.render_dirty, tab.render_error = None, None, False, ""
        return

    uid = tab.uid
    # Asked *before* the snapshot, because the snapshot is the expensive half:
    # ``wsng_bytes`` DEFLATEs every pattern and encodes every sample, on the
    # frame thread. ``submit`` refuses a key already in flight and
    # ``render_dirty`` deliberately stays armed when it does, so a song being
    # rendered was re-serialised in full on every frame until the render landed
    # -- the work thrown away, once per frame, for as long as the render took.
    if ctx.busy(f"sirens-render:{uid}"):
        return

    from .sirens import wsng

    # The snapshot; see the module docstring. Taken on the frame thread, which
    # is where the document is safe to read.
    data = wsng.wsng_bytes(tab.doc)

    # The mix's own mask, read on the frame thread with everything else the
    # task needs: a mute is view state, so the snapshot cannot carry it.
    keep = sirens_mode.audible_channels(tab.doc, tab)
    whole = len(keep) == len(tab.doc.channels)

    def run() -> dict[str, Any]:
        from ..service.errors import invalid_from
        from .sirens import synth, wavout

        try:
            doc = wsng.read_wsng(data)
            samples, loop, marks = (
                synth.render_marked(doc) if whole else synth.render_only(doc, keep)
            )
        except ValueError as exc:
            # Framed, because only a ``ServiceError``'s text survives the task
            # classifier -- and the engine's own sentence (a song past the
            # render ceiling says so, with the ceiling in it) is the half that
            # tells the user what to do about it.
            raise invalid_from(exc, "That song did not render") from exc
        return {
            "pcm": wavout.to_int16(samples),
            "loop": loop,
            "marks": marks,
            "uid": uid,
        }

    tab.rendering = True
    if ctx.submit(f"sirens-render:{uid}", run):
        # Cleared *only* on an accepted submit. The runner refuses a key
        # already in flight, and clearing regardless would drop the note that
        # arrived while the previous render was running.
        tab.render_dirty = False
    else:
        tab.rendering = False


def pump(ctx: Any) -> None:
    """Called from the grid pane's draw, which is the only thing that runs
    every frame in this mode -- the ``motion.py`` idiom."""
    request_render(ctx)
    follow_playhead(ctx)


#: The key prefix an audition carries. **Not ``sirens-render:``**, and that is
#: the decision in this pair of functions: ``TaskRunner.submit`` refuses a key
#: already in flight, so sharing the song's key would make a press of Audition
#: during a re-render do nothing -- and the arm that adopts a ``sirens-render``
#: result puts the samples on ``SongTab.pcm``, which is the buffer Space plays.
#: An effect landing there would replace the song with a coin pickup until the
#: next edit re-armed the renderer, which is the sort of thing a user reports as
#: "the song vanished". So: its own key, its own arm, and the tab's buffer is
#: never touched.
AUDITION_PREFIX = "sirens-audition:"


def audition(ctx: Any, tab: SongTab | None, uid: int) -> bool:
    """Render one sound effect and play it. -> whether the render started.

    The song's render shape, one document object smaller: the snapshot is taken
    on the frame thread, ``synth.render_oneshot`` runs on the task thread, and
    :func:`on_task_done` hands the result straight to the mixer rather than
    storing it. Nothing about an audition outlives the sound, so there is
    nothing on the tab for it to live in.

    The device is checked *before* the submit, because a machine with no card
    would otherwise spend seconds of numpy on a buffer with nowhere to go and
    say nothing about why.
    """
    tab = tab or active(ctx)
    if tab is None or tab.busy:
        return False
    if tab.doc.oneshot(uid) is None:
        return False
    if not sirens_audio.available():
        ctx.toast(sirens_audio.unavailable_reason(), "warn")
        return False

    from .sirens import wsng

    data = wsng.wsng_bytes(tab.doc)
    effect = int(uid)

    def run() -> dict[str, Any]:
        from ..service.errors import invalid_from
        from .sirens import synth, wavout

        try:
            doc = wsng.read_wsng(data)
            samples = synth.render_oneshot(doc, effect)
        except ValueError as exc:
            raise invalid_from(exc, "That sound effect did not render") from exc
        return {"pcm": wavout.to_int16(samples), "oneshot": effect}

    state = ensure(ctx)
    state.play_request += 1
    return bool(ctx.submit(f"{AUDITION_PREFIX}{tab.uid}", run, tag=state.play_request))


#: A note preview's key prefix. ``AUDITION_PREFIX``'s reasoning a third time:
#: its own key, so a note typed during a re-render is not refused, and its own
#: arm, so the buffer never lands on ``SongTab.pcm`` and replaces the song.
#:
#: One key per *tab* rather than per note, deliberately. ``TaskRunner.submit``
#: refuses a key already in flight, so a fast run of notes previews the ones it
#: has finished and drops the rest -- which is the right answer for a preview:
#: queueing them would play a chord of everything typed a second ago.
PREVIEW_PREFIX = "sirens-preview:"

#: How long a previewed note is held, in rows. Four is a beat at the default
#: speed -- long enough to hear an attack and the start of a decay, short enough
#: that a run of typed notes does not become a drone.
PREVIEW_ROWS = 4


def preview_note(ctx: Any, note: int) -> bool:
    """Play the note that was just typed, on the instrument it was stamped with.

    Silent -- and cheap -- in the three cases where it has nothing to say: the
    preview is switched off, the machine has no device, or no instrument is
    selected, in which case the note the grid holds would not sound either.
    None of them toasts: a preview is a courtesy, and a courtesy that interrupts
    typing with a message is worse than one that does not happen.
    """
    state = ensure(ctx)
    tab = state.active
    if tab is None or not state.preview or state.instrument is None:
        return False
    if not sirens_audio.available():
        return False
    if sirens_audio.playing() and sirens_audio.tag() != "preview":
        # **The song wins.** There is one reserved mixer channel, so a preview
        # would cut whatever is on it -- and typing into bar 3 while bar 1
        # plays is exactly what follow mode is for. A preview interrupts only
        # an earlier preview.
        return False

    from .sirens import wsng

    data = wsng.wsng_bytes(tab.doc)
    uid, value, kind = int(state.instrument), int(note), _caret_kind(ctx, tab)

    def run() -> dict[str, Any]:
        from .sirens import synth, wavout

        doc = wsng.read_wsng(data)
        samples = synth.render_note(
            doc, uid, value, kind=kind, rows=PREVIEW_ROWS
        )
        return {"pcm": wavout.to_int16(samples)}

    state.play_request += 1
    return bool(ctx.submit(f"{PREVIEW_PREFIX}{tab.uid}", run, tag=state.play_request))


def _caret_kind(ctx: Any, tab: SongTab) -> str:
    """The voice the caret's channel is, so a preview is the sound the grid
    will make rather than a pulse standing in for a snare."""
    state = ensure(ctx)
    channels = list(tab.doc.channels)
    if 0 <= state.channel < len(channels):
        return str(channels[state.channel].kind)
    return "pulse"


# --- playback -----------------------------------------------------------------


def play(ctx: Any, tab: SongTab | None = None) -> bool:
    """Hand the last render to the device. -> whether it started.

    Refuses rather than renders when the buffer is stale: the pump is a frame
    away, and playing the *previous* version of a bar the user has just edited
    is the one outcome that makes them doubt what they heard.
    """
    tab = tab or active(ctx)
    if tab is None:
        return False
    if not sirens_audio.available():
        ctx.toast(sirens_audio.unavailable_reason(), "warn")
        return False
    if tab.rendering or tab.render_dirty:
        # Stale, whether or not an older buffer exists: the transport reads
        # "Rendering..." and the old bar must not play under it.
        ctx.toast("Still rendering your latest edits -- try again in a moment.", "info")
        return False
    if tab.pcm is None:
        ctx.toast("There is nothing in the order list to play yet.", "error")
        return False
    state = ensure(ctx)
    state.play_request += 1
    looping = bool(state.loop_playback)
    if not sirens_audio.play(tab.pcm, tag=tab.uid, loops=-1 if looping else 0):
        ctx.toast("That song could not be played; see the log for details.", "error")
        return False
    tab.sounding = Sounding(
        marks=tab.marks,
        anchor=0,
        wrap=int(len(tab.pcm)) if looping else None,
        generation=tab.render_generation,
    )
    return True


def play_from_caret(ctx: Any, tab: SongTab | None = None) -> bool:
    """Play from the row the caret is on. -> whether it started.

    The render's own row map says where that row starts in the buffer, so this
    is a slice rather than a second render -- which is the whole reason the map
    exists beyond drawing a highlight. Writing bar 40 of a three-minute song
    and having to hear the first two minutes to check it is the complaint this
    answers.

    A caret on a pattern the order list never reaches has no offset, and says
    so rather than starting from the top: playing something else is a worse
    answer than not playing.
    """
    state = ensure(ctx)
    tab = tab or state.active
    if tab is None or not _playable(ctx, tab):
        return False
    offset = _caret_offset(tab, state)
    if offset is None:
        ctx.toast(
            "The song never reaches this row -- add this pattern to the order "
            "list, or press Play to hear it from the top.",
            "info",
        )
        return False
    state.play_request += 1
    offset = int(offset)
    looping = bool(state.loop_playback)
    if looping:
        # **The whole song repeats, not the tail (S4, 2026-09-05).** This used
        # to hand the mixer ``pcm[offset:]`` with ``loops=-1``, so "from the
        # caret" with loop playback on repeated whatever was left of the song
        # from bar 40 onward and never came back to bar 1 -- M10's bug in Muse,
        # still live here. Rotating the full buffer means the repeat covers the
        # song exactly once per lap; ``Sounding.wrap`` unwinds the rotation when
        # the playhead asks where we are.
        import numpy as np

        buffer = np.concatenate([tab.pcm[offset:], tab.pcm[:offset]]) if offset else tab.pcm
    else:
        buffer = tab.pcm[offset:]
    if len(buffer) == 0:
        return False
    if not sirens_audio.play(buffer, tag=tab.uid, loops=-1 if looping else 0):
        ctx.toast("That song could not be played; see the log for details.", "error")
        return False
    tab.sounding = Sounding(
        marks=tab.marks,
        anchor=offset,
        wrap=int(len(tab.pcm)) if looping else None,
        generation=tab.render_generation,
    )
    return True


def _caret_offset(tab: SongTab, state: Any) -> int | None:
    """Where in the render the caret's row starts, in samples, or ``None``.

    **The order entry decides, not the pattern (S3, 2026-09-05).** A pattern
    used at order entries 00 and 03 is one uid at two places in the song, and
    walking the map for the first mark whose *pattern* matches always found 00 --
    so writing the last chorus and pressing "from the caret" played the first
    one. ``state.order_index`` is preferred when the caret came from the order
    list; falling back to pattern-only is what keeps a caret placed by clicking
    the grid working, where there is no entry to name.
    """
    wanted = state.order_index
    for at, order_index, pattern, row in tab.marks:
        if wanted is not None and order_index != int(wanted):
            continue
        if pattern == state.pattern and row >= state.row:
            return int(at)
    if wanted is None:
        return None
    # The order list moved under the caret (an entry deleted, the list
    # reordered) -- answer for the pattern rather than refusing to play.
    for at, _order_index, pattern, row in tab.marks:
        if pattern == state.pattern and row >= state.row:
            return int(at)
    return None


def play_pattern(ctx: Any, tab: SongTab | None = None) -> bool:
    """Render the caret's pattern alone and play it. -> whether it started.

    ``synth.render_pattern`` has existed since the engine landed with nothing
    calling it. Auditioning one pattern is how a bar gets written -- the order
    list is a later question -- and the sound effect audition beside this one is
    the same shape: its own key, its own arm, and the tab's own buffer never
    touched, so the song is still the song when this finishes.
    """
    state = ensure(ctx)
    tab = tab or state.active
    if tab is None or state.pattern is None:
        return False
    if not sirens_audio.available():
        ctx.toast(sirens_audio.unavailable_reason(), "warn")
        return False
    from .sirens import wsng

    data = wsng.wsng_bytes(tab.doc)
    uid = int(state.pattern)

    def run() -> dict[str, Any]:
        from ..service.errors import invalid_from
        from .sirens import synth, wavout

        try:
            doc = wsng.read_wsng(data)
            samples = synth.render_pattern(doc, uid)
        except ValueError as exc:
            raise invalid_from(exc, "That pattern did not render") from exc
        return {"pcm": wavout.to_int16(samples), "pattern": uid}

    # Keyed on the *tab*, like every other ``sirens-`` key: the arm below looks
    # a tab up from the second segment, and there is one channel anyway, so two
    # patterns auditioning at once is not a thing to make room for.
    state.play_request += 1
    return bool(ctx.submit(f"{PATTERN_PREFIX}{tab.uid}", run, tag=state.play_request))


def _playable(ctx: Any, tab: SongTab) -> bool:
    """The three refusals :func:`play` and :func:`play_from_caret` share."""
    if not sirens_audio.available():
        ctx.toast(sirens_audio.unavailable_reason(), "warn")
        return False
    if tab.rendering or tab.render_dirty:
        ctx.toast("Still rendering your latest edits -- try again in a moment.", "info")
        return False
    if tab.pcm is None:
        ctx.toast("There is nothing in the order list to play yet.", "error")
        return False
    return True


#: A pattern audition's key prefix. ``AUDITION_PREFIX``'s reasoning exactly: a
#: key of its own, so a pattern played during a re-render is not refused, and
#: an arm of its own, so the result never lands on ``SongTab.pcm``.
PATTERN_PREFIX = "sirens-pattern:"


def stop(ctx: Any) -> None:
    """Silence, from any surface. A no-op with no device.

    **Withdraws the request as well as the sound (S1, 2026-09-05).** A preview,
    a pattern audition or a sound effect is rendered on a task thread, and
    ``on_task_done`` used to hand every successful one straight to the mixer
    with no freshness check at all -- so pressing a note and then Stop played
    the note, a second or two after the user had asked for silence. Bumping the
    counter here is what makes those completions stale; see
    ``SirensState.play_request``.
    """
    sirens_audio.stop()
    # ``ctx.state.sirens`` rather than ``ensure``: Stop is on the keymap, so it
    # can be pressed in a session that never opened a song, and building the
    # state to withdraw a request nobody made is ``active``'s rule.
    state = ctx.state.sirens
    if state is not None:
        state.play_request += 1
        for tab in state.docs:
            tab.sounding = None


def toggle_play(ctx: Any, tab: SongTab | None = None) -> bool:
    """What Space does: stop if sounding, start if not."""
    if sirens_audio.playing():
        stop(ctx)
        return True
    return play(ctx, tab)


def playhead_mark(ctx: Any, tab: SongTab | None = None) -> tuple[int, int, int] | None:
    """``(order index, pattern uid, row)`` that is sounding, or ``None``.

    :func:`sirens_audio.position` says how far the mixer has got -- it is the
    only thing that knows, and it is accurate to within a buffer -- and the
    render's own row map says what was playing there. Bisected rather than
    computed: the arithmetic this replaced (seconds over the document's
    seconds-per-row) described one imaginary pattern of unbounded length, so it
    was wrong for any song with an order list longer than one entry, and wrong
    again after every ``Fxx``, ``Bxx`` and ``Dxx``.
    """
    tab = tab or active(ctx)
    if tab is None or sirens_audio.tag() != tab.uid:
        # ``tag`` and not merely ``playing``: there is one channel, so
        # auditioning a sound effect replaces the song on it -- and a playhead
        # bisecting the *song's* row map while a coin pickup sounds is a
        # highlight walking rows nothing is playing.
        return None
    return tab.mark_at(sirens_audio.position(), synth_rate())


def synth_rate() -> int:
    """The render's sample rate. A function so the import stays lazy."""
    from .sirens import synth

    return int(synth.SAMPLE_RATE)


def playhead_row(ctx: Any, tab: SongTab | None = None) -> int | None:
    """Which row *of the pattern on screen* is sounding, or ``None``.

    ``None`` while the song is somewhere else, which is the half the old
    estimate could not express at all: it returned a row number whatever was
    playing, so a two-pattern song highlighted a row of the pattern the user
    was looking at because another pattern had reached it.
    """
    state = ensure(ctx)
    mark = playhead_mark(ctx, tab)
    if mark is None:
        return None
    order_index, pattern, row = mark
    if state.pattern is not None and pattern != state.pattern:
        return None
    if state.order_index is not None and order_index != int(state.order_index):
        # **This entry, not merely this pattern (S3).** A chorus at order
        # entries 00 and 03 lit the grid up during 00 while the caret -- and so
        # the user -- was in 03, which is a highlight that lies about where the
        # song is.
        return None
    return row


def follow_playhead(ctx: Any) -> bool:
    """Move the caret onto the sounding row while following. -> whether it moved.

    Follow mode used to scroll the *view* and leave the caret where it was, so
    the two most ordinary things a person does while a song plays -- watch it,
    and type the next note -- disagreed: the row under the highlight was not
    the row a keystroke wrote to, and on a song with more than one pattern the
    highlight was not even in the pattern being edited. The tracker answer is
    that following moves the caret, so what is under the playhead is what is
    being edited; ``follow`` is off with one click for anyone who wants to type
    into bar 3 while bar 1 plays.
    """
    state = ensure(ctx)
    if not state.follow:
        return False
    mark = playhead_mark(ctx)
    if mark is None:
        return False
    _order_index, pattern, row = mark
    if state.pattern == pattern and state.row == row:
        return False
    state.pattern = pattern
    state.row = row
    # Every other thing that moves the caret clears this; a playhead is no
    # different, and a half-typed instrument number carried onto another row
    # would finish itself there.
    state.digit = 0
    return True
