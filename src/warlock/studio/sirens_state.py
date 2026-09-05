"""Multi-document state for Sirens, without imgui and with one device call.

The device call is :meth:`SirensState.activate`, which stops the mixer on a tab
switch (S6) because there is one channel and the song left behind would go on
sounding under the song brought forward. It is a local import inside that one
method; everything else here is answerable on a box with no sound card, which
is what lets the tests ask it.

The ``inker_state`` / ``plotter_state`` / ``packwright_state`` split, fourth
instance. What is here is everything the mode has to remember that is not in
the document and not on a task thread: which songs are open, which cell the
caret is in, and the last buffer the renderer handed back.

**No view type.** The other three import ``PaintView`` because a zoomable,
pannable 2D viewport has clamping rules worth having once. A tracker grid has
none of that -- it scrolls by rows and columns, which are integers, and the
cursor *is* the scroll anchor. A ``PaintView`` here would be a pan and a zoom
nothing reads.

**The PCM is a task result, not a document field**, exactly as Packwright's
atlas is. ``SongDoc`` holds notes and instruments and derives its audio on
demand -- which is what makes a re-export reproducible -- but rendering a
three-minute song is several seconds of numpy over every tick, which is not
frame-thread work. So the samples and the loop points live here, on the tab,
adopted when the task lands.

**``render_generation`` names which render a sounding buffer came from.** A
re-render replaces the buffer wholesale, and a counter bumped in exactly one
place is the cheapest way to name one: a hash of a 50 MB buffer costs a frame,
and the document's head misses a mute, which is a render change and not a
document change. It is the identity :class:`Sounding` stamps itself with at the
moment playback starts, which is the whole of what it is for -- the sentence
here used to say playback *keyed* on it while nothing read it at all, and
playback in fact keys on ``sirens_audio``'s tag (S2, 2026-09-05).
"""

from __future__ import annotations

import bisect
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import docmodes

#: Bigger than any order index, pattern uid or row, so a bisect key of
#: ``(offset, _LAST, _LAST, _LAST)`` lands past every mark that shares the
#: offset -- which is what makes "the row sounding *now*" the last one that
#: started, rather than the first.
_LAST = float("inf")

WSNG_SUFFIX = ".wsng"

_uids = itertools.count(1)

#: The five cell columns, as the grid names them for a tooltip and for the
#: manual. Indexed by ``sirens.document.NOTE`` and friends, so the order here
#: is the document's order and a test asserts it rather than a comment.
COLUMN_LABELS: tuple[str, ...] = ("Note", "Instrument", "Volume", "Effect", "Param")

#: How many hex digits each column takes from the keyboard, in the document's
#: column order. ``0`` means the column is not hex at all -- the note column is
#: the piano row and the effect column is one letter out of
#: ``synth.EFFECT_NAMES`` -- and ``2`` is the pair of nibbles a byte is typed
#: as, which is what :attr:`SirensState.digit` counts through.
#:
#: Here beside the labels rather than in the mode, because it is the same kind
#: of fact about the same five columns and the grid pane needs it too: the
#: caret narrows to one character exactly where a second keystroke is still
#: owed, and a pane deriving that from its own table would be a second table.
COLUMN_DIGITS: tuple[int, ...] = (0, 2, 1, 0, 2)

#: What ``+``/``-`` move between, and where a new document's caret starts. Two
#: octaves below the note names' midpoint, which is where a bassline and a lead
#: are both one key away.
MIN_OCTAVE, MAX_OCTAVE = 0, 9
DEFAULT_OCTAVE = 4

#: How far the caret falls after a note is entered. Zero means "stay put",
#: which is what a user auditioning one cell wants; one is what writing a
#: melody wants, and is the default every tracker ships.
MIN_STEP, MAX_STEP = 0, 16
DEFAULT_STEP = 1


@dataclass(frozen=True)
class Sounding:
    """The render a buffer *now on the mixer* was built from. Immutable.

    **The map has to be bound to the audio, not to the tab (S2, 2026-09-05).**
    :meth:`SongTab.adopt_render` swaps ``pcm``, ``loop`` and ``marks`` while the
    mixer is still playing a ``Sound`` made from the previous buffer -- the
    device owns a copy, so a re-render cannot disturb it. The playhead used to
    bisect the *live* fields, which passed its tag check and then read the new
    map against the old audio: with follow mode on (the default) that walks the
    caret onto a row nothing is playing. Recording the map at the moment
    playback starts leaves ``adopt_render`` free to swap whenever it likes.

    ``anchor`` is the song sample that buffer index 0 corresponds to -- zero for
    an ordinary Play, the caret's own offset for "from the caret". ``wrap`` is
    the buffer's length in samples when the buffer repeats (the mixer wraps
    modulo its own length), or ``None`` for a one-shot play-through, which is
    what lets a rotated looping buffer be unwound back into song time.
    """

    marks: tuple[tuple[int, int, int, int], ...] = ()
    anchor: int = 0
    wrap: int | None = None
    generation: int = 0

    def mark_at(self, seconds: float, rate: int) -> tuple[int, int, int] | None:
        """Which ``(order index, pattern uid, row)`` is sounding at ``seconds``.

        A bisect of :attr:`marks`, so the answer is what the renderer did rather
        than an estimate: every order entry, every pattern length, every ``Fxx``
        and every jump are already in the offsets. ``None`` before the first row
        and after the last -- the tail of a song is its instruments decaying,
        and highlighting the final row through it says the song is still on it.
        """
        marks = self.marks
        if not marks:
            return None
        offset = int(self.anchor) + int(float(seconds) * int(rate))
        if self.wrap:
            offset %= int(self.wrap)
        if offset < marks[0][0]:
            return None
        index = bisect.bisect_right(marks, (offset, _LAST, _LAST, _LAST)) - 1
        if index < 0:
            return None
        _at, order_index, pattern, row = marks[index]
        return order_index, pattern, row


@dataclass
class SongTab:
    """One tab: a song, where it came from, and the audio it last rendered to."""

    doc: Any
    title: str = "Untitled"
    path: Path | None = None
    uid: str = field(default_factory=lambda: f"sg{next(_uids)}")
    saved_head: int = 0
    saving: bool = False

    # Crash-safety, owned by :mod:`studio.journal` (UX-05). Inker's three
    # fields, verbatim, because they are the same three questions: which file
    # this tab owns under the autosave directory (minted on the first copy, so
    # an untouched tab litters nothing), the history position that copy
    # captured (an undo back to it is not a new edit), and the debounce.
    journal_name: str = ""
    journal_head: int | None = None
    journal_at: float = 0.0

    # The last successful render: ``int16`` ``(n, 2)`` at ``synth.SAMPLE_RATE``,
    # and the loop points the song asked for, in samples. Both ``None`` until a
    # render has landed.
    pcm: np.ndarray | None = None
    loop: tuple[int, int] | None = None
    #: One ``(sample offset, order index, pattern uid, row)`` per row as the
    #: renderer actually played it, ascending in the offset. What a
    #: :class:`Sounding` snapshots; see :func:`~.sirens.synth.render_marked`.
    marks: tuple[tuple[int, int, int, int], ...] = ()

    #: The render whatever this tab last put on the mixer was built from, or
    #: ``None``. Snapshotted at the moment playback starts, never re-read from
    #: the live fields -- see :class:`Sounding` for the swap it survives.
    sounding: Sounding | None = None

    #: Channel uids this song's mix is playing without, and the one channel it
    #: is playing *alone* (``-1`` for none). **View state, not the song**: a
    #: mute is how a person listens to what they are writing, and a ``.wsng``
    #: that remembered one would hand somebody else a song with a missing part.
    #: It reaches the mix through the render (``synth.render_only``), so
    #: toggling one re-renders exactly as an edit does.
    #:
    #: **Per tab, not per app (S5, 2026-09-05).** Channels are identified by
    #: uid and ``document.reserve_uid`` starts each document's count over, so
    #: two tabs opened from the same file carry identical channel uids: a mute
    #: on ``SirensState`` silently applied to the other song's next render, and
    #: because a tab switch does not re-render, the tab that was not touched
    #: went on showing a mix that disagreed with the mask.
    muted: set[int] = field(default_factory=set)
    solo: int = -1
    # Bumped once, where a render is adopted. Playback keys on it.
    render_generation: int = 0
    # A render is in flight. Separate from ``saving`` because a re-render does
    # not stop the user editing -- it only means the audio is a beat behind.
    rendering: bool = False
    # Something changed that the current buffer does not contain. A *flag*
    # pumped from the grid pane's draw rather than a direct submit, for the
    # reason ``PackTab.pack_dirty`` is one: ``TaskRunner.submit`` refuses a key
    # already in flight and nothing re-arms it, so a burst of keystrokes would
    # otherwise leave the last one unrendered for good.
    render_dirty: bool = True
    # What the last render could not do, if anything. Kept so the transport can
    # say so rather than showing a dead Play button that looks like a bug.
    render_error: str = ""

    @property
    def busy(self) -> bool:
        return self.saving

    @property
    def dirty(self) -> bool:
        return self.doc.dirty

    @property
    def label(self) -> str:
        return f"{self.title}###{self.uid}"

    def mark_at(self, seconds: float, rate: int) -> tuple[int, int, int] | None:
        """Which ``(order index, pattern uid, row)`` is sounding at ``seconds``.

        Answered against :attr:`sounding` -- the render the buffer on the mixer
        was actually built from -- and ``None`` when this tab has nothing on it.
        Never against the live :attr:`marks`, which a re-render may already have
        replaced underneath the audio still playing (S2).
        """
        one = self.sounding
        return None if one is None else one.mark_at(seconds, rate)

    def mark_saved(self, head: int | None = None) -> None:
        self.doc.mark_saved(head)
        self.saved_head = self.doc.saved_head
        self.saving = False

    def adopt_render(
        self,
        pcm: np.ndarray,
        loop: tuple[int, int] | None,
        marks: tuple[tuple[int, int, int, int], ...] = (),
    ) -> None:
        """Take a finished render. The one place ``render_generation`` moves.

        **``render_dirty`` is deliberately not touched here.** An edit made
        while a render was in flight set it, and that edit is not in the buffer
        landing now -- clearing it would drop the edit for good, because
        :func:`~.sirens_mode.request_render` clears the flag only on an
        accepted submit and nothing else re-arms it. The flag is cleared at the
        submit, never at the adoption. ``PackTab.adopt_pack`` carries the same
        rule for the same reason.
        """
        self.pcm = pcm
        self.loop = loop
        self.marks = tuple(marks)
        self.render_generation += 1
        self.rendering = False
        self.render_error = ""


@dataclass
class SirensState:
    docs: list[SongTab] = field(default_factory=list)
    active_uid: str = ""

    # --- the caret ------------------------------------------------------------
    #
    # Four numbers rather than a cursor object, because every one of them is
    # clamped against a *different* thing (the order list, the pattern's rows,
    # the document's channels, ``document.COLUMNS``) and an object would either
    # carry the document to clamp against or clamp nothing.
    #
    # ``pattern`` is a **uid**, not an index -- the document's own rule, and for
    # the document's own reason: deleting a pattern must not silently move the
    # caret into a different one.
    pattern: int | None = None
    row: int = 0
    channel: int = 0
    column: int = 0

    #: Which *entry of the order list* the caret is in, or ``None`` when the
    #: caret was put somewhere the order list did not choose. A fifth number
    #: rather than a derived one, because :attr:`pattern` is a uid and a uid
    #: does not answer the question: a chorus used at entries 00 and 03 is one
    #: pattern at two places in the song, so "play from the caret" always
    #: started at 00 and the playhead highlighted the grid during 00 while the
    #: song was in 03 (S3, 2026-09-05). Both lookups prefer it and fall back to
    #: pattern-only, which is what keeps a caret placed from the grid working.
    order_index: int | None = None

    #: Bumped by every request for something to sound, and by ``stop``. Carried
    #: to a render task through ``Done.tag`` and compared on completion, so a
    #: preview, a pattern or a sound effect that finishes rendering *after* the
    #: user pressed Stop is dropped instead of starting the mixer (S1,
    #: 2026-09-05). ``MuseState.audition_job``'s mechanism (M11); a counter
    #: rather than a key because the four things that can sound here are not
    #: all identified by a job id.
    play_request: int = 0

    #: Which nibble of a multi-digit column the *next* hex key fills: ``0`` is
    #: the high one, ``1`` the low. A fifth number rather than a field on the
    #: pane, for the reason the other four are here: a pane is rebuilt from
    #: scratch every frame and owns nothing that outlives one, while a
    #: half-finished entry is by definition the thing that spans frames.
    #:
    #: **Every function that moves the caret clears it**, and that is not
    #: tidiness: the second nibble of one cell landing in the next is a value
    #: the user never typed, in a cell they were not looking at, under an undo
    #: step they will not recognise.
    digit: int = 0
    #: ``history.head`` after the high nibble's own step, or -1. What lets
    #: the low nibble fold the two into one step -- see ``write_hex``.
    digit_head: int = -1

    #: Which octave a letter key types into, and how far the caret falls after.
    octave: int = DEFAULT_OCTAVE
    step: int = DEFAULT_STEP

    #: Whether the grid scrolls to follow playback. Off would make the playhead
    #: leave the screen within a bar; on takes the view away from a user who is
    #: editing bar 40 while bar 1 plays. Default on, because the first thing
    #: anybody does with Space is watch it.
    follow: bool = True

    #: Whether typing a note plays it. On, because a tracker that does not is
    #: one where writing a melody is type, press Space, wait for a re-render,
    #: listen, undo -- and off is a real preference on a machine whose audio is
    #: routed somewhere else. View state: it changes nothing about the song, so
    #: it is never written to a ``.wsng``.
    preview: bool = True

    #: How far the leftmost drawn channel is from the first, in channels. Kept
    #: on the state rather than derived per frame because a pane that scrolled
    #: to the caret would jump back the moment the caret was off screen for a
    #: different reason -- see ``sirens_patterns._first_channel``, which is what
    #: writes it.
    chan_scroll: int = 0

    #: Whether Play repeats the rendered song. View state like the mutes: the
    #: *document's* ``loop_order`` is what the exported WAV's ``smpl`` chunk
    #: says and is a property of the song, while this is how the person writing
    #: it wants to hear it in the next thirty seconds.
    loop_playback: bool = False

    #: The block selection's anchor, as ``(row, channel)``, or ``None``. The
    #: *other* corner is the caret -- which is what makes shift-arrow a
    #: selection with no second cursor to keep in step, and what makes clearing
    #: the selection a single assignment.
    anchor: tuple[int, int] | None = None

    #: The block clipboard: a contiguous ``int16`` ``(rows, channels, columns)``
    #: array, or ``None`` before the first copy. App-level rather than per tab,
    #: for the reason Inker's ``cel_clip`` is (``InkerState``): a bar copied in
    #: one song pastes into another, which is the only thing a clipboard does
    #: that a second order-list entry cannot. Nothing that moves the caret
    #: clears it; only the next copy or cut overwrites it.
    clip: Any = None

    #: Which instrument new notes are stamped with, by uid. ``None`` means the
    #: document has none yet, which a fresh song never is.
    instrument: int | None = None

    #: Which sound effect the effects pane has selected, by uid, or ``None``.
    #: Beside ``instrument`` rather than on the pane because it is the same kind
    #: of state for the same reason -- a pane is rebuilt from scratch every
    #: frame and owns nothing that outlives one -- and because the *grid* is the
    #: effect editor: selecting an effect moves the caret into its pattern, so
    #: the selection and the caret have to be clamped together.
    oneshot: int | None = None

    # --- an envelope gesture in flight ---------------------------------------
    #
    # Four fields rather than a drag object, and here rather than in the pane,
    # for the reason the caret is four numbers: the pane is redrawn from
    # scratch every frame and owns nothing that outlives one, while a drag is
    # by definition the thing that spans frames. Keeping it here is also what
    # lets ``sirens_mode`` open and close the gesture without imgui, which is
    # what makes "a drag is one undo step" a testable claim on a box with no
    # display.
    #
    #: Which sequence the gesture is editing (``"volume"``...), or ``""``.
    env_field: str = ""
    #: What it has hold of: ``"paint"``, ``"loop"`` or ``"release"``.
    env_grip: str = ""
    #: ``len(doc.history)`` when the button went down, so the whole run can be
    #: folded into one step when it comes back up. ``-1`` means no gesture.
    env_depth: int = -1
    #: The last step this drag painted. A pointer moves further than one column
    #: per frame, and without it a fast drag leaves a comb of untouched steps
    #: between the ones the mouse happened to be over on a frame boundary.
    env_step: int = -1

    @property
    def active(self) -> SongTab | None:
        for doc in self.docs:
            if doc.uid == self.active_uid:
                return doc
        return self.docs[-1] if self.docs else None

    @property
    def any_dirty(self) -> bool:
        return any(doc.dirty for doc in self.docs)

    def add(self, doc: SongTab) -> SongTab:
        self.docs.append(doc)
        self.active_uid = doc.uid
        self._reset_caret(doc)
        return doc

    def get(self, uid: str) -> SongTab | None:
        for doc in self.docs:
            if doc.uid == uid:
                return doc
        return None

    def close(self, uid: str) -> bool:
        doc = self.get(uid)
        if doc is None:
            return False
        index = self.docs.index(doc)
        self.docs.remove(doc)
        if self.active_uid == uid:
            self.active_uid = self.docs[min(index, len(self.docs) - 1)].uid if self.docs else ""
            self._reset_caret(self.active)
        return True

    def activate(self, uid: str) -> None:
        if uid != self.active_uid:
            # **The other song stops (S6, 2026-09-05).** There is one mixer
            # channel, so tab A went on sounding under tab B: the playhead
            # correctly declined to draw (the tag names A), but the transport
            # read the *global* ``playing()`` and so offered B a Stop button
            # that silenced A. The one import of a device in this module, and
            # deliberately local: everything else here is answerable with no
            # sound card, which is what lets the tests ask it.
            from . import sirens_audio

            sirens_audio.stop()
            self.play_request += 1
            for tab in self.docs:
                tab.sounding = None
            self.active_uid = uid
            # The caret names a row of the *previous* song's pattern, and a
            # selection anchored in it. Both are meaningless here.
            self._reset_caret(self.active)

    def cycle(self, step: int = 1) -> None:
        if len(self.docs) < 2:
            return
        current = self.active
        index = self.docs.index(current) if current in self.docs else 0
        self.activate(self.docs[(index + step) % len(self.docs)].uid)

    def find_path(self, path: Path) -> SongTab | None:
        """``docmodes.find_path``: the one case-folding body every mode shares."""
        return docmodes.find_path(self.docs, path)

    def _reset_caret(self, tab: SongTab | None) -> None:
        """Put the caret at the top of the tab's first pattern.

        Called wherever the *document under the caret* changes -- an add, a
        close that promoted a neighbour, a tab switch. A caret left pointing at
        another song's pattern uid is not merely stale: ``set_cell`` raises
        ``MISSING_PATTERN`` for it, so the next keystroke is a refusal about a
        pattern the user cannot see.
        """
        self.anchor = None
        self.row = self.channel = self.column = 0
        self.digit = 0
        self.oneshot = None
        self.order_index = None
        if tab is None:
            self.pattern = None
            self.instrument = None
            return
        doc = tab.doc
        self.pattern = doc.patterns[0].uid if doc.patterns else None
        self.instrument = doc.instruments[0].uid if doc.instruments else None

    def selection(self) -> tuple[int, int, int, int] | None:
        """The block as ``(row, channel, rows, channels)``, or ``None``.

        Normalised here rather than at four call sites: an anchor below the
        caret and an anchor above it are the same rectangle, and a transpose
        that only handled one of them would silently do nothing half the time.
        """
        if self.anchor is None:
            return None
        row0, chan0 = self.anchor
        row1, chan1 = self.row, self.channel
        row, chan = min(row0, row1), min(chan0, chan1)
        return (row, chan, abs(row1 - row0) + 1, abs(chan1 - chan0) + 1)


def ensure(ctx: Any) -> SirensState:
    """The mode's state, built on first use.

    Here rather than in ``sirens_mode`` because this and :func:`active` touch
    exactly one thing -- ``ctx.state.sirens`` -- which is this module's whole
    charter, and neither knows a task thread or a mixer exists. The
    ``packwright_state`` shape.
    """
    state = ctx.state.sirens
    if state is None:
        state = SirensState()
        ctx.state.sirens = state
    return state


def active(ctx: Any) -> SongTab | None:
    """The focused tab, or ``None``. Deliberately *not* through :func:`ensure`:
    asking which song is open must not create the state that says none is."""
    state = ctx.state.sirens
    return state.active if state is not None else None


# The same answer in the other document modes; Clay's is on ``stem`` on purpose.
title_for = docmodes.title_for
