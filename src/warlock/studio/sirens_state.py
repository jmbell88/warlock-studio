"""Multi-document state for Sirens, without imgui and without a sound device.

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

**``render_generation`` is what playback keys on.** A re-render replaces the
buffer wholesale, and "should I hand this to the mixer again" gated on a hash
or on the document's head either re-uploads a 50 MB buffer every frame or
misses a change. A counter bumped in exactly one place is the only version of
this that is both.
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
    #: Where in the rendered buffer the sound now playing started, in samples.
    #: Non-zero after "play from here", so the playhead bisects against the
    #: song rather than against the slice that was handed to the mixer.
    play_offset: int = 0

    #: One ``(sample offset, order index, pattern uid, row)`` per row as the
    #: renderer actually played it, ascending in the offset. What the playhead
    #: bisects; see :func:`~.sirens.synth.render_marked`.
    marks: tuple[tuple[int, int, int, int], ...] = ()
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

        A bisect of :attr:`marks`, so the answer is what the renderer did
        rather than an estimate: every order entry, every pattern length, every
        ``Fxx`` and every jump are already in the offsets. ``None`` before the
        first row and after the last -- the tail of a song is its instruments
        decaying, and highlighting the final row through it says the song is
        still on it.
        """
        marks = self.marks
        if not marks:
            return None
        offset = int(float(seconds) * int(rate)) + int(self.play_offset)
        if offset < marks[0][0]:
            return None
        index = bisect.bisect_right(marks, (offset, _LAST, _LAST, _LAST)) - 1
        if index < 0:
            return None
        _at, order_index, pattern, row = marks[index]
        return order_index, pattern, row

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

    #: Channel uids the mix is playing without, and the one channel it is
    #: playing *alone* (``-1`` for none). **View state, not the song**: a mute
    #: is how a person listens to what they are writing, and a ``.wsng`` that
    #: remembered one would hand somebody else a song with a missing part. It
    #: reaches the mix through the render (``synth.render_only``), so toggling
    #: one re-renders exactly as an edit does.
    muted: set[int] = field(default_factory=set)
    solo: int = -1

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
        for doc in self.docs:
            if doc.path is not None and doc.path == path:
                return doc
        return None

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
        if tab is None:
            self.pattern = None
            self.instrument = None
            return
        doc = tab.doc
        self.pattern = doc.patterns[0].uid if doc.patterns else None
        self.instrument = doc.instruments[0].uid if doc.instruments else None

    def selection(self, tab: SongTab | None = None) -> tuple[int, int, int, int] | None:
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
