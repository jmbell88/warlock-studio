"""The song: channels, patterns, an order, one-shots, samples and the history.

## A pattern is a numpy array, not a list of note objects

``cells`` is ``int16`` of shape ``(rows, channels, COLUMNS)``. Everything a
tracker does to a pattern is then a slice: transposing a selection is ``+= 12``
on one plane, clearing a block is a fill, copy and paste are a read and a write,
and an undo step is a pair of sub-arrays rather than a list of cell objects to
walk. A cell-object model would make every one of those a loop, and the loop
would be in the frame thread's path when the user holds a key down.

``int16`` rather than ``int8`` because the effect parameter column is a byte and
the note column has to hold :data:`~.notes.NOTE_RELEASE` at 121 as well as
:data:`~.notes.EMPTY` at ``-1``; and rather than ``int32`` because a pattern is
copied into the undo stack twice per edit and half the bytes is half the budget.

## The order list holds pattern uids

Not indices into :attr:`SongDoc.patterns`. Deleting pattern 3 must not silently
turn every ``3`` in the order into whatever is now third -- which is the classic
tracker data-loss bug, and it is an ``index`` away in either direction.

## An instrument id is per-document and bounded; every other uid is global

The instrument column of a cell is one plane of that ``int16`` array, so what it
holds has to fit in one. :func:`new_uid` is a *process*-global counter, which
means the 32,768th object minted in a session is a number ``set_cell`` cannot
store: numpy 2 raises an ``OverflowError`` from inside a mutator documented to
raise ``ValueError``, and numpy 1.x -- which this build still permits -- wraps
the value silently and writes a different instrument into the song.

So an instrument's id comes from a per-document space bounded by
:data:`MAX_INSTRUMENTS`, minted as the lowest free id (:func:`_free_instrument_id`).
Channels, patterns and one-shots keep the global uid: **none of them is ever
stored in a cell**, so none of them has the ceiling, and a bounded id would only
be a second numbering to keep in step. A bounded id is also what a tracker shows
the user anyway -- slot ``03`` is slot ``03``.

This does not weaken the uid rule; see :func:`_free_instrument_id` for why it
satisfies it exactly, and why reusing a freed id is safe under this history
model rather than merely unlikely.

## One-shots are patterns

A sound effect is a pattern with its own speed and tempo, rendered on the same
channels and exported to its own file. It needs no second document type, no
second synth path and no second undo vocabulary, and a user who has learned to
write a bassline has already learned to write a laser.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..undo import UndoStack
from . import edits as E
from . import instruments as inst
from . import notes

#: The five columns of a cell, in the order the grid draws them.
NOTE, INSTRUMENT, VOLUME, EFFECT, PARAM = range(5)
COLUMNS = 5

#: A row is a sixteenth note. Fixed rather than a document field: it is what
#: makes :attr:`SongDoc.tempo` mean BPM in the sense a musician means it, and a
#: song that redefined it would have a tempo marking that agreed with nothing.
ROWS_PER_BEAT = 4

DEFAULT_TEMPO = 150
DEFAULT_SPEED = 6
DEFAULT_ROWS = 64

MIN_TEMPO, MAX_TEMPO = 20, 400
MIN_SPEED, MAX_SPEED = 1, 31
MIN_ROWS, MAX_ROWS = 1, 256

MAX_CHANNELS = 32
MAX_PATTERNS = 256
MAX_ORDER = 256
MAX_INSTRUMENTS = 128
MAX_ONESHOTS = 64
MAX_SAMPLES = 64
MAX_TITLE_LEN = 64

#: Every refusal this module raises is a ``ValueError`` with one of these, so
#: the mode above can frame them into a toast and let anything else through to
#: the log -- ``packwright/document.py``'s ``MISSING_SOURCE`` rule.
MISSING_PATTERN = "that pattern is not in this song"
MISSING_INSTRUMENT = "that instrument is not in this song"
MISSING_CHANNEL = "that channel is not in this song"
MISSING_ONESHOT = "that sound effect is not in this song"
#: The other kind of refusal this module makes, and the only ceiling a user can
#: reach by working rather than by hand-editing a file. Named beside the
#: ``MISSING_*`` sentences because it is framed the same way: a 129th instrument
#: has to arrive at the mode as a ``ValueError`` it can turn into a toast, not
#: as whatever an exhausted id space would otherwise raise.
FULL_INSTRUMENTS = f"a song holds {MAX_INSTRUMENTS} instruments"

_next_uid = 0


def new_uid() -> int:
    global _next_uid
    _next_uid += 1
    return _next_uid


def reserve_uid(above: int) -> None:
    """Make sure :func:`new_uid` never returns ``above`` or lower again.

    Called by the reader once per document opened, with the largest uid the file
    contained. **Without it, opening a saved song and then adding a pattern is a
    silent corruption**: the counter starts at zero in a fresh process, the new
    pattern is handed a uid the file is already using, and the order list now
    points at two different patterns through one number.

    Bumping a global rather than renumbering on read, deliberately -- a
    renumbering would mean opening a file and saving it produced different
    bytes, which makes every ``.wsng`` in a repository undiffable.
    """
    global _next_uid
    _next_uid = max(_next_uid, int(above))


def _free_instrument_id(instruments: Any) -> int:
    """The lowest id in ``0..MAX_INSTRUMENTS-1`` no instrument here holds.

    **The uid rule survives this.** That rule exists so that reordering a list
    cannot retarget an edit -- an index would move under every insert, and the
    song would retune itself -- and a per-document id satisfies it exactly: it
    is still an identity the instrument carries, still what a cell stores, and
    still unrelated to where in the list the instrument is drawn. What it drops
    is the *process*-global part of :func:`new_uid`, which the instrument column
    cannot store (see this module's docstring) and which nothing here needed.

    **Reusing a freed id is safe under this history model**, and it is worth
    saying why rather than leaving it to be discovered. ``UndoStack.push``
    clears the redo branch, so a step that took a freed id has already
    discarded every step that could have referred to the previous holder. The
    only route back to a freed id is undoing *in order*: remove frees 3, add
    takes 3, undoing the add frees 3 again, and undoing the remove restores the
    original 3 to an id nothing else holds. A collision would need a redo across
    a branch ``push`` has already thrown away, which is not a reachable state.

    Lowest-free rather than highest-plus-one because a tracker's instrument
    numbers are slots the user reads and types: a song that has used and
    dropped a hundred instruments should still add the next one at ``00`` if
    ``00`` is empty, not at ``64``.
    """
    taken = {int(one.uid) for one in instruments}
    for candidate in range(MAX_INSTRUMENTS):
        if candidate not in taken:
            return candidate
    raise ValueError(FULL_INSTRUMENTS)


def empty_cells(rows: int, channels: int) -> np.ndarray:
    """A blank grid. :data:`~.notes.EMPTY` throughout, in every column."""
    return np.full((int(rows), int(channels), COLUMNS), notes.EMPTY, dtype=np.int16)


@dataclass
class Channel:
    """One column group of the grid, and one voice of the mix.

    ``kind`` fixes what can be played here: an instrument written for the noise
    channel on a pulse channel is a refusal rather than a surprising sound.

    ``pan`` is the one thing here that has no hardware ancestor and it earns its
    place: the NES was mono and two pulse channels playing a harmony line down
    the middle turns to mud, while a few tenths of separation makes the same
    arrangement legible. It defaults to centre for every channel, so a document
    nobody has touched sounds exactly as it would have.
    """

    uid: int
    name: str = ""
    kind: str = "pulse"
    pan: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in inst.KINDS:
            raise ValueError(f"{self.kind!r} is not one of {', '.join(inst.KINDS)}")
        self.pan = max(-1.0, min(1.0, float(self.pan)))
        self.name = self.name[:inst.MAX_NAME_LEN]


@dataclass
class Pattern:
    uid: int
    name: str = ""
    cells: np.ndarray = field(default_factory=lambda: empty_cells(DEFAULT_ROWS, 1))

    @property
    def rows(self) -> int:
        return int(self.cells.shape[0])

    @property
    def channels(self) -> int:
        return int(self.cells.shape[1])


@dataclass
class OneShot:
    """A named sound effect: a pattern, and the speed to play it at.

    Its own tempo and speed rather than the song's, because an effect is short
    and its timing has nothing to do with the music it plays over -- a coin
    pickup is forty milliseconds whatever the track is doing.
    """

    uid: int
    name: str = ""
    pattern: int = 0
    tempo: int = DEFAULT_TEMPO
    speed: int = DEFAULT_SPEED

    def __post_init__(self) -> None:
        self.name = self.name[:inst.MAX_NAME_LEN]
        self.tempo = max(MIN_TEMPO, min(MAX_TEMPO, int(self.tempo)))
        self.speed = max(MIN_SPEED, min(MAX_SPEED, int(self.speed)))


def default_channels() -> list[Channel]:
    """The five a new song starts with: the 2A03's, in its order.

    Not a hardware limit -- :func:`SongDoc.add_channel` will add a sixth pulse
    channel and the synth will play it -- but it is the arrangement every piece
    of music in this idiom was written for, and it is what a user opening the
    app expects to find. The two pulses are nudged apart; see :class:`Channel`.
    """
    return [
        Channel(uid=new_uid(), name="Pulse 1", kind="pulse", pan=-0.25),
        Channel(uid=new_uid(), name="Pulse 2", kind="pulse", pan=0.25),
        Channel(uid=new_uid(), name="Triangle", kind="triangle"),
        Channel(uid=new_uid(), name="Noise", kind="noise"),
        Channel(uid=new_uid(), name="Sample", kind="sample"),
    ]


class SongDoc:
    """A song, its history, and every mutation that pushes a step.

    The public methods validate, refuse a no-op, push one edit and apply it. The
    ``_apply``/``_attach``/``_detach`` half is what the edits call and is the
    only code that writes to the fields -- so "what can change" and "what is
    reversible" are the same list by construction.
    """

    def __init__(
        self,
        *,
        channels: list[Channel] | None = None,
        instruments: list[inst.Instrument] | None = None,
        patterns: list[Pattern] | None = None,
        order: list[int] | None = None,
        oneshots: list[OneShot] | None = None,
        samples: dict[str, np.ndarray] | None = None,
        title: str = "",
        author: str = "",
        tempo: int = DEFAULT_TEMPO,
        speed: int = DEFAULT_SPEED,
        loop_order: int = -1,
    ) -> None:
        self.title = title[:MAX_TITLE_LEN]
        self.author = author[:MAX_TITLE_LEN]
        self.tempo = max(MIN_TEMPO, min(MAX_TEMPO, int(tempo)))
        self.speed = max(MIN_SPEED, min(MAX_SPEED, int(speed)))
        #: Which order entry playback returns to at the end. ``-1`` plays once.
        self.loop_order = int(loop_order)
        self.channels = list(channels) if channels is not None else default_channels()
        self.instruments = list(instruments or [])
        self.patterns = list(patterns or [])
        self.order = list(order or [])
        self.oneshots = list(oneshots or [])
        self.samples = dict(samples or {})
        self.history = UndoStack()
        self.saved_head = 0

    # --- identity and state ---------------------------------------------------

    @property
    def dirty(self) -> bool:
        return self.history.head != self.saved_head

    def mark_saved(self, head: int | None = None) -> None:
        """``head`` is the value captured *before* the encode was handed to a
        task, so an edit made during the write still reads as unsaved."""
        self.saved_head = self.history.head if head is None else int(head)

    @property
    def tick_rate(self) -> float:
        """Engine ticks per second. The one place tempo becomes time.

        ``speed`` ticks per row, ``ROWS_PER_BEAT`` rows per beat, ``tempo``
        beats per minute. At the defaults -- 150 BPM, speed 6 -- this is exactly
        60 Hz, which is the rate the whole genre was written against; that it
        falls out of musical units rather than being declared is the point.
        """
        return self.speed * self.tempo * ROWS_PER_BEAT / 60.0

    # --- lookups --------------------------------------------------------------

    def pattern(self, uid: int) -> Pattern | None:
        for one in self.patterns:
            if one.uid == uid:
                return one
        return None

    def instrument(self, uid: int) -> inst.Instrument | None:
        for one in self.instruments:
            if one.uid == uid:
                return one
        return None

    def channel(self, uid: int) -> Channel | None:
        for one in self.channels:
            if one.uid == uid:
                return one
        return None

    def oneshot(self, uid: int) -> OneShot | None:
        for one in self.oneshots:
            if one.uid == uid:
                return one
        return None

    def _require(self, uid: int, finder: Any, message: str) -> Any:
        found = finder(uid)
        if found is None:
            raise ValueError(message)
        return found

    # --- cells ----------------------------------------------------------------

    def set_cells(
        self, pattern: int, row: int, channel: int, column: int, values: Any
    ) -> bool:
        """Write a block of cells. -> whether anything changed.

        The one door for every pattern write: a keystroke is a 1x1x1 block and a
        paste is a large one, and giving them separate paths would be two places
        for the clipping and the no-op check to be written.

        A block that runs off the end is **clipped, not refused**: pasting four
        bars near the bottom of a pattern should put in what fits, which is what
        every tracker does and what a user dragging a selection expects.
        """
        target = self._require(pattern, self.pattern, MISSING_PATTERN)
        block = np.ascontiguousarray(values, dtype=np.int16)
        if block.ndim != 3:
            raise ValueError("a cell block is (rows, channels, columns)")
        row, channel, column = int(row), int(channel), int(column)
        if row < 0 or channel < 0 or column < 0:
            raise ValueError("a cell block starts inside the pattern")
        rows = min(block.shape[0], target.rows - row)
        chans = min(block.shape[1], target.channels - channel)
        cols = min(block.shape[2], COLUMNS - column)
        if rows <= 0 or chans <= 0 or cols <= 0:
            return False
        block = block[:rows, :chans, :cols]
        before = target.cells[row : row + rows, channel : channel + chans, column : column + cols]
        if np.array_equal(before, block):
            # Retyping the note that is already there is a real user action and
            # is not a change. ``packwright/document.py`` states the rule once.
            return False
        self.history.push(
            E.CellsEdit(
                pattern=pattern,
                row=row,
                channel=channel,
                column=column,
                before=before,
                after=block,
            )
        )
        self._apply_cells(pattern, row, channel, column, block)
        return True

    def set_cell(self, pattern: int, row: int, channel: int, column: int, value: int) -> bool:
        """One cell. :meth:`set_cells` with the block written out."""
        return self.set_cells(
            pattern, row, channel, column, np.full((1, 1, 1), int(value), dtype=np.int16)
        )

    def clear_cells(self, pattern: int, row: int, channel: int, rows: int, chans: int) -> bool:
        """Blank a rectangle across every column."""
        return self.set_cells(
            pattern, row, channel, 0, np.full((rows, chans, COLUMNS), notes.EMPTY, dtype=np.int16)
        )

    def transpose(
        self, pattern: int, row: int, channel: int, rows: int, chans: int, by: int
    ) -> bool:
        """Shift the notes in a rectangle by ``by`` semitones.

        Cells that are empty or a command are left alone, and a note that would
        leave the playable range is left where it is rather than clamped to the
        edge -- clamping a chord's top voice turns it into a different chord,
        silently, and the note that did not move is visible.
        """
        target = self._require(pattern, self.pattern, MISSING_PATTERN)
        rows = min(int(rows), target.rows - int(row))
        chans = min(int(chans), target.channels - int(channel))
        if rows <= 0 or chans <= 0:
            return False
        block = target.cells[row : row + rows, channel : channel + chans, NOTE : NOTE + 1].copy()
        moved = block + int(by)
        playable = (block >= 0) & (block <= notes.MAX_NOTE)
        fits = playable & (moved >= 0) & (moved <= notes.MAX_NOTE)
        block = np.where(fits, moved, block)
        return self.set_cells(pattern, row, channel, NOTE, block)

    def _apply_cells(self, pattern: int, row: int, channel: int, column: int, block: Any) -> None:
        target = self.pattern(pattern)
        if target is None:
            return
        rows, chans, cols = block.shape
        target.cells[row : row + rows, channel : channel + chans, column : column + cols] = block

    # --- patterns -------------------------------------------------------------

    def add_pattern(self, rows: int = DEFAULT_ROWS, name: str = "") -> Pattern:
        if len(self.patterns) >= MAX_PATTERNS:
            raise ValueError(f"a song holds {MAX_PATTERNS} patterns")
        rows = max(MIN_ROWS, min(MAX_ROWS, int(rows)))
        pattern = Pattern(
            uid=new_uid(),
            name=name[: inst.MAX_NAME_LEN],
            cells=empty_cells(rows, len(self.channels)),
        )
        index = len(self.patterns)
        self.history.push(E.PatternAddEdit(pattern=pattern, index=index))
        self._attach_pattern(pattern, index)
        return pattern

    def remove_pattern(self, uid: int) -> bool:
        """Delete a pattern, and every order entry that named it.

        One step, for the reason ``PatternRemoveEdit`` states: a song that
        refers to a pattern which does not exist must not be reachable by
        pressing Ctrl+Z once.
        """
        pattern = self._require(uid, self.pattern, MISSING_PATTERN)
        index = self.patterns.index(pattern)
        after = tuple(one for one in self.order if one != uid)
        self.history.push(
            E.PatternRemoveEdit(
                pattern=pattern, index=index, order_before=tuple(self.order), order_after=after
            )
        )
        self._detach_pattern(uid)
        self._apply_order(after)
        return True

    def resize_pattern(self, uid: int, rows: int) -> bool:
        pattern = self._require(uid, self.pattern, MISSING_PATTERN)
        rows = max(MIN_ROWS, min(MAX_ROWS, int(rows)))
        if rows == pattern.rows:
            return False
        after = empty_cells(rows, pattern.channels)
        keep = min(rows, pattern.rows)
        after[:keep] = pattern.cells[:keep]
        self.history.push(E.PatternResizeEdit(pattern=uid, before=pattern.cells, after=after))
        self._apply_pattern_cells(uid, after)
        return True

    def _attach_pattern(self, pattern: Pattern, index: int) -> None:
        self.patterns.insert(index, pattern)

    def _detach_pattern(self, uid: int) -> None:
        self.patterns = [one for one in self.patterns if one.uid != uid]

    def _apply_pattern_cells(self, uid: int, cells: np.ndarray) -> None:
        pattern = self.pattern(uid)
        if pattern is not None:
            pattern.cells = np.ascontiguousarray(cells, dtype=np.int16).copy()

    # --- the order ------------------------------------------------------------

    def set_order(self, order: Any) -> bool:
        """Replace the whole order list. Every order change goes through here.

        Insert, remove, move and retarget are all one assignment away from each
        other over a list of at most :data:`MAX_ORDER` integers, and giving each
        an edit type of its own would be four classes that reverse to the same
        thing.
        """
        after = tuple(int(one) for one in order)[:MAX_ORDER]
        known = {one.uid for one in self.patterns}
        unknown = [one for one in after if one not in known]
        if unknown:
            raise ValueError(MISSING_PATTERN)
        if after == tuple(self.order):
            return False
        self.history.push(E.OrderEdit(before=tuple(self.order), after=after))
        self._apply_order(after)
        return True

    def _apply_order(self, order: tuple[int, ...]) -> None:
        self.order = list(order)
        if self.loop_order >= len(self.order):
            # Not an edit of its own: a loop point past the end of the order is
            # not a state the user chose, it is arithmetic left over from one
            # they did. It rides on whichever step moved the order.
            self.loop_order = -1

    # --- instruments ----------------------------------------------------------

    def add_instrument(self, kind: str = "pulse", name: str = "") -> inst.Instrument:
        if len(self.instruments) >= MAX_INSTRUMENTS:
            raise ValueError(FULL_INSTRUMENTS)
        instrument = inst.default(_free_instrument_id(self.instruments), kind=kind, name=name)
        index = len(self.instruments)
        self.history.push(E.InstrumentAddEdit(instrument=instrument, index=index))
        self._attach_instrument(instrument, index)
        return instrument

    def remove_instrument(self, uid: int) -> bool:
        """Delete an instrument. **The cells that named it are left alone.**

        A pattern cell holding a uid nothing answers to plays silently and can
        be put back by an undo; rewriting every cell in the song to clear the
        reference cannot, and would be a far larger edit than the one the user
        asked for. The instrument list is the place that says which uids are
        real, and the grid draws an unknown one as unknown.
        """
        instrument = self._require(uid, self.instrument, MISSING_INSTRUMENT)
        index = self.instruments.index(instrument)
        self.history.push(E.InstrumentRemoveEdit(instrument=instrument, index=index))
        self._detach_instrument(uid)
        return True

    def update_instrument(self, uid: int, **values: Any) -> bool:
        instrument = self._require(uid, self.instrument, MISSING_INSTRUMENT)
        after = replace(instrument, **values)
        if after == instrument:
            return False
        self.history.push(E.InstrumentEdit(uid=uid, before=instrument, after=after))
        self._apply_instrument(uid, after)
        return True

    def _attach_instrument(self, instrument: inst.Instrument, index: int) -> None:
        self.instruments.insert(index, instrument)

    def _detach_instrument(self, uid: int) -> None:
        self.instruments = [one for one in self.instruments if one.uid != uid]

    def _apply_instrument(self, uid: int, value: inst.Instrument) -> None:
        for i, one in enumerate(self.instruments):
            if one.uid == uid:
                self.instruments[i] = replace(value)
                return

    # --- channels -------------------------------------------------------------

    def add_channel(self, kind: str = "pulse", name: str = "", pan: float = 0.0) -> Channel:
        """Add a voice, widening every pattern in the song by one plane."""
        if len(self.channels) >= MAX_CHANNELS:
            raise ValueError(f"a song holds {MAX_CHANNELS} channels")
        channel = Channel(uid=new_uid(), name=name or kind.capitalize(), kind=kind, pan=pan)
        after = (*self.channels, channel)
        self._push_channels(after, insert_at=len(self.channels), removed=None)
        return channel

    def remove_channel(self, uid: int) -> bool:
        """Remove a voice, and the plane of notes that was written on it."""
        channel = self._require(uid, self.channel, MISSING_CHANNEL)
        if len(self.channels) <= 1:
            raise ValueError("a song needs at least one channel")
        index = self.channels.index(channel)
        after = tuple(one for one in self.channels if one.uid != uid)
        self._push_channels(after, insert_at=None, removed=index)
        return True

    def update_channel(self, uid: int, **values: Any) -> bool:
        """Rename, repan or re-kind one channel. The patterns do not move."""
        channel = self._require(uid, self.channel, MISSING_CHANNEL)
        after = replace(channel, **values)
        if after == channel:
            return False
        channels = tuple(after if one.uid == uid else one for one in self.channels)
        cells = {one.uid: one.cells for one in self.patterns}
        self.history.push(
            E.ChannelsEdit(
                before=tuple(self.channels),
                after=channels,
                cells_before=cells,
                cells_after=cells,
            )
        )
        self._apply_channels(channels, cells)
        return True

    def _push_channels(
        self, after: tuple[Channel, ...], *, insert_at: int | None, removed: int | None
    ) -> None:
        """The shared half of add and remove: rebuild every pattern's grid."""
        cells_before = {one.uid: one.cells for one in self.patterns}
        cells_after: dict[int, np.ndarray] = {}
        for one in self.patterns:
            if insert_at is not None:
                grid = np.insert(one.cells, insert_at, notes.EMPTY, axis=1)
            else:
                grid = np.delete(one.cells, removed, axis=1)
            cells_after[one.uid] = grid
        self.history.push(
            E.ChannelsEdit(
                before=tuple(self.channels),
                after=after,
                cells_before=cells_before,
                cells_after=cells_after,
            )
        )
        self._apply_channels(after, cells_after)

    def _apply_channels(self, channels: tuple[Channel, ...], cells: dict[int, np.ndarray]) -> None:
        self.channels = [replace(one) for one in channels]
        for pattern in self.patterns:
            grid = cells.get(pattern.uid)
            if grid is not None:
                pattern.cells = np.ascontiguousarray(grid, dtype=np.int16).copy()

    # --- one-shots ------------------------------------------------------------

    def add_oneshot(self, name: str = "", rows: int = 8) -> OneShot:
        """A named effect, with a pattern of its own made for it.

        The pattern is created here rather than by the caller because a one-shot
        that points at a pattern shared with the song is an effect whose export
        changes when the music is edited.
        """
        if len(self.oneshots) >= MAX_ONESHOTS:
            raise ValueError(f"a song holds {MAX_ONESHOTS} sound effects")
        depth = self.history.mark()
        pattern = self.add_pattern(rows=rows, name=name or "Effect")
        oneshot = OneShot(
            uid=new_uid(),
            name=name or f"effect{len(self.oneshots) + 1}",
            pattern=pattern.uid,
            tempo=self.tempo,
            speed=self.speed,
        )
        index = len(self.oneshots)
        self.history.push(E.OneShotAddEdit(oneshot=oneshot, index=index))
        self._attach_oneshot(oneshot, index)
        # The pattern and the effect that names it are one gesture: undoing the
        # effect must not leave its pattern behind in the pattern list.
        self.history.collapse_since(depth)
        return oneshot

    def remove_oneshot(self, uid: int) -> bool:
        oneshot = self._require(uid, self.oneshot, MISSING_ONESHOT)
        index = self.oneshots.index(oneshot)
        self.history.push(E.OneShotRemoveEdit(oneshot=oneshot, index=index))
        self._detach_oneshot(uid)
        return True

    def update_oneshot(self, uid: int, **values: Any) -> bool:
        oneshot = self._require(uid, self.oneshot, MISSING_ONESHOT)
        after = replace(oneshot, **values)
        if after == oneshot:
            return False
        self.history.push(E.OneShotEdit(uid=uid, before=oneshot, after=after))
        self._apply_oneshot(uid, after)
        return True

    def _attach_oneshot(self, oneshot: OneShot, index: int) -> None:
        self.oneshots.insert(index, oneshot)

    def _detach_oneshot(self, uid: int) -> None:
        self.oneshots = [one for one in self.oneshots if one.uid != uid]

    def _apply_oneshot(self, uid: int, value: OneShot) -> None:
        for i, one in enumerate(self.oneshots):
            if one.uid == uid:
                self.oneshots[i] = replace(value)
                return

    # --- samples --------------------------------------------------------------

    def set_sample(self, key: str, pcm: Any | None) -> bool:
        """Add, replace or remove one entry of the sample table.

        ``None`` removes. One method because they are one step; see
        :class:`~.edits.SampleEdit`.
        """
        key = str(key)[:inst.MAX_NAME_LEN]
        before = self.samples.get(key)
        after = None if pcm is None else np.ascontiguousarray(pcm, dtype=np.float32).ravel()
        if before is None and after is None:
            return False
        if before is not None and after is not None and np.array_equal(before, after):
            return False
        if after is not None and before is None and len(self.samples) >= MAX_SAMPLES:
            raise ValueError(f"a song holds {MAX_SAMPLES} samples")
        self.history.push(E.SampleEdit(key=key, before=before, after=after))
        self._apply_sample(key, after)
        return True

    def _apply_sample(self, key: str, pcm: np.ndarray | None) -> None:
        if pcm is None:
            self.samples.pop(key, None)
        else:
            self.samples[key] = np.ascontiguousarray(pcm, dtype=np.float32)

    # --- the song's scalars ---------------------------------------------------

    def set_song(self, **values: Any) -> bool:
        """Title, author, tempo, speed or loop point. Only the keys that moved."""
        fields = {"title", "author", "tempo", "speed", "loop_order"}
        unknown = set(values) - fields
        if unknown:
            raise ValueError(f"a song has no {', '.join(sorted(unknown))}")
        after: dict[str, Any] = {}
        for key, value in values.items():
            if key in ("title", "author"):
                value = str(value)[:MAX_TITLE_LEN]
            elif key == "tempo":
                value = max(MIN_TEMPO, min(MAX_TEMPO, int(value)))
            elif key == "speed":
                value = max(MIN_SPEED, min(MAX_SPEED, int(value)))
            else:
                value = int(value)
                if value >= len(self.order):
                    value = -1
            if value != getattr(self, key):
                after[key] = value
        if not after:
            return False
        before = {key: getattr(self, key) for key in after}
        self.history.push(E.SongEdit(before=before, after=after))
        self._apply_song(after)
        return True

    def _apply_song(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            setattr(self, key, value)

    # --- history --------------------------------------------------------------

    def undo(self) -> bool:
        return self.history.undo(self)

    def redo(self) -> bool:
        return self.history.redo(self)


def new_song() -> SongDoc:
    """A song that plays the moment somebody types a note into it.

    One pattern, one instrument per channel kind, and an order that points at
    the pattern -- because a document with an empty order is one where Space
    does nothing and there is no way to find out why. Built by construction
    rather than through the mutators: a brand-new document is not one edit deep,
    it is unmodified, and its history has to be empty for ``dirty`` to be false.
    """
    channels = default_channels()
    pattern = Pattern(uid=new_uid(), name="", cells=empty_cells(DEFAULT_ROWS, len(channels)))
    seen: list[str] = []
    instruments: list[inst.Instrument] = []
    for channel in channels:
        if channel.kind not in seen:
            seen.append(channel.kind)
            # The bounded per-document id, not ``new_uid`` -- see this module's
            # docstring. Built through the same helper ``add_instrument`` uses,
            # so a new song's slots are numbered by the one rule.
            instruments.append(
                inst.default(_free_instrument_id(instruments), kind=channel.kind)
            )
    return SongDoc(
        channels=channels,
        instruments=instruments,
        patterns=[pattern],
        order=[pattern.uid],
    )
