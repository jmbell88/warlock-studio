"""The reversible steps over a song, one class per kind of change.

Every one of them obeys the two rules ``studio/undo.py`` states for all its
users, and neither is restated at each class below.

**A step addresses its subject by uid.** A pattern, an instrument, a channel and
a one-shot are all found by uid and never by list position, so inserting an
instrument at the top of the list cannot retarget an undo of an edit made to the
one that used to be there.

**A step owns its data.** Every array stored here is copied at construction.
A slice of a pattern is a *view* -- it reports its own few hundred bytes to the
undo budget while keeping the whole pattern alive, so a stack of a thousand
one-cell edits could pin a hundred megabytes the budget never sees. ``.copy()``
on the way in is what makes ``cost`` honest.

Separate classes rather than one generic before/after step because
``UndoStack.history`` derives a step's label from its class name: fifteen
classes give a history panel that reads "cells", "pattern add", "order" -- and
one generic class gives a panel that reads "change" fifteen times.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..undo import Edit


def _copy(array: np.ndarray) -> np.ndarray:
    """A private ``int16`` copy, and ``.copy()`` is not decoration here.

    ``np.ascontiguousarray`` alone hands back the *same array* when it is
    already contiguous and already ``int16`` -- which a one-cell slice of a
    pattern is. The step would then hold a view of the live document, so
    ``before`` changes to match ``after`` the instant the edit is applied and
    the undo restores the value it was supposed to reverse. It also makes
    ``cost`` a lie: the view reports ten bytes to the undo budget while pinning
    the whole pattern.
    """
    return np.ascontiguousarray(array, dtype=np.int16).copy()


@dataclass
class CellsEdit(Edit):
    """A rectangle of pattern cells replaced.

    One class for a single keystroke and for a block paste, because they are the
    same operation at two sizes and a separate one-cell type would be a second
    place for the addressing to be written. A single cell is a 1x1x1 block, and
    the undo budget already scales with what a step actually holds.
    """

    pattern: int
    row: int
    channel: int
    column: int
    before: np.ndarray
    after: np.ndarray

    def __post_init__(self) -> None:
        self.before = _copy(self.before)
        self.after = _copy(self.after)
        self.cost = self.before.nbytes + self.after.nbytes

    def undo(self, doc: Any) -> None:
        doc._apply_cells(self.pattern, self.row, self.channel, self.column, self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_cells(self.pattern, self.row, self.channel, self.column, self.after)


@dataclass
class PatternResizeEdit(Edit):
    """A pattern's row count changed. Holds both grids, because shortening one
    throws rows away and there is nowhere else for them to be kept."""

    pattern: int
    before: np.ndarray
    after: np.ndarray

    def __post_init__(self) -> None:
        self.before = _copy(self.before)
        self.after = _copy(self.after)
        self.cost = self.before.nbytes + self.after.nbytes

    def undo(self, doc: Any) -> None:
        doc._apply_pattern_cells(self.pattern, self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_pattern_cells(self.pattern, self.after)


@dataclass
class PatternAddEdit(Edit):
    pattern: Any
    index: int

    def __post_init__(self) -> None:
        self.cost = int(self.pattern.cells.nbytes)

    def undo(self, doc: Any) -> None:
        doc._detach_pattern(self.pattern.uid)

    def redo(self, doc: Any) -> None:
        doc._attach_pattern(self.pattern, self.index)


@dataclass
class PatternRemoveEdit(Edit):
    """A pattern deleted, and the order list as it was before.

    The order is carried here rather than pushed as a second step: removing a
    pattern that the order refers to has to fix the order in the same gesture,
    and splitting that across two Ctrl+Z presses shows a song that refers to a
    pattern which does not exist.
    """

    pattern: Any
    index: int
    order_before: tuple[int, ...]
    order_after: tuple[int, ...]

    def __post_init__(self) -> None:
        self.cost = int(self.pattern.cells.nbytes)

    def undo(self, doc: Any) -> None:
        doc._attach_pattern(self.pattern, self.index)
        doc._apply_order(self.order_before)

    def redo(self, doc: Any) -> None:
        doc._detach_pattern(self.pattern.uid)
        doc._apply_order(self.order_after)


@dataclass
class OrderEdit(Edit):
    before: tuple[int, ...]
    after: tuple[int, ...]

    def undo(self, doc: Any) -> None:
        doc._apply_order(self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_order(self.after)


@dataclass
class InstrumentAddEdit(Edit):
    instrument: Any
    index: int

    def undo(self, doc: Any) -> None:
        doc._detach_instrument(self.instrument.uid)

    def redo(self, doc: Any) -> None:
        doc._attach_instrument(self.instrument, self.index)


@dataclass
class InstrumentRemoveEdit(Edit):
    instrument: Any
    index: int

    def undo(self, doc: Any) -> None:
        doc._attach_instrument(self.instrument, self.index)

    def redo(self, doc: Any) -> None:
        doc._detach_instrument(self.instrument.uid)


@dataclass
class InstrumentEdit(Edit):
    """An instrument's fields changed. Whole values, not a field name and a
    pair: an instrument is a name, a kind and four frozen sequences, so a
    per-field edit type would buy nothing and cost four more classes."""

    uid: int
    before: Any
    after: Any

    def __post_init__(self) -> None:
        self.before = replace(self.before)
        self.after = replace(self.after)

    def undo(self, doc: Any) -> None:
        doc._apply_instrument(self.uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_instrument(self.uid, self.after)


@dataclass
class ChannelsEdit(Edit):
    """The channel list changed, and with it the width of every pattern.

    **The expensive step, and deliberately not decomposed.** Adding a channel
    inserts a plane into every pattern in the document and removing one deletes
    a plane that had notes in it. Both have to undo as one gesture, and the only
    honest way to reverse a delete is to have kept what it deleted -- so this
    holds every pattern's grid, twice, and reports all of it to the undo budget.
    A song of two hundred patterns is a few megabytes here, which is what the
    byte budget in ``studio/undo.py`` exists to arbitrate.
    """

    before: tuple[Any, ...]
    after: tuple[Any, ...]
    cells_before: dict[int, np.ndarray] = field(default_factory=dict)
    cells_after: dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cells_before = {k: _copy(v) for k, v in self.cells_before.items()}
        self.cells_after = {k: _copy(v) for k, v in self.cells_after.items()}
        self.cost = sum(v.nbytes for v in self.cells_before.values()) + sum(
            v.nbytes for v in self.cells_after.values()
        )

    def undo(self, doc: Any) -> None:
        doc._apply_channels(self.before, self.cells_before)

    def redo(self, doc: Any) -> None:
        doc._apply_channels(self.after, self.cells_after)


@dataclass
class SongEdit(Edit):
    """The song's scalars -- title, author, tempo, speed, loop point.

    A dict of only the keys that moved, so undoing a tempo change cannot also
    put back a title the user edited afterwards.
    """

    before: dict[str, Any]
    after: dict[str, Any]

    def undo(self, doc: Any) -> None:
        doc._apply_song(self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_song(self.after)


@dataclass
class OneShotAddEdit(Edit):
    oneshot: Any
    index: int

    def undo(self, doc: Any) -> None:
        doc._detach_oneshot(self.oneshot.uid)

    def redo(self, doc: Any) -> None:
        doc._attach_oneshot(self.oneshot, self.index)


@dataclass
class OneShotRemoveEdit(Edit):
    oneshot: Any
    index: int

    def undo(self, doc: Any) -> None:
        doc._attach_oneshot(self.oneshot, self.index)

    def redo(self, doc: Any) -> None:
        doc._detach_oneshot(self.oneshot.uid)


@dataclass
class OneShotEdit(Edit):
    uid: int
    before: Any
    after: Any

    def __post_init__(self) -> None:
        self.before = replace(self.before)
        self.after = replace(self.after)

    def undo(self, doc: Any) -> None:
        doc._apply_oneshot(self.uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_oneshot(self.uid, self.after)


@dataclass
class SampleEdit(Edit):
    """One entry of the sample table added, replaced or removed.

    ``before``/``after`` are ``None`` for absent, which makes add, replace and
    remove one class: they are the same step with a different pair of ends, and
    three classes would be three places to get the copy rule wrong.
    """

    key: str
    before: np.ndarray | None
    after: np.ndarray | None

    def __post_init__(self) -> None:
        # ``.copy()`` for ``_copy``'s reason: the caller's array may be the one
        # in the document, and a step that shares it reverses to nothing.
        if self.before is not None:
            self.before = np.ascontiguousarray(self.before, dtype=np.float32).copy()
        if self.after is not None:
            self.after = np.ascontiguousarray(self.after, dtype=np.float32).copy()
        self.cost = (0 if self.before is None else self.before.nbytes) + (
            0 if self.after is None else self.after.nbytes
        )

    def undo(self, doc: Any) -> None:
        doc._apply_sample(self.key, self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_sample(self.key, self.after)
