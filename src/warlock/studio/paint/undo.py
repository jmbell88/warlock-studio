"""History as typed edits, not as whole-image snapshots.

The old model copied the entire image before every operation. That was honest
-- a snapshot cannot be wrong about what it restores -- and it stopped scaling
the moment documents grew layers: ten layers at 2048x2048 is 160 MiB a step, so
the 192 MiB budget bought exactly one undo.

The replacement keeps the "cannot be wrong" property where it matters by
storing *exact pixels*, just only the ones that changed: a stroke's edit is the
before/after crops of its dirty rectangle, typically a megabyte or two, which
is a hundred-odd steps inside the same budget. Structural changes (add, remove,
reorder, rename) carry no pixels at all. Only whole-canvas geometry -- rotate,
scale, crop -- falls back to copying, and it redoes by replaying rather than by
storing a second copy.

Every edit addresses its layer by ``uid``, never by index, so an undo issued
after a reorder still lands on the layer the edit was made to.
"""

from __future__ import annotations

import itertools
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# The budget is bytes, because the cost of a step now varies by three orders of
# magnitude between a dab and a crop. The depth bounds are the other half: a
# huge document must still get a few levels, and a document of dabs must not
# grow an unbounded stack of them.
UNDO_BYTES = 192 * 1024 * 1024
UNDO_MAX_DEPTH = 64
UNDO_MIN_DEPTH = 8


# Serial numbers handed out at push time. They exist so a caller can ask "is
# the document where it was when I saved it" -- ``rev`` cannot answer that,
# because it counts *changes* and an undo is a change, so undoing back to the
# saved pixels would still read as unsaved.
_serials = itertools.count(1)


class Edit:
    """One reversible step. ``cost`` is what the budget is spent on."""

    cost: int = 0
    serial: int = 0

    def undo(self, doc: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def redo(self, doc: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class PatchEdit(Edit):
    """Exact pixels of one rectangle of one layer, before and after.

    This is the shape almost every edit takes: strokes, fills, gradients,
    shapes, lifting and committing a floating buffer, transforming a region.
    """

    layer_uid: int
    rect: tuple[int, int, int, int]
    before: np.ndarray
    after: np.ndarray

    def __post_init__(self) -> None:
        self.cost = int(self.before.nbytes + self.after.nbytes)

    def _put(self, doc: Any, pixels: np.ndarray) -> None:
        x0, y0, x1, y1 = self.rect
        layer = doc.stack.by_uid(self.layer_uid)
        layer.pixels[y0:y1, x0:x1] = pixels
        doc.invalidate(self.rect)

    def undo(self, doc: Any) -> None:
        self._put(doc, self.before)

    def redo(self, doc: Any) -> None:
        self._put(doc, self.after)


@dataclass
class LayerAddEdit(Edit):
    """The layer object itself is held, not a copy of it: re-inserting the same
    object is what keeps its uid, and keeping its uid is what lets a patch made
    before the undo still apply after the redo."""

    index: int
    layer: Any

    def undo(self, doc: Any) -> None:
        doc.stack.remove(doc.stack.index_of(self.layer.uid))
        doc.invalidate_all()

    def redo(self, doc: Any) -> None:
        doc.stack.insert(self.index, self.layer)
        doc.invalidate_all()


@dataclass
class LayerRemoveEdit(Edit):
    index: int
    layer: Any

    def __post_init__(self) -> None:
        self.cost = int(self.layer.pixels.nbytes)

    def undo(self, doc: Any) -> None:
        doc.stack.insert(self.index, self.layer)
        doc.invalidate_all()

    def redo(self, doc: Any) -> None:
        doc.stack.remove(doc.stack.index_of(self.layer.uid))
        doc.invalidate_all()


@dataclass
class LayerMoveEdit(Edit):
    layer_uid: int
    index: int
    to: int

    def undo(self, doc: Any) -> None:
        doc.stack.move(doc.stack.index_of(self.layer_uid), self.index)
        doc.invalidate_all()

    def redo(self, doc: Any) -> None:
        doc.stack.move(doc.stack.index_of(self.layer_uid), self.to)
        doc.invalidate_all()


@dataclass
class LayerPropsEdit(Edit):
    """Name, opacity, visibility, blend mode -- anything that is not pixels."""

    layer_uid: int
    before: dict
    after: dict

    def _apply(self, doc: Any, props: dict) -> None:
        layer = doc.stack.by_uid(self.layer_uid)
        for key, value in props.items():
            setattr(layer, key, value)
        doc.invalidate_all()

    def undo(self, doc: Any) -> None:
        self._apply(doc, self.before)

    def redo(self, doc: Any) -> None:
        self._apply(doc, self.after)


def _pack(mask: np.ndarray | None) -> tuple[bytes, tuple[int, int]] | None:
    if mask is None:
        return None
    return zlib.compress(mask.tobytes(), 1), mask.shape


def _unpack(blob: tuple[bytes, tuple[int, int]] | None) -> np.ndarray | None:
    if blob is None:
        return None
    data, shape = blob
    return np.frombuffer(zlib.decompress(data), dtype=np.uint8).reshape(shape).copy()


@dataclass
class SelectionEdit(Edit):
    """A selection change. Masks compress to almost nothing -- they are mostly
    runs of 0 and 255 -- which is the only reason selection changes can afford
    to be undoable at all."""

    before: Any
    after: Any

    def __post_init__(self) -> None:
        self.before = _pack(self.before)
        self.after = _pack(self.after)
        self.cost = sum(len(b[0]) for b in (self.before, self.after) if b)

    def undo(self, doc: Any) -> None:
        doc.set_selection_mask(_unpack(self.before))

    def redo(self, doc: Any) -> None:
        doc.set_selection_mask(_unpack(self.after))


@dataclass
class ReplayEdit(Edit):
    """A whole-canvas operation: flip, rotate, scale, crop, canvas resize.

    Undo restores copied layers (there is no smaller truthful answer when the
    canvas size itself changed); redo *replays* the operation rather than
    holding a second full copy, which halves the cost of the one edit type that
    is expensive. Replay is safe here and nowhere else because these ops are
    pure functions of the document -- no accumulation, no randomness.
    """

    snapshot: list[Any]
    size: tuple[int, int]
    active: int
    replay: Callable[[Any], None]
    selection: Any = None

    def __post_init__(self) -> None:
        self.selection = _pack(self.selection)
        self.cost = sum(int(layer.pixels.nbytes) for layer in self.snapshot)

    def undo(self, doc: Any) -> None:
        doc.restore_snapshot(self.snapshot, self.size, self.active)
        doc.set_selection_mask(_unpack(self.selection))

    def redo(self, doc: Any) -> None:
        self.replay(doc)


@dataclass
class CompoundEdit(Edit):
    """Several edits that must undo together -- committing a floating buffer
    writes pixels *and* drops the selection, and splitting that across two
    Ctrl+Z presses shows the user a state that never existed."""

    edits: list[Edit] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cost = sum(e.cost for e in self.edits)

    def undo(self, doc: Any) -> None:
        for edit in reversed(self.edits):
            edit.undo(doc)

    def redo(self, doc: Any) -> None:
        for edit in self.edits:
            edit.redo(doc)


class UndoStack:
    """Bounded by bytes first, then by a depth floor and a depth ceiling."""

    def __init__(self, budget: int = UNDO_BYTES) -> None:
        self.budget = budget
        self._done: list[Edit] = []
        self._undone: list[Edit] = []

    def __len__(self) -> int:
        return len(self._done)

    @property
    def can_undo(self) -> bool:
        return bool(self._done)

    @property
    def can_redo(self) -> bool:
        return bool(self._undone)

    @property
    def bytes(self) -> int:
        return sum(e.cost for e in self._done) + sum(e.cost for e in self._undone)

    @property
    def head(self) -> int:
        """Identifies the current position in history.

        A save records this and compares against it later, which is what makes
        "dirty" a comparison rather than a latching flag: undo back to the
        saved step and the head is that step's serial again. Serials are
        per-edit, so a *new* branch of the same length is a different head --
        which a plain depth count would miss.
        """
        return self._done[-1].serial if self._done else 0

    def push(self, edit: Edit) -> None:
        edit.serial = next(_serials)
        self._done.append(edit)
        # No history tree: a redo onto a branch the user did not take is worse
        # than no redo at all.
        self._undone.clear()
        self._evict()

    def _evict(self) -> None:
        while len(self._done) > UNDO_MAX_DEPTH:
            self._done.pop(0)
        while len(self._done) > UNDO_MIN_DEPTH and self.bytes > self.budget:
            self._done.pop(0)

    def undo(self, doc: Any) -> bool:
        if not self._done:
            return False
        edit = self._done.pop()
        edit.undo(doc)
        self._undone.append(edit)
        return True

    def redo(self, doc: Any) -> bool:
        if not self._undone:
            return False
        edit = self._undone.pop()
        edit.redo(doc)
        self._done.append(edit)
        return True

    def drop(self) -> None:
        """Discard the newest step without reversing it -- for an operation
        that already undid itself by hand, where leaving the step on the stack
        would make the next Ctrl+Z appear to do nothing."""
        if self._done:
            self._done.pop()

    def clear(self) -> None:
        self._done.clear()
        self._undone.clear()
