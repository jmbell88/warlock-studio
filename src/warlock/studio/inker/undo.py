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

The engine itself -- ``Edit``, ``CompoundEdit``, ``UndoStack``, the serial
counter and the byte budget -- has no opinion about pixels and now lives in
``studio/undo.py``, so Clay can have the same history with its own edit
types without depending on the raster editor. It is re-exported here unchanged:
every module in this package imports those names from this one, and renaming
those imports as part of a move would have made "did the move change
behaviour?" unanswerable from the diff. What stays below is everything that is
genuinely about pixels or layers.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..undo import (
    UNDO_BYTES,
    UNDO_MAX_DEPTH,
    UNDO_MIN_DEPTH,
    CompoundEdit,
    Edit,
    UndoStack,
    _serials,
)

# This list exists to declare the re-export above as intentional -- without it
# every imported name is an unused import (ruff F401) -- so it names the seven
# re-exported names plus the public types defined below. ``_serials`` is in it
# for that reason alone and not because it is public; ``_pack``/``_unpack`` are
# defined here rather than imported, so they need no such declaration and stay
# out, as private helpers ordinarily would.
__all__ = [
    "UNDO_BYTES",
    "UNDO_MAX_DEPTH",
    "UNDO_MIN_DEPTH",
    "CompoundEdit",
    "Edit",
    "LayerAddEdit",
    "LayerMoveEdit",
    "LayerPropsEdit",
    "LayerRemoveEdit",
    "PatchEdit",
    "ReplayEdit",
    "SelectionEdit",
    "UndoStack",
    "_serials",
]


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
        # Own the pixels rather than viewing someone else's. A slice of a
        # full-canvas array reports only its own ``nbytes`` while keeping the
        # whole base alive, so a stroke that costs four kilobytes by the
        # budget's reckoning can pin sixteen megabytes -- invisibly, because
        # eviction is driven by exactly that number.
        if self.before.base is not None:
            self.before = self.before.copy()
        if self.after.base is not None:
            self.after = self.after.copy()
        self.cost = int(self.before.nbytes + self.after.nbytes)

    def _put(self, doc: Any, pixels: np.ndarray) -> None:
        x0, y0, x1, y1 = self.rect
        layer = doc.stack.by_uid(self.layer_uid)
        layer.pixels[y0:y1, x0:x1] = pixels
        doc.invalidate(self.rect, layer_uid=self.layer_uid)

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

    def __post_init__(self) -> None:
        # An undone add holds the *only* reference to a full-canvas layer, so
        # a cost of zero made it invisible to the byte budget: a document could
        # pin an unbounded number of them past the 192 MiB ceiling. Same
        # measurement LayerRemoveEdit already made.
        self.cost = int(self.layer.pixels.nbytes)

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
