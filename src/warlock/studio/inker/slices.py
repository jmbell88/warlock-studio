"""Named rectangles on the canvas: slices, pivots and nine-slice centres.

A slice is metadata *about* the drawing rather than part of it. It carries a
name, a rectangle, optionally the pivot a game engine should place the sprite
by, and optionally the nine-slice centre a UI panel stretches from -- and none
of it is pixels, so nothing here composites, invalidates or costs the undo
budget a byte.

Three decisions shape the model.

**A slice has durable exported identity, so it is addressed by uid.** It ends
up in a sprite-sheet sidecar and in a TexturePacker atlas that some other
program reads, which is exactly the case ``undo.py``'s "address the subject,
never a position" rule exists for: reordering the list, or deleting the slice
above, must not retarget an edit. That is also why the edits below are
per-slice rather than the whole-list snapshot ``anim_edits.TagsEdit`` uses -- a
tag has no uid to name it by *and* is never dragged, while a slice is dragged
by its corner and every frame of that drag would otherwise copy the list.

**Slices live on the document, not on the grid.** A still drawing has slices
too (a nine-slice button is one PNG), so they hang off ``Document.slices`` and
the per-frame overrides are a dictionary inside each one rather than a second
grid beside ``cels``.

**A per-frame override is a whole key, not a delta.** ``SliceKey`` is frozen
and carries the same three fields the slice itself does, so
:meth:`Slice.at` answers with one object whichever case it is in and no caller
has to merge anything. Keys are explicit: nothing here writes one as a side
effect of moving a slice, because a drag that silently keyed the frame it
happened to be on is how an animation ends up with forty slightly different
rectangles nobody asked for.

Coordinates. ``bounds`` is ``x0 y0 x1 y1`` in canvas pixels with the far edge
**exclusive**, the same convention every rectangle in this package uses.
``pivot`` and ``center`` are measured **from the bounds origin**, because that
is what makes them survive a move: dragging a slice across the canvas must not
move its pivot within it. Exports convert to absolute canvas coordinates at the
boundary (see ``sheetout.slices_snapshot``), so a consumer never has to.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .layers import new_uid
from .undo import Edit

__all__ = [
    "SLICE_PROPS",
    "Slice",
    "SliceAddEdit",
    "SliceChangeEdit",
    "SliceKey",
    "SliceRemoveEdit",
    "slice_props",
]

#: What one slice *is*, for the change edit and for the no-op comparison. One
#: list rather than a field-by-field edit type, for ``SettingsEdit``'s reason in
#: Packwright: five values, and five edit types would be five places for the two
#: halves of an undo to disagree.
SLICE_PROPS = ("name", "bounds", "pivot", "center", "keys")


def _rect(value: Any) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in value)
    return (x0, y0, x1, y1)


def _point(value: Any) -> tuple[float, float] | None:
    return None if value is None else (float(value[0]), float(value[1]))


@dataclass(frozen=True)
class SliceKey:
    """What one slice looks like on one frame.

    Frozen, because it is handed straight back out of :meth:`Slice.at` and
    stored inside undo steps: a caller that mutated the answer would be editing
    the document and the history at once, with nothing pushed.
    """

    bounds: tuple[int, int, int, int] = (0, 0, 1, 1)
    pivot: tuple[float, float] | None = None
    center: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounds", _rect(self.bounds))
        object.__setattr__(self, "pivot", _point(self.pivot))
        object.__setattr__(
            self, "center", None if self.center is None else _rect(self.center)
        )

    @property
    def width(self) -> int:
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> int:
        return self.bounds[3] - self.bounds[1]


@dataclass
class Slice:
    """One named rectangle, plus whatever frames override it.

    ``uid`` is minted per process and means nothing in a file -- the ORA writer
    stores frame *indices* for the keys, the ``cels`` precedent -- but within a
    session it is the only thing an edit or a pane may name a slice by.
    """

    name: str = "Slice"
    bounds: tuple[int, int, int, int] = (0, 0, 1, 1)
    pivot: tuple[float, float] | None = None
    center: tuple[int, int, int, int] | None = None
    keys: dict[int, SliceKey] = field(default_factory=dict)
    uid: int = field(default_factory=new_uid)

    def __post_init__(self) -> None:
        self.normalise()

    def normalise(self) -> None:
        """Coerce the three geometry fields to their declared types.

        Public and idempotent because a *drag* is what writes to these: the
        canvas hands over floats straight off the cursor, and everything
        downstream -- the ORA writer, the sidecar, the overlay's own hit test --
        assumes an integer rectangle.
        """
        self.bounds = _rect(self.bounds)
        self.pivot = _point(self.pivot)
        self.center = None if self.center is None else _rect(self.center)

    @property
    def width(self) -> int:
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> int:
        return self.bounds[3] - self.bounds[1]

    def at(self, frame_uid: int | None = None) -> SliceKey:
        """This slice as it stands on a frame -- the key, or the base.

        One object either way, and never None: every reader (the overlay, the
        sheet sidecar, Packwright's metadata) wants "where is it on this frame"
        and none of them should have to know whether the answer came from an
        override. A still document, or a frame that was never keyed, resolves to
        the base values, which is what "no key" means.
        """
        key = None if frame_uid is None else self.keys.get(frame_uid)
        if key is not None:
            return key
        return SliceKey(bounds=self.bounds, pivot=self.pivot, center=self.center)

    def copy(self) -> Slice:
        """A deep copy that **keeps the uid**, for a snapshot to restore by.

        The uid survives for ``LayerAddEdit``'s reason: an undo restores the
        document's state rather than a set of new objects, and every edit
        already on the stack names this slice by the number it has now.
        ``SliceKey`` is frozen, so copying the dictionary is deep enough.
        """
        return Slice(
            name=self.name,
            bounds=self.bounds,
            pivot=self.pivot,
            center=self.center,
            keys=dict(self.keys),
            uid=self.uid,
        )


def slice_props(entry: Slice) -> dict[str, Any]:
    """One slice's five values, detached from it.

    Detached matters: this is what a change edit holds on both sides, and a
    ``keys`` dictionary shared with the live slice would let the next edit write
    through into the step meant to reverse it -- the rule ``_set_tags`` states
    for tags.
    """
    return {
        "name": entry.name,
        "bounds": entry.bounds,
        "pivot": entry.pivot,
        "center": entry.center,
        "keys": dict(entry.keys),
    }


def with_props(entry: Slice, props: dict[str, Any]) -> Slice:
    """``entry`` with some of its five values replaced. Pure; for previews."""
    return replace(entry, **{k: v for k, v in props.items() if k in SLICE_PROPS})


# --- undo steps ---------------------------------------------------------------
#
# All three cost zero, and that is a measurement rather than an oversight: a
# slice is a name, two rectangles and a handful of small tuples, so the byte
# budget -- which exists to stop the history pinning *pixels* the document has
# let go of -- has nothing to say about any of them. ``TagsEdit`` makes the same
# argument for the same reason.


@dataclass
class SliceAddEdit(Edit):
    """The slice object itself is held, not a copy: re-inserting the same object
    is what keeps its uid, and keeping its uid is what lets a change recorded
    before the undo still apply after the redo."""

    index: int
    entry: Slice

    def undo(self, doc: Any) -> None:
        doc._drop_slice(self.entry.uid)

    def redo(self, doc: Any) -> None:
        doc._put_slice(self.index, self.entry)


@dataclass
class SliceRemoveEdit(Edit):
    index: int
    entry: Slice

    def undo(self, doc: Any) -> None:
        doc._put_slice(self.index, self.entry)

    def redo(self, doc: Any) -> None:
        doc._drop_slice(self.entry.uid)


@dataclass
class SliceChangeEdit(Edit):
    """Name, bounds, pivot, centre, keys -- one step for every property change.

    ``uid`` and not an index, which is the whole point of the type: a slice
    dragged, then another one deleted from above it, then Ctrl+Z, must move the
    slice that was dragged.
    """

    uid: int
    before: dict[str, Any]
    after: dict[str, Any]

    def undo(self, doc: Any) -> None:
        doc._apply_slice(self.uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_slice(self.uid, self.after)
