"""The undoable steps a map document is changed by.

On :mod:`warlock.studio.undo`, the engine the raster editor and Clay already
share, and under its two travelling rules:

**Every edit addresses its subject by uid, never by index.** An undo issued
after the layer list has been reordered must still land on the layer the edit
was made to, and an index stopped naming that layer the moment anything moved.
Objects are addressed the same way, by their own uid within a named layer, so
deleting the object above one does not retarget an edit to it.

**An edit owns its arrays.** ``cost`` is what eviction is driven by, and a
numpy slice reports only its own ``nbytes`` while keeping the whole base alive:
a patch that costs four kilobytes by the budget's reckoning can pin a
sixteen-megabyte layer, invisibly. So every array that arrives as a view is
copied on the way in.

The one type whose cost is easy to get wrong is :class:`TilesetAddEdit`. An
undone add holds the *only* reference to a tileset's pixels -- several megabytes
for a large atlas -- so reporting zero would hide it from the byte budget
entirely, which is exactly the mistake ``LayerAddEdit`` was written to avoid in
the raster editor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..undo import Edit


def _own(array: np.ndarray) -> np.ndarray:
    """A contiguous array this edit is the owner of.

    ``base is not None`` is the test for a view, and it is the whole reason
    this function exists rather than a bare ``np.asarray``.
    """
    out = np.ascontiguousarray(array)
    if out.base is not None or out is array:
        out = out.copy()
    return out


@dataclass
class TilePatchEdit(Edit):
    """A rectangle of one tile layer, before and after.

    Dirty-rect rather than whole-layer for the reason the raster editor's patch
    is: a single stamp on a 200x200 map is nine cells, and snapshotting 160 KB
    per click would empty the byte budget in a few dozen strokes.
    """

    layer_uid: int
    x0: int
    y0: int
    before: np.ndarray
    after: np.ndarray

    def __post_init__(self) -> None:
        self.before = _own(self.before)
        self.after = _own(self.after)
        self.cost = int(self.before.nbytes + self.after.nbytes)

    def undo(self, doc: Any) -> None:
        doc._blit(self.layer_uid, self.x0, self.y0, self.before)

    def redo(self, doc: Any) -> None:
        doc._blit(self.layer_uid, self.x0, self.y0, self.after)


def _subtree_bytes(layer: Any) -> int:
    """Every array a layer *and everything under it* owns.

    Duck-typed rather than by ``isinstance``, which is this module's rule
    already (``layer: Any`` everywhere): ``edits`` sits below the model and
    importing the four layer classes to ask what kind one is would invert that.

    Recursive because a group travels as one object. An add or remove of a group
    is one step holding the whole subtree, so while it stands this edit is the
    only thing keeping several megabytes of nested tile layers and image-layer
    pictures alive -- and a cost that stopped at the group would report zero for
    all of it, which is exactly the mistake ``LayerAddEdit`` was written to
    avoid in the raster editor.
    """
    total = int(getattr(getattr(layer, "data", None), "nbytes", 0))
    total += int(getattr(getattr(layer, "pixels", None), "nbytes", 0))
    for child in getattr(layer, "children", None) or ():
        total += _subtree_bytes(child)
    return total


@dataclass
class LayerAddEdit(Edit):
    """Holds the layer object itself, not a copy of it.

    Re-insertion has to bring back the *same* uid, or every patch recorded
    against the layer before it was removed is stranded on a number nothing
    answers to.

    ``parent_uid`` is the group the layer belongs to, or ``None`` for the root
    list. Recorded beside the index rather than being derivable from it, because
    an index alone stopped being an address the moment a layer could be
    somewhere other than the root.
    """

    layer: Any
    index: int
    parent_uid: int | None = None

    def __post_init__(self) -> None:
        self.cost = _subtree_bytes(self.layer)

    def undo(self, doc: Any) -> None:
        doc._detach_layer(self.layer)

    def redo(self, doc: Any) -> None:
        doc._attach_layer(self.layer, self.index, self.parent_uid)


@dataclass
class LayerRemoveEdit(Edit):
    """The mirror of the add. Its cost is the same arrays, for the same reason:
    while the removal stands, this edit is what keeps the subtree alive."""

    layer: Any
    index: int
    parent_uid: int | None = None

    def __post_init__(self) -> None:
        self.cost = _subtree_bytes(self.layer)

    def undo(self, doc: Any) -> None:
        doc._attach_layer(self.layer, self.index, self.parent_uid)

    def redo(self, doc: Any) -> None:
        doc._detach_layer(self.layer)


@dataclass
class LayerMoveEdit(Edit):
    """A reorder *or* a reparent, recorded as the uid and the two places.

    By uid rather than by holding the layer, so that a move undone after an
    unrelated add still finds its subject where the tree actually has it.

    A *place* is ``(parent_uid, index)`` and not an index, because the tree made
    the two halves of one gesture: dragging a layer into a group both changes
    which list it is in and where in that list it lands. Two steps would put a
    state on the undo stack the user never saw -- the layer at the root, at the
    new index -- and one of them would be a position the gesture never visited.
    """

    layer_uid: int
    before: tuple[int | None, int]
    after: tuple[int | None, int]

    def undo(self, doc: Any) -> None:
        doc._relocate(self.layer_uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._relocate(self.layer_uid, self.after)


@dataclass
class LayerPropsEdit(Edit):
    """Name, visibility, opacity and custom properties -- everything about a
    layer that is not its pixels or its position."""

    layer_uid: int
    before: dict[str, Any]
    after: dict[str, Any]

    def __post_init__(self) -> None:
        # Copied: the caller routinely hands us the live dict it is about to
        # write into, and a "before" that mutates with the document restores
        # nothing. One level deeper than a bare ``dict()``, matching
        # ``ObjectPropsEdit``: ``set_layer_props`` builds ``after`` by spreading
        # ``before``, so the two share one ``properties`` mapping whenever that
        # is not the field being changed. Nothing mutates it in place today --
        # ``_apply_layer_props`` rebinds with a fresh copy -- so this is the
        # hazard removed rather than a bug fixed, and it costs one call.
        self.before = _snapshot(self.before)
        self.after = _snapshot(self.after)

    def undo(self, doc: Any) -> None:
        doc._apply_layer_props(self.layer_uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_layer_props(self.layer_uid, self.after)


@dataclass
class MapPropsEdit(Edit):
    """The map's own custom properties.

    Separate from ``LayerPropsEdit`` because it addresses no layer: the map is
    the thing being changed, and giving the layer edit a nullable uid would make
    every reader of it ask which kind it was.
    """

    before: dict[str, Any]
    after: dict[str, Any]

    def __post_init__(self) -> None:
        self.before = dict(self.before)
        self.after = dict(self.after)

    def undo(self, doc: Any) -> None:
        doc._apply_map_properties(self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_map_properties(self.after)


@dataclass
class TilesetAddEdit(Edit):
    """Adding a tileset to the map's list.

    Removal is not an operation the map offers, and the omission is deliberate:
    ``firstgid`` allocation is contiguous, so dropping a set from the middle
    would either renumber every set above it -- invalidating every gid already
    painted -- or leave a hole that the next add would have to reason about.
    Undo therefore only ever has to take back the set it appended, and it does
    so by identity so that a stack which has somehow got out of order fails
    loudly rather than removing somebody else's.
    """

    ref: Any

    def __post_init__(self) -> None:
        self.cost = int(getattr(getattr(self.ref, "tileset", None), "pixels", np.empty(0)).nbytes)

    def undo(self, doc: Any) -> None:
        doc._detach_tileset(self.ref)

    def redo(self, doc: Any) -> None:
        doc._attach_tileset(self.ref)


@dataclass
class ProjectionEdit(Edit):
    """Changing which lattice the map is drawn on.

    Cheap in bytes and enormous in meaning: every cell keeps its gid and moves
    to a different place on screen. Undoable like anything else, but the mode
    only ever offers it on an empty map -- a set drawn for one lattice paints
    the wrong shape into cells already painted for the other.
    """

    before: str
    after: str

    def undo(self, doc: Any) -> None:
        doc._set_projection(self.before)

    def redo(self, doc: Any) -> None:
        doc._set_projection(self.after)


@dataclass
class TilesetReplaceEdit(Edit):
    """Swapping one tileset's art for another of the same shape.

    The smallest relaxation of "tilesets are never removed" that a polish pass
    needs, and it relaxes nothing that rule was defending: the replacement keeps
    the same ``firstgid`` and the same tile count, so every gid already painted
    still resolves to the same row and column of the same grid and simply draws
    the new art. What is refused -- at the call site, by name -- is a different
    tile count, which is the case that *would* renumber.

    The cost is both atlases, because while this edit stands it is the only
    thing keeping the original's pixels alive.
    """

    index: int
    before: Any
    after: Any

    def __post_init__(self) -> None:
        self.cost = sum(
            int(getattr(getattr(ref, "tileset", None), "pixels", np.empty(0)).nbytes)
            for ref in (self.before, self.after)
        )

    def undo(self, doc: Any) -> None:
        doc._swap_tileset(self.index, self.before)

    def redo(self, doc: Any) -> None:
        doc._swap_tileset(self.index, self.after)


@dataclass
class ObjectAddEdit(Edit):
    layer_uid: int
    obj: Any
    index: int

    def undo(self, doc: Any) -> None:
        doc._detach_object(self.layer_uid, self.obj)

    def redo(self, doc: Any) -> None:
        doc._attach_object(self.layer_uid, self.obj, self.index)


@dataclass
class ObjectRemoveEdit(Edit):
    layer_uid: int
    obj: Any
    index: int

    def undo(self, doc: Any) -> None:
        doc._attach_object(self.layer_uid, self.obj, self.index)

    def redo(self, doc: Any) -> None:
        doc._detach_object(self.layer_uid, self.obj)


@dataclass
class ObjectPropsEdit(Edit):
    """A moved, rotated, reshaped, renamed or re-keyed object.

    ``properties`` is a dict inside the snapshot dicts, so both are deep-copied
    one level: a shallow copy would hand undo and redo the same live mapping and
    the two would overwrite each other.

    ``shape`` is *not* copied and must not be: it is a frozen dataclass, so the
    two sides of this step, the document and any open drag session can share
    one safely -- which is what keeps a snapshot per drag frame free of a deep
    copy of the geometry. The ``kind``/``w``/``h`` keys travelling beside it
    are the derived echo the older callers speak; ``_apply_object_props``
    assigns the shape and ignores them.
    """

    layer_uid: int
    obj_uid: int
    before: dict[str, Any]
    after: dict[str, Any]

    def __post_init__(self) -> None:
        self.before = _snapshot(self.before)
        self.after = _snapshot(self.after)

    def undo(self, doc: Any) -> None:
        doc._apply_object_props(self.layer_uid, self.obj_uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_object_props(self.layer_uid, self.obj_uid, self.after)


def _snapshot(values: dict[str, Any]) -> dict[str, Any]:
    out = dict(values)
    if isinstance(out.get("properties"), dict):
        out["properties"] = dict(out["properties"])
    return out


@dataclass
class ResizeEdit(Edit):
    """The whole canvas, before and after.

    A snapshot rather than a patch, and deliberately: a resize changes every
    layer's *shape*, so there is no rectangle that describes it. The cost is
    every array on both sides, which is the honest figure -- a resize is
    expensive and the budget should see that it is.
    """

    before_size: tuple[int, int]
    after_size: tuple[int, int]
    before: dict[int, np.ndarray] = field(default_factory=dict)
    after: dict[int, np.ndarray] = field(default_factory=dict)
    before_objects: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    after_objects: dict[int, list[tuple[float, float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.before = {uid: _own(a) for uid, a in self.before.items()}
        self.after = {uid: _own(a) for uid, a in self.after.items()}
        self.cost = int(
            sum(a.nbytes for a in self.before.values())
            + sum(a.nbytes for a in self.after.values())
        )

    def undo(self, doc: Any) -> None:
        doc._apply_resize(self.before_size, self.before, self.before_objects)

    def redo(self, doc: Any) -> None:
        doc._apply_resize(self.after_size, self.after, self.after_objects)


@dataclass
class TileSizeEdit(Edit):
    """The grid's cell size, before and after, with the objects that rode it.

    Two ints and a handful of coordinates, where :class:`ResizeEdit` is every
    array in the document -- because changing a *cell's size* changes no layer's
    shape and no gid. The tile arrays are deliberately absent: a gid names a
    tile, and how large the cell under it is drawn has nothing to do with which
    tile that is. Snapshotting them would cost the whole document to record
    nothing.

    The object rectangles are here for ``ResizeEdit``'s reason, arrived at from
    the other direction: they are absolute pixels, so a grid whose cells change
    size leaves them describing a different part of the map unless they are
    scaled with it.
    """

    before_size: tuple[int, int]
    after_size: tuple[int, int]
    before_objects: dict[int, list[tuple[float, float, float, float]]] = field(
        default_factory=dict
    )
    after_objects: dict[int, list[tuple[float, float, float, float]]] = field(
        default_factory=dict
    )

    def undo(self, doc: Any) -> None:
        doc._apply_tile_size(self.before_size, self.before_objects)

    def redo(self, doc: Any) -> None:
        doc._apply_tile_size(self.after_size, self.after_objects)
