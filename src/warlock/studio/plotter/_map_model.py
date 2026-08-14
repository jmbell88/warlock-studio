"""The plain data a map is made of: layers, objects, and what a dimension may be.

The leaf of the document's own dependency graph, and the reason the split starts
here. Everything the ``_map_*`` mixins do is *to* one of these types, so a mixin
that needed ``TileLayer`` from ``tilemap`` and ``tilemap`` that needed the mixin
would be a cycle; pulling the types out one level breaks it before it exists.

Nothing here knows about history, and that is the line the file draws: a
:class:`TileLayer` is a rectangle of gids with a name and an opacity, and every
undoable *change* to one lives in :mod:`.edits` and the mixins. ``snapshot`` is
the one method that looks like an exception and is not -- it is what a layer
*is*, handed to an edit that decides what to do with it.

The uid counter is per process, not per document, and there is no reserve step:
``.wmap`` stores indices and mints a fresh uid on read, so nothing is ever
restored onto a number the counter has already issued. One namespace covers
layers *and* objects, because both are undo subjects and a single counter makes
"is this uid a layer or an object" a question nothing has to ask.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_uids = itertools.count(1)

# What an object may be. Tiled has six shapes; the other four (ellipse,
# polygon, polyline, text) are refused by the reader rather than half-drawn,
# so these are the two the editor can actually author.
OBJECT_KINDS = ("rect", "point")

# What a custom property may be is :mod:`.props`' single answer -- this module
# carried a byte-identical second copy of that tuple, re-exported through
# ``tilemap`` and read by nothing, which is exactly how the two would have come
# to disagree the first time one of them gained a type.

MAX_DIMENSION = 4096


def new_uid() -> int:
    """Mint a uid for a layer or an object. Never reused within a process.

    The uid is the *address* every undo step is written against, so reuse is
    not a tidiness question: a recycled uid would let a step recorded against a
    deleted layer land on a different one that happens to wear its number.
    """
    return next(_uids)


def _dimension(value: Any, what: str) -> int:
    size = int(value)
    if size < 1:
        raise ValueError(f"{what} must be at least 1")
    if size > MAX_DIMENSION:
        raise ValueError(f"{what} must be at most {MAX_DIMENSION}")
    return size


@dataclass
class MapObject:
    """One named rectangle or point on an object layer.

    Coordinates are in **pixels**, not tiles, and that is Tiled's convention
    rather than a choice made here: an object is routinely placed off the grid
    on purpose -- a spawn point half a tile in from a wall -- and a tile-space
    position would have no way to say so.
    """

    uid: int
    # Tiled's own persistent object id -- distinct from ``uid`` above, which
    # is a per-process address undo steps are written against and means
    # nothing outside this run. ``0`` is "unassigned yet"; ``MapDoc`` mints a
    # real one from ``next_object_id`` at creation, monotone and never
    # reused, because an object-typed property may go on naming this id after
    # the object it named is deleted. See ``docs/PLOTTER_PLAN.md`` Milestone 2.
    id: int = 0
    name: str = ""
    kind: str = "rect"
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    obj_class: str = ""
    # Tiled lets an individual object be hidden. Modelled rather than refused,
    # because a hidden object is still exactly where it is -- the picture stays
    # right, and drawing it faintly is one branch in the canvas. Contrast
    # ``rotation``, which :mod:`.tmx` refuses outright: an unrotated outline
    # drawn for a rotated object is a *wrong* picture, and a wrong picture is
    # worse than a refusal.
    visible: bool = True
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in OBJECT_KINDS:
            raise ValueError(f"an object is one of {list(OBJECT_KINDS)}, not {self.kind!r}")

    def snapshot(self) -> dict[str, Any]:
        """Everything :class:`~.edits.ObjectPropsEdit` restores."""
        return {
            "name": self.name,
            "kind": self.kind,
            "x": float(self.x),
            "y": float(self.y),
            "w": float(self.w),
            "h": float(self.h),
            "obj_class": self.obj_class,
            "visible": bool(self.visible),
            "properties": dict(self.properties),
        }


@dataclass
class TileLayer:
    uid: int
    name: str
    data: np.ndarray
    # Tiled's own persistent layer id, ``MapObject.id``'s twin: ``0`` until
    # ``MapDoc`` mints one from ``next_layer_id`` at creation, monotone and
    # never reused or decremented on undo.
    id: int = 0
    visible: bool = True
    opacity: float = 1.0
    # Blocks *content* edits only. Renaming, reordering, hiding, changing the
    # opacity and deleting all stay available on a locked layer -- Tiled's
    # semantics, and the useful ones: a lock is there to stop you painting on
    # the wrong layer, not to stop you managing the stack.
    locked: bool = False
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "visible": bool(self.visible),
            "opacity": float(self.opacity),
            "locked": bool(self.locked),
            "properties": dict(self.properties),
        }


@dataclass
class ObjectLayer:
    uid: int
    name: str
    # See ``TileLayer.id`` -- the same field, the same counter's twin
    # (``next_layer_id`` covers both layer kinds, one namespace).
    id: int = 0
    objects: list[MapObject] = field(default_factory=list)
    visible: bool = True
    opacity: float = 1.0
    # Blocks *content* edits only. Renaming, reordering, hiding, changing the
    # opacity and deleting all stay available on a locked layer -- Tiled's
    # semantics, and the useful ones: a lock is there to stop you painting on
    # the wrong layer, not to stop you managing the stack.
    locked: bool = False
    properties: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "visible": bool(self.visible),
            "opacity": float(self.opacity),
            "locked": bool(self.locked),
            "properties": dict(self.properties),
        }


Layer = TileLayer | ObjectLayer
