"""The map document: layers over a fixed grid, and the history over them.

A map is a stack of layers, bottom-first, drawn in list order. Two kinds exist
and they are deliberately different types rather than one type with a mode:
a :class:`TileLayer` is a rectangle of gids the tools paint into, and an
:class:`ObjectLayer` is a list of named rectangles and points that carry typed
properties -- a spawn point, a trigger volume, a camera bound. Nothing an engine
does with the second resembles what it does with the first.

**Dirty is a comparison against ``history.head``, not a flag.** An undo is a
change, so a counter-based check calls a document that has been undone back to
its saved state unsaved forever. :attr:`MapDoc.saved_head` records the head at
save time and :attr:`MapDoc.dirty` compares against it -- the rule every other
document type here already follows.

**A write that changes nothing pushes nothing.** Stamping the same tile onto a
cell that already holds it, filling a rectangle with what is already there,
renaming a layer to its own name: each is a real user action and none of them is
a change, and pushing a no-op step makes a saved document ask to be saved again
the next time anybody clicks.

**The uid counter is per process, not per document, and there is no reserve
step.** ``clay.document`` needs one because a ``.wblk`` stores its uids and the
reader restores them, so the counter has to be pushed past whatever the file
carried. :mod:`.wmap` deliberately stores *indices* instead and mints every uid
fresh on read, so nothing here is ever restored onto a number already issued and
a ``reserve_uid`` would have no case to answer. One namespace covers layers
*and* objects, because both are undo subjects and a single counter makes "is
this uid a layer or an object" a question nothing has to ask.
"""

from __future__ import annotations

from typing import Any

from ..undo import CompoundEdit, Edit, UndoStack
from . import project
from ._map_geometry import GeometryOps
from ._map_layers import LayerOps
from ._map_model import (
    MAX_DIMENSION,
    OBJECT_KINDS,
    PROPERTY_TYPES,
    Layer,
    MapObject,
    ObjectLayer,
    TileLayer,
    _dimension,
    new_uid,
)
from ._map_objects import ObjectOps
from ._map_paint import PaintOps
from ._map_project import ProjectionOps
from ._map_tilesets import TilesetOps
from .tileset import TilesetRef

# Re-exported, not merely imported. ``wmap``, ``tmx``, the panes and the tests
# all say ``from .tilemap import TileLayer``, and this module is where a map's
# vocabulary belongs even now that the dataclasses themselves live one file
# down. Moving the *definitions* out was about breaking a cycle before it
# existed -- every mixin operates on these types -- not about changing where
# anybody looks for them.
__all__ = [
    "MAX_DIMENSION",
    "OBJECT_KINDS",
    "PROPERTY_TYPES",
    "Layer",
    "MapDoc",
    "MapObject",
    "ObjectLayer",
    "TileLayer",
    "new_uid",
]


class MapDoc(ProjectionOps, TilesetOps, LayerOps, PaintOps, GeometryOps, ObjectOps):
    """One tile map, its projection, its tilesets, its layers and its history.

    **Composed from method-only mixins in sibling ``_map_*`` modules**, the
    ``inker.Document`` shape. The class was one file answering to placement,
    lookup, the layer list, tilesets, painting, geometry and objects, and the
    seam between those is real: none of them calls into another except through
    ``self``, which is exactly the condition under which a mixin is a move
    rather than a redesign. What stays here is what is *about the document as a
    whole* -- what it is made of (``__init__``), whether it is saved, and the
    history every mixin pushes onto.
    """

    def __init__(
        self,
        width: int = 32,
        height: int = 32,
        tile_w: int = 32,
        tile_h: int = 32,
        *,
        layers: list[Layer] | None = None,
        tilesets: list[TilesetRef] | None = None,
        projection: str = project.ORTHOGONAL,
    ) -> None:
        self.width = _dimension(width, "width")
        self.height = _dimension(height, "height")
        self.tile_w = _dimension(tile_w, "tile width")
        self.tile_h = _dimension(tile_h, "tile height")
        # Refused by name here rather than at the first draw, for ``_dimension``'s
        # reason: a map placed by arithmetic nothing implements is not a map.
        self.projection = project.check(projection)
        self.layers: list[Layer] = list(layers or [])
        self.tilesets: list[TilesetRef] = list(tilesets or [])
        # Bumped by every hook that mutates the tileset list, so a cache keyed
        # on it is invalidated by an undo exactly as it is by the edit itself --
        # the hooks are what both paths run through. Not serialized and not
        # undoable: it counts changes, and a restored count would let a stale
        # cache match. Starts at 0 and only ever rises.
        self.tileset_epoch = 0
        self.properties: dict[str, Any] = {}
        # Preserved verbatim across a round trip. Neither is honoured by the
        # renderer yet, and writing back something a user set in Tiled is
        # cheaper than explaining why it vanished.
        self.renderorder = "right-down"
        self.backgroundcolor: str | None = None
        self.history = UndoStack()
        self.saved_head = 0
        # View state, and so deliberately not undoable and not serialized: an
        # undoable "which layer am I on" would move the head and make a document
        # ask to be saved because the user clicked a different row.
        self.active_layer: int | None = self.layers[0].uid if self.layers else None
        # The open stroke session, or None. View-adjacent and never serialized:
        # a document is always saved with its strokes closed.
        self._stroke: dict[str, Any] | None = None

    # -- identity ------------------------------------------------------------

    @property
    def dirty(self) -> bool:
        return self.history.head != self.saved_head

    def mark_saved(self, head: int | None = None) -> None:
        """Record the head a save wrote.

        ``head`` is passed by a caller that captured it *before* handing the
        encode to a task thread, which is the only correct value: the document
        may have moved on since, and marking the live head would call those
        later edits saved.
        """
        self.saved_head = self.history.head if head is None else int(head)

    # -- history -------------------------------------------------------------

    def push(self, edit: Edit) -> None:
        """For a caller assembling a :class:`CompoundEdit` of its own."""
        self.history.push(edit)

    def compound(self, edits: list[Edit]) -> None:
        """Push several steps that must undo together, or nothing if empty."""
        if edits:
            self.history.push(CompoundEdit(edits=list(edits)))

    def undo(self) -> bool:
        """One Ctrl+Z, with any open stroke committed first.

        The ``inker/_doc_history.py`` shape. A stroke writes the live array with
        no history at all, so undoing *through* an open session used to reverse
        the step before it while leaving the uncommitted paint on the layer --
        a document whose cells are ahead of its head, and one keypress further
        from the state the user was asking for rather than nearer.

        Committing it clears the redo branch, which is the right cost: a
        half-finished stroke is real work, and the alternative on offer is
        silently discarding it.
        """
        self.end_stroke()
        return self.history.undo(self)

    def redo(self) -> bool:
        """The twin, for :meth:`undo`'s reason -- and it is not symmetric only
        in appearance: redoing with a session open would replay a step onto
        pixels the session has already changed underneath it."""
        self.end_stroke()
        return self.history.redo(self)
