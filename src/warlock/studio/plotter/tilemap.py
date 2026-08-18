"""The map document: layers over a fixed grid, and the history over them.

A map is a tree of layers, bottom-first, drawn in list order. Tile, object,
group and image layers are deliberately different types rather than one type
with a mode. A :class:`TileLayer` is a rectangle of gids the tools paint into;
an :class:`ObjectLayer` holds the eight core Tiled object geometries and their
typed properties. Nothing an engine does with the second resembles the first.

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

**``next_layer_id``/``next_object_id`` are the opposite of the uid counter in
every way that matters, which is the point of having two.** They *are*
document state -- Tiled's own persistent ``id``, the thing an object-typed
property references, and :mod:`.tmx` and :mod:`.wmap` alike write and read
them: a v3 ``.wmap`` carries both counters in its manifest and every layer's
and object's ``id`` verbatim beside them, while a v1 or v2 file, which carries
no ids at all, has them minted on read by the same rule the export door
applies. Monotone and never decremented on
undo: an object property may go on naming a deleted object's id, so undoing
the add that minted it must not let a later add reissue the same number. The
one cost that follows is accepted rather than engineered around -- after an
add-then-undo, a re-save differs from the prior bytes only in these two
fields, and Tiled behaves identically.
"""

from __future__ import annotations

from typing import Any

from ..tilegrid.tileset import TilesetRef, colour_text
from ..undo import CompoundEdit, Edit, UndoStack
from . import project
from ._map_geometry import GeometryOps
from ._map_layers import LayerOps
from ._map_model import (
    BLEND_MODES,
    DECORATION_FIELDS,
    DRAW_ORDERS,
    LEAF_LAYERS,
    MAX_DIMENSION,
    OBJECT_KINDS,
    OPAQUE_WHITE,
    SHAPE_KINDS,
    Capsule,
    Ellipse,
    GroupLayer,
    ImageLayer,
    Layer,
    MapObject,
    ObjectLayer,
    Point,
    Polygon,
    Polyline,
    Rect,
    Shape,
    Text,
    TileLayer,
    TileShape,
    _dimension,
    new_uid,
    shape_for_kind,
    shape_kind,
    shape_size,
)
from ._map_objects import ObjectOps
from ._map_paint import PaintOps
from ._map_project import ProjectionOps
from ._map_tilesets import TilesetOps
from .edits import MapPropsEdit, MapSettingsEdit

# Re-exported, not merely imported. ``wmap``, ``tmx``, the panes and the tests
# all say ``from .tilemap import TileLayer``, and this module is where a map's
# vocabulary belongs even now that the dataclasses themselves live one file
# down. Moving the *definitions* out was about breaking a cycle before it
# existed -- every mixin operates on these types -- not about changing where
# anybody looks for them.
__all__ = [
    "BLEND_MODES",
    "Capsule",
    "DECORATION_FIELDS",
    "DRAW_ORDERS",
    "LEAF_LAYERS",
    "MAX_DIMENSION",
    "OBJECT_KINDS",
    "OPAQUE_WHITE",
    "SHAPE_KINDS",
    "Ellipse",
    "GroupLayer",
    "ImageLayer",
    "Layer",
    "MapDoc",
    "MapObject",
    "ObjectLayer",
    "Point",
    "Polygon",
    "Polyline",
    "Rect",
    "Shape",
    "Text",
    "TileLayer",
    "TileShape",
    "new_uid",
    "shape_for_kind",
    "shape_kind",
    "shape_size",
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
        infinite: bool = False,
    ) -> None:
        self.width = _dimension(width, "width")
        self.height = _dimension(height, "height")
        self.tile_w = _dimension(tile_w, "tile width")
        self.tile_h = _dimension(tile_h, "tile height")
        # Refused by name here rather than at the first draw, for ``_dimension``'s
        # reason: a map placed by arithmetic nothing implements is not a map.
        self.projection = project.check(projection)
        # Kept on the public document even while dense tile layers remain the
        # implementation boundary. File doors refuse ``True`` by name rather
        # than silently writing a finite map.
        self.infinite = bool(infinite)
        self.layers: list[Layer] = list(layers or [])
        self.tilesets: list[TilesetRef] = list(tilesets or [])
        # Bumped by every hook that mutates the tileset list, so a cache keyed
        # on it is invalidated by an undo exactly as it is by the edit itself --
        # the hooks are what both paths run through. Not serialized and not
        # undoable: it counts changes, and a restored count would let a stale
        # cache match. Starts at 0 and only ever rises.
        self.tileset_epoch = 0
        # Document state, not view state: see the module docstring. Both start
        # at 1 because 0 is "unassigned" on a layer or object's own ``id``,
        # matching Tiled's own convention that a real id is never zero.
        self.next_layer_id = 1
        self.next_object_id = 1
        self.properties: dict[str, Any] = {}
        # Tiled's map-level class and parallax reference point. They are file
        # semantics rather than view state: changing either changes how a game
        # interprets or positions the map, so both travel through every codec.
        self.class_name = ""
        self.parallax_origin = (0.0, 0.0)
        # Tiled 1.12 oblique maps offset each row horizontally and each column
        # vertically by these pixel amounts. They remain zero for every older
        # projection and are threaded through the shared lattice.
        self.skew_x = 0
        self.skew_y = 0
        self.stagger_axis = "y"
        self.stagger_index = "odd"
        self.hex_side = 0
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
        # The open object drag, or None. The same thing for objects, and closed
        # at the same three chokepoints.
        self._object_edit: dict[str, Any] | None = None

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

    # -- the map's own properties ---------------------------------------------

    def set_map_properties(self, properties: dict[str, Any]) -> None:
        """Replace the map's custom properties, undoably.

        Whole-replacement rather than key-at-a-time, which is what the object
        and layer property editors already do: a rename is a delete and an add,
        and expressing that as two steps puts a state on the undo stack the user
        never saw. A call that changes nothing pushes nothing, the rule every
        other writer here follows.
        """
        after = dict(properties)
        if after == self.properties:
            return
        self.history.push(MapPropsEdit(before=dict(self.properties), after=after))
        self._apply_map_properties(after)

    def _apply_map_properties(self, values: dict[str, Any]) -> None:
        self.properties = dict(values)

    def map_settings(self) -> dict[str, Any]:
        """The editable Tiled map metadata, as one undo snapshot."""
        return {
            "class_name": str(self.class_name),
            "parallax_origin": tuple(float(value) for value in self.parallax_origin),
            "renderorder": str(self.renderorder),
            "backgroundcolor": self.backgroundcolor,
            "skew_x": int(self.skew_x),
            "skew_y": int(self.skew_y),
        }

    def set_map_settings(self, **values: Any) -> None:
        """Change map-level Tiled metadata in one undoable step."""
        before = self.map_settings()
        after = {**before, **{key: value for key, value in values.items() if key in before}}
        self._apply_map_settings(after)
        normalized = self.map_settings()
        self._apply_map_settings(before)
        if normalized == before:
            return
        self.history.push(MapSettingsEdit(before=before, after=normalized))
        self._apply_map_settings(normalized)

    def _apply_map_settings(self, values: dict[str, Any]) -> None:
        class_name = str(values["class_name"])
        origin = values["parallax_origin"]
        parallax_origin = (float(origin[0]), float(origin[1]))
        order = str(values["renderorder"])
        if order not in project.RENDER_ORDERS:
            raise ValueError(f"unknown render order {order!r}")
        backgroundcolor = colour_text(
            values["backgroundcolor"], "a map background colour"
        )
        skew_x, skew_y = int(values["skew_x"]), int(values["skew_y"])
        self.class_name = class_name
        self.parallax_origin = parallax_origin
        self.renderorder = order
        self.backgroundcolor = backgroundcolor
        self.skew_x, self.skew_y = skew_x, skew_y

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
        self.end_object_edit()
        return self.history.undo(self)

    def redo(self) -> bool:
        """The twin, for :meth:`undo`'s reason -- and it is not symmetric only
        in appearance: redoing with a session open would replay a step onto
        pixels the session has already changed underneath it."""
        self.end_stroke()
        self.end_object_edit()
        return self.history.redo(self)
