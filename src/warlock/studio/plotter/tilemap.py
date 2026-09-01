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

from ..tilegrid import gid as gidlib
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
    MAX_GROWTH,
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
    Stamp,
    Text,
    TileLayer,
    TileShape,
    _dimension,
    new_uid,
    shape_for_kind,
    shape_kind,
    shape_size,
)
from ._map_objects import ObjectOps, object_bounds, objects_in_rect
from ._map_paint import PaintOps
from ._map_project import ProjectionOps
from ._map_tilesets import TilesetOps
from .edits import MapPropsEdit, MapSettingsEdit, StampEdit

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
    "MAX_GROWTH",
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
    "Stamp",
    "Text",
    "TileLayer",
    "TileShape",
    "new_uid",
    "object_bounds",
    "objects_in_rect",
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
        # An infinite map has no fixed rectangle: painting past its edge grows
        # it, and cells may sit at negative coordinates.
        #
        # **Stored as a dense window over the populated extent plus an origin,
        # not as sparse chunks -- and the deviation is a decision.** The format
        # is chunked and the codecs read and write chunks; what differs is only
        # how the *editor* holds them between the two. A dense window means
        # every tool, both renderers, the terrain and wang engines, the stroke
        # session and ``TilePatchEdit`` go on working on the array they already
        # work on, and the growth is an ordinary ``resize`` -- which is already
        # undoable, already moves objects by the right rule, and already has a
        # test corpus. Sparse chunks would have re-answered all of that at once.
        # What it costs is memory on a map painted in two clusters a thousand
        # cells apart, and :data:`MAX_DIMENSION` caps the populated extent for
        # exactly that reason -- the engine's own cap, not the new-map form's
        # ``plotter_setup.MAX_TILES``, because this package may not reach into
        # ``studio`` and the argument is the same either way.
        #
        # It also buys the one property Q's flood needs: a flood is bounded to
        # the window, and the window *is* the populated extent, so "bounded to
        # content bounds" is what the storage says rather than something the
        # tool has to remember to do.
        self.infinite = bool(infinite)
        # Where the stored grid's cell (0, 0) sits in *true* map coordinates.
        # Always (0, 0) on a finite map; on an infinite one it goes negative as
        # the map grows left or up, and it is what the codecs write chunk
        # coordinates against.
        self.origin_x = 0
        self.origin_y = 0
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
        # The nine numbered stamps, by slot. **Document state**, because a
        # stamp is an array of gids and a gid is numbered against this map's
        # firstgids -- see ``Stamp``. Undoable and serialized, unlike the view
        # state further down: a stamp stored and not saved is unsaved work.
        self.stamps: dict[int, Stamp] = {}
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
        # The open *group* drag, or None. A parallel session rather than a
        # widened one -- see ``_map_objects.begin_group_edit`` -- and closed at
        # exactly the same chokepoints, because a group left open is the same
        # document-ahead-of-its-history defect.
        self._group_edit: dict[str, Any] | None = None
        # The open collision-shape drag in the tileset editor, or None. The
        # third session, with the first two's rule: the document moves live and
        # the history moves once, at the release.
        self._tile_meta_edit: dict[str, Any] | None = None

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
            # The three the offset projections read. They have round-tripped
            # through TMX since the day those projections landed and had no UI
            # at all, so a hexagonal map made here always wrote hex_side = 0 --
            # a .tmx this editor produced and Tiled drew wrongly.
            "stagger_axis": str(self.stagger_axis),
            "stagger_index": str(self.stagger_index),
            "hex_side": int(self.hex_side),
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
        stagger_axis = str(values["stagger_axis"])
        if stagger_axis not in ("x", "y"):
            raise ValueError(f"unknown stagger axis {stagger_axis!r}")
        stagger_index = str(values["stagger_index"])
        if stagger_index not in ("odd", "even"):
            raise ValueError(f"unknown stagger index {stagger_index!r}")
        hex_side = max(0, int(values["hex_side"]))
        self.class_name = class_name
        self.parallax_origin = parallax_origin
        self.renderorder = order
        self.backgroundcolor = backgroundcolor
        self.skew_x, self.skew_y = skew_x, skew_y
        self.stagger_axis, self.stagger_index = stagger_axis, stagger_index
        self.hex_side = hex_side

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
        self.end_group_edit()
        self.end_tile_meta_edit()
        return self.history.undo(self)

    def redo(self) -> bool:
        """The twin, for :meth:`undo`'s reason -- and it is not symmetric only
        in appearance: redoing with a session open would replay a step onto
        pixels the session has already changed underneath it."""
        self.end_stroke()
        self.end_object_edit()
        self.end_group_edit()
        self.end_tile_meta_edit()
        return self.history.redo(self)

    # -- stamps ----------------------------------------------------------------

    def set_stamp(self, slot: int, cells: Any, *, name: str | None = None) -> bool:
        """Put a block of gids into a numbered slot. -> whether anything changed.

        ``name=None`` **keeps the name the slot already had**, which is the
        behaviour storing wants: a user who has called slot 3 "roof corner" and
        re-captures a better one has not renamed it, and asking them to retype
        the name every time would make naming a slot not worth doing.

        The array is copied and frozen here rather than trusted from the caller,
        for the reason the props edits copy their dicts: the brush handed in is
        the live one the canvas goes on transforming, and a slot sharing it
        would change under the user when they pressed X.
        """
        import numpy as np

        block = np.array(cells, dtype=gidlib.DTYPE, copy=True)
        block.setflags(write=False)
        before = self.stamps.get(int(slot))
        after = Stamp(
            name=(before.name if before is not None else "") if name is None else str(name),
            cells=block,
        )
        if before is not None and before.name == after.name and np.array_equal(
            before.cells, after.cells
        ):
            # ``TilePatchEdit``'s rule: a write that changes nothing pushes
            # nothing, so re-storing an identical stamp does not dirty the map.
            return False
        self._push_stamp(int(slot), before, after)
        return True

    def rename_stamp(self, slot: int, name: str) -> bool:
        """Give a stored slot a name. -> whether anything changed.

        Refuses an empty slot rather than minting one: a name with no cells
        behind it is a row in the pane that recalls nothing, and the user's
        gesture was to type into a field they should not have been offered.
        """
        before = self.stamps.get(int(slot))
        if before is None or before.name == str(name):
            return False
        self._push_stamp(int(slot), before, Stamp(name=str(name), cells=before.cells))
        return True

    def clear_stamp(self, slot: int) -> bool:
        """Empty a slot. -> whether it held anything."""
        before = self.stamps.get(int(slot))
        if before is None:
            return False
        self._push_stamp(int(slot), before, None)
        return True

    def _push_stamp(self, slot: int, before: Any, after: Any) -> None:
        self._apply_stamp(slot, after)
        self.history.push(StampEdit(slot, before, after))

    def _apply_stamp(self, slot: int, value: Any) -> None:
        """Both directions of :class:`StampEdit`, and the only writer."""
        if value is None:
            self.stamps.pop(int(slot), None)
        else:
            self.stamps[int(slot)] = value

    def step_history(self, index: int) -> bool:
        """Jump to a position in the undo stack. -> whether anything moved.

        What the history panel asks for, and it has to be a method here rather
        than a ``doc.history.step_to(doc, n)`` at the call site: ``step_to``
        walks the *stack's* own undo and redo, which are not
        :meth:`undo`/:meth:`redo` and therefore commit no open session. Called
        straight, a jump made mid-stroke would step over uncommitted paint and
        leave the cells ahead of the head -- the exact defect the two methods
        above exist to prevent, reintroduced by the one caller that reached past
        them.
        """
        self.end_stroke()
        self.end_object_edit()
        self.end_group_edit()
        self.end_tile_meta_edit()
        return self.history.step_to(self, int(index))
