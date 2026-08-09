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

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..undo import CompoundEdit, Edit, UndoStack
from . import gid as gidlib
from .edits import (
    LayerAddEdit,
    LayerMoveEdit,
    LayerPropsEdit,
    LayerRemoveEdit,
    ObjectAddEdit,
    ObjectPropsEdit,
    ObjectRemoveEdit,
    ResizeEdit,
    TilePatchEdit,
    TilesetAddEdit,
)
from .tileset import Tileset, TilesetRef

_uids = itertools.count(1)

# What an object may be. Tiled has six shapes; the other four (ellipse,
# polygon, polyline, text) are refused by the reader rather than half-drawn,
# so these are the two the editor can actually author.
OBJECT_KINDS = ("rect", "point")

# The property types a map, layer or object may carry. Tiled's own set minus
# ``file``, ``object`` and ``class``, which reference things outside the file.
PROPERTY_TYPES = ("string", "int", "float", "bool", "color")

MAX_DIMENSION = 4096


def new_uid() -> int:
    """Mint a uid for a layer or an object. Never reused within a process.

    The uid is the *address* every undo step is written against, so reuse is
    not a tidiness question: a recycled uid would let a step recorded against a
    deleted layer land on a different one that happens to wear its number.
    """
    return next(_uids)


@dataclass
class MapObject:
    """One named rectangle or point on an object layer.

    Coordinates are in **pixels**, not tiles, and that is Tiled's convention
    rather than a choice made here: an object is routinely placed off the grid
    on purpose -- a spawn point half a tile in from a wall -- and a tile-space
    position would have no way to say so.
    """

    uid: int
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
    visible: bool = True
    opacity: float = 1.0
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
            "properties": dict(self.properties),
        }


@dataclass
class ObjectLayer:
    uid: int
    name: str
    objects: list[MapObject] = field(default_factory=list)
    visible: bool = True
    opacity: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "visible": bool(self.visible),
            "opacity": float(self.opacity),
            "properties": dict(self.properties),
        }


Layer = TileLayer | ObjectLayer


class MapDoc:
    """One orthogonal tile map, its tilesets, its layers and its history."""

    def __init__(
        self,
        width: int = 32,
        height: int = 32,
        tile_w: int = 32,
        tile_h: int = 32,
        *,
        layers: list[Layer] | None = None,
        tilesets: list[TilesetRef] | None = None,
    ) -> None:
        self.width = _dimension(width, "width")
        self.height = _dimension(height, "height")
        self.tile_w = _dimension(tile_w, "tile width")
        self.tile_h = _dimension(tile_h, "tile height")
        self.layers: list[Layer] = list(layers or [])
        self.tilesets: list[TilesetRef] = list(tilesets or [])
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

    @property
    def pixel_width(self) -> int:
        return self.width * self.tile_w

    @property
    def pixel_height(self) -> int:
        return self.height * self.tile_h

    # -- lookup --------------------------------------------------------------

    def layer(self, uid: int) -> Layer | None:
        for entry in self.layers:
            if entry.uid == uid:
                return entry
        return None

    def index_of(self, uid: int) -> int:
        for index, entry in enumerate(self.layers):
            if entry.uid == uid:
                return index
        raise KeyError(f"no layer {uid}")

    def tile_layers(self) -> list[TileLayer]:
        return [entry for entry in self.layers if isinstance(entry, TileLayer)]

    def active(self) -> Layer | None:
        return None if self.active_layer is None else self.layer(self.active_layer)

    def set_active_layer(self, uid: int | None) -> None:
        """View state; pushes no step, exactly as ``set_active_layer`` does in
        the raster editor and for the same reason."""
        self.active_layer = None if uid is None else int(uid)

    def resolve(self, cell: int) -> tuple[Tileset, int] | None:
        """``(tileset, local id)`` for one cell, or ``None`` for an empty one.

        Takes the *encoded* value, flags and all, so callers never have to
        remember to mask -- forgetting to is how a flipped tile comes to render
        as an out-of-range id.
        """
        tile_id = int(cell) & gidlib.GID_MASK
        if tile_id == 0:
            return None
        for ref in self.tilesets:
            if ref.holds(tile_id):
                return ref.tileset, ref.local(tile_id)
        return None

    def ref_for(self, tile_id: int) -> TilesetRef | None:
        masked = int(tile_id) & gidlib.GID_MASK
        for ref in self.tilesets:
            if ref.holds(masked):
                return ref
        return None

    @property
    def next_firstgid(self) -> int:
        """Where the next tileset's ids would begin.

        Contiguous, and 1-based because zero means "no tile" in every Tiled
        file ever written.
        """
        return 1 if not self.tilesets else self.tilesets[-1].last_gid + 1

    # -- tilesets ------------------------------------------------------------

    def add_tileset(self, tileset: Tileset, *, source: str = "") -> TilesetRef:
        ref = TilesetRef(firstgid=self.next_firstgid, tileset=tileset, source=source)
        self.history.push(TilesetAddEdit(ref=ref))
        self._attach_tileset(ref)
        return ref

    def _attach_tileset(self, ref: TilesetRef) -> None:
        self.tilesets.append(ref)

    def _detach_tileset(self, ref: TilesetRef) -> None:
        for index, entry in enumerate(self.tilesets):
            if entry is ref:
                del self.tilesets[index]
                return
        raise KeyError("that tileset is not in this map")

    # -- layers --------------------------------------------------------------

    def add_tile_layer(self, name: str = "", *, index: int | None = None) -> TileLayer:
        layer = TileLayer(
            uid=new_uid(),
            name=name or f"Tiles {len(self.tile_layers()) + 1}",
            data=gidlib.empty_layer(self.width, self.height),
        )
        return self._add_layer(layer, index)

    def add_object_layer(self, name: str = "", *, index: int | None = None) -> ObjectLayer:
        layer = ObjectLayer(uid=new_uid(), name=name or "Objects")
        return self._add_layer(layer, index)

    def _add_layer(self, layer: Any, index: int | None) -> Any:
        at = len(self.layers) if index is None else max(0, min(int(index), len(self.layers)))
        self.history.push(LayerAddEdit(layer=layer, index=at))
        self._attach_layer(layer, at)
        self.active_layer = layer.uid
        return layer

    def remove_layer(self, uid: int) -> None:
        index = self.index_of(uid)
        layer = self.layers[index]
        self.history.push(LayerRemoveEdit(layer=layer, index=index))
        self._detach_layer(layer)

    def move_layer(self, uid: int, to_index: int) -> None:
        before = self.index_of(uid)
        after = max(0, min(int(to_index), len(self.layers) - 1))
        if before == after:
            return
        self.history.push(
            LayerMoveEdit(layer_uid=int(uid), before_index=before, after_index=after)
        )
        self._relocate(uid, after)

    def set_layer_props(self, uid: int, **values: Any) -> None:
        """Rename, hide, fade or re-key a layer.

        Only the keys handed in are touched, and a call that changes none of
        them pushes nothing.
        """
        layer = self.layer(uid)
        if layer is None:
            raise KeyError(f"no layer {uid}")
        before = layer.snapshot()
        after = {**before, **{k: v for k, v in values.items() if k in before}}
        if after == before:
            return
        self.history.push(LayerPropsEdit(layer_uid=int(uid), before=before, after=after))
        self._apply_layer_props(uid, after)

    def _attach_layer(self, layer: Any, index: int) -> None:
        self.layers.insert(index, layer)

    def _detach_layer(self, layer: Any) -> None:
        for index, entry in enumerate(self.layers):
            if entry is layer:
                del self.layers[index]
                break
        else:
            raise KeyError("that layer is not in this map")
        if self.active_layer == layer.uid:
            # By uid, so the fallback is a layer that still exists rather than
            # whichever one slid into the removed one's index.
            self.active_layer = self.layers[-1].uid if self.layers else None

    def _relocate(self, uid: int, to_index: int) -> None:
        index = self.index_of(uid)
        layer = self.layers.pop(index)
        self.layers.insert(max(0, min(int(to_index), len(self.layers))), layer)

    def _apply_layer_props(self, uid: int, values: dict[str, Any]) -> None:
        layer = self.layer(uid)
        if layer is None:
            raise KeyError(f"no layer {uid}")
        layer.name = str(values["name"])
        layer.visible = bool(values["visible"])
        layer.opacity = float(values["opacity"])
        layer.properties = dict(values["properties"])

    # -- painting ------------------------------------------------------------

    def write_region(self, uid: int, x0: int, y0: int, after: np.ndarray) -> bool:
        """Put a rectangle of gids into a tile layer, undoably.

        Returns whether anything changed. The diff is taken *here* rather than
        in the tools, so that every path into a layer -- a stamp, a fill, a
        rectangle, a paste -- gets the no-op rule for free rather than each
        remembering to apply it.
        """
        layer = self.layer(uid)
        if not isinstance(layer, TileLayer):
            raise KeyError(f"no tile layer {uid}")
        block = np.ascontiguousarray(after, dtype=gidlib.DTYPE)
        if block.ndim != 2:
            raise ValueError("a tile region is two-dimensional")
        h, w = block.shape
        x0, y0 = int(x0), int(y0)
        if x0 < 0 or y0 < 0 or x0 + w > layer.width or y0 + h > layer.height:
            raise ValueError("that region falls outside the layer")
        before = layer.data[y0 : y0 + h, x0 : x0 + w]
        if np.array_equal(before, block):
            return False
        self.history.push(
            TilePatchEdit(layer_uid=int(uid), x0=x0, y0=y0, before=before, after=block)
        )
        self._blit(uid, x0, y0, block)
        return True

    def _blit(self, uid: int, x0: int, y0: int, block: np.ndarray) -> None:
        layer = self.layer(uid)
        if not isinstance(layer, TileLayer):
            raise KeyError(f"no tile layer {uid}")
        h, w = block.shape
        layer.data[y0 : y0 + h, x0 : x0 + w] = block

    # -- geometry ------------------------------------------------------------

    def resize(self, width: int, height: int, *, offset_x: int = 0, offset_y: int = 0) -> bool:
        """Change the grid, anchoring the old content at ``(offset_x, offset_y)``.

        A snapshot step rather than a patch: every layer's *shape* changes, so
        there is no rectangle that describes it. Objects move with the content,
        because their coordinates are absolute pixels and leaving them put would
        silently detach every trigger volume from the geometry it was drawn
        around.
        """
        width, height = _dimension(width, "width"), _dimension(height, "height")
        dx, dy = int(offset_x), int(offset_y)
        if (width, height) == (self.width, self.height) and (dx, dy) == (0, 0):
            return False

        before = {layer.uid: layer.data for layer in self.tile_layers()}
        after: dict[int, np.ndarray] = {}
        for uid, data in before.items():
            grown = gidlib.empty_layer(width, height)
            # The overlap of the old rectangle, shifted, with the new one.
            sx0, sy0 = max(0, -dx), max(0, -dy)
            tx0, ty0 = max(0, dx), max(0, dy)
            span_w = min(data.shape[1] - sx0, width - tx0)
            span_h = min(data.shape[0] - sy0, height - ty0)
            if span_w > 0 and span_h > 0:
                grown[ty0 : ty0 + span_h, tx0 : tx0 + span_w] = data[
                    sy0 : sy0 + span_h, sx0 : sx0 + span_w
                ]
            after[uid] = grown

        shift_x, shift_y = dx * self.tile_w, dy * self.tile_h
        before_objects: dict[int, list[tuple[float, float]]] = {}
        after_objects: dict[int, list[tuple[float, float]]] = {}
        for layer in self.layers:
            if not isinstance(layer, ObjectLayer):
                continue
            before_objects[layer.uid] = [(o.x, o.y) for o in layer.objects]
            after_objects[layer.uid] = [(o.x + shift_x, o.y + shift_y) for o in layer.objects]

        self.history.push(
            ResizeEdit(
                before_size=(self.width, self.height),
                after_size=(width, height),
                before=before,
                after=after,
                before_objects=before_objects,
                after_objects=after_objects,
            )
        )
        self._apply_resize((width, height), after, after_objects)
        return True

    def _apply_resize(
        self,
        size: tuple[int, int],
        data: dict[int, np.ndarray],
        objects: dict[int, list[tuple[float, float]]],
    ) -> None:
        self.width, self.height = int(size[0]), int(size[1])
        for uid, array in data.items():
            layer = self.layer(uid)
            if isinstance(layer, TileLayer):
                # A copy, or undo and the document would share one array and
                # the next stamp would write into the history.
                layer.data = array.copy()
        for uid, positions in objects.items():
            layer = self.layer(uid)
            if isinstance(layer, ObjectLayer):
                # ``strict``, like every other zip in this package. The two
                # lists are recorded from one walk of the same layer and
                # replayed in LIFO order, so a length mismatch means an object
                # arrived or left between the record and the replay -- at which
                # point the positions no longer describe these objects and
                # truncating quietly leaves the tail sitting on the old grid,
                # detached from the geometry it was drawn around. That is the
                # thing the resize moves objects to prevent.
                for obj, (x, y) in zip(layer.objects, positions, strict=True):
                    obj.x, obj.y = float(x), float(y)

    # -- objects -------------------------------------------------------------

    def add_object(self, layer_uid: int, obj: MapObject, *, index: int | None = None) -> MapObject:
        layer = self.layer(layer_uid)
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {layer_uid}")
        at = len(layer.objects) if index is None else max(0, min(int(index), len(layer.objects)))
        self.history.push(ObjectAddEdit(layer_uid=int(layer_uid), obj=obj, index=at))
        self._attach_object(layer_uid, obj, at)
        return obj

    def remove_object(self, layer_uid: int, obj_uid: int) -> None:
        layer = self.layer(layer_uid)
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {layer_uid}")
        for index, obj in enumerate(layer.objects):
            if obj.uid == obj_uid:
                self.history.push(
                    ObjectRemoveEdit(layer_uid=int(layer_uid), obj=obj, index=index)
                )
                self._detach_object(layer_uid, obj)
                return
        raise KeyError(f"no object {obj_uid}")

    def set_object(self, layer_uid: int, obj_uid: int, **values: Any) -> None:
        layer = self.layer(layer_uid)
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {layer_uid}")
        obj = next((o for o in layer.objects if o.uid == obj_uid), None)
        if obj is None:
            raise KeyError(f"no object {obj_uid}")
        before = obj.snapshot()
        after = {**before, **{k: v for k, v in values.items() if k in before}}
        if after == before:
            return
        if after["kind"] not in OBJECT_KINDS:
            raise ValueError(f"an object is one of {list(OBJECT_KINDS)}")
        self.history.push(
            ObjectPropsEdit(
                layer_uid=int(layer_uid), obj_uid=int(obj_uid), before=before, after=after
            )
        )
        self._apply_object_props(layer_uid, obj_uid, after)

    def _attach_object(self, layer_uid: int, obj: MapObject, index: int) -> None:
        layer = self.layer(layer_uid)
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {layer_uid}")
        layer.objects.insert(index, obj)

    def _detach_object(self, layer_uid: int, obj: MapObject) -> None:
        layer = self.layer(layer_uid)
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {layer_uid}")
        for index, entry in enumerate(layer.objects):
            if entry is obj:
                del layer.objects[index]
                return
        raise KeyError("that object is not on this layer")

    def _apply_object_props(self, layer_uid: int, obj_uid: int, values: dict[str, Any]) -> None:
        layer = self.layer(layer_uid)
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {layer_uid}")
        obj = next((o for o in layer.objects if o.uid == obj_uid), None)
        if obj is None:
            raise KeyError(f"no object {obj_uid}")
        obj.name = str(values["name"])
        obj.kind = str(values["kind"])
        obj.x, obj.y = float(values["x"]), float(values["y"])
        obj.w, obj.h = float(values["w"]), float(values["h"])
        obj.obj_class = str(values["obj_class"])
        obj.visible = bool(values["visible"])
        obj.properties = dict(values["properties"])

    # -- history -------------------------------------------------------------

    def push(self, edit: Edit) -> None:
        """For a caller assembling a :class:`CompoundEdit` of its own."""
        self.history.push(edit)

    def compound(self, edits: list[Edit]) -> None:
        """Push several steps that must undo together, or nothing if empty."""
        if edits:
            self.history.push(CompoundEdit(edits=list(edits)))

    def undo(self) -> bool:
        return self.history.undo(self)

    def redo(self) -> bool:
        return self.history.redo(self)


def _dimension(value: Any, what: str) -> int:
    size = int(value)
    if size < 1:
        raise ValueError(f"{what} must be at least 1")
    if size > MAX_DIMENSION:
        raise ValueError(f"{what} must be at most {MAX_DIMENSION}")
    return size
