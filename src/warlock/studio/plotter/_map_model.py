"""The plain data a map is made of: layers, objects, and what a dimension may be.

The leaf of the *document's* own dependency graph, and the reason the split
starts here. Everything the ``_map_*`` mixins do is *to* one of these types, so
a mixin that needed ``TileLayer`` from ``tilemap`` and ``tilemap`` that needed
the mixin would be a cycle; pulling the types out one level breaks it before it
exists. The one thing it reaches for is :mod:`.tileset`, which is a leaf of its
own (numpy and :mod:`.blob`) and knows nothing about layers -- for the pixel and
colour rules, ``frozen_rgba`` and ``rgba_colour``, rather than for a type. A
private copy of either beside :class:`ImageLayer` is how the two spellings of
"an image is RGBA and nothing may write through it" come to disagree.

Nothing here knows about history, and that is the line the file draws: a
:class:`TileLayer` is a rectangle of gids with a name and an opacity, and every
undoable *change* to one lives in :mod:`.edits` and the mixins. ``snapshot`` is
the one method that looks like an exception and is not -- it is what a layer
*is*, handed to an edit that decides what to do with it.

The ``Shape`` union lives here too, beside the object that carries one, rather
than in a module of its own. It is data with no behaviour and exactly one
consumer, and this package's import set is pinned file by file
(``tests/plotter/test_plotter_imports.py``): a new module for eight frozen
dataclasses would buy a roster entry and a second place to look for what an
object is.

The uid counter is per process, not per document, and there is no reserve step:
``.wmap`` stores indices and mints a fresh uid on read, so nothing is ever
restored onto a number the counter has already issued. One namespace covers
layers *and* objects, because both are undo subjects and a single counter makes
"is this uid a layer or an object" a question nothing has to ask.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..tilegrid.tileset import RGBA, frozen_rgba, rgba_colour

_uids = itertools.count(1)

# What the *editor* can author, which is not the same question as what an
# object may be. The document and canvas model all eight of Tiled's core
# geometries (see ``Shape`` below); this is also the set accepted by the
# ``kind=`` compatibility constructor.
OBJECT_KINDS = (
    "rect",
    "point",
    "ellipse",
    "capsule",
    "polygon",
    "polyline",
    "tile",
    "text",
)

# How an object layer asks to be drawn. Tiled's two, spelled its way.
DRAW_ORDERS = ("topdown", "index")

# What a custom property may be is :mod:`.props`' single answer -- this module
# carried a byte-identical second copy of that tuple, re-exported through
# ``tilemap`` and read by nothing, which is exactly how the two would have come
# to disagree the first time one of them gained a type.

MAX_DIMENSION = 4096

# How far past its own edge one *painted* growth may reach, per side.
#
# ``MAX_DIMENSION`` bounds where an infinite map can end up; it says nothing
# about how it gets there, and one click is enough. The canvas maps a screen
# position to a cell, so zoomed far enough out a single pixel of cursor travel
# is hundreds of cells: a click in the wrong place asked ``grow_to_hold`` to
# reach it, and reaching it means reallocating every tile layer to the full
# 4096 a side. Measured, on a 4-by-4096 map with **one** tile layer, one click
# at cell 4095: **192 MiB peak** between two frames -- the 64 MiB grid itself,
# plus the before and after snapshots the ``ResizeEdit`` needs to make the
# growth undoable. Per layer, for a gesture the user did not mean. Refused, the
# same click costs 1 KiB.
#
# Deliberately per *side* rather than per added area: a side is what a refusal
# can say out loud ("you clicked N cells past the edge") and an area is not.
# 256 columns of a 4096-tall map is ~4 MiB a layer a grid, which is a real
# growth and not a catastrophic one -- and nothing legitimate needs it in one
# press. A drag grows a cell at a time (``_room_for`` runs per painted cell),
# an explicit resize goes through the form and its own cap, and a click that
# genuinely wants to be far away can be made twice.
MAX_GROWTH = 256


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


# --- what an object's geometry is ---------------------------------------------
#
# Tiled's eight object geometries as a tagged union of **frozen** dataclasses,
# and frozen is the load-bearing word. An object edit snapshots the object, a
# drag snapshots it once per frame, and undo holds both sides of every step: a
# mutable shape would have to be deep-copied at each of those points or the
# "before" would change with the document and restore nothing. Immutable, the
# same shape object is safely shared by the document, the open drag session and
# every step on the stack, so ``snapshot`` costs one reference.
#
# Size lives on the shapes that have one rather than on the object, because the
# four that do not -- a point, a polygon, a polyline -- have no width in any
# sense: Tiled writes none, and a stored zero would be a number somebody
# eventually resizes.


def _size(w: Any, h: Any, what: str) -> tuple[float, float]:
    """One shape's size, refused if negative.

    A negative extent is not a drawing that happens to look wrong: it exports
    as a rectangle no engine can read, and the canvas already normalizes a
    corner dragged past its opposite, so nothing legitimate produces one.
    """
    width, height = float(w), float(h)
    if width < 0 or height < 0:
        raise ValueError(f"a {what}'s width and height cannot be negative")
    return width, height


def _points(raw: Any, least: int, what: str) -> tuple[tuple[float, float], ...]:
    """A vertex list as the hashable, comparable form the shapes store.

    Coordinates are **origin-relative**, Tiled's convention: the object's
    ``x``/``y`` is the first vertex's place on the map and every point is an
    offset from it, so moving the object is one addition rather than a walk.
    """
    points = tuple((float(px), float(py)) for px, py in raw)
    if len(points) < least:
        raise ValueError(f"a {what} needs at least {least} points, not {len(points)}")
    return points


@dataclass(frozen=True)
class Rect:
    w: float = 0.0
    h: float = 0.0

    def __post_init__(self) -> None:
        width, height = _size(self.w, self.h, "rect")
        object.__setattr__(self, "w", width)
        object.__setattr__(self, "h", height)


@dataclass(frozen=True)
class Point:
    """A place with no extent. Tiled writes ``<point/>`` and no size."""


@dataclass(frozen=True)
class Ellipse:
    w: float = 0.0
    h: float = 0.0

    def __post_init__(self) -> None:
        width, height = _size(self.w, self.h, "ellipse")
        object.__setattr__(self, "w", width)
        object.__setattr__(self, "h", height)


@dataclass(frozen=True)
class Capsule:
    """Tiled 1.12's capsule, bounded by the object's width and height."""

    w: float = 0.0
    h: float = 0.0

    def __post_init__(self) -> None:
        width, height = _size(self.w, self.h, "capsule")
        object.__setattr__(self, "w", width)
        object.__setattr__(self, "h", height)


@dataclass(frozen=True)
class Polygon:
    points: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", _points(self.points, 3, "polygon"))


@dataclass(frozen=True)
class Polyline:
    points: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.0, 0.0))

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", _points(self.points, 2, "polyline"))


@dataclass(frozen=True)
class TileShape:
    """An object that *is* a tile: one gid, drawn at a size of its own.

    The flip flags ride in the gid exactly as they do in a cell -- one number,
    flags in the top three bits, nothing between here and the renderer strips
    them -- so :mod:`.gid`'s ``compose``/``decompose`` are the whole encoding
    for both. Named ``TileShape`` rather than ``Tile`` because
    :mod:`.tileset` already owns that word.
    """

    gid: int = 0
    w: float = 0.0
    h: float = 0.0

    def __post_init__(self) -> None:
        width, height = _size(self.w, self.h, "tile object")
        object.__setattr__(self, "gid", int(self.gid))
        object.__setattr__(self, "w", width)
        object.__setattr__(self, "h", height)


@dataclass(frozen=True)
class Text:
    """A text object, with Tiled 1.12.2's own defaults field for field.

    Copied rather than chosen: a default that differs from Tiled's would make
    every *unstyled* text object export as a styled one, because a writer only
    emits the attributes that differ from the default.
    """

    text: str = ""
    w: float = 0.0
    h: float = 0.0
    family: str = "sans-serif"
    pixel_size: int = 16
    wrap: bool = False
    color: str = "#000000"
    halign: str = "left"
    valign: str = "top"
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikeout: bool = False
    kerning: bool = True

    def __post_init__(self) -> None:
        width, height = _size(self.w, self.h, "text object")
        object.__setattr__(self, "w", width)
        object.__setattr__(self, "h", height)


Shape = Rect | Point | Ellipse | Capsule | Polygon | Polyline | TileShape | Text

#: Every shape and the name it answers to, which is Tiled's name for it. The
#: tuple form (rather than ``type.__name__.lower()``) is what lets
#: ``TileShape`` report ``"tile"``: the class is only spelled that way to keep
#: clear of :mod:`.tileset`'s ``Tile``, and the *document* should not inherit a
#: naming collision's fingerprint.
SHAPE_KINDS: tuple[tuple[type, str], ...] = (
    (Rect, "rect"),
    (Point, "point"),
    (Ellipse, "ellipse"),
    (Capsule, "capsule"),
    (Polygon, "polygon"),
    (Polyline, "polyline"),
    (TileShape, "tile"),
    (Text, "text"),
)

SHAPES = tuple(cls for cls, _ in SHAPE_KINDS)


def shape_kind(shape: Shape) -> str:
    for cls, name in SHAPE_KINDS:
        if type(shape) is cls:
            return name
    raise ValueError(f"{shape!r} is not one of this document's object shapes")


def shape_size(shape: Shape) -> tuple[float, float]:
    """``(w, h)``, or zeros for the three shapes that have no extent."""
    return (float(getattr(shape, "w", 0.0)), float(getattr(shape, "h", 0.0)))


def resize_shape(shape: Shape, w: Any, h: Any) -> Shape:
    """The same shape at a new size; unchanged if it has no size to set.

    The canvas resizes by dragging a corner and knows only ``w``/``h``, so this
    is what keeps an ellipse an ellipse through a handle drag.
    """
    if not hasattr(shape, "w"):
        return shape
    width, height = float(w), float(h)
    if (shape.w, shape.h) == (width, height):
        return shape
    return replace(shape, w=width, h=height)


def scaled_shape(shape: Shape, scale_x: float, scale_y: float) -> Shape:
    """The same shape with every length multiplied through.

    For the map's *tile size* changing, where the grid keeps its cell counts and
    each cell changes size: an object drawn two cells wide should still be two
    cells wide afterwards, so its extent scales exactly as its position does.

    Both spellings of an extent are handled because the union has both -- four
    shapes state ``w``/``h`` and two state vertices -- and a helper that knew
    only about ``w``/``h`` would silently leave every polygon at its old size,
    which is the one case where the map visibly stops matching itself.
    ``Point`` has no extent and comes back untouched.
    """
    if (scale_x, scale_y) == (1.0, 1.0):
        return shape
    if hasattr(shape, "points"):
        return replace(
            shape, points=tuple((x * scale_x, y * scale_y) for x, y in shape.points)
        )
    if hasattr(shape, "w"):
        return replace(shape, w=shape.w * scale_x, h=shape.h * scale_y)
    return shape


def shape_for_kind(kind: str, w: Any = 0.0, h: Any = 0.0) -> Shape:
    """A useful default geometry for every shape the editor can author."""
    width, height = _size(w, h, f"{kind} object")
    if kind == "rect":
        return Rect(width, height)
    if kind == "point":
        return Point()
    if kind == "ellipse":
        return Ellipse(width, height)
    if kind == "capsule":
        return Capsule(width, height)
    if kind == "polygon":
        return Polygon(((0.0, 0.0), (width, 0.0), (width, height), (0.0, height)))
    if kind == "polyline":
        return Polyline(((0.0, 0.0), (width, height)))
    if kind == "tile":
        return TileShape(gid=0, w=width, h=height)
    if kind == "text":
        return Text("Text", width, height)
    raise ValueError(f"an object is one of {list(OBJECT_KINDS)}, not {kind!r}")


@dataclass(init=False)
class MapObject:
    """One named shape on an object layer.

    Coordinates are in **pixels**, not tiles, and that is Tiled's convention
    rather than a choice made here: an object is routinely placed off the grid
    on purpose -- a spawn point half a tile in from a wall -- and a tile-space
    position would have no way to say so.

    ``kind``, ``w`` and ``h`` are **derived** from :attr:`shape` and read-only.
    They were fields, and the panes, both writers and a good deal of the test
    suite still read them; keeping them as properties is what let the geometry
    become a tagged union in one step instead of a rewrite of every caller.
    Read-only because two spellings of one fact must not be separately
    settable -- ``obj.w = 5`` on a stored field would leave the shape saying
    something else, and the one code path that forgot to update both is the
    whole bug class. Write geometry through ``shape=``, or through
    ``set_object``/``place_object``, which reconcile the two spellings in one
    place (:func:`merged_object_values`).

    The constructor still takes the old ``kind``/``w``/``h`` form -- the whole
    package builds objects that way -- but not *together* with ``shape``:
    saying the geometry twice in one call is two chances to disagree.
    """

    # These annotations are the *field list* -- what ``fields()``, ``repr``,
    # ``__eq__`` and ``dataclasses.replace`` work from. They carry no defaults,
    # because under ``init=False`` no generated constructor could ever consult
    # one: the defaults callers actually get are in ``__init__``'s signature
    # below, and a second set here would be code that cannot run sitting where
    # a reader looks for the answer.
    # ``test_shapes.py`` pins the two lists against each other, since nothing
    # else would notice a field added here and not there.
    uid: int
    # Tiled's own persistent object id -- distinct from ``uid`` above, which
    # is a per-process address undo steps are written against and means
    # nothing outside this run. ``0`` is "unassigned yet"; ``MapDoc`` mints a
    # real one from ``next_object_id`` at creation, monotone and never
    # reused, because an object-typed property may go on naming this id after
    # the object it named is deleted. See ``PLOTTER_PLAN.md`` Milestone 2.
    # Not in ``snapshot()``: an id is the object's identity, not a prop an
    # edit may rewrite -- add/remove edits hold the object, so it survives.
    id: int
    name: str
    x: float
    y: float
    obj_class: str
    # Tiled lets an individual object be hidden. Modelled rather than refused,
    # because a hidden object is still exactly where it is -- the picture stays
    # right, and drawing it faintly is one branch in the canvas.
    visible: bool
    # Tiled 1.12 lets each object fade independently of its layer. Kept as a
    # scalar on the object because it affects tile/text rendering but not the
    # geometry used for hit testing.
    opacity: float
    properties: dict[str, Any]
    # Degrees clockwise about the object's origin, Tiled's own sense.
    rotation: float
    shape: Shape

    def __init__(
        self,
        uid: int,
        name: str = "",
        kind: str | None = None,
        x: float = 0.0,
        y: float = 0.0,
        w: float = 0.0,
        h: float = 0.0,
        obj_class: str = "",
        visible: bool = True,
        opacity: float = 1.0,
        properties: dict[str, Any] | None = None,
        rotation: float = 0.0,
        shape: Shape | None = None,
        id: int = 0,
    ) -> None:
        # Hand-written rather than generated, because the parameters are not
        # the fields: ``kind``/``w``/``h`` go *in* and are never stored, and
        # ``dataclass`` has no way to say that. Positional order is the order
        # it always was, so every existing call site still reads the same --
        # which is why ``id``, a later arrival, sits last rather than beside
        # ``uid`` where the field list keeps it.
        if shape is not None and (kind is not None or w or h):
            raise ValueError("an object takes either a shape or kind/w/h, not both")
        self.uid = uid
        self.id = id
        self.name = name
        self.x = x
        self.y = y
        self.obj_class = obj_class
        self.visible = visible
        self.opacity = opacity
        self.properties = {} if properties is None else properties
        self.rotation = rotation
        if shape is None:
            shape = shape_for_kind("rect" if kind is None else kind, w, h)
        self.shape = shape
        self.__post_init__()

    def __post_init__(self) -> None:
        # Called by hand from ``__init__`` above -- ``init=False`` means
        # ``dataclass`` generates no caller -- and by nothing else except
        # ``dataclasses.replace``, which is exactly where it is also wanted.
        if not isinstance(self.shape, SHAPES):
            raise ValueError(
                f"an object's shape is one of {[name for _, name in SHAPE_KINDS]}, "
                f"not {type(self.shape).__name__}"
            )
        self.x, self.y = float(self.x), float(self.y)
        self.rotation = float(self.rotation)
        self.opacity = float(self.opacity)
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("an object's opacity must be between 0 and 1")

    @property
    def kind(self) -> str:
        return shape_kind(self.shape)

    @property
    def w(self) -> float:
        return shape_size(self.shape)[0]

    @property
    def h(self) -> float:
        return shape_size(self.shape)[1]

    def snapshot(self) -> dict[str, Any]:
        """Everything :class:`~.edits.ObjectPropsEdit` restores.

        ``kind``/``w``/``h`` travel beside ``shape`` as the derived echo they
        are, so that a caller which only knows the old vocabulary -- the
        canvas resizes by ``w``/``h`` -- can still hand ``set_object`` a
        change. :func:`merged_object_values` is what makes the two agree
        again; nothing reads the echo back onto the object.
        """
        return {
            "name": self.name,
            "kind": self.kind,
            "x": float(self.x),
            "y": float(self.y),
            "w": float(self.w),
            "h": float(self.h),
            "rotation": float(self.rotation),
            # By reference. The shape is frozen, so the snapshot, the document
            # and every undo step can hold the same one.
            "shape": self.shape,
            "obj_class": self.obj_class,
            "visible": bool(self.visible),
            "opacity": float(self.opacity),
            "properties": dict(self.properties),
        }


def merged_object_values(before: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """``before`` with ``values`` written over it, geometry reconciled.

    The one place the two spellings of an object's geometry are resolved
    against each other, which is why ``set_object`` and ``place_object`` both
    go through it rather than each spreading a dict. The rules:

    * ``shape=`` says it exactly, and may not arrive with ``kind``/``w``/``h``;
    * ``kind=`` rebuilds a useful default of that core shape;
    * ``w``/``h`` alone resize whatever shape is already there;
    * and the ``kind``/``w``/``h`` keys of the result are then re-derived from
      the shape that won, so the dict handed to an edit is self-consistent
      whichever door it came through.
    """
    given = {k: v for k, v in values.items() if k in before}
    sized = "w" in given or "h" in given
    if "shape" in given:
        if "kind" in given or sized:
            raise ValueError("an object takes either a shape or kind/w/h, not both")
        shape = given["shape"]
        if not isinstance(shape, SHAPES):
            raise ValueError(f"{shape!r} is not one of this document's object shapes")
    elif "kind" in given:
        shape = shape_for_kind(
            str(given["kind"]),
            given.get("w", before["w"]),
            given.get("h", before["h"]),
        )
    elif sized:
        shape = resize_shape(
            before["shape"], given.get("w", before["w"]), given.get("h", before["h"])
        )
    else:
        shape = before["shape"]
    after = {**before, **given, "shape": shape}
    after["opacity"] = float(after["opacity"])
    if not 0.0 <= after["opacity"] <= 1.0:
        raise ValueError("an object's opacity must be between 0 and 1")
    after["rotation"] = float(after["rotation"])
    after["kind"] = shape_kind(shape)
    after["w"], after["h"] = shape_size(shape)
    return after


# --- what every layer kind carries --------------------------------------------
#
# Tiled decorates *every* layer, whatever kind it is, with the same six things,
# and this package spells them the same way for all four. They are declared
# field by field on each class rather than inherited from a base, because a
# dataclass base would have to carry them with defaults and every default field
# must precede every non-default one -- so ``TileLayer.data``, which has no
# sensible default, could not follow them. What keeps the four in step instead
# is this tuple and the two helpers under it: ``snapshot`` and
# ``_apply_layer_props`` both work from the same list, and
# ``tests/plotter/test_scene.py`` asserts all four classes carry it.

#: The tint that changes nothing, and therefore the default: Tiled's own
#: ``#ffffffff``. Multiplying by it is the identity in every channel, which is
#: what lets :mod:`.scene` combine tints unconditionally.
OPAQUE_WHITE: RGBA = (255, 255, 255, 255)

DECORATION_FIELDS = (
    "class_name",
    "blend_mode",
    "tint",
    "offset_x",
    "offset_y",
    "parallax_x",
    "parallax_y",
)

# Tiled 1.12's layer blend modes, in the order used by its UI and format
# documentation. ``normal`` is the identity/default.
BLEND_MODES = (
    "normal",
    "add",
    "multiply",
    "screen",
    "overlay",
    "darken",
    "lighten",
    "color-dodge",
    "color-burn",
    "hard-light",
    "soft-light",
    "difference",
    "exclusion",
)


def _normalize_layer(layer: Any) -> None:
    """Coerce and refuse the shared fields. Called from every kind's post-init.

    ``class_name`` rather than ``class``, which is a keyword, and rather than
    ``obj_class``, which is :class:`MapObject`'s field and would suggest the two
    were one thing: Tiled's ``class`` attribute means the same on both, but a
    layer's and an object's are separate values.
    """
    layer.class_name = str(layer.class_name)
    layer.blend_mode = str(layer.blend_mode)
    if layer.blend_mode not in BLEND_MODES:
        raise ValueError(
            f"a layer blend mode is one of {list(BLEND_MODES)}, not {layer.blend_mode!r}"
        )
    layer.tint = rgba_colour(layer.tint, "a layer tint")
    layer.offset_x = float(layer.offset_x)
    layer.offset_y = float(layer.offset_y)
    layer.parallax_x = float(layer.parallax_x)
    layer.parallax_y = float(layer.parallax_y)


def normalize_layer_values(values: dict[str, Any]) -> dict[str, Any]:
    """One layer-props dict coerced and refused, **at the door**.

    :meth:`~._map_layers.LayerOps.set_layer_props` builds its ``after`` by
    spreading whatever a caller handed in over a snapshot, and then hands that
    dict to an edit *and* to ``_apply_layer_props``. Both of those trusted it:
    the constructor refuses a tint of ``(999, 0, 0, 255)`` and a tint of
    ``(1, 2)``, and nothing stood between ``set_layer_props(uid, tint=…)`` and
    the field, so the setter accepted what the constructor would not. The second
    of those two did not even fail there -- it failed later and elsewhere, in
    :func:`.scene._tint_product`'s strict ``zip``, on the frame thread, one
    resolve after the value was stored.

    So the same coercion :func:`_normalize_layer` applies at construction is
    applied here, and it runs **before** the no-op comparison rather than after:
    a caller spelling an unchanged tint as a list has changed nothing, and a
    dict that compared unequal only by type would push a step for it. Refusing
    here also keeps the raise on the *near* side of ``history.push`` -- a step
    that raises halfway through leaves the stack describing a change the
    document never made, which is the argument the ``draworder`` refusal beside
    it already makes.
    """
    out = dict(values)
    out["name"] = str(out["name"])
    out["class_name"] = str(out["class_name"])
    out["blend_mode"] = str(out["blend_mode"])
    if out["blend_mode"] not in BLEND_MODES:
        raise ValueError(
            f"a layer blend mode is one of {list(BLEND_MODES)}, not {out['blend_mode']!r}"
        )
    out["visible"] = bool(out["visible"])
    out["opacity"] = float(out["opacity"])
    out["locked"] = bool(out["locked"])
    out["tint"] = rgba_colour(out["tint"], "a layer tint")
    for name in ("offset_x", "offset_y", "parallax_x", "parallax_y"):
        out[name] = float(out[name])
    return out


def _layer_snapshot(layer: Any) -> dict[str, Any]:
    """Everything :class:`~.edits.LayerPropsEdit` restores on any layer kind.

    What is deliberately absent is the layer's *contents* -- a tile layer's
    array, a group's children, an image layer's pixels -- which travel with
    add/remove edits instead, and the persistent ``id``, which is the layer's
    identity rather than a property an edit may rewrite.
    """
    return {
        "name": layer.name,
        "class_name": str(layer.class_name),
        "blend_mode": str(layer.blend_mode),
        "visible": bool(layer.visible),
        "opacity": float(layer.opacity),
        "locked": bool(layer.locked),
        "tint": tuple(layer.tint),
        "offset_x": float(layer.offset_x),
        "offset_y": float(layer.offset_y),
        "parallax_x": float(layer.parallax_x),
        "parallax_y": float(layer.parallax_y),
        "properties": dict(layer.properties),
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
    # The six every layer kind carries; see ``DECORATION_FIELDS`` above. The
    # comment sits here and nowhere else, because repeating it on all four
    # classes is how one copy comes to describe something the others do not do.
    # ``offset`` is in pixels and sums down the tree, ``parallax`` is a factor
    # against the camera and multiplies, ``tint`` multiplies per channel.
    class_name: str = ""
    blend_mode: str = "normal"
    tint: RGBA = OPAQUE_WHITE
    offset_x: float = 0.0
    offset_y: float = 0.0
    parallax_x: float = 1.0
    parallax_y: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _normalize_layer(self)

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    def snapshot(self) -> dict[str, Any]:
        return _layer_snapshot(self)


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
    # Tiled's two orders for *drawing* the list, not for storing it: the list
    # is already the manual stacking order, and ``"topdown"`` asks the renderer
    # to sort by ``y`` on top of that so a sprite lower on the map occludes one
    # behind it. Modelled here because it is a property of the document that
    # survives a save; whether a renderer honours it is the renderer's
    # question.
    draworder: str = "topdown"
    # Tiled's editor/display colour for shape outlines. It is not multiplied
    # into tile objects (that is ``tint``); keeping it separate prevents an
    # imported metadata colour from changing rendered pixels.
    color: str | None = None
    class_name: str = ""
    blend_mode: str = "normal"
    tint: RGBA = OPAQUE_WHITE
    offset_x: float = 0.0
    offset_y: float = 0.0
    parallax_x: float = 1.0
    parallax_y: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.draworder not in DRAW_ORDERS:
            raise ValueError(
                f"a draw order is one of {list(DRAW_ORDERS)}, not {self.draworder!r}"
            )
        self.color = None if self.color in (None, "") else str(self.color)
        _normalize_layer(self)

    def snapshot(self) -> dict[str, Any]:
        return {
            **_layer_snapshot(self),
            "draworder": str(self.draworder),
            "color": self.color,
        }


@dataclass
class GroupLayer:
    """A layer that holds layers. It draws nothing of its own.

    **A group is a state its descendants inherit, not a surface anything is
    composited onto.** Tiled's own semantics, and the distinction decides the
    whole design: if a group were a surface, its opacity would apply to the
    *flattened* subtree and two half-opaque siblings inside it would not show
    through each other. They do, here and in Tiled, because the group's numbers
    reach each leaf separately -- which is exactly what :func:`.scene.resolve`
    computes and why nothing else is allowed to work it out for itself.

    ``children`` is not in :meth:`snapshot`, for the reason ``data`` is not in
    :class:`TileLayer`'s: it is the layer's *contents*, and contents travel with
    the add/remove edits that hold the layer object itself.
    """

    uid: int
    name: str
    id: int = 0
    children: list[Layer] = field(default_factory=list)
    visible: bool = True
    opacity: float = 1.0
    locked: bool = False
    class_name: str = ""
    blend_mode: str = "normal"
    tint: RGBA = OPAQUE_WHITE
    offset_x: float = 0.0
    offset_y: float = 0.0
    parallax_x: float = 1.0
    parallax_y: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _normalize_layer(self)

    def snapshot(self) -> dict[str, Any]:
        return _layer_snapshot(self)


def _no_image() -> np.ndarray:
    """The picture an image layer has before it has one.

    Zero-sized rather than ``None``, so every consumer can ask an image layer
    for its pixels' shape without first asking whether there are any -- and a
    ``(0, 0, 4)`` array is still RGBA, so :func:`~.tileset.frozen_rgba`'s
    refusal keeps meaning what it says.
    """
    return frozen_rgba(np.zeros((0, 0, 4), dtype=np.uint8), "an image layer's picture")


@dataclass
class ImageLayer:
    """One picture placed on the map: Tiled's ``<imagelayer>``.

    The pixels are frozen on construction, the :class:`~.tileset.Tileset` rule
    for the :class:`~.tileset.Tileset` reason -- the UI keys a texture upload on
    the array's identity, so an in-place edit would leave the cache holding a
    live key over stale pixels.

    ``source`` is where the picture came from, carried verbatim and never
    resolved (the host's problem, exactly as a ``file`` property's path is), and
    ``repeat_x``/``repeat_y`` are Tiled's two tiling flags: a repeating image
    layer fills the map in that axis rather than sitting once at its offset.
    """

    uid: int
    name: str
    id: int = 0
    pixels: np.ndarray = field(default_factory=_no_image)
    source: str = ""
    repeat_x: bool = False
    repeat_y: bool = False
    visible: bool = True
    opacity: float = 1.0
    locked: bool = False
    class_name: str = ""
    blend_mode: str = "normal"
    tint: RGBA = OPAQUE_WHITE
    offset_x: float = 0.0
    offset_y: float = 0.0
    parallax_x: float = 1.0
    parallax_y: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pixels = frozen_rgba(self.pixels, "an image layer's picture")
        self.source = str(self.source)
        self.repeat_x, self.repeat_y = bool(self.repeat_x), bool(self.repeat_y)
        _normalize_layer(self)

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    def snapshot(self) -> dict[str, Any]:
        return {
            **_layer_snapshot(self),
            "source": str(self.source),
            "repeat_x": bool(self.repeat_x),
            "repeat_y": bool(self.repeat_y),
        }


Layer = TileLayer | ObjectLayer | GroupLayer | ImageLayer

#: The three kinds that are *drawn* -- everything a resolver yields and every
#: renderer's loop body. A group is absent because it draws nothing; asking
#: ``isinstance(layer, LEAF_LAYERS)`` rather than ``not isinstance(layer,
#: GroupLayer)`` is what keeps a fifth kind from silently arriving as a leaf.
LEAF_LAYERS = (TileLayer, ObjectLayer, ImageLayer)
