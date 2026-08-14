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

The ``Shape`` union lives here too, beside the object that carries one, rather
than in a module of its own. It is data with no behaviour and exactly one
consumer, and this package's import set is pinned file by file
(``tests/plotter/test_plotter_imports.py``): a new module for seven frozen
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

_uids = itertools.count(1)

# What the *editor* can author, which is not the same question as what an
# object may be. The document models all seven of Tiled's geometries (see
# ``Shape`` below); this is the subset the canvas has a gesture for and the
# only one the ``kind=`` compat constructor accepts. An ellipse is
# *constructed* -- ``MapObject(shape=Ellipse(...))`` -- never spelled, which
# keeps "a kind string nothing can draw" a refusal while the model stays
# complete.
OBJECT_KINDS = ("rect", "point")

# How an object layer asks to be drawn. Tiled's two, spelled its way.
DRAW_ORDERS = ("topdown", "index")

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


def _dimension(value: Any, what: str) -> int:
    size = int(value)
    if size < 1:
        raise ValueError(f"{what} must be at least 1")
    if size > MAX_DIMENSION:
        raise ValueError(f"{what} must be at most {MAX_DIMENSION}")
    return size


# --- what an object's geometry is ---------------------------------------------
#
# Tiled's seven object geometries as a tagged union of **frozen** dataclasses,
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


Shape = Rect | Point | Ellipse | Polygon | Polyline | TileShape | Text

#: Every shape and the name it answers to, which is Tiled's name for it. The
#: tuple form (rather than ``type.__name__.lower()``) is what lets
#: ``TileShape`` report ``"tile"``: the class is only spelled that way to keep
#: clear of :mod:`.tileset`'s ``Tile``, and the *document* should not inherit a
#: naming collision's fingerprint.
SHAPE_KINDS: tuple[tuple[type, str], ...] = (
    (Rect, "rect"),
    (Point, "point"),
    (Ellipse, "ellipse"),
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


def shape_for_kind(kind: str, w: Any = 0.0, h: Any = 0.0) -> Shape:
    """The compat door: a shape from the two kinds the editor can author.

    Deliberately narrower than :data:`SHAPE_KINDS`. The other five cannot be
    built from a kind and a size at all -- a polygon needs vertices, a tile
    object needs a gid -- and accepting the two that *could* be
    (``ellipse``/``text``) would leave a string parameter as a second way to
    say something the ``shape=`` argument already says exactly.
    """
    if kind == "rect":
        return Rect(*_size(w, h, "rect"))
    if kind == "point":
        return Point()
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

    uid: int
    name: str = ""
    x: float = 0.0
    y: float = 0.0
    obj_class: str = ""
    # Tiled lets an individual object be hidden. Modelled rather than refused,
    # because a hidden object is still exactly where it is -- the picture stays
    # right, and drawing it faintly is one branch in the canvas.
    visible: bool = True
    properties: dict[str, Any] = field(default_factory=dict)
    # Degrees clockwise about the object's origin, Tiled's own sense. The
    # document models it; :mod:`.tmx` still refuses a rotated object at the
    # door, because an unrotated outline drawn for a rotated object is a
    # *wrong* picture and a wrong picture is worse than a refusal. This is the
    # field that refusal will be flipped onto.
    rotation: float = 0.0
    shape: Shape = field(default_factory=Rect)

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
        properties: dict[str, Any] | None = None,
        rotation: float = 0.0,
        shape: Shape | None = None,
    ) -> None:
        # Hand-written rather than generated, because the parameters are not
        # the fields: ``kind``/``w``/``h`` go *in* and are never stored, and
        # ``dataclass`` has no way to say that. Positional order is the order
        # it always was, so every existing call site still reads the same.
        if shape is not None and (kind is not None or w or h):
            raise ValueError("an object takes either a shape or kind/w/h, not both")
        self.uid = uid
        self.name = name
        self.x = x
        self.y = y
        self.obj_class = obj_class
        self.visible = visible
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
            "properties": dict(self.properties),
        }


def merged_object_values(before: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """``before`` with ``values`` written over it, geometry reconciled.

    The one place the two spellings of an object's geometry are resolved
    against each other, which is why ``set_object`` and ``place_object`` both
    go through it rather than each spreading a dict. The rules:

    * ``shape=`` says it exactly, and may not arrive with ``kind``/``w``/``h``;
    * ``kind=`` rebuilds the shape (and is still held to the two kinds the
      editor can author);
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
    after["kind"] = shape_kind(shape)
    after["w"], after["h"] = shape_size(shape)
    return after


@dataclass
class TileLayer:
    uid: int
    name: str
    data: np.ndarray
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
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.draworder not in DRAW_ORDERS:
            raise ValueError(
                f"a draw order is one of {list(DRAW_ORDERS)}, not {self.draworder!r}"
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "visible": bool(self.visible),
            "opacity": float(self.opacity),
            "locked": bool(self.locked),
            "draworder": str(self.draworder),
            "properties": dict(self.properties),
        }


Layer = TileLayer | ObjectLayer
