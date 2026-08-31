"""Click regions and drags in *tile-pixel* space.

One tile drawn at 8-16x is the second canvas this app has, and it is a much
smaller problem than the map's: **no rotation, no layer offset, no pan and no
zoom** -- the square on screen is the tile and nothing else is ever in it. So
this is a local implementation rather than a reach for ``plotter_canvas``'s
machinery, whose every helper carries a ``view``, an ``origin`` and a rotation
that would all be identity here.

Two things live here, and they are separate on purpose:

* :class:`TileView` -- the *only* place a screen coordinate becomes a tile
  pixel or the other way round. Every hit test below takes tile pixels, so a
  caller converts once at the top of a frame and never again. A second
  conversion is how a handle comes to be drawn somewhere it cannot be clicked.
* :func:`nearest_region` -- the generic picker. A *region* is a key and a
  point, and the nearest one within a radius wins. Collision handles and
  polygon vertices are the two callers today; a Wang corner or edge marker is
  the same shape of question about the same square, and this is what it should
  ask rather than growing a second picker.

Everything is pure and every write returns a **new** frozen shape: the caller
decides whether that goes through the undoable door live or at the release.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .tileset import TileEllipse, TilePolygon, TileRect

#: The eight box handles, clockwise from the top left. Named by compass point
#: because that is what says which edges move: ``"nw"`` moves the left and the
#: top, ``"n"`` moves only the top.
BOX_HANDLES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")

#: How near the pointer has to be to a handle or a vertex, in *screen* pixels.
#: Screen rather than tile pixels for the reason the view is drawn at 16x at
#: all: the shapes are in tile pixels, so a radius of one tile pixel is half a
#: mouse-hair on a 16 px tile and a quarter of the tile on a 4 px one.
GRAB_RADIUS = 8.0

#: The smallest a box or an ellipse may be dragged to, in tile pixels. A shape
#: with a zero-sized side is a shape whose handles are all in the same place,
#: which is a shape that can be created and never grabbed again.
MIN_SIDE = 1.0

#: A polygon needs three points to enclose anything, so the third-from-last
#: removal is refused rather than silently leaving a line behind.
MIN_POINTS = 3


@dataclass(frozen=True)
class TileView:
    """The square one tile is drawn in, and the two conversions it owns.

    The tile is *fitted* into the square rather than stretched across it, so a
    16 x 32 tile is drawn half as wide as it is tall and a click at its right
    edge lands on tile pixel 16 rather than on 32. One scale for both axes is
    what makes that true, and it is also what makes a circular ellipse look
    circular.
    """

    origin: tuple[float, float] = (0.0, 0.0)
    side: float = 256.0
    tile_w: int = 16
    tile_h: int = 16

    @property
    def scale(self) -> float:
        """Screen pixels per tile pixel."""
        return float(self.side) / max(1, int(self.tile_w), int(self.tile_h))

    @property
    def size(self) -> tuple[float, float]:
        """How big the tile is actually drawn, which is not always the square."""
        scale = self.scale
        return (max(1, int(self.tile_w)) * scale, max(1, int(self.tile_h)) * scale)

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        scale = self.scale
        return (self.origin[0] + float(x) * scale, self.origin[1] + float(y) * scale)

    def to_tile(self, sx: float, sy: float) -> tuple[float, float]:
        scale = self.scale or 1.0
        return ((float(sx) - self.origin[0]) / scale, (float(sy) - self.origin[1]) / scale)

    def contains(self, sx: float, sy: float) -> bool:
        """Whether a screen point is inside the *tile*, not the square."""
        w, h = self.size
        return (
            self.origin[0] <= float(sx) <= self.origin[0] + w
            and self.origin[1] <= float(sy) <= self.origin[1] + h
        )


def nearest_region(
    regions: Mapping[Any, tuple[float, float]],
    at: tuple[float, float],
    radius: float,
) -> Any | None:
    """The key of the region nearest ``at`` within ``radius``, or ``None``.

    Nearest rather than first-within, because handles overlap on a small shape
    and "the first one in the dict" is an answer that depends on how the dict
    was built. Ties go to the earlier key, which keeps the answer stable.
    """
    best: Any | None = None
    best_d = float(radius) ** 2
    for key, point in regions.items():
        dx = float(point[0]) - float(at[0])
        dy = float(point[1]) - float(at[1])
        distance = dx * dx + dy * dy
        if distance <= best_d and (best is None or distance < best_d):
            best, best_d = key, distance
    return best


# --- what a shape occupies ---------------------------------------------------


def bounds(shape: Any) -> tuple[float, float, float, float]:
    """``(x, y, w, h)`` in tile pixels, for any of the three shapes.

    A polygon has no ``w``/``h`` of its own -- its points are what it is -- so
    this is its points' extent, offset by its origin.
    """
    if isinstance(shape, TilePolygon):
        points = tuple(shape.points)
        if not points:
            return (float(shape.x), float(shape.y), 0.0, 0.0)
        xs = [float(px) for px, _ in points]
        ys = [float(py) for _, py in points]
        return (
            float(shape.x) + min(xs),
            float(shape.y) + min(ys),
            max(xs) - min(xs),
            max(ys) - min(ys),
        )
    return (float(shape.x), float(shape.y), float(shape.w), float(shape.h))


def vertices(shape: Any) -> tuple[tuple[float, float], ...]:
    """A polygon's points in *tile* space rather than relative to its origin."""
    if not isinstance(shape, TilePolygon):
        return ()
    return tuple(
        (float(shape.x) + float(px), float(shape.y) + float(py)) for px, py in shape.points
    )


def vertex_regions(shape: Any) -> dict[int, tuple[float, float]]:
    """Every vertex as a click region keyed by its index."""
    return dict(enumerate(vertices(shape)))


def box_handles(shape: Any) -> dict[str, tuple[float, float]]:
    """The eight resize handles of a box or an ellipse, in tile pixels.

    Empty for a polygon: a polygon is resized by its vertices, and a bounding
    box handle on one would have to scale every point -- a different gesture
    wearing the same grip.
    """
    if isinstance(shape, TilePolygon) or shape is None:
        return {}
    x, y, w, h = bounds(shape)
    mid_x, mid_y = x + w / 2.0, y + h / 2.0
    return {
        "nw": (x, y),
        "n": (mid_x, y),
        "ne": (x + w, y),
        "e": (x + w, mid_y),
        "se": (x + w, y + h),
        "s": (mid_x, y + h),
        "sw": (x, y + h),
        "w": (x, mid_y),
    }


def hit(shape: Any, at: tuple[float, float]) -> bool:
    """Whether a tile-pixel point is *inside* the shape's body."""
    px, py = float(at[0]), float(at[1])
    if isinstance(shape, TilePolygon):
        return _in_polygon(vertices(shape), (px, py))
    x, y, w, h = bounds(shape)
    if isinstance(shape, TileEllipse):
        if w <= 0.0 or h <= 0.0:
            return False
        nx = (px - (x + w / 2.0)) / (w / 2.0)
        ny = (py - (y + h / 2.0)) / (h / 2.0)
        return nx * nx + ny * ny <= 1.0
    return x <= px <= x + w and y <= py <= y + h


def shape_at(shapes: Iterable[Any], at: tuple[float, float]) -> int | None:
    """The index of the topmost shape under ``at``, or ``None``.

    Topmost is *last*, because that is the order they are drawn in: picking the
    first would hand the click to whatever happens to be underneath.
    """
    found = None
    for index, shape in enumerate(shapes):
        if hit(shape, at):
            found = index
    return found


def _in_polygon(points: tuple[tuple[float, float], ...], at: tuple[float, float]) -> bool:
    """Even-odd ray cast. Degenerate outlines (< 3 points) enclose nothing."""
    if len(points) < 3:
        return False
    px, py = at
    inside = False
    for index, (ax, ay) in enumerate(points):
        bx, by = points[index - 1]
        if (ay > py) != (by > py):
            crossing = ax + (py - ay) * (bx - ax) / ((by - ay) or 1e-9)
            if px < crossing:
                inside = not inside
    return inside


# --- moving and resizing -----------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamped_point(
    at: tuple[float, float], tile_w: int, tile_h: int
) -> tuple[float, float]:
    """A tile-pixel point pulled back inside the tile.

    Every gesture clamps, and the reason is that the view *is* the tile: a
    shape dragged past the edge would be invisible and so ungrabbable, which is
    a shape the user has lost rather than moved.
    """
    return (
        _clamp(float(at[0]), 0.0, float(max(1, int(tile_w)))),
        _clamp(float(at[1]), 0.0, float(max(1, int(tile_h)))),
    )


def moved(shape: Any, dx: float, dy: float, tile_w: int, tile_h: int) -> Any:
    """``shape`` shifted by a tile-pixel delta, kept inside the tile."""
    x, y, w, h = bounds(shape)
    limit_x = float(max(1, int(tile_w))) - w
    limit_y = float(max(1, int(tile_h))) - h
    # A shape wider than the tile has no legal range, so it pins to the origin
    # rather than jittering between two clamps.
    new_x = _clamp(x + float(dx), 0.0, max(0.0, limit_x))
    new_y = _clamp(y + float(dy), 0.0, max(0.0, limit_y))
    return dataclasses.replace(shape, x=shape.x + (new_x - x), y=shape.y + (new_y - y))


def resized(
    shape: Any,
    handle: str,
    at: tuple[float, float],
    tile_w: int,
    tile_h: int,
    *,
    minimum: float = MIN_SIDE,
) -> Any:
    """``shape`` with the edges ``handle`` names pulled to ``at``.

    The *opposite* edge stays pinned, which is what makes a resize feel like a
    resize rather than a move. When the drag would collapse the shape it is the
    **moving** edge that gets pushed back to ``minimum``: pushing the pinned one
    would slide the shape out from under the pointer.
    """
    if handle not in BOX_HANDLES or isinstance(shape, TilePolygon):
        return shape
    x, y, w, h = bounds(shape)
    left, top, right, bottom = x, y, x + w, y + h
    ax, ay = clamped_point(at, tile_w, tile_h)
    if "w" in handle:
        left = ax
    if "e" in handle:
        right = ax
    if "n" in handle:
        top = ay
    if "s" in handle:
        bottom = ay
    if right - left < minimum:
        if "w" in handle:
            left = right - minimum
        else:
            right = left + minimum
    if bottom - top < minimum:
        if "n" in handle:
            top = bottom - minimum
        else:
            bottom = top + minimum
    return dataclasses.replace(shape, x=left, y=top, w=right - left, h=bottom - top)


# --- polygon vertices --------------------------------------------------------


def with_vertex(
    shape: Any, index: int, at: tuple[float, float], tile_w: int, tile_h: int
) -> Any:
    """The polygon with vertex ``index`` moved to the tile-pixel point ``at``.

    The origin stays where it was and only the relative point changes, so a
    vertex drag does not silently move every *other* vertex by re-deriving
    ``x``/``y`` from the new extent.
    """
    if not isinstance(shape, TilePolygon) or not 0 <= int(index) < len(shape.points):
        return shape
    ax, ay = clamped_point(at, tile_w, tile_h)
    points = list(shape.points)
    points[int(index)] = (ax - float(shape.x), ay - float(shape.y))
    return dataclasses.replace(shape, points=tuple(points))


def without_vertex(shape: Any, index: int) -> Any | None:
    """The polygon minus one vertex, or ``None`` when that is not a polygon.

    ``None`` rather than a two-point outline: the caller has a refusal to say
    and a silent no-op is the one answer that teaches nothing.
    """
    if not isinstance(shape, TilePolygon) or not 0 <= int(index) < len(shape.points):
        return None
    if len(shape.points) <= MIN_POINTS:
        return None
    points = [p for at, p in enumerate(shape.points) if at != int(index)]
    return dataclasses.replace(shape, points=tuple(points))


def nearest_segment(shape: Any, at: tuple[float, float]) -> int | None:
    """Which edge of the closed outline ``at`` is nearest. -> the index it
    starts at, so a caller inserts *after* it."""
    points = vertices(shape)
    if len(points) < 2:
        return None
    best, best_d = None, math.inf
    for index in range(len(points)):
        distance = _segment_distance(points[index], points[(index + 1) % len(points)], at)
        if distance < best_d:
            best, best_d = index, distance
    return best


def inserted_vertex(shape: Any, at: tuple[float, float], tile_w: int, tile_h: int) -> Any:
    """The polygon with a new vertex at ``at``, on the edge nearest to it.

    On the nearest *edge* rather than appended to the end, which is the whole
    difference between adding a point to an outline and adding one after the
    last one somebody happened to place: appending re-routes two edges at once
    and is almost never what the click meant.
    """
    if not isinstance(shape, TilePolygon):
        return shape
    ax, ay = clamped_point(at, tile_w, tile_h)
    rel = (ax - float(shape.x), ay - float(shape.y))
    points = list(shape.points)
    if len(points) < 2:
        points.append(rel)
        return dataclasses.replace(shape, points=tuple(points))
    edge = nearest_segment(shape, (ax, ay))
    points.insert(int(edge) + 1, rel)
    return dataclasses.replace(shape, points=tuple(points))


def _segment_distance(
    a: tuple[float, float], b: tuple[float, float], at: tuple[float, float]
) -> float:
    ax, ay = a
    bx, by = b
    px, py = at
    dx, dy = bx - ax, by - ay
    length = dx * dx + dy * dy
    if length <= 0.0:
        return math.hypot(px - ax, py - ay)
    t = _clamp(((px - ax) * dx + (py - ay) * dy) / length, 0.0, 1.0)
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


# --- what a new shape starts as ----------------------------------------------


def new_shape(kind: Any, tile_w: int, tile_h: int) -> Any:
    """A shape covering the whole tile, which is the one obviously editable
    starting size -- a zero-sized box is a shape you cannot grab.

    A polygon starts as the same square with a vertex at each corner, so the
    first thing a user can do to it is drag one somewhere else.
    """
    width, height = float(max(1, int(tile_w))), float(max(1, int(tile_h)))
    if kind is TilePolygon:
        return TilePolygon(
            x=0.0,
            y=0.0,
            points=((0.0, 0.0), (width, 0.0), (width, height), (0.0, height)),
        )
    if kind is TileEllipse:
        return TileEllipse(x=0.0, y=0.0, w=width, h=height)
    return TileRect(x=0.0, y=0.0, w=width, h=height)
