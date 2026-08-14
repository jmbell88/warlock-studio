"""Where a cell is, in both projections, and how a click gets back.

One module owns the lattice, for the reason ``gid`` owns the transform flags:
the canvas and the flat renderer must place a cell identically or an export
stops being a picture of the screen, and two spellings of the same arithmetic is
how they come to disagree by one half-tile at some zoom nobody tested.

**Isometric is a change of placement, not of rasterisation.** A cell's image is
still ``tile_w`` by ``tile_h``; the diamond is inscribed in that rectangle, and
the rectangle stays axis-aligned in pixel space. So the draw-list quad keeps its
four axis-aligned corners, the transpose-then-mirror flag permutation is
untouched, and everything downstream of "which pixel rectangle" is unchanged.

**The inverse is exact, not a hit test -- for ortho and iso.** The diamond
lattice is precisely the image of the unit-square lattice under an affine map,
so transforming the point and flooring *is* "which cell contains this" -- there
is no point-in-polygon to get wrong on a shared edge, and a click on a corner
lands in exactly one cell because ``floor`` breaks the tie the same way
everywhere. That claim is scoped to :data:`ORTHOGONAL` and :data:`ISOMETRIC`:
staggered and hexagonal ``cell_at`` is Tiled's reference-point-plus-nearest-
centre test instead, which is not an affine inverse and is not implemented here.

**M5 seam.** :class:`Lattice` carries ``stagger_axis``, ``stagger_index`` and
``hex_side`` so every call site already threads one object instead of five
loose numbers, but the fields are reserved -- nothing in this module reads
them, and staggered/hexagonal maps are still refused at the door by
:mod:`.tmx`. This lands the wide signature migration on a quiet tree so the
staggered/hexagonal math has one seam to land into rather than one per caller.
"""

from __future__ import annotations

import math
from typing import NamedTuple

ORTHOGONAL = "orthogonal"
ISOMETRIC = "isometric"
#: The two this editor draws. Staggered and hexagonal are refused at the door by
#: :mod:`.tmx`, which is what keeps this tuple short enough to branch on.
PROJECTIONS: tuple[str, ...] = (ORTHOGONAL, ISOMETRIC)


class Lattice(NamedTuple):
    """The document numbers every placement function needs, bundled once.

    ``projection``, ``width``, ``height``, ``tile_w`` and ``tile_h`` are the
    five that ortho/iso arithmetic actually reads. ``stagger_axis``,
    ``stagger_index`` and ``hex_side`` are the M5 seam: reserved for the
    staggered/hexagonal projections, carrying Tiled's own defaults, and UNUSED
    -- no function in this module consumes them yet.
    """

    projection: str
    width: int
    height: int
    tile_w: int
    tile_h: int
    stagger_axis: str = "y"
    stagger_index: str = "odd"
    hex_side: int = 0


def check(projection: str) -> str:
    """Refuse an unknown projection by name, the way ``_dimension`` refuses a
    bad size -- the useful moment to say so is when the document is made, not at
    the first draw of a map placed by arithmetic nothing here implements."""
    value = str(projection)
    if value not in PROJECTIONS:
        raise ValueError(
            f"unknown projection {value!r}; this draws {' and '.join(PROJECTIONS)} maps"
        )
    return value


def _origin_x(lat: Lattice) -> float:
    """How far right the lattice sits, so no cell has a negative x.

    Zero for an orthogonal map. For an isometric one the leftmost point of the
    whole map is the west corner of cell ``(0, height - 1)``, so the lattice is
    pushed right by exactly that much and the map's bounding box starts at the
    origin like every other document in the app.
    """
    return (lat.height * lat.tile_w) / 2.0 if lat.projection == ISOMETRIC else 0.0


def map_size(lat: Lattice) -> tuple[int, int]:
    """The pixel extent of the whole map -- Tiled's own bounding box."""
    if lat.projection == ISOMETRIC:
        return (
            (lat.width + lat.height) * lat.tile_w // 2,
            (lat.width + lat.height) * lat.tile_h // 2,
        )
    return (lat.width * lat.tile_w, lat.height * lat.tile_h)


def cell_corner(lat: Lattice, column: float, row: float) -> tuple[float, float]:
    """A *lattice node* in pixels -- the meeting point of four cells.

    Takes fractional coordinates and returns floats because the grid and the
    brush outline want the node at ``(width, height)``, one past the last cell,
    and because a caller drawing a line between two nodes must not have them
    rounded independently.
    """
    if lat.projection == ISOMETRIC:
        half_w, half_h = lat.tile_w / 2.0, lat.tile_h / 2.0
        return (
            (column - row) * half_w + _origin_x(lat),
            (column + row) * half_h,
        )
    return (column * lat.tile_w, row * lat.tile_h)


def cell_origin(lat: Lattice, column: int, row: int) -> tuple[float, float]:
    """The top-left of a cell's *image* rectangle.

    Not the same point as :func:`cell_corner` under an isometric projection:
    the node is the diamond's top vertex and the image is the rectangle the
    diamond is inscribed in, so the rectangle starts half a tile further left.
    Both renderers want this one, and the distinction is the single easiest
    thing to get wrong here.
    """
    x, y = cell_corner(lat, column, row)
    return (x - lat.tile_w / 2.0, y) if lat.projection == ISOMETRIC else (x, y)


def cell_at(lat: Lattice, x: float, y: float) -> tuple[int, int]:
    """Which cell a pixel point falls in. Unclamped, deliberately.

    Every tool in :mod:`.tools` clips its own placement and treats a drag off
    the edge as a legitimate stroke whose visible part lands, so returning a
    clamped cell here would quietly turn a stroke that left the map into one
    that piled up along its border.
    """
    if lat.projection == ISOMETRIC:
        half_w, half_h = lat.tile_w / 2.0, lat.tile_h / 2.0
        # u is (column - row) and v is (column + row), so the two add and
        # subtract back into the pair. This is the affine inverse, which is why
        # no polygon test appears anywhere in this module.
        u = (float(x) - _origin_x(lat)) / half_w
        v = float(y) / half_h
        return (math.floor((u + v) / 2.0), math.floor((v - u) / 2.0))
    return (math.floor(float(x) / lat.tile_w), math.floor(float(y) / lat.tile_h))


def cell_point(lat: Lattice, x: float, y: float) -> tuple[float, float]:
    """:func:`cell_at` without the floor -- a *fractional* cell coordinate.

    What an object needs, since an object is routinely placed off the grid on
    purpose and rounding it to a cell would move every spawn point to a corner.
    """
    if lat.projection == ISOMETRIC:
        half_w, half_h = lat.tile_w / 2.0, lat.tile_h / 2.0
        u = (float(x) - _origin_x(lat)) / half_w
        v = float(y) / half_h
        return ((u + v) / 2.0, (v - u) / 2.0)
    return (float(x) / lat.tile_w, float(y) / lat.tile_h)


def object_to_pixels(lat: Lattice, x: float, y: float) -> tuple[float, float]:
    """A Tiled object's stored position, as a point in this map's pixel plane.

    **The identity for an orthogonal map, and not for an isometric one.** Tiled
    stores an isometric object's ``x``/``y`` in *tile-space units of the map's
    tile height* -- dividing both by ``tileheight`` gives the fractional cell --
    rather than in the projected plane the object is drawn in. Warlock draws
    objects at absolute pixels, which is self-consistent but is not what Tiled
    means by the same two numbers, so a spawn point that made the trip untouched
    would reopen somewhere else entirely and say nothing about it.
    """
    if lat.projection != ISOMETRIC:
        return (float(x), float(y))
    return cell_corner(lat, float(x) / lat.tile_h, float(y) / lat.tile_h)


def object_from_pixels(lat: Lattice, x: float, y: float) -> tuple[float, float]:
    """:func:`object_to_pixels` inverted, for the writer."""
    if lat.projection != ISOMETRIC:
        return (float(x), float(y))
    column, row = cell_point(lat, x, y)
    return (column * lat.tile_h, row * lat.tile_h)


def cell_bounds(
    lat: Lattice,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> tuple[int, int, int, int]:
    """The inclusive cell rectangle covering a pixel rectangle, clamped.

    **Four corners, not two.** Under an isometric projection a screen rectangle
    maps to a *rhombus* in cell space, so the min and max of one diagonal pair
    misses the other two and culls cells that are on screen. The bounding box of
    all four is conservative for isometric and exactly the old answer for
    orthogonal, so there is one code path rather than a branch that only one of
    them is ever tested through.

    Grown by one cell on every side: a tile is drawn from its own origin, and a
    cell whose origin is just off screen can still have pixels on it.
    """
    corners = ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
    cells = [cell_at(lat, px, py) for px, py in corners]
    columns = [c for c, _r in cells]
    rows = [r for _c, r in cells]
    return (
        max(0, min(columns) - 1),
        max(0, min(rows) - 1),
        min(int(lat.width) - 1, max(columns) + 1),
        min(int(lat.height) - 1, max(rows) + 1),
    )


def draw_order(lat: Lattice):
    """Every cell, back to front.

    Row-major for an orthogonal map, and Tiled's ``right-down``. For an
    isometric one screen depth is ``column + row``, and row-major is *not*
    monotone in it -- cell ``(width - 1, 0)`` sits at depth ``width - 1`` and is
    reached before ``(0, 1)`` at depth 1 -- so a tile taller than its cell would
    draw over one in front of it. Ground tiles never overhang, so this is
    invisible today and correct on the day something does.
    """
    width, height = lat.width, lat.height
    if lat.projection == ISOMETRIC:
        for depth in range(width + height - 1):
            lo = max(0, depth - (height - 1))
            hi = min(depth, width - 1)
            for column in range(lo, hi + 1):
                yield (column, depth - column)
        return
    for row in range(height):
        for column in range(width):
            yield (column, row)
