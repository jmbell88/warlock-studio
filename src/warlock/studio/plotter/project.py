"""Where a cell is, in every affine projection, and how a click gets back.

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
OBLIQUE = "oblique"
STAGGERED = "staggered"
HEXAGONAL = "hexagonal"
#: The affine projections this editor draws. Staggered and hexagonal require a
#: nearest-centre hit test and remain explicit refusals at the Tiled door.
PROJECTIONS: tuple[str, ...] = (ORTHOGONAL, ISOMETRIC, OBLIQUE, STAGGERED, HEXAGONAL)

#: The two lattices whose rows (or columns) are offset from each other. Grouped
#: because every piece of arithmetic below treats them the same except for the
#: hex side, which is 0 on a staggered map by definition -- a staggered map is a
#: hexagonal one whose flat run has no length.
OFFSET_PROJECTIONS: tuple[str, ...] = (STAGGERED, HEXAGONAL)

#: Which axis is offset, and which of its lines is the offset one.
STAGGER_AXES: tuple[str, ...] = ("x", "y")
STAGGER_INDICES: tuple[str, ...] = ("odd", "even")
RENDER_ORDERS: tuple[str, ...] = ("right-down", "right-up", "left-down", "left-up")


class Lattice(NamedTuple):
    """The document numbers every placement function needs, bundled once.

    ``projection``, ``width``, ``height``, ``tile_w`` and ``tile_h`` are the
    five that ortho/iso arithmetic reads. ``stagger_axis``, ``stagger_index``
    and ``hex_side`` were reserved for the staggered and hexagonal projections
    and are now **used** by them: which axis is offset, which line of that axis
    is the offset one, and how long the hexagon's flat run is (0 on a staggered
    map, which is the same lattice with no flat run at all).
    """

    projection: str
    width: int
    height: int
    tile_w: int
    tile_h: int
    stagger_axis: str = "y"
    stagger_index: str = "odd"
    hex_side: int = 0
    skew_x: int = 0
    skew_y: int = 0
    render_order: str = "right-down"


# --- the offset lattices ------------------------------------------------------
#
# A staggered map is a grid of diamonds whose every other row (or column) is
# pushed half a tile sideways; a hexagonal map is the same lattice with a flat
# run of ``hex_side`` pixels inserted along the stagger axis, turning each
# diamond into a hexagon. One set of formulas serves both, with ``hex_side = 0``
# recovering the staggered case exactly -- which is why they are one arm here
# rather than two, and why a bug in one cannot be fixed in the other.
#
# **The inverse is not affine on these lattices**, and that is the whole
# difficulty. On an orthogonal, isometric or oblique map, flooring a transformed
# point *is* the cell; here a point near a shared edge belongs to whichever of
# two hexagons actually contains it, and no linear map says which. The refusal
# this replaces said the projections would be "named rather than projected
# approximately", so the bar is an exact hit test rather than a nearest-centre
# guess: :func:`_offset_cell_at` computes the small set of candidates whose
# region can contain the point and tests containment.


def _offset_steps(lat: Lattice) -> tuple[float, float]:
    """How far one cell advances along each axis, in pixels.

    On a y-staggered map a row advances half a tile height (plus half the flat
    run, which is what makes the hexagons meet) and a column advances a whole
    tile width. Transposed on an x-staggered one.
    """
    if lat.stagger_axis == "y":
        return (float(lat.tile_w), (lat.tile_h + lat.hex_side) / 2.0)
    return ((lat.tile_w + lat.hex_side) / 2.0, float(lat.tile_h))


def _staggered(lat: Lattice, line: int) -> bool:
    """Whether this row (or column) is the offset one."""
    return (line % 2 == 1) if lat.stagger_index == "odd" else (line % 2 == 0)


def _offset_origin(lat: Lattice, column: int, row: int) -> tuple[float, float]:
    """The top-left of one cell's image rectangle on an offset lattice."""
    step_x, step_y = _offset_steps(lat)
    if lat.stagger_axis == "y":
        shift = lat.tile_w / 2.0 if _staggered(lat, row) else 0.0
        return (column * step_x + shift, row * step_y)
    shift = lat.tile_h / 2.0 if _staggered(lat, column) else 0.0
    return (column * step_x, row * step_y + shift)


def _offset_size(lat: Lattice) -> tuple[int, int]:
    """The pixel bounding box of a whole offset map.

    The half-tile the staggered lines are pushed by is part of the box, and so
    is the last line's own tile -- which is why this is not simply the step
    times the count.
    """
    step_x, step_y = _offset_steps(lat)
    if lat.stagger_axis == "y":
        width = lat.width * lat.tile_w + (lat.tile_w // 2 if lat.height > 1 else 0)
        height = int((lat.height - 1) * step_y) + lat.tile_h
    else:
        width = int((lat.width - 1) * step_x) + lat.tile_w
        height = lat.height * lat.tile_h + (lat.tile_h // 2 if lat.width > 1 else 0)
    return (max(0, int(width)), max(0, int(height)))


def _contains(lat: Lattice, column: int, row: int, x: float, y: float) -> bool:
    """Is the point inside this cell's hexagon (or diamond)?

    The exact test the refusal's own wording demanded. A hexagon here is a
    rectangle with two triangular ends cut off along the stagger axis, so
    containment is the rectangle test plus one comparison per cut corner -- and
    with ``hex_side = 0`` the rectangle vanishes and the two triangles meet,
    which is exactly a diamond.
    """
    ox, oy = _offset_origin(lat, column, row)
    px, py = float(x) - ox, float(y) - oy
    if not (0.0 <= px <= lat.tile_w and 0.0 <= py <= lat.tile_h):
        return False
    if lat.stagger_axis == "y":
        # The flat run is horizontal, centred vertically; the cut corners are
        # the four above and below it.
        half = lat.tile_w / 2.0
        cut = (lat.tile_h - lat.hex_side) / 2.0
        if cut <= 0.0:
            return True
        if py < cut:
            # The top wedge narrows to the flat run's own width at ``cut``.
            reach = half * (py / cut)
            return abs(px - half) <= reach
        if py > lat.tile_h - cut:
            reach = half * ((lat.tile_h - py) / cut)
            return abs(px - half) <= reach
        return True
    half = lat.tile_h / 2.0
    cut = (lat.tile_w - lat.hex_side) / 2.0
    if cut <= 0.0:
        return True
    if px < cut:
        reach = half * (px / cut)
        return abs(py - half) <= reach
    if px > lat.tile_w - cut:
        reach = half * ((lat.tile_w - px) / cut)
        return abs(py - half) <= reach
    return True


def _offset_cell_at(lat: Lattice, x: float, y: float) -> tuple[int, int]:
    """Which cell a point falls in, by candidate check rather than by inverse.

    A point near a shared edge belongs to whichever hexagon actually contains
    it, and no linear map says which -- so the rough cell is computed from the
    steps, the handful of cells whose region can reach the point are tested, and
    the first that contains it wins. A point on a shared corner lands in exactly
    one cell because the candidates are tried in a fixed order, which is the
    same guarantee the isometric lattice already makes.

    Unclamped, like every other arm: a drag off the edge is a legitimate stroke
    whose visible part lands.
    """
    step_x, step_y = _offset_steps(lat)
    rough_x = int(math.floor(float(x) / step_x)) if step_x else 0
    rough_y = int(math.floor(float(y) / step_y)) if step_y else 0
    for dy in (0, -1, 1, -2, 2):
        for dx in (0, -1, 1):
            column, row = rough_x + dx, rough_y + dy
            if _contains(lat, column, row, x, y):
                return (column, row)
    # Nothing contained it, which happens only in the gaps a rounding error can
    # open at a shared vertex. The rough cell is the honest fallback: it is the
    # nearest by construction and no worse than a nearest-centre answer would be.
    return (rough_x, rough_y)



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


def _oblique_origin(lat: Lattice) -> tuple[float, float]:
    """Offset the skewed lattice so its finite bounding box starts at zero."""
    return (
        -min(0.0, float(lat.height * lat.skew_x)),
        -min(0.0, float(lat.width * lat.skew_y)),
    )


def map_size(lat: Lattice) -> tuple[int, int]:
    """The pixel extent of the whole map -- Tiled's own bounding box."""
    if lat.projection == ISOMETRIC:
        return (
            (lat.width + lat.height) * lat.tile_w // 2,
            (lat.width + lat.height) * lat.tile_h // 2,
        )
    if lat.projection == OBLIQUE:
        return (
            int(abs(lat.height * lat.skew_x) + lat.width * lat.tile_w),
            int(abs(lat.width * lat.skew_y) + lat.height * lat.tile_h),
        )
    if lat.projection in OFFSET_PROJECTIONS:
        return _offset_size(lat)
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
    if lat.projection == OBLIQUE:
        origin_x, origin_y = _oblique_origin(lat)
        return (
            column * lat.tile_w + row * lat.skew_x + origin_x,
            row * lat.tile_h + column * lat.skew_y + origin_y,
        )
    if lat.projection in OFFSET_PROJECTIONS:
        # Fractional coordinates on an offset lattice are only ever asked for by
        # the grid pass, which walks whole cells; the linear form is right for
        # the integer case and the honest approximation elsewhere.
        return _offset_origin(lat, int(math.floor(column)), int(math.floor(row)))
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


def cell_outline(lat: Lattice, column: int, row: int) -> list[tuple[float, float]]:
    """One cell's own outline, in pixels and in order.

    The grid pass draws these rather than two families of straight lines,
    because on an offset lattice there *are* no straight lines running the width
    of the map: every other row is pushed sideways, so a line from ``(0, r)`` to
    ``(width, r)`` crosses cells rather than bounding them.

    Kept here beside the placement arithmetic and not in the pane, for the
    reason everything else here is: the flat renderer wants the same shape the
    canvas draws, and two answers is how the export comes to disagree with the
    picture.
    """
    if lat.projection not in OFFSET_PROJECTIONS:
        x0, y0 = cell_corner(lat, column, row)
        x1, y1 = cell_corner(lat, column + 1, row + 1)
        if lat.projection == ISOMETRIC:
            # The diamond: the four lattice nodes around the cell.
            return [
                cell_corner(lat, column, row),
                cell_corner(lat, column + 1, row),
                cell_corner(lat, column + 1, row + 1),
                cell_corner(lat, column, row + 1),
            ]
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    ox, oy = _offset_origin(lat, column, row)
    width, height = float(lat.tile_w), float(lat.tile_h)
    if lat.stagger_axis == "y":
        cut = (height - lat.hex_side) / 2.0
        half = width / 2.0
        # Clockwise from the top vertex. With ``hex_side = 0`` the two middle
        # pairs coincide and the hexagon collapses to a diamond, which is
        # exactly what a staggered cell is.
        return [
            (ox + half, oy),
            (ox + width, oy + cut),
            (ox + width, oy + height - cut),
            (ox + half, oy + height),
            (ox, oy + height - cut),
            (ox, oy + cut),
        ]
    cut = (width - lat.hex_side) / 2.0
    half = height / 2.0
    return [
        (ox + cut, oy),
        (ox + width - cut, oy),
        (ox + width, oy + half),
        (ox + width - cut, oy + height),
        (ox + cut, oy + height),
        (ox, oy + half),
    ]



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
    if lat.projection == OBLIQUE:
        origin_x, origin_y = _oblique_origin(lat)
        px, py = float(x) - origin_x, float(y) - origin_y
        determinant = lat.tile_w * lat.tile_h - lat.skew_x * lat.skew_y
        if determinant == 0:
            raise ValueError("an oblique map's skew collapses its grid")
        column = (px * lat.tile_h - py * lat.skew_x) / determinant
        row = (py * lat.tile_w - px * lat.skew_y) / determinant
        return (math.floor(column), math.floor(row))
    if lat.projection in OFFSET_PROJECTIONS:
        return _offset_cell_at(lat, x, y)
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
    if lat.projection == OBLIQUE:
        origin_x, origin_y = _oblique_origin(lat)
        px, py = float(x) - origin_x, float(y) - origin_y
        determinant = lat.tile_w * lat.tile_h - lat.skew_x * lat.skew_y
        if determinant == 0:
            raise ValueError("an oblique map's skew collapses its grid")
        return (
            (px * lat.tile_h - py * lat.skew_x) / determinant,
            (py * lat.tile_w - px * lat.skew_y) / determinant,
        )
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
    if lat.render_order not in RENDER_ORDERS:
        raise ValueError(f"unknown render order {lat.render_order!r}")
    if lat.projection == ISOMETRIC:
        for depth in range(width + height - 1):
            lo = max(0, depth - (height - 1))
            hi = min(depth, width - 1)
            for column in range(lo, hi + 1):
                yield (column, depth - column)
        return
    columns = range(width) if lat.render_order.startswith("right") else range(width - 1, -1, -1)
    rows = range(height) if lat.render_order.endswith("down") else range(height - 1, -1, -1)
    for row in rows:
        for column in columns:
            yield (column, row)
