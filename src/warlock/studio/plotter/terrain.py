"""A tileset's declared roles, read back off the cells that use them.

**A terrain is derived from the gid, never stored beside it.** A parallel
per-cell terrain array would be a second source of truth that ``.wmap``, the
undo stack and every Tiled round trip would each have to keep honest, and the
first time one of them missed, the map would paint edges for a field that is not
the one on screen. A terrain set makes the gid answer on its own -- the tileset
says which row is which terrain, and :meth:`Tileset.terrain_of` reads it back --
so a terrain stroke is nothing but an ordinary patch of gids, and a cell someone
stamped by hand with the plain stamp tool joins the field for free.

**Precedence is list position, and membership is "ranked at or above me".**
Blob autotiling encodes *self against not-self*; it cannot say which not-self,
so a single tile in a single layer is only well defined at a two-terrain
boundary unless "not-self" is made unambiguous. Ranking does that. Take grass,
sand and water in that order and put all three around one point:

======  =================  =======  ======  =======  ==========================
cell    membership test    grass    sand    water   picture
======  =================  =======  ======  =======  ==========================
grass   ``rank >= 0``      member   member  member   interior fill, no outline
sand    ``rank >= 1``      no       member  member   outlined against grass
water   ``rank >= 2``      no       no      member   outlined against both
======  =================  =======  ======  =======  ==========================

Every cell gets exactly one tile at a junction of any number of terrains, and
the outline always belongs to the higher-precedence terrain -- a statable art
convention rather than an artifact of the encoding.

**The cost, stated so nobody quietly "fixes" it:** a terrain's edge art cannot
vary by which terrain is on the other side. Sand against grass and sand against
stone draw the same tile. For a set that is a flat fill plus a darker outline
that is right by construction, because the outline is a property of the terrain
rather than of the boundary; when someone eventually wants a shoreline that
differs from a cliff base, the answer is a second terrain set on a second layer,
which this model already supports. Changing the membership test instead would
break every three-terrain junction at once.

Everything here is a pure function over one layer's array returning a
:data:`~.tools.Region`, for :mod:`.tools`' reasons and under its rules: nothing
mutates, nothing pushes a step, a placement wholly off the map is ``None``, and
a computation that changed nothing is ``None`` rather than an empty write.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..tilegrid import blob
from ..tilegrid import gid as gidlib
from ..tilegrid.tileset import TerrainSpec, TilesetRef
from .tools import Region, flood_mask

#: What an empty cell, or one belonging to some other tileset, ranks as. Below
#: every terrain, so nothing is ever outlined against the terrain it is not.
RANK_VOID: int = -1

#: The base set, in precedence order: grass gives way to nothing, water gives
#: way to everything. The colours are flat and the outlines are the same hue
#: darkened -- deliberately plain and deliberately legible at 16 pixels.
#:
#: Nothing in the app reads it since the procedural generator that filled a
#: tileset from it was deleted on 2026-08-18; ``tests/plotter/_terrainset.py``
#: is its only caller now. Kept here rather than moved into the tests because
#: the precedence order is a *statement about this module* -- ``rank_field``
#: and the outlining below are what make "later wins" mean anything -- and a
#: named example of a legal set is what a reader of that rule wants next.
DEFAULT_TERRAINS: tuple[TerrainSpec, ...] = (
    TerrainSpec("Grass", (106, 153, 78, 255), (62, 96, 44, 255)),
    TerrainSpec("Dirt", (156, 122, 84, 255), (98, 74, 48, 255)),
    TerrainSpec("Sand", (214, 195, 132, 255), (156, 136, 82, 255)),
    TerrainSpec("Stone", (140, 142, 150, 255), (86, 88, 96, 255)),
    TerrainSpec("Water", (78, 122, 178, 255), (44, 74, 118, 255)),
)


def rank_field(data: np.ndarray, ref: TilesetRef) -> np.ndarray:
    """Every cell's terrain rank, or :data:`RANK_VOID`.

    ``int16`` rather than ``uint8`` so the void is a real negative and every
    comparison in this module reads as the inequality it is. Vectorised over the
    whole layer because a retile needs one pass per terrain and a Python loop
    over cells would put a 200x200 map inside a mouse-move handler.
    """
    tileset = ref.tileset
    ids = gidlib.tile_ids(np.asarray(data, dtype=gidlib.DTYPE))
    ranks = np.full(ids.shape, RANK_VOID, dtype=np.int16)
    if not tileset.is_terrain_set:
        return ranks
    inside = (ids >= ref.firstgid) & (ids <= ref.last_gid)
    rows = (ids.astype(np.int64) - ref.firstgid) // (
        blob.TILE_COUNT * tileset.phases * tileset.phases
    )
    known = inside & (rows >= 0) & (rows < len(tileset.terrains))
    ranks[known] = rows[known].astype(np.int16)
    return ranks


def terrain_at(data: np.ndarray, x: int, y: int, ref: TilesetRef) -> int | None:
    """The terrain under one point, for an eyedropper. ``None`` off-map or void."""
    height, width = np.asarray(data).shape
    x, y = int(x), int(y)
    if not (0 <= x < width and 0 <= y < height):
        return None
    tile_id = int(data[y, x]) & gidlib.GID_MASK
    if not ref.holds(tile_id):
        return None
    return ref.tileset.terrain_of(ref.local(tile_id))


def gid_for(ref: TilesetRef, terrain: int, blob_index: int) -> int:
    """The encoded cell for one terrain's one blob case.

    Through :func:`gid.compose` rather than by addition, so a set whose ids ran
    past the id space raises here instead of silently setting a transform flag
    and drawing a mirrored tile from nowhere.
    """
    return gidlib.compose(ref.firstgid + ref.tileset.local_for(terrain, blob_index))


def _box(
    cells: Iterable[tuple[int, int]], shape: tuple[int, int], grow: int
) -> tuple[int, int, int, int] | None:
    """The clipped half-open rectangle around some cells, grown by ``grow``."""
    height, width = shape
    xs = [int(x) for x, _y in cells]
    ys = [int(y) for _x, y in cells]
    if not xs:
        return None
    x0 = max(0, min(xs) - grow)
    y0 = max(0, min(ys) - grow)
    x1 = min(width, max(xs) + grow + 1)
    y1 = min(height, max(ys) + grow + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _retile_into(
    work: np.ndarray, ref: TilesetRef, box: tuple[int, int, int, int], outside: bool
) -> None:
    """Recompute the blob case of every terrain cell inside ``box``.

    One pass per terrain rather than per cell: ``ranks >= r`` is the whole
    membership field for rank ``r``, and :func:`blob.indices_from` turns it into
    every cell's case in eight array ORs. Five terrains is five passes, which is
    cheaper than one Python loop over the box.

    **Those passes run over the box grown by one, not over the layer.** A cell
    on the box's edge needs its eight neighbours -- which is why every caller
    grows the box by one before writing it, and why one more ring is read here
    -- but a cell any further away cannot affect the answer, so computing the
    whole layer per painted cell made a terrain drag cost the *map size* rather
    than the brush size. That is exact rather than approximate: the ring's own
    cases are discarded, and where the grown window runs into the map edge the
    padding :func:`blob.masks_from` applies there is the same ``outside`` the
    whole-layer pass would have applied, because it *is* the map edge.
    """
    tileset = ref.tileset
    if not tileset.is_terrain_set:
        return
    x0, y0, x1, y1 = box
    height, width = work.shape
    gx0, gy0 = max(0, x0 - 1), max(0, y0 - 1)
    gx1, gy1 = min(width, x1 + 1), min(height, y1 + 1)
    # Where the box sits inside the grown window.
    ix0, iy0 = x0 - gx0, y0 - gy0
    ix1, iy1 = ix0 + (x1 - x0), iy0 + (y1 - y0)

    grown = work[gy0:gy1, gx0:gx1]
    ranks = rank_field(grown, ref)
    window = ranks[iy0:iy1, ix0:ix1]
    # The phase is the cell's *absolute* map coordinates mod k, so a repaint of
    # any window reproduces the same phases and a grown box equals the whole
    # layer -- a phase derived from window-local coordinates would shift the
    # pattern with the box.
    k = tileset.phases
    phase_grid = (np.arange(y0, y1, dtype=np.int64)[:, None] % k) * k + (
        np.arange(x0, x1, dtype=np.int64)[None, :] % k
    )
    for rank in range(len(tileset.terrains)):
        chosen = window == rank
        if not chosen.any():
            continue
        cases = blob.indices_from(ranks >= rank, outside=outside)[iy0:iy1, ix0:ix1]
        base = ref.firstgid + rank * k * k * blob.TILE_COUNT
        work[y0:y1, x0:x1][chosen] = (
            base + phase_grid[chosen] * blob.TILE_COUNT + cases[chosen]
        ).astype(gidlib.DTYPE)


def _finish(
    data: np.ndarray, work: np.ndarray, box: tuple[int, int, int, int]
) -> Region | None:
    x0, y0, x1, y1 = box
    region = work[y0:y1, x0:x1]
    if np.array_equal(region, data[y0:y1, x0:x1]):
        return None
    return x0, y0, np.ascontiguousarray(region, dtype=gidlib.DTYPE)


def paint_terrain_cells(
    data: np.ndarray,
    cells: Iterable[tuple[int, int]],
    terrain: int,
    ref: TilesetRef,
    *,
    outside: bool = True,
) -> Region | None:
    """Set some cells to a terrain and re-fit them and the ring around them.

    **One region, not a list.** The eight-neighbour fix-up lives inside the same
    rectangle as the paint, so a single :meth:`MapDoc.write_region` is already
    atomic and no compound step is needed -- which is what lets a whole drag
    become one undo step rather than one per cell.
    """
    height, width = np.asarray(data).shape
    inside = [
        (int(x), int(y)) for x, y in cells if 0 <= int(x) < width and 0 <= int(y) < height
    ]
    box = _box(inside, (height, width), 1)
    if box is None:
        return None
    work = np.array(data, dtype=gidlib.DTYPE)
    # The case is a placeholder: the retile below computes the real one from the
    # field this write has just changed. Writing rank-correct nonsense first and
    # fixing it in one pass is what keeps the two steps from disagreeing.
    placeholder = gid_for(ref, terrain, blob.FULL)
    for x, y in inside:
        work[y, x] = placeholder
    _retile_into(work, ref, box, outside)
    return _finish(data, work, box)


def paint_terrain(
    data: np.ndarray,
    x: int,
    y: int,
    terrain: int,
    ref: TilesetRef,
    *,
    radius: int = 0,
    outside: bool = True,
) -> Region | None:
    """One click: a square of side ``2 * radius + 1``, re-fitted with its ring."""
    span = range(-int(radius), int(radius) + 1)
    cells = [(int(x) + dx, int(y) + dy) for dy in span for dx in span]
    return paint_terrain_cells(data, cells, terrain, ref, outside=outside)


def erase_terrain(
    data: np.ndarray,
    x: int,
    y: int,
    ref: TilesetRef,
    *,
    radius: int = 0,
    outside: bool = True,
) -> Region | None:
    """Clear a square back to nothing and re-fit what surrounded it.

    Erasing a terrain cell is not the same as erasing a tile: the hole has to
    grow an outline on everything that now borders it, which is why this is here
    rather than a call to ``tools.erase``.
    """
    height, width = np.asarray(data).shape
    span = range(-int(radius), int(radius) + 1)
    cells = [
        (int(x) + dx, int(y) + dy)
        for dy in span
        for dx in span
        if 0 <= int(x) + dx < width and 0 <= int(y) + dy < height
    ]
    box = _box(cells, (height, width), 1)
    if box is None:
        return None
    work = np.array(data, dtype=gidlib.DTYPE)
    for cx, cy in cells:
        work[cy, cx] = gidlib.EMPTY
    _retile_into(work, ref, box, outside)
    return _finish(data, work, box)


def fill_terrain(
    data: np.ndarray,
    x: int,
    y: int,
    terrain: int,
    ref: TilesetRef,
    *,
    outside: bool = True,
) -> Region | None:
    """Flood the connected run of one terrain, and re-fit the result.

    **The flood is over the rank field, not the gids.** ``tools.flood_fill``
    matches on the full encoded value -- correctly, for its own purpose -- so it
    would stop at the boundary between two of grass's own forty-seven cases and
    fill a one-cell-wide ribbon along an edge. What a user means by "fill the
    grass" is the terrain, so that is what is compared.

    Four-connected, for ``tools.flood_fill``'s reason: a diagonal leak through a
    corner where two fields only touch at a point is how a fill escapes.
    """
    height, width = np.asarray(data).shape
    x, y = int(x), int(y)
    if not (0 <= x < width and 0 <= y < height):
        return None
    ranks = rank_field(data, ref)
    target = int(ranks[y, x])
    if target == int(terrain):
        return None

    # The same reachability question :func:`tools.flood_fill` asks, over a
    # different comparison -- which is exactly why it is one function. The two
    # were a verbatim copy of each other before, differing in nothing but the
    # array being matched.
    ys, xs = np.nonzero(flood_mask(ranks == target, x, y))
    return paint_terrain_cells(
        data, zip(xs.tolist(), ys.tolist(), strict=True), terrain, ref, outside=outside
    )


# --- the general case: a data-driven Wang set --------------------------------
#
# Everything above is the **blob preset**: a two-value collapse laid out
# positionally in 47 columns, whose bar is byte-identity on the whole existing
# terrain corpus. Nothing below touches it. A tileset that carries a foreign
# Wang set -- corner-only, edge-only, or mixed with more than one colour per
# slot -- is painted here instead, by constraint matching rather than by a
# collapse, and the two never meet.
#
# **The invariant is unchanged and is the reason this is a separate path rather
# than a replacement**: membership is still derived from the gid and never
# stored per cell. The wangid is looked up by tile id and the map goes on
# storing gids alone, so a hand-stamped cell joins a field for free here exactly
# as it does there.


def wang_field(data: np.ndarray, ref: TilesetRef, wangset: Any) -> Any:
    """``field_of(x, y)`` over one layer: a cell's wangid, or ``None``.

    ``None`` for off the map, for an empty cell, and for a tile this set says
    nothing about -- all three mean "nothing here has an opinion", which is what
    :func:`~..tilegrid.wang.constraints_from` needs them to mean.

    ``ref.firstgid``, ``ref.last_gid``, ``wangset.tiles`` and the gid mask are
    read once into locals rather than through ``ref.holds`` per neighbour read:
    ``holds`` is ``firstgid <= id <= last_gid`` and ``last_gid`` recomputes
    through a ``max_local_id -> tile_count -> columns/rows -> image_w/h``
    property chain on every call, and one gesture makes on the order of eight
    such reads per touched cell. `docs/measurements/2026-09-06-native-batch-7-candidates.md`
    (B9) measured a 128-cell diagonal drag at 131 ms shipped, 68 ms with this
    hoist -- same answers, by inspection of ``holds``.
    """
    height, width = data.shape
    firstgid, last_gid, tiles, mask = ref.firstgid, ref.last_gid, wangset.tiles, gidlib.GID_MASK

    def field_of(x: int, y: int) -> Any:
        if not (0 <= x < width and 0 <= y < height):
            return None
        value = int(data[y, x]) & mask
        if not value or not (firstgid <= value <= last_gid):
            return None
        local = value - firstgid
        return tiles.get(local)

    return field_of


def _wang_cell(
    work: np.ndarray,
    ref: TilesetRef,
    wangset: Any,
    x: int,
    y: int,
    field: Any = None,
    cache: dict[tuple[tuple[int, int], ...], list[int]] | None = None,
) -> bool:
    """Re-choose one cell's tile from what its neighbours say. Wrote anything?

    **No match leaves the cell untouched** rather than writing a near-miss: a
    wrong tile is the silent half-read in picture form, and a field with a hole
    in it is a thing the user can see and fix.

    ``field`` and ``cache`` are the caller's, when it has a whole box to re-fit.
    ``wangset.matching`` scans every tile in the set and *sorts* the survivors,
    and a fill asks it once per cell -- but the interior of a filled region asks
    the same question at every one of those cells, because an all-one-colour
    neighbourhood has one constraint dict however large it is. Memoising on the
    constraints collapses a fill from one scan-and-sort per cell to one per
    *distinct* neighbourhood, which on a large open area is a handful.
    """
    from ..tilegrid import wang as wanglib

    if field is None:
        field = wang_field(work, ref, wangset)
    wanted = wanglib.constraints_from(field, x, y, wangset)
    if not wanted:
        return False
    if cache is None:
        found = wangset.matching(wanted)
    else:
        key = tuple(sorted(wanted.items()))
        found = cache.get(key)
        if found is None:
            found = wangset.matching(wanted)
            cache[key] = found
    if not found:
        return False
    value = gidlib.DTYPE(ref.firstgid + found[0])
    if work[y, x] == value:
        return False
    work[y, x] = value
    return True


def wang_interior(wangset: Any, colour: int) -> int | None:
    """The local id of the tile whose *used* slots are all one colour, or ``None``.

    A set's own "interior" for a colour, which is what a click asserts. ``None``
    is the honest answer for a set that has no such tile -- a partial set, or a
    colour index nobody declared -- and every caller here turns it into "nothing
    was written" rather than into a near-miss.
    """
    if int(colour) < 1 or int(colour) > len(wangset.colours):
        return None
    found = wangset.matching(dict.fromkeys(wangset.slots, int(colour)))
    return found[0] if found else None


def paint_wang_cells(
    data: np.ndarray,
    cells: Iterable[tuple[int, int]],
    colour: int,
    ref: TilesetRef,
    wangset: Any,
) -> Region | None:
    """Set some cells to a Wang colour and re-fit the ring around them.

    ``colour`` is 1-based into the set's colours, matching a wangid slot.

    The touched cells are set to that colour's interior and every *other* cell
    inside the grown box is re-chosen against them. That is Tiled's terrain
    brush: what the user touched asserts a colour, and the ring reconciles.

    One region rather than a list, for :func:`paint_terrain_cells`' reason: the
    fix-up lives inside the same rectangle as the paint, so a whole gesture is
    one write and one undo step.
    """
    height, width = np.asarray(data).shape
    inside = [
        (int(x), int(y)) for x, y in cells if 0 <= int(x) < width and 0 <= int(y) < height
    ]
    box = _box(inside, (height, width), 1)
    if box is None:
        return None
    local = wang_interior(wangset, colour)
    if local is None:
        return None

    work = np.array(data, dtype=gidlib.DTYPE)
    value = gidlib.DTYPE(ref.firstgid + local)
    # One fancy-index write rather than a Python loop: a *fill* asserts every
    # cell of a flooded region, which on a large map is tens of thousands of
    # them, and the write is the one part of this op that is not sequential.
    if inside:
        xs, ys = zip(*inside, strict=True)
        work[np.asarray(ys), np.asarray(xs)] = value
    asserted = set(inside)
    x0, y0, x1, y1 = box
    # One field closure and one match cache for the whole box, rather than a
    # fresh pair per cell. The re-fit **is** unavoidably sequential -- each cell
    # reads neighbours the loop may already have re-chosen, so there is no pass
    # that computes them all at once and gets the same answer -- so what is
    # shared here is everything that can be, and it is the expensive part. The
    # 2026-09-02 review filed the remaining per-cell Python as a cost; it is
    # the algorithm rather than the implementation. Batch 7 profiled that cost
    # (docs/measurements/2026-09-06-native-batch-7-candidates.md, B9): it was
    # ``TilesetRef.holds``'s property chain, not the loop -- see wang_field.
    field = wang_field(work, ref, wangset)
    cache: dict[tuple[tuple[int, int], ...], list[int]] = {}
    for y in range(y0, y1):
        for x in range(x0, x1):
            if (x, y) not in asserted:
                _wang_cell(work, ref, wangset, x, y, field, cache)
    return _finish(data, work, box)


def paint_wang(
    data: np.ndarray,
    x: int,
    y: int,
    colour: int,
    ref: TilesetRef,
    wangset: Any,
) -> Region | None:
    """One click: paint one cell a Wang colour and re-fit the ring around it."""
    return paint_wang_cells(data, [(x, y)], colour, ref, wangset)


def fill_wang(
    data: np.ndarray,
    x: int,
    y: int,
    colour: int,
    ref: TilesetRef,
    wangset: Any,
) -> Region | None:
    """Flood the connected run of one Wang colour, and re-fit the result.

    **The flood is over the colour field, not the gids** -- :func:`fill_terrain`'s
    rule, for its reason: matching the encoded value would stop at the boundary
    between two of one colour's own tiles and fill a one-cell ribbon. What a user
    means by "fill the grass" is the colour, so that is what is compared.

    A transition tile belongs to no single colour and is therefore its own group,
    which is what makes a fill stop at the edge of the field it was clicked in;
    the ring re-fit then reconciles that edge against what was just laid down.
    """
    height, width = np.asarray(data).shape
    x, y = int(x), int(y)
    if not (0 <= x < width and 0 <= y < height):
        return None
    field = wang_colour_field(data, ref, wangset)
    target = int(field[y, x])
    if target == int(colour):
        return None
    ys, xs = np.nonzero(flood_mask(field == target, x, y))
    return paint_wang_cells(
        data, list(zip(xs.tolist(), ys.tolist(), strict=True)), colour, ref, wangset
    )


#: What a cell whose tile mixes colours -- a transition -- ranks as in
#: :func:`wang_colour_field`. Distinct from the void so a fill clicked on an edge
#: tile floods the edge rather than the empty space beyond it.
WANG_MIXED: int = -1


def wang_colour_field(data: np.ndarray, ref: TilesetRef, wangset: Any) -> np.ndarray:
    """Every cell's single Wang colour, 0 for void and :data:`WANG_MIXED` for a
    transition tile.

    The Wang answer to :func:`rank_field`, and the comparison a fill floods over.
    Vectorised per *tile id* rather than per cell: a set has a few dozen tiles
    and a map has tens of thousands of cells.
    """
    ids = gidlib.tile_ids(np.asarray(data, dtype=gidlib.DTYPE))
    out = np.zeros(ids.shape, dtype=np.int16)
    locals_ = ids.astype(np.int64) - ref.firstgid
    inside = (ids >= ref.firstgid) & (ids <= ref.last_gid)
    for local, wangid in wangset.tiles.items():
        used = {wangid[slot] for slot in wangset.slots if wangid[slot]}
        out[inside & (locals_ == int(local))] = (
            used.pop() if len(used) == 1 else WANG_MIXED
        )
    return out


# A Wang erase is a plain erase, deliberately, and there is no ``erase_wang``
# beside the three above. ``constraints_from`` reads an empty cell as "nothing
# here has an opinion", so a hole constrains none of its neighbours and a re-fit
# around one would re-choose every ring cell against the same evidence it already
# has. The blob path needs ``erase_terrain`` because its collapse is
# *self against not-self* and a hole flips that bit; the general model has no
# such bit to flip.

