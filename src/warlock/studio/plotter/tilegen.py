"""The base ground set, made rather than drawn.

The only module in this package that produces pixels from nothing, and the one
whose determinism is a promise: **the same spec gives the same bytes**, on any
machine and any Python. That is not tidiness. The whole point of generating a
base set is that an artist paints a polished one over it and can diff the two,
and a generator that drifted by a pixel between runs would make that diff
meaningless.

What holds the promise: no RNG anywhere; no floating point in any coverage
decision (the isometric run length is integer floor division); no antialiasing,
which would put the platform's libm in the middle of the answer and is wrong for
the art style besides; and no dependence on dict or hash order, since the blob
table is derived from a ``range`` and the terrain order is the caller's own
tuple.

**The art is deliberately plain: a flat fill with a one-pixel darker outline on
every exposed side.** A ground tile is opaque, so the silhouette never changes
and only the outline moves -- which is why a complete forty-seven-case set is a
few lines rather than a drawing program. The single pixel bitten out of a
*notched* corner is the only thing separating the forty-seven from the sixteen,
and it is what makes two fields meeting diagonally read as two fields rather
than as a staircase.

Isometric reuses every one of those decisions. The cell is a diamond inscribed
in the same rectangle, the grid axes run diagonally on screen, and so the four
cell directions land on the diamond's four sides -- same predicates, one
projection-specific side table. That is why both projections ship together
rather than the second being a rewrite of the first.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import blob, project
from .terrain import DEFAULT_TERRAINS
from .tileset import TerrainSpec, Tileset

#: Coverage codes for :func:`tile_coverage` -- the whole art style, as one
#: small integer per pixel, so a test can assert the shape without a colour.
VOID = 0
FILL = 1
OUTLINE = 2

#: More than this and the atlas is wider than anything will preview usefully,
#: and the precedence chain stops being something a person can hold in mind.
MAX_TERRAINS = 8


@dataclass(frozen=True)
class GenSpec:
    """Everything the generator reads. Frozen and made of plain values, so it is
    hashable, comparable, and safe to hand to a task thread."""

    terrains: tuple[TerrainSpec, ...] = field(default=DEFAULT_TERRAINS)
    tile_w: int = 32
    tile_h: int = 32
    projection: str = project.ORTHOGONAL
    name: str = "Ground"

    def __post_init__(self) -> None:
        object.__setattr__(self, "terrains", tuple(self.terrains))
        object.__setattr__(self, "tile_w", int(self.tile_w))
        object.__setattr__(self, "tile_h", int(self.tile_h))
        object.__setattr__(self, "projection", project.check(self.projection))
        if not self.terrains:
            raise ValueError("a ground set needs at least one terrain")
        if len(self.terrains) > MAX_TERRAINS:
            raise ValueError(f"a ground set holds at most {MAX_TERRAINS} terrains")
        if self.tile_w < 4 or self.tile_h < 4:
            raise ValueError("a generated tile is at least four pixels across")
        names = [entry.name.strip().lower() for entry in self.terrains]
        if "" in names:
            raise ValueError("every terrain needs a name")
        if len(set(names)) != len(names):
            raise ValueError("two terrains cannot share a name")


def _diamond_spans(tile_w: int, tile_h: int) -> tuple[tuple[int, int], ...]:
    """``(lo, hi)`` half-open pixel columns of the diamond, per row.

    Integer arithmetic throughout -- see the module docstring on determinism.
    The half-width comes from the row's distance to the nearer horizontal end,
    which makes the shape symmetric about both axes from one expression rather
    than two that could disagree; the span is then centred by construction, so
    a row is never one pixel off-centre from the row mirroring it.
    """
    spans: list[tuple[int, int]] = []
    for row in range(tile_h):
        nearer = min(row, tile_h - 1 - row) + 1
        half = max(1, (tile_w * nearer) // tile_h)
        lo = max(0, tile_w // 2 - half)
        hi = min(tile_w, tile_w // 2 + half)
        spans.append((lo, hi))
    return tuple(spans)


def tile_coverage(spec: GenSpec, blob_index: int) -> np.ndarray:
    """One tile as :data:`VOID` / :data:`FILL` / :data:`OUTLINE` codes.

    The whole art style lives here, in one array and with no colour in sight, so
    a test can assert "this case has an outline along its north edge" without
    knowing what grass looks like.
    """
    tile_w, tile_h = spec.tile_w, spec.tile_h
    north, east, south, west = blob.open_edges(blob_index)
    ne, se, sw, nw = blob.open_corners(blob_index)
    out = np.zeros((tile_h, tile_w), dtype=np.uint8)

    if spec.projection == project.ISOMETRIC:
        spans = _diamond_spans(tile_w, tile_h)
        upper_last = tile_h // 2 - 1
        for row, (lo, hi) in enumerate(spans):
            if hi <= lo:
                continue
            out[row, lo:hi] = FILL
            # In an isometric lattice +column runs down-right and +row runs
            # down-left, so the four *cell* directions are the diamond's four
            # sides: north is upper-right, east is lower-right, south is
            # lower-left, west is upper-left. Same predicates as the orthogonal
            # branch; only the pixels they reach differ.
            upper = row <= upper_last
            if (west if upper else south):
                out[row, lo] = OUTLINE
            if (north if upper else east):
                out[row, hi - 1] = OUTLINE
        # A notched corner is a *vertex* of the diamond: right is north-east,
        # bottom south-east, left south-west, top north-west. An even-height
        # diamond has two widest rows, so the side vertices are two pixels tall
        # -- marking one of them would leave the tile asymmetric against the
        # case that mirrors it, which is visible as a one-pixel wobble along an
        # otherwise straight run of coastline.
        span_of = [hi - lo for lo, hi in spans]
        widest = [row for row, width in enumerate(span_of) if width == max(span_of)]
        for row in widest:
            if ne:
                out[row, spans[row][1] - 1] = OUTLINE
            if sw:
                out[row, spans[row][0]] = OUTLINE
        for notched, row in ((nw, 0), (se, tile_h - 1)):
            lo, hi = spans[row]
            if notched and hi > lo:
                out[row, lo:hi] = OUTLINE
        return out

    out[:, :] = FILL
    if north:
        out[0, :] = OUTLINE
    if south:
        out[tile_h - 1, :] = OUTLINE
    if west:
        out[:, 0] = OUTLINE
    if east:
        out[:, tile_w - 1] = OUTLINE
    # A notched corner is the narrow case: both flanking sides have a neighbour,
    # so neither drew an outline, and yet the diagonal does not -- leaving one
    # pixel that belongs to the boundary and would otherwise be plain fill.
    if ne:
        out[0, tile_w - 1] = OUTLINE
    if se:
        out[tile_h - 1, tile_w - 1] = OUTLINE
    if sw:
        out[tile_h - 1, 0] = OUTLINE
    if nw:
        out[0, 0] = OUTLINE
    return out


def atlas_pixels(spec: GenSpec) -> np.ndarray:
    """The whole atlas: one row per terrain, one column per blob case."""
    tile_w, tile_h = spec.tile_w, spec.tile_h
    rows, columns = len(spec.terrains), blob.TILE_COUNT
    out = np.zeros((rows * tile_h, columns * tile_w, 4), dtype=np.uint8)
    # The coverage of a case does not depend on the terrain, so it is computed
    # once per case and coloured once per terrain -- forty-seven rasterisations
    # for any number of terrains rather than forty-seven times as many.
    coverage = [tile_coverage(spec, case) for case in range(columns)]
    for row, entry in enumerate(spec.terrains):
        fill = np.array(entry.fill, dtype=np.uint8)
        outline = np.array(entry.outline, dtype=np.uint8)
        y0 = row * tile_h
        for column, codes in enumerate(coverage):
            x0 = column * tile_w
            tile = out[y0 : y0 + tile_h, x0 : x0 + tile_w]
            tile[codes == FILL] = fill
            tile[codes == OUTLINE] = outline
    return out


def generate(spec: GenSpec) -> Tileset:
    """The atlas as a tileset that already knows what its rows mean."""
    return Tileset(
        name=spec.name,
        pixels=atlas_pixels(spec),
        tile_w=spec.tile_w,
        tile_h=spec.tile_h,
        terrains=spec.terrains,
    )


def repolish(original: Tileset, pixels: np.ndarray) -> Tileset:
    """The same terrain set wearing new art.

    Refused by name when the image size changed, because **the roles are
    positional**: an atlas cropped or padded in a paint program is a set whose
    tile ninety-three is no longer sand's interior, and accepting it would
    repaint an entire map wrong while every gid in it stayed valid. The message
    says the size to come back with, because a refusal that does not is a
    refusal that sends someone to a forum.
    """
    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[:2] != original.pixels.shape[:2]:
        got = "x".join(str(part) for part in array.shape[1::-1]) if array.ndim >= 2 else "?"
        raise ValueError(
            f"this atlas is {got}, and the tileset it replaces is "
            f"{original.image_w}x{original.image_h}; resize it and try again"
        )
    # ``replace`` rather than mutation because ``Tileset`` is frozen, and the
    # new array identity is load-bearing: the pane's texture cache is keyed on
    # ``id(tileset.pixels)``, so a replaced set re-uploads with no invalidation
    # call anywhere.
    return replace(original, pixels=array)
