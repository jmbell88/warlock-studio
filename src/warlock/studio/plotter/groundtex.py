"""The same forty-seven cases, drawn in two materials instead of two colours.

:mod:`.tilegen` makes a *plain* set on purpose -- a flat fill with a one-pixel
darker outline -- because it has no art to draw with. This module is the same
geometry handed two seamless textures per terrain: a fill material and a border
material, with the blob edges given real thickness so a shoreline reads as a
rocky rim rather than as a line.

**It still produces no pixels of its own.** Every pixel it writes comes from one
of the two textures the caller passed, chosen by an integer coverage code -- so
this module inherits :mod:`.tilegen`'s determinism promise wholesale: no RNG, no
floating point in any coverage decision, no antialiasing. The AI half of the
feature (prompting SDXL for the two textures, reducing them to tile size) lives
in ``warlock.pipelines.ground`` and never comes near this package.

**A texture must arrive at exactly the tile size, and that is the whole seam
trick.** The generated textures are seamless on a torus of their own period; a
period reduced to precisely ``tile_w`` by ``tile_h`` means a tile samples the
material at its own local coordinates, so two FULL tiles painted side by side on
a map continue the pattern with nothing else in the machinery aware of it. Any
other period would need a per-cell phase, which a tile *atlas* cannot express.

The band geometry is the outline geometry widened: at ``border == 1`` every case
reproduces :func:`.tilegen.tile_coverage`'s footprint exactly, which is what
``tests/plotter/test_groundtex.py`` asserts case by case. That bridge is the
reason there is one art rule here rather than two that drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import blob, project
from .terrain import DEFAULT_TERRAINS
from .tilegen import MAX_TERRAINS, MAX_TILE_EDGE, _diamond_spans
from .tileset import TerrainSpec

#: Coverage codes for :func:`band_coverage`. Deliberately the same integers as
#: :mod:`.tilegen`'s ``VOID``/``FILL``/``OUTLINE``, so the border-one bridge test
#: can compare the two arrays without a translation table in the middle of it.
VOID = 0
FILL = 1
BORDER = 2


def default_border(tile_w: int, tile_h: int) -> int:
    """How thick a rim is, when nobody said.

    An eighth of the shorter edge, which is four pixels on the house 32px tile
    and stays proportionate up and down -- a fixed pixel count would be a hairline
    on a 128px tile and would swallow a 8px one whole. Never less than two, since
    one is :mod:`.tilegen`'s line and the point here is thickness; and never so
    thick that two opposite bands meet in the middle, because a tile whose fill
    has been entirely eaten shows no material at all and every case looks alike.
    """
    shorter = min(int(tile_w), int(tile_h))
    return max(1, min(max(2, shorter // 8), (shorter - 1) // 2))


def band_coverage(
    tile_w: int,
    tile_h: int,
    projection: str,
    blob_index: int,
    border: int,
) -> np.ndarray:
    """One tile as :data:`VOID` / :data:`FILL` / :data:`BORDER` codes.

    :func:`.tilegen.tile_coverage` with the outline widened to ``border``. The
    bands union rather than layer -- there is one border code, so a corner where
    two open sides meet is simply border twice -- which is what keeps a notched
    corner from needing to know whether its flanks drew anything.
    """
    tile_w, tile_h = int(tile_w), int(tile_h)
    border = max(1, int(border))
    north, east, south, west = blob.open_edges(blob_index)
    ne, se, sw, nw = blob.open_corners(blob_index)
    out = np.zeros((tile_h, tile_w), dtype=np.uint8)

    if project.check(projection) == project.ISOMETRIC:
        # A diamond's band runs along a slope, so thickness measured across the
        # row is thickness measured along the edge divided by the slope -- half
        # of it, for a diamond inscribed in its own rectangle. Halving the count
        # here rather than scaling the whole rim keeps a 32px iso tile's rim
        # visually the same weight as an orthogonal one's.
        thick = max(1, border // 2)
        spans = _diamond_spans(tile_w, tile_h)
        upper_last = tile_h // 2 - 1
        for row, (lo, hi) in enumerate(spans):
            if hi <= lo:
                continue
            out[row, lo:hi] = FILL
            run = min(thick, hi - lo)
            # In an isometric lattice +column runs down-right and +row runs
            # down-left, so the four *cell* directions are the diamond's four
            # sides: north is upper-right, east is lower-right, south is
            # lower-left, west is upper-left. Same table as ``tile_coverage``.
            upper = row <= upper_last
            if west if upper else south:
                out[row, lo : lo + run] = BORDER
            if north if upper else east:
                out[row, hi - run : hi] = BORDER
        # A notched corner is a *vertex* of the diamond: right is north-east,
        # bottom south-east, left south-west, top north-west. An even-height
        # diamond has two widest rows, so the side vertices are two pixels tall
        # -- marking one of them would leave the tile asymmetric against the case
        # that mirrors it, visible as a one-pixel wobble along a straight run.
        span_of = [hi - lo for lo, hi in spans]
        widest = [row for row, width in enumerate(span_of) if width == max(span_of)]
        for row in widest:
            lo, hi = spans[row]
            run = min(thick, hi - lo)
            if ne:
                out[row, hi - run : hi] = BORDER
            if sw:
                out[row, lo : lo + run] = BORDER
        for notched, rows in ((nw, range(0, thick)), (se, range(tile_h - thick, tile_h))):
            if not notched:
                continue
            for row in rows:
                if 0 <= row < tile_h:
                    lo, hi = spans[row]
                    if hi > lo:
                        out[row, lo:hi] = BORDER
        return out

    out[:, :] = FILL
    if north:
        out[0:border, :] = BORDER
    if south:
        out[tile_h - border : tile_h, :] = BORDER
    if west:
        out[:, 0:border] = BORDER
    if east:
        out[:, tile_w - border : tile_w] = BORDER
    # A notched corner is the narrow case: both flanking sides have a neighbour,
    # so neither drew a band, and yet the diagonal does not -- leaving a square
    # of the corner that belongs to the boundary and would otherwise be fill.
    if ne:
        out[0:border, tile_w - border : tile_w] = BORDER
    if se:
        out[tile_h - border : tile_h, tile_w - border : tile_w] = BORDER
    if sw:
        out[tile_h - border : tile_h, 0:border] = BORDER
    if nw:
        out[0:border, 0:border] = BORDER
    return out


def _material(pixels: np.ndarray, tile_w: int, tile_h: int, what: str) -> np.ndarray:
    """One texture as ``(tile_h, tile_w, 4)`` uint8, or a refusal naming both
    sizes.

    Refused rather than resized: the caller reduced this texture on purpose and
    to an exact period (see the module docstring), so a texture arriving at the
    wrong size is a bug upstream, and quietly stretching it would break the
    tile-to-tile seam in a way that only shows up as a faint grid on a painted
    map.
    """
    array = np.asarray(pixels, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"{what} must be RGB or RGBA, shaped (h, w, 3 or 4)")
    if array.shape[0] != tile_h or array.shape[1] != tile_w:
        raise ValueError(
            f"{what} is {array.shape[1]}x{array.shape[0]}, and this set's tile is "
            f"{tile_w}x{tile_h}; reduce it to the tile size first"
        )
    if array.shape[2] == 3:
        opaque = np.full((tile_h, tile_w, 4), 255, dtype=np.uint8)
        opaque[:, :, :3] = array
        return opaque
    return np.ascontiguousarray(array)


def compose_atlas(
    tile_w: int,
    tile_h: int,
    projection: str,
    border: int,
    fills,
    borders,
) -> np.ndarray:
    """The whole atlas: one row per terrain, one column per blob case.

    ``fills`` and ``borders`` are one tile-sized texture each per terrain, in the
    terrain order the set declares. Integer-only and RNG-free, so the same
    textures give the same bytes -- the promise :mod:`.tilegen` makes, kept by a
    module whose input happens to have come from a diffusion model.
    """
    fill_list = [
        _material(tex, tile_w, tile_h, f"terrain {i + 1}'s fill texture")
        for i, tex in enumerate(fills)
    ]
    border_list = [
        _material(tex, tile_w, tile_h, f"terrain {i + 1}'s border texture")
        for i, tex in enumerate(borders)
    ]
    if not fill_list:
        raise ValueError("a ground set needs at least one terrain")
    if len(fill_list) != len(border_list):
        raise ValueError(
            f"{len(fill_list)} fill textures and {len(border_list)} border textures; "
            "a ground set needs one of each per terrain"
        )

    columns = blob.TILE_COUNT
    out = np.zeros((len(fill_list) * tile_h, columns * tile_w, 4), dtype=np.uint8)
    # The coverage of a case does not depend on the terrain, so it is computed
    # once per case and painted once per terrain -- forty-seven rasterisations
    # for any number of terrains rather than forty-seven times as many.
    coverage = [
        band_coverage(tile_w, tile_h, projection, case, border) for case in range(columns)
    ]
    for row, (fill_tex, border_tex) in enumerate(zip(fill_list, border_list, strict=True)):
        y0 = row * tile_h
        for column, codes in enumerate(coverage):
            x0 = column * tile_w
            tile = out[y0 : y0 + tile_h, x0 : x0 + tile_w]
            tile[codes == FILL] = fill_tex[codes == FILL]
            tile[codes == BORDER] = border_tex[codes == BORDER]
            # VOID stays fully transparent: outside the diamond is not this
            # cell. Everything else is forced fully *opaque*, which is a second
            # statement and is stated because it is easy to miss: a generated
            # material carrying partial alpha -- which ``_material`` accepts --
            # is flattened to solid here. That is right for a ground tile (a
            # half-transparent floor is a hole in the map) and it means the
            # atlas is not a general compositor for arbitrary RGBA input.
            tile[codes != VOID, 3] = 255
    return out


@dataclass(frozen=True)
class GroundSpec:
    """Everything the compositor reads, validated once.

    :class:`.tilegen.GenSpec`'s rules, read off the same two module constants so
    the two generators cannot come to disagree about how many terrains a set
    holds -- plus the band thickness, which is the only thing this set has that
    a flat one does not.
    """

    terrains: tuple[TerrainSpec, ...] = field(default=DEFAULT_TERRAINS)
    tile_w: int = 32
    tile_h: int = 32
    projection: str = project.ORTHOGONAL
    border: int = 0
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
        if self.tile_w > MAX_TILE_EDGE or self.tile_h > MAX_TILE_EDGE:
            raise ValueError(f"a generated tile is at most {MAX_TILE_EDGE} pixels across")
        names = [entry.name.strip().lower() for entry in self.terrains]
        if "" in names:
            raise ValueError("every terrain needs a name")
        if len(set(names)) != len(names):
            raise ValueError("two terrains cannot share a name")
        # Zero means "whatever suits this tile", which is what the form sends
        # when the user did not touch the control.
        chosen = int(self.border) or default_border(self.tile_w, self.tile_h)
        shorter = min(self.tile_w, self.tile_h)
        if chosen < 1 or 2 * chosen >= shorter:
            raise ValueError(
                f"a {self.tile_w}x{self.tile_h} tile takes a border of "
                f"1..{(shorter - 1) // 2} pixels"
            )
        object.__setattr__(self, "border", chosen)

    def atlas(self, fills, borders) -> np.ndarray:
        """:func:`compose_atlas` with this spec's numbers already filled in."""
        return compose_atlas(self.tile_w, self.tile_h, self.projection, self.border, fills, borders)
