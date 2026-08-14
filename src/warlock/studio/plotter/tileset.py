"""A tileset: one image, sliced into a grid.

Frozen, and its pixels are copied and made read-only on construction, for the
reason ``clay.mesh.Mesh`` is immutable -- the UI keys a GPU texture upload on
``id(tileset.pixels)``, so an in-place edit anywhere would leave the cache
holding a live key over stale pixels and the map would render last week's tiles
forever with nothing in the data to say why.

**The slicing is validated here, not at the first draw.** A geometry that does
not fit the image -- a margin larger than the image, a spacing that leaves a
partial final column -- is a file or a form the user got wrong, and the useful
moment to say so is when the tileset is made. Deferring it produces a tileset
that exists, appears in the list, and draws garbage.

Only *image* tilesets exist here. Tiled's "collection of images" variant, where
each tile is its own file, is refused by :mod:`.tsx` and :mod:`.tmx` rather than
half-supported: the whole model below assumes one atlas and a tile id that is an
index into its grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import blob

RGBA = tuple[int, int, int, int]


def rgba_colour(colour: Any, what: str) -> RGBA:
    """Four channels of 0..255 as the hashable tuple every colour here is.

    Public for :func:`frozen_rgba`'s reason and no other: a second caller
    turned up. :class:`~._map_model.TileLayer` and its three siblings each
    carry a ``tint``, and a private copy of these four lines beside them is how
    one of the two comes to accept a channel the other refuses. ``what`` names
    the thing in the refusal, because "a terrain fill must be four channels" is
    the wrong sentence to show someone tinting a layer.
    """
    values = tuple(int(part) for part in colour)
    if len(values) != 4 or any(part < 0 or part > 255 for part in values):
        raise ValueError(f"{what} must be four channels of 0..255")
    return values  # type: ignore[return-value]


@dataclass(frozen=True)
class TerrainSpec:
    """One terrain a tileset declares: what it is called and what it is made of.

    Here rather than in :mod:`.terrain` because it is a *field* of
    :class:`Tileset` -- the roles are a fact about the atlas, not about any map
    that loads it, so the same generated set embedded in two ``.wmap`` files
    carries them in both. Putting it next door would make the two modules import
    each other.

    **A terrain's position in the tuple is its precedence**, which is the whole
    of how a cell with three terrains around it picks one picture: a cell's blob
    membership is every neighbour ranked at or above its own. So this is a
    tuple, and every serialisation of it is an ordered list.
    """

    name: str
    fill: RGBA
    outline: RGBA

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "fill", rgba_colour(self.fill, "a terrain fill"))
        object.__setattr__(self, "outline", rgba_colour(self.outline, "a terrain outline"))


def frozen_rgba(pixels: Any, what: str = "a tileset image") -> np.ndarray:
    """A private RGBA copy nothing can write through.

    Public, and shared with ``packwright.sources``, which held a byte-identical
    copy differing only in the noun in its error message. The rule is one rule:
    the UI keys a texture upload on the array's identity, so an in-place edit
    would leave the cache holding a live key over stale pixels -- and two
    spellings of it is how one of them comes to be relaxed alone.

    ``what`` names the thing in the refusal, because "a tileset image must be
    RGBA" is the wrong sentence to show someone packing a sprite.
    """
    array = np.ascontiguousarray(pixels, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError(f"{what} must be RGBA, shaped (h, w, 4)")
    array = array.copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Tileset:
    """One sliced atlas. ``name`` is what the map calls it and what a ``.tsx``
    is written under; it is not an identity -- two tilesets may share a name and
    the map distinguishes them by position."""

    name: str
    pixels: np.ndarray
    tile_w: int
    tile_h: int
    spacing: int = 0
    margin: int = 0
    # Preserved verbatim from a .tsx so a round trip does not quietly drop it.
    properties: dict[str, Any] = field(default_factory=dict)
    # Empty for an ordinary atlas. Non-empty makes this a *terrain set*: one row
    # per terrain, one column per blob case, so a tile's role is its position.
    terrains: tuple[TerrainSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "pixels", frozen_rgba(self.pixels))
        for name in ("tile_w", "tile_h", "spacing", "margin"):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "terrains", tuple(self.terrains))
        if self.tile_w < 1 or self.tile_h < 1:
            raise ValueError("a tile must be at least one pixel across")
        if self.spacing < 0 or self.margin < 0:
            raise ValueError("spacing and margin cannot be negative")
        if self.columns < 1 or self.rows < 1:
            raise ValueError(
                f"a {self.image_w}x{self.image_h} image holds no "
                f"{self.tile_w}x{self.tile_h} tiles at margin {self.margin}, "
                f"spacing {self.spacing}"
            )
        # Validated here for the reason the slicing above is: a terrain set
        # whose geometry does not match its declaration is a file or a form
        # somebody got wrong, and the useful moment to say so is now rather than
        # when a paint stroke indexes past the end of a row.
        if self.terrains:
            if self.columns != blob.TILE_COUNT:
                raise ValueError(
                    f"a terrain set is {blob.TILE_COUNT} columns wide, one blob case "
                    f"per column; this one is {self.columns}"
                )
            if self.rows < len(self.terrains):
                raise ValueError(
                    f"{len(self.terrains)} terrains need {len(self.terrains)} rows, "
                    f"and this image holds {self.rows}"
                )

    # -- geometry ------------------------------------------------------------

    @property
    def image_w(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def image_h(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def columns(self) -> int:
        """How many whole tiles fit across.

        Tiled's own formula: the margin is taken off both edges, and each tile
        after the first costs its own width plus one spacing. A partial tile at
        the end is not counted -- it is not a tile.
        """
        span = self.image_w - 2 * self.margin + self.spacing
        step = self.tile_w + self.spacing
        return max(0, span // step) if step else 0

    @property
    def rows(self) -> int:
        span = self.image_h - 2 * self.margin + self.spacing
        step = self.tile_h + self.spacing
        return max(0, span // step) if step else 0

    @property
    def tile_count(self) -> int:
        return self.columns * self.rows

    def tile_rect(self, local_id: int) -> tuple[int, int, int, int]:
        """``(x, y, w, h)`` of one tile within the image, by local id.

        Local ids run row-major from zero, which is what ``firstgid`` is added
        to. Out of range raises: a caller asking for a tile this set does not
        have is a bug upstream, and returning the first tile would draw a
        plausible wrong picture.
        """
        index = int(local_id)
        if index < 0 or index >= self.tile_count:
            raise IndexError(f"tile {index} is outside this tileset (0..{self.tile_count - 1})")
        column, row = index % self.columns, index // self.columns
        return (
            self.margin + column * (self.tile_w + self.spacing),
            self.margin + row * (self.tile_h + self.spacing),
            self.tile_w,
            self.tile_h,
        )

    def tile_pixels(self, local_id: int) -> np.ndarray:
        """One tile's pixels. A read-only *view*, since the base is read-only:
        a caller that wants to write copies, and one that does not pays
        nothing."""
        x, y, w, h = self.tile_rect(local_id)
        return self.pixels[y : y + h, x : x + w]

    # -- terrain -------------------------------------------------------------

    @property
    def is_terrain_set(self) -> bool:
        return bool(self.terrains)

    def terrain_of(self, local_id: int) -> int | None:
        """Which terrain a tile belongs to, or ``None`` if it is not one.

        **This is the only place a cell's terrain comes from.** There is no
        parallel per-cell terrain array anywhere in the package: the gid names a
        tileset and a local id, the local id's row names the terrain, and one
        answer cannot disagree with itself. A stored field would be a second
        source of truth that ``.wmap``, undo and every Tiled round trip would
        each have to keep honest.
        """
        if not self.terrains:
            return None
        index = int(local_id)
        if index < 0 or index >= self.tile_count:
            return None
        row = index // blob.TILE_COUNT
        return row if row < len(self.terrains) else None

    def local_for(self, terrain: int, blob_index: int) -> int:
        """The local id of one terrain's one blob case. The layout, in one line."""
        row, column = int(terrain), int(blob_index)
        if row < 0 or row >= len(self.terrains):
            raise IndexError(f"terrain {row} is outside this set (0..{len(self.terrains) - 1})")
        if column < 0 or column >= blob.TILE_COUNT:
            raise IndexError(f"blob case {column} is outside 0..{blob.TILE_COUNT - 1}")
        return row * blob.TILE_COUNT + column

    def uv(self, local_id: int) -> tuple[float, float, float, float]:
        """``(u0, v0, u1, v1)`` for one tile, for a draw-list quad.

        Here rather than in the pane because it is arithmetic over the slicing
        this class owns, and because a preview, a canvas and a palette grid all
        need the same answer.
        """
        x, y, w, h = self.tile_rect(local_id)
        return (
            x / self.image_w,
            y / self.image_h,
            (x + w) / self.image_w,
            (y + h) / self.image_h,
        )


@dataclass(frozen=True)
class TilesetRef:
    """A tileset as the *map* holds it: the set, plus where its ids begin.

    ``firstgid`` is a property of the map's tileset list rather than of the
    tileset, which is why it lives out here -- the same ``.tsx`` used by two
    maps has a different firstgid in each, and baking it into
    :class:`Tileset` would make that one shared object two.
    """

    firstgid: int
    tileset: Tileset
    # Where the tileset came from, when it came from a file. Carried so a
    # ``.tmx`` export can reference the external ``.tsx`` a user already has
    # rather than writing a second copy of it under a new name.
    source: str = ""

    @property
    def last_gid(self) -> int:
        """The highest gid this reference answers for."""
        return self.firstgid + self.tileset.tile_count - 1

    def holds(self, tile_id: int) -> bool:
        return self.firstgid <= int(tile_id) <= self.last_gid

    def local(self, tile_id: int) -> int:
        return int(tile_id) - self.firstgid
