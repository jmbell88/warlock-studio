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


def _frozen(pixels: Any) -> np.ndarray:
    """A private RGBA copy nothing can write through."""
    array = np.ascontiguousarray(pixels, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError("a tileset image must be RGBA, shaped (h, w, 4)")
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "pixels", _frozen(self.pixels))
        for name in ("tile_w", "tile_h", "spacing", "margin"):
            object.__setattr__(self, name, int(getattr(self, name)))
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
