"""A terrain-carrying tileset, built for a test rather than generated.

``tilegen.generate`` used to be this file. It was product code -- Plotter's
procedural ground generator -- and a dozen painting, export and round-trip tests
had quietly adopted it as a fixture factory because it was the shortest way to
get a ``Tileset`` whose rows meant something. When tile sheets moved to Create
on 2026-08-18 the generator went with them, and those tests turned out to be
about *painting*, which stayed.

So the factory is here now, and being in ``tests/`` is the point: nothing in it
is a claim about what the app draws. The **shape** is the whole contract -- a
row of ``blob.TILE_COUNT`` tiles per terrain per phase, in the order
``Tileset.tile_for`` indexes them -- and every test that used the generator
asserts on gids and cases rather than on pixels. The pixels are deterministic
noise, so a test that accidentally starts depending on them fails the same way
on every machine instead of drifting.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.plotter.terrain import DEFAULT_TERRAINS
from warlock.studio.tilegrid import blob
from warlock.studio.tilegrid.tileset import Tileset, TilesetRef

__all__ = ["DEFAULT_TERRAINS", "terrain_tileset", "terrain_ref"]


def terrain_tileset(
    *,
    terrains=None,
    count: int | None = None,
    tile_w: int = 8,
    tile_h: int = 8,
    name: str = "Ground",
    phases: int = 1,
) -> Tileset:
    """A tileset with the rows a terrain set needs and nothing else.

    ``count`` takes that many of :data:`DEFAULT_TERRAINS`, which is what most
    callers want; ``terrains`` passes a list through verbatim for the few that
    care which ones.
    """
    rows = tuple(terrains if terrains is not None else DEFAULT_TERRAINS[: count or 5])
    if not rows:
        raise ValueError("a terrain set needs at least one terrain")
    # **47 columns, one blob case each; one row per terrain per phase.**
    # ``Tileset.__post_init__`` refuses any other shape for a set that declares
    # terrains, and ``tile_for`` indexes it as
    # ``(terrain * phases**2 + phase) * TILE_COUNT + case`` -- so the row count
    # is what carries the phases, not the width.
    width = blob.TILE_COUNT * tile_w
    height = len(rows) * phases * phases * tile_h
    # Deterministic and cheap, and distinct per pixel so a test that means to
    # tell two tiles apart can. Opaque throughout: a terrain tile is opaque by
    # definition, and a stray alpha would make a compositing test lie.
    flat = (np.arange(width * height * 4, dtype=np.int64) * 7 + 13) % 251
    pixels = flat.astype(np.uint8).reshape(height, width, 4)
    pixels[:, :, 3] = 255
    return Tileset(
        name=name,
        pixels=pixels,
        tile_w=tile_w,
        tile_h=tile_h,
        terrains=rows,
        phases=phases,
    )


def terrain_ref(*, firstgid: int = 1, **kwargs) -> TilesetRef:
    """:func:`terrain_tileset` as a map holds it."""
    return TilesetRef(firstgid=firstgid, tileset=terrain_tileset(**kwargs))
