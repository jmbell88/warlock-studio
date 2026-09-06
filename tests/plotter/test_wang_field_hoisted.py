"""``terrain.wang_field`` hoists ``ref``/``wangset`` reads out of the per-cell
closure (docs/measurements/2026-09-06-native-batch-7-candidates.md, B9): the
old body called ``ref.holds(value)`` on every neighbour read, and ``holds``
recomputes ``last_gid`` through a ``max_local_id -> tile_count ->
columns/rows -> image_w/h`` property chain each time. Two claims: the hoisted
field answers exactly like the old one (parity), and a drag reads the bounds
a small constant number of times rather than once per neighbour (the
regression test, which fails against the unfixed code).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from warlock.studio.plotter import terrain
from warlock.studio.tilegrid import gid as gidlib
from warlock.studio.tilegrid.tileset import Tileset, TilesetRef
from warlock.studio.tilegrid.wang import WangColour, WangSet


def _pixels(size: int = 64) -> np.ndarray:
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., 3] = 255
    return out


def _corner_set() -> WangSet:
    """Two colours over the sixteen corner patterns -- a complete corner set,
    same fixture as tests/plotter/test_wang_paint.py."""
    tiles = {}
    for index in range(16):
        tiles[index] = (
            0,
            1 + ((index >> 0) & 1),
            0,
            1 + ((index >> 1) & 1),
            0,
            1 + ((index >> 2) & 1),
            0,
            1 + ((index >> 3) & 1),
        )
    return WangSet(
        name="Corners",
        kind="corner",
        colours=(WangColour("grass", "#00ff00"), WangColour("sand", "#ffff00")),
        tiles=tiles,
    )


def _ref(wangset: WangSet, firstgid: int = 1) -> TilesetRef:
    return TilesetRef(
        firstgid=firstgid,
        tileset=Tileset(
            name="w", pixels=_pixels(), tile_w=16, tile_h=16, wangsets=(wangset,)
        ),
    )


def _layer(width: int, height: int) -> np.ndarray:
    return gidlib.empty_layer(width, height)


def _shipped_wang_field(data: np.ndarray, ref: TilesetRef, wangset: Any) -> Any:
    """The pre-hoist ``terrain.wang_field`` body, spelled out here rather than
    imported: it calls ``ref.holds`` per neighbour read, which is exactly the
    cost batch 7 measured. Kept verbatim so the parity test compares against
    what actually shipped, not a paraphrase of it."""
    height, width = data.shape

    def field_of(x: int, y: int) -> Any:
        if not (0 <= x < width and 0 <= y < height):
            return None
        value = int(data[y, x]) & gidlib.GID_MASK
        if not value or not ref.holds(value):
            return None
        local = value - ref.firstgid
        return wangset.tiles.get(local)

    return field_of


def _mixed_layer(size: int, ref: TilesetRef, wangset: WangSet) -> np.ndarray:
    """A 24x24 layer with a mix of tiles: every corner pattern the set has,
    tiled across the layer, so a diagonal drag's neighbourhoods are varied
    rather than uniform."""
    data = _layer(size, size)
    tile_ids = sorted(wangset.tiles)
    for y in range(size):
        for x in range(size):
            local = tile_ids[(x + y * 3) % len(tile_ids)]
            data[y, x] = gidlib.DTYPE(ref.firstgid + local)
    return data


def _diagonal_cells(size: int) -> list[tuple[int, int]]:
    return [(i, i) for i in range(size)]


def test_hoisting_the_bounds_out_of_wang_field_leaves_the_refit_unchanged() -> None:
    wangset = _corner_set()
    ref = _ref(wangset)
    data = _mixed_layer(24, ref, wangset)
    cells = _diagonal_cells(24)

    shipped = terrain.wang_field
    try:
        terrain.wang_field = _shipped_wang_field
        before = terrain.paint_wang_cells(np.array(data), cells, 1, ref, wangset)
    finally:
        terrain.wang_field = shipped
    after = terrain.paint_wang_cells(np.array(data), cells, 1, ref, wangset)

    assert before is not None and after is not None
    bx0, by0, bblock = before
    ax0, ay0, ablock = after
    assert (bx0, by0) == (ax0, ay0)
    assert np.array_equal(bblock, ablock), "hoisting the bounds changed the re-fit"


def test_a_drag_reads_the_tileset_bounds_a_constant_number_of_times_not_per_neighbour() -> None:
    """The regression test: it fails against the unfixed ``wang_field``, whose
    closure calls ``ref.holds`` (and so ``last_gid``) on every neighbour read
    -- thousands of times over a 24x24 diagonal drag. The hoisted version reads
    the bounds once per ``wang_field`` call, so the count stays a small
    constant no matter how many cells are touched."""
    wangset = _corner_set()
    base_ref = _ref(wangset)
    calls = {"last_gid": 0}

    class CountingTilesetRef(TilesetRef):
        @property
        def last_gid(self) -> int:
            calls["last_gid"] += 1
            return super().last_gid

    ref = CountingTilesetRef(firstgid=base_ref.firstgid, tileset=base_ref.tileset)
    data = _mixed_layer(24, ref, wangset)
    cells = _diagonal_cells(24)

    region = terrain.paint_wang_cells(data, cells, 1, ref, wangset)
    assert region is not None

    # 24 touched cells plus their rings is on the order of a few hundred
    # neighbour reads; a per-neighbour ``holds`` call would push this into the
    # thousands. One ``wang_field`` call per box makes one read of last_gid.
    assert calls["last_gid"] <= 4, (
        f"last_gid was read {calls['last_gid']} times -- wang_field is not hoisting the bounds"
    )
