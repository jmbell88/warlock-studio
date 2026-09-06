"""materialize memoises oriented() per distinct raw ref.

docs/measurements/2026-08-30-native-batch-6-candidates.md §1 measured this at
a bit-identical 3.1x (444ms -> 142ms at 3200^2 with 8px tiles). These tests
pin bit-identical output against the old un-memoised loop, and pin the memo
itself: the second test fails against the unfixed code because it counts
calls to ``oriented``.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker import tiles
from warlock.studio.inker.tiles import materialize, oriented, strip
from warlock.studio.tilegrid import gid

RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)


def _tile(colour: tuple[int, int, int, int], w: int = 2, h: int = 2) -> np.ndarray:
    tile = np.zeros((h, w, 4), dtype=np.uint8)
    tile[..., 0] = colour[0]
    tile[..., 1] = colour[1]
    tile[..., 2] = colour[2]
    tile[..., 3] = colour[3]
    return tile


def _blank_tile(w: int = 2, h: int = 2) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


def _corner_tile() -> np.ndarray:
    """A 2x2 tile with four distinct corners so all eight symmetries differ."""
    tile = np.zeros((2, 2, 4), dtype=np.uint8)
    tile[0, 0] = (1, 0, 0, 255)
    tile[0, 1] = (0, 1, 0, 255)
    tile[1, 0] = (0, 0, 1, 255)
    tile[1, 1] = (255, 255, 0, 255)
    return tile


def _old_materialize(refs: np.ndarray, ts, size: tuple[int, int]) -> np.ndarray:
    """The pre-memo loop, spelled out here as the reference behaviour."""
    width, height = int(size[0]), int(size[1])
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    grid_h, grid_w = refs.shape
    tile_w, tile_h = ts.tile_w, ts.tile_h
    for row in range(grid_h):
        y0 = row * tile_h
        if y0 >= height:
            break
        for col in range(grid_w):
            x0 = col * tile_w
            if x0 >= width:
                break
            raw = int(refs[row, col])
            local = raw & gid.GID_MASK
            if local >= ts.tile_count:
                local = 0
            tile = oriented(ts.tile_pixels(local), raw)
            h = min(tile.shape[0], height - y0)
            w = min(tile.shape[1], width - x0)
            canvas[y0 : y0 + h, x0 : x0 + w] = tile[:h, :w]
    return canvas


def _mixed_tileset_and_refs():
    """Several tiles, all eight flag combos, an out-of-range id, and a canvas
    size that is not a multiple of the tile size (crop path)."""
    stack = np.stack(
        [_blank_tile(), _corner_tile(), _tile(RED), _tile(GREEN), _tile(BLUE)],
        axis=0,
    )
    ts = strip(stack)

    combos = [
        gid.compose(1, flip_h=fh, flip_v=fv, flip_d=fd)
        for fd in (False, True)
        for fv in (False, True)
        for fh in (False, True)
    ]
    # Row of the 8 flag combos on tile 1, then plain refs to other tiles,
    # then an out-of-range local id (must draw as tile 0 / blank), then a
    # repeat of an earlier raw ref to exercise the memo hit path.
    row0 = combos
    row1 = [2, 3, 4, 99] + combos[:4]
    refs = np.array([row0, row1], dtype=gid.DTYPE)
    return ts, refs


def test_materialize_memo_is_bit_identical_to_the_unmemoised_loop():
    ts, refs = _mixed_tileset_and_refs()
    grid_h, grid_w = refs.shape
    # Canvas not an exact multiple of the tile size -> crop path engaged.
    size = (grid_w * ts.tile_w - 1, grid_h * ts.tile_h - 1)

    got = materialize(refs, ts, size)
    expected = _old_materialize(refs, ts, size)

    assert np.array_equal(got, expected)


def test_materialize_calls_oriented_at_most_once_per_distinct_raw_ref(monkeypatch):
    ts, refs = _mixed_tileset_and_refs()
    grid_h, grid_w = refs.shape
    size = (grid_w * ts.tile_w - 1, grid_h * ts.tile_h - 1)

    calls: dict[int, int] = {}
    real_oriented = oriented

    def counting_oriented(tile, raw):
        calls[raw] = calls.get(raw, 0) + 1
        return real_oriented(tile, raw)

    monkeypatch.setattr(tiles, "oriented", counting_oriented)

    materialize(refs, ts, size)

    distinct_raws = {int(r) for r in refs.flatten()}
    assert set(calls) == distinct_raws
    assert all(count == 1 for count in calls.values())
