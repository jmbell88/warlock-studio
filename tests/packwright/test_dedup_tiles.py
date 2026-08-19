"""Dropping tiles a sheet already carries, at the repack door.

Exact bytes, first-in-reading-order wins, and off by default -- a repack stays
byte-faithful unless somebody asks otherwise. The orientation mode is the same
question asked of the eight dihedral variants, and the tests below pin that a
flipped duplicate survives *without* it, which is what makes the toggle mean
something.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.packwright.sources import (
    dedup_tiles,
    sprites_from_tileset,
    tile_key,
)


def _sheet(cells: list[list[np.ndarray]]) -> np.ndarray:
    """A tile sheet from a grid of equal-sized cells."""
    return np.concatenate([np.concatenate(row, axis=1) for row in cells], axis=0)


def _tile(size: int = 4, *, mark: tuple[int, int] | None = None,
          colour: tuple[int, int, int] = (200, 40, 40)) -> np.ndarray:
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., :3] = (30, 30, 30)
    out[..., 3] = 255
    if mark is not None:
        out[mark[0], mark[1], :3] = colour
    return out


def _sprites(sheet: np.ndarray, tile: int = 4) -> list:
    return sprites_from_tileset(sheet, tile=(tile, tile), prefix="p", name="n")


def test_exact_duplicates_are_dropped() -> None:
    one = _tile(mark=(0, 0))
    two = _tile(mark=(3, 3))
    sheet = _sheet([[one, two], [one.copy(), two.copy()]])
    kept, dropped = dedup_tiles(_sprites(sheet))
    assert dropped == 2
    assert len(kept) == 2


def test_the_reading_order_winner_is_kept() -> None:
    """``tile_key``'s zero-padding makes the lexical order the reading order,
    and which duplicate survives decides whose name the pack carries."""
    one = _tile(mark=(0, 0))
    sheet = _sheet([[one, one.copy()], [one.copy(), _tile(mark=(2, 2))]])
    kept, _dropped = dedup_tiles(_sprites(sheet))
    assert kept[0].key == tile_key("p", 0, 0)
    assert len(kept) == 2


def test_a_flipped_duplicate_survives_without_the_toggle() -> None:
    one = _tile(mark=(0, 0))
    flipped = one[:, ::-1].copy()
    sheet = _sheet([[one, flipped]])
    kept, dropped = dedup_tiles(_sprites(sheet))
    assert dropped == 0
    assert len(kept) == 2


def test_a_flipped_duplicate_drops_with_the_toggle() -> None:
    one = _tile(mark=(0, 0))
    flipped = one[:, ::-1].copy()
    sheet = _sheet([[one, flipped]])
    kept, dropped = dedup_tiles(_sprites(sheet), orientations=True)
    assert dropped == 1
    assert kept[0].key == tile_key("p", 0, 0)


def test_every_rotation_variant_is_caught() -> None:
    one = _tile(mark=(0, 1))
    variants = [one]
    for quarters in (1, 2, 3):
        variants.append(np.ascontiguousarray(np.rot90(one, quarters)))
    sheet = _sheet([variants])
    kept, dropped = dedup_tiles(_sprites(sheet), orientations=True)
    assert dropped == 3
    assert len(kept) == 1


def test_genuinely_different_tiles_are_all_kept() -> None:
    """Distinguished by how *many* pixels are marked, not by where -- two marks
    at (0, 1) and (1, 0) are transposes of one another and would rightly be
    called duplicates by the orientation mode."""
    tiles = []
    for count in range(1, 5):
        tile = _tile()
        for i in range(count):
            tile[i % 4, i // 4, :3] = (200, 40, 40)
        tiles.append(tile)
    sheet = _sheet([tiles[:2], tiles[2:]])
    kept, dropped = dedup_tiles(_sprites(sheet), orientations=True)
    assert dropped == 0
    assert len(kept) == 4


def test_an_rgb_opaque_sheet_keeps_every_cell_before_dedup() -> None:
    """A decode of an RGB file is opaque everywhere, so no cell is empty --
    which is the case where dedup is doing all the work."""
    one = _tile(mark=(0, 0))
    sheet = _sheet([[one, one.copy(), one.copy()]])
    assert len(_sprites(sheet)) == 3
    kept, dropped = dedup_tiles(_sprites(sheet))
    assert (len(kept), dropped) == (1, 2)


def test_dedup_of_nothing_is_nothing() -> None:
    assert dedup_tiles([]) == ([], 0)


def test_a_non_square_tile_transposes_to_a_shape_that_never_collides() -> None:
    """The four transposing variants cost nothing and refuse nothing on a
    non-square tile: their bytes simply do not match the untransposed ones."""
    one = np.zeros((2, 4, 4), dtype=np.uint8)
    one[..., 3] = 255
    one[0, 0, :3] = (200, 40, 40)
    sheet = np.concatenate([one, one.copy()], axis=1)
    sprites = sprites_from_tileset(sheet, tile=(4, 2), prefix="p", name="n")
    kept, dropped = dedup_tiles(sprites, orientations=True)
    assert (len(kept), dropped) == (1, 1)
