"""Slicing an already-made tile sheet into sprites.

The rules under test: full cells only (a remainder strip is outside the grid),
an empty cell is dropped, an all-opaque sheet keeps every cell, and the keys
are zero-padded row-major so the packer's lexical sort is the sheet's reading
order.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.packwright.layout import MAX_SPRITES
from warlock.studio.packwright.sources import (
    sprites_from_tileset,
    tile_key,
    tileset_occupancy,
)


def _sheet(rows: int, columns: int, tile: int = 4, *, empty: set | None = None) -> np.ndarray:
    """A sheet of ``rows x columns`` cells, each carrying one opaque pixel
    unless its ``(row, column)`` is in ``empty``."""
    pixels = np.zeros((rows * tile, columns * tile, 4), dtype=np.uint8)
    for row in range(rows):
        for column in range(columns):
            if empty and (row, column) in empty:
                continue
            pixels[row * tile + 1, column * tile + 1] = (10 * row, 10 * column, 5, 255)
    return pixels


def test_occupancy_reads_the_grid() -> None:
    grid = tileset_occupancy(_sheet(2, 3, empty={(1, 2)}), tile=(4, 4))
    assert grid.shape == (2, 3)
    assert grid.sum() == 5
    assert not grid[1, 2]


def test_a_remainder_strip_is_outside_the_grid() -> None:
    sheet = _sheet(2, 2)
    widened = np.zeros((8, 11, 4), dtype=np.uint8)
    widened[:, :8] = sheet
    widened[0, 10] = (1, 1, 1, 255)  # opaque, but in the 3px remainder
    grid = tileset_occupancy(widened, tile=(4, 4))
    assert grid.shape == (2, 2)
    assert grid.sum() == 4


def test_empty_cells_are_dropped_and_keys_read_in_sheet_order() -> None:
    sprites = sprites_from_tileset(
        _sheet(2, 3, empty={(0, 1), (1, 0)}), tile=(4, 4), prefix="p", name="set"
    )
    assert len(sprites) == 4
    keys = [sprite.key for sprite in sprites]
    assert keys == sorted(keys), "lexical order must equal reading order"
    assert keys[0] == tile_key("p", 0, 0) == "p#tile000x000"
    assert sprites[0].name == "set r1c1"


def test_a_sliced_sprite_carries_its_cell_exactly() -> None:
    sheet = _sheet(1, 2)
    sprites = sprites_from_tileset(sheet, tile=(4, 4), prefix="p")
    assert np.array_equal(sprites[1].pixels, sheet[0:4, 4:8])


def test_an_all_opaque_sheet_keeps_every_cell() -> None:
    sheet = np.full((8, 8, 4), 255, dtype=np.uint8)  # the usual RGB decode
    assert len(sprites_from_tileset(sheet, tile=(4, 4), prefix="p")) == 4


def test_a_degenerate_tile_size_is_refused() -> None:
    with pytest.raises(ValueError):
        tileset_occupancy(_sheet(1, 1), tile=(0, 4))
    with pytest.raises(ValueError):
        sprites_from_tileset(_sheet(1, 1), tile=(4, -1), prefix="p")


def test_a_sheet_smaller_than_one_tile_yields_nothing() -> None:
    assert tileset_occupancy(_sheet(1, 1), tile=(64, 64)).size == 0
    assert sprites_from_tileset(_sheet(1, 1), tile=(64, 64), prefix="p") == []


def test_the_packer_ceiling_is_enforced_before_any_slicing() -> None:
    sheet = np.full((70, 70, 4), 255, dtype=np.uint8)  # 4900 one-pixel cells
    with pytest.raises(ValueError, match=str(MAX_SPRITES)):
        sprites_from_tileset(sheet, tile=(1, 1), prefix="p")
