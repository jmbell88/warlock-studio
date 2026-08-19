"""The separator-band detector: what is a ruled grid and what is only a picture.

Every sheet here is synthesized by :func:`_sheet`, because the property under
test is a relationship between segment sizes and rule darkness, and a real
1024px PNG pins neither -- it would pass while saying nothing about which of the
rules below is load-bearing.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.tilegrid import slicing


def _sheet(
    row_sizes: list[int],
    col_sizes: list[int],
    sep: int = 2,
    line: int = 5,
    fill: int = 200,
    border: bool = False,
) -> np.ndarray:
    """A sheet with the given cell sizes, ruled off by ``sep``-wide dark lines.

    ``line`` is the rule's channel value, ``fill`` the cells'. With ``border``
    the outside is ruled too, which is the case that would mint phantom cells if
    the segment walk did not drop empty ones.
    """
    height = sum(row_sizes) + sep * (len(row_sizes) - 1) + (2 * sep if border else 0)
    width = sum(col_sizes) + sep * (len(col_sizes) - 1) + (2 * sep if border else 0)
    out = np.full((height, width, 4), line, dtype=np.uint8)
    out[:, :, 3] = 255
    y = sep if border else 0
    for r, rh in enumerate(row_sizes):
        x = sep if border else 0
        for c, cw in enumerate(col_sizes):
            out[y:y + rh, x:x + cw, :3] = fill + r * 5 + c
            x += cw + sep
        y += rh + sep
    return out


def test_find_bands_groups_runs() -> None:
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert slicing.find_bands(mask) == [(1, 2), (5, 5), (7, 9)]


def test_find_bands_of_nothing_is_empty() -> None:
    assert slicing.find_bands(np.zeros(8, dtype=bool)) == []


def test_a_uniform_grid_is_detected() -> None:
    grid = slicing.detect_grid(_sheet([16] * 4, [16] * 3))
    assert grid is not None
    assert grid.shape == (4, 3)
    assert grid.tile_count == 12
    assert grid.rows[0] == (0, 15)
    assert grid.cols[1] == (18, 33)


def test_border_rules_make_no_phantom_cells() -> None:
    grid = slicing.detect_grid(_sheet([16] * 4, [16] * 3, border=True))
    assert grid is not None
    assert grid.shape == (4, 3)


def test_slightly_irregular_cells_still_detect() -> None:
    # A CV well under 0.15 -- the generated-sheet case this exists for.
    grid = slicing.detect_grid(_sheet([15, 16, 17, 16], [16, 15, 17]))
    assert grid is not None
    assert grid.shape == (4, 3)


def test_wildly_irregular_cells_are_refused() -> None:
    assert slicing.detect_grid(_sheet([4, 40, 9, 31], [16, 16, 16])) is None


def test_a_flat_image_is_not_a_grid() -> None:
    flat = np.full((64, 64, 4), 200, dtype=np.uint8)
    flat[:, :, 3] = 255
    assert slicing.detect_grid(flat) is None


def test_an_all_dark_image_is_not_a_grid() -> None:
    dark = np.zeros((64, 64, 4), dtype=np.uint8)
    dark[:, :, 3] = 255
    assert slicing.detect_grid(dark) is None


def test_one_axis_only_is_not_a_grid() -> None:
    # Horizontal rules, no vertical ones: strips, not tiles.
    assert slicing.detect_grid(_sheet([16, 16, 16], [64])) is None


def test_darkness_is_the_brightest_channel() -> None:
    # A saturated red rule is not dark: its red channel is 255.
    sheet = _sheet([16] * 3, [16] * 3, line=0)
    rules = sheet[:, :, :3].max(axis=2) == 0
    sheet[rules, 0] = 255
    assert slicing.detect_grid(sheet) is None


def test_the_dark_ratio_edge_is_inclusive() -> None:
    sheet = _sheet([16] * 2, [16] * 2, sep=1, line=0)
    # Punch bright pixels into one rule row until it is exactly at the ratio.
    row = 16
    width = sheet.shape[1]
    bright = width - int(np.ceil(width * slicing.MIN_DARK_RATIO))
    sheet[row, :bright, :3] = 255
    assert slicing.detect_grid(sheet, threshold=20) is not None
    sheet[row, :bright + 1, :3] = 255
    assert slicing.detect_grid(sheet, threshold=20) is None


def test_auto_threshold_prefers_most_tiles() -> None:
    # Rules at 45: only the two highest thresholds see them at all.
    sheet = _sheet([16] * 3, [16] * 3, line=45)
    grid = slicing.detect_grid(sheet)
    assert grid is not None
    assert grid.shape == (3, 3)
    assert grid.threshold == 50


def test_auto_threshold_breaks_ties_by_the_lowest() -> None:
    sheet = _sheet([16] * 3, [16] * 3, line=0)
    grid = slicing.detect_grid(sheet)
    assert grid is not None
    assert grid.threshold == slicing.DARK_THRESHOLDS[0]


def test_a_fixed_threshold_is_used_verbatim() -> None:
    sheet = _sheet([16] * 3, [16] * 3, line=45)
    assert slicing.detect_grid(sheet, threshold=20) is None
    grid = slicing.detect_grid(sheet, threshold=50)
    assert grid is not None
    assert grid.threshold == 50


def test_a_non_image_array_is_refused() -> None:
    with pytest.raises(ValueError):
        slicing.detect_grid(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError):
        slicing.detect_grid(np.zeros((8, 8, 2), dtype=np.uint8))


def test_recompose_shape_and_cell_content() -> None:
    sheet = _sheet([15, 16, 17], [16, 15])
    grid = slicing.detect_grid(sheet)
    assert grid is not None
    out = slicing.recompose(sheet, grid, 8, 8)
    assert out.shape == (24, 16, 4)
    assert out.dtype == np.uint8
    # Cell (r, c) was filled with 200 + 5r + c; every pixel of the redrawn cell
    # carries that same colour, so the crop landed on the content.
    for r in range(3):
        for c in range(2):
            block = out[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8, 0]
            assert np.all(block == 200 + 5 * r + c)


def test_recompose_is_nearest_neighbour() -> None:
    sheet = _sheet([16] * 2, [16] * 2)
    # Give one cell a gradient so an averaging resize would mint new values.
    sheet[0:16, 0:16, :3] = np.arange(16, dtype=np.uint8)[None, :, None] * 15
    grid = slicing.detect_grid(sheet)
    assert grid is not None
    out = slicing.recompose(sheet, grid, 7, 7)
    assert set(np.unique(out[:, :, :3]).tolist()) <= set(np.unique(sheet[:, :, :3]).tolist())


def test_recompose_strips_every_separator_pixel() -> None:
    sheet = _sheet([16] * 3, [16] * 3, sep=3, line=0, border=True)
    grid = slicing.detect_grid(sheet)
    assert grid is not None
    out = slicing.recompose(sheet, grid, 16, 16)
    # No rule pixel survives: the darkest cell fill is 200.
    assert out[:, :, :3].min() >= 200


def test_recompose_never_writes_through_the_source() -> None:
    sheet = _sheet([16] * 2, [16] * 2)
    sheet.flags.writeable = False
    grid = slicing.detect_grid(sheet)
    assert grid is not None
    out = slicing.recompose(sheet, grid, 8, 8)
    out[:] = 7  # would raise if it aliased the read-only source
    assert sheet[0, 0, 0] == 200


def test_recompose_fills_alpha_for_an_rgb_sheet() -> None:
    sheet = _sheet([16] * 2, [16] * 2)[:, :, :3]
    grid = slicing.detect_grid(sheet)
    assert grid is not None
    out = slicing.recompose(sheet, grid, 8, 8)
    assert out.shape[2] == 4
    assert np.all(out[:, :, 3] == 255)


def test_recompose_refuses_a_zero_tile_size() -> None:
    sheet = _sheet([16] * 2, [16] * 2)
    grid = slicing.detect_grid(sheet)
    assert grid is not None
    with pytest.raises(ValueError):
        slicing.recompose(sheet, grid, 0, 8)
