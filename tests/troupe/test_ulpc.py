"""The example sheets as free regression oracles.

Every number here was measured off ``examples/*.png`` first and written down
second. They are validation material only -- no ULPC art ships, and these files
stay out of the package and out of any training set (CC-BY-SA/GPL).

If one of these fails, either the reader's layout table drifted or the example
files were replaced with something that is not a ULPC "full" sheet; both are
worth a loud stop, because everything else in this suite that leans on the
sheets is silently decoding the wrong cells from that point on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from warlock.studio.troupe import ulpc

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
SHEETS = ("male_base_spritesheet.png", "female_base_spritesheet.png")

pytestmark = pytest.mark.skipif(
    not all((EXAMPLES / name).exists() for name in SHEETS),
    reason="the ULPC reference sheets are not checked out",
)


@pytest.fixture(scope="module")
def arrays():
    from PIL import Image

    out = {}
    for name in SHEETS:
        with Image.open(EXAMPLES / name) as im:
            out[name] = np.asarray(im.convert("RGBA"))
    return out


def _occupancy(arr) -> list[int]:
    rows, cols = arr.shape[0] // ulpc.CELL, arr.shape[1] // ulpc.CELL
    return [
        sum(
            1
            for c in range(cols)
            if arr[
                r * ulpc.CELL : (r + 1) * ulpc.CELL,
                c * ulpc.CELL : (c + 1) * ulpc.CELL,
                3,
            ].any()
        )
        for r in range(rows)
    ]


def test_the_layout_table_totals_three_hundred_and_fifty_two_cells():
    assert sum(ulpc.row_frame_counts()) == 352
    assert len(ulpc.cells()) == 352


def test_the_table_is_the_published_full_layout():
    """``(7,8,9,6,13) x 4`` + ``6 + 6`` + ``(2,5,3,3,8,2,13,6) x 4``."""
    assert ulpc.row_frame_counts() == (
        7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 6, 6, 6, 6, 13, 13, 13, 13,
        6, 6,
        2, 2, 2, 2, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 8, 8, 8, 8,
        2, 2, 2, 2, 13, 13, 13, 13, 6, 6, 6, 6,
    )


@pytest.mark.parametrize("name", SHEETS)
def test_both_sheets_are_the_measured_size_and_grid(arrays, name):
    arr = arrays[name]
    assert (arr.shape[1], arr.shape[0]) == ulpc.SHEET_SIZE
    assert (arr.shape[1] // ulpc.CELL, arr.shape[0] // ulpc.CELL) == (13, 54)


@pytest.mark.parametrize("name", SHEETS)
def test_the_sheet_occupancy_matches_the_layout_table_row_for_row(arrays, name):
    assert tuple(_occupancy(arrays[name])) == ulpc.row_frame_counts()


@pytest.mark.parametrize("name", SHEETS)
def test_there_is_no_antialiasing_anywhere(arrays, name):
    """Every pixel is alpha 0 or 255, which is what makes the sheets losslessly
    indexable -- and is the bar Troupe's own pixeliser has to clear."""
    assert set(np.unique(arrays[name][:, :, 3]).tolist()) == {0, 255}


@pytest.mark.parametrize("name", SHEETS)
def test_sixteen_opaque_colours(arrays, name):
    arr = arrays[name]
    opaque = arr[arr[:, :, 3] == 255][:, :3]
    assert len({tuple(p) for p in opaque.tolist()}) == 16


def _walk_mirror_diffs(arr) -> list[tuple[int, int, int, int, int]]:
    """Per walk frame: ``(count, y_min, y_max, x_min, x_max)`` of the pixels a
    west/east mirror leaves behind."""
    out = []
    for c in range(9):
        west = arr[9 * ulpc.CELL : 10 * ulpc.CELL, c * ulpc.CELL : (c + 1) * ulpc.CELL]
        east = arr[11 * ulpc.CELL : 12 * ulpc.CELL, c * ulpc.CELL : (c + 1) * ulpc.CELL]
        diff = (west != east[:, ::-1]).any(axis=-1)
        ys, xs = np.nonzero(diff)
        out.append(
            (int(diff.sum()), int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
        )
    return out


@pytest.mark.parametrize("name", SHEETS)
def test_west_and_east_walk_frames_are_near_exact_mirrors(arrays, name):
    """The property Phase 6's mirror-assisted cleanup rests on. Row 9 is west
    and row 11 is east; mirroring one onto the other leaves only the face --
    a few dozen pixels out of a 64x64 cell, in one horizontal band."""
    for count, y0, _y1, x0, x1 in _walk_mirror_diffs(arrays[name]):
        assert count <= 50, count
        assert y0 >= 17           # never the shoulders or below
        assert x0 >= 24 and x1 <= 41   # the face, and only the face


def test_the_male_sheet_mirrors_to_the_exact_measured_count(arrays):
    """Written down to the pixel because it is the tightest form of the claim
    and the one the plan quotes; the female sheet's face carries a little more
    asymmetry (37-46) and is covered by the looser assertion above."""
    counts = [d[0] for d in _walk_mirror_diffs(arrays[SHEETS[0]])]
    assert set(counts) == {36, 37}


@pytest.mark.parametrize("name", SHEETS)
def test_no_other_shift_explains_the_difference(arrays, name):
    """The mirror is real facial asymmetry, not a centring offset: every
    non-zero horizontal shift is an order of magnitude worse."""
    arr = arrays[name]
    west = arr[9 * ulpc.CELL : 10 * ulpc.CELL, : ulpc.CELL]
    east = arr[11 * ulpc.CELL : 12 * ulpc.CELL, : ulpc.CELL][:, ::-1]
    at_zero = int((west != east).any(axis=-1).sum())
    for shift in (-1, 1):
        rolled = np.roll(east, shift, axis=1)
        assert int((west != rolled).any(axis=-1).sum()) > at_zero * 4


def test_reading_a_sheet_yields_every_named_cell(arrays):
    from PIL import Image

    with Image.open(EXAMPLES / SHEETS[0]) as im:
        decoded = ulpc.read(im)
    assert len(decoded) == 352
    assert ("walk", "west", 0) in decoded
    assert ("hurt", "hurt", 5) in decoded
    assert all(cell.size == (ulpc.CELL, ulpc.CELL) for cell in decoded.values())


def test_a_sheet_that_is_not_the_full_layout_is_refused():
    from PIL import Image

    with (
        Image.new("RGBA", (64, 64)) as im,
        pytest.raises(ValueError, match="full sheet is 832x3456"),
    ):
        ulpc.read(im)


def test_male_and_female_differ_in_almost_every_cell(arrays):
    male, female = arrays[SHEETS[0]], arrays[SHEETS[1]]
    differing = sum(
        1
        for c in ulpc.cells()
        if (
            male[c.box[1] : c.box[3], c.box[0] : c.box[2]]
            != female[c.box[1] : c.box[3], c.box[0] : c.box[2]]
        ).any()
    )
    assert differing >= 300
