"""The wrap helper's own properties, before any tool is wired to it.

Everything above this module -- the brush, the fill, the shape tool, the canvas
-- gets its idea of "the same tile" from these four functions, so the
assertions here are about *coverage of the torus* rather than about any
particular tool: every canvas pixel is written exactly as many times as the
source covers it, an unwrapped axis is the plain clip it always was, and a
point folds to where the pieces say it does.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import tiling

# --- spans ------------------------------------------------------------------


def test_an_unwrapped_span_is_the_plain_clip():
    """Which is the whole of what makes "tiled off" byte-identical rather than
    merely close: the same one slice the brush computed before this existed."""
    assert tiling.spans(-3, 5, 16, False) == [(0, 5, 3)]
    assert tiling.spans(12, 20, 16, False) == [(12, 16, 0)]
    assert tiling.spans(0, 16, 16, False) == [(0, 16, 0)]


def test_a_span_entirely_off_canvas_is_nothing_at_all():
    assert tiling.spans(-9, -1, 8, False) == []
    assert tiling.spans(20, 30, 8, False) == []
    # And with wrap it is *not* nothing: that is the point.
    assert tiling.spans(-9, -1, 8, True) != []


def test_a_wrapped_span_covers_the_source_exactly_once():
    """Every index of the source appears in exactly one piece, and every piece
    is in bounds. A dropped index is a hole in a stroke at the seam; a repeated
    one would be harmless under maximum-coverage and is still a bug."""
    size = 16
    for lo in range(-20, 21):
        hi = lo + 9
        seen: list[int] = []
        for d0, d1, s0 in tiling.spans(lo, hi, size, True):
            assert 0 <= d0 < d1 <= size
            seen.extend(range(s0, s0 + (d1 - d0)))
        assert sorted(seen) == list(range(hi - lo))


def test_a_source_wider_than_the_canvas_covers_every_column():
    """Overlap is expected here, not avoided: coverage is max-accumulated, so a
    brush wider than the tile paints the whole tile rather than raising."""
    covered = np.zeros(8, dtype=int)
    for d0, d1, _s0 in tiling.spans(-4, 20, 8, True):
        covered[d0:d1] += 1
    assert covered.min() >= 1


def test_a_wrapped_piece_reads_the_index_the_shift_implies():
    """dest = src + lo + k*size, spelled as a slice. The one place an off-by-one
    would put every wrapped stamp a pixel out and still look plausible."""
    source = np.arange(8)
    canvas = np.full(8, -1)
    for d0, d1, s0 in tiling.spans(6, 14, 8, True):
        canvas[d0:d1] = source[s0 : s0 + (d1 - d0)]
    # The source starts at 6, so index 0 lands on column 6 and index 2 wraps
    # round to column 0.
    assert canvas.tolist() == [2, 3, 4, 5, 6, 7, 0, 1]


# --- pieces -----------------------------------------------------------------


def test_pieces_off_is_one_clipped_rect_and_both_is_four_at_a_corner():
    size = (16, 16)
    assert tiling.pieces((-2, -2, 6, 6), size, (False, False)) == [((0, 0, 6, 6), (2, 2))]
    corner = tiling.pieces((-2, -2, 6, 6), size, (True, True))
    assert len(corner) == 4
    assert {rect for rect, _off in corner} == {
        (0, 0, 6, 6), (14, 0, 16, 6), (0, 14, 6, 16), (14, 14, 16, 16),
    }


def test_one_axis_wraps_and_the_other_clips():
    """The per-axis negative control. X-only tiling at a corner gives two
    pieces, not four -- the top overhang is thrown away exactly as it was."""
    two = tiling.pieces((-2, -2, 6, 6), (16, 16), (True, False))
    assert {rect for rect, _off in two} == {(0, 0, 6, 6), (14, 0, 16, 6)}


def test_pieces_tile_the_source_rectangle_exactly_once():
    rng = np.random.default_rng(3)
    size = (13, 11)
    for _ in range(40):
        x0 = int(rng.integers(-20, 20))
        y0 = int(rng.integers(-20, 20))
        w = int(rng.integers(1, 9))
        h = int(rng.integers(1, 9))
        seen = np.zeros((h, w), dtype=int)
        for (dx0, dy0, dx1, dy1), (sx, sy) in tiling.pieces(
            (x0, y0, x0 + w, y0 + h), size, (True, True)
        ):
            seen[sy : sy + (dy1 - dy0), sx : sx + (dx1 - dx0)] += 1
        assert seen.min() == seen.max() == 1


# --- canonical --------------------------------------------------------------


@pytest.mark.parametrize(
    ("point", "axes", "expected"),
    [
        ((3.5, 3.5), (True, True), (3.5, 3.5)),
        ((19.5, 3.5), (True, True), (3.5, 3.5)),
        ((-0.5, 3.5), (True, True), (15.5, 3.5)),
        ((19.5, 19.5), (True, False), (3.5, 19.5)),
        ((19.5, 19.5), (False, True), (19.5, 3.5)),
        ((19.5, 19.5), (False, False), (19.5, 19.5)),
    ],
)
def test_canonical_folds_only_the_wrapped_axes(point, axes, expected):
    assert tiling.canonical(point, (16, 16), axes) == pytest.approx(expected)


def test_the_tile_offset_is_whole_tiles_so_sub_pixel_position_survives():
    """A drag subtracts one offset for its whole life, and a brush walks
    fractional positions -- an offset that rounded would quantise every tiled
    stroke to the pixel grid."""
    offset = tiling.tile_offset((33.25, -0.75), (16, 16), (True, True))
    assert offset == (32, -16)
    assert tiling.canonical((33.25, -0.75), (16, 16)) == (1.25, 15.25)


def test_axes_of_names_every_mode_and_refuses_the_rest():
    assert [tiling.axes_of(name) for name in tiling.TILED_AXES] == [
        (False, False), (True, False), (False, True), (True, True)
    ]
    assert tiling.axes_of((True, False)) == (True, False)
    with pytest.raises(ValueError):
        tiling.axes_of("xy")  # the roadmap's spelling; ``both`` is the enum's


# --- fold_coverage ----------------------------------------------------------


def test_fold_coverage_wraps_an_overhang_onto_the_far_edge():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:] = 255
    folded = tiling.fold_coverage(mask, (14, 2), (16, 16), (True, True))
    assert folded is not None
    rect, crop = folded
    # Two columns at the right edge and two wrapped round to the left, so the
    # bbox is the full width.
    assert rect == (0, 2, 16, 6)
    assert crop[:, :2].min() == 255 and crop[:, -2:].min() == 255
    assert crop[:, 2:-2].max() == 0


def test_fold_coverage_takes_the_maximum_where_pieces_overlap():
    """A shape wider than the tile lands on itself. Summing would draw the
    overlap twice as dark; ``np.maximum`` is the rule every other coverage
    buffer in the editor follows."""
    mask = np.full((4, 20), 128, dtype=np.uint8)
    rect, crop = tiling.fold_coverage(mask, (0, 0), (8, 8), (True, True))
    assert rect == (0, 0, 8, 4)
    assert int(crop.max()) == 128


def test_fold_coverage_with_no_wrap_is_the_bbox_of_the_plain_crop():
    """The off path, which the shape tool relies on being unchanged."""
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 1:4] = 255
    rect, crop = tiling.fold_coverage(mask, (0, 0), (8, 8), (False, False))
    assert rect == (1, 2, 4, 5)
    assert np.array_equal(crop, mask[2:5, 1:4])


def test_fold_coverage_of_nothing_is_none():
    assert tiling.fold_coverage(np.zeros((4, 4), np.uint8), (0, 0), (8, 8)) is None


# --- seam seeds -------------------------------------------------------------


def test_seam_seeds_only_appear_on_wrapped_axes():
    reached = np.zeros((4, 4), dtype=bool)
    candidate = np.ones((4, 4), dtype=bool)
    reached[:, 0] = True
    assert tiling.seam_seeds(reached, candidate, (True, False)) == [(3, 0), (3, 1), (3, 2), (3, 3)]
    assert tiling.seam_seeds(reached, candidate, (False, True)) == []


def test_a_seam_seed_is_never_a_pixel_the_flood_already_has():
    """Otherwise the fixpoint below it never terminates."""
    reached = np.ones((4, 4), dtype=bool)
    assert tiling.seam_seeds(reached, np.ones((4, 4), dtype=bool), (True, True)) == []
