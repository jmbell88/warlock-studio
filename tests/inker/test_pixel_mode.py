"""The pixel nibs, the corner filter and the nearest-neighbour resample.

Three separate things with one purpose between them: that a drawing whose pixels
*are* the artwork survives being drawn, scaled and turned. Each of them fails
silently -- a fringe of near-colours, a doubled corner, a blurred sprite -- so
each is asserted on the pixels rather than on the settings that produced them.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio import inker, inker_state
from warlock.studio.inker import brush, transform

# --- the stamp --------------------------------------------------------------


@pytest.mark.parametrize("nib", sorted(brush.PIXEL_NIBS))
@pytest.mark.parametrize("diameter", [1, 2, 3, 4, 7, 16, 33])
def test_a_pixel_nib_has_no_partial_coverage_at_any_size(nib, diameter):
    """The whole of what "pixel" means here, and the reason it is a separate nib
    rather than hardness 1: the soft stamp is antialiased over its last half
    pixel *by design*, which is right for a painted reference and is exactly the
    fringe of near-colours a pixel drawing must not grow."""
    stamp = brush.make_stamp(diameter, 1.0, nib)
    assert set(np.unique(stamp)) <= {0.0, 1.0}


def test_the_soft_nib_is_untouched_and_still_antialiases_its_rim():
    """The regression the two new nibs could most easily cause. Every stroke
    this editor has ever drawn is a soft one."""
    soft = brush.make_stamp(16, 1.0, "soft")
    assert np.any((soft > 0.0) & (soft < 1.0))
    assert np.array_equal(soft, brush.make_stamp(16, 1.0))


def test_a_one_pixel_pencil_stamps_exactly_one_pixel():
    """``<=`` rather than ``<`` at the radius. The strict test leaves the single
    sample of a diameter-1 stamp outside the disc, so the pencil draws nothing
    -- which is the whole nib."""
    assert brush.make_stamp(1, 1.0, "pixel").tolist() == [[1.0]]


def test_a_square_nib_is_solid_and_a_round_one_is_not():
    assert brush.make_stamp(5, 1.0, "square").sum() == 25.0
    assert brush.make_stamp(5, 1.0, "pixel").sum() < 25.0


def test_hardness_cannot_change_a_pixel_nib():
    """Which is why ``_stamp`` passes 1.0 for them: a slider that silently keyed
    the shared stamp cache on a value with no effect is a cache that holds four
    copies of one answer."""
    assert np.array_equal(
        brush.make_stamp(9, 0.1, "pixel"), brush.make_stamp(9, 1.0, "pixel")
    )


# --- the walk ---------------------------------------------------------------


def test_the_pixel_walk_leaves_no_gap_however_fast_the_mouse_moved():
    """The failure the spacing walk has at one pixel: dabs a fraction of a
    *diameter* apart along a float segment, which for a one-pixel nib means a
    gap wherever the cursor moved more than a pixel between frames."""
    pixels = _drawn((40, 40), [(2, 2), (30, 9)], nib="pixel", size=1)
    ys, xs = np.nonzero(pixels[..., 3])
    drawn = {(int(x), int(y)) for x, y in zip(xs, ys, strict=True)}
    # Every step of a Bresenham line is adjacent to the last, so a walk with no
    # gap in it is one whose successive pixels never jump.
    ordered = brush.line_pixels((2, 2), (30, 9))
    assert set(ordered) == drawn


def test_line_pixels_includes_both_ends_and_steps_one_at_a_time():
    walk = brush.line_pixels((0, 0), (4, 2))
    assert walk[0] == (0, 0) and walk[-1] == (4, 2)
    assert all(
        abs(b[0] - a[0]) <= 1 and abs(b[1] - a[1]) <= 1
        for a, b in zip(walk, walk[1:], strict=False)
    )


# --- the corner filter ------------------------------------------------------


def test_is_corner_names_the_elbow_of_a_staircase_step():
    assert brush.is_corner((0, 0), (1, 0), (1, 1))
    assert brush.is_corner((0, 0), (0, 1), (1, 1))


def test_is_corner_leaves_a_straight_run_and_a_true_diagonal_alone():
    assert not brush.is_corner((0, 0), (1, 0), (2, 0))
    assert not brush.is_corner((0, 0), (1, 1), (2, 2))
    # Two apart is not an elbow: dropping the middle would break the line.
    assert not brush.is_corner((0, 0), (1, 0), (2, 1))


def test_pixel_perfect_thins_a_freehand_diagonal_to_one_pixel_a_step():
    """A 45-degree drag drawn by hand arrives as a staircase with a doubled
    pixel at every step. The filter is the difference between a clean diagonal
    and a line that is two pixels thick wherever the hand wobbled."""
    path = [(1, 1), (4, 4)]
    plain = _count(_drawn((16, 16), path, nib="pixel", size=1))
    thin = _count(_drawn((16, 16), path, nib="pixel", size=1, pixel_perfect=True))
    assert plain == 4  # (1,1) (2,2) (3,3) (4,4) -- already clean, nothing to drop
    assert thin == plain

    # A staircase, which is what a hand actually produces.
    stair = [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3)]
    assert _count(_drawn((16, 16), stair, nib="pixel", size=1)) == 5
    assert _count(_drawn((16, 16), stair, nib="pixel", size=1, pixel_perfect=True)) == 3


def test_a_click_still_marks_under_the_corner_filter():
    """The filter holds a pixel back to see whether the next makes it an elbow,
    and at release there is no next. Without the flush a click draws nothing and
    every stroke is a pixel short."""
    doc = inker.Document.blank(8, 8)
    doc.begin_stroke(
        (3.0, 3.0), (255, 0, 0, 255), size=1, nib="pixel", pixel_perfect=True
    )
    assert _count(doc.stack.active.pixels) == 0  # held, not yet drawn
    doc.end_stroke()
    assert _count(doc.stack.active.pixels) == 1


def test_a_pixel_stroke_lands_on_whole_pixels_at_full_alpha():
    """The end-to-end version of the stamp assertion: no partial alpha anywhere
    in a finished stroke, which is what a palette, an outline pass or a colour
    key downstream all depend on."""
    pixels = _drawn((32, 32), [(4.3, 4.7), (20.1, 12.9)], nib="pixel", size=3)
    alpha = pixels[..., 3]
    assert set(np.unique(alpha)) <= {0, 255}


def test_an_odd_pixel_nib_is_centred_on_the_pixel_it_is_on():
    """The rounding the soft path uses is half a pixel out for an even
    diameter, which a soft rim hides and a hard one does not."""
    pixels = _drawn((11, 11), [(5, 5)], nib="square", size=3)
    on = np.argwhere(pixels[..., 3] > 0)
    assert on[:, 0].min() == 4 and on[:, 0].max() == 6
    assert on[:, 1].min() == 4 and on[:, 1].max() == 6


def _drawn(canvas, path, **kw):
    doc = inker.Document.blank(*canvas)
    doc.begin_stroke((float(path[0][0]), float(path[0][1])), (255, 0, 0, 255), **kw)
    for point in path[1:]:
        doc.stroke_to((float(point[0]), float(point[1])))
    doc.end_stroke()
    return doc.stack.active.pixels


def _count(pixels):
    return int(np.count_nonzero(pixels[..., 3]))


# --- resampling -------------------------------------------------------------


def test_nearest_scaling_introduces_no_colour_that_was_not_there():
    """A filtered scale of a sprite comes back with thousands of new colours in
    it, which is the whole reason this option exists."""
    source = np.zeros((4, 4, 4), dtype=np.uint8)
    source[..., 3] = 255
    source[:2, :2, 0] = 255
    source[2:, 2:, 1] = 255

    before = {tuple(c) for c in source.reshape(-1, 4)}
    nearest = transform.scale(source, (16, 16), resample="nearest")
    assert {tuple(c) for c in nearest.reshape(-1, 4)} == before
    assert len({tuple(c) for c in transform.scale(source, (16, 16)).reshape(-1, 4)}) > len(
        before
    )


def test_nearest_scaling_by_a_whole_factor_is_an_exact_block_copy():
    source = np.arange(2 * 2 * 4, dtype=np.uint8).reshape(2, 2, 4)
    doubled = transform.scale(source, (4, 4), resample="nearest")
    assert np.array_equal(doubled, np.repeat(np.repeat(source, 2, axis=0), 2, axis=1))


def test_nearest_does_not_round_trip_through_premultiplied_alpha():
    """The reason ``_resample`` grew a ``straight`` path. Nearest mixes nothing,
    so premultiplying and dividing back out would be pure loss -- it moves the
    colour of every partially transparent pixel, which is the one thing a copy
    must not do."""
    source = np.array([[[200, 100, 50, 3]]], dtype=np.uint8)
    assert np.array_equal(transform.scale(source, (2, 2), resample="nearest")[0, 0], source[0, 0])


def test_a_quarter_turn_by_nearest_is_lossless():
    rng = np.random.default_rng(7)
    source = rng.integers(0, 256, (8, 8, 4), dtype=np.uint8)
    turned = transform.rotate(source, 90.0, resample="nearest")
    assert sorted(map(tuple, turned.reshape(-1, 4))) == sorted(
        map(tuple, source.reshape(-1, 4))
    )


def test_the_smooth_path_is_untouched():
    """Every existing caller passes no ``resample`` and must get exactly what it
    got before."""
    rng = np.random.default_rng(11)
    source = rng.integers(0, 256, (8, 8, 4), dtype=np.uint8)
    assert np.array_equal(
        transform.scale(source, (5, 5)), transform.scale(source, (5, 5), resample="smooth")
    )


# --- shape constraints ------------------------------------------------------


def test_shift_makes_a_rectangle_square_in_whichever_quadrant_it_is_drawn():
    for point, expected in (
        ((10, 4), (10, 10)),
        ((4, 10), (10, 10)),
        ((-10, -4), (-10, -10)),
        ((-4, 10), (-10, 10)),
    ):
        _, p1 = inker_state.shape_endpoints("rect", (0, 0), point, constrain=True)
        assert p1 == expected


def test_shift_snaps_a_line_to_an_eighth_of_a_turn_keeping_its_length():
    _, p1 = inker_state.shape_endpoints("line", (0, 0), (10, 1), constrain=True)
    assert p1 == pytest.approx((10.0498, 0.0), abs=1e-3)
    _, p1 = inker_state.shape_endpoints("line", (0, 0), (10, 9), constrain=True)
    # 45 degrees, and equal sides -- which is what makes it a clean diagonal.
    assert p1[0] == pytest.approx(p1[1])


def test_alt_expands_a_shape_about_the_press_rather_than_from_it():
    p0, p1 = inker_state.shape_endpoints("rect", (10, 10), (14, 12), from_centre=True)
    assert (p0, p1) == ((6.0, 8.0), (14, 12))


def test_the_two_modifiers_compose_with_the_press_still_at_the_centre():
    """Constrain first, then mirror. The other order squares a box that has
    already been mirrored, which moves the centre off the pressed point -- the
    one thing "from centre" promises."""
    p0, p1 = inker_state.shape_endpoints(
        "rect", (10, 10), (16, 12), constrain=True, from_centre=True
    )
    assert p1 == (16, 16)
    assert p0 == (4.0, 4.0)
    assert ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2) == (10.0, 10.0)


def test_no_modifiers_is_the_drag_exactly_as_it_was():
    assert inker_state.shape_endpoints("rect", (1, 2), (3, 4)) == ((1, 2), (3, 4))


def test_a_zero_length_line_is_left_alone_rather_than_given_an_angle():
    assert inker_state.constrain_line((5, 5), (5, 5)) == (5, 5)
