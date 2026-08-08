"""The stamp, the spacing walk, and the modes that read instead of write."""

from __future__ import annotations

import math

import numpy as np
import pytest

from warlock.studio.inker import brush

RED = (255, 0, 0, 255)


def _layer(size=(32, 32), colour=(255, 255, 255, 255)):
    pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    pixels[:, :] = colour
    return pixels


def _stroke(pixels, **kw):
    kw.setdefault("colour", RED)
    return brush.StrokeState(
        layer_uid=1,
        size=(pixels.shape[1], pixels.shape[0]),
        before=pixels.copy(),
        **kw,
    )


# --- the stamp --------------------------------------------------------------


def test_a_hard_stamp_is_solid_in_the_middle_and_soft_only_at_the_rim():
    """Even at hardness 1: a stamp that is exactly 0/1 gives every diagonal a
    staircase, which is the difference between a brush and a bitmap."""
    stamp = brush.make_stamp(16, 1.0)
    assert stamp[8, 8] == pytest.approx(1.0)
    assert stamp[0, 0] == pytest.approx(0.0)
    assert np.any((stamp > 0.05) & (stamp < 0.95))


def test_a_soft_stamp_falls_off_from_the_centre():
    stamp = brush.make_stamp(32, 0.0)
    assert stamp[16, 16] > stamp[16, 24] > stamp[16, 30]


def test_a_stamp_is_round_not_square():
    stamp = brush.make_stamp(16, 1.0)
    assert stamp[8, 0] > stamp[0, 0]


def test_stamps_are_cached_so_a_stroke_does_not_rebuild_one_per_dab():
    a = brush.make_stamp(12, 0.5)
    assert brush.make_stamp(12, 0.5) is a


def test_a_one_pixel_brush_still_produces_a_stamp():
    assert brush.make_stamp(1, 1.0).shape == (1, 1)


# --- the walk ---------------------------------------------------------------


def test_spacing_is_carried_between_segments_so_density_is_speed_independent():
    """Sampling per segment instead of per distance makes a slow drag darker
    than a fast one -- the most visible bug a naive painter has."""
    fast = _layer()
    slow = _layer()
    one = _stroke(fast, diameter=6, hardness=1.0)
    one.begin((2, 16), fast)
    one.to((30, 16), fast)
    many = _stroke(slow, diameter=6, hardness=1.0)
    many.begin((2, 16), slow)
    for x in range(3, 31):
        many.to((x, 16), slow)
    assert np.array_equal(fast, slow)


def test_a_press_with_no_drag_lays_down_exactly_one_dab():
    pixels = _layer()
    stroke = _stroke(pixels, diameter=8, hardness=1.0)
    stroke.begin((16, 16), pixels)
    assert tuple(pixels[16, 16]) == RED
    assert tuple(pixels[0, 0]) == (255, 255, 255, 255)


def test_a_zero_length_move_does_not_stamp_again():
    pixels = _layer()
    stroke = _stroke(pixels, diameter=4, hardness=1.0)
    stroke.begin((8, 8), pixels)
    rect = stroke.dirty
    stroke.to((8, 8), pixels)
    assert stroke.dirty == rect


def test_the_dirty_rectangle_grows_to_cover_the_whole_stroke():
    pixels = _layer((64, 64))
    stroke = _stroke(pixels, diameter=4)
    stroke.begin((10, 10), pixels)
    stroke.to((40, 40), pixels)
    x0, y0, x1, y1 = stroke.dirty
    assert x0 <= 8 and y0 <= 8 and x1 >= 42 and y1 >= 42


def test_a_stroke_entirely_off_canvas_marks_nothing():
    pixels = _layer((8, 8))
    stroke = _stroke(pixels, diameter=2)
    stroke.begin((-50, -50), pixels)
    assert stroke.dirty is None


# --- modes ------------------------------------------------------------------


def test_opacity_is_a_property_of_the_stroke_not_of_each_dab():
    pixels = _layer((32, 32), (0, 0, 0, 255))
    stroke = _stroke(pixels, diameter=10, hardness=1.0, opacity=0.5)
    stroke.begin((4, 16), pixels)
    stroke.to((28, 16), pixels)
    stroke.to((4, 16), pixels)
    assert 120 <= int(pixels[16, 16][0]) <= 136


def test_erasing_lowers_alpha_and_leaves_the_colour_alone():
    pixels = _layer((16, 16), (10, 20, 30, 255))
    stroke = _stroke(pixels, diameter=6, hardness=1.0, mode="erase")
    stroke.begin((8, 8), pixels)
    assert int(pixels[8, 8][3]) == 0
    assert tuple(pixels[8, 8][:3]) == (10, 20, 30)


def test_blur_softens_an_edge_instead_of_painting_over_it():
    pixels = _layer((32, 32), (0, 0, 0, 255))
    pixels[:, 16:] = (255, 255, 255, 255)
    stroke = _stroke(pixels, diameter=16, hardness=1.0, mode="blur", strength=1.0)
    stroke.begin((16, 16), pixels)
    edge = [int(pixels[16, x][0]) for x in range(12, 20)]
    assert edge == sorted(edge)
    assert any(20 < v < 235 for v in edge)


def test_smudge_drags_colour_along_the_stroke():
    pixels = _layer((32, 32), (255, 255, 255, 255))
    pixels[:, :8] = (0, 0, 0, 255)
    stroke = _stroke(pixels, diameter=10, hardness=1.0, mode="smudge", strength=0.9)
    stroke.begin((4, 16), pixels)
    stroke.to((20, 16), pixels)
    assert int(pixels[16, 12][0]) < 255


def test_blur_and_smudge_read_the_live_layer_so_a_second_pass_does_more():
    """They are accumulation tools; the coverage-recompute path the paint
    modes use would make the second pass identical to the first."""
    pixels = _layer((32, 32), (0, 0, 0, 255))
    pixels[:, 16:] = (255, 255, 255, 255)
    stroke = _stroke(pixels, diameter=16, hardness=1.0, mode="blur", strength=1.0)
    stroke.begin((16, 16), pixels)
    once = int(pixels[16, 14][0])
    for _ in range(4):
        stroke.begin((16, 16), pixels)
    assert int(pixels[16, 14][0]) > once


# --- symmetry ---------------------------------------------------------------


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("none", [(6, 6)]),
        ("x", [(6, 6), (25, 6)]),
        ("y", [(6, 6), (6, 25)]),
        ("xy", [(6, 6), (25, 6), (6, 25), (25, 25)]),
    ],
)
def test_symmetry_puts_a_dab_at_every_reflection(mode, expected):
    pixels = _layer((32, 32))
    stroke = _stroke(pixels, diameter=4, hardness=1.0, symmetry=mode)
    stroke.begin((6, 6), pixels)
    for x, y in expected:
        assert tuple(pixels[y, x]) == RED, (mode, x, y)
    marked = int(((pixels == np.array(RED, dtype=np.uint8)).all(axis=2)).sum())
    assert marked >= 4 * len(expected)


def test_symmetry_applies_to_erasing_too_because_it_mirrors_positions():
    pixels = _layer((32, 32), (0, 0, 0, 255))
    stroke = _stroke(pixels, diameter=4, hardness=1.0, mode="erase", symmetry="x")
    stroke.begin((6, 6), pixels)
    assert int(pixels[6, 6][3]) == 0
    assert int(pixels[6, 25][3]) == 0


# --- clipping ---------------------------------------------------------------


def test_a_selection_clips_the_brush_with_the_same_multiply_it_clips_a_fill():
    from warlock.studio.inker.selection import SelectionMask

    pixels = _layer((32, 32))
    clip = SelectionMask.from_rect((32, 32), (0, 0, 16, 32))
    stroke = _stroke(pixels, diameter=12, hardness=1.0, clip=clip)
    stroke.begin((16, 16), pixels)
    assert tuple(pixels[16, 14]) == RED
    assert tuple(pixels[16, 18]) == (255, 255, 255, 255)


def test_a_feathered_clip_softens_the_brush_rather_than_cutting_it():
    from warlock.studio.inker.selection import SelectionMask

    pixels = _layer((64, 64), (255, 255, 255, 255))
    clip = SelectionMask.from_rect((64, 64), (0, 0, 32, 64)).feathered(4.0)
    stroke = _stroke(pixels, diameter=48, hardness=1.0, clip=clip)
    stroke.begin((32, 32), pixels)
    ramp = [int(pixels[32, x][1]) for x in range(24, 40)]
    assert ramp == sorted(ramp)
    assert any(20 < v < 235 for v in ramp)


# --- symmetry axis, radial symmetry, stabilisation and taper (Ink10-12) ------


def _canvas(size=64):
    return np.zeros((size, size, 4), dtype=np.uint8)


def test_the_default_axis_reflects_column_zero_onto_the_last_column():
    """``(w - 1) / 2`` rather than ``w / 2``: the only placement where a
    symmetric drawing is symmetric to the pixel, and the one the fixed formula
    this replaced produced."""
    assert brush.default_axis((16, 9)) == (7.5, 4.0)
    assert brush._mirror((0.0, 3.0), (16, 9), "x") == [(0.0, 3.0), (15.0, 3.0)]


def test_a_moved_axis_reflects_about_the_axis_rather_than_the_centre():
    points = brush._mirror((2.0, 5.0), (32, 32), "x", axis=(10.0, 10.0))
    assert points == [(2.0, 5.0), (18.0, 5.0)]


def test_both_axes_still_give_four_points():
    points = brush._mirror((1.0, 2.0), (32, 32), "xy", axis=(10.0, 20.0))
    assert set(points) == {(1.0, 2.0), (19.0, 2.0), (1.0, 38.0), (19.0, 38.0)}


def test_radial_symmetry_gives_one_point_per_way_including_the_original():
    for ways in (2, 3, 6, 12):
        points = brush._mirror((5.0, 0.0), (32, 32), "radial", axis=(0.0, 0.0), radial=ways)
        assert len(points) == ways
        assert points[0] == (5.0, 0.0)


def test_every_radial_point_is_the_same_distance_from_the_axis():
    axis = (16.0, 16.0)
    points = brush._mirror((16.0, 4.0), (32, 32), "radial", axis=axis, radial=8)
    radii = [math.hypot(x - axis[0], y - axis[1]) for x, y in points]
    assert max(radii) - min(radii) < 1e-4


def test_two_way_radial_is_a_half_turn_rather_than_a_mirror():
    """A rotation, not a reflection: the difference shows on anything that is
    not itself symmetric, which is most drawings."""
    points = brush._mirror((4.0, 1.0), (32, 32), "radial", axis=(0.0, 0.0), radial=2)
    assert points[1][0] == pytest.approx(-4.0)
    assert points[1][1] == pytest.approx(-1.0)


def test_the_number_of_ways_is_clamped_rather_than_trusted():
    assert len(brush._mirror((1.0, 1.0), (8, 8), "radial", radial=1)) == brush.MIN_RADIAL
    assert len(brush._mirror((1.0, 1.0), (8, 8), "radial", radial=999)) == brush.MAX_RADIAL


def test_a_stroke_with_the_new_options_off_is_identical_to_one_without_them():
    """The bar for every option added here: the default path is the old path.

    Not "close" -- the same bytes. A stabiliser that lagged by a hair or a
    taper that rounded a diameter differently would change every stroke in the
    app for people who never touched either control.
    """
    walk = [(4.0, 4.0), (12.0, 9.0), (28.0, 11.0), (40.0, 30.0)]

    plain = _canvas()
    old = brush.StrokeState(layer_uid=1, size=(64, 64), before=plain.copy(), colour=RED)
    old.begin(walk[0], plain)
    for point in walk[1:]:
        old.to(point, plain)

    fresh = _canvas()
    new = brush.StrokeState(
        layer_uid=1,
        size=(64, 64),
        before=fresh.copy(),
        colour=RED,
        stabilise=0.0,
        speed_taper=0.0,
        axis=None,
    )
    new.begin(walk[0], fresh)
    for point in walk[1:]:
        new.to(point, fresh)

    assert np.array_equal(plain, fresh)


def test_stabilisation_makes_the_brush_lag_the_cursor():
    target = _canvas()
    stroke = brush.StrokeState(
        layer_uid=1, size=(64, 64), before=target.copy(), colour=RED, stabilise=0.8
    )
    stroke.begin((4.0, 4.0), target)
    stroke.to((44.0, 4.0), target)
    assert stroke._last is not None
    assert 4.0 < stroke._last[0] < 44.0, "it moved, but not all the way"
    assert stroke._target == (44.0, 4.0), "while the cursor is where it was put"


def test_a_stabilised_brush_catches_up_when_the_cursor_stops():
    """An exponential lag, so it arrives -- a fixed-length rolling average
    would leave the stroke permanently short of the last point."""
    target = _canvas()
    stroke = brush.StrokeState(
        layer_uid=1, size=(64, 64), before=target.copy(), colour=RED, stabilise=0.8
    )
    stroke.begin((4.0, 4.0), target)
    for _ in range(60):
        stroke.to((44.0, 4.0), target)
    assert stroke._last is not None
    assert stroke._last[0] == pytest.approx(44.0, abs=0.5)


def test_stabilisation_never_stops_the_brush_entirely():
    stroke = brush.StrokeState(
        layer_uid=1, size=(8, 8), before=_canvas(8), colour=RED, stabilise=5.0
    )
    assert stroke.stabilise == brush.MAX_STABILISE


def test_a_fast_segment_taper_thins_the_dabs():
    """Same flick, taper on and off.

    Compared against the *untapered* stroke rather than against a slow one,
    because both begin with a full-diameter dab at the press -- a press is not
    moving, so it has no speed to be thinned by, and the maximum over the whole
    canvas would be that dab either way.
    """
    def flick(taper: float):
        target = _canvas()
        stroke = brush.StrokeState(
            layer_uid=1,
            size=(64, 64),
            before=target.copy(),
            colour=RED,
            diameter=20,
            hardness=1.0,
            speed_taper=taper,
        )
        stroke.begin((32.0, 10.0), target)
        stroke.to((32.0, 50.0), target)
        return target

    tapered, plain = flick(0.9), flick(0.0)
    assert int(tapered[..., 3].sum()) > 0
    # Row 48 is near the end of the flick and well clear of the press.
    assert _row_width(tapered, 48) < _row_width(plain, 48)
    assert _row_width(plain, 48) == 20


def _row_width(pixels, row: int) -> int:
    return int((pixels[row, :, 3] > 0).sum())


def test_taper_never_reaches_a_zero_diameter():
    """A stamp with no diameter is a stroke that stops."""
    target = _canvas()
    stroke = brush.StrokeState(
        layer_uid=1, size=(64, 64), before=target.copy(), colour=RED,
        diameter=2, hardness=1.0, speed_taper=1.0,
    )
    stroke.begin((5.0, 32.0), target)
    for step in range(1, 8):
        stroke.to((5.0 + step * 8.0, 32.0), target)
    assert int(target[..., 3].sum()) > 0
