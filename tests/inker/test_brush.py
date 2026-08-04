"""The stamp, the spacing walk, and the modes that read instead of write."""

from __future__ import annotations

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
