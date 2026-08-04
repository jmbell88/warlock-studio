"""Gradients and whole-plane geometry, as pure functions."""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.paint import gradient as grad
from warlock.studio.paint import transform as tf

RED = (255, 0, 0, 255)


def _plane(w=8, h=4, colour=(255, 255, 255, 255)):
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :] = colour
    return out


# --- gradients --------------------------------------------------------------


def test_a_linear_ramp_is_zero_at_one_end_and_one_at_the_other():
    t = grad.ramp((16, 1), (0, 0), (15, 0))
    assert t[0, 0] == pytest.approx(0.0)
    assert t[0, 15] == pytest.approx(1.0)


def test_a_ramp_is_clamped_past_both_ends_rather_than_wrapping():
    t = grad.ramp((16, 1), (4, 0), (8, 0))
    assert t[0, 0] == pytest.approx(0.0)
    assert t[0, 15] == pytest.approx(1.0)


def test_a_radial_ramp_grows_with_distance_from_the_centre():
    t = grad.ramp((32, 32), (16, 16), (16, 0), "radial")
    assert t[16, 16] < t[16, 8] < t[16, 0]


def test_a_degenerate_gradient_is_all_end_colour_rather_than_a_divide_by_zero():
    assert np.allclose(grad.ramp((4, 4), (2, 2), (2, 2)), 1.0)
    assert np.allclose(grad.ramp((4, 4), (2, 2), (2, 2), "radial"), 1.0)


def test_an_unknown_gradient_kind_is_a_programming_error():
    with pytest.raises(ValueError):
        grad.ramp((4, 4), (0, 0), (1, 1), "conical")


def test_the_alpha_ramp_comes_back_as_a_weight_not_as_the_colours_alpha():
    """Otherwise "foreground to transparent" fades toward black instead of
    fading out: the colour has to stay the colour all the way along."""
    rgba, weight = grad.render((4, 1), (0, 0), (3, 0), RED, (255, 0, 0, 0))
    assert np.allclose(rgba[0, :, 0], 1.0)
    assert weight[0, 0] == pytest.approx(1.0)
    assert weight[0, 3] == pytest.approx(0.0)


# --- flips and turns --------------------------------------------------------


def test_a_flip_moves_the_pixels_and_is_its_own_inverse():
    plane = _plane()
    plane[:, 0] = RED
    once = tf.flip(plane, "horizontal")
    assert tuple(once[0, 7]) == RED
    assert np.array_equal(tf.flip(once, "horizontal"), plane)


def test_a_vertical_flip_is_not_a_horizontal_one():
    plane = _plane()
    plane[0, :] = RED
    assert tuple(tf.flip(plane, "vertical")[3, 0]) == RED


def test_an_unknown_flip_axis_is_a_programming_error():
    with pytest.raises(ValueError):
        tf.flip(_plane(), "diagonal")


def test_a_quarter_turn_swaps_the_dimensions_and_four_of_them_are_identity():
    plane = _plane()
    turned = tf.rotate90(plane)
    assert turned.shape[:2] == (8, 4)
    assert np.array_equal(tf.rotate90(plane, 4), plane)


def test_a_quarter_turn_loses_nothing_at_all():
    rng = np.random.default_rng(2)
    plane = rng.integers(0, 256, (5, 9, 4), dtype=np.uint8)
    assert np.array_equal(tf.rotate90(tf.rotate90(plane, 3)), plane)


# --- resampling -------------------------------------------------------------


def test_scaling_up_and_back_down_keeps_a_flat_colour_flat():
    plane = _plane(8, 8, (30, 60, 90, 255))
    out = tf.scale(tf.scale(plane, (32, 32)), (8, 8))
    assert np.allclose(out.astype(int), plane.astype(int), atol=2)


def test_resampling_does_not_bleed_transparent_pixels_into_the_edge():
    """Straight alpha plus a bilinear filter drags the colour of fully
    transparent pixels into everything they touch -- black fringing on every
    scaled sprite. Premultiplying first is the whole fix."""
    plane = np.zeros((16, 16, 4), dtype=np.uint8)
    plane[4:12, 4:12] = RED  # opaque red on transparent *black*
    out = tf.scale(plane, (32, 32))
    lit = out[..., 3] > 128
    assert np.all(out[lit][:, 0] > 200)
    assert np.all(out[lit][:, 1] < 40)


def test_a_mask_resamples_without_gaining_a_channel():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    out = tf.scale(mask, (16, 16))
    assert out.ndim == 2 and out.shape == (16, 16)
    assert int(out[8, 8]) > 200


def test_an_arbitrary_rotation_keeps_the_canvas_size_unless_asked_to_expand():
    plane = _plane(16, 16)
    assert tf.rotate(plane, 30.0).shape[:2] == (16, 16)
    assert tf.rotate(plane, 30.0, expand=True).shape[0] > 16


# --- crop and canvas --------------------------------------------------------


def test_cropping_keeps_what_was_inside_the_rectangle():
    plane = _plane(16, 16)
    plane[4:8, 4:8] = RED
    out = tf.crop(plane, (4, 4, 8, 8))
    assert out.shape[:2] == (4, 4)
    assert tuple(out[0, 0]) == RED


def test_growing_the_canvas_places_the_pixels_rather_than_stretching_them():
    plane = _plane(4, 4, RED)
    out = tf.resize_canvas(plane, (8, 8), (2, 2))
    assert tuple(out[3, 3]) == RED
    assert int(out[0, 0][3]) == 0


def test_a_negative_offset_crops_from_the_top_left():
    plane = _plane(8, 8)
    plane[0, 0] = RED
    out = tf.resize_canvas(plane, (8, 8), (-2, -2))
    assert int(out[0, 0][3]) == 255
    assert tuple(out[6, 6]) == (0, 0, 0, 0)


def test_shrinking_the_canvas_discards_what_falls_outside_it():
    plane = _plane(8, 8, RED)
    out = tf.resize_canvas(plane, (4, 4))
    assert out.shape[:2] == (4, 4)
    assert tuple(out[3, 3]) == RED


def test_a_canvas_resize_that_overlaps_nothing_gives_an_empty_plane():
    out = tf.resize_canvas(_plane(4, 4, RED), (4, 4), (100, 100))
    assert out.max() == 0
