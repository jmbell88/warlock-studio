"""Gradients and whole-plane geometry, as pure functions."""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import gradient as grad
from warlock.studio.inker import transform as tf

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


# --- multi-stop gradients (Ink6) ---------------------------------------------


def test_two_colours_and_a_two_stop_list_are_the_same_call():
    """The shorthand builds the list rather than having arithmetic of its own,
    which is what stops the two disagreeing in the last ulp at the ends."""
    size, p0, p1 = (16, 1), (0, 0), (15, 0)
    a_rgba, a_weight = grad.render(size, p0, p1, RED, (0, 0, 255, 0))
    b_rgba, b_weight = grad.render(
        size, p0, p1, stops=[(0.0, RED), (1.0, (0, 0, 255, 0))]
    )
    assert np.array_equal(a_rgba, b_rgba)
    assert np.array_equal(a_weight, b_weight)


def test_a_middle_stop_is_reached_where_it_was_placed():
    rgba, _ = grad.render(
        (11, 1),
        (0, 0),
        (10, 0),
        stops=[(0.0, (0, 0, 0, 255)), (0.5, (255, 0, 0, 255)), (1.0, (0, 0, 0, 255))],
    )
    assert rgba[0, 5, 0] == pytest.approx(1.0, abs=0.02)
    assert rgba[0, 0, 0] == pytest.approx(0.0, abs=0.02)
    assert rgba[0, 10, 0] == pytest.approx(0.0, abs=0.02)


def test_a_gradient_can_fade_out_in_the_middle_and_back_in():
    """Which is the whole reason the weight is per stop rather than per end."""
    _rgba, weight = grad.render(
        (11, 1),
        (0, 0),
        (10, 0),
        stops=[(0.0, RED), (0.5, (255, 0, 0, 0)), (1.0, RED)],
    )
    assert weight[0, 5] == pytest.approx(0.0, abs=0.02)
    assert weight[0, 0] == pytest.approx(1.0, abs=0.02)
    assert weight[0, 10] == pytest.approx(1.0, abs=0.02)


def test_stops_are_sorted_rather_than_required_sorted():
    """The UI's list is in the order stops were added, and dragging one past
    another must not invert the ramp."""
    forward = grad.sample([(0.0, RED), (1.0, (0, 0, 255, 255))], np.linspace(0, 1, 9))
    shuffled = grad.sample([(1.0, (0, 0, 255, 255)), (0.0, RED)], np.linspace(0, 1, 9))
    assert np.array_equal(forward, shuffled)


def test_one_stop_is_a_flat_colour_rather_than_an_error():
    """Dragging a two-stop gradient down to one is something a user does on the
    way to a three-stop one; refusing mid-edit would be refusing a keystroke."""
    out = grad.sample([(0.3, RED)], np.linspace(0, 1, 5))
    assert out.shape == (5, 4)
    assert np.allclose(out[:, 0], 1.0)


def test_no_stops_at_all_is_a_programming_error():
    with pytest.raises(ValueError):
        grad.sample([], np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError):
        grad.render((4, 1), (0, 0), (3, 0))


def test_a_stop_list_is_clamped_past_both_ends():
    out = grad.sample([(0.25, RED), (0.75, (0, 0, 0, 255))], np.array([0.0, 1.0]))
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(0.0)


# --- the resize anchor (Ink7) ------------------------------------------------


def test_the_default_anchor_is_the_corner_the_old_offset_meant():
    assert tf.anchor_offset((8, 8), (16, 16), "top-left") == (0, 0)


def test_centring_splits_the_slack_and_rounds_rather_than_flooring():
    assert tf.anchor_offset((8, 8), (16, 16), "centre") == (4, 4)
    assert tf.anchor_offset((8, 8), (10, 10), "centre") == (1, 1)
    # An odd growth: one extra pixel, and it goes after rather than being lost.
    assert tf.anchor_offset((8, 8), (11, 11), "centre") == (2, 2)


def test_the_far_corner_puts_every_pixel_of_slack_before_the_image():
    assert tf.anchor_offset((8, 4), (12, 10), "bottom-right") == (4, 6)


def test_shrinking_gives_a_negative_offset_which_is_a_crop():
    """Growing and cropping are one operation, so an anchor is one number
    either way -- which is what ``resize_canvas`` already meant by a negative
    offset."""
    assert tf.anchor_offset((16, 16), (8, 8), "centre") == (-4, -4)
    assert tf.anchor_offset((16, 16), (8, 8), "top-left") == (0, 0)


def test_an_unknown_anchor_falls_back_to_the_corner():
    """It comes out of a settings dictionary, so a name this build does not
    carry is a stale file rather than a bug worth raising over."""
    assert tf.anchor_offset((8, 8), (16, 16), "middle-ish") == (0, 0)


def test_every_named_anchor_places_the_image_inside_the_new_canvas():
    for name in tf.ANCHORS:
        ox, oy = tf.anchor_offset((8, 6), (20, 14), name)
        assert 0 <= ox <= 12 and 0 <= oy <= 8, name


def test_a_centred_canvas_growth_puts_the_image_in_the_middle():
    from warlock.studio import inker

    doc = inker.Document.blank(4, 4)
    doc.stack.active.pixels[:, :] = RED
    doc.resize_canvas((8, 8), anchor="centre")
    pixels = doc.stack.active.pixels
    assert doc.size == (8, 8)
    assert tuple(pixels[4, 4]) == RED
    assert tuple(pixels[0, 0]) == (0, 0, 0, 0)
    assert tuple(pixels[7, 7]) == (0, 0, 0, 0)


def test_an_explicit_offset_still_wins_over_the_anchor():
    """The general form, and every caller that already computed one keeps
    working unchanged."""
    from warlock.studio import inker

    doc = inker.Document.blank(4, 4)
    doc.stack.active.pixels[:, :] = RED
    doc.resize_canvas((8, 8), (0, 0), anchor="centre")
    assert tuple(doc.stack.active.pixels[0, 0]) == RED


# --- whole-number magnification (C13a) ---------------------------------------


def test_upscale_repeats_every_pixel_exactly():
    """Not a resize with a nearest filter: Pillow picks a source pixel per
    destination pixel by rounding a ratio, which puts a one-pixel jitter into
    the block edges. ``np.repeat`` places every block exactly."""
    pixels = np.array(
        [[[1, 2, 3, 255], [4, 5, 6, 255]]], dtype=np.uint8
    )
    out = tf.upscale(pixels, 3)
    assert out.shape == (3, 6, 4)
    assert np.array_equal(out[0, 0], pixels[0, 0])
    assert np.array_equal(out[2, 2], pixels[0, 0])
    assert np.array_equal(out[0, 3], pixels[0, 1])
    # Every block is uniform, i.e. nothing was blended anywhere.
    assert len(np.unique(out.reshape(-1, 4), axis=0)) == 2


def test_upscale_by_one_is_the_untouched_path():
    pixels = np.zeros((2, 2, 4), dtype=np.uint8)
    assert tf.upscale(pixels, 1) is pixels


def test_upscale_floors_at_one_rather_than_emptying_the_image():
    pixels = np.zeros((2, 2, 4), dtype=np.uint8)
    assert tf.upscale(pixels, 0).shape == (2, 2, 4)
    assert tf.upscale(pixels, -4).shape == (2, 2, 4)


def test_upscale_preserves_alpha_untouched():
    """No premultiply round trip anywhere: nothing is mixed, so dividing a
    rounded product back out would move colours for no reason at all."""
    pixels = np.array([[[200, 100, 50, 3]]], dtype=np.uint8)
    out = tf.upscale(pixels, 4)
    assert set(np.unique(out[..., 3]).tolist()) == {3}
    assert np.array_equal(out[2, 2], pixels[0, 0])
