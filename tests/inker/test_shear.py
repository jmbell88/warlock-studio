"""Skew: the third thing a free transform can do to a floating buffer.

A shear is the one transform whose *order* is part of the contract, because it
does not commute with a rotation -- shearing a turned rectangle and turning a
sheared one are different pictures. ``FloatingBuffer.transform`` fixes that
order at scale, then shear, then rotate, and the assertions below pin both the
arithmetic and the order.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from warlock.studio.inker import transform as tf
from warlock.studio.inker.document import Document
from warlock.studio.inker.selection import FloatingBuffer, SelectionMask

RED = (255, 0, 0, 255)


def _plane(width: int = 8, height: int = 8) -> np.ndarray:
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[:, :] = RED
    return pixels


# --- the function ------------------------------------------------------------


def test_no_slant_is_a_copy_and_not_a_resample():
    plane = _plane()
    out = tf.shear(plane, (0.0, 0.0))
    assert np.array_equal(out, plane)
    assert out is not plane


def test_a_horizontal_slant_widens_by_the_tangent():
    """``x' = x + tan(sx) y``, so an 8-tall plane sheared 45 degrees gains
    exactly its own height in width."""
    out = tf.shear(_plane(8, 8), (45.0, 0.0), resample="nearest")
    assert out.shape[:2] == (8, 16)


def test_a_vertical_slant_heightens_by_the_tangent():
    out = tf.shear(_plane(8, 8), (0.0, 45.0), resample="nearest")
    assert out.shape[:2] == (16, 8)


def test_the_slant_goes_the_way_the_sign_says():
    """The matrix is *inverted* before Pillow sees it, because ``AFFINE`` walks
    the destination and asks where each pixel came from. Handing over the
    forward matrix shears the picture the other way -- a bug that looks like a
    sign error in the UI."""
    plane = np.zeros((8, 8, 4), dtype=np.uint8)
    plane[:, 0:2] = RED  # a bar down the left edge
    out = tf.shear(plane, (45.0, 0.0), resample="nearest")

    # A positive horizontal slant pushes the *bottom* rows right, so the bar's
    # occupied columns move rightward as the row index grows.
    def first(row) -> int:
        got = np.flatnonzero(out[row, :, 3])
        return int(got[0])

    assert first(1) < first(6)


def test_a_slant_grows_the_plane_rather_than_clipping_the_corners():
    """A one-axis shear preserves area (``det`` is 1 when the other axis is
    zero), so every pixel that went in is still there -- it has only moved."""
    plane = _plane(8, 8)
    out = tf.shear(plane, (20.0, 0.0), resample="nearest")
    assert out.shape[0] == 8 and out.shape[1] > 8
    covered = int(np.count_nonzero(out[..., 3]))
    assert abs(covered - 64) <= 8


def test_a_two_axis_slant_contracts_by_the_determinant():
    """``det = 1 - tan(sx)tan(sy)`` really is the area factor -- the same
    number ``SHEAR_MIN_DET`` puts a floor under, measured here rather than
    asserted, so the floor is a bound on something real."""
    out = tf.shear(_plane(8, 8), (20.0, 20.0), resample="nearest")
    det = 1.0 - math.tan(math.radians(20.0)) ** 2
    covered = int(np.count_nonzero(out[..., 3]))
    assert abs(covered - 64 * det) <= 8


def test_a_mask_shears_too():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    out = tf.shear(mask, (30.0, 0.0), resample="nearest")
    assert out.ndim == 2
    assert out.shape[0] == 8 and out.shape[1] > 8


def test_each_axis_is_clamped_to_the_per_axis_bound():
    """``SHEAR_MAX`` bounds each number the panel can send, and that is all it
    does -- it is deliberately *not* what keeps the transform invertible."""
    assert tf.SHEAR_MAX < 90.0
    out = tf.shear(_plane(), (400.0, -400.0), resample="nearest")
    same = tf.shear(_plane(), (tf.SHEAR_MAX, -tf.SHEAR_MAX), resample="nearest")
    assert np.array_equal(out, same)


def test_the_singular_pair_is_inside_the_per_axis_bound():
    """The bug the old docstring hid. ``det = 1 - tan(sx)tan(sy)`` is zero at
    (45, 45), and 45 is well *under* ``SHEAR_MAX`` -- so the per-axis clamp
    never came near preventing it, and the pair is reachable straight from the
    Slant fields."""
    assert tf.SHEAR_MAX > 45.0
    assert 1.0 - math.tan(math.radians(45.0)) ** 2 == pytest.approx(0.0, abs=1e-12)


def test_a_degenerate_pair_comes_back_unslanted():
    """A refusal, not an approximation: the plane would collapse onto a line
    and there is no inverse to sample through."""
    plane = _plane(8, 8)
    assert np.array_equal(tf.shear(plane, (45.0, 45.0), resample="nearest"), plane)
    assert np.array_equal(tf.shear(plane, (-45.0, -45.0), resample="nearest"), plane)


def test_a_near_degenerate_pair_is_refused_too():
    """"Not quite singular" is not a state worth rendering. At (44, 44) the
    determinant is 0.067, so the sprite would come back as a sliver a fifteenth
    of its area with every pixel sampled from a plane squeezed flat."""
    plane = _plane(8, 8)
    det = 1.0 - math.tan(math.radians(44.0)) ** 2
    assert 0.0 < det < tf.SHEAR_MIN_DET
    assert np.array_equal(tf.shear(plane, (44.0, 44.0), resample="nearest"), plane)


def test_the_floor_lets_an_ordinary_two_axis_slant_through():
    """The guard has to be a floor on the *area factor* rather than a ban on
    two axes: a modest slant on both is an ordinary thing to ask for."""
    det = 1.0 - math.tan(math.radians(30.0)) ** 2
    assert det > tf.SHEAR_MIN_DET
    out = tf.shear(_plane(8, 8), (30.0, 30.0), resample="nearest")
    assert out.shape[:2] != (8, 8)


def test_opposite_signed_slants_never_approach_the_floor():
    """They multiply to a *negative* tangent product, so the determinant grows
    past one rather than shrinking towards zero -- which is exactly why the old
    test at (60, -60) proved nothing whatever about the singularity."""
    for angle in (30.0, 45.0, tf.SHEAR_MAX):
        det = 1.0 + math.tan(math.radians(angle)) ** 2
        assert det > tf.SHEAR_MIN_DET


def test_a_filtered_slant_does_not_bleed_transparent_colour_into_the_edge():
    """The premultiply round trip ``_resample`` owns, reached from here too."""
    plane = np.zeros((16, 16, 4), dtype=np.uint8)
    plane[4:12, 4:12] = (255, 255, 255, 255)
    out = tf.shear(plane, (15.0, 0.0))
    lit = out[..., 3] > 0
    assert lit.any()
    # Nothing anywhere near black: a straight-alpha filter would have dragged
    # the transparent black surround into every edge pixel.
    assert int(out[..., :3][lit].min()) > 100


# --- through a floating buffer -----------------------------------------------


def _floating(doc: Document) -> FloatingBuffer:
    doc.select(SelectionMask.from_rect(doc.size, (2, 2, 14, 14)))
    assert doc.lift()
    assert doc.floating is not None
    return doc.floating


def _doc() -> Document:
    doc = Document.blank(16, 16)
    doc.stack.active.pixels[:, :] = RED
    doc.invalidate_all()
    return doc


def test_a_buffer_reports_itself_transformed_once_it_is_slanted():
    doc = _doc()
    buf = _floating(doc)
    assert not buf.transformed
    assert doc.transform_floating(shear=(10.0, 0.0))
    assert buf.transformed


def test_a_slant_re_renders_from_the_lifted_pixels_rather_than_compounding():
    """Dragging a slant back and forth is one shear, not a chain of them."""
    doc = _doc()
    buf = _floating(doc)
    doc.transform_floating(shear=(20.0, 0.0), resample="nearest")
    doc.transform_floating(shear=(10.0, 0.0), resample="nearest")
    once = buf.pixels.copy()

    doc.transform_floating(shear=(0.0, 0.0), resample="nearest")
    doc.transform_floating(shear=(10.0, 0.0), resample="nearest")

    assert np.array_equal(buf.pixels, once)


def test_the_order_is_scale_then_shear_then_rotate():
    """Asserted against the composition done by hand. If the buffer applied
    them in any other order the two sizes would disagree, because a shear and a
    rotation do not commute."""
    doc = _doc()
    buf = _floating(doc)
    doc.transform_floating(
        scale=(2.0, 1.0), shear=(20.0, 0.0), angle=30.0, resample="nearest"
    )

    source = buf.source
    height, width = source.shape[:2]
    expected = tf.scale(
        source, (round(width * 2.0), round(height * 1.0)), resample="nearest"
    )
    expected = tf.shear(expected, (20.0, 0.0), resample="nearest")
    expected = tf.rotate(expected, 30.0, expand=True, resample="nearest")

    assert np.array_equal(buf.pixels, expected)


def test_the_centre_is_held_fixed_through_a_slant():
    doc = _doc()
    buf = _floating(doc)
    before = buf.centre
    doc.transform_floating(shear=(25.0, 10.0), resample="nearest")
    assert math.dist(buf.centre, before) <= 1.0


def test_a_slanted_buffer_commits_where_it_is_shown():
    doc = _doc()
    _floating(doc)
    doc.transform_floating(shear=(15.0, 0.0), resample="nearest")
    assert doc.commit_floating()
    assert doc.floating is None


def test_base_size_is_the_lifted_size_not_the_transformed_one():
    """What the numeric width and height fields divide by. Using the current
    size instead would compound every keystroke against the last one."""
    doc = _doc()
    buf = _floating(doc)
    lifted = buf.size
    doc.transform_floating(scale=(3.0, 3.0), resample="nearest")
    assert buf.size != lifted
    assert buf.base_size == lifted
