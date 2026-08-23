"""The one render both transform doors go through.

``selection.render_transform`` is a module function rather than a method for a
stated reason -- it has **two** callers that must not drift, the floating
buffer's live render and the per-cel replay a ranged commit runs, and "two
spellings would mean a transform that previewed one way and landed another"
(``docs/INVARIANTS.md``, the selection section).

It had no direct test. Its callers were covered, which is the coverage that
cannot catch the thing it exists to prevent: if the function is wrong, both
callers are wrong together and agree with each other perfectly.
"""

from __future__ import annotations

import inspect

import numpy as np

from warlock.studio.inker import selection
from warlock.studio.inker.selection import render_transform


def _square(size: int = 8, at=(2, 2, 6, 6)) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.zeros((size, size, 4), np.uint8)
    x0, y0, x1, y1 = at
    pixels[y0:y1, x0:x1] = (255, 0, 0, 255)
    mask = np.zeros((size, size), np.uint8)
    mask[y0:y1, x0:x1] = 255
    return pixels, mask


def test_the_identity_transform_changes_nothing():
    pixels, mask = _square()
    out, out_mask = render_transform(pixels, mask, 0.0, (1.0, 1.0), (0.0, 0.0), "nearest")
    assert np.array_equal(out, pixels)
    assert np.array_equal(out_mask, mask)


def test_the_mask_is_carried_through_every_stage():
    """The mask is transformed alongside the pixels, never left behind: a
    composite reads it for coverage, so a mask a stage skipped would clip the
    result to the shape it had before that stage."""
    pixels, mask = _square()
    for angle, scale, shear in (
        (0.0, (2.0, 2.0), (0.0, 0.0)),
        (0.0, (1.0, 1.0), (0.3, 0.0)),
        (30.0, (1.0, 1.0), (0.0, 0.0)),
        (30.0, (1.5, 0.5), (0.2, 0.1)),
    ):
        out, out_mask = render_transform(pixels, mask, angle, scale, shear, "nearest")
        assert out.shape[:2] == out_mask.shape[:2], (angle, scale, shear)


def test_scale_then_shear_then_rotate_is_the_order():
    """They do not commute, and the panel's three fields have to mean one
    picture. Asserted by *difference*: a render with the rotation folded in
    first is a different array, so an implementation that reordered the stages
    could not go on matching this."""
    pixels, mask = _square()
    both = render_transform(pixels, mask, 40.0, (2.0, 1.0), (0.0, 0.0), "nearest")[0]

    from warlock.studio.inker import transform as tf

    scaled_first = tf.rotate(
        tf.scale(pixels, (16, 8), resample="nearest"), 40.0, expand=True, resample="nearest"
    )
    rotated_first = tf.scale(
        tf.rotate(pixels, 40.0, expand=True, resample="nearest"), (16, 8), resample="nearest"
    )
    assert np.array_equal(both, scaled_first)
    assert both.shape != rotated_first.shape or not np.array_equal(both, rotated_first)


def test_a_nearest_render_keeps_the_mask_hard():
    """``resample`` reaches the mask too, and the docstring says why: a
    nearest-neighbour rotation whose mask was filtered would keep a one-pixel
    band of partial coverage all the way round, so the hard edge the setting
    exists to preserve would be thrown away at the composite."""
    pixels, mask = _square()
    _, out_mask = render_transform(pixels, mask, 37.0, (1.0, 1.0), (0.0, 0.0), "nearest")
    assert set(np.unique(out_mask).tolist()) <= {0, 255}


def test_a_smooth_render_is_allowed_partial_coverage():
    pixels, mask = _square()
    _, out_mask = render_transform(pixels, mask, 37.0, (1.0, 1.0), (0.0, 0.0), "smooth")
    assert len(set(np.unique(out_mask).tolist())) > 2


def test_a_scale_that_rounds_to_nothing_still_leaves_a_pixel():
    """``max(1, ...)`` on both axes: a zero-sized plane is not a picture, and
    every caller downstream indexes it."""
    pixels, mask = _square()
    out, out_mask = render_transform(pixels, mask, 0.0, (0.001, 0.001), (0.0, 0.0), "nearest")
    assert out.shape[:2] == (1, 1)
    assert out_mask.shape[:2] == (1, 1)


def test_it_stays_a_module_function_with_two_callers():
    """The structural half of the invariant. A method, or a second copy, is the
    failure mode the module function exists to rule out."""
    assert inspect.isfunction(render_transform)
    doc_selection = inspect.getsource(
        __import__(
            "warlock.studio.inker._doc_selection", fromlist=["_doc_selection"]
        )
    )
    assert "render_transform(" in doc_selection
    assert "render_transform(" in inspect.getsource(selection.FloatingBuffer)
