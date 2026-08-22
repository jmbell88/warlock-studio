"""Wave 6.6's colour tools: curves, a custom kernel, harmonies, remap.

Two decisions are pinned here rather than left to the code.

**A curve is three handles, not a spline.** A spline is a document-level object
with control points to serialize, undo, preset and round-trip, and none of the
formats this app writes has anywhere to put one. Three numbers are a filter
parameter like every other filter's -- and the same three the hand actually
pulls a curve at.

**Remap is not set-palette.** Setting a palette re-matches every pixel to the
nearest new colour, so the drawing keeps its colours and gets new indices;
remapping keeps the indices, so it keeps its structure and slot 4 becomes
whatever slot 4 now is. That is what makes a palette swap a recolour rather
than a re-quantisation.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio import inker
from warlock.studio.inker import filters, indexed


def _flat(colour, size=(4, 4)):
    pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    pixels[...] = colour
    return pixels


# --- curves -----------------------------------------------------------------


def test_a_flat_curve_changes_nothing():
    pixels = _flat((100, 120, 140, 255))
    assert np.array_equal(filters.curves(pixels), pixels)


def test_pulling_the_midtones_up_lightens_a_midtone():
    pixels = _flat((128, 128, 128, 255))
    out = filters.curves(pixels, midtones=0.5)
    assert int(out[0, 0, 0]) > 128


def test_a_curve_leaves_the_two_ends_where_they_are():
    """It passes through 0 and 1 unchanged however hard the handles are
    pulled, which is what stops a curve folding back on itself."""

    for colour in ((0, 0, 0, 255), (255, 255, 255, 255)):
        out = filters.curves(_flat(colour), midtones=1.0, shadows=1.0, highlights=1.0)
        assert tuple(int(c) for c in out[0, 0]) == colour


def test_a_channel_weight_of_zero_leaves_that_channel_alone():
    """Which is what makes the control a *target* rather than a second
    amount -- and what a curve needs to warm a picture at all."""

    pixels = _flat((128, 128, 128, 255))
    out = filters.curves(pixels, midtones=0.6, green=0.0, blue=0.0)
    assert int(out[0, 0, 0]) > 128
    assert int(out[0, 0, 1]) == 128
    assert int(out[0, 0, 2]) == 128


# --- the convolution --------------------------------------------------------


def test_the_identity_kernel_is_the_identity():
    pixels = np.random.default_rng(5).integers(0, 256, (6, 5, 4), dtype=np.uint8)
    assert np.array_equal(filters.convolve(pixels), pixels)


def test_a_box_kernel_is_normalised_by_its_own_sum():
    """An unnormalised blur brightens, which is not what the numbers say."""

    pixels = _flat((100, 100, 100, 255), (5, 5))
    out = filters.convolve(pixels, **{f"m{y}{x}": 1.0 for y in range(3) for x in range(3)})
    assert int(out[2, 2, 0]) == 100


def test_an_edge_detect_sums_to_zero_and_is_left_alone():
    """There the point is the difference rather than the level, so
    normalising would be dividing by nothing."""

    pixels = _flat((100, 100, 100, 255), (5, 5))
    out = filters.convolve(
        pixels, m01=-1.0, m10=-1.0, m11=4.0, m12=-1.0, m21=-1.0
    )
    assert int(out[2, 2, 0]) == 0


def test_the_kernel_is_offered_as_nine_scalars():
    """``FILTERS`` is a table of scalar parameters and the popup builds itself
    from it -- a filter needing a widget of its own would be the one entry the
    extension point could not serve."""

    defaults, _func = filters.FILTERS["convolution"]
    assert set(defaults) == {f"m{y}{x}" for y in range(3) for x in range(3)}
    assert all(key in filters.RANGES for key in defaults)


# --- harmonies and shades ---------------------------------------------------


def test_a_triad_is_three_colours_a_third_of_the_wheel_apart():
    out = indexed.harmony((255, 0, 0, 255), "triad")
    assert out == [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]


def test_a_harmony_keeps_the_alpha_and_the_first_entry_is_the_colour():
    for kind in indexed.HARMONIES:
        out = indexed.harmony((30, 120, 200, 128), kind)
        assert out[0] == (30, 120, 200, 128)
        assert all(colour[3] == 128 for colour in out)


def test_an_unknown_harmony_answers_the_colour_rather_than_raising():
    assert indexed.harmony((1, 2, 3, 4), "nonsense") == [(1, 2, 3, 4)]


def test_shades_run_from_black_through_the_colour_to_white():
    ramp = indexed.shades((255, 0, 0, 255), 5)
    assert ramp[0] == (0, 0, 0, 255)
    assert ramp[2] == (255, 0, 0, 255)
    assert ramp[-1] == (255, 255, 255, 255)


# --- used, unused, remap ----------------------------------------------------


def _indexed_doc():
    doc = inker.Document.blank(4, 4)
    doc.set_palette([(0, 0, 0, 255), (255, 0, 0, 255), (0, 255, 0, 255)])
    doc.stack.active.pixels[...] = (255, 0, 0, 255)
    doc.convert_to_indexed(list(doc.palette), "nearest")
    return doc


def test_used_and_unused_are_two_readings_of_one_histogram():
    doc = _indexed_doc()
    used, unused = doc.used_slots(), doc.unused_slots()
    assert set(used) & set(unused) == set()
    assert sorted(used + unused) == list(range(len(doc.palette)))


def test_remapping_keeps_every_index_and_changes_the_colours():
    doc = _indexed_doc()
    before = doc.stack.active.indices.copy()
    assert doc.remap_palette([(0, 0, 0, 255), (0, 0, 255, 255), (0, 255, 0, 255)])
    assert np.array_equal(doc.stack.active.indices, before)
    assert tuple(int(c) for c in doc.stack.active.pixels[0, 0]) == (0, 0, 255, 255)


def test_remapping_a_document_that_is_not_indexed_is_refused_by_name():
    doc = inker.Document.blank(4, 4)
    with pytest.raises(ValueError, match="indexed"):
        doc.remap_palette([(1, 2, 3, 255)])


def test_remapping_to_the_same_palette_changes_nothing():
    doc = _indexed_doc()
    assert doc.remap_palette(list(doc.palette)) is False
