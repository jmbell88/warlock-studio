"""The four FX staples: invert, replace colour, outline and despeckle.

They join a file whose whole shape is "pure function of pixels", so most of what
is here is arithmetic on a five-pixel array. Two rules are worth more than the
arithmetic and are what this file is really for.

**The alpha rule and its one exception.** Everything in ``filters.py`` leaves
alpha alone except blur, sharpen and an outline *placed outside* -- and that
last one is new, so it is asserted precisely rather than in spirit: the set of
pixels whose alpha moved must be exactly the ring, and every other pixel must be
untouched byte for byte.

**The identity rule.** A filter at its declared defaults changes nothing, which
is what lets the popup preview on the frame it opens. Invert is the awkward one
-- nobody opens Invert to invert no channels -- and the way that is resolved
(the popup seeds the toggles, the defaults table does not) is a decision with a
test rather than a compromise.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio import inker
from warlock.studio.inker import filters, indexed


def _flat(colour, size=(4, 4)) -> np.ndarray:
    out = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    out[:, :] = colour
    return out


def _dot(size=(5, 5), at=(2, 2), colour=(255, 255, 255, 255)) -> np.ndarray:
    """One lit pixel on emptiness -- the smallest shape an outline can have."""
    out = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    out[at[1], at[0]] = colour
    return out


# --- invert -----------------------------------------------------------------


def test_invert_flips_the_channels_that_are_on_and_only_those():
    pixels = _flat((10, 20, 30, 128))
    assert tuple(filters.invert(pixels, red=1.0)[0, 0]) == (245, 20, 30, 128)
    assert tuple(filters.invert(pixels, green=1.0, blue=1.0)[0, 0]) == (10, 235, 225, 128)
    assert tuple(filters.invert(pixels, red=1.0, green=1.0, blue=1.0)[0, 0]) == (
        245,
        235,
        225,
        128,
    )


def test_invert_is_its_own_inverse():
    """255 - c is exact in 8-bit, so this is equality and not near-equality --
    which is the property that makes a single-channel invert usable as a way of
    *looking* at a drawing rather than as an edit."""
    pixels = np.random.default_rng(2).integers(0, 256, (6, 5, 4), dtype=np.uint8)
    once = filters.invert(pixels, red=1.0, green=1.0, blue=1.0)
    assert np.array_equal(filters.invert(once, red=1.0, green=1.0, blue=1.0), pixels)


def test_invert_never_touches_alpha():
    pixels = np.random.default_rng(3).integers(0, 256, (6, 5, 4), dtype=np.uint8)
    out = filters.invert(pixels, red=1.0, green=1.0, blue=1.0)
    assert np.array_equal(out[..., 3], pixels[..., 3])


def test_the_defaults_are_the_identity_and_the_popup_seeds_all_three():
    """The two halves of the awkward case, stated together because either one
    alone reads like a bug. A popup that opened on the identity would look
    broken; a defaults table that was not the identity would make the live
    preview change the drawing before the user chose anything.
    """
    assert filters.FILTERS["invert"][0] == {"red": 0.0, "green": 0.0, "blue": 0.0}
    assert filters.popup_values("invert") == {"red": 1.0, "green": 1.0, "blue": 1.0}
    pixels = _flat((10, 20, 30, 128))
    assert np.array_equal(filters.apply_named("invert", pixels), pixels)


def test_popup_values_is_the_defaults_wherever_nothing_is_seeded():
    """The general rule invert was the first case of, and the matte pack the
    next three: a filter nobody opens to do nothing gets a popup seed, and every
    other filter's popup opens exactly on its identity defaults."""
    for name in filters.FILTERS:
        if name not in filters.POPUP_VALUES:
            assert filters.popup_values(name) == filters.FILTERS[name][0], name


def test_every_seeded_popup_still_has_identity_defaults():
    """The half that would be a real bug if it slipped: a defaults table that
    was not the identity makes the live preview change the drawing on the frame
    the popup opens, before the user has chosen anything."""
    pixels = _flat((10, 20, 30, 128))
    for name in filters.POPUP_VALUES:
        assert filters.popup_values(name) != filters.FILTERS[name][0], name
        assert np.array_equal(filters.apply_named(name, pixels), pixels), name


# --- replace colour ---------------------------------------------------------


def test_replace_colour_rewrites_the_match_and_keeps_both_alphas():
    """``new``'s alpha is ignored and the pixel's own is kept: this is a
    recolour, so an antialiased edge of the replaced shade stays antialiased."""
    pixels = np.zeros((1, 3, 4), dtype=np.uint8)
    pixels[0, 0] = (200, 30, 30, 255)
    pixels[0, 1] = (200, 30, 30, 90)
    pixels[0, 2] = (30, 200, 30, 255)

    out = filters.replace_colour(pixels, old=(200, 30, 30, 255), new=(0, 0, 255, 7))
    assert tuple(out[0, 0]) == (0, 0, 255, 255)
    assert tuple(out[0, 1]) == (0, 0, 255, 90)
    assert tuple(out[0, 2]) == (30, 200, 30, 255)


def test_replace_colour_skips_fully_transparent_pixels():
    """A straight-alpha buffer keeps whatever colour was painted under a zero
    alpha -- ``paint_colour`` leaves it there -- so matching on it would
    "replace" pixels nobody can see and, worse, would do it invisibly."""
    pixels = np.zeros((1, 1, 4), dtype=np.uint8)
    pixels[0, 0] = (200, 30, 30, 0)
    out = filters.replace_colour(pixels, old=(200, 30, 30, 255), new=(0, 0, 255, 255))
    assert tuple(out[0, 0]) == (200, 30, 30, 0)


def test_tolerance_is_a_radius_in_rgb_and_bites_at_its_edge():
    pixels = np.zeros((1, 3, 4), dtype=np.uint8)
    pixels[0, 0] = (100, 100, 100, 255)
    pixels[0, 1] = (103, 104, 100, 255)  # distance 5 exactly
    pixels[0, 2] = (100, 100, 106, 255)  # distance 6

    out = filters.replace_colour(
        pixels, old=(100, 100, 100, 255), new=(255, 0, 0, 255), tolerance=5.0
    )
    assert tuple(out[0, 0])[:3] == (255, 0, 0)
    assert tuple(out[0, 1])[:3] == (255, 0, 0)
    assert tuple(out[0, 2])[:3] == (100, 100, 106)


def test_at_zero_tolerance_it_is_exactly_the_palette_remap():
    """The parity that makes it safe for one program to have both.

    Recolouring a palette slot goes through ``indexed.remap`` on the write path
    of an indexed document, so the two cannot share an implementation -- remap
    has to stay a plain equality. They can and must agree about *which pixels
    are that colour*, or a user recolouring by hand and a user editing the slot
    get two different sets of pixels out of the same drawing.
    """
    rng = np.random.default_rng(11)
    pixels = rng.integers(0, 4, (12, 9, 4), dtype=np.uint8) * 64
    old, new = (0, 64, 128, 255), (255, 192, 0, 255)
    assert np.array_equal(
        filters.replace_colour(pixels, old=old, new=new, tolerance=0.0),
        indexed.remap(pixels, old, new),
    )


# --- outline ----------------------------------------------------------------


def test_outline_at_size_zero_is_the_identity_and_a_copy():
    pixels = _dot()
    out = filters.outline(pixels, colour=(255, 0, 0, 255), size=0.0)
    assert np.array_equal(out, pixels)
    assert out is not pixels


def test_an_outside_outline_adds_alpha_and_that_is_the_only_exception():
    """The precise form of the one stated exception to the alpha rule.

    Not "alpha increased somewhere" -- the *set* of pixels whose alpha moved is
    asserted to be exactly the ring, every one of them is asserted to arrive at
    the outline colour's own alpha, and every pixel outside the ring is asserted
    to be untouched byte for byte. A loose version of this test would pass on an
    outline that also nudged the shape's own edge, which is the failure that
    would look like "the outline eats a pixel of my drawing".
    """
    pixels = _dot()
    out = filters.outline(
        pixels, colour=(255, 0, 0, 200), size=1.0, place="outside", corners=4
    )

    ring = np.zeros((5, 5), dtype=bool)
    ring[1, 2] = ring[3, 2] = ring[2, 1] = ring[2, 3] = True

    moved = out[..., 3] != pixels[..., 3]
    assert np.array_equal(moved, ring), "alpha may move on the ring and nowhere else"
    assert set(np.unique(out[..., 3][ring]).tolist()) == {200}
    assert np.array_equal(out[~ring], pixels[~ring]), "and nothing else moved at all"
    assert tuple(out[1, 2]) == (255, 0, 0, 200)


def test_an_inside_outline_touches_no_alpha_at_all():
    """Placed inside, it recolours pixels that were already there -- including a
    half-covered edge pixel, whose coverage it must leave alone. An inside
    outline that wrote its colour's alpha would sharpen every soft edge it
    touched."""
    pixels = np.zeros((7, 7, 4), dtype=np.uint8)
    pixels[1:6, 1:6] = (10, 200, 10, 255)
    pixels[1, 1] = (10, 200, 10, 90)  # a soft corner of the blob
    out = filters.outline(pixels, colour=(0, 0, 0, 255), size=1.0, place="inside")

    assert np.array_equal(out[..., 3], pixels[..., 3])
    # The blob's own border is recoloured, at the coverage it already had, and
    # the middle of the blob is not.
    assert tuple(out[1, 1]) == (0, 0, 0, 90)
    assert tuple(out[1, 3]) == (0, 0, 0, 255)
    assert tuple(out[3, 3]) == (10, 200, 10, 255)


def test_corners_choose_a_diamond_or_a_square_step():
    """The whole of what 4 versus 8 means, on the shape where they differ: the
    diagonal neighbour of a single pixel."""
    pixels = _dot()
    four = filters.outline(pixels, colour=(255, 0, 0, 255), size=1.0, corners=4)
    eight = filters.outline(pixels, colour=(255, 0, 0, 255), size=1.0, corners=8)
    assert int(four[1, 1, 3]) == 0
    assert int(eight[1, 1, 3]) == 255
    assert int(four[1, 2, 3]) == int(eight[1, 2, 3]) == 255


def test_size_is_a_number_of_single_pixel_steps():
    pixels = _dot(size=(9, 9), at=(4, 4))
    out = filters.outline(pixels, colour=(255, 0, 0, 255), size=3.0, corners=4)
    assert int(out[1, 4, 3]) == 255  # three steps up
    assert int(out[0, 4, 3]) == 0  # four is past the ring


def test_wrap_carries_the_outline_across_the_seam():
    """A tiled document has no edge, so an outline that stopped at the border
    would put a seam exactly where the tile joins itself. ``np.roll``, not an
    import of ``tiling.py`` -- that module is about previewing a tiled canvas.
    """
    pixels = _dot(size=(5, 5), at=(0, 2))
    plain = filters.outline(pixels, colour=(255, 0, 0, 255), size=1.0, corners=4)
    tiled = filters.outline(
        pixels, colour=(255, 0, 0, 255), size=1.0, corners=4, wrap=1.0
    )
    assert int(plain[2, 4, 3]) == 0, "clipped to the buffer it was handed"
    assert int(tiled[2, 4, 3]) == 255, "and wrapped onto the far side when tiled"


def test_an_inside_outline_does_not_draw_along_the_edge_of_its_crop():
    """A filter sees the session rect and nothing else, so what is past the edge
    is *unknown*, not empty. Treating it as empty would draw an inner outline
    along a selection's boundary through the middle of a filled area, which is a
    line the user did not ask for and cannot see the cause of.
    """
    filled = _flat((255, 255, 255, 255), size=(5, 5))
    out = filters.outline(filled, colour=(0, 0, 0, 255), size=1.0, place="inside")
    assert np.array_equal(out, filled), "a crop with no visible edge has no inside edge"


# --- despeckle --------------------------------------------------------------


def test_despeckle_at_zero_is_the_identity_and_a_copy():
    pixels = _flat((10, 20, 30, 40))
    out = filters.despeckle(pixels, speck=0.0)
    assert np.array_equal(out, pixels)
    assert out is not pixels


def test_despeckle_deletes_a_lone_pixel_and_leaves_an_edge_hard():
    """Both halves matter: a blur would do the first and ruin the second, which
    is the entire reason this is a median."""
    pixels = _flat((20, 20, 20, 255), size=(8, 6))
    pixels[:, 4:] = (220, 220, 220, 255)
    pixels[2, 1] = (255, 0, 0, 255)  # the speck

    out = filters.despeckle(pixels, speck=1.0)
    assert tuple(out[2, 1])[:3] == (20, 20, 20), "the speck is gone"
    assert tuple(out[2, 3])[:3] == (20, 20, 20)
    assert tuple(out[2, 4])[:3] == (220, 220, 220), "and the edge is still a step"


def test_despeckle_premultiplies_so_invisible_colour_stays_invisible():
    """The blur's rule, and it bites harder here: a median over straight-alpha
    channels can *pick* the invisible colour outright rather than averaging a
    little of it in."""
    pixels = np.zeros((5, 5, 4), dtype=np.uint8)
    pixels[:, :] = (0, 255, 0, 0)  # invisible green everywhere
    pixels[1:4, 1:4] = (255, 0, 0, 255)  # an opaque red blob on top of it
    out = filters.despeckle(pixels, speck=1.0)
    assert int(out[2, 2, 1]) < 40, "the invisible green must not reach the visible red"


def test_despeckle_leaves_a_flat_region_alone():
    pixels = _flat((80, 120, 200, 255), size=(7, 7))
    out = filters.despeckle(pixels, speck=2.0)
    assert np.abs(out[..., :3].astype(int) - np.array([80, 120, 200])).max() <= 1


def test_despeckle_cannot_be_asked_for_a_window_it_cannot_afford():
    """Two locks on the same door, because a median is a rank filter -- it sorts
    every window -- and this runs on the frame thread: ``preview_filter``
    recomputes on every parameter change, and a dragged slider makes one per
    frame. A 65x65 window, which the shared ``radius`` span would have reached,
    is seconds per call on a full-size layer.

    The slider cannot reach past the ceiling, *and* the function clamps, because
    the slider is not the only caller: ``apply_named`` fills parameters in from
    the values the panel remembers per filter, so an entry from a build with a
    different span -- or a script, or this test -- must not be able to ask.
    """
    low, high = filters.RANGES["speck"]
    assert (low, high) == (0.0, (filters.DESPECKLE_MAX - 1) / 2)

    pixels = np.random.default_rng(9).integers(0, 256, (12, 12, 4), dtype=np.uint8)
    at_ceiling = filters.despeckle(pixels, speck=high)
    assert np.array_equal(filters.despeckle(pixels, speck=32.0), at_ceiling)
    # And the ceiling is a real filter rather than a clamp to nothing.
    assert not np.array_equal(at_ceiling, pixels)


# --- through the registry and the session -----------------------------------


@pytest.mark.parametrize("name", ["invert", "replace colour", "outline", "despeckle"])
def test_each_new_filter_is_reachable_by_name_at_its_defaults(name):
    pixels = np.random.default_rng(5).integers(0, 256, (6, 5, 4), dtype=np.uint8)
    assert np.array_equal(filters.apply_named(name, pixels), pixels), name


def test_an_outline_through_a_document_session_is_one_undo_step():
    """The integration shape: the session copies a rect, previews into the
    layer and commits one patch -- and this is the filter that writes *alpha*
    through all of that, which is the path a colour filter never exercised."""
    doc = inker.Document.blank(6, 6)
    doc.stack.active.pixels[2:4, 2:4] = (255, 255, 255, 255)
    doc.invalidate_all()
    depth = len(doc.history)

    assert doc.begin_filter() == (0, 0, 6, 6)
    doc.preview_filter("outline", colour=(255, 0, 0, 255), size=1.0, place="outside")
    assert doc.commit_filter() is True

    assert len(doc.history) == depth + 1
    assert tuple(doc.stack.active.pixels[1, 2]) == (255, 0, 0, 255)
    doc.undo()
    assert int(doc.stack.active.pixels[1, 2, 3]) == 0


def test_a_colour_parameter_survives_the_preview_memo():
    """The session keys its memo on ``(name, sorted(params))``, so a tuple-valued
    parameter has to be hashable -- and a colour that was a list would raise out
    of a popup rather than out of a test."""
    doc = inker.Document.blank(4, 4)
    doc.stack.active.pixels[:, :] = (200, 30, 30, 255)
    doc.begin_filter()
    doc.preview_filter("replace colour", old=(200, 30, 30, 255), new=(0, 0, 255, 255))
    doc.preview_filter("replace colour", old=(200, 30, 30, 255), new=(0, 0, 255, 255))
    assert doc.commit_filter() is True
    assert tuple(doc.stack.active.pixels[0, 0]) == (0, 0, 255, 255)
