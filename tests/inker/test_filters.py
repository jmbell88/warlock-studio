"""Whole-layer filters: the arithmetic, and the session that applies it.

The functions are pure, so every rule about what a filter does to a pixel is a
plain assertion on a tiny array. What the document adds is a *session* -- one
undo step however many times a slider moved -- and that is tested through the
history rather than through the pixels, because the pixels would look right
either way.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio import inker
from warlock.studio.inker import filters


def _flat(colour, size=(4, 4)) -> np.ndarray:
    out = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    out[:, :] = colour
    return out


# --- tone -------------------------------------------------------------------


def test_every_filter_is_the_identity_at_its_declared_defaults():
    """Which is what makes opening a popup safe: the preview runs immediately
    and must not change a pixel until a slider is moved."""
    pixels = np.random.default_rng(3).integers(0, 256, (6, 5, 4), dtype=np.uint8)
    for name in filters.FILTERS:
        out = filters.apply_named(name, pixels)
        assert np.array_equal(out, pixels), name


def test_brightness_moves_colour_and_leaves_alpha_alone():
    pixels = _flat((100, 100, 100, 128))
    out = filters.brightness_contrast(pixels, brightness=0.2)
    assert int(out[0, 0, 0]) == 151  # 100 + 0.2*255, rounded
    assert int(out[0, 0, 3]) == 128


def test_contrast_pivots_on_mid_grey_rather_than_on_the_image():
    """A pivot that moved with the content would make one slider do different
    things to two halves of one drawing."""
    grey = filters.brightness_contrast(_flat((128, 128, 128, 255)), contrast=0.5)
    assert int(grey[0, 0, 0]) in (128, 129), "mid-grey is the fixed point"
    dark = filters.brightness_contrast(_flat((64, 64, 64, 255)), contrast=0.5)
    assert int(dark[0, 0, 0]) < 64


def test_levels_maps_the_black_and_white_points_onto_the_ends():
    ramp = np.zeros((1, 3, 4), dtype=np.uint8)
    ramp[0, :, :3] = np.array([[50], [100], [150]])
    ramp[..., 3] = 255
    out = filters.levels(ramp, black=50, white=150)
    assert int(out[0, 0, 0]) == 0
    assert int(out[0, 2, 0]) == 255
    assert 120 <= int(out[0, 1, 0]) <= 136


def test_a_degenerate_levels_range_does_not_raise():
    """A slider can be dragged there, and a filter that raised mid-drag would
    take the popup down with it."""
    out = filters.levels(_flat((200, 10, 10, 255)), black=200, white=100)
    assert out.shape == (4, 4, 4)
    assert np.all(np.isfinite(out.astype(float)))


def test_gamma_lifts_the_midtones_without_moving_the_ends():
    ramp = np.zeros((1, 3, 4), dtype=np.uint8)
    ramp[0, :, :3] = np.array([[0], [128], [255]])
    ramp[..., 3] = 255
    out = filters.levels(ramp, gamma=2.0)
    assert int(out[0, 0, 0]) == 0
    assert int(out[0, 2, 0]) == 255
    assert int(out[0, 1, 0]) > 128


# --- colour -----------------------------------------------------------------


def test_a_full_hue_turn_is_the_identity():
    pixels = np.random.default_rng(9).integers(0, 256, (5, 5, 4), dtype=np.uint8)
    out = filters.hue_saturation(pixels, hue=1.0)
    assert np.abs(out.astype(int) - pixels.astype(int)).max() <= 1


def test_a_third_of_a_turn_takes_red_to_green_and_green_to_blue():
    red = filters.hue_saturation(_flat((255, 0, 0, 255)), hue=1.0 / 3.0)
    assert tuple(red[0, 0])[:3] == (0, 255, 0)
    green = filters.hue_saturation(_flat((0, 255, 0, 255)), hue=1.0 / 3.0)
    assert tuple(green[0, 0])[:3] == (0, 0, 255)


def test_a_grey_pixel_stays_grey_however_far_saturation_is_pushed():
    """Saturation is undefined for a grey, and inventing a hue for it is how a
    desaturated drawing comes back tinted."""
    out = filters.hue_saturation(_flat((90, 90, 90, 255)), saturation=1.0, hue=0.25)
    assert tuple(out[0, 0])[:3] == (90, 90, 90)


def test_full_desaturation_leaves_equal_channels():
    out = filters.hue_saturation(_flat((200, 30, 60, 255)), saturation=-1.0)
    r, g, b = (int(c) for c in out[0, 0][:3])
    assert r == g == b


def test_lightness_can_reach_both_ends():
    assert tuple(filters.hue_saturation(_flat((120, 40, 40, 255)), lightness=1.0)[0, 0])[:3] == (
        255,
        255,
        255,
    )
    assert tuple(filters.hue_saturation(_flat((120, 40, 40, 255)), lightness=-1.0)[0, 0])[:3] == (
        0,
        0,
        0,
    )


def test_a_colour_filter_never_changes_alpha():
    pixels = np.random.default_rng(11).integers(0, 256, (7, 7, 4), dtype=np.uint8)
    for name in ("brightness / contrast", "hue / saturation", "levels"):
        params = {key: 0.4 for key in filters.FILTERS[name][0]}
        out = filters.apply_named(name, pixels, **params)
        assert np.array_equal(out[..., 3], pixels[..., 3]), name


# --- spatial ----------------------------------------------------------------


def test_blur_premultiplies_so_an_edge_does_not_pick_up_invisible_colour():
    """The case the premultiply exists for, built the way the app builds it.

    ``paint_colour`` leaves paint under a transparent pixel with the alpha at
    zero -- invisible and correct -- so a straight-alpha blur would drag that
    colour into the visible edge as a halo.
    """
    pixels = np.zeros((1, 9, 4), dtype=np.uint8)
    pixels[0, :4] = (0, 255, 0, 0)  # invisible green, exactly as a stroke leaves it
    pixels[0, 4:] = (255, 0, 0, 255)

    out = filters.blur(pixels, radius=2.0)
    edge = out[0, 4]
    assert int(edge[3]) < 255, "the alpha does blur -- that is most of why you blur a layer"
    assert int(edge[1]) < 40, "but the invisible green must not bleed into it"


def test_blur_at_zero_radius_is_the_identity_and_a_copy():
    pixels = _flat((10, 20, 30, 40))
    out = filters.blur(pixels, radius=0.0)
    assert np.array_equal(out, pixels)
    assert out is not pixels


def test_blur_conserves_a_flat_region():
    """Edge padding, not zero padding: a flat colour must not darken at the
    border, which is the classic way a blur betrays its padding mode."""
    out = filters.blur(_flat((80, 120, 200, 255), size=(9, 9)), radius=3.0)
    assert np.abs(out[..., :3].astype(int) - np.array([80, 120, 200])).max() <= 1


def test_sharpen_raises_local_contrast_and_leaves_alpha_alone():
    pixels = np.zeros((1, 8, 4), dtype=np.uint8)
    pixels[0, :4] = (60, 60, 60, 255)
    pixels[0, 4:] = (190, 190, 190, 255)
    soft = filters.blur(pixels, radius=2.0)
    out = filters.sharpen(soft, amount=1.5, radius=2.0)
    step_before = int(soft[0, 4, 0]) - int(soft[0, 3, 0])
    step_after = int(out[0, 4, 0]) - int(out[0, 3, 0])
    assert step_after > step_before
    assert np.array_equal(out[..., 3], soft[..., 3])


# --- the registry -----------------------------------------------------------


def test_every_filter_declares_a_complete_call():
    for name, (defaults, func) in filters.FILTERS.items():
        assert func(_flat((1, 2, 3, 4)), **defaults).shape == (4, 4, 4), name


def test_every_parameter_is_drawable_exactly_one_way():
    """The panel picks a widget per parameter off these four tables, so a
    parameter in none of them is one nobody can set and a parameter in two is a
    widget that depends on the order the pane happens to test them in.

    The numeric case keeps its old assertion as well: a default outside its own
    slider's span is a popup that changes a pixel the moment it opens.
    """
    for name, (defaults, _func) in filters.FILTERS.items():
        # **Per filter, not per name** (6.6): ``invert`` takes red/green/blue
        # as three checkboxes and ``curves`` takes the same three names as
        # per-channel weights. They are the same idea at the precision each
        # filter can use, so the pane asks per filter and so does this.
        kinds = (
            filters.COLOUR_PARAMS,
            filters.toggles_for(name),
            filters.CHOICE_PARAMS,
        )
        for key, value in defaults.items():
            named = [table for table in kinds if key in table]
            assert len(named) <= 1, key
            if named:
                continue
            assert key in filters.RANGES, key
            low, high = filters.RANGES[key]
            assert low <= value <= high, key


def test_every_choice_parameter_defaults_to_one_of_its_choices():
    for defaults, _func in filters.FILTERS.values():
        for key, value in defaults.items():
            if key in filters.CHOICE_PARAMS:
                assert value in filters.CHOICE_PARAMS[key], key


def test_an_unknown_parameter_is_dropped_rather_than_raising():
    """The panel remembers values per filter; a renamed parameter must not turn
    a stale entry into a TypeError out of a slider drag."""
    out = filters.apply_named("blur", _flat((1, 1, 1, 255)), radius=0.0, gone=7)
    assert out.shape == (4, 4, 4)


def test_an_unknown_filter_is_a_programming_error():
    with pytest.raises(KeyError):
        filters.apply_named("posterise", _flat((1, 1, 1, 1)))


# --- the session ------------------------------------------------------------


def test_a_whole_filter_session_is_exactly_one_undo_step():
    doc = inker.Document.blank(8, 8)
    doc.stack.active.pixels[:, :] = (100, 100, 100, 255)
    doc.invalidate_all()
    # Depth, not ``head``: head is the top step's *serial* and serials are
    # global, so "one more" is a fact about the stack rather than about the
    # number.
    depth = len(doc.history)

    assert doc.begin_filter() == (0, 0, 8, 8)
    for value in (0.1, 0.2, 0.3):
        doc.preview_filter("brightness / contrast", brightness=value)
    assert doc.commit_filter() is True

    assert len(doc.history) == depth + 1
    doc.undo()
    assert int(doc.stack.active.pixels[0, 0, 0]) == 100


def test_a_preview_recomputes_from_the_original_rather_than_compounding():
    """Three drags of a brightness slider must not be three brightenings --
    the stroke's coverage rule, applied to a filter."""
    doc = inker.Document.blank(4, 4)
    doc.stack.active.pixels[:, :] = (100, 100, 100, 255)
    doc.begin_filter()
    doc.preview_filter("brightness / contrast", brightness=0.1)
    doc.preview_filter("brightness / contrast", brightness=0.1)
    doc.preview_filter("brightness / contrast", brightness=0.1)
    doc.commit_filter()
    assert int(doc.stack.active.pixels[0, 0, 0]) == 126  # 100 + 0.1*255, once


def test_cancelling_puts_the_pixels_back_and_pushes_nothing():
    doc = inker.Document.blank(4, 4)
    doc.stack.active.pixels[:, :] = (100, 100, 100, 255)
    keep = doc.stack.active.pixels.copy()
    head = doc.history.head

    doc.begin_filter()
    doc.preview_filter("brightness / contrast", brightness=0.5)
    assert not np.array_equal(doc.stack.active.pixels, keep)
    assert doc.cancel_filter() is True

    assert np.array_equal(doc.stack.active.pixels, keep)
    assert doc.history.head == head


def test_applying_a_filter_that_changed_nothing_pushes_nothing():
    """Open, move nothing, press Apply: a document must not ask to be saved."""
    doc = inker.Document.blank(4, 4)
    head = doc.history.head
    doc.begin_filter()
    doc.preview_filter("blur", radius=0.0)
    assert doc.commit_filter() is True
    assert doc.history.head == head


def test_a_filter_is_clipped_to_the_selection_and_faded_by_a_feathered_one():
    doc = inker.Document.blank(8, 8)
    doc.stack.active.pixels[:, :] = (100, 100, 100, 255)
    doc.select(inker.SelectionMask.from_rect((8, 8), (0, 0, 4, 8)))
    doc.begin_filter()
    doc.preview_filter("brightness / contrast", brightness=0.5)
    doc.commit_filter()

    pixels = doc.stack.active.pixels
    assert int(pixels[0, 0, 0]) > 200, "inside the selection"
    assert int(pixels[0, 6, 0]) == 100, "outside it, untouched"


def test_a_filter_respects_the_layers_alpha_lock():
    doc = inker.Document.blank(8, 8)
    doc.stack.active.pixels[2:6, 2:6] = (200, 30, 30, 255)
    doc.stack.active.alpha_lock = True
    before = doc.stack.active.pixels[..., 3].copy()

    doc.begin_filter()
    doc.preview_filter("blur", radius=3.0)
    doc.commit_filter()
    assert np.array_equal(doc.stack.active.pixels[..., 3], before)


def test_beginning_a_second_session_cancels_the_first():
    doc = inker.Document.blank(4, 4)
    doc.stack.active.pixels[:, :] = (100, 100, 100, 255)
    keep = doc.stack.active.pixels.copy()
    doc.begin_filter()
    doc.preview_filter("brightness / contrast", brightness=0.5)
    doc.begin_filter()
    assert np.array_equal(doc.stack.active.pixels, keep)


def test_committing_or_cancelling_without_a_session_is_a_no_op():
    doc = inker.Document.blank(4, 4)
    assert doc.commit_filter() is False
    assert doc.cancel_filter() is False
    assert doc.preview_filter("blur") is False
