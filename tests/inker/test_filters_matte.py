"""The matte-cleanup four: what a BiRefNet cutout needs and the eraser was.

A cutout lands with a halo -- a rim of pixels whose colour is a blend of the
subject and whatever was behind it, at partial coverage -- and with stray single
pixels the reduction invented. These four are the ones that fix it, and what is
asserted here is what each does to a *synthesized* halo, not to a real one: the
fixtures are built so a wrong answer is one assertion away rather than a
judgement call about how a photograph looks.

The existing ``FILTERS``-parametrized sweeps in ``test_fx_filters.py`` already
cover the identity-at-defaults and shape properties for all four; nothing here
duplicates them.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker import filters

SUBJECT = (200, 60, 40, 255)
BACKDROP = (30, 200, 90)


def _blob(size: int = 9, rim: int = 128) -> np.ndarray:
    """A solid square with a one-pixel semi-transparent rim in a blended colour.

    The rim's colour is halfway between the subject and a green backdrop, which
    is exactly the artefact: a defringe must replace it with the subject's own
    colour and must leave its alpha where it is.
    """
    out = np.zeros((size, size, 4), dtype=np.uint8)
    blend = tuple((SUBJECT[i] + BACKDROP[i]) // 2 for i in range(3))
    out[2:size - 2, 2:size - 2, :3] = blend
    out[2:size - 2, 2:size - 2, 3] = rim
    out[3:size - 3, 3:size - 3, :3] = SUBJECT[:3]
    out[3:size - 3, 3:size - 3, 3] = 255
    return out


# --- alpha threshold ---------------------------------------------------------


def test_a_threshold_of_zero_is_bit_identity() -> None:
    blob = _blob()
    assert np.array_equal(filters.alpha_threshold(blob, threshold=0.0), blob)


def test_the_snap_is_exact_at_the_boundary() -> None:
    """At or above becomes opaque; below becomes a hole. The equality case is
    the one a rewrite gets wrong."""
    plane = np.zeros((1, 4, 4), dtype=np.uint8)
    plane[0, :, 3] = (126, 127, 128, 129)
    out = filters.alpha_threshold(plane, threshold=128.0)
    assert out[0, :, 3].tolist() == [0, 0, 255, 255]


def test_the_threshold_leaves_colour_alone() -> None:
    blob = _blob()
    out = filters.alpha_threshold(blob, threshold=128.0)
    assert np.array_equal(out[..., :3], blob[..., :3])


# --- defringe ----------------------------------------------------------------


def test_defringe_recolours_the_halo_from_the_subject() -> None:
    blob = _blob()
    rim = (blob[..., 3] > 0) & (blob[..., 3] < 255)
    assert rim.any()
    out = filters.defringe(blob, fringe=2.0)
    assert {tuple(int(v) for v in c) for c in out[..., :3][rim]} == {SUBJECT[:3]}


def test_defringe_leaves_every_alpha_untouched() -> None:
    """It is a colour filter, not an alpha exception -- the coverage is what
    the matte decided and this has no opinion about it."""
    blob = _blob()
    out = filters.defringe(blob, fringe=4.0)
    assert np.array_equal(out[..., 3], blob[..., 3])


def test_defringe_only_ever_writes_colours_that_were_already_opaque() -> None:
    """Copy, never average: a blend is precisely the halo being removed, and on
    a palette-locked document a copied colour is already a palette member."""
    blob = _blob()
    opaque = {tuple(int(v) for v in c) for c in blob[..., :3][blob[..., 3] == 255]}
    out = filters.defringe(blob, fringe=3.0)
    written = {tuple(int(v) for v in c) for c in out[..., :3][blob[..., 3] > 0]}
    assert written <= opaque


def test_defringe_of_zero_is_bit_identity() -> None:
    blob = _blob()
    assert np.array_equal(filters.defringe(blob, fringe=0.0), blob)


def test_defringe_with_no_opaque_pixel_anywhere_changes_nothing() -> None:
    ghost = np.zeros((5, 5, 4), dtype=np.uint8)
    ghost[..., :3] = 100
    ghost[..., 3] = 40
    assert np.array_equal(filters.defringe(ghost, fringe=4.0), ghost)


# --- grow / shrink matte -----------------------------------------------------


def _coverage(pixels: np.ndarray) -> int:
    return int((pixels[..., 3] == 255).sum())


def test_growing_the_matte_increases_coverage_monotonically() -> None:
    blob = _blob()
    base = _coverage(blob)
    one = _coverage(filters.matte_grow(blob, grow=1.0))
    two = _coverage(filters.matte_grow(blob, grow=2.0))
    assert base < one < two


def test_shrinking_the_matte_decreases_coverage_monotonically() -> None:
    blob = _blob()
    base = _coverage(blob)
    one = _coverage(filters.matte_grow(blob, grow=-1.0))
    two = _coverage(filters.matte_grow(blob, grow=-2.0))
    assert two < one < base


def test_grown_pixels_copy_a_neighbours_colour_and_take_full_alpha() -> None:
    blob = _blob()
    out = filters.matte_grow(blob, grow=1.0)
    added = (out[..., 3] == 255) & (blob[..., 3] != 255)
    assert added.any()
    existing = {tuple(int(v) for v in c) for c in blob[..., :3][blob[..., 3] == 255]}
    assert {tuple(int(v) for v in c) for c in out[..., :3][added]} <= existing


def test_grow_of_zero_is_bit_identity() -> None:
    blob = _blob()
    assert np.array_equal(filters.matte_grow(blob, grow=0.0), blob)


# Grow-then-shrink is deliberately NOT asserted as an identity: a dilation
# rounds a corner off and no erosion puts it back. Monotonicity per sign above
# is the property that actually holds.


# --- remove orphans ----------------------------------------------------------


def _field(size: int = 7) -> np.ndarray:
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., :3] = (40, 40, 40)
    out[..., 3] = 255
    return out


def test_a_lone_pixel_takes_its_neighbours_colour() -> None:
    field = _field()
    field[3, 3, :3] = (250, 10, 10)
    out = filters.remove_orphans(field, orphans=1.0)
    assert tuple(int(v) for v in out[3, 3, :3]) == (40, 40, 40)


def test_a_two_pixel_pair_survives() -> None:
    """The contrast with despeckle, and the reason both exist: a median deletes
    one-pixel detail wholesale, this deletes only friendless pixels."""
    field = _field()
    field[3, 3, :3] = (250, 10, 10)
    field[3, 4, :3] = (250, 10, 10)
    out = filters.remove_orphans(field, orphans=1.0)
    assert tuple(int(v) for v in out[3, 3, :3]) == (250, 10, 10)
    assert tuple(int(v) for v in out[3, 4, :3]) == (250, 10, 10)


def test_an_isolated_hole_is_left_alone() -> None:
    """Transparent pixels are left alone in both roles: an isolated hole is a
    silhouette decision, not an artefact."""
    field = _field()
    field[3, 3, 3] = 0
    out = filters.remove_orphans(field, orphans=1.0)
    assert out[3, 3, 3] == 0
    assert np.array_equal(out[..., 3], field[..., 3])


def test_removing_orphans_off_is_bit_identity() -> None:
    field = _field()
    field[3, 3, :3] = (250, 10, 10)
    assert np.array_equal(filters.remove_orphans(field, orphans=0.0), field)


def test_a_tiny_image_is_left_alone() -> None:
    tiny = np.zeros((2, 2, 4), dtype=np.uint8)
    tiny[..., 3] = 255
    assert np.array_equal(filters.remove_orphans(tiny, orphans=1.0), tiny)


def test_a_lone_pixel_with_no_opaque_neighbour_is_left_alone() -> None:
    lone = np.zeros((5, 5, 4), dtype=np.uint8)
    lone[2, 2] = (250, 10, 10, 255)
    assert np.array_equal(filters.remove_orphans(lone, orphans=1.0), lone)


# --- the registry picks all four up ------------------------------------------


def test_all_four_are_reachable_through_the_registry() -> None:
    """The module's own contract: a new filter is an entry here and no edit in
    any pane -- which is what buys live preview, alpha lock and one-undo commit.
    """
    for name in ("alpha threshold", "defringe", "grow / shrink matte", "remove orphans"):
        assert name in filters.FILTERS
        blob = _blob()
        assert filters.apply_named(name, blob).shape == blob.shape
