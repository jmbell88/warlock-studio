"""The index-plane arithmetic, on its own, before any document touches it.

Every test here is about the one thing an index plane buys that constrain-on-
write cannot: **identity**. Two palette slots holding the same colour are one
colour in an RGBA plane and two slots in an index plane, and that difference is
what makes a palette reorder exact, a duplicate-swatch ``.aseprite`` file
round-trip, and "which slot did I paint with" answerable.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import dither
from warlock.studio.inker import index_plane as ixp

# Slots 1 and 3 are the *same red*. Every duplicate-identity assertion below
# leans on that, and on slot 0 being the transparent one while carrying an
# opaque black -- which is how Aseprite writes a palette, and the trap the
# "never a nearest candidate" rule exists to avoid.
PALETTE = [(0, 0, 0, 255), (200, 20, 20, 255), (20, 200, 20, 255), (200, 20, 20, 255)]


def _pixels(rows):
    return np.asarray(rows, dtype=np.uint8)


# --- lut --------------------------------------------------------------------


def test_the_transparent_slot_materialises_as_a_hole_without_being_edited():
    """A projection, not an edit. Aseprite gives the transparent index a
    displayable colour; honouring it literally would paint the hole, and
    rewriting the stored entry would lose what a writer has to put back."""
    table = ixp.lut(PALETTE, 0)
    assert table[0].tolist() == [0, 0, 0, 0]
    assert PALETTE[0] == (0, 0, 0, 255), "the caller's table was mutated"
    assert table[1].tolist() == [200, 20, 20, 255]


def test_a_palette_over_the_cap_is_refused_by_name():
    with pytest.raises(ValueError, match="at most 256"):
        ixp.lut([(0, 0, 0, 255)] * 257)


def test_an_empty_palette_is_refused():
    with pytest.raises(ValueError, match="at least one colour"):
        ixp.lut([])


def test_a_palette_entry_without_alpha_is_opaque():
    """RGB triples arrive from ``.gpl`` imports and from hand-written tests. The
    honest default for "no alpha stated" is opaque, not zero -- zero would make
    an imported palette invisible."""
    assert ixp.lut([(1, 2, 3)], transparent=-1)[0].tolist() == [1, 2, 3, 255]


# --- resolve ----------------------------------------------------------------


def test_alpha_is_the_only_road_to_the_transparent_index():
    """The rule the whole design rests on. Slot 0 is an opaque black in the
    table and a *drawn* black must not collapse into it -- so a black pixel
    resolves to some other slot, and only sub-threshold alpha reaches 0."""
    table = ixp.lut(PALETTE, 0)
    px = _pixels([[(0, 0, 0, 255), (0, 0, 0, 0)]])
    out = ixp.resolve(px, table, 0)
    assert out[0, 1] == 0, "a fully transparent pixel is the transparent index"
    assert out[0, 0] != 0, "a painted black must not become a hole"


@pytest.mark.parametrize(
    ("alpha", "hole"),
    [(0, True), (1, True), (127, True), (128, False), (129, False), (255, False)],
)
def test_the_alpha_threshold_is_the_midpoint(alpha, hole):
    """128, and the boundary is inclusive on the opaque side. A soft nib rounds
    to the shape the user drew rather than fattening or eating an edge."""
    table = ixp.lut(PALETTE, 0)
    out = ixp.resolve(_pixels([[(20, 200, 20, alpha)]]), table, 0)
    assert bool(out[0, 0] == 0) is hole


def test_duplicate_slots_fall_to_lowest_index_without_a_preference():
    """Deterministic, and stated: output no gesture authored -- a filter, a
    paste, a gradient -- has no slot to prefer, so it must not depend on
    iteration order."""
    table = ixp.lut(PALETTE, 0)
    out = ixp.resolve(_pixels([[(200, 20, 20, 255)]]), table, 0)
    assert out[0, 0] == 1


def test_prefer_makes_the_painted_slot_the_one_that_lands():
    """The gap that would otherwise be visible: painting with slot 3 in a table
    where slot 1 is the same red would silently record slot 1, and the next
    recolour of slot 3 would leave the stroke behind."""
    table = ixp.lut(PALETTE, 0)
    px = _pixels([[(200, 20, 20, 255), (20, 200, 20, 255)]])
    out = ixp.resolve(px, table, 0, prefer=((200, 20, 20, 255), 3))
    assert out[0, 0] == 3, "the preferred slot wins its own colour"
    assert out[0, 1] == 2, "and changes nothing about any other colour"


def test_prefer_cannot_smuggle_a_pixel_into_the_transparent_slot():
    """Otherwise "paint with the transparent swatch" would punch holes through
    a mechanism that is supposed to have exactly one entrance."""
    table = ixp.lut(PALETTE, 0)
    out = ixp.resolve(_pixels([[(0, 0, 0, 255)]]), table, 0, prefer=((0, 0, 0, 255), 0))
    assert out[0, 0] != 0


def test_a_one_colour_palette_is_degenerate_rather_than_a_refusal():
    """Its only slot is also its transparent one. There is no wrong answer to
    give, and raising here would refuse a legal (if pointless) document."""
    table = ixp.lut([(9, 9, 9, 255)], 0)
    assert ixp.resolve(_pixels([[(255, 0, 0, 255)]]), table, 0)[0, 0] == 0


def test_transparent_none_makes_every_slot_a_candidate():
    """The conversion helper's shape: a candidate sub-table with no hole in it."""
    table = ixp.lut(PALETTE, transparent=-1)
    assert ixp.resolve(_pixels([[(0, 0, 0, 255)]]), table, None)[0, 0] == 0


def test_resolve_and_materialize_round_trip_an_already_indexed_picture():
    """The property the funnel depends on: pixels that are exactly on the table
    resolve back to slots that materialise back to the same pixels."""
    table = ixp.lut(PALETTE, 0)
    plane = np.asarray([[1, 2, 3], [3, 1, 2]], dtype=np.uint8)
    pixels = ixp.materialize(plane, table)
    assert np.array_equal(ixp.materialize(ixp.resolve(pixels, table, 0), table), pixels)


def test_materialize_returns_a_fresh_contiguous_array():
    """It is assigned straight onto ``Layer.pixels``, which the compositor
    slices and the texture uploader reads; a fancy-indexed view would surprise
    both."""
    table = ixp.lut(PALETTE, 0)
    out = ixp.materialize(np.zeros((2, 2), np.uint8), table)
    assert out.flags["C_CONTIGUOUS"] and out.base is None


def test_materialize_clips_rather_than_raises_on_a_stale_index():
    """A plane can outlive its table by one step of a history walk. Showing a
    wrong colour for a frame is recoverable; raising out of the middle of an
    undo is not."""
    table = ixp.lut(PALETTE[:2], 0)
    out = ixp.materialize(np.asarray([[7]], np.uint8), table)
    assert out[0, 0].tolist() == table[1].tolist()


# --- remaps -----------------------------------------------------------------


def test_a_move_permutation_is_exact_and_invertible():
    """The payoff of storing slots: reordering a palette is a table applied to
    the plane, with no colour comparison anywhere in it -- so the two identical
    reds stay two different slots across the move and back."""
    forward, inverse = ixp.permutation_tables(len(PALETTE), 0, 2)
    plane = np.asarray([[0, 1, 2, 3]], dtype=np.uint8)
    moved = ixp.apply_remap(plane, forward)
    assert np.array_equal(ixp.apply_remap(moved, inverse), plane)
    # And the map agrees with what the list operation does to the table.
    table = list(PALETTE)
    table.insert(2, table.pop(0))
    assert [table[int(i)] for i in moved[0]] == [PALETTE[int(i)] for i in plane[0]]


def test_a_merge_map_shifts_everything_above_the_removed_slot():
    """The table it is applied to has had an entry taken out of it, so a map
    that only redirected the dead slot would leave every slot above it pointing
    one colour too high."""
    forward = ixp.merge_table(4, 1, 2)
    assert forward.tolist() == [0, 1, 1, 2]


def test_a_merge_into_itself_is_refused_by_name():
    with pytest.raises(ValueError, match="merge into itself"):
        ixp.merge_table(4, 1, 1)


def test_the_last_slot_cannot_be_removed():
    with pytest.raises(ValueError, match="last slot"):
        ixp.merge_table(1, 0, 0)


def test_histogram_counts_slots_and_not_colours():
    """The measurement ``indexed.histogram`` cannot make: it counts by colour,
    so it reports both identical reds as holding every red pixel. This is what
    makes deleting an unused duplicate safe."""
    from warlock.studio.inker import indexed as ix

    plane = np.asarray([[1, 1, 3]], dtype=np.uint8)
    assert ixp.histogram(plane, len(PALETTE)) == [0, 2, 0, 1]
    by_colour = ix.histogram(ixp.materialize(plane, ixp.lut(PALETTE, 0)), PALETTE)
    assert by_colour[1] == by_colour[3] == 3, "the colour count cannot tell them apart"


# --- dither.convert_indices -------------------------------------------------


@pytest.mark.parametrize("method", dither.METHODS)
def test_every_method_agrees_with_the_rgba_conversion_on_visible_pixels(method):
    """The parity pin. ``convert`` stays byte-identical for palette-constrained
    RGB and ``convert_indices`` serves indexed mode, and they must be the same
    picture -- a second copy of the dithering arithmetic is how a document
    converted through one door and repainted through the other comes to
    disagree along an edge.

    Compared against ``convert`` on the **candidate** table (the palette without
    its transparent slot), which is the sub-table the indexed path is defined to
    dither over.
    """
    rng = np.random.default_rng(7)
    px = rng.integers(0, 256, size=(16, 24, 4), dtype=np.uint8)
    candidates = [c for i, c in enumerate(PALETTE) if i != 0]

    plane = dither.convert_indices(px, PALETTE, method, transparent=0)
    mine = ixp.materialize(plane, ixp.lut(PALETTE, 0))
    theirs = dither.convert(px, candidates, method)

    visible = px[..., 3] >= ixp.OPAQUE_THRESHOLD
    assert np.array_equal(mine[..., :3][visible], theirs[..., :3][visible])
    assert (plane[~visible] == 0).all(), "everything else is the transparent index"


def test_the_transparent_slot_is_never_dithered_into():
    """Dithering picks freely between candidates, so a transparent slot left in
    the running would scatter holes through a solid area wherever its colour
    happened to win the threshold."""
    px = np.zeros((8, 8, 4), np.uint8)
    px[..., :3] = (10, 10, 10)
    px[..., 3] = 255
    for method in dither.METHODS:
        plane = dither.convert_indices(px, PALETTE, method, transparent=0)
        assert not (plane == 0).any(), method


def test_convert_indices_is_deterministic():
    rng = np.random.default_rng(3)
    px = rng.integers(0, 256, size=(12, 12, 4), dtype=np.uint8)
    for method in dither.METHODS:
        a = dither.convert_indices(px, PALETTE, method)
        b = dither.convert_indices(px, PALETTE, method)
        assert np.array_equal(a, b), method


def test_convert_indices_refuses_an_oversized_palette_by_name():
    px = np.zeros((1, 1, 4), np.uint8)
    with pytest.raises(ValueError, match="at most 256"):
        dither.convert_indices(px, [(0, 0, 0, 255)] * 257)
