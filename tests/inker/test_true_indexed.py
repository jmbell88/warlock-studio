"""True indexed mode: index planes as the record, and what that buys.

The whole file is an argument that **identity** is the feature. Palette slots 1
and 3 of :data:`PALETTE` are the same red throughout, and almost every test here
would pass in palette-constrained RGB except for the assertion that tells the
two slots apart -- which is the one that matters.

``check_materialized`` is called after every operation rather than once at the
end. Drift between an index plane and its pixels is silent: the document looks
right until it is saved, undone or reordered, and then the indices win.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import Document
from warlock.studio.inker.undo import ColorStateEdit, CompoundEdit, IndexPatchEdit, IndexRemapEdit

BLACK = (0, 0, 0, 255)
RED = (200, 20, 20, 255)
GREEN = (20, 200, 20, 255)
BLUE = (20, 20, 200, 255)

#: Slot 0 is the transparent index *and* carries an opaque black, which is how
#: Aseprite writes one. Slots 1 and 3 are the same red.
PALETTE = [BLACK, RED, GREEN, RED]


def _doc(width: int = 4, height: int = 4) -> Document:
    doc = Document.blank(width, height)
    doc.stack.active.pixels[:, :] = RED
    doc.convert_to_indexed(PALETTE)
    doc.check_materialized()
    return doc


def _mark_duplicate(doc: Document, x: int = 0, y: int = 0) -> None:
    """Put one pixel in the *other* red slot, by hand.

    By hand and not by painting, deliberately: this is setting up the state a
    ``.aseprite`` import or a ``prefer``-ed stroke produces, and the point of
    the tests below is what happens to it afterwards.
    """
    layer = doc.stack.active
    layer.indices[y, x] = 3
    doc._rematerialize(layer)


# --- entering and leaving ---------------------------------------------------


def test_converting_installs_planes_a_mode_and_a_table_in_one_step():
    doc = Document.blank(4, 4)
    doc.stack.active.pixels[:, :] = RED
    head = doc.history.head

    assert doc.convert_to_indexed(PALETTE) is True

    assert doc.color_mode == "indexed" and doc.is_indexed and not doc.is_palette_locked
    assert doc.palette == PALETTE and doc.transparent_index == 0
    assert doc.stack.active.indices is not None
    assert (doc.stack.active.indices == 1).all()
    doc.check_materialized()
    assert doc.history.head != head

    doc.undo()
    assert doc.color_mode == "rgb" and doc.palette is None
    assert doc.stack.active.indices is None
    assert tuple(doc.stack.active.pixels[0, 0]) == RED


def test_a_conversion_is_one_undo_step_across_every_layer_and_frame():
    """What the user did was change the document's mode, once."""
    doc = Document.blank(4, 4)
    doc.stack.active.pixels[:, :] = RED
    doc.add_layer()
    doc.stack.active.pixels[:, :] = GREEN
    doc.add_frame()
    head = doc.history.head

    doc.convert_to_indexed(PALETTE)
    doc.check_materialized()
    assert all(layer.indices is not None for layer in doc.anim.unique_cel_layers())

    doc.undo()
    assert doc.history.head == head
    assert all(layer.indices is None for layer in doc.anim.unique_cel_layers())


def test_leaving_indexed_mode_keeps_the_pixels_and_the_table():
    """Lossless by construction -- the pixels *are* the materialisation. What it
    loses is identity, which is the whole content of the warning the UI gives."""
    doc = _doc()
    _mark_duplicate(doc)
    before = doc.stack.active.pixels.copy()

    assert doc.convert_to_rgb() is True
    assert doc.color_mode == "rgb"
    assert doc.stack.active.indices is None
    assert doc.palette == PALETTE, "the table the user authored survives the mode change"
    assert doc.is_palette_locked, "which makes it palette-constrained RGB"
    assert np.array_equal(doc.stack.active.pixels, before)

    doc.undo()
    assert doc.is_indexed and doc.stack.active.indices[0, 0] == 3


def test_converting_to_what_it_already_is_pushes_nothing():
    doc = _doc()
    head = doc.history.head
    assert doc.convert_to_indexed(PALETTE) is False
    assert doc.history.head == head


def test_a_dither_is_let_through_the_no_op_door():
    """Re-running one is a real request -- it is how a user compares matrices --
    and it does move pixels. ``convert_to_palette``'s rule, restated."""
    doc = _doc()
    head = doc.history.head
    assert doc.convert_to_indexed(PALETTE, "bayer4") is True
    assert doc.history.head != head


def test_an_oversized_palette_is_refused_by_name():
    doc = Document.blank(2, 2)
    with pytest.raises(ValueError, match="at most 256"):
        doc.convert_to_indexed([BLACK] * 257)
    assert doc.color_mode == "rgb"


def test_a_transparent_index_outside_the_table_is_refused_by_name():
    doc = Document.blank(2, 2)
    with pytest.raises(ValueError, match="must name a slot"):
        doc.convert_to_indexed(PALETTE, transparent=9)


# --- identity through the palette ops ---------------------------------------


def test_recolouring_a_duplicate_slot_reaches_its_pixels_and_no_others():
    """**The test this whole wave exists for.** In palette-constrained RGB the
    two reds are one colour and this repaints both; here they are two slots."""
    doc = _doc()
    _mark_duplicate(doc)

    assert doc.recolour_slot(3, BLUE) is True
    doc.check_materialized()
    assert tuple(doc.stack.active.pixels[0, 0]) == BLUE
    assert tuple(doc.stack.active.pixels[0, 1]) == RED
    assert (doc.stack.active.indices == np.array([[3, 1, 1, 1]] * 1 + [[1] * 4] * 3)).all()


def test_a_recolour_moves_no_index_at_all():
    """Which is what makes it instant: one row of the lookup table changes and
    the whole clip repaints, in both directions."""
    doc = _doc()
    _mark_duplicate(doc)
    planes = doc.stack.active.indices.copy()

    doc.recolour_slot(1, BLUE)
    assert np.array_equal(doc.stack.active.indices, planes)
    doc.undo()
    assert np.array_equal(doc.stack.active.indices, planes)
    doc.check_materialized()


def test_a_recolour_is_one_colour_state_edit_and_nothing_else():
    doc = _doc()
    doc.recolour_slot(1, BLUE)
    assert isinstance(doc.history.top, ColorStateEdit)


def test_moving_a_slot_permutes_every_plane_exactly_and_undoes_exactly():
    doc = _doc()
    _mark_duplicate(doc)
    before = doc.stack.active.indices.copy()

    assert doc.move_slot(0, 2) is True
    doc.check_materialized()
    assert isinstance(doc.history.top, CompoundEdit)
    assert isinstance(doc.history.top.edits[1], IndexRemapEdit)
    # Slot 3 did not move, so the marked pixel is still in it -- and still
    # distinguishable from the identical red now sitting at slot 0.
    assert doc.stack.active.indices[0, 0] == 3
    assert doc.stack.active.indices[0, 1] == 0

    doc.undo()
    assert np.array_equal(doc.stack.active.indices, before)
    assert doc.palette == PALETTE
    doc.check_materialized()


def test_the_transparent_index_moves_with_its_slot():
    """A position, not a colour. A reorder that left it behind would point it at
    whatever slid into that position -- turning one swatch into holes across
    every frame at once, with no visible cause."""
    doc = _doc()
    assert doc.transparent_index == 0
    doc.move_slot(0, 2)
    assert doc.transparent_index == 2
    assert doc.palette[2] == BLACK
    doc.check_materialized()
    doc.undo()
    assert doc.transparent_index == 0


def test_sorting_keeps_two_identical_swatches_apart():
    doc = _doc()
    _mark_duplicate(doc)
    marked = doc.palette[3]

    assert doc.sort_palette("luma") is True
    doc.check_materialized()
    # The pixel still points at *its own* slot, wherever the sort put it.
    slot = int(doc.stack.active.indices[0, 0])
    assert doc.palette[slot] == marked
    assert slot != int(doc.stack.active.indices[0, 1])


def test_a_sort_in_indexed_mode_is_undoable():
    """``move_slot``'s argument at table scale: it rewrites every plane in the
    document, so calling it "presentation" would be the wart."""
    doc = _doc()
    head = doc.history.head
    doc.sort_palette("luma")
    assert doc.history.head != head
    doc.undo()
    assert doc.palette == PALETTE and doc.history.head == head


def test_removing_a_slot_merges_into_its_nearest_survivor_and_undoes_exactly():
    doc = _doc()
    _mark_duplicate(doc)
    before = doc.stack.active.indices.copy()

    assert doc.remove_slot(1) is True
    doc.check_materialized()
    assert len(doc.palette) == 3
    # Every red pixel is now in the one surviving red slot.
    slot = int(doc.stack.active.indices[0, 0])
    assert doc.palette[slot] == RED
    assert (doc.stack.active.indices == slot).all()

    doc.undo()
    assert np.array_equal(doc.stack.active.indices, before), "the merge undoes to exact indices"
    assert doc.palette == PALETTE
    doc.check_materialized()


def test_a_merge_never_lands_on_the_transparent_slot():
    """It would turn a solid area into a hole for no reason the user could see,
    and it is the one slot whose colour is not a claim about anything."""
    doc = Document.blank(3, 3)
    doc.stack.active.pixels[:, :] = (1, 1, 1, 255)
    # Slot 0 is a near-black transparent index; slot 1 is the near-black the
    # pixels are painted in, so nearest-by-colour would pick slot 0.
    doc.convert_to_indexed([(0, 0, 0, 255), (1, 1, 1, 255), GREEN])
    doc.remove_slot(1)
    doc.check_materialized()
    assert not (doc.stack.active.indices == doc.transparent_index).any()


def test_the_transparent_slot_cannot_be_removed():
    doc = _doc()
    with pytest.raises(ValueError, match="transparent slot cannot be removed"):
        doc.remove_slot(0)


def test_removing_a_slot_below_the_transparent_one_shifts_it_down():
    doc = _doc()
    doc.set_transparent_index(2)
    assert doc.transparent_index == 2
    doc.remove_slot(1)
    assert doc.transparent_index == 1
    assert doc.palette[1] == GREEN
    doc.check_materialized()


def test_adding_a_slot_is_undoable_in_indexed_mode():
    doc = _doc()
    head = doc.history.head
    assert doc.add_slot(BLUE) is True
    assert doc.history.head != head and doc.palette[-1] == BLUE
    doc.undo()
    assert doc.palette == PALETTE


def test_a_ramp_shifts_every_slot_above_the_insertion_point():
    doc = _doc()
    _mark_duplicate(doc)

    assert doc.insert_ramp(1, 2, 2) is True
    doc.check_materialized()
    # The marked pixel was in slot 3; two colours went in after slot 1.
    assert doc.stack.active.indices[0, 0] == 5
    assert doc.palette[5] == RED
    assert doc.stack.active.indices[0, 1] == 1

    doc.undo()
    assert doc.stack.active.indices[0, 0] == 3 and doc.palette == PALETTE


def test_the_palette_cap_is_refused_by_name_rather_than_truncated():
    doc = Document.blank(2, 2)
    doc.convert_to_indexed([(i, i, i, 255) for i in range(256)])
    with pytest.raises(ValueError, match="at most 256"):
        doc.add_slot(BLUE)
    assert len(doc.palette) == 256


def test_setting_the_transparent_index_moves_the_holes():
    doc = _doc()
    doc.stack.active.indices[0, 0] = 2
    doc._rematerialize(doc.stack.active)

    assert doc.set_transparent_index(2) is True
    doc.check_materialized()
    assert tuple(doc.stack.active.pixels[0, 0]) == (0, 0, 0, 0), "slot 2 is now the hole"
    assert tuple(doc.stack.active.pixels[0, 1]) == RED
    doc.undo()
    assert tuple(doc.stack.active.pixels[0, 0]) == GREEN


def test_usage_counts_slots_rather_than_colours():
    """The measurement the colour histogram cannot make. A duplicate always
    looks used to it, so a user could never tell which one to delete."""
    doc = _doc()
    _mark_duplicate(doc)
    counts = doc.palette_usage()
    assert counts == [0, 15, 0, 1]


# --- the funnel -------------------------------------------------------------


def test_a_stroke_lands_on_the_slot_the_user_picked():
    """``paint_slot`` is what closes the last visible gap: without it, painting
    with slot 3 in a table where slot 1 is the same red silently records slot 1,
    and the next recolour of slot 3 leaves the stroke behind."""
    doc = _doc()
    doc.paint_slot = 3
    doc.begin_stroke((1, 1), RED, size=1, nib="pixel")
    doc.end_stroke()
    doc.check_materialized()
    assert doc.stack.active.indices[1, 1] == 3
    assert doc.stack.active.indices[0, 0] == 1


def test_a_stroke_with_no_preference_falls_to_the_lowest_duplicate():
    doc = _doc()
    doc.paint_slot = None
    doc.begin_stroke((1, 1), RED, size=1, nib="pixel")
    doc.end_stroke()
    assert doc.stack.active.indices[1, 1] == 1


def test_a_stroke_pushes_an_index_patch_and_undoes_to_exact_indices():
    doc = _doc()
    _mark_duplicate(doc)
    before = doc.stack.active.indices.copy()

    doc.paint_slot = 2
    doc.begin_stroke((1, 1), GREEN, size=1, nib="pixel")
    doc.end_stroke()
    assert isinstance(doc.history.top, IndexPatchEdit)
    doc.check_materialized()

    doc.undo()
    assert np.array_equal(doc.stack.active.indices, before)
    doc.check_materialized()


def test_a_soft_nib_is_legal_and_thresholds_to_the_shape_that_was_drawn():
    """Soft brushes stay legal at the engine: the funnel thresholds them, so no
    invariant can break. What the user gets is a hard edge, which is what the
    mode is for."""
    doc = _doc()
    doc.begin_stroke((2, 2), GREEN, size=4, hardness=0.0, nib="soft")
    doc.end_stroke()
    doc.check_materialized()
    alpha = doc.stack.active.pixels[..., 3]
    assert set(np.unique(alpha).tolist()) <= {0, 255}, "no partial alpha survives the funnel"


def test_a_write_that_thresholds_back_to_where_it_started_pushes_nothing():
    """The no-op rule, reached by different arithmetic than the RGBA path's: a
    gesture whose every pixel resolves to the slot it was already in changed
    nothing, so it must not move the history head."""
    doc = _doc()
    head = doc.history.head
    doc.paint_slot = 1
    doc.begin_stroke((1, 1), RED, size=1, nib="pixel")
    doc.end_stroke()
    assert doc.history.head == head
    doc.check_materialized()


def test_a_no_op_write_on_a_fresh_frame_takes_its_cel_back_out():
    doc = _doc()
    doc.add_frame()
    head = doc.history.head
    doc.paint_slot = 0
    doc.begin_stroke((1, 1), (0, 0, 0, 0), size=1, nib="pixel")
    doc.end_stroke()
    assert doc.history.head == head


def test_a_fresh_cel_is_a_plane_of_the_transparent_index_not_of_zero():
    """Only the same slot by coincidence. On a document whose transparent index
    is 2, autovivifying a plane of zeroes paints a canvas of solid slot-0
    colour the instant a stroke starts."""
    doc = _doc()
    doc.set_transparent_index(2)
    doc.add_frame()
    doc.begin_stroke((1, 1), RED, size=1, nib="pixel")
    doc.end_stroke()
    doc.check_materialized()
    cel = doc.stack.active
    assert cel.indices is not None
    assert cel.indices[3, 3] == 2
    assert tuple(cel.pixels[3, 3]) == (0, 0, 0, 0)


def test_erasing_reaches_the_transparent_index_and_nothing_else_does():
    """The one entrance. An eraser cuts alpha, the funnel reads alpha, and the
    hole it produces is the transparent slot -- while a *painted* black, which
    is the colour slot 0 holds, resolves somewhere else entirely."""
    doc = _doc()
    doc.begin_stroke((1, 1), RED, size=2, nib="pixel", mode="erase")
    doc.end_stroke()
    doc.check_materialized()
    assert doc.stack.active.indices[1, 1] == 0
    assert tuple(doc.stack.active.pixels[1, 1]) == (0, 0, 0, 0)

    doc.begin_stroke((3, 3), BLACK, size=1, nib="pixel")
    doc.end_stroke()
    doc.check_materialized()
    assert doc.stack.active.indices[3, 3] != 0


# --- geometry ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("run", "where"),
    [
        (lambda d: d.flip("horizontal"), (0, 3)),
        (lambda d: d.flip("vertical"), (3, 0)),
        (lambda d: d.rotate90(1), (3, 0)),
        (lambda d: d.crop((0, 0, 3, 3)), (0, 0)),
    ],
)
def test_exact_geometry_permutes_indices_rather_than_re_inferring_them(run, where):
    """Flip, quarter turn and crop move pixels without inventing any, so the
    index plane takes the same permutation and duplicate slots survive. Had they
    been re-resolved from the colours instead, slot 3 would come back as slot 1
    and the difference would be invisible until the next recolour."""
    doc = _doc()
    _mark_duplicate(doc)

    run(doc)
    doc.check_materialized()
    assert doc.stack.active.indices[where] == 3

    doc.undo()
    assert doc.stack.active.indices[0, 0] == 3
    doc.check_materialized()


def test_a_canvas_resize_fills_with_the_transparent_index():
    doc = _doc()
    doc.set_transparent_index(2)
    doc.resize_canvas((6, 6), (1, 1))
    doc.check_materialized()
    assert doc.stack.active.indices[0, 0] == 2
    assert tuple(doc.stack.active.pixels[0, 0]) == (0, 0, 0, 0)


def test_a_smooth_scale_re_resolves_and_says_so():
    """The one stated place indices come back from colours. A resample invents
    colours that were never in the table, so there is no permutation to apply --
    and duplicate identity is genuinely lost, which is why this is the only op
    allowed to do it."""
    doc = _doc(8, 8)
    _mark_duplicate(doc)
    doc.scale((4, 4), resample="smooth")
    doc.check_materialized()
    assert doc.stack.active.indices is not None
    assert not (doc.stack.active.indices == 3).any()


def test_nearest_scale_also_re_resolves_but_stays_on_the_table():
    doc = _doc(8, 8)
    doc.scale((4, 4), resample="nearest")
    doc.check_materialized()
    assert (doc.stack.active.indices == 1).all()


# --- links ------------------------------------------------------------------


def test_a_linked_cel_shares_one_index_plane():
    """For free, and that is the point: a link is the same ``Layer`` object in
    two grid slots, so there is exactly one plane to share and no machinery."""
    doc = _doc()
    doc.add_frame(link=True)
    cels = list(doc.anim.unique_cel_layers())
    assert len(cels) == 1

    doc.paint_slot = 2
    doc.begin_stroke((1, 1), GREEN, size=1, nib="pixel")
    doc.end_stroke()
    doc.check_materialized()

    doc.set_current_frame(0)
    assert doc.stack.active.indices[1, 1] == 2


def test_a_whole_document_op_touches_a_linked_plane_once():
    doc = _doc()
    doc.add_frame(link=True)
    _mark_duplicate(doc)
    doc.flip("horizontal")
    doc.check_materialized()
    assert doc.stack.active.indices[0, 3] == 3, "flipped once, not twice"


# --- the negative control ---------------------------------------------------


def test_an_rgb_document_is_untouched_by_any_of_this():
    """The standing control for the whole wave. A document that never enters a
    new mode must behave, byte for byte, as it did before these fields existed.
    """
    doc = Document.blank(4, 4)
    doc.begin_stroke((1, 1), RED, size=2)
    doc.end_stroke()
    before = doc.stack.active.pixels.copy()
    head = doc.history.head

    assert doc.color_mode == "rgb"
    assert doc.stack.active.indices is None
    assert not doc.is_indexed and not doc.is_palette_locked
    doc.check_materialized()  # a no-op on an RGB document

    doc.undo()
    doc.redo()
    assert np.array_equal(doc.stack.active.pixels, before)
    assert doc.history.head == head


def test_palette_constrained_rgb_keeps_its_old_step_shapes():
    """A slot move stays free and unrecorded there; only indexed mode charges
    for it, and only because it moves index data."""
    from warlock.studio.inker.undo import PaletteEdit

    doc = Document.blank(4, 4)
    doc.stack.active.pixels[:, :] = RED
    doc.set_palette([BLACK, RED, GREEN])
    assert doc.is_palette_locked and not doc.is_indexed

    head = doc.history.head
    assert doc.move_slot(0, 2) is True
    assert doc.history.head == head, "no step, exactly as before"

    doc.recolour_slot(0, BLUE)
    assert isinstance(doc.history.top, CompoundEdit)
    assert isinstance(doc.history.top.edits[0], PaletteEdit)


def test_check_materialized_catches_drift():
    """The guard proves it can fail. A test-only assert that cannot detect the
    thing it is for is worse than none: it reads as coverage."""
    doc = _doc()
    doc.stack.active.pixels[0, 0] = BLUE
    with pytest.raises(AssertionError, match="drifted from its index plane"):
        doc.check_materialized()


def test_the_undo_budget_sees_an_index_plane():
    """An edit that under-reports its size is one the byte budget cannot evict."""
    from warlock.studio.inker.undo import _plane_bytes

    doc = _doc(8, 8)
    layer = doc.stack.active
    assert layer.plane_bytes == layer.pixels.nbytes + layer.indices.nbytes
    assert _plane_bytes(layer) == layer.plane_bytes


def test_a_layer_copy_owns_its_own_index_plane():
    doc = _doc()
    copy = doc.stack.active.copy()
    copy.indices[0, 0] = 2
    assert doc.stack.active.indices[0, 0] == 1


def test_a_mismatched_index_plane_is_refused_at_construction():
    from warlock.studio.inker.layers import Layer

    with pytest.raises(ValueError, match="the size of the layer"):
        Layer(pixels=np.zeros((4, 4, 4), np.uint8), indices=np.zeros((2, 2), np.uint8))
    with pytest.raises(ValueError, match=r"\(H, W\) uint8"):
        Layer(pixels=np.zeros((4, 4, 4), np.uint8), indices=np.zeros((4, 4), np.int32))


def test_applying_an_index_patch_to_a_planeless_layer_is_refused_by_name():
    """A half-applied history walk is worse than a stopped one, and the guard
    exists to say which happened."""
    doc = _doc()
    edit = IndexPatchEdit(
        doc.stack.active.uid, (0, 0, 2, 2), np.zeros((2, 2), np.uint8), np.ones((2, 2), np.uint8)
    )
    doc.stack.active.indices = None
    with pytest.raises(ValueError, match="no index plane"):
        edit.redo(doc)
