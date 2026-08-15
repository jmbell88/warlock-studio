"""A palette op is a colour map, and the *table* is part of the step.

Two latent bugs, both of them only reachable through the palette, and both fixed
here.

**A live selection made every palette op raise.** ``Document._map_planes`` was
written for geometry, where the marquee has to rotate with the pixels, and it
put the selection mask through the same function as the planes. ``indexed.snap``
and ``indexed.remap`` take ``(H, W, 4)`` uint8 and raise on anything else, so a
mask -- 2-D, by definition -- took the op down from the middle, with the new
table already assigned to the document and the pixels half rewritten.

**The table was in no undo step.** ``ReplayEdit`` restores pixels, and the
palette is not pixels: undoing a recolour put the old colours back into a
document still claiming the new table, and the next stroke snapped them right
back to the palette the user had just undone.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker.document import Document
from warlock.studio.inker.selection import SelectionMask
from warlock.studio.inker.undo import CompoundEdit, PaletteEdit, ReplayEdit

BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
RED = (255, 0, 0, 255)
RAMP = [BLACK, WHITE, RED]


def _doc(colour=(200, 200, 200, 255)) -> Document:
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[:, :] = colour
    return doc


def _marquee(doc: Document) -> np.ndarray:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    doc.select(SelectionMask(mask), "replace")
    assert doc.mask is not None
    return doc.mask.mask.copy()


# --- the selection is not a colour plane -------------------------------------


def test_indexing_with_a_marquee_up_neither_raises_nor_moves_the_mask():
    doc = _doc()
    before = _marquee(doc)

    assert doc.set_palette(RAMP) is True

    assert doc.mask is not None
    assert np.array_equal(doc.mask.mask, before)


def test_recolouring_a_slot_with_a_marquee_up_still_works():
    doc = _doc(BLACK)
    doc.set_palette(RAMP)
    before = _marquee(doc)

    assert doc.recolour_slot(0, (10, 20, 30, 255)) is True

    assert tuple(doc.stack.active.pixels[0, 0]) == (10, 20, 30, 255)
    assert np.array_equal(doc.mask.mask, before)


def test_removing_a_slot_with_a_marquee_up_still_works():
    doc = _doc(RED)
    doc.set_palette(RAMP)
    before = _marquee(doc)

    assert doc.remove_slot(2) is True

    assert doc.palette == [BLACK, WHITE]
    assert np.array_equal(doc.mask.mask, before)


def test_the_conversion_ignores_the_selection_it_leaves_alone():
    """The mask survives the op *and* does not narrow it: a mode change is a
    statement about the whole document, so the pixels outside the marquee are
    snapped too. Aseprite does the same, and the alternative -- a document half
    on its palette -- is not a state the mode can describe."""
    doc = _doc((250, 250, 250, 255))
    doc.stack.active.pixels[0, 0] = (5, 5, 5, 255)
    _marquee(doc)

    doc.set_palette(RAMP)

    assert tuple(doc.stack.active.pixels[0, 0]) == BLACK  # outside the marquee
    assert tuple(doc.stack.active.pixels[3, 3]) == WHITE  # inside it


def test_a_geometry_op_still_takes_its_mask_with_it():
    """The other half of the split: ``mask_fn`` defaults to the pixel function,
    so a rotate is exactly as it was."""
    doc = _doc()
    _marquee(doc)

    doc.rotate90(1)

    assert doc.mask is not None
    rotated = np.zeros((8, 8), dtype=np.uint8)
    rotated[2:6, 2:6] = 255  # the marquee is square and centred, so it maps onto itself
    assert np.array_equal(doc.mask.mask, rotated)
    assert int(doc.mask.mask.sum()) == 16 * 255


# --- the table is part of the step -------------------------------------------


def test_indexing_is_one_step_that_carries_the_table():
    doc = _doc()
    depth = len(doc.history)

    doc.set_palette(RAMP)

    assert len(doc.history) == depth + 1
    top = doc.history.top
    assert isinstance(top, CompoundEdit)
    assert [type(e) for e in top.edits] == [PaletteEdit, ReplayEdit]


def test_undoing_an_index_puts_the_document_back_to_not_indexed():
    doc = _doc()
    doc.set_palette(RAMP)

    doc.undo()

    assert doc.palette is None
    assert doc.is_indexed is False
    assert tuple(doc.stack.active.pixels[0, 0]) == (200, 200, 200, 255)


def test_redoing_an_index_brings_the_table_back_with_the_pixels():
    doc = _doc()
    doc.set_palette(RAMP)
    doc.undo()

    doc.redo()

    assert doc.palette == RAMP
    assert tuple(doc.stack.active.pixels[0, 0]) == WHITE


def test_undoing_a_recolour_restores_the_slot_and_not_only_the_pixels():
    doc = _doc(BLACK)
    doc.set_palette(RAMP)
    doc.recolour_slot(0, (10, 20, 30, 255))

    doc.undo()

    assert doc.palette == RAMP
    assert tuple(doc.stack.active.pixels[0, 0]) == BLACK


def test_the_next_write_after_an_undo_snaps_to_the_restored_table():
    """The bug, stated as the user meets it: without the table in the step,
    undoing a slot edit left the document claiming the *new* palette, and one
    stroke put every pixel straight back onto the colour just undone."""
    doc = _doc(BLACK)
    doc.set_palette(RAMP)
    doc.recolour_slot(0, (10, 20, 30, 255))
    doc.undo()

    doc.write_colour((0, 0, 2, 2), (2, 2, 2, 255), np.ones((2, 2), dtype=np.float32))

    assert tuple(doc.stack.active.pixels[0, 0]) == BLACK


def test_undoing_a_removed_slot_brings_the_swatch_back():
    doc = _doc(RED)
    doc.set_palette(RAMP)
    doc.remove_slot(2)

    doc.undo()

    assert doc.palette == RAMP
    assert tuple(doc.stack.active.pixels[0, 0]) == RED


def test_the_history_holds_its_own_copy_of_the_table():
    """A step that held the live list would be edited by the next ``add_slot``
    -- the table is one mutable object the document keeps appending to."""
    doc = _doc()
    doc.set_palette(RAMP)
    doc.add_slot((7, 7, 7, 255))

    doc.undo()

    assert doc.palette is None


def test_a_table_only_edit_still_pushes_no_step():
    """Order and additions repaint nothing, so they stay off the stack -- the
    rule ``move_slot`` has always followed."""
    doc = _doc()
    doc.set_palette(RAMP)
    depth = len(doc.history)

    doc.add_slot((7, 7, 7, 255))
    doc.move_slot(0, 1)
    doc.set_palette(None)

    assert len(doc.history) == depth
