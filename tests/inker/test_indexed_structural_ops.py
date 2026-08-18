"""Structural ops on a truly indexed document: the plane follows every write.

The bug class this file pins: an op that writes ``pixels`` directly and records
a raw ``PatchEdit`` leaves ``layer.indices`` describing the picture from before
it -- silent until the document is saved, undone or rematerialised, and then
the indices win and the deleted (or merged, or matted) pixels come back. And an
op that *mints* a ``Layer`` inside a still-indexed document without a plane
opens a gap ``check_materialized`` cannot see into: the layer is skipped, and
the ORA writer stores RGBA in an archive that claims indexed.

So every test here makes the same three assertions the funnel already earns for
tool writes: ``check_materialized()`` passes after the op, the ORA round trip
carries the post-op picture, and undo restores exactly.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker import Document
from warlock.studio.inker.ora import read_ora, write_ora
from warlock.studio.inker.selection import SelectionMask

BLACK = (0, 0, 0, 255)
RED = (200, 20, 20, 255)
GREEN = (20, 200, 20, 255)
BLUE = (20, 20, 200, 255)

#: Slot 0 is the transparent index; slots 1 and 3 are the same red, because
#: identity surviving is what the mode is for.
PALETTE = [BLACK, RED, GREEN, RED]


def _doc(width: int = 4, height: int = 4) -> Document:
    doc = Document.blank(width, height)
    doc.stack.active.pixels[:, :] = RED
    doc.convert_to_indexed(PALETTE)
    doc.check_materialized()
    return doc


def _mask(doc: Document, box: tuple[int, int, int, int] = (0, 0, 2, 2)) -> SelectionMask:
    width, height = doc.size
    mask = np.zeros((height, width), dtype=np.uint8)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = 255
    return SelectionMask(mask)


def _roundtrip(doc: Document, tmp_path) -> Document:
    path = tmp_path / "structural.ora"
    write_ora(doc, path)
    back = read_ora(path)
    back.check_materialized()
    return back


# --- F1: selection delete and lift ------------------------------------------


def test_deleting_a_selection_reaches_the_index_plane(tmp_path):
    doc = _doc()
    before = doc.stack.active.indices.copy()
    doc.select(_mask(doc))

    assert doc.delete_selection() is True
    doc.check_materialized()
    assert (doc.stack.active.indices[0:2, 0:2] == 0).all(), "deleted pixels are the hole"
    assert (doc.stack.active.indices[2:, :] == 1).all()

    back = _roundtrip(doc, tmp_path)
    assert np.array_equal(back.stack.active.indices, doc.stack.active.indices), (
        "the save records the deletion rather than resurrecting it"
    )

    doc.undo()
    assert np.array_equal(doc.stack.active.indices, before)
    doc.check_materialized()


def test_a_rematerialisation_does_not_resurrect_a_deletion():
    """The on-screen half of the same bug: any palette op rematerialises every
    plane, and a stale one painted the deleted pixels straight back."""
    doc = _doc()
    doc.select(_mask(doc))
    doc.delete_selection()

    doc._rematerialize_all()
    assert (doc.stack.active.pixels[0:2, 0:2, 3] == 0).all()


def test_a_lift_cuts_the_index_plane_and_cancel_restores_it_exactly():
    doc = _doc()
    before = doc.stack.active.indices.copy()
    doc.select(_mask(doc))

    assert doc.lift() is True
    doc.check_materialized()
    assert (doc.stack.active.indices[0:2, 0:2] == 0).all(), "the hole is cut in the plane"

    assert doc.cancel_floating() is True
    assert np.array_equal(doc.stack.active.indices, before)
    doc.check_materialized()


def test_a_lifted_move_commits_and_undoes_to_exact_indices(tmp_path):
    doc = _doc()
    before = doc.stack.active.indices.copy()
    doc.select(_mask(doc))
    doc.lift()
    # One step diagonally, so the landing overlaps the hole: the refilled
    # corner is a real change for the commit to record. (Landing red on red
    # is the funnel's no-op and pushes nothing.)
    doc.move_floating(1, 1)

    assert doc.commit_floating() is True
    doc.check_materialized()
    assert doc.stack.active.indices[0, 0] == 0
    assert doc.stack.active.indices[0, 1] == 0
    assert doc.stack.active.indices[1, 0] == 0
    assert doc.stack.active.indices[1, 1] == 1, "the landing refilled the hole's corner"
    assert (doc.stack.active.indices[2:, :] == 1).all()

    back = _roundtrip(doc, tmp_path)
    assert np.array_equal(back.stack.active.indices, doc.stack.active.indices)

    doc.undo()  # the commit
    doc.undo()  # the lift's cut
    assert np.array_equal(doc.stack.active.indices, before)
    doc.check_materialized()


def test_a_cut_promotion_keeps_the_plane_honest():
    """``layer_from_selection(cut=True)`` folds ``_cut_selection_patch`` into a
    compound, so it rides the same fix -- and one Ctrl+Z reverses the pair."""
    doc = _doc()
    before = doc.stack.active.indices.copy()
    doc.select(_mask(doc))

    assert doc.layer_from_selection(cut=True) is True
    doc.check_materialized()
    assert len(doc.stack) == 2
    assert (doc.stack[0].indices[0:2, 0:2] == 0).all()
    assert doc.stack[1].indices is not None, "the promoted layer is born with a plane"

    doc.undo()
    assert len(doc.stack) == 1
    assert np.array_equal(doc.stack.active.indices, before)
    doc.check_materialized()


# --- F2: merge down ---------------------------------------------------------


def test_merge_down_re_resolves_the_lower_plane(tmp_path):
    doc = _doc()
    lower_before = doc.stack[0].indices.copy()
    doc.add_layer()
    doc.paint_slot = 2
    doc.begin_stroke((3, 3), GREEN, size=1, nib="pixel")
    doc.end_stroke()
    doc.check_materialized()

    assert doc.merge_down() is True
    doc.check_materialized()
    assert len(doc.stack) == 1
    assert doc.stack.active.indices[3, 3] == 2, "the merged stroke is in the plane"
    assert doc.stack.active.indices[0, 0] == 1

    back = _roundtrip(doc, tmp_path)
    assert np.array_equal(back.stack.active.indices, doc.stack.active.indices), (
        "the save carries the merged artwork, not the pre-merge indices"
    )

    doc.undo()
    assert len(doc.stack) == 2
    assert np.array_equal(doc.stack[0].indices, lower_before)
    assert doc.stack[1].indices[3, 3] == 2
    doc.check_materialized()


def test_a_track_merge_mints_cels_with_planes(tmp_path):
    doc = _doc()
    doc.add_frame()
    doc.add_layer()  # a second track, on an animated document
    doc.paint_slot = 2
    doc.begin_stroke((3, 3), GREEN, size=1, nib="pixel")
    doc.end_stroke()
    doc.check_materialized()

    assert doc.merge_down() is True
    doc.check_materialized()
    assert all(cel.indices is not None for cel in doc.anim.unique_cel_layers())
    assert doc.stack.active.indices[3, 3] == 2

    back = _roundtrip(doc, tmp_path)
    assert all(cel.indices is not None for cel in back.anim.unique_cel_layers())

    doc.undo()
    doc.check_materialized()
    assert len(doc.anim.tracks) == 2


# --- F3: a live float across a palette insertion ----------------------------


def test_insert_ramp_commits_the_float_before_the_table_moves():
    """Committed after the reassignment, the buffer resolved into new-table
    numbering and was then pushed through a remap sized to the old table --
    clamped, so the floated pixels landed as the wrong colour, silently
    self-consistent. Every sibling op commits the float first; now this one
    does too."""
    doc = Document.blank(4, 4)
    doc.stack.active.pixels[:, :] = GREEN
    doc.convert_to_indexed([BLACK, RED, GREEN, BLUE])
    original = doc.stack.active.indices.copy()
    doc.select(_mask(doc))
    doc.lift()
    doc.move_floating(2, 2)

    assert doc.insert_ramp(0, 1, 4) is True
    doc.check_materialized()
    green_slot = doc.palette.index(GREEN)
    assert doc.stack.active.indices[3, 3] == green_slot, (
        "the floated pixels landed in their own colour's slot"
    )
    assert tuple(doc.stack.active.pixels[3, 3]) == GREEN

    doc.undo()  # the ramp's remap
    doc.undo()  # the float's commit
    doc.undo()  # the lift's cut
    assert np.array_equal(doc.stack.active.indices, original)
    doc.check_materialized()


# --- F4: the matte ----------------------------------------------------------


def test_apply_matte_reaches_the_index_plane(tmp_path):
    doc = _doc()
    before = doc.stack.active.indices.copy()
    alpha = np.full((4, 4), 255, dtype=np.uint8)
    alpha[0, :] = 0

    assert doc.apply_matte(alpha) is True
    doc.check_materialized()
    assert (doc.stack.active.indices[0, :] == 0).all(), "matted pixels are the hole"
    assert (doc.stack.active.indices[1:, :] == 1).all()

    back = _roundtrip(doc, tmp_path)
    assert np.array_equal(back.stack.active.indices, doc.stack.active.indices)

    doc.undo()
    assert np.array_equal(doc.stack.active.indices, before)
    doc.check_materialized()


def test_a_matte_that_thresholds_back_to_itself_pushes_nothing():
    """The funnel's no-op rule, inherited: a weight that leaves every pixel over
    the threshold resolves to the same slots, so there is no step -- and the
    materialisation is put back rather than left half-faded."""
    doc = _doc()
    head = doc.history.head
    alpha = np.full((4, 4), 200, dtype=np.uint8)  # 255 * 200/255 = 200 >= 128

    assert doc.apply_matte(alpha) is False
    assert doc.history.head == head
    doc.check_materialized()


# --- F5: minted layers get planes -------------------------------------------


def test_flatten_mints_a_layer_with_a_plane(tmp_path):
    doc = _doc()
    doc.add_layer()
    doc.paint_slot = 2
    doc.begin_stroke((3, 3), GREEN, size=1, nib="pixel")
    doc.end_stroke()

    doc.flatten_layers()
    doc.check_materialized()
    assert len(doc.stack) == 1
    assert doc.stack.active.indices is not None, "the flattened layer is born with a plane"
    assert doc.stack.active.indices[3, 3] == 2

    back = _roundtrip(doc, tmp_path)
    assert back.stack.active.indices is not None
    assert np.array_equal(back.stack.active.indices, doc.stack.active.indices)

    doc.undo()
    assert len(doc.stack) == 2
    doc.check_materialized()

    doc.redo()
    doc.check_materialized()
    assert doc.stack.active.indices is not None, "replay mints the plane again"


def test_flatten_grid_mints_planes_for_every_frame():
    doc = _doc()
    doc.add_frame()
    doc.add_layer()
    doc.paint_slot = 2
    doc.begin_stroke((3, 3), GREEN, size=1, nib="pixel")
    doc.end_stroke()

    doc.flatten_layers()
    doc.check_materialized()
    assert len(doc.anim.tracks) == 1
    assert all(cel.indices is not None for cel in doc.anim.unique_cel_layers())

    doc.undo()
    doc.check_materialized()
    assert len(doc.anim.tracks) == 2


def test_paste_as_layer_snaps_onto_the_table_and_gets_a_plane(tmp_path):
    doc = _doc()
    pasted = np.zeros((2, 2, 4), dtype=np.uint8)
    pasted[:, :] = (25, 190, 25, 255)  # near-green, off the table

    assert doc.paste_as_layer(pasted) is True
    doc.check_materialized()
    layer = doc.stack.active
    assert layer.indices is not None, "the pasted layer is born with a plane"
    assert layer.indices[0, 0] == 2, "off-table colours resolve like any commit"
    assert tuple(layer.pixels[0, 0]) == GREEN

    back = _roundtrip(doc, tmp_path)
    assert back.stack[1].indices is not None
    assert np.array_equal(back.stack[1].indices, layer.indices)

    doc.undo()
    assert len(doc.stack) == 1
    doc.check_materialized()


def test_add_layer_mints_a_plane_of_the_transparent_index(tmp_path):
    """``_ensure_cel_for``'s fresh-cel rule, on the still branch: without a
    plane, every stroke on the new layer falls through the funnel's RGBA path
    and the layer lives outside the mode it is in."""
    doc = _doc()
    doc.set_transparent_index(2)
    doc.add_layer()
    layer = doc.stack.active
    assert layer.indices is not None
    assert (layer.indices == 2).all(), "the transparent index, not zero"
    doc.check_materialized()

    doc.paint_slot = 3
    doc.begin_stroke((3, 3), RED, size=1, nib="pixel")
    doc.end_stroke()
    doc.check_materialized()
    assert layer.indices[3, 3] == 3, "the stroke went through the indexed funnel"

    back = _roundtrip(doc, tmp_path)
    assert back.stack[1].indices is not None
    assert back.stack[1].indices[3, 3] == 3


def test_palette_usage_counts_a_minted_layer_per_slot():
    """The quiet second symptom: a planeless layer sent ``palette_usage`` to the
    colour histogram, where the two red slots are indistinguishable and each
    reports every pixel of either. A flatten re-resolves (divergence 21's
    accepted cost -- the duplicate collapses to the lowest slot), but what it
    mints is a *plane*, so the count is per-slot and sums to the canvas."""
    doc = _doc()
    doc.add_layer()
    doc.paint_slot = 2
    doc.begin_stroke((3, 3), GREEN, size=1, nib="pixel")
    doc.end_stroke()

    doc.flatten_layers()
    doc.check_materialized()
    assert doc.palette_usage() == [0, 15, 1, 0], "exact, and it sums to the 16 pixels"
