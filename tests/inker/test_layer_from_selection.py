"""Layer from selection, reselect, and moving a selection's edges.

The three are one file because they are one story: a selection is a region of
the canvas, and these are the three things a user reaches for once they have
drawn one -- put it back, slide it, promote it. What they share is that each of
them is *exactly one* undo step, which for the first is the whole difficulty:
the cut and the new layer must undo together or a Ctrl+Z shows a hole with
nothing above it.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker.document import Document
from warlock.studio.inker.selection import SelectionMask

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def _doc() -> Document:
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[:, :] = RED
    doc.invalidate_all()
    return doc


def _box(doc: Document, rect=(2, 2, 6, 6)) -> None:
    doc.select(SelectionMask.from_rect(doc.size, rect))


# --- layer from selection ----------------------------------------------------


def test_a_copy_leaves_the_source_alone():
    doc = _doc()
    _box(doc)
    assert doc.layer_from_selection(cut=False)

    assert len(doc.stack) == 2
    assert tuple(doc.stack[0].pixels[3, 3]) == RED
    assert tuple(doc.stack[1].pixels[3, 3]) == RED
    # Outside the selection the new layer is empty, not a rectangle of source.
    assert int(doc.stack[1].pixels[0, 0][3]) == 0


def test_a_cut_takes_the_pixels_with_it():
    doc = _doc()
    _box(doc)
    assert doc.layer_from_selection(cut=True)

    assert int(doc.stack[0].pixels[3, 3][3]) == 0
    assert tuple(doc.stack[1].pixels[3, 3]) == RED
    # And only inside the selection.
    assert int(doc.stack[0].pixels[0, 0][3]) == 255


def test_the_new_layer_lines_up_with_what_it_came_from():
    """Placed at the selection's own offset rather than at the top-left, which
    is the whole difference between this and a paste."""
    doc = _doc()
    doc.select(SelectionMask.from_rect(doc.size, (5, 5, 7, 7)))
    assert doc.layer_from_selection()

    made = doc.stack[1]
    assert int(made.pixels[5, 5][3]) == 255
    assert int(made.pixels[0, 0][3]) == 0


def test_a_feathered_selection_makes_a_feathered_layer():
    """The pixels carry the mask in their alpha, through the same
    ``_masked_alpha`` a lift and the clipboard both use."""
    doc = _doc()
    _box(doc)
    doc.feather_selection(1.5)
    assert doc.layer_from_selection()

    alpha = doc.stack[1].pixels[..., 3]
    assert 0 < int(alpha.max()) <= 255
    assert len(set(alpha.ravel().tolist())) > 2


def test_the_cut_and_the_add_are_one_undo_step():
    doc = _doc()
    _box(doc)
    head = doc.history.head

    assert doc.layer_from_selection(cut=True)
    assert doc.undo()

    assert doc.history.head == head
    assert len(doc.stack) == 1
    assert tuple(doc.stack[0].pixels[3, 3]) == RED


def test_a_copy_is_one_step_too():
    doc = _doc()
    _box(doc)
    assert doc.layer_from_selection(cut=False)
    assert doc.undo()
    assert len(doc.stack) == 1


def test_with_no_selection_it_does_nothing():
    doc = _doc()
    head = doc.history.head
    assert doc.layer_from_selection() is False
    assert doc.history.head == head
    assert len(doc.stack) == 1


def test_an_empty_selection_does_nothing():
    doc = _doc()
    doc.select(SelectionMask(np.zeros((8, 8), dtype=np.uint8)))
    assert doc.layer_from_selection() is False


def test_on_an_animated_document_the_cel_lands_on_the_current_frame_only():
    """The new pixels are a thing that happens at a moment, not a thing that is
    true for the whole clip -- ``paste_as_layer``'s rule, shared through
    ``_add_layer_edit``."""
    doc = _doc()
    doc.add_frame()
    doc.add_frame()
    doc.set_current_frame(1)
    _box(doc)

    assert doc.layer_from_selection()

    anim = doc.anim
    assert anim is not None
    track = anim.tracks[doc.stack.active_index]
    slots = [anim.cels.get((track.uid, frame.uid)) for frame in anim.frames]
    assert [cel is not None for cel in slots] == [False, True, False]


def test_a_cut_is_refused_on_a_locked_layer_but_a_copy_is_not():
    doc = _doc()
    _box(doc)
    doc.stack.active.locked = True

    assert doc.layer_from_selection(cut=True) is False
    assert len(doc.stack) == 1
    # Reading is always legal, so the copy half still works.
    assert doc.layer_from_selection(cut=False)
    assert len(doc.stack) == 2


def test_paste_as_layer_still_works_through_the_refactor():
    """``_add_layer_edit`` returns the step instead of pushing it, and this is
    the caller that used to *be* that code."""
    doc = _doc()
    _box(doc)
    assert doc.copy()
    assert doc.paste_as_layer()
    assert len(doc.stack) == 2
    assert doc.undo()
    assert len(doc.stack) == 1


def test_delete_selection_still_pushes_one_step_through_the_refactor():
    doc = _doc()
    _box(doc)
    head = doc.history.head
    assert doc.delete_selection()
    assert doc.history.head != head
    assert doc.undo()
    assert doc.history.head == head
    assert tuple(doc.stack[0].pixels[3, 3]) == RED


def test_deleting_a_selection_over_nothing_pushes_nothing():
    """The ``[]`` answer from ``_cut_selection_patch``: a step that changes
    nothing must not move the head."""
    doc = Document.blank(8, 8)
    _box(doc)
    head = doc.history.head
    assert doc.delete_selection()
    assert doc.history.head == head


# --- reselect ----------------------------------------------------------------


def test_reselect_brings_back_the_dismissed_selection():
    doc = _doc()
    _box(doc)
    wanted = doc.mask.mask.copy()
    doc.deselect()
    assert doc.mask is None

    assert doc.reselect()
    assert np.array_equal(doc.mask.mask, wanted)


def test_reselect_with_nothing_remembered_does_nothing():
    doc = _doc()
    head = doc.history.head
    assert doc.reselect() is False
    assert doc.history.head == head


def test_reselect_is_an_ordinary_undoable_step():
    doc = _doc()
    _box(doc)
    doc.deselect()
    head = doc.history.head

    assert doc.reselect()
    assert doc.history.head != head
    assert doc.undo()
    assert doc.mask is None


def test_a_stale_shaped_mask_is_refused_after_a_resize():
    """It is a per-pixel statement about a canvas that no longer exists, and
    the two honest answers are "nothing" and "guess"."""
    doc = _doc()
    _box(doc)
    doc.deselect()
    doc.resize_canvas((16, 16))

    assert doc.reselect() is False
    assert doc.mask is None


def test_undo_does_not_overwrite_what_reselect_remembers():
    """A Ctrl+Z is the user putting the selection back, so there is nothing for
    a reselect to restore -- and recording one there would throw away the mask
    they actually meant."""
    doc = _doc()
    _box(doc)
    wanted = doc.mask.mask.copy()
    doc.deselect()
    doc.select_all()
    doc.undo()  # back to no selection, via ``set_selection_mask``

    assert doc.reselect()
    assert np.array_equal(doc.mask.mask, wanted)


# --- moving a selection's edges ----------------------------------------------


def test_a_translated_mask_slides_by_whole_pixels():
    mask = SelectionMask.from_rect((8, 8), (1, 1, 3, 3))
    moved = mask.translated(2, 3)
    assert moved.bounds == (3, 4, 5, 6)


def test_what_leaves_the_canvas_does_not_come_back_round_the_other_side():
    """Zero-filled, not rolled: wrapping would put coverage exactly where the
    user dragged the selection away from."""
    mask = SelectionMask.from_rect((8, 8), (0, 0, 2, 2))
    moved = mask.translated(-4, 0)
    assert moved.bounds is None
    assert int(moved.mask.max()) == 0


def test_a_translation_keeps_soft_edges_soft():
    """Whole pixels, so nothing is resampled: a feathered edge arrives with the
    same values it left with, three pixels along."""
    mask = SelectionMask.from_rect((16, 16), (4, 4, 12, 12)).feathered(2.0)
    moved = mask.translated(3, 0)
    assert np.array_equal(moved.mask[8, 3:], mask.mask[8, :-3])
    assert len(set(moved.mask[8].tolist())) > 2


def test_moving_the_selection_leaves_the_pixels_alone():
    """The edges only. That is the whole point of the gesture: the move tool
    moves pixels, and this moves the region that names them."""
    doc = _doc()
    _box(doc)
    before = doc.stack[0].pixels.copy()

    doc.select(doc.mask.translated(1, 1))

    assert np.array_equal(doc.stack[0].pixels, before)
    assert doc.mask.bounds == (3, 3, 7, 7)


def test_a_zero_translation_pushes_nothing():
    """``select`` already refuses a step that changes nothing, and a mask-move
    that never left the press point is exactly that."""
    doc = _doc()
    _box(doc)
    head = doc.history.head
    doc.select(doc.mask.translated(0, 0))
    assert doc.history.head == head


# --- what the canvas measures ------------------------------------------------


def test_the_drag_offset_is_measured_against_the_press():
    """One function serves the ants preview and the release, so what is drawn
    while the drag runs and what ``select`` is handed cannot disagree -- and it
    is measured against the press rather than accumulated per frame, the rule
    the transform handles already follow."""
    from types import SimpleNamespace

    from warlock.studio.panes import inker_canvas

    state = SimpleNamespace(
        drag_kind="mask-move", drag_anchor=(4.0, 4.0), last_point=(6.4, 1.6)
    )
    assert inker_canvas._mask_shift(state) == (2, -2)
    assert inker_canvas._mask_shift(state, (10.0, 4.0)) == (6, 0)

    state.drag_kind = "marquee"
    assert inker_canvas._mask_shift(state) == (0, 0)


def test_the_chords_are_the_photoshop_way_round():
    """The plain chord is the *non-destructive* one. The obvious reading is the
    opposite -- plain does the whole thing, Shift does less -- which is exactly
    why this is pinned rather than left to the comment: the point of a shortcut
    is that it matches the hand that already knows it."""
    from types import SimpleNamespace

    from warlock.studio import inker_mode

    calls: list[bool] = []
    doc = SimpleNamespace(
        layer_from_selection=lambda *, cut: calls.append(cut), mask=object()
    )
    tab = SimpleNamespace(busy=False, doc=doc)
    state = SimpleNamespace(active=tab, tip=None, say=lambda *a, **k: None)
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))

    from warlock.studio import inker_ops

    # The plain chord is an op; only the shifted half is still a branch, because
    # an op carries one key and these two differ by more than a modifier.
    inker_ops.run(ctx, inker_ops.get("copy_to_layer"))
    inker_mode._ctrl_key(None, None, tab, doc, "j", None, shift=True)

    assert calls == [False, True]  # Ctrl+J copies, Ctrl+Shift+J cuts
    assert "j" in inker_mode._MUTATING_CTRL


def test_the_chords_are_refused_while_the_tab_is_busy():
    """It adds a layer and may cut pixels, so it waits for a save the way every
    other mutating chord does."""
    from types import SimpleNamespace

    from warlock.studio import inker_mode

    calls: list[bool] = []
    doc = SimpleNamespace(layer_from_selection=lambda *, cut: calls.append(cut))
    inker_mode._ctrl_key(
        None, None, SimpleNamespace(busy=True), doc, "j", None, shift=False
    )
    assert calls == []
