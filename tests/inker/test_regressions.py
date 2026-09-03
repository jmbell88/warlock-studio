"""Regressions in the paint engine's history and composite cache.

Each test here pins one defect that was reproducible against the shipped
package. They are grouped by the invariant they defend rather than by the
module they touch, because every one of them is a case where two modules
disagreed about who owns a piece of state -- uids, the redo stack, or the
cached composite of the layers below the active one.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker.document import Document
from warlock.studio.inker.selection import SelectionMask
from warlock.studio.inker.undo import PatchEdit


def _doc(width: int = 16, height: int = 16) -> Document:
    """An opaque grey canvas: alpha 255 everywhere, so a lift's alpha-cut and
    its restoration are both visible in channel 3."""
    doc = Document.blank(width, height)
    doc.stack.active.pixels[..., :3] = 10
    doc.stack.active.pixels[..., 3] = 255
    doc.invalidate_all()
    return doc


def _paint(doc: Document, value: int, rect=(2, 2, 6, 6)) -> None:
    """One patch-shaped edit on the active layer, pushed as one undo step."""
    x0, y0, x1, y1 = rect
    layer = doc.stack.active
    before = layer.pixels[y0:y1, x0:x1].copy()
    layer.pixels[y0:y1, x0:x1, :3] = value
    doc._commit_patch(layer, rect, before)


# -- layer identity across a whole-canvas op ---------------------------------


def test_a_whole_canvas_undo_keeps_every_layer_uid():
    doc = _doc()
    before = [layer.uid for layer in doc.stack]
    doc.flip("horizontal")
    doc.undo()
    assert [layer.uid for layer in doc.stack] == before


def test_undoing_past_a_flip_still_finds_the_layer_it_painted():
    """The reproduced crash: the flip's undo re-made the stack with fresh uids,
    so the stroke's patch underneath it addressed a layer that no longer
    existed and the second Ctrl+Z raised KeyError."""
    doc = _doc()
    _paint(doc, 200)
    doc.flip("horizontal")

    assert doc.undo() is True  # the flip
    assert doc.undo() is True  # the stroke -- used to raise KeyError
    assert int(doc.stack.active.pixels[3, 3, 0]) == 10


def test_a_replayed_flatten_keeps_one_stable_uid():
    """Redo replays the op rather than storing a copy, so a flatten that minted
    a new uid each replay would strand any patch made above it."""
    doc = _doc()
    doc.add_layer()
    doc.flatten_layers()
    first = doc.stack[0].uid
    doc.undo()
    doc.redo()
    assert doc.stack[0].uid == first


def test_a_snapshot_restore_does_not_alias_the_snapshot_pixels():
    """Restoring must copy: undo, paint, undo again has to give back the
    original pixels, not the ones the second edit wrote into the snapshot."""
    doc = _doc()
    doc.flip("horizontal")
    doc.undo()
    _paint(doc, 99)
    doc.undo()
    assert int(doc.stack.active.pixels[3, 3, 0]) == 10


# -- steps that change nothing are not steps ---------------------------------


def test_reselecting_the_same_region_pushes_nothing():
    """Dirty is a comparison against ``history.head``, so a step that changes
    nothing makes a saved document ask to be saved again and spends a Ctrl+Z
    doing nothing. The both-None case was guarded; an identical *non*-empty
    mask -- a marquee redrawn over itself, Select All twice, feather 0 -- was
    not."""
    doc = _doc()
    rect = SelectionMask.from_rect(doc.size, (2, 2, 6, 6))
    doc.select(rect)
    depth = len(doc.history)

    doc.select(SelectionMask.from_rect(doc.size, (2, 2, 6, 6)))
    doc.select_all()
    doc.select_all()

    assert len(doc.history) == depth + 1, "only the select-all is a real change"


# -- the floating buffer's half of history -----------------------------------


def test_cancelling_a_paste_leaves_earlier_history_alone():
    """A pasted float never pushed an edit, so undoing one popped whatever the
    user did before it."""
    doc = _doc()
    _paint(doc, 200)
    doc.select(SelectionMask.from_rect(doc.size, (0, 0, 4, 4)))
    doc.copy()
    doc.select(None)

    assert doc.paste((8, 8)) is True
    assert doc.cancel_floating() is True
    assert doc.floating is None
    assert int(doc.stack.active.pixels[3, 3, 0]) == 200


def test_one_ctrl_z_over_a_lifted_float_undoes_one_step():
    """``undo`` cancelled the float (itself an undo of the lift) and then undid
    again, so a single Ctrl+Z ate two steps."""
    doc = _doc()
    _paint(doc, 200)
    doc.select(SelectionMask.from_rect(doc.size, (2, 2, 6, 6)))
    doc.lift()

    assert doc.undo() is True
    assert doc.floating is None
    # The lift is reversed, and the paint under it survives.
    assert int(doc.stack.active.pixels[3, 3, 3]) == 255
    assert int(doc.stack.active.pixels[3, 3, 0]) == 200


def test_cancelling_a_lift_cannot_be_redone_into_an_empty_cut():
    """Esc after Ctrl+T put the lift's alpha-cut back on the redo stack with no
    float left to restore, which erased the region outright."""
    doc = _doc()
    doc.select(SelectionMask.from_rect(doc.size, (2, 2, 6, 6)))
    doc.lift()
    doc.cancel_floating()

    doc.redo()
    assert int(doc.stack.active.pixels[3, 3, 3]) == 255


def test_cancelling_a_lift_reverses_the_lift_and_not_whatever_is_newest():
    """``lifted`` said *that* a history entry existed, not which one, and
    ``UndoStack.undo`` pops the top -- so any step pushed while the buffer
    floated was reversed instead. Nothing forces a commit first: every
    selection op pushes one.

    The cost was total: the lifted pixels were dropped, the alpha-cut stayed,
    and ``redoable=False`` made both unrecoverable.
    """
    doc = _doc()
    _paint(doc, 200)
    doc.select(SelectionMask.from_rect(doc.size, (2, 2, 6, 6)))
    doc.lift()
    assert int(doc.stack.active.pixels[3, 3, 3]) == 0  # the hole is cut

    # A selection op while the buffer floats -- pushes a SelectionEdit on top
    # of the lift's PatchEdit.
    doc.invert_selection()

    assert doc.cancel_floating() is True
    # The lift is what got reversed: the pixels are back, colour and all.
    assert int(doc.stack.active.pixels[3, 3, 3]) == 255
    assert int(doc.stack.active.pixels[3, 3, 0]) == 200


def test_cancelling_a_lift_leaves_the_step_pushed_over_it_alone():
    """The other half: reversing the lift out of order must not also swallow
    the edit that was sitting on top of it."""
    doc = _doc()
    doc.select(SelectionMask.from_rect(doc.size, (2, 2, 6, 6)))
    doc.lift()
    doc.select_all()
    selected = doc.mask

    doc.cancel_floating()

    assert doc.mask is selected  # the select-all still stands
    doc.undo()
    assert doc.mask is not selected  # ...and is still the next thing to undo


def test_a_paste_clears_the_redo_stack():
    doc = _doc()
    _paint(doc, 200)
    doc.select(SelectionMask.from_rect(doc.size, (0, 0, 4, 4)))
    doc.copy()
    doc.undo()  # the selection
    doc.undo()  # the paint

    doc.paste((8, 8))
    doc.cancel_floating()
    assert doc.redo() is False
    assert int(doc.stack.active.pixels[3, 3, 0]) == 10


def test_subtracting_from_nothing_does_not_spend_an_undo_step():
    """It correctly does nothing to the mask, but pushed a SelectionEdit whose
    before and after are both None anyway -- so an Alt-drag on an unselected
    canvas marked a saved document dirty and made the next Ctrl+Z a no-op."""
    doc = _doc()
    head, steps = doc.history.head, len(doc.history)

    doc.select(SelectionMask.from_rect(doc.size, (2, 2, 6, 6)), op="subtract")

    assert doc.mask is None
    assert (doc.history.head, len(doc.history)) == (head, steps)


def test_a_selection_that_genuinely_changes_still_records_a_step():
    """The other direction, so the early return is not simply "never push"."""
    doc = _doc()
    steps = len(doc.history)

    doc.select(SelectionMask.from_rect(doc.size, (2, 2, 6, 6)))

    assert doc.mask is not None
    assert len(doc.history) == steps + 1


# -- the below-cache ----------------------------------------------------------


def test_a_patch_under_the_active_layer_repairs_the_composite():
    """``_below`` is the cached composite of everything under the active layer.
    An undo landing there wrote real pixels but reused the stale cache, so the
    document -- and every save derived from it -- kept showing the old ones."""
    doc = _doc()
    bottom = doc.stack.active
    doc.add_layer()
    doc.set_active_layer(0)
    _paint(doc, 200)
    # Switching up caches _below as "the painted layer 0", which the undo below
    # then makes stale.
    doc.set_active_layer(1)

    assert doc.undo() is True
    assert int(bottom.pixels[3, 3, 0]) == 10
    assert int(doc.composite[3, 3, 0]) == 10
    assert int(doc.flatten()[3, 3, 0]) == 10


# -- undo cost accounting -----------------------------------------------------


def test_a_patch_owns_its_pixels_rather_than_viewing_a_whole_canvas():
    """``end_stroke`` sliced the full pre-stroke copy, so a 4 KiB dab reported
    4 KiB while pinning the entire 16 MiB base array alive."""
    doc = _doc(64, 64)
    doc.begin_stroke((10.0, 10.0), (255, 0, 0, 255), size=4)
    doc.stroke_to((12.0, 12.0))
    doc.end_stroke()

    edit = doc.history._done[-1]
    assert isinstance(edit, PatchEdit)
    assert edit.before.base is None
    assert edit.after.base is None
    assert edit.cost == edit.before.nbytes + edit.after.nbytes


# -- selection algebra --------------------------------------------------------


def test_subtracting_from_no_selection_selects_nothing():
    doc = _doc()
    doc.select(SelectionMask.from_rect(doc.size, (2, 2, 6, 6)), "subtract")
    assert doc.mask is None


def test_intersecting_with_no_selection_selects_nothing():
    doc = _doc()
    doc.select(SelectionMask.from_rect(doc.size, (2, 2, 6, 6)), "intersect")
    assert doc.mask is None


def test_adding_to_no_selection_adopts_the_mask():
    doc = _doc()
    doc.select(SelectionMask.from_rect(doc.size, (2, 2, 6, 6)), "add")
    assert doc.mask is not None
    assert doc.mask.bounds == (2, 2, 6, 6)


# -- layer properties ---------------------------------------------------------


def test_an_opacity_change_is_one_undo_step():
    """The panel mutated the layer live and then asked the document to record a
    change from the value it had just written, so nothing was ever pushed."""
    doc = _doc()
    layer = doc.stack.active
    depth = len(doc.history)

    layer.opacity = 0.25  # what the slider does while dragging
    assert doc.set_layer_props(opacity=0.25, was={"opacity": 1.0}) is True
    assert len(doc.history) == depth + 1

    doc.undo()
    assert layer.opacity == 1.0


def test_an_opacity_change_moves_the_saved_head():
    doc = _doc()
    head = doc.history.head
    doc.stack.active.opacity = 0.5
    doc.set_layer_props(opacity=0.5, was={"opacity": 1.0})
    assert doc.history.head != head


def test_set_layer_props_still_ignores_a_change_to_the_same_value():
    doc = _doc()
    depth = len(doc.history)
    assert doc.set_layer_props(opacity=1.0) is False
    assert len(doc.history) == depth


# -- marquee rounding ---------------------------------------------------------


def test_a_right_to_left_marquee_covers_the_same_pixels_as_a_left_to_right_one():
    from warlock.studio.panes import inker_canvas

    forward = inker_canvas.marquee_rect((2.3, 2.3), (10.7, 10.7))
    backward = inker_canvas.marquee_rect((10.7, 10.7), (2.3, 2.3))
    assert forward == backward
    assert forward == (2, 2, 11, 11)


def test_a_click_inside_a_select_tool_is_a_click_and_not_a_one_pixel_drag():
    """``marquee_rect`` floors one corner and ceils the other -- correctly, so
    a drag rounds outward whichever way it was drawn -- which makes a press and
    release at the same point a 1x1 rectangle rather than an empty one. The
    "click deselects" branch was therefore unreachable, and every stray click
    left a one-pixel selection nothing could be painted outside of."""
    from warlock.studio.panes import inker_canvas

    assert inker_canvas.is_click((6.3, 4.2), (6.9, 4.8)) is True
    assert inker_canvas.marquee_rect((6.3, 4.2), (6.9, 4.8)) == (6, 4, 7, 5), (
        "which is why the rectangle alone cannot answer it"
    )
    assert inker_canvas.is_click((6.3, 4.2), (7.1, 4.2)) is False
    assert inker_canvas.is_click((6.3, 4.2), (6.3, 5.0)) is False


def test_an_overlay_line_lands_inside_one_device_pixel():
    """imgui centres a one-pixel line on the coordinate, so a grid line at a
    whole number covers half of two columns and is antialiased across both --
    a 16 px grid over pixel art came out as a grey haze."""
    from warlock.studio.panes import inker_canvas

    assert inker_canvas.crisp((40.0, 12.0)) == (40.5, 12.5)
    assert inker_canvas.crisp((40.7, 12.2)) == (40.5, 12.5)
    assert inker_canvas.crisp((-1.0, -0.5)) == (-1.0 + 0.5, -1.0 + 0.5)


def test_the_composite_matches_a_from_scratch_flatten_after_an_undo():
    """A belt-and-braces check on the cache: whatever path got us here, the
    cached composite must equal one computed with no cache at all."""
    doc = _doc()
    doc.add_layer()
    doc.set_active_layer(0)
    _paint(doc, 200)
    doc.set_active_layer(1)
    doc.undo()

    fresh = np.zeros_like(doc.composite)
    width, height = doc.size
    from warlock.studio.inker import composite as cp

    fresh[:] = cp.to_uint8(doc.stack.composite_region((0, 0, width, height)))
    assert np.array_equal(doc.composite, fresh)
