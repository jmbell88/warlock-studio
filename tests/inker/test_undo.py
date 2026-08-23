"""History, against a stub document.

The point of the rewrite is that a step now costs its dirty rectangle instead
of the whole canvas, so the assertions worth making are: a patch restores the
exact pixels, an edit finds its layer by uid after a reorder, and eviction
drops the oldest step rather than refusing the newest.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import undo as U
from warlock.studio.inker.layers import Layer, LayerStack


class FakeDoc:
    def __init__(self, stack: LayerStack) -> None:
        self.stack = stack
        self.selection = None
        self.dirty: list = []

    def layer_by_uid(self, uid):
        # The real document looks in the animation grid when the uid is not in
        # the current stack; a still document, which is all this stub models,
        # has nowhere else to look and so is exactly ``stack.by_uid``.
        return self.stack.by_uid(uid)

    def invalidate(self, rect, *, layer_uid=None):
        self.dirty.append(rect)

    def invalidate_all(self):
        self.dirty.append(None)

    def set_selection_mask(self, mask):
        self.selection = mask

    def restore_snapshot(self, layers, size, active, grid=None, slices=None):
        self.stack = LayerStack(list(layers), active)
        self.slices = [] if slices is None else [entry.copy() for entry in slices]


def _doc(n=2, size=(8, 8)):
    w, h = size
    layers = []
    rng = np.random.default_rng(3)
    for _ in range(n):
        layers.append(Layer(pixels=rng.integers(0, 256, (h, w, 4), dtype=np.uint8)))
    return FakeDoc(LayerStack(layers, 0))


# --- patches ----------------------------------------------------------------


def test_a_patch_restores_the_exact_pixels_it_replaced():
    doc = _doc()
    layer = doc.stack[0]
    rect = (2, 2, 6, 5)
    before = layer.pixels[2:5, 2:6].copy()
    layer.pixels[2:5, 2:6] = 0
    after = layer.pixels[2:5, 2:6].copy()
    edit = U.PatchEdit(layer.uid, rect, before, after)

    edit.undo(doc)
    assert np.array_equal(layer.pixels[2:5, 2:6], before)
    edit.redo(doc)
    assert np.array_equal(layer.pixels[2:5, 2:6], after)


def test_a_patch_costs_its_rectangle_and_not_the_canvas():
    doc = _doc(size=(512, 512))
    crop = doc.stack[0].pixels[0:16, 0:16].copy()
    edit = U.PatchEdit(doc.stack[0].uid, (0, 0, 16, 16), crop, crop.copy())
    assert edit.cost == 2 * 16 * 16 * 4
    assert edit.cost < doc.stack[0].pixels.nbytes


def test_a_patch_follows_its_layer_across_a_reorder():
    """Indices move; uids do not. Addressing by index would make an undo after
    a reorder paint the patch onto a different layer."""
    doc = _doc(3)
    layer = doc.stack[0]
    before = layer.pixels[0:2, 0:2].copy()
    layer.pixels[0:2, 0:2] = 0
    edit = U.PatchEdit(layer.uid, (0, 0, 2, 2), before, layer.pixels[0:2, 0:2].copy())

    doc.stack.move(0, 2)
    edit.undo(doc)
    assert np.array_equal(doc.stack[2].pixels[0:2, 0:2], before)


def test_a_patch_marks_only_its_own_rectangle_dirty():
    doc = _doc()
    crop = doc.stack[0].pixels[1:3, 1:3].copy()
    U.PatchEdit(doc.stack[0].uid, (1, 1, 3, 3), crop, crop).undo(doc)
    assert doc.dirty == [(1, 1, 3, 3)]


# --- structural edits -------------------------------------------------------


def test_adding_and_undoing_a_layer_returns_the_same_object_on_redo():
    doc = _doc(1)
    added = Layer.empty(8, 8)
    doc.stack.insert(1, added)
    edit = U.LayerAddEdit(1, added)
    edit.undo(doc)
    assert len(doc.stack) == 1
    edit.redo(doc)
    assert doc.stack[1] is added
    # An undone add holds the only reference to a full-canvas layer, so it has
    # to report that to the byte budget: a cost of zero let an unbounded number
    # of them sit past the ceiling, invisible to eviction.
    assert edit.cost == added.pixels.nbytes


def test_removing_a_layer_keeps_its_pixels_so_the_undo_is_exact():
    doc = _doc(2)
    gone = doc.stack[0]
    doc.stack.remove(0)
    edit = U.LayerRemoveEdit(0, gone)
    edit.undo(doc)
    assert doc.stack[0] is gone
    assert edit.cost == gone.pixels.nbytes


def test_a_move_undoes_back_to_the_original_order():
    doc = _doc(3)
    uids = [layer.uid for layer in doc.stack]
    doc.stack.move(0, 2)
    U.LayerMoveEdit(uids[0], 0, 2).undo(doc)
    assert [layer.uid for layer in doc.stack] == uids


def test_a_props_edit_sets_only_the_keys_it_names():
    doc = _doc(1)
    layer = doc.stack[0]
    edit = U.LayerPropsEdit(layer.uid, {"opacity": 1.0}, {"opacity": 0.4})
    edit.redo(doc)
    assert layer.opacity == 0.4
    assert layer.visible is True
    edit.undo(doc)
    assert layer.opacity == 1.0


# --- selection and replay ---------------------------------------------------


def test_a_selection_edit_round_trips_the_mask_through_compression():
    doc = _doc()
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 255
    edit = U.SelectionEdit(None, mask)
    edit.redo(doc)
    assert np.array_equal(doc.selection, mask)
    edit.undo(doc)
    assert doc.selection is None


def test_a_selection_mask_costs_far_less_than_it_measures():
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:400, 100:400] = 255
    assert U.SelectionEdit(None, mask).cost < mask.nbytes // 50


def test_a_replay_edit_restores_by_copy_and_redoes_by_replaying():
    doc = _doc(2)
    snapshot = [layer.copy() for layer in doc.stack]
    calls = []
    edit = U.ReplayEdit(snapshot, (8, 8), 0, lambda d: calls.append(d))
    doc.stack[0].pixels[:] = 0
    edit.undo(doc)
    assert doc.stack[0].pixels.max() > 0
    edit.redo(doc)
    assert calls == [doc]


def test_a_compound_edit_undoes_its_parts_in_reverse():
    order = []

    class Noop(U.Edit):
        def __init__(self, tag):
            self.tag = tag

        def undo(self, doc):
            order.append(("undo", self.tag))

        def redo(self, doc):
            order.append(("redo", self.tag))

    edit = U.CompoundEdit([Noop("a"), Noop("b")])
    edit.undo(None)
    edit.redo(None)
    assert order == [("undo", "b"), ("undo", "a"), ("redo", "a"), ("redo", "b")]


# --- the stack ---------------------------------------------------------------


def _patch(doc, n=1):
    crop = doc.stack[0].pixels[0:n, 0:n].copy()
    return U.PatchEdit(doc.stack[0].uid, (0, 0, n, n), crop, crop.copy())


def test_undo_and_redo_report_whether_they_did_anything():
    doc = _doc()
    stack = U.UndoStack()
    assert not stack.undo(doc)
    assert not stack.redo(doc)
    stack.push(_patch(doc))
    assert stack.undo(doc)
    assert stack.redo(doc)


def test_a_new_edit_discards_the_redo_branch():
    doc = _doc()
    stack = U.UndoStack()
    stack.push(_patch(doc))
    stack.undo(doc)
    assert stack.can_redo
    stack.push(_patch(doc))
    assert not stack.can_redo


def test_the_byte_budget_evicts_the_oldest_step_but_never_past_the_floor():
    doc = _doc(size=(64, 64))
    stack = U.UndoStack(budget=1)
    for _ in range(U.UNDO_MIN_DEPTH + 5):
        stack.push(_patch(doc, 32))
    assert len(stack) == U.UNDO_MIN_DEPTH


def test_a_document_of_dabs_still_stops_at_the_depth_cap():
    doc = _doc(size=(4, 4))
    stack = U.UndoStack()
    for _ in range(U.UNDO_MAX_DEPTH + 10):
        stack.push(_patch(doc))
    assert len(stack) == U.UNDO_MAX_DEPTH


def test_many_stroke_sized_patches_fit_where_one_old_snapshot_did_not():
    """The whole point: 2048^2 x 10 layers was 160 MiB a step. A stroke's
    rectangle is a couple of megabytes, so the same budget buys a session."""
    doc = _doc(size=(256, 256))
    stack = U.UndoStack()
    for _ in range(U.UNDO_MAX_DEPTH):
        stack.push(_patch(doc, 128))
    assert len(stack) == U.UNDO_MAX_DEPTH
    assert stack.bytes < U.UNDO_BYTES


def test_drop_discards_the_newest_step_without_reversing_it():
    doc = _doc()
    stack = U.UndoStack()
    stack.push(_patch(doc))
    stack.drop()
    assert not stack.can_undo


def test_clearing_empties_both_directions():
    doc = _doc()
    stack = U.UndoStack()
    stack.push(_patch(doc))
    stack.undo(doc)
    stack.clear()
    assert not stack.can_undo and not stack.can_redo


@pytest.mark.parametrize("edit_type", [U.PatchEdit, U.LayerAddEdit])
def test_every_edit_type_is_an_edit(edit_type):
    assert issubclass(edit_type, U.Edit)


# --- one_step ---------------------------------------------------------------


def test_one_step_leaves_a_lone_edit_unwrapped():
    """Not merely tidiness. The history panel labels a step by its class name,
    so an unwrapped step reads as what it did and a ``CompoundEdit`` of one
    reads as "compound" -- which is why every one of the nine call sites this
    replaces had to make the same choice, and why nine copies of a choice is how
    one of them comes to differ."""
    from warlock.studio.inker.undo import CompoundEdit, LayerFlagEdit, one_step

    lone = LayerFlagEdit(1, {"background": False}, {"background": True})
    assert one_step([lone]) is lone

    second = LayerFlagEdit(2, {"reference": False}, {"reference": True})
    step = one_step([lone, second])
    assert isinstance(step, CompoundEdit)
    assert list(step.edits) == [lone, second]


def test_one_step_refuses_an_empty_list():
    """A caller with nothing to push must not push: a step that changes nothing
    moves the history head and asks the user to save a gesture that did not
    happen."""
    import pytest

    from warlock.studio.inker.undo import one_step

    with pytest.raises(ValueError):
        one_step([])


def test_the_idiom_is_not_written_out_anywhere_any_more():
    import pathlib

    root = (
        pathlib.Path(__file__).resolve().parents[2]
        / "src"
        / "warlock"
        / "studio"
        / "inker"
    )
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if path.name == "undo.py":
            continue
        assert "if len(edits) == 1 else" not in source, path.name
