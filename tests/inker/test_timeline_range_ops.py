"""Layer verbs over a timeline *range* rather than one row.

``extend_range`` was written, tested and wired to nothing: Shift-click did not
reach it, and every verb on the row menu addressed ``[index]``. So a range was
selectable and then acted on one row at a time -- which is worse than no range,
because the highlight says otherwise.

One gesture is one undo step here for the reason it is everywhere else in this
package: deleting a block of eight layers must not cost eight Ctrl+Z.
"""

from __future__ import annotations

import numpy as np

from warlock.studio import inker


def _doc(count=5):
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    doc = inker.Document.from_pixels(pixels)
    for index in range(count - 1):
        doc.add_layer(f"L{index + 1}")
    return doc


def _names(doc):
    return [layer.name for layer in doc.stack]


# --- the primitive ----------------------------------------------------------


def test_a_run_of_steps_collapses_into_one():
    doc = _doc(4)
    before = len(doc.history)
    with doc.one_gesture():
        doc.set_layer_props(1, name="a")
        doc.set_layer_props(2, name="b")
    assert len(doc.history) == before + 1
    doc.undo()
    assert _names(doc)[1:3] == ["L1", "L2"]


def test_a_gesture_that_changed_nothing_pushes_nothing():
    doc = _doc(2)
    before = len(doc.history)
    with doc.one_gesture():
        pass
    assert len(doc.history) == before


def test_a_gesture_with_one_step_in_it_stays_that_step():
    """A lone ``CompoundEdit`` around one edit reads as "compound" in the
    history panel where the edit itself reads as what it did."""
    from warlock.studio.undo import CompoundEdit

    doc = _doc(2)
    with doc.one_gesture():
        doc.set_layer_props(1, name="a")
    assert not isinstance(doc.history.top, CompoundEdit)


# --- the range verbs --------------------------------------------------------


def test_removing_a_run_of_layers_is_one_step():
    doc = _doc(5)
    assert doc.remove_layers([1, 2, 3])
    assert _names(doc) == ["Background", "L4"]
    doc.undo()
    assert _names(doc) == ["Background", "L1", "L2", "L3", "L4"]


def test_removing_every_layer_is_refused():
    doc = _doc(3)
    assert not doc.remove_layers([0, 1, 2])
    assert len(doc.stack) == 3


def test_duplicating_a_run_is_one_step_and_keeps_the_order():
    doc = _doc(3)
    assert doc.duplicate_layers([1, 2])
    assert len(doc.stack) == 5
    doc.undo()
    assert _names(doc) == ["Background", "L1", "L2"]


def test_merging_a_run_leaves_one_layer_in_one_step():
    doc = _doc(5)
    assert doc.merge_range(1, 3)
    assert _names(doc) == ["Background", "L1", "L4"]
    doc.undo()
    assert _names(doc) == ["Background", "L1", "L2", "L3", "L4"]


def test_merging_a_single_row_range_is_a_plain_merge_down():
    doc = _doc(3)
    assert doc.merge_range(2, 2)
    assert len(doc.stack) == 2


def test_merging_a_range_that_starts_at_the_background_is_refused():
    """There is nothing under row 0 for the run to land on."""
    doc = _doc(3)
    assert not doc.merge_range(0, 1)
    assert len(doc.stack) == 3


# --- what a clicked verb acts on -------------------------------------------


class _Tab:
    def __init__(self, doc, range_sel=None):
        self.doc = doc
        self.range_sel = range_sel


def test_a_verb_acts_on_one_row_when_there_is_no_range():
    from warlock.studio.panes import inker_timeline

    doc = _doc(4)
    assert inker_timeline.row_targets(_Tab(doc), doc, 2) == [2]


def test_a_verb_acts_on_the_whole_block_when_the_click_is_inside_it():
    from warlock.studio.panes import inker_timeline

    doc = _doc(5)
    doc.ensure_animation()
    tab = _Tab(doc, (1, 3, 0, 0))
    assert inker_timeline.row_targets(tab, doc, 2) == [1, 2, 3]


def test_a_click_outside_the_block_acts_on_that_row_alone():
    from warlock.studio.panes import inker_timeline

    doc = _doc(5)
    doc.ensure_animation()
    tab = _Tab(doc, (1, 3, 0, 0))
    assert inker_timeline.row_targets(tab, doc, 4) == [4]
