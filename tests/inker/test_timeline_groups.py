"""Group headers in the timeline: the fold, and what it hides.

``TabDoc.collapsed_groups`` was declared and never read for as long as groups
have existed, so a folder could be made and never shut. The plan is a pure
function over the document and that set for the reason ``cell_index`` is pure:
a row that should not have been drawn is a *list* mistake, and asserting it
against a list is the only way to know it holds.
"""

from __future__ import annotations

import numpy as np

from warlock.studio import inker
from warlock.studio.panes import inker_timeline


def _doc(count=4):
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    doc = inker.Document.from_pixels(pixels)
    for index in range(count - 1):
        doc.add_layer(f"L{index + 1}")
    return doc


def _kinds(plan):
    return [(entry.kind, entry.depth) for entry in plan]


def test_a_stack_with_no_groups_is_one_row_per_layer():
    doc = _doc(3)
    plan = inker_timeline.row_plan(doc)
    assert _kinds(plan) == [("track", 0)] * 3
    assert [entry.index for entry in plan] == [0, 1, 2]


def test_a_group_gets_a_header_row_above_its_members():
    doc = _doc(4)
    node = doc.group_layers([1, 2])
    plan = inker_timeline.row_plan(doc)
    assert _kinds(plan) == [
        ("track", 0),
        ("group", 0),
        ("track", 1),
        ("track", 1),
        ("track", 0),
    ]
    header = plan[1]
    assert header.uid == node.uid
    assert [entry.index for entry in plan if entry.kind == "track"] == [0, 1, 2, 3]


def test_collapsing_a_group_hides_its_members_and_keeps_its_header():
    doc = _doc(4)
    node = doc.group_layers([1, 2])
    plan = inker_timeline.row_plan(doc, collapsed={node.uid})
    assert _kinds(plan) == [("track", 0), ("group", 0), ("track", 0)]
    assert [entry.index for entry in plan if entry.kind == "track"] == [0, 3]


def test_a_nested_group_draws_its_header_one_step_deeper():
    doc = _doc(4)
    outer = doc.group_layers([1, 2])
    inner = doc.group_layers([2])
    plan = inker_timeline.row_plan(doc)
    assert _kinds(plan) == [
        ("track", 0),
        ("group", 0),
        ("track", 1),
        ("group", 1),
        ("track", 2),
        ("track", 0),
    ]
    assert plan[1].uid == outer.uid
    assert plan[3].uid == inner.uid


def test_collapsing_the_outer_group_hides_the_inner_header_too():
    doc = _doc(4)
    outer = doc.group_layers([1, 2])
    doc.group_layers([2])
    plan = inker_timeline.row_plan(doc, collapsed={outer.uid})
    assert _kinds(plan) == [("track", 0), ("group", 0), ("track", 0)]


def test_a_filtered_out_group_takes_its_header_with_it():
    """The name filter runs over layers; a folder whose every layer was
    filtered away has nothing left to be a folder of."""
    doc = _doc(4)
    doc.group_layers([1, 2])
    plan = inker_timeline.row_plan(doc, indices=[0, 3])
    assert _kinds(plan) == [("track", 0), ("track", 0)]


def test_a_group_with_one_matching_layer_keeps_its_header():
    doc = _doc(4)
    node = doc.group_layers([1, 2])
    plan = inker_timeline.row_plan(doc, indices=[1])
    assert _kinds(plan) == [("group", 0), ("track", 1)]
    assert plan[0].uid == node.uid


# --- the fold itself --------------------------------------------------------


class _Tab:
    def __init__(self):
        self.collapsed_groups: set[int] = set()


def test_the_fold_toggles_and_is_per_tab_view_state():
    tab = _Tab()
    inker_timeline.toggle_fold(tab, 7)
    assert tab.collapsed_groups == {7}
    inker_timeline.toggle_fold(tab, 7)
    assert tab.collapsed_groups == set()


def test_a_dissolved_group_leaves_no_fold_behind():
    """A uid that is not a group any more would keep a row folded shut that no
    header can reopen -- there is no header."""
    doc = _doc(4)
    node = doc.group_layers([1, 2])
    tab = _Tab()
    inker_timeline.toggle_fold(tab, node.uid)
    doc.ungroup(node.uid)
    inker_timeline.forget_folds(tab, doc)
    assert tab.collapsed_groups == set()
    assert _kinds(inker_timeline.row_plan(doc, collapsed=tab.collapsed_groups)) == [
        ("track", 0)
    ] * 4
