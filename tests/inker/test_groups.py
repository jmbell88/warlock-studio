"""Layer groups: the tree, the fold, and the invariant that makes it sound.

The flat stack stays authoritative for paint order and the tree hangs beside
it, so almost everything here is really one assertion made from several angles:
**a group's leaves are contiguous in stack order and spans nest**. That is what
lets a group be a slice of the painter's order rather than a scattering of it,
and every op that could break it is exercised below -- create, delete, reorder,
merge, move in, move out.

``groups.check()`` is the pure statement of it and is used as the oracle
throughout, because a test that asserted the *outcome* of each op would be
re-deriving the invariant once per op and would miss the op nobody thought of.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import groups as gp
from warlock.studio.inker.document import Document
from warlock.studio.inker.selection import SelectionMask

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def _doc(layers: int = 4) -> Document:
    doc = Document.blank(8, 8)
    doc.stack[0].name = "L0"
    doc.stack[0].pixels[:, :] = RED
    for i in range(1, layers):
        doc.add_layer(f"L{i}")
        doc.stack[i].pixels[0:2, 0:2] = BLUE
    doc.invalidate_all()
    return doc


def _ok(doc: Document) -> None:
    gp.check(doc.groups, doc.group_of, doc.member_uids())


def _names(doc: Document) -> list[str]:
    return [layer.name for layer in doc.stack]


# --- the pure tree -----------------------------------------------------------


def test_resolve_folds_visibility_opacity_and_the_lock():
    outer = gp.GroupNode(name="outer", opacity=0.5)
    inner = gp.GroupNode(name="inner", opacity=0.5, locked=True)
    groups = {outer.uid: outer, inner.uid: inner}
    group_of = {inner.uid: outer.uid, 7: inner.uid}

    visible, opacity, locked = gp.resolve(groups, group_of, 7)
    assert visible is True
    assert opacity == pytest.approx(0.25)
    assert locked is True

    outer.visible = False
    assert gp.resolve(groups, group_of, 7)[0] is False


def test_a_member_of_nothing_resolves_to_the_neutral_triple():
    assert gp.resolve({}, {}, 7) == (True, 1.0, False)


def test_a_dangling_parent_is_skipped_rather_than_refused():
    """An ORA can arrive with one, and the layer is still a layer."""
    assert gp.resolve({}, {7: 99}, 7) == (True, 1.0, False)


def test_ancestry_does_not_hang_on_a_cycle():
    """Called on every composite of every layer, so a malformed tree from a
    file must cost a wrong picture at worst, never a hang."""
    assert gp.ancestry({1: 2, 2: 1}, 1) == [2, 1]


def test_check_reports_a_cycle_a_non_contiguous_span_and_an_empty_group():
    a, b = gp.GroupNode(uid=101), gp.GroupNode(uid=102)
    groups = {a.uid: a, b.uid: b}

    with pytest.raises(ValueError, match="inside itself"):
        gp.check({a.uid: a}, {a.uid: a.uid, 1: a.uid}, [1])
    with pytest.raises(ValueError, match="not contiguous"):
        gp.check({a.uid: a}, {1: a.uid, 3: a.uid}, [1, 2, 3])
    with pytest.raises(ValueError, match="empty"):
        gp.check(groups, {1: a.uid}, [1])
    with pytest.raises(ValueError, match="which is gone"):
        gp.check({}, {1: 999}, [1])


def test_a_snapshot_of_the_tree_does_not_follow_later_edits():
    node = gp.GroupNode(name="before")
    nodes, _ = gp.copy_tree({node.uid: node}, {})
    node.name = "after"
    assert nodes[node.uid].name == "before"


# --- creating and dissolving -------------------------------------------------


def test_grouping_a_run_of_layers():
    doc = _doc()
    node = doc.group_layers([1, 2], name="Ink")
    assert node is not None
    _ok(doc)
    assert gp.leaves_of(doc.group_of, doc.member_uids(), node.uid) == [
        doc.stack[1].uid,
        doc.stack[2].uid,
    ]


def test_a_run_that_is_not_contiguous_is_refused():
    """Moving layers is a visible change to the painter's order, and a grouping
    gesture that silently reordered somebody's stack would be worse than one
    that says no."""
    doc = _doc()
    head = doc.history.head
    assert doc.group_layers([0, 2]) is None
    assert doc.history.head == head
    assert not doc.groups


def test_grouping_inside_a_group_nests():
    doc = _doc(4)
    outer = doc.group_layers([1, 2, 3], name="outer")
    inner = doc.group_layers([2, 3], name="inner")
    assert inner is not None
    _ok(doc)
    assert doc.group_of[inner.uid] == outer.uid
    assert gp.ancestry(doc.group_of, doc.stack[2].uid) == [inner.uid, outer.uid]


def test_a_run_that_straddles_a_boundary_is_refused():
    doc = _doc(4)
    doc.group_layers([2, 3], name="top")
    head = doc.history.head
    assert doc.group_layers([1, 2]) is None
    assert doc.history.head == head


def test_grouping_is_one_undo_step():
    doc = _doc()
    node = doc.group_layers([1, 2])
    assert doc.undo()
    assert node.uid not in doc.groups
    assert not doc.group_of
    assert doc.redo()
    assert node.uid in doc.groups
    _ok(doc)


def test_ungrouping_leaves_the_layers_where_they_are():
    doc = _doc()
    node = doc.group_layers([1, 2])
    order = _names(doc)
    assert doc.ungroup(node.uid)
    assert _names(doc) == order
    assert not doc.groups
    assert doc.undo()
    assert node.uid in doc.groups
    _ok(doc)


def test_dissolving_a_nested_group_reparents_rather_than_orphans():
    """Its layers belong to the folder it was inside, not to the root."""
    doc = _doc(4)
    outer = doc.group_layers([1, 2, 3], name="outer")
    inner = doc.group_layers([2, 3], name="inner")
    assert doc.ungroup(inner.uid)
    _ok(doc)
    assert doc.group_of[doc.stack[2].uid] == outer.uid


def test_group_properties_are_one_step_and_a_no_op_pushes_nothing():
    doc = _doc()
    node = doc.group_layers([1, 2])
    head = doc.history.head
    assert doc.set_group_props(node.uid, name="Renamed", opacity=0.5)
    assert doc.history.head != head
    assert doc.set_group_props(node.uid, name="Renamed", opacity=0.5) is False
    assert doc.undo()
    assert (node.name, node.opacity) == ("Group", 1.0)


# --- the fold ----------------------------------------------------------------


def test_hiding_a_group_hides_what_is_in_it():
    doc = _doc(2)
    doc.stack[1].pixels[:, :] = BLUE
    node = doc.group_layers([1])
    doc.invalidate_all()
    assert tuple(doc.composite[4, 4]) == BLUE

    doc.set_group_props(node.uid, visible=False)
    assert tuple(doc.composite[4, 4]) == RED


def test_group_opacity_multiplies_the_layers_own():
    doc = _doc(2)
    doc.stack[1].pixels[:, :] = BLUE
    doc.stack[1].opacity = 0.5
    node = doc.group_layers([1])
    doc.set_group_props(node.uid, opacity=0.5)
    # Pass-through: the layer is composited at 25% against the red beneath it.
    assert int(doc.composite[4, 4][2]) == pytest.approx(64, abs=2)


def test_a_group_lock_refuses_writes_to_what_is_inside_it():
    doc = _doc(2)
    node = doc.group_layers([1])
    doc.stack.active_index = 1
    assert doc.write_locked() is False

    doc.set_group_props(node.uid, locked=True)
    assert doc.write_locked() is True
    assert doc.write_colour((0, 0, 4, 4), BLUE, np.ones((4, 4), dtype=np.float32)) is False


def test_the_fold_is_absent_on_a_document_with_no_groups():
    """None rather than a list of neutral pairs, so the composite takes its
    original path by identity rather than multiplying by 1.0 per layer."""
    doc = _doc()
    assert doc.group_fold() is None
    assert doc.stack.group_fold is None
    doc.group_layers([1])
    assert doc.stack.group_fold is not None


# --- the placeholder trap ----------------------------------------------------


def test_hiding_a_group_hides_an_empty_cel_too():
    """The trap this whole feature could have shipped with. A materialised
    empty slot is a placeholder carrying a per-slot uid, not the track's, so
    keying membership on ``layer.uid`` would un-group every empty cel -- and
    nothing would look wrong until somebody drew on one, at which point a
    drawing would appear inside a hidden group."""
    doc = _doc(2)
    doc.add_frame(copy=True)
    assert doc.clear_cel(1, 1)  # frame 2's upper slot is empty, its lower is not
    doc.set_current_frame(0)
    node = doc.group_layers([1])
    doc.set_group_props(node.uid, visible=False)
    doc.set_current_frame(1)

    fold = doc.stack.group_fold
    assert fold is not None
    # Row 1 is a placeholder on this frame, and it is still inside the group.
    assert doc.anim.is_placeholder(doc.stack[1])
    assert fold[1][0] is False

    # And a stroke drawn there stays hidden, rather than appearing out of a
    # folder the user has closed.
    doc.stack.active_index = 1
    doc.write_colour((0, 0, 8, 8), BLUE, np.ones((8, 8), dtype=np.float32))
    assert tuple(doc.composite[4, 4]) == RED


def test_membership_is_keyed_on_the_track_not_the_cel():
    doc = _doc(2)
    doc.add_frame()
    node = doc.group_layers([1])
    assert doc.group_of[doc.anim.tracks[1].uid] == node.uid
    _ok(doc)


# --- the ops that could break contiguity -------------------------------------


def test_a_new_layer_joins_the_group_the_active_one_is_in():
    doc = _doc(4)
    node = doc.group_layers([1, 2], name="Ink")
    doc.stack.active_index = 1
    doc.add_layer("New")
    _ok(doc)
    assert doc.group_of[doc.stack[2].uid] == node.uid


def test_a_pasted_layer_joins_the_group_the_active_one_is_in():
    """``_add_layer_edit`` inserts above the active row exactly as
    ``add_layer`` does, so a paste made while working inside a folder used to
    land membership-less in the *middle* of that folder's span -- which is the
    contiguity invariant being false with nothing to say so."""
    doc = _doc(4)
    node = doc.group_layers([1, 2], name="Ink")
    doc.stack.active_index = 1
    doc.select(SelectionMask.from_rect(doc.size, (0, 0, 4, 4)))
    assert doc.copy()

    assert doc.paste_as_layer()

    _ok(doc)
    assert doc.group_of[doc.stack[2].uid] == node.uid


def test_a_layer_from_selection_joins_the_group_it_came_from():
    doc = _doc(4)
    node = doc.group_layers([1, 2], name="Ink")
    doc.stack.active_index = 1
    doc.select(SelectionMask.from_rect(doc.size, (0, 0, 4, 4)))

    assert doc.layer_from_selection(cut=False)

    _ok(doc)
    assert doc.group_of[doc.stack[2].uid] == node.uid


def test_a_cut_layer_from_selection_inside_a_group_is_still_one_step():
    doc = _doc(4)
    doc.group_layers([1, 2], name="Ink")
    doc.stack.active_index = 1
    doc.select(SelectionMask.from_rect(doc.size, (0, 0, 4, 4)))
    head = doc.history.head

    assert doc.layer_from_selection(cut=True)
    _ok(doc)

    assert doc.undo()
    assert doc.history.head == head
    assert len(doc.stack) == 4
    _ok(doc)


def test_a_duplicate_joins_its_sources_group():
    doc = _doc(3)
    node = doc.group_layers([1], name="Ink")
    doc.duplicate_layer(1)
    _ok(doc)
    assert doc.group_of[doc.stack[2].uid] == node.uid


def test_deleting_the_last_member_dissolves_the_group():
    """Empty groups are disallowed: one has no span, composites nothing, and
    would sit in the panel as a folder that cannot be refilled."""
    doc = _doc(3)
    node = doc.group_layers([1], name="Ink")
    doc.stack.active_index = 1
    assert doc.remove_layer(1)
    assert node.uid not in doc.groups
    _ok(doc)

    assert doc.undo()
    assert node.uid in doc.groups
    assert doc.group_of[doc.stack[1].uid] == node.uid
    _ok(doc)


def test_deleting_one_of_two_members_keeps_the_group():
    doc = _doc(4)
    node = doc.group_layers([1, 2], name="Ink")
    assert doc.remove_layer(2)
    assert node.uid in doc.groups
    _ok(doc)


def test_a_reorder_across_a_boundary_takes_the_membership_with_it():
    doc = _doc(4)
    node = doc.group_layers([2, 3], name="Top")
    # Row 0 is outside; move it up into the middle of the group's span.
    assert doc.move_layer(0, 2)
    _ok(doc)
    assert doc.group_of.get(doc.stack[2].uid) == node.uid


def test_a_reorder_out_of_a_group_drops_the_membership():
    doc = _doc(4)
    doc.group_layers([2, 3], name="Top")
    moved = doc.stack[2].uid
    assert doc.move_layer(2, 0)
    _ok(doc)
    assert moved not in doc.group_of


def test_a_reorder_is_still_one_undo_step():
    doc = _doc(4)
    doc.group_layers([2, 3], name="Top")
    order = _names(doc)
    head = doc.history.head
    assert doc.move_layer(0, 2)
    assert doc.undo()
    assert doc.history.head == head
    assert _names(doc) == order
    _ok(doc)


def test_merging_across_a_boundary_keeps_the_lowers_membership():
    doc = _doc(4)
    node = doc.group_layers([1, 2], name="Ink")
    # Row 3 is outside the group; merge it down onto row 2, which is inside.
    doc.stack.active_index = 3
    assert doc.merge_down(3)
    _ok(doc)
    assert doc.group_of.get(doc.stack[2].uid) == node.uid


def test_merging_the_last_member_out_of_a_group_dissolves_it():
    doc = _doc(3)
    node = doc.group_layers([2], name="Top")
    assert doc.merge_down(2)
    assert node.uid not in doc.groups
    _ok(doc)


def test_flatten_takes_the_whole_tree_with_it_and_undo_puts_it_back():
    doc = _doc(3)
    node = doc.group_layers([1, 2], name="Ink")
    doc.flatten_layers()
    assert not doc.groups
    assert not doc.group_of

    assert doc.undo()
    assert node.uid in doc.groups
    _ok(doc)


# --- moving in and out -------------------------------------------------------


def test_moving_a_layer_into_a_group_moves_it_in_the_stack_too():
    doc = _doc(4)
    node = doc.group_layers([2, 3], name="Top")
    moved = doc.stack[0].uid
    assert doc.move_into_group(0, node.uid)
    _ok(doc)
    # It landed at the top of the span and joined the group.
    assert doc.stack[3].uid == moved
    assert doc.group_of[moved] == node.uid


def test_moving_in_and_the_move_are_one_undo_step():
    doc = _doc(4)
    node = doc.group_layers([2, 3], name="Top")
    order = _names(doc)
    head = doc.history.head
    assert doc.move_into_group(0, node.uid)
    assert doc.undo()
    assert doc.history.head == head
    assert _names(doc) == order
    _ok(doc)


def test_moving_the_last_member_out_dissolves_the_group():
    doc = _doc(3)
    node = doc.group_layers([1], name="Ink")
    assert doc.move_into_group(1, None)
    assert node.uid not in doc.groups
    _ok(doc)


def test_a_move_that_changes_nothing_is_refused():
    doc = _doc(3)
    node = doc.group_layers([1], name="Ink")
    head = doc.history.head
    assert doc.move_into_group(1, node.uid) is False
    assert doc.history.head == head


def test_a_group_may_not_be_moved_into_its_own_subtree():
    """A group inside itself is not a state any op should be able to reach, so
    the refusal is by name rather than by ``ancestry`` quietly stopping."""
    doc = _doc(4)
    outer = doc.group_layers([1, 2, 3], name="outer")
    assert gp.descends_from(doc.group_of, doc.stack[2].uid, outer.uid)
    # The panel's route: dropping a row of ``outer`` onto ``outer`` is a no-op
    # rather than a cycle, and the guard is what makes a *group* payload safe.
    assert doc.move_into_group(2, outer.uid) is False


def test_taking_a_layer_out_of_the_middle_of_a_span_moves_it_clear():
    """The harder direction. Dropping the membership alone would leave the
    group's leaves in two halves with a stranger between them."""
    doc = _doc(5)
    node = doc.group_layers([1, 2, 3], name="Ink")
    middle = doc.stack[2].uid

    assert doc.move_into_group(2, None)

    _ok(doc)
    assert middle not in doc.group_of
    assert gp.leaves_of(doc.group_of, doc.member_uids(), node.uid) == [
        doc.stack[1].uid,
        doc.stack[2].uid,
    ]


def test_taking_a_layer_out_of_the_end_of_a_span_leaves_it_where_it_is():
    doc = _doc(4)
    doc.group_layers([1, 2], name="Ink")
    bottom = doc.stack[1].uid
    assert doc.move_into_group(1, None)
    _ok(doc)
    assert doc.stack[1].uid == bottom


def test_taking_a_layer_out_of_a_nested_group_clears_the_outer_span_too():
    doc = _doc(5)
    outer = doc.group_layers([1, 2, 3], name="outer")
    inner = doc.group_layers([2, 3], name="inner")
    middle = doc.stack[2].uid

    assert doc.move_into_group(2, None)

    _ok(doc)
    assert middle not in doc.group_of
    assert outer.uid in doc.groups
    assert inner.uid in doc.groups


def test_moving_from_an_inner_group_to_its_parent():
    doc = _doc(5)
    outer = doc.group_layers([1, 2, 3], name="outer")
    inner = doc.group_layers([2, 3], name="inner")
    moved = doc.stack[2].uid

    assert doc.move_into_group(2, outer.uid)

    _ok(doc)
    assert doc.group_of[moved] == outer.uid
    assert inner.uid in doc.groups


def test_a_duplicated_track_keeps_both_locks():
    """The still branch copies every property (``Layer.copy`` does), and this
    list stopping at four unlocked the duplicate on an animated document and
    nowhere else."""
    doc = _doc(2)
    doc.add_frame(copy=True)
    doc.set_layer_props(1, alpha_lock=True, locked=True)
    doc.duplicate_layer(1)
    assert doc.anim is not None
    assert (doc.anim.tracks[2].alpha_lock, doc.anim.tracks[2].locked) == (True, True)


def test_flattening_an_animated_grid_honours_a_hidden_group():
    """The flatten has to see the fold, or a hidden folder's layers reappear in
    the flattened cel -- and it has to see it *before* the tree is cleared."""
    doc = _doc(2)
    doc.stack[1].pixels[:, :] = BLUE
    doc.add_frame(copy=True)
    doc.set_current_frame(0)
    node = doc.group_layers([1], name="Ink")
    doc.set_group_props(node.uid, visible=False)

    doc.flatten_layers()

    assert not doc.groups
    anim = doc.anim
    assert anim is not None
    for frame in anim.frames:
        cel = anim.cels[(anim.tracks[0].uid, frame.uid)]
        assert tuple(cel.pixels[4, 4]) == RED
