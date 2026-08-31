"""Selecting several objects at once, and dragging them as one.

Three claims, and every one of them is driven through ``_object_input`` rather
than through a helper, because that dispatch is where a drag kind is chosen and
a drawn-but-dead control is this codebase's most common historical defect:

* the **marquee** belongs to the Select tool and to nothing else -- an insert
  tool's empty-space drag has always drawn an object and still must;
* **Shift** and **Ctrl** toggle one object in or out of the set, and neither
  starts a drag;
* a **group drag is one undo step**, built out of the ``ObjectPropsEdit`` a
  single-object drag already pushes, and it is addressed by uid -- so a reorder
  that moves the objects around underneath an open session changes nothing
  about what the undo restores.

What is deliberately *not* here: a bulk property edit across the set. The
Properties pane says "N objects selected" and offers move and delete, which is
the whole of this pass.
"""

from __future__ import annotations

import pytest

from warlock.studio.panes import plotter_canvas as canvas
from warlock.studio.plotter import tilemap

from ._drive import Scene


@pytest.fixture
def scene(monkeypatch):
    return Scene(monkeypatch)


def _three(scene):
    """Three 16x16 rectangles on a row, well clear of each other."""
    a = scene.add(kind="rect", x=0.0, y=0.0, w=16.0, h=16.0, name="a")
    b = scene.add(kind="rect", x=64.0, y=0.0, w=16.0, h=16.0, name="b")
    c = scene.add(kind="rect", x=128.0, y=0.0, w=16.0, h=16.0, name="c")
    scene.state.select_object(None)
    return a, b, c


def _centre(obj):
    return (obj.x + obj.w * 0.5, obj.y + obj.h * 0.5)


# --- the marquee --------------------------------------------------------------


def test_a_drag_over_empty_space_sweeps_up_everything_it_touches(scene):
    a, b, c = _three(scene)
    scene.frame((0.0, 40.0), click=True)
    scene.frame((90.0, 60.0), down=True)
    scene.frame((90.0, -8.0), release=True)

    assert scene.state.selected_objects == {a.uid, b.uid}, "and not the far one"


def test_the_band_is_published_while_the_drag_runs_and_dropped_at_release(scene):
    """The rectangle the renderer draws. Recomputing it in the draw would put
    the band a frame behind the pointer, which is why it is state."""
    _three(scene)
    scene.frame((10.0, 100.0), click=True)
    assert scene.state.object_marquee == (10.0, 100.0, 10.0, 100.0)
    scene.frame((50.0, 140.0), down=True)
    assert scene.state.object_marquee == (10.0, 100.0, 50.0, 140.0)
    scene.frame((50.0, 140.0), release=True)
    assert scene.state.object_marquee is None


def test_a_marquee_that_touches_nothing_clears_the_selection(scene):
    a, _b, _c = _three(scene)
    scene.state.select_object(a.uid)
    scene.frame((0.0, 200.0), click=True)
    scene.frame((40.0, 240.0), release=True)
    assert scene.state.selected_objects == set()


def test_a_marquee_drawn_upwards_selects_the_same_objects(scene):
    """The band is normalised, so a drag that ends above and left of where it
    began is a rectangle rather than an empty one."""
    a, b, _c = _three(scene)
    scene.frame((90.0, 60.0), click=True)
    scene.frame((-8.0, -8.0), release=True)
    assert scene.state.selected_objects == {a.uid, b.uid}


def test_shift_marquee_adds_to_what_was_already_selected(scene):
    a, b, c = _three(scene)
    scene.state.select_object(c.uid)
    scene.frame((0.0, 40.0), click=True, shift=True)
    scene.frame((90.0, -8.0), release=True, shift=True)
    assert scene.state.selected_objects == {a.uid, b.uid, c.uid}


def test_a_marquee_selects_on_a_locked_layer(scene):
    """Selecting has always been allowed on a locked layer -- it is how you read
    an object's properties -- and a marquee only selects."""
    a, _b, _c = _three(scene)
    scene.doc.set_layer_props(scene.layer.uid, locked=True)
    scene.frame((-8.0, -8.0), click=True)
    scene.frame((40.0, 40.0), release=True)
    assert scene.state.selected_objects == {a.uid}
    assert scene.state.drag_kind == ""


def test_the_marquee_takes_a_rotated_object_the_band_visibly_crosses(scene):
    """``object_bounds`` turns the corners. A box taken from ``w``/``h`` alone
    would miss the half of a turned rectangle that sticks out."""
    obj = scene.add(kind="rect", x=100.0, y=100.0, w=64.0, h=8.0, name="beam")
    scene.doc.set_object(scene.layer.uid, obj.uid, rotation=90.0)
    scene.state.select_object(None)
    # Straight down from the origin corner is where the turned bar now lies,
    # and nowhere near its unrotated box.
    scene.frame((96.0, 150.0), click=True)
    scene.frame((104.0, 160.0), release=True)
    assert scene.state.selected_objects == {obj.uid}


# --- the insert tools are untouched -------------------------------------------


def test_an_insert_tool_still_draws_an_object_on_an_empty_space_drag(monkeypatch):
    """The regression this wave was most able to cause. Every insert tool is the
    object gesture with a shape already chosen, and empty space means *draw*
    there -- only the Select tool sweeps a band."""
    scene = Scene(monkeypatch, tool="object_rect")
    scene.state.object_shape = "rect"
    before = len(scene.doc.layer(scene.layer.uid).objects)
    depth = len(scene.doc.history)

    scene.frame((32.0, 32.0), click=True)
    assert scene.state.drag_kind == "object", "the insert drag, not a marquee"
    assert scene.state.object_marquee is None
    scene.frame((80.0, 80.0), release=True)

    objects = scene.doc.layer(scene.layer.uid).objects
    assert len(objects) == before + 1
    assert len(scene.doc.history) == depth + 1
    assert scene.state.selected_objects == {objects[-1].uid}


def test_the_select_tool_draws_nothing_on_an_empty_space_drag(scene):
    before = len(scene.doc.layer(scene.layer.uid).objects)
    depth = len(scene.doc.history)
    scene.frame((32.0, 200.0), click=True)
    scene.frame((80.0, 240.0), release=True)
    assert len(scene.doc.layer(scene.layer.uid).objects) == before
    assert len(scene.doc.history) == depth


def test_an_insert_tool_on_a_locked_layer_still_refuses_out_loud(monkeypatch):
    toasts = []
    scene = Scene(monkeypatch, tool="object_rect")
    scene.ctx.toast = lambda message, *_a, **_k: toasts.append(message)
    scene.doc.set_layer_props(scene.layer.uid, locked=True)
    scene.frame((32.0, 32.0), click=True)
    assert scene.state.drag_kind == ""
    assert toasts and "locked" in toasts[-1]


# --- click to toggle ----------------------------------------------------------


def test_shift_click_adds_a_second_object_to_the_set(scene):
    a, b, _c = _three(scene)
    scene.frame(_centre(a), click=True)
    assert scene.state.selected_objects == {a.uid}
    scene.frame(_centre(b), click=True, shift=True)
    assert scene.state.selected_objects == {a.uid, b.uid}


def test_ctrl_click_toggles_too_and_takes_a_selected_object_back_out(scene):
    a, b, _c = _three(scene)
    scene.state.select_objects([a.uid, b.uid])
    scene.frame(_centre(a), click=True, ctrl=True)
    assert scene.state.selected_objects == {b.uid}


def test_a_toggle_starts_no_drag_at_all(scene):
    """A Shift+click that armed a move would drag the object the user was only
    adding to the set, on the first pixel of hand wobble."""
    a, b, _c = _three(scene)
    scene.frame(_centre(a), click=True)
    scene.frame(_centre(a), release=True)
    scene.frame(_centre(b), click=True, shift=True)
    assert scene.state.drag_kind == ""
    assert scene.doc.editing_group is False and scene.doc.editing_object is False


def test_a_click_inside_the_group_that_never_moved_collapses_it_to_one(scene):
    """The way back out of a multi-selection. The press keeps the set so the
    drag can happen; the release narrows it when the drag did not."""
    a, b, _c = _three(scene)
    scene.state.select_objects([a.uid, b.uid])
    scene.frame(_centre(b), click=True)
    assert scene.state.selected_objects == {a.uid, b.uid}, "still whole on the press"
    scene.frame(_centre(b), release=True)
    assert scene.state.selected_objects == {b.uid}


def test_a_group_drag_does_not_collapse_the_set(scene):
    a, b, _c = _three(scene)
    scene.state.select_objects([a.uid, b.uid])
    at = _centre(b)
    scene.frame(at, click=True)
    scene.frame((at[0] + 24.0, at[1]), down=True)
    scene.frame((at[0] + 24.0, at[1]), release=True)
    assert scene.state.selected_objects == {a.uid, b.uid}


def test_pressing_inside_the_group_keeps_the_group(scene):
    """Otherwise a multi-selection could never be dragged: the press that began
    the drag would have thrown the rest of the set away first."""
    a, b, _c = _three(scene)
    scene.state.select_objects([a.uid, b.uid])
    scene.frame(_centre(a), click=True)
    assert scene.state.selected_objects == {a.uid, b.uid}
    assert scene.state.drag_kind == "object-group"


# --- the group drag -----------------------------------------------------------


def test_dragging_one_member_moves_every_member_by_the_same_offset(scene):
    a, b, c = _three(scene)
    scene.state.select_objects([a.uid, c.uid])
    scene.frame(_centre(a), click=True)
    at = _centre(a)
    scene.frame((at[0] + 32.0, at[1] + 16.0), down=True)
    scene.frame((at[0] + 32.0, at[1] + 16.0), release=True)

    assert (scene.object(a.uid).x, scene.object(a.uid).y) == (32.0, 16.0)
    assert (scene.object(c.uid).x, scene.object(c.uid).y) == (160.0, 16.0)
    assert (scene.object(b.uid).x, scene.object(b.uid).y) == (64.0, 0.0), (
        "and the object that was not selected did not move"
    )


def test_the_whole_group_drag_is_one_undo_step_that_restores_every_object(scene):
    a, _b, c = _three(scene)
    scene.state.select_objects([a.uid, c.uid])
    depth = len(scene.doc.history)
    at = _centre(a)
    scene.frame(at, click=True)
    # Several held frames, which is what a real drag is: one step for the
    # gesture, not one per frame.
    scene.frame((at[0] + 10.0, at[1]), down=True)
    scene.frame((at[0] + 20.0, at[1] + 5.0), down=True)
    scene.frame((at[0] + 48.0, at[1] + 48.0), down=True)
    scene.frame((at[0] + 48.0, at[1] + 48.0), release=True)

    assert len(scene.doc.history) == depth + 1, "one step for the whole gesture"
    assert (scene.object(a.uid).x, scene.object(a.uid).y) == (48.0, 48.0)

    scene.doc.undo()
    assert (scene.object(a.uid).x, scene.object(a.uid).y) == (0.0, 0.0)
    assert (scene.object(c.uid).x, scene.object(c.uid).y) == (128.0, 0.0), (
        "both of them, off one Ctrl+Z"
    )
    scene.doc.redo()
    assert (scene.object(a.uid).x, scene.object(a.uid).y) == (48.0, 48.0)
    assert (scene.object(c.uid).x, scene.object(c.uid).y) == (176.0, 48.0)


def test_a_group_edit_survives_a_reorder_underneath_it(scene):
    """**Uid addressing, stated as the thing that breaks without it.** An object
    layer's list order *is* its draw order, and Raise/Lower rewrite it. A
    session that had recorded positions in that list would, after a reorder mid
    drag, move and then restore whichever objects had inherited those slots."""
    a, b, c = _three(scene)
    scene.state.select_objects([a.uid, c.uid])
    at = _centre(a)
    scene.frame(at, click=True)
    assert scene.doc.editing_group is True

    # Raise ``a`` twice: the list goes a, b, c -> b, c, a, so every index the
    # session could have stored now names a different object.
    scene.doc.reorder_object(scene.layer.uid, a.uid, 1)
    scene.doc.reorder_object(scene.layer.uid, a.uid, 1)
    assert [o.name for o in scene.doc.layer(scene.layer.uid).objects] == ["b", "c", "a"]

    scene.frame((at[0] + 32.0, at[1]), down=True)
    scene.frame((at[0] + 32.0, at[1]), release=True)

    assert (scene.object(a.uid).x, scene.object(c.uid).x) == (32.0, 160.0)
    assert scene.object(b.uid).x == 64.0, "the object in between never moved"

    scene.doc.undo()
    assert (scene.object(a.uid).x, scene.object(c.uid).x) == (0.0, 128.0)
    assert scene.object(b.uid).x == 64.0


def test_a_group_press_that_never_moved_pushes_nothing(scene):
    a, _b, c = _three(scene)
    scene.state.select_objects([a.uid, c.uid])
    depth = len(scene.doc.history)
    scene.frame(_centre(a), click=True)
    scene.frame(_centre(a), release=True)
    assert len(scene.doc.history) == depth


def test_a_locked_layer_arms_no_group_drag(scene):
    a, _b, c = _three(scene)
    scene.state.select_objects([a.uid, c.uid])
    scene.doc.set_layer_props(scene.layer.uid, locked=True)
    scene.frame(_centre(a), click=True)
    assert scene.state.drag_kind == ""
    assert scene.doc.editing_group is False


def test_snapping_moves_the_offset_and_not_each_object(scene):
    """Snapping every member's own corner would pull the arrangement apart. The
    delta snaps instead, so the group lands on the grid *and* keeps its shape --
    here two objects five pixels off the grid stay exactly five pixels off it,
    64 apart, after a snapped move."""
    a = scene.add(kind="rect", x=5.0, y=5.0, w=16.0, h=16.0)
    b = scene.add(kind="rect", x=69.0, y=5.0, w=16.0, h=16.0)
    scene.state.select_objects([a.uid, b.uid])
    scene.state.snap = "grid"
    at = _centre(a)
    scene.frame(at, click=True)
    scene.frame((at[0] + 20.0, at[1]), down=True)  # 20 -> one 16px cell
    assert (scene.object(a.uid).x, scene.object(b.uid).x) == (21.0, 85.0)


# --- what a group is not ------------------------------------------------------


def test_the_rotation_gizmo_falls_back_to_a_single_selection(scene):
    """Wave 5's grip is drawn for exactly one object, and pressing where it
    would be for a *member* of a group starts the group move instead -- a
    multi-object rotate is not in this pass, and half of one would be worse
    than none."""
    a, b, _c = _three(scene)
    grip = canvas._rotate_grip(scene.tab.view, (0.0, 0.0), a)
    assert grip is not None

    scene.state.select_object(a.uid)
    scene.frame(grip, click=True)
    assert scene.state.drag_kind == "object-rotate"
    scene.frame(grip, release=True)

    scene.state.select_objects([a.uid, b.uid])
    scene.frame(grip, click=True)
    assert scene.state.drag_kind != "object-rotate"


def test_the_resize_handles_belong_to_a_single_selection_too(scene):
    a, b, _c = _three(scene)
    corner = canvas._handle_corners(a)["se"]
    scene.state.select_object(a.uid)
    scene.frame(corner, click=True)
    assert scene.state.drag_kind == "object-resize"
    scene.frame(corner, release=True)

    scene.state.select_objects([a.uid, b.uid])
    scene.frame(corner, click=True)
    assert scene.state.drag_kind == "object-group", "the group moves, it does not resize"


# --- the selection set itself -------------------------------------------------


def test_the_primary_accessor_is_read_only_and_names_the_last_click(scene):
    a, b, _c = _three(scene)
    state = scene.state
    assert state.selected_object is None
    state.select_object(a.uid)
    assert state.selected_object == a.uid
    state.toggle_object(b.uid)
    assert state.selected_object == b.uid, "the one most recently clicked"
    assert state.selected_objects == {a.uid, b.uid}
    with pytest.raises(AttributeError):
        state.selected_object = a.uid


def test_a_toggle_that_empties_the_set_leaves_no_primary_behind(scene):
    a, _b, _c = _three(scene)
    scene.state.select_object(a.uid)
    scene.state.toggle_object(a.uid)
    assert scene.state.selected_objects == set()
    assert scene.state.selected_object is None


def test_switching_documents_forgets_the_whole_set(scene):
    """``_forget_document_state``'s rule reaching the set rather than a scalar:
    half a selection carried into another map would name objects it does not
    have."""
    from warlock.studio import plotter_state

    a, b, _c = _three(scene)
    scene.state.select_objects([a.uid, b.uid])
    scene.state.add(
        plotter_state.PlotterDoc(doc=tilemap.MapDoc(4, 4, 16, 16), title="Other")
    )
    assert scene.state.selected_objects == set()


# --- the engine, without a pointer --------------------------------------------


def test_a_group_edit_refuses_a_uid_the_layer_does_not_hold(scene):
    a, _b, _c = _three(scene)
    with pytest.raises(KeyError):
        scene.doc.begin_group_edit(scene.layer.uid, [a.uid, 999999])
    assert scene.doc.editing_group is False, "and opens no partial session"


def test_ending_a_group_edit_is_idempotent(scene):
    a, _b, c = _three(scene)
    scene.doc.begin_group_edit(scene.layer.uid, [a.uid, c.uid])
    scene.doc.move_group(8.0, 0.0)
    assert scene.doc.end_group_edit() is True
    assert scene.doc.end_group_edit() is False
    assert scene.doc.end_group_edit() is False


def test_moving_a_group_outside_a_session_refuses_rather_than_writing(scene):
    """``place_object``'s rule: a caller that forgot to open a session would
    otherwise push a step a frame and only be caught by counting the stack."""
    _three(scene)
    with pytest.raises(RuntimeError):
        scene.doc.move_group(1.0, 1.0)


def test_an_undo_closes_an_open_group_edit_first(scene):
    """The chokepoint rule: a session left open is a document whose objects are
    ahead of its history."""
    a, _b, c = _three(scene)
    scene.doc.begin_group_edit(scene.layer.uid, [a.uid, c.uid])
    scene.doc.move_group(24.0, 0.0)
    scene.doc.undo()
    assert scene.doc.editing_group is False
    assert scene.object(a.uid).x == 0.0, "the open move was committed and then undone"


def test_a_member_deleted_mid_drag_does_not_take_the_rest_down_with_it(scene):
    a, _b, c = _three(scene)
    scene.doc.begin_group_edit(scene.layer.uid, [a.uid, c.uid])
    scene.doc.remove_object(scene.layer.uid, c.uid)
    scene.doc.move_group(16.0, 0.0)
    assert scene.doc.end_group_edit() is True
    assert scene.object(a.uid).x == 16.0


def test_deleting_a_group_is_one_step_that_brings_them_all_back(scene):
    a, b, c = _three(scene)
    depth = len(scene.doc.history)
    assert scene.doc.remove_objects(scene.layer.uid, [a.uid, c.uid]) == 2
    assert [o.name for o in scene.doc.layer(scene.layer.uid).objects] == ["b"]
    assert len(scene.doc.history) == depth + 1

    scene.doc.undo()
    assert [o.name for o in scene.doc.layer(scene.layer.uid).objects] == ["a", "b", "c"], (
        "in their original draw order, which is what the recorded indices are for"
    )


def test_removing_nothing_pushes_nothing(scene):
    _three(scene)
    depth = len(scene.doc.history)
    assert scene.doc.remove_objects(scene.layer.uid, []) == 0
    assert len(scene.doc.history) == depth


def test_object_bounds_and_the_band_agree_about_a_point(scene):
    """A point has no extent, so its box is a dot at its own position -- and a
    band drawn across it still takes it."""
    point = scene.add(kind="point", x=40.0, y=40.0)
    assert tilemap.object_bounds(scene.object(point.uid)) == (40.0, 40.0, 40.0, 40.0)
    assert tilemap.objects_in_rect(
        scene.doc.layer(scene.layer.uid).objects, (30.0, 30.0, 50.0, 50.0)
    ) == [point.uid]
