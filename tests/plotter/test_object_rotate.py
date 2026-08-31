"""The object rotation gizmo, and the one thing it must not do.

``MapObject.rotation`` has round-tripped through every format since objects
landed, and until now the only way to author it was to type a number into the
sidebar's *Rotation* field. The grip on the canvas is the gesture, and it has a
single failure mode worth a file of its own: **the turn is about
``(obj.x, obj.y)``, the object's origin corner, not about the middle of the
rectangle you can see.** Those two fields are the unrotated origin and
``_rotated`` turns every drawn point about them, so rotating about the centre
means writing ``x``/``y`` as well -- and any drag that writes a position and an
angle from the same pointer sample makes the object leap on its first frame.
The tests below drive the real press/drag/release through the real dispatch and
assert the corner never moves.
"""

from __future__ import annotations

import math

import pytest

from warlock.studio import plotter_state
from warlock.studio.panes import plotter_canvas as canvas

from ._drive import Scene


@pytest.fixture
def scene(monkeypatch):
    return Scene(monkeypatch)


def _grip(scene, obj):
    """Where the grip is drawn, in screen space -- which at identity view is
    map space. Taken from the production function rather than recomputed: a
    test that aimed at its own idea of the position would pass while the drawn
    grip sat somewhere unclickable."""
    return canvas._rotate_grip(scene.tab.view, (0.0, 0.0), obj)


def _rect(scene, **kwargs):
    return scene.add(kind="rect", x=32.0, y=32.0, w=32.0, h=16.0, **kwargs)


# --- the grip exists and is reachable ----------------------------------------


def test_the_grip_floats_above_the_top_edge_and_follows_the_rotation(scene):
    obj = _rect(scene)
    grip = _grip(scene, obj)
    top_mid = canvas._rotated(obj, obj.w * 0.5, 0.0)
    assert grip[0] == pytest.approx(top_mid[0])
    assert grip[1] < top_mid[1], "above the edge, not inside the body"

    scene.doc.set_object(scene.layer.uid, obj.uid, rotation=180.0)
    turned = _grip(scene, scene.object(obj.uid))
    assert turned[1] > canvas._rotated(scene.object(obj.uid), obj.w * 0.5, 0.0)[1], (
        "upside down, the grip is below the (now bottom) edge"
    )


def test_a_point_and_a_polygon_get_no_grip(scene):
    """``_handle_at``'s gate, shared: a point has no extent to turn and a
    polygon is reshaped by its vertices."""
    from warlock.studio.plotter.tilemap import Polygon

    point = scene.add(kind="point", x=10.0, y=10.0)
    assert canvas._rotate_grip(scene.tab.view, (0.0, 0.0), point) is None
    poly = scene.add(shape=Polygon(((0.0, 0.0), (8.0, 0.0), (8.0, 8.0))), x=0.0, y=0.0)
    assert canvas._rotate_grip(scene.tab.view, (0.0, 0.0), poly) is None


def test_the_grip_stays_the_same_screen_distance_away_at_every_zoom(scene):
    """A fixed *map* offset would collapse into the outline zoomed out, exactly
    where the grip is hardest to hit -- the reason the resize handles are sized
    in ``sp`` too."""
    obj = _rect(scene)
    far = _grip(scene, obj)
    top = canvas._rotated(obj, obj.w * 0.5, 0.0)
    at_one = math.dist(far, top)

    scene.tab.view.zoom = 8.0
    grip = _grip(scene, obj)
    top_screen = canvas.inker_state.to_screen(scene.tab.view, (0.0, 0.0), *top)
    assert math.dist(grip, top_screen) == pytest.approx(at_one)


# --- the gesture -------------------------------------------------------------


def test_pressing_the_grip_arms_a_rotate_rather_than_a_resize_or_a_move(scene):
    obj = _rect(scene)
    scene.frame(_grip(scene, obj), click=True)
    assert scene.state.drag_kind == "object-rotate"


def test_the_object_does_not_jump_on_the_first_frame_of_the_drag(scene):
    """The trap, stated as the user meets it. The grip is not at the object's
    zero-degree bearing, so a gesture that wrote ``atan2`` straight would snap
    the object through fifty-odd degrees before the pointer had moved at all.
    """
    obj = _rect(scene)
    grip = _grip(scene, obj)
    scene.frame(grip, click=True)
    scene.frame(grip, down=True)
    after = scene.object(obj.uid)
    assert after.rotation == pytest.approx(0.0)
    assert (after.x, after.y) == (32.0, 32.0)


def test_the_turn_follows_the_pointer_by_the_angle_it_moved(scene):
    obj = _rect(scene)
    grip = _grip(scene, obj)
    scene.frame(grip, click=True)
    # A point due right of the origin corner: bearing zero.
    scene.frame((obj.x + 100.0, obj.y), down=True)
    moved = -canvas._pointer_angle(obj, grip) % 360.0
    assert scene.object(obj.uid).rotation == pytest.approx(moved)


def test_the_origin_corner_is_the_fixed_point_and_the_centre_is_not(scene):
    """The whole file in one assertion. Under an origin-corner rotation
    ``(x, y)`` never moves and the visual centre swings; under the centre
    rotation this is not, both would be the other way round."""
    obj = _rect(scene)
    before_centre = canvas._rotated(obj, obj.w * 0.5, obj.h * 0.5)
    grip = _grip(scene, obj)
    scene.frame(grip, click=True)
    scene.frame((obj.x, obj.y + 100.0), down=True)  # straight below: bearing 90
    scene.frame((obj.x, obj.y + 100.0), release=True)

    after = scene.object(obj.uid)
    assert (after.x, after.y) == (32.0, 32.0), "the origin corner never moves"
    assert canvas._handle_corners(after)["nw"] == (32.0, 32.0)
    centre = canvas._rotated(after, after.w * 0.5, after.h * 0.5)
    assert math.dist(centre, before_centre) > 1.0, "and the centre swings around it"


def test_a_quarter_turn_puts_the_far_corner_where_the_origin_says_it_should(scene):
    """An exact case, worked by hand: 90 degrees clockwise about (32, 32) sends
    the corner at +32 in x to +32 in y."""
    obj = _rect(scene)
    scene.doc.begin_object_edit(scene.layer.uid, obj.uid)
    scene.doc.place_object(rotation=90.0)
    scene.doc.end_object_edit()
    corners = canvas._handle_corners(scene.object(obj.uid))
    assert corners["ne"][0] == pytest.approx(32.0)
    assert corners["ne"][1] == pytest.approx(64.0)


def test_the_whole_drag_is_one_undo_step_and_undoing_restores_the_angle(scene):
    obj = _rect(scene)
    depth = len(scene.doc.history)
    grip = _grip(scene, obj)
    scene.frame(grip, click=True)
    scene.frame((obj.x + 100.0, obj.y + 40.0), down=True)
    scene.frame((obj.x + 100.0, obj.y + 90.0), down=True)
    scene.frame((obj.x + 100.0, obj.y + 90.0), release=True)

    assert scene.object(obj.uid).rotation != 0.0
    assert len(scene.doc.history) == depth + 1, "one step for the gesture, not one a frame"
    scene.doc.undo()
    assert scene.object(obj.uid).rotation == pytest.approx(0.0)


def test_a_press_that_never_moved_pushes_nothing(scene):
    obj = _rect(scene)
    depth = len(scene.doc.history)
    grip = _grip(scene, obj)
    scene.frame(grip, click=True)
    scene.frame(grip, release=True)
    assert len(scene.doc.history) == depth


# --- snapping ----------------------------------------------------------------


def test_snapping_rounds_the_angle_to_fifteen_degrees(scene):
    """Tiled's step, and the same setting the positional gestures read -- one
    snap control, not one per kind of drag."""
    obj = _rect(scene)
    scene.state.snap = "grid"
    grip = _grip(scene, obj)
    scene.frame(grip, click=True)
    scene.frame((obj.x + 100.0, obj.y + 37.0), down=True)
    angle = scene.object(obj.uid).rotation
    assert angle % plotter_state.ROTATE_SNAP_DEGREES == pytest.approx(0.0)


def test_ctrl_inverts_it_for_rotation_exactly_as_it_does_for_a_move(scene):
    obj = _rect(scene)
    grip = _grip(scene, obj)

    scene.state.snap = "off"
    scene.frame(grip, click=True)
    scene.frame((obj.x + 100.0, obj.y + 37.0), down=True, ctrl=True)
    assert scene.object(obj.uid).rotation % 15.0 == pytest.approx(0.0), (
        "off plus Ctrl snaps, which is what Ctrl always meant"
    )
    scene.frame((obj.x + 100.0, obj.y + 37.0), release=True)

    scene.doc.set_object(scene.layer.uid, obj.uid, rotation=0.0)
    scene.state.snap = "grid"
    scene.frame(grip, click=True)
    scene.frame((obj.x + 100.0, obj.y + 37.0), down=True, ctrl=True)
    assert scene.object(obj.uid).rotation % 15.0 != pytest.approx(0.0), (
        "on plus Ctrl is the escape hatch"
    )


# --- the grip does not steal the other gestures ------------------------------


def test_the_resize_handles_still_win_where_they_are_drawn(scene):
    obj = _rect(scene)
    corner = canvas._handle_corners(obj)["se"]
    scene.frame(corner, click=True)
    assert scene.state.drag_kind == "object-resize"


def test_clicking_the_body_still_moves(scene):
    obj = _rect(scene)
    scene.frame((obj.x + obj.w * 0.5, obj.y + obj.h * 0.5), click=True)
    assert scene.state.drag_kind == "object-move"


def test_a_locked_layer_offers_no_grip_press(scene):
    obj = _rect(scene)
    scene.doc.set_layer_props(scene.layer.uid, locked=True)
    scene.frame(_grip(scene, obj), click=True)
    assert scene.state.drag_kind != "object-rotate"
