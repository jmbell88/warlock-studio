"""The viewport's two pieces of chrome, asserted without a viewport.

Both are pure, and that is what this file is for: "does the +X ball sit on the
right when the camera is at the front" and "does edge mode mention the loop
shortcut" are questions a headless test can ask and a screenshot cannot be made
to fail on.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio import clay_hints, clay_ops, clay_state
from warlock.studio.viewer.camera import Camera

SIZE = 84.0
CENTRE = SIZE * 0.5


def _at(name: str) -> list[clay_hints.AxisBall]:
    camera = Camera()
    assert camera.look_along(name)
    return clay_hints.axis_layout(camera.view(), SIZE)


# --- the navigation widget ---------------------------------------------------


def test_there_are_six_ends_and_three_of_them_are_lettered():
    balls = _at("front")
    assert len(balls) == 6
    assert sorted(ball.label for ball in balls if ball.label) == ["X", "Y", "Z"]
    assert all(ball.positive for ball in balls if ball.label), (
        "only the positive end of an axis carries its letter"
    )


@pytest.mark.parametrize("name", sorted(Camera.AXIS_VIEWS))
def test_every_ball_puts_the_camera_where_its_axis_points(name):
    """The pairing of a ball to a ``Camera.AXIS_VIEWS`` name, checked against
    the camera rather than against the table that declares it.

    Looking *from* the +Z side is what "front" means, so with the camera there
    the +Z ball must be the one nearest the viewer -- and dead centre, because
    an axis pointing straight at the eye projects to a point.
    """
    balls = _at(name)
    nearest = balls[-1]
    assert nearest.view == name
    assert nearest.x == pytest.approx(CENTRE, abs=0.5)
    assert nearest.y == pytest.approx(CENTRE, abs=0.5)
    assert nearest.depth == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("name", sorted(Camera.AXIS_VIEWS))
def test_the_opposite_ball_is_hidden_behind_it(name):
    """They land on the same pixel, and which is on top is the whole of what
    tells the reader which way they are looking. Back to front, so the far one
    is drawn first."""
    balls = _at(name)
    assert balls[0].depth == pytest.approx(-1.0, abs=1e-6)
    assert balls[0].x == pytest.approx(balls[-1].x, abs=0.5)


def test_the_balls_are_ordered_back_to_front():
    balls = clay_hints.axis_layout(Camera().view(), SIZE)
    assert [ball.depth for ball in balls] == sorted(ball.depth for ball in balls)


def test_x_is_to_the_right_and_y_is_up_at_a_front_on_camera():
    """Screen y grows downward, which is the one conversion the layout owes and
    the easiest to get backwards."""
    balls = {ball.view: ball for ball in _at("front")}
    assert balls["right"].x > CENTRE
    assert balls["left"].x < CENTRE
    assert balls["top"].y < CENTRE, "up the screen is a smaller y"
    assert balls["bottom"].y > CENTRE


def test_every_ball_stays_inside_the_box():
    for name in Camera.AXIS_VIEWS:
        for ball in _at(name):
            assert 0.0 <= ball.x <= SIZE, (name, ball)
            assert 0.0 <= ball.y <= SIZE, (name, ball)


def test_the_layout_scales_with_the_box():
    small = clay_hints.axis_layout(Camera().view(), 40.0)
    big = clay_hints.axis_layout(Camera().view(), 80.0)
    for a, b in zip(small, big, strict=True):
        assert b.x == pytest.approx(a.x * 2.0)
        assert b.y == pytest.approx(a.y * 2.0)


def test_a_degenerate_matrix_still_produces_six_balls():
    """A pane must not raise on the first frame, before a camera exists."""
    assert len(clay_hints.axis_layout(np.eye(4), SIZE)) == 6


# --- the hint line -----------------------------------------------------------


@pytest.mark.parametrize("mode", clay_ops.ALL_MODES)
@pytest.mark.parametrize("tool", [key for key, _label, _key in clay_state.TOOLS])
def test_every_mode_and_tool_pair_has_a_hint(mode, tool):
    line = clay_hints.hint(mode, tool)
    assert line and "  " not in line
    # The two navigation buttons are on every line: they are what a newcomer to
    # a 3D viewport asks about first and what a manual is least open at.
    assert "Alt+LMB orbit" in line and "MMB pan" in line


def test_the_element_modes_advertise_the_verbs_only_they_have():
    """The complaint this answers: none of these is a button, so a user who has
    not read chapter 30 cannot discover that edge mode does anything vertex
    mode does not."""
    assert "Alt+click loop" in clay_hints.hint("edge", "select")
    assert "Ctrl+Alt+click ring" in clay_hints.hint("edge", "select")
    assert "Alt+click loop" in clay_hints.hint("face", "select")
    assert "Ctrl+Alt+click ring" not in clay_hints.hint("face", "select"), (
        "a ring is an edge idea; offering it on faces is offer-then-refuse"
    )
    for mode in clay_ops.ELEMENT_MODES:
        assert "L linked" in clay_hints.hint(mode, "select")


def test_object_mode_offers_neither_the_element_verbs_nor_edit_mode_twice():
    line = clay_hints.hint("object", "select")
    assert "Tab edit" in line
    assert "loop" not in line and "linked" not in line


def test_each_transform_tool_names_its_own_key():
    assert "G move" in clay_hints.hint("object", "move")
    # E, not R: ``clay_mode.TOOL_KEYS`` binds R to Scale and ``DRAG_KEYS`` has
    # no rotate key at all, so "R rotate" named a key that scaled.
    assert "E rotate" in clay_hints.hint("object", "rotate")
    assert "S scale" in clay_hints.hint("object", "scale")
    assert clay_hints.hint("object", "select") == clay_hints.hint("object", "select")


def test_a_live_drag_replaces_the_line_rather_than_adding_to_it():
    """Mid-drag the only keys that mean anything are the ones that constrain,
    commit or cancel it, and a line still offering "Tab object" would be
    offering a key that is not listened to."""
    line = clay_hints.hint("edge", "move", dragging=True, drag_kind="move")
    assert line.startswith("Move")
    assert "X/Y/Z lock" in line and "Esc/RMB cancel" in line
    assert "Tab" not in line and "Alt+click loop" not in line


def test_the_drag_line_names_which_drag_is_running():
    for kind in ("move", "rotate", "scale"):
        assert clay_hints.hint("object", "move", dragging=True, drag_kind=kind).startswith(
            kind.capitalize()
        )
    # And says something rather than nothing when the kind is unknown.
    assert clay_hints.hint("object", "move", dragging=True).startswith("Drag")


def test_an_unknown_mode_falls_back_rather_than_raising():
    assert clay_hints.hint("nonsense", "select") == clay_hints.hint("object", "select")


# --- the keys the line names -------------------------------------------------


def test_keys_named_finds_the_chords_and_the_bare_letters():
    found = clay_hints.keys_named(clay_hints.hint("edge", "move"))
    assert {"L", "G", "Tab", "LMB", "MMB", "Alt+click", "Ctrl+Alt+click"} <= found


def test_keys_named_does_not_read_english_as_a_binding():
    """"drag a ring" -- the article is not the A key. A single letter counts
    only when it is capital, which is how this app writes every binding."""
    assert clay_hints.keys_named("R rotate . drag a ring") == {"R"}
    assert "X" in clay_hints.keys_named("X/Y/Z lock")
    assert clay_hints.keys_named("grow/shrink the selection") == set()


def test_every_key_the_line_names_is_a_key_the_mode_listens_to():
    """The parity that matters: a hint naming a binding nothing implements is
    worse than no hint, because it is read as a promise."""
    from warlock.studio import clay_mode

    letters = set()
    for mode in clay_ops.ALL_MODES:
        for tool, _label, _key in clay_state.TOOLS:
            letters |= {
                key.lower()
                for key in clay_hints.keys_named(clay_hints.hint(mode, tool))
                if len(key) == 1 and key.isupper()
            }
    known = set(clay_mode.TOOL_KEYS) | {"g", "s", "r", "l"}
    assert letters <= known, sorted(letters - known)
