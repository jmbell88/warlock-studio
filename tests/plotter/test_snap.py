"""The snap setting, and what Ctrl means now that there is one.

Snapping used to be Ctrl-gated and nothing else: hold Ctrl and a move, a resize
or a vertex drag landed on a cell corner; let go and it did not. There was no
setting, so a user who wanted every object on the grid held a modifier down for
the length of every gesture, and one who wanted Tiled's *Snap to Pixels* could
not have it at all.

**The decision this file pins is that Ctrl inverts the setting.** Not "Ctrl
still forces grid snapping", not "Ctrl now disables snapping": the momentary
opposite of whatever is set, which is what every editor with a persisted snap
does. The consequence worth stating is that the default is unchanged behaviour
-- with the setting at ``"off"``, Ctrl means exactly what it always meant -- so
this landed without moving anybody's hands.
"""

from __future__ import annotations

import pytest

from warlock.studio import plotter_state
from warlock.studio.panes import plotter_canvas as canvas

from ._drive import Scene


@pytest.fixture
def scene(monkeypatch):
    return Scene(monkeypatch, tile=16)


# --- the policy, as a table ---------------------------------------------------


@pytest.mark.parametrize(
    ("setting", "ctrl", "expected"),
    [
        ("off", False, "off"),
        # The compatibility row: what Ctrl has always done, unchanged.
        ("off", True, "grid"),
        ("grid", False, "grid"),
        ("grid", True, "off"),
        ("pixel", False, "pixel"),
        ("pixel", True, "off"),
        # A setting from a future the reader is not in: fall back rather than
        # snap to something nobody asked for.
        ("nonsense", False, "off"),
    ],
)
def test_ctrl_inverts_the_setting(setting, ctrl, expected):
    assert plotter_state.snap_mode(setting, ctrl) == expected


def test_the_three_modes_are_the_three_the_sidebar_offers():
    from warlock.studio.panes import plotter_tools

    assert tuple(key for key, _label in plotter_tools.SNAP_LABELS) == plotter_state.SNAP_MODES
    for key in plotter_state.SNAP_MODES:
        assert "Ctrl" in plotter_tools.SNAP_TIPS[key], (
            "a modifier nobody documents is a modifier nobody finds"
        )


def test_a_fresh_state_snaps_to_nothing():
    """The default has to be today's behaviour or every existing map's objects
    move the first time somebody nudges one."""
    assert plotter_state.PlotterState().snap == "off"


# --- through the real gesture -------------------------------------------------


def _rect(scene):
    # Deliberately off the grid: 20 is not a multiple of the 16px tile, so a
    # snapped drag and an unsnapped one cannot accidentally agree.
    return scene.add(kind="rect", x=20.0, y=20.0, w=32.0, h=32.0)


def _drag_body_to(scene, obj, target, *, ctrl: bool = False):
    """Grab the object in the middle and put that grab point at ``target``."""
    grab = (obj.x + obj.w * 0.5, obj.y + obj.h * 0.5)
    offset = (grab[0] - obj.x, grab[1] - obj.y)
    scene.frame(grab, click=True)
    scene.frame((target[0] + offset[0], target[1] + offset[1]), down=True, ctrl=ctrl)


@pytest.mark.parametrize(
    ("setting", "ctrl", "expected"),
    [
        ("off", False, (69.0, 45.0)),
        ("off", True, (64.0, 32.0)),
        ("grid", False, (64.0, 32.0)),
        ("grid", True, (69.0, 45.0)),
        ("pixel", False, (69.0, 45.0)),
    ],
)
def test_a_move_lands_where_the_setting_says(scene, setting, ctrl, expected):
    obj = _rect(scene)
    scene.state.snap = setting
    _drag_body_to(scene, obj, (69.0, 45.0), ctrl=ctrl)
    after = scene.object(obj.uid)
    assert (after.x, after.y) == pytest.approx(expected)


def test_pixel_snapping_is_the_whole_map_pixel(scene):
    """Which is invisible against an integer drag -- so the drag here is not
    one. Tiled's third setting exists for maps whose objects are smaller than a
    tile, where the cell corner is far too coarse to place anything."""
    obj = _rect(scene)
    scene.state.snap = "pixel"
    _drag_body_to(scene, obj, (69.4, 45.6))
    after = scene.object(obj.uid)
    assert (after.x, after.y) == (69.0, 46.0)

    scene.state.snap = "off"
    scene.doc.end_object_edit()
    _drag_body_to(scene, obj, (69.4, 45.6))
    assert (scene.object(obj.uid).x, scene.object(obj.uid).y) == pytest.approx((69.4, 45.6))


def test_a_resize_reads_the_same_setting(scene):
    obj = _rect(scene)
    scene.state.snap = "grid"
    corner = canvas._handle_corners(obj)["se"]
    scene.frame(corner, click=True)
    assert scene.state.drag_kind == "object-resize"
    scene.frame((70.0, 70.0), down=True)
    after = scene.object(obj.uid)
    # The pinned corner is the object's own (20, 20); the dragged one lands on
    # the cell corner at (64, 64).
    assert (after.x, after.y) == pytest.approx((20.0, 20.0))
    assert (after.w, after.h) == pytest.approx((44.0, 44.0))


def test_a_vertex_reads_the_same_setting(scene):
    from warlock.studio.plotter.tilemap import Polygon

    obj = scene.add(
        shape=Polygon(((0.0, 0.0), (32.0, 0.0), (32.0, 32.0))), x=0.0, y=0.0
    )
    scene.state.snap = "grid"
    scene.frame((32.0, 0.0), click=True)
    assert scene.state.drag_kind == "object-vertex"
    scene.frame((70.0, 5.0), down=True)
    assert scene.object(obj.uid).shape.points[1] == pytest.approx((64.0, 0.0))


# --- the two chords -----------------------------------------------------------


def _key(name: str, *, ctrl: bool = False, shift: bool = False):
    import pygame

    mods = (pygame.KMOD_CTRL if ctrl else 0) | (pygame.KMOD_SHIFT if shift else 0)
    return pygame.event.Event(pygame.KEYDOWN, key=getattr(pygame, f"K_{name}"), mod=mods)


def test_ctrl_shift_g_toggles_grid_snapping_the_way_tiled_spells_it(plotter_ctx):
    ctx, state = plotter_ctx
    from warlock.studio import plotter_mode

    assert plotter_mode.handle_key(ctx, _key("g", ctrl=True, shift=True)) is True
    assert state.snap == "grid"
    plotter_mode.handle_key(ctx, _key("g", ctrl=True, shift=True))
    assert state.snap == "off"


def test_ctrl_shift_p_toggles_pixel_snapping_and_the_two_do_not_fight(plotter_ctx):
    ctx, state = plotter_ctx
    from warlock.studio import plotter_mode

    plotter_mode.handle_key(ctx, _key("p", ctrl=True, shift=True))
    assert state.snap == "pixel"
    # Grid from pixel is a straight replacement, not a cycle through off: the
    # two chords behave like Tiled's two checkable rows.
    plotter_mode.handle_key(ctx, _key("g", ctrl=True, shift=True))
    assert state.snap == "grid"


def test_plain_ctrl_g_still_toggles_the_grid_and_not_the_snap(plotter_ctx):
    ctx, state = plotter_ctx
    from warlock.studio import plotter_mode

    plotter_mode.handle_key(ctx, _key("g", ctrl=True))
    assert state.grid is False and state.snap == "off"
