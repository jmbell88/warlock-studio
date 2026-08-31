"""Collision shapes that can actually be moved, resized and reshaped.

Until this wave the Collision tab could *add* a shape and *clear* the lot, and
nothing else: ``_add_shape`` hard-coded the geometry to the whole tile and the
view under it was an ``imgui.dummy``, which is a picture rather than a control.
So every collision shape any map made here carried was the same full-tile box,
and ``TilePolygon`` -- which has round-tripped through ``.tsx`` and ``.wmap``
the whole time -- had no author.

Two halves, tested two ways.

* The arithmetic is in :mod:`warlock.studio.tilegrid.picking`, which is
  headless and pure, and is asserted directly.
* The *gesture* is in ``plotter_tileset_editor._collision_input``, and every
  test of it below goes through that real dispatch with the shared synthetic
  pointer (``tests/plotter/_drive.TileScene``) -- press, held frames, release.
  A control that is drawn and does nothing is this codebase's most common
  historical defect, and a unit test of a helper is exactly what fails to
  catch it.
"""

from __future__ import annotations

import pytest

from warlock.studio.panes import plotter_tileset_editor as editor
from warlock.studio.tilegrid import picking
from warlock.studio.tilegrid.tileset import TileEllipse, TilePolygon, TileRect

from ._drive import TileScene


@pytest.fixture
def scene(monkeypatch):
    return TileScene(monkeypatch)


# --- the view is the one place screen and tile pixels meet -------------------


def test_the_tile_is_fitted_into_the_square_rather_than_stretched():
    """A 16 x 32 tile is drawn half as wide as it is tall, so a click at its
    right edge lands on tile pixel 16 and not on 32."""

    view = picking.TileView(origin=(10.0, 20.0), side=256.0, tile_w=16, tile_h=32)
    assert view.scale == 8.0
    assert view.size == (128.0, 256.0)
    assert view.to_screen(16.0, 32.0) == (10.0 + 128.0, 20.0 + 256.0)
    assert view.to_tile(10.0 + 128.0, 20.0 + 256.0) == (16.0, 32.0)
    assert view.contains(10.0 + 127.0, 20.0 + 1.0)
    assert not view.contains(10.0 + 129.0, 20.0 + 1.0), "past the tile, inside the square"


def test_round_tripping_a_point_through_the_view_returns_it():
    view = picking.TileView(origin=(3.5, -2.0), side=200.0, tile_w=25, tile_h=25)
    for point in ((0.0, 0.0), (12.5, 3.25), (25.0, 25.0)):
        assert view.to_tile(*view.to_screen(*point)) == pytest.approx(point)


# --- the generic picker Wave 8's Wang regions will reuse ----------------------


def test_the_nearest_region_wins_rather_than_the_first_one_listed():
    regions = {"far": (10.0, 0.0), "near": (1.0, 0.0)}
    assert picking.nearest_region(regions, (0.0, 0.0), 20.0) == "near"


def test_a_region_beyond_the_radius_is_not_picked():
    assert picking.nearest_region({"a": (10.0, 0.0)}, (0.0, 0.0), 5.0) is None
    assert picking.nearest_region({}, (0.0, 0.0), 5.0) is None


def test_a_tie_goes_to_the_earlier_key_so_the_answer_is_stable():
    regions = {"first": (3.0, 4.0), "second": (-3.0, -4.0)}
    assert picking.nearest_region(regions, (0.0, 0.0), 10.0) == "first"


# --- what a shape occupies ---------------------------------------------------


def test_a_polygons_bounds_are_its_points_offset_by_its_origin():
    poly = TilePolygon(x=4.0, y=2.0, points=((0.0, 0.0), (6.0, 0.0), (3.0, 5.0)))
    assert picking.bounds(poly) == (4.0, 2.0, 6.0, 5.0)
    assert picking.vertices(poly) == ((4.0, 2.0), (10.0, 2.0), (7.0, 7.0))


def test_an_ellipse_is_hit_inside_its_curve_and_not_in_its_corner():
    ellipse = TileEllipse(x=0.0, y=0.0, w=16.0, h=16.0)
    assert picking.hit(ellipse, (8.0, 8.0))
    assert not picking.hit(ellipse, (0.5, 0.5)), "the box corner is outside the curve"
    assert picking.hit(TileRect(x=0.0, y=0.0, w=16.0, h=16.0), (0.5, 0.5))


def test_a_polygon_is_hit_inside_its_outline_only():
    triangle = TilePolygon(x=0.0, y=0.0, points=((0.0, 0.0), (16.0, 0.0), (0.0, 16.0)))
    assert picking.hit(triangle, (2.0, 2.0))
    assert not picking.hit(triangle, (14.0, 14.0))
    assert not picking.hit(TilePolygon(points=((0.0, 0.0), (1.0, 1.0))), (0.5, 0.5))


def test_the_topmost_shape_takes_the_click():
    """Last drawn is on top, so it is the one a click means -- picking the
    first would hand every click to whatever happens to be underneath."""

    shapes = (
        TileRect(x=0.0, y=0.0, w=16.0, h=16.0),
        TileRect(x=4.0, y=4.0, w=4.0, h=4.0),
    )
    assert picking.shape_at(shapes, (5.0, 5.0)) == 1
    assert picking.shape_at(shapes, (12.0, 12.0)) == 0
    assert picking.shape_at(shapes, (99.0, 99.0)) is None


# --- the gestures, through the real dispatch ---------------------------------


def test_a_new_box_covers_the_tile_and_arrives_selected(scene):
    shape = scene.add(TileRect)
    assert shape == TileRect(x=0.0, y=0.0, w=16.0, h=16.0)
    assert scene.state.tileset_shape == 0, "selected, or its handles are not drawn"


def test_pressing_a_shape_selects_it_and_dragging_moves_it(scene):
    scene.add(TileRect)
    # Shrink it first so there is somewhere to move to.
    scene.drag(scene.handle("se"), (8.0, 8.0))
    assert scene.selected() == TileRect(x=0.0, y=0.0, w=8.0, h=8.0)

    scene.drag((2.0, 2.0), (6.0, 5.0))
    assert scene.selected() == TileRect(x=4.0, y=3.0, w=8.0, h=8.0)


def test_a_move_keeps_the_grab_offset_so_the_shape_does_not_leap(scene):
    """Pressing at the middle of a shape and moving one pixel moves the shape
    one pixel -- it does not put its corner under the pointer."""

    scene.add(TileRect)
    scene.drag(scene.handle("se"), (8.0, 8.0))
    scene.frame((6.0, 6.0), click=True)
    scene.frame((7.0, 6.0), down=True)
    assert picking.bounds(scene.selected())[:2] == (1.0, 0.0)
    scene.frame((7.0, 6.0), release=True)


def test_a_drag_that_leaves_the_tile_is_clamped_rather_than_lost(scene):
    scene.add(TileRect)
    scene.drag(scene.handle("se"), (8.0, 8.0))
    scene.drag((4.0, 4.0), (400.0, 400.0), hovered=True)
    assert picking.bounds(scene.selected()) == (8.0, 8.0, 8.0, 8.0)


def test_each_of_the_eight_handles_moves_the_edges_it_names(scene):
    scene.add(TileRect)
    scene.drag(scene.handle("se"), (12.0, 12.0))
    scene.drag(scene.handle("nw"), (4.0, 4.0))
    assert scene.selected() == TileRect(x=4.0, y=4.0, w=8.0, h=8.0)
    scene.drag(scene.handle("n"), (0.0, 2.0))
    assert scene.selected() == TileRect(x=4.0, y=2.0, w=8.0, h=10.0), "only the top moved"
    scene.drag(scene.handle("e"), (14.0, 0.0))
    assert scene.selected() == TileRect(x=4.0, y=2.0, w=10.0, h=10.0), "only the right"


def test_a_resize_pins_the_opposite_edge_for_every_handle():
    """Straight arithmetic over all eight, which is the half a gesture test
    would have to spell out eight times."""

    shape = TileRect(x=4.0, y=4.0, w=8.0, h=8.0)
    for handle in picking.BOX_HANDLES:
        out = picking.resized(shape, handle, (6.0, 6.0), 16, 16)
        x, y, w, h = picking.bounds(out)
        if "w" not in handle:
            assert x == 4.0, handle
        if "e" not in handle:
            assert x + w == 12.0, handle
        if "n" not in handle:
            assert y == 4.0, handle
        if "s" not in handle:
            assert y + h == 12.0, handle


def test_a_resize_cannot_collapse_a_shape_to_nothing(scene):
    """A zero-sized box is a shape whose handles are all in one place, which is
    a shape that can be made and never grabbed again."""

    scene.add(TileEllipse)
    scene.drag(scene.handle("se"), (0.0, 0.0))
    _x, _y, w, h = picking.bounds(scene.selected())
    assert (w, h) == (picking.MIN_SIDE, picking.MIN_SIDE)
    assert picking.bounds(scene.selected())[:2] == (0.0, 0.0), "the pinned corner held"


def test_an_ellipse_resizes_and_stays_an_ellipse(scene):
    scene.add(TileEllipse)
    scene.drag(scene.handle("se"), (10.0, 6.0))
    assert scene.selected() == TileEllipse(x=0.0, y=0.0, w=10.0, h=6.0)


def test_clicking_empty_space_drops_the_selection(scene):
    scene.add(TileRect)
    scene.drag(scene.handle("se"), (6.0, 6.0))
    scene.frame((12.0, 12.0), click=True)
    assert scene.state.tileset_shape is None
    assert scene.state.tileset_drag == ""


def test_a_handle_beats_the_body_underneath_it(scene):
    """The grips sit *on* the outline and so overlap the body. Checking the
    body first would make every one of them unreachable."""

    scene.add(TileRect)
    scene.drag(scene.handle("se"), (8.0, 8.0))
    scene.frame(scene.handle("se"), click=True)
    assert scene.state.tileset_drag == "se"
    scene.frame(scene.handle("se"), release=True)


def test_a_press_outside_the_view_is_not_a_gesture(scene):
    """``hovered`` is imgui's answer about the region, and the dispatch has to
    respect it or a click on the button below would start a drag."""

    scene.add(TileRect)
    scene.state.tileset_shape = None
    scene.frame((8.0, 8.0), click=True, hovered=False)
    assert scene.state.tileset_shape is None and scene.state.tileset_drag == ""


def test_a_drag_that_never_moved_pushes_no_step(scene):
    """A click that changed nothing is not an edit -- pushing an empty step
    makes a saved map ask to be saved again."""

    scene.add(TileRect)
    head = scene.doc.history.head
    scene.drag((8.0, 8.0), (8.0, 8.0))
    assert scene.doc.history.head == head


# --- one drag is one undo step ------------------------------------------------


def test_a_whole_drag_is_a_single_undo_step(scene):
    """Sixty writes a second, one step. Undo puts the shape back where the
    press found it rather than one frame earlier."""

    scene.add(TileRect)
    scene.drag(scene.handle("se"), (8.0, 8.0))
    before = scene.selected()
    head = scene.doc.history.head

    scene.frame((4.0, 4.0), click=True)
    for step in range(1, 5):
        scene.frame((4.0 + step, 4.0 + step), down=True)
    scene.frame((8.0, 8.0), release=True)
    assert scene.doc.history.head == head + 1, "one step for five moved frames"
    assert scene.selected() != before

    scene.doc.undo()
    assert scene.shapes[0] == before


def test_the_document_moves_live_during_the_drag_not_only_at_the_release(scene):
    scene.add(TileRect)
    scene.drag(scene.handle("se"), (8.0, 8.0))
    scene.frame((4.0, 4.0), click=True)
    scene.frame((6.0, 4.0), down=True)
    assert picking.bounds(scene.shapes[0])[0] == 2.0, "the document, not a preview"
    scene.frame((6.0, 4.0), release=True)


def test_an_undo_mid_drag_closes_the_session_rather_than_stepping_over_it(scene):
    """The third session on the document, with the first two's chokepoints: a
    document whose shapes are ahead of its head is the defect they prevent."""

    scene.add(TileRect)
    scene.frame((8.0, 8.0), click=True)
    scene.frame((8.0, 8.0), down=True)
    assert scene.doc._tile_meta_edit is not None
    scene.doc.undo()
    assert scene.doc._tile_meta_edit is None


def test_losing_the_shape_mid_drag_ends_the_gesture(scene):
    """Undo, Clear or a tab switch can take the shape out from under an open
    drag; the next frame has to close the session, not index past the end."""

    scene.add(TileRect)
    scene.frame((8.0, 8.0), click=True)
    scene.doc.set_tile_meta(0, 0, scene.meta.__class__())
    scene.frame((9.0, 9.0), down=True)
    assert scene.state.tileset_drag == "" and scene.doc._tile_meta_edit is None


def test_a_live_write_outside_a_session_is_refused(scene):
    """The pair exists so a drag is one step. A caller that lost its session
    and fell back to a recorded write would start pushing one step per frame,
    which is the defect arriving silently."""

    with pytest.raises(RuntimeError):
        scene.doc.live_tile_meta(scene.meta)


# --- polygons -----------------------------------------------------------------


def test_a_new_polygon_has_a_corner_at_each_corner_of_the_tile(scene):
    poly = scene.add(TilePolygon)
    assert isinstance(poly, TilePolygon)
    assert picking.vertices(poly) == (
        (0.0, 0.0),
        (16.0, 0.0),
        (16.0, 16.0),
        (0.0, 16.0),
    )


def test_a_polygon_corner_can_be_dragged(scene):
    scene.add(TilePolygon)
    scene.drag((16.0, 0.0), (10.0, 4.0))
    assert picking.vertices(scene.selected())[1] == (10.0, 4.0)
    assert picking.vertices(scene.selected())[0] == (0.0, 0.0), "and no other moved"


def test_dragging_a_corner_is_one_undo_step_too(scene):
    scene.add(TilePolygon)
    before = scene.selected()
    head = scene.doc.history.head
    scene.frame((16.0, 0.0), click=True)
    scene.frame((14.0, 2.0), down=True)
    scene.frame((10.0, 4.0), down=True)
    scene.frame((10.0, 4.0), release=True)
    assert scene.doc.history.head == head + 1
    scene.doc.undo()
    assert scene.shapes[0] == before


def test_ctrl_clicking_an_edge_adds_a_corner_on_that_edge(scene):
    """On the nearest edge rather than appended to the end: appending re-routes
    two edges at once and is almost never what the click meant."""

    scene.add(TilePolygon)
    scene.frame((8.0, 0.0), click=True, ctrl=True)
    assert picking.vertices(scene.selected()) == (
        (0.0, 0.0),
        (8.0, 0.0),
        (16.0, 0.0),
        (16.0, 16.0),
        (0.0, 16.0),
    )


def test_alt_clicking_a_corner_removes_it(scene):
    scene.add(TilePolygon)
    scene.frame((16.0, 16.0), click=True, alt=True)
    assert picking.vertices(scene.selected()) == (
        (0.0, 0.0),
        (16.0, 0.0),
        (0.0, 16.0),
    )


def test_the_third_corner_is_refused_by_name_rather_than_ignored(scene):
    scene.add(TilePolygon)
    scene.frame((16.0, 16.0), click=True, alt=True)
    head = scene.doc.history.head
    scene.frame((16.0, 0.0), click=True, alt=True)
    assert len(scene.selected().points) == 3
    assert scene.doc.history.head == head, "and nothing was written"
    assert scene.toasts and "three" in scene.toasts[-1][0]


def test_a_polygon_has_no_box_handles(scene):
    """A bounding-box grip on a polygon would have to scale every point, which
    is a different gesture wearing the same grip."""

    poly = scene.add(TilePolygon)
    assert picking.box_handles(poly) == {}
    assert picking.resized(poly, "se", (4.0, 4.0), 16, 16) == poly


def test_a_polygon_body_still_drags(scene):
    scene.add(TilePolygon)
    # A new polygon fills the tile, so there is nowhere for it to move to --
    # pull each corner in first, then drag the body it now encloses.
    for corner, to in (
        ((0.0, 0.0), (2.0, 2.0)),
        ((16.0, 0.0), (6.0, 2.0)),
        ((16.0, 16.0), (6.0, 6.0)),
        ((0.0, 16.0), (2.0, 6.0)),
    ):
        scene.drag(corner, to)
    assert picking.bounds(scene.selected()) == (2.0, 2.0, 4.0, 4.0)
    scene.drag((4.0, 4.0), (6.0, 7.0))
    assert picking.bounds(scene.selected()) == (4.0, 5.0, 4.0, 4.0)


# --- the shape of the pane ----------------------------------------------------


def test_the_view_is_a_region_and_no_longer_a_dummy():
    """The gap this wave closed, asserted where it was: a ``dummy`` is a
    rectangle of nothing and imgui will not say whether it is hovered."""

    import inspect

    source = inspect.getsource(editor._collision_tab)
    assert "invisible_button" in source
    assert "imgui.dummy(" not in source


def test_the_gesture_runs_on_the_mode_state_the_app_actually_builds(
    plotter_ctx, monkeypatch
):
    """``TileScene`` builds a ``PlotterState()`` directly, which is fast and
    which would go on passing if ``plotter_state.ensure`` stopped producing one
    the tab could hold. This drives the same dispatch over the shared
    ``plotter_ctx`` fixture instead -- the real mode, the real ensure, the real
    toast -- so the two halves are pinned to each other."""

    import numpy as np

    from warlock.studio.tilegrid.picking import TileView
    from warlock.studio.tilegrid.tileset import Tileset

    from ._drive import Mouse

    ctx, state = plotter_ctx
    tab = state.active
    pixels = np.zeros((16, 64, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    tab.doc.add_tileset(Tileset(name="Set", tile_w=16, tile_h=16, pixels=pixels))
    state.editing_tileset = 0
    state.tileset_tab = "Collision"
    assert editor.active(ctx) is True

    editor._add_shape(state, tab, 0, 0, tab.doc.tilesets[0].tileset.meta_of(0), TileRect)
    view = TileView(origin=(0.0, 0.0), side=editor.COLLISION_VIEW, tile_w=16, tile_h=16)
    mouse = Mouse()
    mouse.install(monkeypatch)
    head = tab.doc.history.head
    for at, flags in (
        ((16.0, 16.0), {"clicked": True}),
        ((8.0, 8.0), {"down": True}),
        ((8.0, 8.0), {"released": True}),
    ):
        mouse.at = view.to_screen(*at)
        mouse.clicked = {0: bool(flags.get("clicked")), 1: False, 2: False}
        mouse.down = {0: bool(flags.get("clicked") or flags.get("down")), 1: False, 2: False}
        mouse.released = {0: bool(flags.get("released")), 1: False, 2: False}
        editor._collision_input(ctx, state, tab, 0, 0, view, True)
    assert tab.doc.tilesets[0].tileset.meta_of(0).collision == (
        TileRect(x=0.0, y=0.0, w=8.0, h=8.0),
    )
    assert tab.doc.history.head == head + 1


def test_every_gesture_writes_through_the_documents_own_door():
    """No second write path: the dispatch only ever calls ``set_tile_meta``
    for a one-shot edit or the session pair for a drag."""

    import inspect

    source = inspect.getsource(editor._collision_input)
    assert "with_meta" not in source and "_apply_tile_meta" not in source
    for door in ("set_tile_meta", "begin_tile_meta_edit", "live_tile_meta"):
        assert door in source
