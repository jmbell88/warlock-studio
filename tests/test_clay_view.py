"""The Clay viewport: uploads, the cache, picking and framing.

The cache is the interesting half. The 3D pane shows one loaded GLB and can
rebuild the lot on any change; a Clay document is many objects, one of which
changes while the rest do not, so "only what changed was rebuilt" is a property
worth asserting rather than assuming -- and it is sound only because ``Mesh``
is frozen, which is what makes identity a valid cache key.

Everything that needs a context is skipped where there is no GPU, per the
``gl`` fixture; picking, framing and the cache key itself do not, and are
asserted headlessly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from warlock.studio import clay_view
from warlock.studio.clay import document as bd
from warlock.studio.clay import primitives as bp
from warlock.studio.viewer import math3d as m3


class _State:
    def __init__(self, tool: str = "select") -> None:
        self.tool = tool
        self.snap = False
        self.snap_translate = 0.125
        self.snap_rotate = 15.0
        self.snap_vertex = False


class _Ctx:
    """The *app* ctx, which is not the GL one. ``ClayView`` takes both, and
    the only thing it wants from this one is the selected transform tool."""

    def __init__(self, tool: str = "select") -> None:
        self.state = type("S", (), {"clay": _State(tool)})()


def _doc(*, count: int = 2) -> bd.ClayDoc:
    doc = bd.ClayDoc()
    for i in range(count):
        doc.add_object(
            bd.Obj(
                uid=bd.new_uid(),
                name=f"obj{i}",
                mesh=bp.box(),
                translation=m3.vec3(float(i) * 3.0, 0.0, 0.0),
            )
        )
    return doc


@pytest.fixture
def view(gl):
    v = clay_view.ClayView(gl, _Ctx())
    yield v
    v.release()


RECT = (0.0, 0.0, 128.0, 96.0)


# --- rendering ---------------------------------------------------------------


def test_an_authored_document_renders_something(view) -> None:
    doc = _doc(count=1)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    pixels = np.asarray(view.screenshot().convert("RGB"), dtype="i4")
    # Not "is not black": the background is deliberately not black. What says
    # a mesh was drawn is that the frame is not one flat colour.
    assert pixels.reshape(-1, 3).std(axis=0).max() > 2.0


def test_an_empty_document_still_draws_a_frame(view) -> None:
    view.draw(bd.ClayDoc(), RECT, 0.0)
    assert view.viewport.texture is not None


# --- the cache ---------------------------------------------------------------


def test_only_the_object_whose_mesh_changed_is_rebuilt(view) -> None:
    """The key is ``id(obj.mesh)``, which is valid precisely because ``Mesh``
    is frozen and every op is ``Mesh -> Mesh``: a changed mesh is a different
    object, and an unchanged one is the same object."""
    doc = _doc(count=3)
    view.sync(doc)
    assert view.rebuilds == 3

    doc.set_mesh(doc.objects[1].uid, bp.cone())
    view.sync(doc)
    assert view.rebuilds == 4


def test_moving_an_object_rebuilds_nothing(view) -> None:
    """A transform is a uniform, not a buffer. Putting it in the key would
    rebuild every vertex buffer in the scene on every frame of a drag."""
    doc = _doc(count=2)
    view.sync(doc)
    before = view.rebuilds

    doc.set_transform(doc.objects[0].uid, translation=(5.0, 1.0, 2.0))
    view.sync(doc)
    assert view.rebuilds == before


def test_a_palette_change_rebuilds_everything_once(view) -> None:
    """A material is shared, so a palette edit reaches every object that uses
    it -- and ``set_material`` replaces the entry, which is what the identity
    key sees."""
    doc = _doc(count=2)
    view.sync(doc)
    before = view.rebuilds

    doc.set_material(0, bd.default_material("changed"))
    view.sync(doc)
    assert view.rebuilds == before + 2


def test_syncing_an_unchanged_document_twice_rebuilds_nothing(view) -> None:
    doc = _doc(count=3)
    view.sync(doc)
    before = view.rebuilds
    view.sync(doc)
    view.sync(doc)
    assert view.rebuilds == before


def test_hiding_an_object_releases_its_upload(view) -> None:
    """``visible=False`` means it does not render, does not export and is not
    picked. Keeping its buffers would make one of those three only half true."""
    doc = _doc(count=2)
    view.sync(doc)
    hidden = doc.objects[0]

    doc.set_props(hidden.uid, visible=False)
    view.sync(doc)
    assert hidden.uid not in view._cache


def test_deleting_an_object_releases_its_upload(view) -> None:
    doc = _doc(count=2)
    view.sync(doc)
    uid = doc.objects[0].uid

    doc.remove_object(uid)
    view.sync(doc)
    assert uid not in view._cache
    assert len(view._cache) == 1


def test_releasing_twice_does_not_raise(gl) -> None:
    """Teardown runs on a path that can already have torn down -- a failed
    startup, or a mode switch racing a close."""
    v = clay_view.ClayView(gl, _Ctx())
    v.release()
    v.clear()


# --- the imgui registration rule ---------------------------------------------


def test_the_outgoing_texture_is_forgotten_before_a_resize(view, monkeypatch) -> None:
    """``Viewport.resize`` releases its texture and makes a new one, and the
    imgui backend maps GL names to moderngl objects. Releasing without
    forgetting leaves it holding a dead object under a name the driver is free
    to reissue, which is how an unrelated image starts rendering as this one."""
    forgotten: list[Any] = []
    monkeypatch.setattr(view, "_forget", forgotten.append)

    doc = _doc(count=1)
    view.draw(doc, (0.0, 0.0, 64.0, 64.0), 0.0)
    first = view.viewport.texture
    view.draw(doc, (0.0, 0.0, 96.0, 64.0), 0.0)

    assert first in forgotten
    assert view.viewport.texture is not first


def test_a_redraw_at_the_same_size_forgets_nothing(view, monkeypatch) -> None:
    """A docked panel reports its size every frame; forgetting and
    re-registering the same texture each time would be pure churn."""
    doc = _doc(count=1)
    view.draw(doc, RECT, 0.0)  # the first draw resizes off the 16x16 default

    forgotten: list[Any] = []
    monkeypatch.setattr(view, "_forget", forgotten.append)
    view.draw(doc, RECT, 0.0)
    view.draw(doc, RECT, 0.0)
    assert forgotten == []


def test_release_forgets_the_texture_before_freeing_it(gl, monkeypatch) -> None:
    v = clay_view.ClayView(gl, _Ctx())
    order: list[str] = []
    monkeypatch.setattr(v, "_forget", lambda tex: order.append("forget"))
    real_release = v.viewport.release

    def release() -> None:
        order.append("release")
        real_release()

    monkeypatch.setattr(v.viewport, "release", release)
    v.release()
    assert order == ["forget", "release"]


# --- picking -----------------------------------------------------------------


def test_clicking_a_box_selects_it(view) -> None:
    doc = _doc(count=1)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    centre = (RECT[2] * 0.5, RECT[3] * 0.5)
    assert view.pick(doc, centre) == doc.objects[0].uid


def test_clicking_empty_space_hits_nothing(view) -> None:
    doc = _doc(count=1)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    assert view.pick(doc, (2.0, 2.0)) is None


def test_a_press_on_empty_space_clears_the_selection(view) -> None:
    import pygame

    doc = _doc(count=1)
    doc.select([doc.objects[0].uid])
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(2, 2))
    assert view.handle_event(doc, event, True) is True
    assert doc.selection == set()


def test_a_press_on_a_box_selects_it(view) -> None:
    import pygame

    doc = _doc(count=1)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    pos = (int(RECT[2] * 0.5), int(RECT[3] * 0.5))
    event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)
    view.handle_event(doc, event, True)
    assert doc.selection == {doc.objects[0].uid}


def test_a_hidden_object_cannot_be_picked(view) -> None:
    doc = _doc(count=1)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    doc.set_props(doc.objects[0].uid, visible=False)

    assert view.pick(doc, (RECT[2] * 0.5, RECT[3] * 0.5)) is None


def test_the_nearer_of_two_overlapping_objects_is_picked(view) -> None:
    doc = bd.ClayDoc()
    far = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="far", mesh=bp.box(), translation=m3.vec3(0, 0, -6))
    )
    near = doc.add_object(bd.Obj(uid=bd.new_uid(), name="near", mesh=bp.box()))

    view._rect = RECT
    view.camera.set_target(m3.vec3())
    view.camera.set_position(m3.vec3(0.0, 0.0, 8.0))
    view.camera.aspect = RECT[2] / RECT[3]

    hit = view.pick(doc, (RECT[2] * 0.5, RECT[3] * 0.5))
    assert hit == near.uid
    assert hit != far.uid


# --- framing -----------------------------------------------------------------


def _inside_frustum(camera, point: np.ndarray) -> bool:
    clip = camera.projection() @ camera.view() @ np.append(point, 1.0)
    w = clip[3]
    return bool(w > 0 and all(abs(clip[i]) <= w for i in range(3)))


def test_framing_puts_the_whole_bounding_box_on_screen(view) -> None:
    doc = _doc(count=3)
    view._rect = RECT
    view.camera.aspect = RECT[2] / RECT[3]
    view.frame_selection(doc)

    lo, hi = view.world_bounds(doc)
    corners = [
        np.array([x, y, z])
        for x in (lo[0], hi[0])
        for y in (lo[1], hi[1])
        for z in (lo[2], hi[2])
    ]
    assert all(_inside_frustum(view.camera, c) for c in corners)


def test_framing_a_selection_frames_only_the_selection(view) -> None:
    doc = _doc(count=3)
    view._rect = RECT
    view.camera.aspect = RECT[2] / RECT[3]
    doc.select([doc.objects[0].uid])
    view.frame_selection(doc)

    lo, hi = view.world_bounds(doc, selected_only=True)
    assert np.allclose(view.camera.target, (lo + hi) * 0.5)


def test_framing_an_empty_document_is_a_no_op(view) -> None:
    assert view.frame_selection(bd.ClayDoc()) == 0.0


def test_world_bounds_accounts_for_the_transform(view) -> None:
    doc = bd.ClayDoc()
    doc.add_object(
        bd.Obj(
            uid=bd.new_uid(),
            name="A",
            mesh=bp.box(),
            translation=m3.vec3(10.0, 0.0, 0.0),
            scale=m3.vec3(4.0, 4.0, 4.0),
        )
    )
    lo, hi = view.world_bounds(doc)
    assert np.allclose(lo, [8.0, -2.0, -2.0])
    assert np.allclose(hi, [12.0, 2.0, 2.0])


# --- gizmos ------------------------------------------------------------------


def test_no_gizmo_is_active_for_the_select_tool(view) -> None:
    doc = _doc(count=1)
    doc.select([doc.objects[0].uid])
    assert view.active_gizmo(doc) is None


def test_no_gizmo_is_active_with_nothing_selected(view) -> None:
    view.app_ctx.state.clay.tool = "move"
    assert view.active_gizmo(_doc(count=1)) is None


@pytest.mark.parametrize(
    ("tool", "attr"),
    [("move", "translate_gizmo"), ("rotate", "rotate_gizmo"), ("scale", "scale_gizmo")],
)
def test_each_transform_tool_drives_its_own_gizmo(view, tool: str, attr: str) -> None:
    doc = _doc(count=1)
    doc.select([doc.objects[0].uid])
    view.app_ctx.state.clay.tool = tool
    assert view.active_gizmo(doc) is getattr(view, attr)


def test_a_gizmo_drag_records_one_history_step_per_object(view) -> None:
    """Applied in place while dragging and committed on release: a step per
    mouse-move would fill the undo stack with a hundred entries for one drag,
    and the intermediate positions are not states worth stepping back through."""
    import pygame

    doc = _doc(count=1)
    obj = doc.objects[0]
    doc.select([obj.uid])
    view.app_ctx.state.clay.tool = "move"
    view._rect = RECT

    view._grab = "gizmo"
    view._drag_start = {obj.uid: tuple(np.array(v, copy=True) for v in obj.trs())}
    depth = len(doc.history)
    obj.translation = np.array([2.0, 0.0, 0.0])  # what the drag would have done

    view.handle_event(doc, pygame.event.Event(pygame.MOUSEBUTTONUP, button=1), False)
    assert len(doc.history) == depth + 1

    doc.undo()
    assert np.allclose(obj.translation, [0.0, 0.0, 0.0])


def _drive(view, doc, gizmo, axis, rays) -> None:
    """Run a real multi-event drag through ``_drag_gizmo``.

    Through the gizmo's own ``update``, which is the whole point: a drag
    driven by writing ``obj.translation`` directly -- the shape the step-count
    test above uses -- cannot see what the gizmo actually hands back, and a
    single-``update`` drag cannot tell an increment from a total.
    """
    assert gizmo.begin(axis, *rays[0])
    view._begin_gizmo_drag(doc)
    for i in range(1, len(rays)):
        view._ray = lambda _local, _r=rays[i]: _r
        view._drag_gizmo(doc, (0.0, 0.0))


def _down_at(x: float, y: float):
    """A ray straight down -Z from above the z=0 plane."""
    return (np.array([x, y, 5.0]), np.array([0.0, 0.0, -1.0]))


def test_a_rotate_drag_keeps_every_increment_not_just_the_last(view) -> None:
    """``RotateGizmo.update`` returns the increment *since the last update*, so
    composing it against the press-time rotation keeps only the final
    mouse-move: a 90-degree sweep used to land wherever the last frame
    travelled, and the commit then recorded a near-no-op undo step."""
    import math

    doc = _doc(count=1)
    obj = doc.objects[0]
    doc.select([obj.uid])
    view.app_ctx.state.clay.tool = "rotate"
    view._rect = RECT

    gizmo = view.rotate_gizmo
    gizmo.place(np.zeros(3), m3.identity(), view.camera, int(RECT[3]))
    r = gizmo.scale
    rays = [
        _down_at(r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
        for a in range(0, 91, 15)
    ]
    _drive(view, doc, gizmo, "z", rays)

    swept = gizmo.drag.accumulated
    assert abs(abs(swept) - math.radians(90.0)) < 1e-9
    assert np.allclose(obj.rotation, m3.quat_from_axis_angle([0.0, 0.0, 1.0], swept))


def test_a_translate_drag_displaces_each_object_rather_than_placing_it(view) -> None:
    """``TranslateGizmo.update`` returns the *gizmo's* new world position.
    Assigning it to every selected object collapsed a multi-selection onto one
    point, and teleported any single object whose origin was not already under
    the gizmo."""
    doc = _doc(count=2)  # boxes at x=0 and x=3; the gizmo sits between them
    doc.select([o.uid for o in doc.objects])
    view.app_ctx.state.clay.tool = "move"
    view._rect = RECT

    gizmo = view.translate_gizmo
    gizmo.place(np.array([1.5, 0.0, 0.0]), m3.identity(), view.camera, int(RECT[3]))
    _drive(view, doc, gizmo, "x", [_down_at(1.5, 0.0), _down_at(2.5, 0.0), _down_at(3.5, 0.0)])

    assert np.allclose(doc.objects[0].translation, [2.0, 0.0, 0.0])
    assert np.allclose(doc.objects[1].translation, [5.0, 0.0, 0.0])


def test_a_drag_on_an_object_deleted_midway_does_not_raise(view) -> None:
    import pygame

    doc = _doc(count=1)
    obj = doc.objects[0]
    doc.select([obj.uid])
    view.app_ctx.state.clay.tool = "move"
    view._grab = "gizmo"
    view._drag_start = {obj.uid: tuple(np.array(v, copy=True) for v in obj.trs())}

    doc.remove_object(obj.uid)
    view.handle_event(doc, pygame.event.Event(pygame.MOUSEBUTTONUP, button=1), False)


# --- element modes: input, overlays and the live drag (T15-T17) --------------


@pytest.fixture(autouse=True)
def headless_mods(monkeypatch):
    """``pygame.key.get_mods`` raises with no display; stub it at zero.

    Autouse because every press now reads the modifier state -- see
    ``clay_view``'s module docstring on why it is read from the platform rather
    than shadowed off KEYDOWN/KEYUP -- and a test that forgot would exercise the
    fallback rather than the code.
    """
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0, raising=False)
    return monkeypatch


def _hold(monkeypatch, mods: int) -> None:
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: mods, raising=False)


def _face_mode(doc: bd.ClayDoc) -> None:
    doc.set_element_mode("face")


def _centre() -> tuple[int, int]:
    return (int(RECT[2] * 0.5), int(RECT[3] * 0.5))


def _press(view, doc, pos, button: int = 1) -> None:
    import pygame

    view.handle_event(
        doc, pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=button, pos=pos), True
    )


def _release(view, doc, pos, button: int = 1) -> None:
    import pygame

    view.handle_event(
        doc, pygame.event.Event(pygame.MOUSEBUTTONUP, button=button, pos=pos), True
    )


def test_clicking_a_face_in_face_mode_selects_that_face(view) -> None:
    doc = _doc(count=1)
    _face_mode(doc)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    _press(view, doc, _centre())
    uid = doc.objects[0].uid
    assert doc.selection == {uid}
    assert len(doc.element_sel_of(uid).faces) == 1


def test_shift_adds_and_ctrl_subtracts(view, headless_mods) -> None:
    import pygame

    doc = _doc(count=1)
    _face_mode(doc)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    uid = doc.objects[0].uid

    _press(view, doc, _centre())
    first = doc.element_sel_of(uid).faces.tolist()
    assert first

    _hold(headless_mods, pygame.KMOD_CTRL)
    _press(view, doc, _centre())
    assert first[0] not in doc.element_sel_of(uid).faces.tolist()

    _hold(headless_mods, pygame.KMOD_SHIFT)
    _press(view, doc, _centre())
    assert first[0] in doc.element_sel_of(uid).faces.tolist()


def test_clicking_empty_space_with_the_select_tool_starts_a_marquee(view) -> None:
    doc = _doc(count=1)
    _face_mode(doc)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    _press(view, doc, (2, 2))
    assert view._grab == "marquee"
    assert view.marquee is not None


def test_a_marquee_over_the_whole_viewport_takes_every_face(view) -> None:
    import pygame

    doc = _doc(count=1)
    _face_mode(doc)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    far = (int(RECT[2]) - 1, int(RECT[3]) - 1)
    _press(view, doc, (1, 1))
    view.handle_event(doc, pygame.event.Event(pygame.MOUSEMOTION, pos=far), True)
    _release(view, doc, far)
    assert view.marquee is None
    assert len(doc.element_sel_of(doc.objects[0].uid).faces) == 6


def test_a_zero_area_marquee_clears_the_selection(view) -> None:
    doc = _doc(count=1)
    _face_mode(doc)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    _press(view, doc, _centre())
    assert doc.element_sel

    _press(view, doc, (2, 2))
    _release(view, doc, (2, 2))
    assert doc.element_sel == {}


def test_alt_drag_orbits_instead_of_selecting(view, headless_mods) -> None:
    import pygame

    doc = _doc(count=1)
    _face_mode(doc)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    _hold(headless_mods, pygame.KMOD_ALT)
    _press(view, doc, _centre())
    assert view._grab == "orbit"
    assert doc.element_sel == {}


def test_the_middle_button_pans_and_the_right_one_no_longer_does(view) -> None:
    doc = _doc(count=1)
    view.draw(doc, RECT, 0.0)
    _press(view, doc, _centre(), button=2)
    assert view._grab == "pan"

    view._grab = None
    _press(view, doc, _centre(), button=3)
    assert view._grab is None, "RMB pan is gone; the right button is the menu's"


def test_a_right_click_asks_for_the_menu_and_a_right_drag_does_not(view) -> None:
    doc = _doc(count=1)
    view.draw(doc, RECT, 0.0)

    _press(view, doc, (40, 40), button=3)
    _release(view, doc, (41, 41), button=3)
    assert view.menu_request == (41.0, 41.0)

    view.menu_request = None
    _press(view, doc, (40, 40), button=3)
    _release(view, doc, (90, 90), button=3)
    assert view.menu_request is None


def test_hovering_reports_an_element_and_moving_off_clears_it(view) -> None:
    import pygame

    doc = _doc(count=1)
    _face_mode(doc)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    view.handle_event(doc, pygame.event.Event(pygame.MOUSEMOTION, pos=_centre()), True)
    assert view.hover_element is not None
    view.handle_event(doc, pygame.event.Event(pygame.MOUSEMOTION, pos=(1, 1)), True)
    assert view.hover_element is None


def test_the_screen_cache_reprojects_only_when_something_moved(view) -> None:
    doc = _doc(count=1)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    uid = doc.objects[0].uid

    first = view.screen_of(doc, uid)
    assert view.screen_of(doc, uid) is first, "nothing moved, so nothing reprojected"

    doc.set_transform(uid, translation=(1.0, 0.0, 0.0))
    assert view.screen_of(doc, uid) is not first


def test_the_gizmo_sits_at_the_selected_elements_centroid(view) -> None:
    from warlock.studio.clay import elements as el

    doc = _doc(count=1)
    _face_mode(doc)
    uid = doc.objects[0].uid
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    centre = view.selection_centre(doc)
    # Face 0 of the box is the -Y face, so the centroid is half a metre below
    # the object's own centre rather than at it.
    assert centre is not None
    assert centre[1] == pytest.approx(-0.5)


def test_the_select_tool_shows_no_gizmo_in_an_element_mode(view) -> None:
    from warlock.studio.clay import elements as el

    doc = _doc(count=1)
    _face_mode(doc)
    doc.set_element_sel(doc.objects[0].uid, el.ElementSel(faces=[0]))
    view.app_ctx.state.clay.tool = "select"
    assert view.active_gizmo(doc) is None
    view.app_ctx.state.clay.tool = "move"
    assert view.active_gizmo(doc) is not None


def test_an_element_drag_previews_without_rebuilding_or_pushing(view) -> None:
    from warlock.studio.clay import elements as el

    doc = _doc(count=1)
    _face_mode(doc)
    uid = doc.objects[0].uid
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    view.app_ctx.state.clay.tool = "move"
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    rebuilds, depth = view.rebuilds, len(doc.history)
    view._begin_element_drag(doc)
    centre = view.element_centre(doc)
    view._preview_element_drag(doc, centre + np.array([0.0, 1.0, 0.0]), view.state)

    assert view.rebuilds == rebuilds, "a preview writes buffers, it does not rebuild"
    assert len(doc.history) == depth, "and it pushes nothing"
    assert np.allclose(doc.by_uid(uid).mesh.positions, bp.box().positions), (
        "the document mesh is untouched until the release"
    )


def test_releasing_an_element_drag_pushes_one_step_per_object(view) -> None:
    from warlock.studio.clay import elements as el

    doc = _doc(count=1)
    _face_mode(doc)
    uid = doc.objects[0].uid
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    view.app_ctx.state.clay.tool = "move"
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    before = doc.by_uid(uid).mesh
    view._begin_element_drag(doc)
    view._preview_element_drag(
        doc, view.element_centre(doc) + np.array([0.0, -1.0, 0.0]), view.state
    )
    assert doc.by_uid(uid).mesh is before, "the preview leaves the document alone"

    depth = len(doc.history)
    view._grab = "gizmo"
    view._release_drag(doc)
    assert len(doc.history) == depth + 1

    moved = doc.by_uid(uid).mesh.positions
    touched = el.affected_verts(before, doc.element_sel_of(uid))
    assert np.allclose(moved[touched] - before.positions[touched], [0.0, -1.0, 0.0])

    doc.undo()
    assert np.allclose(doc.by_uid(uid).mesh.positions, before.positions)


def test_a_zero_movement_element_drag_pushes_nothing(view) -> None:
    from warlock.studio.clay import elements as el

    doc = _doc(count=1)
    _face_mode(doc)
    uid = doc.objects[0].uid
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    view.app_ctx.state.clay.tool = "move"
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    view._begin_element_drag(doc)
    depth = len(doc.history)
    view._grab = "gizmo"
    view._release_drag(doc)
    assert len(doc.history) == depth
    assert uid not in view._cache, "the previewed buffers are evicted, not left stale"


# --- the keyboard's half of a drag, and snapping (Clay18, Clay19) ------------


def _moving_face(view, doc) -> int:
    """One face selected in face mode with the move tool, drawn once."""
    from warlock.studio.clay import elements as el

    _face_mode(doc)
    uid = doc.objects[0].uid
    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    view.app_ctx.state.clay.tool = "move"
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    return uid


def test_an_axis_lock_narrows_a_live_element_drag(view) -> None:
    doc = _doc(count=1)
    _moving_face(view, doc)
    view._begin_gizmo_drag(doc)
    view._begin_element_drag(doc)
    centre = view.element_centre(doc)

    view.drag_input.key("y")
    narrowed = view._narrow(doc, centre + np.array([1.0, 2.0, 3.0]), view.state, (0.0, 0.0))
    assert np.allclose(narrowed - centre, [0.0, 2.0, 0.0])


def test_a_typed_value_reaches_the_drag_and_the_hud_says_so(view) -> None:
    doc = _doc(count=1)
    _moving_face(view, doc)
    view._begin_gizmo_drag(doc)
    view._begin_element_drag(doc)
    centre = view.element_centre(doc)
    for ch in ("x", "2"):
        view.drag_input.key(ch)
    narrowed = view._narrow(doc, centre + np.array([9.0, 9.0, 9.0]), view.state, (0.0, 0.0))
    assert np.allclose(narrowed - centre, [2.0, 0.0, 0.0])
    assert "[X]" in view.drag_hud and "2.000" in view.drag_hud


def test_the_drag_keyboard_is_ignored_when_no_drag_is_under_way(view) -> None:
    """The key handler asks ``dragging`` first, so a stray X outside a drag must
    not quietly arm a lock for the next one."""
    doc = _doc(count=1)
    assert not view.dragging
    assert view.drag_key(doc, "x") is False
    assert view.drag_input.axis == ""


def test_a_lock_is_dropped_at_the_release_rather_than_carried_forward(view) -> None:
    doc = _doc(count=1)
    _moving_face(view, doc)
    view._begin_gizmo_drag(doc)
    view.drag_input.key("z")
    view._release_drag(doc)
    assert view.drag_input.axis == ""
    assert view.drag_hud == ""


def test_cancelling_a_drag_puts_the_objects_back_and_records_nothing(view) -> None:
    doc = _doc(count=1)
    uid = doc.objects[0].uid
    doc.select([uid])
    view.app_ctx.state.clay.tool = "move"
    view.draw(doc, RECT, 0.0)
    before = np.array(doc.by_uid(uid).translation, copy=True)

    view._begin_gizmo_drag(doc)
    depth = len(doc.history)
    doc.by_uid(uid).translation = before + np.array([5.0, 0.0, 0.0])
    assert view.cancel_drag(doc)

    assert np.allclose(doc.by_uid(uid).translation, before)
    assert len(doc.history) == depth
    assert not view.dragging


def test_cancelling_an_element_drag_drops_the_previewed_buffers(view) -> None:
    """The cached mesh identity has not changed, so nothing else would ever
    rebuild them and the object would keep rendering the abandoned preview."""
    doc = _doc(count=1)
    uid = _moving_face(view, doc)
    view._begin_gizmo_drag(doc)
    view._preview_element_drag(
        doc, view.element_centre(doc) + np.array([0.0, 1.0, 0.0]), view.state
    )
    depth = len(doc.history)
    assert view.cancel_drag(doc)
    assert uid not in view._cache
    assert len(doc.history) == depth
    assert np.allclose(doc.by_uid(uid).mesh.positions, bp.box().positions)


def test_a_move_snaps_onto_the_vertex_under_the_cursor(view) -> None:
    doc = _doc(count=2)
    _moving_face(view, doc)
    view.app_ctx.state.clay.snap_vertex = True
    view._begin_gizmo_drag(doc)
    view._begin_element_drag(doc)

    other = doc.objects[1]
    screen = view.screen_of(doc, other.uid)
    index = int(np.argmin(screen.depth))
    at = (float(screen.xy[index][0]), float(screen.xy[index][1]))
    world = (view._world(other) @ np.append(other.mesh.positions[index].astype("f8"), 1.0))[:3]

    narrowed = view._narrow(doc, np.zeros(3), view.state, at)
    assert np.allclose(narrowed, world, atol=1e-6)
    assert "snapped" in view.drag_hud


def test_a_move_never_snaps_onto_the_geometry_it_is_moving(view) -> None:
    """The worst failure available, because it looks like the feature working:
    the drag would track the cursor exactly and report a snap."""
    doc = _doc(count=1)
    uid = _moving_face(view, doc)
    view.app_ctx.state.clay.snap_vertex = True
    view._begin_gizmo_drag(doc)
    view._begin_element_drag(doc)

    screen = view.screen_of(doc, uid)
    moving = view._element_drags[uid].verts
    index = int(moving[0])
    at = (float(screen.xy[index][0]), float(screen.xy[index][1]))
    # Something else may still be within the radius -- what must never happen is
    # landing on one of the vertices the drag is carrying.
    found = view._snap_vertex(doc, at)
    matrix = view._world(doc.by_uid(uid))
    carried = [
        (matrix @ np.append(doc.by_uid(uid).mesh.positions[int(v)].astype("f8"), 1.0))[:3]
        for v in moving
    ]
    assert found is None or not any(np.allclose(found, c) for c in carried)


def test_an_object_move_never_snaps_onto_its_own_geometry(view) -> None:
    """The element path's rule, on the object path: in object mode the whole
    object rides the drag, so every one of its vertices -- reprojected at the
    live transform -- would track the cursor exactly and report a snap."""
    doc = _doc(count=1)
    uid = doc.objects[0].uid
    doc.select([uid])
    view.app_ctx.state.clay.tool = "move"
    view.app_ctx.state.clay.snap_vertex = True
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    view._begin_gizmo_drag(doc)

    screen = view.screen_of(doc, uid)
    index = int(np.argmin(screen.depth))
    at = (float(screen.xy[index][0]), float(screen.xy[index][1]))
    assert view._snap_vertex(doc, at) is None, "the only object is the one moving"


def test_only_the_owning_button_releases_a_grab(view) -> None:
    """The press half guards exactly this; the release half has to mirror it.
    An MMB release or a wheel tick (buttons 4/5) mid-LMB-drag used to commit
    the gizmo drag early, and the real LMB-up then found no grab."""
    import pygame

    doc = _doc(count=1)
    _moving_face(view, doc)
    view._begin_gizmo_drag(doc)
    assert view._grab == "gizmo"

    for button in (2, 4, 5):
        up = pygame.event.Event(pygame.MOUSEBUTTONUP, button=button, pos=_centre())
        assert view.handle_event(doc, up, True) is True
        assert view._grab == "gizmo", "a stray button-up must not end the drag"

    up = pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=_centre())
    assert view.handle_event(doc, up, True) is True
    assert view._grab is None


def test_a_pan_is_released_by_the_middle_button_alone(view) -> None:
    doc = _doc(count=1)
    _press(view, doc, _centre(), button=2)
    assert view._grab == "pan"
    import pygame

    up = pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=_centre())
    assert view.handle_event(doc, up, True) is True
    assert view._grab == "pan", "an LMB release must not end a pan"
    up = pygame.event.Event(pygame.MOUSEBUTTONUP, button=2, pos=_centre())
    assert view.handle_event(doc, up, True) is True
    assert view._grab is None


def test_an_object_rotation_snap_quantises_the_delta_not_the_orientation(view) -> None:
    """``ops.snap_rotation``'s own docstring: "rotate this by fifteen degrees
    about the ring I grabbed" leaves an already-placed object where it was put.
    Quantising the absolute orientation instead swung a 37-degree object to 30
    the moment a snapped drag began, even with the mouse held still."""
    doc = _doc(count=1)
    obj = doc.objects[0]
    obj.rotation = m3.quat_from_axis_angle(
        np.array([0.0, 1.0, 0.0]), np.radians(37.0)
    )
    was = tuple(np.array(v, copy=True) for v in obj.trs())
    view.app_ctx.state.clay.snap = True
    view.app_ctx.state.clay.tool = "rotate"

    view._apply(obj, was, m3.quat_identity(), view.state)
    assert np.allclose(obj.rotation, was[1]), "a zero delta moves nothing"

    delta = m3.quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.radians(14.0))
    view._apply(obj, was, delta, view.state)
    snapped = m3.quat_mul(
        m3.quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.radians(15.0)), was[1]
    )
    assert np.allclose(obj.rotation, snapped), "the delta snaps to the grid"


def test_an_explicit_constraint_beats_a_snap(view) -> None:
    """A user who has typed a number has said exactly where the thing goes, and
    moving it onto a nearby vertex instead would be the app overruling them."""
    doc = _doc(count=2)
    _moving_face(view, doc)
    view.app_ctx.state.clay.snap_vertex = True
    view._begin_gizmo_drag(doc)
    view._begin_element_drag(doc)
    centre = view.element_centre(doc)

    other = doc.objects[1]
    screen = view.screen_of(doc, other.uid)
    index = int(np.argmin(screen.depth))
    at = (float(screen.xy[index][0]), float(screen.xy[index][1]))

    for ch in ("x", "1"):
        view.drag_input.key(ch)
    narrowed = view._narrow(doc, centre + np.array([4.0, 4.0, 4.0]), view.state, at)
    assert np.allclose(narrowed - centre, [1.0, 0.0, 0.0])
    assert view._snap_point is None


def test_a_soft_falloff_carries_the_neighbours_part_of_the_way(view) -> None:
    """The whole feature in one assertion: the selected corners move fully, the
    ones behind them move less, and the box bends instead of tearing."""
    from warlock.studio.clay import elements as el

    doc = _doc(count=1)
    uid = _moving_face(view, doc)
    before = doc.by_uid(uid).mesh.positions.copy()
    selected = el.affected_verts(doc.by_uid(uid).mesh, doc.element_sel_of(uid))

    view.app_ctx.state.clay.proportional = True
    view.app_ctx.state.clay.proportional_radius = 2.0
    view._begin_gizmo_drag(doc)
    view._preview_element_drag(
        doc, view.element_centre(doc) + np.array([0.0, 1.0, 0.0]), view.state
    )
    view._release_drag(doc)

    moved = doc.by_uid(uid).mesh.positions - before
    assert np.allclose(moved[selected][:, 1], 1.0, atol=1e-5)
    others = np.setdiff1d(np.arange(len(before)), selected)
    assert (moved[others][:, 1] > 1e-3).all()
    assert (moved[others][:, 1] < 1.0 - 1e-3).all()


def test_a_hard_selection_is_what_a_zero_radius_still_means(view) -> None:
    from warlock.studio.clay import elements as el

    doc = _doc(count=1)
    uid = _moving_face(view, doc)
    before = doc.by_uid(uid).mesh.positions.copy()
    selected = el.affected_verts(doc.by_uid(uid).mesh, doc.element_sel_of(uid))

    view.app_ctx.state.clay.proportional = True
    view.app_ctx.state.clay.proportional_radius = 0.0
    view._begin_gizmo_drag(doc)
    view._preview_element_drag(
        doc, view.element_centre(doc) + np.array([0.0, 1.0, 0.0]), view.state
    )
    view._release_drag(doc)

    moved = doc.by_uid(uid).mesh.positions - before
    others = np.setdiff1d(np.arange(len(before)), selected)
    assert np.allclose(moved[others], 0.0)


def test_element_overlays_are_built_and_released_with_the_mode(view) -> None:
    from warlock.studio.clay import elements as el

    doc = _doc(count=1)
    _face_mode(doc)
    doc.set_element_sel(doc.objects[0].uid, el.ElementSel(faces=[0]))
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    assert view._overlays, "an element mode draws its overlay"

    doc.set_element_mode("object")
    view.draw(doc, RECT, 0.0)
    assert view._overlays == {}, "object mode releases them"


def test_repeated_element_draws_reuse_the_overlays_gl_objects(view) -> None:
    """The draw list is a pure function of the overlay's cache key, so drawing
    an unchanged frame again must mint no GL objects. Each ``indexed`` call
    used to append a fresh IBO and VAO per draw per frame, released only on a
    key change -- a leak at frame rate for as long as the cursor held still."""
    from warlock.studio.clay import elements as el

    doc = _doc(count=1)
    _face_mode(doc)
    doc.set_element_sel(doc.objects[0].uid, el.ElementSel(faces=[0]))
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    overlay = next(iter(view._overlays.values()))
    count = len(overlay._vaos)
    assert count > 0

    view.draw(doc, RECT, 0.0)
    view.draw(doc, RECT, 0.0)
    assert next(iter(view._overlays.values())) is overlay, "the key did not change"
    assert len(overlay._vaos) == count


def test_an_object_with_nothing_selected_keeps_its_overlay_across_frames(view) -> None:
    """The key held ``id(sel)``, and ``element_sel_of`` synthesises a fresh
    ``empty()`` for an object with nothing selected -- a temporary nothing
    keeps alive. So the key changed every frame (a full re-upload per
    unselected object, forever) or, once the allocator reissued the address,
    matched a stale overlay and left a new selection invisible."""
    doc = _doc(count=1)
    _face_mode(doc)  # nothing selected inside it
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    overlay = next(iter(view._overlays.values()))
    view.draw(doc, RECT, 0.0)
    view.draw(doc, RECT, 0.0)

    assert next(iter(view._overlays.values())) is overlay


def test_selecting_inside_that_object_does_change_the_key(view) -> None:
    """The other half: the cache must still notice a real change."""
    from warlock.studio.clay import elements as el

    doc = _doc(count=1)
    _face_mode(doc)
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    before = next(iter(view._overlays.values()))

    doc.set_element_sel(doc.objects[0].uid, el.ElementSel(faces=[0]))
    view.draw(doc, RECT, 0.0)

    assert next(iter(view._overlays.values())) is not before


# --- textured documents reach the GPU unchanged (T23) ------------------------


def test_a_textured_document_uploads_its_uvs_and_its_maps(view) -> None:
    """Verification, not construction: ``_build`` -> ``to_primitives`` ->
    ``GpuPrimitive``/``GpuMaterial`` already carried both. What this pins is
    that an imported asset actually reaches the GPU with them rather than
    silently losing one on the way through the Clay-specific half."""
    import numpy as np

    from warlock.studio.clay import mesh as cm
    from warlock.studio.viewer import gltf

    plane = bp.plane(size=(2.0, 2.0))
    n = len(plane.loops)
    textured = cm.Mesh(
        positions=plane.positions,
        loops=plane.loops,
        starts=plane.starts,
        material=plane.material,
        smooth=plane.smooth,
        uv=np.stack(
            [np.arange(n, dtype="f4") / n, np.zeros(n, dtype="f4")], axis=1
        ),
    )
    image = (2, 2, bytes(range(16)))
    doc = bd.ClayDoc(materials=[gltf.Material(name="baked", base_color=image)])
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="P", mesh=textured))

    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    entry = view._cache[doc.objects[0].uid]
    _node, gpu = entry.gpu.draws[0]
    assert gpu.primitive.uvs is not None
    assert "HAS_BASE_COLOR_MAP" in gpu.defines
    assert "base_color" in gpu.material.textures

    pixels = np.asarray(view.screenshot().convert("RGB"), dtype="i4")
    assert pixels.reshape(-1, 3).std(axis=0).max() > 2.0


# --- the hover overlay --------------------------------------------------------


def test_moving_the_hover_rebuilds_only_the_hover_draw(view) -> None:
    """Hover used to be in the overlay's cache key, so crossing onto the next
    face released the whole overlay and built it again -- the position buffer
    re-uploaded and the guide wireframe's index buffer, two indices per edge and
    megabytes on an import, minted from scratch, on every mouse move."""
    doc = _doc(count=1)
    _face_mode(doc)
    uid = doc.objects[0].uid
    view.frame_selection(doc)
    view.hover_element = (uid, 0)
    view.draw(doc, RECT, 0.0)

    overlay = view._overlays[uid]
    guides, positions = list(overlay._vaos), overlay.pos_vbo
    assert len(overlay._hover_vaos) == 1, "the hovered face is drawn"

    view.hover_element = (uid, 1)
    view.draw(doc, RECT, 0.0)

    assert view._overlays[uid] is overlay, "the overlay survived the cursor moving"
    assert overlay._vaos == guides, "and so did the guides and the selection"
    assert overlay.pos_vbo is positions, "the vertices were not re-uploaded"
    assert len(overlay._hover_vaos) == 1, "one hover buffer, not two"


def test_the_hover_draw_appears_and_goes_away(view) -> None:
    """The other half: splitting the key must not cost the overlay its hover."""
    doc = _doc(count=1)
    _face_mode(doc)
    uid = doc.objects[0].uid
    view.frame_selection(doc)

    view.hover_element = None
    view.draw(doc, RECT, 0.0)
    without = len(view._element_overlays(doc))
    assert view._overlays[uid]._hover_vaos == []

    view.hover_element = (uid, 2)
    hovered = len(view._element_overlays(doc))
    assert hovered == without + 1, "the hovered face is one more draw"

    view.hover_element = None
    assert len(view._element_overlays(doc)) == without
    assert view._overlays[uid]._hover_vaos == [], "and its buffer went with it"


def test_every_element_mode_hovers_without_a_rebuild(view) -> None:
    for mode in ("vertex", "edge", "face"):
        doc = _doc(count=1)
        doc.set_element_mode(mode)
        uid = doc.objects[0].uid
        view.frame_selection(doc)
        view.hover_element = (uid, 0)
        view.draw(doc, RECT, 0.0)
        overlay = view._overlays[uid]
        guides = list(overlay._vaos)

        view.hover_element = (uid, 1)
        view.draw(doc, RECT, 0.0)
        assert view._overlays[uid] is overlay, mode
        assert overlay._vaos == guides, mode
        assert len(overlay._hover_vaos) == 1, mode


def test_a_selection_change_still_rebuilds_the_overlay(view) -> None:
    """The key lost hover, not its job."""
    from warlock.studio.clay import elements as el

    doc = _doc(count=1)
    _face_mode(doc)
    uid = doc.objects[0].uid
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)
    overlay = view._overlays[uid]

    doc.set_element_sel(uid, el.ElementSel(faces=[0]))
    view.draw(doc, RECT, 0.0)
    assert view._overlays[uid] is not overlay


def test_a_hover_does_not_outlive_the_mesh_it_indexed(view) -> None:
    """``hover_element`` names an index into a particular mesh's elements. A
    keyboard op that shrinks the edge set left next frame's overlay reading
    ``edge_verts[hover]`` past the end -- an IndexError out of ``draw()`` on
    the frame loop -- and vertex mode sent the stale index to the GPU."""
    doc = _doc(count=1)
    doc.set_element_mode("edge")
    uid = doc.objects[0].uid
    view.draw(doc, RECT, 0.0)

    # The read-site guard: an out-of-range hover draws nothing, never raises.
    view.hover_element = (uid, 10_000)
    view._render_dirty = True
    view.draw(doc, RECT, 0.0)

    # The choke point: replacing the mesh clears a hover naming the old one.
    view.hover_element = (uid, 10_000)
    doc.set_mesh(uid, bp.plane())
    view.draw(doc, RECT, 0.0)
    assert view.hover_element is None


def test_a_projection_toggle_invalidates_the_screen_cache(view) -> None:
    """The projection *kind* moves every projected point without moving the
    camera, so a key without it served stale positions to pick, hover and the
    marquee until the camera happened to move (Ctrl+5)."""
    doc = _doc(count=1)
    uid = doc.objects[0].uid
    view.frame_selection(doc)
    view.draw(doc, RECT, 0.0)

    before = view.screen_of(doc, uid)
    assert view.screen_of(doc, uid) is before, "an unchanged camera hits"
    view.camera.orthographic = True
    assert view.screen_of(doc, uid) is not before, "the toggle misses"


def test_the_nearer_of_two_overlapping_edges_is_picked(view) -> None:
    """Edge mode ranked candidates with a constant key, so with two objects'
    edges under the cursor the earlier one in ``doc.objects`` always won --
    here the far box, added first, one twentieth of a unit behind."""
    from warlock.studio.clay import pick as clay_pick
    from warlock.studio.clay.adjacency import adjacency

    doc = bd.ClayDoc()
    far = doc.add_object(
        bd.Obj(uid=bd.new_uid(), name="far", mesh=bp.box(), translation=m3.vec3(0, 0, -0.05))
    )
    near = doc.add_object(bd.Obj(uid=bd.new_uid(), name="near", mesh=bp.box()))
    doc.element_mode = "edge"

    view._rect = RECT
    view.camera.set_target(m3.vec3())
    view.camera.set_position(m3.vec3(0.0, 0.0, 8.0))
    view.camera.aspect = RECT[2] / RECT[3]

    screen = view.screen_of(doc, near.uid)
    edges = adjacency(near.mesh).edge_verts
    front = [
        i
        for i, (a, b) in enumerate(edges)
        if near.mesh.positions[a][2] > 0 and near.mesh.positions[b][2] > 0
    ]
    assert front, "a box has front-face edges"
    a, b = edges[front[0]]
    # Three pixels outside the silhouette, so the ray misses both boxes and
    # the occlusion test admits every edge in reach: the ranking alone decides.
    mid = (screen.xy[a] + screen.xy[b]) * 0.5
    centre = np.array([RECT[2] * 0.5, RECT[3] * 0.5])
    outward = (mid - centre) / np.linalg.norm(mid - centre)
    point = tuple(mid + outward * 3.0)
    assert view.pick_face(doc, point) is None
    # Both boxes offer an edge here; the test is only meaningful if they do.
    far_screen = view.screen_of(doc, far.uid)
    assert clay_pick.nearest_edge(far_screen, adjacency(far.mesh).edge_verts, point) is not None

    picked = view.pick_element(doc, point)
    assert picked is not None and picked[0] == near.uid


# --- the pivot, and the keyboard's half of a drag (Clay W3) -------------------


def test_a_rotate_orbits_the_selection_median_rather_than_each_origin(view) -> None:
    """The picture used to lie.

    The gizmo is drawn at ``selection_centre`` -- the median of what is
    selected -- while ``_apply`` wrote only ``obj.rotation``, so dragging the
    ring with two objects selected spun each of them in place around a ring
    drawn somewhere neither of them was.
    """
    doc = _doc(count=2)
    uids = [obj.uid for obj in doc.objects]
    doc.select(uids)
    before = [np.array(doc.by_uid(uid).translation) for uid in uids]
    pivot = view.selection_centre(doc)

    view._begin_gizmo_drag(doc)
    view._drag_origin = np.asarray(pivot, dtype="f8")
    half_turn = m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), np.pi)
    for uid, was in view._drag_start.items():
        view._apply(doc.by_uid(uid), was, half_turn, view.state)

    after = [np.array(doc.by_uid(uid).translation) for uid in uids]
    # A half turn about the median swaps two objects placed either side of it.
    assert np.allclose(after[0], before[1], atol=1e-6), (before, after)
    assert np.allclose(after[1], before[0], atol=1e-6)


def test_a_rotate_of_one_object_about_its_own_centre_leaves_it_put(view) -> None:
    """The correction must not move what was already right: an object whose
    origin *is* the pivot stays where it is, whatever it is turned by."""
    doc = _doc(count=1)
    uid = doc.objects[0].uid
    doc.select([uid])
    before = np.array(doc.by_uid(uid).translation)

    view._begin_gizmo_drag(doc)
    view._drag_origin = before.astype("f8")
    turn = m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), 0.7)
    view._apply(doc.by_uid(uid), view._drag_start[uid], turn, view.state)

    assert np.allclose(doc.by_uid(uid).translation, before, atol=1e-9)


def test_a_scale_moves_the_objects_toward_the_pivot(view) -> None:
    doc = _doc(count=2)
    uids = [obj.uid for obj in doc.objects]
    doc.select(uids)
    pivot = np.asarray(view.selection_centre(doc), dtype="f8")
    before = [np.array(doc.by_uid(uid).translation) for uid in uids]

    view.state.tool = "scale"
    view._begin_gizmo_drag(doc)
    view._drag_origin = pivot
    for uid, was in view._drag_start.items():
        view._apply(doc.by_uid(uid), was, np.array([0.5, 0.5, 0.5]), view.state)

    for uid, was in zip(uids, before, strict=True):
        assert np.allclose(
            doc.by_uid(uid).translation, pivot + (was - pivot) * 0.5, atol=1e-9
        )
    view.state.tool = "select"


def test_g_starts_a_drag_with_no_handle_grabbed(view) -> None:
    """Every transform went through grabbing a coloured arrow, which means
    finding it, which means never moving an object without first looking at the
    gizmo rather than at the model."""
    doc = _doc(count=1)
    doc.select([doc.objects[0].uid])
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (64.0, 48.0)

    assert view.begin_keyboard_drag(doc, "move")
    assert view.dragging
    assert view._grab == "keydrag"


def test_a_keyboard_drag_moves_the_selection_as_the_pointer_moves(view) -> None:
    doc = _doc(count=1)
    uid = doc.objects[0].uid
    doc.select([uid])
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (64.0, 48.0)
    before = np.array(doc.by_uid(uid).translation)

    view.begin_keyboard_drag(doc, "move")
    view._motion(doc, (90.0, 48.0))

    assert not np.allclose(doc.by_uid(uid).translation, before)


def test_a_keyboard_drag_needs_something_to_drag(view) -> None:
    doc = _doc(count=1)
    doc.select([])
    view.draw(doc, RECT, 0.0)

    assert not view.begin_keyboard_drag(doc, "move")
    assert not view.dragging


def test_a_keyboard_drag_will_not_start_over_a_live_one(view) -> None:
    doc = _doc(count=1)
    doc.select([doc.objects[0].uid])
    view.draw(doc, RECT, 0.0)

    assert view.begin_keyboard_drag(doc, "move")
    assert not view.begin_keyboard_drag(doc, "scale")


def test_escape_puts_a_keyboard_drag_back(view) -> None:
    doc = _doc(count=1)
    uid = doc.objects[0].uid
    doc.select([uid])
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (64.0, 48.0)
    before = np.array(doc.by_uid(uid).translation)

    view.begin_keyboard_drag(doc, "move")
    view._motion(doc, (100.0, 60.0))
    assert view.cancel_drag(doc)

    assert np.allclose(doc.by_uid(uid).translation, before)
    assert not view.dragging
    assert view._key_kind == ""


def test_a_left_press_commits_a_keyboard_drag(view) -> None:
    """It holds no button, so a press is how it ends rather than a release."""
    doc = _doc(count=1)
    uid = doc.objects[0].uid
    doc.select([uid])
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (64.0, 48.0)
    head = doc.history.head

    view.begin_keyboard_drag(doc, "move")
    view._motion(doc, (100.0, 48.0))
    view._press(doc, 1, (100.0, 48.0))

    assert not view.dragging
    assert doc.history.head != head, "one step for the whole gesture"


def test_a_right_press_cancels_one(view) -> None:
    doc = _doc(count=1)
    uid = doc.objects[0].uid
    doc.select([uid])
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (64.0, 48.0)
    before = np.array(doc.by_uid(uid).translation)

    view.begin_keyboard_drag(doc, "move")
    view._motion(doc, (100.0, 48.0))
    view._press(doc, 3, (100.0, 48.0))

    assert not view.dragging
    assert np.allclose(doc.by_uid(uid).translation, before)


def test_an_axis_lock_narrows_a_keyboard_drag(view) -> None:
    """The lock, the typed value and the snap are ``_narrow``'s, above both
    drag paths, so neither has to learn what a lock is."""
    doc = _doc(count=1)
    uid = doc.objects[0].uid
    doc.select([uid])
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (64.0, 48.0)
    before = np.array(doc.by_uid(uid).translation)

    view.begin_keyboard_drag(doc, "move")
    assert view.drag_key(doc, "x")
    view._motion(doc, (100.0, 70.0))

    moved = doc.by_uid(uid).translation - before
    assert abs(moved[1]) < 1e-9 and abs(moved[2]) < 1e-9, moved


def test_r_mid_drag_switches_the_transform_and_starts_it_afresh(view) -> None:
    """A rotate that began from a half-finished move would carry that move into
    its result, with no way to undo one without the other."""
    doc = _doc(count=1)
    uid = doc.objects[0].uid
    doc.select([uid])
    view.draw(doc, RECT, 0.0)
    view._last_mouse = (64.0, 48.0)
    before = np.array(doc.by_uid(uid).translation)

    view.begin_keyboard_drag(doc, "move")
    view._motion(doc, (100.0, 48.0))
    assert not np.allclose(doc.by_uid(uid).translation, before)

    assert view.drag_key(doc, "r")
    assert view._key_kind == "rotate"
    assert np.allclose(doc.by_uid(uid).translation, before), "the move was undone"


def test_switching_to_the_transform_already_running_does_nothing(view) -> None:
    doc = _doc(count=1)
    doc.select([doc.objects[0].uid])
    view.draw(doc, RECT, 0.0)

    view.begin_keyboard_drag(doc, "move")
    anchor = view._key_anchor
    view.drag_key(doc, "g")

    assert view._key_kind == "move"
    assert anchor is view._key_anchor
