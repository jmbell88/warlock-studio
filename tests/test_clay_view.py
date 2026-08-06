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
