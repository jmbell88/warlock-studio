"""The Build viewport: uploads, the cache, picking and framing.

The cache is the interesting half. The 3D pane shows one loaded GLB and can
rebuild the lot on any change; a Build document is many objects, one of which
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

from warlock.studio import build_view
from warlock.studio.build import document as bd
from warlock.studio.build import primitives as bp
from warlock.studio.viewer import math3d as m3


class _State:
    def __init__(self, tool: str = "select") -> None:
        self.tool = tool
        self.snap = False
        self.snap_translate = 0.125
        self.snap_rotate = 15.0


class _Ctx:
    """What ``BuildView`` reads from its ctx: a moderngl context and the mode
    state it takes the current tool from."""

    def __init__(self, gl: Any, tool: str = "select") -> None:
        self._gl = gl
        self.state = type("S", (), {"build": _State(tool)})()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gl, name)


def _doc(*, count: int = 2) -> bd.BuildDoc:
    doc = bd.BuildDoc()
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
    v = build_view.BuildView(_Ctx(gl))
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
    view.draw(bd.BuildDoc(), RECT, 0.0)
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
    v = build_view.BuildView(_Ctx(gl))
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
    v = build_view.BuildView(_Ctx(gl))
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
    doc = bd.BuildDoc()
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
    assert view.frame_selection(bd.BuildDoc()) == 0.0


def test_world_bounds_accounts_for_the_transform(view) -> None:
    doc = bd.BuildDoc()
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
    view.ctx.state.build.tool = "move"
    assert view.active_gizmo(_doc(count=1)) is None


@pytest.mark.parametrize(
    ("tool", "attr"),
    [("move", "translate_gizmo"), ("rotate", "rotate_gizmo"), ("scale", "scale_gizmo")],
)
def test_each_transform_tool_drives_its_own_gizmo(view, tool: str, attr: str) -> None:
    doc = _doc(count=1)
    doc.select([doc.objects[0].uid])
    view.ctx.state.build.tool = tool
    assert view.active_gizmo(doc) is getattr(view, attr)


def test_a_gizmo_drag_records_one_history_step_per_object(view) -> None:
    """Applied in place while dragging and committed on release: a step per
    mouse-move would fill the undo stack with a hundred entries for one drag,
    and the intermediate positions are not states worth stepping back through."""
    import pygame

    doc = _doc(count=1)
    obj = doc.objects[0]
    doc.select([obj.uid])
    view.ctx.state.build.tool = "move"
    view._rect = RECT

    view._grab = "gizmo"
    view._drag_start = {obj.uid: tuple(np.array(v, copy=True) for v in obj.trs())}
    depth = len(doc.history)
    obj.translation = np.array([2.0, 0.0, 0.0])  # what the drag would have done

    view.handle_event(doc, pygame.event.Event(pygame.MOUSEBUTTONUP, button=1), False)
    assert len(doc.history) == depth + 1

    doc.undo()
    assert np.allclose(obj.translation, [0.0, 0.0, 0.0])


def test_a_drag_on_an_object_deleted_midway_does_not_raise(view) -> None:
    import pygame

    doc = _doc(count=1)
    obj = doc.objects[0]
    doc.select([obj.uid])
    view.ctx.state.build.tool = "move"
    view._grab = "gizmo"
    view._drag_start = {obj.uid: tuple(np.array(v, copy=True) for v in obj.trs())}

    doc.remove_object(obj.uid)
    view.handle_event(doc, pygame.event.Event(pygame.MOUSEBUTTONUP, button=1), False)
