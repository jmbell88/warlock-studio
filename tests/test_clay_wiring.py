"""C2: the live ClayView must be reachable *through the ctx*.

Eight call sites read ``getattr(ctx, "clay_view", None)`` -- the drag keyboard
(axis locks, typed values, Esc, Enter), the Ctrl+digit axis views, Frame in the
ops registry, the wireframe toggle, and the per-tab camera that ``camera_of``
refreshes on every save. Every one of them was inert, because the view lived
only on ``App.clay_view`` and ``Ctx`` never carried the field: ``getattr`` with
a default is precisely the spelling that turns a missing wire into silence
rather than an AttributeError. The tests here pin the mirror at both ends --
the field exists, ``_ensure_build_view`` assigns it, teardown clears it -- and
then walk the call sites against a stub view hung on a stub ctx, proving they
work the moment the wire is there.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest

from warlock.studio import clay_mode, clay_state, main
from warlock.studio.app_ctx import Ctx
from warlock.studio.clay import document as bd
from warlock.studio.clay import primitives as bp


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    """``pygame.key.get_mods`` needs a video system; there is none in a test."""
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


def _tab_ctx() -> SimpleNamespace:
    """A ctx with one open document, no GL and no App anywhere near it."""
    doc = bd.ClayDoc()
    doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    state = clay_state.ClayState()
    state.add(clay_state.ClayTab(doc=doc, saved_head=doc.history.head))
    return SimpleNamespace(state=SimpleNamespace(clay=state), clay_view=None)


class _StubView:
    """What the keyboard path needs a view to be: dragging, and recording."""

    def __init__(self) -> None:
        self.dragging = True
        self.calls: list[tuple[str, Any]] = []
        self.camera = None

    def cancel_drag(self, doc: Any) -> None:
        self.calls.append(("cancel", doc))

    def _release_drag(self, doc: Any) -> None:
        self.calls.append(("release", doc))

    def drag_key(self, doc: Any, name: str) -> bool:
        self.calls.append(("key", name))
        return True


# --- the field and the mirror ------------------------------------------------


def test_the_ctx_declares_a_clay_view_field_defaulting_to_none() -> None:
    """The call sites read ``getattr(ctx, "clay_view", None)``; a field the
    dataclass does not declare is one nothing ever assigns, which is exactly
    how all eight of them came to be inert."""
    fields = {f.name: f for f in dataclasses.fields(Ctx)}
    assert "clay_view" in fields
    assert fields["clay_view"].default is None


def test_ensure_build_view_mirrors_the_view_onto_the_ctx(monkeypatch) -> None:
    """The App owns the view; the panes and clay_mode only ever see the ctx.
    Without the mirror the two disagree for the life of the process."""
    from warlock.studio import clay_view as cv

    built: list[Any] = []

    class _Fake:
        def __init__(self, ctx: Any, app_ctx: Any = None) -> None:
            built.append(self)

    monkeypatch.setattr(cv, "ClayView", _Fake)
    app = SimpleNamespace(clay_view=None, ctx=object(), app_ctx=SimpleNamespace(clay_view=None))

    view = main.App._ensure_build_view(app)
    assert view is app.clay_view
    assert app.app_ctx.clay_view is app.clay_view

    # A second call returns the same view and keeps the mirror intact.
    assert main.App._ensure_build_view(app) is view
    assert app.app_ctx.clay_view is view
    assert len(built) == 1


def test_teardown_clears_the_ctx_mirror(monkeypatch) -> None:
    """A released view left on the ctx would hand the call sites dead GL
    objects, where None is the answer every one of them already refuses."""
    released: list[bool] = []
    ctx = SimpleNamespace(
        settings=SimpleNamespace(flush=lambda: None),
        textures=None,
        clay_view=SimpleNamespace(release=lambda: released.append(True)),
    )
    app = SimpleNamespace(
        fps=SimpleNamespace(frames=0),
        app_ctx=ctx,
        viewer=None,
        clay_view=ctx.clay_view,
        poser_viewer=None,
        imgui_renderer=None,
        runtime=SimpleNamespace(shutdown=lambda: None),
    )
    main.App.teardown(app)
    assert released == [True]
    assert ctx.clay_view is None


# --- the call sites, against a view that is wired ----------------------------


def test_esc_reaches_cancel_drag_through_the_ctx() -> None:
    import pygame

    ctx = _tab_ctx()
    ctx.clay_view = _StubView()
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0)
    assert clay_mode.handle_key(ctx, event) is True
    assert [c[0] for c in ctx.clay_view.calls] == ["cancel"]


def test_enter_reaches_release_drag_through_the_ctx() -> None:
    import pygame

    ctx = _tab_ctx()
    ctx.clay_view = _StubView()
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0)
    assert clay_mode.handle_key(ctx, event) is True
    assert [c[0] for c in ctx.clay_view.calls] == ["release"]


def test_an_axis_key_reaches_drag_key_through_the_ctx() -> None:
    """X during a drag is the axis lock, not a tool key -- but only while the
    drag's owner is reachable, which is the whole of C2."""
    import pygame

    ctx = _tab_ctx()
    ctx.clay_view = _StubView()
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_x, mod=0)
    assert clay_mode.handle_key(ctx, event) is True
    assert ("key", "x") in ctx.clay_view.calls


def test_camera_of_refreshes_the_tab_from_the_live_camera() -> None:
    """The half broken since v0.0.13: saving the tab you are looking at must
    store where the camera is *now*, not wherever it sat when you last
    switched away."""
    camera = SimpleNamespace(theta=1.5, phi=0.25, distance=9.0, target=(1.0, 2.0, 3.0))
    ctx = SimpleNamespace(clay_view=SimpleNamespace(camera=camera))
    tab = clay_state.ClayTab(doc=bd.ClayDoc())

    view = clay_mode.camera_of(ctx, tab)
    assert view is tab.view
    assert tab.view.yaw == 1.5
    assert tab.view.pitch == 0.25
    assert tab.view.distance == 9.0
    assert tab.view.target == (1.0, 2.0, 3.0)


def test_camera_of_without_a_view_still_answers_from_the_tab() -> None:
    """Headless callers -- and the first frame, before the viewport exists --
    get the stored view rather than a raise."""
    tab = clay_state.ClayTab(doc=bd.ClayDoc())
    tab.view.yaw = 0.9
    assert clay_mode.camera_of(SimpleNamespace(clay_view=None), tab).yaw == 0.9


# --- H20: the GPU cache's identity key ---------------------------------------
#
# An entry is keyed on ``id(obj.mesh)``, which is sound only while that id
# cannot be recycled: nothing in the entry's ``model`` keeps the Mesh
# *instance* alive (``_submesh`` builds new arrays on the way to the GPU), so a
# freed mesh's address coming back on different geometry would make
# ``entry.key == key`` true of a stale upload -- the viewport drawing the old
# shape forever with nothing in the data to say why.


def test_a_gpu_entry_holds_the_mesh_its_key_names(monkeypatch) -> None:
    from warlock.studio import clay_view as cv

    monkeypatch.setattr(
        cv.scenelib, "GpuModel", lambda ctx, model: SimpleNamespace(release=lambda: None)
    )
    doc = bd.ClayDoc()
    obj = doc.add_object(bd.Obj(uid=bd.new_uid(), name="A", mesh=bp.box()))
    key = cv._object_key(obj, cv._materials_key(doc))

    entry = cv.ClayView._build(SimpleNamespace(ctx=None), obj, doc, key)
    assert entry.mesh is obj.mesh
    assert entry.key[0] == id(entry.mesh), "the held mesh is the one the key identifies"


# --- the primitive icon table ------------------------------------------------


def test_primitive_icons_covers_every_generator_exactly() -> None:
    """The ``.get(name, icons.BOX)`` fallback keeps the pane drawing when the
    table is short -- which is also what hides the shortfall, so a tenth
    generator would silently wear a box forever. This is the coverage check the
    fallback cannot be: one icon per generator, and no icon for a generator
    that no longer exists."""
    from warlock.studio.panes import clay_tools

    assert set(clay_tools.PRIMITIVE_ICONS) == set(bp.GENERATORS)


def test_every_camera_attribute_the_clay_view_reaches_exists():
    """``read_from``/``write_to`` name four private goal attributes on Camera,
    and the ``getattr`` fallbacks in ``read_from`` would turn a Camera rename
    into a silent wrong answer rather than an AttributeError. So every name the
    two methods reach is pinned against a real Camera here."""
    import inspect
    import re

    from warlock.studio.viewer.camera import Camera

    source = inspect.getsource(clay_state.CameraView.read_from) + inspect.getsource(
        clay_state.CameraView.write_to
    )
    names = set(re.findall(r"camera\.(\w+)", source))
    names |= set(re.findall(r'getattr\(camera, "(\w+)"', source))
    camera = Camera()
    missing = sorted(n for n in names if not hasattr(camera, n))
    assert not missing, f"CameraView names Camera attributes that do not exist: {missing}"
    # The scan found the private half at all -- an empty match set would pass
    # the assert above while checking nothing.
    assert "_goal_theta" in names
