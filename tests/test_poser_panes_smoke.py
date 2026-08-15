"""Poser's two panes, built for real, once -- the test_studio_smoke idiom.

Not a screenshot test: it asserts that a frame containing each pane can be
*built* in its empty, populated and mid-session states -- no missing begin/end
pair, no attribute that moved. The harness is test_studio_smoke's, module-scoped
here and restoring the previous imgui context on the way out, because two live
harnesses in one merged run would otherwise fight over the current-context
pointer.
"""

from __future__ import annotations

import pytest

from warlock.studio.app_ctx import Ctx
from warlock.studio.jobs_cache import JobsCache
from warlock.studio.settings import Settings
from warlock.studio.state import AppState
from warlock.studio.viewer import math3d as m3
from warlock.studio.viewer.gltf import Model, Node
from warlock.studio.viewer.pose import PoseEditor


@pytest.fixture(scope="module")
def imgui_ctx(gl):
    """An imgui context with a real renderer, over the standalone GL context.

    The renderer is needed even though nothing is presented: imgui 1.92 hands
    its font atlas to the backend, and a context whose backend never claims it
    cannot finish a frame.
    """
    from imgui_bundle import imgui

    from warlock.studio import imgui_backend, theme, widgets

    prev_ctx = imgui.get_current_context()
    prev_screen = type(gl).__dict__.get("screen")
    # A standalone context has no default framebuffer; the renderer targets
    # ctx.screen, so give it one that exists.
    fbo = gl.simple_framebuffer((1600, 950))
    fbo.use()
    type(gl).screen = property(lambda _self: fbo)

    # Every collapsing section forced open: a frame of collapsed headings would
    # build without ever touching the code this test exists to exercise.
    prev_force = widgets.FORCE_SECTIONS_OPEN
    widgets.FORCE_SECTIONS_OPEN = True

    ctx = imgui.create_context()
    io = imgui.get_io()
    # See ``test_studio_smoke``'s fixture: a persisted collapsed flag in
    # ``imgui.ini`` survives the process and silently empties every window.
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    theme.apply(imgui)
    renderer = imgui_backend.ImguiRenderer(gl)
    yield imgui, renderer
    renderer.shutdown()
    imgui.destroy_context(ctx)
    widgets.FORCE_SECTIONS_OPEN = prev_force
    if prev_screen is not None:
        type(gl).screen = prev_screen
    if prev_ctx is not None:
        imgui.set_current_context(prev_ctx)


@pytest.fixture
def app_ctx(gl, svc, tmp_path, imgui_ctx):
    from warlock.studio import textures
    from warlock.studio.runtime import Runtime
    from warlock.studio.tasks import TaskRunner
    from warlock.studio.viewer_embed import Viewer

    runtime = Runtime(svc.config)
    runtime.store = svc.store
    runtime.tasks = TaskRunner(workers=1)
    viewer = Viewer(gl)
    ctx = Ctx(
        svc=svc,
        runtime=runtime,
        state=AppState(),
        cache=JobsCache(svc),
        tasks=runtime.tasks,
        settings=Settings.load(tmp_path),
        viewer=viewer,
        textures=textures.ThumbnailCache(gl),
    )
    ctx.rig_default = "humanoid"
    yield ctx
    viewer.release()
    ctx.textures.release()
    runtime.tasks.shutdown(wait=False)


def _frame(imgui_ctx, build):
    """Run one complete imgui frame around ``build``."""
    imgui, renderer = imgui_ctx
    imgui.new_frame()
    imgui.set_next_window_size((1200, 900))
    imgui.begin("##host")
    build()
    imgui.end()
    imgui.render()
    renderer.render(imgui.get_draw_data())


class _PoserViewer:
    """The surface the panes touch, over a real editor on a meshless armature
    -- the shape the preview loads as, minus the file and the GL."""

    def __init__(self) -> None:
        nodes = [
            Node(name="rig", children=[1]),
            Node(name="hips", translation=m3.vec3(0.0, 0.53, 0.0), children=[2, 3, 4]),
            Node(name="spine", translation=m3.vec3(0.0, 0.07, 0.0)),
            Node(name="arm.L", translation=m3.vec3(-0.2, 0.1, 0.0)),
            Node(name="arm.R", translation=m3.vec3(0.2, 0.1, 0.0)),
        ]
        self.editor = PoseEditor()
        self.editor.bind(Model(nodes, roots=[0], meshes=[], skins=[]),
                         ["hips", "spine", "arm.L", "arm.R"])
        self.editor.mirror_pairs = [["arm.L", "arm.R"]]
        self.editor.root = "hips"
        self.pose_mode = True
        self.selected_bone = None


def test_the_poser_panes_build_without_rigging(app_ctx, imgui_ctx):
    """The Blender-missing branch, which is the state a bare install opens in."""
    from warlock.studio.panes import poser_controls, poser_library

    app_ctx.rigging_available = False
    _frame(imgui_ctx, lambda: poser_library.draw(app_ctx))
    _frame(imgui_ctx, lambda: poser_controls.draw(app_ctx))


def test_the_controls_pane_builds_while_the_preview_loads(app_ctx, imgui_ctx):
    from warlock.studio.panes import poser_controls

    app_ctx.rigging_available = True
    assert app_ctx.poser_viewer is None
    _frame(imgui_ctx, lambda: poser_controls.draw(app_ctx))


def test_the_poser_panes_build_with_a_session_and_a_library(app_ctx, imgui_ctx):
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_controls, poser_library

    app_ctx.rigging_available = True
    state = poser_mode.ensure(app_ctx)
    state.poses = [
        {"id": "0123456789ab", "name": "Crouch", "bones": {}},
        {"id": "0123456789ac", "name": "Leap", "bones": {}},
    ]
    state.presets = [{"name": "idle", "bones": {}}]
    app_ctx.poser_viewer = _PoserViewer()
    _frame(imgui_ctx, lambda: poser_library.draw(app_ctx))
    _frame(imgui_ctx, lambda: poser_controls.draw(app_ctx))

    # And mid-edit: the unsaved banner, the highlighted row, the root-selected
    # checkbox and the nonzero offset line are each their own branch.
    viewer = app_ctx.poser_viewer
    viewer.editor.current = "0123456789ab"
    viewer.editor.dirty = True
    viewer.editor.selected = "hips"
    viewer.selected_bone = "hips"
    viewer.editor.root_translate = True
    viewer.editor.set_root_translation([0.1, 0.0, 0.2])
    _frame(imgui_ctx, lambda: poser_library.draw(app_ctx))
    _frame(imgui_ctx, lambda: poser_controls.draw(app_ctx))


def test_drawing_the_library_pane_pumps_the_refresh_flag(app_ctx, imgui_ctx):
    """The per-frame half of the refresh idiom is wired through this pane's
    draw, so a drawn frame with the flag up must submit the list."""
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_library

    app_ctx.rigging_available = True
    state = poser_mode.ensure(app_ctx)
    state.refresh_dirty = True
    _frame(imgui_ctx, lambda: poser_library.draw(app_ctx))
    assert state.refresh_dirty is False, "cleared because the submit was accepted"
    assert state.loading is True
