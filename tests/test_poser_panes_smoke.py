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


# --- the clip editor ----------------------------------------------------------


def _library() -> dict:
    """A clip library shaped exactly as ``service.clips.library`` returns one."""
    return {
        "template": "humanoid",
        "space": "delta",
        "edited": False,
        "poses": [
            {"name": "contact A", "bones": {"hips": [0.0, 0.0, 0.0, 1.0]}},
            {"name": "passing", "bones": {"spine": [0.0, 0.0, 0.0, 1.0]}},
            {"name": "contact B", "bones": {"arm.L": [0.0, 0.0, 0.0, 1.0]}},
        ],
        "clips": [
            {
                "name": "walk",
                "keys": ["contact A", "passing", "contact B"],
                "segments": [2, 2, 2],
                "closed": True,
                "easing": "linear",
                "space": "delta",
            }
        ],
    }


def test_the_clip_pane_builds_without_rigging(app_ctx, imgui_ctx):
    from warlock.studio.panes import poser_clips

    app_ctx.rigging_available = False
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))


def test_the_clip_pane_builds_for_a_skeleton_with_no_clips(app_ctx, imgui_ctx):
    """The state every template but the humanoid opens in, and the one a
    collapsed heading would hide."""
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_clips

    app_ctx.rigging_available = True
    state = poser_mode.ensure(app_ctx)
    state.clips = {"template": state.template, "clips": [], "poses": [], "edited": False}
    state.clips_dirty_flag = False
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))


def test_the_clip_pane_builds_over_a_clip_and_a_session(app_ctx, imgui_ctx):
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_clips

    app_ctx.rigging_available = True
    app_ctx.poser_viewer = _PoserViewer()
    state = poser_mode.ensure(app_ctx)
    state.clips_dirty_flag = False
    poser_mode.adopt_clips(app_ctx, _library())
    assert state.clip == "walk"
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))


def test_the_clip_pane_builds_while_scrubbing_and_while_unsaved(app_ctx, imgui_ctx):
    """Three branches that only exist in the middle of a session: the
    in-between banner, the Back-to-key button and the unsaved marker."""
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_clips

    app_ctx.rigging_available = True
    app_ctx.poser_viewer = _PoserViewer()
    state = poser_mode.ensure(app_ctx)
    state.clips_dirty_flag = False
    poser_mode.adopt_clips(app_ctx, _library())
    poser_mode.scrub(app_ctx, 3)
    assert state.frame == 3
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))

    state.clips_unsaved = True
    state.clips["edited"] = True
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))


def test_the_clip_pane_builds_when_the_clip_will_not_expand(app_ctx, imgui_ctx):
    """Mid-edit inconsistency is ordinary -- the segments briefly do not match
    the keys -- and it must draw a reason rather than raise into a frame."""
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_clips

    app_ctx.rigging_available = True
    app_ctx.poser_viewer = _PoserViewer()
    state = poser_mode.ensure(app_ctx)
    state.clips_dirty_flag = False
    poser_mode.adopt_clips(app_ctx, _library())
    state.open_clip()["segments"] = [2]
    poser_mode.rebuild_frames(app_ctx)
    assert state.clips_error and not state.frames
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))


def test_drawing_the_clip_pane_pumps_its_own_refresh_flag(app_ctx, imgui_ctx):
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_clips

    app_ctx.rigging_available = True
    state = poser_mode.ensure(app_ctx)
    state.clips_dirty_flag = True
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))
    assert state.clips_dirty_flag is False
    assert state.clips_loading is True


def test_an_unsaved_editor_is_never_reloaded_underneath_the_user(app_ctx, imgui_ctx):
    """A background refresh landing on unsaved keys would discard them without
    anyone asking, so the pump refuses while there is work in the editor."""
    from warlock.studio import poser_mode
    from warlock.studio.panes import poser_clips

    app_ctx.rigging_available = True
    state = poser_mode.ensure(app_ctx)
    poser_mode.adopt_clips(app_ctx, _library())
    state.clips_unsaved = True
    state.clips_dirty_flag = True
    _frame(imgui_ctx, lambda: poser_clips.draw(app_ctx))
    assert state.clips_loading is False
    assert state.clips_dirty_flag is True, "still wanted, just not now"
