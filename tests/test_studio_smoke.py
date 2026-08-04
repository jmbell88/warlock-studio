"""Every pane, built for real, once.

Not a screenshot test -- it asserts nothing about what anything looks like. It
asserts that a frame containing every panel can be *built*: no missing imgui
begin/end pair, no attribute that moved, no pane reaching for state that is
only set once a job exists. That class of mistake is otherwise found by opening
the app, and there are eight panels.
"""

from __future__ import annotations

import pytest

from warlock.service import jobs as svc_jobs
from warlock.studio.app_ctx import Ctx
from warlock.studio.jobs_cache import JobsCache
from warlock.studio.settings import Settings
from warlock.studio.state import AppState, Eta


@pytest.fixture(scope="session")
def imgui_ctx(gl):
    """An imgui context with a real renderer, over the standalone GL context.

    The renderer is needed even though nothing is presented: imgui 1.92 hands
    its font atlas to the backend, and a context whose backend never claims it
    cannot finish a frame.
    """
    from imgui_bundle import imgui

    from warlock.studio import imgui_backend, theme

    # A standalone context has no default framebuffer; the renderer targets
    # ctx.screen, so give it one that exists.
    fbo = gl.simple_framebuffer((1600, 950))
    fbo.use()
    type(gl).screen = property(lambda _self: fbo)

    # Every collapsing section forced open: a frame of collapsed headings would
    # build without ever touching the code this test exists to exercise.
    from warlock.studio import widgets

    widgets.FORCE_SECTIONS_OPEN = True

    imgui.create_context()
    io = imgui.get_io()
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    theme.apply(imgui)
    renderer = imgui_backend.ImguiRenderer(gl)
    yield imgui, renderer
    renderer.shutdown()
    imgui.destroy_context()


@pytest.fixture
def app_ctx(gl, svc, tmp_path, imgui_ctx):
    from warlock.service import sheets as svc_sheets
    from warlock.service import system as svc_system
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
    ctx.guidance = svc_system.guidance_catalog()
    ctx.sheet_options = svc_sheets.sheet_options()
    ctx.base_models = [("turbo", "SDXL-Turbo")]
    ctx.style_loras = [("", "none"), ("render3d", "3D render")]
    ctx.rig_templates = [{"key": "biped", "label": "Biped"}]
    ctx.rig_default = "biped"
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


def _seeded(ctx, **overrides):
    """A finished job with a mesh, selected."""
    job_id = svc_jobs.create_job(ctx.svc, kind="text", prompt="a barrel")["id"]
    job_dir = ctx.svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"x")
    (job_dir / "model.glb").write_bytes(b"x")
    ctx.svc.store.set_status(job_id, "done")
    if overrides:
        ctx.svc.store.merge_params(job_id, overrides)
    ctx.cache.invalidate()
    ctx.cache.tick()
    ctx.state.select(job_id)
    return job_id


# --- panes ------------------------------------------------------------------


def test_the_2d_pane_builds_with_an_empty_form(app_ctx, imgui_ctx):
    from warlock.studio.panes import settings_2d

    _frame(imgui_ctx, lambda: settings_2d.draw(app_ctx))


def test_the_2d_pane_builds_with_advanced_open_and_a_lora_chosen(app_ctx, imgui_ctx):
    from warlock.studio.panes import settings_2d

    app_ctx.state.form_2d["style_lora"] = "render3d"
    app_ctx.state.form_2d["prompt"] = "a barrel"
    app_ctx.state.preview = {"prompt": "a barrel, fantasy", "tokens": 12, "chunks": 1}
    app_ctx.state.history = ["a barrel", "a sword"]
    _frame(imgui_ctx, lambda: settings_2d.draw(app_ctx))


def test_the_3d_pane_builds_with_and_without_rigging(app_ctx, imgui_ctx):
    from warlock.studio.panes import settings_3d

    _frame(imgui_ctx, lambda: settings_3d.draw(app_ctx))
    app_ctx.rigging_available = True
    app_ctx.state.form_3d["rig"] = True
    _frame(imgui_ctx, lambda: settings_3d.draw(app_ctx))


def test_the_library_builds_empty_and_populated(app_ctx, imgui_ctx):
    from warlock.studio.panes import library

    _frame(imgui_ctx, lambda: library.draw(app_ctx))
    job_id = _seeded(app_ctx)
    app_ctx.state.checked.add(job_id)
    app_ctx.export_dir = "D:/somewhere"
    _frame(imgui_ctx, lambda: library.draw(app_ctx))


def test_the_inspector_builds_for_every_status(app_ctx, imgui_ctx):
    from warlock.studio.panes import inspector

    _frame(imgui_ctx, lambda: inspector.draw(app_ctx))  # nothing selected
    job_id = _seeded(app_ctx, mesh_report={"verdict": "good", "reasons": []})
    _frame(imgui_ctx, lambda: inspector.draw(app_ctx))
    app_ctx.state.mode = "3d"
    _frame(imgui_ctx, lambda: inspector.draw(app_ctx))
    app_ctx.svc.store.set_status(job_id, "error", "it broke")
    app_ctx.cache.invalidate()
    app_ctx.cache.tick()
    _frame(imgui_ctx, lambda: inspector.draw(app_ctx))


def test_the_retarget_panel_builds_with_and_without_a_reconstruction(
    app_ctx, imgui_ctx, monkeypatch
):
    from warlock.studio.panes import retarget_panel

    job_id = _seeded(app_ctx)
    app_ctx.state.mode = "3d"
    job = app_ctx.cache.get(job_id)
    # No source.glb: there is nothing to rebuild from and the section hides.
    _frame(imgui_ctx, lambda: retarget_panel.draw(app_ctx, job))

    (app_ctx.svc.job_dir(job_id) / "source.glb").write_bytes(b"x")
    (app_ctx.svc.job_dir(job_id) / "rig.glb").write_bytes(b"x")
    app_ctx.cache.invalidate()
    app_ctx.cache.tick()
    rigged = app_ctx.cache.get(job_id)
    _frame(imgui_ctx, lambda: retarget_panel.draw(app_ctx, rigged))

    # And, with the binary present, the full tier list plus the custom budget's
    # number input. Without it the pane pins itself to "raw", so this path is
    # otherwise unreachable in a checkout that has no vendored gltfpack.
    monkeypatch.setattr(retarget_panel, "_gltfpack_available", lambda ctx: True)
    retarget_panel._form(app_ctx, job_id)["profile"] = "custom"
    _frame(imgui_ctx, lambda: retarget_panel.draw(app_ctx, rigged))


def test_the_pose_panel_builds_rigged_and_unrigged(app_ctx, imgui_ctx):
    from warlock.studio.panes import pose_panel

    job_id = _seeded(app_ctx)
    job = app_ctx.cache.get(job_id)
    _frame(imgui_ctx, lambda: pose_panel.draw(app_ctx, job))
    job["files"] = ["model.glb", "rig.glb"]
    app_ctx.state.preview["poses"] = [{"id": "abcdef012345", "name": "idle", "bones": {}}]
    _frame(imgui_ctx, lambda: pose_panel.draw(app_ctx, job))


def test_the_sheet_panel_builds_with_poses_and_a_clip(app_ctx, imgui_ctx):
    from warlock.studio.panes import sheet_panel

    job_id = _seeded(app_ctx)
    job = app_ctx.cache.get(job_id)
    job["files"] = ["model.glb", "rig.glb"]
    app_ctx.state.preview["poses"] = [
        {"id": "abcdef012345", "name": "idle"},
        {"id": "abcdef012346", "name": "walk"},
    ]
    _frame(imgui_ctx, lambda: sheet_panel.draw(app_ctx, job))
    form = app_ctx.state.preview["sheet_form"]
    form["clip"] = True
    form["clip_from"] = "abcdef012345"
    form["clip_to"] = "abcdef012346"
    _frame(imgui_ctx, lambda: sheet_panel.draw(app_ctx, job))


def test_the_overlay_builds_with_a_toolbar_and_a_banner(app_ctx, imgui_ctx):
    from warlock.studio.panes import overlay

    eta = Eta()
    app_ctx.state.last_error = "trellis: the exe is missing"
    _frame(
        imgui_ctx,
        lambda: (overlay.doctor_banner(app_ctx), overlay.toolbar(app_ctx),
                 overlay.progress_card(app_ctx, eta), overlay.placeholder(app_ctx)),
    )


def test_toasts_and_dialogs_build(app_ctx, imgui_ctx):
    from warlock.studio import dialogs, widgets

    app_ctx.state.toast("finished", "info")
    app_ctx.state.toast("it broke", "error")
    app_ctx.confirms.ask(dialogs.Confirm(title="Sure?", message="Really?"))
    app_ctx.prompts.ask(dialogs.Prompt(title="Name it", label="Name"))
    imgui, renderer = imgui_ctx
    imgui.new_frame()
    widgets.toasts(app_ctx.state, (1600, 950))
    app_ctx.confirms.draw()
    app_ctx.prompts.draw()
    imgui.render()
    renderer.render(imgui.get_draw_data())


def test_the_whole_frame_builds_at_once(app_ctx, imgui_ctx):
    """The real layout: three panes side by side, as main.py assembles them."""
    from warlock.studio.panes import inspector, library, overlay, settings_2d

    _seeded(app_ctx)
    imgui, renderer = imgui_ctx

    def build():
        imgui.begin_child("settings", (340, 0), imgui.ChildFlags_.borders.value)
        settings_2d.draw(app_ctx)
        imgui.separator()
        library.draw(app_ctx)
        imgui.end_child()
        imgui.same_line()
        imgui.begin_child("viewport", (400, 0), imgui.ChildFlags_.borders.value)
        overlay.toolbar(app_ctx)
        overlay.placeholder(app_ctx)
        imgui.end_child()
        imgui.same_line()
        imgui.begin_child("inspector", (340, 0), imgui.ChildFlags_.borders.value)
        inspector.draw(app_ctx)
        imgui.end_child()

    _frame(imgui_ctx, build)
    del renderer


def test_the_landing_screen_builds_in_each_of_its_views(app_ctx, imgui_ctx):
    from warlock.studio.panes import landing

    _frame(imgui_ctx, lambda: landing.draw(app_ctx))
    app_ctx.state.landing_view = "open"
    _seeded(app_ctx)
    _frame(imgui_ctx, lambda: landing.draw(app_ctx))
    app_ctx.state.landing_view = "profiles"
    _frame(imgui_ctx, lambda: landing.draw(app_ctx))


def test_the_profile_manager_builds_listing_and_editing(app_ctx, imgui_ctx):
    from warlock.studio import profiles
    from warlock.studio.panes import profiles_panel

    _frame(imgui_ctx, lambda: profiles_panel.draw(app_ctx))  # nothing saved yet
    profiles.save_profile(app_ctx.settings, "props", {"base_model": "turbo", "genre": "fantasy"})
    profiles.set_active(app_ctx.settings, "props")
    _frame(imgui_ctx, lambda: profiles_panel.draw(app_ctx))
    app_ctx.state.profile_draft = profiles.capture(app_ctx.state.form_2d)
    app_ctx.state.profile_draft["style_lora"] = "render3d"
    app_ctx.state.profile_draft_name = "props"
    _frame(imgui_ctx, lambda: profiles_panel.draw(app_ctx))


def test_the_2d_pane_builds_with_a_reference_chosen(app_ctx, imgui_ctx):
    """The conditioning group is hidden until ref_path is set, so the empty-form
    smoke test never reaches it."""
    from warlock.studio.panes import settings_2d

    form = app_ctx.state.form_2d
    form["prompt"] = "a barrel"
    form["ref_path"] = "D:/pictures/knight.png"
    form["ip_adapter"] = "plus"
    # A CFG base, so the Structure group draws rather than its muted note.
    form["base_model"] = "sdxl_cfg"
    form["control"] = "canny"
    _frame(imgui_ctx, lambda: settings_2d.draw(app_ctx))

    # And again on a base that cannot run a ControlNet: the other branch.
    form["base_model"] = "turbo"
    _frame(imgui_ctx, lambda: settings_2d.draw(app_ctx))


def test_the_inspector_builds_with_a_reference_report(app_ctx, imgui_ctx):
    from warlock.studio.panes import inspector

    _seeded(app_ctx)
    job_id = app_ctx.state.selected
    app_ctx.svc.store.merge_params(
        job_id,
        {
            "reference_report": {
                "ok": False,
                "reasons": ["There is more than one object in the reference."],
                "warnings": ["The subject touches the edge of the frame."],
                "occupancy": 0.62,
            },
            "control_hint": {"kind": "canny", "edge_fraction": 0.031},
            "ip_adapter": "plus",
            "ip_scale": 0.6,
        },
    )
    app_ctx.cache.invalidate()
    app_ctx.cache.tick()
    _frame(imgui_ctx, lambda: inspector.draw(app_ctx))


def test_paint_mode_builds_and_gives_its_textures_back(app_ctx, imgui_ctx):
    """The panes that own GL objects of their own. _seeded writes a byte, not
    a PNG, so this seeds a real image -- opening one is a decode."""
    from warlock.studio import inker_mode, inker_state
    from warlock.studio.panes import (
        paint_bridge,
        paint_canvas,
        paint_colors,
        paint_layers,
        paint_tools,
    )

    job_id = _reference_job(app_ctx)
    app_ctx.state.mode = "paint"
    state = inker_mode.ensure(app_ctx)

    def build() -> None:
        paint_tools.draw(app_ctx)
        paint_colors.draw(app_ctx)
        paint_canvas.draw(app_ctx)
        paint_layers.draw(app_ctx)
        paint_bridge.draw(app_ctx)

    # Empty first: the "nothing open" branch is what a user sees on arrival.
    _frame(imgui_ctx, build)

    loaded = inker_mode._load_job(app_ctx.svc, job_id)
    inker_mode.on_task_done(app_ctx, _done(f"inker-open:{job_id}", loaded))
    tab = state.active
    assert tab is not None and tab.job_id == job_id
    _frame(imgui_ctx, build)

    # Every tool once: each has its own options branch.
    for tool, _label, _key in inker_state.TOOLS:
        state.tool = tool
        _frame(imgui_ctx, build)

    # A second layer, a selection and a floating buffer: the other textures.
    tab.doc.add_layer()
    tab.doc.select_all()
    tab.doc.lift()
    _frame(imgui_ctx, build)

    # Free transform takes the canvas over, so it is its own set of branches:
    # the handle overlay, the numeric row, and every panel while modal.
    tab.doc.commit_floating()
    inker_mode.begin_transform(app_ctx, tab)
    assert state.transforming and tab.doc.floating is not None
    _frame(imgui_ctx, build)
    tab.doc.rotate_floating(30.0)
    tab.doc.transform_floating(scale=(1.5, 1.5))
    _frame(imgui_ctx, build)
    inker_mode.end_transform(app_ctx, commit=True)
    assert not state.transforming and tab.doc.floating is None
    _frame(imgui_ctx, build)
    assert app_ctx.state.preview.get(f"paint_tex:{tab.uid}:composite") is not None
    assert app_ctx.state.preview.get(f"paint_tex:{tab.uid}:floating") is not None

    # Dirty, so closing asks first -- and the question is what stops a stray
    # click on the tab's x from losing an unsaved painting.
    uid = tab.uid
    inker_mode.request_close(app_ctx, tab)
    assert app_ctx.confirms.pending is not None
    app_ctx.confirms.pending.on_confirm()
    app_ctx.confirms.pending = None
    assert state.active is None
    assert not [k for k in app_ctx.state.preview if k.startswith(f"paint_tex:{uid}:")]
    inker_mode.release_all(app_ctx)


def test_a_finished_mesh_is_not_offered_paint(app_ctx, imgui_ctx):
    """Paint edits the *generated reference*, and a mesh job's input.png is
    whatever it was reconstructed from."""
    from warlock.studio import inker_mode

    job_id = _seeded(app_ctx)
    app_ctx.state.mode = "2d"
    assert not inker_mode.can_edit_job(app_ctx, app_ctx.cache.get(job_id))


def _done(key, result):
    from warlock.studio.tasks import Done

    return Done(key=key, result=result)


def _reference_job(app_ctx) -> str:
    """A finished reference with a real PNG on disk."""
    import io

    from PIL import Image

    job_id = svc_jobs.create_job(
        app_ctx.svc, kind="text", prompt="a barrel", output="reference"
    )["id"]
    job_dir = app_ctx.svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    Image.new("RGBA", (8, 8), (40, 80, 160, 255)).save(buf, "PNG")
    (job_dir / "input.png").write_bytes(buf.getvalue())
    app_ctx.svc.store.set_status(job_id, "done")
    app_ctx.cache.invalidate()
    app_ctx.cache.tick()
    app_ctx.state.select(job_id)
    return job_id


def test_the_widget_kit_builds_every_new_widget(app_ctx, imgui_ctx):
    """The Phase-2 widgets (segmented control, toggle, pill, card, buttons,
    empty state, animated toasts) all draw through the real backend."""
    from warlock.studio import icons, widgets

    imgui, renderer = imgui_ctx
    state = app_ctx.state
    state.toast("hello")
    state.toast("bad thing", level="error")

    def build():
        widgets.segmented_control("seg", [("a", "Alpha"), ("b", "Beta")], "a")
        widgets.toggle("Wireframe", True, tag="wf")
        widgets.icon_button(icons.CAMERA, "Screenshot", enabled=False)
        widgets.icon_button(icons.TRASH, "Delete", danger=True)
        widgets.status_pill("running")
        widgets.primary_button("Generate", (-1, 34))
        widgets.destructive_button("Delete", (150, 0))
        widgets.labeled_combo("Art style", "", [("", "art style..."), ("x", "X")])
        with widgets.card("c1", (300, 90)):
            imgui.text("card body")
        widgets.empty_state(icons.BOX, "Nothing here", "Do a thing.")
        widgets.toasts(state, (1600, 950))

    # Twice: the motion dict has state after the first frame, so the second
    # exercises the animated paths rather than the first-sighting snap.
    for _ in range(2):
        _frame(imgui_ctx, build)
