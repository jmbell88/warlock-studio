"""Every pane, built for real, once.

Not a screenshot test -- it asserts nothing about what anything looks like. It
asserts that a frame containing every panel can be *built*: no missing imgui
begin/end pair, no attribute that moved, no pane reaching for state that is
only set once a job exists. That class of mistake is otherwise found by opening
the app, and there are eight panels.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from warlock.service import jobs as svc_jobs
from warlock.studio.app_ctx import Ctx
from warlock.studio.jobs_cache import JobsCache
from warlock.studio.settings import Settings
from warlock.studio.state import AppState, Eta, Filters


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
    ctx.guidance = svc_system.guidance_catalog(svc)
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


def test_the_library_offers_a_way_to_the_failed_jobs(app_ctx, imgui_ctx):
    """A failed job says why it failed in the inspector and nowhere else, so
    the only route to the reason was to already know which card to click."""
    from warlock.studio.panes import library

    job_id = _seeded(app_ctx)
    app_ctx.svc.store.set_status(job_id, "error", error="it broke")
    app_ctx.cache.invalidate()
    app_ctx.cache.tick()
    assert app_ctx.state.filters.failures(app_ctx.cache.jobs) == 1
    _frame(imgui_ctx, lambda: library.draw(app_ctx))


def test_the_failure_affordance_is_absent_once_the_filter_is_already_on_errors():
    """It exists to *reach* the failed jobs. Left on screen while they are the
    only thing showing, it is a button whose click changes nothing.

    Needs no imgui frame precisely because it draws nothing: the cache here
    raises if the count is ever computed, which is what proves the guard runs
    before the work rather than merely before the button.
    """
    from warlock.studio.panes import library

    class Detonating:
        @property
        def jobs(self):
            raise AssertionError("counted the failures after the early return")

    ctx = SimpleNamespace(
        state=SimpleNamespace(filters=Filters(status="error")), cache=Detonating()
    )
    library._failures(ctx)


def test_the_library_filter_row_fits_the_sidebar(app_ctx, imgui_ctx):
    """Not "it builds" -- where it builds *to*.

    The sidebar is a fixed 300 design px and a child window clips rather than
    wraps, so a row of controls wider than that is drawn past the right edge
    and can be neither seen nor clicked. Three 110 px combos and two square
    buttons came to 417 into 290, which is how the favourites star spent its
    whole life invisible. Asserted against the content region rather than
    against an arrangement, so the row is free to be laid out any way that
    fits.
    """
    imgui, _renderer = imgui_ctx
    from warlock.studio import layout as layout_mod
    from warlock.studio.panes import library
    from warlock.studio.tokens import sp

    measured: list[float] = []

    def build() -> None:
        if layout_mod.pane_child("library", (sp(layout_mod.SIDEBAR_W), 0)):
            right = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
            library._filters(app_ctx, [])
            measured.append(imgui.get_item_rect_max().x - right)
        imgui.end_child()

    _frame(imgui_ctx, build)
    assert measured and measured[0] <= 1.0, f"the filter row overflows by {measured[0]:.0f} px"


def test_a_library_cards_action_row_stays_inside_the_card(app_ctx, imgui_ctx):
    """A reference has no mesh report, so ``quality_badge`` draws nothing --
    and the ``same_line`` in front of it was then inherited by the action row,
    which started 73 px to the right on the status pill's line and put the
    favourite star off the edge of the card, where it could not be clicked."""
    imgui, _renderer = imgui_ctx
    from warlock.studio import layout as layout_mod
    from warlock.studio.panes import library
    from warlock.studio.tokens import sp

    app_ctx.rigging_available = True
    _seeded(app_ctx)  # no mesh_report and no mesh_audit: an ordinary reference
    measured: list[float] = []
    real = library._card_actions

    def spy(ctx, job):
        right = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
        real(ctx, job)
        measured.append(imgui.get_item_rect_max().x - right)

    library._card_actions = spy
    try:
        _frame(
            imgui_ctx,
            lambda: (
                layout_mod.pane_child("library", (sp(layout_mod.SIDEBAR_W), 0))
                and library.draw(app_ctx),
                imgui.end_child(),
            ),
        )
    finally:
        library._card_actions = real
    assert measured and measured[0] <= 0.0, f"the action row overflows by {measured[0]:.0f} px"


def _seed_findings(ctx, param, value):
    """A findings.json with one bucket deep enough to hint."""
    import json
    from pathlib import Path

    bench = Path(ctx.svc.config.bench_dir)
    bench.mkdir(parents=True, exist_ok=True)
    (bench / "findings.json").write_text(
        json.dumps(
            {"params": {param: {str(value): {"n": 8, "accepts": 6, "wilson_low": 0.41}}}}
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("pane", "param", "value"),
    [("settings_2d", "art_style", "nes"), ("settings_3d", "platform", "pc")],
)
def test_an_evidence_hint_stays_inside_the_pane(app_ctx, imgui_ctx, pane, param, value):
    """The hints are the whole visible payoff of the observation corpus, and
    both panes drew them with a bare ``same_line`` after a control that had
    taken the full width -- which puts the cursor *on* the right edge, so the
    text went past it and was clipped away entirely. 147 px past, on the 3D
    pane; 63 px past the column, in the 2D pane's guidance grid."""
    imgui, _renderer = imgui_ctx
    import importlib

    from warlock.studio import layout as layout_mod
    from warlock.studio import widgets
    from warlock.studio.tokens import sp

    module = importlib.import_module(f"warlock.studio.panes.{pane}")
    _seed_findings(app_ctx, param, value)
    (app_ctx.state.form_2d if pane == "settings_2d" else app_ctx.state.form_3d)[param] = value

    measured: list[float] = []
    real = widgets.hint_text

    def spy(text):
        right = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
        real(text)
        measured.append(imgui.get_item_rect_max().x - right)

    widgets.hint_text = spy
    try:
        _frame(
            imgui_ctx,
            lambda: (
                layout_mod.pane_child(pane, (sp(layout_mod.SIDEBAR_W), 0))
                and module.draw(app_ctx),
                imgui.end_child(),
            ),
        )
    finally:
        widgets.hint_text = real
    assert measured, "the hint never drew -- the fixture no longer produces one"
    assert max(measured) <= 1.0, f"a hint overflows by {max(measured):.0f} px"


def test_no_pane_continues_a_line_that_has_no_room_left(app_ctx, imgui_ctx):
    """The class of bug the panes kept hitting, guarded once for all of them.

    ``same_line`` after an item drawn at full width leaves the cursor *on* the
    content region's right edge, and a child window clips rather than wraps --
    so whatever is drawn next is not squeezed, it is gone. Four controls were
    living out there when this was written: the library's favourites star, its
    select-all tick, its Prune button, and every evidence hint and ``(?)`` in
    the two generate panes. None of them raised anything, and each looked
    exactly like a feature nobody had built.

    The threshold is "no room at all", not "not much room": a tight row is a
    judgement call, and a control drawn past the edge is not. The one exempt
    caller is ``widgets.same_line_or_wrap``, which asks this same question
    itself and starts a new line when the answer is no.
    """
    imgui, _renderer = imgui_ctx
    import traceback

    from warlock.studio import layout as layout_mod
    from warlock.studio.panes import (
        clay_bridge,
        clay_outliner,
        clay_props,
        clay_tools,
        inker_bridge,
        inker_colors,
        inker_layers,
        inker_tools,
        inspector,
        library,
        pose_panel,
        profiles_panel,
        retarget_panel,
        settings_2d,
        settings_3d,
        sheet_panel,
    )
    from warlock.studio.tokens import sp

    job_id = _seeded(app_ctx)
    app_ctx.state.mode = "3d"
    app_ctx.rigging_available = True
    app_ctx.state.form_3d["rig"] = True
    job = app_ctx.cache.get(job_id)
    panes = [
        ("settings-2d", lambda: settings_2d.draw(app_ctx)),
        ("settings-3d", lambda: settings_3d.draw(app_ctx)),
        ("library", lambda: library.draw(app_ctx)),
        ("inspector", lambda: inspector.draw(app_ctx)),
        ("retarget", lambda: retarget_panel.draw(app_ctx, job)),
        ("pose", lambda: pose_panel.draw(app_ctx, job)),
        ("sheet", lambda: sheet_panel.draw(app_ctx, job)),
        ("profiles", lambda: profiles_panel.draw(app_ctx)),
        ("clay-tools", lambda: clay_tools.draw(app_ctx)),
        ("clay-props", lambda: clay_props.draw(app_ctx)),
        ("clay-outliner", lambda: clay_outliner.draw(app_ctx)),
        ("clay-bridge", lambda: clay_bridge.draw(app_ctx)),
        ("inker-tools", lambda: inker_tools.draw(app_ctx)),
        ("inker-layers", lambda: inker_layers.draw(app_ctx)),
        ("inker-colors", lambda: inker_colors.draw(app_ctx)),
        ("inker-bridge", lambda: inker_bridge.draw(app_ctx)),
    ]

    offenders: dict[str, float] = {}
    real = imgui.same_line

    def spy(*args, **kwargs):
        real(*args, **kwargs)
        avail = imgui.get_content_region_avail().x
        if avail >= 1.0:
            return
        caller = traceback.extract_stack()[-2]
        if caller.name == "same_line_or_wrap":
            return
        where = f"{Path(caller.filename).name}:{caller.lineno}"
        offenders[where] = min(offenders.get(where, avail), avail)

    imgui.same_line = spy
    try:
        for pane_id, build in panes:
            _frame(
                imgui_ctx,
                lambda pane_id=pane_id, build=build: (
                    layout_mod.pane_child(pane_id, (sp(layout_mod.SIDEBAR_W), 0)) and build(),
                    imgui.end_child(),
                ),
            )
    finally:
        imgui.same_line = real
    assert not offenders, f"drawn past the right edge: {offenders}"


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

    # And the branch for "the editor is open on some other asset", which
    # _sync_viewer's deliberate early-return while posing makes reachable by
    # clicking any other card.
    app_ctx.viewer.pose_mode = True
    app_ctx.viewer.pose_job_id = "ffffffffffff"
    _frame(imgui_ctx, lambda: pose_panel.draw(app_ctx, job))
    app_ctx.viewer.pose_mode = False
    app_ctx.viewer.pose_job_id = None


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


def test_the_sheet_preview_advances_a_cell_per_frame(app_ctx, imgui_ctx):
    """The preview draws and reads back per cell, so it is spread over frames
    rather than done in the one the button was pressed. Drawn here with a
    strip genuinely in progress, which is the branch a static frame misses."""
    from PIL import Image

    from warlock.studio.panes import sheet_panel

    job_id = _seeded(app_ctx)
    job = app_ctx.cache.get(job_id)
    job["files"] = ["model.glb", "rig.glb"]

    class _Strip:
        def __init__(self) -> None:
            self.steps = 0
            self.released = False
            self.image = Image.new("RGBA", (32, 8), (0, 0, 0, 0))
            self.index = 0

        def step(self) -> bool:
            self.steps += 1
            return self.steps >= 2

        def release(self) -> None:
            self.released = True

    strip = _Strip()
    app_ctx.viewer._strip = strip
    # ``has_model`` gates the whole section and the fixture's viewer is empty;
    # a stand-in is enough, since nothing in this path dereferences it.
    app_ctx.viewer.gpu = object()

    try:
        assert app_ctx.viewer.stripping is True
        _frame(imgui_ctx, lambda: sheet_panel.draw(app_ctx, job))
        assert strip.steps == 1, "one cell per frame, not all of them"
        assert app_ctx.viewer.stripping is True

        _frame(imgui_ctx, lambda: sheet_panel.draw(app_ctx, job))
        assert strip.steps == 2
        assert app_ctx.viewer.stripping is False, "finished, and let go of"
    finally:
        app_ctx.viewer.gpu = None
        app_ctx.viewer._strip = None
        app_ctx.state.preview.pop("sheet_strip", None)


def test_the_overlay_builds_with_a_toolbar_and_a_banner(app_ctx, imgui_ctx):
    from warlock.studio.panes import overlay

    eta = Eta()
    app_ctx.state.note_error("trellis: the exe is missing")
    app_ctx.state.note_error("The GPU worker is not running. Restart Warlock.")
    _frame(
        imgui_ctx,
        lambda: (overlay.doctor_banner(app_ctx), overlay.toolbar(app_ctx),
                 overlay.progress_card(app_ctx, eta), overlay.placeholder(app_ctx)),
    )


def test_toasts_and_dialogs_build(app_ctx, imgui_ctx):
    from warlock.studio import dialogs, widgets

    app_ctx.state.toast("finished", "info")
    app_ctx.state.toast("it broke", "error")
    app_ctx.state.toast("Something went wrong; see the log.", "error", action="log")
    app_ctx.state.toast("an action nothing draws", "error", action="teleport")
    app_ctx.confirms.ask(dialogs.Confirm(title="Sure?", message="Really?"))
    app_ctx.prompts.ask(dialogs.Prompt(title="Name it", label="Name"))
    imgui, renderer = imgui_ctx
    imgui.new_frame()
    widgets.toasts(app_ctx.state, (1600, 950), on_action=lambda _name: None)
    app_ctx.confirms.draw()
    app_ctx.prompts.draw()
    imgui.render()
    renderer.render(imgui.get_draw_data())


def test_the_whole_frame_builds_at_once(app_ctx, imgui_ctx):
    """The real layout: three panes side by side, as main.py assembles them.

    Through ``layout.pane_child`` for the left sidebar and viewport, and
    through ``main._right_column`` -- the same function ``App._build_ui``
    calls -- for the right one, rather than a second hand-copy of its
    inspector/library split sitting next to the function that exists to
    prevent exactly that.
    """
    from warlock.studio import layout as layout_mod
    from warlock.studio import main as main_mod
    from warlock.studio.panes import inspector, library, overlay, settings_2d

    _seeded(app_ctx)
    imgui, renderer = imgui_ctx

    def build():
        layout_mod.pane_child("settings", (340, 0))
        settings_2d.draw(app_ctx)
        imgui.end_child()
        imgui.same_line()
        layout_mod.pane_child("viewport", (400, 0))
        overlay.toolbar(app_ctx)
        overlay.placeholder(app_ctx)
        imgui.end_child()
        imgui.same_line()
        lay = layout_mod.Layout(app_ctx.settings)
        main_mod._right_column(
            app_ctx, lay, 340.0, inspector_draw=inspector.draw, library_draw=library.draw
        )

    _frame(imgui_ctx, build)
    del renderer


def test_the_right_sidebar_splits_inspector_and_library_by_settings_share(app_ctx, imgui_ctx):
    """Calls ``main._right_column`` directly -- the same function
    ``App._build_ui`` calls for the right sidebar -- rather than a
    reimplementation of its arithmetic, so a regression in main.py itself (a
    hardcoded ratio, or the panes swapped) is caught here rather than only on
    screen.

    The expected split is computed from the splitter's grip width and the two
    item-spacing gaps around it, not padded into a wide tolerance: the avail
    height measured before the split recombines exactly into
    ``inspector + spacing + grip + spacing + library``, so hardcoding the
    ratio (or dropping the splitter) throws the measured heights off by more
    than a pixel.
    """
    from warlock.studio import layout as layout_mod
    from warlock.studio import main as main_mod
    from warlock.studio.tokens import sp

    imgui, _renderer = imgui_ctx
    lay = layout_mod.Layout(app_ctx.settings)
    assert lay.settings_share == 0.55  # the untouched default this test relies on

    avail_before: list[float] = []
    tops: list[float] = []
    bottoms: list[float] = []

    def build():
        avail_before.append(imgui.get_content_region_avail().y)
        main_mod._right_column(
            app_ctx,
            lay,
            300.0,
            inspector_draw=lambda _ctx: tops.append(imgui.get_window_size().y),
            library_draw=lambda _ctx: bottoms.append(imgui.get_window_size().y),
        )

    _frame(imgui_ctx, build)

    avail_y = avail_before[0]
    spacing = imgui.get_style().item_spacing.y
    grip = sp(layout_mod.GRIP)
    expected_top = avail_y * lay.settings_share
    expected_bottom = avail_y - expected_top - grip - 2 * spacing

    assert tops[0] == pytest.approx(expected_top, abs=1.0)
    assert bottoms[0] == pytest.approx(expected_bottom, abs=1.0)


def test_the_landing_screen_builds_in_each_of_its_views(app_ctx, imgui_ctx):
    from warlock.studio.panes import landing

    _frame(imgui_ctx, lambda: landing.draw(app_ctx))
    app_ctx.state.landing_view = "open"
    _seeded(app_ctx)
    _frame(imgui_ctx, lambda: landing.draw(app_ctx))
    app_ctx.state.landing_view = "profiles"
    _frame(imgui_ctx, lambda: landing.draw(app_ctx))


def test_the_manual_builds_embedded(app_ctx, imgui_ctx):
    """As a mode, not a window: no begin/end of its own to get wrong.

    The loader falls back to the repo's docs/manual in this checkout, so this
    parses and draws the real chapters.
    """
    from warlock.studio.manual import render

    _frame(imgui_ctx, lambda: render.draw_body(app_ctx))
    app_ctx.state.manual.open_at("08-shortcuts", None)
    _frame(imgui_ctx, lambda: render.draw_body(app_ctx))


def test_the_settings_pane_builds(app_ctx, imgui_ctx):
    """Twice: once bare, once after the model lists are populated, because the
    pane reads them off the Ctx with getattr and both shapes must build."""
    from warlock.studio.panes import app_settings

    app_ctx.base_models = []
    app_ctx.style_loras = []
    _frame(imgui_ctx, lambda: app_settings.draw(app_ctx))
    app_ctx.base_models = [("turbo", "SDXL-Turbo"), ("x", "X - weights missing")]
    app_ctx.style_loras = [("", "no style LoRA"), ("ink", "Ink")]
    app_ctx.rigging_available = True
    _frame(imgui_ctx, lambda: app_settings.draw(app_ctx))


def test_the_settings_pane_help_button_stays_inside_the_pane(app_ctx, imgui_ctx, monkeypatch):
    """help_button is a same_line, and same_line returns to the *previous* row.

    Drawn as the pane's first widget it landed on the row above -- in the app
    that is the mode switch, so it overlapped the health dot and, being
    submitted second, could not be clicked. Every other pane escapes this only
    because a section heading precedes it, so the guard has to be a measurement
    rather than a call-order convention.
    """
    from imgui_bundle import imgui

    from warlock.studio import widgets
    from warlock.studio.panes import app_settings

    seen: dict[str, float] = {}
    real = widgets.icon_button

    def spy(*args, **kwargs):
        seen.setdefault("help_y", imgui.get_cursor_screen_pos().y)
        return real(*args, **kwargs)

    monkeypatch.setattr(widgets, "icon_button", spy)

    def build():
        imgui.button("stand-in for the mode switch")
        seen["bar_y"] = imgui.get_item_rect_min().y
        app_settings.draw(app_ctx)

    _frame(imgui_ctx, build)
    assert "help_y" in seen, "the pane drew no help button at all"
    assert seen["help_y"] > seen["bar_y"], seen


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


def test_a_non_sdxl_base_disables_the_style_lora_control_and_says_why(app_ctx, imgui_ctx):
    """Disabled with a reason, not hidden: the form holds a style picked under
    another base, and hiding the control would make that selection vanish with
    no explanation of why the submit is now refused."""
    from warlock.studio.panes import settings_2d

    form = app_ctx.state.form_2d
    form["prompt"] = "a barrel"
    form["base_model"] = "sdxl_cfg"
    form["style_lora"] = "render3d"
    assert settings_2d.lora_note(app_ctx, form) is None
    assert not settings_2d.validate(form)

    form["base_model"] = "flux_klein"
    note = settings_2d.lora_note(app_ctx, form)
    assert note is not None
    # It has to name bases the user can actually find in the picker.
    assert "SDXL" in note
    assert "Style LoRAs need an SDXL model." in settings_2d.validate(form)
    # And the disabled branch has to draw.
    _frame(imgui_ctx, lambda: settings_2d.draw(app_ctx))

    # A tile is refused on the same grounds, and independently of the LoRA.
    form["style_lora"] = ""
    form["output"] = "tile"
    assert "Seamless tiles need an SDXL model." in settings_2d.validate(form)


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
        inker_bridge,
        inker_canvas,
        inker_colors,
        inker_layers,
        inker_tools,
    )

    job_id = _reference_job(app_ctx)
    app_ctx.state.mode = "inker"
    state = inker_mode.ensure(app_ctx)

    def build() -> None:
        inker_tools.draw(app_ctx)
        inker_colors.draw(app_ctx)
        inker_canvas.draw(app_ctx)
        inker_layers.draw(app_ctx)
        inker_bridge.draw(app_ctx)

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
    assert app_ctx.state.preview.get(f"inker_tex:{tab.uid}:composite") is not None
    assert app_ctx.state.preview.get(f"inker_tex:{tab.uid}:floating") is not None

    # Dirty, so closing asks first -- and the question is what stops a stray
    # click on the tab's x from losing an unsaved painting.
    uid = tab.uid
    inker_mode.request_close(app_ctx, tab)
    assert app_ctx.confirms.pending is not None
    app_ctx.confirms.pending.on_confirm()
    app_ctx.confirms.pending = None
    assert state.active is None
    assert not [k for k in app_ctx.state.preview if k.startswith(f"inker_tex:{uid}:")]
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


# --- Clay mode -------------------------------------------------------------


def _clay_tab(app_ctx, *, objects: int = 2):
    """A Clay document with objects in it, adopted as the active tab."""
    from warlock.studio import clay_mode
    from warlock.studio.clay import document as bd
    from warlock.studio.clay import primitives as bp

    doc = bd.ClayDoc()
    for i in range(objects):
        doc.add_object(
            bd.Obj(uid=bd.new_uid(), name=f"obj{i}", mesh=bp.box(), generator="box",
                   params={"size": (1.0, 1.0, 1.0)})
        )
    return clay_mode.adopt(app_ctx, doc, title="Scene")


def test_the_clay_panes_build_with_nothing_open(app_ctx, imgui_ctx):
    """Every one of them has to survive the state the mode opens in."""
    from warlock.studio.panes import clay_bridge, clay_outliner, clay_props, clay_tools

    for pane in (clay_tools, clay_props, clay_outliner, clay_bridge):
        _frame(imgui_ctx, lambda pane=pane: pane.draw(app_ctx))


def test_the_clay_panes_build_with_a_document_and_a_selection(app_ctx, imgui_ctx):
    from warlock.studio.panes import clay_bridge, clay_outliner, clay_props, clay_tools

    tab = _clay_tab(app_ctx)
    tab.doc.select([tab.doc.objects[0].uid])
    for pane in (clay_tools, clay_props, clay_outliner, clay_bridge):
        _frame(imgui_ctx, lambda pane=pane: pane.draw(app_ctx))


def test_the_clay_panes_build_while_a_save_is_in_flight(app_ctx, imgui_ctx):
    """``saving`` puts every mutating control inside ``begin_disabled``, and an
    unbalanced disable stack is exactly the class of mistake this file exists
    to catch."""
    from warlock.studio.panes import clay_bridge, clay_outliner, clay_props, clay_tools

    tab = _clay_tab(app_ctx)
    tab.doc.select([tab.doc.objects[0].uid])
    tab.saving = True
    for pane in (clay_tools, clay_props, clay_outliner, clay_bridge):
        _frame(imgui_ctx, lambda pane=pane: pane.draw(app_ctx))


def test_the_clay_context_menu_and_its_parameter_popup_build(app_ctx, imgui_ctx):
    """``clay_menu`` was the one drawn pane named in no test at all, and it
    owns the parameter popup: the pending-op lifecycle, the clamp to a param's
    range, the stale-name recovery and the saving gate."""
    from types import SimpleNamespace

    from warlock.studio import clay_mode, clay_ops
    from warlock.studio.panes import clay_menu

    tab = _clay_tab(app_ctx)
    tab.doc.select([tab.doc.objects[0].uid])
    view = SimpleNamespace(menu_request=None)

    _frame(imgui_ctx, lambda: clay_menu.draw(app_ctx, view))
    view.menu_request = (10.0, 10.0)
    _frame(imgui_ctx, lambda: clay_menu.draw(app_ctx, view))
    assert view.menu_request is None, "the request is consumed by the frame that opens it"

    # The parameter popup, including the keyboard path's open request.
    state = clay_mode.ensure(app_ctx)
    parameterised = next(op for op in clay_ops.OPS if op.params)
    state.pending_op = parameterised.name
    state.open_op_popup = True
    _frame(imgui_ctx, lambda: clay_menu.draw(app_ctx, view))
    assert state.open_op_popup is False, "a request must not outlive its frame"

    # And a name from an op that no longer exists clears itself rather than
    # leaving the mode holding a request it can never act on.
    state.pending_op = "no-such-op"
    state.open_op_popup = True
    _frame(imgui_ctx, lambda: clay_menu.draw(app_ctx, view))
    assert state.pending_op == ""
    assert state.open_op_popup is False

    # Saving greys every row rather than swallowing the click.
    tab.saving = True
    _frame(imgui_ctx, lambda: clay_menu.draw(app_ctx, view))


def test_the_clay_properties_pane_builds_for_a_frozen_object(app_ctx, imgui_ctx):
    """Phase 2's state: no generator, so the panel shows counts instead of
    parameters. Unreachable from the UI today and drawn here anyway, because it
    is one line away from being reachable."""
    from warlock.studio.panes import clay_props

    tab = _clay_tab(app_ctx, objects=1)
    obj = tab.doc.objects[0]
    tab.doc.set_props(obj.uid, generator=None)
    tab.doc.select([obj.uid])
    _frame(imgui_ctx, lambda: clay_props.draw(app_ctx))


def test_the_clay_properties_pane_builds_for_every_generator(app_ctx, imgui_ctx):
    """The parameter widgets come off the registry, so every default type in it
    has to have a widget -- a float, an int and a tuple today."""
    from warlock.studio.clay import document as bd
    from warlock.studio.clay import primitives as bp
    from warlock.studio.panes import clay_props

    tab = _clay_tab(app_ctx, objects=0)
    for name, (defaults, build) in bp.GENERATORS.items():
        obj = bd.Obj(
            uid=bd.new_uid(), name=name, mesh=build(**defaults),
            generator=name, params=dict(defaults),
        )
        tab.doc.add_object(obj)
        tab.doc.select([obj.uid])
        _frame(imgui_ctx, lambda: clay_props.draw(app_ctx))


def test_the_clay_outliner_builds_with_a_rename_in_flight(app_ctx, imgui_ctx):
    from warlock.studio import clay_mode
    from warlock.studio.panes import clay_outliner

    tab = _clay_tab(app_ctx)
    clay_mode.ensure(app_ctx).renaming = tab.doc.objects[0].uid
    _frame(imgui_ctx, lambda: clay_outliner.draw(app_ctx))


def test_the_clay_properties_pane_enumerates_a_generator_it_has_never_seen(
    app_ctx, imgui_ctx, monkeypatch
):
    """A seventh primitive must need no edit to the pane.

    Asserted by registering one and *drawing* it: the claim is about what the
    pane enumerates, and a chain of ``if name == "box"`` would satisfy any
    source check while showing this object no parameters at all. The widget
    labels are read back off the frame, so a default type with no widget shows
    up as a missing label rather than as a pane that merely did not crash.
    """
    from warlock.studio.clay import document as bd
    from warlock.studio.clay import primitives as bp
    from warlock.studio.panes import clay_props

    def wedge(width: float = 2.0, steps: int = 3, footprint=(1.0, 1.0)):
        return bp.box(size=(width, 1.0, 1.0))

    defaults = {"width": 2.0, "steps": 3, "footprint": (1.0, 1.0)}
    monkeypatch.setitem(bp.GENERATORS, "wedge", (defaults, wedge))

    tab = _clay_tab(app_ctx, objects=0)
    obj = bd.Obj(
        uid=bd.new_uid(), name="Wedge", mesh=wedge(**defaults),
        generator="wedge", params=dict(defaults),
    )
    tab.doc.add_object(obj)
    tab.doc.select([obj.uid])

    seen: list[str] = []
    real = clay_props._widget

    def spy(key, value, default):
        seen.append(key)
        return real(key, value, default)

    monkeypatch.setattr(clay_props, "_widget", spy)
    _frame(imgui_ctx, lambda: clay_props.draw(app_ctx))
    assert seen == ["width", "steps", "footprint"]


def test_the_clay_viewport_draws_through_the_real_imgui_backend(app_ctx, imgui_ctx, gl):
    """The one part of the Clay workspace the pane tests cannot reach.

    ``widgets.texture_ref`` has to *register* the viewport texture with the
    backend as well as wrap it -- an id the renderer does not know maps to no
    moderngl object, and every image in the UI comes out as the font atlas. A
    frame that draws it for real is the only thing that says so.
    """
    from imgui_bundle import imgui

    from warlock.studio import clay_view, widgets

    tab = _clay_tab(app_ctx)
    view = clay_view.ClayView(gl, app_ctx)
    try:
        view.frame_selection(tab.doc)

        def build():
            texture = view.draw(tab.doc, (0.0, 0.0, 320.0, 240.0), 1 / 60)
            imgui.image(widgets.texture_ref(texture), (320, 240), (0, 1), (1, 0))

        _frame(imgui_ctx, build)
    finally:
        view.release()


def test_a_built_document_renders_the_flat_reference_trellis_is_given(app_ctx, gl):
    """No grid, no gizmos, no overlays, on a plain background: trellis is being
    handed a subject, and a grid line in the picture is a subject too."""
    from warlock.studio import clay_view
    from warlock.studio.viewer import capture, glctx

    tab = _clay_tab(app_ctx)
    view = clay_view.ClayView(gl, app_ctx)
    target = glctx.Viewport(gl, (128, 128))
    try:
        view.frame_selection(tab.doc)
        view.sync(tab.doc)
        view.renderer.draw(
            target,
            view.camera,
            view._composite(tab.doc),
            flat=True,
            show_grid=False,
            background=(1.0, 1.0, 1.0, 1.0),
            overlays=[],
        )
        png = capture.png_bytes(target)
    finally:
        target.release()
        view.release()

    assert png.startswith(b"\x89PNG")
    import io as _io

    from PIL import Image

    with Image.open(_io.BytesIO(png)) as im:
        pixels = im.convert("RGB")
        colours = pixels.getcolors(maxcolors=1 << 16) or []
    # White is there (the background) and so is something else (the subject).
    assert any(colour == (255, 255, 255) for _n, colour in colours)
    assert len(colours) > 1


# --- review -----------------------------------------------------------------


class _ReviewApp:
    """The Review pane's drawing methods, unbound, over a stub app.

    They live on ``App`` rather than in a pane module (Review draws its own
    three columns), and none of them touches anything of ``self`` beyond the
    other methods here -- so this is enough to build them for real.
    """

    from warlock.studio import main as _main

    _review_runs = _main.App._review_runs
    _review_delete_button = _main.App._review_delete_button
    _review_form = _main.App._review_form
    _review_units = _main.App._review_units
    _review_verdict = _main.App._review_verdict
    _review_findings = _main.App._review_findings
    _save_vector_preset = _main.App._save_vector_preset


def _review_state(ctx, *, with_units=True):
    from warlock.studio import review_mode

    state = review_mode.ensure(ctx)
    units = []
    if with_units:
        job_id = _seeded(ctx, mesh_report={"triangles": 10, "watertight": True})
        units = [
            {
                "job_id": job_id,
                "label": "lora_weight=0.6 s1",
                "status": "done",
                "params": ctx.svc.store.get(job_id)["params"],
                "dir": ctx.svc.job_dir(job_id),
                "verdict": "reject",
                "reasons": ["holes"],
            }
        ]
    state.sweeps = [
        {"id": review_mode.RECENT_ID, "label": "Recent, unreviewed",
         "prompt": "", "units": [], "todo": 0},
        {"id": "abcdef012345", "label": "lora sweep", "prompt": "a barrel",
         "units": units, "todo": 0},
    ]
    state.sweep_id = "abcdef012345"
    state.units = units
    return state


def test_the_review_panes_build_with_a_sweep_and_a_unit(app_ctx, imgui_ctx):
    from warlock.studio import review_mode

    app = _ReviewApp()
    state = _review_state(app_ctx)
    state.form.prompt = "a barrel"
    state.form.axes = [{"param": "lora_weight", "values": "0.6, 1.2"}]

    _frame(
        imgui_ctx,
        lambda: (
            app._review_runs(app_ctx, state, review_mode),
            app._review_units(state, review_mode),
            app._review_verdict(app_ctx, state, review_mode),
        ),
    )


def test_the_review_panes_build_with_nothing_recorded(app_ctx, imgui_ctx):
    from warlock.studio import review_mode

    app = _ReviewApp()
    state = _review_state(app_ctx, with_units=False)

    _frame(
        imgui_ctx,
        lambda: (
            app._review_runs(app_ctx, state, review_mode),
            app._review_units(state, review_mode),
            app._review_verdict(app_ctx, state, review_mode),
        ),
    )


def test_the_review_pane_builds_a_findings_table(app_ctx, imgui_ctx):
    """The section only appears once a configuration has enough verdicts, so a
    frame with an empty findings.json never touches the rows."""
    import json

    from warlock.studio import review_mode

    bench = app_ctx.svc.config.bench_dir
    bench.mkdir(parents=True, exist_ok=True)
    (bench / "findings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "params": {},
                "vectors": [
                    {"key": "abc123", "vector": {"lora_weight": 0.9, "platform": "pc"},
                     "n": 8, "accepts": 6, "accept_rate": 0.75,
                     "top_reasons": [["holes", 2]], "jobs": []}
                ],
            }
        ),
        encoding="utf-8",
    )
    app = _ReviewApp()
    state = _review_state(app_ctx)

    _frame(imgui_ctx, lambda: app._review_verdict(app_ctx, state, review_mode))


def test_the_2d_pane_builds_with_a_saved_vector_preset(app_ctx, imgui_ctx):
    from warlock.studio import vector_presets
    from warlock.studio.panes import settings_2d

    vector_presets.save_preset(
        app_ctx.settings, "chests", {"genre": "fantasy", "platform": "pc"}
    )
    _frame(imgui_ctx, lambda: settings_2d.draw(app_ctx))

    # And the Forget branch, which only appears once one has been applied --
    # presets could be saved and applied but never removed, and nothing capped
    # the list.
    app_ctx.state.preview["vector_preset"] = "chests"
    _frame(imgui_ctx, lambda: settings_2d.draw(app_ctx))

    settings_2d._forget_vector_preset(app_ctx, "chests")
    assert vector_presets.list_presets(app_ctx.settings) == {}
    assert "vector_preset" not in app_ctx.state.preview


def test_the_inspector_builds_its_verdict_section_armed_and_not(app_ctx, imgui_ctx):
    from warlock.studio.panes import inspector

    app_ctx.state.mode = "3d"
    job_id = _seeded(app_ctx)
    app_ctx.svc.store.set_stage(job_id, "model")
    app_ctx.cache.invalidate()
    app_ctx.cache.tick()
    job = app_ctx.svc.store.get(job_id)

    _frame(imgui_ctx, lambda: inspector._verdict(app_ctx, job))
    inspector.arm_verdict(app_ctx.state, job_id)
    _frame(imgui_ctx, lambda: inspector._verdict(app_ctx, job))


def test_the_bulk_bar_says_how_much_of_the_selection_is_off_screen(app_ctx, imgui_ctx):
    """``state.checked`` is not pruned when the filters change -- deliberately,
    because ticking across two filters is a real way to use this. What is not
    defensible is the destructive path describing a smaller act than it
    performs, so the count and the confirm both name what is no longer shown."""
    imgui, _renderer = imgui_ctx
    from warlock.studio.panes import library

    kept = _seeded(app_ctx)
    gone = _seeded(app_ctx)
    app_ctx.svc.store.set_status(gone, "error", "it broke")
    app_ctx.cache.invalidate()
    app_ctx.cache.tick()
    app_ctx.state.checked.update({kept, gone})
    # A filter the second job no longer matches, exactly as narrowing one by
    # hand after a select-all would leave it.
    app_ctx.state.filters.status = "done"
    shown = app_ctx.cache.visible(app_ctx.state.filters)
    assert {j["id"] for j in shown} == {kept}

    drawn: list[str] = []
    real = imgui.text
    imgui.text = lambda s: (drawn.append(s), real(s))[1]
    try:
        _frame(imgui_ctx, lambda: library._bulk(app_ctx, shown))
    finally:
        imgui.text = real

    assert "2 selected (1 not shown)" in drawn
    assert "1 of them are not in the list you can see." in library._delete_message(2, 1)
    # And with nothing hidden it stays the sentence it always was.
    assert library._delete_message(2, 0) == "2 jobs and everything derived from them."


@pytest.mark.parametrize("scale", [1.0, 1.75])
def test_the_mode_switchs_right_hand_strip_stays_inside_the_window(
    app_ctx, imgui_ctx, scale
):
    """The readout, the ? and the health dot are one right-aligned strip, and
    its width is measured rather than reserved as a constant.

    A constant is what was there (``sp(64)``, sized for two controls), and a
    ``same_line`` past the content region clips rather than wraps -- so a strip
    that outgrew its reservation would not be squeezed, it would be gone. The
    text's width is a function of the DPI scale, the font *and* how many digits
    the readings happen to have, which is why this runs at more than one scale.
    """
    imgui, _renderer = imgui_ctx
    from warlock.studio import main as main_mod
    from warlock.studio import tokens
    from warlock.studio import widgets as widgets_mod
    from warlock.studio.fps import FpsMeter
    from warlock.studio.panes import overlay

    fake = SimpleNamespace(app_ctx=app_ctx, fps=FpsMeter())
    fake._shortcuts_popup = lambda: main_mod.App._shortcuts_popup(fake)
    fake._diagnostics_popup = lambda checks: main_mod.App._diagnostics_popup(fake, checks)
    fake._request_quit = lambda: None
    fake.fps.record(1 / 60)

    rects: list[tuple[str, tuple[float, float, float, float]]] = []
    edge: list[float] = []

    def record(name: str) -> None:
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        rects.append((name, (lo.x, lo.y, hi.x, hi.y)))

    real_icon = widgets_mod.icon_button
    real_readout = overlay.status_readout

    def icon_spy(label, *args, **kwargs):
        result = real_icon(label, *args, **kwargs)
        record(f"icon:{label}")
        return result

    def readout_spy(text, tooltip):
        real_readout(text, tooltip)
        record("readout")

    def build():
        edge.append(imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x)
        main_mod.App._mode_switch(fake)
        record("health")

    old_scale = tokens.SCALE
    tokens.set_scale(scale)
    widgets_mod.icon_button = icon_spy
    overlay.status_readout = readout_spy
    try:
        _frame(imgui_ctx, build)
    finally:
        widgets_mod.icon_button = real_icon
        overlay.status_readout = real_readout
        tokens.set_scale(old_scale)

    assert {name for name, _ in rects} >= {"readout", "icon:?", "health"}, rects
    right = edge[0]
    for name, rect in rects:
        assert rect[2] <= right + 1.0, (
            f"{name} ends {rect[2] - right:.0f} px past the content edge; "
            "a clipped control is an invisible one"
        )
    # And the readout does not run under the buttons it sits beside -- the
    # overlap guard only covers icon buttons, and text against a button is the
    # same bug with neither of them an icon.
    by_name = dict(rects)
    assert by_name["readout"][2] <= by_name["icon:?"][0] + 1.0


def test_no_two_of_a_panes_icon_buttons_are_drawn_on_top_of_each_other(
    app_ctx, imgui_ctx
):
    """The other half of the overflow guard, and the half that misses.

    ``same_line`` past the edge draws a control where it cannot be seen;
    ``same_line`` onto a spot another control already holds draws it where it
    cannot be *told apart*, and the later item takes the click. The library's
    (?) landed exactly on its select-all tick -- restored as a pair (the
    manual integrity test enforces the entry and the call site) but never
    looked at, because ``render.help_button`` right-aligns with an
    unconditional ``same_line`` and the filter row had already put the tick at
    the right edge. So the tick's clicks opened the manual.

    Icon buttons only: they are square, unlabelled and interchangeable at a
    glance, which is what makes an overlap between two of them invisible.
    """
    imgui, _renderer = imgui_ctx

    from warlock.studio import layout as layout_mod
    from warlock.studio import widgets as widgets_mod
    from warlock.studio.manual import render as manual_render
    from warlock.studio.panes import library
    from warlock.studio.tokens import sp

    _seeded(app_ctx)
    rects: list[tuple[str, tuple[float, float, float, float]]] = []
    real = widgets_mod.icon_button

    def spy(label, *args, **kwargs):
        result = real(label, *args, **kwargs)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        rects.append((label.split("##")[-1], (lo.x, lo.y, hi.x, hi.y)))
        return result

    widgets_mod.icon_button = spy
    manual_render.widgets.icon_button = spy
    try:
        imgui.new_frame()
        imgui.set_next_window_size((sp(layout_mod.SIDEBAR_W) + 24, 900))
        imgui.begin("##host")
        if layout_mod.pane_child("library", (sp(layout_mod.SIDEBAR_W), 0)):
            library.draw(app_ctx)
        imgui.end_child()
        imgui.end()
        imgui.render()
    finally:
        widgets_mod.icon_button = real
        manual_render.widgets.icon_button = real

    for i, (name_a, a) in enumerate(rects):
        for name_b, b in rects[i + 1 :]:
            overlaps = a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
            assert not overlaps, (
                f"{name_a} at {a} and {name_b} at {b} occupy the same pixels; "
                "the one drawn second takes the click"
            )
