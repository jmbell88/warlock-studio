"""Every pane, built for real, once.

Not a screenshot test -- it asserts nothing about what anything looks like. It
asserts that a frame containing every panel can be *built*: no missing imgui
begin/end pair, no attribute that moved, no pane reaching for state that is
only set once a job exists. That class of mistake is otherwise found by opening
the app, and there are eight panels.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from warlock.service import jobs as svc_jobs
from warlock.studio import icons, toolbar
from warlock.studio.app_ctx import Ctx
from warlock.studio.jobs_cache import JobsCache
from warlock.studio.settings import Settings
from warlock.studio.state import AppState, Eta, Filters


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


@pytest.fixture(scope="module")
def imgui_ctx(gl):
    """An imgui context with a real renderer, over the standalone GL context.

    The renderer is needed even though nothing is presented: imgui 1.92 hands
    its font atlas to the backend, and a context whose backend never claims it
    cannot finish a frame.

    **Module-scoped, not session-scoped, and that is a correctness fix rather
    than a tuning one.** Only this file uses this fixture, so the two scopes
    cost exactly the same -- but a session-scoped context stays alive after the
    last test here, and the next file in this xdist worker that builds its own
    then has *two* contexts over the one GL context, which is a Windows access
    violation rather than a failure. It reproduced on unmodified master:
    ``uv run pytest -n 0 tests/test_studio_smoke.py tests/test_studio_controls.py``
    dies in ``test_component_gallery_builds_every_state``. The default run
    never showed it because ``--dist loadfile`` happened to put those two files
    in different workers, and "happened to" is not an invariant -- adding one
    unrelated test file re-deals that hand.
    """
    from imgui_bundle import imgui

    from warlock.studio import imgui_backend, theme

    # A standalone context has no default framebuffer; the renderer targets
    # ctx.screen, so give it one that exists.
    prev_screen = type(gl).__dict__.get("screen")
    fbo = gl.simple_framebuffer((1600, 950))
    fbo.use()
    type(gl).screen = property(lambda _self: fbo)

    # Every collapsing section forced open: a frame of collapsed headings would
    # build without ever touching the code this test exists to exercise.
    #
    # Put back on the way out, which it never used to be. It is a module global
    # and this fixture outlived the file, so every later test in the worker drew
    # its sections forced open -- the same shape as the ``vram._published`` leak
    # ``tests/conftest.py`` resets, and invisible for the same reason.
    from warlock.studio import widgets

    prev_force = widgets.FORCE_SECTIONS_OPEN
    widgets.FORCE_SECTIONS_OPEN = True

    imgui.create_context()
    io = imgui.get_io()
    # **No ini.** imgui's default is to persist every window's position, size
    # and *collapsed* flag to ``imgui.ini`` in the working directory, and to
    # load it back on the next context -- so a test that synthesises a click
    # which happens to land on a title bar twice inside the double-click window
    # collapses the shared host window, writes that to disk, and every later
    # run of the file finds a window that submits no items at all. That is
    # eighteen failures in files nobody touched, reproducing from a gitignored
    # artefact, which is the worst kind. Tests supply their own geometry
    # anyway; nothing here wants last run's.
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    theme.apply(imgui)
    renderer = imgui_backend.ImguiRenderer(gl)
    yield imgui, renderer
    renderer.shutdown()
    imgui.destroy_context()
    widgets.FORCE_SECTIONS_OPEN = prev_force
    if prev_screen is not None:
        type(gl).screen = prev_screen


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


def test_the_first_run_overlay_builds_and_dismisses(app_ctx, imgui_ctx, monkeypatch):
    from warlock.studio.panes import first_run

    app_ctx.first_run = True
    app_ctx.first_run_info = {
        "gpu_name": "Test GPU",
        "vram_total_gib": 24.0,
        "vram_free_gib": 20.0,
        "three_d": {"ready": True, "detail": "Ready"},
        "images": {"ready": True, "detail": "Ready"},
        "rigging": {"ready": False, "detail": "Needs Blender (bpy)."},
        "rows": [
            {"label": "TRELLIS.2 GGUF weights", "present": False},
            {"label": "SDXL 1.0", "present": False},
        ],
        "total_gib": 23.1,
        "download_gib": 23.1,
        "disk_refusal": None,
    }
    monkeypatch.setattr(first_run.widgets, "primary_button", lambda *_a, **_k: False)
    monkeypatch.setattr(
        first_run.controls,
        "button",
        lambda label, *_a, **_k: label == "Not now",
    )

    _frame(imgui_ctx, lambda: first_run.draw(app_ctx))
    assert app_ctx.first_run is False
    assert first_run.marker_path(app_ctx.svc.config).is_file()
    _frame(imgui_ctx, lambda: first_run.draw(app_ctx))


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


def test_the_2d_pane_builds_every_output_kind(app_ctx, imgui_ctx):
    """Each of the three draws a different set of sections, and two of them
    swap the count radios for a bare seed field. Drawn rather than reasoned
    about because the failure this catches is a layout one: ``same_line`` past
    the content edge puts a control nowhere, and no pure test sees it."""
    from warlock.studio.panes import settings_2d

    app_ctx.state.form_2d["prompt"] = "mossy dungeon"
    for asset_type in ("model_3d", "seamless_tile", "tileset_top_down"):
        app_ctx.state.form_2d["asset_type"] = asset_type
        _frame(imgui_ctx, lambda: settings_2d.draw(app_ctx))


def test_the_2d_pane_builds_both_arms_of_the_sheet_output(app_ctx, imgui_ctx):
    """The two arms draw different controls off the same form dict, and the
    sprite arm's are the ones a stale tile-arm value can reach."""
    from warlock.service import tilesheets as svc_tilesheets
    from warlock.studio.panes import settings_2d

    app_ctx.state.form_2d["prompt"] = "a hooded ranger"
    # Every view the service offers, off the service's own list rather than a
    # literal here: the pair this used to hold could not have grown a third
    # entry without somebody remembering to edit it, which is exactly how a new
    # view gets shipped having never been drawn in a frame.
    arms = [("tile", view) for view in svc_tilesheets.VIEWS]
    # ...plus the old spelling a stored job can still carry, and the sprite
    # arm, whose controls are the ones a stale tile-arm value reaches.
    arms += [("tile", "orthogonal"), ("sprite", "top_down")]
    for sheet_type, view in arms:
        app_ctx.state.form_2d["asset_type"] = (
            "sprite_turnaround"
            if sheet_type == "sprite"
            else {
                "orthogonal": "tileset_top_down",
                "top_down": "tileset_top_down",
                "three_quarter": "tileset_three_quarter",
                "isometric": "tileset_isometric",
            }[view]
        )
        _frame(imgui_ctx, lambda: settings_2d.draw(app_ctx))


def test_the_2d_pane_builds_every_tile_layout(app_ctx, imgui_ctx):
    """Three layouts, three different sets of controls, drawn in a real frame.

    Off the service's own list rather than a literal here, for the reason the
    view sweep above gives -- and with Advanced open, because the tile size is
    the one control whose *menu* is a function of the layout and a segmented row
    whose current value is off its own menu is the failure no pure test sees.
    """
    from warlock.service import tilesheets as svc_tilesheets
    from warlock.studio.panes import settings_2d

    app_ctx.state.form_2d["prompt"] = "mossy dungeon"
    app_ctx.state.form_2d["asset_type"] = "tileset"
    app_ctx.state.form_2d["materials"] = "mossy stone\ncracked earth"
    app_ctx.state.form_2d["inner_terrain"] = "wet grass"
    app_ctx.state.form_2d["outer_terrain"] = "dark water"
    app_ctx.state.create_advanced = True
    for mode in svc_tilesheets.TILE_MODES:
        app_ctx.state.form_2d["tile_mode"] = mode
        # A 48 px tile is legal for the grid and not for the other two: the
        # stored value the seamless layouts have to survive drawing.
        app_ctx.state.form_2d["tile_size"] = "48"
        _frame(imgui_ctx, lambda: settings_2d.draw(app_ctx))


def test_the_2d_pane_draws_the_pixel_look_on_both_arms(app_ctx, imgui_ctx, tmp_path):
    """The palette picker, the dither box and the sprite arm's outline row, in a
    real frame, **with a palette installed**.

    The install is the point. The combo is drawn only when the directory has
    something in it, so a sweep run against an empty palette folder never
    executes that branch at all -- which is exactly how a control ships having
    never been drawn. Both arms, because only one of them draws the outline row
    and only one of them lists palettes through ``sprite_palettes``.
    """
    from warlock.studio.panes import inspector, settings_2d

    directory = tmp_path / "smoke-palettes"
    directory.mkdir(exist_ok=True)
    (directory / "duo.hex").write_text("#1a1c2c\n#f4f4f4\n", encoding="utf-8")
    app_ctx.svc.config.palette_dir = directory
    app_ctx.state.palettes = None
    # Pinned, not assumed. Without this the whole test passes against an empty
    # directory with the combo never drawn -- which is the shape of every
    # fixture in this repo that has quietly stopped covering its branch.
    assert inspector.palette_names(app_ctx) == ["duo"]
    app_ctx.state.form_2d["prompt"] = "a hooded ranger"
    app_ctx.state.form_2d["palette"] = "duo"
    app_ctx.state.form_2d["dither"] = True
    app_ctx.state.create_advanced = True
    for asset_type, outline in (
        ("tileset_top_down", "none"),
        ("sprite_turnaround", "none"),
        ("sprite_turnaround", "inner"),
        ("sprite_turnaround", "outer"),
    ):
        app_ctx.state.form_2d["asset_type"] = asset_type
        app_ctx.state.form_2d["outline"] = outline
        _frame(imgui_ctx, lambda: settings_2d.draw(app_ctx))


def test_the_sheet_output_pins_the_count_to_one(app_ctx, imgui_ctx):
    """Both doors refuse a batch, so the radios are not drawn -- and the value
    is persisted, so a 4 left over from the Object output has to be written
    back rather than merely ignored."""
    from warlock.studio.panes import settings_2d

    app_ctx.state.form_2d["prompt"] = "mossy dungeon"
    app_ctx.state.form_2d["count"] = 4
    app_ctx.state.form_2d["asset_type"] = "tileset_top_down"
    _frame(imgui_ctx, lambda: settings_2d.draw(app_ctx))
    assert app_ctx.state.form_2d["count"] == 1


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
        json.dumps({"params": {param: {str(value): {"n": 8, "accepts": 6, "wilson_low": 0.41}}}}),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("pane", "param", "value"),
    [("settings_2d", "base_model", "turbo"), ("settings_3d", "platform", "pc")],
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
    # The base-model combo -- and so the hint attached to it -- is an Advanced
    # control now: automatic routing names the resolved recipe and returns
    # before the combo is reached, so the hint has nothing to hang off.
    app_ctx.state.form_2d["model_mode"] = "advanced"

    measured: list[float] = []
    # The 3D pane draws its hints through hint_text; the 2D pane's surviving
    # hints (base_model/style_lora/lora_weight) go through secondary after a
    # same_line, so both are spied.
    real_hint, real_secondary = widgets.hint_text, widgets.secondary

    def _spying(real):
        def spy(text):
            right = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
            real(text)
            measured.append(imgui.get_item_rect_max().x - right)

        return spy

    widgets.hint_text = _spying(real_hint)
    widgets.secondary = _spying(real_secondary)
    try:
        _frame(
            imgui_ctx,
            lambda: (
                layout_mod.pane_child(pane, (sp(layout_mod.SIDEBAR_W), 0)) and module.draw(app_ctx),
                imgui.end_child(),
            ),
        )
    finally:
        widgets.hint_text = real_hint
        widgets.secondary = real_secondary
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
        app_settings,
        candidates_panel,
        clay_bridge,
        clay_outliner,
        clay_props,
        clay_tools,
        inker_colors,
        inker_generate,
        inker_menu,
        inker_picker,
        inker_tools,
        inspector,
        library,
        pose_panel,
        retarget_panel,
        settings_2d,
        settings_3d,
        sheet_panel,
    )
    from warlock.studio.tokens import sp

    job_id = _seeded(app_ctx)
    app_ctx.state.mode = "create"
    app_ctx.state.create_stage = "mesh"
    app_ctx.rigging_available = True
    app_ctx.state.form_3d["rig"] = True
    job = app_ctx.cache.get(job_id)
    # An undecided candidate group, so the picker's own row (a fixed-width
    # button then ``same_line`` then its status) is measured too.
    candidate = app_ctx.svc.store.create(
        "image",
        "a barrel",
        {"mesh_seed": 11},
        stage="model",
        candidate_group="grp",
        candidate_index=0,
    )
    app_ctx.svc.store.set_status(candidate, "done")
    app_ctx.cache.invalidate()
    app_ctx.cache.tick()
    app_ctx.state.select(candidate)
    app_ctx.model_rows = _model_rows()
    def _settings_health() -> None:
        """The Health category, with rows in it.

        Its own entry because the walk draws whichever category is remembered,
        and the default is Appearance -- so the two shapes this test exists for
        would never be reached. A check row chains an icon, a name, a dash and
        a wrapped detail with ``same_line``, and the actions under it are a row
        of buttons in a 640-wide column.
        """
        from warlock.doctor import Check

        before = app_ctx.state.preview.get(app_settings.CATEGORY_SLOT)
        app_ctx.state.preview[app_settings.CATEGORY_SLOT] = "health"
        app_ctx.runtime.checks = [
            Check("trellis-server.exe", True, "found", fatal=False),
            Check(
                "Blender (rigging)",
                False,
                "bpy is not installed; rigging, posing and sprite sheets are unavailable",
                fatal=False,
            ),
            Check("trellis port", False, "already in use", fatal=True),
        ]
        app_ctx.state.note_error("the worker exited before it reported anything")
        app_ctx.state.dismiss_errors()
        try:
            app_settings.draw(app_ctx)
        finally:
            if before is None:
                app_ctx.state.preview.pop(app_settings.CATEGORY_SLOT, None)
            else:
                app_ctx.state.preview[app_settings.CATEGORY_SLOT] = before

    panes = [
        ("candidates", lambda: candidates_panel.draw(app_ctx)),
        ("app-settings-health", _settings_health),
        # Joined the list when its model list stopped being read-only: a row is
        # now a checkbox, a coloured label and a button, and the sidebar it is
        # drawn in is 300 px.
        ("app-settings", lambda: app_settings.draw(app_ctx)),
        ("settings-2d", lambda: settings_2d.draw(app_ctx)),
        ("settings-3d", lambda: settings_3d.draw(app_ctx)),
        ("library", lambda: library.draw(app_ctx)),
        ("inspector", lambda: inspector.draw(app_ctx)),
        ("retarget", lambda: retarget_panel.draw(app_ctx, job)),
        ("pose", lambda: pose_panel.draw(app_ctx, job)),
        ("sheet", lambda: sheet_panel.draw(app_ctx, job)),
        ("clay-tools", lambda: clay_tools.draw(app_ctx)),
        ("clay-props", lambda: clay_props.draw(app_ctx)),
        ("clay-outliner", lambda: clay_outliner.draw(app_ctx)),
        ("clay-bridge", lambda: clay_bridge.draw(app_ctx)),
        ("inker-tools", lambda: inker_tools.draw(app_ctx)),
        ("inker-colors", lambda: inker_colors.draw(app_ctx)),
        # The two panes this wave added. Both are sidebar panes with headings,
        # a help button and full-width controls, which is exactly the shape
        # this walk measures.
        ("inker-picker", lambda: inker_picker.draw(app_ctx)),
        ("inker-generate", lambda: inker_generate.draw(app_ctx)),
        ("inker-menu", lambda: inker_menu.draw_popups(app_ctx)),
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
    app_ctx.state.mode = "create"
    app_ctx.state.create_stage = "mesh"
    _frame(imgui_ctx, lambda: inspector.draw(app_ctx))
    app_ctx.svc.store.set_status(job_id, "error", "it broke")
    app_ctx.cache.invalidate()
    app_ctx.cache.tick()
    _frame(imgui_ctx, lambda: inspector.draw(app_ctx))


def test_the_candidate_picker_builds_running_and_finished(app_ctx, imgui_ctx):
    """The picker is drawn from ``inspector.draw``, above the header, so it has
    to build with nothing selected as well as with a candidate selected -- the
    rows it lists are hidden from the library, and it is the only way back to
    them."""
    from warlock.studio.panes import inspector

    app_ctx.state.mode = "create"
    app_ctx.state.create_stage = "mesh"
    ids = []
    for index in range(2):
        job_id = app_ctx.svc.store.create(
            "image",
            "a barrel",
            {"mesh_seed": 7 + index},
            stage="model",
            candidate_group="grp",
            candidate_index=index,
        )
        ids.append(job_id)
    app_ctx.cache.invalidate()
    app_ctx.cache.tick()
    # The rows really do reach the picker through the cache -- without this the
    # frames below would draw nothing at all and pass regardless.
    from warlock.studio import candidates as candidates_mod

    assert candidates_mod.pending(app_ctx.cache.jobs) is not None

    # Nothing selected, one member still queued: Keep is not offered yet.
    app_ctx.state.select(None)
    _frame(imgui_ctx, lambda: inspector.draw(app_ctx))
    # A candidate selected, with the group settled: Keep is live.
    for job_id in ids:
        app_ctx.svc.store.set_status(job_id, "done")
    app_ctx.cache.invalidate()
    app_ctx.cache.tick()
    app_ctx.state.select(ids[0])
    _frame(imgui_ctx, lambda: inspector.draw(app_ctx))
    # And a member that failed, which offers no Keep of its own.
    app_ctx.svc.store.set_status(ids[1], "error", "it broke")
    app_ctx.cache.invalidate()
    app_ctx.cache.tick()
    app_ctx.state.select(ids[1])
    _frame(imgui_ctx, lambda: inspector.draw(app_ctx))


def test_the_retarget_panel_builds_with_and_without_a_reconstruction(
    app_ctx, imgui_ctx, monkeypatch
):
    from warlock.studio.panes import retarget_panel

    job_id = _seeded(app_ctx)
    app_ctx.state.mode = "create"
    app_ctx.state.create_stage = "mesh"
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
            # The pane reads a "3 of 16" off these two (L104).
            self.yaws = [0.0, 90.0]

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
        assert app_ctx.viewer.strip_progress == (0, 2)
        assert app_ctx.viewer.stripping is True

        _frame(imgui_ctx, lambda: sheet_panel.draw(app_ctx, job))
        assert strip.steps == 2
        assert app_ctx.viewer.stripping is False, "finished, and let go of"
        # *Let go of* is not enough: the finished image is a CPU-side PIL
        # object, but the strip still owns its render target -- a 64-square
        # texture, two multisample renderbuffers and two framebuffers -- and
        # the moderngl context sets no ``gc_mode``, so dropping the reference
        # frees none of it. ``cancel_sheet_strip`` released and the success
        # path did not, which leaked one viewport per completed preview.
        assert strip.released, "the finished strip released its render target"
        assert app_ctx.state.preview["sheet_strip"] is strip.image

        # And the release happens exactly once: the field is cleared first, so
        # the cancel inside teardown finds nothing left to free.
        strip.released = False
        app_ctx.viewer.cancel_sheet_strip()
        assert strip.released is False, "no second release from the cancel path"
    finally:
        app_ctx.viewer.gpu = None
        app_ctx.viewer._strip = None
        app_ctx.state.preview.pop("sheet_strip", None)
        sheet_panel.release_strip_texture(app_ctx)


def test_the_overlay_builds_with_a_toolbar_and_a_banner(app_ctx, imgui_ctx):
    from warlock.studio.panes import overlay

    eta = Eta()
    app_ctx.state.note_error("trellis: the exe is missing")
    app_ctx.state.note_error("The GPU worker is not running. Restart Warlock.")
    _frame(
        imgui_ctx,
        lambda: (
            overlay.doctor_banner(app_ctx),
            overlay.toolbar(app_ctx),
            overlay.progress_card(app_ctx, eta),
            overlay.placeholder(app_ctx),
        ),
    )


def test_the_overlay_offers_clear_only_where_there_is_something_to_clear(app_ctx, imgui_ctx):
    """The button's whole predicate, per stage.

    Asked per stage rather than as one ``has_model or reference``, because the
    two stages draw from two different fields -- so a Clear offered on the
    Reference stage because a *mesh* happened to be loaded would appear to do
    nothing at all.
    """
    from warlock.studio.panes import overlay

    app_ctx.clear_viewport = lambda: None
    was_mode = app_ctx.state.mode
    # ``create_stages.at`` asks the mode *and* the stage together, deliberately
    # -- ``create_stage`` is not cleared on a mode change -- so a stage set
    # without the mode is not "on the Reference stage" at all.
    app_ctx.state.mode = "create"
    try:
        app_ctx.state.create_stage = "reference"
        assert overlay._has_content(app_ctx, app_ctx.viewer) is False
        app_ctx.viewer.gpu = object()
        assert overlay._has_content(app_ctx, app_ctx.viewer) is False, (
            "a mesh is not the Reference stage's canvas"
        )
        app_ctx.state.create_stage = "mesh"
        assert overlay._has_content(app_ctx, app_ctx.viewer) is True

        # And it draws, on both stages, with content present.
        app_ctx.viewer.reference = object()
        for stage in ("reference", "mesh"):
            app_ctx.state.create_stage = stage
            _frame(imgui_ctx, lambda: overlay.toolbar(app_ctx))
    finally:
        app_ctx.viewer.gpu = None
        app_ctx.viewer.reference = None
        app_ctx.clear_viewport = None
        app_ctx.state.mode = was_mode


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


def test_the_landing_screen_builds_empty_and_with_something_to_resume(app_ctx, imgui_ctx):
    """Both states, because Home's empty state is the frame that shows none of
    its Resume rows -- which is the half a seeded smoke run never reaches."""
    from warlock.studio.panes import landing

    _frame(imgui_ctx, lambda: landing.draw(app_ctx))
    _seeded(app_ctx)
    _frame(imgui_ctx, lambda: landing.draw(app_ctx))


def test_the_landing_screen_builds_with_unsaved_work_to_offer(app_ctx, imgui_ctx):
    """The third state, and the one no seeded run reaches: a crash copy waiting.

    It used to be a modal in front of this screen and is a section at the top of
    it now, so it is part of what Home has to be able to draw -- including the
    unadoptable branch, which is a kind this build has no provider for and must
    render as a greyed row rather than vanish.
    """
    from warlock.studio import journal
    from warlock.studio.panes import landing

    app_ctx.state.recovery = [
        journal.Recovered(path=Path("sketch-pd9.ora"), kind="inker", title="sketch", at=1.0),
        journal.Recovered(path=Path("scene.wmap"), kind="plotter", title="scene", at=2.0),
        journal.Recovered(
            path=Path("mystery.bin"), kind="from-the-future", title="mystery", at=3.0
        ),
    ]
    journal.ensure_providers()
    imgui, _renderer = imgui_ctx
    _frame(imgui_ctx, lambda: landing.draw(app_ctx))

    # Drawn, not merely survived: a section that returned early would pass a
    # build-it-and-see test silently, which is the whole failure mode here.
    labels = _drawn_labels(imgui, lambda: landing.draw(app_ctx), "##home-recovery")
    assert _index_of(labels, "Recover##sketch-pd9.ora") >= 0
    assert _index_of(labels, "Recover##scene.wmap") >= 0
    assert _index_of(labels, "Discard all") >= 0
    assert _index_of(labels, "Recover##mystery.bin") == -1, "no provider, no button"


def test_the_library_builds_as_its_own_mode(app_ctx, imgui_ctx):
    """It was a sub-view of Home behind an enum; it is a single-pane mode now,
    drawn by the same call Home used to make."""
    from warlock.studio.panes import library

    _seeded(app_ctx)
    app_ctx.state.mode = "library"
    _frame(imgui_ctx, lambda: library.draw(app_ctx))


def test_the_manual_builds_embedded(app_ctx, imgui_ctx):
    """The body, in whatever host is current: no begin/end of its own to get
    wrong.

    The loader falls back to the repo's docs/manual in this checkout, so this
    parses and draws the real chapters.
    """
    from warlock.studio.manual import render

    _frame(imgui_ctx, lambda: render.draw_body(app_ctx))
    # A real chapter key, and *asserted* to be one rather than written out.
    # This said "08-shortcuts", then "12-shortcuts", and neither has ever been
    # the name of anything -- the chapter was 14 when the second spelling was
    # written and has moved three times since, most recently when the tutorials
    # reserved 01-19. ``open_at`` stores whatever it is given and the
    # drawn page falls back to "could not be loaded", so both spellings smoked
    # the not-found path rather than the renderer this test exists for, and
    # neither renumbering that moved the chapter could fail here. Looking the
    # key up means the next one cannot go stale in silence either.
    from warlock.studio.manual import loader

    shortcuts = next(c.key for c in loader.chapters() if c.key.endswith("-shortcuts"))
    app_ctx.state.manual.open_at(shortcuts, None)
    _frame(imgui_ctx, lambda: render.draw_body(app_ctx))


def test_every_step_of_every_tour_builds(app_ctx, imgui_ctx):
    """Nineteen steps, each drawn for real, in a real GL frame.

    The tour is data plus one renderer, so the thing that can break is a step
    the renderer cannot draw -- a body it cannot wrap, a chapter link on a step
    with no chapter, an anchor whose rect is missing. Two of those only happen
    on *some* steps, which is why this walks all of them rather than the first.

    Drawn with no anchor recorded, deliberately: nothing has marked a control
    in this frame, so every step takes the ``hole is None`` path. That is the
    honest worst case -- a step pointing at a control inside a collapsed
    section is exactly this -- and it is the path a first-step-only test would
    never reach.
    """
    from warlock.studio.panes import tour as tour_pane
    from warlock.studio.tour import TOURS

    for one in TOURS:
        tour_pane.start(app_ctx, one.key)
        assert app_ctx.state.tour.running, one.key
        for index in range(len(one)):
            app_ctx.state.tour.index = index
            _frame(imgui_ctx, lambda: tour_pane.draw(app_ctx))
        tour_pane.stop(app_ctx)
    assert not app_ctx.state.tour.running


def test_the_tour_draws_nothing_when_none_is_running(app_ctx, imgui_ctx):
    """The overlay is called every frame, from every mode, forever."""

    from warlock.studio.panes import tour as tour_pane

    app_ctx.state.tour.stop()
    _frame(imgui_ctx, lambda: tour_pane.draw(app_ctx))
    assert not tour_pane.is_open(app_ctx)


def test_a_tour_whose_steps_went_away_stops_rather_than_raising(app_ctx, imgui_ctx):
    """The index comes back from settings and the scripts are edited.

    A saved index past the end of a shortened tour must end the tour, not take
    the frame -- an exception here is one the user cannot dismiss, because
    dismissing it is the frame that crashed.
    """
    from warlock.studio.panes import tour as tour_pane
    from warlock.studio.tour import TOURS

    tour_pane.start(app_ctx, TOURS[0].key)
    app_ctx.state.tour.index = 9_999
    _frame(imgui_ctx, lambda: tour_pane.draw(app_ctx))
    assert not app_ctx.state.tour.running


def test_a_chapter_with_a_screenshot_uploads_and_draws_it(app_ctx, imgui_ctx):
    """The manual can show a control now, not only describe it.

    Through a real GL frame because that is where the parts meet: the parser
    makes an ``Image`` block, the renderer resolves it against
    ``loader.manual_dir()`` and asks the thumbnail cache for it at a *screenshot*
    cap -- and the cap being 256 px was the whole reason images could not be
    added before, since a screenshot reduced to 256 px is unreadable.

    The chapter is *found* rather than named. This asked for the one starting
    "02-", which was Home for as long as Home was chapter 2 and became an
    unrelated chapter the moment the tutorials reserved 01-19 -- a
    ``StopIteration`` in a test whose subject has nothing to do with numbering.
    Any chapter carrying a screenshot exercises the same seam, and only three of
    them do, so the search is also the assertion that some chapter still does.
    """
    from warlock.studio.manual import loader, parser, render

    found = (
        (c.key, b)
        for c in loader.chapters()
        for b in parser.parse(loader.load(c.key))
        if isinstance(b, parser.Image)
    )
    key, image = next(found, (None, None))
    assert image is not None, "no chapter carries a screenshot for this test to draw"
    images = [image]
    assert (loader.manual_dir() / image.path).is_file()

    app_ctx.state.manual.open_at(key, None)
    _frame(imgui_ctx, lambda: render.draw_body(app_ctx))

    texture = app_ctx.textures.get(
        f"manual:{images[0].path}",
        loader.manual_dir() / images[0].path,
        max_side=render.IMAGE_MAX_SIDE,
    )
    assert texture is not None
    assert max(texture.width, texture.height) > 256


def test_a_missing_screenshot_degrades_to_its_alt_text(app_ctx, imgui_ctx):
    """Which is why alt text is required rather than optional: the chapters
    ship and the captures are generated, so "not there yet" is an ordinary
    state and must read as a caption rather than as damage."""
    from warlock.studio.manual import parser, render

    block = parser.Image("The Inker toolbar", "img/not-generated-yet.png")
    _frame(imgui_ctx, lambda: render._draw_image(app_ctx, block))


def test_the_toc_tree_draws_its_sections_and_follows_the_scroll(app_ctx, imgui_ctx):
    """The TOC's second level, through a real frame.

    ``tests/manual/test_sections.py`` owns the three decisions behind it, all
    of them pure. What only a frame can cover is the drawing: the indent /
    unindent pairing around a row, the id suffix that keeps two chapters'
    identically-named sections apart, and the scroll read that has to happen
    inside the page child rather than the host.
    """
    from warlock.studio.manual import loader, parser, render

    ms = app_ctx.state.manual
    inker = next(c.key for c in loader.chapters() if c.key.endswith("-inker"))
    sections = loader.sections(parser.parse(loader.load(inker)))
    assert len(sections) > 5, "this test wants a chapter with a real tree"

    # The current chapter expands.
    ms.open_at(inker)
    assert ms.anchor is None, "a chapter navigation carries no section"
    _frame(imgui_ctx, lambda: render.draw_body(app_ctx))

    # A section navigation lights that row on the frame it happens, before the
    # scroll it asks for has landed anywhere.
    ms.open_at(inker, sections[-1].anchor)
    assert ms.anchor == sections[-1].anchor
    _frame(imgui_ctx, lambda: render.draw_body(app_ctx))

    # And a search expands every matching chapter instead of only the open one.
    ms.search = "palette"
    _frame(imgui_ctx, lambda: render.draw_body(app_ctx))
    ms.search = "zzz-matches-nothing"
    _frame(imgui_ctx, lambda: render.draw_body(app_ctx))
    ms.search = ""


def test_the_manual_overlay_builds_over_a_mode(app_ctx, imgui_ctx):
    """Its own window now (the UI redesign, wave 3), which is a begin/end to get
    wrong -- and it is drawn *outside* the host, so it cannot be smoked by the
    pane pass that covers everything else.

    Closed as well as open: the early return is the branch every frame in the
    app but a handful takes.
    """
    from warlock.studio.manual import render

    imgui, renderer = imgui_ctx

    def frame() -> None:
        imgui.new_frame()
        imgui.set_next_window_size((1200, 900))
        imgui.begin("##host")
        imgui.text("behind")
        imgui.end()
        render.draw_overlay(app_ctx)
        imgui.render()
        renderer.render(imgui.get_draw_data())

    app_ctx.state.manual.open = False
    frame()
    from warlock.studio.manual import loader

    shortcuts = next(c.key for c in loader.chapters() if c.key.endswith("-shortcuts"))
    render.open_at(app_ctx, (shortcuts, None))
    assert app_ctx.state.manual.open
    # Twice: the first is the appearing frame, whose alpha and rise come from
    # ``popover_enter``; the second is the settled one.
    frame()
    frame()
    render.close(app_ctx)
    frame()


def _model_rows() -> list[dict]:
    """One row of each shape the pane draws: present, missing-and-fetchable,
    and missing-with-nothing-to-fetch."""
    return [
        {
            "row_key": "base:turbo",
            "kind": "base",
            "key": "turbo",
            "label": "SDXL-Turbo (fast)",
            "present": True,
            "size_gib": 7.0,
            "downloadable": True,
            # A present row also carries what removing it would free -- and
            # whether removing it would free anything at all.
            "removable": True,
            "freed_gib": 7.0,
        },
        {
            "row_key": "base:flux_klein",
            "kind": "base",
            "key": "flux_klein",
            "label": "FLUX.2 klein-base 4B (full CFG)",
            "present": False,
            "size_gib": 16.0,
            "downloadable": True,
            # The pane must tolerate the key's *absence* everywhere else -- it
            # is written only when the service had a resolved VRAM plan -- so
            # exactly one row carries it.
            "vram": "tight",
        },
        {
            "row_key": "lora:pixelxl",
            "kind": "lora",
            "key": "pixelxl",
            "label": "Pixel art (pixel-art-xl)",
            "present": False,
            "size_gib": 0.2,
            "downloadable": True,
        },
        {
            "row_key": "matting:none",
            "kind": "matting",
            "key": "none",
            "label": "Something with no fetch",
            "present": False,
            "size_gib": 0.0,
            "downloadable": False,
        },
    ]


def test_the_settings_pane_builds(app_ctx, imgui_ctx):
    """Three times: bare, populated, and with a download in flight, because the
    pane reads its lists off the Ctx with getattr and all three must build."""
    from warlock.studio.panes import app_settings

    app_ctx.base_models = []
    app_ctx.style_loras = []
    app_ctx.model_rows = []
    _frame(imgui_ctx, lambda: app_settings.draw(app_ctx))
    app_ctx.base_models = [("turbo", "SDXL-Turbo"), ("x", "X - weights missing")]
    app_ctx.style_loras = [("", "no style LoRA"), ("ink", "Ink")]
    app_ctx.model_rows = _model_rows()
    app_ctx.model_picks.add("lora:pixelxl")
    app_ctx.rigging_available = True
    _frame(imgui_ctx, lambda: app_settings.draw(app_ctx))

    # The in-flight branch: a busy row draws a bar instead of its button, and
    # every other row's button is disabled. Submitted through the real runner
    # so the progress dict is exercised the way the pane reads it.
    import threading

    gate = threading.Event()
    key = "download:base:flux_klein"
    assert app_ctx.tasks.submit(key, gate.wait)
    app_ctx.tasks.set_progress(key, 42.0, "6.7 of ~16.0 GB")
    assert app_ctx.tasks.progress(key)["percent"] == 42.0
    try:
        _frame(imgui_ctx, lambda: app_settings.draw(app_ctx))
    finally:
        gate.set()


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


def test_the_settings_pane_draws_one_category_at_a_time(app_ctx, imgui_ctx, monkeypatch):
    """The UI redesign, wave 4.1. One category body per frame, whatever the count.

    The pane used to draw them all stacked, which is the shape the segmented
    control replaced -- so the assertion worth making is not that the switch
    renders but that the *other bodies do not run*.
    """
    from warlock.studio.panes import app_settings

    drawn: list[str] = []
    for name in ("_interface", "_models", "_storage", "_health", "_layout", "_config"):
        monkeypatch.setattr(app_settings, name, lambda _ctx, _n=name: drawn.append(_n))

    for key, expected in [
        ("appearance", ["_interface"]),
        ("models", ["_models"]),
        ("storage", ["_storage"]),
        ("health", ["_health"]),
        ("advanced", ["_layout", "_config"]),
    ]:
        drawn.clear()
        app_ctx.state.preview[app_settings.CATEGORY_SLOT] = key
        _frame(imgui_ctx, lambda: app_settings.draw(app_ctx))
        assert drawn == expected, key

    # An unknown key -- a settings file from a build with a category this one
    # does not have -- falls back rather than drawing an empty pane.
    drawn.clear()
    app_ctx.state.preview[app_settings.CATEGORY_SLOT] = "nonsense"
    _frame(imgui_ctx, lambda: app_settings.draw(app_ctx))
    assert drawn == ["_interface"]


def test_the_settings_column_is_bounded_and_centred(app_ctx, imgui_ctx):
    """A form stretched across a 5K monitor puts the label and its control at
    opposite ends of the desk. Narrower than the bound, it takes what it has."""
    from imgui_bundle import imgui

    from warlock.studio import tokens
    from warlock.studio.panes import app_settings

    seen: dict[str, tuple[float, float]] = {}

    def build(width: float, tag: str) -> None:
        imgui.begin_child(f"probe-{tag}", (width, 200))
        start = imgui.get_cursor_pos_x()
        got = app_settings._centre(tokens.sp(app_settings.CONTENT_W))
        seen[tag] = (got, imgui.get_cursor_pos_x() - start)
        # An item afterwards, or the cursor move is a boundary imgui asserts
        # about -- and the real caller's ``begin_child`` is exactly that item.
        imgui.dummy((got, 1))
        imgui.end_child()

    _frame(imgui_ctx, lambda: (build(1100, "wide"), build(300, "narrow")))
    wide_w, wide_indent = seen["wide"]
    assert wide_w == tokens.sp(app_settings.CONTENT_W)
    assert wide_indent > 0, "a wide window left the column against the left edge"
    narrow_w, narrow_indent = seen["narrow"]
    assert narrow_w < tokens.sp(app_settings.CONTENT_W)
    assert narrow_indent == 0, "a narrow window indented a column that already fills it"


def test_the_bulk_deletes_left_the_library_footer_for_settings(app_ctx, imgui_ctx):
    """The UI redesign, wave 4.1: the confirms stay in ``library``, the buttons do not.

    Under a scrolling list of assets, "Clean library..." reads as an action on
    the assets you can see; and that footer is the narrowest row in the app's
    narrowest column, which is how Prune landed off the panel's right edge
    twice in one file's history.
    """
    import inspect

    from warlock.studio.panes import app_settings, library

    footer = inspect.getsource(library._storage)
    assert "ask_prune" not in footer and "ask_clean" not in footer
    # The figure it *does* still draw describes the list above it.
    assert "job_dirs" in footer

    storage = inspect.getsource(app_settings._storage)
    assert "library.ask_prune" in storage
    assert "library.ask_clean" in storage
    # One measurement of the trash, not a second opinion about it.
    assert "library.measure_trash" in storage

    app_ctx.state.preview[app_settings.CATEGORY_SLOT] = "storage"
    _frame(imgui_ctx, lambda: app_settings.draw(app_ctx))
    _frame(imgui_ctx, lambda: app_settings.draw(app_ctx))


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
    from warlock import models

    form["base_model"] = "sdxl_cfg"
    form["style_lora"] = "render3d"
    assert settings_2d.lora_note(app_ctx, form) is None
    assert not settings_2d.validate(form)

    form["base_model"] = "flux_klein"
    note = settings_2d.lora_note(app_ctx, form)
    if note is not None:
        # No adapter in the registry fits this architecture: the whole control
        # is inert, and the note has to name bases the user can find in the
        # picker.
        assert settings_2d._base_labels(app_ctx, models.lora_bases()) in note
    else:
        # Some adapter fits it, so the control is live and merely narrowed --
        # and the stale SDXL selection must still be listed, or the value that
        # keeps Generate off is the one control the user cannot see.
        assert "render3d" in [k for k, _ in settings_2d.lora_options(app_ctx, form)]
        assert settings_2d.lora_filter_note(app_ctx, form) is not None
    assert "The style LoRA is not fitted to this model's architecture." in settings_2d.validate(
        form
    )
    # And whichever branch it is has to draw.
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
    from imgui_bundle import imgui

    from warlock.studio import inker_mode, inker_state
    from warlock.studio.panes import (
        inker_bridge,
        inker_canvas,
        inker_colors,
        inker_tiles,
        inker_tools,
    )
    from warlock.studio.tokens import sp

    job_id = _reference_job(app_ctx)
    app_ctx.state.mode = "inker"
    state = inker_mode.ensure(app_ctx)
    wants_filter: list[int] = []
    wants_convert: list[int] = []

    def build() -> None:
        """Three columns, as the app lays them out -- see the animated test.

        Stacked in one column the canvas child ends up below the tools pane and
        imgui culls a child pushed past the visible area, so a row added to the
        tools pane silently stops the canvas drawing and the texture assertions
        below fail with nothing to do with textures.
        """
        if imgui.begin_child("##paint-left", (sp(300), 0)):
            inker_tools.draw(app_ctx)
            # Between the two, as the workspace lays it out: the panel appears
            # with the document's first tileset, so most of this walk draws its
            # "nothing yet" branch and the block at the end draws the picker.
            inker_tiles.draw(app_ctx)
            inker_colors.draw(app_ctx)
            # The conversion popup is opened *and* begun from the colours pane,
            # so its opener belongs here rather than in the right column: imgui
            # namespaces a popup id by the id stack that opened it, and the two
            # panes are different child windows.
            if wants_convert:
                wants_convert.pop()
                inker_bridge.open_convert(app_ctx, tab)
        imgui.end_child()
        imgui.same_line()
        if imgui.begin_child("##paint-centre", (sp(560), 0)):
            inker_canvas.draw(app_ctx)
            # From inside the same id stack the popup is drawn in: imgui
            # namespaces a popup id by the stack that opened it, and the
            # dialogs the retired bridge panel kept are drawn by the canvas
            # (W2.2). ``open_popup`` outside a frame altogether is an access
            # violation rather than a no-op.
            if wants_filter:
                wants_filter.pop()
                inker_bridge._open_filter(app_ctx, tab)
        imgui.end_child()
        imgui.same_line()
        # The right column is the preview alone now: the layers panel is
        # deleted and its rows are the timeline's (W2.5a).

    # Empty first: the "nothing open" branch is what a user sees on arrival.
    _frame(imgui_ctx, build)

    loaded = inker_mode._load_job(app_ctx.svc, job_id)
    inker_mode.on_task_done(app_ctx, _done(f"inker-open:{job_id}", loaded))
    tab = state.active
    assert tab is not None and tab.job_id == job_id
    _frame(imgui_ctx, build)

    # A selected slice with everything on it, so the slice tool's section draws
    # its whole body rather than the "no slices yet" line -- and so the overlay
    # draws a rectangle, a pivot and a dashed centre on the canvas.
    chosen = tab.doc.add_slice((2, 2, 20, 20), pivot=(4.0, 8.0), center=(4, 4, 12, 12))
    state.slice_uid = chosen.uid

    # Every tool once: each has its own options branch.
    for tool, _label, _key in inker_state.TOOLS:
        state.tool = tool
        _frame(imgui_ctx, build)

    # The filter popup: a live preview that runs every frame while it is up,
    # then a cancel, which has to put the pixels back rather than commit them.
    state.tool = "brush"
    before = tab.doc.stack.active.pixels.copy()
    head = tab.doc.history.head
    wants_filter.append(1)
    _frame(imgui_ctx, build)
    assert state.filter_uid == tab.uid
    state.filter_params[state.filter_name] = {"brightness": 0.4, "contrast": 0.0}
    _frame(imgui_ctx, build)  # the popup body, and one preview
    assert not np.array_equal(tab.doc.stack.active.pixels, before), "the preview shows"
    assert tab.doc.history.head == head, "and pushes nothing while it is only a preview"

    # Every filter's popup body once. The FX staples brought parameters a slider
    # cannot hold -- a colour, an on/off, a choice between two numbers -- and
    # each is a branch of ``_filter_control`` that only a real frame walks.
    # Selecting one also seeds its values, which is where Invert's popup turns
    # its three toggles on.
    from warlock.studio.inker import filters as inker_filters

    for name in inker_filters.FILTERS:
        state.filter_name = name
        _frame(imgui_ctx, build)

    tab.doc.cancel_filter()
    state.filter_uid = ""
    assert np.array_equal(tab.doc.stack.active.pixels, before)
    assert tab.doc.history.head == head

    # Indexed colour: the palette section has a whole second half that only
    # exists once a document has a table, and every control in it acts on the
    # *document* rather than on the session swatch row.
    assert inker_mode.index_to(app_ctx, tab, [(0, 0, 0, 255), (255, 255, 255, 255)])
    assert tab.doc.is_palette_locked
    _frame(imgui_ctx, build)
    state.palette_slot = 1
    _frame(imgui_ctx, build)
    state.palette_usage = (tab.doc.rev, tab.doc.palette_usage())
    _frame(imgui_ctx, build)
    # Multi-slot selection and the two table-only ops it feeds, drawn: the sort
    # combo, the direction box, the ramp slider and the note under them.
    state.select_slot(0)
    state.select_slot(1, ctrl=True)
    _frame(imgui_ctx, build)
    assert tab.doc.insert_ramp(0, 1, 2)
    _frame(imgui_ctx, build)
    # The shading tool's options, which only exist once a document has a table:
    # the tool loop above ran before this one was indexed, so the live branch --
    # the direction radios and the ramp note -- is drawn here and nowhere else.
    # Both directions and both ramp notes, since each is a line the other frame
    # does not draw.
    state.tool = "shade"
    _frame(imgui_ctx, build)
    state.shade_dir = -1
    state.palette_slots = []
    _frame(imgui_ctx, build)
    state.tool = "brush"
    assert inker_mode.index_to(app_ctx, tab, None)
    assert not tab.doc.is_palette_locked
    _frame(imgui_ctx, build)

    # The conversion popup: the filter popup's shape against a whole-document
    # session, so the same three things are asserted -- the preview shows, it
    # pushes nothing while it is only a preview, and a cancel puts the pixels
    # back rather than committing them.
    # A ramp first, so the document has colours to lose: indexing it above left
    # it on two, where every conversion is the identity and the preview would
    # be indistinguishable from nothing happening.
    width, _height = tab.doc.size
    tab.doc.gradient((0, 0), (width - 1, 0), (0, 0, 0, 255), (255, 255, 255, 255))
    convert_before = tab.doc.stack.active.pixels.copy()
    convert_head = tab.doc.history.head
    wants_convert.append(1)
    _frame(imgui_ctx, build)
    assert state.convert_uid == tab.uid and state.convert_table
    # Two colours, which is what the Colours slider at its floor would build:
    # the canvas here is small enough that a default-sized table is the exact
    # set of colours already in it, and a conversion onto that is the identity.
    state.convert_table = [(0, 0, 0, 255), (255, 255, 255, 255)]
    state.convert_method = "bayer4"
    _frame(imgui_ctx, build)  # the popup body, and one preview
    assert not np.array_equal(tab.doc.stack.active.pixels, convert_before)
    assert tab.doc.history.head == convert_head
    assert tab.doc.palette is None, "a preview installs no table"

    # A *save* while the popup is up must not write the preview. `_settle` at
    # the top of every serialising path cancels the session and clears the flag,
    # so the pixels are the user's own again before anything is encoded.
    inker_mode._settle(app_ctx, tab)
    assert state.convert_uid == ""
    assert tab.doc._convert is None
    assert np.array_equal(tab.doc.stack.active.pixels, convert_before)
    assert tab.doc.history.head == convert_head
    _frame(imgui_ctx, build)

    # And the same session against a *second* tab. The popup is one imgui window
    # over one shared state object while the session lives on one document, so
    # switching tabs with it up used to cancel on whichever tab was now in front
    # -- leaving the previewed one holding pixels nobody would ever answer for.
    from warlock.studio.inker.document import Document as _Document

    wants_convert.append(1)
    _frame(imgui_ctx, build)
    assert state.convert_uid == tab.uid
    state.convert_table = [(0, 0, 0, 255), (255, 255, 255, 255)]
    _frame(imgui_ctx, build)
    assert not np.array_equal(tab.doc.stack.active.pixels, convert_before), "previewing"

    # Opening a tab switches to it -- the tab bar carries ``auto_select_new_tabs``
    # -- which is the gesture, and it is why this is driven through ``_adopt``
    # rather than through ``state.activate``: the bar would put the new tab back
    # in front on the very next frame anyway.
    other = inker_mode._adopt(app_ctx, state, _Document.blank(8, 8), path=None, title="second")
    other_before = other.doc.stack.active.pixels.copy()
    # imgui registers the new tab this frame and selects it on the next, so the
    # switch is two frames rather than one -- and how many is imgui's business,
    # not this test's.
    for _ in range(4):
        _frame(imgui_ctx, build)
        if inker_mode.active(app_ctx) is other:
            break
    assert inker_mode.active(app_ctx) is other, "the pane is drawing the new tab"
    assert state.convert_uid == "", "the stranded session is settled"
    assert tab.doc._convert is None
    assert np.array_equal(tab.doc.stack.active.pixels, convert_before), "on its own doc"
    assert np.array_equal(other.doc.stack.active.pixels, other_before), "not the new one"
    assert other.doc._convert is None

    state.close(other.uid)
    for _ in range(4):
        _frame(imgui_ctx, build)
        if inker_mode.active(app_ctx) is tab:
            break
    assert inker_mode.active(app_ctx) is tab

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

    # A tileset and a tilemap layer, last because a document holding one
    # refuses the whole-document colour conversions above by name. This is the
    # only state where the tile panel draws its picker rather than its "nothing
    # yet" branch -- and the only one that uploads an atlas texture, which the
    # close below has to give back with the rest.
    from warlock.studio.inker.tiles import strip as _strip

    tiles = np.zeros((3, 8, 8, 4), dtype=np.uint8)
    tiles[1, ..., 0] = tiles[1, ..., 3] = 255
    tiles[2, ..., 2] = tiles[2, ..., 3] = 255
    slot = tab.doc.add_tileset(_strip(tiles))
    tab.doc.add_tilemap_layer(slot.uid)
    state.tool = "tile"
    _frame(imgui_ctx, build)
    # Each behaviour once: the toggle's selected button is a pushed style
    # colour and the note under it is a different sentence per mode.
    for behavior in ("auto", "stack", "manual"):
        tab.doc.tile_behavior = behavior
        _frame(imgui_ctx, build)
    assert app_ctx.state.preview.get(f"inker_tex:{tab.uid}:tileset{slot.uid}") is not None
    state.tool = "brush"

    # Dirty, so closing asks first -- and the question is what stops a stray
    # click on the tab's x from losing an unsaved painting.
    uid = tab.uid
    inker_mode.request_close(app_ctx, tab)
    assert app_ctx.confirms.pending is not None
    app_ctx.confirms.pending.on_confirm()
    app_ctx.confirms.dismiss()
    assert state.active is None
    assert not [k for k in app_ctx.state.preview if k.startswith(f"inker_tex:{uid}:")]
    inker_mode.release_all(app_ctx)


def test_the_context_bar_draws_every_tools_own_options(app_ctx, imgui_ctx):
    """One frame per tool, so a widget key added to ``CONTEXT_WIDGETS`` with no
    drawing case behind it fails here rather than on the bar it belongs to.

    The coverage test asserts the *table* is complete; this asserts the table
    can be drawn. The bucket's three Aseprite options were the case that made
    the difference matter -- the exercise pass presses tool *groups* and leaves
    the bar showing the brush's widgets, so nothing rendered them."""
    from warlock.studio import inker, inker_mode, inker_state
    from warlock.studio.panes import inker_context

    app_ctx.state.mode = "inker"
    state = inker_mode.ensure(app_ctx)
    tab = inker_mode._adopt(
        app_ctx, state, inker.Document.blank(16, 16), path=None, title="tools"
    )
    for tool, _label, _key in inker_state.TOOLS:
        state.set_tool(tool)
        assert state.tool == tool
        _frame(imgui_ctx, lambda: inker_context.draw(app_ctx, state, tab))


def test_the_timeline_draws_group_headers_open_and_folded(app_ctx, imgui_ctx):
    """The header row is the only new *widget* in the grid, and nothing else
    renders it: the engine tests assert the row plan, and the exercise pass
    walks a blank document, which has no folders to draw. So this draws one --
    open, folded, nested and with a tag band under it -- because a header that
    unbalances the id stack takes the frame loop down and the plan alone
    cannot say."""

    from warlock.studio import inker, inker_mode
    from warlock.studio.panes import inker_timeline

    app_ctx.state.mode = "inker"
    state = inker_mode.ensure(app_ctx)
    tab = inker_mode._adopt(
        app_ctx, state, inker.Document.blank(16, 16), path=None, title="grouped"
    )
    doc = tab.doc
    for name in ("lines", "flats", "shadow"):
        doc.add_layer(name)
    outer = doc.group_layers([1, 2])
    inner = doc.group_layers([2])
    doc.add_frame()
    doc.anim.tags.append(inker.Tag(name="walk", start=0, end=1))

    def frame() -> None:
        _frame(imgui_ctx, lambda: inker_timeline.draw(app_ctx))

    frame()
    inker_timeline.toggle_fold(tab, inner.uid)
    frame()
    inker_timeline.toggle_fold(tab, outer.uid)
    frame()
    assert tab.collapsed_groups == {inner.uid, outer.uid}

    # A dissolved group must not leave its fold behind: the next draw is what
    # sweeps it, and a uid folded shut with no header to reopen it would hide
    # rows for the rest of the session.
    doc.ungroup(outer.uid)
    frame()
    assert outer.uid not in tab.collapsed_groups


def test_the_animated_inker_builds_and_gives_its_frame_textures_back(app_ctx, imgui_ctx):
    """The timeline strip, onion skinning and the playback draw path.

    Onion and playback each upload a texture per frame under the same
    ``inker_tex:{uid}:`` prefix the composite uses, which is the whole reason
    they were put there -- a fifty-frame tab is fifty textures, and the release
    at the end is what says the prefix sweep really does collect them.
    """
    from imgui_bundle import imgui

    from warlock.studio import inker, inker_mode
    from warlock.studio.panes import (
        inker_canvas,
        inker_colors,
        inker_timeline,
        inker_tools,
    )
    from warlock.studio.tokens import sp

    job_id = _reference_job(app_ctx)
    app_ctx.state.mode = "inker"
    state = inker_mode.ensure(app_ctx)

    def build() -> None:
        """The three columns the app actually lays these panes out in.

        Not a nicety. Stacked in one column, the canvas child ends up below
        the tools and colours panes, and imgui *culls* a child pushed past the
        visible area -- a culled canvas uploads no textures, so a single row
        added to the tools pane upstream failed this test with an empty texture
        list and nothing whatsoever to do with frames. Columns give each pane
        the full height, which is what the app gives them.
        """
        if imgui.begin_child("##smoke-left", (sp(300), 0)):
            inker_tools.draw(app_ctx)
            inker_colors.draw(app_ctx)
        imgui.end_child()
        imgui.same_line()
        if imgui.begin_child("##smoke-centre", (sp(560), 0)):
            inker_canvas.draw(app_ctx)
            inker_timeline.draw(app_ctx)
        imgui.end_child()
        # The right column is the preview alone now (W2.5a).

    def frame() -> None:
        _frame(imgui_ctx, build)

    loaded = inker_mode._load_job(app_ctx.svc, job_id)
    inker_mode.on_task_done(app_ctx, _done(f"inker-open:{job_id}", loaded))
    tab = state.active
    assert tab is not None

    # Still: the timeline draws nothing at all, which is what keeps a
    # non-animated document's layout byte-for-byte what it always was.
    frame()
    assert not [k for k in app_ctx.state.preview if ":frame" in k]

    inker_mode.animate(app_ctx, tab)
    assert tab.doc.anim is not None and len(tab.doc.anim.frames) == 2
    tab.doc.add_frame(link=True)
    tab.doc.anim.tags.append(inker.Tag(name="walk", start=0, end=2))
    frame()

    # Onion on: the neighbours upload.
    state.onion = True
    frame()
    assert [k for k in app_ctx.state.preview if ":frame" in k]

    # Playing: the canvas draws a cached flatten instead of the composite, and
    # every editing control is refused.
    inker_mode.toggle_play(app_ctx, tab)
    assert tab.playing and tab.busy
    for _ in range(3):
        frame()
    inker_mode.stop_play(tab)
    assert not tab.playing
    frame()

    # The tag row's rename field, which replaces the label in place rather than
    # opening a modal -- so it is a widget that only exists on some frames and
    # would otherwise never be built by anything.
    state.tag_editing = 0
    state.tag_name = "walk"
    frame()
    state.tag_editing = -1
    frame()

    # A deleted frame's texture is released rather than living until the tab is
    # closed: a clip built up and cut down over a session leaks one full-canvas
    # texture per frame that ever went. The drain is read in ``composite``, so
    # it takes one drawn frame.
    gone = tab.doc.anim.frames[-1].uid
    assert any(f"frame{gone}" in key for key in app_ctx.state.preview)
    tab.doc.remove_frame(len(tab.doc.anim.frames) - 1)
    frame()
    assert not [key for key in app_ctx.state.preview if f"frame{gone}" in key]

    uid = tab.uid
    inker_mode.request_close(app_ctx, tab)
    if app_ctx.confirms.pending is not None:
        app_ctx.confirms.pending.on_confirm()
        app_ctx.confirms.dismiss()
    assert not [k for k in app_ctx.state.preview if k.startswith(f"inker_tex:{uid}:")]
    inker_mode.release_all(app_ctx)


def test_a_finished_mesh_is_not_offered_paint(app_ctx, imgui_ctx):
    """Paint edits the *generated reference*, and a mesh job's input.png is
    whatever it was reconstructed from."""
    from warlock.studio import inker_mode

    job_id = _seeded(app_ctx)
    app_ctx.state.mode = "create"
    app_ctx.state.create_stage = "reference"
    assert not inker_mode.can_edit_job(app_ctx, app_ctx.cache.get(job_id))


def _done(key, result):
    from warlock.studio.tasks import Done

    return Done(key=key, result=result)


def _reference_job(app_ctx) -> str:
    """A finished reference with a real PNG on disk."""
    import io

    from PIL import Image

    job_id = svc_jobs.create_job(app_ctx.svc, kind="text", prompt="a barrel", output="reference")[
        "id"
    ]
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


def test_the_quiet_registers_build(app_ctx, imgui_ctx):
    """The UI redesign wave 2's three additions: a button with no resting fill, a
    borderless glyph, a primary that has stopped being accent-coloured because
    it cannot be pressed, and the shared "nothing open" screen."""
    from warlock.studio import icons, widgets

    def build():
        widgets.ghost_button("View all")
        widgets.ghost_button("Cancel", enabled=False, reason="Nothing to cancel")
        widgets.icon_button(icons.UNDO, "Undo", borderless=True)
        widgets.small_icon_button(icons.COPY, "Copy", borderless=True)
        widgets.primary_button("Generate", enabled=False, reason="A prompt is required")
        widgets.empty_state(
            icons.BOX,
            "Nothing here",
            "A hint long enough to need the wrap the centring used to measure "
            "without, which is what put its first line off centre.",
            action=("Do the thing", lambda: None),
        )
        widgets.nothing_open(
            "Start a canvas, open an image, or send one here from the library.",
            [("New canvas", lambda: None), ("Open a file...", lambda: None)],
            recent_paths=["/tmp/a.ora", "/tmp/b.ora"],
            on_open=lambda _path: None,
        )

    for _ in range(2):
        _frame(imgui_ctx, build)


@pytest.mark.parametrize("scale", [1.0, 1.5, 1.75])
def test_a_toolbar_row_never_draws_past_the_pane_it_is_in(app_ctx, imgui_ctx, scale):
    """The whole reason ``studio/toolbar.py`` exists.

    ``same_line`` past the content region clips rather than wraps, so a row
    that outgrows its pane does not get cramped -- its last buttons are simply
    not there. Every workspace's toolbar was written as a chain of them, and
    the defect is invisible at scale 1.0 in a wide window, which is the only
    configuration the rest of this file draws.
    """
    from warlock.studio import icons, tokens
    from warlock.studio import toolbar as toolbar_mod
    from warlock.studio import widgets as widgets_mod

    imgui, _renderer = imgui_ctx
    items = [
        toolbar_mod.Item("new", "New", icons.PLUS, pinned=True),
        toolbar_mod.Item("open", "Open", icons.FOLDER_OPEN),
        toolbar_mod.Item("save", "Save", icons.SAVE, priority=1),
        toolbar_mod.Item("copy", "Duplicate", icons.COPY, priority=1),
        toolbar_mod.Item("export", "Export PNG", icons.DOWNLOAD, priority=2),
        toolbar_mod.Item("settings", "Settings", icons.SETTINGS, priority=2),
    ]

    rects: list[tuple[str, float]] = []
    edge: list[float] = []
    real_icon = widgets_mod.icon_button
    real_ghost = widgets_mod.ghost_button

    def spy(real, name):
        def wrapper(label, *args, **kwargs):
            result = real(label, *args, **kwargs)
            rects.append((f"{name}:{label}", imgui.get_item_rect_max().x))
            return result

        return wrapper

    def build():
        imgui.begin_child("bar-host", (tokens.sp(300), tokens.sp(60)))
        edge.append(imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x)
        toolbar_mod.toolbar("smoke-bar", items)
        imgui.end_child()

    old_scale = tokens.SCALE
    tokens.set_scale(scale)
    widgets_mod.icon_button = spy(real_icon, "icon")
    widgets_mod.ghost_button = spy(real_ghost, "ghost")
    try:
        # Twice: the first frame has no hover history, and the tier chosen on
        # it is what the second frame is measured at.
        for _ in range(2):
            _frame(imgui_ctx, build)
    finally:
        widgets_mod.icon_button = real_icon
        widgets_mod.ghost_button = real_ghost
        tokens.set_scale(old_scale)

    assert rects, "the row drew nothing at all"
    right = edge[-1]
    for name, item_right in rects:
        assert item_right <= right + 1.0, (
            f"{name} ends {item_right - right:.0f} px past the pane edge; "
            "a clipped control is an invisible one"
        )


# --- Clay mode -------------------------------------------------------------


def _clay_tab(app_ctx, *, objects: int = 2):
    """A Clay document with objects in it, adopted as the active tab."""
    from warlock.studio import clay_mode
    from warlock.studio.clay import document as bd
    from warlock.studio.clay import primitives as bp

    doc = bd.ClayDoc()
    for i in range(objects):
        doc.add_object(
            bd.Obj(
                uid=bd.new_uid(),
                name=f"obj{i}",
                mesh=bp.box(),
                generator="box",
                params={"size": (1.0, 1.0, 1.0)},
            )
        )
    return clay_mode.adopt(app_ctx, doc, title="Scene")


def test_the_clay_panes_build_with_nothing_open(app_ctx, imgui_ctx):
    """Every one of them has to survive the state the mode opens in."""
    from warlock.studio.panes import (
        clay_bridge,
        clay_header,
        clay_outliner,
        clay_props,
        clay_tools,
    )

    # ``clay_header`` is the strip over the viewport rather than a sidebar
    # pane, and it walks with them for the reason they walk: an unbalanced
    # disable stack or a popup left open is the same defect wherever it is.
    for pane in (clay_tools, clay_props, clay_outliner, clay_bridge, clay_header):
        _frame(imgui_ctx, lambda pane=pane: pane.draw(app_ctx))


def test_the_clay_panes_build_with_a_document_and_a_selection(app_ctx, imgui_ctx):
    from warlock.studio.panes import (
        clay_bridge,
        clay_header,
        clay_outliner,
        clay_props,
        clay_tools,
    )

    tab = _clay_tab(app_ctx)
    tab.doc.select([tab.doc.objects[0].uid])
    # ``clay_header`` is the strip over the viewport rather than a sidebar
    # pane, and it walks with them for the reason they walk: an unbalanced
    # disable stack or a popup left open is the same defect wherever it is.
    for pane in (clay_tools, clay_props, clay_outliner, clay_bridge, clay_header):
        _frame(imgui_ctx, lambda pane=pane: pane.draw(app_ctx))


def test_the_clay_panes_build_while_a_save_is_in_flight(app_ctx, imgui_ctx):
    """``saving`` puts every mutating control inside ``begin_disabled``, and an
    unbalanced disable stack is exactly the class of mistake this file exists
    to catch."""
    from warlock.studio.panes import (
        clay_bridge,
        clay_header,
        clay_outliner,
        clay_props,
        clay_tools,
    )

    tab = _clay_tab(app_ctx)
    tab.doc.select([tab.doc.objects[0].uid])
    tab.saving = True
    # ``clay_header`` is the strip over the viewport rather than a sidebar
    # pane, and it walks with them for the reason they walk: an unbalanced
    # disable stack or a popup left open is the same defect wherever it is.
    for pane in (clay_tools, clay_props, clay_outliner, clay_bridge, clay_header):
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
            uid=bd.new_uid(),
            name=name,
            mesh=build(**defaults),
            generator=name,
            params=dict(defaults),
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
        uid=bd.new_uid(),
        name="Wedge",
        mesh=wedge(**defaults),
        generator="wedge",
        params=dict(defaults),
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
    _review_remove_reviewed = _main.App._review_remove_reviewed
    _retention_tick = _main.App._retention_tick
    _review_form = _main.App._review_form
    _review_axis_values = _main.App._review_axis_values
    _review_judging_card = _main.App._review_judging_card
    _review_judging_report = _main.App._review_judging_report
    _review_judging_controls = _main.App._review_judging_controls
    _review_units = _main.App._review_units
    _review_verdict = _main.App._review_verdict
    _review_findings = _main.App._review_findings
    _review_labels = _main.App._review_labels
    _review_label_panel = _main.App._review_label_panel
    _LABEL_CELL = _main.App._LABEL_CELL


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
        {
            "id": review_mode.RECENT_ID,
            "label": "Recent, unreviewed",
            "prompt": "",
            "units": [],
            "todo": 0,
        },
        {
            "id": "abcdef012345",
            "label": "lora sweep",
            "prompt": "a barrel",
            "units": units,
            "todo": 0,
            # The two fields the removal controls read. Carried here so the
            # bulk button and its retention tick are actually *built* by the
            # smoke pass rather than skipped -- an unexercised branch behind a
            # falsy default is how the Phase 5 thumbnail defect survived 9,348
            # passing tests.
            "retained": 1,
            "removable": True,
        },
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


def test_the_review_panes_build_with_a_judging_pass_running(app_ctx, imgui_ctx):
    """The entry card gives way to the in-pane controls, and the verdict panel
    grows an Accept/Reject pair above the grade row it does not replace."""
    from warlock.studio import review_mode

    app = _ReviewApp()
    state = _review_state(app_ctx)
    state.sweeps[1]["todo"] = 1
    state.units[0]["verdict"] = None
    state.judging = review_mode.JudgingPass(total=3, filed=1, order=["abcdef012345"])

    _frame(
        imgui_ctx,
        lambda: (
            app._review_runs(app_ctx, state, review_mode),
            app._review_units(state, review_mode),
            app._review_verdict(app_ctx, state, review_mode),
        ),
    )


def test_the_review_panes_build_the_judging_entry_card(app_ctx, imgui_ctx):
    from warlock.studio import review_mode

    app = _ReviewApp()
    state = _review_state(app_ctx)
    state.sweeps[1]["todo"] = 2
    assert review_mode.todo_total(state) == 2

    _frame(imgui_ctx, lambda: app._review_runs(app_ctx, state, review_mode))


def test_the_review_panes_build_the_judging_report(app_ctx, imgui_ctx):
    from warlock.studio import review_mode

    app = _ReviewApp()
    state = _review_state(app_ctx)
    review_mode.start_judging(app_ctx)
    state.judging = review_mode.JudgingPass(total=1, filed=1, order=["abcdef012345"])
    review_mode.finish_judging(app_ctx)
    assert state.judging_report is not None

    _frame(imgui_ctx, lambda: app._review_runs(app_ctx, state, review_mode))


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


def _label_pass(ctx, stage="blank", rows=3):
    """A labelling pass with real images on disk, as the listing task builds it."""
    import io as _io

    from PIL import Image as _Image

    from warlock.studio import review_mode

    state = review_mode.ensure(ctx)
    made = []
    for i in range(rows):
        job_id = _seeded(ctx)
        job_dir = ctx.svc.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        image = job_dir / "reference.png"
        buf = _io.BytesIO()
        _Image.new("RGBA", (8, 8), (40, 80, 160, 255)).save(buf, "PNG")
        image.write_bytes(buf.getvalue())
        made.append(
            {
                "job_id": job_id,
                "prompt": f"a rogue {i}",
                "status": "error" if i == 0 else "done",
                "image": image,
                "verdict": "accept" if i == 1 else None,
            }
        )
    state.labels = review_mode.LabelPass(
        stage=stage,
        rows=made,
        status={"positives": 1, "negatives": 0, "needed": 8, "trained": False},
    )
    return state


def test_the_labelling_grid_builds_and_uploads_one_cell_per_frame(app_ctx, imgui_ctx):
    """Three cells, three frames: ``StripRender``'s rule, and the reason the grid
    draws a placeholder for a cell whose texture has not been uploaded yet."""
    from warlock.studio import review_mode

    app = _ReviewApp()
    state = _label_pass(app_ctx)

    for expected in (1, 2, 3):
        _frame(
            imgui_ctx,
            lambda: (
                app._review_labels(app_ctx, state, review_mode),
                app._review_label_panel(app_ctx, state, review_mode),
            ),
        )
        assert state.labels.uploaded == expected
    # And a fourth frame is not an error, it simply has nothing left to admit.
    _frame(imgui_ctx, lambda: app._review_labels(app_ctx, state, review_mode))
    assert state.labels.uploaded == 3


def test_the_labelling_grid_builds_with_nothing_left_to_label(app_ctx, imgui_ctx):
    from warlock.studio import review_mode

    app = _ReviewApp()
    state = _label_pass(app_ctx, rows=0)

    _frame(
        imgui_ctx,
        lambda: (
            app._review_labels(app_ctx, state, review_mode),
            app._review_label_panel(app_ctx, state, review_mode),
        ),
    )


def test_the_labelling_grid_builds_while_the_listing_is_still_reading(app_ctx, imgui_ctx):
    from warlock.studio import review_mode

    app = _ReviewApp()
    state = _label_pass(app_ctx, rows=0)
    state.labels.loading = True

    _frame(imgui_ctx, lambda: app._review_labels(app_ctx, state, review_mode))


def test_the_runs_pane_offers_both_labelling_passes(app_ctx, imgui_ctx):
    from warlock.studio import main, review_mode

    app = _ReviewApp()
    state = _review_state(app_ctx)

    seen: list[str] = []
    imgui, _renderer = imgui_ctx
    real = imgui.selectable

    def spy(label, *args, **kwargs):
        seen.append(label)
        return real(label, *args, **kwargs)

    imgui.selectable = spy
    try:
        _frame(imgui_ctx, lambda: app._review_runs(app_ctx, state, review_mode))
    finally:
        imgui.selectable = real
    for title in main._LABEL_TITLES.values():
        assert any(title in text for text in seen), title


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
                    {
                        "key": "abc123",
                        "vector": {"lora_weight": 0.9, "platform": "pc"},
                        "n": 8,
                        "accepts": 6,
                        "accept_rate": 0.75,
                        "top_reasons": [["holes", 2]],
                        "jobs": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    app = _ReviewApp()
    state = _review_state(app_ctx)

    _frame(imgui_ctx, lambda: app._review_verdict(app_ctx, state, review_mode))


def test_the_2d_pane_builds_with_stale_vector_preset_settings(app_ctx, imgui_ctx):
    # The vector-preset save mechanism retired with the taxonomy; a
    # studio_settings.json still carrying old entries must not break the pane.
    from warlock.studio.panes import settings_2d

    app_ctx.settings.set("vector_presets", {"chests": {"genre": "fantasy", "platform": "pc"}})
    _frame(imgui_ctx, lambda: settings_2d.draw(app_ctx))


def test_the_inspector_builds_its_verdict_section_with_and_without_staged_tags(app_ctx, imgui_ctx):
    from warlock.studio.panes import inspector

    app_ctx.state.mode = "create"
    app_ctx.state.create_stage = "mesh"
    job_id = _seeded(app_ctx)
    app_ctx.svc.store.set_stage(job_id, "model")
    app_ctx.cache.invalidate()
    app_ctx.cache.tick()
    job = app_ctx.svc.store.get(job_id)

    _frame(imgui_ctx, lambda: inspector._verdict(app_ctx, job))
    # A staged tag draws its button in the selected style, which is a different
    # code path through ``tag_toggles`` -- the push/pop pair has to balance.
    inspector.toggle_tag(app_ctx.state, job_id, "holes")
    inspector.toggle_tag(app_ctx.state, job_id, "clean-shape")
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


@pytest.mark.parametrize("scale", [1.0, 1.5, 1.75])
def test_the_rail_fits_the_resize_floor_at_every_scale(app_ctx, imgui_ctx, scale):
    """What the header's three tests were about, asked of the control that
    replaced it (the UI redesign, wave 3).

    Those three pinned a right-aligned strip whose width had to be *measured*
    rather than reserved, because ``same_line`` past the content region clips
    rather than wraps -- a control drawn out there is simply gone. A column has
    no such strip: it is one width for every item, and what can go wrong
    instead is that it does not fit the window's own floor, or that its items
    overlap each other down the y axis. Both are asserted here, at the resize
    floor, at the scales the old tests ran at plus the one in the middle.
    """
    imgui, _renderer = imgui_ctx
    from warlock.doctor import Check
    from warlock.studio import main as main_mod
    from warlock.studio import modes as modes_mod
    from warlock.studio import rail as rail_mod
    from warlock.studio import tokens

    # Something failing, so the footer draws its badge: the widest the rail
    # ever is vertically, and the case the old health-label test was about.
    app_ctx.runtime.checks = [
        Check("trellis-server.exe", False, "not found", fatal=True),
        Check("CUDA", False, "torch.cuda.is_available() is False", fatal=False),
    ]
    fake = SimpleNamespace(app_ctx=app_ctx, layout=SimpleNamespace(rail="icons"))
    fake._set_mode = lambda key: None

    boxes: list[tuple[str, float, float]] = []
    real_item = rail_mod._item

    def spy(key, label, icon, box, **kwargs):
        result = real_item(key, label, icon, box, **kwargs)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        boxes.append((key, lo.y, hi.y))
        return result

    old_scale = tokens.SCALE
    tokens.set_scale(scale)
    rail_mod._item = spy
    try:
        imgui.new_frame()
        # The resize floor: what the window cannot be shrunk below, and so the
        # size every mode has to survive. Undecorated, because the host window
        # is (``no_decoration`` in ``_build_ui``) and a title bar this test
        # invented would be ~40 px of height the app does not spend.
        imgui.set_next_window_size(main_mod.MIN_SIZE)
        imgui.begin(
            "##min-window",
            None,
            imgui.WindowFlags_.no_decoration.value | imgui.WindowFlags_.no_saved_settings.value,
        )
        rail_mod.tick(fake.layout)
        top = imgui.get_cursor_screen_pos().y
        bottom = top + imgui.get_content_region_avail().y
        rail_mod.draw(fake, app_ctx)
        imgui.end()
        imgui.render()
    finally:
        rail_mod._item = real_item
        tokens.set_scale(old_scale)

    keys = [key for key, _lo, _hi in boxes]
    assert set(keys) >= set(modes_mod.KEYS), (
        f"missing rail items: {set(modes_mod.KEYS) - set(keys)}"
    )
    assert "health" not in keys, "the rail contains workspace destinations only"
    for key, lo, hi in boxes:
        assert lo >= top - 1.0 and hi <= bottom + 1.0, (
            f"{key} is drawn from {lo:.0f} to {hi:.0f} outside the rail's "
            f"{top:.0f}..{bottom:.0f} at UI scale {scale}; an item outside the "
            "column is an unreachable mode"
        )
    ordered = sorted(boxes, key=lambda row: row[1])
    for (a, _alo, ahi), (b, blo, _bhi) in zip(ordered, ordered[1:], strict=False):
        assert ahi <= blo + 1.0, f"{a} and {b} overlap in the rail at UI scale {scale}"


def test_the_health_badge_is_drawn_only_when_something_is_failing(app_ctx, imgui_ctx):
    """A permanent "OK" badge is noise: the app said "Everything checks out" in
    the corner of every frame it ever drew. The diagnostics list stays reachable
    when it is green through the palette, which is where a thing you
    occasionally want and never need belongs."""
    from warlock.doctor import Check
    from warlock.studio import status_bar

    app_ctx.runtime.checks = [Check("trellis-server.exe", True, "found", fatal=True)]
    app_ctx.state.errors = []
    assert not any(item.key == "health" for item in status_bar.items(app_ctx))

    app_ctx.runtime.checks = [Check("CUDA", False, "unavailable", fatal=False)]
    assert any(item.key == "health" for item in status_bar.items(app_ctx))


def test_the_rail_carries_a_door_to_the_shortcut_sheet(app_ctx, imgui_ctx):
    """Ctrl+/ and the palette both open the sheet, and both are things you have
    to already know. The header's ``?`` button was its only visible door and
    went with the header, so the footer carries one -- and it sets the same
    one-shot flag, because a popup opened inside the rail child would be
    scoped to the child."""
    from warlock.studio import menus

    row = next(row for row in menus.specs(app_ctx) if row.identity == "command:shortcuts")
    assert row.path == ("Help",)
    app_ctx.state.shortcuts_requested = False
    row.callback()
    assert app_ctx.state.shortcuts_requested


def test_the_expanded_rail_gives_way_before_the_columns_do(app_ctx, imgui_ctx):
    """The preference is untouched by the room: a window dragged narrow draws
    the collapsed rail and dragging it back restores the labels, because what
    the user chose and what fits are two different facts.

    Without the rule the inspector is the column that falls off the edge --
    UX-01's whole story, one column further left.
    """
    imgui, _renderer = imgui_ctx
    from warlock.studio import main as main_mod
    from warlock.studio import motion, tokens
    from warlock.studio import rail as rail_mod

    layout = SimpleNamespace(rail="labels")
    old_scale = tokens.SCALE
    io = imgui.get_io()
    old_display = (io.display_size.x, io.display_size.y)
    seen: dict[str, float] = {}
    try:
        for name, size, scale in (
            ("wide", (2400.0, 1200.0), 1.0),
            ("floor", (float(main_mod.MIN_SIZE[0]), float(main_mod.MIN_SIZE[1])), 1.5),
        ):
            tokens.set_scale(scale)
            # The *viewport*, not a window: the rail is measured before the
            # host is begun, so what it asks about is the display it will be
            # drawn on -- which in the app is the window itself.
            io.display_size = size
            imgui.new_frame()
            imgui.begin(f"##{name}")
            # Settled rather than eased: the width animates, and what is being
            # asserted is the target it is heading for.
            motion.forget(rail_mod._WIDTH_KEY)
            seen[name] = rail_mod.tick(layout)
            imgui.end()
            imgui.render()
    finally:
        io.display_size = old_display
        tokens.set_scale(old_scale)
        motion.forget(rail_mod._WIDTH_KEY)

    assert seen["wide"] == rail_mod.RAIL_EXPANDED_W
    assert seen["floor"] == rail_mod.RAIL_W
    assert layout.rail == "labels", "the preference is not what gave way"


def test_no_two_of_a_panes_icon_buttons_are_drawn_on_top_of_each_other(app_ctx, imgui_ctx):
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
    # (name, left, right, frame_padding.x, button_text_align.x, em) captured
    # *inside* the helper, because the whole question is what it pushes.
    geometry: list[tuple[str, float, float, float, float, float]] = []
    real = widgets_mod.icon_button
    real_small = widgets_mod.small_icon_button
    real_button = imgui.button

    def button_spy(label, *args, **kwargs):
        style = imgui.get_style()
        pad_x = float(style.frame_padding.x)
        align_x = float(style.button_text_align.x)
        em = float(imgui.get_font_size())
        result = real_button(label, *args, **kwargs)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        geometry.append((label.split("##")[-1], lo.x, hi.x, pad_x, align_x, em))
        return result

    def _watch(fn):
        def spy(label, *args, **kwargs):
            imgui.button = button_spy
            try:
                result = fn(label, *args, **kwargs)
            finally:
                imgui.button = real_button
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            rects.append((label.split("##")[-1], (lo.x, lo.y, hi.x, hi.y)))
            return result

        return spy

    widgets_mod.icon_button = _watch(real)
    widgets_mod.small_icon_button = _watch(real_small)
    manual_render.widgets.icon_button = widgets_mod.icon_button
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
        widgets_mod.small_icon_button = real_small
        manual_render.widgets.icon_button = real
        imgui.button = real_button

    # Where the glyph actually lands inside the square, by imgui's own rule:
    # the label gets ``width - 2 * frame_padding.x``, and ``RenderTextClipped``
    # clamps the alignment offset at zero rather than centring into a gap that
    # has gone negative. A Lucide glyph advances about a full em, so an em is
    # what the square has to be able to hold -- the theme's 9 px of horizontal
    # padding leaves less than that, which pinned every icon in the app to the
    # left edge of its button with its right-hand pixels clipped off. The
    # headless atlas has no Lucide in it, so this measures the *rule* against a
    # full-em glyph rather than against whichever box this machine substitutes.
    assert geometry, "no icon button was drawn; the spy is measuring nothing"
    for name, x0, x1, pad_x, align_x, em in geometry:
        inner = (x1 - x0) - 2 * pad_x
        assert inner >= em - 0.5, (
            f"{name}: {inner:.1f} px of text rect in a {x1 - x0:.1f} px button "
            f"cannot hold a {em:.1f} px glyph; it is clipped on the right"
        )
        offset = max(0.0, align_x * (inner - em))
        glyph_centre = x0 + pad_x + offset + em * 0.5
        assert abs(glyph_centre - (x0 + x1) * 0.5) <= 0.5, (
            f"{name}: a glyph lands centred on {glyph_centre:.1f} in a button "
            f"spanning {x0:.1f}..{x1:.1f}, which is off-centre by "
            f"{glyph_centre - (x0 + x1) * 0.5:.1f} px"
        )

    for i, (name_a, a) in enumerate(rects):
        for name_b, b in rects[i + 1 :]:
            overlaps = a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
            assert not overlaps, (
                f"{name_a} at {a} and {name_b} at {b} occupy the same pixels; "
                "the one drawn second takes the click"
            )


def test_two_rows_with_the_same_icon_and_tooltip_keep_separate_hover_state(imgui_ctx):
    """Every visible layer's eye button in Inker draws the identical icon glyph
    and the identical "Hide this layer" tooltip -- the only thing that tells
    two rows' buttons apart to imgui is the ``push_id`` each row wraps them in.

    The hover-fade cache in :func:`widgets._glyph_button` keyed itself off the
    icon and tooltip text alone, with nothing from that id stack -- so every
    row sharing a glyph and a tooltip shared one animation slot. Hovering one
    row's eye then painted a *different* row's eye as hovered, because
    ``note_hover`` overwrites the shared slot mid-frame and whichever row of
    the pair draws second in that same frame reads the just-written value back
    out before it has a chance to decay -- the "wrong icon lights up" bug.
    """
    imgui, _renderer = imgui_ctx
    from warlock.studio import widgets

    seen: list[str] = []
    real_note_hover = widgets.note_hover

    def spy(key, hovered):
        seen.append(key)
        real_note_hover(key, hovered)

    widgets.note_hover = spy
    try:
        imgui.get_io().add_mouse_pos_event(-1, -1)
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((400, 400))
        imgui.begin("##hover-collision", None, imgui.WindowFlags_.no_decoration.value)
        imgui.push_id("row-a")
        widgets.icon_button("##visible", "Hide this layer", borderless=True)
        imgui.pop_id()
        imgui.push_id("row-b")
        widgets.icon_button("##visible", "Hide this layer", borderless=True)
        imgui.pop_id()
        imgui.end()
        imgui.render()
        _renderer.render(imgui.get_draw_data())
    finally:
        widgets.note_hover = real_note_hover

    assert len(seen) == 2
    assert seen[0] != seen[1], (
        "two different rows' eye buttons shared one hover-animation key "
        f"({seen[0]!r}); hovering one paints the other"
    )


@pytest.mark.parametrize("stage", ["reference", "tile", "model", "rig", "sheet"])
def test_a_library_card_says_which_kind_of_asset_it_is(app_ctx, imgui_ctx, stage):
    """The card never read ``stage``, and the only tells were a 72 px
    thumbnail, the "from a reference" note and the *absence* of a quality
    badge -- so a picture and the mesh made from it looked the same at a
    glance. Asserted through the chip the card actually draws rather than
    through the table it reads: a badge nothing calls is the failure here."""
    imgui, _renderer = imgui_ctx
    from warlock.studio import layout as layout_mod
    from warlock.studio import theme
    from warlock.studio import widgets as widgets_mod
    from warlock.studio.panes import library
    from warlock.studio.tokens import sp

    job_id = _seeded(app_ctx)
    app_ctx.svc.store.set_stage(job_id, stage)
    app_ctx.cache.invalidate()
    app_ctx.cache.tick()

    chips: list[tuple[str, tuple[float, ...]]] = []
    real = widgets_mod._chip

    def spy(label, colour, fill):
        chips.append((label, tuple(colour)))
        return real(label, colour, fill)

    widgets_mod._chip = spy
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
        widgets_mod._chip = real

    icon, word = widgets_mod.STAGE_BADGES[stage]
    drawn = dict(chips)
    assert f"{icon} {word}" in drawn, f"no {stage} badge among {list(drawn)}"
    # Colour is the status pill's encoding and only its: two hues side by side
    # on one line fight rather than add, so the kind chip is always muted.
    assert drawn[f"{icon} {word}"] == tuple(theme.rgba(theme.MUTED))


def test_the_stage_badges_tell_pixels_from_geometry_by_glyph():
    """The icon is what carries the distinction, so no glyph may serve both
    sides of it -- and the words stay ASCII, because imgui's default atlas is
    Basic Latin plus Latin-1 and anything outside it renders as a box."""
    from warlock.studio import widgets

    flat = {widgets.STAGE_BADGES[k][0] for k in ("reference", "tile")}
    solid = {widgets.STAGE_BADGES[k][0] for k in ("model", "rig", "sheet")}
    assert not (flat & solid), "a glyph means both an image and a mesh"
    glyphs = [icon for icon, _ in widgets.STAGE_BADGES.values()]
    assert len(set(glyphs)) == len(glyphs), "two stages share one glyph"
    for _icon, word in widgets.STAGE_BADGES.values():
        assert word.isascii() and word.islower()


def _overflow_labels(app_ctx, imgui_ctx) -> list[str]:
    """Every menu item the library's overflow popup offers for the one card.

    The popup is opened from inside the frame rather than clicked: imgui lets
    ``open_popup`` and ``begin_popup`` happen in one frame under the same id
    stack, which is what the ellipsis button already relies on.
    """
    imgui, _renderer = imgui_ctx
    from warlock.studio import layout as layout_mod
    from warlock.studio.panes import library
    from warlock.studio.tokens import sp

    labels: list[str] = []
    real_overflow = library._overflow
    real_menu = imgui.menu_item

    def menu_spy(label, *args, **kwargs):
        labels.append(label)
        return real_menu(label, *args, **kwargs)

    def overflow_spy(ctx, job):
        imgui.open_popup("more")
        imgui.menu_item = menu_spy
        try:
            real_overflow(ctx, job)
        finally:
            imgui.menu_item = real_menu

    library._overflow = overflow_spy
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
        library._overflow = real_overflow
        imgui.menu_item = real_menu
    return labels


def test_the_library_offers_a_finished_reference_to_the_inker(app_ctx, imgui_ctx):
    """The 2D half of Open in Clay. Inker could already open a job's reference
    and reuse an open tab for it; nothing in the library said so."""
    from warlock.studio import inker_mode

    job_id = _seeded(app_ctx)
    app_ctx.svc.store.set_stage(job_id, "reference")
    app_ctx.cache.invalidate()
    app_ctx.cache.tick()
    assert inker_mode.can_edit_job(app_ctx, app_ctx.cache.get(job_id))
    assert "Open in Inker" in _overflow_labels(app_ctx, imgui_ctx)


def test_the_library_does_not_offer_the_inker_a_mesh(app_ctx, imgui_ctx):
    """Never offer an action the loader would refuse: Inker opens the pixels a
    reference *is*, and a model-stage job's ``input.png`` is the input to a
    mesh, not the asset. Open in Clay is the item that belongs on that card."""
    job_id = _seeded(app_ctx)
    app_ctx.svc.store.set_stage(job_id, "model")
    app_ctx.cache.invalidate()
    app_ctx.cache.tick()
    labels = _overflow_labels(app_ctx, imgui_ctx)
    assert "Open in Inker" not in labels
    assert "Open in Clay" in labels


def test_editing_a_big_mesh_in_clay_still_asks_first(app_ctx):
    """trellis output is 177k-299k triangles, so this confirm fires on nearly
    every reconstruction -- which is the feature, not a nuisance. Raising the
    threshold past what the pipeline produces would delete it while leaving it
    in the file."""
    from warlock.studio import clay_mode

    assert clay_mode.SLOW_TRIANGLES == 200_000
    asked: list = []
    app_ctx.confirms = SimpleNamespace(ask=asked.append)
    clay_mode._adopt_import(app_ctx, {"doc": None, "title": "big", "triangles": 250_000})
    assert asked and "250,000 faces" in asked[0].message


# --- section I: the palette, the library keyboard and the drop target --------


def test_the_command_palette_builds_and_runs_a_command(app_ctx, imgui_ctx):
    """A modal drawn over whatever mode is up, so it has to build from a mode
    that owns its whole window as well as from a generate mode."""
    from warlock.studio.panes import palette

    _seeded(app_ctx)
    for mode in ("home", "3d", "inker"):
        app_ctx.state.mode = mode
        palette.toggle(app_ctx)
        assert app_ctx.state.palette_open
        _frame(imgui_ctx, lambda: palette.draw(app_ctx))
        app_ctx.state.palette_query = "wire"
        _frame(imgui_ctx, lambda: palette.draw(app_ctx))
        # An asset row as well, which is the other half of the list.
        app_ctx.state.palette_query = "barrel"
        _frame(imgui_ctx, lambda: palette.draw(app_ctx))
        palette.close(app_ctx)
        _frame(imgui_ctx, lambda: palette.draw(app_ctx))
        assert not app_ctx.state.palette_open


def test_the_palette_rows_put_the_commands_above_the_assets(app_ctx):
    from warlock.studio.panes import palette

    _seeded(app_ctx)
    app_ctx.state.palette_query = "barrel"
    kinds = [kind for kind, _item in palette.rows(app_ctx)]
    assert "asset" in kinds
    assert kinds.index("asset") == kinds.count("command")


def test_arrow_keys_walk_the_library_and_ask_it_to_scroll(app_ctx):
    """I79. The list walked is the one on screen -- filtered and sorted -- so
    Down always goes to the card below."""
    from warlock.studio.panes import library

    first = _seeded(app_ctx)
    second = _seeded(app_ctx)
    order = [job["id"] for job in app_ctx.cache.visible(app_ctx.state.filters)]
    assert set(order) == {first, second}

    app_ctx.state.select(order[0])
    library.select_relative(app_ctx, 1)
    assert app_ctx.state.selected == order[1]
    assert app_ctx.state.library_scroll_to == order[1]
    # Clamped at both ends rather than wrapped.
    library.select_relative(app_ctx, 1)
    assert app_ctx.state.selected == order[1]
    library.select_relative(app_ctx, -1)
    library.select_relative(app_ctx, -1)
    assert app_ctx.state.selected == order[0]


def test_arrow_keys_with_nothing_selected_enter_from_the_near_end(app_ctx):
    from warlock.studio.panes import library

    _seeded(app_ctx)
    _seeded(app_ctx)
    order = [job["id"] for job in app_ctx.cache.visible(app_ctx.state.filters)]
    app_ctx.state.select(None)
    library.select_relative(app_ctx, 1)
    assert app_ctx.state.selected == order[0]
    app_ctx.state.select(None)
    library.select_relative(app_ctx, -1)
    assert app_ctx.state.selected == order[-1]


def test_the_full_window_library_builds_in_every_column_arrangement(app_ctx, imgui_ctx):
    """The UI redesign, wave 4.4. Four states, because the middle column's width is
    a function of whether the inspector is there and the grid is a function of
    which pile it is drawing."""
    from warlock.studio.panes import library_full

    # Empty: the grid reuses the library's own empty state.
    app_ctx.state.select(None)
    _frame(imgui_ctx, lambda: library_full.draw(app_ctx))

    first = _seeded(app_ctx)
    _seeded(app_ctx)
    # A selection resolves, so the inspector column appears and the grid narrows.
    app_ctx.state.select(first)
    _frame(imgui_ctx, lambda: library_full.draw(app_ctx))
    # ...and a selection that the cache cannot resolve does not.
    app_ctx.state.select("no-such-job")
    _frame(imgui_ctx, lambda: library_full.draw(app_ctx))
    # A bulk selection puts the toolbar under the grid.
    app_ctx.state.select(first)
    app_ctx.state.checked.add(first)
    _frame(imgui_ctx, lambda: library_full.draw(app_ctx))
    app_ctx.state.checked.clear()
    # The trash is a different pile with a different footer.
    app_ctx.state.filters.trash = True
    try:
        _frame(imgui_ctx, lambda: library_full.draw(app_ctx))
    finally:
        app_ctx.state.filters.trash = False


def test_the_grid_publishes_its_column_count_for_the_keyboard(app_ctx, imgui_ctx):
    """Up and Down move by a *row*, which nothing can know until the grid has
    been laid out -- so the grid says how wide it was and the keyboard reads it
    back. A pane that has never drawn answers one, not zero: zero would make
    both keys do nothing at all."""
    from warlock.studio.panes import library, library_full

    app_ctx.state.preview.pop(library.COLUMNS_SLOT, None)
    assert library.columns(app_ctx) == 1

    order = [_seeded(app_ctx) for _ in range(5)]
    app_ctx.state.select(None)
    _frame(imgui_ctx, lambda: library_full.draw(app_ctx))
    count = library.columns(app_ctx)
    assert count >= 1

    shown = [job["id"] for job in app_ctx.cache.visible(app_ctx.state.filters)]
    assert set(shown) == set(order)
    app_ctx.state.select(shown[0])
    library.select_grid(app_ctx, 1, 0)
    assert app_ctx.state.selected == shown[1]
    library.select_grid(app_ctx, 0, 1)
    # One row down, clamped at the end of the list -- the same clamp
    # ``select_relative`` applies, because it is the one doing the moving.
    assert app_ctx.state.selected == shown[min(1 + count, len(shown) - 1)]


def test_only_a_finished_reference_can_be_dragged(app_ctx):
    """A card that lifts and a slot that refuses it is worse than a card that
    does not lift -- so the predicate is stated once and both use it."""
    from warlock.studio.panes import library

    assert library.can_drag({"stage": "reference", "status": "done"})
    assert not library.can_drag({"stage": "reference", "status": "running"})
    assert not library.can_drag({"stage": "model", "status": "done"})


def test_the_3d_source_slot_builds_while_a_drag_is_in_flight(app_ctx, imgui_ctx):
    """The outline is drawn from the group's rect, which only exists after
    ``end_group`` -- a frame with a drag in flight is the one that exercises
    it."""
    from warlock.studio.panes import settings_3d

    job_id = _seeded(app_ctx)
    app_ctx.state.mode = "create"
    app_ctx.state.create_stage = "mesh"
    app_ctx.state.dragging_job = job_id
    _frame(imgui_ctx, lambda: settings_3d.draw(app_ctx))
    app_ctx.state.dragging_job = None
    _frame(imgui_ctx, lambda: settings_3d.draw(app_ctx))


def test_the_toast_stack_and_its_history_build(app_ctx, imgui_ctx):
    """Every level, the over-cap counter, an action button, and the history in
    the diagnostics popup -- all of which draw only when something has been
    raised, which is why nothing had built them."""
    from warlock.studio import widgets

    for level in ("info", "success", "warn", "error"):
        app_ctx.toast(f"a {level} notice", level)
    app_ctx.toast("with an action", "success", action="show", action_arg="nope")
    for index in range(widgets.TOAST_VISIBLE):
        app_ctx.toast(f"filler {index}")
    seen: list[tuple] = []
    _frame(
        imgui_ctx,
        lambda: widgets.toasts(app_ctx.state, (1200.0, 900.0), on_action=lambda *a: seen.append(a)),
    )
    assert len(app_ctx.state.toasts) == widgets.TOAST_VISIBLE + 5


def test_the_placeholder_builds_in_every_mode(app_ctx, imgui_ctx):
    from warlock.studio import modes
    from warlock.studio.panes import overlay

    for mode in modes.KEYS:
        app_ctx.state.mode = mode
        _frame(imgui_ctx, lambda: overlay.placeholder(app_ctx))


def test_a_card_with_no_thumbnail_builds_its_placeholder(app_ctx, imgui_ctx):
    from warlock.studio.panes import library

    job_id = _seeded(app_ctx)
    # No thumb.png was written, so every card takes the placeholder path.
    app_ctx.state.mode = "create"
    app_ctx.state.create_stage = "mesh"
    _frame(imgui_ctx, lambda: library.draw(app_ctx))
    assert "thumb.png" not in (app_ctx.cache.get(job_id).get("files") or [])


def test_the_library_builds_in_every_view_and_density(app_ctx, imgui_ctx):
    """J85/J89/J91. The trash view, the compact rows and the date headings are
    all branches nothing else builds."""
    from warlock.service import jobs as svc_jobs
    from warlock.studio.panes import library
    from warlock.studio.state import SORTS

    job_id = _seeded(app_ctx)
    app_ctx.state.mode = "create"
    app_ctx.state.create_stage = "mesh"
    for key, _label in SORTS:
        app_ctx.state.filters.sort = key
        for descending in (True, False):
            app_ctx.state.filters.descending = descending
            _frame(imgui_ctx, lambda: library.draw(app_ctx))
    app_ctx.state.filters.sort = "newest"
    app_ctx.settings.set("library_compact", True)
    _frame(imgui_ctx, lambda: library.draw(app_ctx))
    app_ctx.settings.set("library_compact", False)

    # And the trash, with something in it and a bulk selection armed.
    svc_jobs.trash_job(app_ctx.svc, job_id)
    app_ctx.cache.invalidate()
    app_ctx.cache.tick()
    app_ctx.state.filters.trash = True
    _frame(imgui_ctx, lambda: library.draw(app_ctx))
    app_ctx.state.checked.add(job_id)
    _frame(imgui_ctx, lambda: library.draw(app_ctx))
    app_ctx.state.checked.clear()
    app_ctx.state.filters.trash = False


def test_the_prune_confirm_builds_its_keep_count(app_ctx, imgui_ctx):
    """O116: a confirm with a widget in it, which nothing else exercises."""
    from warlock.studio.panes import library

    library.ask_prune(app_ctx)
    assert app_ctx.confirms.pending is not None
    _frame(imgui_ctx, lambda: app_ctx.confirms.draw())
    app_ctx.confirms.dismiss()


def test_the_panel_search_boxes_build_and_hide_themselves(app_ctx, imgui_ctx):
    """J86. The box only appears once a list is long enough to search, and the
    query is cleared when it is not -- a filter running with nothing on screen
    to say so looks like a panel that has lost its contents."""
    from warlock.studio import widgets

    seen: list[str] = []

    def build():
        seen.append(widgets.list_filter(app_ctx, "demo", 20))

    _frame(imgui_ctx, build)
    app_ctx.state.list_filters["demo"] = "chest"
    _frame(imgui_ctx, build)
    assert widgets.list_filter(app_ctx, "demo", 3) == ""
    assert "demo" not in app_ctx.state.list_filters


@pytest.mark.parametrize("palette", ["light", "pixel"])
def test_the_whole_frame_builds_under_every_palette(app_ctx, imgui_ctx, palette):
    """M105. The palette is resolved live through ``theme.__getattr__``, so a
    switch reaches every hand-drawn rect -- and every one of those call sites
    is a place a missing name would raise on the frame somebody switched.

    Dark is not in the list because every other test in this file already
    builds under it; what is worth the frames is each palette somebody has to
    *switch into*.
    """
    from warlock.studio import theme, tokens
    from warlock.studio.panes import app_settings, inspector, landing, library

    imgui, _renderer = imgui_ctx
    _seeded(app_ctx)
    try:
        tokens.set_theme(palette)
        theme.apply(imgui)
        for mode, draw in (
            ("home", lambda: landing.draw(app_ctx)),
            ("create", lambda: library.draw(app_ctx)),
            ("create", lambda: inspector.draw(app_ctx)),
            ("settings", lambda: app_settings.draw(app_ctx)),
        ):
            app_ctx.state.mode = mode
            _frame(imgui_ctx, draw)
    finally:
        tokens.set_theme("dark")
        theme.apply(imgui)


def test_the_home_resume_rows_build_with_the_keyboard_cursor_on_each(app_ctx, imgui_ctx):
    from warlock.studio import recents
    from warlock.studio.panes import landing

    app_ctx.state.mode = "home"
    _seeded(app_ctx)
    for index in range(4):
        recents.remember(app_ctx.settings, "clay", f"f{index}.wblk", when=float(index))
    for index in range(len(landing.rows(app_ctx))):
        app_ctx.state.home_index = index
        _frame(imgui_ctx, lambda: landing.draw(app_ctx))


def test_the_3d_form_builds_with_a_custom_budget(app_ctx, imgui_ctx, monkeypatch):
    """K94/K95: the disabled single-option path *and* the custom-triangles
    widget, which the shipped tier list never reaches."""
    from warlock.studio.panes import settings_3d

    app_ctx.state.mode = "create"
    app_ctx.state.create_stage = "mesh"
    _frame(imgui_ctx, lambda: settings_3d.draw(app_ctx))
    monkeypatch.setattr(settings_3d, "PROFILES", [("raw", "Raw"), ("custom", "Custom...")])
    app_ctx.state.form_3d["profile"] = "custom"
    _frame(imgui_ctx, lambda: settings_3d.draw(app_ctx))


# --- Plotter and Packwright ---------------------------------------------------


def _tileset(size: int = 32, tile: int = 16):
    from warlock.studio.tilegrid.tileset import Tileset

    pixels = np.zeros((size, size, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    pixels[0, 0] = (9, 9, 9, 255)
    return Tileset(name="terrain", pixels=pixels, tile_w=tile, tile_h=tile)


def test_plotter_builds_empty_and_with_a_map(app_ctx, imgui_ctx):
    """Both branches of every pane. The empty state is the frame that shows
    none of the mode's controls, which is exactly why it is drawn first."""
    from imgui_bundle import imgui

    from warlock.studio import plotter_mode, plotter_state
    from warlock.studio.panes import (
        plotter_bridge,
        plotter_canvas,
        plotter_layers,
        plotter_textures,
        plotter_tileset,
        plotter_tools,
    )
    from warlock.studio.plotter.tilemap import MapObject, new_uid
    from warlock.studio.tilegrid import gid
    from warlock.studio.tokens import sp

    app_ctx.state.mode = "plotter"
    state = plotter_mode.ensure(app_ctx)

    def build() -> None:
        # Three columns, as the app lays them out -- Tiled's arrangement since
        # the toolbar landed: Properties over the map file on the left, the
        # layer stack over the tileset palette on the right. Stacked in one
        # column the canvas child ends up below a sidebar, and imgui culls a
        # child pushed past the visible area -- so a row added to a pane would
        # silently stop the canvas drawing and its textures with it.
        #
        # **The tools pane is not drawn here.** It is the strip across the top
        # of the centre column now, and ``plotter_canvas.draw`` draws it: a
        # second call in a sidebar would put two bars carrying one set of imgui
        # ids in one frame, which is a collision rather than a coverage win.
        if imgui.begin_child("##plot-left", (sp(300), 0)):
            # The Properties pane is a *separate* door in ``skeletons.py``, so
            # a build that only called ``draw`` never entered the object form
            # or the multi-selection summary at all.
            plotter_layers.draw_properties(app_ctx)
            plotter_bridge.draw(app_ctx)
        imgui.end_child()
        imgui.same_line()
        if imgui.begin_child("##plot-centre", (sp(560), 0)):
            plotter_canvas.draw(app_ctx)
        imgui.end_child()
        imgui.same_line()
        if imgui.begin_child("##plot-right", (sp(300), 0)):
            plotter_layers.draw(app_ctx)
            plotter_tileset.draw(app_ctx)
        imgui.end_child()

    _frame(imgui_ctx, build)  # nothing open

    tab = plotter_mode.new_document(app_ctx, (8, 8, 16, 16))
    _frame(imgui_ctx, build)  # a map, but no tileset: the "add one first" branch

    ref = tab.doc.add_tileset(_tileset())
    layer = tab.doc.tile_layers()[0]
    cells = np.zeros((8, 8), gid.DTYPE)
    cells[0, 0] = gid.compose(ref.firstgid)
    cells[1, 1] = gid.compose(ref.firstgid + 1, flip_h=True, flip_d=True)
    tab.doc.write_region(layer.uid, 0, 0, cells)
    state.brush = np.array([[ref.firstgid]], gid.DTYPE)
    _frame(imgui_ctx, build)

    # The draw loop resolves each id to its tileset through a memo rather than
    # rescanning the list per visible cell. Asserted here because this is the
    # only place the real ``_layers`` runs: both painted ids must have landed on
    # the one tileset, and the flipped cell must resolve like the plain one --
    # the memo is keyed on the *masked* id, so it would collide otherwise.
    epoch, resolved = plotter_canvas._TILESET_MEMO[tab.uid]
    assert epoch == tab.doc.tileset_epoch
    assert resolved[ref.firstgid] == 0
    assert resolved[ref.firstgid + 1] == 0

    # Every tool, twice over: once with no terrain set on the map (the Terrain
    # tool's empty branch) and once with one and a terrain in hand (the swatch
    # grid). Neither branch executes anywhere else.
    for tool, _label, _letter in plotter_state.TOOLS:
        state.tool = tool
        _frame(imgui_ctx, build)

    # Both Shape modes: the picker only draws while Shape is the tool in hand,
    # so neither button rasterises anywhere else.
    state.tool = "shape"
    for mode, _label, _glyph in plotter_tools.SHAPES:
        state.shape_mode = mode
        _frame(imgui_ctx, build)
    state.tool = "stamp"

    # A locked layer: the padlock swaps glyph and the object form draws its
    # read-only branch, neither of which rasterises anywhere else.
    tab.doc.set_layer_props(layer.uid, locked=True)
    _frame(imgui_ctx, build)
    tab.doc.set_layer_props(layer.uid, locked=False)

    # And a marquee, which is an overlay with no other route to a frame.
    state.select = (1, 1, 4, 4)
    _frame(imgui_ctx, build)
    state.select = None

    from plotter._terrainset import terrain_tileset

    from warlock.studio.plotter import terrain as terrainlib

    ground = tab.doc.add_tileset(terrain_tileset(tile_w=16, tile_h=16))
    state.terrain = (tab.doc.tilesets.index(ground), 0)
    tab.doc.set_active_layer(layer.uid)
    tab.doc.write_region(layer.uid, *terrainlib.paint_terrain(layer.data, 4, 4, 0, ground))
    for tool, _label, _letter in plotter_state.TOOLS:
        state.tool = tool
        _frame(imgui_ctx, build)

    from warlock.studio import widgets

    # Both property editors, with a property in each so the value row and the
    # remove button rasterise rather than only the empty new-key form.
    from warlock.studio.plotter.tsx import Prop

    widgets.request_open("plotter/map-props")
    widgets.request_open("plotter/layer-props")
    tab.doc.set_map_properties({"theme": Prop("string", "cave"), "hard": Prop("bool", True)})
    tab.doc.set_layer_props(
        layer.uid, properties={"kind": Prop("string", "floor"), "cost": Prop("int", 3)}
    )
    tab.doc.set_active_layer(layer.uid)
    state.tool = "stamp"
    _frame(imgui_ctx, build)

    # A selected rect object, which is the only way the resize handles draw --
    # and the same object on a locked layer, where they must not.
    objects = tab.doc.add_object_layer("Things")
    zone = tab.doc.add_object(
        objects.uid,
        MapObject(uid=new_uid(), name="zone", kind="rect", x=8, y=8, w=48, h=32),
    )
    tab.doc.set_active_layer(objects.uid)
    state.tool = "object"
    state.select_object(zone.uid)
    _frame(imgui_ctx, build)
    tab.doc.set_layer_props(objects.uid, locked=True)
    _frame(imgui_ctx, build)
    tab.doc.set_layer_props(objects.uid, locked=False)
    tab.doc.set_active_layer(layer.uid)
    state.tool = "stamp"

    # And an isometric map, which is the only way ``_backdrop``, ``_layers``,
    # ``_grid``, ``_cursor`` and ``_visible_range`` take their diamond branch.
    iso = plotter_mode.new_document(app_ctx, (6, 6, 32, 16))
    iso_set = terrain_tileset(tile_w=32, tile_h=16)
    iso.doc.set_projection("isometric", adding=iso_set)
    iso_layer = iso.doc.tile_layers()[0]
    iso.doc.write_region(
        iso_layer.uid,
        *terrainlib.paint_terrain(iso_layer.data, 2, 2, 0, iso.doc.tilesets[-1]),
    )
    state.terrain = (len(iso.doc.tilesets) - 1, 0)
    for tool, _label, _letter in plotter_state.TOOLS:
        state.tool = tool
        _frame(imgui_ctx, build)
    state.activate(tab.uid)
    state.tool = "stamp"

    # The minimap toggled off, which is its own branch -- every frame above drew
    # it, in both projections.
    state.minimap = False
    _frame(imgui_ctx, build)
    state.minimap = True

    # An object layer with something selected: the properties form and the
    # typed-property editor are a whole branch nothing above reaches.
    objects = tab.doc.add_object_layer("Things")
    obj = tab.doc.add_object(
        objects.uid, MapObject(uid=new_uid(), name="spawn", kind="rect", w=16, h=16)
    )
    state.select_object(obj.uid)
    tab.doc.set_active_layer(objects.uid)
    _frame(imgui_ctx, build)

    # And a *multi*-selection: the "N objects selected" summary is a second
    # branch of the same pane, with its own destructive button, and nothing
    # above it reaches it -- locked as well as not, because the lock replaces
    # that button with a reason.
    other = tab.doc.add_object(
        objects.uid,
        MapObject(uid=new_uid(), name="exit", kind="rect", x=32, y=32, w=16, h=16),
    )
    state.select_objects([obj.uid, other.uid])
    _frame(imgui_ctx, build)
    tab.doc.set_layer_props(objects.uid, locked=True)
    _frame(imgui_ctx, build)
    tab.doc.set_layer_props(objects.uid, locked=False)
    state.select_object(obj.uid)

    prefix = f"{plotter_textures.PREFIX}{tab.uid}:"
    assert [k for k in app_ctx.state.preview if k.startswith(prefix)], "a texture was made"
    plotter_mode.release_all(app_ctx)
    assert not [k for k in app_ctx.state.preview if k.startswith(plotter_textures.PREFIX)]


def test_packwright_builds_empty_and_with_an_atlas(app_ctx, imgui_ctx):
    from imgui_bundle import imgui

    from warlock.studio import packwright_mode
    from warlock.studio.packwright import compose as composelib
    from warlock.studio.packwright import layout as laylib
    from warlock.studio.packwright.sources import Sprite
    from warlock.studio.panes import (
        packwright_bridge,
        packwright_items,
        packwright_preview,
        packwright_settings,
        packwright_sources,
        packwright_textures,
    )
    from warlock.studio.tokens import sp

    app_ctx.state.mode = "packwright"
    packwright_mode.ensure(app_ctx)

    def build() -> None:
        if imgui.begin_child("##pack-left", (sp(300), 0)):
            packwright_sources.draw(app_ctx)
            packwright_settings.draw(app_ctx)
        imgui.end_child()
        imgui.same_line()
        if imgui.begin_child("##pack-centre", (sp(560), 0)):
            packwright_preview.draw(app_ctx)
        imgui.end_child()
        imgui.same_line()
        if imgui.begin_child("##pack-right", (sp(300), 0)):
            packwright_items.draw(app_ctx)
            packwright_bridge.draw(app_ctx)
        imgui.end_child()

    _frame(imgui_ctx, build)  # nothing open

    tab = packwright_mode.new_document(app_ctx)
    tab.pack_dirty = False  # the pane pumps a repack; there is nothing to pack
    _frame(imgui_ctx, build)  # a document with no sources

    for index in range(4):
        block = np.zeros((6 + index, 8, 4), dtype=np.uint8)
        block[1:-1, 1:-1] = (200, 30, 30, 255)
        tab.doc.add_source(Sprite(key=f"s{index}", name=f"s{index}", pixels=block))

    def repack(mode: str) -> None:
        """What the pack task does, inline. The task runner is real here and
        its result arrives on a later frame, so the adoption is done by hand --
        ``adopt_pack`` is the same call ``on_task_done`` makes."""
        tab.doc.set_settings(mode=mode)
        result = laylib.layout(tab.doc.sprites(), tab.doc.settings)
        tab.adopt_pack(result, composelib.compose(tab.doc.sprites(), result))
        tab.pack_dirty = False

    for mode in ("grid", "maxrects"):
        repack(mode)
        _frame(imgui_ctx, build)

    packwright_mode.ensure(app_ctx).selected = tab.doc.sources[0].uid
    _frame(imgui_ctx, build)

    # And the failure branch, which must say why rather than show an empty list
    # -- an empty list reads as success, which is the worst outcome here.
    tab.pack_error = "these do not fit"
    _frame(imgui_ctx, build)

    prefix = f"{packwright_textures.PREFIX}{tab.uid}:"
    assert [k for k in app_ctx.state.preview if k.startswith(prefix)], "a texture was made"
    packwright_mode.release_all(app_ctx)
    assert not [k for k in app_ctx.state.preview if k.startswith(packwright_textures.PREFIX)]


@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_a_segmented_control_takes_its_compact_labelling_rather_than_clipping(imgui_ctx, scale):
    """A clipped segment is an unreachable choice.

    ``segmented_control`` lays its segments out from ``origin`` with no wrap and
    no scroll, so a control wider than the region it is in does not compress --
    the last segments are simply drawn off the edge. That is why it has a
    ``compact`` labelling and takes it all-or-nothing.

    This used to be asked of the *mode switch*, which was thirteen segments and
    the app's whole navigation; the rail replaced it in the UI redesign, wave 3 and
    the rail's own fit is pinned above. What survives is the widget's rule,
    asked of a control wide enough to need it -- and it survives because the
    control does: wave 4 gives it the Settings pane's categories.
    """
    imgui, renderer = imgui_ctx
    from warlock.studio import icons, tokens
    from warlock.studio import widgets as widgets_mod

    options = [
        ("appearance", f"{icons.PALETTE} Appearance"),
        ("models", f"{icons.BOX} Models"),
        ("storage", f"{icons.FOLDER_OPEN} Storage"),
        ("advanced", f"{icons.SETTINGS} Advanced"),
    ]
    compact = [
        ("appearance", icons.PALETTE),
        ("models", icons.BOX),
        ("storage", icons.FOLDER_OPEN),
        ("advanced", icons.SETTINGS),
    ]
    measured: list[tuple[float, float]] = []

    old_scale = tokens.SCALE
    tokens.set_scale(scale)
    try:
        imgui.new_frame()
        # Narrow enough that the labelled form cannot fit at any scale, which
        # is the branch being asserted.
        imgui.set_next_window_size((240, 200))
        imgui.begin("##narrow")
        available = imgui.get_content_region_avail().x
        widgets_mod.segmented_control(
            "seg-fit", options, "appearance", compact=compact, max_width=available
        )
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        measured.append((hi.x - lo.x, available))
        imgui.end()
        imgui.render()
        renderer.render(imgui.get_draw_data())
    finally:
        tokens.set_scale(old_scale)

    drawn, available = measured[-1]
    assert drawn <= available, (
        f"the control is {drawn:.0f}px wide in a {available:.0f}px region at UI "
        f"scale {scale}; a clipped segment is an unreachable choice"
    )


def test_the_create_pane_builds_at_every_stage(app_ctx, imgui_ctx):
    """The merged mode, through the real dispatch: the rail plus whichever
    panel the stage names. Every stage in one test because the whole claim of
    wave 5 is that they are one pane with a breadcrumb -- and through
    ``main._stage_pane`` rather than a copy of it, so a stage that the rail
    offers and the dispatch has no branch for fails here."""
    from warlock.studio import create_stages, main

    _seeded(app_ctx)
    app_ctx.rigging_available = True
    app_ctx.state.mode = create_stages.MODE
    app = SimpleNamespace(app_ctx=app_ctx)
    for stage in create_stages.STAGES:
        app_ctx.state.create_stage = stage

        def build():
            main.App._stage_rail(app, app_ctx)
            main._stage_pane(app_ctx)

        _frame(imgui_ctx, build)
        assert app_ctx.state.create_stage == stage, "the rail moved on its own"


def test_the_inspector_builds_at_every_stage(app_ctx, imgui_ctx):
    """The other half of the merged mode. The inspector carries no tabs in
    Create -- the rail is the tab bar -- so a stage with no ``_STAGE_SECTIONS``
    entry silently shows a header and nothing else, and only a build of all
    five says so."""
    from warlock.studio import create_stages
    from warlock.studio.panes import inspector

    _seeded(app_ctx)
    app_ctx.rigging_available = True
    app_ctx.state.mode = create_stages.MODE
    for stage in create_stages.STAGES:
        assert stage in inspector._STAGE_SECTIONS, stage
        app_ctx.state.create_stage = stage
        _frame(imgui_ctx, lambda: inspector.draw(app_ctx))


def test_the_create_pane_builds_at_every_stage_with_nothing_selected(app_ctx, imgui_ctx):
    """The empty case, which is the one a first run sees: no selection, and
    four stages of which three are about an asset that does not exist."""
    from warlock.studio import create_stages, main

    app_ctx.state.select(None)
    app_ctx.state.mode = create_stages.MODE
    app = SimpleNamespace(app_ctx=app_ctx)
    for stage in create_stages.STAGES:
        app_ctx.state.create_stage = stage
        _frame(imgui_ctx, lambda: (main.App._stage_rail(app, app_ctx), main._stage_pane(app_ctx)))


def _stage_items(blocked=()):
    from warlock.studio import create_stages

    return [
        (
            stage,
            create_stages.LABELS[stage],
            create_stages.ICONS[stage],
            "a tile has no subject to reconstruct" if stage in blocked else None,
        )
        for stage in create_stages.STAGES
    ]


def _click(imgui_ctx, build, at):
    """Press and release over ``at``, one frame each. -> what ``build`` returned.

    imgui resolves a click on release over the item that was pressed, so a
    button needs two frames to be clicked and a single-frame test would assert
    that nothing is ever clickable.
    """
    imgui, renderer = imgui_ctx
    io = imgui.get_io()
    out = None
    for down in (True, False):
        io.add_mouse_pos_event(at[0], at[1])
        io.add_mouse_button_event(0, down)
        imgui.new_frame()
        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((1200, 900))
        # Undecorated, and under its own name. The context is session-scoped,
        # so a shared window that a synthetic double-click happened to collapse
        # would stay collapsed for every later test in the file -- and a
        # collapsed window submits no items at all, which reads as "nothing is
        # ever clickable" rather than as a broken fixture.
        imgui.begin("##click-host", None, imgui.WindowFlags_.no_decoration.value)
        out = build()
        imgui.end()
        imgui.render()
        renderer.render(imgui.get_draw_data())
    return out


@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_the_stage_rail_compacts_rather_than_clipping_a_stage(imgui_ctx, scale):
    """``segmented_control``'s rule, asked of the rail that borrowed its idiom.

    The column it sits above is the *narrowest* sidebar the app offers, at the
    scales the app is looked at, because that is the case the fit has to hold
    in -- and a stage drawn off the edge of the column is a step of the
    pipeline with no way in.
    """
    imgui, renderer = imgui_ctx
    from warlock.studio import layout as layout_mod
    from warlock.studio import tokens
    from warlock.studio import widgets as widgets_mod

    old_scale = tokens.SCALE
    tokens.set_scale(scale)
    try:
        imgui.new_frame()
        imgui.set_next_window_size((tokens.sp(layout_mod.SIDEBAR_WIDTHS["narrow"]), 200))
        imgui.begin("##narrow")
        available = imgui.get_content_region_avail().x
        widgets_mod.stage_rail(
            "rail-fit", _stage_items(), "reference", done="mesh", max_width=available
        )
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        drawn = hi.x - lo.x
        imgui.end()
        imgui.render()
        renderer.render(imgui.get_draw_data())
    finally:
        tokens.set_scale(old_scale)

    assert drawn <= available, (
        f"the rail is {drawn:.0f}px wide in a {available:.0f}px column at UI "
        f"scale {scale}; a clipped stage is an unreachable stage"
    )


def _rail_probe(imgui_ctx, rail_id, items, current, done):
    """A builder that draws the rail and records where its second segment is,
    so a click is aimed at measured geometry rather than at a guess -- the
    padding, the font and therefore the segment widths are all scale-dependent.
    """
    imgui, _renderer = imgui_ctx
    from warlock.studio import widgets as widgets_mod

    seen: dict[str, float] = {}

    def build():
        out = widgets_mod.stage_rail(rail_id, items, current, done=done)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        # Just inside the right edge, so the point is in the *last* segment
        # whatever the widths are -- they follow the labels, and the labels
        # change as stages are added.
        seen["x"] = hi.x - 2.0
        seen["y"] = (lo.y + hi.y) * 0.5
        return out

    return build, seen


def test_a_stage_rail_segment_can_be_picked(imgui_ctx):
    from warlock.studio import create_stages

    build, seen = _rail_probe(imgui_ctx, "rail-pick", _stage_items(), "reference", "mesh")
    _click(imgui_ctx, build, (-100.0, -100.0))
    assert seen, "the rail never drew"
    assert _click(imgui_ctx, build, (seen["x"], seen["y"])) == create_stages.STAGES[-1]


def test_a_blocked_stage_cannot_be_picked(imgui_ctx):
    """The reason is a tooltip, not a refusal after the fact. A blocked
    segment is still an item -- it has to be hoverable to carry its sentence --
    so the click is *dropped* rather than the button not being drawn."""
    from warlock.studio import create_stages

    last = create_stages.STAGES[-1]
    build, seen = _rail_probe(
        imgui_ctx, "rail-block", _stage_items(blocked={last}), "reference", "reference"
    )
    _click(imgui_ctx, build, (-100.0, -100.0))
    assert seen, "the rail never drew"
    assert _click(imgui_ctx, build, (seen["x"], seen["y"])) == "reference"


# --- UX.md Phase 5: the GPU tier --------------------------------------------
#
# The rest of the phase's rules are assertable headlessly and live in
# ``tests/test_ux_phase45.py``. These three are the half that needs a context,
# a renderer and a framebuffer -- that a sprite really uploads, that a card
# really draws through it, and that the backdrop really captures the frame --
# which is exactly what the fixtures in this file are.


def test_a_shadow_sprite_uploads_and_a_card_draws_through_it(imgui_ctx):
    """The textured path end to end. Everything below the upload is pure and
    pinned elsewhere; what this adds is that the texture builds on a real
    context and that ``widgets.shadow`` takes the sliced branch rather than
    quietly falling back to the drawn bands for the life of the session."""
    from warlock.studio import shadows, tokens, widgets

    shadows.release_all()
    sprite = shadows.sprite(tokens.RADIUS_L, shadows.SPREAD)
    assert sprite is not None, "no sprite on a live renderer means the fallback forever"
    assert sprite.texture.size == (sprite.size, sprite.size)
    assert not shadows._BROKEN

    drawn = []

    def build():
        imgui_mod, _renderer = imgui_ctx
        with widgets.card("smoke/shadowed", (320.0, 120.0)):
            imgui_mod.text("card")
        drawn.append(
            widgets._sliced_shadow(
                (10.0, 10.0),
                (330.0, 130.0),
                tokens.RADIUS_L,
                1.0,
                1.0,
                imgui_mod.get_window_draw_list(),
            )
        )

    _frame(imgui_ctx, build)
    assert drawn == [True]
    shadows.release_all()


def test_a_squircle_sprite_uploads_and_is_refused_below_its_own_threshold(imgui_ctx):
    """UX.md's own rule about where the shape stops being visible, on the side
    of it that needs a renderer to answer."""
    from warlock.studio import surfaces, tokens

    surfaces.release_all()
    sprite = surfaces.sprite(tokens.RADIUS_L)
    assert sprite is not None
    assert sprite.size == 2 * sprite.corner + 1
    assert surfaces.sprite(surfaces.MIN_RADIUS - 1.0) is None
    surfaces.release_all()


def test_the_backdrop_captures_the_frame_and_skips_the_frames_it_was_used_on(imgui_ctx, gl):
    """The one rule the whole design rests on: a floating surface must never
    sample a blur of itself, which is enforced by *not capturing* on a frame
    something asked for the backdrop."""
    from warlock.studio import vibrancy

    vibrancy.release_all()
    try:
        vibrancy.capture(gl, (1600.0, 950.0))
        assert vibrancy._PIPELINE is not None
        assert vibrancy._PIPELINE.size == vibrancy._target_size((1600.0, 950.0))
        # A blur runs on demand and hands back something imgui can draw.
        assert vibrancy.backdrop() is not None
        # Having been used, the next capture is skipped -- and the throttle is
        # not what did it, so the clock is moved past the interval first.
        vibrancy._LAST_CAPTURE = 0.0
        before = vibrancy._LAST_CAPTURE
        vibrancy.capture(gl, (1600.0, 950.0))
        assert before == vibrancy._LAST_CAPTURE
        # Unused now, so the frame after it captures again.
        vibrancy.capture(gl, (1600.0, 950.0))
        assert before < vibrancy._LAST_CAPTURE
        assert not vibrancy._BROKEN
    finally:
        vibrancy.release_all()
        gl.screen.use()


def test_a_staged_tag_is_the_only_thing_that_repaints_a_toggle(app_ctx, imgui_ctx):
    """The selected-toggle branch, which is the whole signal the control gives.

    Everything else in the smoke suite asserts that a pane *builds*, and this
    branch builds identically whether or not it paints -- so a push that went to
    the wrong colour slot, or never happened, would be invisible here and
    invisible in the GL pass. The staged fill was in fact verified by eye
    through ``scripts/screenshot_modes.py --review``; this is what keeps it.

    Asserted on the *calls* rather than on pixels: pushed only when something is
    staged, and every push popped. A framebuffer colour comparison would be more
    direct and far more brittle -- it would fail on any palette change, which is
    not what is being protected.
    """
    imgui, _renderer = imgui_ctx
    from warlock.studio import widgets

    pushed: list = []
    real_push = imgui.push_style_color
    real_pop = imgui.pop_style_color

    def spy_push(index, colour):
        pushed.append(index)
        return real_push(index, colour)

    popped = [0]

    def spy_pop(count=1):
        popped[0] += count
        return real_pop(count)

    imgui.push_style_color = spy_push
    imgui.pop_style_color = spy_pop
    try:
        _frame(imgui_ctx, lambda: widgets.tag_toggles("t", [], True))
        assert pushed == [], "nothing staged must push nothing"

        pushed.clear()
        popped[0] = 0
        _frame(imgui_ctx, lambda: widgets.tag_toggles("t", ["holes"], True))
    finally:
        imgui.push_style_color = real_push
        imgui.pop_style_color = real_pop

    assert pushed, "a staged tag must push a colour"
    assert imgui.Col_.button.value in pushed
    # The hover slot too, or a staged tag drops back to the default fill under
    # the pointer -- the one moment the user is deciding whether to unstage it.
    assert imgui.Col_.button_hovered.value in pushed
    assert popped[0] == len(pushed), "every push is popped"


# --- the "install what this needs" gate --------------------------------------


def test_the_model_gate_is_silent_on_a_host_that_has_everything(app_ctx, imgui_ctx):
    """An empty ``model_rows`` says nothing rather than locking both features.

    The direction that matters: a snapshot that has not arrived yet must not
    read as "everything is missing" and grey out a working install.
    """
    from warlock.service import sprites as svc_sprites
    from warlock.studio.panes import model_gate

    app_ctx.model_rows = []
    assert model_gate.missing(app_ctx, svc_sprites.SPRITE_ROWS) == []
    locked: list[bool] = []
    _frame(imgui_ctx, lambda: locked.append(model_gate.draw(app_ctx, svc_sprites.SPRITE_ROWS)))
    assert locked == [False]


def test_the_model_gate_names_only_the_absent_rows(app_ctx, imgui_ctx):
    from warlock.studio.panes import model_gate

    app_ctx.model_rows = _model_rows()
    rows = model_gate.missing(app_ctx, ("base:turbo", "lora:pixelxl", "base:nope"))
    assert [r["row_key"] for r in rows] == ["lora:pixelxl"]
    locked: list[bool] = []
    _frame(
        imgui_ctx, lambda: locked.append(model_gate.draw(app_ctx, ("base:turbo", "lora:pixelxl")))
    )
    assert locked == [True]


def test_requesting_an_install_ticks_the_rows_and_opens_settings(app_ctx):
    """The click body, without a frame. ``set_mode`` rather than a direct
    ``state.mode`` write -- ``tests/test_mode_writes.py`` scans for the other."""
    from warlock.studio.panes import model_gate

    app_ctx.state.mode = "create"
    app_ctx.state.create_stage = "reference"
    app_ctx.model_picks.add("metric:dinov2")
    model_gate.request_install(app_ctx, ("lora:pixelxl", "control:canny"))
    assert app_ctx.state.mode == "settings"
    assert app_ctx.model_picks == {"metric:dinov2", "lora:pixelxl", "control:canny"}
    # set_mode's bookkeeping, which a direct write skips.
    assert app_ctx.state.previous_mode == "create"


def test_the_sprite_form_locks_its_submit_while_weights_are_missing(app_ctx, imgui_ctx):
    """The pane half, through the real ``_submit``: with the pixel LoRA absent
    the gate draws and the button is disabled."""
    from warlock.studio.panes import sprite_panel

    app_ctx.model_rows = _model_rows()
    job_id = _seeded(app_ctx)
    form = {
        "job_id": job_id,
        "sheet_type": "turnaround",
        "logical_size": 64,
        "colors": 32,
        "seed_a": 1,
        "seed_b": 2,
    }
    _frame(imgui_ctx, lambda: sprite_panel._submit(app_ctx, job_id, form))
    # Nothing was submitted: a disabled button cannot be pressed, and the frame
    # does not click anything anyway. What is asserted is that it *builds*.
    assert not app_ctx.busy(f"sprite:{job_id}")


def test_the_sprite_button_and_note_state_what_the_door_will_actually_do(
    app_ctx, imgui_ctx
):
    """The two shapes the old literals were wrong about, drawn for real.

    A turnaround at 64px is the pair the panel always claimed; an eight-direction
    action at 32px is **one** draft of **eight** generations. Both are drawn
    through ``_submit`` as well as asserted as strings, because a note nothing
    draws is a note that can go stale unnoticed.
    """
    from warlock.service import sprites as svc_sprites
    from warlock.studio.panes import sprite_panel

    app_ctx.model_rows = []
    job_id = _seeded(app_ctx)
    offered = [t["key"] for t in svc_sprites.sprite_options()["sheet_types"]]
    # The precondition, asserted: the planned kinds are discovered from the guide
    # templates on disk, so without this the action half of the test could pass
    # by testing a menu that has only turnarounds in it.
    assert "attack8" in offered, f"attack8 is not offered; the menu is {offered}"

    pair = svc_sprites.sprite_cost("turnaround", 64)
    single = svc_sprites.sprite_cost("attack8", 32)

    assert sprite_panel.submit_label(pair) == "Generate 2 drafts"
    assert sprite_panel.submit_label(single) == "Generate 1 draft"
    assert sprite_panel.submit_label(pair) != sprite_panel.submit_label(single)

    pair_note = sprite_panel.cost_text(pair)
    single_note = sprite_panel.cost_text(single)
    assert pair_note != single_note
    assert "2 full image generations" in pair_note
    assert "about 40 seconds" in pair_note
    assert "8 full image generations" in single_note
    assert "one per direction" in single_note
    assert "about 3 minutes" in single_note
    # The sentence the button contradicted: one draft must never say "two".
    assert "two" not in single_note.lower()

    for sheet_type, size in (("turnaround", 64), ("attack8", 32), ("attack8", 64)):
        form = {
            "job_id": job_id,
            "sheet_type": sheet_type,
            "logical_size": size,
            "colors": 32,
            "seed_a": 1,
            "seed_b": 2,
        }
        _frame(imgui_ctx, lambda f=form: sprite_panel._submit(app_ctx, job_id, f))
    assert not app_ctx.busy(f"sprite:{job_id}")

    # The third shape above: a size this action cannot be drawn at says so and
    # promises no number at all.
    plan = svc_sprites.sprite_cost("attack8", 64)
    assert not plan["drawable"]
    assert sprite_panel.cost_text(plan) == plan["refusal"]
    assert "1536x1024 band" in plan["refusal"]
    assert sprite_panel.submit_label(plan) == "Generate"


def test_the_settings_pane_draws_a_fit_badge_and_a_recommendation(app_ctx, imgui_ctx):
    """The two W4 surfaces, in the pane that owns them. What is asserted is
    that the pane builds with a ``vram`` verdict on one row and none on the
    others -- the absence path is the one that would crash a naive read."""
    from warlock.studio.panes import app_settings

    rows = _model_rows()
    rows[0]["vram"] = "no"
    app_ctx.model_rows = rows
    app_ctx.recommended_base_label = "SDXL 1.0 (full CFG, structural control)"
    _frame(imgui_ctx, lambda: app_settings.draw(app_ctx))
    # And with no plan at all: no key on any row, no recommendation line.
    for row in rows:
        row.pop("vram", None)
    app_ctx.recommended_base_label = ""
    _frame(imgui_ctx, lambda: app_settings.draw(app_ctx))


def test_starting_a_removal_submits_under_the_remove_prefix(app_ctx):
    """The click body without a frame. The prefix is load-bearing: the pane's
    busy check and the App's task-done handler both switch on it."""
    from warlock.studio import app_ctx as app_ctx_mod
    from warlock.studio.panes import app_settings

    app_settings._start_removal(app_ctx, "lora:pixelxl")
    key = app_ctx_mod.remove_key("lora:pixelxl")
    assert key == "remove:lora:pixelxl"
    assert app_ctx.tasks.is_busy(key) or app_ctx.tasks.collect()


def test_a_present_row_with_nothing_to_free_draws_no_trash_button(app_ctx, imgui_ctx):
    """A recipe whose every file is shared has nothing to offer, and a button
    that refused on click would be worse than no button."""
    from warlock.studio.panes import app_settings

    rows = _model_rows()
    rows[0]["removable"] = False
    app_ctx.model_rows = rows
    _frame(imgui_ctx, lambda: app_settings.draw(app_ctx))
    # And with it removable again, plus a confirm raised by hand: the dialog
    # body is the other half of the click and must build too.
    rows[0]["removable"] = True
    app_settings._confirm_removal(app_ctx, rows[0])
    assert app_ctx.confirms.pending is not None


# --- the UI redesign, wave 4.2: the rows rewritten onto ``toolbar`` ---------------
#
# The rows wave 4.2 rewrote onto ``toolbar``, measured rather than eyeballed.
#
# Two questions, asked of every rewritten row.
#
# **Does it fit?** ``toolbar.plan`` is pure arithmetic over measured widths, so
# the honest test is to measure the row's real items in a real font at each scale
# the screenshot pass runs at, and then check the planned spread against a range
# of pane widths. A row that fits is one whose drawn items plus the overflow
# button are inside the region -- and the *only* legitimate way to exceed it is a
# row whose pinned items alone are wider than the pane, which the module docstring
# says out loud and which is asserted here as an exemption rather than assumed.
#
# **Is every key wired?** The rows are data now, and the failure that invites is
# a key in the item list that no branch of the dispatcher matches -- a button that
# draws, highlights, clicks and does nothing at all. So every key each row
# publishes is checked against the source of the function that answers it.
#
# The item lists are rebuilt here rather than imported from the panes. Extracting
# them into a shared table would make this test assert against a function written
# for it; rebuilding them means a pane that renames a key without renaming it here
# fails the second test, which is the point.

SCALES = (1.0, 1.5, 1.75)
#: The pane widths a row is asked about, in physical pixels. 320 is narrower
#: than any real pane and is there to reach the all-pinned exemption; 1600 is a
#: maximised centre column.
WIDTHS = (320.0, 480.0, 720.0, 1000.0, 1600.0)


def _transform_items():
    return [
        toolbar.Item("fliph", "Flip H", icons.FLIP_HORIZONTAL, priority=1),
        toolbar.Item("flipv", "Flip V", priority=1),
        toolbar.Item("ccw", "-90", priority=1),
        toolbar.Item("cw", "+90", icons.ROTATE_CW, priority=1),
        toolbar.Item("apply", "Apply", icons.CHECK, primary=True, pinned=True),
        toolbar.Item("cancel", "Cancel", icons.X, pinned=True),
    ]


def _transport_items():
    from warlock.studio.panes import inker_timeline

    items = [
        toolbar.Item(key, label, tooltip=tip, pinned=True)
        for key, label, tip in inker_timeline._STEPS
    ]
    items.append(toolbar.Item("play", "Play", icons.PLAY, pinned=True))
    items += [
        toolbar.Item(key, label, tooltip=tip, pinned=True)
        for key, label, tip in inker_timeline._STEPS_AFTER
    ]
    items += [
        toolbar.Item("add", "Frame", icons.PLUS, priority=1),
        toolbar.Item("copy", "Copy", icons.COPY, priority=1),
        toolbar.Item("link", "Link", priority=1),
        toolbar.Item("remove", "Delete frame", icons.TRASH, danger=True, pinned=True, priority=1),
    ]
    return items


def _export_items():
    return [
        toolbar.Item("sheet", "Export sheet", icons.GRID),
        toolbar.Item("gif", "Export GIF", icons.FILM),
        toolbar.Item("pngs", "Export PNGs", icons.IMAGE),
    ]


def _bulk_items():
    # The workshop shape, which is the wider of the two -- the trash's row is
    # Clear/Restore/Delete permanently and is a subset of this one's demands.
    return [
        toolbar.Item("clear", "Clear", icons.X, pinned=True),
        toolbar.Item("retry", "Try again (12)", icons.REFRESH),
        toolbar.Item("zip", "Export zip...", icons.DOWNLOAD, priority=1),
        toolbar.Item("folder", "Save to project", icons.SAVE, priority=1),
        toolbar.Item("delete", "Delete", icons.TRASH, danger=True, pinned=True),
    ]


def _rows():
    """``(label, items, dispatcher)`` for every row wave 4.2 rewrote."""
    from warlock.studio.panes import inker_canvas, inker_timeline, library

    return [
        ("inker-transform", _transform_items(), inker_canvas._transform_action),
        ("inker-transport", _transport_items(), inker_timeline._frame_action),
        ("inker-timeline-out", _export_items(), inker_timeline._export_action),
        ("library-bulk", _bulk_items(), library._bulk_action),
    ]


def _spread(items, tiers, full, icon, overflow_w, gap):
    shown = [i for i in range(len(items)) if tiers[i] != toolbar.MENU]
    total = sum(full[i] if tiers[i] == toolbar.FULL else icon[i] for i in shown)
    if shown:
        total += gap * (len(shown) - 1)
    if len(shown) != len(items):
        total += overflow_w + (gap if shown else 0.0)
    return total


def _assert_fits(imgui, items, label, scale):
    gap = imgui.get_style().item_spacing.x
    overflow_w = imgui.get_frame_height()
    full, icon = toolbar._measure(items)
    pinned = [i for i, item in enumerate(items) if item.pinned]
    # The narrowest the row can ever be: every pinned item as a glyph, plus the
    # overflow button that holds everything else -- which is the width a row
    # forgets to budget for, and the reason ``plan`` takes it as an argument.
    floor = sum(icon[i] for i in pinned) + gap * max(len(pinned) - 1, 0)
    if len(pinned) != len(items):
        floor += overflow_w + (gap if pinned else 0.0)
    for avail in WIDTHS:
        tiers = toolbar.plan(
            full,
            icon,
            [item.priority for item in items],
            [item.pinned for item in items],
            avail,
            overflow_w,
            gap=gap,
        )
        got = _spread(items, tiers, full, icon, overflow_w, gap)
        if got > avail:
            # The one documented way out: everything left is pinned, so there
            # is nothing further to give and drawing the truth beats pretending.
            assert floor > avail, (
                f"{label} at scale {scale} overflows {avail:.0f} px by "
                f"{got - avail:.0f} px with room still to give"
            )


@pytest.mark.parametrize("scale", SCALES)
def test_every_rewritten_row_fits_or_is_all_pinned(imgui_ctx, scale):
    from warlock.studio import theme, tokens

    imgui, renderer = imgui_ctx
    old = tokens.SCALE
    tokens.set_scale(scale)
    theme.apply(imgui)
    try:
        imgui.new_frame()
        imgui.set_next_window_size((1200, 900))
        imgui.begin("##host")
        for label, items, _dispatch in _rows():
            _assert_fits(imgui, items, label, scale)
        imgui.end()
        imgui.render()
        renderer.render(imgui.get_draw_data())
    finally:
        tokens.set_scale(old)
        theme.apply(imgui)


def test_every_key_a_row_publishes_is_one_its_dispatcher_answers():
    """A key with no branch is a button that draws, clicks and does nothing."""
    for label, items, dispatch in _rows():
        source = inspect.getsource(dispatch)
        for item in items:
            assert f'"{item.key}"' in source, f"{label}: nothing answers {item.key!r}"


def test_every_symmetry_button_actually_moves_the_setting():
    """The context bar's symmetry group -- what the canvas's view row became.

    Not one of ``_rows()``: it is a ``trailing`` block rather than a list of
    ``toolbar.Item``, because ``toolbar`` draws items before fields and as
    items these five landed in the middle of the tool's own settings. So it is
    checked by *pressing* it rather than by scanning a dispatcher's source for
    each key, which is the stronger check anyway.
    """
    from warlock.studio.inker import brush
    from warlock.studio.panes import inker_context

    # ``_symmetry_hit`` persists, and ``inker_mode.persist`` reads the whole
    # session off the context -- so the context here carries no Inker state at
    # all, which is its documented early return.
    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=None),
        settings=SimpleNamespace(get=lambda *a: None, set=lambda *a: None),
    )
    for axis, _label, _tip in inker_context.SYMMETRY_TOGGLES:
        state = SimpleNamespace(
            symmetry="none", symmetry_axis=None, radial_count=6, swatches=[]
        )
        inker_context._symmetry_hit(ctx, state, f"sym/{axis}")
        assert brush.axes_of(state.symmetry) == (axis,), axis
        # And again is off: these are toggles, not a radio group.
        inker_context._symmetry_hit(ctx, state, f"sym/{axis}")
        assert brush.axes_of(state.symmetry) == ()

    state = SimpleNamespace(
        symmetry="x+y+diag", symmetry_axis=(3.0, 4.0), radial_count=12, swatches=[]
    )
    inker_context._symmetry_hit(ctx, state, f"sym/{inker_context.SYMMETRY_RESET}")
    assert brush.axes_of(state.symmetry) == ()
    assert state.symmetry_axis is None
    assert state.radial_count == brush.DEFAULT_RADIAL

    # A key this dispatcher does not own is left alone rather than swallowed:
    # it shares a toolbar with the tool's own panels and dynamics buttons.
    state = SimpleNamespace(symmetry="x", symmetry_axis=None, radial_count=6, swatches=[])
    inker_context._symmetry_hit(ctx, state, "panels")
    assert state.symmetry == "x"


# --- no two items may claim one imgui id --------------------------------------
#
# imgui addresses every item by an id hashed from its label and the id stack, and
# routes hover, focus and activation by that id alone. Two items sharing one is
# not a cosmetic clash: the second never reliably acts, because the first
# already answers for the id. Plotter's Resize section shipped that way -- a
# ``header("Resize")`` and a ``primary_button("Resize")`` in one window -- and
# the button did nothing, which read as "the map cannot be resized".
#
# The probe is the same question imgui asks itself: resolve each label through
# ``get_id`` at the moment it is submitted, so the id stack is accounted for,
# and look for one id claimed twice in a frame. imgui's own warning is a popup
# in a debug build and invisible in a release one, so this is the only place the
# conflict is an error rather than a red rectangle nobody screenshots.

_ID_WIDGETS = (
    "button",
    "small_button",
    "invisible_button",
    "arrow_button",
    "checkbox",
    "radio_button",
    "slider_int",
    "slider_float",
    "drag_int",
    "drag_float",
    "input_text",
    # widgets.input_text calls this one whenever a field has a hint, which
    # every search box in the app does -- so a harness without it is blind to
    # exactly the control that filters a list.
    "input_text_with_hint",
    "input_int",
    "input_float",
    "combo",
    "begin_combo",
    "selectable",
    "collapsing_header",
    "tree_node",
    "color_edit3",
    "color_edit4",
    "image_button",
    "menu_item",
    "begin_child",
    # Tabs are a *selection* control here, not decoration: Plotter's tileset
    # picker is a tab strip, so a harness blind to them cannot see which tileset
    # the pane offers -- which is how the picker assertion below came to pass by
    # matching a combo label after the combo was gone.
    "begin_tab_bar",
    "begin_tab_item",
)


def _claimed_ids(imgui, build, title):
    """Draw ``build`` in its own window; return {id: [(call, label), ...]}.

    Its *own* window, never the shared ``##host``: an id stack is seeded per
    window, so two panes that are separate windows in the app cannot collide and
    a probe that stacked them would report conflicts the user can never meet.
    """
    seen: list[tuple[int, str, str]] = []
    originals = {name: getattr(imgui, name) for name in _ID_WIDGETS if hasattr(imgui, name)}

    def wrap(fn):
        def recorded(label, *args, **kwargs):
            if isinstance(label, str):
                seen.append((imgui.get_id(label), fn.__name__, label))
            return fn(label, *args, **kwargs)

        return recorded

    for name, fn in originals.items():
        setattr(imgui, name, wrap(fn))
    try:
        imgui.new_frame()
        imgui.set_next_window_size((420, 900))
        imgui.begin(title)
        build()
        imgui.end()
        imgui.render()
    finally:
        for name, fn in originals.items():
            setattr(imgui, name, fn)
    claims: dict[int, list[tuple[str, str]]] = {}
    for ident, call, label in seen:
        claims.setdefault(ident, []).append((call, label))
    return {k: v for k, v in claims.items() if len(v) > 1}


def test_the_new_map_dialog_opens_once_per_request(app_ctx, imgui_ctx):
    """``setup_pending`` is a request, not a state. Left set, the dialog would
    re-open itself every frame and could never be cancelled -- and the five
    doors that raise it are spread across four windows, so the only place that
    can honour it is the one that draws it."""
    from warlock.studio import plotter_mode
    from warlock.studio.panes import plotter_canvas

    state = plotter_mode.ensure(app_ctx)
    plotter_mode.ask_new_document(app_ctx)
    assert state.setup_pending is True

    _frame(imgui_ctx, lambda: plotter_canvas.setup_popup(app_ctx, state))
    assert state.setup_pending is False
    # And a second frame with nothing pending draws without reopening it.
    _frame(imgui_ctx, lambda: plotter_canvas.setup_popup(app_ctx, state))
    assert state.setup_pending is False


@pytest.mark.parametrize(
    "pane",
    ["tools", "tileset", "layers", "bridge", "setup"],
)
def test_no_two_plotter_items_claim_one_imgui_id(app_ctx, imgui_ctx, pane):
    from warlock.studio import plotter_mode
    from warlock.studio.panes import (
        plotter_bridge,
        plotter_canvas,
        plotter_layers,
        plotter_tileset,
        plotter_tools,
    )

    imgui, _renderer = imgui_ctx
    tab = plotter_mode.new_document(app_ctx, (8, 8, 16, 16))
    # With a tileset attached, so the picker, the terrain rows and the "back
    # from Inker" rows are all submitted rather than skipped by an empty branch.
    tab.doc.add_tileset(_tileset())
    state = plotter_mode.ensure(app_ctx)

    def setup(_ctx):
        # The New map dialog, open: it is a form of buttons and fields inside a
        # window that already has a "Create" and a "Cancel", which is exactly
        # the shape the Resize collision had.
        state.setup_pending = True
        plotter_canvas.setup_popup(app_ctx, state)

    draw = {
        "tools": plotter_tools.draw,
        "tileset": plotter_tileset.draw,
        "layers": plotter_layers.draw,
        "bridge": plotter_bridge.draw,
        "setup": setup,
    }[pane]
    title = f"##idprobe-{pane}"
    _claimed_ids(imgui, lambda: draw(app_ctx), title)  # settle headers and combos
    conflicts = _claimed_ids(imgui, lambda: draw(app_ctx), title)
    detail = "; ".join(
        f"{ident}: " + " == ".join(f"{call}({label!r})" for call, label in items)
        for ident, items in conflicts.items()
    )
    assert not conflicts, f"{pane}: {detail}"


def _drawn_labels(imgui, build, title):
    """Every label the pane submits, in draw order.

    The same wrapping trick ``_claimed_ids`` uses, recording the *sequence*
    rather than the ids: a pane's reading order is the order it submits its
    items in, and that is the only thing a headless test can see of a layout.
    """
    seen: list[str] = []
    originals = {name: getattr(imgui, name) for name in _ID_WIDGETS if hasattr(imgui, name)}

    def wrap(fn):
        def recorded(label, *args, **kwargs):
            if isinstance(label, str):
                seen.append(label)
            return fn(label, *args, **kwargs)

        return recorded

    for name, fn in originals.items():
        setattr(imgui, name, wrap(fn))
    try:
        imgui.new_frame()
        imgui.set_next_window_size((420, 900))
        imgui.begin(title)
        build()
        imgui.end()
        imgui.render()
    finally:
        for name, fn in originals.items():
            setattr(imgui, name, fn)
    return seen


def _index_of(labels, needle):
    for index, label in enumerate(labels):
        if needle in label:
            return index
    return -1


def test_plotter_tileset_pane_leads_with_the_picker(app_ctx, imgui_ctx):
    """What a map is painted *with* comes before how a tileset gets onto it.

    The pane used to open with "Add from a file..." and a generator header,
    which put ways of *acquiring* a tileset above the picker for the one
    already loaded -- so the control used on every single click sat below
    controls used once per map. The generators were retired on 2026-08-18 when
    tile sheets moved to Create.

    The ordering rule is satisfied more strongly since the footer bar landed:
    the acquisition doors are not in the column at all. They were about a
    hundred pixels of full-width buttons *below* the picker -- one per open
    Inker document -- and they are four rows behind the bar's ``...`` now, at
    the cost of one glyph. So what this asserts is that the strip leads, the
    bar follows it, and none of the four is drawn in the column.
    """
    from warlock.studio import plotter_mode
    from warlock.studio.panes import plotter_tileset

    imgui, _renderer = imgui_ctx
    tab = plotter_mode.new_document(app_ctx, (8, 8, 16, 16))
    tab.doc.add_tileset(_tileset())

    title = "##tileset-order"
    _drawn_labels(imgui, lambda: plotter_tileset.draw(app_ctx), title)  # settle headers
    labels = _drawn_labels(imgui, lambda: plotter_tileset.draw(app_ctx), title)

    # The picker is a *tab strip* since W4.6 -- one click per tileset instead of
    # a combo's two -- so what leads the pane is the strip and its first tab.
    picker = _index_of(labels, plotter_tileset.TABS)
    assert _index_of(labels, "terrain###ts0") >= 0, labels
    assert picker >= 0, labels
    bar = _index_of(labels, "Fit")
    assert bar > picker, f"the strip leads and the bar follows it: {labels}"
    # And the picker itself comes after the bar rather than under a column of
    # buttons: the bar is one row, which is what buys the picker the pane.
    assert _index_of(labels, "##palette-sources") > picker, labels
    for gone in ("Add from a file", "Polish in Inker", "Reload the image"):
        assert _index_of(labels, gone) == -1, f"{gone} belongs behind the ...: {labels}"
    # Nothing in Plotter generates a tileset any more.
    assert _index_of(labels, "Generate a ground set") == -1, labels


def test_plotter_tileset_pane_is_only_the_file_door_when_empty(app_ctx, imgui_ctx):
    """With no tileset attached there is nothing to pick and nothing to polish,
    so the one way of getting one is the whole pane."""
    from warlock.studio import plotter_mode
    from warlock.studio.panes import plotter_tileset

    imgui, _renderer = imgui_ctx
    plotter_mode.new_document(app_ctx, (8, 8, 16, 16))

    title = "##tileset-order-empty"
    _drawn_labels(imgui, lambda: plotter_tileset.draw(app_ctx), title)
    labels = _drawn_labels(imgui, lambda: plotter_tileset.draw(app_ctx), title)

    assert _index_of(labels, "Add from a file") >= 0, labels
    assert _index_of(labels, "Polish in Inker") == -1, "nothing to polish with no tileset"
    assert _index_of(labels, "Generate a ground set") == -1, labels


def test_the_sheet_popup_pump_runs_before_the_empty_map_return(app_ctx, imgui_ctx):
    """A ruled sheet arriving onto a map with *no* tilesets is the commonest
    way this feature is reached -- it is how a map gets its first tileset -- so
    a pump placed after the pane's empty-map early return would park megabytes
    that no popup ever asks about."""
    import numpy as np

    from warlock.studio import plotter_mode
    from warlock.studio.panes import plotter_tileset
    from warlock.studio.tilegrid import slicing

    imgui, _renderer = imgui_ctx
    tab = plotter_mode.new_document(app_ctx, (8, 8, 16, 16))
    assert not tab.doc.tilesets

    sheet = np.zeros((50, 50, 4), dtype=np.uint8)
    sheet[:, :, 3] = 255
    for r in range(3):
        for c in range(3):
            y, x = r * 17, c * 17
            sheet[y : y + 16, x : x + 16, :3] = 200
    grid = slicing.detect_grid(sheet)
    assert grid is not None

    state = plotter_mode.ensure(app_ctx)
    state.sheet_import = (tab.uid, "sheet", "", sheet, grid)
    state.sheet_import_open = False

    _drawn_labels(imgui, lambda: plotter_tileset.draw(app_ctx), "##sheet-pump")
    assert state.sheet_import_open is True, "the pump must precede the empty-map return"


def test_a_frame_without_the_sheet_popup_drops_the_pixels(app_ctx, imgui_ctx):
    """imgui closes a popup on a click outside, and the parked sheet is
    megabytes: the frame that finds the popup gone is what releases them."""
    import numpy as np

    from warlock.studio import plotter_mode
    from warlock.studio.panes import plotter_tileset
    from warlock.studio.tilegrid import slicing

    imgui, _renderer = imgui_ctx
    tab = plotter_mode.new_document(app_ctx, (8, 8, 16, 16))

    sheet = np.zeros((50, 50, 4), dtype=np.uint8)
    sheet[:, :, 3] = 255
    for r in range(3):
        for c in range(3):
            y, x = r * 17, c * 17
            sheet[y : y + 16, x : x + 16, :3] = 200
    grid = slicing.detect_grid(sheet)

    state = plotter_mode.ensure(app_ctx)
    state.sheet_import = (tab.uid, "sheet", "", sheet, grid)
    # Already flagged open, so the pump does not re-open it; the popup itself
    # was never begun, which is exactly the click-outside shape.
    state.sheet_import_open = True

    _drawn_labels(imgui, lambda: plotter_tileset.draw(app_ctx), "##sheet-gone")
    assert state.sheet_import is None
    assert state.sheet_import_open is False


def test_the_resize_popup_shows_a_detected_pixel_grid(app_ctx):
    """An SDXL "pixel art" render at 1024 squared whose real art is 64 squared
    is the commonest AI artefact after a ragged tilesheet, and the measurement
    already existed on the export path -- this is the editor reaching it."""
    import numpy as np

    from warlock.studio import inker_mode
    from warlock.studio.inker.document import Document as _Doc
    from warlock.studio.panes import inker_bridge

    rng = np.random.default_rng(5)
    palette = np.array(
        [[20, 20, 30], [200, 60, 40], [40, 160, 90], [230, 220, 190]], dtype=np.uint8
    )
    cells = np.zeros((12, 12, 4), dtype=np.uint8)
    cells[..., :3] = palette[rng.integers(0, len(palette), (12, 12))]
    cells[..., 3] = 255
    art = np.repeat(np.repeat(cells, 8, axis=0), 8, axis=1)

    state = inker_mode.ensure(app_ctx)
    tab = inker_mode._adopt(app_ctx, state, _Doc.blank(96, 96), path=None, title="render")
    tab.doc.stack.active.pixels[:] = art
    tab.doc.invalidate_all()

    inker_bridge._measure_pixel_grid(app_ctx, tab)
    found = app_ctx.state.preview[f"inker_grid:{tab.uid}"]
    assert found["scale"] == 8, found

    # And an ordinary drawing measures nothing, so the popup is unchanged for it.
    plain = inker_mode._adopt(app_ctx, state, _Doc.blank(96, 96), path=None, title="plain")
    ramp = np.linspace(0, 255, 96).astype(np.uint8)
    plain.doc.stack.active.pixels[..., :3] = ramp[None, :, None]
    plain.doc.stack.active.pixels[..., 3] = 255
    plain.doc.invalidate_all()
    inker_bridge._measure_pixel_grid(app_ctx, plain)
    assert app_ctx.state.preview[f"inker_grid:{plain.uid}"]["scale"] is None


def test_the_seam_readout_and_wrap_button_appear_only_in_tiled_mode(app_ctx):
    """Both belong to the wrap, so neither exists on a drawing nobody is
    tiling -- and the readout is cached against the undo serial, because a
    full-image diff per frame is the frame-thread stall the task layer exists
    to avoid."""
    import numpy as np

    from warlock.studio import inker_mode
    from warlock.studio.inker.document import Document as _Doc
    from warlock.studio.panes import inker_canvas

    state = inker_mode.ensure(app_ctx)
    tab = inker_mode._adopt(app_ctx, state, _Doc.blank(32, 32), path=None, title="tile")
    rng = np.random.default_rng(2)
    tab.doc.stack.active.pixels[..., :3] = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    tab.doc.stack.active.pixels[..., 3] = 255
    tab.doc.invalidate_all()

    tab.tiled = "off"
    assert inker_canvas.seam_text(app_ctx, tab) is None

    tab.tiled = "both"
    first = inker_canvas.seam_text(app_ctx, tab)
    assert first is not None and first[1].startswith("seam x")

    # Cached: a second call at the same undo serial does not recompute, and the
    # entry is keyed on that serial.
    serial, worst = app_ctx.state.preview[f"inker_seam:{tab.uid}"]
    assert serial == tab.doc.history.head
    assert inker_canvas.seam_text(app_ctx, tab) == first

    # An edit moves the serial, so the next call measures again.
    tab.doc.offset_layer(16, 16)
    assert app_ctx.state.preview[f"inker_seam:{tab.uid}"][0] == serial
    inker_canvas.seam_text(app_ctx, tab)
    assert app_ctx.state.preview[f"inker_seam:{tab.uid}"][0] == tab.doc.history.head
    assert worst is not None


def test_a_rendered_sheet_offers_both_hand_offs(app_ctx, imgui_ctx):
    """The sheet used to dead-end at Save PNG, though everything needed to open
    it in either editor already existed."""
    from warlock.studio.panes import sheet_panel

    imgui, _renderer = imgui_ctx
    job_id = _seeded(app_ctx)
    job = app_ctx.cache.get(job_id)
    job["files"] = ["model.glb"]
    app_ctx.state.preview["sheets"] = [
        {
            "id": "s1",
            "name": "turnaround",
            "columns": 4,
            "rows": 2,
            "frame_size": 16,
            "cells": [{"index": i} for i in range(8)],
        }
    ]

    labels = _drawn_labels(imgui, lambda: sheet_panel.draw(app_ctx, job), "##sheet-hand")
    assert _index_of(labels, "Open in Inker") >= 0, labels
    assert _index_of(labels, "Add to Packwright") >= 0, labels


def test_the_offset_and_autocrop_controls_render(app_ctx, imgui_ctx):
    from warlock.studio import plotter_mode

    imgui, _renderer = imgui_ctx
    tab = plotter_mode.new_document(app_ctx, (8, 8, 16, 16))
    tab.doc.add_tileset(_tileset())
    app_ctx.settings.set("plotter/resize", True)

    # The form is the Map > Resize dialog now (W3.1), so it is driven through
    # the popup's own function rather than through the pane: two hundred lines
    # of accordion for a form answered once per map is a dialog.
    state = plotter_mode.ensure(app_ctx)
    state.resize_pending = True

    def frame() -> None:
        from warlock.studio.panes import plotter_tools as pane

        pane.resize_popup(app_ctx, state, tab)

    _drawn_labels(imgui, frame, "##k-geom")
    labels = _drawn_labels(imgui, frame, "##k-geom")
    assert _index_of(labels, "Move X") >= 0, labels
    assert _index_of(labels, "Autocrop to content") >= 0, labels
    assert _index_of(labels, "Scope") >= 0, labels


def test_the_wand_row_renders_and_no_dead_generator_route_remains(app_ctx, imgui_ctx):
    """`plotter/generate` has had no registered owner since the generator was
    deleted on 2026-08-18 -- a route that was live and did nothing."""
    import subprocess
    from pathlib import Path as _Path

    from warlock.studio import plotter_mode
    from warlock.studio.panes import plotter_tools

    imgui, _renderer = imgui_ctx
    plotter_mode.new_document(app_ctx, (8, 8, 16, 16))
    labels = _drawn_labels(imgui, lambda: plotter_tools.draw(app_ctx), "##k-wand")
    # By its id, not by the letter "W": that letter was being matched against
    # the tile-size field's *Width* box, which has since moved into the Map >
    # Resize dialog (W3.1) -- so the assertion passed for four months without
    # ever looking at the Wand button. The id changed shape with the pill:
    # ``controls.segmented_choice`` keys its segments ``<control id>/<key>``.
    assert any("##plotter-tool/wand" in label for label in labels), labels
    assert _index_of(labels, "Open the generator") == -1, labels

    root = _Path(__file__).resolve().parents[1] / "src"
    found = subprocess.run(
        ["git", "grep", "-n", "plotter/generate", "--", str(root)],
        capture_output=True,
        text=True,
        cwd=str(root.parent),
    )
    assert found.stdout.strip() == "", found.stdout


def test_the_menu_bar_and_status_bar_actually_render(app_ctx, imgui_ctx):
    """The half of the editor shell that ``test_editor_shell`` cannot reach.

    That file tests ``menus.specs`` and ``status_bar.items`` -- the pure data
    behind the two surfaces -- and nothing anywhere tested the imgui half:
    ``begin_menu_bar``/``end_menu_bar`` pairing, the style vars pushed around
    it, and the status bar's ``begin_child``. Every other pane in this app is
    smoke-rendered; these two were the exception, which is a poor thing for the
    two newest modules in the shell to be.

    The assertion is the frame completing. An unbalanced style stack or a
    missing ``end_menu_bar`` does not raise where it happens -- it corrupts the
    draw data and surfaces later, so ``imgui.render()`` running clean over both
    is the real check, and ``imgui.end()`` would already have thrown if the
    window stack were wrong.
    """
    from warlock.studio import menus, status_bar

    imgui, renderer = imgui_ctx
    imgui.new_frame()
    imgui.set_next_window_size((1200, 900))
    # The menu-bar flag is the whole reason this cannot use ``_frame``:
    # ``begin_menu_bar`` returns False in a window without it, and a test that
    # only ever exercised the early return would be the same gap again.
    imgui.begin("##shell-host", None, imgui.WindowFlags_.menu_bar.value)
    menus.draw(app_ctx)
    status_bar.draw(app_ctx)
    imgui.end()
    imgui.render()
    renderer.render(imgui.get_draw_data())

    # And the stacks are back where they started, which is the thing a leaked
    # push_style_var would break for whatever drew next rather than for this.
    assert imgui.get_current_context() is not None


# --- W4: Plotter's six small gaps, drawn for real ----------------------------
#
# Each of these was a capability with no control in front of it, so each test
# below asserts about the *control*: that it is submitted into a real frame, and
# that pressing it reaches the door. A test that called the door instead would
# have passed before any of this was built.


def _plotter_tab(app_ctx, tilesets=("terrain",)):
    from warlock.studio import plotter_mode

    tab = plotter_mode.new_document(app_ctx, (8, 8, 16, 16))
    for name in tilesets:
        tab.doc.add_tileset(_tileset())
        tab.doc.tilesets[-1] = dataclasses.replace(
            tab.doc.tilesets[-1],
            tileset=dataclasses.replace(tab.doc.tilesets[-1].tileset, name=name),
        )
    return tab


def test_the_object_toolbox_draws_a_capsule_button(app_ctx, imgui_ctx):
    """The shape was drawable, hit-testable and writable by four codecs, and
    reachable only by hand-editing a file."""
    from warlock.studio.panes import plotter_tools

    imgui, _renderer = imgui_ctx
    tab = _plotter_tab(app_ctx)
    layer = tab.doc.add_object_layer()
    tab.doc.set_active_layer(layer.uid)

    title = "##capsule-tool"
    _drawn_labels(imgui, lambda: plotter_tools.draw(app_ctx), title)
    labels = _drawn_labels(imgui, lambda: plotter_tools.draw(app_ctx), title)
    assert any("##plotter-tool/object_capsule" in label for label in labels), labels


def test_the_undo_history_popover_lists_the_stack_and_jumps(app_ctx, imgui_ctx):
    """The step count was on screen and was not a control; the stack behind it
    had no panel at all."""
    import numpy as np

    from warlock.studio import controls, plotter_mode
    from warlock.studio.panes import plotter_bridge
    from warlock.studio.tilegrid import gid as gidlib

    imgui, _renderer = imgui_ctx
    tab = _plotter_tab(app_ctx)
    layer = tab.doc.tile_layers()[0]
    for column in range(3):
        tab.doc.write_region(
            layer.uid, column, 0, np.array([[column + 1]], gidlib.DTYPE)
        )
    depth = len(tab.doc.history)

    def frame() -> None:
        imgui.open_popup(plotter_bridge.HISTORY_POPUP)
        plotter_bridge._history_popup(app_ctx, tab)

    labels = _drawn_labels(imgui, frame, "##plotter-history")
    assert _index_of(labels, "(the map as opened)") >= 0, labels
    # One row per step, and the head marked.
    assert sum(1 for label in labels if "##plotter-undo" in label) == depth + 1, labels
    assert any(label.startswith("tile patch  <") for label in labels), labels

    # And a row is a jump. ``controls.selectable`` is what a row *is*, so
    # replacing it is what clicking one is -- the popup, the loop and the
    # dispatch are all still the real thing.
    original = controls.selectable
    target = "##plotter-undo0"

    def fake_selectable(label, selected=False, *args, **kwargs):
        original(label, selected, *args, **kwargs)
        return (target in label, selected)

    controls.selectable = fake_selectable
    try:
        _frame(imgui_ctx, frame)
    finally:
        controls.selectable = original

    assert len(tab.doc.history) == 1, "the click did not move the head"
    assert int(layer.data[0, 0]) == 0
    assert plotter_mode.ensure(app_ctx).selected_object is None


def test_go_to_coordinate_draws_a_dialog_that_moves_the_view(app_ctx, imgui_ctx):
    """Menu -> flag -> popup -> door -> pan, every leg through the real code."""
    from warlock.studio import plotter_mode, widgets
    from warlock.studio.panes import plotter_canvas

    imgui, _renderer = imgui_ctx
    tab = _plotter_tab(app_ctx)
    state = plotter_mode.ensure(app_ctx)
    state.goto_pending = True

    labels = _drawn_labels(
        imgui, lambda: plotter_canvas.goto_popup(app_ctx, state, tab), "##goto"
    )
    assert _index_of(labels, "Column##goto-x") >= 0, labels
    assert _index_of(labels, "Row##goto-y") >= 0, labels

    app_ctx.state.preview[f"plotter_goto:{tab.uid}"] = {"x": 6, "y": 5}
    original = widgets.primary_button
    widgets.primary_button = lambda label, *a, **k: "goto-apply" in label
    try:
        state.goto_pending = True
        _frame(imgui_ctx, lambda: plotter_canvas.goto_popup(app_ctx, state, tab))
    finally:
        widgets.primary_button = original

    assert state.goto_cell == (6, 5)

    # And the canvas turns that into a pan on the frame it reads it.
    before = tab.view.pan
    _frame(imgui_ctx, lambda: plotter_canvas.draw(app_ctx))
    assert state.goto_cell is None, "the flag is cleared by the frame that uses it"
    assert tab.view.pan != before


def test_the_tileset_strip_draws_one_tab_per_tileset(app_ctx, imgui_ctx):
    from warlock.studio.panes import plotter_tileset

    imgui, _renderer = imgui_ctx
    _plotter_tab(app_ctx, tilesets=("Grass", "Dungeon", "props"))

    title = "##tileset-tabs"
    _drawn_labels(imgui, lambda: plotter_tileset.draw(app_ctx), title)
    labels = _drawn_labels(imgui, lambda: plotter_tileset.draw(app_ctx), title)
    for index, name in enumerate(("Grass", "Dungeon", "props")):
        assert _index_of(labels, f"{name}###ts{index}") >= 0, labels
    # Three tilesets is under the filter's threshold, so no box.
    assert _index_of(labels, "##find-plotter-tilesets") == -1, labels


def test_choosing_a_tab_changes_the_tileset_and_drops_the_brush(app_ctx, imgui_ctx):
    """imgui owns the click; what this owns is what it does with the answer.

    ``begin_tab_item`` is replaced so a *named* tab reports itself open, which
    is exactly what a press does -- the strip, the reconciliation and the brush
    rule are all the real code.
    """
    import imgui_bundle
    import numpy as np

    from warlock.studio import plotter_mode
    from warlock.studio.panes import plotter_tileset
    from warlock.studio.tilegrid import gid as gidlib

    imgui, _renderer = imgui_ctx
    _plotter_tab(app_ctx, tilesets=("Grass", "Dungeon", "props"))
    state = plotter_mode.ensure(app_ctx)
    state.brush = np.array([[gidlib.compose(1)]], gidlib.DTYPE)

    # One settling frame first, so the strip has reported where it is and stops
    # forcing its own selection -- which is the state a user's click arrives in.
    _frame(imgui_ctx, lambda: plotter_tileset.draw(app_ctx))
    assert state.tileset_index == 0

    original = imgui.begin_tab_item

    def fake_tab_item(label, p_open=None, flags=0):
        # What a press does, said to imgui in imgui's own terms: the tab bar,
        # the selection and everything the pane does with the answer are real.
        if "###ts2" in label:
            flags |= imgui.TabItemFlags_.set_selected.value
        return original(label, p_open, flags)

    imgui_bundle.imgui.begin_tab_item = fake_tab_item
    try:
        # Twice: ``SetSelected`` writes ``NextSelectedTabId``, which the tab bar
        # honours at its *next* layout -- imgui's own one-frame latency, and the
        # reason the pane may not assume the tab it flagged is the tab it gets.
        _frame(imgui_ctx, lambda: plotter_tileset.draw(app_ctx))
        _frame(imgui_ctx, lambda: plotter_tileset.draw(app_ctx))
    finally:
        imgui_bundle.imgui.begin_tab_item = original

    assert state.tileset_index == 2
    assert state.brush is None, "a brush is numbered against the set it came from"


def test_the_tileset_filter_appears_and_narrows_the_strip(app_ctx, imgui_ctx):
    """At the count where the strip starts scrolling, and not before."""
    from warlock.studio.panes import plotter_tileset

    imgui, _renderer = imgui_ctx
    names = ("Grass", "grass cliff", "Dungeon", "props", "water", "sand", "ice", "lava")
    _plotter_tab(app_ctx, tilesets=names)

    title = "##tileset-filter"
    _drawn_labels(imgui, lambda: plotter_tileset.draw(app_ctx), title)
    labels = _drawn_labels(imgui, lambda: plotter_tileset.draw(app_ctx), title)
    assert _index_of(labels, "##find-plotter-tilesets") >= 0, labels

    app_ctx.state.list_filters[plotter_tileset.FILTER_TAG] = "grass"
    labels = _drawn_labels(imgui, lambda: plotter_tileset.draw(app_ctx), title)
    tabs = [label for label in labels if "###ts" in label]
    # The two matches, and the set in hand (index 0, which is also a match).
    assert tabs == ["Grass###ts0", "grass cliff###ts1"], tabs

    # A query that matches nothing still leaves the tileset being painted with.
    app_ctx.state.list_filters[plotter_tileset.FILTER_TAG] = "zzz"
    labels = _drawn_labels(imgui, lambda: plotter_tileset.draw(app_ctx), title)
    assert [label for label in labels if "###ts" in label] == ["Grass###ts0"], labels


def _drain_task(runner, key, timeout=10.0):
    """Wait for one submitted key to come back. -> its ``Done``.

    The task runner is a real thread pool here, which is the point: a browser
    that submits nothing, or submits under a key nothing lands, would pass a
    test that only looked at the frame it was clicked on.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for done in runner.poll():
            if done.key == key:
                return done
        time.sleep(0.01)
    raise AssertionError(f"{key} never finished")


def test_the_palette_folder_browser_actually_loads_a_palette(app_ctx, imgui_ctx, monkeypatch):
    """**Pressed, not called.** The browser is found in the frame it drew, its
    button is clicked where imgui put it, and the palette has to arrive in the
    swatch row through the same task and the same landing branch the file
    dialog's import uses.

    A control that draws correctly and does nothing is this codebase's most
    common historical defect, and a test that calls the handler directly is
    precisely the test that would not notice one: the wiring is the half that
    breaks. ``probe`` is what makes the button addressable -- positions are
    read from imgui rather than computed, for the reason its own docstring
    gives.
    """
    from warlock.studio import inker_mode, probe
    from warlock.studio.panes import inker_colors

    folder = Path(app_ctx.svc.config.palette_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "wave9.hex").write_text("112233\n445566\n", encoding="utf-8")
    # The listing is remembered per directory version; a directory written to
    # inside one 15.6 ms tick is exactly the case that remembering guards.
    app_ctx.state.palettes = None

    state = inker_mode.ensure(app_ctx)
    state.swatches = []
    monkeypatch.setattr(probe, "ENABLED", True)

    def build():
        inker_colors._palette_folder(app_ctx, state)

    # Located inside ``_click``'s own host window, and not in ``_frame``'s:
    # they are different windows at different positions, and a rect read from
    # one is empty space in the other.
    probe.begin_frame()
    _click(imgui_ctx, build, (-100.0, -100.0))
    buttons = [one for one in probe.census() if one.text == "Load"]
    assert buttons, [one.text for one in probe.census()]
    assert buttons[-1].visible and buttons[-1].enabled, buttons[-1]

    probe.begin_frame()
    _click(imgui_ctx, build, buttons[-1].centre)
    done = _drain_task(app_ctx.tasks, "inker-palette")
    assert done.error is None, done.message
    inker_mode.on_task_done(app_ctx, done)
    assert state.swatches == [(0x11, 0x22, 0x33, 255), (0x44, 0x55, 0x66, 255)]


def test_the_palette_folder_browser_says_so_when_the_folder_is_empty(
    app_ctx, imgui_ctx, monkeypatch
):
    """The ordinary state of a fresh install: a muted line naming the formats,
    and no control at all -- never a button that submits an empty load."""
    from warlock.studio import inker_mode, probe
    from warlock.studio.panes import inker_colors

    folder = Path(app_ctx.svc.config.palette_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.iterdir():
        stale.unlink()
    app_ctx.state.palettes = None
    state = inker_mode.ensure(app_ctx)
    monkeypatch.setattr(probe, "ENABLED", True)

    def build():
        inker_colors._palette_folder(app_ctx, state)

    probe.begin_frame()
    _frame(imgui_ctx, build)
    assert [one.text for one in probe.census()] == []

    # The same pane, one file later: the empty branch is a fact about the
    # folder and not a browser that never draws anything.
    (folder / "wave9.gpl").write_text("GIMP Palette\n1 2 3\n", encoding="utf-8")
    app_ctx.state.palettes = None
    probe.begin_frame()
    _frame(imgui_ctx, build)
    assert "Load" in [one.text for one in probe.census()]


def test_the_plotter_layer_bar_and_the_rename_field_both_render(app_ctx, imgui_ctx):
    """Two branches of the layers pane with no other route to a frame.

    The bar is where every layer verb lives since wave B -- add, duplicate,
    raise, lower, lock, delete -- and the rename field replaces the row's label
    while it is being typed. Neither draws unless it is asked for, so neither
    rasterises in the workspace frame above.
    """
    from warlock.studio import plotter_mode
    from warlock.studio.panes import plotter_layers

    imgui, _renderer = imgui_ctx
    tab = plotter_mode.new_document(app_ctx, (8, 8, 16, 16))
    state = plotter_mode.ensure(app_ctx)
    layer = tab.doc.tile_layers()[0]
    tab.doc.set_active_layer(layer.uid)

    title = "##layer-bar"
    labels = _drawn_labels(imgui, lambda: plotter_layers.draw(app_ctx), title)
    for key in ("##plotter-layer-add", "##plotter-layer-dup", "##plotter-layer-del"):
        assert _index_of(labels, key) >= 0, labels
    # The bar is under the list, which is the whole point of the footer child:
    # thirty layers must not push Delete off the bottom of the pane.
    assert _index_of(labels, "##plotter-layer-add") > _index_of(
        labels, f"plotter-layer/{layer.uid}"
    ), labels

    # And the rename branch, which swaps the label for a field.
    state.renaming_layer = int(layer.uid)
    labels = _drawn_labels(imgui, lambda: plotter_layers.draw(app_ctx), title)
    assert _index_of(labels, f"##plotter-rename-{layer.uid}") >= 0, labels
    # A second frame, so the field is drawn with a draft already parked rather
    # than only on the frame that creates one.
    _drawn_labels(imgui, lambda: plotter_layers.draw(app_ctx), title)
    state.renaming_layer = 0

    # A saving tab greys the bar rather than hiding it, and the rename field is
    # not offered at all -- a text box that discards what is typed is worse
    # than one that is not there.
    # ``busy`` is derived from ``saving``, which is the flag a save sets.
    tab.saving = True
    state.renaming_layer = int(layer.uid)
    _drawn_labels(imgui, lambda: plotter_layers.draw(app_ctx), title)
    tab.saving = False
    state.renaming_layer = 0


def test_the_plotter_stamp_ghost_draws_the_brush_under_the_pointer(app_ctx, imgui_ctx):
    """The cursor's three parts, which need a *hovered* canvas to run at all.

    The workspace frame above never puts the pointer over the map, so the hover
    wash, the ghost and the footprint outline had no route to a frame -- and the
    ghost resolves tilesets and uploads textures, which is exactly the kind of
    code a smoke frame exists to execute.
    """
    import numpy as np

    from warlock.studio import plotter_mode
    from warlock.studio.panes import plotter_canvas
    from warlock.studio.tilegrid import gid

    tab = plotter_mode.new_document(app_ctx, (8, 8, 16, 16))
    state = plotter_mode.ensure(app_ctx)
    ref = tab.doc.add_tileset(_tileset())
    # A 2x2 brush with a hole and a flipped corner: the ghost draws each cell
    # through ``_cell_quad``, so the flag path is what a second copy of the
    # arithmetic would get wrong.
    state.brush = np.array(
        [
            [gid.compose(ref.firstgid), 0],
            [gid.compose(ref.firstgid + 1, flip_h=True), gid.compose(ref.firstgid + 2)],
        ],
        gid.DTYPE,
    )
    state.tool = "stamp"

    def build() -> None:
        plotter_canvas.draw(app_ctx)

    # Over the middle of the host window, which is where the canvas is.
    _click(imgui_ctx, build, (400, 400))
    # Random mode draws one tile rather than the block, because one is what
    # lands -- the other branch of the ghost.
    state.random_mode = True
    _click(imgui_ctx, build, (400, 400))
    state.random_mode = False
    # And every other tool's wash, which is the branch that is not the stamp.
    for tool in ("erase", "fill", "select", "wand"):
        state.tool = tool
        _click(imgui_ctx, build, (400, 400))
    state.tool = "stamp"


#: What Clay's centre column actually gets at the app's default 1600x950. Two
#: 300 dp sidebars and a 70 dp mode rail leave about this -- the number
#: ``inker_context`` measured for the same reason and wrote down.
CLAY_CENTRE_AT_DEFAULT = 835.0


def _clay_header_tiers(imgui, avail: float) -> list[str]:
    """The tier ``toolbar`` would choose for each header entry at ``avail`` px."""
    from warlock.studio import clay_state, toolbar
    from warlock.studio.panes import clay_header

    state = clay_state.ClayState()
    items = clay_header._items(state)
    fields = [clay_header._tool_field(state)]
    style = imgui.get_style()
    square = imgui.get_frame_height()
    full = [
        imgui.calc_text_size(item.label).x + style.frame_padding.x * 2.0
        for item in items
    ]
    icon = [square for _item in items]
    for field in fields:
        wide, narrow = field.widths()
        full.append(wide)
        icon.append(narrow)
    entries = [*items, *fields]
    return toolbar.plan(
        full,
        icon,
        [entry.priority for entry in entries],
        [entry.pinned for entry in entries],
        avail,
        square,
        gap=style.item_spacing.x,
    )


def test_the_clay_header_fits_at_the_default_window(app_ctx, imgui_ctx):
    """Every label on, at 1600x950.

    ``toolbar`` degrades rather than clips, so a header that does not fit does
    not *look* broken -- it looks like a header whose every label is a hover
    away, on the machine everybody uses. Measured here rather than in
    ``tests/test_clay_header.py`` because measuring a font needs a live imgui
    context and that file must not build a second one.
    """
    from warlock.studio import toolbar

    imgui, _renderer = imgui_ctx
    # ``_frame`` returns nothing, so the reading is carried out in a list.
    out: list[list[str]] = []
    _frame(
        imgui_ctx,
        lambda: out.append(_clay_header_tiers(imgui, CLAY_CENTRE_AT_DEFAULT)),
    )
    assert set(out[0]) == {toolbar.FULL}, out[0]


def test_the_clay_header_gives_up_labels_before_controls(app_ctx, imgui_ctx):
    """The ordering the module rests on: a label is cheaper to lose than a
    control, so a genuinely narrow centre drops to glyphs rather than hiding a
    button."""
    from warlock.studio import toolbar

    imgui, _renderer = imgui_ctx
    out: list[list[str]] = []
    _frame(imgui_ctx, lambda: out.append(_clay_header_tiers(imgui, 120.0)))
    assert toolbar.ICON in out[0] or toolbar.MENU in out[0]


def test_the_clay_hud_draws_its_widget_and_its_line(app_ctx, imgui_ctx):
    """The axis ball and the hint line, neither of which the pane walk reaches:
    the widget needs a live viewport to read a camera off."""
    from types import SimpleNamespace

    from warlock.studio import clay_mode
    from warlock.studio.panes import clay_hud
    from warlock.studio.viewer.camera import Camera

    tab = _clay_tab(app_ctx)
    view = SimpleNamespace(camera=Camera())
    rect = (10.0, 10.0, 600.0, 400.0)

    labels = _drawn_labels(
        imgui_ctx[0],
        lambda: clay_hud.axis_widget(app_ctx, view, rect),
        "##clay-axis",
    )
    for name in ("front", "back", "right", "left", "top", "bottom"):
        assert _index_of(labels, f"##clay-axis/{name}") >= 0, labels

    # And with no viewport at all, which is the first frame.
    _frame(imgui_ctx, lambda: clay_hud.axis_widget(app_ctx, None, rect))

    # The line, in every mode and mid-drag.
    state = clay_mode.ensure(app_ctx)
    for mode in ("object", "vertex", "edge", "face"):
        tab.doc.set_element_mode(mode)
        _frame(imgui_ctx, lambda: clay_hud.hint_line(app_ctx))
    state.drag_kind = "move"
    _frame(imgui_ctx, lambda: clay_hud.hint_line(app_ctx))
    state.drag_kind = ""


def test_the_plotter_properties_table_draws_every_branch(app_ctx, imgui_ctx):
    """The name/value table, in each of the four things it can be showing.

    Three of these have no other route to a frame: the map table draws only
    with *no* layer selected, a folded container draws only once the chevron
    has been pressed, and the ``-`` in the footer is live only with a top-level
    row selected.
    """
    from warlock.studio import plotter_mode
    from warlock.studio.panes import plotter_layers
    from warlock.studio.plotter.tsx import Prop

    imgui, _renderer = imgui_ctx
    tab = plotter_mode.new_document(app_ctx, (8, 8, 16, 16))
    state = plotter_mode.ensure(app_ctx)
    doc = tab.doc
    layer = doc.tile_layers()[0]

    title = "##props-table"

    # 1. No layer at all -- the map's own table, which replaced a sentence
    #    telling the user to go and select something.
    doc.set_active_layer(None)
    labels = _drawn_labels(imgui, lambda: plotter_layers.draw_properties(app_ctx), title)
    # By the fields rather than by the table id: ``begin_table`` is not one of
    # the id widgets ``_drawn_labels`` wraps, so what a table *is* has to be
    # asserted through what it contains. The map branch draws the property
    # editor and no layer fields; the layer branch is the other way round.
    assert _index_of(labels, "##add-prop") >= 0, labels
    assert _index_of(labels, "##layer-name") == -1, labels

    # 2. A layer, with a class property and a list property on it.
    doc.set_active_layer(layer.uid)
    doc.set_layer_props(
        layer.uid,
        properties={
            "theme": Prop("string", "cave"),
            "loot": Prop("class", {"gold": Prop("int", 3)}, propertytype="Loot"),
            "tags": Prop("list", [Prop("string", "wet")]),
        },
    )
    labels = _drawn_labels(imgui, lambda: plotter_layers.draw_properties(app_ctx), title)
    for field in ("##layer-name", "##layer-class", "##layer-opacity", "##layer-blend"):
        assert _index_of(labels, field) >= 0, labels

    # 3. The class unfolded, and one row selected so the ``-`` goes live.
    form = plotter_layers._prop_form(app_ctx, f"plotter_layer_prop:{layer.uid}")
    form["selected"] = "/theme"
    _frame(imgui_ctx, lambda: plotter_layers.draw_properties(app_ctx))
    # And folded, which is the other half of the chevron.
    form["folded"].add("/loot")
    _frame(imgui_ctx, lambda: plotter_layers.draw_properties(app_ctx))
    # A nested selection, where the ``-`` refuses by name rather than acting.
    form["folded"].discard("/loot")
    form["selected"] = "/loot/gold"
    _frame(imgui_ctx, lambda: plotter_layers.draw_properties(app_ctx))
    form["selected"] = ""

    # 4. Every object shape's own rows, which the layer branch never reaches.
    objects = doc.add_object_layer()
    doc.set_active_layer(objects.uid)
    for kind in ("rect", "point", "ellipse", "text", "tile"):
        obj = plotter_layers.add_object(doc, objects, kind, 8.0, 8.0, 16.0, 16.0)
        state.select_object(obj.uid)
        _frame(imgui_ctx, lambda: plotter_layers.draw_properties(app_ctx))
    state.select_object(None)
