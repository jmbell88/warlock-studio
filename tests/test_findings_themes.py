"""The cross-cutting themes of the 2026-09-02 review, pinned.

The themes recur in more than one mode and were cheaper to fix once; the
entries were struck from the findings file as they were built, per the
repository's rule that a built thing is deleted rather than ticked, and this
is what keeps them fixed.
"""

from __future__ import annotations

from types import SimpleNamespace

# --- T4. a greyed control names the gate that actually failed ----------------


def test_the_progress_card_cancel_says_why_it_is_grey():
    from warlock.studio.panes import overlay

    assert overlay.cancel_reason(True, False) == ""
    assert "Cancelling" in overlay.cancel_reason(True, True)
    assert "stopped" in overlay.cancel_reason(False, False)


def test_the_plotter_layer_menu_does_not_blame_a_save_for_a_missing_layer():
    """It passed ``BUSY`` for the ``active``/``many`` gates, so "Delete layer"
    on a one-layer map said the map was being written."""
    from warlock.studio.panes import plotter_menu as menu

    idle = SimpleNamespace(busy=False)
    busy = SimpleNamespace(busy=True)

    assert menu._layer_reason(None) == menu.NO_MAP
    assert menu._layer_reason(busy) == menu.BUSY
    assert menu._layer_reason(idle) == ""
    assert (
        menu._layer_reason(idle, active=None, need_active=True) == menu.NO_LAYER
    )
    assert (
        menu._layer_reason(idle, active=0, need_active=True, need_many=True, many=False)
        == menu.ONE_LAYER
    )
    assert (
        menu._layer_reason(idle, active=0, need_active=True, need_many=True, many=True)
        == ""
    )
    # Busy still wins over the shape gates: it is the one that will pass on its
    # own, so it is the one worth waiting for.
    assert (
        menu._layer_reason(busy, active=None, need_active=True) == menu.BUSY
    )


def test_the_inker_tileset_doors_do_not_say_open_a_drawing_while_one_is_open():
    from warlock.studio import inker_mode

    assert inker_mode._no_document_reason(None) == "Open a drawing first."
    saving = SimpleNamespace(saving=True)
    assert "being written" in inker_mode._no_document_reason(saving)


def test_a_failed_result_row_says_it_failed_and_can_be_rerun():
    """The tray said "not ready yet" and disabled Rerun on rows that had
    already stopped, while the library card offered "Try again" on the same
    ones."""
    from warlock.studio import generation_workspace as gw

    assert "not ready yet" in gw._why_not_finished({}, "running")
    assert "cancelled" in gw._why_not_finished({}, "cancelled")
    assert "boom" in gw._why_not_finished({"error": "boom"}, "error")
    assert "failed" in gw._why_not_finished({}, "error")
    assert {"error", "cancelled"} <= gw._FAILED


def test_the_tray_opens_a_result_through_the_one_door():
    """``ctx.state.select`` left ``source_job`` stale on a reference and showed
    a mesh result's ``input.png`` on the Reference stage. There is one router,
    and every "Open" goes through it."""
    import inspect

    from warlock.studio import generation_workspace as gw

    body = inspect.getsource(gw._result_card)
    assert "asset_open.open_asset(ctx, job)" in body
    assert "ctx.state.select(" not in body


# --- T6. work nobody looks at does not run per frame -------------------------


def test_the_menu_bar_evaluates_no_gate_until_a_menu_is_open():
    """Every command's ``enabled`` ran sixty times a second at a closed bar --
    including "Empty the trash", which scans the whole job cache page."""
    from warlock.studio import menus, palette

    asked: list[str] = []

    def fake_commands(_ctx):
        return [
            palette.Command(
                key="quit",
                label="Quit",
                group="Application",
                run=lambda _ctx: None,
                enabled=lambda _ctx: asked.append("quit") is None,
            )
        ]

    ctx = SimpleNamespace(state=SimpleNamespace(mode="home"))
    original = palette.commands
    palette.commands = fake_commands
    try:
        shape = menus.specs(ctx, evaluate=False)
        assert asked == []
        assert [row.identity for row in shape] == ["command:quit"]
        menus.specs(ctx)
        assert asked == ["quit"]
    finally:
        palette.commands = original


def test_home_builds_its_resume_list_once_per_job_page():
    """Three things index this list every frame -- the count, the grid and the
    keyboard -- and the asset half walks the whole cache page for each."""
    from warlock.studio import recents
    from warlock.studio.panes import landing
    from warlock.studio.state import AppState

    class Settings:
        def get(self, key, default=None):
            return {recents.SETTING: []}.get(key, default)

        def set(self, key, value):
            pass

    jobs = [{"id": "a", "status": "done", "stage": "model", "created_at": 2.0}]
    cache = SimpleNamespace(jobs=jobs, _generation=1)
    ctx = SimpleNamespace(state=AppState(), settings=Settings(), cache=cache)

    first = landing.rows(ctx)
    assert landing.rows(ctx) is first

    cache._generation = 2
    assert landing.rows(ctx) is not first

    # A context whose cache cannot count generations is never memoised: the key
    # could not tell two job pages apart, and would serve one the other's rows.
    plain = SimpleNamespace(state=AppState(), settings=Settings(), cache=SimpleNamespace(jobs=jobs))
    assert landing.rows(plain) is not landing.rows(plain)


def test_the_review_inspector_does_not_stat_the_reference_every_frame(tmp_path):
    from warlock.studio import review_mode

    review_mode._REFERENCE_CACHE.clear()
    (tmp_path / "reference.png").write_bytes(b"x")
    unit = {"dir": str(tmp_path)}

    assert review_mode.reference_path(unit) == tmp_path / "reference.png"
    (tmp_path / "reference.png").unlink()
    # Remembered outright: the file a finished sweep unit was made from does
    # not move, and the alternative is two stats a frame forever.
    assert review_mode.reference_path(unit) == tmp_path / "reference.png"


def test_a_missing_reference_is_re_asked_rather_than_remembered(tmp_path):
    from warlock.studio import review_mode

    review_mode._REFERENCE_CACHE.clear()
    unit = {"dir": str(tmp_path)}
    assert review_mode.reference_path(unit) is None
    review_mode._REFERENCE_CACHE[str(tmp_path)] = (None, -1e9)
    (tmp_path / "input.png").write_bytes(b"x")
    assert review_mode.reference_path(unit) == tmp_path / "input.png"


def test_the_reference_stage_validates_once_a_frame_not_twice():
    """The command bar and the plan block under it both ask what is wrong, and
    both answers have to agree -- which one evaluation guarantees."""
    from warlock.studio import create_assets
    from warlock.studio.panes import settings_2d
    from warlock.studio.state import AppState, default_form_2d

    calls: list[int] = []
    original = settings_2d.validate

    def counting(form):
        calls.append(1)
        return original(form)

    form = default_form_2d()
    create_assets.sync_legacy_fields(form)
    ctx = SimpleNamespace(state=AppState())
    ctx.state.problems_cache = None
    settings_2d.validate = counting
    try:
        first = settings_2d.problems_for(ctx, form)
        assert settings_2d.problems_for(ctx, form) is first
        assert len(calls) == 1
        ctx.state.frame_index += 1
        settings_2d.problems_for(ctx, form)
        assert len(calls) == 2
    finally:
        settings_2d.validate = original


def test_a_primitive_measures_its_own_box_once():
    """``Model.bounds`` claimed it did not touch every vertex; the transform
    did not, but the ``min``/``max`` behind it did, on every frame."""
    import numpy as np

    from warlock.studio.viewer.gltf import Primitive

    prim = Primitive(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype="f4"),
        indices=np.array([0, 1, 0], dtype="u4"),
    )
    low, high = prim.box()
    assert list(high) == [1.0, 2.0, 3.0]
    prim.positions[:] = 99.0
    assert prim.box()[1] is high

    empty = Primitive(
        positions=np.zeros((0, 3), dtype="f4"), indices=np.zeros((0,), dtype="u4")
    )
    assert empty.box() is None


# --- section 2: the shell ----------------------------------------------------


def test_a_draw_buffer_reaches_the_gpu_without_being_copied_first():
    """``ctypes.string_at`` copied every vertex and index buffer into a fresh
    ``bytes`` before moderngl copied it into the VBO, once per command list per
    frame."""
    import ctypes

    from warlock.studio import imgui_backend

    payload = (ctypes.c_ubyte * 4)(1, 2, 3, 4)
    view = imgui_backend._as_bytes(ctypes.addressof(payload), 4)
    assert bytes(view) == b"\x01\x02\x03\x04"
    # The same memory, not a snapshot of it -- which is the whole point.
    payload[0] = 9
    assert bytes(view) == b"\x09\x02\x03\x04"


def _shell(mode: str = "create"):
    """``tests/test_mode_keys.py::_app``, which is the smallest thing
    ``_shortcut`` needs to route a key."""
    from types import MethodType

    from warlock.studio import main
    from warlock.studio.state import AppState

    state = AppState()
    state.mode = state.mode_observed = state.previous_mode = mode
    app = SimpleNamespace(
        app_ctx=SimpleNamespace(state=state, cache=SimpleNamespace(get=lambda _id: None)),
        viewer=SimpleNamespace(
            pose_mode=False,
            frame=lambda: None,
            set_wireframe=lambda v: None,
            set_turntable=lambda v: None,
            exit_compare=lambda: None,
        ),
    )
    for name in ("_note_mode", "_set_mode", "_escape_mode"):
        setattr(app, name, MethodType(getattr(main.App, name), app))
    return app


def _press(app, key, mod=0):
    import pygame

    from warlock.studio import main

    main.App._shortcut(app, pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod))


def test_the_command_palette_owns_the_keyboard_while_it_is_up():
    """``palette_open`` was never one of ``modal_open``'s answers, so every
    chord leaked through: Ctrl+Enter with the palette open in Create queued a
    generation behind it."""
    import pygame

    app = _shell()
    state = app.app_ctx.state

    _press(app, pygame.K_F10)
    assert state.show_fps is True

    state.palette_open = True
    _press(app, pygame.K_F10)
    assert state.show_fps is True  # not toggled back: the key never arrived


def test_ctrl_k_is_the_one_key_the_palette_does_not_swallow():
    """It is the way out, which is the whole argument for exempting it."""
    import pygame

    from warlock.studio.panes import palette

    app = _shell()
    app.app_ctx.state.palette_open = True
    _press(app, pygame.K_k, pygame.KMOD_CTRL)
    assert app.app_ctx.state.palette_open is False
    assert palette.close is not None  # the toggle really is the palette's


def test_the_manual_overlay_owns_the_keyboard_except_for_escape():
    """With the reference open, Delete in Create trashed the selected asset
    unconfirmed and a tool letter switched Inker's tool underneath it."""
    import pygame

    app = _shell()
    state = app.app_ctx.state
    state.manual.open = True

    _press(app, pygame.K_F10)
    assert state.show_fps is False

    # Esc still reaches the branch that closes it -- that branch is why Esc is
    # exempt in the first place.
    _press(app, pygame.K_ESCAPE)
    assert state.manual.open is False


def test_delete_is_bound_in_library_mode_as_well_as_the_create_sidebar():
    """The shortcuts sheet advertised it in both and only one had it."""
    import pygame

    from warlock.studio.panes import library

    app = _shell("library")
    app.app_ctx.state.selected = "job-1"
    deleted: list[str] = []
    original = library.delete_asset
    library.delete_asset = lambda _ctx, job_id: deleted.append(job_id)
    try:
        _press(app, pygame.K_DELETE)
    finally:
        library.delete_asset = original
    assert deleted == ["job-1"]


def test_the_caption_is_marked_by_any_unsaved_document_not_only_a_pose():
    from warlock.studio import docmodes
    from warlock.studio.state import AppState

    state = AppState()
    ctx = SimpleNamespace(state=state)
    assert docmodes.any_unsaved(ctx) is False

    # ``getattr`` rather than each mode's ``ensure``: asking must not create
    # the state that says no.
    state.plotter = SimpleNamespace(any_dirty=True)
    assert docmodes.any_unsaved(ctx) is True
    assert set(docmodes.DOC_MODES) == {
        "inker",
        "clay",
        "plotter",
        "packwright",
        "sirens",
    }


def test_the_prompt_dialog_grabs_the_keyboard_once_so_tab_can_leave_it():
    """``set_keyboard_focus_here`` ran on every frame the field was not active,
    which is every frame after a Tab: the focus snapped straight back."""
    import inspect

    from warlock.studio import dialogs

    body = inspect.getsource(dialogs.PromptQueue.draw)
    assert "if not prompt._focused:" in body
    assert "if not imgui.is_any_item_active():" not in body
    from dataclasses import fields

    assert "_focused" in {f.name for f in fields(dialogs.Prompt)}


def test_the_three_host_scope_popups_draw_under_a_guard():
    """They draw outside every pane, so a raise inside one took the whole
    frame down -- and the layouts popup is where finding 1 shipped from."""
    import inspect

    from warlock.studio import main

    body = inspect.getsource(main.App._build_ui)
    for key in ("shell/layouts", "shell/gallery", "shell/shortcuts"):
        assert f'"{key}"' in body


def test_one_rule_routes_the_pointer_into_all_three_viewports():
    from warlock.studio import main

    assert main._takes_pointer(None, True) is True
    assert main._takes_pointer(None, False) is False
    assert main._takes_pointer(SimpleNamespace(dragging=True), False) is True
    assert main._takes_pointer(SimpleNamespace(dragging=False), False) is False


def test_the_persistence_half_clamps_a_width_the_way_the_splitter_does():
    """``layouts`` cannot import ``layout``, so it had re-spelled the range as
    literals in three places -- and a splitter and the file it saves into
    disagreeing about the ceiling is a width that will not round-trip."""
    from warlock.studio import layout, tokens

    assert layout.PANEL_MIN is tokens.PANEL_MIN
    assert layout.PANEL_MAX is tokens.PANEL_MAX
    assert layout.SIDEBAR_WIDTHS is tokens.SIDEBAR_WIDTHS
    assert tokens.clamp_panel(10.0) == tokens.PANEL_MIN
    assert tokens.clamp_panel(9999.0) == tokens.PANEL_MAX


def test_no_pane_imports_the_frame_loop():
    """``modal_open`` and the version string both lived in ``main``, so the
    tour and Home imported the shell for one helper each."""
    from pathlib import Path

    import warlock.studio

    panes = Path(warlock.studio.__file__).parent / "panes"
    offenders = [
        path.name
        for path in panes.glob("*.py")
        if "from ..main import" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


# --- section 3: Create, Review, Library, Home --------------------------------


def test_a_revert_reloads_the_picture_off_the_frame_thread():
    """``_nudge_viewer`` ran the PNG decode inline, on the frame a revert
    lands -- which is exactly the frame something was pressed on."""
    from pathlib import Path

    from warlock.studio import viewer_embed

    submitted: list[tuple] = []
    viewer = SimpleNamespace(pending=None, parse_reference=lambda p: (p, b""))
    ctx = SimpleNamespace(
        viewer=viewer,
        submit=lambda key, fn, *a, **kw: submitted.append((key, a)) is None or True,
    )
    assert viewer_embed.request_reference(ctx, Path("x.png")) is True
    assert submitted[0][0] == viewer_embed.LOAD_KEY
    assert viewer.pending == Path("x.png")

    # A refused submit means a load is already in flight; its result is checked
    # against ``pending``, so this one is dropped rather than queued.
    ctx.submit = lambda *a, **kw: False
    assert viewer_embed.request_reference(ctx, Path("y.png")) is False
    assert viewer.pending is None


def test_the_key_the_frame_loop_lands_is_the_one_a_mode_submits_under():
    from warlock.studio import main, viewer_embed

    assert main.VIEWER_KEY == viewer_embed.LOAD_KEY


def test_a_vram_refusal_survives_long_enough_to_be_read():
    """It was a fading toast while the plan block a few pixels away went on
    saying "Ready to generate" -- and ``vram.shortfall_message`` is a
    multi-remedy sentence a toast cannot hold."""
    from warlock.studio.state import AppState

    state = AppState()
    assert state.submit_refusal == ""


def test_the_remesh_line_is_not_a_ranking():
    """INVARIANTS forbids presenting ``hole_worst`` as a quality scale, and
    "kept the best of 12.3%, 4.5%" is exactly that."""
    from warlock.studio import quality

    lines = quality.remesh_line([{"worst": 0.123}, {"worst": 0.045}])
    joined = " ".join(lines)
    assert "best" not in joined
    assert "silhouette openness" in joined
    assert "12.3%" in joined and "4.5%" in joined
    assert "kept the lowest (4.5%)" in joined
    assert "not a quality score" in joined

    # One attempt is not a remesh, so there is nothing to say.
    assert quality.remesh_line([{"worst": 0.5}]) == []

    # A kept reading the audit cannot tell from a solid slab carries the caveat.
    assert quality.UNINFORMATIVE_CAVEAT in quality.remesh_line(
        [{"worst": 0.5}, {"worst": 0.001}]
    )
    assert "unmeasured" in " ".join(quality.remesh_line([{"worst": None}, {"worst": 0.5}]))


def test_one_caveat_wording_in_one_headless_place():
    from warlock.studio import quality, review_mode, widgets

    assert widgets.AUDIT_UNINFORMATIVE is quality.AUDIT_UNINFORMATIVE
    assert quality.caveat_for(0.001) == quality.UNINFORMATIVE_CAVEAT
    assert quality.caveat_for(0.5) == ""
    assert quality.caveat_for(None) == ""
    assert "from .widgets import" not in inspect_source(review_mode.mesh_lines)


def inspect_source(fn):
    import inspect

    return inspect.getsource(fn)


def test_every_sweep_axis_explains_itself():
    """Three of fourteen had tooltips -- the three that had just been added --
    which teaches the reader that the tooltips are decoration."""
    from warlock.service.sweeps import KWARG_AXES
    from warlock.studio.review_mode import AXIS_HELP

    assert set(AXIS_HELP) == set(KWARG_AXES)
    assert all(len(text) > 30 for text in AXIS_HELP.values())


def test_the_tray_and_the_shell_agree_about_whether_there_is_a_tray():
    """``should_draw`` said yes from the first finished job onward, so the
    viewer lost ``tray_height`` permanently and the tray's own empty state was
    unreachable -- while a corpus of candidate rows reserved the strip and drew
    that empty state into it."""
    from warlock.studio import generation_workspace as gw

    empty = SimpleNamespace(cache=SimpleNamespace(jobs=[], active=None))
    assert gw.should_draw(empty) is False

    candidate = SimpleNamespace(
        cache=SimpleNamespace(
            jobs=[{"id": "a", "status": "done", "candidate_group": "g"}], active=None
        )
    )
    # Not a *result* row -- but a pending group is its own reason to draw.
    assert gw.should_draw(candidate) == (
        gw.candidates_mod.pending(candidate.cache.jobs) is not None
    )

    done = SimpleNamespace(
        cache=SimpleNamespace(jobs=[{"id": "a", "status": "done"}], active=None)
    )
    assert gw.should_draw(done) is True
    assert gw.should_draw(SimpleNamespace(cache=None)) is False


def test_deleting_several_losers_is_one_toast_and_one_undo():
    """Choosing between eight attempts finished with seven stacked toasts and
    no way to put them all back at once."""
    from warlock.studio.panes import library
    from warlock.studio.state import AppState

    toasts: list[tuple] = []
    ctx = SimpleNamespace(
        state=AppState(),
        svc=None,
        submit=lambda *a, **kw: True,
        toast=lambda *a, **kw: toasts.append(a),
    )
    library.delete_assets(ctx, ["a", "b", "c"])
    assert len(toasts) == 1
    assert toasts[0][0] == "Moved 3 to trash."
    assert toasts[0][3] == "a b c"

    restored: list[str] = []
    ctx.submit = lambda key, fn, *a, **kw: restored.append(key) is None or True
    library.restore_asset(ctx, "a b c")
    assert restored == ["restore:a", "restore:b", "restore:c"]


def test_the_export_rail_segment_does_not_import_imgui_to_answer():
    """``create_stages``' docstring says it imports nothing from imgui, and it
    reached ``widgets`` -- which imports imgui at module scope -- per frame."""
    import inspect

    from warlock.studio import artifacts, create_stages

    assert "widgets" not in inspect.getsource(create_stages._reached_export)
    assert artifacts.artifacts_for({"stage": "reference"})
    assert artifacts.artifacts_for({"stage": "tilesheet"}) == (
        ("input.png", "Tile sheet PNG"),
    )


def test_the_two_public_names_the_tray_and_the_footer_share():
    from warlock.studio import generation_workspace as gw
    from warlock.studio.panes import library

    assert callable(gw.queue_position)
    assert callable(library.copy_settings)


# --- section 4: first run, docs, tour, accessibility -------------------------


def test_the_two_colours_a_wash_and_a_knob_paint_with_are_in_the_palette():
    """Both were hard-coded white: ~1.3:1 against the light theme's EDGE track
    for the knob, and invisible over a light viewport for the wash.
    ``test_accessibility`` measures ``tokens.PALETTES`` and nothing else, so a
    literal is a colour no test can see."""
    from warlock.studio import tokens

    for name, palette in tokens.PALETTES.items():
        assert "KNOB" in palette, name
        assert "WASH" in palette, name
    assert tokens.PALETTES["dark"]["KNOB"] != tokens.PALETTES["light"]["KNOB"]


def test_no_pane_paints_a_toggle_knob_with_a_literal():
    import inspect

    from warlock.studio import controls, widgets
    from warlock.studio.panes import clay_hud

    for module in (controls, widgets, clay_hud):
        assert "0xFFFFFF" not in inspect.getsource(module)
    assert "(1.0, 1.0, 1.0, 0.2)" not in inspect.getsource(clay_hud)


def test_the_tour_welcome_step_mentions_music_and_the_rail_step_a_key():
    """Sirens shipped and the welcome step still listed four things, and the
    "click the rail" step had no keyboard path at all."""
    from warlock.studio.tour import scripts

    steps = {step.id: step for tour in scripts.TOURS for step in tour.steps}
    assert "music" in steps["welcome"].body
    assert "Ctrl+K" in steps["rail"].body
    assert "Ctrl+K" in steps["open-create"].body


def test_the_tour_card_can_be_driven_from_the_keyboard():
    import inspect

    from warlock.studio.panes import tour

    body = inspect.getsource(tour)
    assert "imgui.Key.right_arrow" in body
    assert "imgui.Key.left_arrow" in body
    assert "imgui.Key.enter" in body


def test_the_engines_do_not_have_stale_top_level_packages():
    """``src/warlock/{sirens,plotter,packwright}`` held nothing but
    ``__pycache__``; the engines live under ``studio/``."""
    from pathlib import Path

    import warlock

    root = Path(warlock.__file__).parent
    for name in ("sirens", "plotter", "packwright"):
        assert not (root / name).exists()
        assert (root / "studio" / name).is_dir()
