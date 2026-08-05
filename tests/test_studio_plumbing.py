"""Controls that submitted work whose result nothing consumed.

Every case here is the same shape: a button existed, a service call ran, and
the answer went nowhere -- so the feature looked present and did nothing. The
dispatch is driven for real, with a stand-in app and a fake completed task,
because "no branch matched" is exactly what a rendered frame cannot show --
and equally what a substring in the method's source cannot rule out.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from warlock.studio import main
from warlock.studio.state import AppState


class FakeCache:
    def __init__(self) -> None:
        self.jobs: list[dict[str, Any]] = []
        self.storage: dict[str, Any] = {}
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1


class FakeApp:
    """Enough of ``App`` for ``_on_task_done`` to run against.

    The bound methods it calls are recorded rather than performed: what is
    under test is which of them a given task key reaches, not what they then
    do -- each of those has its own test.
    """

    def __init__(self, *, pose_mode: bool = False, selected: str | None = None) -> None:
        self.toasts: list[tuple[str, str]] = []
        self.calls: list[str] = []
        state = AppState()
        state.selected = selected
        self.app_ctx = SimpleNamespace(
            state=state,
            cache=FakeCache(),
            toast=lambda message, level="info": self.toasts.append((message, level)),
            submit=lambda key, run: self.calls.append(f"submit:{key}") or True,
        )
        self.viewer = SimpleNamespace(
            pose_mode=pose_mode,
            path="model.glb",
            editor=SimpleNamespace(dirty=True),
        )

    # recorded, not performed
    def _refresh_rig_side_data(self) -> None:
        self.calls.append("_refresh_rig_side_data")

    def _reload_viewer(self) -> None:
        self.calls.append("_reload_viewer")

    def _request_storage(self) -> None:
        self.calls.append("_request_storage")

    def _sync_viewer(self) -> None:
        self.calls.append("_sync_viewer")

    def dispatch(self, key: str, result: Any = None) -> None:
        main.App._on_task_done(self, SimpleNamespace(key=key, result=result, ok=True))


# --- results that had no branch ----------------------------------------------


def test_the_trellis_log_result_is_stored_where_the_inspector_reads_it():
    """The one diagnostic for "the 3D engine stopped unexpectedly". The button
    submitted, the service answered, and nothing ever wrote ``trellis_log``."""
    from warlock.studio.panes import inspector

    app = FakeApp()
    app.dispatch("trellis-log", {"text": "CUDA error: out of memory"})

    assert app.app_ctx.state.preview["trellis_log"] == "CUDA error: out of memory"
    # And the reader still keys on the same name.
    assert 'preview.get("trellis_log")' in inspect.getsource(inspector._error)


def test_deleting_a_sheet_refreshes_the_list():
    """``sheet-del:`` does not start with ``sheet:``, and _sync_viewer's
    early-return meant nothing else refetched: a deleted sheet stayed on screen
    with live-looking buttons."""
    app = FakeApp()
    app.dispatch("sheet-del:aaaaaaaaaaaa")

    assert "_refresh_rig_side_data" in app.calls


def test_rendering_a_sheet_refreshes_the_list_and_the_job_rows():
    app = FakeApp()
    app.dispatch("sheet:aaaaaaaaaaaa")

    assert "_refresh_rig_side_data" in app.calls
    assert app.app_ctx.cache.invalidated == 1


def test_a_bulk_export_reports_where_it_went():
    """Single-artifact saves have always toasted; a zip or folder export
    finished with no visible outcome at all."""
    app = FakeApp()
    app.dispatch("export-obj:aaaaaaaaaaaa", "D:/out/barrel.zip")

    assert any("D:/out/barrel.zip" in message for message, _ in app.toasts)


def test_a_retarget_reloads_the_viewer_and_says_what_went_stale():
    app = FakeApp()
    app.dispatch("retarget:aaaaaaaaaaaa", {"stale": ["rig.glb", "rig.json"]})

    assert "_reload_viewer" in app.calls
    assert app.app_ctx.cache.invalidated == 1
    assert any("2 rig artifact(s)" in message for message, _ in app.toasts)


def test_a_retarget_with_nothing_stale_does_not_warn_about_rig_artifacts():
    app = FakeApp()
    app.dispatch("retarget:aaaaaaaaaaaa", {"stale": []})

    assert [m for m, _ in app.toasts] == ["Mesh rebuilt."]


def test_a_delete_remeasures_storage_but_a_rename_does_not():
    """The walk is expensive; only the keys that change bytes on disk pay."""
    deleted, renamed = FakeApp(), FakeApp()
    deleted.dispatch("delete:aaaaaaaaaaaa")
    renamed.dispatch("rename:aaaaaaaaaaaa")

    assert "_request_storage" in deleted.calls
    assert "_request_storage" not in renamed.calls
    assert deleted.app_ctx.cache.invalidated == renamed.app_ctx.cache.invalidated == 1


def test_side_data_for_a_job_that_is_no_longer_selected_is_dropped():
    """It arrives after the click moved on; showing it would offer another
    asset's poses against this one."""
    app = FakeApp(selected="bbbbbbbbbbbb")
    app.dispatch("poses:aaaaaaaaaaaa", {"poses": [{"id": "0" * 12}], "bones": ["root"]})

    assert "poses" not in app.app_ctx.state.preview


def test_side_data_for_the_selected_job_lands():
    app = FakeApp(selected="aaaaaaaaaaaa")
    app.dispatch("poses:aaaaaaaaaaaa", {"poses": [{"id": "0" * 12}], "bones": ["root"]})

    assert app.app_ctx.state.preview["bones"] == ["root"]


def test_reloading_the_viewer_defeats_the_same_path_shortcut():
    """model.glb is rewritten under its own name, so _sync_viewer's
    ``self.viewer.path == wanted`` short-circuit is exactly wrong here."""
    app = FakeApp()
    main.App._reload_viewer(app)

    assert app.viewer.path is None
    assert "_sync_viewer" in app.calls


# --- choices that were collected and dropped ---------------------------------


def _calls_to(module: Any, name: str) -> list[ast.Call]:
    """Every call in ``module`` whose callee ends in ``.name`` or is ``name``.

    Parsed rather than grepped: a substring count breaks on reformatting and on
    a legitimate third call site, while saying nothing about what the calls
    actually pass.
    """
    tree = ast.parse(inspect.getsource(module))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        got = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if got == name:
            out.append(node)
    return out


def test_rigging_an_existing_mesh_forwards_the_chosen_skeleton():
    """The skeleton combo applied only to rig-on-generate; the library's Rig
    action passed nothing, so the config default always won."""
    from warlock.studio.panes import library

    # create_rig is never called directly here -- it is handed to ctx.submit --
    # so the call to inspect is the submit that carries it.
    submits = [
        call
        for call in _calls_to(library, "submit")
        if any(
            isinstance(a, ast.Attribute) and a.attr == "create_rig" for a in call.args
        )
    ]
    assert submits, "no rig submission found in library.py"
    for call in submits:
        assert "template" in {kw.arg for kw in call.keywords}, (
            "a rig submission that forwards no template silently uses the "
            "config default, ignoring the 3D form's skeleton combo"
        )


def test_the_skeleton_comes_from_the_3d_form():
    from warlock.studio.panes import library
    from warlock.studio.state import AppState

    class Ctx:
        state = AppState()

    ctx = Ctx()
    assert library._skeleton(ctx) is None  # unset means "let the service decide"
    ctx.state.form_3d["rig_template"] = "quadruped"
    assert library._skeleton(ctx) == "quadruped"


def test_a_sheet_can_be_named():
    """The service validated, stored and displayed a name that no control could
    ever set."""
    from warlock.studio.panes import sheet_panel

    assert "sheet-name" in inspect.getsource(sheet_panel._controls)


def test_the_sheet_form_is_rebuilt_when_the_selection_moves():
    """Half the form is pose ids, and those belong to the rig they were fitted
    to -- carrying them across submits another job's poses."""
    from warlock.studio.panes import sheet_panel
    from warlock.studio.state import AppState

    class Ctx:
        state = AppState()
        sheet_options = {"defaults": {}}

    ctx = Ctx()
    first = sheet_panel._form(ctx, "aaaaaaaaaaaa")
    first["poses"].add("deadbeefcafe")
    second = sheet_panel._form(ctx, "bbbbbbbbbbbb")
    assert second["poses"] == set()
    assert sheet_panel._form(ctx, "bbbbbbbbbbbb") is second  # stable within a job


def test_the_sheet_sidecar_can_be_exported():
    """A PNG with no cell map is a grid an importer cannot address."""
    from warlock.studio.panes import sheet_panel

    assert "Save JSON" in inspect.getsource(sheet_panel._saved)
    assert "get_sheet" in inspect.getsource(sheet_panel._save_sidecar)


def test_the_composed_prompt_preview_retries_when_its_request_was_refused():
    """``submit`` refuses a key already in flight. Clearing the dirty flag
    regardless dropped the edit made during a slow first preview."""
    from warlock.studio.panes import settings_2d

    source = inspect.getsource(settings_2d._preview)
    assert "if ctx.submit(" in source
    assert source.index("if ctx.submit(") < source.index("state.preview_dirty_at = 0.0")


# --- caps and messages that disagreed with the service ------------------------


def test_the_profile_editor_uses_the_service_cap():
    from warlock.service import validation
    from warlock.studio.panes import profiles_panel

    # The cap argument must be the service's constant, not a literal that can
    # drift away from it. Asserted on the call rather than as "2000 does not
    # appear", which fails on any unrelated number in the function.
    tree = ast.parse(inspect.getsource(profiles_panel).lstrip())
    caps = [
        call.args[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_editor"
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "multiline"
    ]
    assert caps, "no multiline prompt field found in the profile editor"
    for cap in caps:
        assert isinstance(cap, ast.Attribute) and cap.attr == "MAX_PROMPT", (
            "a literal cap here means the refusal arrives at submit time, "
            "against a profile the user had already saved"
        )
    assert validation.MAX_PROMPT == 1000


def test_the_pose_panel_does_not_ask_for_a_rig_that_cannot_be_made():
    from warlock.studio.panes import pose_panel

    source = inspect.getsource(pose_panel.draw)
    assert "ctx.rigging_available" in source


def test_the_pose_dirty_flag_is_cleared_when_the_save_lands():
    """Not when the save is *submitted*: clearing it there lets the user walk
    away from a pose that was never written."""
    app = FakeApp(pose_mode=True)
    app.dispatch("pose-save:aaaaaaaaaaaa")

    assert app.viewer.editor.dirty is False
    assert "_refresh_rig_side_data" in app.calls


def test_a_pose_save_that_is_still_in_flight_leaves_the_flag_set():
    """The submit side of the same rule: ``_save`` must not clear it itself."""
    from warlock.studio.panes import pose_panel

    assert "dirty = False" not in inspect.getsource(pose_panel._save)


def test_a_pose_delete_does_not_clear_the_dirty_flag():
    """Only ``pose-save:`` proves anything reached disk; the other pose- keys
    share the prefix and must not be read as a save landing."""
    app = FakeApp(pose_mode=True)
    app.dispatch("pose-del:aaaaaaaaaaaa")

    assert app.viewer.editor.dirty is True
    assert "_refresh_rig_side_data" in app.calls


def test_a_reference_image_can_be_saved():
    """A promoted job copies input.png and then had no way to give it back."""
    from warlock.studio import widgets

    assert "input.png" in dict(widgets.ARTIFACTS)


# --- copying a recipe back into the form --------------------------------------


def test_copying_settings_fills_the_form_from_a_jobs_params():
    from warlock.studio.state import form_from_params

    form = form_from_params(
        {
            "prompt": "a barrel",
            "style_lora": "render3d",
            "lora_weight": 1,  # JSON round-trips a float as an int
            "seed": 42,
            "material": "wood",
        }
    )
    assert form["prompt"] == "a barrel"
    assert form["style_lora"] == "render3d"
    assert form["lora_weight"] == 1.0
    assert form["seed"] == 42
    assert form["material"] == "wood"


def test_copying_settings_cannot_smuggle_a_derived_value_into_the_form():
    """The form is the allowlist: anything the worker recorded about a finished
    job's artifacts must not come back as a submitted field."""
    from warlock.studio.state import form_from_params

    form = form_from_params(
        {
            "prompt": "a barrel",
            "composed_prompt": "a barrel, fantasy, 4k",
            "mesh_report": {"triangles": 1},
            "transform": {"scale": 2.0},
            "hand_edited": True,
        }
    )
    assert set(form) & {"composed_prompt", "mesh_report", "transform", "hand_edited"} == set()


def _copy_ctx():
    from types import SimpleNamespace

    from warlock.studio.state import default_form_2d

    return SimpleNamespace(
        state=SimpleNamespace(form_2d=default_form_2d(), mode="3d"),
        toast=lambda _text: None,
    )


def test_copying_a_tiles_settings_keeps_it_a_tile():
    """The one field params cannot carry: tile-ness lives in the stage.

    A form filled from params alone opens in Object mode, which for a texture
    means the pane offers a taxonomy the prompt compiler will discard and a
    second stage the job cannot have.
    """
    from warlock.studio.panes import library

    ctx = _copy_ctx()
    library._copy_settings(ctx, {"stage": "tile", "params": {"prompt": "cobblestone"}})
    assert ctx.state.form_2d["output"] == "tile"
    assert ctx.state.mode == "2d"


def test_copying_a_references_settings_makes_an_object():
    from warlock.studio.panes import library

    ctx = _copy_ctx()
    ctx.state.form_2d["output"] = "tile"
    library._copy_settings(ctx, {"stage": "reference", "params": {"prompt": "a barrel"}})
    assert ctx.state.form_2d["output"] == "reference"


def test_copying_settings_survives_a_junk_value():
    from warlock.studio.state import form_from_params

    form = form_from_params({"seed": "not a number", "prompt": "ok"})
    assert form["prompt"] == "ok"
    assert isinstance(form["seed"], int)


# --- the retarget control ------------------------------------------------------


def test_the_retarget_panel_offers_only_raw_without_gltfpack():
    from warlock.studio.panes import retarget_panel

    assert retarget_panel.TIERS[0][0] == "raw"
    assert {t[0] for t in retarget_panel.TIERS} == {
        "raw",
        "draft",
        "standard",
        "detailed",
        "custom",
    }


@pytest.mark.parametrize(
    "triangles,ok",
    [(50_000, True), (100, False), (500_000, False)],
)
def test_the_retarget_panel_states_the_services_own_range(triangles, ok):
    from warlock.studio.panes import retarget_panel

    problems = retarget_panel.validate(
        {"profile": "custom", "custom_triangles": triangles}
    )
    assert (problems == []) is ok


def test_a_named_tier_needs_no_custom_count():
    from warlock.studio.panes import retarget_panel

    assert retarget_panel.validate({"profile": "raw", "custom_triangles": 0}) == []


def test_the_retarget_panel_calls_the_service_function_that_had_no_caller():
    from warlock.studio.panes import retarget_panel

    source = inspect.getsource(retarget_panel._submit)
    assert "svc_jobs.optimize_job" in source
    assert "custom_triangles" in source


def test_an_unmeasured_reroll_attempt_is_not_called_a_refusal():
    """The attempts line has three states, not two: a measurement that never
    ran is not a verdict, and labelling it "refused" invents one."""
    from warlock.studio.panes import inspector

    assert inspector._attempt_verdict({"seed": 1, "ok": True, "reasons": []}) == "kept"
    assert inspector._attempt_verdict({"seed": 2, "ok": False, "reasons": ["x"]}) == "refused"
    assert (
        inspector._attempt_verdict({"seed": 3, "ok": True, "reasons": [], "measured": False})
        == "not measured"
    )
    # Rows written before the third key existed, and anything malformed, still
    # render: the inspector draws a finished job and must never be the thing
    # that fails it.
    assert inspector._attempt_verdict({}) == "refused"
    assert inspector._attempt_verdict(None) == "?"


# --- remesh is offered in two places, and they must agree --------------------


def test_a_finished_tile_is_not_offered_a_remesh():
    """The context menu's gate, which is the call site that was missed.

    A tile has no subject to reconstruct, so ``rerun_job`` refuses to remesh
    one. Right-click -> Remesh therefore reached the service, was refused and
    came back as an error toast -- the exact failure the ``rerollable`` check
    beside it exists to prevent.
    """
    from warlock.studio.panes import library

    tile = {"id": "a", "stage": "tile", "kind": "text", "files": ["input.png"]}
    assert library._remeshable(tile) is False


def test_a_finished_reference_is_still_offered_a_remesh():
    from warlock.studio.panes import library

    reference = {"id": "a", "stage": "reference", "kind": "text", "files": ["input.png"]}
    assert library._remeshable(reference) is True


def test_a_job_with_no_image_is_not_offered_a_remesh():
    from warlock.studio.panes import library

    assert library._remeshable({"id": "a", "stage": "model", "files": []}) is False


def test_the_retry_ladder_rerolls_a_tile_rather_than_remeshing_it():
    """The other call site, through the action it actually submits."""
    from warlock.studio.panes import library

    calls: list[dict] = []

    class Ctx:
        svc = object()

        def submit(self, key, fn, *args, **kwargs):
            calls.append({"key": key, "fn": fn, "kwargs": kwargs})

    job = {"id": "a", "stage": "tile", "kind": "text", "files": ["input.png"]}
    library.run_action(Ctx(), job, "retry")

    assert [c["kwargs"]["mode"] for c in calls] == ["reroll"]


def test_both_remesh_call_sites_go_through_the_one_predicate():
    """Stated once so it cannot be honoured in one place and not the other.

    The regression this pins is not a wrong answer, it is a *second* copy of
    the rule: the menu item and the retry ladder held the same expression, one
    of them learned about tiles and the other did not.
    """
    from warlock.studio.panes import library

    assert len(_calls_to(library, "_remeshable")) == 2
    source = inspect.getsource(library)
    assert source.count('"input.png" in') == 1, (
        "a second inline input.png gate has appeared; route it through "
        "_remeshable instead"
    )


# --- Build mode's panes ------------------------------------------------------


def test_the_properties_pane_never_lists_a_generator_by_name():
    """The registry is data precisely so the pane is not a chain of names."""
    from warlock.studio.build import primitives as bp
    from warlock.studio.panes import build_props

    source = inspect.getsource(build_props)
    for name in bp.GENERATORS:
        assert f'"{name}"' not in source, f"{name} is hardcoded in build_props"


def test_every_build_pane_gates_its_controls_on_saving():
    """The rule Inker had to learn: a save encodes the live document on a task
    thread, so a control that restructures it mid-encode writes a file
    describing a document that never existed."""
    from warlock.studio.panes import build_bridge, build_outliner, build_props, build_tools

    for pane in (build_tools, build_props, build_outliner, build_bridge):
        source = inspect.getsource(pane)
        assert "saving" in source, f"{pane.__name__} does not consult tab.saving"


def test_the_bridge_offers_both_output_paths_and_they_are_different_calls():
    """Two genuinely different things: the exact geometry, or a picture trellis
    reinterprets. A bridge that wired both to one call would look complete."""
    from warlock.studio.panes import build_bridge

    source = inspect.getsource(build_bridge)
    assert "export_asset" in source
    assert "send_to_3d" in source


def test_the_tools_pane_mirrors_through_ops_rather_than_negating_a_scale():
    """A negative node scale is one line and is how an inside-out asset ships:
    glTF readers disagree about whether it flips the winding."""
    from warlock.studio.panes import build_tools

    source = inspect.getsource(build_tools)
    assert "ops.mirror" in source
    assert "scale=-" not in source


def test_the_outliner_addresses_every_row_by_uid():
    """An index stops naming the thing it named the moment anything moves, and
    the outliner is the thing that moves them."""
    from warlock.studio.panes import build_outliner

    tree = ast.parse(inspect.getsource(build_outliner))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("set_props", "remove_object", "select")
    ]
    assert calls, "the outliner changes nothing at all"
    for call in calls:
        rendered = ast.unparse(call)
        assert "uid" in rendered, f"{rendered} does not address its object by uid"


# --- Build as the eighth mode ------------------------------------------------


def test_build_claims_its_own_task_prefix_on_both_paths():
    """A key nothing claims is a result delivered nowhere -- and on the failure
    path it is a tab left read-only forever."""
    source = inspect.getsource(main.App._collect_tasks)
    assert 'startswith("build-")' in source
    assert "build_mode.on_task_failed" in source
    assert "build_mode.on_task_done" in inspect.getsource(main.App._on_task_done)


def test_build_takes_first_refusal_on_the_keyboard():
    """Build owns the centre pane, so the global F/W/S bindings -- which act on
    the asset viewer -- must be shadowed rather than run underneath it."""
    source = inspect.getsource(main.App._shortcut)
    build_at = source.index("build_mode.handle_key")
    viewer_at = source.index("self.viewer.frame()")
    assert build_at < viewer_at
    # And it returns, so nothing below belongs to Build.
    assert "return" in source[build_at : source.index("inker_mode.handle_key")]


def test_the_quit_guard_asks_about_built_geometry_too():
    """One chain, nested: ConfirmQueue holds a single pending question, so
    three asked side by side would silently drop two."""
    source = inspect.getsource(main.App._request_quit)
    for guard in ("inker_mode.guard", "build_mode.guard", "pose_panel.guard"):
        assert guard in source
    assert source.index("inker_mode.guard") < source.index("build_mode.guard")
    assert source.index("build_mode.guard") < source.index("pose_panel.guard")


def test_a_dropped_glb_is_refused_in_build_mode():
    """Reading a GLB back into editable objects is not Build Phase 1, and a
    frozen one-object document would be a different feature wearing its name."""
    source = inspect.getsource(main.App._on_drop)
    assert "WBLK_SUFFIX" in source
    assert ".wblk" in source


def test_build_persists_its_recent_list_and_no_mode():
    source = inspect.getsource(main.App)
    assert "build_mode.persist" in source
    # The guard the whole app is under; restated here because Build is the
    # newest place that could have broken it.
    assert 'settings.set("mode"' not in source


def test_the_send_to_3d_render_carries_no_grid_gizmo_or_overlay():
    """trellis is being handed a *subject*. A grid line in the picture is a
    subject too, and it comes back as geometry nobody asked for."""
    source = inspect.getsource(main.App._render_build_reference)
    assert "show_grid=False" in source
    assert "overlays=[]" in source
    assert "flat=True" in source


def test_the_send_to_3d_render_happens_on_the_frame_thread():
    """It needs the GL context. Only the service call goes to a task thread,
    which is the shape inker_mode.send_to_3d already has."""
    source = inspect.getsource(main.App._build_send_to_3d)
    assert "submit" not in source
    assert "upload_bytes" in source
    assert "submit" in inspect.getsource(
        __import__("warlock.studio.panes.settings_3d", fromlist=["x"]).upload_bytes
    )


def test_both_upload_paths_read_the_same_form():
    """A form field honoured for a dropped file and ignored for a rendered one
    is the shape of bug this consolidation exists to prevent."""
    from warlock.studio.panes import settings_3d

    assert len(_calls_to(settings_3d, "_upload_kwargs")) == 2


def test_build_is_a_workspace_rather_than_a_single_pane():
    from warlock.studio import main as main_mod
    from warlock.studio import modes

    assert "build" in modes.WORKSPACE_MODES
    assert "build" not in main_mod._SINGLE_PANE_MODES
    assert "build" not in modes.VIEWPORT_MODES
    assert "build" in modes.WORK_MODES


def test_the_asset_viewer_never_sees_the_mouse_in_build_mode():
    """Both viewports would orbit on one drag otherwise."""
    source = inspect.getsource(main.App._events)
    build_at = source.index('ctx.state.mode == "build"')
    viewer_at = source.index("self.viewer.handle_event")
    assert build_at < viewer_at
