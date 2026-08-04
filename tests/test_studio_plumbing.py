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
