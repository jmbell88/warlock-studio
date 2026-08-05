"""Review mode's controller: launching sweeps, and the verdict loop over them.

Nothing here draws. What is asserted is the part a pane cannot be trusted to
get right on its own: that Accept writes exactly one verdict and moves on, that
Reject *waits* for a reason rather than recording a bare rejection, that Skip
writes nothing at all, that a sweep launch is all-or-nothing, and that the
first bucket is the finished assets nobody has judged -- which is what makes
ordinary use feed the same findings a deliberate sweep does.

The mode-set membership is here too, because Review is the first mode that owns
its own centre pane *and* borrows the shared asset viewer: it must be a work
mode (it takes keys) and a workspace mode (it fills the window), and must not
be a viewport mode, or ``_sync_viewer`` would reload the selected library asset
over the sweep unit on screen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from warlock.service import verdicts as svc_verdicts
from warlock.studio import review_mode
from warlock.studio.state import AppState

# --- the harness -------------------------------------------------------------


class FakeCtx:
    """Runs a submitted callable inline, so the test sees what the task thread
    would have done without needing one."""

    def __init__(self, svc: Any, *, accept: bool = True) -> None:
        self.svc = svc
        self.runtime = _Runtime(svc.config)
        self.state = AppState()
        self.state.review = None
        self.settings = _Settings()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.accept = accept
        self.result: Any = None

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        if not self.accept:
            return False
        self.result = run(*args)
        return True

    def toast(self, message: str, kind: str = "info") -> None:
        self.toasts.append((message, kind))


class _Runtime:
    def __init__(self, config: Any) -> None:
        self.config = config


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _Done:
    def __init__(self, key: str, result: Any = None) -> None:
        self.key = key
        self.result = result


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    """``pygame.key.get_mods`` needs a video system; there is none in a test."""
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


@pytest.fixture
def ctx(svc):
    return FakeCtx(svc)


def _mesh(svc, name, **params):
    """A finished model job with a mesh and a reference on disk."""
    job_id = svc.store.create("image", name, params, stage="model", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"glTF-not-really")
    (job_dir / "reference.png").write_bytes(b"png-not-really")
    return job_id


def _sweep(svc, label="lora", n=2, **params):
    sweep_id = svc.store.create_sweep(label, "a chest", {"axes": []})
    ids = []
    for i in range(n):
        job_id = svc.store.create(
            "text", "a chest", params, stage="model", status="done",
            sweep_id=sweep_id, sweep_unit=f"unit{i}",
        )
        job_dir = svc.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "model.glb").write_bytes(b"glTF-not-really")
        (job_dir / "reference.png").write_bytes(b"png-not-really")
        ids.append(job_id)
    return sweep_id, ids


def _scanned(ctx) -> Any:
    """A whole scan: the submit, and the result coming back. ``scan`` only does
    the first half -- applying the result is ``on_task_done``'s job."""
    review_mode.scan(ctx)
    review_mode.on_task_done(ctx, _Done(review_mode.SCAN_KEY, ctx.result))
    return ctx.state.review


def _press(ctx, key_name: str) -> bool:
    import pygame

    key = getattr(pygame, f"K_{key_name}")
    return review_mode.handle_key(ctx, pygame.event.Event(pygame.KEYDOWN, key=key))


# --- the mode sets -----------------------------------------------------------


def test_review_is_a_work_mode_and_a_workspace_but_not_a_viewport_mode():
    from warlock.studio import main, modes

    assert "review" in modes.KEYS
    assert "review" in modes.WORK_MODES
    assert "review" in modes.WORKSPACE_MODES
    assert "review" not in modes.VIEWPORT_MODES
    assert "review" not in main._SINGLE_PANE_MODES


def test_the_three_categories_still_partition_the_modes():
    from warlock.studio import main, modes

    categories = [
        set(main._SINGLE_PANE_MODES),
        set(modes.VIEWPORT_MODES),
        set(modes.WORKSPACE_MODES),
    ]
    assert set().union(*categories) == set(modes.KEYS)
    for i, first in enumerate(categories):
        for second in categories[i + 1 :]:
            assert not (first & second)


def test_review_mode_imports_no_imgui():
    """The clay_mode.py rule: state and logic here, drawing in main.py, so
    every one of these tests runs without a GL context."""
    source = Path(review_mode.__file__).read_text("utf-8")
    assert "import imgui" not in source
    assert "from imgui" not in source
    assert "moderngl" not in source


# --- scanning ----------------------------------------------------------------


def test_a_scan_finds_the_sweeps_and_opens_the_first_bucket(ctx, svc):
    sweep_id, ids = _sweep(svc)
    state = _scanned(ctx)

    assert [s["id"] for s in state.sweeps] == [review_mode.RECENT_ID, sweep_id]
    assert state.sweep_id == review_mode.RECENT_ID
    assert state.scanning is False
    assert [u["job_id"] for u in state.sweeps[1]["units"]] == ids


def test_the_first_bucket_is_the_finished_assets_nobody_has_judged(ctx, svc):
    """Ordinary daily use feeds the same findings pool a sweep does."""
    plain = _mesh(svc, "a chest")
    judged = _mesh(svc, "a sword")
    svc_verdicts.record_verdict(svc, judged, verdict="accept")
    _sweep(svc)

    state = _scanned(ctx)
    recent = state.sweeps[0]
    assert recent["id"] == review_mode.RECENT_ID
    # Not the judged one, and not the sweep's units -- they are reviewed under
    # their own sweep.
    assert [u["job_id"] for u in recent["units"]] == [plain]


def test_a_scan_with_nothing_recorded_leaves_an_empty_but_usable_state(ctx):
    state = _scanned(ctx)

    assert [s["id"] for s in state.sweeps] == [review_mode.RECENT_ID]
    assert state.units == []
    assert review_mode.current(state) is None
    # Every key path has to survive the empty case rather than raise.
    assert _press(ctx, "a") is False
    review_mode.record(ctx, "accept")


def test_the_scan_runs_off_the_frame_thread_under_one_claimable_key(ctx):
    """The app claims task results by prefix: a key without one is a result
    delivered nowhere."""
    review_mode.scan(ctx)
    assert ctx.submitted == [review_mode.SCAN_KEY]
    assert review_mode.SCAN_KEY.startswith("review-")
    assert review_mode.DELETE_KEY.startswith("review-")
    assert review_mode.FINDINGS_KEY.startswith("review-")


def test_a_second_scan_while_one_is_in_flight_is_not_submitted(ctx):
    review_mode.ensure(ctx).scanning = True
    review_mode.scan(ctx)
    assert ctx.submitted == []


def test_a_refused_or_failed_scan_clears_the_scanning_flag(svc):
    """``scanning`` gates every button and key, so leaving it set makes the
    mode permanently inert with no way back short of a restart."""
    refused = FakeCtx(svc, accept=False)
    review_mode.scan(refused)
    assert review_mode.ensure(refused).scanning is False

    ctx = FakeCtx(svc)
    review_mode.ensure(ctx).scanning = True
    review_mode.on_task_failed(ctx, _Done(review_mode.SCAN_KEY))
    assert ctx.state.review.scanning is False


def test_a_rescan_keeps_the_sweep_that_is_open(ctx, svc):
    sweep_id, _ = _sweep(svc)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    assert state.sweep_id == sweep_id

    _scanned(ctx)
    assert ctx.state.review.sweep_id == sweep_id
    assert len(ctx.state.review.units) == 2


def test_a_scan_carries_the_verdicts_already_recorded(ctx, svc):
    sweep_id, ids = _sweep(svc)
    svc_verdicts.record_verdict(svc, ids[0], verdict="accept")

    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    assert state.units[0]["verdict"] == "accept"
    assert state.units[1]["verdict"] is None
    assert state.sweeps[1]["todo"] == 1


def test_opening_a_sweep_starts_on_the_first_unit_with_no_verdict(ctx, svc):
    """A review session is resumed far more often than it is started, and
    landing back on unit one every time is what makes resuming useless."""
    sweep_id, ids = _sweep(svc, n=3)
    svc_verdicts.record_verdict(svc, ids[0], verdict="accept")

    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    assert state.index == 1


# --- the verdict loop --------------------------------------------------------


def test_accept_writes_one_verdict_and_advances(ctx, svc):
    sweep_id, ids = _sweep(svc)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    assert _press(ctx, "a") is True

    recorded = svc.store.latest_verdicts()
    assert [(r["job_id"], r["verdict"]) for r in recorded] == [(ids[0], "accept")]
    assert state.index == 1
    # And the findings recompute is a task, not frame-thread work.
    assert review_mode.FINDINGS_KEY in ctx.submitted


def test_a_verdict_carries_the_jobs_config_vector(ctx, svc):
    sweep_id, _ = _sweep(svc, lora_weight=0.6)
    _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    review_mode.record(ctx, "accept")

    assert svc.store.latest_verdicts()[0]["vector"] == {
        "lora_weight": 0.6, "stage": "model"
    }


def test_reject_waits_for_a_reason_key(ctx, svc):
    sweep_id, _ = _sweep(svc)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    assert _press(ctx, "r") is True
    assert state.pending_reject is True
    assert svc.store.latest_verdicts() == []

    assert _press(ctx, "1") is True
    recorded = svc.store.latest_verdicts()
    assert recorded[0]["verdict"] == "reject"
    assert recorded[0]["reasons"] == [review_mode.REASON_KEYS["1"]]
    assert state.pending_reject is False


def test_a_reason_key_with_nothing_armed_writes_nothing(ctx, svc):
    sweep_id, _ = _sweep(svc)
    _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    assert _press(ctx, "1") is False
    assert svc.store.latest_verdicts() == []


def test_escape_disarms_a_pending_reject(ctx, svc):
    sweep_id, _ = _sweep(svc)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    _press(ctx, "r")

    assert _press(ctx, "ESCAPE") is True
    assert state.pending_reject is False
    assert svc.store.latest_verdicts() == []


def test_every_reason_key_names_a_real_reason():
    assert set(review_mode.REASON_KEYS.values()) == set(svc_verdicts.REASONS)
    assert list(review_mode.REASON_KEYS) == ["1", "2", "3", "4", "5"]


def test_skip_writes_nothing_and_advances(ctx, svc):
    sweep_id, _ = _sweep(svc)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    assert _press(ctx, "s") is True
    assert state.index == 1
    assert svc.store.latest_verdicts() == []


def test_recording_advances_past_units_that_already_have_a_verdict(ctx, svc):
    sweep_id, ids = _sweep(svc, n=3)
    svc_verdicts.record_verdict(svc, ids[1], verdict="accept")
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    assert state.index == 0
    review_mode.record(ctx, "accept")
    assert state.index == 2


def test_recording_wraps_to_work_left_behind_the_cursor(ctx, svc):
    sweep_id, _ = _sweep(svc, n=3)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    review_mode.step(state, 2)

    review_mode.record(ctx, "accept")
    assert state.index == 0


def test_the_last_unit_stays_put_when_there_is_nothing_left_to_do(ctx, svc):
    sweep_id, _ = _sweep(svc, n=1)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    review_mode.record(ctx, "accept")
    assert state.index == 0


def test_a_re_review_supersedes_rather_than_duplicating(ctx, svc):
    sweep_id, ids = _sweep(svc, n=1)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    review_mode.record(ctx, "accept")
    review_mode.record(ctx, "reject", ("holes",))

    latest = svc.store.latest_verdicts()
    assert len(latest) == 1
    assert latest[0]["verdict"] == "reject"
    assert state.units[0]["verdict"] == "reject"


def test_the_arrows_navigate_and_clamp_at_both_ends(ctx, svc):
    sweep_id, _ = _sweep(svc, n=2)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    assert _press(ctx, "LEFT") is True
    assert state.index == 0
    _press(ctx, "RIGHT")
    _press(ctx, "RIGHT")
    assert state.index == 1


def test_navigating_disarms_a_pending_reject(ctx, svc):
    sweep_id, _ = _sweep(svc, n=2)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    _press(ctx, "r")

    _press(ctx, "RIGHT")
    assert state.pending_reject is False


def test_no_key_does_anything_while_a_scan_is_in_flight(ctx, svc):
    sweep_id, _ = _sweep(svc)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    state.scanning = True

    for key in ("a", "r", "s", "1", "LEFT"):
        assert _press(ctx, key) is False
    assert svc.store.latest_verdicts() == []


# --- what the pane reads -----------------------------------------------------


def test_the_paths_the_pane_draws_are_inside_the_jobs_own_directory(ctx, svc):
    job_id = _mesh(svc, "a chest")
    state = _scanned(ctx)
    unit = review_mode.current(state)

    assert review_mode.model_path(unit) == svc.job_dir(job_id) / "model.glb"
    assert review_mode.reference_path(unit) == svc.job_dir(job_id) / "reference.png"


def test_the_reference_falls_back_to_the_uploaded_input(ctx, svc):
    job_id = _mesh(svc, "a chest")
    (svc.job_dir(job_id) / "reference.png").unlink()
    (svc.job_dir(job_id) / "input.png").write_bytes(b"png-not-really")

    state = _scanned(ctx)
    unit = review_mode.current(state)
    assert review_mode.reference_path(unit).name == "input.png"


def test_a_missing_mesh_is_a_path_that_does_not_exist_not_a_crash(ctx, svc):
    job_id = _mesh(svc, "a chest")
    (svc.job_dir(job_id) / "model.glb").unlink()

    state = _scanned(ctx)
    unit = review_mode.current(state)
    assert review_mode.model_path(unit).exists() is False


def test_the_thumbnail_id_is_the_job_id(ctx, svc):
    """ThumbnailCache keys on (id, mtime). A job id is globally unique, which
    is what the old run-qualified unit key was working around."""
    job_id = _mesh(svc, "a chest")
    state = _scanned(ctx)
    assert review_mode.cache_id(review_mode.current(state)) == f"review:{job_id}"


def test_the_mesh_summary_is_read_off_the_jobs_own_params(ctx, svc):
    """No archived job.json any more: the row *is* the record, so there is
    nothing to cache and nothing to go stale."""
    _mesh(
        svc, "a chest",
        mesh_report={"triangles": 12000, "watertight": True, "materials": 2},
        mesh_audit={"verdict": "clean"},
    )
    state = _scanned(ctx)
    lines = review_mode.mesh_lines(review_mode.current(state))
    assert lines == [
        "12,000 triangles", "watertight: yes", "2 material(s)", "silhouette: clean"
    ]


# --- launching ---------------------------------------------------------------


def test_launching_queues_a_sweep_and_opens_it(ctx, svc):
    form = review_mode.ensure(ctx).form
    form.prompt = "a wooden chest"
    form.seeds = "1, 2"
    form.axes = [{"param": "trellis_band", "values": "4, 8"}]

    assert review_mode.preview_units(ctx.state.review) == 6
    assert review_mode.launch(ctx) is True

    sweeps = svc.store.list_sweeps()
    assert len(sweeps) == 1
    assert sweeps[0]["units"] == 6
    assert ctx.state.review.sweep_id == sweeps[0]["id"]
    # Values typed as text reach the validator as the type it demands.
    units = svc.store.sweep_jobs(sweeps[0]["id"])
    assert {u["params"].get("trellis_band") for u in units} == {None, 4, 8}


def test_a_malformed_form_toasts_rather_than_queueing(ctx, svc):
    form = review_mode.ensure(ctx).form
    form.prompt = ""
    form.axes = [{"param": "trellis_band", "values": "8"}]

    assert review_mode.launch(ctx) is False
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
    assert svc.store.list_sweeps() == []


def test_a_refused_unit_refuses_the_whole_sweep_with_its_name(ctx, svc):
    form = review_mode.ensure(ctx).form
    form.prompt = "a chest"
    form.seeds = "1"
    form.axes = [{"param": "trellis_band", "values": "8, 999"}]

    assert review_mode.launch(ctx) is False
    assert "trellis_band=999 s1" in ctx.toasts[-1][0]
    assert svc.store.list_sweeps() == []
    assert svc.store.list(50) == []


def test_an_axis_with_no_values_is_refused(ctx):
    form = review_mode.ensure(ctx).form
    form.prompt = "a chest"
    form.axes = [{"param": "trellis_band", "values": ""}]

    assert review_mode.preview_units(ctx.state.review) == -1
    assert review_mode.launch(ctx) is False


def test_the_baseline_is_captured_from_the_forms_the_user_already_tuned(ctx):
    ctx.state.form_2d["genre"] = "fantasy"
    ctx.state.form_2d["style_lora"] = "some-lora"
    ctx.state.form_2d["lora_weight"] = 0.75
    ctx.state.form_3d["platform"] = "pc"
    ctx.state.form_3d["reference_prep"] = True

    base = review_mode.capture_base(ctx)
    assert base["genre"] == "fantasy"
    assert base["lora_weight"] == 0.75
    # The 3D pane's platform is the geometry resolution and wins.
    assert base["platform"] == "pc"
    assert base["reference_prep"] is True
    # The prompt is not a setting: it belongs to the sweep, not to its base.
    assert "prompt" not in base


def test_seeds_are_parsed_and_a_typo_is_a_toast_not_a_crash(ctx):
    assert review_mode.parse_seeds("1, 2,3") == (1, 2, 3)
    with pytest.raises(ValueError):
        review_mode.parse_seeds("")
    with pytest.raises(ValueError):
        review_mode.parse_seeds("nope")


# --- deleting ----------------------------------------------------------------


def test_deleting_a_sweep_goes_through_the_task_runner(ctx, svc):
    sweep_id, ids = _sweep(svc)
    _scanned(ctx)

    assert review_mode.delete(ctx, sweep_id) is True
    assert review_mode.DELETE_KEY in ctx.submitted
    assert svc.store.get(ids[0]) is None
    assert svc.store.list_sweeps() == []


def test_the_recent_bucket_cannot_be_deleted(ctx, svc):
    _mesh(svc, "a chest")
    _scanned(ctx)
    assert review_mode.delete(ctx, review_mode.RECENT_ID) is False


def test_a_finished_delete_says_what_it_kept(ctx, svc):
    sweep_id, ids = _sweep(svc, n=1)
    _scanned(ctx)
    svc_verdicts.record_verdict(svc, ids[0], verdict="accept")
    review_mode.delete(ctx, sweep_id)
    review_mode.on_task_done(ctx, _Done(review_mode.DELETE_KEY, {"deleted": 1}))

    assert "Verdicts and findings kept" in ctx.toasts[-1][0]
    # The whole reason the vector is snapshotted onto the verdict row.
    assert len(svc.store.latest_verdicts()) == 1


# --- housekeeping ------------------------------------------------------------


def test_the_mode_state_is_built_lazily(ctx):
    assert ctx.state.review is None
    assert review_mode.ensure(ctx) is ctx.state.review


def test_review_persists_nothing_at_all(ctx, svc):
    """Not the mode -- which ``test_no_mode_is_persisted_anywhere`` pins -- and
    not the open sweep either: a stored sweep id would outlive the sweep it
    names, and a deleted one would open on nothing with no way to say so."""
    _mesh(svc, "a chest")
    _scanned(ctx)
    review_mode.record(ctx, "accept")

    assert ctx.settings.store == {}
    source = Path(review_mode.__file__).read_text("utf-8")
    assert "ctx.settings" not in source


# --- the plumbing in main.py -------------------------------------------------


def test_review_claims_its_own_task_prefix_on_both_paths():
    import inspect

    from warlock.studio import main

    collect = inspect.getsource(main.App._collect_tasks)
    assert 'startswith("review-")' in collect
    assert "review_mode.on_task_failed" in collect
    assert "review_mode.on_task_done" in inspect.getsource(main.App._on_task_done)


def test_review_takes_first_refusal_on_the_keyboard():
    import inspect

    from warlock.studio import main

    source = inspect.getsource(main.App._shortcut)
    review_at = source.index("review_mode.handle_key")
    assert review_at < source.index("self.viewer.frame()")
    assert "return" in source[review_at : source.index("inker_mode.handle_key")]


def test_the_workspace_decides_what_to_load_the_way_sync_viewer_does():
    """Against ``viewer.path``, never against a remembered unit key. No
    ``loaded_key`` anywhere, because a second copy of "what is loaded" is how
    the two answers drift apart in the first place."""
    import inspect

    from warlock.studio import main

    source = inspect.getsource(main.App._review_load)
    assert "self.viewer.path == wanted" in source
    assert "loaded_key" not in inspect.getsource(main.App._review_viewport)
    assert "loaded_key" not in Path(review_mode.__file__).read_text("utf-8")


def test_the_workspace_asks_whether_there_is_a_unit_before_asking_the_viewer():
    import inspect

    from warlock.studio import main

    source = inspect.getsource(main.App._review_viewport)
    assert source.index("if unit is None") < source.index("self.viewer.has_model")


def test_the_workspace_refuses_to_load_over_the_pose_editor():
    import inspect

    from warlock.studio import main

    source = inspect.getsource(main.App._review_viewport)
    pose_at = source.index("self.viewer.pose_mode")
    assert pose_at < source.index("self._review_load")
    assert "return" in source[pose_at : source.index("self._review_load")]


def test_the_workspace_clears_a_stale_compare_split_before_drawing():
    import inspect

    from warlock.studio import main

    source = inspect.getsource(main.App._review_viewport)
    comparing_at = source.index("ctx.state.comparing")
    assert comparing_at < source.index("review_mode.current(state)")
    guarded = source[comparing_at : source.index("review_mode.current(state)")]
    assert "ctx.state.comparing = None" in guarded
    assert "self.viewer.exit_compare()" in guarded


def test_arriving_in_review_scans_and_arriving_repeatedly_does_not():
    import inspect

    from warlock.studio import main

    source = inspect.getsource(main.App._build_ui)
    assert "review_mode.scan" in source
    assert 'self._last_mode and ctx.state.mode == "review"' in source


# --- loading the mesh (the shared viewer) ------------------------------------


class _FakeViewer:
    """Only the four members ``_review_load`` touches. No GL, no imgui."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self.model: Any = None
        self.pose_mode = False
        self.loads: list[Path] = []
        self.cleared = 0

    @property
    def has_model(self) -> bool:
        return self.model is not None

    def load_model(self, path: Path) -> None:
        self.loads.append(Path(path))
        self.path = Path(path)
        self.model = object()

    def clear(self) -> None:
        self.cleared += 1
        self.model = None
        self.path = None


class _FakeApp:
    """``_review_load`` unbound, over a fake viewer -- the whole decision it
    makes is which path to hand the viewer, which needs no window."""

    from warlock.studio import main as _main

    _review_load = _main.App._review_load

    def __init__(self, ctx: Any) -> None:
        self.viewer = _FakeViewer()
        self.app_ctx = ctx


def test_switching_sweeps_reloads_the_unit(ctx, svc):
    first, _ = _sweep(svc, "one", n=1)
    second, _ = _sweep(svc, "two", n=1)
    state = _scanned(ctx)
    app = _FakeApp(ctx)

    review_mode.open_sweep(ctx, first)
    app._review_load(review_mode.current(state), review_mode)
    review_mode.open_sweep(ctx, second)
    app._review_load(review_mode.current(state), review_mode)

    assert len(app.viewer.loads) == 2
    assert app.viewer.loads[0] != app.viewer.loads[1]


def test_coming_back_from_3d_reloads_the_unit_that_is_still_selected(ctx, svc):
    """3D loads a library asset into this same viewer. A marker keyed on the
    unit still matched on the way back, so Review drew the library's asset
    under its own verdict buttons."""
    _mesh(svc, "a chest")
    state = _scanned(ctx)
    app = _FakeApp(ctx)
    unit = review_mode.current(state)

    app._review_load(unit, review_mode)
    app.viewer.load_model(Path(svc.config.data_dir) / "some-job" / "model.glb")
    app._review_load(unit, review_mode)

    assert app.viewer.path == review_mode.model_path(unit)


def test_showing_the_same_unit_again_does_not_reload_it(ctx, svc):
    """A comparison that never matched would decode a GLB every frame."""
    _mesh(svc, "a chest")
    state = _scanned(ctx)
    app = _FakeApp(ctx)

    for _ in range(3):
        app._review_load(review_mode.current(state), review_mode)
    assert len(app.viewer.loads) == 1


def test_a_unit_with_no_mesh_is_attempted_once_not_every_frame(ctx, svc):
    """``viewer.path`` is set even with nothing to show, or an errored unit
    re-clears the viewer sixty times a second."""
    job_id = _mesh(svc, "a chest")
    (svc.job_dir(job_id) / "model.glb").unlink()
    state = _scanned(ctx)
    unit = review_mode.current(state)
    app = _FakeApp(ctx)

    for _ in range(3):
        app._review_load(unit, review_mode)
    assert app.viewer.cleared == 1
    assert app.viewer.has_model is False


def test_a_mesh_that_will_not_open_is_reported_once(ctx, svc):
    """A toast per frame is not a report, it is a wall."""
    _mesh(svc, "a chest")
    state = _scanned(ctx)
    unit = review_mode.current(state)
    app = _FakeApp(ctx)

    def boom(path):
        raise ValueError("not a GLB")

    app.viewer.load_model = boom
    for _ in range(3):
        app._review_load(unit, review_mode)

    assert len(ctx.toasts) == 1
    assert ctx.toasts[0][1] == "error"
