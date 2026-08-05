"""Review mode's controller: the verdict loop over a sweep's units.

Nothing here draws. What is asserted is the part a pane cannot be trusted to
get right on its own: that Accept writes exactly one verdict and moves on, that
Reject *waits* for a reason rather than recording a bare rejection, that Skip
writes nothing at all, and that the param/value a verdict is filed under comes
from the item record the run wrote rather than from the unit key -- the report
joins on those two, so a verdict filed under the wrong pair is invisible to it.

The mode-set membership is here too, because Review is the first mode that owns
its own centre pane *and* borrows the shared asset viewer: it must be a work
mode (it takes keys) and a workspace mode (it fills the window), and must not
be a viewport mode, or ``_sync_viewer`` would reload the selected library asset
over the sweep unit on screen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from warlock.bench import manifest as manifest_mod
from warlock.bench import runner as runner_mod
from warlock.bench import verdicts as verdicts_mod
from warlock.config import Config
from warlock.studio import review_mode

# --- the harness -------------------------------------------------------------


class FakeCtx:
    """Runs a submitted callable inline, so the test sees what the task thread
    would have done without needing one."""

    def __init__(self, config: Any, *, accept: bool = True) -> None:
        self.runtime = _Runtime(config)
        self.state = _AppState()
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


class _AppState:
    def __init__(self) -> None:
        self.review = None
        self.mode = "home"


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
    """``pygame.key.get_mods`` needs a video system; there is none in a test.

    Patched for every test here rather than only in the ones that press a key,
    so a new key test cannot fail for a reason unrelated to what it checks.
    """
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


@pytest.fixture(autouse=True)
def _no_dialogs(monkeypatch):
    """No native picker ever runs in a test: ``FakeCtx`` runs a submitted
    callable inline, and a modal picker on that path hangs the suite."""
    from warlock.studio import dialogs

    monkeypatch.setattr(dialogs, "open_file", lambda *a, **k: None)
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: None)


@pytest.fixture
def config(tmp_path):
    return Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i",
        bench_dir=tmp_path / "bench",
    )


def _sweep_run(config, name, units, *, job_params=None) -> Path:
    """A sweep run directory as ``sweep.run_sweep`` leaves one behind."""
    run_dir = Path(config.bench_dir) / "runs" / name
    run_dir.mkdir(parents=True)
    manifest_mod.write_manifest(
        run_dir,
        {"version": 1, "sweep": {"key": "x", "axes": [{"param": "lora_weight"}]}},
    )
    lines = []
    for unit in units:
        record = {
            "key": unit["key"],
            "param": unit.get("param"),
            "value": unit.get("value"),
            "status": unit.get("status", "done"),
        }
        lines.append(json.dumps(record))
        item_dir = run_dir / "items" / unit["key"]
        item_dir.mkdir(parents=True)
        (item_dir / "model.glb").write_bytes(b"glTF-not-really")
        (item_dir / "reference.png").write_bytes(b"png-not-really")
        (item_dir / "job.json").write_text(
            json.dumps({"id": unit["key"], "params": job_params or {}}), encoding="utf-8"
        )
    (run_dir / runner_mod.ITEMS_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def _units(*keys) -> list[dict[str, Any]]:
    return [
        {"key": key, "param": "lora_weight", "value": 0.6 + i * 0.3}
        for i, key in enumerate(keys)
    ]


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
    """It takes keys, it fills the window with its own three columns, and it
    borrows the shared viewer for a file ``_sync_viewer`` knows nothing about --
    listing it under VIEWPORT_MODES would have the selection reload the
    library's asset over the sweep unit on screen."""
    from warlock.studio import main, modes

    assert "review" in modes.KEYS
    assert "review" in modes.WORK_MODES
    assert "review" in modes.WORKSPACE_MODES
    assert "review" not in modes.VIEWPORT_MODES
    assert "review" not in main._SINGLE_PANE_MODES


def test_the_three_categories_still_partition_the_modes():
    """``_build_ui``'s dispatch ends in a bare else, so a mode in none of the
    three would silently draw Inker rather than fail."""
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
    """The build_mode.py rule: state and logic here, drawing in main.py, so
    every one of these tests runs without a GL context."""
    source = Path(review_mode.__file__).read_text("utf-8")
    assert "import imgui" not in source
    assert "from imgui" not in source
    assert "moderngl" not in source


# --- scanning ----------------------------------------------------------------


def test_a_scan_finds_the_sweep_runs_and_opens_one(config):
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", _units("a", "b"))

    state = _scanned(ctx)
    assert [run["label"] for run in state.runs] == ["20260101-sweep-x"]
    assert state.run_dir is not None and state.run_dir.name == "20260101-sweep-x"
    assert [unit["key"] for unit in state.units] == ["a", "b"]
    assert state.scanning is False


def test_a_scan_with_no_sweep_runs_leaves_an_empty_but_usable_state(config):
    ctx = FakeCtx(config)
    state = _scanned(ctx)

    assert state.runs == []
    assert state.units == []
    assert review_mode.current(state) is None
    # Every key path has to survive the empty case rather than raise.
    assert _press(ctx, "a") is False
    review_mode.record(ctx, "accept")


def test_the_scan_runs_off_the_frame_thread_under_one_claimable_key(config):
    """The app claims task results by prefix: a key without one is a result
    delivered nowhere."""
    ctx = FakeCtx(config)
    review_mode.scan(ctx)
    assert ctx.submitted == [review_mode.SCAN_KEY]
    assert review_mode.SCAN_KEY.startswith("review-")


def test_a_second_scan_while_one_is_in_flight_is_not_submitted(config):
    ctx = FakeCtx(config, accept=True)
    state = review_mode.ensure(ctx)
    state.scanning = True

    review_mode.scan(ctx)
    assert ctx.submitted == []


def test_a_refused_or_failed_scan_clears_the_scanning_flag(config):
    """``scanning`` gates every button and key, so leaving it set makes the
    mode permanently inert with no way back short of a restart."""
    refused = FakeCtx(config, accept=False)
    review_mode.scan(refused)
    assert review_mode.ensure(refused).scanning is False

    ctx = FakeCtx(config)
    review_mode.ensure(ctx).scanning = True
    review_mode.on_task_failed(ctx, _Done(review_mode.SCAN_KEY))
    assert ctx.state.review.scanning is False


def test_a_rescan_keeps_the_run_that_is_open(config):
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", _units("a", "b"))
    _sweep_run(config, "20260202-sweep-y", _units("c"))

    state = _scanned(ctx)
    other = next(run["dir"] for run in state.runs if run["label"] == "20260202-sweep-y")
    review_mode.open_run(ctx, other)
    assert state.run_dir == other

    _scanned(ctx)
    assert ctx.state.review.run_dir == other
    assert [unit["key"] for unit in ctx.state.review.units] == ["c"]


def test_a_scan_carries_the_verdicts_already_recorded(config):
    ctx = FakeCtx(config)
    run_dir = _sweep_run(config, "20260101-sweep-x", _units("a", "b"))
    verdicts_mod.append_verdict(
        run_dir, unit="a", source="human", verdict="accept", param="lora_weight", value=0.6
    )

    state = _scanned(ctx)
    assert state.units[0]["verdict"] == "accept"
    assert state.units[1]["verdict"] is None
    # And the run list says how much is left to do.
    assert state.runs[0]["todo"] == 1


def test_opening_a_run_starts_on_the_first_unit_with_no_verdict(config):
    """A review session is resumed far more often than it is started, and
    landing back on unit one every time is what makes resuming useless."""
    ctx = FakeCtx(config)
    run_dir = _sweep_run(config, "20260101-sweep-x", _units("a", "b", "c"))
    verdicts_mod.append_verdict(
        run_dir, unit="a", source="human", verdict="accept", param="lora_weight", value=0.6
    )

    state = _scanned(ctx)
    assert review_mode.current(state)["key"] == "b"


# --- verdicts ----------------------------------------------------------------


def test_accept_writes_one_verdict_and_advances(config):
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", _units("a", "b"))
    state = _scanned(ctx)

    assert _press(ctx, "a") is True

    written = verdicts_mod.read_verdicts(state.run_dir)
    assert len(written) == 1
    assert written[0]["unit"] == "a"
    assert written[0]["verdict"] == "accept"
    assert written[0]["source"] == review_mode.SOURCE
    assert review_mode.current(state)["key"] == "b"


def test_a_verdict_is_filed_under_the_param_and_value_the_run_recorded(config):
    """``report.aggregate`` groups on (param, str(value)); a verdict filed
    under the unit key instead is invisible to the findings table."""
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", _units("a"))
    state = _scanned(ctx)

    review_mode.record(ctx, "accept")
    written = verdicts_mod.read_verdicts(state.run_dir)[0]
    assert written["param"] == "lora_weight"
    assert written["value"] == 0.6


def test_reject_waits_for_a_reason_key(config):
    """R arms; nothing is written until a reason is picked, because a bare
    rejection tells the findings table nothing it can tally."""
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", _units("a", "b"))
    state = _scanned(ctx)

    assert _press(ctx, "r") is True
    assert state.pending_reject is True
    assert verdicts_mod.read_verdicts(state.run_dir) == []
    assert review_mode.current(state)["key"] == "a"

    assert _press(ctx, "2") is True
    written = verdicts_mod.read_verdicts(state.run_dir)
    assert len(written) == 1
    assert written[0]["verdict"] == "reject"
    assert written[0]["reasons"] == [verdicts_mod.REASONS[1]]
    assert state.pending_reject is False
    assert review_mode.current(state)["key"] == "b"


def test_a_reason_key_with_nothing_armed_writes_nothing(config):
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", _units("a"))
    state = _scanned(ctx)

    assert _press(ctx, "3") is False
    assert verdicts_mod.read_verdicts(state.run_dir) == []


def test_escape_disarms_a_pending_reject(config):
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", _units("a"))
    state = _scanned(ctx)

    _press(ctx, "r")
    assert _press(ctx, "ESCAPE") is True
    assert state.pending_reject is False
    assert verdicts_mod.read_verdicts(state.run_dir) == []


def test_every_reason_key_names_a_real_reason(config):
    assert len(review_mode.REASON_KEYS) == len(verdicts_mod.REASONS)
    assert set(review_mode.REASON_KEYS.values()) == set(verdicts_mod.REASONS)


def test_skip_writes_nothing_and_advances(config):
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", _units("a", "b"))
    state = _scanned(ctx)

    assert _press(ctx, "s") is True
    assert verdicts_mod.read_verdicts(state.run_dir) == []
    assert review_mode.current(state)["key"] == "b"


def test_recording_advances_past_units_that_already_have_a_verdict(config):
    """The advance is to the next thing to *do*, not the next row: a session
    that walks back onto its own finished work asks the same question twice."""
    ctx = FakeCtx(config)
    run_dir = _sweep_run(config, "20260101-sweep-x", _units("a", "b", "c"))
    verdicts_mod.append_verdict(
        run_dir, unit="b", source="human", verdict="accept", param="lora_weight", value=0.9
    )
    state = _scanned(ctx)
    review_mode.step(state, -1)  # back onto "a", which is still unverdicted
    assert review_mode.current(state)["key"] == "a"

    review_mode.record(ctx, "accept")
    assert review_mode.current(state)["key"] == "c"


def test_the_last_unit_stays_put_when_there_is_nothing_left_to_do(config):
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", _units("a"))
    state = _scanned(ctx)

    review_mode.record(ctx, "accept")
    assert review_mode.current(state)["key"] == "a"
    assert state.units[0]["verdict"] == "accept"
    assert state.runs[0]["todo"] == 0


def test_a_re_review_supersedes_rather_than_duplicating(config):
    """``verdicts.latest`` keys on (unit, source), so a changed mind is a
    second append and the newest one wins."""
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", _units("a"))
    state = _scanned(ctx)

    review_mode.record(ctx, "accept")
    review_mode.record(ctx, "reject", ("holes",))

    latest = verdicts_mod.latest(state.run_dir)
    assert latest[("a", review_mode.SOURCE)]["verdict"] == "reject"
    assert state.units[0]["verdict"] == "reject"


# --- navigation --------------------------------------------------------------


def test_the_arrows_navigate_and_clamp_at_both_ends(config):
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", _units("a", "b"))
    state = _scanned(ctx)

    assert _press(ctx, "LEFT") is True
    assert review_mode.current(state)["key"] == "a"
    _press(ctx, "RIGHT")
    assert review_mode.current(state)["key"] == "b"
    _press(ctx, "RIGHT")
    assert review_mode.current(state)["key"] == "b"


def test_navigating_disarms_a_pending_reject(config):
    """The armed state belongs to the unit that was on screen when R was
    pressed; carrying it to the next one rejects the wrong mesh."""
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", _units("a", "b"))
    state = _scanned(ctx)

    _press(ctx, "r")
    _press(ctx, "RIGHT")
    assert state.pending_reject is False


def test_no_key_does_anything_while_a_scan_is_in_flight(config):
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", _units("a"))
    state = _scanned(ctx)
    state.scanning = True

    _press(ctx, "a")
    assert verdicts_mod.read_verdicts(state.run_dir) == []


# --- what the pane reads -----------------------------------------------------


def test_the_archived_job_is_read_once_and_cached_on_the_unit(config):
    """A small read per navigation is fine on the frame thread; a read per
    frame is not."""
    ctx = FakeCtx(config)
    _sweep_run(
        config,
        "20260101-sweep-x",
        _units("a"),
        job_params={"mesh_report": {"triangles": 40000, "watertight": True}},
    )
    state = _scanned(ctx)
    unit = review_mode.current(state)
    assert unit["job"]["params"]["mesh_report"]["triangles"] == 40000

    # Deleting the file must not change the answer: it was read once.
    (unit["dir"] / "job.json").unlink()
    review_mode.load_job(unit)
    assert unit["job"] is not None


def test_a_unit_with_no_archived_job_reads_as_empty_rather_than_raising(config):
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", _units("a"))
    state = _scanned(ctx)
    unit = state.units[0]
    unit["job"] = None
    (unit["dir"] / "job.json").unlink()

    review_mode.load_job(unit)
    assert unit["job"] == {}
    assert review_mode.mesh_lines(unit) == []


def test_the_mesh_summary_is_read_off_the_archived_params(config):
    ctx = FakeCtx(config)
    _sweep_run(
        config,
        "20260101-sweep-x",
        _units("a"),
        job_params={
            "mesh_report": {"triangles": 40000, "watertight": False, "materials": 1},
            "mesh_audit": {"verdict": "holes"},
        },
    )
    state = _scanned(ctx)
    lines = review_mode.mesh_lines(review_mode.current(state))

    assert any("40" in line for line in lines)
    assert any("holes" in line for line in lines)


def test_the_paths_the_pane_draws_are_inside_the_units_own_directory(config):
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", _units("a"))
    state = _scanned(ctx)
    unit = review_mode.current(state)

    assert review_mode.model_path(unit) == unit["dir"] / "model.glb"
    assert review_mode.reference_path(unit) == unit["dir"] / "reference.png"


def test_the_reference_falls_back_to_the_uploaded_input(config):
    """A sweep unit copies whichever of the two the job had: a text job writes
    reference.png, an upload writes input.png and no reference at all."""
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", _units("a"))
    state = _scanned(ctx)
    unit = review_mode.current(state)
    (unit["dir"] / "reference.png").unlink()
    (unit["dir"] / "input.png").write_bytes(b"png-not-really")

    assert review_mode.reference_path(unit) == unit["dir"] / "input.png"


def test_a_missing_mesh_is_a_path_that_does_not_exist_not_a_crash(config):
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", [{"key": "a", "status": "error"}])
    state = _scanned(ctx)
    unit = review_mode.current(state)
    (unit["dir"] / "model.glb").unlink()

    assert review_mode.model_path(unit).exists() is False
    assert unit["status"] == "error"


# --- state -------------------------------------------------------------------


def test_the_mode_state_is_built_lazily(config):
    """``AppState`` deliberately knows nothing about it, and a session that
    never opens Review should not pay for it."""
    ctx = FakeCtx(config)
    assert ctx.state.review is None
    assert review_mode.ensure(ctx) is ctx.state.review


def test_review_persists_nothing_at_all(config):
    """Not the mode -- which ``test_no_mode_is_persisted_anywhere`` pins -- and
    not the open run either: a stored run directory would outlive the sweep it
    names, and a pruned bench dir would open on nothing with no way to say so.
    """
    ctx = FakeCtx(config)
    _sweep_run(config, "20260101-sweep-x", _units("a"))
    _scanned(ctx)
    review_mode.record(ctx, "accept")

    assert ctx.settings.store == {}
    source = Path(review_mode.__file__).read_text("utf-8")
    assert "ctx.settings" not in source


# --- the plumbing in main.py -------------------------------------------------


def test_review_claims_its_own_task_prefix_on_both_paths():
    """A key nothing claims is a result delivered nowhere -- and on the failure
    path it is a mode left inert forever, because ``scanning`` gates
    everything."""
    import inspect

    from warlock.studio import main

    collect = inspect.getsource(main.App._collect_tasks)
    assert 'startswith("review-")' in collect
    assert "review_mode.on_task_failed" in collect
    assert "review_mode.on_task_done" in inspect.getsource(main.App._on_task_done)


def test_review_takes_first_refusal_on_the_keyboard():
    """It owns the centre pane, so the global F/W/S bindings -- which act on
    the asset viewer -- must be shadowed rather than run underneath it, and the
    branch returns whether or not handle_key consumed the key."""
    import inspect

    from warlock.studio import main

    source = inspect.getsource(main.App._shortcut)
    review_at = source.index("review_mode.handle_key")
    assert review_at < source.index("self.viewer.frame()")
    assert "return" in source[review_at : source.index("inker_mode.handle_key")]


def test_the_workspace_loads_a_mesh_on_a_change_of_unit_not_every_frame():
    import inspect

    from warlock.studio import main

    source = inspect.getsource(main.App._review_viewport)
    assert "loaded_key" in source
    assert "load_model" not in source  # it is behind the guard, in _review_load
    assert "load_model" in inspect.getsource(main.App._review_load)


def test_arriving_in_review_scans_and_arriving_repeatedly_does_not():
    """Driven off the mode change rather than off "the list is empty", which
    would submit a walk of the bench directory every frame on a machine that
    has never run a sweep."""
    import inspect

    from warlock.studio import main

    source = inspect.getsource(main.App._build_ui)
    assert "review_mode.scan" in source
    assert 'self._last_mode and ctx.state.mode == "review"' in source
