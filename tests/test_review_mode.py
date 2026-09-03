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
        from warlock.service import system as svc_system

        self.svc = svc
        self.runtime = _Runtime(svc.config)
        # The real catalog, because ``axis_options`` reads it and a fake that
        # answered ``{}`` would let every param quietly fall back to a text
        # field -- which is exactly the state these tests exist to detect.
        self.guidance = svc_system.guidance_catalog(svc)
        self.state = AppState()
        self.state.review = None
        self.settings = _Settings()
        self.submitted: list[str] = []
        self.tags: list[Any] = []
        self.toasts: list[tuple[str, str]] = []
        self.accept = accept
        self.result: Any = None

    def submit(self, key: str, run: Any, *args: Any, tag: Any = None, **kw: Any) -> bool:
        # ``tag`` and ``**kw`` because the real ``Ctx.submit`` takes them, and a
        # fake that refuses one fails on the call site rather than on the
        # behaviour -- the same lesson ``toast`` learned above.
        self.submitted.append(key)
        self.tags.append(tag)
        if not self.accept:
            return False
        self.result = run(*args, **kw)
        return True

    def toast(self, message: str, kind: str = "info", **extra: Any) -> None:
        # ``**extra`` for ``action``/``action_arg``: the real Ctx has taken them
        # since toasts grew an "Open the log" button, and a fake that refuses
        # them fails on the call site rather than on the behaviour.
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
    def __init__(
        self, key: str, result: Any = None, tag: Any = None, error: Any = None
    ) -> None:
        self.key = key
        self.result = result
        # The real ``Done`` carries one, and the cleanup toast names the sweep
        # off it -- a fake without the field would make that branch untestable.
        self.tag = tag
        self.error = error

    @property
    def ok(self) -> bool:
        return self.error is None


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


def _press(ctx, key_name: str, mod: int = 0) -> bool:
    """One keydown. ``mod`` is always supplied, because ``handle_key`` reads it
    off the event -- ``pygame.key.name`` returns "1" whether or not Ctrl is
    held, so the modifier is the only thing that says which of three things a
    digit means."""
    import pygame

    key = getattr(pygame, f"K_{key_name}")
    return review_mode.handle_key(
        ctx, pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod)
    )


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
    svc_verdicts.record_verdict(svc, judged, grade=3)
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
    review_mode.record(ctx, 3)


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
    svc_verdicts.record_verdict(svc, ids[0], grade=3)

    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    assert state.units[0]["verdict"] == "accept"
    assert state.units[1]["verdict"] is None
    assert state.sweeps[1]["todo"] == 1


def test_opening_a_sweep_starts_on_the_first_unit_with_no_verdict(ctx, svc):
    """A review session is resumed far more often than it is started, and
    landing back on unit one every time is what makes resuming useless."""
    sweep_id, ids = _sweep(svc, n=3)
    svc_verdicts.record_verdict(svc, ids[0], grade=3)

    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    assert state.index == 1


# --- the verdict loop --------------------------------------------------------

# Filled in at import time from pygame rather than written as literals: the
# modifier bits are pygame's, and a hand-copied constant is a second spelling
# of a number this test suite exists to check the first one against.
import pygame as _pygame  # noqa: E402

_CTRL = _pygame.KMOD_LCTRL
_SHIFT = _pygame.KMOD_LSHIFT


def test_a_grade_key_writes_one_verdict_and_advances(ctx, svc):
    sweep_id, ids = _sweep(svc)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    assert _press(ctx, "4") is True

    recorded = svc.store.latest_verdicts()
    assert [(r["job_id"], r["grade"], r["verdict"]) for r in recorded] == [
        (ids[0], 4, "accept")
    ]
    assert state.index == 1
    # And the findings recompute is asked for rather than done here: it reads
    # every verdict and writes a file, neither of which belongs on the frame
    # thread. ``pump_findings`` is what turns the request into the task.
    assert ctx.state.findings_dirty is True
    assert review_mode.FINDINGS_KEY not in ctx.submitted

    review_mode.pump_findings(ctx)

    assert review_mode.FINDINGS_KEY in ctx.submitted
    assert ctx.state.findings_dirty is False


def test_a_burst_of_verdicts_is_not_swallowed_by_the_one_in_flight(ctx, svc):
    """The regression: ``submit`` refuses a key already in flight and nothing
    re-armed, so five presses in a second recomputed over the set as it stood
    at the first and dropped the rest until the *next* verdict."""
    sweep_id, _ = _sweep(svc)
    _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    ctx.accept = False  # stand in for "a recompute is already running"
    review_mode.record(ctx, 3)
    review_mode.pump_findings(ctx)
    assert ctx.state.findings_dirty is True

    ctx.accept = True
    review_mode.pump_findings(ctx)
    assert ctx.state.findings_dirty is False
    assert ctx.submitted.count(review_mode.FINDINGS_KEY) == 2


def test_a_verdict_carries_the_jobs_config_vector(ctx, svc):
    sweep_id, _ = _sweep(svc, lora_weight=0.6)
    _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    review_mode.record(ctx, 3)

    assert svc.store.latest_verdicts()[0]["vector"] == {
        "lora_weight": 0.6, "stage": "model"
    }


def test_r_arms_the_sign_and_the_next_digit_is_negative(ctx, svc):
    sweep_id, _ = _sweep(svc)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    assert _press(ctx, "r") is True
    assert state.pending_negative is True
    assert svc.store.latest_verdicts() == []

    assert _press(ctx, "2") is True
    recorded = svc.store.latest_verdicts()
    assert recorded[0]["grade"] == -2
    assert recorded[0]["verdict"] == "reject"
    # The arm belongs to the unit that was on screen when it was pressed.
    assert state.pending_negative is False


def test_a_bare_digit_is_positive(ctx, svc):
    sweep_id, _ = _sweep(svc)
    _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    assert _press(ctx, "5") is True
    assert svc.store.latest_verdicts()[0]["grade"] == 5


def test_zero_is_its_own_key_and_a_real_answer(ctx, svc):
    """No sign to arm, and "no opinion either way" is an answer rather than a
    refusal to give one -- so it records, and the row it records is *answered*,
    which is what stops ``advance`` from offering it again."""
    sweep_id, _ = _sweep(svc)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    assert _press(ctx, "0") is True
    row = svc.store.latest_verdicts()[0]
    assert row["grade"] == 0
    assert row["verdict"] == "reject"  # below the usable cut, as +0 must be
    assert state.units[0]["verdict"] is not None


def test_the_whole_scale_is_reachable_from_the_keyboard(ctx, svc):
    sweep_id, _ = _sweep(svc, n=10)
    _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    filed = []
    for magnitude in range(1, 6):
        for negative in (False, True):
            if negative:
                _press(ctx, "r")
            _press(ctx, str(magnitude))
            filed.append(-magnitude if negative else magnitude)

    recorded = sorted(r["grade"] for r in svc.store.latest_verdicts())
    assert recorded == sorted(filed)
    assert set(recorded) == set(range(-5, 6)) - {0}


def test_escape_disarms_the_sign_and_drops_the_staged_tags(ctx, svc):
    sweep_id, _ = _sweep(svc)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    _press(ctx, "r")
    _press(ctx, "1", mod=_CTRL)

    assert _press(ctx, "ESCAPE") is True
    assert state.pending_negative is False
    assert state.pending_tags == []
    assert svc.store.latest_verdicts() == []


def test_every_tag_key_names_a_real_tag():
    assert set(review_mode.GOOD_TAG_KEYS.values()) == set(svc_verdicts.GOOD_TAGS)
    assert set(review_mode.BAD_TAG_KEYS.values()) == set(svc_verdicts.BAD_TAGS)
    assert list(review_mode.GOOD_TAG_KEYS) == ["1", "2", "3", "4", "5"]
    assert list(review_mode.BAD_TAG_KEYS) == ["1", "2", "3", "4", "5"]
    assert list(review_mode.GRADE_KEYS.values()) == [1, 2, 3, 4, 5]


def test_the_grade_scale_is_derived_from_vectors_everywhere():
    """``vectors.py`` owns GRADE_MIN/GRADE_MAX; the button row and the grade
    keyboard must be derived from it, so widening the scale there moves the UI
    with it rather than leaving it drawing last week's buttons."""
    from warlock import vectors
    from warlock.studio import widgets

    assert tuple(range(vectors.GRADE_MIN, vectors.GRADE_MAX + 1)) == widgets.GRADES
    assert {str(i): i for i in range(1, vectors.GRADE_MAX + 1)} == review_mode.GRADE_KEYS


def test_the_tag_rows_and_the_tag_keyboard_share_one_order():
    """The positional contract: the digit that presses a tag button is the
    digit that names its tag. Both sides must iterate the vocabulary tuples
    themselves -- a hand-ordered row would silently disagree with Ctrl/Shift+1-5.
    """
    import inspect

    from warlock import vectors
    from warlock.studio import widgets

    assert list(review_mode.GOOD_TAG_KEYS.values()) == list(vectors.GOOD_TAGS)
    assert list(review_mode.BAD_TAG_KEYS.values()) == list(vectors.BAD_TAGS)
    source = inspect.getsource(widgets.tag_toggles)
    # The widget iterates the vocabulary objects, not a re-spelling of them,
    # and draws Good before Bad -- the order the modifiers read in.
    assert "verdicts_mod.GOOD_TAGS" in source
    assert "verdicts_mod.BAD_TAGS" in source
    assert source.index("GOOD_TAGS") < source.index("BAD_TAGS")


def test_tags_are_toggled_by_modifier_and_ride_the_next_grade(ctx, svc):
    sweep_id, _ = _sweep(svc)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    # Ctrl is good, Shift is bad, and the digit is positional in both.
    assert _press(ctx, "1", mod=_CTRL) is True
    assert _press(ctx, "1", mod=_SHIFT) is True
    assert state.pending_tags == [svc_verdicts.GOOD_TAGS[0], svc_verdicts.BAD_TAGS[0]]
    assert svc.store.latest_verdicts() == []

    assert _press(ctx, "4") is True
    row = svc.store.latest_verdicts()[0]
    assert row["grade"] == 4
    assert row["reasons"] == [svc_verdicts.GOOD_TAGS[0], svc_verdicts.BAD_TAGS[0]]
    assert state.units[0]["tags"] == row["reasons"]
    # Staged state belongs to the unit that is now behind us.
    assert state.pending_tags == []


def test_a_tag_key_pressed_twice_removes_it(ctx, svc):
    sweep_id, _ = _sweep(svc)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    _press(ctx, "3", mod=_SHIFT)
    assert state.pending_tags == [svc_verdicts.BAD_TAGS[2]]
    _press(ctx, "3", mod=_SHIFT)
    assert state.pending_tags == []


def test_a_good_tag_is_legal_on_a_negative_grade(ctx, svc):
    """Tags stopped being reasons a reviewer rejected. A mesh can have a clean
    shape and still be unusable, and that pair is the finding."""
    sweep_id, _ = _sweep(svc)
    _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    _press(ctx, "1", mod=_CTRL)
    _press(ctx, "r")
    _press(ctx, "4")

    row = svc.store.latest_verdicts()[0]
    assert row["grade"] == -4
    assert row["reasons"] == [svc_verdicts.GOOD_TAGS[0]]


def test_a_modified_digit_never_files_a_grade(ctx, svc):
    """Tags are checked before grades, or Ctrl+1 would file a +1 and stage
    nothing at all."""
    sweep_id, _ = _sweep(svc)
    _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    _press(ctx, "1", mod=_CTRL)
    _press(ctx, "2", mod=_SHIFT)
    _press(ctx, "0", mod=_CTRL)
    assert svc.store.latest_verdicts() == []


def test_a_records_nothing_at_all(ctx, svc):
    """The old Accept key. Deliberately not remapped onto +3: silently filing
    the mildest usable grade for somebody reaching for the old key is a wrong
    number rather than a missing one."""
    sweep_id, _ = _sweep(svc)
    _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    assert _press(ctx, "a") is False
    assert svc.store.latest_verdicts() == []


def test_moving_clears_the_staged_tags_as_well_as_the_sign(ctx, svc):
    """Both belong to the unit that was on screen. A tag surviving a step is
    filed against a mesh nobody was looking at when they chose it."""
    sweep_id, _ = _sweep(svc, n=3)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    for move in (
        lambda: review_mode.step(state, 1),
        lambda: review_mode.advance(state),
        lambda: review_mode.open_sweep(ctx, sweep_id),
    ):
        _press(ctx, "r")
        _press(ctx, "2", mod=_SHIFT)
        assert state.pending_negative is True
        assert state.pending_tags
        move()
        assert state.pending_negative is False
        assert state.pending_tags == []


def test_grade_text_is_signed_except_at_zero():
    assert review_mode.grade_text(4) == "+4"
    assert review_mode.grade_text(-2) == "-2"
    assert review_mode.grade_text(0) == "0"
    assert review_mode.grade_text(None) == ""
    # bool is an int, and True would otherwise render as +1.
    assert review_mode.grade_text(True) == ""


def test_skip_writes_nothing_and_advances(ctx, svc):
    sweep_id, _ = _sweep(svc)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    assert _press(ctx, "s") is True
    assert state.index == 1
    assert svc.store.latest_verdicts() == []


def test_recording_advances_past_units_that_already_have_a_verdict(ctx, svc):
    sweep_id, ids = _sweep(svc, n=3)
    svc_verdicts.record_verdict(svc, ids[1], grade=3)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    assert state.index == 0
    review_mode.record(ctx, 3)
    assert state.index == 2


def test_recording_wraps_to_work_left_behind_the_cursor(ctx, svc):
    sweep_id, _ = _sweep(svc, n=3)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    review_mode.step(state, 2)

    review_mode.record(ctx, 3)
    assert state.index == 0


def test_the_last_unit_stays_put_when_there_is_nothing_left_to_do(ctx, svc):
    sweep_id, _ = _sweep(svc, n=1)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    review_mode.record(ctx, 3)
    assert state.index == 0


def test_a_re_review_supersedes_rather_than_duplicating(ctx, svc):
    """Two units, not one, and that is now load-bearing: judging the *last*
    unit of a sweep triggers the automatic cleanup, which removes the job rows
    a re-grade would have to name. The supersede rule under test is about the
    verdict table and is unchanged; this only keeps the sweep unfinished long
    enough to exercise it."""
    sweep_id, ids = _sweep(svc, n=2)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    review_mode.record(ctx, 3)
    review_mode.step(state, -state.index)
    review_mode.record(ctx, -3, ("holes",))

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


def test_navigating_disarms_the_pending_sign(ctx, svc):
    sweep_id, _ = _sweep(svc, n=2)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    _press(ctx, "r")

    _press(ctx, "RIGHT")
    assert state.pending_negative is False


def test_no_key_does_anything_while_a_scan_is_in_flight(ctx, svc):
    sweep_id, _ = _sweep(svc)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    state.scanning = True

    for key in ("r", "s", "0", "1", "LEFT"):
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
    # ``mesh_audit`` as the worker actually writes it. This used to pass
    # ``{"verdict": "clean"}`` and assert the line it produced, which pinned a
    # branch nothing has ever fed: ``queue._audit_mesh`` stores worst, mean,
    # faces and resolution, so the reader was dead from the day it was typed
    # (P120, and the same mistake ``report["verdict"]`` made in the badge).
    _mesh(
        svc, "a chest",
        mesh_report={"triangles": 12000, "watertight": True, "materials": 2},
        mesh_audit={"worst": 0.0304, "mean": 0.01, "faces": 12000, "resolution": 1024},
    )
    state = _scanned(ctx)
    lines = review_mode.mesh_lines(review_mode.current(state))
    assert lines == [
        "12,000 triangles",
        "watertight: yes",
        "2 material(s)",
        "see-through at worst view: 3.0%",
    ]


# --- blind review ------------------------------------------------------------
#
# The 2026-08-07 review that produced the bg_removal signal was unblinded and
# single-reviewer, and the pane shows a unit's arm in two places (the unit list
# and the verdict header). A confirm run has to hide both -- and the *order*,
# because ``sweeps.expand`` enqueues the baseline first and then one unit per
# axis value, so position alone names the arm in a two-arm sweep.


def test_a_units_presented_name_is_its_arm_when_the_review_is_open(ctx, svc):
    sweep_id, _ = _sweep(svc, n=2)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    assert [review_mode.label(state, u) for u in state.units] == ["unit0", "unit1"]


def test_a_blind_review_shows_no_units_arm_anywhere(ctx, svc):
    sweep_id, ids = _sweep(svc, n=4)
    state = _scanned(ctx)
    review_mode.set_blind(ctx, True)
    review_mode.open_sweep(ctx, sweep_id)

    shown = [review_mode.label(state, u) for u in state.units]
    assert all("unit" not in name for name in shown)
    # Still one name per unit, and distinct, or a reviewer cannot say which of
    # four they are looking at.
    assert len(set(shown)) == 4
    assert all(any(name.endswith(job_id[:6]) for job_id in ids) for name in shown)


def test_the_blind_order_does_not_depend_on_the_order_units_were_queued(ctx, svc):
    """``expand`` puts the baseline first, so queue order names the arm."""
    units = [{"job_id": f"job{i:02x}"} for i in range(8)]

    assert review_mode.blind_order(units) == review_mode.blind_order(units[::-1])


def test_the_blind_order_is_the_same_in_every_process(ctx, svc):
    """A stable digest, not ``hash()``, which is salted per process: a reviewer
    resuming tomorrow would get a different order, and the order is what makes
    "#3f2a1b" mean the same unit as it did yesterday.

    The expected order is recomputed here rather than pasted, so the assertion
    is about *which* digest is used and not about a literal nobody can check.
    """
    import hashlib

    units = [{"job_id": f"job{i:02x}"} for i in range(8)]
    expected = sorted(units, key=lambda u: hashlib.sha1(u["job_id"].encode()).digest())

    assert review_mode.blind_order(units) == expected
    # And it genuinely is a reordering of this fixture, not a no-op sort.
    assert review_mode.blind_order(units) != units


def test_turning_blinding_on_reorders_the_open_sweep(ctx, svc):
    sweep_id, _ = _sweep(svc, n=8)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    queued = [u["job_id"] for u in state.units]

    review_mode.set_blind(ctx, True)

    assert [u["job_id"] for u in state.units] != queued
    assert sorted(u["job_id"] for u in state.units) == sorted(queued)
    # And back: the toggle is not one-way.
    review_mode.set_blind(ctx, False)
    assert [u["job_id"] for u in state.units] == queued


def test_a_rescan_keeps_the_blind_order_it_was_showing(ctx, svc):
    sweep_id, _ = _sweep(svc, n=8)
    state = _scanned(ctx)
    review_mode.set_blind(ctx, True)
    review_mode.open_sweep(ctx, sweep_id)
    shown = [u["job_id"] for u in state.units]

    _scanned(ctx)

    assert [u["job_id"] for u in state.units] == shown


def test_blinding_does_not_hide_the_verdicts_already_recorded(ctx, svc):
    """What is hidden is the arm, not the reviewer's own answers -- the unit
    list's marks are how a session is resumed."""
    sweep_id, ids = _sweep(svc, n=4)
    svc_verdicts.record_verdict(svc, ids[0], grade=3)
    state = _scanned(ctx)
    review_mode.set_blind(ctx, True)
    review_mode.open_sweep(ctx, sweep_id)

    recorded = {u["job_id"]: u["verdict"] for u in state.units}
    assert recorded[ids[0]] == "accept"


def test_blinding_is_not_persisted(ctx, svc):
    """``ReviewState`` persists nothing, and a stored blind flag would be the
    first thing to reach for -- a review resumed unblinded without saying so
    is worse than one that starts unblinded every time."""
    _sweep(svc, n=2)
    _scanned(ctx)
    review_mode.set_blind(ctx, True)

    assert ctx.settings.store == {}
    assert review_mode.ReviewState().blind is False


# --- the labelling pass ------------------------------------------------------
#
# The quality-judge programme's UI half. A grid of images with two keys, in
# Review beside the
# verdict loop, because the judge is meant to improve as the corpus is reviewed
# -- the analogue of a vision system's teach mode. Four rules from the plan are
# asserted here and every one of them is a bug that has already happened once
# somewhere else in this app.


def _image_job(svc, stage="model", status="done", prompt="a rogue"):
    job_id = svc.store.create("image", prompt, {}, stage=stage, status=status)
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "reference.png").write_bytes(b"png-not-really")
    return job_id


def _labels_open(ctx, stage="blank"):
    """A whole open: the submit, and the listing coming back. ``open_labels``
    only does the first half -- applying the result is ``on_task_done``'s job,
    exactly as it is for a scan."""
    review_mode.open_labels(ctx, stage)
    review_mode.on_task_done(ctx, _Done(review_mode.LABELS_KEY, ctx.result))
    return ctx.state.review.labels


def test_a_labelling_pass_lists_the_images_that_question_has_not_reached(ctx, svc):
    first, second = _image_job(svc), _image_job(svc)
    state = _scanned(ctx)

    _labels_open(ctx)

    assert state.labels is not None
    assert {row["job_id"] for row in state.labels.rows} == {first, second}
    assert state.labels.stage == "blank"


def test_a_and_r_label_with_no_reason_step(ctx, svc):
    """Five reason classes is far more than a first corpus can support, and
    reasons are a mesh-stage concept: what a blank probe learns is one bit."""
    job_id = _image_job(svc)
    _scanned(ctx)
    _labels_open(ctx)

    assert _press(ctx, "a") is True

    row = svc.store.latest_verdicts()[0]
    assert (row["job_id"], row["verdict"], row["stage"]) == (job_id, "accept", "blank")
    assert row["reasons"] == []


def test_labelling_advances_and_a_labelled_image_is_marked_not_removed(ctx, svc):
    """Removed would renumber the grid under the cursor mid-pass, which at the
    rate this is meant to be worked through is how the wrong image gets judged."""
    ids = [_image_job(svc) for _ in range(3)]
    _scanned(ctx)
    _labels_open(ctx)
    order = [row["job_id"] for row in ctx.state.review.labels.rows]

    _press(ctx, "a")

    labels = ctx.state.review.labels
    assert [row["job_id"] for row in labels.rows] == order
    assert labels.rows[0]["verdict"] == "accept"
    assert labels.index == 1
    assert set(order) == set(ids)


def test_labelling_asks_for_a_retrain_through_a_flag_never_a_submit(ctx, svc):
    """``TaskRunner.submit`` refuses a key already in flight and nothing re-arms
    it, so a burst of labels used to train once on the set as it stood at the
    first press and drop the rest. It is the ``findings_dirty`` bug exactly, and
    the fix is the same: mark, and let the frame loop pump it."""
    _image_job(svc)
    _scanned(ctx)
    _labels_open(ctx)

    _press(ctx, "a")

    assert ctx.state.judge_dirty == "blank"
    assert review_mode.TRAIN_KEY not in ctx.submitted


def test_the_pump_submits_once_and_only_clears_the_flag_when_accepted(ctx, svc):
    ctx.state.judge_dirty = "blank"

    review_mode.pump_judge(ctx)
    assert ctx.submitted.count(review_mode.TRAIN_KEY) == 1
    assert ctx.state.judge_dirty is None

    # And a refused submit leaves the request standing, for the next frame.
    ctx.state.judge_dirty = "blank"
    ctx.accept = False
    review_mode.pump_judge(ctx)
    assert ctx.state.judge_dirty == "blank"


def test_thumbnails_are_uploaded_one_per_frame(ctx, svc):
    """``viewer/sheet.StripRender``'s lesson, larger: a draw plus a synchronous
    upload sixteen times in one frame is a visible freeze, and this grid is a
    hundred cells."""
    for _ in range(5):
        _image_job(svc)
    _scanned(ctx)
    _labels_open(ctx)
    labels = ctx.state.review.labels

    assert review_mode.next_thumbnail(labels) is not None
    assert labels.uploaded == 1
    for _ in range(4):
        review_mode.next_thumbnail(labels)
    assert labels.uploaded == 5
    # Nothing left to do, and asking again is free rather than an error.
    assert review_mode.next_thumbnail(labels) is None


def test_the_review_list_sorts_by_the_judges_score_and_never_filters_by_it(ctx, svc):
    """The filter-bubble guard, structurally: if a judge hid what it disliked,
    its mistakes would become invisible and nobody would learn it was wrong.
    Sorting shows the same set in a more useful order."""
    units = [
        {"job_id": "a", "verdict": None, "score": 0.1},
        {"job_id": "b", "verdict": None, "score": None},
        {"job_id": "c", "verdict": None, "score": 0.9},
    ]

    ordered = review_mode.by_score(units)

    assert [u["job_id"] for u in ordered] == ["c", "a", "b"]
    assert len(ordered) == len(units)


def test_labelling_keys_do_nothing_when_no_pass_is_open(ctx, svc):
    """The verdict loop owns the keyboard the rest of the time, and it reads a
    digit as a grade rather than as a label."""
    _mesh(svc, "a chest")
    _scanned(ctx)

    _press(ctx, "3")

    row = svc.store.latest_verdicts()[0]
    assert row["stage"] == "model"
    assert row["grade"] == 3


def test_closing_a_labelling_pass_returns_to_the_verdict_loop(ctx, svc):
    _image_job(svc)
    _mesh(svc, "a chest")
    _scanned(ctx)
    _labels_open(ctx)

    review_mode.close_labels(ctx)

    assert ctx.state.review.labels is None
    assert _press(ctx, "3") is True
    assert svc.store.latest_verdicts()[0]["stage"] == "model"


# --- launching ---------------------------------------------------------------


def test_launching_queues_a_sweep_and_opens_it(ctx, svc):
    form = review_mode.ensure(ctx).form
    form.prompt = "a wooden chest"
    form.seeds = "1, 2"
    form.axes = [{"param": "trellis_band", "values": "4, 8"}]

    assert review_mode.preview_units(ctx.state.review) == 6
    assert review_mode.launch(ctx) is True
    # The insert batch is a task (T2): the form is locked until it lands, and
    # the selection moves only when it does.
    assert ctx.state.review.form.submitting is True
    assert ctx.submitted[-1] == review_mode.LAUNCH_KEY
    review_mode.on_task_done(ctx, _Done(review_mode.LAUNCH_KEY, ctx.result))
    assert ctx.state.review.form.submitting is False

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
    ctx.state.form_2d["style_lora"] = "some-lora"
    ctx.state.form_2d["lora_weight"] = 0.75
    ctx.state.form_3d["platform"] = "pc"
    ctx.state.form_3d["reference_prep"] = True

    base = review_mode.capture_base(ctx)
    assert base["style_lora"] == "some-lora"
    assert base["lora_weight"] == 0.75
    # platform is the 3D pane's geometry-resolution select.
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
    svc_verdicts.record_verdict(svc, ids[0], grade=3)
    review_mode.delete(ctx, sweep_id)
    review_mode.on_task_done(ctx, _Done(review_mode.DELETE_KEY, {"deleted": 1}))

    assert "Verdicts and findings kept" in ctx.toasts[-1][0]
    # The whole reason the vector is snapshotted onto the verdict row.
    assert len(svc.store.latest_verdicts()) == 1


# --- removing sweeps that are finished with -----------------------------------


def test_removable_ids_skips_the_recent_bucket_and_anything_outstanding(ctx, svc):
    """The recent bucket's units are ordinary library rows: prune's job."""
    _mesh(svc, "a chest")
    sweep_id, ids = _sweep(svc, n=2)
    state = _scanned(ctx)

    # Nothing judged yet, so nothing is removable.
    assert review_mode.removable_ids(state) == []

    for job_id in ids:
        svc_verdicts.record_verdict(svc, job_id, grade=-3)
    state = _scanned(ctx)
    assert review_mode.removable_ids(state) == [sweep_id]
    assert review_mode.RECENT_ID not in review_mode.removable_ids(state)


def test_a_unit_that_errored_does_not_hold_its_sweep_back(ctx, svc):
    """``removable`` and ``todo`` are different questions, deliberately.

    An errored unit can never be graded, so it owes no verdict -- but it keeps
    ``todo`` above zero for ever, which is how a failed sweep would become
    permanently unremovable.
    """
    sweep_id, ids = _sweep(svc, n=2)
    svc_verdicts.record_verdict(svc, ids[0], grade=-3)
    svc.store.set_status(ids[1], "error")
    state = _scanned(ctx)

    entry = next(s for s in state.sweeps if s["id"] == sweep_id)
    assert entry["todo"] == 1, "it is still unjudged, and the count says so"
    assert entry["removable"] is True
    assert review_mode.removable_ids(state) == [sweep_id]


def test_a_blank_label_alone_does_not_make_a_unit_look_mesh_judged(ctx, svc):
    """The ``stage=`` that was missing from ``_collect``.

    ``verdicts_for`` with no stage means *every* stage, and Review is a
    one-stage caller -- so a unit labelled at the blank stage but never
    mesh-graded read as judged, dropped out of ``todo``, and could trigger the
    automatic cleanup on a sweep nobody had graded.
    """
    sweep_id, ids = _sweep(svc, n=1)
    # An image label is binary and takes a verdict, where a mesh grade takes a
    # number -- which is the whole reason the two stages must not be conflated.
    svc_verdicts.record_verdict(svc, ids[0], verdict="accept", stage="blank")
    state = _scanned(ctx)

    entry = next(s for s in state.sweeps if s["id"] == sweep_id)
    assert entry["todo"] == 1
    assert entry["removable"] is False
    assert review_mode.removable_ids(state) == []


def test_the_removal_plan_uses_blind_labels_under_blinding(ctx, svc):
    """Naming the arms of a blinded pass in a confirm un-blinds them."""
    sweep_id, ids = _sweep(svc, label="lora weight", n=1)
    svc_verdicts.record_verdict(svc, ids[0], grade=-3)
    state = _scanned(ctx)

    plan = review_mode.removal_plan(state, [sweep_id])
    assert plan["labels"] == ["lora weight"]
    assert plan["sweeps"] == 1 and plan["units"] == 1

    state.blind = True
    blinded = review_mode.removal_plan(state, [sweep_id])
    assert blinded["labels"] == [f"#{sweep_id[:6]}"]
    assert "lora weight" not in blinded["labels"]


def test_removing_reviewed_sweeps_goes_through_the_task_runner(ctx, svc):
    sweep_id, ids = _sweep(svc, n=1)
    svc_verdicts.record_verdict(svc, ids[0], grade=-3)
    state = _scanned(ctx)

    assert review_mode.remove_reviewed(ctx, review_mode.removable_ids(state)) is True
    assert review_mode.REMOVE_KEY in ctx.submitted
    assert review_mode.REMOVE_KEY.startswith("review-"), "or it is delivered nowhere"
    assert svc.store.list_sweeps() == []
    # The whole point: the analytics, not the keepsake.
    assert len(svc.store.latest_verdicts()) == 1


def test_removing_nothing_submits_nothing(ctx, svc):
    _scanned(ctx)
    assert review_mode.remove_reviewed(ctx, []) is False
    assert review_mode.remove_reviewed(ctx, [review_mode.RECENT_ID]) is False


def test_a_finished_removal_counts_the_rows_not_just_the_folders(ctx, svc):
    """"8 asset folders" says nothing about the thing the user asked to be rid
    of, which is the rows cluttering the list."""
    _scanned(ctx)
    review_mode.on_task_done(
        ctx,
        _Done(review_mode.REMOVE_KEY, {"sweeps": 3, "deleted": 8}),
    )
    said = ctx.toasts[-1][0]
    assert "Removed" in said and "3 sweep(s)" in said and "8 asset folder(s)" in said


def test_a_removal_that_kept_units_points_at_the_tick(ctx, svc):
    """The old sentence pointed at the library because no other route existed.

    There is one now, and it is the thing that actually changes the outcome --
    so an invitation to press again would still be a lie, but an invitation to
    tick the box is not.
    """
    _scanned(ctx)
    review_mode.on_task_done(
        ctx, _Done(review_mode.DELETE_KEY, {"deleted": 1, "kept": 2})
    )
    said = ctx.toasts[-1][0]
    assert "kept 2" in said
    assert "accepted or labelled" in said
    assert "Delete again" not in said


def test_a_failed_removal_says_that_nothing_went(ctx, svc):
    """The service validates the whole set first, so this is true and the user
    is about to press the button again."""
    _scanned(ctx)
    review_mode.on_task_failed(ctx, _Done(review_mode.REMOVE_KEY, None))
    assert "Nothing was deleted" in ctx.toasts[-1][0]


def test_the_retention_override_reaches_cleanup_sweep(ctx, svc):
    """Ticked, a per-sweep delete takes the units retention would have kept."""
    sweep_id, ids = _sweep(svc, n=2)
    svc_verdicts.record_verdict(svc, ids[0], grade=3)  # an accept: retained
    svc_verdicts.record_verdict(svc, ids[1], grade=-3)
    _scanned(ctx)

    review_mode.delete(ctx, sweep_id)
    assert svc.store.get(ids[0]) is not None, "the default keeps it"

    review_mode.delete(ctx, sweep_id, drop_retained=True)
    assert svc.store.get(ids[0]) is None
    assert svc.store.list_sweeps() == []
    assert len(svc.store.latest_verdicts()) == 2


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
    review_mode.record(ctx, 3)

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
        self.pending: Path | None = None
        self.model: Any = None
        self.pose_mode = False
        self.loads: list[Path] = []
        self.cleared = 0

    @property
    def has_model(self) -> bool:
        return self.model is not None

    def parse_model(self, path: Path) -> Any:
        """The task-thread half; ``loads`` counts parses, which is what a
        decode-per-frame regression would inflate."""
        self.loads.append(Path(path))
        return ("parsed", Path(path))

    def adopt_model(self, model: Any, path: Path) -> None:
        self.pending = None
        self.path = Path(path)
        self.model = model

    def load_model(self, path: Path) -> None:
        self.adopt_model(self.parse_model(path), path)

    def clear(self) -> None:
        self.cleared += 1
        self.model = None
        self.path = None
        self.pending = None


class _FakeApp:
    """``_review_load`` unbound, over a fake viewer -- the whole decision it
    makes is which path to hand the viewer, which needs no window. ``show``
    is one frame plus the task landing: the parse is submitted from the draw
    and adopted when it returns (T2), and the fake ctx runs it inline."""

    from warlock.studio import main as _main

    _review_load = _main.App._review_load
    _adopt_review_model = _main.App._adopt_review_model

    def __init__(self, ctx: Any) -> None:
        self.viewer = _FakeViewer()
        self.app_ctx = ctx

    def show(self, unit: Any) -> None:
        before = len(self.app_ctx.submitted)
        self._review_load(unit, review_mode)
        if len(self.app_ctx.submitted) > before:
            key = self.app_ctx.submitted[-1]
            assert key == self._main.REVIEW_MESH_KEY
            self._adopt_review_model(_Done(key, self.app_ctx.result, tag=self.app_ctx.tags[-1]))


def test_switching_sweeps_reloads_the_unit(ctx, svc):
    first, _ = _sweep(svc, "one", n=1)
    second, _ = _sweep(svc, "two", n=1)
    state = _scanned(ctx)
    app = _FakeApp(ctx)

    review_mode.open_sweep(ctx, first)
    app.show(review_mode.current(state))
    review_mode.open_sweep(ctx, second)
    app.show(review_mode.current(state))

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

    app.show(unit)
    app.viewer.load_model(Path(svc.config.data_dir) / "some-job" / "model.glb")
    app.show(unit)

    assert app.viewer.path == review_mode.model_path(unit)


def test_showing_the_same_unit_again_does_not_reload_it(ctx, svc):
    """A comparison that never matched would decode a GLB every frame."""
    _mesh(svc, "a chest")
    state = _scanned(ctx)
    app = _FakeApp(ctx)

    for _ in range(3):
        app.show(review_mode.current(state))
    assert len(app.viewer.loads) == 1


def test_a_unit_whose_parse_is_in_flight_is_not_submitted_again(ctx, svc):
    """Between the submit and the landing the draw runs every frame; the
    ``pending`` marker is what keeps those frames from re-parsing."""
    _mesh(svc, "a chest")
    state = _scanned(ctx)
    app = _FakeApp(ctx)
    unit = review_mode.current(state)

    for _ in range(3):
        app._review_load(unit, review_mode)
    assert len(app.viewer.loads) == 1
    assert app.viewer.pending == review_mode.model_path(unit)
    assert app.viewer.has_model is False


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

    app._review_load(unit, review_mode)
    key = ctx.submitted[-1]
    app._adopt_review_model(_Done(key, tag=ctx.tags[-1], error=ValueError("not a GLB")))
    for _ in range(3):
        app._review_load(unit, review_mode)

    assert len(ctx.toasts) == 1
    assert ctx.toasts[0][1] == "error"
    assert len(app.viewer.loads) == 1, "tried once"


# --- the judge, wired in -----------------------------------------------------
#
# The helper's ``score_jobs`` was reachable from nothing and pinned by a test
# that called it on a literal list, which is false confidence: the sort worked
# and no unit ever carried a score. These are about the wiring.


def test_opening_a_sweep_asks_for_scores_rather_than_computing_them(ctx, svc):
    """A DINOv2 pass per unit is a task, never a frame -- and a flag, never a
    direct submit, for ``findings_dirty``'s reason."""
    _sweep(svc)
    state = _scanned(ctx)
    ctx.submitted.clear()

    review_mode.open_sweep(ctx, state.sweeps[-1]["id"])

    assert ctx.state.review_scores_dirty is True
    assert review_mode.SCORE_KEY not in ctx.submitted


def test_the_score_request_is_only_cleared_once_a_submit_is_accepted(svc):
    """The ``pump_findings`` bug, which this is the third instance of: a refused
    submit that cleared the flag dropped the request with nothing to re-arm it,
    and the request that matters is the last one."""
    ctx = FakeCtx(svc)
    _sweep(svc)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, state.sweeps[-1]["id"])

    ctx.accept = False
    review_mode.pump_scores(ctx)
    assert ctx.state.review_scores_dirty is True

    ctx.accept = True
    review_mode.pump_scores(ctx)
    assert ctx.state.review_scores_dirty is False


def test_with_no_probe_the_pump_asks_once_and_stops(ctx, svc):
    """``score_jobs`` answers ``{}`` when there is nothing fitted, and every
    unit is marked as asked -- otherwise the pump requests the same rows on
    every frame for the rest of the session."""
    _sweep(svc)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, state.sweeps[-1]["id"])

    review_mode.pump_scores(ctx)
    review_mode.on_task_done(ctx, _Done(review_mode.SCORE_KEY, ctx.result))

    assert ctx.result == {}
    assert all(unit["score"] is None for unit in state.units)
    assert review_mode.unscored(state.units) == []


def test_scores_merge_onto_the_open_units_without_reordering_them(ctx, svc):
    """The ``LabelPass`` lesson: a list that resorts under the cursor is how the
    wrong thing gets judged. Order is applied when a sweep is opened, once."""
    _sweep(svc, n=3)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, state.sweeps[-1]["id"])
    before = [unit["job_id"] for unit in state.units]

    state.score_request = list(before)
    review_mode.adopt_scores(
        state, {before[0]: 0.1, before[1]: 0.9, before[2]: 0.5}
    )

    assert [unit["job_id"] for unit in state.units] == before
    assert [unit["score"] for unit in state.units] == [0.1, 0.9, 0.5]


def test_a_score_written_here_is_visible_on_the_sweep_entry_too(ctx, svc):
    """The unit dicts are shared with the ``sweeps`` list on purpose -- which is
    also why ``clear_scores`` has to sweep both."""
    _sweep(svc, n=2)
    state = _scanned(ctx)
    entry = state.sweeps[-1]
    review_mode.open_sweep(ctx, entry["id"])
    state.score_request = [state.units[0]["job_id"]]
    review_mode.adopt_scores(state, {state.units[0]["job_id"]: 0.75})

    assert any(unit.get("score") == 0.75 for unit in entry["units"])

    review_mode.clear_scores(state)
    assert all("score" not in unit for unit in entry["units"])


def test_with_no_scores_the_order_is_character_for_character_what_it_was(ctx, svc):
    """The no-probe case is the common one, so it has to be a no-op: ``by_score``
    over score-less rows is a stable sort and must not move anything."""
    _sweep(svc, n=3)
    state = _scanned(ctx)
    entry = state.sweeps[-1]
    expected = [unit["job_id"] for unit in entry["units"]]

    review_mode.open_sweep(ctx, entry["id"])

    assert [unit["job_id"] for unit in state.units] == expected


def test_a_blind_review_neither_sorts_by_score_nor_shows_one(ctx, svc):
    """A score-derived order is a quality channel that re-identifies the arms,
    and an AI opinion on screen anchors the independent judgement a blind review
    exists to collect. So blinding wins wholesale."""
    _sweep(svc, n=3)
    state = _scanned(ctx)
    entry = state.sweeps[-1]
    for unit in entry["units"]:
        unit["score"] = 0.9 if unit is entry["units"][-1] else 0.1

    review_mode.set_blind(ctx, True)
    review_mode.open_sweep(ctx, entry["id"])

    assert [u["job_id"] for u in state.units] == [
        u["job_id"] for u in review_mode.blind_order(entry["units"])
    ]
    assert review_mode.score_line(state, state.units[0]) == ""

    # And it asks for nothing, so a blind session never spends a forward pass
    # producing a number it may not show.
    ctx.submitted.clear()
    review_mode.request_scores(ctx)
    review_mode.pump_scores(ctx)
    assert review_mode.SCORE_KEY not in ctx.submitted


def test_the_score_line_names_the_question_it_answers(ctx, svc):
    """A bare percentage beside a mesh reads as an opinion about the mesh, and
    this probe has never seen one."""
    _sweep(svc, n=1)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, state.sweeps[-1]["id"])
    state.units[0]["score"] = 0.82

    line = review_mode.score_line(state, state.units[0])

    assert "82%" in line
    assert review_mode.SCORE_QUESTION in line
    assert line.isascii(), "imgui's default atlas is Basic Latin plus Latin-1"


def test_a_row_the_judge_had_no_opinion_about_shows_nothing(ctx, svc):
    _sweep(svc, n=1)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, state.sweeps[-1]["id"])
    state.units[0]["score"] = None
    assert review_mode.score_line(state, state.units[0]) == ""


def test_retraining_the_blank_probe_throws_every_score_away_and_asks_again(ctx, svc):
    """The ``warlockc`` staleness rule: an absent answer is obvious, one
    silently computed by a probe that has since changed its mind is not."""
    _sweep(svc, n=2)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, state.sweeps[-1]["id"])
    state.score_request = [state.units[0]["job_id"]]
    review_mode.adopt_scores(state, {state.units[0]["job_id"]: 0.6})
    ctx.state.review_scores_dirty = False

    review_mode.on_task_done(
        ctx,
        _Done(review_mode.TRAIN_KEY, {"stage": review_mode.SCORE_STAGE,
                                      "trained": True, "usable": 40, "labels": 40}),
    )

    assert all("score" not in unit for unit in state.units)
    assert ctx.state.review_scores_dirty is True


def test_the_judge_files_no_verdict_of_its_own_yet(ctx, svc):
    """Deliberate, and a decision rather than an omission: filing needs a
    probability-to-accept threshold, and a threshold is a constant the stored
    corpus is then keyed on, which owes a measurement document first. The name
    it will file under is decided now so it is decided once."""
    _sweep(svc, n=2)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, state.sweeps[-1]["id"])
    review_mode.pump_scores(ctx)
    review_mode.on_task_done(ctx, _Done(review_mode.SCORE_KEY, ctx.result))

    assert review_mode.SOURCE_AI.startswith("ai:")
    assert all(row["source"] != review_mode.SOURCE_AI for row in svc.store.latest_verdicts())


def test_a_failed_scoring_run_is_silent_and_leaves_no_scores(ctx, svc):
    """Advisory: a failure to have an opinion is not a failure of the review,
    and a toast would make it look like one.

    The rows are left *unmarked* rather than marked as scored-with-nothing, or
    ``unscored`` would call them answered and nothing would ever ask again."""
    _sweep(svc, n=2)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, state.sweeps[-1]["id"])
    review_mode.pump_scores(ctx)
    ctx.toasts.clear()

    review_mode.on_task_failed(ctx, _Done(review_mode.SCORE_KEY))

    assert ctx.toasts == []
    assert all("score" not in unit for unit in state.units)
    assert review_mode.unscored(state.units) == [u["job_id"] for u in state.units]


def test_a_score_lands_on_the_rows_it_was_asked_about_not_on_whatever_is_open(
    ctx, svc
):
    """The user can open another sweep while the pass runs.

    Marking whatever happens to be open when the answer returns ticks off units
    nobody scored -- and ``unscored`` reads the same key, so those rows are then
    never asked about again. The request is carried instead, and because the
    unit dicts are shared with the ``sweeps`` entries the answer still lands on
    the sweep the user has moved away from.
    """
    _sweep(svc, label="first", n=2)
    _sweep(svc, label="second", n=2)
    state = _scanned(ctx)
    first, second = state.sweeps[1], state.sweeps[2]

    review_mode.open_sweep(ctx, first["id"])
    review_mode.pump_scores(ctx)
    asked = list(state.score_request)
    assert asked == [u["job_id"] for u in first["units"]]

    # Away before the answer comes back.
    review_mode.open_sweep(ctx, second["id"])
    review_mode.on_task_done(
        ctx, _Done(review_mode.SCORE_KEY, {job_id: 0.5 for job_id in asked})
    )

    assert all(unit.get("score") == 0.5 for unit in first["units"])
    assert all("score" not in unit for unit in second["units"])
    assert review_mode.unscored(state.units) == [u["job_id"] for u in second["units"]]


# --- the sweep form's legibility ---------------------------------------------
#
# The form used to be a combo of thirty raw param names beside a free-text box
# and a bare job count. Nothing here changes what is *stored* -- every control
# still writes the same comma-separated string, and ``build_plan``/``_coerce``
# are untouched -- so what is asserted is that the form can now say what each
# param accepts, and that a param it cannot describe degrades to the old field
# rather than to a broken one.


def test_every_axis_param_either_resolves_or_has_a_help_line(ctx):
    """The complement, deliberately. A param that stops resolving is caught by
    having no help line rather than by somebody noticing an empty box."""
    for row in review_mode.axis_options(ctx):
        assert row["kind"] != "text" or row["help"], (
            f"{row['param']} is a free-text field with nothing explaining it"
        )


def test_every_sweep_axis_param_is_described(ctx):
    from warlock.service import sweeps as svc_sweeps

    described = {row["param"] for row in review_mode.axis_options(ctx)}
    assert described == set(svc_sweeps.axis_params())


def test_a_taxonomy_param_offers_the_catalogs_options(ctx):
    rows = {row["param"]: row for row in review_mode.axis_options(ctx)}
    platform = rows["platform"]
    assert platform["kind"] == "options"
    assert platform["options"], "the catalog has platform options"
    assert all(o["key"] and o["label"] for o in platform["options"])


def test_a_numeric_param_carries_its_range_and_a_default(ctx):
    rows = {row["param"]: row for row in review_mode.axis_options(ctx)}
    weight = rows["lora_weight"]
    assert weight["kind"] == "number"
    low, high = weight["range"]
    assert low < high
    assert weight["default"] is not None


def test_the_params_the_guidance_catalog_cannot_answer_come_from_the_sweeps_block(ctx):
    """``resolution``, ``profile`` and the two trellis knobs are ``create_job``
    kwargs rather than taxonomy fields, so their legal values live in
    ``validation`` and ``pipelines.optimize`` -- gathered into the catalog by
    ``guidance_catalog`` rather than read from a pane."""
    rows = {row["param"]: row for row in review_mode.axis_options(ctx)}
    assert rows["resolution"]["kind"] == "options"
    assert "1024" in [o["key"] for o in rows["resolution"]["options"]]
    assert rows["profile"]["kind"] == "options"
    assert "raw" in [o["key"] for o in rows["profile"]["options"]]
    assert rows["trellis_band"]["kind"] == "number"
    assert rows["trellis_tex_res"]["kind"] == "number"


def test_a_switch_is_a_switch(ctx):
    rows = {row["param"]: row for row in review_mode.axis_options(ctx)}
    assert rows["reference_prep"]["kind"] == "bool"


def test_axis_options_falls_back_to_free_text_with_no_catalog(svc):
    """The old behaviour survives whole: a headless caller, or a ctx built
    before the catalog was fetched, gets the text field every row used to be.

    ``reference_prep`` is the one exception and needs no catalog: on/off is a
    fact about the param rather than a list somebody has to supply.
    """
    bare = FakeCtx(svc)
    bare.guidance = {}
    rows = {row["param"]: row for row in review_mode.axis_options(bare)}
    assert rows["platform"]["kind"] == "text"
    assert rows["lora_weight"]["kind"] == "text"
    assert rows["reference_prep"]["kind"] == "bool"
    assert not any(r["kind"] == "options" for r in rows.values()), (
        "an options row with no options draws nothing at all"
    )


def test_the_preview_line_names_the_multiplication(ctx):
    state = review_mode.ensure(ctx)
    state.form.prompt = "a wooden chest"
    state.form.seeds = "1, 2"
    state.form.axes = [{"param": "lora_weight", "values": "0.4, 0.6, 0.9"}]

    line = review_mode.preview_line(state)

    assert line.startswith("8 jobs:")
    assert "baseline" in line
    assert "lora_weight x 3 value(s)" in line
    assert "2 seed(s)" in line


def test_the_preview_line_uses_the_labels_it_is_handed(ctx):
    """Label resolution is the pane's -- this module may not import one."""
    state = review_mode.ensure(ctx)
    state.form.prompt = "a chest"
    state.form.seeds = "1"
    state.form.axes = [{"param": "lora_weight", "values": "0.4, 0.9"}]

    line = review_mode.preview_line(state, {"lora_weight": "Style strength"})

    assert "Style strength x 2 value(s)" in line


def test_the_preview_line_is_empty_for_a_form_that_cannot_be_planned(ctx):
    """Half an axis, which is what a form mid-typing routinely is."""
    state = review_mode.ensure(ctx)
    state.form.prompt = "a chest"
    state.form.axes = [{"param": "lora_weight", "values": ""}]
    assert review_mode.preview_line(state) == ""


def test_the_preview_line_is_basic_latin_only(ctx):
    """imgui's default atlas, the same rule ``score_line`` follows."""
    state = review_mode.ensure(ctx)
    state.form.prompt = "a chest"
    state.form.axes = [{"param": "lora_weight", "values": "0.4, 0.9"}]
    assert review_mode.preview_line(state).isascii()


def test_a_stored_spec_is_summarised_for_the_list():
    spec = {
        "axes": [{"param": "lora_weight", "values": [0.4, 0.6, 0.9]}],
        "seeds": [1, 2],
    }
    assert review_mode.spec_summary(spec) == "varies lora_weight (0.4, 0.6, 0.9) - 2 seed(s)"


def test_a_spec_that_is_not_one_summarises_to_nothing():
    """A sweep row written by an older build, or the recent bucket, which has
    no spec at all."""
    assert review_mode.spec_summary(None) == ""
    assert review_mode.spec_summary({}) == ""


def test_the_scan_carries_each_sweeps_spec(ctx, svc):
    """``list_sweeps`` has always parsed it and this bucket always dropped it,
    which left the list identifying a run by whatever name was typed at the
    time."""
    from warlock.service import sweeps as svc_sweeps

    svc.store.create_sweep("test2", "a chest", {"axes": [{"param": "lora_weight",
                                                          "values": [0.4, 0.9]}],
                                                "seeds": [1]})
    state = _scanned(ctx)

    sweeps = [s for s in state.sweeps if s["id"] != review_mode.RECENT_ID]
    assert sweeps and sweeps[0]["spec"]["axes"][0]["param"] == "lora_weight"
    assert "varies lora_weight" in review_mode.spec_summary(sweeps[0]["spec"])
    del svc_sweeps


def test_the_recent_bucket_has_no_spec_to_show(ctx):
    state = _scanned(ctx)
    recent = state.sweeps[0]
    assert recent["id"] == review_mode.RECENT_ID
    assert review_mode.spec_summary(recent.get("spec")) == ""


# --- the guided judging pass -------------------------------------------------


def _two_sweeps(svc):
    """Two sweeps, returned **in the order the list presents them**.

    ``list_sweeps`` is newest-first, so the one created second is the one a
    pass opens first -- and a test that assumed creation order would be
    asserting against an order the user never sees.
    """
    older, _ = _sweep(svc, "one", n=2)
    newer, _ = _sweep(svc, "two", n=2)
    return newer, older


def test_todo_total_counts_every_bucket(ctx, svc):
    _two_sweeps(svc)
    state = _scanned(ctx)
    assert review_mode.todo_total(state) == 4


def test_starting_a_pass_opens_the_first_bucket_with_work(ctx, svc):
    first, _second = _two_sweeps(svc)
    state = _scanned(ctx)
    state.sweep_id = None

    assert review_mode.start_judging(ctx) is True

    assert state.judging is not None
    assert state.judging.total == 4
    assert state.judging.filed == 0
    # The recent bucket has no work here, so the first sweep is where it lands.
    assert state.sweep_id == first
    assert review_mode.RECENT_ID not in state.judging.order


def test_starting_a_pass_disarms(ctx, svc):
    """The armed sign belongs to whatever was on screen before the pass."""
    _two_sweeps(svc)
    state = _scanned(ctx)
    state.pending_negative = True
    state.pending_tags = ["holes"]

    review_mode.start_judging(ctx)

    assert state.pending_negative is False
    assert state.pending_tags == []


def test_a_pass_refuses_with_nothing_to_judge(ctx, svc):
    state = _scanned(ctx)
    assert review_mode.todo_total(state) == 0
    assert review_mode.start_judging(ctx) is False
    assert state.judging is None


def test_a_pass_refuses_while_a_scan_is_running(ctx, svc):
    _two_sweeps(svc)
    state = _scanned(ctx)
    state.scanning = True
    assert review_mode.start_judging(ctx) is False


def test_accept_and_reject_file_the_binary_grades(ctx, svc):
    """Read back through the store, not off the in-memory unit: the point is
    that the pass writes real verdicts through the same door everything else
    does."""
    from warlock.vectors import BINARY_GRADES

    sweep_id, _ids = _sweep(svc, n=2)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    state.judging = review_mode.JudgingPass(total=2, order=[sweep_id])

    assert _press(ctx, "a") is True
    assert _press(ctx, "r") is True

    grades = sorted(v["grade"] for v in svc.store.latest_verdicts())
    assert grades == [BINARY_GRADES["reject"], BINARY_GRADES["accept"]]


def test_a_bare_a_outside_a_pass_still_records_nothing(ctx, svc):
    """The pin the pass must not weaken. ``a`` is bound because a loop the user
    entered on purpose says on screen what it does -- not because the key is
    free."""
    sweep_id, _ = _sweep(svc)
    _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    assert _press(ctx, "a") is False
    assert svc.store.latest_verdicts() == []


def test_r_inside_a_pass_rejects_rather_than_arming_a_sign(ctx, svc):
    """A binary loop has no magnitude to arm. The negative grades stay on the
    digits and the pane's own buttons."""
    sweep_id, _ = _sweep(svc, n=2)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    state.judging = review_mode.JudgingPass(total=2, order=[sweep_id])

    _press(ctx, "r")

    assert state.pending_negative is False
    assert len(svc.store.latest_verdicts()) == 1


def test_a_grade_pressed_inside_a_pass_still_files_and_counts(ctx, svc):
    """The eleven-point scale is the power path and keeps working."""
    sweep_id, _ = _sweep(svc, n=3)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    state.judging = review_mode.JudgingPass(total=3, order=[sweep_id])

    _press(ctx, "5")

    assert svc.store.latest_verdicts()[0]["grade"] == 5
    assert state.judging.filed == 1


def test_skip_inside_a_pass_files_nothing_and_moves_on(ctx, svc):
    sweep_id, _ = _sweep(svc, n=3)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    state.judging = review_mode.JudgingPass(total=3, order=[sweep_id])
    before = state.index

    assert _press(ctx, "s") is True

    assert svc.store.latest_verdicts() == []
    assert state.index != before
    assert state.judging is not None, "skipping does not end the pass"


def test_a_pass_moves_from_one_bucket_to_the_next(ctx, svc):
    first, second = _two_sweeps(svc)
    state = _scanned(ctx)
    review_mode.start_judging(ctx)
    assert state.sweep_id == first

    _press(ctx, "a")
    _press(ctx, "a")

    assert state.sweep_id == second
    assert state.judging is not None


def test_a_pass_ends_when_every_bucket_is_judged(ctx, svc):
    _two_sweeps(svc)
    state = _scanned(ctx)
    review_mode.start_judging(ctx)
    for _ in range(4):
        _press(ctx, "a")

    assert state.judging is None
    assert state.judging_report is not None
    assert state.judging_report["filed"] == 4
    assert state.judging_report["remaining"] == 0


def test_escape_ends_a_pass_early_and_counts_what_is_left(ctx, svc):
    """A pass abandoned halfway is exactly when the summary is most useful."""
    _two_sweeps(svc)
    state = _scanned(ctx)
    review_mode.start_judging(ctx)
    _press(ctx, "a")

    assert _press(ctx, "ESCAPE") is True

    assert state.judging is None
    report = state.judging_report
    assert report["filed"] == 1
    assert report["remaining"] == 3


def test_escape_outside_a_pass_still_only_disarms(ctx, svc):
    sweep_id, _ = _sweep(svc, n=2)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    state.pending_negative = True

    _press(ctx, "ESCAPE")

    assert state.pending_negative is False
    assert state.judging_report is None


def test_the_report_is_tallied_from_state_rather_than_the_store(ctx, svc, monkeypatch):
    """Never a DB read on the frame thread: ``_collect`` already fetched the
    units and ``record`` keeps them current."""
    sweep_id, _ = _sweep(svc, n=2)
    state = _scanned(ctx)
    review_mode.start_judging(ctx)
    _press(ctx, "a")

    def boom(*_a, **_kw):
        raise AssertionError("the report must not read the store")

    monkeypatch.setattr(svc.store, "latest_verdicts", boom)
    monkeypatch.setattr(svc.store, "sweep_jobs", boom)
    review_mode.end_judging(ctx)

    row = next(r for r in state.judging_report["sweeps"] if r["id"] == sweep_id)
    assert row["accepted"] == 1
    assert row["total"] == 2
    assert row["mean_grade"] == 3.0


def test_a_blind_report_stays_blind(ctx, svc):
    """Naming the arms at the end would un-blind every unit after the fact,
    which is most of what blinding was protecting."""
    sweep_id, _ = _sweep(svc, "lora-weight-arms", n=2)
    state = _scanned(ctx)
    review_mode.set_blind(ctx, True)
    review_mode.start_judging(ctx)
    _press(ctx, "a")
    review_mode.end_judging(ctx)

    labels = [row["label"] for row in state.judging_report["sweeps"]]
    assert "lora-weight-arms" not in labels
    assert labels == [f"#{sweep_id[:6]}"]


def test_dismissing_the_report_clears_it(ctx, svc):
    _sweep(svc, n=1)
    _scanned(ctx)
    review_mode.start_judging(ctx)
    _press(ctx, "a")
    state = ctx.state.review
    assert state.judging_report is not None

    review_mode.dismiss_report(ctx)

    assert state.judging_report is None


def test_nothing_about_a_pass_is_persisted(ctx, svc):
    """``ReviewState`` persists nothing at all, and the pass is no exception:
    a stored one would resume against buckets since judged or deleted."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(review_mode.ReviewState)}
    assert {"judging", "judging_report"} <= fields
    assert ctx.settings.store == {}


# --- automatic cleanup once a sweep is judged --------------------------------


def test_judging_the_last_unit_submits_a_cleanup(ctx, svc):
    sweep_id, _ = _sweep(svc, n=2)
    _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    review_mode.record(ctx, 3)
    assert review_mode.CLEANUP_KEY not in ctx.submitted
    review_mode.record(ctx, 3)

    assert review_mode.CLEANUP_KEY in ctx.submitted


def test_cleanup_fires_for_hand_grading_outside_a_pass(ctx, svc):
    """The requirement is about the sweep rather than about how it was judged:
    a hand-graded sweep has been judged just as thoroughly as a pass-driven
    one."""
    sweep_id, _ = _sweep(svc, n=1)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)
    assert state.judging is None

    review_mode.record(ctx, -3)

    assert review_mode.CLEANUP_KEY in ctx.submitted


def test_cleanup_actually_removes_the_assets(ctx, svc):
    """``FakeCtx`` runs a submit inline, so this is the whole round trip."""
    sweep_id, ids = _sweep(svc, n=1)
    _scanned(ctx)
    review_mode.open_sweep(ctx, sweep_id)

    review_mode.record(ctx, 3)

    assert svc.store.get(ids[0]) is None
    assert not svc.job_dir(ids[0]).exists()
    # And what was learned survives, which is the whole licence for removing
    # the files at all.
    assert len(svc.store.latest_verdicts()) == 1


def test_cleanup_refuses_the_recent_bucket(ctx, svc):
    """Its units are ordinary library rows that happen not to carry a verdict
    yet; deleting them because they were judged would delete the user's assets.
    Pruning the library is prune's job and has its own confirmation."""
    _mesh(svc, "a chest")
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, review_mode.RECENT_ID)

    assert review_mode.auto_cleanup(ctx, review_mode.RECENT_ID) is False
    review_mode.record(ctx, 3)
    assert review_mode.CLEANUP_KEY not in ctx.submitted
    assert state.sweeps[0]["todo"] == 0


def test_a_finished_cleanup_toasts_and_rescans(ctx, svc):
    _sweep(svc, n=1)
    _scanned(ctx)
    ctx.toasts.clear()

    review_mode.on_task_done(
        ctx,
        _Done(
            review_mode.CLEANUP_KEY,
            {"ok": True, "deleted": 3, "remaining": 0, "kept": 0},
            tag="lora",
        ),
    )

    said = " ".join(text for text, _kind in ctx.toasts)
    assert "Cleaned up lora" in said
    assert "Verdicts and findings kept" in said
    assert review_mode.SCAN_KEY in ctx.submitted


def test_a_cleanup_that_could_not_finish_says_so(ctx, svc):
    _scanned(ctx)
    ctx.toasts.clear()

    review_mode.on_task_done(
        ctx,
        _Done(
            review_mode.CLEANUP_KEY,
            {"ok": True, "deleted": 1, "remaining": 2, "kept": 0},
            tag="lora",
        ),
    )

    said = " ".join(text for text, _kind in ctx.toasts)
    assert "still finishing" in said


def test_a_failed_cleanup_is_toasted_rather_than_swallowed(ctx, svc):
    """Nobody pressed a button for it, but the entry card promised it would
    happen -- silence would leave a failed promise looking like a kept one."""
    _scanned(ctx)
    ctx.toasts.clear()

    review_mode.on_task_failed(ctx, _Done(review_mode.CLEANUP_KEY, None))

    assert any(kind == "error" for _text, kind in ctx.toasts)


def test_the_cleanup_key_carries_the_review_prefix():
    """The app claims a result by prefix; a key without one is delivered
    nowhere."""
    assert review_mode.CLEANUP_KEY.startswith("review-")


# --- the pass and the rescan its own cleanup triggers -------------------------
#
# Both of these are one bug found by driving the real app, and it deadlocked the
# feature outright. Judging a sweep out fires a cleanup; the cleanup lands as a
# task and rescans; the rescan's fallback opened the *first* bucket, which by
# then was an emptied recent bucket with no units. ``record`` returns early with
# no current unit and ``judging_skip`` needs one too -- so the only two things
# that moved the pass along could not run, and it sat there with work still
# outstanding in another sweep and no key that did anything.


def test_a_cleanups_rescan_does_not_drop_the_bucket_the_pass_moved_to(ctx, svc):
    """A manual delete removes the sweep you were looking at, so forgetting it
    is right. A pass's cleanup fires for a sweep the pass has already walked
    past, and clearing the selection there is what sent the rescan's fallback
    to the wrong bucket."""
    newer, older = _two_sweeps(svc)
    state = _scanned(ctx)
    review_mode.start_judging(ctx)
    state.sweep_id = older

    review_mode.on_task_done(
        ctx,
        _Done(review_mode.CLEANUP_KEY, {"ok": True, "deleted": 2, "remaining": 0, "kept": 0}),
    )

    assert state.sweep_id == older, "the pass keeps the bucket it had moved to"
    del newer


def test_a_delete_outside_a_pass_still_drops_the_selection(ctx, svc):
    """The pin the fix above must not weaken."""
    newer, _older = _two_sweeps(svc)
    state = _scanned(ctx)
    review_mode.open_sweep(ctx, newer)
    assert state.judging is None

    review_mode.on_task_done(
        ctx,
        _Done(review_mode.DELETE_KEY, {"ok": True, "deleted": 2, "remaining": 0, "kept": 0}),
    )

    # The rescan runs inline in this fake, so the selection has already fallen
    # back to the first bucket rather than staying on a sweep that is going.
    assert state.sweep_id != newer or not any(s["id"] == newer for s in state.sweeps)


def test_a_pass_parked_on_an_empty_bucket_moves_itself_on(ctx, svc):
    """The deadlock, reproduced directly: no unit on screen, so neither of the
    pass's other two pumps can run."""
    newer, older = _two_sweeps(svc)
    state = _scanned(ctx)
    review_mode.start_judging(ctx)
    # An emptied bucket, which is exactly what a cleaned sweep and a drained
    # recent bucket both look like after a rescan.
    empty = {"id": "empty", "label": "Empty", "prompt": "", "units": [], "todo": 0}
    state.sweeps = [empty] + [s for s in state.sweeps if s["id"] in (newer, older)]
    state.sweep_id, state.units, state.index = "empty", [], 0
    assert review_mode.current(state) is None

    review_mode.on_task_done(ctx, _Done(review_mode.SCAN_KEY, state.sweeps))

    assert state.judging is not None, "there is still work; the pass must not end"
    assert state.sweep_id in (newer, older)
    assert review_mode.current(state) is not None, "a unit is on screen again"


def test_a_scan_that_leaves_nothing_to_judge_ends_the_pass(ctx, svc):
    """The other half of the same pump: a pass whose work all disappeared has
    to finish and report, not sit on an empty list."""
    _two_sweeps(svc)
    state = _scanned(ctx)
    review_mode.start_judging(ctx)
    for sweep in state.sweeps:
        sweep["units"], sweep["todo"] = [], 0
    state.units, state.index = [], 0

    review_mode.on_task_done(ctx, _Done(review_mode.SCAN_KEY, state.sweeps))

    assert state.judging is None
    assert state.judging_report is not None


def test_the_pump_does_not_re_clean_a_sweep_that_is_already_gone(ctx, svc):
    """``_pump_judging`` runs after a scan now, so it meets buckets that have
    been cleaned already and buckets that were never sweeps."""
    newer, _older = _two_sweeps(svc)
    state = _scanned(ctx)
    review_mode.start_judging(ctx)
    entry = next(s for s in state.sweeps if s["id"] == newer)
    entry["units"], entry["todo"] = [], 0
    state.sweep_id, state.units, state.index = newer, [], 0
    ctx.submitted.clear()

    review_mode.on_task_done(ctx, _Done(review_mode.SCAN_KEY, state.sweeps))

    assert review_mode.CLEANUP_KEY not in ctx.submitted


def test_a_whole_pass_survives_the_cleanups_it_triggers(ctx, svc):
    """End to end, and it is the *delivery* that makes it worth having.

    ``FakeCtx`` runs a submitted callable inline but does not route its result
    back through ``on_task_done`` -- which is the App's job -- so a pass driven
    by keys alone never meets the rescan its own cleanup causes, and that
    rescan is the whole bug. This loop plays the App's part: after each press,
    every task the pass submitted is delivered, in order, exactly as the frame
    loop delivers them.
    """
    _two_sweeps(svc)
    _mesh(svc, "a wooden chest")
    state = _scanned(ctx)
    outstanding = review_mode.todo_total(state)
    assert outstanding == 5

    def deliver() -> None:
        """Hand back every result the last action produced."""
        for _ in range(8):
            pending = [k for k in ctx.submitted if k.startswith("review-")]
            if not pending:
                return
            ctx.submitted.clear()
            for key in pending:
                if key == review_mode.SCAN_KEY:
                    review_mode.on_task_done(ctx, _Done(key, _collect_now(ctx)))
                elif key == review_mode.CLEANUP_KEY:
                    review_mode.on_task_done(
                        ctx,
                        _Done(key, {"ok": True, "deleted": 1, "remaining": 0, "kept": 0}),
                    )

    review_mode.start_judging(ctx)
    ctx.submitted.clear()
    for _ in range(30):
        if state.judging is None:
            break
        assert review_mode.current(state) is not None, (
            f"stalled with no unit on screen: sweep_id={state.sweep_id} "
            f"todo={review_mode.todo_total(state)}"
        )
        _press(ctx, "a")
        deliver()

    assert state.judging is None, "the pass finished by itself"
    assert state.judging_report["remaining"] == 0
    assert review_mode.todo_total(state) == 0


def _collect_now(ctx: Any) -> list[dict[str, Any]]:
    """What a scan would have found, read now rather than when it was asked
    for -- the store has moved on since the submit."""
    return review_mode._collect(ctx.svc)
