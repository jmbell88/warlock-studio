"""The trash bar's figure: ``service.jobs.trash_size`` finally has a reader.

The function was written for this bar and then never wired to it -- its only
occurrence in the tree was its own ``def`` -- so the trash view offered "Empty
trash..." and no way to know whether pressing it would free four gigabytes or
nothing. Its sibling ``empty_trash`` had been wired the whole time, which is
what made the gap easy to miss.

Two properties are worth pinning and neither is about the number itself
(``service.jobs`` owns that, and ``tests/test_service.py`` pins it). The first
is that the walk never happens on the frame thread: ``trash_size`` opens a
directory per trashed job, which is the stall ``attach_files`` exists to avoid,
so the pane submits it and draws the last answer. The second is that it is
*re-armed* rather than armed once -- ``TaskRunner.submit`` refuses a key already
in flight and nothing else would re-arm it, which is the ``findings_dirty``
lesson, and here the trash changes every time a row is restored or trashed.
"""

from __future__ import annotations

import time
from typing import Any

from warlock.studio.panes import library
from warlock.studio.state import AppState


class FakeCtx:
    """A ctx whose ``submit`` runs inline, which is the point of the fake.

    The real one is a four-thread pool; running the callable here makes the
    answer land in the same call and keeps the test about *when* the pane asks
    rather than about ``TaskRunner``. ``accept=False`` stands in for the runner
    refusing a key already in flight.
    """

    def __init__(self, svc: Any, *, accept: bool = True) -> None:
        self.svc = svc
        self.state = AppState()
        self.accept = accept
        self.submits: list[str] = []

    def submit(self, key: str, fn: Any, *args: Any, **kwargs: Any) -> bool:
        self.submits.append(key)
        if not self.accept:
            return False
        fn(*args, **kwargs)
        return True


def _trashed(svc, count):
    """``count`` jobs in the trash, each with a directory holding some bytes."""
    rows = []
    for index in range(count):
        job_id = svc.store.create("image", "a chest", {}, stage="reference", status="done")
        job_dir = svc.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "input.png").write_bytes(b"x" * 1000)
        svc.store.set_deleted_if_not_running(job_id, time.time())
        rows.append({"id": job_id, "index": index})
    return rows


# --- when the walk happens ----------------------------------------------------


def test_the_walk_is_a_task_and_never_the_frame_threads_work(svc):
    ctx = FakeCtx(svc)
    rows = _trashed(svc, 3)

    assert library.measure_trash(ctx, rows) is None  # nothing measured yet
    assert ctx.submits == [library.TRASH_SIZE_KEY]

    answer = library.measure_trash(ctx, rows)
    assert answer["count"] == 3
    assert answer["bytes"] >= 3000


def test_a_measured_trash_is_not_re_walked_every_frame(svc):
    ctx = FakeCtx(svc)
    rows = _trashed(svc, 2)

    library.measure_trash(ctx, rows)  # the first ask does the walk
    for _ in range(10):
        assert library.measure_trash(ctx, rows)["count"] == 2

    assert ctx.submits == [library.TRASH_SIZE_KEY]


def test_restoring_a_row_re_arms_the_measurement(svc):
    """The re-arm is a comparison against the loaded page, not a flag: nothing
    else would notice, and a figure that never moved would be worse than none
    -- it would be believed."""
    ctx = FakeCtx(svc)
    rows = _trashed(svc, 3)
    library.measure_trash(ctx, rows)
    assert library.measure_trash(ctx, rows)["count"] == 3

    svc.store.set_deleted_if_not_running(rows[0]["id"], None)
    remaining = rows[1:]

    # The frame that notices asks and still draws the old figure -- the answer
    # lands on a task thread, which in the real app is some frames later.
    assert library.measure_trash(ctx, remaining)["count"] == 3
    assert library.measure_trash(ctx, remaining)["count"] == 2
    assert ctx.submits == [library.TRASH_SIZE_KEY] * 2


def test_a_refused_submit_is_retried_rather_than_dropped(svc):
    """``TaskRunner.submit`` refuses a key already in flight and nothing else
    re-arms it, so the ask has to be repeated for as long as it is stale --
    otherwise the first frame after a restore is the only one that ever asks,
    and a refusal there strands the figure at its old value forever."""
    ctx = FakeCtx(svc, accept=False)
    rows = _trashed(svc, 1)

    for _ in range(3):
        assert library.measure_trash(ctx, rows) is None

    assert ctx.submits == [library.TRASH_SIZE_KEY] * 3


def test_a_stale_answer_is_still_drawn_while_the_next_walk_runs(svc):
    """A number that vanishes while it is being re-taken is a row that changes
    height every time something is restored."""
    ctx = FakeCtx(svc)
    rows = _trashed(svc, 2)
    library.measure_trash(ctx, rows)
    ctx.accept = False  # the next walk is refused, i.e. still in flight

    assert library.measure_trash(ctx, rows[:1])["count"] == 2


# --- what the bar says --------------------------------------------------------


def test_the_summary_counts_assets_and_says_how_much_disk(svc):
    assert library.trash_summary({"count": 12, "bytes": 0}).startswith("12 assets - ")
    assert library.trash_summary({"count": 1, "bytes": 0}).startswith("1 asset - ")
    assert library.trash_summary({"count": 0, "bytes": 0}).startswith("0 assets - ")


def test_nothing_measured_yet_is_no_line_at_all():
    """Rather than "0 assets", which is a claim -- and the wrong one, on the
    frames between opening the trash and the walk finishing."""
    assert library.trash_summary(None) is None


def test_the_summary_survives_a_malformed_answer():
    """It is read on the frame thread from a dict a task thread wrote, and the
    posture everywhere else in the panes is to draw something wrong-but-safe
    rather than raise inside a frame."""
    assert library.trash_summary({}) == "0 assets - 0 B"
