"""Nothing writes UI state from a task thread -- the doors the review named.

The 2026-09-02 review's theme T3 is T2's mirror image. T2 was about expensive
work on the frame thread; this is about the answer coming back the wrong way:
a task that reaches into the object the frame loop is reading sixty times a
second and amends it in place. Two sites stood after the Clay uid counter was
locked -- ``journal.write``'s three mark attributes and ``jobs_cache``'s
storage sizes -- and both are now a *reading* handed back through
``Done.result`` with a frame-thread half that applies it.

The guard is behavioural: the task half runs on a real worker thread, and the
test asserts the state it must not touch is exactly as it was when the thread
joins. A regression that moves the write back inside the task shows up as a
mark, a size or a generation that moved without anybody adopting anything.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from warlock.service import jobs as svc_jobs
from warlock.studio import journal
from warlock.studio.jobs_cache import JobsCache

WORKER = "warlock-task-test"


def _on_worker(fn: Any) -> Any:
    """Run ``fn`` on a thread named like the runner's, and hand back its result."""
    box: dict[str, Any] = {}

    def go() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
            box["error"] = exc

    thread = threading.Thread(target=go, name=WORKER)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


# --- the journal --------------------------------------------------------------


class _Slot:
    def __init__(self) -> None:
        self.uid = "s1"
        self.title = "thing"
        self.body = b"a"
        self.head = 1
        self.journal_name = ""
        self.journal_head = None
        self.journal_at = 0.0


class _DeferredCtx:
    """A ctx whose ``submit`` queues rather than runs, so the test owns both
    halves and can say which thread each ran on."""

    def __init__(self, root: Path) -> None:
        self.svc = SimpleNamespace(config=SimpleNamespace(autosave_dir=root))
        self.state = SimpleNamespace()
        self.queued: list[tuple[str, Any]] = []
        self.toasts: list[tuple[str, str]] = []

    def submit(self, key: str, run: Any, *args: Any, **kwargs: Any) -> bool:
        self.queued.append((key, lambda: run(*args, **kwargs)))
        return True

    def toast(self, text: str, level: str = "info", **_kw: Any) -> None:
        self.toasts.append((text, level))


@pytest.fixture
def probe(monkeypatch):
    """A registered journal provider over one slot, removed again afterwards."""
    slots: list[_Slot] = []
    provider = journal.Provider(
        kind="probe",
        ext=".probe",
        label="probe",
        slots=lambda ctx: list(slots),
        uid_of=lambda s: s.uid,
        title_of=lambda s: s.title,
        head_of=lambda s: s.head,
        encode=lambda s: s.body,
        adopt=lambda ctx, path, meta: True,
    )
    before = dict(journal._PROVIDERS)
    journal.register(provider)
    yield SimpleNamespace(provider=provider, slots=slots)
    journal._PROVIDERS.clear()
    journal._PROVIDERS.update(before)


def _armed(tmp_path, probe):
    """A ctx and a slot with one write queued and nothing written yet."""
    ctx = _DeferredCtx(tmp_path)
    slot = _Slot()
    probe.slots.append(slot)
    journal.pump(ctx, now=9_000.0)  # first sight arms the debounce
    journal.pump(ctx, now=10_000.0)
    assert len(ctx.queued) == 1
    return ctx, slot


def test_the_write_task_leaves_the_slot_alone(tmp_path, probe):
    """The three attributes are read every frame with no lock, so a task
    setting them can be seen half-applied: a new ``head`` beside the old
    ``name`` is a slot claiming a copy under a filename holding a different
    one."""
    ctx, slot = _armed(tmp_path, probe)
    name, at = slot.journal_name, slot.journal_at

    _on_worker(ctx.queued[0][1])

    assert slot.journal_head is None, "the head is not the task's to advance"
    assert (slot.journal_name, slot.journal_at) == (name, at)
    assert (tmp_path / name).read_bytes() == b"a", "the copy itself did land"


def test_the_frame_thread_half_advances_the_mark(tmp_path, probe):
    ctx, slot = _armed(tmp_path, probe)
    key, run = ctx.queued[0]

    result = _on_worker(run)
    journal.on_task_done(ctx, SimpleNamespace(key=key, result=result))

    assert slot.journal_head == slot.head
    assert slot.journal_name and slot.journal_at == 10_000.0


def test_a_drop_before_the_result_lands_is_not_undone_by_it(tmp_path, probe):
    """``drop`` runs on the frame thread and so does the adopt, so the two can
    no longer interleave at all -- but a result can still arrive *after* a
    drop, and a mark applied then would offer a saved document's copy back on
    the next launch."""
    ctx, slot = _armed(tmp_path, probe)
    key, run = ctx.queued[0]
    result = _on_worker(run)

    journal.drop(ctx, slot)
    journal.on_task_done(ctx, SimpleNamespace(key=key, result=result))

    assert slot.journal_name == "" and slot.journal_head is None
    assert journal.recoverable(ctx) == []


def test_a_failed_write_hands_back_nothing_to_adopt(tmp_path, probe, monkeypatch):
    ctx, slot = _armed(tmp_path, probe)

    def full_disk(*_args: Any) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(journal, "_write_pair", full_disk)
    with pytest.raises(OSError):
        _on_worker(ctx.queued[0][1])

    assert slot.journal_head is None, "a write that failed is not a copy"


# --- the storage walk ---------------------------------------------------------


def test_measuring_storage_amends_nothing_from_the_task_thread(svc):
    """``_dir_sizes`` backs the library's size sort and ``_sizes_generation``
    is in the ``visible`` memo's key, so a task amending either mid-frame
    re-sorts a list under the reader looking at it."""
    svc_jobs.create_job(svc, kind="text", prompt="a")
    cache = JobsCache(svc)
    generation = cache._sizes_generation

    reading = _on_worker(cache.measure)

    assert cache._dir_sizes == {}, "the walk publishes nothing"
    assert cache._sizes_generation == generation
    assert cache.storage == {}
    assert "sizes" in reading

    cache.adopt_storage(reading)
    assert cache._sizes_generation == generation + 1
    assert cache.storage["job_dirs"] >= 0


def test_folding_one_job_in_amends_nothing_from_the_task_thread(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a")["id"]
    cache = JobsCache(svc)
    cache.adopt_storage(cache.measure())
    before = dict(cache._dir_sizes)
    generation = cache._sizes_generation
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"x" * 1000)

    reading = _on_worker(lambda: cache.measure_one(job_id))

    assert cache._dir_sizes == before, "the fold is the frame thread's to apply"
    assert cache._sizes_generation == generation

    cache.adopt_storage(reading)
    assert cache.storage["bytes"] >= sum(before.values()) + 1000


def test_a_directory_that_vanished_is_folded_out_rather_than_left_behind(svc):
    """The other half of the fold, and the one a dict amendment made easy to
    lose: a pruned job's entry has to go, or its bytes stay in the total."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a")["id"]
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"x" * 1000)
    cache = JobsCache(svc)
    cache.adopt_storage(cache.measure())
    assert cache.storage["bytes"] >= 1000

    (job_dir / "model.glb").unlink()
    job_dir.rmdir()
    cache.adopt_storage(cache.measure_one(job_id))

    assert job_dir.name not in cache._dir_sizes
    assert cache.storage["bytes"] == 0


def test_a_failed_measurement_keeps_the_last_good_figure_and_says_why(svc, monkeypatch):
    from warlock.studio import jobs_cache as mod

    cache = JobsCache(svc)
    cache.adopt_storage({"sizes": {"a": 10}})
    assert cache.storage == {"job_dirs": 1, "bytes": 10}

    def boom(_svc):
        raise OSError("the data directory vanished")

    monkeypatch.setattr(mod.svc_jobs, "storage_sizes", boom)
    cache.adopt_storage(_on_worker(cache.measure))

    assert cache.storage == {"job_dirs": 1, "bytes": 10}
    assert "vanished" in (cache.storage_error or "")
