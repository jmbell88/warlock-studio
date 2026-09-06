"""What the queue and the store do when the machine underneath them misbehaves.

The 2026-08-24 audit asked whether the job pipeline is *correct*. These ask what
it does when it is not: a terminal write that fails, a store that answers every
question with an exception, a cancel that lands one line after the artifact has
been published, and a backup that used to park the frame thread while it walked
the database out to disk.

Each case here is one where the failure was *silent* -- a row stranded in
``running``, a worker that stayed "alive" with nothing running, a sheet deleted
out from under the user -- which is the property being pinned rather than the
mechanism.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from warlock import db as db_mod
from warlock import rigging
from warlock.config import Config
from warlock.db import JobStore
from warlock.queue import Worker


@pytest.fixture
def worker(tmp_path, fake_pipelines):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    yield w
    store.close()


async def _wait_until(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


# --- the dispatch loop -------------------------------------------------------


async def test_a_store_that_never_answers_stops_the_worker_and_says_so(worker):
    """The loop caught everything, slept and looped, which is right for a
    hiccup and wrong for a wall.

    A corrupt page or a full disk left ``Runtime.alive`` True and ``fatal``
    unset: every job sat in ``queued``, nothing on screen said why, and one
    traceback a second went to the disk that had caused it.
    ``main._check_worker`` already knew how to report a dead worker exactly
    once and well -- it simply had nothing to read.
    """
    from warlock import queue as queue_mod

    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("database disk image is malformed")

    worker.store.next_queued = broken
    # The back-off between attempts, shrunk. Five real POLL_INTERVALs is five
    # seconds of a test sitting still, and the interval is not the property --
    # the escalation is.
    monkeypatch_interval = queue_mod.POLL_INTERVAL
    queue_mod.POLL_INTERVAL = 0.001
    try:
        worker.start()
        await _wait_until(lambda: worker.fatal is not None)
    finally:
        queue_mod.POLL_INTERVAL = monkeypatch_interval

    assert len(calls) >= queue_mod.LOOP_FAILURE_LIMIT
    assert "malformed" in str(worker.fatal)
    await _wait_until(lambda: not worker.alive)
    await worker.shutdown()


async def test_one_bad_iteration_does_not_stop_the_worker(worker):
    """The other half, and the reason the counter is *consecutive*. The loop
    was written to survive a DB hiccup and must go on doing so."""
    real = worker.store.next_queued
    failed = []

    def flaky():
        if not failed:
            failed.append(1)
            raise RuntimeError("database is locked")
        return real()

    worker.store.next_queued = flaky
    worker.start()
    await _wait_until(lambda: len(failed) == 1)
    await asyncio.sleep(0.05)
    assert worker.fatal is None and worker.alive
    await worker.shutdown()


async def test_a_terminal_write_that_loses_a_race_is_retried(worker, monkeypatch):
    """``store.finish`` raising left the row in ``running``, and the next
    launch's ``reconcile_startup`` calls that "interrupted by shutdown" -- so a
    mesh that generated perfectly was reported to the user as a crash, with
    nothing but the presence of a model.glb to tell the two apart."""
    from warlock import queue as queue_mod

    monkeypatch.setattr(queue_mod, "POLL_INTERVAL", 0.001)
    real_finish = worker.store.finish
    attempts = []

    def flaky(job_id, status, error=None):
        attempts.append(status)
        if len(attempts) < 3:
            raise RuntimeError("database is locked")
        return real_finish(job_id, status, error)

    monkeypatch.setattr(worker.store, "finish", flaky)
    job_id = worker.store.create("text", "a knight", {"seed": 1})
    worker.store.claim(job_id)
    assert await worker._finish_job(job_id, "done", None) is True
    assert len(attempts) == 3
    assert worker.store.get(job_id)["status"] == "done"


async def test_a_terminal_write_that_never_lands_is_reported_rather_than_swallowed(
    worker, monkeypatch
):
    """Nothing here can save the row. What it must not do is fail quietly:
    the exception goes back to the dispatch loop, whose counter is what turns
    a persistently unwritable store into a visible dead worker."""
    from warlock import queue as queue_mod

    monkeypatch.setattr(queue_mod, "POLL_INTERVAL", 0.001)

    def never(job_id, status, error=None):
        raise RuntimeError("no space left on device")

    monkeypatch.setattr(worker.store, "finish", never)
    with pytest.raises(RuntimeError):
        await worker._finish_job("nope", "done", None)


# --- publishing and cancelling ----------------------------------------------


def _function_source(module: Any, name: str) -> str:
    """The source of one method of the module's ``*Ops`` mixin.

    By hand rather than through ``inspect.getsource`` on the function object:
    these are mixin methods on a class whose name differs per module, and the
    question here is about the *text between two calls*, which is exactly what
    a source slice is for.
    """
    source = inspect.getsource(module)
    start = source.index(f"def {name}(")
    tree = ast.parse(source)
    ends = [
        node.end_lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name
    ]
    assert ends, f"{module.__name__} has no {name}"
    lines = source.splitlines(keepends=True)
    return source[start : sum(len(line) for line in lines[: ends[0]])]


#: The stages that publish onto a served name, and the call that publishes it.
#: Every one has to commit the cancel token: once the artifact is on the served
#: name a cancel cannot take it back, so recording the row as "cancelled" is a
#: lie about the file on disk -- and for the two sheet kinds it is worse than a
#: lie, because ``_discard_artifacts`` then deletes a sheet the user can see.
PUBLISHERS = [
    ("warlock._q_rig", "_rig", "finalize_rig"),
    ("warlock._q_sprite", "_pixel_sheet", "_publish_text"),
    ("warlock._q_sprite", "_retexture", "os.replace"),
    ("warlock._q_troupe", "_charsheet", "_publish_text"),
]


@pytest.mark.parametrize("module,func,publish", PUBLISHERS, ids=lambda v: str(v))
def test_every_served_publish_commits_the_cancel_token(module, func, publish):
    """A scan, because two of these four had no commit and nothing said so.

    ``_retexture`` in particular has no cheap end-to-end harness -- it wants a
    resident SDXL pipe, ten Blender renders and a texture bake -- and it is the
    one whose publish is *irreversible*: it replaces another job's served
    ``model.glb``, so the skin the user had is simply gone. Reading the source
    is the honest way to hold that, in the idiom of the twenty-odd other
    structural scans in this suite.
    """
    import importlib

    body = _function_source(importlib.import_module(module), func)
    call = publish.split(".")[-1]
    assert call in body, f"{module}.{func} no longer publishes through {publish}"
    commit_at = body.find("_cancel.commit()", body.index(call))
    assert commit_at != -1, (
        f"{module}.{func} publishes a served artifact and never commits the "
        "cancel token; a cancel in its tail records the row as cancelled with "
        "the artifact on disk"
    )


async def test_a_cancel_after_a_character_sheet_is_published_records_it_as_done(
    worker, monkeypatch
):
    """The sharpest form of the missing commit.

    ``_discard_artifacts``'s sheet branch deletes the *served* pair, not temps
    -- so a cancel landing after the sidecar has been published recorded the
    row "cancelled" and then unlinked a character sheet Troupe was already
    drawing. Two hundred and fifty-six rendered frames, gone, with the row
    saying it never happened.
    """
    from PIL import Image

    from warlock import queue as queue_mod
    from warlock.pipelines import charsheet, pixelize, pixelsheet
    from warlock.pipelines import sheet as sheetlib

    source = worker.store.create("image", "a ranger", {"seed": 1}, stage="model")
    source_dir = worker.config.job_dir(source)
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "model.glb").write_bytes(b"fake-glb")
    (source_dir / "rig.glb").write_bytes(b"fake-rig")
    worker.store.set_status(source, "done")

    async def fake_render(glb, layout, cells, *, pack_target, **kwargs):
        pack_target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (8, 8), (1, 2, 3, 255)).save(pack_target, "PNG")
        # Three values, not two: ``_render_charsheet`` grew the projected
        # sockets on 2026-09-05 for the character effects pass. Empty here
        # because this fixture has no ``character.json``, which is the same
        # thing a sheet built before characters existed carries -- so this fake
        # keeps standing for the case the test is actually about.
        return {"pivot": (0.0, 0.0)}, {}, {}

    async def no_roots(*_args):
        return {}, None

    monkeypatch.setattr(worker, "_render_charsheet", fake_render)
    monkeypatch.setattr(worker, "_charsheet_roots", no_roots)
    monkeypatch.setattr(
        pixelsheet, "quantize_shared", lambda atlas, colors: (atlas, ["#010203"])
    )
    monkeypatch.setattr(
        pixelize,
        "pixelize_atlas",
        lambda atlas, **kwargs: (Image.new("RGBA", (8, 8), (1, 2, 3, 255)), {}),
    )
    monkeypatch.setattr(sheetlib, "measure_trim", lambda _image: None)

    sheet_id = rigging.new_id()
    job_id = worker.store.create(
        "charsheet",
        None,
        {"source_job": source, "sheet_id": sheet_id, "logical_size": 16},
    )

    # The cancel lands the instant the sidecar -- the completion marker -- is
    # on disk, which is the window the commit exists to close.
    real_publish = queue_mod._publish_text

    def publish_then_cancel(path, text):
        real_publish(path, text)
        if worker._cancel is not None:
            worker._cancel.event.set()

    monkeypatch.setattr(queue_mod, "_publish_text", publish_then_cancel)

    worker.start()
    await _wait_until(
        lambda: worker.store.get(job_id)["status"] in ("done", "error", "cancelled"),
        timeout=60.0,
    )
    await worker.shutdown()

    row = worker.store.get(job_id)
    assert row["status"] == "done", row["error"]
    assert rigging.sheet_path(source_dir, sheet_id).exists()
    assert rigging.sheet_png_path(source_dir, sheet_id).exists()
    assert charsheet  # imported for the fakes above to be the right shapes


# --- stable ids --------------------------------------------------------------


@pytest.mark.parametrize("kind", ["sheet", "charsheet"])
async def test_a_sheet_row_without_an_id_is_refused_rather_than_given_one(
    worker, kind
):
    """``_discard_artifacts`` deletes this kind's served pair by ``sheet_id``,
    a pure function of a stable id. That is safe only while every door mints a
    fresh one -- and both workers used to paper over a missing id with ``or
    new_id()``, which is exactly how the assumption would have been broken
    without anything going red."""
    source = worker.store.create("text", "a knight", {"seed": 1})
    source_dir = worker.config.job_dir(source)
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "model.glb").write_bytes(b"fake-glb")
    (source_dir / "rig.glb").write_bytes(b"fake-rig")
    worker.store.set_status(source, "done")

    job_id = worker.store.create(kind, None, {"source_job": source})
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "error")
    await worker.shutdown()
    assert "sheet_id" in (worker.store.get(job_id)["error"] or "")


def test_no_worker_mints_a_sheet_id_of_its_own() -> None:
    """The rule, as a scan, because the refusals above only cover the two
    spellings that exist today."""
    root = Path(__file__).resolve().parents[1] / "src" / "warlock"
    offenders = []
    for path in sorted(root.glob("_q_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
                continue
            text = ast.unparse(node)
            if "sheet_id" in text and "new_id()" in text:
                offenders.append(f"{path.name}:{node.lineno}: {text}")
    assert not offenders, (
        "a worker minting its own sheet id makes _discard_artifacts unsafe: "
        + "; ".join(offenders)
    )


# --- the failure the user actually sees --------------------------------------


def test_an_autosave_failure_says_it_was_the_autosave() -> None:
    """``_collect_tasks`` routes a failure by key prefix, and there was no
    branch for ``journal:``.

    That mattered little while the journal recorded its mark on *submit*: a
    failed write was invisible either way. It matters now that the mark waits
    for the write, because this toast is the only signal the user gets that the
    crash copy the app promised does not exist -- and the sentence it carried
    was "Something went wrong; see the log for details", which names neither
    the autosave nor what to do about it.
    """
    from warlock.studio import main
    from warlock.studio.state import AppState

    toasts: list[tuple[str, str, Any]] = []
    done = SimpleNamespace(
        key="journal:inker:tab-1",
        ok=False,
        error=OSError("no space left on device"),
        message="Something went wrong; see the log for details.",
        action="log",
        tag=None,
    )
    app = SimpleNamespace(
        app_ctx=SimpleNamespace(
            tasks=SimpleNamespace(poll=lambda: [done]),
            state=AppState(),
            toast=lambda text, level="info", action=None: toasts.append(
                (text, level, action)
            ),
        )
    )
    main.App._collect_tasks(app)

    assert len(toasts) == 1
    text, level, action = toasts[0]
    assert "Autosave" in text and "recovery copy" in text
    assert level == "error" and action == "log"


# --- the store ---------------------------------------------------------------


def test_a_backup_does_not_hold_the_store_lock(tmp_path, monkeypatch):
    """``backup_to`` was the one lock-then-blocking-IO path in the store.

    The frame thread reads this store directly, so a library backup -- a page
    walk over the whole database, out to disk -- queued the job list behind it
    for the duration. ``deferred_commits`` declines to hold the lock across the
    caller's file writes for exactly this reason and says so.
    """
    store = JobStore(tmp_path / "jobs.sqlite")
    try:
        store.create("text", "a knight", {"seed": 1})
        entered, release = threading.Event(), threading.Event()
        real_connect = db_mod.sqlite3.connect

        class _Slow:
            def __init__(self, conn: Any) -> None:
                self._conn = conn

            def backup(self, target: Any) -> None:
                entered.set()
                assert release.wait(10.0)
                self._conn.backup(target)

            def close(self) -> None:
                self._conn.close()

        def connect(path, *args, **kwargs):
            conn = real_connect(path, *args, **kwargs)
            return _Slow(conn) if Path(path) == Path(store._path) else conn

        monkeypatch.setattr(db_mod.sqlite3, "connect", connect)
        dest = tmp_path / "backup.sqlite"
        backup = threading.Thread(target=lambda: store.backup_to(dest), daemon=True)
        backup.start()
        assert entered.wait(10.0), "the backup is in flight"

        answered = threading.Event()
        threading.Thread(
            target=lambda: (store.list(), answered.set()), daemon=True
        ).start()
        assert answered.wait(3.0), (
            "the store answered a reader while a backup was walking it out to "
            "disk -- it did not, and the frame thread was the reader"
        )
        release.set()
        backup.join(10.0)
        assert dest.exists()
    finally:
        store.close()


# --- writers with no callers -------------------------------------------------


def test_the_viewport_has_no_write_in_place_helper() -> None:
    """``capture.save_png`` wrote straight onto the path it was handed -- the
    opposite of the rule every real capture path follows -- and the only caller
    it had anywhere was the test that tested it. A trap sitting in a module
    four callers import."""
    from warlock.studio.viewer import capture

    assert not hasattr(capture, "save_png")
    assert hasattr(capture, "png_bytes")
