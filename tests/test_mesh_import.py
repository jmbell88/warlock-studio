"""Bringing a mesh you already have into the library, and saying why you can't.

Three defects found together on 2026-08-30, by a user who dropped a rigged GLB
onto Clay and got nothing they could act on.

**There was no way to import a premade mesh at all.** Geometry entered the
library exactly two ways -- reconstructed by the pipeline, or built in Clay and
handed back through ``jobs.import_mesh``. A ``.glb`` a user already had could
reach only Clay, which converts it into an editable *document* rather than an
asset, and refuses a rigged one outright because it has no skinning. So the
supplied-base-mesh path that Troupe's whole intake assumes had no door, while
``import_mesh`` sat written, tested, and reachable from one caller.

**Clay's refusal never reached the screen.** ``OpError``'s docstring says it is
"a user-facing refusal: shown as a toast... the message is the whole user
interface for it" -- and ``tasks.py`` kept an error's own message only for
``ServiceError``, so every Clay open and import failure arrived as *Something
went wrong; see the log for details.* The sentence naming the cause went to
``warlock.log`` alone, which is where the user had to go to find it.

**And the refusal named a destination that cannot take the file.** It said
"Open it in Create instead"; Create accepts ``DROPPABLE_IMAGES`` and refuses a
``.glb`` with a sentence about images.

Run with: uv run pytest tests/test_mesh_import.py -n 0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from warlock.service.errors import Invalid
from warlock.studio.clay.elements import OpError
from warlock.studio.panes import library

# --- the refusal reaches the user -------------------------------------------


def _raise(exc):
    raise exc


def _drain(runner, count, timeout=5.0):
    import time

    out = []
    deadline = time.monotonic() + timeout
    while len(out) < count and time.monotonic() < deadline:
        out.extend(runner.poll())
        time.sleep(0.01)
    assert len(out) == count, f"only {len(out)} of {count} tasks finished"
    return out


def test_an_op_error_keeps_its_own_sentence():
    """The defect, at the layer that dropped it.

    ``OpError`` is a ``ValueError`` and was therefore not a ``ServiceError``,
    so it fell to the branch that replaces the message wholesale. Clay writes
    these refusals to be read -- "this GLB is rigged, and Clay has no
    skinning" names both the cause and what to do -- and the user saw none of
    it.
    """
    from warlock.studio.tasks import TaskRunner

    runner = TaskRunner(workers=1)
    try:
        runner.submit("clay-import:x", _raise, OpError("This GLB is rigged."))
        (done,) = _drain(runner, 1)
    finally:
        runner.shutdown()

    assert done.message == "This GLB is rigged."
    # Never the log: an ``OpError`` is a refusal written for this exact case,
    # and there is nothing further to find there. ``Failed`` is the one that
    # points at the log, and it is a ServiceError.
    assert done.action is None


def test_a_plain_exception_still_defers_to_the_log():
    """The generic branch is intact -- this widened one case, not the rule.

    An unexpected exception has no sentence worth showing, and inventing one
    from ``str(exc)`` would put a traceback fragment in a toast.
    """
    from warlock.studio.tasks import TaskRunner

    runner = TaskRunner(workers=1)
    try:
        runner.submit("whatever", _raise, RuntimeError("index out of range"))
        (done,) = _drain(runner, 1)
    finally:
        runner.shutdown()

    assert done.action == "log"
    assert "index out of range" not in done.message


def test_clay_sends_a_rigged_mesh_somewhere_that_can_take_it():
    """The refusal's remedy has to exist.

    It said "Open it in Create instead", and Create takes images only -- so the
    one sentence the user got named a dead end. Asserted on the words rather
    than by driving the UI because the words *are* the interface here, which is
    what ``OpError``'s docstring says.
    """
    from warlock.studio.clay import glbimport

    source = Path(glbimport.__file__).read_text(encoding="utf-8")
    refusal = source.split("model.skins", 1)[1].split(")", 1)[0]
    assert "Library" in refusal
    assert "Create" not in refusal, "Create accepts images, not meshes"


# --- the door itself ---------------------------------------------------------


def test_the_import_key_is_one_key():
    """Unkeyed by path, so a double-click cannot open two pickers.

    ``submit`` deduplicates on the key while a task is in flight, which is the
    whole mechanism -- a key carrying the path would defeat it for the case it
    exists to cover.
    """
    calls: list[tuple] = []
    ctx = _ctx(calls)
    library.pick_and_import_mesh(ctx)
    library.import_mesh_path(ctx, Path("a.glb"))
    assert [c[0] for c in calls] == [library.IMPORT_MESH_KEY] * 2


def test_a_cancelled_picker_imports_nothing(monkeypatch):
    """``None`` is a cancel and says nothing.

    ``dialogs.open_file`` returns ``None`` only for a cancel -- a picker that
    failed to open raises -- so this branch cannot swallow a real failure.
    """
    monkeypatch.setattr(library.dialogs, "open_file", lambda *a, **k: None)
    called: list[Any] = []
    monkeypatch.setattr(library.svc_jobs, "import_mesh", lambda *a, **k: called.append(a))

    assert library._pick_and_import(_ctx([])) is None
    assert called == []


def test_a_mesh_over_the_ceiling_is_refused_before_it_is_read(tmp_path, monkeypatch):
    """Checked against the file, not the bytes.

    ``import_mesh`` re-checks what it is handed, so this is a cheaper first
    gate rather than the rule -- but reading a two-gigabyte file into memory to
    discover it is too big is how a refusal becomes a swap storm.
    """
    from warlock.service.validation import MAX_MESH_BYTES

    big = tmp_path / "huge.glb"
    big.write_bytes(b"x")
    monkeypatch.setattr(
        Path, "stat", lambda self: type("S", (), {"st_size": MAX_MESH_BYTES + 1})()
    )
    read: list[Any] = []
    monkeypatch.setattr(Path, "read_bytes", lambda self: read.append(self) or b"")

    with pytest.raises(Invalid) as exc:
        library._read_and_import(_ctx([]), big)
    assert "MB" in str(exc.value)
    assert read == [], "the file was read before the size was checked"


def test_an_imported_mesh_becomes_an_ordinary_asset(svc, tmp_path):
    """The payoff, end to end through the real service door.

    Named and prompted from the file's stem: the name is what the library
    shows and the prompt is what every "what was this?" reader falls back to,
    so a file called ``knight_base`` answers both rather than leaving one
    blank.
    """
    from conftest import _tiny_glb

    mesh = tmp_path / "knight_base.glb"
    mesh.write_bytes(_tiny_glb())

    out = library._read_and_import(_ctx([], svc=svc), mesh)

    job = svc.store.get(out["id"])
    assert job is not None
    assert job["status"] == "done"
    assert job["prompt"] == "knight_base"
    assert (svc.job_dir(out["id"]) / "model.glb").is_file()


# --- the drop router ---------------------------------------------------------


def test_a_dropped_glb_is_imported_on_home_and_in_the_library():
    """Read off the source, ``test_sirens_mode.py``'s rule for this router.

    Home and the Library only. Create is deliberately excluded: a mesh dropped
    mid-generation is ambiguous between "start from this" and "put this in my
    library", and the branch below it already answers that for images.
    """
    import inspect

    from warlock.studio import main

    source = inspect.getsource(main.App._on_drop)
    branch = source.split('ctx.state.mode in ("home", "library")', 1)[1]
    branch = branch.split("DROPPABLE_IMAGES", 1)[0]
    assert "library.import_mesh_path" in branch
    assert ".glb" in branch


def _ctx(calls: list, svc: Any = None) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        svc=svc,
        submit=lambda key, fn, *args: calls.append((key, fn, args)),
        toast=lambda *a, **k: None,
    )
