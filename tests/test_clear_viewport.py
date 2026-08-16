"""Clearing the Create canvas, and making the clear survive the next tick.

The interesting half is not the emptying. ``App._sync_viewer`` runs from four
places and again off the cache tick within a few seconds, and it decides what to
show from the *selection* -- so a plain ``viewer.clear()`` is undone almost
immediately and the button looks broken. What stops it is the short-circuit
``viewer.path == wanted``: the clear leaves the path naming what would be
loaded, so the sync agrees there is nothing to do. That is ``_review_load``'s
clear-then-pin idiom applied at the other end of the same mechanism, and these
tests drive a real ``_sync_viewer`` tick over the top of the clear to assert it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from warlock.studio import main
from warlock.studio.state import AppState

JOB = {"id": "job-1", "status": "done", "kind": "mesh", "files": ["input.png", "model.glb"]}


class _FakeViewer:
    """Only the members ``_clear_viewport`` and ``_sync_viewer`` touch."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self.pending: Path | None = None
        self.reference: Any = None
        self.model: Any = None
        self.pose_mode = False
        self.comparing = False
        self.strips_cancelled = 0
        self.exited_compare = 0
        self.loaded_references: list[Path] = []

    @property
    def has_model(self) -> bool:
        return self.model is not None

    def clear(self) -> None:
        self.model = None
        self.path = None
        self.pending = None

    def clear_reference(self) -> None:
        self.reference = None

    def load_reference(self, path: Path) -> None:
        self.loaded_references.append(Path(path))
        self.reference = object()

    def cancel_sheet_strip(self) -> None:
        self.strips_cancelled += 1

    def exit_compare(self) -> None:
        self.exited_compare += 1

    def parse_model(self, path: Path) -> Any:  # pragma: no cover - never run inline
        raise AssertionError("the parse must go to a task, not run here")


class _Ctx:
    def __init__(self, job_dir: Path) -> None:
        self.state = AppState()
        self.state.mode = "create"
        self.state.selected = JOB["id"]
        self._job_dir = job_dir
        self.submitted: list[tuple[str, Any]] = []
        self.toasts: list[str] = []
        self.cache = SimpleNamespace(get=lambda job_id: JOB if job_id == JOB["id"] else None)

    def job(self, job_id: str | None = None) -> Any:
        return JOB

    def job_dir(self, job_id: str) -> Path:
        return self._job_dir

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kw: Any) -> bool:
        self.submitted.append((key, tag))
        return True

    def toast(self, text: str, level: str = "info", *a: Any, **kw: Any) -> None:
        self.toasts.append(text)


class _FakeApp:
    """The two App methods under test, unbound over the fakes above."""

    _clear_viewport = main.App._clear_viewport
    _sync_viewer = main.App._sync_viewer
    _refresh_rig_side_data = staticmethod(lambda: None)

    def __init__(self, ctx: _Ctx) -> None:
        self.viewer = _FakeViewer()
        self.app_ctx = ctx


@pytest.fixture
def app(tmp_path):
    (tmp_path / "input.png").write_bytes(b"")
    (tmp_path / "model.glb").write_bytes(b"")
    ctx = _Ctx(tmp_path)
    return _FakeApp(ctx), ctx, tmp_path


def _at_mesh(ctx: _Ctx) -> None:
    ctx.state.create_stage = "mesh"


def _at_reference(ctx: _Ctx) -> None:
    ctx.state.create_stage = "reference"


# --- the mesh stage ----------------------------------------------------------


def test_clearing_the_mesh_empties_the_viewer(app):
    fake, ctx, _dir = app
    _at_mesh(ctx)
    fake.viewer.model = object()
    fake.viewer.path = _dir / "model.glb"

    fake._clear_viewport()

    assert not fake.viewer.has_model


def test_the_cleared_mesh_stays_cleared_through_a_sync_tick(app):
    """The pin. Without it the next tick decides the selection "should" be
    showing model.glb and dispatches a parse for it."""
    fake, ctx, dir_ = app
    _at_mesh(ctx)
    fake.viewer.model = object()
    fake.viewer.path = dir_ / "model.glb"

    fake._clear_viewport()
    fake._sync_viewer()

    assert not fake.viewer.has_model
    assert ctx.submitted == []
    assert fake.viewer.pending is None


def test_clearing_the_mesh_drops_a_parse_still_in_flight(app):
    """``_adopt_model`` checks its result against ``pending``; leaving it set
    would land the parse on the emptied viewport a moment later."""
    fake, ctx, dir_ = app
    _at_mesh(ctx)
    fake.viewer.pending = dir_ / "model.glb"

    fake._clear_viewport()

    assert fake.viewer.pending is None


def test_clearing_the_mesh_cancels_a_strip_in_flight(app):
    """A strip renders off the mesh that is about to go."""
    fake, ctx, _dir = app
    _at_mesh(ctx)
    fake.viewer.model = object()

    fake._clear_viewport()

    assert fake.viewer.strips_cancelled == 1


def test_clearing_the_mesh_leaves_a_comparison_reconciled(app):
    fake, ctx, _dir = app
    _at_mesh(ctx)
    fake.viewer.model = object()
    ctx.state.comparing = "other-job"

    fake._clear_viewport()

    assert ctx.state.comparing is None
    assert fake.viewer.exited_compare == 1


# --- the reference stage -----------------------------------------------------


def test_clearing_the_reference_empties_only_the_reference(app):
    fake, ctx, dir_ = app
    _at_reference(ctx)
    fake.viewer.reference = object()
    fake.viewer.model = object()
    fake.viewer.path = dir_ / "input.png"

    fake._clear_viewport()

    assert fake.viewer.reference is None
    assert fake.viewer.has_model


def test_the_cleared_reference_stays_cleared_through_a_sync_tick(app):
    fake, ctx, dir_ = app
    _at_reference(ctx)
    fake.viewer.reference = object()
    fake.viewer.path = dir_ / "input.png"

    fake._clear_viewport()
    fake._sync_viewer()

    assert fake.viewer.reference is None
    assert fake.viewer.loaded_references == []


# --- the pin releases on its own --------------------------------------------


def test_changing_stage_brings_the_asset_back(app):
    """The pin is ``path == wanted`` and nothing more, so it lapses the moment
    ``wanted`` changes -- which is what makes this need no cleanup anywhere."""
    fake, ctx, dir_ = app
    _at_reference(ctx)
    fake.viewer.reference = object()
    fake.viewer.path = dir_ / "input.png"
    fake._clear_viewport()

    _at_mesh(ctx)
    fake._sync_viewer()

    assert ctx.submitted and ctx.submitted[0][1] == dir_ / "model.glb"


def test_clearing_does_nothing_outside_a_viewport_mode(app):
    """Review borrows this same viewer for its sweep units; a clear routed
    from anywhere else would empty a canvas this helper knows nothing about."""
    fake, ctx, _dir = app
    ctx.state.mode = "review"
    fake.viewer.model = object()

    fake._clear_viewport()

    assert fake.viewer.has_model
    assert fake.viewer.strips_cancelled == 0
