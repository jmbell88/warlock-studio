"""Plotter's side of an AI ground set: submit, watch, adopt.

The wiring, not the pixels. Three things have to hold or the feature is worse
than not having it: the submission goes on its own key and parks a job id on the
tab; the watch clears that id **before** it submits the adoption, so a refused
adoption is a refusal rather than a refusal sixty times a second; and an atlas
painted for another map's geometry is refused by name instead of repainting
every cell wrong while every gid in it stays valid.

``FakeCtx`` never routes a submit through ``on_task_done`` -- it runs the
callable inline and hands back the result -- so every handler here is driven
directly, which is also what the real app does one frame later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image

from warlock.studio import plotter_mode
from warlock.studio.panes import plotter_tileset as pane
from warlock.studio.plotter import blob


class FakeCtx:
    def __init__(self, svc: Any = None) -> None:
        self.svc = svc
        self.state = _AppState()
        self.settings = _Settings()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.cache = _Cache()
        self.viewer = None
        self.result: Any = None

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        self.result = run(*args)
        return True

    def toast(self, message: str, kind: str = "info") -> None:
        self.toasts.append((message, kind))


class _AppState:
    def __init__(self) -> None:
        self.plotter = None
        self.mode = "home"
        self.previous_mode = "home"
        self.mode_observed = "home"
        self.preview: dict[str, Any] = {}


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _Cache:
    def __init__(self) -> None:
        self.invalidated = 0
        self.rows: dict[str, dict] = {}

    def invalidate(self) -> None:
        self.invalidated += 1

    def get(self, job_id: str | None) -> dict | None:
        return None if job_id is None else self.rows.get(job_id)


class _Done:
    def __init__(self, key: str, result: Any = None) -> None:
        self.key = key
        self.result = result
        self.message = ""


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


def _tab(ctx: FakeCtx, *, tile_w: int = 16, tile_h: int = 16) -> Any:
    return plotter_mode.new_document(ctx, (4, 4, tile_w, tile_h))


def _rows(count: int = 2) -> list[dict]:
    names = ("stone", "water", "lava")
    return [
        {
            "name": names[i],
            "material": f"{names[i]} surface",
            "edge": "",
            "color": [30 + i * 30, 90, 140, 255],
        }
        for i in range(count)
    ]


def _painted(svc, job_id: str, *, terrains: int = 2, tile_w: int = 16, tile_h: int = 16):
    """A finished ground job with a real 47-column atlas on disk."""
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((terrains * tile_h, blob.TILE_COUNT * tile_w, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    Image.fromarray(pixels, "RGBA").save(job_dir / "input.png")


# --- submitting --------------------------------------------------------------


def test_painting_goes_on_its_own_key_and_queues_a_job(svc):
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    plotter_mode.paint_ground_set(ctx, theme="a damp dungeon", rows=_rows(2))

    assert ctx.submitted == [f"plotter-ground:{tab.uid}"]
    assert ctx.result["textures"] == 4
    row = svc.store.get(ctx.result["ground_job"])
    assert row["kind"] == "ground_set"
    # The map's geometry is captured at submit time, not read at adoption.
    assert row["params"]["ground"]["tile_w"] == 16


def test_the_tab_remembers_the_job_it_is_waiting_on(svc):
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    plotter_mode.paint_ground_set(ctx, theme="a damp dungeon", rows=_rows(2))
    plotter_mode.on_task_done(ctx, _Done(f"plotter-ground:{tab.uid}", ctx.result))

    assert tab.ground_job == ctx.result["ground_job"]
    # And the map is still editable: painting takes minutes and there is no
    # reason to freeze the document for them.
    assert not tab.busy


def test_painting_with_no_map_open_says_so(svc):
    ctx = FakeCtx(svc)
    plotter_mode.ensure(ctx)
    plotter_mode.paint_ground_set(ctx, theme="a theme", rows=_rows(1))
    assert ctx.submitted == []
    assert ctx.toasts and ctx.toasts[-1][1] == "error"


# --- watching ----------------------------------------------------------------


def test_a_finished_job_is_adopted_as_a_terrain_set(svc):
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    plotter_mode.paint_ground_set(ctx, theme="a damp dungeon", rows=_rows(2))
    job_id = ctx.result["ground_job"]
    plotter_mode.on_task_done(ctx, _Done(f"plotter-ground:{tab.uid}", ctx.result))
    _painted(svc, job_id)
    ctx.cache.rows[job_id] = dict(svc.store.get(job_id), status="done")

    before = tab.doc.history.head
    pane.ground_watch(ctx, tab)
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    assert tab.ground_job is None
    assert len(tab.doc.tilesets) == 1
    arrived = tab.doc.tilesets[0].tileset
    # A *terrain* set, not forty-seven anonymous tiles -- which is the whole
    # reason this is not the library's "use as tileset" button.
    assert [entry.name for entry in arrived.terrains] == ["stone", "water"]
    assert arrived.columns == blob.TILE_COUNT
    # One undo step, so Ctrl+Z takes the whole set back.
    tab.doc.undo()
    assert tab.doc.tilesets == []
    assert tab.doc.history.head == before


def test_the_watch_costs_nothing_while_the_job_is_still_running(svc):
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    plotter_mode.paint_ground_set(ctx, theme="a theme", rows=_rows(1))
    job_id = ctx.result["ground_job"]
    tab.ground_job = job_id
    ctx.submitted.clear()
    for status in ("queued", "running"):
        ctx.cache.rows[job_id] = dict(svc.store.get(job_id), status=status)
        pane.ground_watch(ctx, tab)
    assert ctx.submitted == []
    assert tab.ground_job == job_id


def test_an_unknown_job_leaves_the_watch_alone(svc):
    """The cache is a page of history, so the row may simply not be there yet."""
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    tab.ground_job = "deadbeefcafe"
    pane.ground_watch(ctx, tab)
    assert tab.ground_job == "deadbeefcafe"


@pytest.mark.parametrize("status", ["error", "cancelled"])
def test_a_failed_or_cancelled_paint_clears_the_watch_and_says_so(svc, status):
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    plotter_mode.paint_ground_set(ctx, theme="a theme", rows=_rows(1))
    job_id = ctx.result["ground_job"]
    tab.ground_job = job_id
    ctx.submitted.clear()
    ctx.cache.rows[job_id] = dict(svc.store.get(job_id), status=status, error="boom")

    pane.ground_watch(ctx, tab)

    assert tab.ground_job is None
    assert ctx.submitted == []
    assert ctx.toasts


def test_a_refused_adoption_does_not_retry_every_frame(svc):
    """The watch is cleared *before* the adoption is submitted, so a geometry
    mismatch is a refusal rather than a refusal sixty times a second."""
    ctx = FakeCtx(svc)
    tab = _tab(ctx, tile_w=16, tile_h=16)
    plotter_mode.paint_ground_set(ctx, theme="a theme", rows=_rows(1))
    job_id = ctx.result["ground_job"]
    tab.ground_job = job_id
    ctx.cache.rows[job_id] = dict(svc.store.get(job_id), status="done")
    # The map was resized while it painted.
    tab.doc.set_tile_size(32, 32)
    ctx.submitted.clear()

    pane.ground_watch(ctx, tab)
    pane.ground_watch(ctx, tab)

    assert tab.ground_job is None
    assert ctx.submitted == []
    message = ctx.toasts[-1][0]
    assert "16x16" in message and "32x32" in message
    assert ctx.toasts[-1][1] == "error"


def test_a_watch_on_a_tab_that_is_no_longer_active_is_safe(svc):
    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    plotter_mode.paint_ground_set(ctx, theme="a theme", rows=_rows(1))
    job_id = ctx.result["ground_job"]
    tab.ground_job = job_id
    _painted(svc, job_id, terrains=1)
    ctx.cache.rows[job_id] = dict(svc.store.get(job_id), status="done")
    plotter_mode.close_tab(ctx, tab.uid)
    ctx.submitted.clear()

    # The watch went with the tab: the pane only ever draws the active one, so
    # there is nothing left to call it. The job itself finished into the library
    # exactly like any other, which is the whole of what a closed tab costs.
    assert plotter_mode.ensure(ctx).get(tab.uid) is None
    assert ctx.submitted == []


def test_a_ground_job_is_never_serialised_with_the_map(svc):
    """It is a watch, not a property of the document: it lives on the *tab*, so
    it cannot reach a file and cannot make the map ask to be saved."""
    from warlock.studio.plotter import wmap as wmaplib

    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    tab.doc.mark_saved()
    tab.ground_job = "deadbeefcafe"
    assert not tab.dirty
    assert b"deadbeefcafe" not in wmaplib.wmap_bytes(tab.doc)
