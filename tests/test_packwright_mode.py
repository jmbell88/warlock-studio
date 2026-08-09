"""Packwright's controller.

Two things carry this file. The save rules every editor here shares -- a failed
save clears the lock, and the head a save records is the one the encode wrote.
And one of its own: **the repack flag**, which is the ``findings_dirty`` lesson
applied a second time. ``TaskRunner.submit`` refuses a key already in flight and
nothing re-arms it, so a flag cleared regardless of whether the submit was
accepted silently drops every edit made while a pack was running.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from warlock.studio import packwright_mode
from warlock.studio.packwright import wpack
from warlock.studio.packwright.sources import Sprite


class FakeCtx:
    def __init__(self, svc: Any = None, *, accept: bool = True) -> None:
        self.svc = svc
        self.state = _AppState()
        self.settings = _Settings()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.confirms = _Confirms()
        self.cache = _Cache()
        self.viewer = None
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


class _AppState:
    def __init__(self) -> None:
        self.packwright = None
        self.inker = None
        self.mode = "home"
        self.preview: dict[str, Any] = {}


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _Confirms:
    def __init__(self) -> None:
        self.pending: Any = None

    def ask(self, confirm: Any) -> None:
        self.pending = confirm


class _Cache:
    def __init__(self) -> None:
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1


class _Done:
    def __init__(self, key: str, result: Any = None, message: str = "") -> None:
        self.key = key
        self.result = result
        self.message = message


def _sprite(key: str, w: int = 8, h: int = 6) -> Sprite:
    pixels = np.zeros((h, w, 4), dtype=np.uint8)
    pixels[1:-1, 1:-1] = (200, 30, 30, 255)
    return Sprite(key=key, name=key, pixels=pixels)


def _tab(ctx: FakeCtx, *, sources: int = 3) -> Any:
    tab = packwright_mode.new_document(ctx)
    for index in range(sources):
        tab.doc.add_source(_sprite(f"s{index}"))
    tab.doc.mark_saved()
    tab.pack_dirty = True
    return tab


def _pack(ctx: FakeCtx, tab: Any) -> None:
    packwright_mode.request_pack(ctx, tab)
    packwright_mode.on_task_done(ctx, _Done(f"packwright-pack:{tab.uid}", ctx.result))


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


# --- packing ------------------------------------------------------------------


def test_a_pack_produces_a_layout_and_an_atlas():
    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)
    assert tab.layout is not None and tab.atlas is not None
    assert len(tab.layout.frames) == 3
    assert tab.atlas.shape[2] == 4
    assert not tab.packing and not tab.pack_dirty


def test_adopting_a_pack_bumps_the_generation_exactly_once():
    """The texture cache keys on it, so it has to move for a repack and not for
    anything else."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)
    first = tab.pack_generation
    tab.pack_dirty = True
    _pack(ctx, tab)
    assert tab.pack_generation == first + 1


def test_the_dirty_flag_is_cleared_only_when_the_submit_is_accepted():
    """The regression this flag exists for. The runner refuses a key already in
    flight; clearing regardless would drop the edit that arrived while the
    previous pack was running, permanently, because nothing re-arms it."""
    ctx = FakeCtx(accept=False)
    tab = _tab(ctx)
    packwright_mode.request_pack(ctx, tab)
    assert ctx.submitted == [f"packwright-pack:{tab.uid}"]
    assert tab.pack_dirty is True
    assert tab.packing is False


def test_a_clean_document_does_not_resubmit_every_frame():
    """``pump`` runs from the preview pane's draw, so it is called sixty times a
    second -- it has to be free when there is nothing to do."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)
    ctx.submitted.clear()
    for _ in range(10):
        packwright_mode.pump(ctx)
    assert ctx.submitted == []


def test_every_document_edit_re_arms_the_pack():
    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)

    packwright_mode.set_settings(ctx, tab, padding=6)
    assert tab.pack_dirty
    _pack(ctx, tab)

    packwright_mode.remove_source(ctx, tab.doc.sources[0].uid, tab)
    assert tab.pack_dirty


def test_a_failed_pack_clears_packing_and_says_why():
    """An empty items list that looks like success is the worst outcome of a
    pack that could not fit."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.packing = True
    packwright_mode.on_task_failed(
        ctx, _Done(f"packwright-pack:{tab.uid}", message="does not fit")
    )
    assert not tab.packing
    assert tab.pack_error == "does not fit"


def test_an_empty_document_packs_to_nothing_rather_than_raising():
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    tab.pack_dirty = True
    packwright_mode.request_pack(ctx, tab)
    assert ctx.submitted == []
    assert tab.layout is None and not tab.pack_dirty


def test_impossible_settings_are_refused_with_the_reason():
    """``PackSettings`` validates on construction, so the pane never has to."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.set_settings(ctx, tab, padding=1, extrude=4)
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
    assert "twice extrude" in ctx.toasts[-1][0]
    assert tab.doc.settings.extrude == 0


# --- sources ------------------------------------------------------------------


def test_a_batch_skips_what_is_already_there_rather_than_refusing():
    """Dropping twenty files of which one is already in the atlas should add
    nineteen. The document's own refusal stays the authority on what may
    coexist; this is the caller deciding what to ask for."""
    ctx = FakeCtx()
    tab = _tab(ctx, sources=2)
    added = packwright_mode._add_sprites(ctx, tab, [_sprite("s0"), _sprite("new")])
    assert added == 1
    assert len(tab.doc.sources) == 3


def test_adding_with_nothing_open_says_so():
    ctx = FakeCtx()
    packwright_mode.ask_add_sources(ctx)
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
    assert ctx.submitted == []


def test_an_inker_document_contributes_one_sprite_per_frame():
    from warlock.studio.inker.document import Document

    ctx = FakeCtx()
    tab = _tab(ctx, sources=0)
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[:] = (255, 0, 0, 255)
    doc.add_frame(copy=True)
    inker_tab = type("T", (), {"doc": doc, "title": "walk.ora", "uid": "pd1"})()

    packwright_mode.add_inker_document(ctx, inker_tab)
    assert [s.key for s in tab.doc.sprites()] == ["walk#frame0000", "walk#frame0001"]


def test_removing_the_selected_source_clears_the_selection():
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = packwright_mode.ensure(ctx)
    uid = tab.doc.sources[0].uid
    state.selected = uid
    packwright_mode.remove_source(ctx, uid, tab)
    assert state.selected is None


# --- saving -------------------------------------------------------------------


def test_a_save_writes_the_file_and_marks_the_document_clean(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.doc.set_settings(padding=6)
    path = tmp_path / "atlas.wpack"
    packwright_mode.save_to(ctx, tab, path)
    packwright_mode.on_task_done(ctx, _Done(f"packwright-save:{tab.uid}", ctx.result))

    assert path.exists() and not tab.dirty and not tab.saving
    assert tab.title == "atlas.wpack"
    assert wpack.read_wpack(path.read_bytes()).settings.padding == 6


def test_a_save_records_the_head_the_encode_wrote():
    ctx = FakeCtx()
    tab = _tab(ctx)
    head = tab.doc.history.head
    tab.doc.set_settings(padding=8)
    packwright_mode.on_task_done(
        ctx, _Done(f"packwright-save:{tab.uid}", {"head": head, "path": ""})
    )
    assert tab.dirty


def test_a_failed_save_clears_the_lock():
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True
    packwright_mode.on_task_failed(ctx, _Done(f"packwright-save:{tab.uid}"))
    assert not tab.saving


def test_a_refused_submit_clears_the_lock_too(tmp_path):
    ctx = FakeCtx(accept=False)
    tab = _tab(ctx)
    packwright_mode.save_to(ctx, tab, tmp_path / "a.wpack")
    assert not tab.saving


# --- exporting ----------------------------------------------------------------


def test_a_grid_export_writes_the_png_the_json_and_a_tsx(tmp_path, monkeypatch):
    from warlock.studio import dialogs

    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.set_settings(ctx, tab, mode="grid")
    _pack(ctx, tab)
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "out.png")

    packwright_mode.export_files(ctx, tab)
    packwright_mode.on_task_done(ctx, _Done(f"packwright-export:{tab.uid}", ctx.result))
    assert (tmp_path / "out.png").exists()
    assert (tmp_path / "out.json").exists()
    # A grid pack *is* a tileset, which is the whole payoff of the two modes
    # being genuinely different.
    assert (tmp_path / "out.tsx").exists()


def test_a_maxrects_export_writes_no_tsx(tmp_path, monkeypatch):
    from warlock.studio import dialogs

    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.set_settings(ctx, tab, mode="maxrects")
    _pack(ctx, tab)
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "out.png")

    packwright_mode.export_files(ctx, tab)
    packwright_mode.on_task_done(ctx, _Done(f"packwright-export:{tab.uid}", ctx.result))
    assert (tmp_path / "out.json").exists()
    assert not (tmp_path / "out.tsx").exists()


def test_exporting_before_anything_is_packed_says_so():
    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.export_files(ctx, tab)
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
    assert not tab.saving


# --- the guard ----------------------------------------------------------------


def test_the_guard_asks_once_for_however_many_are_dirty():
    """One question for all of them: asking per document would put the user in
    front of a queue they answered the same way each time."""
    ctx = FakeCtx()
    first = _tab(ctx)
    first.doc.set_settings(padding=6)
    calls: list[str] = []
    assert packwright_mode.guard(ctx, "quit", lambda: calls.append("go")) is False
    assert calls == []
    assert "One atlas has" in ctx.confirms.pending.message

    second = _tab(ctx)
    second.doc.set_settings(padding=8)
    packwright_mode.guard(ctx, "quit", lambda: calls.append("go"))
    assert "2 atlases have" in ctx.confirms.pending.message


def test_the_guard_is_silent_before_the_mode_has_ever_been_opened():
    ctx = FakeCtx()
    calls: list[str] = []
    packwright_mode.guard(ctx, "quit", lambda: calls.append("go"))
    assert calls == ["go"] and ctx.state.packwright is None


def test_closing_a_dirty_tab_asks_first():
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.doc.set_settings(padding=6)
    packwright_mode.close_tab(ctx, tab.uid)
    assert ctx.confirms.pending is not None
    ctx.confirms.pending.on_confirm()
    assert packwright_mode.ensure(ctx).get(tab.uid) is None


# --- keys ---------------------------------------------------------------------


def test_r_forces_a_repack(monkeypatch):
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)
    assert packwright_mode.handle_key(ctx, event) is True
    assert tab.pack_dirty


def test_undo_re_arms_the_pack(monkeypatch):
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx)
    _pack(ctx, tab)
    monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_CTRL)
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_z)
    assert packwright_mode.handle_key(ctx, event) is True
    assert tab.pack_dirty


def test_delete_removes_the_selected_source(monkeypatch):
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = packwright_mode.ensure(ctx)
    state.selected = tab.doc.sources[1].uid
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DELETE)
    assert packwright_mode.handle_key(ctx, event) is True
    assert len(tab.doc.sources) == 2


def test_a_key_release_is_never_consumed():
    import pygame

    ctx = FakeCtx()
    _tab(ctx)
    up = pygame.event.Event(pygame.KEYUP, key=pygame.K_r)
    assert packwright_mode.handle_key(ctx, up) is False


def test_recent_files_persist():
    ctx = FakeCtx()
    doc = packwright_mode.new_document(ctx).doc
    path = Path("/tmp/atlas.wpack")
    packwright_mode.adopt(ctx, doc, path=path)
    assert packwright_mode.recent_paths(ctx) == [str(path)]
