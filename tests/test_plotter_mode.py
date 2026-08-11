"""Plotter's controller: the rules the other editors had to learn first.

Nothing here is about tiles. It is about the two things that made a paint tab go
permanently read-only and permanently dirty, both of which follow from saving
being a *state* rather than a call that returns: a failed save has to clear that
state, and the head a save records has to be the one the encode actually wrote.
Plotter inherits both, and these tests are why the inheritance is real rather
than intended.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from warlock.studio import plotter_mode
from warlock.studio.plotter import gid, tmx, wmap
from warlock.studio.plotter.tilemap import MapObject, new_uid
from warlock.studio.plotter.tileset import Tileset


class FakeCtx:
    """Runs a submitted callable inline, so the test sees what the task thread
    would have done without needing one."""

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
        self.plotter = None
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
    def __init__(self, key: str, result: Any = None) -> None:
        self.key = key
        self.result = result
        self.message = ""


def _tileset(name: str = "terrain") -> Tileset:
    pixels = np.zeros((32, 32, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    pixels[0, 0] = (9, 9, 9, 255)
    return Tileset(name=name, pixels=pixels, tile_w=16, tile_h=16)


def _tab(ctx: FakeCtx, *, dirty: bool = False, tileset: bool = True) -> Any:
    tab = plotter_mode.new_document(ctx, (4, 4, 16, 16))
    if tileset:
        tab.doc.add_tileset(_tileset())
        tab.doc.mark_saved()
    if dirty:
        layer = tab.doc.tile_layers()[0]
        tab.doc.write_region(layer.uid, 0, 0, np.array([[1]], gid.DTYPE))
    return tab


def _save(ctx: FakeCtx, tab: Any, path: Path) -> None:
    plotter_mode.save_to(ctx, tab, path, "wmap")
    plotter_mode.on_task_done(ctx, _Done(f"plotter-save:{tab.uid}", ctx.result))


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


# --- documents ----------------------------------------------------------------


def test_a_new_document_opens_clean_and_with_something_to_paint_on():
    """A map with no layer has nothing to paint into and no row in the layers
    panel, which reads as broken rather than as empty."""
    ctx = FakeCtx()
    tab = plotter_mode.new_document(ctx)
    assert not tab.dirty
    assert len(tab.doc.tile_layers()) == 1
    assert tab.doc.active_layer == tab.doc.layers[0].uid


def test_state_is_built_lazily_and_remembers_recent_files():
    ctx = FakeCtx()
    assert ctx.state.plotter is None
    plotter_mode.ensure(ctx)
    assert ctx.state.plotter is not None

    tab = _tab(ctx)
    path = Path("/tmp/level.wmap")
    plotter_mode.adopt(ctx, tab.doc, path=path)
    assert plotter_mode.recent_paths(ctx) == [str(path)]


def test_opening_an_already_open_path_focuses_rather_than_forking():
    """Two tabs over one path would race on save."""
    ctx = FakeCtx()
    first = plotter_mode.adopt(ctx, _tab(ctx).doc, path=Path("/tmp/a.wmap"))
    plotter_mode.adopt(ctx, _tab(ctx).doc, path=Path("/tmp/b.wmap"))
    ctx.submitted.clear()
    plotter_mode.open_path(ctx, Path("/tmp/a.wmap"))
    assert ctx.submitted == []
    assert plotter_mode.active(ctx) is first


def test_closing_a_tab_leaves_you_on_the_neighbour():
    ctx = FakeCtx()
    first, second, third = _tab(ctx), _tab(ctx), _tab(ctx)
    state = plotter_mode.ensure(ctx)
    state.activate(second.uid)
    plotter_mode.close_tab(ctx, second.uid)
    assert state.active is third
    assert {tab.uid for tab in state.docs} == {first.uid, third.uid}


def test_closing_a_dirty_tab_asks_first():
    ctx = FakeCtx()
    tab = _tab(ctx, dirty=True)
    plotter_mode.close_tab(ctx, tab.uid)
    assert ctx.confirms.pending is not None
    assert plotter_mode.ensure(ctx).get(tab.uid) is tab
    ctx.confirms.pending.on_confirm()
    assert plotter_mode.ensure(ctx).get(tab.uid) is None


def test_switching_tabs_drops_the_palette_and_the_object_selection():
    """Both name things in the *previous* document: a tileset index and an
    object uid the new map does not have."""
    ctx = FakeCtx()
    first = _tab(ctx)
    second = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    state.brush = np.array([[1]], gid.DTYPE)
    state.selected_object = 99
    state.terrain = (0, 2)
    state.activate(first.uid)
    assert state.brush is None and state.selected_object is None
    # The terrain names a *particular* map's tilesets, exactly as the palette
    # index does, so it goes the same way.
    assert state.terrain is None
    # And opening a new one is the same arrival, so it resets too.
    state.brush = np.array([[1]], gid.DTYPE)
    _tab(ctx)
    assert state.brush is None
    assert second is not None


# --- saving -------------------------------------------------------------------


def test_a_save_writes_the_file_and_marks_the_document_clean(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx, dirty=True)
    path = tmp_path / "level.wmap"
    _save(ctx, tab, path)

    assert path.exists()
    assert not tab.dirty and not tab.saving
    assert tab.path == path and tab.title == "level.wmap"
    back = wmap.read_wmap(path.read_bytes())
    assert (back.width, back.height) == (4, 4)


def test_a_save_records_the_head_the_encode_wrote_not_a_later_one():
    """The document routinely moves on while a save runs on a task thread, and
    marking the live head would call those later edits saved."""
    ctx = FakeCtx()
    tab = _tab(ctx, dirty=True)
    head = tab.doc.history.head
    layer = tab.doc.tile_layers()[0]
    tab.doc.write_region(layer.uid, 2, 2, np.array([[1]], gid.DTYPE))
    plotter_mode.on_task_done(
        ctx, _Done(f"plotter-save:{tab.uid}", {"head": head, "path": "", "format": "wmap"})
    )
    assert tab.dirty


def test_a_failed_save_clears_the_lock():
    """``busy`` gates every control that changes the document, so without this
    one failed write makes the tab read-only until it is closed."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True
    plotter_mode.on_task_failed(ctx, _Done(f"plotter-save:{tab.uid}"))
    assert not tab.saving


def test_a_refused_submit_clears_the_lock_too(tmp_path):
    """The runner refuses a key already in flight; leaving the flag set is what
    makes a tab read-only forever after a double press."""
    ctx = FakeCtx(accept=False)
    tab = _tab(ctx)
    plotter_mode.save_to(ctx, tab, tmp_path / "a.wmap", "wmap")
    assert not tab.saving


def test_a_cancelled_dialog_leaves_the_document_alone():
    ctx = FakeCtx()
    tab = _tab(ctx, dirty=True)
    tab.saving = True
    plotter_mode.on_task_done(ctx, _Done(f"plotter-saveas:{tab.uid}", None))
    assert not tab.saving and tab.dirty and tab.path is None


def test_a_map_opened_from_tiled_saves_back_to_tiled(tmp_path):
    """Warlock does not silently convert a file you brought from Tiled into one
    Tiled cannot open."""
    ctx = FakeCtx()
    source = _tab(ctx)
    files = tmx.tmx_export(source.doc)
    for name, blob in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)

    plotter_mode.open_path(ctx, tmp_path / "map.tmx")
    plotter_mode.on_task_done(ctx, _Done("plotter-open:1", ctx.result))
    opened = plotter_mode.active(ctx)
    assert opened.file_format == "tmx"


# --- tilesets -----------------------------------------------------------------


def test_a_loaded_tileset_is_added_to_the_open_map(tmp_path):
    from PIL import Image

    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    png = tmp_path / "grass.png"
    Image.fromarray(np.zeros((32, 32, 4), np.uint8), "RGBA").save(png)

    plotter_mode.add_tileset_path(ctx, png)
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))
    assert len(tab.doc.tilesets) == 1
    assert tab.doc.tilesets[0].tileset.name == "grass"
    # Sliced at the *map's* tile size, which is the only default needing no
    # dialog.
    assert tab.doc.tilesets[0].tileset.tile_w == tab.doc.tile_w


def test_adding_a_tileset_does_not_touch_the_saving_flag():
    """It is not a save, so ``saving`` was never set -- and clearing it here
    would unlock a tab that really is mid-write."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True
    plotter_mode.on_task_done(
        ctx, _Done(f"plotter-tileset:{tab.uid}", {"tileset": _tileset("b"), "source": ""})
    )
    assert tab.saving


def test_adding_a_tileset_with_nothing_open_says_so():
    ctx = FakeCtx()
    plotter_mode.ask_add_tileset(ctx)
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
    assert ctx.submitted == []


# --- the guard ----------------------------------------------------------------


def test_the_guard_proceeds_when_nothing_is_dirty():
    ctx = FakeCtx()
    _tab(ctx)
    calls: list[str] = []
    assert plotter_mode.guard(ctx, "quit", lambda: calls.append("go")) is True
    assert calls == ["go"] and ctx.confirms.pending is None


def test_the_guard_asks_once_for_however_many_are_dirty():
    """One question for all of them: asking per document would put the user in
    front of a queue they answered the same way each time."""
    ctx = FakeCtx()
    _tab(ctx, dirty=True)
    _tab(ctx, dirty=True)
    calls: list[str] = []
    assert plotter_mode.guard(ctx, "quit", lambda: calls.append("go")) is False
    assert calls == []
    assert "2 maps have" in ctx.confirms.pending.message


def test_the_guard_is_silent_before_the_mode_has_ever_been_opened():
    """``ensure`` builds state lazily, so the quit chain must not create it."""
    ctx = FakeCtx()
    calls: list[str] = []
    plotter_mode.guard(ctx, "quit", lambda: calls.append("go"))
    assert calls == ["go"] and ctx.state.plotter is None


# --- keys ---------------------------------------------------------------------


def _key(name: str, *, ctrl: bool = False, shift: bool = False):
    import pygame

    mods = (pygame.KMOD_CTRL if ctrl else 0) | (pygame.KMOD_SHIFT if shift else 0)
    return pygame.event.Event(pygame.KEYDOWN, key=getattr(pygame, f"K_{name}")), mods


def test_a_tool_letter_picks_that_tool(monkeypatch):
    import pygame

    ctx = FakeCtx()
    _tab(ctx)
    state = plotter_mode.ensure(ctx)
    event, mods = _key("g")
    monkeypatch.setattr(pygame.key, "get_mods", lambda: mods)
    assert plotter_mode.handle_key(ctx, event) is True
    assert state.tool == "fill"


def test_undo_is_refused_while_the_tab_is_busy(monkeypatch):
    """Consumed, not passed through: the key belongs to this mode either way,
    and letting it fall through would act on a viewport Plotter has replaced."""
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx, dirty=True)
    tab.saving = True
    head = tab.doc.history.head
    event, mods = _key("z", ctrl=True)
    monkeypatch.setattr(pygame.key, "get_mods", lambda: mods)
    assert plotter_mode.handle_key(ctx, event) is True
    assert tab.doc.history.head == head


def test_undo_works_when_it_is_not(monkeypatch):
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx, dirty=True)
    event, mods = _key("z", ctrl=True)
    monkeypatch.setattr(pygame.key, "get_mods", lambda: mods)
    plotter_mode.handle_key(ctx, event)
    assert not tab.dirty


def test_a_key_release_is_never_consumed():
    import pygame

    ctx = FakeCtx()
    _tab(ctx)
    up = pygame.event.Event(pygame.KEYUP, key=pygame.K_g)
    assert plotter_mode.handle_key(ctx, up) is False


# --- objects ------------------------------------------------------------------


def test_an_object_round_trips_through_a_save(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx)
    layer = tab.doc.add_object_layer("Things")
    tab.doc.add_object(
        layer.uid, MapObject(uid=new_uid(), name="spawn", kind="point", x=3, y=4)
    )
    path = tmp_path / "level.wmap"
    _save(ctx, tab, path)
    back = wmap.read_wmap(path.read_bytes())
    assert back.layers[1].objects[0].name == "spawn"


# --- terrain sets -------------------------------------------------------------


def _spec(**kwargs):
    from warlock.studio.plotter import tilegen

    return tilegen.GenSpec(**{"tile_w": 8, "tile_h": 8, **kwargs})


def test_generating_with_nothing_open_says_so():
    ctx = FakeCtx()
    plotter_mode.generate_terrain_set(ctx, _spec())
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
    assert not ctx.submitted


def test_generating_a_set_goes_through_the_tileset_key_so_nothing_new_routes_it():
    """The result *is* a tileset for this tab and ``on_task_done`` already
    adopts one; a key of its own would be a second copy of that branch."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    plotter_mode.generate_terrain_set(ctx, _spec())
    assert ctx.submitted[-1] == f"plotter-tileset:{tab.uid}"


def test_a_generated_set_arrives_as_one_tileset_and_one_undo_step():
    from warlock.studio.plotter import blob

    ctx = FakeCtx()
    tab = _tab(ctx)
    depth = len(tab.doc.history)
    before = len(tab.doc.tilesets)
    plotter_mode.generate_terrain_set(ctx, _spec())
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))
    assert len(tab.doc.tilesets) == before + 1
    assert len(tab.doc.history) == depth + 1
    assert tab.doc.tilesets[-1].tileset.columns == blob.TILE_COUNT
    tab.doc.undo()
    assert len(tab.doc.tilesets) == before


def test_a_terrain_set_puts_its_first_terrain_in_your_hand():
    """The thing that just arrived is the thing you are holding, which is what
    setting the palette index already says for an ordinary tileset."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    plotter_mode.generate_terrain_set(ctx, _spec())
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))
    assert plotter_mode.ensure(ctx).terrain == (len(tab.doc.tilesets) - 1, 0)


def test_generating_an_isometric_set_makes_the_map_isometric_in_the_same_step():
    """Two steps would leave a Ctrl+Z on a map whose only tileset is drawn for
    the lattice it is no longer on."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    depth = len(tab.doc.history)
    before = len(tab.doc.tilesets)
    plotter_mode.generate_terrain_set(ctx, _spec(projection="isometric"))
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))
    assert tab.doc.projection == "isometric"
    assert len(tab.doc.history) == depth + 1
    tab.doc.undo()
    assert tab.doc.projection == "orthogonal"
    assert len(tab.doc.tilesets) == before


def test_an_arriving_tileset_refits_the_view():
    """An isometric map's pixel extent is not width times tile width, so a fit
    computed against the old projection is the wrong frame."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.view.fitted = True
    plotter_mode.generate_terrain_set(ctx, _spec(projection="isometric"))
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))
    assert tab.view.fitted is False


def test_sending_an_atlas_back_from_inker_keeps_every_painted_cell():
    from warlock.studio.plotter import terrain

    ctx = FakeCtx()
    tab = _tab(ctx)
    plotter_mode.generate_terrain_set(ctx, _spec())
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))
    index = len(tab.doc.tilesets) - 1
    ref = tab.doc.tilesets[index]
    layer = tab.doc.tile_layers()[0]
    tab.doc.write_region(layer.uid, *terrain.paint_terrain(layer.data, 2, 2, 1, ref))
    before = layer.data.copy()

    painted = np.array(ref.tileset.pixels)
    painted[..., 2] = 111
    plotter_mode.tileset_from_inker(ctx, _FakeInkerDoc(painted), index=index)
    assert np.array_equal(layer.data, before)
    assert int(tab.doc.tilesets[index].tileset.pixels[..., 2].max()) == 111
    assert [t.name for t in tab.doc.tilesets[index].tileset.terrains] == [
        t.name for t in ref.tileset.terrains
    ]


def test_an_atlas_of_a_different_size_is_refused_rather_than_renumbering_the_map():
    ctx = FakeCtx()
    tab = _tab(ctx)
    plotter_mode.generate_terrain_set(ctx, _spec())
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))
    index = len(tab.doc.tilesets) - 1
    before = tab.doc.tilesets[index].tileset.pixels
    plotter_mode.tileset_from_inker(
        ctx, _FakeInkerDoc(np.zeros((4, 4, 4), dtype=np.uint8)), index=index
    )
    assert ctx.toasts[-1][1] == "error"
    assert tab.doc.tilesets[index].tileset.pixels is before


class _FakeInkerDoc:
    """Only what ``tileset_from_inker`` touches: a title and a flatten."""

    def __init__(self, pixels: np.ndarray) -> None:
        self._pixels = pixels
        self.title = "atlas"

    def flatten(self, *, matte: bool = True) -> np.ndarray:
        return self._pixels
