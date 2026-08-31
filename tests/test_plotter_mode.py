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

from warlock.studio import plotter_io, plotter_mode
from warlock.studio.plotter import tmx, wmap
from warlock.studio.plotter.tilemap import MapObject, new_uid
from warlock.studio.tilegrid import gid
from warlock.studio.tilegrid.tileset import Tileset


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
        # ``set_mode`` writes both of these, and ``ask_new_document`` switches
        # modes so Home and the palette can reach a dialog that is Plotter's.
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


def test_one_file_spelled_two_ways_is_one_tab(tmp_path):
    """On Windows ``Level.WMAP`` and ``level.wmap`` are the same file, and
    ``Path.__eq__`` says they are not -- so the recents list and a drop used to
    fork into two tabs that then raced on save."""
    ctx = FakeCtx()
    (tmp_path / "Level.WMAP").write_bytes(b"")
    first = plotter_mode.adopt(ctx, _tab(ctx).doc, path=tmp_path / "Level.WMAP")
    ctx.submitted.clear()
    plotter_mode.open_path(ctx, tmp_path / "level.wmap")
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
    state.select_object(99)
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


def test_an_export_refusal_is_toasted_rather_than_raised_on_the_frame_thread(
    tmp_path, monkeypatch
):
    """A named exporter refusal cannot be allowed to take down the UI thread."""
    ctx = FakeCtx()
    tab = _tab(ctx)

    def refuse(_doc, _format):
        raise tmx.TiledUnsupported("a synthetic future feature", exporting=True)

    monkeypatch.setattr(plotter_io, "_encode", refuse)

    plotter_mode.save_to(ctx, tab, tmp_path / "a.tmx", "tmx")
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
    assert "a synthetic future feature" in ctx.toasts[-1][0]
    assert not tab.saving, "a refused encode must not leave the tab locked"


class _FutureLayer:
    """A fifth layer kind, arriving before ``.wmap`` can hold it.

    The two tests below used to reach the ``.wmap`` writer door with a *group*,
    which version 3 stores. The door is still there and still has to be caught
    by name on the frame thread -- chunked storage and the next layer kind both
    arrive behind it -- so the refusal is provoked with the one thing that
    still reaches it rather than the tests being deleted along with the
    refusals they were written for.
    """


def test_a_wmap_writer_door_refusal_is_toasted_too_rather_than_raised(tmp_path):
    """The ``.wmap`` door raises a ``WmapUnstorable`` -- our own format's limit
    is not a Tiled feature -- so a guard that only caught ``TiledUnsupported``
    would let a ``.wmap`` save crash the window while a ``.tmx`` save of the
    same document toasted politely."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.doc.layers.append(_FutureLayer())

    plotter_mode.save_to(ctx, tab, tmp_path / "a.wmap", "wmap")
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
    assert "no entry for" in ctx.toasts[-1][0]
    assert not tab.saving


def test_exporting_a_map_the_wmap_writer_refuses_toasts_rather_than_raising():
    """``export_to_library`` is the one other frame-thread encode: it writes the
    ``.wmap`` beside the render so ``Open in Plotter`` can reopen the real
    document, and it does so before the task starts."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.doc.layers.append(_FutureLayer())

    plotter_mode.export_library(ctx, tab)
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
    assert "no entry for" in ctx.toasts[-1][0]
    assert not tab.saving


def test_a_map_with_a_group_now_saves_rather_than_toasting(tmp_path):
    """Flipped. The two refusals above were written when the ``.wmap``
    manifest was a flat list; version 3's entries are recursive, so the save
    path that used to toast a tree now writes one."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.doc.add_group_layer("G")

    plotter_mode.save_to(ctx, tab, tmp_path / "a.wmap", "wmap")
    assert not [t for t in ctx.toasts if t[1] == "error"]


def test_a_cancelled_dialog_leaves_the_document_alone():
    ctx = FakeCtx()
    tab = _tab(ctx, dirty=True)
    tab.saving = True
    plotter_mode.on_task_done(ctx, _Done(f"plotter-saveas:{tab.uid}", None))
    assert not tab.saving and tab.dirty and tab.path is None


def test_a_save_leaves_no_staging_file_behind(tmp_path):
    """The temporary is a dotfile and it is removed in a ``finally``: the old
    ``level.wmap.tmp`` sat in the folder the user picked, sorted right beside
    the file it is a fragment of."""
    ctx = FakeCtx()
    tab = _tab(ctx, dirty=True)
    _save(ctx, tab, tmp_path / "level.wmap")
    assert {p.name for p in tmp_path.iterdir()} == {"level.wmap"}


def test_a_failed_write_strands_no_temporary(tmp_path, monkeypatch):
    """Nothing removed the staging file when the replace never happened, so a
    full disk or a locked target left a fragment in the user's folder."""
    ctx = FakeCtx()
    tab = _tab(ctx, dirty=True)

    def boom(*_args, **_kwargs):
        raise OSError("no room")

    monkeypatch.setattr(plotter_io.atomic.os, "replace", boom)
    with pytest.raises(OSError):
        plotter_io._write({"map.wmap": b"x"}, tmp_path / "level.wmap")
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert tab is not None


def test_an_export_stages_every_file_before_replacing_any(tmp_path, monkeypatch):
    """``packwright_io._write``'s rule, which this one was written from and
    never brought up to.

    A Tiled export is ``map.tmx`` plus a ``.tsx`` and a ``.png`` per tileset.
    Replacing them one at a time meant an encode that raised on the third left
    the map already written on top of the previous export, pointing at tilesets
    that were stale or not there -- a set of files that say they belong
    together and do not.
    """
    (tmp_path / "level.tmx").write_bytes(b"old map")
    (tmp_path / "tiles.tsx").write_bytes(b"old tileset")
    calls: list[str] = []
    real = plotter_io.atomic.os.replace

    def counted(src, dst):
        calls.append(Path(dst).name)
        if len(calls) == 1:
            raise OSError("no room")
        return real(src, dst)

    monkeypatch.setattr(plotter_io.atomic.os, "replace", counted)
    with pytest.raises(OSError):
        plotter_io._write(
            {"map.tmx": b"new map", "tiles.tsx": b"new tileset"},
            tmp_path / "level.tmx",
        )
    # The first replace is the one that raised, so nothing landed -- and in
    # particular the *second* file was never written on its own.
    assert (tmp_path / "level.tmx").read_bytes() == b"old map"
    assert (tmp_path / "tiles.tsx").read_bytes() == b"old tileset"
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_a_failing_encode_of_a_later_file_writes_none_of_them(tmp_path, monkeypatch):
    """The half-of-one-export-and-half-of-another case, from the staging side:
    a write that raises partway leaves every previous target untouched."""
    (tmp_path / "level.tmx").write_bytes(b"old map")

    calls: list[str] = []
    real_write = Path.write_bytes

    def counted(self, blob):
        calls.append(self.name)
        if len(calls) == 2:
            raise OSError("no room")
        return real_write(self, blob)

    monkeypatch.setattr(Path, "write_bytes", counted)
    with pytest.raises(OSError):
        plotter_io._write(
            {"map.tmx": b"new map", "tiles.tsx": b"new tileset"},
            tmp_path / "level.tmx",
        )
    assert (tmp_path / "level.tmx").read_bytes() == b"old map"
    assert not (tmp_path / "tiles.tsx").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


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


# --- what a file may point at -------------------------------------------------
#
# The paths inside a ``.tmx`` or ``.tsx`` come from a file, not from the user,
# and this layer is the one that turns them into a read.


@pytest.mark.parametrize("source", ["C:\\evil.png", "\\\\host\\share\\x.png", "/etc/passwd"])
def test_a_tileset_source_that_leaves_the_document_is_refused(source, tmp_path):
    with pytest.raises(ValueError, match="absolute path"):
        plotter_io._resolve_source(tmp_path, source)


def test_an_engine_refusal_reaches_the_user_inside_a_sentence(tmp_path):
    """``this file uses group layers, which Plotter does not support`` is
    precise and has no subject. The frame supplies one and keeps the detail,
    which is the only part that says which feature."""
    from warlock.service.errors import Invalid

    # An unrecognised orientation, and it is the *third* feature this test has
    # asked about: hexagonal left the refusal list when the editor learned to
    # project it, and infinite left when it learned to chunk. That is the only
    # reason a refusal ever moves, and this test is about the *framing* rather
    # than about any one feature -- so it wants whatever is still genuinely
    # refused, and an orientation outside ``project.PROJECTIONS`` is a refusal
    # by construction rather than one waiting on effort.
    path = tmp_path / "hostile.tmx"
    path.write_bytes(
        b'<map version="1.10" orientation="spiral" width="1" '
        b'height="1" tilewidth="16" tileheight="16"/>'
    )
    with pytest.raises(Invalid) as exc:
        plotter_io._load(path)
    assert "This map could not be opened" in str(exc.value)
    assert "spiral" in str(exc.value)
    assert exc.value.field == "file"


def test_a_file_past_the_ceiling_is_refused_before_it_is_read(tmp_path, monkeypatch):
    """One answer to "how big may a map document be", shared with the service's
    own upload cap rather than invented a second time here."""
    from warlock.service import files as svc_files
    from warlock.service.errors import TooLarge

    path = tmp_path / "level.wmap"
    path.write_bytes(b"PK\x03\x04" + b"\0" * 64)
    monkeypatch.setattr(svc_files, "MAX_MAP_SOURCE_BYTES", 8)
    with pytest.raises(TooLarge) as exc:
        plotter_io._load(path)
    assert exc.value.field == "file" and "level.wmap" in str(exc.value)


def test_a_relative_source_may_climb_because_tiled_projects_do(tmp_path):
    """A tileset folder beside a maps folder is the normal Tiled layout, so
    containment to the map's own directory would refuse ordinary projects."""
    from PIL import Image

    maps = tmp_path / "maps"
    tilesets = tmp_path / "tilesets"
    maps.mkdir()
    tilesets.mkdir()
    Image.fromarray(np.zeros((32, 32, 4), np.uint8), "RGBA").save(tilesets / "t.png")
    (tilesets / "t.tsx").write_text(
        '<tileset name="t" tilewidth="16" tileheight="16">'
        '<image source="t.png" width="32" height="32"/></tileset>',
        encoding="utf-8",
    )
    loaded = plotter_io._loaders(maps)["tsx_loader"]("../tilesets/t.tsx")
    assert loaded.name == "t" and loaded.tile_w == 16


# --- the library --------------------------------------------------------------


def test_the_exported_render_is_the_source_beside_it_not_a_later_document(svc, monkeypatch):
    """The encode is on the frame thread because serialising reads the live
    document; the *render* is not, and it reparses those bytes rather than
    reading the document a second time -- so an edit made while the task runs
    cannot appear in the picture without appearing in the source too."""
    from PIL import Image

    from warlock.studio.plotter.render import render_map

    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    layer = tab.doc.tile_layers()[0]
    tab.doc.write_region(layer.uid, 0, 0, np.array([[1]], gid.DTYPE))

    encoded: list[bytes] = []
    real = wmap.read_wmap

    def spy(data: bytes):
        encoded.append(bytes(data))
        # An edit landing *between* the encode and the render is the window the
        # old shape composited inside.
        tab.doc.write_region(layer.uid, 3, 3, np.array([[1]], gid.DTYPE))
        return real(data)

    monkeypatch.setattr(wmap, "read_wmap", spy)
    plotter_mode.export_library(ctx, tab)

    assert encoded, "the task reparses what the frame thread encoded"
    job_id = ctx.result["job_id"]
    with Image.open(svc.job_dir(job_id) / "input.png") as image:
        rendered = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    assert np.array_equal(rendered, render_map(real(encoded[0])))
    # And the picture is genuinely not the later document.
    assert not np.array_equal(rendered, render_map(tab.doc))


# --- the canvas's own arithmetic ----------------------------------------------
#
# ``_apply`` and ``_corner_uvs`` are imgui-free and pure, so the pane imports
# headlessly and the two decisions it owns can be asserted without a window.


def test_picking_from_the_second_tileset_selects_the_second_tileset():
    """``list.index`` on a ``TilesetRef`` compares ndarrays; it only ever
    returned the right answer by short-circuiting on the firstgid."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.doc.add_tileset(_tileset("second"))
    state = plotter_mode.ensure(ctx)
    state.tool = "pick"
    layer = tab.doc.tile_layers()[0]
    second = tab.doc.tilesets[1]
    tab.doc.write_region(
        layer.uid, 0, 0, np.array([[second.firstgid]], gid.DTYPE)
    )
    plotter_canvas._apply(ctx, state, tab, (0, 0))
    assert state.tileset_index == 1


def _stamping(ctx: FakeCtx):
    """A tab with the stamp tool and a one-tile brush in hand."""
    tab = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    state.tool = "stamp"
    state.brush = np.array([[tab.doc.tilesets[0].firstgid]], gid.DTYPE)
    return tab, state, tab.doc.tile_layers()[0]


def test_a_shift_click_line_is_one_undo_step():
    """The whole point of drawing it inside the open session: forty cells must
    cost one Ctrl+Z, exactly as a forty-cell drag does."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, layer = _stamping(ctx)
    depth = len(tab.doc.history)

    tab.doc.begin_stroke(layer.uid)
    plotter_canvas._apply_line(ctx, state, tab, (0, 0), (3, 0))
    tab.doc.end_stroke()

    assert len(tab.doc.history) == depth + 1
    assert int((layer.data[0, 0:4] != 0).sum()) == 4
    tab.doc.undo()
    assert not layer.data.any()


def test_a_line_with_nothing_in_hand_toasts_once():
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, _layer = _stamping(ctx)
    state.brush = None
    plotter_canvas._apply_line(ctx, state, tab, (0, 0), (3, 3))
    assert len(ctx.toasts) == 1


def test_a_fast_drag_paints_the_cells_it_skipped_over():
    """A drag is sampled once a frame, so a fast one arrives with gaps. Without
    interpolation the stroke comes out dotted."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, layer = _stamping(ctx)
    tab.doc.begin_stroke(layer.uid)
    plotter_canvas._apply_drag(ctx, state, tab, (0, 0))
    plotter_canvas._apply_drag(ctx, state, tab, (3, 3))  # three cells in one frame
    tab.doc.end_stroke()

    assert [int(layer.data[i, i] != 0) for i in range(4)] == [1, 1, 1, 1]


def test_a_drag_that_stays_in_one_cell_paints_it_once():
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, layer = _stamping(ctx)
    depth = len(tab.doc.history)
    tab.doc.begin_stroke(layer.uid)
    for _ in range(5):
        plotter_canvas._apply_drag(ctx, state, tab, (2, 2))
    tab.doc.end_stroke()
    assert len(tab.doc.history) == depth + 1
    assert int((layer.data != 0).sum()) == 1


def test_the_line_origin_follows_the_last_stamp():
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, _layer = _stamping(ctx)
    assert state.last_paint is None
    plotter_canvas._apply_drag(ctx, state, tab, (2, 3))
    assert state.last_paint == (2, 3)


def test_the_line_origin_does_not_survive_a_tab_switch():
    """It names a cell in a particular map; a Shift+click in another one would
    otherwise draw from somewhere the user never clicked."""
    ctx = FakeCtx()
    first = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    state.last_paint = (2, 3)
    second = plotter_mode.new_document(ctx, (4, 4, 16, 16))
    assert state.last_paint is None

    state.last_paint = (1, 1)
    state.activate(first.uid)
    assert state.last_paint is None
    assert second is not None


def test_a_marquee_drag_selects_and_a_plain_click_clears_it():
    """Tiled's gesture, including the last part: a click with no drag is how a
    selection is dropped without reaching for a menu."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    state.tool = "select"

    # Press, drag, release -- driven directly, since the gesture branches on
    # imgui's mouse state.
    state.drag_kind = "select"
    state.drag_anchor = (1, 1)
    state.select = plotter_canvas._normalized((1, 1), (1, 1))
    state.select = plotter_canvas._normalized(state.drag_anchor, (3, 2))
    assert state.select == (1, 1, 3, 2)

    # Dragged back past the anchor: still lowest-first.
    state.select = plotter_canvas._normalized(state.drag_anchor, (0, 0))
    assert state.select == (0, 0, 1, 1)
    assert tab is not None


def test_a_selection_is_clamped_to_the_map_it_is_used_on():
    """Clamped at use, not when stored: a resize is undoable and the marquee is
    not, so a rect trimmed at store time could not grow back."""
    ctx = FakeCtx()
    tab = _tab(ctx)  # 4x4
    state = plotter_mode.ensure(ctx)

    state.select = (2, 2, 99, 99)
    assert state.selection_in(tab.doc) == (2, 2, 3, 3)

    state.select = (-5, -5, 1, 1)
    assert state.selection_in(tab.doc) == (0, 0, 1, 1)

    state.select = (10, 10, 20, 20)
    assert state.selection_in(tab.doc) is None, "wholly off the map is nothing"

    state.select = None
    assert state.selection_in(tab.doc) is None


def test_escape_cancels_one_thing_at_a_time(monkeypatch):
    """Staged, outermost first: a press that abandons a drag must not also throw
    away the marquee the user spent a gesture placing."""

    ctx = FakeCtx()
    _tab(ctx)
    state = plotter_mode.ensure(ctx)
    state.select = (0, 0, 2, 2)
    state.select_object(7)
    state.drag_kind = "paint"
    event = _key("ESCAPE")

    assert plotter_mode.handle_key(ctx, event) is True
    assert state.drag_kind == ""
    assert state.selected_object == 7 and state.select == (0, 0, 2, 2)

    plotter_mode.handle_key(ctx, event)
    assert state.selected_object is None
    assert state.select == (0, 0, 2, 2)

    plotter_mode.handle_key(ctx, event)
    assert state.select is None

    # Consumed even with nothing left to drop: Esc belongs to this mode.
    assert plotter_mode.handle_key(ctx, event) is True


def test_ctrl_a_selects_everything_and_ctrl_d_drops_it(monkeypatch):

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    head = tab.doc.history.head

    event = _key("a", ctrl=True)
    assert plotter_mode.handle_key(ctx, event) is True
    assert state.select == (0, 0, tab.doc.width - 1, tab.doc.height - 1)

    event = _key("d", ctrl=True)
    plotter_mode.handle_key(ctx, event)
    assert state.select is None

    # Inker's spelling deselects too.
    state.select = (0, 0, 1, 1)
    event = _key("a", ctrl=True, shift=True)
    plotter_mode.handle_key(ctx, event)
    assert state.select is None

    # None of it touched the document.
    assert tab.doc.history.head == head
    assert not tab.dirty


def test_a_selection_does_not_survive_a_tab_switch():
    ctx = FakeCtx()
    first = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    state.select = (0, 0, 2, 2)
    plotter_mode.new_document(ctx, (4, 4, 16, 16))
    assert state.select is None

    state.select = (0, 0, 1, 1)
    state.activate(first.uid)
    assert state.select is None


def _painted(ctx: FakeCtx):
    """A stamping tab whose layer is filled with a known tile."""
    tab, state, layer = _stamping(ctx)
    value = int(tab.doc.tilesets[0].firstgid)
    layer.data[:, :] = value
    tab.doc.mark_saved()
    return tab, state, layer, value


def test_copy_then_paste_puts_the_block_in_the_brush(monkeypatch):
    """Paste is Tiled's model -- the block becomes the brush and the tool
    becomes Stamp -- so every rule the stamp already has applies unchanged."""

    ctx = FakeCtx()
    tab, state, layer, value = _painted(ctx)
    state.select = (0, 0, 1, 1)

    event = _key("c", ctrl=True)
    assert plotter_mode.handle_key(ctx, event) is True
    assert state.clipboard.shape == (2, 2)
    assert state.clipboard_doc == tab.uid

    state.tool = "select"
    event = _key("v", ctrl=True)
    plotter_mode.handle_key(ctx, event)
    assert state.tool == "stamp"
    assert np.array_equal(state.brush, np.full((2, 2), value, gid.DTYPE))
    assert not tab.dirty, "a copy and a paste write nothing"


def test_the_clipboard_is_an_owned_copy():
    """The layer goes on being painted; a clipboard tracking it would paste
    whatever the map looks like now."""
    ctx = FakeCtx()
    tab, state, layer, _value = _painted(ctx)
    state.select = (0, 0, 1, 1)
    plotter_mode._copy(ctx, state, tab, cut=False)
    layer.data[:, :] = 0
    assert state.clipboard.any()


def test_a_cut_clears_the_cells_in_one_step():
    ctx = FakeCtx()
    tab, state, layer, _value = _painted(ctx)
    state.select = (0, 0, 1, 1)
    depth = len(tab.doc.history)

    plotter_mode._copy(ctx, state, tab, cut=True)
    assert len(tab.doc.history) == depth + 1
    assert not layer.data[0:2, 0:2].any()
    assert layer.data[2:, 2:].all(), "and nothing outside it"
    assert state.clipboard.all(), "the copy was taken before the clear"


def test_pasting_into_another_map_is_refused_by_name():
    """Gids are numbered against a map's own firstgids, so the same number is a
    different tile here -- the paste would land silently wrong."""
    ctx = FakeCtx()
    first, state, _layer, _value = _painted(ctx)
    state.select = (0, 0, 1, 1)
    plotter_mode._copy(ctx, state, first, cut=False)

    second = plotter_mode.new_document(ctx, (4, 4, 16, 16))
    plotter_mode._paste(ctx, state, second)
    assert state.brush is None
    assert ctx.toasts and "another map" in ctx.toasts[-1][0]
    assert first.title in ctx.toasts[-1][0], "the refusal names its source"


def test_copying_with_no_selection_says_so():
    ctx = FakeCtx()
    tab, state, _layer, _value = _painted(ctx)
    state.select = None
    plotter_mode._copy(ctx, state, tab, cut=False)
    assert state.clipboard is None
    assert ctx.toasts and "Select" in ctx.toasts[-1][0]


def test_delete_clears_the_selected_cells_in_one_step():
    ctx = FakeCtx()
    tab, state, layer, _value = _painted(ctx)
    state.select = (1, 1, 2, 2)
    depth = len(tab.doc.history)
    plotter_mode._delete(ctx, state, tab)
    assert len(tab.doc.history) == depth + 1
    assert not layer.data[1:3, 1:3].any()
    assert layer.data[0, 0]


def test_delete_removes_the_object_when_the_object_tool_is_held():
    """Decided by the tool in hand, so a marquee left over from earlier does not
    quietly erase tiles instead."""
    ctx = FakeCtx()
    tab, state, _layer, _value = _painted(ctx)
    obj_layer = tab.doc.add_object_layer("Things")
    tab.doc.set_active_layer(obj_layer.uid)
    obj = MapObject(uid=new_uid(), name="spawn", kind="point", x=1, y=1)
    tab.doc.add_object(obj_layer.uid, obj)
    state.tool = "object"
    state.select_object(obj.uid)
    state.select = (0, 0, 3, 3)

    plotter_mode._delete(ctx, state, tab)
    assert not tab.doc.layers[-1].objects
    assert state.selected_object is None


def test_a_selection_constrains_a_stamp():
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, layer = _stamping(ctx)
    state.select = (0, 0, 0, 0)
    state.brush = np.full((3, 3), tab.doc.tilesets[0].firstgid, gid.DTYPE)
    plotter_canvas._apply(ctx, state, tab, (0, 0))
    assert int((layer.data != 0).sum()) == 1, "a 3x3 stamp cut to the one cell"


def test_a_selection_does_not_constrain_a_terrain_refit():
    """A blob re-fit reaches the eight cells around what it touched by design.
    Trimming it to the marquee would leave that ring showing the edge art of a
    neighbour that is no longer what it was -- a visibly broken field, which is
    worse than a tool reaching one cell past a selection."""
    from warlock.studio.panes import plotter_canvas
    from warlock.studio.plotter import terrain as terrainlib

    ctx = FakeCtx()
    tab, state, ref = _terrain_tab(ctx)
    layer = tab.doc.tile_layers()[0]
    cells = [(x, y) for y in range(4) for x in range(4)]
    tab.doc.write_region(
        layer.uid, *terrainlib.paint_terrain_cells(layer.data, cells, 0, ref)
    )
    neighbour = int(layer.data[1, 1])

    # A marquee around the single cell being erased. (1, 1) is outside it.
    state.select = (2, 2, 2, 2)
    state.tool = "erase"
    plotter_canvas._apply(ctx, state, tab, (2, 2))

    assert int(layer.data[2, 2]) == 0
    assert int(layer.data[1, 1]) != neighbour, "the ring re-fit past the marquee"


def test_a_selection_does_constrain_a_plain_erase():
    """The other half of the rule: an erase that is *not* a re-fit is an
    ordinary placement, and gets cut down like any other."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, layer, _value = _painted(ctx)
    state.tool = "erase"
    state.select = (0, 0, 0, 0)
    plotter_canvas._apply(ctx, state, tab, (2, 2))
    assert layer.data.all(), "the erase fell outside the marquee and landed nowhere"


def test_a_selection_constrains_a_shape_fill():
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, layer = _stamping(ctx)
    state.tool = "shape"
    state.shape_mode = "rect"
    state.select = (0, 0, 1, 1)
    plotter_canvas._apply_shape(ctx, state, tab, (0, 0), (3, 3))
    assert int((layer.data != 0).sum()) == 4


def test_a_cut_is_refused_while_the_tab_is_busy(monkeypatch):

    ctx = FakeCtx()
    tab, state, layer, _value = _painted(ctx)
    state.select = (0, 0, 1, 1)
    tab.saving = True
    event = _key("x", ctrl=True)
    assert plotter_mode.handle_key(ctx, event) is True
    assert layer.data.all(), "nothing was cut"
    assert state.clipboard is None


def test_the_minimap_is_sized_to_its_longest_edge():
    """A wide map and a tall one both fit the same box, and neither is
    stretched -- the scale comes from whichever edge is longer."""
    from warlock.studio.panes import plotter_canvas
    from warlock.studio.tokens import sp

    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.doc.resize(40, 10)
    _x, _y, w, h = plotter_canvas._minimap_rect(tab, (800.0, 600.0))
    assert w == pytest.approx(sp(plotter_canvas._MINIMAP_MAX))
    assert h == pytest.approx(w / 4), "aspect kept"

    tab.doc.resize(10, 40)
    _x, _y, w2, h2 = plotter_canvas._minimap_rect(tab, (800.0, 600.0))
    assert h2 == pytest.approx(sp(plotter_canvas._MINIMAP_MAX))
    assert w2 == pytest.approx(h2 / 4)


def test_the_minimap_sits_inside_the_bottom_right_of_the_pane():
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab = _tab(ctx)
    x, y, w, h = plotter_canvas._minimap_rect(tab, (800.0, 600.0))
    assert x > 0 and x + w < 800.0
    assert y > 0 and y + h < 600.0


def test_the_minimap_cache_is_keyed_on_the_head_and_the_layer_count():
    """The head moves for every edit, which is exactly when the picture
    changes; the layer count catches an add whose step the head also moved."""
    from warlock.studio.plotter import render as plotter_render

    ctx = FakeCtx()
    tab = _tab(ctx)
    layer = tab.doc.tile_layers()[0]
    first = (tab.doc.history.head, len(tab.doc.layers))

    tab.doc.write_region(layer.uid, 0, 0, np.array([[1]], gid.DTYPE))
    assert (tab.doc.history.head, len(tab.doc.layers)) != first

    # And the picture really does change with it.
    tab.doc.undo()
    blank = plotter_render.minimap(tab.doc)
    tab.doc.redo()
    assert not np.array_equal(plotter_render.minimap(tab.doc), blank)


def test_a_resize_pins_the_opposite_corner():
    """Named by the corner that moves; the other stays still."""
    from warlock.studio.panes import plotter_canvas

    class Obj:
        x, y, w, h = 10.0, 20.0, 30.0, 40.0
        kind = "rect"

    obj = Obj()
    assert plotter_canvas._opposite("nw", obj) == (40.0, 60.0)
    assert plotter_canvas._opposite("se", obj) == (10.0, 20.0)
    assert plotter_canvas._opposite("ne", obj) == (10.0, 60.0)
    assert plotter_canvas._opposite("sw", obj) == (40.0, 20.0)


def test_dragging_a_corner_past_its_opposite_flips_rather_than_going_negative():
    """A negative size draws as nothing and exports as a rectangle no engine
    can read, so the rect is normalized as it goes."""
    from warlock.studio.panes import plotter_canvas

    # Pinned at (40, 60); the pointer crosses well past it.
    class Unrotated:
        rotation = 0.0

    x, y, w, h = plotter_canvas._resized(Unrotated(), (40.0, 60.0), (100.0, 90.0))
    assert (x, y, w, h) == (40.0, 60.0, 60.0, 30.0)
    assert w >= 0 and h >= 0


def test_resizing_a_rotated_object_keeps_the_pinned_corner_still():
    """The whole point of pinning a corner: it must not move.

    ``x``/``y``/``w``/``h`` are the object's *unrotated* origin and extent, so
    a resize that took the min and max of two map-space points wrote a
    rectangle that -- once ``_rotated`` put the rotation back -- was neither
    the size asked for nor in the place it was grabbed. Asserted through
    ``_handle_corners`` rather than against the raw fields, because what has to
    hold is a statement about where the corner ends up on screen.
    """
    import math

    from warlock.studio.panes import plotter_canvas

    class Obj:
        x, y, w, h = 100.0, 100.0, 40.0, 20.0
        rotation = 30.0

    obj = Obj()
    # Pin the north-west corner and drag the south-east one somewhere new.
    fixed = plotter_canvas._opposite("se", obj)
    target = plotter_canvas._rotated(obj, 90.0, 50.0)
    obj.x, obj.y, obj.w, obj.h = plotter_canvas._resized(obj, fixed, target)

    moved = plotter_canvas._handle_corners(obj)
    assert math.isclose(moved["nw"][0], fixed[0], abs_tol=1e-9)
    assert math.isclose(moved["nw"][1], fixed[1], abs_tol=1e-9)
    # And the dragged corner lands under the pointer, which is the other half
    # of what a resize promises.
    assert math.isclose(moved["se"][0], target[0], abs_tol=1e-9)
    assert math.isclose(moved["se"][1], target[1], abs_tol=1e-9)
    # The extent is the diagonal measured in the object's own frame.
    assert math.isclose(obj.w, 90.0, abs_tol=1e-9)
    assert math.isclose(obj.h, 50.0, abs_tol=1e-9)


def test_only_a_rect_has_resize_handles():
    """A point has no corners; its position *is* the object."""
    from warlock.studio.panes import plotter_canvas

    class Point:
        x, y, w, h = 5.0, 5.0, 0.0, 0.0
        kind = "point"

    assert plotter_canvas._handle_at(None, None, Point(), (0.0, 0.0)) is None


def test_a_moved_object_is_one_undo_step_through_the_document():
    """The canvas drives the session; this pins the shape of what it drives."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    layer = tab.doc.add_object_layer("Things")
    obj = MapObject(uid=new_uid(), name="zone", kind="rect", x=0, y=0, w=8, h=8)
    tab.doc.add_object(layer.uid, obj)
    depth = len(tab.doc.history)

    tab.doc.begin_object_edit(layer.uid, obj.uid)
    for step in range(1, 20):
        tab.doc.place_object(x=float(step), y=float(step))
    tab.doc.end_object_edit()

    assert len(tab.doc.history) == depth + 1
    assert (obj.x, obj.y) == (19.0, 19.0)
    tab.doc.undo()
    assert (obj.x, obj.y) == (0.0, 0.0)


def test_painting_a_locked_layer_toasts_and_pushes_nothing():
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, layer = _stamping(ctx)
    tab.doc.set_layer_props(layer.uid, locked=True)
    depth = len(tab.doc.history)

    plotter_canvas._apply(ctx, state, tab, (0, 0))
    assert not layer.data.any()
    assert len(tab.doc.history) == depth
    assert ctx.toasts and "locked" in ctx.toasts[-1][0]


def test_the_engine_still_writes_to_a_locked_layer():
    """The lock is enforced at the studio layer on purpose. ``write_region`` has
    to go on working, or an undo could not put back what was there before the
    lock -- and neither could the reader that opens a file with one set."""
    ctx = FakeCtx()
    tab, _state, layer = _stamping(ctx)
    tab.doc.set_layer_props(layer.uid, locked=True)
    assert tab.doc.write_region(layer.uid, 0, 0, np.array([[9]], gid.DTYPE)) is True
    assert int(layer.data[0, 0]) == 9


def test_a_lock_stops_a_cut_but_not_a_copy():
    """A lock blocks content edits; reading one changes nothing. Refusing the
    copy would make a lock a reason you cannot get at your own tiles."""
    ctx = FakeCtx()
    tab, state, layer, _value = _painted(ctx)
    tab.doc.set_layer_props(layer.uid, locked=True)
    state.select = (0, 0, 1, 1)

    plotter_mode._copy(ctx, state, tab, cut=False)
    assert state.clipboard is not None, "copying is allowed"

    state.clipboard = None
    plotter_mode._copy(ctx, state, tab, cut=True)
    assert state.clipboard is None
    assert layer.data.all(), "nothing was cut"
    assert "locked" in ctx.toasts[-1][0]


def test_a_lock_stops_delete():
    ctx = FakeCtx()
    tab, state, layer, _value = _painted(ctx)
    tab.doc.set_layer_props(layer.uid, locked=True)
    state.select = (0, 0, 1, 1)
    plotter_mode._delete(ctx, state, tab)
    assert layer.data.all()
    assert "locked" in ctx.toasts[-1][0]


def test_a_lock_stops_removing_an_object_but_not_selecting_it():
    ctx = FakeCtx()
    tab, state, _layer, _value = _painted(ctx)
    obj_layer = tab.doc.add_object_layer("Things")
    tab.doc.set_active_layer(obj_layer.uid)
    obj = MapObject(uid=new_uid(), name="spawn", kind="point", x=1, y=1)
    tab.doc.add_object(obj_layer.uid, obj)
    tab.doc.set_layer_props(obj_layer.uid, locked=True)

    state.tool = "object"
    state.select_object(obj.uid)
    plotter_mode._delete(ctx, state, tab)
    assert tab.doc.layers[-1].objects, "the object is still there"
    assert state.selected_object == obj.uid, "and still selected, so it can be read"
    assert "locked" in ctx.toasts[-1][0]


def _locked_object_layer(ctx: FakeCtx) -> tuple[Any, Any, Any, Any]:
    """A tab whose active layer is a locked object layer holding one object."""
    tab, state, _layer, _value = _painted(ctx)
    obj_layer = tab.doc.add_object_layer("Things")
    tab.doc.set_active_layer(obj_layer.uid)
    obj = MapObject(uid=new_uid(), name="spawn", kind="point", x=1, y=1)
    tab.doc.add_object(obj_layer.uid, obj)
    tab.doc.set_layer_props(obj_layer.uid, locked=True)
    state.tool = "object"
    state.select_object(obj.uid)
    return tab, state, obj_layer, obj


def test_a_lock_stops_cutting_an_object_but_not_copying_it():
    """Cutting removes the object, which is a content edit; and a refused cut
    copies nothing, or the user would paste what they believe they moved."""
    ctx = FakeCtx()
    tab, state, obj_layer, obj = _locked_object_layer(ctx)

    plotter_mode._copy(ctx, state, tab, cut=True)
    assert obj_layer.objects, "nothing was cut"
    assert state.clipboard is None, "and a refused cut copies nothing"
    assert "locked" in ctx.toasts[-1][0]

    plotter_mode._copy(ctx, state, tab, cut=False)
    assert isinstance(state.clipboard, plotter_mode.ObjectClip), "copying is allowed"


def test_a_lock_stops_pasting_an_object():
    """A paste adds an object, which is a content edit like any other."""
    ctx = FakeCtx()
    tab, state, obj_layer, obj = _locked_object_layer(ctx)
    state.clipboard = plotter_mode.ObjectClip(obj=obj)
    state.clipboard_doc = tab.uid

    plotter_mode._paste(ctx, state, tab)
    assert len(obj_layer.objects) == 1, "nothing landed"
    assert "locked" in ctx.toasts[-1][0]


def test_a_lock_stops_duplicating_an_object():
    ctx = FakeCtx()
    tab, state, obj_layer, _obj = _locked_object_layer(ctx)

    plotter_mode._duplicate_object(ctx, state, tab)
    assert len(obj_layer.objects) == 1, "nothing was added"
    assert "locked" in ctx.toasts[-1][0]


def _grouped(doc: Any, layer: Any, *, locked: bool = True) -> Any:
    """``layer`` moved into a fresh group, and the *group* is what gets locked.

    The v3 reader can hand this shape back from a file, so it is reachable
    without the tree pane ever being asked to build it.
    """
    group = doc.add_group_layer("Locked")
    doc.move_layer(layer.uid, 0, parent_uid=group.uid)
    doc.set_layer_props(group.uid, locked=bool(locked))
    doc.set_active_layer(layer.uid)
    return group


def _fake_imgui(monkeypatch, mouse: tuple[float, float], *, clicked: bool = True) -> None:
    """Just enough of imgui for ``_object_input``: where the pointer is and what
    the left button did. The canvas imports it inside the function, so the whole
    module is swapped rather than an attribute patched."""
    import sys
    from types import SimpleNamespace

    fake = SimpleNamespace(
        get_mouse_pos=lambda: SimpleNamespace(x=float(mouse[0]), y=float(mouse[1])),
        is_mouse_clicked=lambda _button: bool(clicked),
        is_mouse_down=lambda _button: False,
        is_mouse_released=lambda _button: False,
        get_io=lambda: SimpleNamespace(key_ctrl=False, key_shift=False, key_alt=False),
    )
    monkeypatch.setitem(sys.modules, "imgui_bundle", SimpleNamespace(imgui=fake))


def test_a_lock_on_a_group_stops_painting_inside_it():
    """The lock the canvas draws is the *resolved* one -- a group's lock is
    inherited -- so the lock the input path enforces has to be the same one, or
    a layer whose handles are hidden goes on taking paint."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, layer = _stamping(ctx)
    _grouped(tab.doc, layer)
    depth = len(tab.doc.history)

    plotter_canvas._apply(ctx, state, tab, (0, 0))

    assert not layer.data.any(), "the lock above it refused the write"
    assert len(tab.doc.history) == depth
    assert ctx.toasts and "locked" in ctx.toasts[-1][0]


def test_a_lock_on_a_group_stops_a_handle_drag_inside_it(monkeypatch):
    """The handles are not drawn on an inherited lock; without this the hit test
    still fires where one would sit, and the drag starts on an invisible grip."""
    from warlock.studio import inker_state
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    obj_layer = tab.doc.add_object_layer("Things")
    obj = MapObject(uid=new_uid(), name="zone", kind="rect", x=0, y=0, w=16, h=16)
    tab.doc.add_object(obj_layer.uid, obj)
    _grouped(tab.doc, obj_layer)
    state.select_object(obj.uid)
    origin = (0.0, 0.0)
    _fake_imgui(monkeypatch, inker_state.to_screen(tab.view, origin, obj.x, obj.y))

    plotter_canvas._object_input(ctx, state, tab, origin, True)

    assert state.drag_kind == "", "no resize began on a layer under a locked group"


def test_a_lock_on_a_group_stops_drawing_a_new_object_inside_it(monkeypatch):
    from warlock.studio import inker_state
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    obj_layer = tab.doc.add_object_layer("Things")
    _grouped(tab.doc, obj_layer)
    origin = (0.0, 0.0)
    _fake_imgui(monkeypatch, inker_state.to_screen(tab.view, origin, 40.0, 40.0))

    plotter_canvas._object_input(ctx, state, tab, origin, True)

    assert state.drag_kind == ""
    assert ctx.toasts and "locked" in ctx.toasts[-1][0]


def test_locking_leaves_the_rest_of_the_stack_alone():
    """Tiled's semantics: a lock is there to stop you painting on the wrong
    layer, not to stop you managing the stack."""
    ctx = FakeCtx()
    tab, _state, layer = _stamping(ctx)
    tab.doc.set_layer_props(layer.uid, locked=True)

    tab.doc.set_layer_props(layer.uid, name="Renamed", visible=False, opacity=0.5)
    assert (layer.name, layer.visible, layer.opacity) == ("Renamed", False, 0.5)

    tab.doc.set_layer_props(layer.uid, locked=False)
    assert layer.locked is False, "and unlocking is always available"


def test_the_shape_tool_fills_whichever_shape_is_chosen():
    """One tool with a mode, not two tools: the gesture, the preview and the
    undo step are the same and only the set of cells differs."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, layer = _stamping(ctx)
    state.tool = "shape"

    state.shape_mode = "rect"
    plotter_canvas._apply_shape(ctx, state, tab, (0, 0), (3, 3))
    assert int((layer.data != 0).sum()) == 16, "a rect fills its corners"

    tab.doc.undo()
    state.shape_mode = "ellipse"
    plotter_canvas._apply_shape(ctx, state, tab, (0, 0), (3, 3))
    assert int((layer.data != 0).sum()) == 12, "an ellipse does not"
    assert layer.data[0, 0] == 0


def test_the_shape_preview_outlines_the_box_it_would_fill():
    """Pure lattice arithmetic, like ``_corner_uvs``: the outline is measured to
    the *far* edge of the last cell, not its near edge, or the preview sits one
    cell short of what lands."""
    from warlock.studio.panes import plotter_canvas

    corners = plotter_canvas._shape_points("rect", (1, 2), (4, 5))
    assert corners == [(1.0, 2.0), (5.0, 2.0), (5.0, 6.0), (1.0, 6.0)]

    # Dragged the other way round, the same outline.
    assert plotter_canvas._shape_points("rect", (4, 5), (1, 2)) == corners


def test_the_ellipse_preview_is_sampled_inside_the_same_box():
    from warlock.studio.panes import plotter_canvas

    points = plotter_canvas._shape_points("ellipse", (0, 0), (7, 7))
    assert len(points) == plotter_canvas._ELLIPSE_SAMPLES
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    assert min(xs) == pytest.approx(0.0) and max(xs) == pytest.approx(8.0)
    assert min(ys) == pytest.approx(0.0) and max(ys) == pytest.approx(8.0)
    # Every sample is on the ellipse, which is what makes it a preview of the
    # fill rather than a shape that merely fits the same box.
    for x, y in points:
        assert ((x - 4.0) / 4.0) ** 2 + ((y - 4.0) / 4.0) ** 2 == pytest.approx(1.0)


def test_a_shape_fill_is_one_undo_step():
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, _layer = _stamping(ctx)
    state.tool = "shape"
    state.shape_mode = "ellipse"
    depth = len(tab.doc.history)
    plotter_canvas._apply_shape(ctx, state, tab, (0, 0), (3, 3))
    assert len(tab.doc.history) == depth + 1


def test_the_shape_tool_still_needs_something_in_hand():
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, layer = _stamping(ctx)
    state.tool = "shape"
    state.brush = None
    plotter_canvas._apply_shape(ctx, state, tab, (0, 0), (2, 2))
    assert len(ctx.toasts) == 1
    assert not layer.data.any()


def test_a_line_needs_shift_a_stamp_and_somewhere_to_start_from():
    """All three, because each rules out a different wrong line: no shift is an
    ordinary click, another tool has no "from" to draw from, and no last cell
    means the user has not placed anything in this map yet."""
    from warlock.studio.panes import plotter_canvas

    class Io:
        def __init__(self, shift):
            self.key_shift = shift

    ctx = FakeCtx()
    _tab(ctx)
    state = plotter_mode.ensure(ctx)
    state.tool = "stamp"
    state.last_paint = (1, 1)

    assert plotter_canvas._line_pending(state, Io(True)) is True
    assert plotter_canvas._line_pending(state, Io(False)) is False

    state.tool = "erase"
    assert plotter_canvas._line_pending(state, Io(True)) is False

    state.tool = "stamp"
    state.last_paint = None
    assert plotter_canvas._line_pending(state, Io(True)) is False


def test_clearing_a_drag_keeps_the_line_origin():
    """``drag_last_cell`` belongs to the gesture and ``last_paint`` outlives it
    -- the whole point of the second is to be there on the *next* click."""
    ctx = FakeCtx()
    _tab(ctx)
    state = plotter_mode.ensure(ctx)
    state.last_paint = (2, 3)
    state.drag_last_cell = (2, 3)
    state.clear_drag()
    assert state.drag_last_cell is None
    assert state.last_paint == (2, 3)


def _terrain_tab(ctx: FakeCtx):
    """A tab with a terrain set adopted and that terrain in hand."""
    from plotter._terrainset import terrain_tileset

    tab = _tab(ctx, tileset=False)
    plotter_mode.on_task_done(
        ctx,
        _Done(f"plotter-tileset:{tab.uid}", {
            "tileset": terrain_tileset(tile_w=8, tile_h=8),
            "source": "",
            "uid": tab.uid,
        }),
    )
    return tab, plotter_mode.ensure(ctx), tab.doc.tilesets[-1]


def test_erasing_a_terrain_cell_re_fits_what_surrounded_it():
    """A terrain hole has to grow an outline on everything that now borders it,
    or the field keeps the edge art of a neighbour that is no longer there."""
    from warlock.studio.panes import plotter_canvas
    from warlock.studio.plotter import terrain as terrainlib

    ctx = FakeCtx()
    tab, state, ref = _terrain_tab(ctx)
    layer = tab.doc.tile_layers()[0]
    cells = [(x, y) for y in range(4) for x in range(4)]
    tab.doc.write_region(
        layer.uid, *terrainlib.paint_terrain_cells(layer.data, cells, 0, ref)
    )
    neighbour = int(layer.data[1, 1])
    depth = len(tab.doc.history)

    state.tool = "erase"
    plotter_canvas._apply(ctx, state, tab, (2, 2))
    assert int(layer.data[2, 2]) == 0
    assert int(layer.data[1, 1]) != neighbour, "the ring grew an edge against the hole"
    # One region, so one step: the eight-neighbour fix-up lives inside the same
    # rectangle as the erase.
    assert len(tab.doc.history) == depth + 1
    tab.doc.undo()
    assert int(layer.data[1, 1]) == neighbour and int(layer.data[2, 2]) != 0


def test_erasing_a_plain_cell_on_a_terrain_map_is_still_a_plain_erase():
    """Self-selecting per cell is the only rule that makes one eraser correct on
    a mixed map; the alternative is a second eraser and a user deciding which."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, ref = _terrain_tab(ctx)
    tab.doc.add_tileset(_tileset("plain"))
    plain = tab.doc.tilesets[-1]
    layer = tab.doc.tile_layers()[0]
    tab.doc.write_region(
        layer.uid, 2, 2, np.array([[plain.firstgid]], gid.DTYPE)
    )
    tab.doc.write_region(layer.uid, 0, 0, np.array([[ref.firstgid]], gid.DTYPE))
    untouched = int(layer.data[0, 0])

    state.tool = "erase"
    plotter_canvas._apply(ctx, state, tab, (2, 2))
    assert int(layer.data[2, 2]) == 0
    assert int(layer.data[0, 0]) == untouched, "a plain erase re-fits nothing"


def test_fill_with_a_terrain_in_hand_floods_instead_of_refusing():
    """Fill used to toast "Pick a tile from the tileset first" whenever the
    brush was empty, including with a terrain unambiguously in hand."""
    from warlock.studio.panes import plotter_canvas
    from warlock.studio.plotter import terrain as terrainlib

    ctx = FakeCtx()
    tab, state, ref = _terrain_tab(ctx)
    layer = tab.doc.tile_layers()[0]
    block = [(x, y) for y in range(1, 4) for x in range(1, 4)]
    tab.doc.write_region(
        layer.uid, *terrainlib.paint_terrain_cells(layer.data, block, 0, ref)
    )
    state.tool = "fill"
    state.brush = None
    state.terrain = (len(tab.doc.tilesets) - 1, 2)
    ctx.toasts.clear()

    plotter_canvas._apply(ctx, state, tab, (2, 2))
    ranks = terrainlib.rank_field(layer.data, ref)
    assert {int(ranks[y, x]) for x, y in block} == {2}
    assert not [t for t in ctx.toasts if t[1] == "error"]


def test_fill_with_a_tile_in_hand_is_the_plain_flood_it_always_was():
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, ref = _terrain_tab(ctx)
    layer = tab.doc.tile_layers()[0]
    state.tool = "fill"
    state.brush = np.array([[ref.firstgid]], gid.DTYPE)
    plotter_canvas._apply(ctx, state, tab, (0, 0))
    # A raw flood puts the *picked gid* in every cell, blob case and all --
    # which is exactly what a terrain fill would not do.
    assert (layer.data == ref.firstgid).all()


def test_fill_with_neither_still_says_so():
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    state.tool = "fill"
    state.brush = None
    state.terrain = None
    plotter_canvas._apply(ctx, state, tab, (0, 0))
    assert ctx.toasts[-1][1] == "error"


def test_the_canvas_and_the_flat_renderer_agree_about_every_flag():
    """Two renderers exist and neither is redundant, so the thing that has to be
    pinned is that they *agree*. Both apply the flags transpose-then-mirror;
    ``render.orient`` does it to pixels and ``_corner_uvs`` to four UV corners,
    which is what makes the diagonal flip drawable at all. This is here rather
    than in ``tests/plotter/`` because it reaches into a pane -- ``_corner_uvs``
    is imgui-free and pure, so importing it headlessly is safe.
    """
    from warlock.studio.panes.plotter_canvas import _corner_uvs
    from warlock.studio.plotter.render import orient

    # Four distinct corners, so every permutation is distinguishable.
    tile = np.zeros((2, 2, 4), dtype=np.uint8)
    for index, (row, column) in enumerate(((0, 0), (0, 1), (1, 1), (1, 0))):
        tile[row, column] = (index + 1, 0, 0, 255)
    # ``_corner_uvs`` returns TL, TR, BR, BL over the unit square; each corner
    # UV names the source pixel that ends up drawn at that screen corner.
    screen_corners = ((0, 0), (0, 1), (1, 1), (1, 0))  # TL, TR, BR, BL as (row, column)

    for mask in range(8):
        flip_h, flip_v, flip_d = bool(mask & 1), bool(mask & 2), bool(mask & 4)
        drawn = orient(tile, flip_h, flip_v, flip_d)
        uvs = _corner_uvs((0.0, 0.0, 1.0, 1.0), flip_h, flip_v, flip_d)
        for (row, column), (u, v) in zip(screen_corners, uvs, strict=True):
            source = tile[int(v), int(u)]
            assert np.array_equal(drawn[row, column], source), (
                f"flags h={flip_h} v={flip_v} d={flip_d} disagree at {(row, column)}"
            )


def test_the_tileset_memo_is_kept_until_the_epoch_moves():
    """Which tileset owns an id is a linear scan, asked once per visible cell
    per layer. It is memoised, and ``tileset_epoch`` is the only thing that may
    invalidate it. Imgui-free like ``_corner_uvs``, so it is safe headlessly.
    """
    from warlock.studio.panes import plotter_canvas

    plotter_canvas.forget_all()
    memo = plotter_canvas._index_memo("tab-a", 0)
    memo[7] = 3
    # Same epoch: the very same dict, so a seeded answer is what the draw loop
    # reads back -- this is the test that the memo is consulted at all.
    assert plotter_canvas._index_memo("tab-a", 0) is memo
    assert plotter_canvas._index_memo("tab-a", 0)[7] == 3

    moved = plotter_canvas._index_memo("tab-a", 1)
    assert moved is not memo
    assert moved == {}


def test_the_tileset_memo_remembers_that_an_id_belongs_to_nothing():
    """``None`` is the expensive answer, not the missing one: every cell painted
    from a since-detached tileset scans the whole list to reach it."""
    from warlock.studio.panes import plotter_canvas

    plotter_canvas.forget_all()
    memo = plotter_canvas._index_memo("tab-a", 0)
    memo[9] = None
    assert 9 in memo
    assert memo.get(9, plotter_canvas._UNMEMOISED) is None


def test_each_document_memoises_separately():
    from warlock.studio.panes import plotter_canvas

    plotter_canvas.forget_all()
    plotter_canvas._index_memo("tab-a", 0)[1] = 0
    plotter_canvas._index_memo("tab-b", 0)[1] = 1
    assert plotter_canvas._index_memo("tab-a", 0)[1] == 0
    assert plotter_canvas._index_memo("tab-b", 0)[1] == 1

    plotter_canvas.forget_doc("tab-a")
    assert plotter_canvas._index_memo("tab-a", 0) == {}
    assert plotter_canvas._index_memo("tab-b", 0)[1] == 1


def test_closing_a_tab_drops_its_tileset_memo():
    """A tab uid is never reissued, so a memo left behind leaks rather than
    merely going stale -- released at the moment the textures are."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab = _tab(ctx)
    plotter_canvas.forget_all()
    plotter_canvas._index_memo(tab.uid, tab.doc.tileset_epoch)[1] = 0
    assert tab.uid in plotter_canvas._TILESET_MEMO

    plotter_mode.close_tab(ctx, tab.uid)
    assert tab.uid not in plotter_canvas._TILESET_MEMO


class _FakeTexture:
    def __init__(self) -> None:
        self.filter = None
        self.released = False

    def release(self) -> None:
        self.released = True


class _FakeGL:
    NEAREST = 0x2600

    def texture(self, size: Any, components: int, data: bytes) -> _FakeTexture:
        return _FakeTexture()


class _FakeViewer:
    def __init__(self) -> None:
        self.ctx = _FakeGL()


def test_a_replaced_tileset_gets_a_fresh_texture_and_a_kept_one_does_not():
    """The staleness stamp is ``tileset_epoch``, a monotonic counter -- it was
    ``id(pixels)`` before, and CPython recycles an id after GC, so a
    replacement atlas landing on a recycled id false-matched and the stale
    atlas drew forever."""
    from warlock.studio.panes import plotter_textures

    ctx = FakeCtx()
    tab = _tab(ctx)
    ctx.viewer = _FakeViewer()
    doc = tab.doc

    first = plotter_textures.tileset_texture(
        ctx, tab.uid, 0, doc.tilesets[0].tileset, doc.tileset_epoch
    )
    again = plotter_textures.tileset_texture(
        ctx, tab.uid, 0, doc.tilesets[0].tileset, doc.tileset_epoch
    )
    assert again is first and not first.released

    doc.replace_tileset(0, _tileset("repainted"))
    fresh = plotter_textures.tileset_texture(
        ctx, tab.uid, 0, doc.tilesets[0].tileset, doc.tileset_epoch
    )
    assert fresh is not first
    assert first.released


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
    """One KEYDOWN carrying its own modifier state. ``handle_key`` reads
    ``event.mod`` (``main._shortcut``'s UX-12 rule), never ``get_mods()``."""
    import pygame

    mods = (pygame.KMOD_CTRL if ctrl else 0) | (pygame.KMOD_SHIFT if shift else 0)
    return pygame.event.Event(pygame.KEYDOWN, key=getattr(pygame, f"K_{name}"), mod=mods)


# --- the tileset palette's zoom -----------------------------------------------
#
# The picker used to floor its zoom at 1:1 and submit the atlas at full width
# into a child with no horizontal scrollbar, so every column past the pane's
# edge was clipped away with no way to reach it. These are the pure half of the
# fix: the pane still owns the viewport, but the arithmetic that decides the
# scale is here and assertable without a window.


def test_the_palette_ladder_is_reciprocal_below_one_and_whole_above():
    """The pixel-art rule, and here it is about NEAREST rather than taste.

    The atlas is uploaded with nearest filtering, so a scale like 0.7 samples a
    non-uniform subset of source pixels and the palette grid beats against
    itself. 1/N samples uniformly and merely gets smaller.
    """
    from warlock.studio import plotter_state as ps

    ladder = ps.PALETTE_ZOOM_LADDER
    assert ladder == tuple(sorted(ladder))
    assert 1.0 in ladder
    for rung in ladder:
        whole = 1.0 / rung if rung < 1.0 else rung
        assert abs(whole - round(whole)) < 1e-9, rung

    # Not ``ZOOM_LADDER``, and the divergence is the point: that table bottoms
    # out at 5%, which puts a 16px tile at 0.8 screen pixels -- unhittable.
    from warlock.studio import inker_state

    assert ladder != inker_state.ZOOM_LADDER
    assert 16 * ladder[0] >= 1.0


def test_fitting_picks_the_largest_rung_that_shows_the_whole_atlas():
    """"View the entire tile set", which is what the pane could not do."""
    from warlock.studio import plotter_state as ps

    # A 1024px atlas in a 300px pane: must go below 1:1, which the old
    # ``max(1, ...)`` floor made impossible.
    zoom = ps.palette_fit_zoom(1024, 1024, 300.0, 300.0)
    assert zoom < 1.0
    assert 1024 * zoom <= 300.0
    assert zoom in ps.PALETTE_ZOOM_LADDER

    # Both axes, not just width: a tall atlas in a wide pane fits on height.
    tall = ps.palette_fit_zoom(64, 2048, 600.0, 260.0)
    assert 2048 * tall <= 260.0

    # A small atlas is allowed to magnify rather than being pinned at 1:1.
    assert ps.palette_fit_zoom(32, 32, 300.0, 260.0) > 1.0


def test_fitting_answers_rather_than_dividing_by_zero():
    """A tileset with no pixels is a load that failed, not a crash in the loop."""
    from warlock.studio import plotter_state as ps

    floor = ps.PALETTE_ZOOM_LADDER[0]
    assert ps.palette_fit_zoom(0, 0, 300.0, 260.0) == floor
    assert ps.palette_fit_zoom(64, 64, 0.0, 0.0) == floor
    # Nothing fits even at the floor: take the floor and let the well scroll.
    assert ps.palette_fit_zoom(100_000, 100_000, 300.0, 260.0) == floor


def test_a_zoom_nudge_steps_strictly_past_a_fitted_scale():
    """Why the rung rule is shared rather than reimplemented.

    A fit-derived zoom routinely sits *between* two rungs, and a
    nearest-then-step rule would answer a press labelled "in" by zooming out.
    """
    from warlock.studio import plotter_state as ps

    ladder = ps.PALETTE_ZOOM_LADDER
    between = (ladder[3] + ladder[4]) / 2.0
    assert ps.palette_zoom_rung(between, 1) > between
    assert ps.palette_zoom_rung(between, -1) < between
    # The ends hold rather than running off the table.
    assert ps.palette_zoom_rung(ladder[0], -1) == ladder[0]
    assert ps.palette_zoom_rung(ladder[-1], 1) == ladder[-1]
    # On a rung, a step moves to the neighbour rather than standing still.
    assert ps.palette_zoom_rung(1.0, 1) > 1.0
    assert ps.palette_zoom_rung(1.0, -1) < 1.0


def test_the_palette_zoom_is_remembered_per_tileset():
    """One number for the mode would leave a reader staring at a corner.

    The zoom that shows a 1024px terrain set whole is the zoom that shows one
    tile of a 64px four-tile set, so switching the combo must not carry a scale
    across with it.
    """
    ctx = FakeCtx()
    tab = _tab(ctx)

    assert tab.palette_zoom == {}, "a missing key is what means 'fit'"
    tab.palette_zoom[0] = 2.0
    tab.palette_zoom[1] = 0.25
    assert tab.palette_zoom[0] == 2.0


def test_a_tool_letter_picks_that_tool(monkeypatch):
    """Every letter, derived from ``TOOLS`` rather than one hard-coded pair:
    this test named G/fill and so had to be edited by hand when the keymap moved
    to Tiled's letters, which is exactly the drift the derivation prevents."""

    from warlock.studio import plotter_state

    ctx = FakeCtx()
    _tab(ctx)
    state = plotter_mode.ensure(ctx)
    for key, _label, letter in plotter_state.TOOLS:
        state.tool = "stamp" if key != "stamp" else "erase"
        event = _key(letter.lower())
        assert plotter_mode.handle_key(ctx, event) is True, letter
        assert state.tool == key, f"{letter} should pick {key}"


def test_no_tool_letter_collides_with_another_binding():
    """X, Y and Z transform the brush and are read before the tool table, so a
    tool that took one of them would be unreachable rather than ambiguous."""
    from warlock.studio import plotter_mode as mode
    from warlock.studio import plotter_state

    letters = [letter.lower() for _k, _l, letter in plotter_state.TOOLS]
    assert len(letters) == len(set(letters)), "two tools share a letter"
    assert not set(letters) & set(mode._BRUSH_TRANSFORMS), "a tool shadows a brush transform"


def test_undo_is_refused_while_the_tab_is_busy(monkeypatch):
    """Consumed, not passed through: the key belongs to this mode either way,
    and letting it fall through would act on a viewport Plotter has replaced."""

    ctx = FakeCtx()
    tab = _tab(ctx, dirty=True)
    tab.saving = True
    head = tab.doc.history.head
    event = _key("z", ctrl=True)
    assert plotter_mode.handle_key(ctx, event) is True
    assert tab.doc.history.head == head


def test_undo_works_when_it_is_not(monkeypatch):

    ctx = FakeCtx()
    tab = _tab(ctx, dirty=True)
    event = _key("z", ctrl=True)
    plotter_mode.handle_key(ctx, event)
    assert not tab.dirty


def test_ctrl_shift_z_redoes_the_way_inker_and_clay_accept(monkeypatch):
    """Plotter was Ctrl+Y only, so a user arriving from either of the other two
    editors found their redo silently doing nothing."""

    ctx = FakeCtx()
    tab = _tab(ctx, dirty=True)
    dirty_data = tab.doc.tile_layers()[0].data.copy()

    event = _key("z", ctrl=True)
    plotter_mode.handle_key(ctx, event)
    assert not tab.dirty

    event = _key("z", ctrl=True, shift=True)
    plotter_mode.handle_key(ctx, event)
    assert tab.dirty
    assert np.array_equal(tab.doc.tile_layers()[0].data, dirty_data)


def test_the_two_empty_states_name_the_same_droppable_suffixes():
    """Two hand-written copies existed and they already disagreed."""
    from warlock.studio import plotter_state

    assert plotter_state.MAP_SUFFIX_TEXT == ".wmap / .tmx / .tmj"


def test_a_key_release_is_never_consumed():
    import pygame

    ctx = FakeCtx()
    _tab(ctx)
    up = pygame.event.Event(pygame.KEYUP, key=pygame.K_g, mod=0)
    assert plotter_mode.handle_key(ctx, up) is False


def test_x_y_and_z_transform_the_brush_in_hand(monkeypatch):

    ctx = FakeCtx()
    _tab(ctx)
    state = plotter_mode.ensure(ctx)
    state.brush = np.array([[1, 2], [3, 4]], gid.DTYPE)

    event = _key("x")
    assert plotter_mode.handle_key(ctx, event) is True
    assert np.array_equal(gid.tile_ids(state.brush), [[2, 1], [4, 3]])

    event = _key("y")
    plotter_mode.handle_key(ctx, event)
    assert np.array_equal(gid.tile_ids(state.brush), [[4, 3], [2, 1]])


def test_shift_z_turns_the_brush_the_other_way(monkeypatch):
    """Three clockwise turns rather than a counter-clockwise routine of its
    own: Z then Shift+Z must be the identity."""

    ctx = FakeCtx()
    _tab(ctx)
    state = plotter_mode.ensure(ctx)
    original = np.array([[1, 2], [3, 4]], gid.DTYPE)
    state.brush = original.copy()

    event = _key("z")
    plotter_mode.handle_key(ctx, event)
    assert not np.array_equal(state.brush, original)

    event = _key("z", shift=True)
    plotter_mode.handle_key(ctx, event)
    assert np.array_equal(state.brush, original)


def test_a_brush_transform_is_not_refused_while_the_tab_is_busy(monkeypatch):
    """The brush is view state: transforming it writes nothing and pushes no
    step, so it stays outside the gate that ``Ctrl+Z`` sits behind."""

    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True
    state = plotter_mode.ensure(ctx)
    state.brush = np.array([[1, 2]], gid.DTYPE)
    head = tab.doc.history.head

    event = _key("x")
    assert plotter_mode.handle_key(ctx, event) is True
    assert np.array_equal(gid.tile_ids(state.brush), [[2, 1]])
    assert tab.doc.history.head == head
    assert not tab.dirty


def test_a_brush_transform_with_nothing_in_hand_does_nothing(monkeypatch):

    ctx = FakeCtx()
    _tab(ctx)
    state = plotter_mode.ensure(ctx)
    state.brush = None
    event = _key("x")
    assert plotter_mode.handle_key(ctx, event) is False
    assert state.brush is None


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
#
# Plotter no longer *makes* one -- the procedural and AI generators were both
# retired on 2026-08-18 when tile sheets moved to Create. What is still
# Plotter's and is still tested here is the *arrival*: a terrain-carrying
# tileset landing on the shared ``plotter-tileset:`` key, and everything
# ``on_task_done`` does with one.


def _arrive(ctx, tab, *, projection=None, **kwargs):
    """Land a terrain set on ``tab`` the way every door here lands one."""
    from plotter._terrainset import terrain_tileset

    kwargs.setdefault("tile_w", 8)
    kwargs.setdefault("tile_h", 8)
    result = {
        "tileset": terrain_tileset(**kwargs),
        "source": "",
        "uid": tab.uid,
    }
    if projection is not None:
        result["projection"] = projection
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", result))
    return tab.doc.tilesets[-1]


def test_an_arriving_set_is_one_tileset_and_one_undo_step():
    from warlock.studio.tilegrid import blob

    ctx = FakeCtx()
    tab = _tab(ctx)
    depth = len(tab.doc.history)
    before = len(tab.doc.tilesets)
    _arrive(ctx, tab)
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
    _arrive(ctx, tab)
    assert plotter_mode.ensure(ctx).terrain == (len(tab.doc.tilesets) - 1, 0)


def test_a_set_and_its_projection_arrive_in_the_same_step():
    """Two steps would leave a Ctrl+Z on a map whose only tileset is drawn for
    the lattice it is no longer on."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    depth = len(tab.doc.history)
    before = len(tab.doc.tilesets)
    _arrive(ctx, tab, projection="isometric", tile_w=32, tile_h=16)
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
    _arrive(ctx, tab, projection="isometric", tile_w=32, tile_h=16)
    assert tab.view.fitted is False


def test_sending_an_atlas_back_from_inker_keeps_every_painted_cell():
    from warlock.studio.plotter import terrain

    ctx = FakeCtx()
    tab = _tab(ctx)
    _arrive(ctx, tab)
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
    _arrive(ctx, tab)
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


# --- the property rows the layers pane draws ----------------------------------
#
# The pane itself needs a frame to draw, but what it *offers* and what it makes
# of a value it cannot edit are plain functions, and both are the half that goes
# wrong when the property model grows a type.


def test_the_new_property_row_offers_scalar_and_recursive_types():
    """Class and list start empty, then unfold into the recursive editor."""
    from warlock.studio.panes import plotter_layers

    assert plotter_layers.AUTHORABLE_TYPES == (
        "string",
        "int",
        "float",
        "bool",
        "color",
        "file",
        "object",
        "class",
        "list",
    )
    assert plotter_layers._blank_value("object") == 0
    assert plotter_layers._blank_value("file") == ""
    assert plotter_layers._blank_value("class") == {}
    assert plotter_layers._blank_value("list") == []


def test_a_container_property_gets_a_compact_summary_for_its_editor_header():
    from warlock.studio.panes import plotter_layers
    from warlock.studio.plotter.props import Prop

    npc = Prop("class", {"hp": Prop("int", 3), "name": Prop("string", "Bob")}, propertytype="NPC")
    assert plotter_layers._summary(npc) == "NPC (2 members)"
    assert plotter_layers._summary(Prop("list", [Prop("int", 1)])) == "list (1 item)"


def test_object_property_choices_use_persistent_ids_in_recursive_layers():
    from warlock.studio.panes import plotter_layers

    ctx = FakeCtx()
    tab = _tab(ctx)
    group = tab.doc.add_group_layer("Nested")
    layer = tab.doc.add_object_layer("Things", parent_uid=group.uid)
    obj = tab.doc.add_object(
        layer.uid, MapObject(uid=new_uid(), name="Spawn", kind="point")
    )
    assert plotter_layers.object_options(tab.doc) == [
        ("0", "None"),
        (str(obj.id), f"Spawn ({obj.id})"),
    ]


# --- asking before making -----------------------------------------------------


def test_ask_new_document_asks_and_makes_nothing():
    """The whole point of the door. It used to *be* ``new_document``, so a map
    existed at 32x32x32 orthogonal before anybody was offered the choice -- and
    two of those numbers cannot be taken back once tiles are on the map."""
    ctx = FakeCtx()
    plotter_mode.ask_new_document(ctx)
    state = plotter_mode.ensure(ctx)
    assert state.setup_pending is True
    assert state.docs == []


def test_ask_new_document_switches_to_plotter():
    """Home and the command palette are not Plotter panes, and the dialog is
    Plotter's -- so the door has to carry the caller to where it is drawn."""
    ctx = FakeCtx()
    ctx.state.mode = "home"
    plotter_mode.ask_new_document(ctx)
    assert ctx.state.mode == "plotter"


def test_a_new_document_takes_the_projection_it_is_given():
    ctx = FakeCtx()
    tab = plotter_mode.new_document(ctx, (4, 4, 64, 32), projection="isometric")
    assert tab.doc.projection == "isometric"
    assert (tab.doc.tile_w, tab.doc.tile_h) == (64, 32)


def test_a_new_document_still_opens_clean_and_undoes_to_nothing():
    """The ``history.clear()`` contract, re-pinned because the projection
    argument now runs before it."""
    ctx = FakeCtx()
    tab = plotter_mode.new_document(ctx, (4, 4, 16, 16), projection="isometric")
    assert not tab.doc.dirty
    # Nothing to undo: the Ground layer is added before the history is cleared.
    assert tab.doc.undo() is False


def test_every_new_map_door_asks_rather_than_inventing():
    """Five doors, one of them a keyboard shortcut, and the bug they all had was
    the same: calling ``new_document`` straight. A source scan rather than five
    click tests, because what matters is that no *sixth* door reintroduces it.
    """
    import inspect

    from warlock.studio import palette
    from warlock.studio.panes import landing, plotter_bridge, plotter_canvas

    for module in (palette, landing, plotter_bridge, plotter_canvas, plotter_mode):
        source = inspect.getsource(module)
        for number, line in enumerate(source.splitlines(), 1):
            if "new_document(ctx)" not in line or "ask_new_document" in line:
                continue
            assert "plotter_mode" not in line and "def " not in line, (
                f"{module.__name__}:{number} makes a map without asking: {line.strip()}"
            )


def test_create_builds_the_map_the_form_describes(monkeypatch):
    """The dialog's one job. ``_create`` is what Create is wired to, so this is
    the assertion that the numbers on screen are the numbers in the map."""
    from warlock.studio import plotter_setup
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    form = plotter_setup.blank_form()
    form.update(
        width=12, height=9, tile_w=64, tile_h=32,
        projection="isometric", next=plotter_setup.NEXT_EMPTY,
    )
    plotter_canvas._create(ctx, form)

    tab = plotter_mode.ensure(ctx).active
    assert (tab.doc.width, tab.doc.height) == (12, 9)
    assert (tab.doc.tile_w, tab.doc.tile_h) == (64, 32)
    assert tab.doc.projection == "isometric"


def test_create_opens_the_tileset_door_the_form_chose(monkeypatch):
    """A new map cannot be painted until it has a tileset, so the dialog offers
    both doors rather than leaving the user to find them. Monkeypatched because
    one of them opens an OS picker."""
    from warlock.studio import plotter_setup, widgets
    from warlock.studio.panes import plotter_canvas

    asked: list[str] = []
    monkeypatch.setattr(plotter_mode, "ask_add_tileset", lambda _c: asked.append("file"))
    monkeypatch.setattr(widgets, "request_open", lambda key: asked.append(key))

    for choice, expected in (
        (plotter_setup.NEXT_FILE, "file"),
        (plotter_setup.NEXT_EMPTY, None),
    ):
        asked.clear()
        form = plotter_setup.blank_form()
        form["next"] = choice
        plotter_canvas._create(FakeCtx(), form)
        assert asked == ([expected] if expected else [])


def test_create_clamps_a_number_the_field_would_have_accepted(monkeypatch):
    """The cap is enforced where the map is built, not only where it is typed:
    the form is restored from ``state.preview`` and could carry anything."""
    from warlock.studio import plotter_setup
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    form = plotter_setup.blank_form()
    form.update(width=99999, next=plotter_setup.NEXT_EMPTY)
    plotter_canvas._create(ctx, form)
    assert plotter_mode.ensure(ctx).active.doc.width == plotter_setup.MAX_TILES


# --- ruled tilesheets (Part A) -------------------------------------------------


def _ruled_sheet(
    rows: int = 3, cols: int = 3, cell: int = 16, sep: int = 2, fill: int = 200
) -> np.ndarray:
    """A sheet whose cells are ruled off by dark separator lines."""
    height = rows * cell + sep * (rows - 1)
    width = cols * cell + sep * (cols - 1)
    out = np.zeros((height, width, 4), dtype=np.uint8)
    out[:, :, 3] = 255
    for r in range(rows):
        for c in range(cols):
            y, x = r * (cell + sep), c * (cell + sep)
            out[y:y + cell, x:x + cell, :3] = fill + r * 5 + c
    return out


def _save_png(path: Path, pixels: np.ndarray) -> Path:
    from PIL import Image

    Image.fromarray(pixels, "RGBA").save(path)
    return path


def test_a_ruled_sheet_parks_for_confirmation(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    png = _save_png(tmp_path / "sheet.png", _ruled_sheet())

    plotter_mode.add_tileset_path(ctx, png)
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    state = plotter_mode.ensure(ctx)
    # Nothing has been decided: the detection is a suggestion, so the document
    # is untouched until the popup is answered.
    assert len(tab.doc.tilesets) == 0
    assert state.sheet_import is not None
    assert state.sheet_import[0] == tab.uid
    assert state.sheet_import[4].shape == (3, 3)
    assert state.sheet_import_open is False


def test_a_plain_image_still_slices_blind(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    flat = np.full((32, 32, 4), 200, dtype=np.uint8)
    flat[:, :, 3] = 255
    png = _save_png(tmp_path / "flat.png", flat)

    plotter_mode.add_tileset_path(ctx, png)
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    assert plotter_mode.ensure(ctx).sheet_import is None
    assert len(tab.doc.tilesets) == 1
    assert tab.doc.tilesets[0].tileset.tile_w == tab.doc.tile_w


def test_importing_a_detected_sheet_recomposes_at_map_tile_size(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    png = _save_png(tmp_path / "sheet.png", _ruled_sheet())
    plotter_mode.add_tileset_path(ctx, png)
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    assert plotter_mode.import_detected_sheet(ctx) is True

    state = plotter_mode.ensure(ctx)
    assert state.sheet_import is None
    assert state.sheet_import_open is False
    ref = tab.doc.tilesets[0]
    assert ref.tileset.tile_w == tab.doc.tile_w
    assert ref.tileset.tile_count == 9
    # A recomposed atlas is no longer the file on disk, so a ``.tmx`` export
    # must not reference it.
    assert ref.source == ""
    # Every separator pixel is gone: the darkest cell fill is 200.
    assert ref.tileset.pixels[:, :, :3].min() >= 200


def test_the_blind_answer_matches_todays_slice(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    sheet = _ruled_sheet()
    png = _save_png(tmp_path / "sheet.png", sheet)
    plotter_mode.add_tileset_path(ctx, png)
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    assert plotter_mode.import_sheet_blind(ctx) is True

    ref = tab.doc.tilesets[0]
    assert np.array_equal(np.asarray(ref.tileset.pixels), sheet)
    assert ref.source == str(png)
    assert plotter_mode.ensure(ctx).sheet_import is None


def test_a_parked_sheet_for_a_closed_tab_is_dropped(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    png = _save_png(tmp_path / "sheet.png", _ruled_sheet())
    plotter_mode.add_tileset_path(ctx, png)
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    state = plotter_mode.ensure(ctx)
    state.close(tab.uid)
    assert plotter_mode.import_detected_sheet(ctx) is False
    assert state.sheet_import is None


def test_switching_tabs_drops_the_parked_sheet(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    png = _save_png(tmp_path / "sheet.png", _ruled_sheet())
    plotter_mode.add_tileset_path(ctx, png)
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    state = plotter_mode.ensure(ctx)
    other = plotter_mode.new_document(ctx, (4, 4, 16, 16))
    state.activate(other.uid)
    assert state.sheet_import is None
    assert state.sheet_import_open is False


def test_importing_with_nothing_parked_is_a_no_op():
    ctx = FakeCtx()
    _tab(ctx, tileset=False)
    assert plotter_mode.import_detected_sheet(ctx) is False
    assert plotter_mode.import_sheet_blind(ctx) is False


def test_use_as_tileset_routes_through_detection(tmp_path, monkeypatch):
    from warlock.studio import plotter_tilesets

    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    png = _save_png(tmp_path / "input.png", _ruled_sheet())
    monkeypatch.setattr(
        "warlock.service.files.job_dir_file", lambda svc, job_id, name: png
    )
    assert plotter_tilesets is not None

    plotter_mode.use_as_tileset(ctx, {"id": "j1", "name": "sheet"})
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    state = plotter_mode.ensure(ctx)
    assert state.sheet_import is not None
    assert state.sheet_import[1] == "sheet"


def test_a_tsx_never_parks(tmp_path):
    from PIL import Image

    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    Image.fromarray(_ruled_sheet(), "RGBA").save(tmp_path / "t.png")
    sheet = _ruled_sheet()
    tsx_path = tmp_path / "t.tsx"
    tsx_path.write_text(
        f'<tileset name="t" tilewidth="16" tileheight="16">'
        f'<image source="t.png" width="{sheet.shape[1]}" height="{sheet.shape[0]}"/>'
        f"</tileset>",
        encoding="utf-8",
    )

    plotter_mode.add_tileset_path(ctx, tsx_path)
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    assert plotter_mode.ensure(ctx).sheet_import is None
    assert len(tab.doc.tilesets) == 1


# --- tile-size mismatch at the library door (Part E2) --------------------------


def _library(monkeypatch, tmp_path, pixels: np.ndarray, sidecar: dict | None):
    """A fake job directory holding an input.png and maybe a sheet.json."""
    import json

    _save_png(tmp_path / "input.png", pixels)
    if sidecar is not None:
        (tmp_path / "sheet.json").write_text(json.dumps(sidecar), encoding="utf-8")

    def job_dir_file(svc, job_id, name):
        return tmp_path / name

    monkeypatch.setattr("warlock.service.files.job_dir_file", job_dir_file)


def _flat(size: int = 64) -> np.ndarray:
    out = np.full((size, size, 4), 200, dtype=np.uint8)
    out[:, :, 3] = 255
    # A little variation so the image is not uniformly one colour.
    out[::2, ::2, :3] = 210
    return out


def test_a_sheet_generated_at_another_size_parks(tmp_path, monkeypatch):
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    _library(monkeypatch, tmp_path, _flat(), {"tile_w": 32, "tile_h": 32})

    plotter_mode.use_as_tileset(ctx, {"id": "j1", "name": "sheet"})
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    state = plotter_mode.ensure(ctx)
    assert state.sheet_import is not None
    assert state.sheet_import[4] == plotter_mode.SheetMismatch(32, 32)
    assert len(tab.doc.tilesets) == 0


def test_a_sheet_generated_at_the_map_size_parks_nothing(tmp_path, monkeypatch):
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    _library(monkeypatch, tmp_path, _flat(), {"tile_w": 16, "tile_h": 16})

    plotter_mode.use_as_tileset(ctx, {"id": "j1", "name": "sheet"})
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    assert plotter_mode.ensure(ctx).sheet_import is None
    assert len(tab.doc.tilesets) == 1


def test_a_job_without_a_sidecar_never_parks(tmp_path, monkeypatch):
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    _library(monkeypatch, tmp_path, _flat(), None)

    plotter_mode.use_as_tileset(ctx, {"id": "j1", "name": "sheet"})
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    assert plotter_mode.ensure(ctx).sheet_import is None
    assert len(tab.doc.tilesets) == 1


def test_importing_a_mismatched_sheet_recomposes_onto_the_map_grid(tmp_path, monkeypatch):
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    # 64 px sheet, four 32 px cells, each a flat distinguishable colour.
    sheet = np.zeros((64, 64, 4), dtype=np.uint8)
    sheet[:, :, 3] = 255
    for r in range(2):
        for c in range(2):
            sheet[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32, :3] = 100 + r * 40 + c * 20
    _library(monkeypatch, tmp_path, sheet, {"tile_w": 32, "tile_h": 32})

    plotter_mode.use_as_tileset(ctx, {"id": "j1", "name": "sheet"})
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))
    assert plotter_mode.import_detected_sheet(ctx) is True

    ref = tab.doc.tilesets[0]
    assert ref.tileset.tile_w == 16
    # Four cells, each redrawn at 16 -- a 32x32 atlas, not the blind 4x4 cut.
    assert ref.tileset.tile_count == 4
    assert np.asarray(ref.tileset.pixels).shape == (32, 32, 4)
    assert ref.source == ""


def test_the_alternate_button_matches_todays_blind_slice(tmp_path, monkeypatch):
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    sheet = _flat()
    _library(monkeypatch, tmp_path, sheet, {"tile_w": 32, "tile_h": 32})

    plotter_mode.use_as_tileset(ctx, {"id": "j1", "name": "sheet"})
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))
    assert plotter_mode.import_sheet_blind(ctx) is True

    ref = tab.doc.tilesets[0]
    assert np.array_equal(np.asarray(ref.tileset.pixels), sheet)
    assert ref.tileset.tile_w == 16


# --- terrain-set inference (Part F) --------------------------------------------


def _blob_sheet(tile: int = 16, cols: int = 47) -> np.ndarray:
    """A 47-case blob sheet, drawn from ``blob`` itself.

    Derived rather than typed for ``tests/tilegrid/test_roles.py``'s reason: the
    fixture and the analyzer then agree about what a role looks like by
    construction.
    """
    from warlock.studio.tilegrid import blob

    depth = max(1, tile // 8) + 1
    rows = -(-blob.TILE_COUNT // cols)
    out = np.zeros((rows * tile, cols * tile, 4), dtype=np.uint8)
    for case in range(blob.TILE_COUNT):
        y, x = (case // cols) * tile, (case % cols) * tile
        cell = out[y:y + tile, x:x + tile]
        cell[:, :] = (60, 140, 70, 255)
        north, east, south, west = blob.open_edges(case)
        if north:
            cell[:depth, :] = 0
        if south:
            cell[tile - depth:, :] = 0
        if west:
            cell[:, :depth] = 0
        if east:
            cell[:, tile - depth:] = 0
        ne, se, sw, nw = blob.open_corners(case)
        if ne:
            cell[:depth, tile - depth:] = 0
        if se:
            cell[tile - depth:, tile - depth:] = 0
        if sw:
            cell[tile - depth:, :depth] = 0
        if nw:
            cell[:depth, :depth] = 0
    return out


def test_a_blob_sheet_parks_as_a_terrain_set(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    png = _save_png(tmp_path / "blob.png", _blob_sheet())

    plotter_mode.add_tileset_path(ctx, png)
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    state = plotter_mode.ensure(ctx)
    assert state.sheet_import is not None
    parked = state.sheet_import[4]
    assert isinstance(parked, plotter_mode.SheetTerrain)
    assert parked.tile == (16, 16)
    assert parked.roles.tile_count == 47
    assert len(tab.doc.tilesets) == 0


def test_importing_a_terrain_set_reorders_and_enables_painting(tmp_path):
    from warlock.studio.plotter import terrain as terrainlib

    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    png = _save_png(tmp_path / "blob.png", _blob_sheet())
    plotter_mode.add_tileset_path(ctx, png)
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    assert plotter_mode.import_sheet_terrain(ctx) is True

    ref = tab.doc.tilesets[0]
    assert len(ref.tileset.terrains) == 1
    # The reorder *is* the role assignment: 47 columns, one row.
    assert ref.tileset.columns == 47
    assert ref.tileset.rows == 1
    # A reordered atlas no longer matches any file.
    assert ref.source == ""
    # And the palette has it in hand, which is what the terrain hand-off does.
    state = plotter_mode.ensure(ctx)
    assert state.terrain == (0, 0)

    layer = tab.doc.tile_layers()[0]
    region = terrainlib.paint_terrain(layer.data, 1, 1, 0, ref)
    assert region is not None, "the imported set can actually be painted with"
    x0, y0, block = region
    tab.doc.write_region(layer.uid, x0, y0, block)
    assert int(tab.doc.tile_layers()[0].data[1, 1]) != 0


def test_the_plain_button_slices_the_terrain_sheet_as_ordinary_tiles(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    sheet = _blob_sheet()
    png = _save_png(tmp_path / "blob.png", sheet)
    plotter_mode.add_tileset_path(ctx, png)
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    assert plotter_mode.import_sheet_blind(ctx) is True

    ref = tab.doc.tilesets[0]
    assert ref.tileset.terrains == ()
    assert np.array_equal(np.asarray(ref.tileset.pixels), sheet)
    assert ref.source == str(png)


# --- a *generated* terrain set, landed by its record --------------------------
#
# The inference above cannot see one of these and no threshold will make it:
# ``roles._background_masks`` wants transparency or one dominant ring colour, and
# a generated terrain set is two opaque textures composited edge to edge. The
# sidecar is the only thing that can land it, which is why these tests build the
# record with the very function the worker writes it with.


def _material(seed: int, tile: int = 16) -> np.ndarray:
    """One opaque, noisy, seamless-ish material -- what SDXL returns, reduced.

    Noisy on purpose: a flat colour would give the inference a ring colour to
    find, which is exactly the property a real material does not have.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((tile, tile, 4), dtype=np.uint8)
    out[:, :, :3] = rng.integers(40, 210, size=(tile, tile, 3), dtype=np.uint8)
    out[:, :, 3] = 255
    return out


def _generated_terrain_atlas(tile: int = 16) -> np.ndarray:
    from warlock.pipelines import tilemask

    return tilemask.blob_atlas(_material(1, tile), _material(2, tile), tile, seed=7)


def _generated_terrain_sidecar(tile: int = 16) -> dict:
    """The sheet.json a terrain job writes, from the writer itself.

    Built through ``tileatlas.atlas_sidecar`` rather than typed out, so this
    fixture cannot describe a record the worker does not produce -- the failure
    that makes a form test pass over a submit nobody can make.
    """
    import dataclasses
    import time

    from warlock.pipelines import tileatlas

    geom = tileatlas.terrain_geometry(tile, "top_down")
    cells = tuple(
        dataclasses.replace(cell, prompt="wet grass", variant=1, seed=7)
        for cell in geom.cells
    )
    return tileatlas.atlas_sidecar(
        geom,
        created=time.time(),
        materials=cells,
        terrains=(
            {
                "name": "wet grass",
                "fill": [106, 153, 78, 255],
                "outline": [63, 91, 46, 255],
            },
        ),
        mask={"seed": 7},
    )


def test_a_generated_terrain_set_lands_by_its_record(tmp_path, monkeypatch):
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    atlas = _generated_terrain_atlas()
    _library(monkeypatch, tmp_path, atlas, _generated_terrain_sidecar())

    plotter_mode.use_as_tileset(ctx, {"id": "j1", "name": "coast"})
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    # No popup: there is nothing to ask about a set that says what it is.
    state = plotter_mode.ensure(ctx)
    assert state.sheet_import is None
    ref = tab.doc.tilesets[0]
    assert [t.name for t in ref.tileset.terrains] == ["wet grass"]
    assert ref.tileset.columns == 47
    assert ref.tileset.rows == 1
    assert np.array_equal(np.asarray(ref.tileset.pixels), atlas)
    # And the palette has it in hand, which is the terrain hand-off.
    assert state.terrain == (0, 0)


def test_the_same_set_without_its_record_cannot_be_recognised(tmp_path, monkeypatch):
    """The documented reason the record exists, pinned as a measurement.

    ``infer_roles`` finds a background from transparency or from one dominant
    ring colour; two opaque textures composited edge to edge have neither, so
    every cell's silhouette reads as noise, fewer than the forty-seven roles are
    claimed and the inference returns ``None`` on a set that is perfectly formed.
    If this ever starts returning a set, the record has stopped being the *only*
    thing that can land one -- and this test is where to say so.
    """
    from warlock.studio.tilegrid import roles as rolelib

    atlas = _generated_terrain_atlas()
    assert rolelib.infer_roles(atlas, 16, 16) is None

    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    _library(monkeypatch, tmp_path, atlas, {"tile_w": 16, "tile_h": 16})

    plotter_mode.use_as_tileset(ctx, {"id": "j1", "name": "coast"})
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    assert plotter_mode.ensure(ctx).sheet_import is None
    assert tab.doc.tilesets[0].tileset.terrains == ()


def test_a_generated_terrain_set_at_another_size_is_resampled_not_parked(
    tmp_path, monkeypatch
):
    """The mismatch arm's shape on a grid that is known rather than detected:
    cut on the record's forty-seven columns and redrawn at the map's size. The
    popup is not the answer here -- a terrain set sliced at the wrong size is
    not a set at all, and the record says exactly where its cells are."""
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    _library(
        monkeypatch, tmp_path, _generated_terrain_atlas(32), _generated_terrain_sidecar(32)
    )

    plotter_mode.use_as_tileset(ctx, {"id": "j1", "name": "coast"})
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    state = plotter_mode.ensure(ctx)
    assert state.sheet_import is None
    ref = tab.doc.tilesets[0]
    assert (ref.tileset.tile_w, ref.tileset.tile_h) == (16, 16)
    assert ref.tileset.columns == 47
    assert len(ref.tileset.terrains) == 1
    assert np.asarray(ref.tileset.pixels).shape == (16, 47 * 16, 4)
    # A resampled atlas is no longer the file on disk.
    assert ref.source == ""


def test_a_record_the_image_does_not_fit_falls_back_to_the_ordinary_path(
    tmp_path, monkeypatch
):
    """A sidecar from one job beside another job's image. The record is ignored
    rather than refused: the sheet still lands, sliced as it would be with no
    sidecar at all."""
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    sidecar = _generated_terrain_sidecar()
    _library(monkeypatch, tmp_path, _flat(), sidecar)

    plotter_mode.use_as_tileset(ctx, {"id": "j1", "name": "coast"})
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    assert plotter_mode.ensure(ctx).sheet_import is None
    assert tab.doc.tilesets[0].tileset.terrains == ()


def test_a_terrain_record_is_all_or_nothing(tmp_path, monkeypatch):
    """A terrain's position in the list is its precedence, so a list with one
    entry quietly dropped is not a smaller set -- it is a different one."""
    from warlock.studio import plotter_tilesets

    sidecar = _generated_terrain_sidecar()
    sidecar["terrains"] = [sidecar["terrains"][0], {"name": "broken"}]
    assert plotter_tilesets._recorded_terrains(sidecar["terrains"]) == ()

    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    _library(monkeypatch, tmp_path, _generated_terrain_atlas(), sidecar)

    plotter_mode.use_as_tileset(ctx, {"id": "j1", "name": "coast"})
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    assert tab.doc.tilesets[0].tileset.terrains == ()


def test_the_seamless_sidecars_own_spelling_of_the_view_is_read(tmp_path, monkeypatch):
    """The grid path writes ``projection`` and the seamless path writes ``view``.
    Reading only the first left every generated tileset with no lattice at all,
    which is the silent half of "the sidecar records it and nothing reads it"."""
    from warlock.studio import plotter_tilesets
    from warlock.studio.plotter import project

    _library(monkeypatch, tmp_path, _flat(), _generated_terrain_sidecar())
    record = plotter_tilesets._recorded_sheet(FakeCtx(), "j1")
    assert record is not None
    assert record.lattice == project.ORTHOGONAL
    assert record.layout == plotter_tilesets.BLOB_LAYOUT


def test_a_sidecar_from_before_the_planner_was_deleted_still_opens(
    tmp_path, monkeypatch
):
    """Grid sheets drawn before 2026-08-29 carry a ``workflow`` block: the
    compiled tileset plan ``_q_tilesheet`` used to write beside a picture that
    had none of that structure. The writer is gone; the files are not, and the
    rule for removing a sidecar key is that a sheet already on disk keeps
    opening exactly as it did. This reader takes the keys it knows and ignores
    the rest, so it does -- and the block is still in the file afterwards,
    because nothing here rewrites it.
    """
    import json

    from warlock.pipelines import tilesheet
    from warlock.studio import plotter_tilesets
    from warlock.studio.plotter import project

    sidecar = tilesheet.sheet_sidecar(
        prompt="a damp dungeon",
        tile_w=16,
        tile_h=16,
        view="top_down",
        colors=8,
        palette=["#000000"],
        recipe={"base_model": "sdxl_cfg"},
        created=1.0,
    )
    # As the deleted writer wrote it: the block sat beside the geometry rather
    # than inside it, which is why removing it leaves every other key in place.
    sidecar["workflow"] = {
        "version": 2,
        "mode": "collection",
        "resolved_mode": "materials",
        "cells": [{"index": 0, "prompt": "moss", "variant": 1, "seed": 11}],
        "roles": [],
    }
    _library(monkeypatch, tmp_path, _flat(), sidecar)

    record = plotter_tilesets._recorded_sheet(FakeCtx(), "j1")
    assert record is not None
    assert (record.tile_w, record.tile_h) == (16, 16)
    assert record.lattice == project.ORTHOGONAL
    stored = json.loads((tmp_path / "sheet.json").read_text(encoding="utf-8"))
    assert stored["workflow"] == sidecar["workflow"]


def test_a_sheet_with_too_few_cells_never_parks_as_terrain(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    small = _blob_sheet()[:, : 20 * 16]
    png = _save_png(tmp_path / "few.png", small)

    plotter_mode.add_tileset_path(ctx, png)
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    assert plotter_mode.ensure(ctx).sheet_import is None
    assert len(tab.doc.tilesets) == 1
    assert tab.doc.tilesets[0].tileset.terrains == ()


# --- painting an infinite map -------------------------------------------------


def _infinite(ctx: FakeCtx):
    """A stamping tab whose map has no edge."""
    tab = plotter_mode.new_document(ctx, (4, 4, 16, 16), infinite=True)
    tab.doc.add_tileset(_tileset())
    tab.doc.mark_saved()
    state = plotter_mode.ensure(ctx)
    state.tool = "stamp"
    state.brush = np.array([[tab.doc.tilesets[0].firstgid]], gid.DTYPE)
    return tab, state


def test_painting_past_the_edge_of_an_infinite_map_grows_it():
    """The gesture the whole feature is for. The cell keeps its *true*
    coordinate; it is the window that moved under it."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state = _infinite(ctx)
    plotter_canvas._apply(ctx, state, tab, (-2, -1))
    assert (tab.doc.origin_x, tab.doc.origin_y) == (-2, -1)
    layer = tab.doc.tile_layers()[0]
    assert int(layer.data[0, 0]) != 0
    assert not ctx.toasts


def test_a_stamp_at_the_edge_makes_room_for_its_whole_footprint():
    """The brush's footprint, not the cell under the cursor: growing by one
    cell would clip the rest of a 3x3 stamp."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state = _infinite(ctx)
    state.brush = np.full((3, 3), tab.doc.tilesets[0].firstgid, gid.DTYPE)
    plotter_canvas._apply(ctx, state, tab, (3, 3))
    assert (tab.doc.width, tab.doc.height) == (6, 6)
    assert int((tab.doc.tile_layers()[0].data != 0).sum()) == 9


def test_painting_past_a_finite_map_still_clips():
    """The flag is the only thing that decides it, and a finite map is exactly
    as it was."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, _layer = _stamping(ctx)
    plotter_canvas._apply(ctx, state, tab, (-2, -1))
    assert (tab.doc.width, tab.doc.height) == (4, 4)
    assert int((tab.doc.tile_layers()[0].data != 0).sum()) == 0


def test_a_refused_growth_says_so_instead_of_stopping_silently():
    """A refusal reaching the user. A stroke that quietly stopped at an
    invisible edge is the failure this toast exists to prevent.

    A click this far out is now caught by the *per-press* cap rather than the
    extent cap -- reaching cell 100,000 would mean reallocating every tile
    layer to the full 4096 a side between two frames -- and the assertion here
    is about the refusal arriving, not about which of the two it was. The
    extent cap has its own case below."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state = _infinite(ctx)
    plotter_canvas._apply(ctx, state, tab, (100_000, 0))
    assert len(ctx.toasts) == 1
    assert "past the edge" in ctx.toasts[0][0]
    assert (tab.doc.width, tab.doc.height) == (4, 4)
    assert int((tab.doc.tile_layers()[0].data != 0).sum()) == 0


def test_the_extent_cap_also_reaches_the_user():
    """The other refusal out of the same door. A map already near the cap
    growing a modest amount is what the per-press cap lets through and the
    extent cap then stops -- so both have to say something, and they come out
    of the one ``except ValueError`` in ``_room_for``."""
    from warlock.studio.panes import plotter_canvas
    from warlock.studio.plotter import tilemap

    ctx = FakeCtx()
    tab, state = _infinite(ctx)
    tab.doc.resize(tilemap.MAX_DIMENSION - 100, 4)
    plotter_canvas._apply(ctx, state, tab, (tilemap.MAX_DIMENSION + 10, 0))
    assert len(ctx.toasts) == 1
    assert "extent" in ctx.toasts[0][0]


def test_a_growth_carries_the_selection_and_the_line_anchor_with_it():
    """Every cell-space view field is an *index* into the window, and a growth
    slides the window. Left alone they all point one growth to the left of
    where the user put them."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state = _infinite(ctx)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    state.set_selection((1, 1, 2, 2), mask)
    state.last_paint = (1, 1)
    plotter_canvas._apply(ctx, state, tab, (-2, -1))
    assert state.select == (3, 2, 4, 3)
    assert state.last_paint == (3, 2)
    assert bool(state.select_mask[2, 3])
    assert state.select_mask.shape == (tab.doc.height, tab.doc.width)


def test_a_drag_that_grows_the_map_interpolates_in_the_new_window():
    """The line is drawn from ``drag_last_cell``, which the growth moved. Drawn
    before the growth it would start one growth to the left of the cell the
    pointer was actually in."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state = _infinite(ctx)
    tab.doc.begin_stroke(tab.doc.tile_layers()[0].uid)
    plotter_canvas._apply_drag(ctx, state, tab, (1, 0))
    plotter_canvas._apply_drag(ctx, state, tab, (-2, 0))  # off the left edge
    tab.doc.end_stroke()
    data = tab.doc.tile_layers()[0].data
    # Four cells in one unbroken run: true coordinates -2..1 on row 0.
    assert [int(value != 0) for value in data[0, :4].tolist()] == [1, 1, 1, 1]


def test_a_shift_click_line_survives_the_growth_it_causes():
    """``last_paint`` is the anchor and the growth moves it, so the line lands
    between the two cells the user actually clicked."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state = _infinite(ctx)
    # Through the drag path, because that is what sets ``last_paint`` -- the
    # anchor a Shift+click draws from, and what ``_line_pending`` requires
    # before the pane ever calls ``_apply_line``.
    plotter_canvas._apply_drag(ctx, state, tab, (1, 0))
    plotter_canvas._apply_line(ctx, state, tab, state.last_paint, (-2, 0))
    data = tab.doc.tile_layers()[0].data
    assert [int(value != 0) for value in data[0, :4].tolist()] == [1, 1, 1, 1]


# --- crash recovery -------------------------------------------------------------


def test_a_recovered_map_reads_dirty_and_close_asks(tmp_path):
    """``read_wmap`` hands back a document already marked saved, and
    ``PlotterDoc.dirty`` delegates to the document -- so an adopt that only
    wrote ``tab.saved_head`` produced a *clean* recovered tab: one unprompted
    close skipped the confirm and ``drop()`` deleted the journal copy, the only
    surviving copy of the work."""
    from warlock.studio.plotter.tilemap import MapDoc

    doc = MapDoc(4, 4, 16, 16)
    doc.add_tile_layer("Ground")
    path = tmp_path / "level.wmap"
    path.write_bytes(wmap.wmap_bytes(doc))

    ctx = FakeCtx()
    assert plotter_mode._journal_adopt(ctx, path, {"title": "level"}) is True
    tab = ctx.state.plotter.docs[-1]
    assert tab.dirty is True, "recovered work is unsaved by definition"

    plotter_mode.close_tab(ctx, tab.uid)
    assert ctx.confirms.pending is not None, "closing recovered work must ask"
    assert ctx.state.plotter.get(tab.uid) is not None, "still open until answered"


def test_an_aux_file_named_like_the_map_is_not_routed_onto_it(tmp_path):
    """``startswith("map.")`` also matched an image layer whose source happened
    to be ``map.png``, which then overwrote the map at the user's chosen path
    with the picture -- or the other way round, depending on dict order."""
    plotter_io._write(
        {"map.tmx": b"the map", "map.png": b"the picture"}, tmp_path / "level.tmx"
    )
    assert (tmp_path / "level.tmx").read_bytes() == b"the map"
    assert (tmp_path / "map.png").read_bytes() == b"the picture"


def test_paste_and_duplicate_are_refused_while_the_tab_is_busy():
    """Ctrl+V writes the document since ``ObjectClip`` (an object paste is an
    ``add_object``) and Ctrl+J adds a duplicate, so both sit behind the busy
    gate with their mutating peers."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True
    state = plotter_mode.ensure(ctx)
    state.clipboard = np.array([[1]], gid.DTYPE)
    state.clipboard_doc = tab.uid

    assert plotter_mode.handle_key(ctx, _key("v", ctrl=True)) is True
    assert state.brush is None, "a busy tab takes no paste"
    assert plotter_mode.handle_key(ctx, _key("j", ctrl=True)) is True
    assert not tab.dirty


def test_every_plotter_tool_has_its_own_icon():
    """Wand had no entry, fell through to the ``icons.SQUARE`` default, and so
    drew Shape's glyph two buttons along -- while ``icons.WAND`` sat on
    Terrain. A missing entry here is a *wrong* picture, not a blank one, so the
    table's completeness and its distinctness are both pinned."""
    from warlock.studio import plotter_state
    from warlock.studio.panes import plotter_tools

    keys = [key for key, _label, _letter in plotter_state.TOOLS]
    assert set(plotter_tools._ICONS) == set(keys)
    glyphs = [plotter_tools._ICONS[key] for key in keys]
    assert len(set(glyphs)) == len(glyphs), "two tools cannot share one glyph"


def test_use_as_tileset_with_no_map_starts_one_rather_than_refusing():
    """The Library offers this for any asset carrying an ``input.png`` and
    cannot know whether a map is open, so the old error toast was an offer the
    app took straight back -- and six navigation steps for the user to put
    right. It asks rather than invents, because a map's tile size is the number
    this very action slices on."""
    ctx = FakeCtx()
    job = {"id": "j1", "name": "sheet"}

    plotter_mode.use_as_tileset(ctx, job)

    state = plotter_mode.ensure(ctx)
    assert not ctx.toasts, "starting the map is the answer, not an error"
    assert state.setup_pending is True
    assert state.pending_job_tileset == job
    assert ctx.state.mode == "plotter"


def test_the_new_map_dialog_hands_the_waiting_asset_to_the_map_it_makes(
    tmp_path, monkeypatch
):
    """``plotter_canvas._create`` is the far side of the dialog: it calls back
    into ``use_as_tileset`` with a tab to work on, and the pending asset wins
    over the form's own "Then" choice."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    png = _save_png(tmp_path / "input.png", _ruled_sheet())
    monkeypatch.setattr("warlock.service.files.job_dir_file", lambda svc, job_id, name: png)
    asked: list[int] = []
    monkeypatch.setattr(plotter_mode, "ask_add_tileset", lambda ctx: asked.append(1))

    plotter_mode.use_as_tileset(ctx, {"id": "j1", "name": "sheet"})
    form = {"projection": "orthogonal", "width": 8, "height": 8, "tile_w": 16, "tile_h": 16}
    form["next"] = "file"
    plotter_canvas._create(ctx, form)

    state = plotter_mode.ensure(ctx)
    assert state.docs, "the dialog made the map"
    assert state.pending_job_tileset is None, "the one-shot is cleared"
    assert not asked, "the picker would be a second tileset nobody asked for"
    tab = plotter_mode.active(ctx)
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))
    assert state.sheet_import is not None


def test_an_isometric_sheet_on_a_painted_square_map_parks(tmp_path, monkeypatch):
    """The sidecar has recorded the view since tile sheets existed and nothing
    ever read it back, so a diamond sheet was sliced onto a square grid with
    nothing anywhere saying so. A 32x32 isometric sheet and a 32x32 top-down one
    measure identically, which is why the cell size cannot catch this."""
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False, dirty=True)
    _library(
        monkeypatch, tmp_path, _flat(), {"tile_w": 16, "tile_h": 16, "projection": "isometric"}
    )

    plotter_mode.use_as_tileset(ctx, {"id": "j1", "name": "sheet"})
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    state = plotter_mode.ensure(ctx)
    assert state.sheet_import is not None
    assert state.sheet_import[4] == plotter_mode.SheetLattice("isometric", "orthogonal")
    assert len(tab.doc.tilesets) == 0


def test_an_isometric_sheet_on_an_unpainted_map_brings_its_lattice(tmp_path, monkeypatch):
    """An empty map has no lattice worth defending, and ``land_tileset`` has
    carried the arm for exactly this since the ground generator was deleted."""
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False)
    _library(
        monkeypatch, tmp_path, _flat(), {"tile_w": 16, "tile_h": 16, "projection": "isometric"}
    )

    plotter_mode.use_as_tileset(ctx, {"id": "j1", "name": "sheet"})
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    assert plotter_mode.ensure(ctx).sheet_import is None
    assert tab.doc.projection == "isometric"
    assert len(tab.doc.tilesets) == 1


def test_a_three_quarter_sheet_is_not_a_lattice_mismatch(tmp_path, monkeypatch):
    """3/4 is square and sits on the orthogonal grid exactly as top-down does --
    the tilt is in the art. Reporting it as a mismatch would be a popup in front
    of every 3/4 import there will ever be."""
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False, dirty=True)
    _library(
        monkeypatch,
        tmp_path,
        _flat(),
        {"tile_w": 16, "tile_h": 16, "projection": "three_quarter"},
    )

    plotter_mode.use_as_tileset(ctx, {"id": "j1", "name": "sheet"})
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    assert plotter_mode.ensure(ctx).sheet_import is None
    assert tab.doc.projection == "orthogonal"
    assert len(tab.doc.tilesets) == 1


def test_the_old_orthogonal_spelling_in_a_sidecar_is_not_a_mismatch(tmp_path, monkeypatch):
    """Sidecars written before the vocabulary widened carry it, and every one of
    them is on the square lattice."""
    ctx = FakeCtx()
    tab = _tab(ctx, tileset=False, dirty=True)
    _library(
        monkeypatch, tmp_path, _flat(), {"tile_w": 16, "tile_h": 16, "projection": "orthogonal"}
    )

    plotter_mode.use_as_tileset(ctx, {"id": "j1", "name": "sheet"})
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))

    assert plotter_mode.ensure(ctx).sheet_import is None
    assert len(tab.doc.tilesets) == 1


# --- a foreign Wang set as a terrain the tool can lay down ---------------------
#
# The Terrain picker offers a foreign Wang set's colours beside the blob preset,
# encoded as a *negative* rank. Everything below is about that encoding
# surviving the trip from the picker to the painter: it used to reach
# ``Tileset.local_for`` as a blob rank and raise ``IndexError: terrain -1 is
# outside this set (0..-1)``, and the emptier the terrain list the more certainly
# it got there.


def _wang_tileset(name: str = "wang") -> Tileset:
    """Sixteen tiles over the complete two-colour corner set, and *no* blob
    terrains -- which is what a Tiled ``.tsx`` carrying a genuine Wang set is."""
    from warlock.studio.tilegrid.wang import WangColour, WangSet

    tiles = {
        index: (
            0,
            1 + ((index >> 0) & 1),
            0,
            1 + ((index >> 1) & 1),
            0,
            1 + ((index >> 2) & 1),
            0,
            1 + ((index >> 3) & 1),
        )
        for index in range(16)
    }
    wangset = WangSet(
        name="Corners",
        kind="corner",
        colours=(WangColour("grass", "#00ff00"), WangColour("sand", "#ffff00")),
        tiles=tiles,
    )
    pixels = np.zeros((64, 64, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return Tileset(name=name, pixels=pixels, tile_w=16, tile_h=16, wangsets=(wangset,))


def _wang_tab(ctx: FakeCtx):
    """A tab whose only tileset carries a Wang set and declares no blob terrains."""
    tab = _tab(ctx, tileset=False)
    ref = tab.doc.add_tileset(_wang_tileset())
    tab.doc.mark_saved()
    return tab, plotter_mode.ensure(ctx), ref


def test_the_terrain_tool_paints_a_wang_colour():
    """The bug, through the path it actually took: the picker's encoding reached
    the blob painter and raised out of the frame loop."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, ref = _wang_tab(ctx)
    layer = tab.doc.tile_layers()[0]
    state.tool = "terrain"
    state.terrain = (0, -1)  # the picker's encoding of Wang colour 1

    plotter_canvas._apply(ctx, state, tab, (2, 2))

    assert ctx.toasts == []
    value = int(layer.data[2, 2]) & gid.GID_MASK
    assert value, "the cell was painted"
    wangset = ref.tileset.wangsets[0]
    assert wangset.wangid_of(value - ref.firstgid) == (0, 1, 0, 1, 0, 1, 0, 1)


def test_the_second_wang_colour_is_the_second_negative_rank():
    """``-1 - colour_index``: the decode has to be the picker's encode, or the
    tool lays down a colour the user did not click."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, ref = _wang_tab(ctx)
    layer = tab.doc.tile_layers()[0]
    state.tool = "terrain"
    state.terrain = (0, -2)

    plotter_canvas._apply(ctx, state, tab, (2, 2))

    value = int(layer.data[2, 2]) & gid.GID_MASK
    assert ref.tileset.wangsets[0].wangid_of(value - ref.firstgid) == (
        0, 2, 0, 2, 0, 2, 0, 2
    )


def test_a_wang_colour_that_does_not_exist_is_refused_rather_than_painted():
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, _ref = _wang_tab(ctx)
    layer = tab.doc.tile_layers()[0]
    state.tool = "terrain"
    state.terrain = (0, -9)

    plotter_canvas._apply(ctx, state, tab, (2, 2))

    assert int(layer.data[2, 2]) == 0
    assert ctx.toasts and ctx.toasts[-1][1] == "error"


def test_filling_with_a_wang_colour_in_hand_floods_instead_of_raising():
    """Fill's terrain branch reads the same state the terrain tool does, so it
    had the same crash and needs the same dispatch."""
    from warlock.studio.panes import plotter_canvas

    ctx = FakeCtx()
    tab, state, ref = _wang_tab(ctx)
    layer = tab.doc.tile_layers()[0]
    state.tool = "fill"
    state.brush = None
    state.terrain = (0, -1)

    plotter_canvas._apply(ctx, state, tab, (1, 1))

    assert ctx.toasts == []
    assert int((layer.data != 0).sum()) == layer.data.size, "the empty run flooded"
    wangset = ref.tileset.wangsets[0]
    value = int(layer.data[1, 1]) & gid.GID_MASK
    assert wangset.wangid_of(value - ref.firstgid) == (0, 1, 0, 1, 0, 1, 0, 1)


def test_the_picker_arms_a_terrain_the_canvas_can_paint_with():
    """No click is needed to reach the bug: the section auto-selects its first
    entry, and on a Wang-only map that entry is a negative rank."""
    from warlock.studio.panes import plotter_canvas, plotter_tools

    ctx = FakeCtx()
    tab, state, _ref = _wang_tab(ctx)
    layer = tab.doc.tile_layers()[0]
    entries = plotter_tools.terrains_of(tab.doc)
    assert entries and entries[0][1] < 0, "the only rows this map has are Wang colours"

    state.terrain = plotter_tools.first_terrain(entries)
    state.tool = "terrain"
    plotter_canvas._apply(ctx, state, tab, (2, 2))

    assert ctx.toasts == []
    assert int(layer.data[2, 2]) != 0


# --- W4: the six small gaps ---------------------------------------------------
#
# Every one of these is a control that had no entry point. The tests below drive
# the entry point rather than the thing behind it: what was missing was never the
# capability, and a test that called the capability would have passed before the
# gap was closed.


class _Key:
    """A pygame KEYDOWN with no modifiers."""

    def __init__(self, key: int) -> None:
        import pygame

        self.key = key
        self.mod = 0
        self.type = pygame.KEYDOWN


def test_the_capsule_letter_reaches_the_canvas_through_the_key_handler():
    """The keyboard is one of the two doors onto a tool, and the one a headless
    test can drive end to end. ``C`` was not bound to anything at all."""
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    layer = tab.doc.add_object_layer()
    tab.doc.set_active_layer(layer.uid)

    assert plotter_mode.handle_key(ctx, _Key(pygame.K_c)) is True
    assert state.tool == "object_capsule"
    # What the canvas reads when the drag is released.
    assert state.object_shape == "capsule"


def test_go_to_coordinate_asks_the_canvas_to_centre_on_a_cell():
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    assert state.goto_cell is None

    assert plotter_mode.go_to_cell(ctx, tab, 2, 3) == (2, 3)
    assert state.goto_cell == (2, 3)


def test_go_to_coordinate_clamps_rather_than_refusing():
    """A number remembered from a bigger map is a navigation, not an error."""
    ctx = FakeCtx()
    tab = _tab(ctx)  # 4 x 4

    assert plotter_mode.go_to_cell(ctx, tab, 99, -7) == (3, 0)
    assert plotter_mode.ensure(ctx).goto_cell == (3, 0)
    assert ctx.toasts == []


def test_a_pending_jump_is_dropped_when_the_tab_changes():
    """It names a cell in the map being left, ``last_paint``'s own rule."""
    ctx = FakeCtx()
    first = _tab(ctx)
    plotter_mode.go_to_cell(ctx, first, 1, 1)
    _tab(ctx)
    assert plotter_mode.ensure(ctx).goto_cell is None
    plotter_mode.go_to_cell(ctx, plotter_mode.ensure(ctx).active, 2, 2)
    plotter_mode.ensure(ctx).activate(first.uid)
    assert plotter_mode.ensure(ctx).goto_cell is None


def test_a_history_jump_drops_the_object_selection():
    """A jump can undo an object out of existence, so the Properties pane must
    not be left pointing at a uid no layer holds -- ``undo``'s rule."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    layer = tab.doc.add_object_layer()
    obj = tab.doc.add_object(
        layer.uid, MapObject(uid=new_uid(), name="spawn", kind="point", x=1, y=1)
    )
    state.select_object(obj.uid)
    depth = len(tab.doc.history)

    assert plotter_mode.step_history(ctx, tab, depth - 1) is True
    assert state.selected_object is None
    assert tab.doc.layers[-1].objects == []


def test_reloading_a_tileset_image_replaces_the_art_in_place():
    """Tileset > Reload the image..., the door that did not exist.

    Re-importing the file made a *second* tileset with its own firstgid, which
    every gid already painted on the map ignored. This swaps the art under the
    ids instead.
    """
    ctx = FakeCtx()
    tab = _tab(ctx)
    before = tab.doc.tilesets[0]
    painted = tab.doc.tile_layers()[0]
    tab.doc.write_region(
        painted.uid, 0, 0, np.array([[before.firstgid + 1]], gid.DTYPE)
    )

    fresh = np.zeros((32, 32, 4), dtype=np.uint8)
    fresh[..., 3] = 255
    fresh[5, 5] = (1, 2, 3, 255)
    plotter_mode.on_task_done(
        ctx,
        _Done(
            f"plotter-tileset-image:{tab.uid}",
            {"replace": (0, "atlas.png", fresh), "uid": tab.uid},
        ),
    )

    after = tab.doc.tilesets[0]
    assert len(tab.doc.tilesets) == 1, "a reload must not append a tileset"
    assert after.firstgid == before.firstgid
    assert after.tileset.tile_count == before.tileset.tile_count
    assert int(after.tileset.pixels[5, 5, 0]) == 1
    # The cell still names the same tile, which is the whole point.
    assert int(painted.data[0, 0]) == before.firstgid + 1
    assert tab.doc.history.can_undo


def test_reloading_an_atlas_of_a_different_size_is_refused_by_name():
    """The roles are positional: a cropped atlas is a set whose tile 93 is not
    the tile the map thinks it is."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    before = tab.doc.tilesets[0].tileset

    plotter_mode.on_task_done(
        ctx,
        _Done(
            f"plotter-tileset-image:{tab.uid}",
            {"replace": (0, "small.png", np.zeros((16, 16, 4), dtype=np.uint8))},
        ),
    )

    assert tab.doc.tilesets[0].tileset is before
    assert ctx.toasts and ctx.toasts[-1][1] == "error"
    assert "32x32" in ctx.toasts[-1][0]


def test_the_reload_door_picks_a_file_on_its_own_key(monkeypatch):
    """Not the ``plotter-tileset:`` arrival key the other five doors share: this
    one replaces rather than appends, and sharing the key would make a reload and
    an add refuse each other as duplicates."""
    from warlock.studio import dialogs, plotter_tilesets

    ctx = FakeCtx()
    tab = _tab(ctx)
    fresh = np.zeros((32, 32, 4), dtype=np.uint8)
    calls: list[str] = []

    def _picked(title: str, filters: Any) -> Any:
        calls.append(title)
        return Path("atlas.png")

    monkeypatch.setattr(dialogs, "open_file", _picked)
    monkeypatch.setattr(plotter_tilesets, "_decode", lambda path: fresh)
    plotter_mode.ask_replace_tileset(ctx, 0)

    assert calls, "the door never opened a picker"
    assert ctx.submitted == [f"plotter-tileset-image:{tab.uid}"]
    assert ctx.result["replace"][0] == 0
    assert ctx.result["uid"] == tab.uid


def test_the_reload_door_refuses_an_index_the_map_does_not_have():
    ctx = FakeCtx()
    _tab(ctx, tileset=False)
    plotter_mode.ask_replace_tileset(ctx, 0)
    assert ctx.submitted == []
    assert ctx.toasts and ctx.toasts[-1][1] == "error"


def _menu_rows(ctx: Any, state: Any, tab: Any, menu: str, hit: str, monkeypatch) -> None:
    """Draw one menu's rows with ``hit`` pressed, through the real dispatch.

    ``controls.menu_item`` is what a row *is*, so replacing it is what a click
    is: everything between the press and the action -- the label, the enabled
    gate, the branch it lands in -- is the code under test. A test that called
    the action directly would pass on a menu with no row in it at all, which is
    exactly the defect these two rows exist to close.
    """
    from warlock.studio import controls
    from warlock.studio.panes import plotter_menu

    pressed: list[str] = []

    def fake_menu_item(label, shortcut="", selected=False, enabled=True, **kwargs):
        if label.startswith(f"{hit}##"):
            pressed.append(label)
            return (bool(enabled), selected)
        return (False, selected)

    monkeypatch.setattr(controls, "menu_item", fake_menu_item)
    # The only other imgui call a row list makes, and it needs a live context.
    monkeypatch.setattr(controls, "menu_separator", lambda: None)
    plotter_menu._rows(ctx, state, tab, menu)
    assert pressed, f"no row named {hit!r} was drawn under {menu}"


def test_the_map_menu_has_a_go_to_coordinate_row_that_acts(monkeypatch):
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    _menu_rows(ctx, state, tab, "Map", "Go to coordinate...", monkeypatch)
    assert state.goto_pending is True


def test_the_tileset_menu_has_a_reload_row_that_acts(monkeypatch):
    from warlock.studio import plotter_tilesets

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = plotter_mode.ensure(ctx)
    state.tileset_index = 0
    seen: list[int] = []
    monkeypatch.setattr(
        plotter_tilesets, "ask_replace_tileset", lambda _ctx, index: seen.append(index)
    )
    _menu_rows(ctx, state, tab, "Tileset", "Reload the image...", monkeypatch)
    assert seen == [0]
