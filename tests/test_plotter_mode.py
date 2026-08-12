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

    monkeypatch.setattr(plotter_io.os, "replace", boom)
    with pytest.raises(OSError):
        plotter_io._write({"map.wmap": b"x"}, tmp_path / "level.wmap")
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert tab is not None


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

    path = tmp_path / "hostile.tmx"
    path.write_bytes(
        b'<map version="1.10" orientation="hexagonal" width="1" height="1" '
        b'tilewidth="16" tileheight="16"/>'
    )
    with pytest.raises(Invalid) as exc:
        plotter_io._load(path)
    assert "This map could not be opened" in str(exc.value)
    assert "hexagonal" in str(exc.value)
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
    """A tab with a generated terrain set adopted and that terrain in hand."""
    tab = _tab(ctx, tileset=False)
    plotter_mode.generate_terrain_set(ctx, _spec())
    plotter_mode.on_task_done(ctx, _Done(f"plotter-tileset:{tab.uid}", ctx.result))
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


def test_ctrl_shift_z_redoes_the_way_inker_and_clay_accept(monkeypatch):
    """Plotter was Ctrl+Y only, so a user arriving from either of the other two
    editors found their redo silently doing nothing."""
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx, dirty=True)
    dirty_data = tab.doc.tile_layers()[0].data.copy()

    event, mods = _key("z", ctrl=True)
    monkeypatch.setattr(pygame.key, "get_mods", lambda: mods)
    plotter_mode.handle_key(ctx, event)
    assert not tab.dirty

    event, mods = _key("z", ctrl=True, shift=True)
    monkeypatch.setattr(pygame.key, "get_mods", lambda: mods)
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
    up = pygame.event.Event(pygame.KEYUP, key=pygame.K_g)
    assert plotter_mode.handle_key(ctx, up) is False


def test_x_y_and_z_transform_the_brush_in_hand(monkeypatch):
    import pygame

    ctx = FakeCtx()
    _tab(ctx)
    state = plotter_mode.ensure(ctx)
    state.brush = np.array([[1, 2], [3, 4]], gid.DTYPE)

    event, mods = _key("x")
    monkeypatch.setattr(pygame.key, "get_mods", lambda: mods)
    assert plotter_mode.handle_key(ctx, event) is True
    assert np.array_equal(gid.tile_ids(state.brush), [[2, 1], [4, 3]])

    event, mods = _key("y")
    monkeypatch.setattr(pygame.key, "get_mods", lambda: mods)
    plotter_mode.handle_key(ctx, event)
    assert np.array_equal(gid.tile_ids(state.brush), [[4, 3], [2, 1]])


def test_shift_z_turns_the_brush_the_other_way(monkeypatch):
    """Three clockwise turns rather than a counter-clockwise routine of its
    own: Z then Shift+Z must be the identity."""
    import pygame

    ctx = FakeCtx()
    _tab(ctx)
    state = plotter_mode.ensure(ctx)
    original = np.array([[1, 2], [3, 4]], gid.DTYPE)
    state.brush = original.copy()

    event, mods = _key("z")
    monkeypatch.setattr(pygame.key, "get_mods", lambda: mods)
    plotter_mode.handle_key(ctx, event)
    assert not np.array_equal(state.brush, original)

    event, mods = _key("z", shift=True)
    monkeypatch.setattr(pygame.key, "get_mods", lambda: mods)
    plotter_mode.handle_key(ctx, event)
    assert np.array_equal(state.brush, original)


def test_a_brush_transform_is_not_refused_while_the_tab_is_busy(monkeypatch):
    """The brush is view state: transforming it writes nothing and pushes no
    step, so it stays outside the gate that ``Ctrl+Z`` sits behind."""
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True
    state = plotter_mode.ensure(ctx)
    state.brush = np.array([[1, 2]], gid.DTYPE)
    head = tab.doc.history.head

    event, mods = _key("x")
    monkeypatch.setattr(pygame.key, "get_mods", lambda: mods)
    assert plotter_mode.handle_key(ctx, event) is True
    assert np.array_equal(gid.tile_ids(state.brush), [[2, 1]])
    assert tab.doc.history.head == head
    assert not tab.dirty


def test_a_brush_transform_with_nothing_in_hand_does_nothing(monkeypatch):
    import pygame

    ctx = FakeCtx()
    _tab(ctx)
    state = plotter_mode.ensure(ctx)
    state.brush = None
    event, mods = _key("x")
    monkeypatch.setattr(pygame.key, "get_mods", lambda: mods)
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
