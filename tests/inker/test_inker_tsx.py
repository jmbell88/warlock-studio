"""Tileset export/import + the Plotter handoff -- Wave 3, Chunk 3.6.

An Inker tileset IS a ``tilegrid.Tileset`` (Chunk 3.1's ``doc.tilesets``), so
this suite is mostly proving the *absence* of conversion: the ``.tsx`` this
writes reads back through the one Tiled reader in the repo bit-exact, the
pair it writes is internally consistent, and the object the map ends up
holding is the very one Inker built rather than a copy of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from warlock.studio import inker_mode, plotter_mode, plotter_tilesets
from warlock.studio.inker.document import Document
from warlock.studio.inker.tiles import strip
from warlock.studio.inker_state import InkerDoc, InkerState
from warlock.studio.plotter import tsx as tsxlib
from warlock.studio.plotter.tilemap import MapDoc
from warlock.studio.plotter_state import PlotterDoc, PlotterState
from warlock.studio.tilegrid.tileset import TerrainSpec, Tileset

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


# --- fixtures ------------------------------------------------------------------


def _tile(colour: tuple[int, int, int, int], w: int = 4, h: int = 4) -> np.ndarray:
    tile = np.zeros((h, w, 4), dtype=np.uint8)
    tile[..., 0], tile[..., 1], tile[..., 2], tile[..., 3] = colour
    return tile


def _blank_tile(w: int = 4, h: int = 4) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


def _strip_tileset(*colours: tuple[int, int, int, int], w: int = 4, h: int = 4) -> Tileset:
    """An Inker-authored atlas: a required-blank tile plus one per colour,
    exactly the shape ``convert_layer_to_tilemap`` mints."""
    stack = np.stack([_blank_tile(w, h), *[_tile(c, w, h) for c in colours]], axis=0)
    return strip(stack)


def _terrain_tileset(terrains: int = 2, k: int = 1) -> Tileset:
    from warlock.studio.tilegrid import blob

    tile = 4
    specs = tuple(
        TerrainSpec(f"T{i}", (10 * i, 200, 0, 255), (0, 90, 0, 255)) for i in range(terrains)
    )
    pixels = np.zeros((terrains * k * k * tile, blob.TILE_COUNT * tile, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return Tileset(
        name="ground", pixels=pixels, tile_w=tile, tile_h=tile, terrains=specs, phases=k
    )


class _Ctx:
    """Runs a submitted callable inline -- the shape every inker_mode test in
    this repo uses so the test sees what the task thread would have done
    without needing one."""

    def __init__(self, state: InkerState) -> None:
        self.state = _AppState(state)
        self.toasts: list[tuple[str, str]] = []
        self.submitted: list[str] = []
        self.accept = True
        self.result: Any = None

    def toast(self, message: str, kind: str = "info", **_kw: Any) -> None:
        self.toasts.append((message, kind))

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        if not self.accept:
            return False
        self.result = run(*args)
        return True


class _AppState:
    def __init__(self, inker_state: InkerState) -> None:
        self.inker = inker_state
        self.plotter: PlotterState | None = None


def _open(*colours: tuple[int, int, int, int]):
    doc = Document.blank(8, 8)
    slot = doc.add_tileset(_strip_tileset(*colours))
    tab = InkerDoc(doc=doc, title="atlas.ora")
    state = InkerState()
    state.add(tab)
    ctx = _Ctx(state)
    return ctx, state, tab, slot


class _Done:
    def __init__(self, key: str, result: Any) -> None:
        self.key = key
        self.result = result


def _plotter_tab(ctx: _Ctx) -> PlotterDoc:
    map_doc = MapDoc(4, 4, 16, 16)
    map_doc.add_tile_layer("Ground")
    tab = PlotterDoc(doc=map_doc, title="Untitled")
    ctx.state.plotter = PlotterState()
    ctx.state.plotter.add(tab)
    return tab


# --- export ----------------------------------------------------------------


def test_tsx_bytes_of_an_inker_authored_strip_open_back_bit_exact(monkeypatch, tmp_path):
    from warlock.studio import dialogs

    ctx, state, tab, slot = _open(RED, BLUE)
    dest = tmp_path / "atlas.tsx"
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: dest)

    inker_mode.export_tileset(ctx, tab, index=0)

    from PIL import Image

    png_path = dest.with_suffix(".png")
    image = np.asarray(Image.open(png_path).convert("RGBA"))
    back = tsxlib.read_tsx(dest.read_bytes(), image)

    assert np.array_equal(back.pixels, slot.tileset.pixels)
    assert (back.tile_w, back.tile_h) == (slot.tileset.tile_w, slot.tileset.tile_h)
    assert back.tile_count == slot.tileset.tile_count
    assert back.columns == slot.tileset.columns


def test_export_writes_both_files_as_a_consistent_pair(monkeypatch, tmp_path):
    from warlock.studio import dialogs

    ctx, state, tab, slot = _open(RED)
    dest = tmp_path / "mine.tsx"
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: dest)

    inker_mode.export_tileset(ctx, tab, index=0)

    assert dest.exists()
    png_path = dest.with_suffix(".png")
    assert png_path.exists()
    assert tsxlib.tsx_source(dest.read_bytes()) == png_path.name


def test_export_reads_the_tileset_before_the_picker(monkeypatch, tmp_path):
    """``export_palette``'s own rule: serialising after an unbounded modal
    would write whatever the document changed to while it was up."""
    from warlock.studio import dialogs

    ctx, state, tab, slot = _open(RED)
    dest = tmp_path / "atlas.tsx"
    original = np.array(slot.tileset.pixels)

    def fake_save(title: str, default: str, filters: Any = None) -> Path:
        # Grow the tileset *while the picker is up*; the export must not see it.
        tab.doc._apply_tileset_grow(slot.uid, _tile(BLUE)[np.newaxis, ...])
        return dest

    monkeypatch.setattr(dialogs, "save_file", fake_save)
    inker_mode.export_tileset(ctx, tab, index=0)

    from PIL import Image

    image = np.asarray(Image.open(dest.with_suffix(".png")).convert("RGBA"))
    assert np.array_equal(image, original)


def test_a_cancelled_export_picker_writes_nothing(monkeypatch, tmp_path):
    from warlock.studio import dialogs

    ctx, state, tab, slot = _open(RED)
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: None)

    inker_mode.export_tileset(ctx, tab, index=0)

    assert list(tmp_path.iterdir()) == []


def test_exporting_with_no_document_is_refused():
    ctx = _Ctx(InkerState())
    inker_mode.export_tileset(ctx, None, index=0)
    assert ctx.submitted == []
    assert ctx.toasts and ctx.toasts[-1][1] == "error"


# --- import ------------------------------------------------------------------


def test_import_places_a_terrain_set_with_terrains_intact(monkeypatch, tmp_path):
    from warlock.studio import dialogs

    source = _terrain_tileset(terrains=2, k=1)
    tsx_path = tmp_path / "ground.tsx"
    png_path = tmp_path / "ground.png"
    tsx_path.write_bytes(tsxlib.tsx_bytes(source, image_name="ground.png"))
    from warlock.studio.plotter import pngio

    png_path.write_bytes(pngio.png_bytes(source.pixels))
    monkeypatch.setattr(dialogs, "open_file", lambda *a, **k: tsx_path)

    ctx, state, tab, _existing = _open(RED)
    before = len(tab.doc.tilesets)

    inker_mode.import_tileset(ctx, tab)
    inker_mode.on_task_done(
        ctx, _Done(f"inker-tileset-import:{tab.uid}", ctx.result)
    )

    assert len(tab.doc.tilesets) == before + 1
    landed = tab.doc.tilesets[-1].tileset
    assert [t.name for t in landed.terrains] == ["T0", "T1"]
    assert landed.phases == source.phases


def test_a_cancelled_import_picker_changes_nothing(monkeypatch):
    from warlock.studio import dialogs

    monkeypatch.setattr(dialogs, "open_file", lambda *a, **k: None)
    ctx, state, tab, _slot = _open(RED)
    before = len(tab.doc.tilesets)

    inker_mode.import_tileset(ctx, tab)
    inker_mode.on_task_done(
        ctx, _Done(f"inker-tileset-import:{tab.uid}", ctx.result)
    )

    assert len(tab.doc.tilesets) == before
    assert not ctx.toasts


def test_importing_with_no_document_is_refused():
    ctx = _Ctx(InkerState())
    inker_mode.import_tileset(ctx, None)
    assert ctx.submitted == []
    assert ctx.toasts and ctx.toasts[-1][1] == "error"


def test_an_unsupported_tsx_is_refused_by_name(monkeypatch, tmp_path):
    from warlock.service.errors import Invalid
    from warlock.studio import dialogs

    tsx_path = tmp_path / "weird.tsx"
    tsx_path.write_text(
        '<?xml version="1.0"?>\n'
        '<tileset name="w" tilewidth="4" tileheight="4" tilecount="1" columns="1">'
        '<image source="weird.png" width="4" height="4"/>'
        '<terraintypes/>'
        "</tileset>",
        encoding="utf-8",
    )
    monkeypatch.setattr(dialogs, "open_file", lambda *a, **k: tsx_path)
    ctx, state, tab, _slot = _open(RED)

    with pytest.raises(Invalid, match="terrain types"):
        inker_mode.import_tileset(ctx, tab)


# --- the Plotter handoff -------------------------------------------------------


def _land(ctx: _Ctx, plotter_tab: PlotterDoc) -> None:
    """Route ``use_inker_tileset``'s submit through ``on_task_done``, exactly
    as the real task runner would -- the same landing every one of the four
    other tileset-arrival doors shares."""
    plotter_mode.on_task_done(
        ctx, _Done(f"plotter-tileset:{plotter_tab.uid}", ctx.result)
    )


def test_handoff_hands_the_frozen_object_across_zero_copy():
    ctx, state, tab, slot = _open(RED)
    plotter_tab = _plotter_tab(ctx)

    plotter_tilesets.use_inker_tileset(ctx, tab.doc, slot.uid)
    _land(ctx, plotter_tab)

    ref = plotter_tab.doc.tilesets[-1]
    assert ref.tileset is slot.tileset


def test_an_inker_edit_after_handoff_leaves_the_plotter_object_unchanged():
    """Snapshot semantics: Inker never mutates a frozen tileset in place, so
    a later edit mints a new object and the map keeps drawing the old one."""
    ctx, state, tab, slot = _open(RED)
    plotter_tab = _plotter_tab(ctx)
    plotter_tilesets.use_inker_tileset(ctx, tab.doc, slot.uid)
    _land(ctx, plotter_tab)
    ref = plotter_tab.doc.tilesets[-1]
    before_pixels = np.array(ref.tileset.pixels)

    tab.doc._apply_tileset_tiles(slot.uid, [(1, _tile(BLUE))])

    assert ref.tileset is not tab.doc.tileset_slot(slot.uid).tileset
    assert np.array_equal(ref.tileset.pixels, before_pixels)


def test_handoff_with_no_plotter_tab_is_refused():
    ctx, state, tab, slot = _open(RED)
    plotter_tilesets.use_inker_tileset(ctx, tab.doc, slot.uid)
    assert ctx.submitted == []
    assert ctx.toasts and ctx.toasts[-1][1] == "error"


def test_handoff_of_an_unknown_uid_adds_nothing():
    ctx, state, tab, slot = _open(RED)
    plotter_tab = _plotter_tab(ctx)
    before = len(plotter_tab.doc.tilesets)

    plotter_tilesets.use_inker_tileset(ctx, tab.doc, slot.uid + 9999)
    _land(ctx, plotter_tab)

    assert len(plotter_tab.doc.tilesets) == before
