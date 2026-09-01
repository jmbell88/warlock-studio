"""The tileset editor sheet, and the refusal that guards a removal.

The refusal *is* the feature. A gid is a firstgid plus a local id, so dropping
a tileset out from under painted cells does not clear them -- it renumbers what
they mean. And "it is still in use" is a sentence a user cannot act on, so the
message carries the count and the layer.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from warlock.studio import plotter_state
from warlock.studio.plotter.tilemap import MapDoc
from warlock.studio.tilegrid.tileset import Tileset


def _tileset(name: str = "Overworld", tiles: int = 4) -> Tileset:
    pixels = np.zeros((8, 8 * tiles, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return Tileset(name=name, tile_w=8, tile_h=8, pixels=pixels)


def _doc() -> MapDoc:
    doc = MapDoc(4, 4, 8, 8)
    doc.add_tile_layer(name="Ground")
    return doc


def test_an_unused_tileset_can_be_removed_and_the_step_undone():
    doc = _doc()
    doc.add_tileset(_tileset())
    head = doc.history.head
    doc.remove_tileset(0)
    assert doc.tilesets == []
    doc.undo()
    assert len(doc.tilesets) == 1 and doc.history.head == head


def test_a_used_tileset_is_refused_by_name_with_a_count_and_a_layer():
    doc = _doc()
    ref = doc.add_tileset(_tileset())
    layer = doc.layers[0]
    data = layer.data.copy()
    data[0, :2] = ref.firstgid
    doc.write_region(layer.uid, 0, 0, data)
    with pytest.raises(ValueError) as raised:
        doc.remove_tileset(0)
    message = str(raised.value)
    assert "Overworld" in message and "Ground" in message
    assert "2" in message
    assert len(doc.tilesets) == 1, "and nothing was removed"


def test_a_survivor_keeps_its_firstgid_so_painted_cells_still_mean_what_they_meant():
    """A hole in gid space is legal -- wmap requires only that they increase."""

    doc = _doc()
    first = doc.add_tileset(_tileset("A"))
    second = doc.add_tileset(_tileset("B"))
    before = second.firstgid
    assert before > first.firstgid
    doc.remove_tileset(0)
    assert doc.tilesets[0].firstgid == before


def test_usage_counts_across_every_layer():
    doc = _doc()
    ref = doc.add_tileset(_tileset())
    doc.add_tile_layer(name="Detail")
    for layer in doc.layers:
        data = layer.data.copy()
        data[0, 0] = ref.firstgid
        doc.write_region(layer.uid, 0, 0, data)
    used, where = doc.tileset_usage(0)
    assert used == 2 and where in {"Ground", "Detail"}


def test_a_tileset_held_only_by_a_stamp_is_still_in_use():
    """The stamps are document state -- stored in the map, written to ``.wmap``
    since VERSION 11 -- and they hold gids exactly as a layer does. Counting
    only the layers let a tileset that nothing had painted yet but a stamp still
    named be removed; ``next_firstgid`` then reuses the range, and recalling the
    stamp paints the *new* tileset's tiles with no sign anything happened."""
    doc = _doc()
    ref = doc.add_tileset(_tileset())
    doc.set_stamp(1, np.full((1, 2), ref.firstgid, dtype=np.uint32), name="roof")

    used, where = doc.tileset_usage(0)
    assert used == 2
    assert "stamp 1" in where and "roof" in where
    with pytest.raises(ValueError, match="Overworld"):
        doc.remove_tileset(0)
    assert len(doc.tilesets) == 1


def test_reading_a_wmap_refuses_a_stamp_whose_tileset_is_gone():
    """``_validate`` checked the layers and the tile objects and not the stamps,
    so a map carrying a stamp nothing accounts for opened without complaint and
    only went wrong on recall -- where the user's gesture was a number key and
    there is nothing useful to say."""
    from warlock.studio.plotter import wmap

    doc = _doc()
    ref = doc.add_tileset(_tileset())
    doc.set_stamp(2, np.full((1, 1), ref.firstgid, dtype=np.uint32))
    assert wmap.read_wmap(wmap.wmap_bytes(doc)).stamps[2] is not None

    # Past the end of every tileset this map has.
    doc.set_stamp(2, np.full((1, 1), ref.last_gid + 9, dtype=np.uint32))
    with pytest.raises(ValueError, match="stamp 2"):
        wmap.read_wmap(wmap.wmap_bytes(doc))


def test_the_sheet_is_off_until_a_tileset_is_chosen():
    from warlock.studio.panes import plotter_tileset_editor

    doc = _doc()
    tab = plotter_state.PlotterDoc(doc=doc, title="m")
    state = plotter_state.PlotterState()
    state.add(tab)
    ctx = SimpleNamespace(state=SimpleNamespace(plotter=state))
    assert plotter_tileset_editor.active(ctx) is False
    doc.add_tileset(_tileset())
    state.editing_tileset = 0
    assert plotter_tileset_editor.active(ctx) is True
    # An index the list cannot honour is not a sheet: the map stays on screen.
    state.editing_tileset = 7
    assert plotter_tileset_editor.active(ctx) is False


def test_the_editor_offers_no_reordering():
    """Order *is* firstgid order, baked into every painted cell, so reordering
    means renumbering the map. Tiled reorders its tabs, not its ids."""

    import inspect

    from warlock.studio.panes import plotter_tileset_editor

    source = inspect.getsource(plotter_tileset_editor)
    assert "move_tileset" not in source
    assert "reorder" not in source.lower().split('"""')[2]
