"""What Plotter's status line reads, in the order Tiled reads it.

The line is a pure function of the state and the tab, which is what lets its
*order* be asserted at all: the ordering claim -- nearest-first, the reading
that moves with the mouse at the near end rather than the far one -- is not
something a screenshot can be made to fail on.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from warlock.studio.panes.plotter_canvas import status_bits
from warlock.studio.plotter.tilemap import MapDoc
from warlock.studio.tilegrid import gid as gidlib
from warlock.studio.tilegrid.tileset import Tileset


def _tileset(name: str = "terrain", size: int = 32, tile: int = 16) -> Tileset:
    pixels = np.zeros((size, size, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return Tileset(name=name, pixels=pixels, tile_w=tile, tile_h=tile)


def _map(with_tileset: bool = True):
    doc = MapDoc(8, 6, 16, 16)
    ref = doc.add_tileset(_tileset()) if with_tileset else None
    layer = doc.add_tile_layer()
    doc.set_active_layer(layer.uid)
    return doc, layer, ref


def _state(**kwargs):
    base = {"hover_cell": None, "select": None, "tool": "stamp", "terrain": None}
    base.update(kwargs)
    return SimpleNamespace(**base)


def _tab(doc, busy: bool = False):
    return SimpleNamespace(doc=doc, busy=busy, view=SimpleNamespace(zoom=1.0))


# --- the order ---------------------------------------------------------------


def test_the_line_reads_nearest_first():
    """Where the pointer is, then what is under it, then what you are painting
    on and with, then the map. It used to open with the size and the projection
    and finish with the cell -- the one reading that moves with the mouse, at
    the far end of the line."""
    doc, layer, ref = _map()
    cells = np.zeros((6, 8), gidlib.DTYPE)
    cells[2, 3] = gidlib.compose(ref.firstgid + 1)
    doc.write_region(layer.uid, 0, 0, cells)

    bits = status_bits(_state(hover_cell=(3, 2)), _tab(doc))

    assert bits[0] == "3, 2"
    assert bits[1].startswith("tile ")
    assert bits.index(f"{doc.width} x {doc.height}") > bits.index("tool stamp")
    assert bits[-1] == doc.projection


def test_the_map_facts_come_last_because_they_do_not_move():
    bits = status_bits(_state(), _tab(_map()[0]))
    assert bits[-2:] == ["8 x 6", "orthogonal"]


# --- the tile under the pointer ----------------------------------------------


def test_the_tile_under_the_pointer_is_named_by_its_local_id_and_tileset():
    """A gid is an artefact of how the map packs its tilesets together; the
    number the tile is called in the atlas is the one a person can act on."""
    doc, layer, ref = _map()
    cells = np.zeros((6, 8), gidlib.DTYPE)
    cells[1, 1] = gidlib.compose(ref.firstgid + 3)
    doc.write_region(layer.uid, 0, 0, cells)

    bits = status_bits(_state(hover_cell=(1, 1)), _tab(doc))

    assert "tile 3 (terrain)" in bits


def test_a_flipped_tile_still_reads_as_its_own_id():
    """``resolve`` takes the encoded value flags and all, so a flipped cell must
    not read as an out-of-range id -- which is what forgetting to mask does."""
    doc, layer, ref = _map()
    cells = np.zeros((6, 8), gidlib.DTYPE)
    cells[0, 0] = gidlib.compose(ref.firstgid + 2, flip_h=True, flip_d=True)
    doc.write_region(layer.uid, 0, 0, cells)

    assert "tile 2 (terrain)" in status_bits(_state(hover_cell=(0, 0)), _tab(doc))


def test_an_empty_cell_says_nothing_rather_than_zero():
    doc, _layer, _ref = _map()
    bits = status_bits(_state(hover_cell=(4, 4)), _tab(doc))
    assert bits[0] == "4, 4"
    assert not any(bit.startswith("tile ") for bit in bits)


def test_the_reading_comes_off_the_active_layer_not_the_topmost_one():
    """The line sits beside a tool that acts on the active layer, so a reading
    taken from a layer above it would name a tile the next click will not
    replace."""
    doc, lower, ref = _map()
    upper = doc.add_tile_layer()
    low = np.zeros((6, 8), gidlib.DTYPE)
    low[0, 0] = gidlib.compose(ref.firstgid + 1)
    doc.write_region(lower.uid, 0, 0, low)
    high = np.zeros((6, 8), gidlib.DTYPE)
    # Local 2, not 5: a 32x32 atlas at 16 px holds four tiles, and an id past
    # the end resolves to nothing -- which would have made this pass for the
    # wrong reason.
    high[0, 0] = gidlib.compose(ref.firstgid + 2)
    doc.write_region(upper.uid, 0, 0, high)

    doc.set_active_layer(lower.uid)
    assert "tile 1 (terrain)" in status_bits(_state(hover_cell=(0, 0)), _tab(doc))
    doc.set_active_layer(upper.uid)
    assert "tile 2 (terrain)" in status_bits(_state(hover_cell=(0, 0)), _tab(doc))


def test_an_object_layer_in_hand_reads_no_tile():
    doc, _layer, _ref = _map()
    obj = doc.add_object_layer()
    doc.set_active_layer(obj.uid)

    bits = status_bits(_state(hover_cell=(0, 0), tool="object"), _tab(doc))

    assert not any(bit.startswith("tile ") for bit in bits)
    assert bits[0] == "0, 0"


# --- what is suppressed ------------------------------------------------------


@pytest.mark.parametrize("cell", [None, (-1, 0), (99, 0), (0, 99)])
def test_a_pointer_off_the_map_reads_no_cell(cell):
    """Otherwise the line flickers while the pointer is over the chrome."""
    bits = status_bits(_state(hover_cell=cell), _tab(_map()[0]))
    assert not any("," in bit for bit in bits)


def test_the_selection_size_is_reported_when_there_is_one():
    doc, _layer, _ref = _map()
    assert "sel 3 x 2" in status_bits(_state(select=(1, 1, 3, 2)), _tab(doc))
    assert not any(bit.startswith("sel ") for bit in status_bits(_state(), _tab(doc)))


def test_a_saving_tab_says_so_last():
    doc, _layer, _ref = _map()
    assert status_bits(_state(), _tab(doc, busy=True))[-1] == "saving"


def test_a_map_with_no_layer_selected_still_reads():
    doc, layer, _ref = _map()
    doc.set_active_layer(None)
    bits = status_bits(_state(), _tab(doc))
    assert "no layer" in bits
