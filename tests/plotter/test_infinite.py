"""Infinite maps: the window, the origin, and the chunks at the file doors.

The document holds a **dense window over the populated extent plus an origin**
rather than sparse chunks, and that deviation from the format is what these
tests are mostly about: the chunked shape lives at the Tiled codec's door, so
the statements worth making are that a true coordinate survives every hop, that
growth is one undoable step, and that the two shapes agree at the seam.

The corpus gate (``test_fixture_corpus``) already reads ``infinite-112`` through
both readers, both writers and ``.wmap``; what is here is the behaviour a
fixture cannot state.
"""

from __future__ import annotations

import base64
import gzip
import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from warlock.studio.plotter import tilemap, tmx, tools, wmap
from warlock.studio.plotter.tilemap import new_uid
from warlock.studio.tilegrid import gid as gidlib
from warlock.studio.tilegrid.tileset import Tileset


def _pixels() -> np.ndarray:
    array = np.zeros((32, 32, 4), dtype=np.uint8)
    array[..., 3] = 255
    return array


LOADERS = {
    "image_loader": lambda _s: _pixels(),
    "tsx_loader": lambda _s: Tileset(name="t", pixels=_pixels(), tile_w=16, tile_h=16),
}


_FIXTURES = Path(__file__).parent / "fixtures" / "tiled"


def _corpus_loaders(extra: dict[str, bytes] | None = None) -> dict[str, object]:
    """The fixture directory's own loaders, and optionally an export read back
    out of memory rather than off disk."""
    from ._corpus import loaders_for

    return loaders_for(_FIXTURES, extra=extra or {})


def _doc(width: int = 4, height: int = 4) -> tilemap.MapDoc:
    doc = tilemap.MapDoc(width, height, 16, 16, infinite=True)
    doc.add_tile_layer("Ground")
    doc.add_tileset(Tileset(name="t", pixels=_pixels(), tile_w=16, tile_h=16))
    doc.history.clear()
    return doc


# --- the window and its origin ------------------------------------------------


def test_a_true_coordinate_survives_a_growth():
    """The whole arrangement in one assertion. The window slides under the
    content, so the cell painted at (1, 1) is still at (1, 1) afterwards --
    at a different *index*, which is the point of keeping an origin at all."""
    doc = _doc()
    doc.tile_layers()[0].data[1, 1] = 3
    doc.grow_to_hold(-2, -2, -2, -2)
    row, column = np.argwhere(doc.tile_layers()[0].data != 0)[0].tolist()
    assert (column + doc.origin_x, row + doc.origin_y) == (1, 1)


def test_growth_is_one_undo_step_and_takes_the_origin_with_it():
    """A shape undone without its origin is every cell at a different true
    coordinate, and the next save writes the wrong chunks."""
    doc = _doc()
    assert doc.grow_to_hold(-3, -1, 0, 0) is True
    assert (doc.width, doc.height, doc.origin_x, doc.origin_y) == (7, 5, -3, -1)
    assert len(doc.history) == 1
    doc.history.undo(doc)
    assert (doc.width, doc.height, doc.origin_x, doc.origin_y) == (4, 4, 0, 0)
    doc.history.redo(doc)
    assert (doc.width, doc.height, doc.origin_x, doc.origin_y) == (7, 5, -3, -1)


def test_a_finite_map_never_grows():
    """Painting past a finite map's edge is clipped, exactly as it always was.
    The flag is the only thing that decides it."""
    doc = _doc()
    doc.set_infinite(False, crop=False)
    assert doc.grow_to_hold(-5, -5, 20, 20) is False
    assert (doc.width, doc.height, doc.origin_x, doc.origin_y) == (4, 4, 0, 0)


def test_the_populated_extent_is_capped():
    """An infinite map does not mean an unbounded allocation -- the same
    on-a-slip-of-the-keyboard argument a finite map's size cap makes."""
    doc = _doc()
    with pytest.raises(ValueError, match="extent"):
        doc.grow_to_hold(0, 0, 10_000, 0)


def test_a_growth_moves_objects_with_the_window():
    """``resize``'s own rule, inherited rather than re-answered. The document is
    window-relative throughout -- an object's pixels are measured from the
    window's corner, not from a true origin -- so when the window slides two
    cells left, everything on it slides with it and the object stays over the
    cell it was drawn on."""
    doc = _doc()
    layer = doc.add_object_layer("O")
    obj = doc.add_object(
        layer.uid, tilemap.MapObject(uid=new_uid(), name="spawn", x=32.0, y=16.0)
    )
    doc.grow_to_hold(-2, 0, 0, 0)
    assert (obj.x, obj.y) == (64.0, 16.0)


def test_an_object_position_converts_at_the_tiled_door_and_nowhere_else():
    """A file gives object positions in *true* pixels, which is the space its
    chunk coordinates are in. Converting at the codec is what lets both
    renderers, every tool and ``.wmap`` stay window-relative and unaware."""
    doc = tmx.read_tmx(
        (_FIXTURES / "infinite-112.tmx").read_bytes(), **_corpus_loaders()
    )
    obj = next(
        layer.objects[0]
        for layer in doc.all_layers()
        if isinstance(layer, tilemap.ObjectLayer)
    )
    # The file says (-64, -64) and the window starts at cell (-16, -16) of 32px
    # cells, so the object sits 448px into the window.
    assert (obj.x, obj.y) == (448.0, 448.0)
    assert b'x="-64.0" y="-64.0"' in tmx.tmx_export(doc)["map.tmx"]


# --- the tools ----------------------------------------------------------------


def test_a_flood_is_bounded_by_the_window():
    """The property the dense window buys for free. A flood over genuinely
    unbounded empty space does not terminate; here the window *is* the
    populated extent, so "bounded to content" is what the storage says."""
    doc = _doc(3, 3)
    layer = doc.tile_layers()[0]
    region = tools.flood_fill(layer.data, 1, 1, 2)
    assert region is not None
    x0, y0, block = region
    assert (x0, y0) == (0, 0)
    assert block.shape == (3, 3)
    assert int((block == 2).sum()) == 9


def test_a_stroke_spanning_a_chunk_boundary_is_one_step():
    """Chunks are a file shape, so a stroke that crosses one is not a special
    case -- and this is the test that says so rather than assuming it."""
    doc = _doc(40, 4)
    layer = doc.tile_layers()[0]
    doc.begin_stroke(layer.uid)
    for x in range(0, 40):
        doc.stroke_write(x, 0, np.array([[3]], gidlib.DTYPE))
    doc.end_stroke()
    assert len(doc.history) == 1
    assert int((doc.tile_layers()[0].data == 3).sum()) == 40


# --- the file doors -----------------------------------------------------------


def test_an_empty_chunk_is_not_written():
    """The format's own shape, and what makes an erase shrink a file: a map
    painted in two clusters writes two clusters of chunks, not the rectangle
    between them."""
    doc = _doc(40, 40)
    doc.tile_layers()[0].data[0, 0] = 1
    doc.tile_layers()[0].data[39, 39] = 1
    chunks = tmx.chunks_of(doc.tile_layers()[0].data, doc.origin_x, doc.origin_y)
    assert len(chunks) == 2
    assert {(x, y) for x, y, _b in chunks} == {(0, 0), (32, 32)}


def test_chunks_are_aligned_in_true_coordinates():
    """So the same content saved from two windows -- before and after a growth
    -- writes the same chunks, rather than chunks offset by the growth."""
    doc = _doc(4, 4)
    doc.tile_layers()[0].data[1, 1] = 1
    before = tmx.chunks_of(doc.tile_layers()[0].data, doc.origin_x, doc.origin_y)
    doc.grow_to_hold(-20, -20, -20, -20)
    after = tmx.chunks_of(doc.tile_layers()[0].data, doc.origin_x, doc.origin_y)
    kept = [(x, y, b) for x, y, b in after if (x, y) == before[0][:2]]
    assert len(kept) == 1
    assert np.array_equal(kept[0][2], before[0][2])


def test_a_layer_with_no_chunks_is_a_map_that_still_opens():
    """Zero-sized arrays are not a thing this engine has anywhere else, and a
    map with nothing painted yet still has to open."""
    body = '<layer id="1" name="L"><data encoding="csv"></data></layer>'
    data = (
        '<map version="1.10" orientation="orthogonal" infinite="1" width="0" '
        'height="0" tilewidth="16" tileheight="16">'
        '<tileset firstgid="1" source="t.tsx"/>' + body + "</map>"
    ).encode()
    doc = tmx.read_tmx(data, **LOADERS)
    assert (doc.width, doc.height) == (1, 1)


def test_two_layers_at_different_extents_share_one_window():
    """Layers are read one at a time and each carries its own chunk extent, so
    the document's window is their union -- and the layer read first has to
    move when a later one reaches further left."""
    layer = (
        '<layer id="{i}" name="L{i}"><data encoding="csv">'
        '<chunk x="{x}" y="0" width="1" height="1">1</chunk>'
        "</data></layer>"
    )
    body = layer.format(i=1, x=0) + layer.format(i=2, x=-8)
    data = (
        '<map version="1.10" orientation="orthogonal" infinite="1" width="1" '
        'height="1" tilewidth="16" tileheight="16">'
        '<tileset firstgid="1" source="t.tsx"/>' + body + "</map>"
    ).encode()
    doc = tmx.read_tmx(data, **LOADERS)
    assert (doc.origin_x, doc.width) == (-8, 9)
    first, second = doc.tile_layers()
    assert int(first.data[0, 8]) == 1
    assert int(second.data[0, 0]) == 1
    assert not len(doc.history)


@pytest.mark.parametrize(
    ("encoding", "compression"),
    [("csv", ""), ("base64", ""), ("base64", "zlib"), ("base64", "gzip")],
)
def test_a_chunk_reads_in_every_encoding(encoding, compression):
    """Chunked data is not a fourth encoding; it is the same three inside a
    different element, and the chunk reader must not have quietly hard-wired
    the one the exporter happens to write."""
    block = np.zeros((1, 1), dtype=gidlib.DTYPE)
    block[0, 0] = 2
    text = "2" if encoding == "csv" else _encoded(block, compression)
    attrs = f'encoding="{encoding}"'
    if compression:
        attrs += f' compression="{compression}"'
    body = (
        f'<layer id="1" name="L"><data {attrs}>'
        f'<chunk x="6" y="7" width="1" height="1">{text}</chunk>'
        "</data></layer>"
    )
    data = (
        '<map version="1.10" orientation="orthogonal" infinite="1" width="1" '
        'height="1" tilewidth="16" tileheight="16">'
        '<tileset firstgid="1" source="t.tsx"/>' + body + "</map>"
    ).encode()
    doc = tmx.read_tmx(data, **LOADERS)
    assert (doc.origin_x, doc.origin_y) == (6, 7)
    assert int(doc.tile_layers()[0].data[0, 0]) == 2


def _encoded(block: np.ndarray, compression: str) -> str:
    raw = block.astype("<u4").tobytes()
    if compression == "zlib":
        raw = zlib.compress(raw)
    elif compression == "gzip":
        raw = gzip.compress(raw)
    return base64.b64encode(raw).decode()


def test_our_own_export_reads_back_with_its_cells_where_they_were():
    """The round trip the corpus gate makes for the fixture, made here for a
    document the editor grew rather than one a file described."""
    doc = _doc(4, 4)
    doc.tile_layers()[0].data[1, 1] = 2
    doc.grow_to_hold(-20, -20, -20, -20)
    files = tmx.tmx_export(doc)
    back = tmx.read_tmx(files["map.tmx"], **_corpus_loaders(files))
    row, column = np.argwhere(back.tile_layers()[0].data == 2)[0].tolist()
    assert (column + back.origin_x, row + back.origin_y) == (1, 1)


def test_wmap_stores_the_origin_and_gates_the_version():
    """The reserved key activating exactly as reserved -- and the gate, because
    an old reader that ignored ``origin`` would open the map with every cell at
    a different true coordinate."""
    doc = _doc()
    doc.grow_to_hold(-5, -6, 0, 0)
    manifest = json.loads(wmap.manifest_json(doc))
    assert manifest["infinite"] is True
    assert manifest["origin"] == [-5, -6]
    assert manifest["version"] == wmap.VERSION
    back = wmap.read_wmap(wmap.wmap_bytes(doc))
    assert (back.origin_x, back.origin_y) == (-5, -6)


# --- the conversion -----------------------------------------------------------


def test_making_a_map_infinite_keeps_every_cell_where_it_is():
    """Nothing to re-chunk: the rectangle it has becomes the window it starts
    with, so the conversion loses nothing and asks nothing."""
    doc = tilemap.MapDoc(4, 4, 16, 16)
    doc.add_tile_layer("Ground")
    doc.tile_layers()[0].data[2, 3] = 1
    doc.history.clear()
    assert doc.set_infinite(True) is True
    assert (doc.width, doc.height, doc.origin_x, doc.origin_y) == (4, 4, 0, 0)
    assert int(doc.tile_layers()[0].data[2, 3]) == 1


def test_giving_an_infinite_map_a_fixed_size_crops_to_content():
    """The only honest rectangle to pick: the window may carry slack an erase
    left behind, and the cells sit at true coordinates a finite map cannot
    express."""
    doc = _doc(6, 6)
    doc.tile_layers()[0].data[2, 2] = 1
    doc.tile_layers()[0].data[3, 4] = 1
    assert doc.set_infinite(False) is True
    assert (doc.width, doc.height) == (3, 2)
    assert (doc.origin_x, doc.origin_y) == (0, 0)
    assert int(doc.tile_layers()[0].data[0, 0]) == 1


def test_the_conversion_is_one_undo_step_even_when_it_crops():
    """Two Ctrl+Z presses would show a cropped map that is still infinite --
    a state that never existed."""
    doc = _doc(6, 6)
    doc.tile_layers()[0].data[2, 2] = 1
    doc.set_infinite(False)
    assert len(doc.history) == 1
    doc.history.undo(doc)
    assert doc.infinite is True
    assert (doc.width, doc.height) == (6, 6)


def test_an_empty_infinite_map_is_not_cropped_to_nothing():
    """There is no content to crop to, and a 1x1 map is not what "make this
    finite" meant."""
    doc = _doc(6, 6)
    assert doc.set_infinite(False) is True
    assert (doc.width, doc.height) == (6, 6)


def test_converting_to_what_it_already_is_does_nothing():
    """The no-op rule, which on the undo stack is the difference between a
    step and a Ctrl+Z that appears to do nothing."""
    doc = _doc()
    assert doc.set_infinite(True) is False
    assert not len(doc.history)
