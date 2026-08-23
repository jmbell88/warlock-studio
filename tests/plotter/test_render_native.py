"""The tile-blit kernel: bit-identical to the loop it replaces, or declining.

``_blit_cells_native`` answers only ``_over``'s masked-copy branch -- binary
source alpha, full opacity, normal mode -- because that is what a tileset
actually is and it is where the 4.3 seconds were. Everything here is either
"the two paths produce the same bytes" or "this is a case it must refuse", and
the second half matters as much as the first: a kernel that quietly answered a
partial-alpha layer would be wrong in a way no export byte would show until a
soft-edged tile met a second layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock import native
from warlock.studio.plotter import render
from warlock.studio.plotter.tilemap import MapDoc
from warlock.studio.tilegrid import gid
from warlock.studio.tilegrid.tileset import Tileset

needs_dll = pytest.mark.skipif(not native.available(), reason="warlockc.dll not built")


def _sheet(tiles: int = 8, size: int = 8, *, binary: bool = True, seed: int = 5):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size, size * tiles, 4), dtype=np.uint8)
    if binary:
        pixels[..., 3] = np.where(rng.random((size, size * tiles)) > 0.4, 255, 0)
    else:
        pixels[..., 3] = rng.integers(1, 255, size=(size, size * tiles), dtype=np.uint8)
    return Tileset(name="t", pixels=pixels, tile_w=size, tile_h=size)


def _doc(cells: int = 6, tiles: int = 8, size: int = 8, *, layers: int = 2, **kw):
    rng = np.random.default_rng(11)
    doc = MapDoc(cells, cells, size, size)
    doc.add_tileset(_sheet(tiles, size, **kw))
    for _ in range(layers):
        layer = doc.add_tile_layer()
        # Zeros left in deliberately: an empty cell is skipped entirely, and a
        # kernel fed one anyway would paint tile 0 across the gaps.
        data = rng.integers(0, tiles + 1, size=(cells, cells)).astype(gid.DTYPE)
        doc.write_region(layer.uid, 0, 0, data)
    return doc


def _both(doc, monkeypatch) -> tuple[np.ndarray, np.ndarray]:
    got = render.render_map(doc)
    monkeypatch.setattr(native, "available", lambda: False)
    want = render.render_map(doc)
    return got, want


@needs_dll
def test_the_fallback_is_genuinely_taken_when_the_seam_is_closed(monkeypatch):
    """Without this the monkeypatch could stop toggling anything and every
    comparison below would be one code path measured against itself."""
    calls: list[int] = []
    real = native.blit_cells
    monkeypatch.setattr(
        native, "blit_cells", lambda *a: (calls.append(1), real(*a))[1]
    )
    render.render_map(_doc())
    assert calls, "the kernel was never reached with the DLL present"

    monkeypatch.setattr(native, "available", lambda: False)
    calls.clear()
    render.render_map(_doc())
    assert not calls, "the numpy path still called the kernel"


@needs_dll
def test_a_binary_alpha_map_composites_to_the_same_bytes(monkeypatch):
    got, want = _both(_doc(), monkeypatch)
    assert np.array_equal(got, want), "bit-identical, not merely close"


@needs_dll
def test_the_both_clear_quirk_is_reproduced_rather_than_tidied_up(monkeypatch):
    """``_over`` writes rgb 0 where source *and* destination are both fully
    clear, discarding colour stored under a zero alpha. It is a quirk and it is
    the contract: a kernel that sensibly left those pixels alone would differ
    from the reference on every transparent pixel of every map.
    """
    doc = MapDoc(2, 1, 2, 2)
    # A tile that is entirely transparent but carries colour underneath.
    pixels = np.zeros((2, 4, 4), dtype=np.uint8)
    pixels[..., :3] = 200
    doc.add_tileset(Tileset(name="t", pixels=pixels, tile_w=2, tile_h=2))
    layer = doc.add_tile_layer()
    doc.write_region(layer.uid, 0, 0, np.array([[1, 2]], gid.DTYPE))

    got, want = _both(doc, monkeypatch)
    assert np.array_equal(got, want)
    assert not got.any(), "the discarded colour is the quirk being asserted"


@needs_dll
def test_a_tile_taller_than_its_cell_is_clipped_and_ordered_the_same(monkeypatch):
    """Tiled anchors an oversized tile by its bottom left, so it grows upward
    out of its cell and overlaps its neighbour. That overlap is why the kernel
    is fed the cells in draw order rather than grouped by tile -- grouping
    would be faster and would paint the overlaps the wrong way round.
    """
    doc = MapDoc(3, 3, 4, 4)
    doc.add_tileset(_sheet(tiles=4, size=8))  # 8px tiles on a 4px grid
    layer = doc.add_tile_layer()
    rng = np.random.default_rng(2)
    doc.write_region(
        layer.uid, 0, 0, rng.integers(1, 5, size=(3, 3)).astype(gid.DTYPE)
    )
    got, want = _both(doc, monkeypatch)
    assert np.array_equal(got, want)


@needs_dll
def test_a_flipped_tile_composites_the_same(monkeypatch):
    """Orientation is resolved in Python and memoised on ``(id, flags)``; the
    kernel only ever sees the already-oriented pixels. Asserted because that
    division of labour is exactly the sort a later reader moves."""
    doc = MapDoc(2, 2, 8, 8)
    doc.add_tileset(_sheet())
    layer = doc.add_tile_layer()
    data = np.array(
        [
            [1, 2 | gid.FLIP_H],
            [3 | gid.FLIP_V, 4 | gid.FLIP_D],
        ],
        dtype=gid.DTYPE,
    )
    doc.write_region(layer.uid, 0, 0, data)
    got, want = _both(doc, monkeypatch)
    assert np.array_equal(got, want)


@needs_dll
def test_a_partial_alpha_tileset_is_refused_and_falls_back(monkeypatch):
    """The kernel answers one branch of ``_over`` and must decline the rest."""
    calls: list[int] = []
    real = native.blit_cells
    monkeypatch.setattr(native, "blit_cells", lambda *a: (calls.append(1), real(*a))[1])
    doc = _doc(binary=False)
    got = render.render_map(doc)
    assert not calls, "a soft-edged tileset went through the masked-copy kernel"

    monkeypatch.setattr(native, "available", lambda: False)
    assert np.array_equal(got, render.render_map(doc))


@needs_dll
def test_a_layer_below_full_opacity_is_refused_and_falls_back(monkeypatch):
    calls: list[int] = []
    real = native.blit_cells
    monkeypatch.setattr(native, "blit_cells", lambda *a: (calls.append(1), real(*a))[1])
    doc = _doc(layers=1)
    doc.layers[0].opacity = 0.5
    got = render.render_map(doc)
    assert not calls

    monkeypatch.setattr(native, "available", lambda: False)
    assert np.array_equal(got, render.render_map(doc))


@needs_dll
def test_an_empty_layer_reaches_the_kernel_with_nothing_to_do(monkeypatch):
    doc = MapDoc(3, 2, 4, 4)
    doc.add_tileset(_sheet())
    doc.add_tile_layer()
    got, want = _both(doc, monkeypatch)
    assert np.array_equal(got, want)
    assert not got.any()
