"""Wave 3 chunk 3.4: ``.ora`` persistence for tilesets and tilemap cels.

``tiles.json`` (``ora.TILES_MEMBER``/``ora.TILES_VERSION``) plus two auxiliary
member kinds -- ``data/tileset{i}.png`` (each strip) and
``data/tilerefs{n}.u32`` (raw little-endian uint32 refs planes) -- carry
everything ``inker/tiles.py`` and ``_doc_tiles.py`` cannot recover from a plain
cel PNG: the atlases, the track bindings and each cel's refs. Ordinary cel
PNGs are still written and read as honest RGBA, so any failure here costs tile
*structure* -- never a pixel or a frame.

``_assert_synced`` is the Wave 3 risk-item helper, copied from
``test_tile_edits.py`` (``tests/inker`` has no conftest): after a round trip,
every ``TilemapCel``'s ``pixels`` must equal ``materialize(refs, ts, size)``.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import pytest

from warlock.studio.inker import ora
from warlock.studio.inker.document import Document
from warlock.studio.inker.layers import Layer
from warlock.studio.inker.tiles import TilemapCel, materialize, strip

RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)
YELLOW = (255, 255, 0, 255)
WHITE = (255, 255, 255, 255)


def _tile(colour: tuple[int, int, int, int], w: int = 4, h: int = 4) -> np.ndarray:
    tile = np.zeros((h, w, 4), dtype=np.uint8)
    tile[..., 0], tile[..., 1], tile[..., 2], tile[..., 3] = colour
    return tile


def _blank_tile(w: int = 4, h: int = 4) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


def _tileset(*colours: tuple[int, int, int, int]):
    return strip(np.stack([_blank_tile(), *[_tile(c) for c in colours]], axis=0))


def _assert_synced(doc: Document) -> None:
    """Every tilemap cel's pixels agree with ``materialize(refs, ts, size)``."""
    layers = doc.stack if doc.anim is None else doc.anim.unique_cel_layers()
    for layer in layers:
        if isinstance(layer, TilemapCel):
            slot = doc.tileset_slot(layer.tileset_uid)
            want = materialize(layer.refs, slot.tileset, layer.size)
            assert np.array_equal(layer.pixels, want), f"cel {layer.uid} drifted from its refs"


def _rewrite_member(path: Path, member: str, data: bytes) -> None:
    """Replace one member's bytes, keeping every other member and the
    archive's order -- for exercising a single corrupted member without
    hand-building a whole ``.ora``."""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        items = {name: zf.read(name) for name in names}
    items[member] = data
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.writestr(name, items[name])


# -- still-document fixture: Background / Tilemap / Ink ----------------------


def _still_doc() -> Document:
    """8x8, 4x4 tiles (a 2x2 grid): a plain Background, a tilemap layer over
    its own tileset with two placed tiles, and a plain Ink layer on top."""
    doc = Document.blank(8, 8)
    ts = doc.add_tileset(_tileset(RED, GREEN))
    cel = doc.add_tilemap_layer(ts.uid, name="Tiles")
    doc.place_tiles(cel.uid, (0, 0), np.array([[1, 2]], dtype=np.uint32))
    doc.add_layer(name="Ink")
    doc.stack.active.pixels[0:2, 0:2] = WHITE
    return doc


# -- animated fixture: plain track + 2 tilemap tracks, one linked cel --------


def _animated_doc() -> Document:
    """8x8, 4x4 tiles: a plain raster track and two tilemap tracks (their own
    tilesets) across two frames. Track A's cel is linked into both frames;
    track B's cels are independent -- the one distinction the round trip has
    to preserve."""
    doc = Document.blank(8, 8)
    anim = doc.ensure_animation()

    ts_a = doc.add_tileset(_tileset(RED, GREEN))
    ts_b = doc.add_tileset(_tileset(BLUE, YELLOW))

    doc._ensure_cel_for(doc.stack[0].uid)
    doc.stack[0].pixels[1:3, 1:3] = WHITE

    layer_a = doc.add_tilemap_layer(ts_a.uid, name="Tiles A")
    from warlock.studio.tilegrid import gid

    doc.place_tiles(layer_a.uid, (0, 0), np.array([[gid.compose(2, flip_h=True)]], dtype=np.uint32))

    layer_b = doc.add_tilemap_layer(ts_b.uid, name="Tiles B")
    doc.place_tiles(layer_b.uid, (0, 0), np.array([[1]], dtype=np.uint32))

    track_b_uid = anim.tracks[doc.stack.index_of(layer_b.uid)].uid

    frame2 = doc.add_frame(link=True)

    # Un-link track B on frame 2 alone: a fresh, independent copy with its
    # own refs -- everything else (track A, the plain track) stays linked.
    old_b2 = anim.cels[(track_b_uid, frame2.uid)]
    new_b2 = old_b2.copy()
    new_b2.refs[0, 1] = gid.compose(2, flip_v=True)
    new_b2_ts = doc.tileset_slot(new_b2.tileset_uid).tileset
    new_b2.pixels = materialize(new_b2.refs, new_b2_ts, new_b2.size)
    anim.cels[(track_b_uid, frame2.uid)] = new_b2
    doc.stack.layers[doc.stack.index_of(layer_b.uid)] = new_b2

    return doc


# -- animated round trip -------------------------------------------------------


def test_animated_round_trip_preserves_tile_structure(tmp_path: Path):
    doc = _animated_doc()
    _assert_synced(doc)

    path = tmp_path / "a.ora"
    ora.write_ora(doc, path)
    back = ora.read_ora(path)

    assert back.anim is not None
    assert len(back.tilesets) == len(doc.tilesets) == 2
    _assert_synced(back)

    for i, slot in enumerate(doc.tilesets):
        back_slot = back.tilesets[i]
        assert np.array_equal(back_slot.tileset.pixels, slot.tileset.pixels)
        assert back_slot.tileset.tile_w == slot.tileset.tile_w
        assert back_slot.tileset.tile_h == slot.tileset.tile_h
        # A fresh process mints its own uids -- indices are the record, not
        # this one.
        assert back_slot.uid != slot.uid

    assert len(back.anim.tracks) == len(doc.anim.tracks) == 3
    assert back.anim.tracks[0].tileset_uid is None
    assert back.anim.tracks[1].tileset_uid == back.tilesets[0].uid
    assert back.anim.tracks[2].tileset_uid == back.tilesets[1].uid

    frame1, frame2 = back.anim.frames[0], back.anim.frames[1]
    track_a_uid, track_b_uid = back.anim.tracks[1].uid, back.anim.tracks[2].uid
    back_a1 = back.anim.cels[(track_a_uid, frame1.uid)]
    back_a2 = back.anim.cels[(track_a_uid, frame2.uid)]
    back_b1 = back.anim.cels[(track_b_uid, frame1.uid)]
    back_b2 = back.anim.cels[(track_b_uid, frame2.uid)]

    assert isinstance(back_a1, TilemapCel)
    assert isinstance(back_b1, TilemapCel)
    assert isinstance(back_b2, TilemapCel)
    assert back_a1 is back_a2, "the linked tilemap cel must still be linked"
    assert back_b1 is not back_b2, "the independent tilemap cel must not have been linked"

    src_anim = doc.anim
    orig_track_a_uid = src_anim.tracks[1].uid
    orig_track_b_uid = src_anim.tracks[2].uid
    orig_frame1_uid = src_anim.frames[0].uid
    orig_frame2_uid = src_anim.frames[1].uid
    orig_a1 = src_anim.cels[(orig_track_a_uid, orig_frame1_uid)]
    orig_b1 = src_anim.cels[(orig_track_b_uid, orig_frame1_uid)]
    orig_b2 = src_anim.cels[(orig_track_b_uid, orig_frame2_uid)]

    assert np.array_equal(back_a1.refs, orig_a1.refs)
    assert np.array_equal(back_b1.refs, orig_b1.refs)
    assert np.array_equal(back_b2.refs, orig_b2.refs)
    assert np.array_equal(back_a1.pixels, orig_a1.pixels)
    assert np.array_equal(back_b1.pixels, orig_b1.pixels)
    assert np.array_equal(back_b2.pixels, orig_b2.pixels)

    # The plain raster track round-trips as an ordinary layer, untouched by
    # any of this.
    plain0 = back.anim.cels[(back.anim.tracks[0].uid, frame1.uid)]
    assert not isinstance(plain0, TilemapCel)
    assert np.array_equal(
        plain0.pixels, src_anim.cels[(src_anim.tracks[0].uid, orig_frame1_uid)].pixels
    )


# -- still-document round trip ------------------------------------------------


def test_still_document_round_trip(tmp_path: Path):
    doc = _still_doc()
    _assert_synced(doc)

    path = tmp_path / "s.ora"
    ora.write_ora(doc, path)
    back = ora.read_ora(path)

    assert back.anim is None
    assert len(back.tilesets) == 1
    _assert_synced(back)

    assert not isinstance(back.stack[0], TilemapCel)
    assert isinstance(back.stack[1], TilemapCel)
    assert not isinstance(back.stack[2], TilemapCel)

    orig = doc.stack[1]
    assert np.array_equal(back.stack[1].refs, orig.refs)
    assert np.array_equal(back.stack[1].pixels, orig.pixels)
    assert np.array_equal(
        back.tileset_slot(back.stack[1].tileset_uid).tileset.pixels,
        doc.tileset_slot(orig.tileset_uid).tileset.pixels,
    )
    assert np.array_equal(back.stack[0].pixels, doc.stack[0].pixels)
    assert np.array_equal(back.stack[2].pixels, doc.stack[2].pixels)


# -- containment: a wrong version drops structure, keeps pixels --------------


def test_wrong_tiles_version_drops_structure_but_keeps_pixels(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    doc = _still_doc()
    path = tmp_path / "v.ora"
    ora.write_ora(doc, path)

    with zipfile.ZipFile(path) as zf:
        payload = json.loads(zf.read(ora.TILES_MEMBER))
    payload["version"] = 999
    _rewrite_member(path, ora.TILES_MEMBER, json.dumps(payload).encode("utf-8"))

    with caplog.at_level(logging.WARNING):
        back = ora.read_ora(path)

    assert back.tilesets == []
    assert not isinstance(back.stack[1], TilemapCel)
    assert isinstance(back.stack[1], Layer)
    # The picture survives intact -- the materialization the writer recorded,
    # not a re-derivation of anything.
    assert np.array_equal(back.stack[1].pixels, doc.stack[1].pixels)
    assert np.array_equal(back.stack[0].pixels, doc.stack[0].pixels)
    assert np.array_equal(back.stack[2].pixels, doc.stack[2].pixels)
    assert any(ora.TILES_MEMBER in record.message for record in caplog.records)


# -- containment: a corrupt tilerefs blob drops structure, keeps pixels ------


def test_corrupt_tilerefs_member_drops_structure_but_keeps_pixels(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    doc = _still_doc()
    path = tmp_path / "r.ora"
    ora.write_ora(doc, path)

    _rewrite_member(path, "data/tilerefs0.u32", b"\x01\x02")  # wrong length for the grid

    with caplog.at_level(logging.WARNING):
        back = ora.read_ora(path)

    assert back.tilesets == []
    assert not isinstance(back.stack[1], TilemapCel)
    assert np.array_equal(back.stack[1].pixels, doc.stack[1].pixels)
    assert any(ora.TILES_MEMBER in record.message for record in caplog.records)


# -- an unreferenced tileset is not garbage -----------------------------------


def test_unreferenced_tileset_survives(tmp_path: Path):
    doc = Document.blank(8, 8)
    doc.add_tileset(_tileset(RED))  # never bound to any track or cel

    path = tmp_path / "u.ora"
    ora.write_ora(doc, path)
    back = ora.read_ora(path)

    assert back.anim is None
    assert len(back.tilesets) == 1
    assert np.array_equal(back.tilesets[0].tileset.pixels, doc.tilesets[0].tileset.pixels)
    assert not any(isinstance(layer, TilemapCel) for layer in back.stack)


# -- ORA-side half of the end-to-end chain (Task 6 does the rest) ------------


def test_reopened_document_still_supports_tile_edits(tmp_path: Path):
    """What comes back is a live document, not just correct static fields --
    the funnel keeps working on it. The rest of the chain (``.aseprite``
    import onto this same model) is Task 6's; this is the ORA half."""
    doc = _still_doc()
    path = tmp_path / "chain.ora"
    ora.write_ora(doc, path)
    back = ora.read_ora(path)

    cel = back.stack[1]
    assert isinstance(cel, TilemapCel)
    assert back.place_tiles(cel.uid, (1, 1), np.array([[1]], dtype=np.uint32))
    _assert_synced(back)
