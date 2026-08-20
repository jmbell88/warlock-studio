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
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest

from warlock.studio.inker import asein, aseout, ora
from warlock.studio.inker.document import Document
from warlock.studio.inker.layers import Layer
from warlock.studio.inker.tiles import TilemapCel, materialize, strip
from warlock.studio.tilegrid import gid

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


# -- containment: a wrong-shaped tiles.json drops structure, keeps pixels ----


def test_tiles_json_as_a_list_drops_structure_but_keeps_pixels(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Valid JSON, wrong top-level shape: ``payload.get("version")`` on a
    list is an ``AttributeError``, not a ``KeyError``/``ValueError`` -- the
    one member-read exception tuple in this module missing it, unlike
    ``_read_colour``'s (the in-file precedent). Uncaught, that crashes the
    whole ORA open instead of costing tile structure alone.
    """
    doc = _still_doc()
    path = tmp_path / "l.ora"
    ora.write_ora(doc, path)

    _rewrite_member(path, ora.TILES_MEMBER, json.dumps([1, 2, 3]).encode("utf-8"))

    with caplog.at_level(logging.WARNING):
        back = ora.read_ora(path)

    assert back.tilesets == []
    assert not isinstance(back.stack[1], TilemapCel)
    assert isinstance(back.stack[1], Layer)
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


# -- a minimal .aseprite fixture, for the Task 6 end-to-end chain ------------
#
# A small subset of ``test_asein.py``'s own byte-builders, kept local rather
# than imported across test modules (``tests/inker`` has no ``__init__.py``,
# so a cross-module import here would be a path hazard the rest of the suite
# does not take on) -- just enough to build one animated file with a tileset,
# a tilemap layer carrying a flipped ref, and a linked tilemap cel.


def _ase_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<H", len(raw)) + raw


def _ase_chunk(kind: int, payload: bytes) -> bytes:
    return struct.pack("<IH", len(payload) + 6, kind) + payload


def _ase_frame(chunks: list[bytes], duration: int = 100) -> bytes:
    body = b"".join(chunks)
    return (
        struct.pack("<IHHHHI", len(body) + 16, 0xF1FA, len(chunks), duration, 0, 0)
        + body
    )


def _ase_header(frames: int, width: int, height: int) -> bytes:
    head = struct.pack(
        "<IHHHHHIHIIB3sHBBhhHH",
        0,
        0xA5E0,
        frames,
        width,
        height,
        32,
        1,
        100,
        0,
        0,
        0,
        b"\0\0\0",
        0,
        1,
        1,
        0,
        0,
        0,
        0,
    )
    return head + b"\0" * 84


def _ase_file(header: bytes, frames: list[bytes]) -> bytes:
    body = header + b"".join(frames)
    return struct.pack("<I", len(body)) + body[4:]


def _ase_layer(name: str, *, kind: int = 0, tileset: int | None = None) -> bytes:
    body = struct.pack("<HHHHHHB3s", 1 | 2, kind, 0, 0, 0, 0, 255, b"\0\0\0") + _ase_string(
        name
    )
    if tileset is not None:
        body += struct.pack("<I", tileset)
    return _ase_chunk(0x2004, body)


def _ase_tileset(tileset_id: int, tile_w: int, tile_h: int, tiles: list[bytes]) -> bytes:
    body = struct.pack("<IIIHHh", tileset_id, 2, len(tiles), tile_w, tile_h, 1) + b"\0" * 14
    body += _ase_string("tiles")
    raw = b"".join(tiles)
    compressed = zlib.compress(raw)
    body += struct.pack("<I", len(compressed)) + compressed
    return _ase_chunk(0x2023, body)


def _ase_tilemap_cel(layer: int, refs: np.ndarray) -> bytes:
    grid_h, grid_w = refs.shape
    compressed = zlib.compress(refs.astype("<u4").tobytes())
    body = (
        struct.pack("<HhhBHh5s", layer, 0, 0, 255, 3, 0, b"\0" * 5)
        + struct.pack("<HHH", grid_w, grid_h, 32)
        + struct.pack("<IIII", 0x1FFFFFFF, 0x80000000, 0x40000000, 0x20000000)
        + b"\0" * 10
        + compressed
    )
    return _ase_chunk(0x2005, body)


def _ase_linked_cel(layer: int, frame: int) -> bytes:
    body = struct.pack("<HhhBHh5s", layer, 0, 0, 255, 1, 0, b"\0" * 5)
    body += struct.pack("<H", frame)
    return _ase_chunk(0x2005, body)


def _ase_rgba(w: int, h: int, colour: tuple[int, int, int, int]) -> bytes:
    return bytes(colour) * (w * h)


def test_an_imported_tilemap_document_round_trips_through_ora_bit_exact(tmp_path: Path):
    """The end-to-end chain this wave's gate is about: an ``.aseprite`` with a
    tileset, a tilemap layer carrying a flipped ref, and a linked tilemap cel,
    opened through :func:`asein.document_from_aseprite`, saved to ``.ora`` and
    reopened -- refs, flags and the tileset binding all bit-exact, and the
    link still a link."""
    blank = _ase_rgba(2, 2, (0, 0, 0, 0))
    art = _ase_rgba(2, 2, (11, 22, 33, 255))
    ref_value = gid.compose(1, flip_h=True)
    data = _ase_file(
        _ase_header(2, 4, 2),
        [
            _ase_frame(
                [
                    _ase_tileset(5, 2, 2, [blank, art]),
                    _ase_layer("Tiles", kind=2, tileset=5),
                    _ase_tilemap_cel(0, np.array([[ref_value, 0]], dtype=np.uint32)),
                ]
            ),
            _ase_frame([_ase_linked_cel(0, 0)]),
        ],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert warnings == []
    _assert_synced(doc)

    path = tmp_path / "chain.aseprite.ora"
    ora.write_ora(doc, path)
    back = ora.read_ora(path)
    _assert_synced(back)

    assert len(back.tilesets) == 1
    orig_track = doc.anim.tracks[0]
    back_track = back.anim.tracks[0]
    assert back_track.tileset_uid == back.tilesets[0].uid

    orig_frames, back_frames = doc.anim.frames, back.anim.frames
    orig_first = doc.anim.cels[(orig_track.uid, orig_frames[0].uid)]
    orig_second = doc.anim.cels[(orig_track.uid, orig_frames[1].uid)]
    back_first = back.anim.cels[(back_track.uid, back_frames[0].uid)]
    back_second = back.anim.cels[(back_track.uid, back_frames[1].uid)]

    assert orig_second is orig_first, "the source .aseprite link must resolve to one object"
    assert back_second is back_first, "the link must survive the ORA round trip too"
    assert np.array_equal(back_first.refs, orig_first.refs)
    assert int(back_first.refs[0, 0]) == ref_value
    assert np.array_equal(back_first.pixels, orig_first.pixels)


def test_the_import_edit_export_chain_comes_back_through_aseout(tmp_path: Path):
    """Wave 5's extension of the chain above, all the way round: the same
    ``.aseprite`` opened, saved to ``.ora``, reopened, written back **out** as
    an ``.aseprite`` by :mod:`~warlock.studio.inker.aseout` and imported once
    more. Refs, flag bits, strip pixels and the track binding are identical at
    every stop, and the linked cel is still one object at the end of it.

    The ORA leg is deliberately in the middle rather than skipped: it is the
    format the editor actually saves in, so a tileset id or a refs plane that
    only survives an ``.aseprite``-to-``.aseprite`` hop would still be lost by
    the ordinary way a user works.
    """
    blank = _ase_rgba(2, 2, (0, 0, 0, 0))
    art = _ase_rgba(2, 2, (11, 22, 33, 255))
    ref_value = gid.compose(1, flip_h=True)
    data = _ase_file(
        _ase_header(2, 4, 2),
        [
            _ase_frame(
                [
                    _ase_tileset(5, 2, 2, [blank, art]),
                    _ase_layer("Tiles", kind=2, tileset=5),
                    _ase_tilemap_cel(0, np.array([[ref_value, 0]], dtype=np.uint32)),
                ]
            ),
            _ase_frame([_ase_linked_cel(0, 0)]),
        ],
    )
    opened, warnings = asein.document_from_aseprite(data)
    assert warnings == []

    path = tmp_path / "chain.ora"
    ora.write_ora(opened, path)
    reopened = ora.read_ora(path)

    exported = aseout.aseprite_bytes(reopened)
    back, out_warnings = asein.document_from_aseprite(exported)
    assert out_warnings == []
    _assert_synced(back)

    # The tileset, pixel for pixel -- the strip is what a lost id or a
    # re-encoded atlas would show up in first.
    assert len(back.tilesets) == 1
    assert np.array_equal(
        back.tilesets[0].tileset.pixels, opened.tilesets[0].tileset.pixels
    )
    assert (back.tilesets[0].tileset.tile_w, back.tilesets[0].tileset.tile_h) == (2, 2)

    track = back.anim.tracks[0]
    assert track.tileset_uid == back.tilesets[0].uid

    first = back.anim.cels[(track.uid, back.anim.frames[0].uid)]
    second = back.anim.cels[(track.uid, back.anim.frames[1].uid)]
    assert second is first, "the link must survive the whole chain"

    source = opened.anim.cels[(opened.anim.tracks[0].uid, opened.anim.frames[0].uid)]
    assert np.array_equal(first.refs, source.refs)
    assert int(first.refs[0, 0]) == ref_value
    assert np.array_equal(first.pixels, source.pixels)

    # And once more round: the exported bytes are a fixed point, so nothing in
    # the chain depends on a uid or a dictionary order that changed on the way.
    assert aseout.aseprite_bytes(back) == exported

# -- the ORA half of the end-to-end chain -------------------------------------


def test_reopened_document_still_supports_tile_edits(tmp_path: Path):
    """What comes back is a live document, not just correct static fields --
    the funnel keeps working on it. This is the ORA half of the chain; the
    other half (``.aseprite`` import onto this same model) is the round trip
    above."""
    doc = _still_doc()
    path = tmp_path / "chain.ora"
    ora.write_ora(doc, path)
    back = ora.read_ora(path)

    cel = back.stack[1]
    assert isinstance(cel, TilemapCel)
    assert back.place_tiles(cel.uid, (1, 1), np.array([[1]], dtype=np.uint32))
    _assert_synced(back)


# --- the diagonal mask at the import doors, and the two empties ---------------


def test_ora_read_strips_diagonal_flips_on_a_non_square_tileset(tmp_path: Path):
    """A file from before the refs door was sealed: the D bit over a
    non-square tileset degrades to the unturned placement -- left in, a later
    Auto-mode commit would read neighbour-cell pixels back into the atlas."""
    doc = Document.blank(8, 8)
    blank = np.zeros((2, 4, 4), dtype=np.uint8)
    red = np.zeros((2, 4, 4), dtype=np.uint8)
    red[:] = RED
    slot = doc.add_tileset(strip(np.stack([blank, red], axis=0)))  # 4x2 tiles
    cel = doc.add_tilemap_layer(slot.uid, name="Tiles")
    doc.place_tiles(cel.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    path = tmp_path / "wide.ora"
    ora.write_ora(doc, path)

    with zipfile.ZipFile(path) as zf:
        payload = json.loads(zf.read(ora.TILES_MEMBER))
        member = payload["cels"][0]["refs"]
        refs = np.frombuffer(zf.read(member), dtype="<u4").copy()
    refs[0] |= np.uint32(gid.FLIP_D)
    _rewrite_member(path, member, refs.tobytes())

    back = ora.read_ora(path)
    cell = next(layer for layer in back.stack if isinstance(layer, TilemapCel))
    assert int(cell.refs.flat[0]) == 1, "the D bit stripped, the id kept"
    assert not (np.asarray(cell.refs) & np.uint32(gid.FLIP_D)).any()
    _assert_synced(back)


def test_a_corrupt_tileset_strip_drops_structure_and_keeps_pixels(tmp_path: Path):
    """A tileset member holding non-image bytes is Pillow's
    ``UnidentifiedImageError`` -- an ``OSError`` -- and used to crash the open
    instead of costing the member, against the reader's stated contract."""
    doc = _still_doc()
    path = tmp_path / "strip.ora"
    ora.write_ora(doc, path)
    with zipfile.ZipFile(path) as zf:
        member = json.loads(zf.read(ora.TILES_MEMBER))["tilesets"][0]["data"]
    _rewrite_member(path, member, b"this is not a png")

    back = ora.read_ora(path)
    assert back.tilesets == []
    assert not any(isinstance(layer, TilemapCel) for layer in back.stack)
    doc.invalidate_all()  # ``_still_doc`` writes Ink's pixels directly
    assert np.array_equal(back.composite, doc.composite)


def test_a_0xffffffff_cell_imports_as_empty_with_a_warning():
    """The format's other empty: a tileset without the "tile ID 0 is empty"
    flag stores erased cells as 0xFFFFFFFF, which the mask arithmetic would
    otherwise read as a huge id wearing every flag."""
    blank = _ase_rgba(2, 2, (0, 0, 0, 0))
    art = _ase_rgba(2, 2, (11, 22, 33, 255))
    data = _ase_file(
        _ase_header(1, 4, 2),
        [
            _ase_frame(
                [
                    _ase_tileset(5, 2, 2, [blank, art]),
                    _ase_layer("Tiles", kind=2, tileset=5),
                    _ase_tilemap_cel(0, np.array([[1, 0xFFFFFFFF]], dtype=np.uint32)),
                ]
            )
        ],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert any("0xffffffff" in warning for warning in warnings)
    cel = next(
        layer
        for layer in (doc.stack if doc.anim is None else doc.anim.unique_cel_layers())
        if isinstance(layer, TilemapCel)
    )
    assert int(cel.refs[0, 0]) == 1
    assert int(cel.refs[0, 1]) == 0
    _assert_synced(doc)


def test_import_drops_diagonal_flips_on_a_non_square_tileset():
    blank = _ase_rgba(4, 2, (0, 0, 0, 0))
    art = _ase_rgba(4, 2, (11, 22, 33, 255))
    data = _ase_file(
        _ase_header(1, 8, 2),
        [
            _ase_frame(
                [
                    _ase_tileset(5, 4, 2, [blank, art]),
                    _ase_layer("Tiles", kind=2, tileset=5),
                    _ase_tilemap_cel(
                        0,
                        np.array(
                            [[gid.compose(1, flip_d=True), 1]], dtype=np.uint32
                        ),
                    ),
                ]
            )
        ],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert any("diagonal" in warning for warning in warnings)
    cel = next(
        layer
        for layer in (doc.stack if doc.anim is None else doc.anim.unique_cel_layers())
        if isinstance(layer, TilemapCel)
    )
    assert int(cel.refs[0, 0]) == 1, "D dropped, the id kept"
    assert int(cel.refs[0, 1]) == 1
    _assert_synced(doc)


def _tileset_chunk_flags(data: bytes) -> list[int]:
    """Every 0x2023 chunk's flags word, walked off the raw bytes."""
    out: list[int] = []
    at = 128  # the header's own size
    while at < len(data):
        frame_size, magic, _old, _dur, _pad, chunk_count = struct.unpack_from(
            "<IHHHHI", data, at
        )
        assert magic == 0xF1FA
        chunk_at = at + 16
        for _ in range(chunk_count):
            size, kind = struct.unpack_from("<IH", data, chunk_at)
            if kind == 0x2023:
                _tileset_id, flags = struct.unpack_from("<II", data, chunk_at + 6)
                out.append(flags)
            chunk_at += size
        at += frame_size
    return out


def test_the_written_tileset_chunk_declares_tile_zero_empty():
    """Flag 4 says what ``tiles.py`` already promises -- gid 0 *is* the blank
    tile. Without it real Aseprite treats the file as the pre-release layout
    whose empty cell is 0xFFFFFFFF, and a re-save there could hand
    ``_remap_tile_refs`` that value as a tile id."""
    data = aseout.aseprite_bytes(_still_doc())
    flags = _tileset_chunk_flags(data)
    assert flags, "the document writes at least one tileset chunk"
    assert all(word == (2 | 4) for word in flags)
