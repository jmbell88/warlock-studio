"""The headless tile model: strips, refs and materialization.

Wave 3 chunk 3.2a. ``tiles.py`` is deliberately thin over ``tilegrid`` -- the
shared leaf owns the gid word and the sliced-atlas type, and this module only
adds what a *cel* needs on top of it: a mutable holder for the frozen
``Tileset`` (:class:`TilesetSlot`), a ``Layer`` subclass whose picture is
derived from a refs plane (:class:`TilemapCel`), and the pure functions that
build/edit a vertical-strip atlas and turn refs back into pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import composite as cp
from warlock.studio.inker.anim_edits import charged, pixel_bytes
from warlock.studio.inker.animation import TRACK_PROPS, Track
from warlock.studio.inker.document import Document
from warlock.studio.inker.tiles import (
    TilemapCel,
    TilesetSlot,
    blank_strip,
    content_key,
    grid_shape,
    grow,
    materialize,
    shrink,
    strip,
    with_tiles,
)
from warlock.studio.tilegrid import gid

RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)
YELLOW = (255, 255, 0, 255)


def _doc(width: int = 16, height: int = 16) -> Document:
    return Document.blank(width, height)


def _tile(colour: tuple[int, int, int, int], w: int = 2, h: int = 2) -> np.ndarray:
    tile = np.zeros((h, w, 4), dtype=np.uint8)
    tile[..., 0] = colour[0]
    tile[..., 1] = colour[1]
    tile[..., 2] = colour[2]
    tile[..., 3] = colour[3]
    return tile


def _blank_tile(w: int = 2, h: int = 2) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


def _corner_tile() -> np.ndarray:
    """A 2x2 tile with four distinct corners -- the trivial-stabilizer motif
    that makes all eight square symmetries produce distinct pictures."""
    tile = np.zeros((2, 2, 4), dtype=np.uint8)
    tile[0, 0] = (1, 0, 0, 255)
    tile[0, 1] = (0, 1, 0, 255)
    tile[1, 0] = (0, 0, 1, 255)
    tile[1, 1] = (255, 255, 0, 255)
    return tile


def _slot(tile_w: int = 4, tile_h: int = 4) -> TilesetSlot:
    return TilesetSlot(tileset=blank_strip(tile_w, tile_h))


# -- TilesetSlot --------------------------------------------------------------


def test_tileset_slot_mints_its_own_uid():
    slot1 = _slot()
    slot2 = _slot()
    assert slot1.uid != slot2.uid


# -- strip / blank_strip geometry ---------------------------------------------


def test_blank_strip_is_one_transparent_tile():
    ts = blank_strip(4, 3)
    assert ts.tile_w == 4
    assert ts.tile_h == 3
    assert ts.columns == 1
    assert ts.rows == 1
    assert ts.tile_count == 1
    assert not ts.pixels.any()


def test_strip_builds_a_single_column_tileset_with_exact_tile_rects():
    stack = np.stack([_blank_tile(), _tile(RED), _tile(GREEN)], axis=0)
    ts = strip(stack)
    assert ts.columns == 1
    assert ts.rows == 3
    assert ts.tile_count == 3
    assert ts.tile_rect(0) == (0, 0, 2, 2)
    assert ts.tile_rect(1) == (0, 2, 2, 2)
    assert ts.tile_rect(2) == (0, 4, 2, 2)
    assert np.array_equal(ts.tile_pixels(1), _tile(RED))
    assert np.array_equal(ts.tile_pixels(2), _tile(GREEN))


def test_strip_rejects_a_non_blank_first_tile():
    stack = np.stack([_tile(RED), _blank_tile()], axis=0)
    with pytest.raises(ValueError):
        strip(stack)


# -- grow / shrink / with_tiles ------------------------------------------------


def test_grow_appends_and_is_a_frozen_replace():
    ts = blank_strip(2, 2)
    before_pixels_id = id(ts.pixels)
    added = np.stack([_tile(RED), _tile(GREEN)], axis=0)
    grown = grow(ts, added)

    assert grown.tile_count == 3
    assert id(grown.pixels) != before_pixels_id, "a frozen-replace is a new identity"
    assert np.array_equal(grown.tile_pixels(1), _tile(RED))
    assert np.array_equal(grown.tile_pixels(2), _tile(GREEN))
    # The original is untouched -- growing never mutates in place.
    assert ts.tile_count == 1


def test_shrink_undoes_grow():
    ts = blank_strip(2, 2)
    grown = grow(ts, np.stack([_tile(RED), _tile(GREEN)], axis=0))

    shrunk = shrink(grown, 1)
    assert shrunk.tile_count == 1
    assert id(shrunk.pixels) != id(grown.pixels)
    assert np.array_equal(shrunk.pixels, ts.pixels)


def test_shrink_refuses_to_drop_the_blank_tile():
    ts = blank_strip(2, 2)
    with pytest.raises(ValueError):
        shrink(ts, 0)


def test_with_tiles_edits_in_place_as_a_frozen_replace():
    ts = grow(blank_strip(2, 2), np.stack([_tile(RED), _tile(GREEN)], axis=0))
    before_id = id(ts.pixels)

    edited = with_tiles(ts, [(1, _tile(BLUE))])
    assert id(edited.pixels) != before_id
    assert np.array_equal(edited.tile_pixels(1), _tile(BLUE))
    # Tile 2 is untouched by an edit that only named tile 1.
    assert np.array_equal(edited.tile_pixels(2), _tile(GREEN))
    # The input tileset is unaffected -- with_tiles never mutates in place.
    assert np.array_equal(ts.tile_pixels(1), _tile(RED))


# -- grid_shape / content_key ---------------------------------------------------


def test_grid_shape_ceil_divides_height_then_width():
    # width=10 -> ceil(10/4)=3 columns; height=6 -> ceil(6/4)=2 rows.
    assert grid_shape((10, 6), 4, 4) == (2, 3)
    assert grid_shape((8, 8), 4, 4) == (2, 2)


def test_content_key_is_the_tile_bytes():
    a = _tile(RED)
    b = _tile(RED).copy()
    c = _tile(GREEN)
    assert content_key(a) == content_key(b)
    assert content_key(a) != content_key(c)
    assert content_key(a) == a.tobytes()


# -- materialize ----------------------------------------------------------------


def test_materialize_places_tiles_at_grid_positions():
    stack = np.stack([_blank_tile(), _tile(RED), _tile(GREEN)], axis=0)
    ts = strip(stack)
    refs = np.array([[1, 2]], dtype=gid.DTYPE)  # one row, two columns
    canvas = materialize(refs, ts, (4, 2))

    expected = np.zeros((2, 4, 4), dtype=np.uint8)
    expected[:, 0:2] = _tile(RED)
    expected[:, 2:4] = _tile(GREEN)
    assert np.array_equal(canvas, expected)


def test_materialize_treats_ref_zero_and_tile_zero_identically():
    stack = np.stack([_blank_tile(), _tile(RED)], axis=0)
    ts = strip(stack)
    refs = np.array([[0]], dtype=gid.DTYPE)
    canvas = materialize(refs, ts, (2, 2))
    assert np.array_equal(canvas, _blank_tile())


def test_materialize_treats_an_out_of_range_local_id_as_blank():
    stack = np.stack([_blank_tile(), _tile(RED)], axis=0)
    ts = strip(stack)
    # local id 9 is outside this tileset (0..1) -- must not raise.
    refs = np.array([[9]], dtype=gid.DTYPE)
    canvas = materialize(refs, ts, (2, 2))
    assert np.array_equal(canvas, _blank_tile())


def test_materialize_crops_at_the_canvas_edge():
    stack = np.stack([_blank_tile(4, 4), _tile(BLUE, 4, 4)], axis=0)
    ts = strip(stack)
    refs = np.full((2, 2), 1, dtype=gid.DTYPE)
    # A 6x6 canvas over a 4x4 tile grid: the grid is 2x2 (ceil(6/4)), and the
    # second row/column of tiles is cut to 2 pixels.
    canvas = materialize(refs, ts, (6, 6))
    assert canvas.shape == (6, 6, 4)
    expected = np.broadcast_to(np.array(BLUE, dtype=np.uint8), (6, 6, 4))
    assert np.array_equal(canvas, expected)


def _expected_corner(flip_h: bool, flip_v: bool, flip_d: bool) -> np.ndarray:
    """Hand-computed transpose-then-mirror, independent of ``materialize``."""
    tile = _corner_tile()
    if flip_d:
        tile = np.transpose(tile, (1, 0, 2))
    if flip_h:
        tile = tile[:, ::-1]
    if flip_v:
        tile = tile[::-1, :]
    return tile


@pytest.mark.parametrize("flip_h", [False, True])
@pytest.mark.parametrize("flip_v", [False, True])
@pytest.mark.parametrize("flip_d", [False, True])
def test_materialize_applies_all_eight_symmetries_transpose_then_mirror(
    flip_h, flip_v, flip_d
):
    stack = np.stack([_blank_tile(), _corner_tile()], axis=0)
    ts = strip(stack)
    ref = gid.compose(1, flip_h=flip_h, flip_v=flip_v, flip_d=flip_d)
    refs = np.array([[ref]], dtype=gid.DTYPE)
    canvas = materialize(refs, ts, (2, 2))
    assert np.array_equal(canvas, _expected_corner(flip_h, flip_v, flip_d))


# -- TilemapCel -----------------------------------------------------------------


def test_tilemap_cel_requires_a_refs_plane():
    with pytest.raises(ValueError):
        TilemapCel(pixels=cp.empty(4, 4))


def test_tilemap_cel_requires_uint32_two_dimensional_refs():
    with pytest.raises(ValueError):
        TilemapCel(pixels=cp.empty(4, 4), refs=np.zeros((2, 2), dtype=np.int32))
    with pytest.raises(ValueError):
        TilemapCel(pixels=cp.empty(4, 4), refs=np.zeros((2, 2, 1), dtype=gid.DTYPE))


def test_tilemap_cel_plane_bytes_includes_refs():
    cel = TilemapCel(pixels=cp.empty(4, 4), refs=np.zeros((2, 2), dtype=gid.DTYPE))
    assert cel.plane_bytes == cel.pixels.nbytes + cel.refs.nbytes


def test_tilemap_cel_copy_deep_copies_refs_and_preserves_tileset_uid():
    cel = TilemapCel(
        pixels=cp.empty(4, 4),
        refs=np.zeros((2, 2), dtype=gid.DTYPE),
        tileset_uid=7,
    )
    dup = cel.copy()
    assert isinstance(dup, TilemapCel)
    assert dup.refs is not cel.refs
    assert np.array_equal(dup.refs, cel.refs)
    assert dup.tileset_uid == 7
    assert dup.uid != cel.uid

    dup_with_uid = cel.copy(uid=99)
    assert dup_with_uid.uid == 99


# -- Track.tileset_uid ------------------------------------------------------


def test_tileset_uid_is_not_a_copied_down_track_prop():
    assert "tileset_uid" not in TRACK_PROPS
    assert "tileset_uid" not in Track().props()


def test_track_default_tileset_uid_is_none():
    assert Track().tileset_uid is None


# -- Document.tilesets / tile_behavior --------------------------------------


def test_document_starts_with_no_tilesets_and_manual_behavior():
    doc = _doc()
    assert doc.tilesets == []
    assert doc.tile_behavior == "manual"


# -- _ensure_cel_for autovivifies a TilemapCel -------------------------------


def test_track_with_tileset_uid_autovivifies_a_tilemap_cel_keeping_placeholder_uid():
    doc = _doc()
    anim = doc.ensure_animation()
    slot = _slot(4, 4)
    doc.tilesets.append(slot)
    anim.tracks[0].tileset_uid = slot.uid

    doc.add_frame()  # a blank second frame: track 0's slot is a placeholder
    placeholder_uid = doc.stack[0].uid

    doc._ensure_cel_for(placeholder_uid)
    cel = doc.stack[0]

    assert isinstance(cel, TilemapCel)
    assert cel.uid == placeholder_uid
    assert cel.tileset_uid == slot.uid
    assert cel.refs.dtype == gid.DTYPE
    assert cel.refs.shape == grid_shape(doc.size, 4, 4)
    assert not cel.refs.any()


def test_a_track_without_a_tileset_uid_still_autovivifies_an_ordinary_layer():
    doc = _doc()
    doc.ensure_animation()
    doc.add_frame()
    doc._ensure_cel_for(doc.stack[0].uid)
    assert not isinstance(doc.stack[0], TilemapCel)


def test_a_dangling_tileset_uid_autovivifies_nothing():
    """A track whose ``tileset_uid`` names no slot in ``self.tilesets`` is an
    impossible state, not a normal one to paper over -- ``_ensure_cel_for``
    refuses exactly as it does for its other guard conditions: no cel is
    invented, the slot stays a placeholder, and nothing is queued to commit."""
    doc = _doc()
    anim = doc.ensure_animation()
    anim.tracks[0].tileset_uid = 999999  # no matching TilesetSlot was ever added

    doc.add_frame()
    placeholder = doc.stack[0]
    assert doc.anim.is_placeholder(placeholder)

    doc._ensure_cel_for(placeholder.uid)

    assert doc.stack[0] is placeholder
    assert doc.anim.is_placeholder(doc.stack[0])
    assert (anim.tracks[0].uid, anim.frame.uid) not in anim.cels
    assert doc._pending_cels == []


def test_linked_tilemap_cel_shares_refs():
    doc = _doc()
    anim = doc.ensure_animation()
    slot = _slot(4, 4)
    doc.tilesets.append(slot)
    anim.tracks[0].tileset_uid = slot.uid

    doc.add_frame()
    doc._ensure_cel_for(doc.stack[0].uid)
    cel = doc.stack[0]
    assert isinstance(cel, TilemapCel)

    doc.add_frame(link=True)  # frame 3 links frame 2's cels
    frame3 = anim.frame
    linked = anim.cels[(anim.tracks[0].uid, frame3.uid)]
    assert linked is cel
    assert linked.refs is cel.refs


def test_charged_counts_a_tilemap_cels_refs_via_plane_bytes():
    doc = _doc()
    anim = doc.ensure_animation()
    slot = _slot(4, 4)
    doc.tilesets.append(slot)
    anim.tracks[0].tileset_uid = slot.uid

    doc.add_frame()
    doc._ensure_cel_for(doc.stack[0].uid)
    cel = doc.stack[0]

    expected = cel.pixels.nbytes + cel.refs.nbytes
    assert pixel_bytes([cel]) == expected
    assert charged([cel], True) == expected
    assert charged([cel], False) == 0
