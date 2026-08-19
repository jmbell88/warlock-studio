"""Tests for chunk 3.3: the funnel's tilemap branch, Manual/Auto/Stack, the
raster<->tilemap conversions, and the door refusals every op that writes
``layer.pixels`` before a guard could fire now owes a tilemap layer.

``_assert_synced`` is the Wave 3 risk-item helper, copied from
``test_tile_edits.py`` (``tests/inker`` has no conftest): after every op in
this suite, every ``TilemapCel``'s ``pixels`` must equal
``materialize(refs, bound_tileset, size)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker.document import Document
from warlock.studio.inker.layers import Layer
from warlock.studio.inker.selection import FloatingBuffer, SelectionMask
from warlock.studio.inker.tiles import TilemapCel, materialize, strip
from warlock.studio.tilegrid import gid
from warlock.studio.undo import CompoundEdit

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
GREEN = (0, 255, 0, 255)
WHITE = (255, 255, 255, 255)


def _doc(width: int = 8, height: int = 8) -> Document:
    return Document.blank(width, height)


def _tile(colour: tuple[int, int, int, int], w: int = 4, h: int = 4) -> np.ndarray:
    tile = np.zeros((h, w, 4), dtype=np.uint8)
    tile[..., 0], tile[..., 1], tile[..., 2], tile[..., 3] = colour
    return tile


def _blank_tile(w: int = 4, h: int = 4) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


def _tileset(*tiles: np.ndarray):
    return strip(np.stack([_blank_tile(), *tiles], axis=0))


def _corner_tile() -> np.ndarray:
    """An asymmetric 4x4 tile: no square symmetry maps it onto itself."""
    tile = np.zeros((4, 4, 4), dtype=np.uint8)
    tile[0, 0] = WHITE
    tile[0, 1] = RED
    tile[1, 0] = GREEN
    return tile


def _assert_synced(doc: Document) -> None:
    """Every tilemap cel's pixels agree with ``materialize(refs, ts, size)``."""
    layers = doc.stack if doc.anim is None else doc.anim.unique_cel_layers()
    for layer in layers:
        if isinstance(layer, TilemapCel):
            slot = doc.tileset_slot(layer.tileset_uid)
            want = materialize(layer.refs, slot.tileset, layer.size)
            assert np.array_equal(layer.pixels, want), f"cel {layer.uid} drifted from its refs"


def _still_tilemap(doc: Document, *tiles: np.ndarray):
    slot = doc.add_tileset(_tileset(*tiles))
    cel = doc.add_tilemap_layer(slot.uid)
    return slot, cel


def _paint(doc: Document, rect: tuple[int, int, int, int], colour) -> bool:
    """Write a flat colour through the ordinary paint door, which ends at the
    funnel -- the whole point of the routing being *at* ``_commit_patch``."""
    x0, y0, x1, y1 = rect
    weight = np.ones((y1 - y0, x1 - x0), dtype=np.float32)
    return doc.write_colour(rect, colour, weight)


def _activate(doc: Document, layer) -> None:
    doc.stack.active_index = doc.stack.index_of(layer.uid)


def _setup(behavior: str, *tiles: np.ndarray):
    """A still 8x8 document, 4x4 tiles (a 2x2 grid), cell (0,0) holding tile 1."""
    doc = _doc()
    slot, cel = _still_tilemap(doc, *tiles)
    doc.place_tiles(cel.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    doc.tile_behavior = behavior
    _activate(doc, cel)
    doc.history.clear()
    return doc, slot, cel


# -- manual -------------------------------------------------------------------


def test_manual_mode_reverts_the_write_and_pushes_nothing():
    doc, slot, cel = _setup("manual", _tile(RED))

    assert _paint(doc, (0, 0, 2, 2), BLUE) is True
    # The tool wrote RGBA into the cel; manual mode puts it straight back.
    assert tuple(int(c) for c in cel.pixels[0, 0]) == RED
    assert not doc.history.can_undo
    assert slot.tileset.tile_count == 2
    assert np.array_equal(slot.tileset.tile_pixels(1), _tile(RED))
    _assert_synced(doc)


def test_an_unknown_behaviour_falls_back_to_manual():
    doc, slot, cel = _setup("sideways", _tile(RED))

    assert _paint(doc, (0, 0, 2, 2), BLUE) is True
    assert tuple(int(c) for c in cel.pixels[0, 0]) == RED
    assert not doc.history.can_undo
    _assert_synced(doc)


def test_manual_mode_discards_a_cel_the_write_autovivified():
    doc = _doc()
    anim = doc.ensure_animation()
    slot = doc.add_tileset(_tileset(_tile(RED)))
    anim.tracks[0].tileset_uid = slot.uid
    doc.add_frame()  # frame 2: track 0 is a placeholder
    doc.tile_behavior = "manual"
    doc.stack.active_index = 0
    doc.history.clear()

    assert _paint(doc, (0, 0, 4, 4), BLUE) is True
    assert doc.anim.is_placeholder(doc.stack[0])
    assert doc._pending_cels == []
    assert not doc.history.can_undo


# -- auto ---------------------------------------------------------------------


def test_auto_mode_edits_the_tile_in_place():
    doc, slot, cel = _setup("auto", _tile(RED))

    assert _paint(doc, (0, 0, 1, 1), BLUE) is True
    assert slot.tileset.tile_count == 2  # nothing appended
    assert tuple(int(c) for c in slot.tileset.tile_pixels(1)[0, 0]) == BLUE
    assert tuple(int(c) for c in slot.tileset.tile_pixels(1)[1, 1]) == RED
    assert int(cel.refs[0, 0]) == 1  # the ref did not move
    _assert_synced(doc)


def test_auto_mode_updates_every_placement_on_every_layer():
    doc = _doc()
    slot = doc.add_tileset(_tileset(_tile(RED)))
    cel_a = doc.add_tilemap_layer(slot.uid, name="A")
    cel_b = doc.add_tilemap_layer(slot.uid, name="B")
    doc.place_tiles(cel_a.uid, (0, 0), np.array([[1, 1]], dtype=np.uint32))
    doc.place_tiles(cel_b.uid, (1, 1), np.array([[1]], dtype=np.uint32))
    doc.tile_behavior = "auto"
    _activate(doc, cel_a)
    doc.history.clear()

    assert _paint(doc, (0, 0, 1, 1), BLUE) is True

    # The other placement in this same cel, and the placement on the other
    # layer, both show the edited tile.
    assert tuple(int(c) for c in cel_a.pixels[0, 4]) == BLUE
    assert tuple(int(c) for c in cel_b.pixels[4, 4]) == BLUE
    _assert_synced(doc)


def test_auto_mode_undo_reverts_the_tile_and_every_placement_in_one_step():
    doc = _doc()
    slot = doc.add_tileset(_tileset(_tile(RED)))
    cel_a = doc.add_tilemap_layer(slot.uid, name="A")
    cel_b = doc.add_tilemap_layer(slot.uid, name="B")
    doc.place_tiles(cel_a.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    doc.place_tiles(cel_b.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    doc.tile_behavior = "auto"
    _activate(doc, cel_a)
    doc.history.clear()

    assert _paint(doc, (0, 0, 1, 1), BLUE) is True
    assert len(doc.history) == 1

    doc.history.undo(doc)
    assert np.array_equal(slot.tileset.tile_pixels(1), _tile(RED))
    assert tuple(int(c) for c in cel_a.pixels[0, 0]) == RED
    assert tuple(int(c) for c in cel_b.pixels[0, 0]) == RED
    _assert_synced(doc)

    doc.history.redo(doc)
    assert tuple(int(c) for c in cel_a.pixels[0, 0]) == BLUE
    assert tuple(int(c) for c in cel_b.pixels[0, 0]) == BLUE
    _assert_synced(doc)


def test_auto_mode_dedups_onto_an_existing_tile_by_re_pointing():
    doc, slot, cel = _setup("auto", _tile(RED), _tile(BLUE))

    assert _paint(doc, (0, 0, 4, 4), BLUE) is True
    assert slot.tileset.tile_count == 3  # nothing appended
    # Tile 1 was NOT edited -- the ref moved to the tile that already matched.
    assert np.array_equal(slot.tileset.tile_pixels(1), _tile(RED))
    assert int(cel.refs[0, 0]) == 2
    _assert_synced(doc)


def test_auto_mode_appends_when_painting_an_empty_cell():
    """Tile 0 is the required-blank slot, so Auto cannot edit it in place --
    painting an empty cell mints a tile, as Aseprite does."""
    doc, slot, cel = _setup("auto", _tile(RED))

    assert _paint(doc, (4, 0, 8, 4), BLUE) is True
    assert slot.tileset.tile_count == 3
    assert np.array_equal(slot.tileset.tile_pixels(0), _blank_tile())
    assert int(cel.refs[0, 1]) == 2
    _assert_synced(doc)


# -- stack --------------------------------------------------------------------


def test_stack_mode_appends_and_re_points_leaving_the_original_untouched():
    doc, slot, cel = _setup("stack", _tile(RED))

    assert _paint(doc, (0, 0, 4, 4), BLUE) is True
    assert slot.tileset.tile_count == 3
    assert np.array_equal(slot.tileset.tile_pixels(1), _tile(RED))  # untouched
    assert np.array_equal(slot.tileset.tile_pixels(2), _tile(BLUE))
    assert int(cel.refs[0, 0]) == 2
    _assert_synced(doc)

    doc.history.undo(doc)
    assert slot.tileset.tile_count == 2
    assert int(cel.refs[0, 0]) == 1
    _assert_synced(doc)


def test_stack_mode_never_appends_identical_content_twice():
    doc, slot, cel = _setup("stack", _tile(RED))
    doc.place_tiles(cel.uid, (1, 0), np.array([[1]], dtype=np.uint32))
    doc.history.clear()

    # One gesture covering both cells: both resolve to the same new content.
    assert _paint(doc, (0, 0, 8, 4), BLUE) is True
    assert slot.tileset.tile_count == 3
    assert int(cel.refs[0, 0]) == 2
    assert int(cel.refs[0, 1]) == 2
    _assert_synced(doc)


def test_stack_mode_re_points_to_an_existing_tile_rather_than_appending():
    doc, slot, cel = _setup("stack", _tile(RED), _tile(BLUE))

    assert _paint(doc, (0, 0, 4, 4), BLUE) is True
    assert slot.tileset.tile_count == 3  # nothing appended
    assert int(cel.refs[0, 0]) == 2
    _assert_synced(doc)


# -- flipped placements -------------------------------------------------------


def test_paint_over_a_flip_h_placement_edits_the_canonical_tile():
    doc, slot, cel = _setup("auto", _corner_tile())
    doc.place_tiles(
        cel.uid, (0, 0), np.array([[1 | gid.FLIP_H]], dtype=np.uint32)
    )
    doc.history.clear()

    # Canvas (0,0) is the *mirrored* tile's top-left, which is the canonical
    # tile's top-RIGHT: column 3.
    assert _paint(doc, (0, 0, 1, 1), BLUE) is True
    tile = slot.tileset.tile_pixels(1)
    assert tuple(int(c) for c in tile[0, 3]) == BLUE
    assert tuple(int(c) for c in tile[0, 0]) == WHITE  # untouched
    assert int(cel.refs[0, 0]) == (1 | gid.FLIP_H)  # the flag survives
    _assert_synced(doc)


def test_paint_over_a_flip_d_placement_edits_the_canonical_tile():
    doc, slot, cel = _setup("auto", _corner_tile())
    doc.place_tiles(
        cel.uid, (0, 0), np.array([[1 | gid.FLIP_D]], dtype=np.uint32)
    )
    doc.history.clear()

    # FLIP_D transposes, so canvas (x=1, y=0) is the canonical tile's (x=0, y=1).
    assert _paint(doc, (1, 0, 2, 1), BLUE) is True
    tile = slot.tileset.tile_pixels(1)
    assert tuple(int(c) for c in tile[1, 0]) == BLUE
    assert tuple(int(c) for c in tile[0, 1]) == RED  # untouched
    assert int(cel.refs[0, 0]) == (1 | gid.FLIP_D)
    _assert_synced(doc)


def test_paint_over_a_flip_h_v_placement_edits_the_canonical_tile():
    doc, slot, cel = _setup("auto", _corner_tile())
    doc.place_tiles(
        cel.uid, (0, 0), np.array([[1 | gid.FLIP_H | gid.FLIP_V]], dtype=np.uint32)
    )
    doc.history.clear()

    assert _paint(doc, (0, 0, 1, 1), BLUE) is True
    tile = slot.tileset.tile_pixels(1)
    assert tuple(int(c) for c in tile[3, 3]) == BLUE
    _assert_synced(doc)


# -- multi-tile ----------------------------------------------------------------


def test_a_stroke_across_four_cells_edits_all_of_them_as_one_step():
    doc, slot, cel = _setup("stack", _tile(RED))
    doc.place_tiles(cel.uid, (0, 0), np.full((2, 2), 1, dtype=np.uint32))
    doc.history.clear()

    assert _paint(doc, (2, 2, 6, 6), BLUE) is True
    assert len(doc.history) == 1
    # Four cells, four distinct quadrant patterns -> four new tiles.
    assert slot.tileset.tile_count == 6
    assert set(int(r) for r in cel.refs.ravel()) == {2, 3, 4, 5}
    _assert_synced(doc)

    doc.history.undo(doc)
    assert slot.tileset.tile_count == 2
    assert set(int(r) for r in cel.refs.ravel()) == {1}
    _assert_synced(doc)


def test_a_write_that_changes_nothing_pushes_nothing():
    doc, slot, cel = _setup("auto", _tile(RED))

    assert _paint(doc, (0, 0, 4, 4), RED) is True
    assert not doc.history.can_undo
    assert slot.tileset.tile_count == 2
    _assert_synced(doc)


# -- conversions ---------------------------------------------------------------


def _painted_doc() -> tuple[Document, Layer]:
    doc = _doc()
    layer = doc.stack.active
    layer.pixels[0:4, 0:4] = np.array(RED, dtype=np.uint8)
    layer.pixels[0:2, 4:6] = np.array(BLUE, dtype=np.uint8)
    doc.history.clear()
    return doc, layer


def test_convert_layer_to_tilemap_then_to_raster_round_trips_bit_exact():
    doc, layer = _painted_doc()
    original = layer.pixels.copy()
    uid = layer.uid

    assert doc.convert_layer_to_tilemap(uid, 4, 4) is True
    cel = doc.stack.by_uid(uid)
    assert isinstance(cel, TilemapCel)
    assert np.array_equal(cel.pixels, original)
    _assert_synced(doc)

    assert doc.convert_layer_to_raster(uid) is True
    back = doc.stack.by_uid(uid)
    assert type(back) is Layer
    assert back.uid == uid
    assert np.array_equal(back.pixels, original)


def test_convert_layer_to_tilemap_dedups_and_maps_transparent_to_zero():
    doc = _doc()
    layer = doc.stack.active
    # Two identical 4x4 cells on the top row, nothing on the bottom.
    layer.pixels[0:4, 0:4] = np.array(RED, dtype=np.uint8)
    layer.pixels[0:4, 4:8] = np.array(RED, dtype=np.uint8)
    doc.history.clear()

    assert doc.convert_layer_to_tilemap(layer.uid, 4, 4) is True
    cel = doc.stack.by_uid(layer.uid)
    slot = doc.tileset_slot(cel.tileset_uid)
    assert slot.tileset.tile_count == 2  # blank + the one distinct cell
    assert np.array_equal(cel.refs, np.array([[1, 1], [0, 0]], dtype=np.uint32))
    _assert_synced(doc)


def test_convert_layer_to_tilemap_is_one_undoable_step():
    doc, layer = _painted_doc()
    original = layer.pixels.copy()

    assert doc.convert_layer_to_tilemap(layer.uid, 4, 4) is True
    assert len(doc.history) == 1
    assert isinstance(doc.history.top, CompoundEdit)

    doc.history.undo(doc)
    back = doc.stack.by_uid(layer.uid)
    assert back is layer
    assert doc.tilesets == []
    assert np.array_equal(back.pixels, original)

    doc.history.redo(doc)
    assert isinstance(doc.stack.by_uid(layer.uid), TilemapCel)
    _assert_synced(doc)


def test_convert_layer_to_tilemap_pads_a_canvas_that_is_not_tile_divisible():
    doc = _doc(6, 6)  # a 6x6 canvas at 4x4 tiles -> a 2x2 ceil-covering grid
    layer = doc.stack.active
    layer.pixels[:, :] = np.array(RED, dtype=np.uint8)
    original = layer.pixels.copy()
    doc.history.clear()

    assert doc.convert_layer_to_tilemap(layer.uid, 4, 4) is True
    cel = doc.stack.by_uid(layer.uid)
    assert cel.refs.shape == (2, 2)
    assert np.array_equal(cel.pixels, original)
    _assert_synced(doc)


def test_convert_layer_to_tilemap_on_an_animated_track_replaces_every_cel():
    doc = _doc()
    doc.ensure_animation()
    doc.stack.active.pixels[0:4, 0:4] = np.array(RED, dtype=np.uint8)
    doc.add_frame()
    doc._ensure_cel_for(doc.stack[0].uid)
    doc.stack[0].pixels[4:8, 4:8] = np.array(BLUE, dtype=np.uint8)
    uid = doc.stack[0].uid
    doc.history.clear()

    assert doc.convert_layer_to_tilemap(uid, 4, 4) is True
    track = doc.anim.tracks[0]
    assert track.tileset_uid is not None
    cels = [
        doc.anim.cels[(track.uid, frame.uid)]
        for frame in doc.anim.frames
        if (track.uid, frame.uid) in doc.anim.cels
    ]
    assert len(cels) == 2
    assert all(isinstance(cel, TilemapCel) for cel in cels)
    _assert_synced(doc)

    doc.history.undo(doc)
    assert doc.anim.tracks[0].tileset_uid is None
    assert doc.tilesets == []


def test_convert_layer_to_tilemap_keeps_a_linked_cel_linked():
    doc = _doc()
    doc.ensure_animation()
    doc.stack.active.pixels[0:4, 0:4] = np.array(RED, dtype=np.uint8)
    doc.add_frame(link=True)
    track = doc.anim.tracks[0]
    uid = doc.stack[0].uid
    doc.history.clear()

    assert doc.convert_layer_to_tilemap(uid, 4, 4) is True
    a = doc.anim.cels[(track.uid, doc.anim.frames[0].uid)]
    b = doc.anim.cels[(track.uid, doc.anim.frames[1].uid)]
    assert a is b
    assert isinstance(a, TilemapCel)
    _assert_synced(doc)


def test_convert_layer_to_raster_returns_false_for_a_plain_layer():
    doc, layer = _painted_doc()
    assert doc.convert_layer_to_raster(layer.uid) is False


def test_convert_layer_to_tilemap_returns_false_for_a_tilemap_layer():
    doc, slot, cel = _setup("manual", _tile(RED))
    assert doc.convert_layer_to_tilemap(cel.uid, 4, 4) is False


def test_convert_layer_to_tilemap_refuses_a_locked_layer():
    doc, layer = _painted_doc()
    layer.locked = True
    assert doc.convert_layer_to_tilemap(layer.uid, 4, 4) is False


def test_a_converted_layer_paints_through_the_tilemap_branch():
    doc, layer = _painted_doc()
    assert doc.convert_layer_to_tilemap(layer.uid, 4, 4) is True
    doc.tile_behavior = "stack"
    _activate(doc, doc.stack.by_uid(layer.uid))
    doc.history.clear()

    assert _paint(doc, (0, 0, 1, 1), GREEN) is True
    _assert_synced(doc)


# -- indexed interplay ---------------------------------------------------------


def test_an_indexed_document_routes_a_tilemap_stroke_through_the_tile_branch():
    doc = _doc()
    doc.convert_to_indexed([(0, 0, 0, 0), RED, BLUE], transparent=0)
    slot, cel = _still_tilemap(doc, _tile(RED))
    doc.place_tiles(cel.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    doc.tile_behavior = "auto"
    _activate(doc, cel)
    doc.history.clear()

    assert cel.indices is None  # the recorded divergence: the strip stays RGBA
    assert _paint(doc, (0, 0, 1, 1), BLUE) is True
    assert cel.indices is None
    assert tuple(int(c) for c in slot.tileset.tile_pixels(1)[0, 0]) == BLUE
    doc.check_materialized()
    _assert_synced(doc)


# -- the other two colour modes ------------------------------------------------


def test_a_grayscale_document_flattens_the_canonical_tile_to_luma():
    """The funnel's grayscale branch sits *after* the tilemap divert, so it is
    applied here or nowhere: without it a blue stroke put ``[0,0,255,255]`` into
    the tileset and every placement of that tile drew it in colour."""
    doc = _doc()
    doc.convert_to_grayscale()
    slot, cel = _still_tilemap(doc, _tile(RED))
    doc.place_tiles(cel.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    doc.tile_behavior = "auto"
    _activate(doc, cel)
    doc.history.clear()

    assert _paint(doc, (0, 0, 1, 1), BLUE) is True
    r, g, b, a = (int(c) for c in slot.tileset.tile_pixels(1)[0, 0])
    assert r == g == b
    assert (r, g, b) != (0, 0, 255)
    assert a == 255
    _assert_synced(doc)

    # Still one step, and the undo lands on the pre-stroke tile.
    assert len(doc.history) == 1
    doc.history.undo(doc)
    assert np.array_equal(slot.tileset.tile_pixels(1), _tile(RED))
    _assert_synced(doc)


def test_a_palette_constrained_document_snaps_the_canonical_tile():
    """Reachable exactly as the review found it: an indexed document keeps its
    table through ``convert_to_rgb`` (which stays legal beside a tilemap layer),
    so the palette constraint outlives the index planes."""
    doc = _doc()
    doc.convert_to_indexed([(0, 0, 0, 0), RED, GREEN], transparent=0)
    slot, cel = _still_tilemap(doc, _tile(RED))
    assert doc.convert_to_rgb() is True
    assert doc.palette  # the table survived the mode change
    doc.place_tiles(cel.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    doc.tile_behavior = "stack"
    _activate(doc, cel)
    doc.history.clear()

    assert _paint(doc, (0, 0, 1, 1), BLUE) is True
    painted = tuple(int(c) for c in slot.tileset.tile_pixels(2)[0, 0])
    assert painted != BLUE  # off-palette blue never reaches the atlas
    # ``snap`` matches on colour and rides alpha through unchanged, so the
    # membership test is over the table's RGB and not its RGBA.
    assert painted[:3] in {tuple(c)[:3] for c in doc.palette}
    _assert_synced(doc)

    assert len(doc.history) == 1
    doc.history.undo(doc)
    assert slot.tileset.tile_count == 2
    assert int(cel.refs[0, 0]) == 1
    _assert_synced(doc)


def test_a_palette_snap_leaves_the_rest_of_the_tile_alone():
    """The constraint is scoped to the written rect, as the funnel's is: a tile
    holding off-palette content an import brought in is not rewritten wholesale
    because one pixel of one placement was touched."""
    doc = _doc()
    off_palette = _tile(RED)
    off_palette[3, 3] = (7, 9, 11, 255)  # on no table
    slot = doc.add_tileset(_tileset(off_palette))
    cel = doc.add_tilemap_layer(slot.uid)
    doc.palette = [(0, 0, 0, 0), RED, GREEN]
    doc.place_tiles(cel.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    doc.tile_behavior = "auto"
    _activate(doc, cel)
    doc.history.clear()

    assert _paint(doc, (0, 0, 1, 1), GREEN) is True
    tile = slot.tileset.tile_pixels(1)
    assert tuple(int(c) for c in tile[0, 0]) == GREEN
    assert tuple(int(c) for c in tile[3, 3]) == (7, 9, 11, 255)
    _assert_synced(doc)


# -- door refusals -------------------------------------------------------------


def test_merge_down_refuses_a_tilemap_layer_before_touching_the_lower_one():
    doc = _doc()
    lower = doc.stack.active
    lower.pixels[:, :] = np.array(RED, dtype=np.uint8)
    before = lower.pixels.copy()
    _still_tilemap(doc, _tile(BLUE))

    with pytest.raises(ValueError, match="tilemap"):
        doc.merge_down(doc.stack.index_of(doc.stack[1].uid))
    assert np.array_equal(lower.pixels, before)
    assert len(doc.stack) == 2
    _assert_synced(doc)


def test_merge_down_refuses_when_the_tilemap_is_the_lower_layer():
    doc = _doc()
    slot, cel = _still_tilemap(doc, _tile(RED))
    doc.add_layer(name="Above")
    before = cel.pixels.copy()

    with pytest.raises(ValueError, match="tilemap"):
        doc.merge_down(doc.stack.index_of(doc.stack[-1].uid))
    assert np.array_equal(cel.pixels, before)
    _assert_synced(doc)


def test_apply_matte_refuses_a_document_holding_a_tilemap_layer():
    doc = _doc()
    slot, cel = _still_tilemap(doc, _tile(RED))
    doc.place_tiles(cel.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    before = cel.pixels.copy()
    other = doc.stack[0].pixels.copy()

    with pytest.raises(ValueError, match="tilemap"):
        doc.apply_matte(np.zeros((8, 8), dtype=np.uint8))
    assert np.array_equal(cel.pixels, before)
    assert np.array_equal(doc.stack[0].pixels, other)
    _assert_synced(doc)


def test_convert_to_indexed_refuses_a_document_holding_a_tilemap_layer():
    doc = _doc()
    slot, cel = _still_tilemap(doc, _tile(RED))
    doc.place_tiles(cel.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    before = cel.pixels.copy()

    with pytest.raises(ValueError, match="tilemap"):
        doc.convert_to_indexed([(0, 0, 0, 0), RED])
    assert doc.color_mode == "rgb"
    assert np.array_equal(cel.pixels, before)
    _assert_synced(doc)


def test_convert_to_grayscale_refuses_a_document_holding_a_tilemap_layer():
    doc = _doc()
    slot, cel = _still_tilemap(doc, _tile(RED))
    doc.place_tiles(cel.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    before = cel.pixels.copy()

    with pytest.raises(ValueError, match="tilemap"):
        doc.convert_to_grayscale()
    assert doc.color_mode == "rgb"
    assert np.array_equal(cel.pixels, before)


def test_convert_to_palette_refuses_a_document_holding_a_tilemap_layer():
    doc = _doc()
    slot, cel = _still_tilemap(doc, _tile(RED))
    doc.place_tiles(cel.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    before = cel.pixels.copy()

    with pytest.raises(ValueError, match="tilemap"):
        doc.convert_to_palette([(0, 0, 0, 0), BLUE])
    assert np.array_equal(cel.pixels, before)


def test_convert_to_rgb_is_still_allowed_beside_a_tilemap_layer():
    """The pixel-rewriting conversions are refused; leaving indexed mode
    rewrites nothing, so it stays legal -- a document that could not leave the
    mode it was in when the layer was added would be wedged."""
    doc = _doc()
    doc.convert_to_indexed([(0, 0, 0, 0), RED])
    slot, cel = _still_tilemap(doc, _tile(RED))
    assert doc.convert_to_rgb() is True
    _assert_synced(doc)


def test_lift_refuses_a_tilemap_layer_before_cutting_it():
    doc, slot, cel = _setup("auto", _tile(RED))
    before = cel.pixels.copy()
    doc.mask = SelectionMask.from_rect(doc.size, (0, 0, 4, 4))

    with pytest.raises(ValueError, match="tilemap"):
        doc.lift()
    assert np.array_equal(cel.pixels, before)
    assert doc.floating is None
    _assert_synced(doc)


def test_delete_selection_refuses_a_tilemap_layer_before_cutting_it():
    doc, slot, cel = _setup("auto", _tile(RED))
    before = cel.pixels.copy()
    doc.mask = SelectionMask.from_rect(doc.size, (0, 0, 4, 4))

    with pytest.raises(ValueError, match="tilemap"):
        doc.delete_selection()
    assert np.array_equal(cel.pixels, before)
    _assert_synced(doc)


def test_commit_floating_range_refuses_a_tilemap_cel_in_the_range():
    doc = _doc()
    anim = doc.ensure_animation()
    doc.stack.active.pixels[0:4, 0:4] = np.array(RED, dtype=np.uint8)
    slot = doc.add_tileset(_tileset(_tile(BLUE)))
    doc.add_tilemap_layer(slot.uid, name="Tiles")
    doc.add_frame()
    doc.stack.active_index = 0
    doc.mask = SelectionMask.from_rect(doc.size, (0, 0, 4, 4))
    doc.lift()
    assert doc.floating is not None
    f = doc.anim.current

    with pytest.raises(ValueError, match="tilemap"):
        doc.commit_floating_range(0, len(anim.tracks) - 1, 0, f)
    # The buffer is still floating -- the refusal fired before the teardown.
    assert doc.floating is not None
    _assert_synced(doc)


def test_commit_floating_onto_a_tilemap_layer_goes_through_the_funnel():
    """The one float path that is *not* refused: an ordinary commit ends at
    ``_commit_patch``, so it earns the tilemap behaviour for free."""
    doc, slot, cel = _setup("stack", _tile(RED))
    doc.floating = FloatingBuffer(
        pixels=np.full((4, 4, 4), 255, dtype=np.uint8),
        mask=np.full((4, 4), 255, dtype=np.uint8),
        offset=(0, 0),
        layer_uid=cel.uid,
    )

    assert doc.commit_floating() is True
    assert slot.tileset.tile_count == 3  # stack mode appended the new content
    _assert_synced(doc)
