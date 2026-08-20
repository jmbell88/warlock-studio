"""Tests for ``tile_edits.py`` + ``_doc_tiles.py``: the tileset list, the refs
door and the geometry/range refusals a tilemap layer earns until Chunk 3.7
teaches refs-aware permutation.

``_assert_synced`` is the Wave 3 risk-item test helper the brief calls for:
after every op in this suite, every ``TilemapCel``'s ``pixels`` must equal
``materialize(refs, bound_tileset, size)`` -- the funnel that keeps the two in
step is singular by design, and this is what would catch it if it were not.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import transform as tf
from warlock.studio.inker.document import Document
from warlock.studio.inker.selection import SelectionMask
from warlock.studio.inker.tiles import TilemapCel, materialize, strip
from warlock.studio.tilegrid import gid
from warlock.studio.undo import CompoundEdit

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def _doc(width: int = 8, height: int = 8) -> Document:
    return Document.blank(width, height)


def _tile(colour: tuple[int, int, int, int], w: int = 4, h: int = 4) -> np.ndarray:
    tile = np.zeros((h, w, 4), dtype=np.uint8)
    tile[..., 0], tile[..., 1], tile[..., 2], tile[..., 3] = colour
    return tile


def _blank_tile(w: int = 4, h: int = 4) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


def _tileset(*colours: tuple[int, int, int, int], w: int = 4, h: int = 4):
    stack = np.stack([_blank_tile(w, h), *[_tile(c, w, h) for c in colours]], axis=0)
    return strip(stack)


def _assert_synced(doc: Document) -> None:
    """Every tilemap cel's pixels agree with ``materialize(refs, ts, size)``."""
    layers = doc.stack if doc.anim is None else doc.anim.unique_cel_layers()
    for layer in layers:
        if isinstance(layer, TilemapCel):
            slot = doc.tileset_slot(layer.tileset_uid)
            want = materialize(layer.refs, slot.tileset, layer.size)
            assert np.array_equal(layer.pixels, want), f"cel {layer.uid} drifted from its refs"


def _still_tilemap(doc: Document, *colours: tuple[int, int, int, int]):
    slot = doc.add_tileset(_tileset(*colours))
    cel = doc.add_tilemap_layer(slot.uid)
    return slot, cel


# -- add_tileset / remove_tileset / tileset_slot -----------------------------


def test_add_tileset_pushes_one_step_and_undo_removes_the_slot():
    doc = _doc()
    slot = doc.add_tileset(_tileset(RED))
    assert len(doc.tilesets) == 1
    assert doc.tilesets[0] is slot
    assert doc.history.can_undo

    doc.history.undo(doc)
    assert doc.tilesets == []

    doc.history.redo(doc)
    assert len(doc.tilesets) == 1
    assert doc.tilesets[0] is slot
    assert doc.tileset_slot(slot.uid) is slot


def test_remove_tileset_undo_redo_restores_the_same_slot_object():
    doc = _doc()
    slot = doc.add_tileset(_tileset(RED))
    doc.history.clear()

    assert doc.remove_tileset(slot.uid) is True
    assert doc.tilesets == []

    doc.history.undo(doc)
    assert len(doc.tilesets) == 1
    # The unlink rule: the same object comes back, not a rebuild -- a track
    # binding recorded against this uid before the remove must still resolve.
    assert doc.tilesets[0] is slot

    doc.history.redo(doc)
    assert doc.tilesets == []


def test_remove_tileset_returns_false_for_an_unknown_uid():
    doc = _doc()
    assert doc.remove_tileset(999_999) is False


def test_remove_tileset_refuses_while_a_track_binds_it():
    doc = _doc()
    anim = doc.ensure_animation()
    slot = doc.add_tileset(_tileset(RED))
    anim.tracks[0].tileset_uid = slot.uid  # bound, no cel drawn yet
    with pytest.raises(ValueError):
        doc.remove_tileset(slot.uid)


def test_remove_tileset_refuses_while_a_cel_binds_it():
    doc = _doc()
    anim = doc.ensure_animation()
    slot = doc.add_tileset(_tileset(RED))
    anim.tracks[0].tileset_uid = slot.uid
    doc._ensure_cel_for(doc.stack[0].uid)
    with pytest.raises(ValueError):
        doc.remove_tileset(slot.uid)


def test_tileset_slot_raises_key_error_for_an_unknown_uid():
    doc = _doc()
    with pytest.raises(KeyError):
        doc.tileset_slot(123_456)


# -- add_tilemap_layer --------------------------------------------------------


def test_add_tilemap_layer_on_a_still_document_inserts_a_tilemap_cel():
    doc = _doc()
    slot, cel = _still_tilemap(doc, RED)
    assert isinstance(cel, TilemapCel)
    assert cel.tileset_uid == slot.uid
    assert cel.name == "Tilemap"
    assert not cel.refs.any()
    _assert_synced(doc)

    doc.history.undo(doc)
    assert not any(layer is cel for layer in doc.stack)
    doc.history.redo(doc)
    assert any(layer is cel for layer in doc.stack)


def test_add_tilemap_layer_on_an_animated_document_adds_a_bound_track():
    doc = _doc()
    doc.ensure_animation()
    slot = doc.add_tileset(_tileset(RED))
    doc.add_tilemap_layer(slot.uid, name="Ground")

    track = doc.anim.tracks[doc.stack.active_index]
    assert track.tileset_uid == slot.uid
    assert track.name == "Ground"
    assert (track.uid, doc.anim.frame.uid) not in doc.anim.cels  # sparse: no cel yet

    doc.history.undo(doc)
    assert slot.uid not in [t.tileset_uid for t in doc.anim.tracks]


# -- place_tiles ---------------------------------------------------------------


def test_place_tiles_writes_refs_and_pixels_as_one_step():
    doc = _doc()
    slot, cel = _still_tilemap(doc, RED)
    doc.history.clear()

    patch = np.array([[1, 0]], dtype=np.uint32)
    assert doc.place_tiles(cel.uid, (0, 0), patch) is True
    assert doc.history.can_undo
    assert not isinstance(doc.history.top, CompoundEdit)  # the cel already existed
    assert cel.refs[0, 0] == 1
    assert cel.refs[0, 1] == 0
    _assert_synced(doc)

    doc.history.undo(doc)
    assert not cel.refs.any()
    _assert_synced(doc)

    doc.history.redo(doc)
    assert cel.refs[0, 0] == 1
    _assert_synced(doc)


def test_place_tiles_clips_a_patch_that_runs_past_the_grid_edge():
    doc = _doc()  # 8x8 canvas, 4x4 tiles -> a 2x2 grid
    slot, cel = _still_tilemap(doc, RED)
    doc.history.clear()

    patch = np.array([[1, 1, 1]], dtype=np.uint32)  # 3 wide, grid is 2 wide
    assert doc.place_tiles(cel.uid, (0, 0), patch) is True
    assert cel.refs.shape == (2, 2)
    assert cel.refs[0, 0] == 1
    assert cel.refs[0, 1] == 1
    _assert_synced(doc)


def test_place_tiles_no_op_pushes_nothing():
    doc = _doc()
    slot, cel = _still_tilemap(doc, RED)
    doc.history.clear()

    zeros = np.zeros((2, 2), dtype=np.uint32)
    assert doc.place_tiles(cel.uid, (0, 0), zeros) is False
    assert not doc.history.can_undo
    _assert_synced(doc)


def test_place_tiles_entirely_off_grid_pushes_nothing():
    doc = _doc()
    slot, cel = _still_tilemap(doc, RED)
    doc.history.clear()

    patch = np.array([[1]], dtype=np.uint32)
    assert doc.place_tiles(cel.uid, (99, 99), patch) is False
    assert not doc.history.can_undo


def test_place_tiles_refuses_a_non_tilemap_layer():
    doc = _doc()
    layer = doc.add_layer()
    with pytest.raises(ValueError):
        doc.place_tiles(layer.uid, (0, 0), np.array([[1]], dtype=np.uint32))


def test_place_tiles_refuses_a_local_id_the_tileset_does_not_have():
    """Blank *today* and a different tile tomorrow.

    ``materialize`` draws an out-of-range ref as blank, so an over-range id
    writes something that looks right until the tileset grows past it -- at
    which point every one of those cells silently starts drawing whatever tile
    landed on that number. The door is where that has to be caught, because by
    the time it draws there is nothing left to say it was ever wrong.
    """
    doc = _doc()
    _slot, cel = _still_tilemap(doc, RED)  # tiles 0 and 1
    with pytest.raises(ValueError, match="outside"):
        doc.place_tiles(cel.uid, (0, 0), np.array([[2]], dtype=np.uint32))
    assert not cel.refs.any()
    _assert_synced(doc)


def test_place_tiles_reads_the_id_through_the_gid_mask():
    """The flag bits are not part of the id, so a flipped tile 1 is in range
    and a flipped tile 9 is not."""
    doc = _doc()
    _slot, cel = _still_tilemap(doc, RED)
    from warlock.studio.tilegrid import gid

    assert doc.place_tiles(cel.uid, (0, 0), np.array([[1 | gid.FLIP_D]], dtype=np.uint32))
    with pytest.raises(ValueError, match="outside"):
        doc.place_tiles(cel.uid, (1, 0), np.array([[9 | gid.FLIP_D]], dtype=np.uint32))
    _assert_synced(doc)


def test_place_tiles_range_check_runs_before_any_autovivify():
    """The funnel discipline: a refused write must not leave a cel behind with
    no undo step to take it back out."""
    doc = _doc()
    anim = doc.ensure_animation()
    slot = doc.add_tileset(_tileset(RED))
    anim.tracks[0].tileset_uid = slot.uid
    doc.add_frame()
    placeholder_uid = doc.stack[0].uid
    head = doc.history.head
    with pytest.raises(ValueError, match="outside"):
        doc.place_tiles(placeholder_uid, (0, 0), np.array([[5]], dtype=np.uint32))
    assert doc.history.head == head
    assert doc._pending_cels == []
    assert not isinstance(doc.stack[0], TilemapCel)


def test_place_tiles_autovivify_and_write_is_one_compound_step():
    doc = _doc()
    anim = doc.ensure_animation()
    slot = doc.add_tileset(_tileset(RED))
    anim.tracks[0].tileset_uid = slot.uid
    doc.add_frame()  # frame 2: track 0 is a placeholder
    placeholder_uid = doc.stack[0].uid
    doc.history.clear()

    patch = np.array([[1]], dtype=np.uint32)
    assert doc.place_tiles(placeholder_uid, (0, 0), patch) is True
    assert len(doc.history) == 1
    assert isinstance(doc.history.top, CompoundEdit)
    cel = doc.stack[0]
    assert isinstance(cel, TilemapCel)
    _assert_synced(doc)

    doc.history.undo(doc)
    assert doc.anim.is_placeholder(doc.stack[0])

    doc.history.redo(doc)
    assert isinstance(doc.stack[0], TilemapCel)
    _assert_synced(doc)


def test_place_tiles_on_a_plain_tracks_placeholder_refuses_before_autovivifying():
    """The regression this guards: on an animated document, a placeholder
    behind a *plain* (non-tileset) track must be refused without ever
    autovivifying a real ``Layer`` for it -- a refusal that fired only after
    ``_ensure_cel_for`` ran would leave a materialized cel with no undo step
    and a dangling ``CelSetEdit`` in ``_pending_cels`` that would wrongly
    attach itself to whatever the next successful commit happened to be.
    """
    doc = _doc()
    doc.ensure_animation()  # track 0 is an ordinary raster track
    doc.add_frame()  # frame 2: track 0 is a placeholder
    placeholder_uid = doc.stack[0].uid
    doc.history.clear()

    with pytest.raises(ValueError):
        doc.place_tiles(placeholder_uid, (0, 0), np.array([[1]], dtype=np.uint32))

    assert doc.anim.is_placeholder(doc.stack[0])
    assert doc._pending_cels == []
    assert not doc.history.can_undo

    # A subsequent legitimate op still pushes exactly one step -- nothing
    # orphaned by the refusal rides along with it.
    doc.add_layer(name="Fresh")
    assert len(doc.history) == 1
    assert not isinstance(doc.history.top, CompoundEdit)


def test_place_tiles_after_reordering_the_stack_still_lands_by_uid():
    doc = _doc()
    slot, cel = _still_tilemap(doc, RED)
    doc.add_layer(name="Above")
    doc.move_layer(doc.stack.index_of(cel.uid), 0)
    doc.history.clear()

    patch = np.array([[1]], dtype=np.uint32)
    assert doc.place_tiles(cel.uid, (0, 0), patch) is True
    _assert_synced(doc)

    doc.history.undo(doc)
    assert not cel.refs.any()
    _assert_synced(doc)


# -- shared-invalidation risk item --------------------------------------------


def test_apply_tileset_tiles_rematerializes_two_layers_sharing_one_tileset():
    doc = _doc()
    slot = doc.add_tileset(_tileset(RED))
    cel_a = doc.add_tilemap_layer(slot.uid, name="A")
    cel_b = doc.add_tilemap_layer(slot.uid, name="B")
    doc.place_tiles(cel_a.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    doc.place_tiles(cel_b.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    _assert_synced(doc)

    doc._apply_tileset_tiles(slot.uid, [(1, _tile(BLUE))])

    assert tuple(int(c) for c in cel_a.pixels[0, 0]) == BLUE
    assert tuple(int(c) for c in cel_b.pixels[0, 0]) == BLUE
    _assert_synced(doc)


def test_apply_tileset_tiles_stamps_every_frame_holding_a_linked_cel():
    doc = _doc()
    anim = doc.ensure_animation()
    slot = doc.add_tileset(_tileset(RED))
    anim.tracks[0].tileset_uid = slot.uid
    doc.add_frame()  # a fresh placeholder frame for the now-bound track
    doc._ensure_cel_for(doc.stack[0].uid)
    cel = doc.stack[0]
    assert isinstance(cel, TilemapCel)
    doc.place_tiles(cel.uid, (0, 0), np.array([[1]], dtype=np.uint32))

    doc.add_frame(link=True)  # frame 3 links frame 2's cel
    frame1, frame2 = anim.frames[1], anim.frame
    stamp1_before = doc.frame_stamp(frame1.uid)
    stamp2_before = doc.frame_stamp(frame2.uid)
    _assert_synced(doc)

    doc._apply_tileset_tiles(slot.uid, [(1, _tile(BLUE))])

    assert doc.frame_stamp(frame1.uid) > stamp1_before
    assert doc.frame_stamp(frame2.uid) > stamp2_before
    assert tuple(int(c) for c in cel.pixels[0, 0]) == BLUE
    _assert_synced(doc)


def test_apply_tileset_grow_appends_and_undo_truncates():
    doc = _doc()
    slot, cel = _still_tilemap(doc, RED)
    before_count = slot.tileset.tile_count

    doc._apply_tileset_grow(slot.uid, np.stack([_tile(BLUE)], axis=0))
    assert slot.tileset.tile_count == before_count + 1
    _assert_synced(doc)

    doc._apply_tileset_grow(slot.uid, before_count)
    assert slot.tileset.tile_count == before_count
    _assert_synced(doc)


# -- geometry refusals ---------------------------------------------------------


def _still_tilemap_doc() -> Document:
    doc = _doc()
    _still_tilemap(doc, RED)
    return doc


# The Aseprite parity programme's Wave 3 left "refs-aware flip/rotate (the flag algebra)"
# open, and these two tests pinned the blanket refusal that stood in for it. The
# refusal has now *moved* rather than been deleted -- the ledger rule at the top
# of ``asein.py``, applied to geometry: the editor learned to model the thing, so
# what refuses now are the narrower cases underneath it, raised by name.


@pytest.mark.parametrize("axis", ["horizontal", "vertical"])
def test_flip_mirrors_the_refs_and_turns_every_cells_flags(axis: str):
    """The arrangement *and* the tiles, or a flipped map is a mirrored layout of
    unmirrored tiles -- the classic version of this bug.

    The assertion that matters is the last one: the picture a flipped tilemap
    draws is exactly the flip of the picture it drew before. That is the whole
    correctness claim of the flag algebra, and it is checked against the
    *pixels* rather than a hand-derived flag table, because a table is the thing
    being derived."""
    doc = _doc(8, 8)
    _still_tilemap(doc, RED, BLUE)
    cel = doc.stack[doc.stack.active_index]
    cel.refs[0, 0] = gid.compose(1)
    cel.refs[0, 1] = gid.compose(2, flip_h=True)
    cel.refs[1, 0] = gid.compose(2, flip_d=True)
    doc._repaint_tiles(cel, (0, 0, 2, 2))
    before = cel.pixels.copy()
    doc.history.clear()

    doc.flip(axis)
    _assert_synced(doc)
    after = doc.stack[doc.stack.active_index]
    assert np.array_equal(after.pixels, tf.flip(before, axis))


def test_a_quarter_turn_carries_a_tilemap_layer_with_it():
    """Counter-clockwise, matching ``transform.rotate90``. Same bar: the turned
    map draws the turn of what it drew."""
    doc = _doc(8, 8)
    _still_tilemap(doc, RED, BLUE)
    cel = doc.stack[doc.stack.active_index]
    cel.refs[0, 0] = gid.compose(1)
    cel.refs[0, 1] = gid.compose(2, flip_v=True)
    doc._repaint_tiles(cel, (0, 0, 2, 2))
    before = cel.pixels.copy()
    doc.history.clear()

    doc.rotate90(1)
    _assert_synced(doc)
    after = doc.stack[doc.stack.active_index]
    assert np.array_equal(after.pixels, tf.rotate90(before, 1))


def test_four_quarter_turns_and_two_flips_are_the_identity():
    """The property a permutation owes: applied enough times it comes back.
    Cheap to state, and it catches a rule that is self-consistent but wrong."""
    doc = _doc(8, 8)
    _still_tilemap(doc, RED, BLUE)
    cel = doc.stack[doc.stack.active_index]
    cel.refs[0, 0] = gid.compose(1, flip_d=True)
    cel.refs[1, 1] = gid.compose(2, flip_h=True, flip_v=True)
    doc._repaint_tiles(cel, (0, 0, 2, 2))
    refs = cel.refs.copy()
    doc.history.clear()

    for _ in range(4):
        doc.rotate90(1)
    assert np.array_equal(doc.stack[doc.stack.active_index].refs, refs)
    doc.flip("horizontal")
    doc.flip("horizontal")
    assert np.array_equal(doc.stack[doc.stack.active_index].refs, refs)
    _assert_synced(doc)


def test_a_flip_of_a_tilemap_is_one_undo_step():
    """It goes through ``_replay`` like every other geometry op, so it owes the
    same one-gesture-one-Ctrl+Z contract -- and undoing it must put the *refs*
    back, not only the picture."""
    doc = _doc(8, 8)
    _still_tilemap(doc, RED, BLUE)
    cel = doc.stack[doc.stack.active_index]
    cel.refs[0, 0] = gid.compose(1)
    doc._repaint_tiles(cel, (0, 0, 2, 2))
    refs = cel.refs.copy()
    pixels = cel.pixels.copy()
    doc.history.clear()

    doc.flip("horizontal")
    assert not np.array_equal(doc.stack[doc.stack.active_index].refs, refs)
    doc.history.undo(doc)
    back = doc.stack[doc.stack.active_index]
    assert np.array_equal(back.refs, refs)
    assert np.array_equal(back.pixels, pixels)
    _assert_synced(doc)


def test_a_flip_refuses_a_canvas_that_is_not_a_whole_number_of_tiles():
    """The one case the algebra genuinely cannot reach, and why: ``materialize``
    crops the grid to the canvas, so on a ragged canvas the visible pixel a flip
    wants at x=0 is one the crop threw away. No flag reconciles that, so it is
    refused by name -- and the message names the thing the user can do about
    it."""
    doc = _doc(10, 8)  # 10 wide at 4px tiles: two whole columns and a stub
    _still_tilemap(doc, RED)
    with pytest.raises(ValueError, match="tile-aligned canvas"):
        doc.flip("horizontal")
    # The *other* axis is fine: 8 tall is two whole rows.
    doc.flip("vertical")
    _assert_synced(doc)


def test_a_quarter_turn_refuses_non_square_tiles():
    """A turned 8x4 tile is 4x8 and a tileset stores one tile size, so the
    rotated art would have nowhere to live. A flip has no such problem, which is
    why only the rotation asks."""
    doc = _doc(16, 16)
    slot = doc.add_tileset(_tileset(RED, w=8, h=4))
    doc.add_tilemap_layer(slot.uid)
    with pytest.raises(ValueError, match="square tiles"):
        doc.rotate90(1)
    # Mirroring is still fine -- 16 is a whole number of both 8 and 4.
    doc.flip("horizontal")
    doc.flip("vertical")
    _assert_synced(doc)


def test_a_refused_flip_has_changed_nothing_at_all():
    """The refusal is built before ``commit_floating``, so a document that says
    no is a document that did nothing -- not one that committed a floating
    buffer on its way to raising."""
    doc = _doc(10, 8)
    _still_tilemap(doc, RED)
    cel = doc.stack[doc.stack.active_index]
    cel.refs[0, 0] = gid.compose(1)
    doc._repaint_tiles(cel, (0, 0, 2, 2))
    doc.history.clear()
    refs = cel.refs.copy()
    pixels = cel.pixels.copy()
    head = doc.history.head

    with pytest.raises(ValueError):
        doc.flip("horizontal")
    assert doc.history.head == head
    assert np.array_equal(doc.stack[doc.stack.active_index].refs, refs)
    assert np.array_equal(doc.stack[doc.stack.active_index].pixels, pixels)


def test_scale_refuses_a_document_with_a_tilemap_layer():
    doc = _still_tilemap_doc()
    with pytest.raises(ValueError):
        doc.scale((16, 16))


def test_crop_refuses_a_document_with_a_tilemap_layer():
    doc = _still_tilemap_doc()
    with pytest.raises(ValueError):
        doc.crop((0, 0, 4, 4))


def test_crop_to_selection_refuses_a_document_with_a_tilemap_layer():
    doc = _still_tilemap_doc()
    doc.mask = SelectionMask.from_rect(doc.size, (0, 0, 4, 4))
    with pytest.raises(ValueError):
        doc.crop_to_selection()


# -- canvas resize: the one geometry op that re-grids rather than refusing ------
#
# Chunk 3.7's half of the deferral. A resize moves no pixel *within* a tile, so
# a tile-aligned offset is a pure pad/crop of the refs plane -- which is why
# this one is modelled and the flips, the rotations, the scale and the crop are
# still refused (their permutation would have to turn every cell's flag bits by
# the eight-symmetry algebra, which nothing here does yet).


def test_resize_canvas_pads_the_refs_grid_at_the_top_left_anchor():
    doc = _still_tilemap_doc()
    cel = doc.stack.active
    doc.place_tiles(cel.uid, (0, 0), np.array([[1, 1], [1, 1]], dtype=np.uint32))
    doc.resize_canvas((16, 16))
    assert doc.size == (16, 16)
    assert cel.refs.shape == (4, 4)
    assert np.array_equal(cel.refs[:2, :2], np.ones((2, 2), dtype=np.uint32))
    # The new room is blank, not a repeat of the old grid.
    assert not cel.refs[2:, :].any()
    assert not cel.refs[:, 2:].any()
    _assert_synced(doc)


def test_resize_canvas_honours_the_anchor_in_tile_units():
    doc = _still_tilemap_doc()
    cel = doc.stack.active
    doc.place_tiles(cel.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    doc.resize_canvas((16, 16), anchor="bottom-right")
    # An 8px offset at a 4px tile is two whole cells.
    assert cel.refs.shape == (4, 4)
    assert int(cel.refs[2, 2]) == 1
    assert not cel.refs[:2, :].any()
    _assert_synced(doc)


def test_resize_canvas_crops_the_refs_grid():
    doc = _still_tilemap_doc()
    cel = doc.stack.active
    doc.place_tiles(cel.uid, (1, 1), np.array([[1]], dtype=np.uint32))
    doc.resize_canvas((4, 4))
    assert cel.refs.shape == (1, 1)
    assert int(cel.refs[0, 0]) == 0
    _assert_synced(doc)


def test_resize_canvas_refuses_an_offset_that_is_not_tile_aligned():
    doc = _still_tilemap_doc()
    with pytest.raises(ValueError, match="tile-aligned"):
        doc.resize_canvas((16, 16), offset=(2, 0))
    # Refused at the door: nothing moved.
    assert doc.size == (8, 8)
    assert doc.stack.active.refs.shape == (2, 2)
    _assert_synced(doc)


def test_resize_canvas_refuses_a_non_aligned_anchor_offset_by_name():
    doc = _still_tilemap_doc()
    with pytest.raises(ValueError, match="tile-aligned"):
        doc.resize_canvas((14, 14), anchor="bottom-right")
    assert doc.size == (8, 8)


def test_undoing_a_resize_puts_the_refs_grid_back():
    doc = _still_tilemap_doc()
    cel = doc.stack.active
    doc.place_tiles(cel.uid, (0, 0), np.array([[1, 1], [1, 1]], dtype=np.uint32))
    doc.resize_canvas((16, 16))
    doc.undo()
    assert doc.size == (8, 8)
    back = doc.stack.active
    assert back.refs.shape == (2, 2)
    assert np.array_equal(back.refs, np.ones((2, 2), dtype=np.uint32))
    _assert_synced(doc)


def test_resize_canvas_regrids_every_cel_of_an_animated_document():
    doc = _animated_tilemap_doc()
    cel = doc.stack[0]
    doc.place_tiles(cel.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    doc.resize_canvas((16, 16))
    for layer in doc.anim.unique_cel_layers():
        if isinstance(layer, TilemapCel):
            assert layer.refs.shape == (4, 4)
    _assert_synced(doc)


def test_resize_canvas_still_checks_alignment_for_a_bound_but_empty_track():
    """A track bound to a tileset with no cel drawn on it yet is one
    autovivify away from a grid, so its tile size still decides the answer."""
    doc = _doc()
    anim = doc.ensure_animation()
    slot = doc.add_tileset(_tileset(RED))
    anim.tracks[0].tileset_uid = slot.uid
    with pytest.raises(ValueError, match="tile-aligned"):
        doc.resize_canvas((16, 16), offset=(1, 1))


def test_a_track_bound_to_a_tileset_counts_before_any_cel_exists():
    """The structural binding alone puts the document on the tilemap path: an op
    that ran before the first tile was placed would otherwise be one autovivify
    away from a ``TilemapCel`` whose ``pixels`` disagree with its ``refs``.

    It no longer *refuses* -- a flip is modelled now -- so what this asserts is
    that the bare binding is still what the alignment check consults. An 8x8
    canvas at 4px tiles is aligned and passes; a 10-wide one is refused by name,
    and only ``_holds_tilemap`` seeing the binding makes that happen."""
    doc = _doc()
    anim = doc.ensure_animation()
    slot = doc.add_tileset(_tileset(RED))
    anim.tracks[0].tileset_uid = slot.uid
    doc.flip("horizontal")
    _assert_synced(doc)

    ragged = _doc(10, 8)
    anim = ragged.ensure_animation()
    slot = ragged.add_tileset(_tileset(RED))
    anim.tracks[0].tileset_uid = slot.uid
    with pytest.raises(ValueError, match="tile-aligned canvas"):
        ragged.flip("horizontal")


# -- range-op refusals ----------------------------------------------------------


def _animated_tilemap_doc() -> Document:
    doc = _doc()
    anim = doc.ensure_animation()
    slot = doc.add_tileset(_tileset(RED))
    anim.tracks[0].tileset_uid = slot.uid
    # ``ensure_animation`` converted the document's one Background layer into
    # frame 1's cel *before* the track was bound, so that cel is an ordinary
    # ``Layer``, not a ``TilemapCel`` -- a fresh frame is what actually holds
    # a placeholder for the now-bound track, the same setup
    # ``test_tiles.py::test_track_with_tileset_uid_autovivifies_a_tilemap_cel``
    # uses.
    doc.add_frame()
    doc._ensure_cel_for(doc.stack[0].uid)
    return doc


# The range half of the same move. ``_permute_range`` used to refuse a tilemap
# outright; it now takes a ``refs_fn`` and refuses only where a caller has no
# refs answer to give -- so flip, rotate and a tile-aligned wrapping shift are
# modelled, and ``fill_range``/``filter_range`` (which *paint*) still refuse,
# permanently and for a different reason: painting a tilemap goes through the
# tile funnel, which is a mechanism rather than a gap.


def test_flip_range_carries_a_tilemap_tracks_refs():
    """The same flag algebra as the whole-canvas flip, one cel at a time -- and
    the same bar: the range's picture afterwards is the flip of its picture
    before."""
    doc = _animated_tilemap_doc()
    f = doc.anim.current
    cel = doc.layer_by_uid(doc.stack[0].uid)
    assert isinstance(cel, TilemapCel)
    cel.refs[0, 0] = gid.compose(1)
    doc._repaint_tiles(cel, (0, 0, 2, 2))
    before = cel.pixels.copy()
    doc.history.clear()

    assert doc.flip_range("horizontal", 0, 0, f, f) is True
    _assert_synced(doc)
    after = doc.layer_by_uid(doc.stack[0].uid)
    assert np.array_equal(after.pixels, tf.flip(before, "horizontal"))


def test_rotate_range_carries_a_tilemap_tracks_refs():
    doc = _animated_tilemap_doc()
    f = doc.anim.current
    cel = doc.layer_by_uid(doc.stack[0].uid)
    cel.refs[0, 0] = gid.compose(1)
    doc._repaint_tiles(cel, (0, 0, 2, 2))
    before = cel.pixels.copy()
    doc.history.clear()

    assert doc.rotate_range(1, 0, 0, f, f) is True
    _assert_synced(doc)
    after = doc.layer_by_uid(doc.stack[0].uid)
    assert np.array_equal(after.pixels, tf.rotate90(before, 1))


def test_a_range_permutation_of_a_tilemap_is_one_undo_step():
    """It goes back through ``TileRefsEdit``, not a pixel patch -- so undo puts
    the *record* back and the picture is re-derived from it."""
    doc = _animated_tilemap_doc()
    f = doc.anim.current
    cel = doc.layer_by_uid(doc.stack[0].uid)
    cel.refs[0, 0] = gid.compose(1)
    doc._repaint_tiles(cel, (0, 0, 2, 2))
    refs = cel.refs.copy()
    doc.history.clear()

    assert doc.flip_range("horizontal", 0, 0, f, f) is True
    doc.history.undo(doc)
    assert np.array_equal(doc.layer_by_uid(doc.stack[0].uid).refs, refs)
    _assert_synced(doc)


def test_rotate_range_refuses_non_square_tiles_even_on_a_square_canvas():
    """A square *canvas* does not imply square *tiles*: this one is 16x16 with
    8x4 tiles, which the canvas check waves through and the turn cannot do."""
    doc = _doc(16, 16)
    anim = doc.ensure_animation()
    slot = doc.add_tileset(_tileset(RED, w=8, h=4))
    anim.tracks[0].tileset_uid = slot.uid
    doc.add_frame()
    doc._ensure_cel_for(doc.stack[0].uid)
    f = anim.current
    with pytest.raises(ValueError, match="square tiles"):
        doc.rotate_range(1, 0, 0, f, f)


def test_shift_range_refuses_a_tilemap_track_that_does_not_wrap():
    """An unwrapped shift vacates pixels and fills them transparent; on a
    tilemap that means "these cells now point at no tile", which is a content
    decision this op has no mandate to make. Refused by name."""
    doc = _animated_tilemap_doc()
    f = doc.anim.current
    with pytest.raises(ValueError, match="must wrap"):
        doc.shift_range(4, 0, False, 0, 0, f, f)


def test_shift_range_refuses_a_distance_that_is_not_whole_tiles():
    """A sub-tile shift re-cuts every cell against a new phase, which rewrites
    the tileset -- a different operation wearing a shift's name, exactly the
    argument ``_tile_regrid`` makes about an unaligned resize offset."""
    doc = _animated_tilemap_doc()
    f = doc.anim.current
    with pytest.raises(ValueError, match="tile-aligned distance"):
        doc.shift_range(1, 0, True, 0, 0, f, f)


def test_a_tile_aligned_wrapping_shift_rolls_the_grid():
    """No cell's art changes, so no flag moves -- which is why this one needs no
    algebra where the flip and the turn do. Just a roll."""
    doc = _animated_tilemap_doc()
    f = doc.anim.current
    cel = doc.layer_by_uid(doc.stack[0].uid)
    cel.refs[0, 0] = gid.compose(1)
    doc._repaint_tiles(cel, (0, 0, 2, 2))
    before = cel.pixels.copy()
    doc.history.clear()

    assert doc.shift_range(4, 0, True, 0, 0, f, f) is True
    _assert_synced(doc)
    after = doc.layer_by_uid(doc.stack[0].uid)
    assert np.array_equal(after.pixels, tf.translate(before, 4, 0, wrap=True))


def test_fill_range_refuses_a_tilemap_track():
    doc = _animated_tilemap_doc()
    f = doc.anim.current
    with pytest.raises(ValueError):
        doc.fill_range((255, 0, 0, 255), 0, 0, f, f)


def test_filter_range_refuses_a_tilemap_track():
    doc = _animated_tilemap_doc()
    f = doc.anim.current
    with pytest.raises(ValueError):
        doc.filter_range("invert", {}, 0, 0, f, f)


def test_filter_range_refusal_leaves_every_touched_cel_untouched():
    """The funnel discipline, restated for ``filter_range``: a refused write
    must not leave the tilemap cel's pixels desynced from its refs, and must
    not leave an *earlier* cel in the same range filtered with no undo step
    to take it back out -- ``filter_range`` only pushes its edits after the
    whole loop, so a raise partway through must not have written anything
    permanent at all. The raster track is placed at index 0 and the tilemap
    track at index 1 specifically so the raster cel is the one visited first.
    """
    doc = _doc()
    anim = doc.ensure_animation()  # track 0: an ordinary raster track
    raster_cel = doc.stack[0]
    raster_cel.pixels[...] = 255
    slot = doc.add_tileset(_tileset(RED))
    tile_track_layer = doc.add_tilemap_layer(slot.uid, name="Ground")  # track 1
    assert doc.place_tiles(tile_track_layer.uid, (0, 0), np.array([[1]], dtype=np.uint32))
    tile_cel = doc.layer_by_uid(tile_track_layer.uid)
    assert isinstance(tile_cel, TilemapCel)

    raster_before = raster_cel.pixels.copy()
    tile_before = tile_cel.pixels.copy()
    head = doc.history.head

    f = anim.current
    with pytest.raises(ValueError):
        doc.filter_range("invert", {"red": 1.0, "green": 1.0, "blue": 1.0}, 0, 1, f, f)

    assert doc.history.head == head
    assert np.array_equal(raster_cel.pixels, raster_before)
    assert np.array_equal(tile_cel.pixels, tile_before)
    _assert_synced(doc)


def test_patch_edit_for_refuses_a_tilemap_cel():
    doc = _animated_tilemap_doc()
    cel = doc.stack[0]
    assert isinstance(cel, TilemapCel)
    with pytest.raises(ValueError):
        doc._patch_edit_for(cel, (0, 0, 4, 4), cel.pixels[0:4, 0:4].copy())
