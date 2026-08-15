"""Slices: uid-addressed history, explicit keys, and geometry that follows.

Four of these are the failure modes the design was chosen to avoid, so they are
worth naming rather than reading as coverage.

*Uid-addressed undo* is the whole reason a slice does not use the tag list's
whole-list snapshot: reordering or deleting must not retarget a step.

*A step that changes nothing is not pushed*, because ``dirty`` is a comparison
against ``history.head`` and a saved document must not ask to be saved for a
gesture that did nothing.

*A key survives a frame delete and its undo*, which is the accepted small leak
``_placeholder_uids`` already takes: nothing purges keys when a frame goes, so
an undone delete -- which re-inserts the same ``Frame`` object and therefore the
same uid -- finds its override still there.

*A quarter turn is exact*, because a slice that drifted a pixel per rotation
would be wrong in a file somebody else reads and invisible in the image.
"""

from __future__ import annotations

from warlock.studio.inker import transform as tf
from warlock.studio.inker.document import Document
from warlock.studio.inker.slices import Slice, SliceKey


def _doc(width: int = 32, height: int = 16) -> Document:
    return Document.blank(width, height)


# --- the model ----------------------------------------------------------------


def test_a_slice_resolves_to_its_base_when_the_frame_is_not_keyed():
    entry = Slice(name="hit", bounds=(2, 3, 10, 11), pivot=(4.0, 8.0))
    assert entry.at(None) == SliceKey(bounds=(2, 3, 10, 11), pivot=(4.0, 8.0))
    assert entry.at(12345).bounds == (2, 3, 10, 11)


def test_a_keyed_frame_resolves_to_the_key():
    key = SliceKey(bounds=(0, 0, 4, 4), pivot=(1.0, 1.0))
    entry = Slice(bounds=(2, 2, 8, 8), keys={7: key})
    assert entry.at(7) is key
    assert entry.at(8).bounds == (2, 2, 8, 8)


def test_a_copy_keeps_the_uid_and_detaches_the_keys():
    entry = Slice(bounds=(0, 0, 4, 4), keys={1: SliceKey(bounds=(1, 1, 2, 2))})
    clone = entry.copy()
    assert clone.uid == entry.uid
    clone.keys[2] = SliceKey(bounds=(0, 0, 1, 1))
    assert set(entry.keys) == {1}


# --- history ------------------------------------------------------------------


def test_adding_and_removing_a_slice_is_one_step_each():
    doc = _doc()
    entry = doc.add_slice((1, 1, 5, 5), name="a")
    assert [s.name for s in doc.slices] == ["a"]
    doc.undo()
    assert doc.slices == []
    doc.redo()
    assert [s.uid for s in doc.slices] == [entry.uid]

    doc.remove_slice(entry.uid)
    assert doc.slices == []
    doc.undo()
    assert [s.uid for s in doc.slices] == [entry.uid]


def test_an_undone_add_puts_the_same_object_back():
    """Which is what lets a change recorded before the undo still apply after
    the redo -- the ``LayerAddEdit`` rule, one level up."""
    doc = _doc()
    entry = doc.add_slice((1, 1, 5, 5))
    doc.undo()
    doc.redo()
    assert doc.slices[0] is entry


def test_undo_after_a_reorder_lands_on_the_slice_the_edit_was_made_to():
    """The reason a slice is addressed by uid. Dragging the second slice, then
    deleting the first, then Ctrl+Z twice must move the slice that was dragged
    -- an index would have named its neighbour by then."""
    doc = _doc()
    first = doc.add_slice((0, 0, 4, 4), name="a")
    second = doc.add_slice((8, 8, 12, 12), name="b")

    doc.set_slice(second.uid, bounds=(9, 9, 13, 13))
    doc.remove_slice(first.uid)
    assert [s.name for s in doc.slices] == ["b"]

    doc.undo()  # the removal
    doc.undo()  # the move
    assert doc.slice_by_uid(second.uid).bounds == (8, 8, 12, 12)
    assert doc.slice_by_uid(first.uid).bounds == (0, 0, 4, 4)


def test_a_change_that_changes_nothing_pushes_nothing():
    doc = _doc()
    entry = doc.add_slice((1, 1, 5, 5), name="a")
    head = doc.history.head
    assert doc.set_slice(entry.uid, name="a", bounds=(1, 1, 5, 5)) is False
    assert doc.history.head == head
    assert doc.set_slice(entry.uid) is False
    assert doc.history.head == head


def test_a_drag_records_what_it_was_told_it_was_before():
    """``was`` is how a live-mutated drag becomes one step: the canvas moves the
    slice every frame so the overlay follows the cursor, and hands back the
    properties as they stood at the press."""
    doc = _doc()
    entry = doc.add_slice((1, 1, 5, 5))
    before = {"bounds": (1, 1, 5, 5)}
    entry.bounds = (6, 2, 10, 6)  # what the drag did, frame by frame
    assert doc.set_slice(entry.uid, was=before) is True
    assert len(doc.history) == 2
    doc.undo()
    assert doc.slice_by_uid(entry.uid).bounds == (1, 1, 5, 5)


def test_a_new_slice_is_clamped_into_the_canvas_rather_than_refused():
    doc = _doc(16, 16)
    entry = doc.add_slice((-4, -4, 200, 200))
    assert entry.bounds == (0, 0, 16, 16)


def test_an_undone_change_does_not_share_its_keys_with_the_document():
    """The ``_set_tags`` rule: the edit holds both halves and may replay any
    number of times, so a later edit must not write through into them."""
    doc = _doc()
    entry = doc.add_slice((0, 0, 8, 8))
    doc.set_slice(entry.uid, keys={5: SliceKey(bounds=(1, 1, 3, 3))})
    doc.undo()
    doc.redo()
    doc.slices[0].keys[9] = SliceKey(bounds=(0, 0, 2, 2))
    doc.undo()
    doc.redo()
    assert set(doc.slices[0].keys) == {5}


# --- keys ---------------------------------------------------------------------


def test_keying_a_frame_is_explicit_and_starts_from_what_is_resolved():
    doc = _doc()
    doc.add_frame()
    frame = doc.anim.frames[0]
    entry = doc.add_slice((2, 2, 6, 6), pivot=(1.0, 2.0))

    # Moving the slice keys nothing at all.
    doc.set_slice(entry.uid, bounds=(3, 3, 7, 7))
    assert doc.slices[0].keys == {}

    assert doc.set_slice_key(entry.uid, frame.uid) is True
    key = doc.slices[0].keys[frame.uid]
    assert (key.bounds, key.pivot) == ((3, 3, 7, 7), (1.0, 2.0))


def test_a_key_survives_a_frame_delete_and_its_undo():
    doc = _doc()
    doc.add_frame()
    doc.add_frame()
    gone = doc.anim.frames[1]
    entry = doc.add_slice((0, 0, 4, 4))
    doc.set_slice_key(entry.uid, gone.uid, key=SliceKey(bounds=(5, 5, 9, 9)))

    doc.remove_frame(1)
    doc.undo()
    back = doc.anim.frames[1]
    assert back.uid == gone.uid
    assert doc.slices[0].at(back.uid).bounds == (5, 5, 9, 9)


def test_clearing_a_key_that_is_not_there_pushes_nothing():
    doc = _doc()
    entry = doc.add_slice((0, 0, 4, 4))
    head = doc.history.head
    assert doc.set_slice_key(entry.uid, 999, clear=True) is False
    assert doc.history.head == head


# --- geometry -----------------------------------------------------------------


def test_a_quarter_turn_moves_a_slice_exactly():
    doc = _doc(32, 16)
    doc.add_slice((2, 3, 10, 7), pivot=(1.0, 2.0))
    doc.rotate90(1)
    assert doc.size == (16, 32)
    # ``(x, y) -> (y, w - x)`` on the two corners, ordered: x spans [3, 7),
    # y spans [32 - 10, 32 - 2) = [22, 30).
    assert doc.slices[0].bounds == (3, 22, 7, 30)
    # And the pivot with it: it sat at canvas (3, 5), which maps to (5, 29).
    assert doc.slices[0].pivot == (5.0 - 3.0, 29.0 - 22.0)


def test_four_quarter_turns_are_the_identity():
    doc = _doc(32, 16)
    entry = doc.add_slice((2, 3, 10, 7), pivot=(1.5, 2.5))
    for _ in range(4):
        doc.rotate90(1)
    assert doc.slices[0].bounds == (2, 3, 10, 7)
    assert doc.slices[0].pivot == (1.5, 2.5)
    assert doc.slices[0].uid == entry.uid


def test_a_flip_mirrors_a_slice_and_its_pivot():
    doc = _doc(32, 16)
    doc.add_slice((2, 3, 10, 7), pivot=(1.0, 2.0))
    doc.flip("horizontal")
    assert doc.slices[0].bounds == (22, 3, 30, 7)
    # The pivot was at canvas x = 3, which mirrors to 29 -- seven in from the
    # slice's new origin, where it was one in from the old.
    assert doc.slices[0].pivot == (7.0, 2.0)


def test_a_scale_carries_the_slices_with_it():
    doc = _doc(32, 16)
    doc.add_slice((4, 4, 8, 8), pivot=(2.0, 2.0))
    doc.scale((64, 32))
    assert doc.slices[0].bounds == (8, 8, 16, 16)
    assert doc.slices[0].pivot == (4.0, 4.0)


def test_a_crop_clamps_a_slice_to_a_pixel_rather_than_deleting_it():
    """A slice has a name, a pivot and a nine-slice centre that a crop cannot
    recover, so it is clamped to 1x1 rather than dropped."""
    doc = _doc(32, 16)
    doc.add_slice((24, 8, 30, 14), name="far")
    doc.crop((0, 0, 8, 8))
    assert doc.size == (8, 8)
    entry = doc.slices[0]
    assert entry.name == "far"
    assert entry.bounds == (7, 7, 8, 8)


def test_a_crop_translates_a_slice_that_survives_it():
    doc = _doc(32, 16)
    doc.add_slice((10, 4, 18, 12))
    doc.crop((8, 2, 24, 14))
    assert doc.slices[0].bounds == (2, 2, 10, 10)


def test_undoing_a_crop_restores_the_slices_it_clamped():
    doc = _doc(32, 16)
    entry = doc.add_slice((24, 8, 30, 14), name="far", pivot=(1.0, 1.0))
    doc.crop((0, 0, 8, 8))
    doc.undo()
    assert doc.size == (32, 16)
    back = doc.slices[0]
    assert (back.uid, back.bounds, back.pivot) == (entry.uid, (24, 8, 30, 14), (1.0, 1.0))


def test_redoing_a_crop_maps_the_slices_again():
    doc = _doc(32, 16)
    doc.add_slice((24, 8, 30, 14))
    doc.crop((0, 0, 8, 8))
    doc.undo()
    doc.redo()
    assert doc.slices[0].bounds == (7, 7, 8, 8)


def test_geometry_carries_the_per_frame_keys_too():
    doc = _doc(32, 16)
    doc.add_frame()
    frame = doc.anim.frames[0]
    entry = doc.add_slice((0, 0, 4, 4))
    doc.set_slice_key(entry.uid, frame.uid, key=SliceKey(bounds=(2, 3, 10, 7)))
    doc.rotate90(1)
    assert doc.slices[0].keys[frame.uid].bounds == (3, 22, 7, 30)


def test_a_canvas_resize_translates_a_slice_by_the_offset():
    doc = _doc(16, 16)
    doc.add_slice((2, 2, 6, 6))
    doc.resize_canvas((32, 32), anchor="centre")
    assert doc.slices[0].bounds == (10, 10, 14, 14)


def test_a_slice_edit_never_touches_the_composite():
    """``rev`` ticks so a pane redraws; the dirty rectangle stays clean, because
    a slice is not pixels and repainting the canvas to move a rectangle nobody
    paints with would throw away every cached frame in the document."""
    doc = _doc()
    doc.take_dirty()
    rev = doc.rev
    entry = doc.add_slice((1, 1, 5, 5))
    doc.set_slice(entry.uid, name="renamed")
    assert doc.rev > rev
    assert doc.take_dirty() is None


# --- the pure mappers ---------------------------------------------------------


def test_a_rectangle_is_ordered_before_it_is_rounded():
    """Flooring the origin and ceiling the far edge only rounds outward while
    the pair is already the right way up."""
    assert tf.rect_from_points((3.2, 9.7), (0.4, 1.1)) == (0, 1, 4, 10)


def test_clamping_never_produces_an_empty_rectangle():
    for rect in ((-9, -9, -4, -4), (40, 40, 60, 60), (5, 5, 5, 5)):
        x0, y0, x1, y1 = tf.clamp_rect(rect, (16, 16))
        assert 0 <= x0 < x1 <= 16 and 0 <= y0 < y1 <= 16


def test_the_rect_mapper_and_the_plane_agree_about_a_quarter_turn():
    """The mapper is derived from ``np.rot90``'s own index arithmetic, so this
    checks the derivation against the array rather than restating the formula:
    one pixel's own 1x1 rectangle, mapped, has to be where that pixel went.

    A *point* comparison would not do it. Bounds are exclusive at the far edge,
    so a pixel's top-left corner maps to one of the four corners of its new box
    -- ``(6, 2)`` here comes out at ``(2, 3)``, which is the bottom-left of the
    right answer. Mapping the pair and re-ordering is exactly what removes that
    trap, and it is the reason ``rect_from_points`` exists.
    """
    import numpy as np

    plane = np.zeros((5, 9, 4), dtype=np.uint8)
    plane[2, 6] = (1, 2, 3, 4)
    turned = tf.rotate90(plane, 1)
    rect = tf.rotate90_rect((6, 2, 7, 3), (9, 5), 1)
    assert rect == (2, 2, 3, 3)
    assert tuple(turned[rect[1], rect[0]]) == (1, 2, 3, 4)
