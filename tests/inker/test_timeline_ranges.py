"""Range edits over the animation grid, headless.

One step per op, and the assertions are deliberately about the four things a
range edit is easy to get wrong and impossible to notice: that undo restores
frame *uids* rather than equal-looking frames, that it restores **link
identity** (``is``, not ``==`` -- two equal copies where the document had one
shared object is the failure that only shows up on the next stroke), that a
shared cel is charged to the byte budget once, and that an op issued after a
reorder lands where the user is looking rather than where the indices used to
point.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker._doc_ranges import clamp_span
from warlock.studio.inker.document import Document
from warlock.studio.inker.selection import FloatingBuffer

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
GREEN = (0, 255, 0, 255)


def _paint(doc: Document, colour: tuple[int, int, int, int]) -> None:
    weight = np.ones((2, 2), dtype=np.float32)
    assert doc.write_colour((0, 0, 2, 2), colour, weight)


def _clip(frames: int = 4, tracks: int = 1) -> Document:
    """A clip with a real cel on every slot, and nothing linked."""
    doc = Document.blank(4, 4)
    for _ in range(tracks - 1):
        doc.add_layer()
    for index in range(frames):
        if index:
            doc.add_frame()
        for track in range(tracks):
            doc.set_active_layer(track)
            _paint(doc, (10 * index + track, 0, 0, 255))
    doc.history.clear()
    return doc


#: Slots 1 and 3 are the same red, which is the whole point: a permutation
#: keeps them apart and a re-resolve collapses them onto the lower one.
BLACK = (0, 0, 0, 255)
INDEXED_PALETTE = [BLACK, RED, GREEN, RED]


def _indexed_clip(frames: int = 2) -> Document:
    """A clip in true indexed mode, with one pixel parked in the *other* red.

    By hand rather than by painting, ``test_true_indexed``'s way: this is the
    state an ``.aseprite`` import produces, and what the tests below ask is
    what a range op does to it.
    """
    doc = _clip(frames)
    assert doc.convert_to_indexed(INDEXED_PALETTE)
    for index in range(frames):
        cel = _cel(doc, 0, index)
        cel.indices[0, 3] = 3
        doc._rematerialize(cel)
    doc.check_materialized()
    doc.history.clear()
    return doc


def _uids(doc: Document) -> list[int]:
    return [frame.uid for frame in doc.anim.frames]


def _cel(doc: Document, track: int, frame: int):
    anim = doc.anim
    return anim.cels.get((anim.tracks[track].uid, anim.frames[frame].uid))


# --- the clamp ---------------------------------------------------------------


def test_a_span_is_sorted_clamped_and_may_miss_entirely():
    assert clamp_span(3, 1, 10) == (1, 3)
    assert clamp_span(-4, 2, 10) == (0, 2)
    assert clamp_span(8, 40, 10) == (8, 9)
    # Wholly past the end, and an empty grid: both are "no cells", not a crash.
    assert clamp_span(12, 40, 10) is None
    assert clamp_span(0, 0, 0) is None


# --- remove ------------------------------------------------------------------


def test_removing_a_span_is_one_step_and_undo_restores_the_same_frames():
    doc = _clip(5)
    before = _uids(doc)
    assert doc.remove_range(1, 3)
    assert len(doc.anim.frames) == 2
    assert _uids(doc) == [before[0], before[4]]
    assert len(doc.history) == 1

    assert doc.history.undo(doc)
    # The uids, not merely the count: a re-insert that minted fresh frames
    # would strand every cel key and every stamp recorded against them.
    assert _uids(doc) == before


def test_removing_the_whole_timeline_is_refused_and_pushes_nothing():
    doc = _clip(3)
    assert not doc.remove_range(0, 2)
    assert len(doc.history) == 0
    assert len(doc.anim.frames) == 3


def test_a_removed_range_restores_its_cels_and_their_links():
    doc = _clip(2)
    doc.add_frame(link=True)  # frame 2 shares frame 1's cels
    doc.history.clear()
    shared = _cel(doc, 0, 1)
    assert _cel(doc, 0, 2) is shared

    assert doc.remove_range(1, 2)
    assert doc.history.undo(doc)
    assert _cel(doc, 0, 1) is _cel(doc, 0, 2)
    assert _cel(doc, 0, 1) is shared


def test_a_linked_cel_removed_through_two_slots_is_charged_once():
    doc = _clip(2)
    doc.add_frame(link=True)
    doc.history.clear()
    plane = _cel(doc, 0, 1).pixels.nbytes

    assert doc.remove_range(1, 2)
    # Both frames held the same object, so the history is left holding one
    # plane -- charging twice would evict real steps for memory nobody has.
    assert doc.history.bytes == plane


def test_removing_a_range_charges_nothing_for_a_cel_still_linked_outside_it():
    doc = _clip(2)
    doc.add_frame(link=True)
    doc.history.clear()
    # Frames 1 and 2 share; only frame 2 goes, so the object stays alive.
    assert doc.remove_range(2, 2)
    assert doc.history.bytes == 0


# --- clear -------------------------------------------------------------------


def test_clearing_a_rect_empties_only_the_occupied_slots_it_covers():
    doc = _clip(3, tracks=2)
    assert doc.clear_range(0, 0, 0, 1)
    assert _cel(doc, 0, 0) is None
    assert _cel(doc, 0, 1) is None
    assert _cel(doc, 1, 0) is not None
    assert _cel(doc, 0, 2) is not None
    assert len(doc.history) == 1

    assert doc.history.undo(doc)
    assert _cel(doc, 0, 0) is not None
    assert _cel(doc, 0, 1) is not None


def test_clearing_an_empty_rect_pushes_nothing():
    doc = _clip(2)
    assert doc.clear_range(0, 0, 0, 1)
    head = doc.history.head
    assert not doc.clear_range(0, 0, 0, 1)
    assert doc.history.head == head


# --- flip and rotate ----------------------------------------------------------
#
# Whole cels, deliberately: the selection weights a *fill* and a *filter*, but a
# weighted permutation is not a thing -- half-mirroring a region would have to
# invent what goes in the half it did not mirror.


def test_flipping_a_range_is_one_step_and_leaves_the_rest_alone():
    doc = _clip(4)
    before = [_cel(doc, 0, i).pixels.copy() for i in range(4)]

    assert doc.flip_range("horizontal", 0, 0, 1, 2)
    for index in (1, 2):
        assert np.array_equal(_cel(doc, 0, index).pixels, before[index][:, ::-1])
    for index in (0, 3):
        assert np.array_equal(_cel(doc, 0, index).pixels, before[index])
    assert len(doc.history) == 1

    assert doc.history.undo(doc)
    for index in range(4):
        assert np.array_equal(_cel(doc, 0, index).pixels, before[index])


def test_a_linked_cel_is_flipped_once_however_many_slots_hold_it():
    """Twice the flip is not the flip -- it is the identity, which is worse:
    the op would silently do nothing at all on a linked span of even length."""
    doc = _clip(1)
    doc.add_frame(link=True)
    doc.add_frame(link=True)
    doc.history.clear()
    shared = _cel(doc, 0, 0)
    assert _cel(doc, 0, 2) is shared
    once = shared.pixels[:, ::-1].copy()

    assert doc.flip_range("horizontal", 0, 0, 0, 2)
    assert np.array_equal(shared.pixels, once)
    assert len(doc.history) == 1


def test_flipping_an_empty_range_pushes_nothing():
    doc = _clip(2)
    assert doc.clear_range(0, 0, 0, 1)
    doc.history.clear()
    assert not doc.flip_range("horizontal", 0, 0, 0, 1)
    assert len(doc.history) == 0


def test_flipping_a_range_that_changes_nothing_pushes_nothing():
    """A cel already symmetric about the axis. ``dirty`` is a comparison
    against the history head, so a step that changed nothing would make a
    saved document ask to be saved."""
    doc = _clip(2)
    for index in range(2):
        _cel(doc, 0, index).pixels[:, :] = RED
    doc.history.clear()
    assert not doc.flip_range("horizontal", 0, 0, 0, 1)
    assert len(doc.history) == 0


def test_flipping_a_range_commits_a_float_before_it_reads_the_cels():
    doc = _clip(2)
    assert doc.clear_range(0, 0, 1, 1)  # frame 1 is empty
    doc.set_current_frame(1)
    _float_over(doc)
    doc.history.clear()

    assert doc.flip_range("horizontal", 0, 0, 0, 1)
    assert doc.floating is None
    landed = _cel(doc, 0, 1)
    assert landed is not None
    # The commit's cel was flipped with the rest rather than missed: the green
    # square went down at the left and is now at the right.
    assert int(landed.pixels[0, 3, 1]) == 255


def test_flipping_an_indexed_range_is_an_exact_index_permutation():
    """Never a re-resolve. Slots 1 and 3 hold the same red, so a resolve would
    land the pixel on slot 1 and the duplicate identity would be gone -- and
    nothing on screen would change, which is what makes it silent."""
    doc = _indexed_clip(2)
    before = [_cel(doc, 0, i).indices.copy() for i in range(2)]

    assert doc.flip_range("horizontal", 0, 0, 0, 1)
    for index in range(2):
        assert np.array_equal(_cel(doc, 0, index).indices, before[index][:, ::-1])
        assert int(_cel(doc, 0, index).indices[0, 0]) == 3
    doc.check_materialized()
    assert len(doc.history) == 1

    assert doc.history.undo(doc)
    doc.check_materialized()
    assert np.array_equal(_cel(doc, 0, 0).indices, before[0])


def test_rotating_a_range_by_a_quarter_turns_every_cel_it_covers():
    doc = _clip(2)
    before = [_cel(doc, 0, i).pixels.copy() for i in range(2)]
    assert doc.rotate_range(1, 0, 0, 0, 1)
    for index in range(2):
        assert np.array_equal(
            _cel(doc, 0, index).pixels, np.rot90(before[index], 1)
        )
    assert len(doc.history) == 1


def test_rotating_a_range_on_a_non_square_canvas_is_refused_by_name():
    """A quarter turn transposes the plane, and a cel is a full-canvas array:
    turning one of them and not the canvas would leave the grid holding planes
    of two shapes. Refused rather than silently resized."""
    doc = _clip(2)
    doc.resize_canvas((4, 6))
    doc.history.clear()

    with pytest.raises(ValueError, match="square"):
        doc.rotate_range(1, 0, 0, 0, 1)
    assert len(doc.history) == 0


def test_rotating_a_range_by_180_works_on_any_canvas():
    doc = _clip(2)
    doc.resize_canvas((4, 6))
    doc.history.clear()
    before = _cel(doc, 0, 0).pixels.copy()

    assert doc.rotate_range(2, 0, 0, 0, 1)
    assert np.array_equal(_cel(doc, 0, 0).pixels, np.rot90(before, 2))


def test_rotating_a_range_by_a_whole_turn_pushes_nothing():
    doc = _clip(2)
    assert not doc.rotate_range(0, 0, 0, 0, 1)
    assert not doc.rotate_range(4, 0, 0, 0, 1)
    assert len(doc.history) == 0


def test_rotating_an_indexed_range_is_an_exact_index_permutation():
    doc = _indexed_clip(2)
    before = _cel(doc, 0, 0).indices.copy()
    assert doc.rotate_range(1, 0, 0, 0, 0)
    assert np.array_equal(_cel(doc, 0, 0).indices, np.rot90(before, 1))
    assert int(_cel(doc, 0, 0).indices.max()) == 3
    doc.check_materialized()


# --- shift ---------------------------------------------------------------------


def test_a_wrapped_shift_of_a_full_cycle_pushes_nothing():
    """The canvas is 4 wide, so shifting by 4 with wrap is the identity -- and
    an op that changes nothing must not move the history head."""
    doc = _clip(2)
    assert not doc.shift_range(4, 0, True, 0, 0, 0, 1)
    assert len(doc.history) == 0


def test_a_shift_by_nothing_pushes_nothing():
    doc = _clip(2)
    assert not doc.shift_range(0, 0, True, 0, 0, 0, 1)
    assert not doc.shift_range(0, 0, False, 0, 0, 0, 1)
    assert len(doc.history) == 0


def test_a_wrapped_shift_round_trips_exactly():
    doc = _clip(2)
    before = [_cel(doc, 0, i).pixels.copy() for i in range(2)]

    assert doc.shift_range(1, 2, True, 0, 0, 0, 1)
    assert not np.array_equal(_cel(doc, 0, 0).pixels, before[0])
    assert doc.shift_range(-1, -2, True, 0, 0, 0, 1)
    for index in range(2):
        assert np.array_equal(_cel(doc, 0, index).pixels, before[index])
    # Two gestures, two steps -- and the second is not a no-op, so it pushed.
    assert len(doc.history) == 2


def test_an_unwrapped_shift_vacates_with_transparent_pixels():
    doc = _clip(1)
    doc.add_frame()
    doc.history.clear()
    assert doc.shift_range(2, 0, False, 0, 0, 0, 0)
    assert (_cel(doc, 0, 0).pixels[:, :2, 3] == 0).all()


def test_shifting_an_indexed_range_vacates_with_the_transparent_index():
    """Not slot 0. They are the same slot only by coincidence, and a document
    whose transparent index is 7 would vacate into a band of slot-0 colour."""
    doc = _indexed_clip(2)
    assert doc.shift_range(1, 0, False, 0, 0, 0, 1)
    assert (_cel(doc, 0, 0).indices[:, 0] == doc.transparent_index).all()
    doc.check_materialized()


def test_shifting_an_indexed_range_with_wrap_keeps_the_duplicate_slot():
    doc = _indexed_clip(2)
    before = _cel(doc, 0, 0).indices.copy()
    assert doc.shift_range(1, 0, True, 0, 0, 0, 1)
    assert np.array_equal(_cel(doc, 0, 0).indices, np.roll(before, 1, axis=1))
    assert int(_cel(doc, 0, 0).indices.max()) == 3
    doc.check_materialized()


def test_shifting_a_linked_cel_touches_it_once():
    doc = _clip(1)
    doc.add_frame(link=True)
    doc.add_frame(link=True)
    doc.history.clear()
    shared = _cel(doc, 0, 0)
    once = np.roll(shared.pixels, 1, axis=1).copy()

    assert doc.shift_range(1, 0, True, 0, 0, 0, 2)
    assert np.array_equal(shared.pixels, once)
    assert len(doc.history) == 1


def test_an_unwrapped_shift_off_the_edge_is_still_one_undoable_step():
    doc = _clip(2)
    before = _cel(doc, 0, 0).pixels.copy()
    assert doc.shift_range(99, 0, False, 0, 0, 0, 1)
    assert _cel(doc, 0, 0).pixels.max() == 0
    assert doc.history.undo(doc)
    assert np.array_equal(_cel(doc, 0, 0).pixels, before)


# --- duplicate ---------------------------------------------------------------


def test_duplicating_a_span_lands_after_it_and_copies_pixels():
    doc = _clip(3)
    before = _uids(doc)
    assert doc.duplicate_range(0, 1)
    assert len(doc.anim.frames) == 5
    assert _uids(doc)[:2] == before[:2]
    assert _uids(doc)[4] == before[2]
    # An independent copy: painting the duplicate must not change the source.
    source, copy = _cel(doc, 0, 0), _cel(doc, 0, 2)
    assert copy is not source
    assert np.array_equal(copy.pixels, source.pixels)

    assert doc.history.undo(doc)
    assert _uids(doc) == before


def test_duplicating_preserves_internal_links_without_linking_to_the_source():
    doc = _clip(1)
    doc.add_frame(link=True)  # frames 0 and 1 share one cel
    doc.history.clear()
    source = _cel(doc, 0, 0)

    assert doc.duplicate_range(0, 1)
    left, right = _cel(doc, 0, 2), _cel(doc, 0, 3)
    # One copy for the pair -- the link inside the span survived -- and it is
    # not the source, so editing the duplicate leaves the original alone.
    assert left is right
    assert left is not source


def test_a_linked_duplicate_shares_the_source_objects_and_costs_nothing():
    doc = _clip(2)
    doc.history.clear()
    assert doc.duplicate_range(0, 1, link=True)
    assert _cel(doc, 0, 2) is _cel(doc, 0, 0)
    assert doc.history.bytes == 0


def test_a_copied_duplicate_charges_each_new_plane_once():
    doc = _clip(1)
    doc.add_frame(link=True)
    doc.history.clear()
    plane = _cel(doc, 0, 0).pixels.nbytes
    assert doc.duplicate_range(0, 1)
    assert doc.history.bytes == plane


def test_duplicating_at_an_explicit_index_reads_the_source_before_inserting():
    doc = _clip(3)
    third = _cel(doc, 0, 2)
    assert doc.duplicate_range(2, 2, at=0)
    assert len(doc.anim.frames) == 4
    assert np.array_equal(_cel(doc, 0, 0).pixels, third.pixels)
    assert _cel(doc, 0, 3) is third


# --- move and reverse --------------------------------------------------------


def test_moving_a_span_puts_its_first_frame_at_the_named_index():
    doc = _clip(5)
    before = _uids(doc)
    assert doc.move_range(3, 4, 0)
    assert _uids(doc) == [before[3], before[4], before[0], before[1], before[2]]
    assert len(doc.history) == 1
    assert doc.history.undo(doc)
    assert _uids(doc) == before


def test_reversing_a_span_reverses_only_it():
    doc = _clip(5)
    before = _uids(doc)
    assert doc.reverse_range(1, 3)
    assert _uids(doc) == [before[0], before[3], before[2], before[1], before[4]]
    assert doc.history.undo(doc)
    assert _uids(doc) == before


def test_a_reorder_that_changes_nothing_pushes_nothing():
    doc = _clip(3)
    assert not doc.move_range(0, 1, 0)
    assert not doc.reverse_range(2, 2)
    assert len(doc.history) == 0


def test_a_reorder_keeps_the_playhead_on_the_same_frame():
    doc = _clip(4)
    doc.set_current_frame(3)
    watched = doc.anim.frames[3].uid
    assert doc.move_range(3, 3, 0)
    assert doc.anim.frames[doc.anim.current].uid == watched
    assert doc.anim.current == 0


def test_a_reorder_moves_no_pixels_and_keeps_every_link():
    doc = _clip(2)
    doc.add_frame(link=True)
    doc.history.clear()
    shared = _cel(doc, 0, 1)
    assert doc.reverse_range(0, 2)
    # Cels are keyed by frame uid, so a permutation is free -- and the shared
    # object is still shared, wherever its two frames ended up.
    assert _cel(doc, 0, 0) is shared
    assert _cel(doc, 0, 1) is shared
    assert doc.history.bytes == 0


def test_an_op_after_a_reorder_lands_on_the_frames_now_at_those_indices():
    doc = _clip(4)
    assert doc.reverse_range(0, 3)
    watched = doc.anim.frames[0].uid
    assert doc.remove_range(0, 0)
    assert watched not in _uids(doc)


# --- link and unlink ---------------------------------------------------------


def test_linking_a_range_shares_the_earliest_cel_across_it():
    doc = _clip(3)
    first = _cel(doc, 0, 0)
    assert doc.link_range(0, 0, 0, 2)
    assert _cel(doc, 0, 1) is first
    assert _cel(doc, 0, 2) is first
    assert len(doc.history) == 1

    assert doc.history.undo(doc)
    assert _cel(doc, 0, 1) is not first
    assert _cel(doc, 0, 2) is not first


def test_linking_an_already_linked_range_pushes_nothing():
    doc = _clip(3)
    assert doc.link_range(0, 0, 0, 2)
    head = doc.history.head
    assert not doc.link_range(0, 0, 0, 2)
    assert doc.history.head == head


def test_linking_a_track_with_no_cels_in_the_span_does_nothing_to_it():
    doc = _clip(2, tracks=2)
    assert doc.clear_range(1, 1, 0, 1)
    assert not doc.link_range(1, 1, 0, 1)


def test_unlinking_a_range_mints_its_copies_once_so_redo_returns_them():
    doc = _clip(1)
    doc.add_frame(link=True)
    doc.add_frame(link=True)
    doc.history.clear()

    assert doc.unlink_range(0, 0, 1, 2)
    minted = (_cel(doc, 0, 1), _cel(doc, 0, 2))
    assert minted[0] is not minted[1]
    assert doc.history.undo(doc)
    assert doc.history.redo(doc)
    # The *same* objects, uids included: a redo that copied again would strand
    # every patch recorded against the first copies.
    assert (_cel(doc, 0, 1), _cel(doc, 0, 2)) == minted


def test_unlinking_charges_the_copies_it_minted_and_nothing_else():
    """The copies are new memory an undo leaves the history alone holding, so
    they are charged even though they are in the grid right now -- and the
    original they came from is charged once, or not at all while a slot outside
    the range is still pointing at it."""
    doc = _clip(1)
    doc.add_frame(link=True)
    doc.add_frame(link=True)
    doc.history.clear()
    plane = _cel(doc, 0, 0).pixels.nbytes

    # Frames 1 and 2 copied; frame 0 keeps the original, so it is not released.
    assert doc.unlink_range(0, 0, 1, 2)
    assert doc.history.bytes == 2 * plane


def test_unlinking_a_whole_link_also_charges_the_original_once():
    doc = _clip(1)
    doc.add_frame(link=True)
    doc.history.clear()
    plane = _cel(doc, 0, 0).pixels.nbytes

    # Both slots copied, so nothing points at the original any more: two new
    # planes plus the stranded one, charged once and not once per slot.
    assert doc.unlink_range(0, 0, 0, 1)
    assert doc.history.bytes == 3 * plane


def test_unlinking_a_range_with_nothing_shared_pushes_nothing():
    doc = _clip(3)
    assert not doc.unlink_range(0, 0, 0, 2)
    assert len(doc.history) == 0


# --- durations ---------------------------------------------------------------


def test_setting_a_range_duration_is_one_step_over_the_frames_that_change():
    doc = _clip(4)
    doc.set_frame_duration(1, 250)
    doc.history.clear()
    assert doc.set_range_duration(0, 2, 250)
    assert [f.duration_ms for f in doc.anim.frames] == [250, 250, 250, 100]
    assert len(doc.history) == 1

    assert doc.history.undo(doc)
    assert [f.duration_ms for f in doc.anim.frames] == [100, 250, 100, 100]


def test_setting_a_duration_every_frame_already_has_pushes_nothing():
    doc = _clip(3)
    assert not doc.set_range_duration(0, 2, 100)
    assert len(doc.history) == 0


# --- the cel clipboard -------------------------------------------------------


def test_a_clip_dedupes_planes_and_records_the_slots_that_share_them():
    doc = _clip(1)
    doc.add_frame(link=True)
    clip = doc.copy_cels(0, 0, 0, 1)
    assert clip.frames == 2
    assert len(clip.planes) == 1
    assert clip.slots == {(0, 0): 0, (0, 1): 0}


def test_a_clip_owns_its_pixels_and_does_not_follow_later_edits():
    doc = _clip(2)
    doc.set_current_frame(0)
    clip = doc.copy_cels(0, 0, 0, 0)
    taken = clip.planes[0].pixels.copy()
    _paint(doc, GREEN)
    assert np.array_equal(clip.planes[0].pixels, taken)


def test_pasting_a_clip_restores_its_internal_links_as_one_object():
    doc = _clip(1)
    doc.add_frame(link=True)
    clip = doc.copy_cels(0, 0, 0, 1)
    doc.add_frame()
    doc.add_frame()
    doc.history.clear()

    assert doc.paste_cels(clip, 0, 2)
    assert _cel(doc, 0, 2) is _cel(doc, 0, 3)
    # Pasted objects are fresh: editing them must not edit what was copied.
    assert _cel(doc, 0, 2) is not _cel(doc, 0, 0)
    assert len(doc.history) == 1


def test_a_pasted_shared_plane_is_charged_once():
    doc = _clip(2)
    doc.add_frame(link=True)  # frames 1 and 2 share one cel
    clip = doc.copy_cels(0, 0, 1, 2)
    doc.add_frame()
    doc.add_frame()
    doc.history.clear()

    # Onto empty slots, so the only pixels the history is left holding are the
    # pasted plane's -- and there is one of it, however many slots show it.
    assert doc.paste_cels(clip, 0, 3)
    assert doc.history.bytes == clip.planes[0].pixels.nbytes


def test_pasting_past_the_end_conjures_frames_inside_the_same_step():
    doc = _clip(3)
    clip = doc.copy_cels(0, 0, 0, 2)
    doc.history.clear()

    assert doc.paste_cels(clip, 0, 2)
    assert len(doc.anim.frames) == 5
    assert len(doc.history) == 1

    # One Ctrl+Z takes the cels *and* the frames they needed.
    assert doc.history.undo(doc)
    assert len(doc.anim.frames) == 3


def test_pasting_a_clip_from_another_size_is_refused():
    doc = _clip(2)
    clip = doc.copy_cels(0, 0, 0, 1)
    other = Document.blank(8, 8)
    other.add_frame()
    other.history.clear()
    assert not other.paste_cels(clip, 0, 0)
    assert len(other.history) == 0


def test_copying_a_rect_off_the_grid_is_none_and_pasting_nothing_is_refused():
    doc = _clip(2)
    assert doc.copy_cels(0, 0, 9, 12) is None
    assert not doc.paste_cels(None, 0, 0)


# --- the commit-first rule ---------------------------------------------------


def _float_over(doc: Document) -> None:
    """A floating buffer sitting over the current frame's active slot.

    Built by hand rather than through ``paste``: what matters is only that a
    commit is pending and that committing it will *autovivify* the cel under
    it, which is exactly what a paste onto an empty frame does.
    """
    pixels = np.zeros((2, 2, 4), dtype=np.uint8)
    pixels[:] = (0, 255, 0, 255)
    doc.floating = FloatingBuffer(
        pixels=pixels,
        mask=np.full((2, 2), 255, dtype=np.uint8),
        offset=(0, 0),
        layer_uid=doc.stack.active.uid,
    )


def test_clearing_a_range_commits_a_float_before_it_reads_the_slots():
    """Every op here is clamp -> commit_floating -> mutate, and this is the
    order that matters: committing autovivifies the cel it lands in, so a slot
    set read *first* misses it and leaves a cel behind in a cleared rect."""
    doc = _clip(2)
    assert doc.clear_range(0, 0, 1, 1)  # frame 1 is now empty
    doc.set_current_frame(1)
    _float_over(doc)

    assert doc.clear_range(0, 0, 0, 1)
    assert doc.floating is None
    assert _cel(doc, 0, 0) is None
    assert _cel(doc, 0, 1) is None


def test_linking_a_range_commits_a_float_before_it_picks_the_earliest_cel():
    doc = _clip(2)
    assert doc.clear_range(0, 0, 0, 0)  # frame 0 is now empty
    doc.set_current_frame(0)
    _float_over(doc)

    assert doc.link_range(0, 0, 0, 1)
    assert doc.floating is None
    # The cel the commit conjured is the earliest occupied one, so it is the
    # one shared across the span -- reading the slots first would have picked
    # frame 1's instead and thrown the new pixels away.
    assert _cel(doc, 0, 0) is _cel(doc, 0, 1)
    assert int(_cel(doc, 0, 0).pixels[0, 0, 1]) == 255


def test_setting_a_range_duration_commits_a_float_like_every_other_op():
    doc = _clip(2)
    _float_over(doc)
    assert doc.set_range_duration(0, 1, 250)
    assert doc.floating is None


def test_a_still_document_refuses_every_range_op():
    doc = Document.blank(4, 4)
    assert not doc.remove_range(0, 0)
    assert not doc.clear_range(0, 0, 0, 0)
    assert not doc.duplicate_range(0, 0)
    assert not doc.move_range(0, 0, 1)
    assert not doc.reverse_range(0, 0)
    assert not doc.link_range(0, 0, 0, 0)
    assert not doc.unlink_range(0, 0, 0, 0)
    assert not doc.set_range_duration(0, 0, 50)
    assert not doc.flip_range("horizontal", 0, 0, 0, 0)
    assert not doc.rotate_range(1, 0, 0, 0, 0)
    assert not doc.shift_range(1, 0, True, 0, 0, 0, 0)
    assert doc.copy_cels(0, 0, 0, 0) is None
    assert len(doc.history) == 0
