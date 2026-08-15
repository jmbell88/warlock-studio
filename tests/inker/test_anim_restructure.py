"""Merge-down and flatten on an animated document.

Both were refused outright for a version, and the reason was a real question:
what is a merge of a linked cel with an unlinked one? Everything below is the
answer, asserted the only way it can be -- with ``is``. A link in this editor is
two slots holding *one object*, so "the link survived" is an identity check and
nothing else; two equal copies pass a pixel comparison and are the bug.

The other half is what the history holds. A merge mints one ``Layer`` per
distinct pair of inputs, and several slots legitimately name it, so the byte
budget must count it once -- charging per slot evicts real history to make room
for a number describing memory that exists once.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker.document import Document

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
GREEN = (0, 255, 0, 255)


def _grid(frames: int = 3) -> Document:
    """A two-track, ``frames``-frame document with cels everywhere."""
    doc = Document.blank(8, 8)
    doc.stack[0].pixels[:, :] = RED
    doc.add_layer("Ink")
    doc.stack[1].pixels[0:4, 0:4] = BLUE
    # Animated before the frames are added, so a one-frame grid is still a
    # grid: ``add_frame`` is what animates a still document, and a test that
    # asked for one frame would otherwise get no ``anim`` at all.
    doc.ensure_animation()
    for _ in range(frames - 1):
        doc.add_frame(copy=True)
    doc.set_current_frame(0)
    return doc


def _cel(doc: Document, track: int, frame: int):
    anim = doc.anim
    assert anim is not None
    return anim.cels.get((anim.tracks[track].uid, anim.frames[frame].uid))


def _link(doc: Document, track: int, frames: tuple[int, ...]) -> None:
    """Point every named frame's slot at the first one's cel."""
    anim = doc.anim
    assert anim is not None
    source = _cel(doc, track, frames[0])
    for index in frames[1:]:
        anim.cels[(anim.tracks[track].uid, anim.frames[index].uid)] = source
    doc._anim_changed()


# --- merge down --------------------------------------------------------------


def test_a_merge_runs_across_every_frame():
    doc = _grid(3)
    assert doc.merge_down(1)
    anim = doc.anim
    assert anim is not None
    assert len(anim.tracks) == 1
    assert len(anim.frames) == 3
    for index in range(3):
        merged = _cel(doc, 0, index)
        assert merged is not None
        assert tuple(merged.pixels[1, 1]) == BLUE
        assert tuple(merged.pixels[6, 6]) == RED


def test_frames_sharing_both_cels_share_one_merged_cel():
    """The whole doctrine in one assertion. Frames 0 and 1 hold the same lower
    cel and the same upper cel, so they hold one merged cel afterwards."""
    doc = _grid(3)
    _link(doc, 0, (0, 1, 2))
    _link(doc, 1, (0, 1))

    assert doc.merge_down(1)

    assert _cel(doc, 0, 0) is _cel(doc, 0, 1)
    assert _cel(doc, 0, 0) is not _cel(doc, 0, 2)


def test_a_frame_whose_upper_differs_gets_its_own_merged_cel():
    doc = _grid(2)
    _link(doc, 0, (0, 1))
    # The upper row starts linked; unlink frame 1 and paint it differently.
    _link(doc, 1, (0, 1))
    doc.set_current_frame(1)
    assert doc.unlink_cel(1, 1)
    _cel(doc, 1, 1).pixels[4:8, 4:8] = GREEN

    assert doc.merge_down(1)

    assert _cel(doc, 0, 0) is not _cel(doc, 0, 1)
    assert tuple(_cel(doc, 0, 1).pixels[6, 6]) == GREEN
    assert tuple(_cel(doc, 0, 0).pixels[6, 6]) == RED


def test_a_slot_with_no_upper_cel_keeps_the_lower_cel_it_had():
    """Identity, not equality: minting a copy for a slot with nothing to merge
    into it would break the lower row's own links and cost a plane a frame."""
    doc = _grid(2)
    _link(doc, 0, (0, 1))
    lower = _cel(doc, 0, 0)
    doc.set_current_frame(1)
    assert doc.clear_cel(1, 1)

    assert doc.merge_down(1)

    assert _cel(doc, 0, 1) is lower
    assert _cel(doc, 0, 0) is not lower  # frame 0 really was merged


def test_the_lower_cels_are_never_mutated_in_place():
    """A lower cel may be linked across frames where the upper is not, so
    writing merged pixels into it pushes one frame's merge into another."""
    doc = _grid(2)
    _link(doc, 0, (0, 1))
    lower = _cel(doc, 0, 0)
    before = lower.pixels.copy()
    doc.set_current_frame(1)
    assert doc.clear_cel(1, 1)

    assert doc.merge_down(1)

    assert np.array_equal(lower.pixels, before)


def test_the_lower_tracks_opacity_and_blend_are_baked():
    doc = _grid(2)
    anim = doc.anim
    assert anim is not None
    anim.tracks[0].opacity = 0.5
    anim.tracks[0].blend = "multiply"

    assert doc.merge_down(1)

    assert anim.tracks[0].opacity == 1.0
    assert anim.tracks[0].blend == "normal"


def test_the_whole_merge_is_one_undo_step_and_it_restores_the_links():
    doc = _grid(3)
    _link(doc, 0, (0, 1, 2))
    _link(doc, 1, (0, 1))
    lower, upper = _cel(doc, 0, 0), _cel(doc, 1, 0)
    head = doc.history.head

    assert doc.merge_down(1)
    assert doc.history.head != head

    assert doc.undo()
    assert doc.history.head == head
    anim = doc.anim
    assert anim is not None
    assert len(anim.tracks) == 2
    # The same objects, in the same slots: one Ctrl+Z, and the link structure
    # is the one the document had.
    assert _cel(doc, 0, 0) is lower
    assert _cel(doc, 0, 1) is lower
    assert _cel(doc, 0, 2) is lower
    assert _cel(doc, 1, 0) is upper
    assert _cel(doc, 1, 1) is upper


def test_a_redo_lands_on_the_same_merged_objects():
    """The merged cels are minted once at op time and held by the edit. A redo
    that merged again would hand back new identities and strand every patch
    recorded above."""
    doc = _grid(2)
    _link(doc, 0, (0, 1))
    _link(doc, 1, (0, 1))
    assert doc.merge_down(1)
    merged = _cel(doc, 0, 0)
    uid = merged.uid

    assert doc.undo()
    assert doc.redo()

    assert _cel(doc, 0, 0) is merged
    assert _cel(doc, 0, 1) is merged
    assert _cel(doc, 0, 0).uid == uid


def test_a_shared_merged_cel_is_charged_for_once():
    """Two slots, one merged object. Charging per slot would spend the byte
    budget twice on memory that exists once."""
    doc = _grid(2)
    _link(doc, 0, (0, 1))
    _link(doc, 1, (0, 1))
    plane = doc.stack[0].pixels.nbytes
    doc.history._done.clear()
    doc.history._undone.clear()

    assert doc.merge_down(1)

    # Three planes and no more: the merged one, the single lower cel it
    # consumed, and the single upper cel the track removal released. The second
    # slot's ``CelSetEdit`` introduces none of them and charges nothing --
    # without that it would charge for the first two a second time.
    assert doc.history.bytes == 3 * plane


def test_a_merge_with_nothing_in_the_upper_row_is_just_a_removal():
    doc = _grid(2)
    assert doc.clear_cel(1, 0)
    doc.set_current_frame(1)
    assert doc.clear_cel(1, 1)
    lower = _cel(doc, 0, 0)

    assert doc.merge_down(1)

    anim = doc.anim
    assert anim is not None
    assert len(anim.tracks) == 1
    assert _cel(doc, 0, 0) is lower


def test_a_hidden_upper_track_contributes_nothing():
    doc = _grid(1)
    doc.set_layer_props(1, visible=False)
    assert doc.merge_down(1)
    assert tuple(_cel(doc, 0, 0).pixels[1, 1]) == RED


# --- flatten -----------------------------------------------------------------


def test_flatten_collapses_the_grid_to_one_track():
    doc = _grid(3)
    doc.flatten_layers()
    anim = doc.anim
    assert anim is not None
    assert len(anim.tracks) == 1
    assert len(anim.frames) == 3
    assert len(doc.stack) == 1
    for index in range(3):
        assert tuple(_cel(doc, 0, index).pixels[1, 1]) == BLUE


def test_flatten_partitions_frames_by_the_cels_they_hold():
    """Two frames whose whole stacks are the same objects flatten to the same
    picture, so they share one cel; a frame that differs gets its own."""
    doc = _grid(3)
    _link(doc, 0, (0, 1, 2))
    _link(doc, 1, (0, 1))
    doc.set_current_frame(2)
    _cel(doc, 1, 2).pixels[4:8, 4:8] = GREEN

    doc.flatten_layers()

    assert _cel(doc, 0, 0) is _cel(doc, 0, 1)
    assert _cel(doc, 0, 0) is not _cel(doc, 0, 2)


def test_flatten_leaves_durations_and_tags_alone():
    doc = _grid(2)
    assert doc.set_frame_duration(0, 250)
    assert doc.add_tag("walk", 0, 1)

    doc.flatten_layers()

    anim = doc.anim
    assert anim is not None
    assert anim.frames[0].duration_ms == 250
    assert [(tag.name, tag.start, tag.end) for tag in anim.tags] == [("walk", 0, 1)]


def test_undoing_a_flatten_restores_the_link_structure():
    doc = _grid(3)
    _link(doc, 0, (0, 1, 2))
    background = _cel(doc, 0, 0)
    ink = _cel(doc, 1, 0)

    doc.flatten_layers()
    assert doc.undo()

    anim = doc.anim
    assert anim is not None
    assert len(anim.tracks) == 2
    # ``ReplayEdit`` restores by slot *index* into its snapshot list, so the
    # three slots that held one object hold one object again.
    assert _cel(doc, 0, 0) is _cel(doc, 0, 1) is _cel(doc, 0, 2)
    assert np.array_equal(_cel(doc, 0, 0).pixels, background.pixels)
    assert np.array_equal(_cel(doc, 1, 0).pixels, ink.pixels)


def test_a_redone_flatten_lands_on_the_same_uids():
    """Replay must be a pure function of the document, and both the uids and
    the frame partition are the parts that are not -- so both are drawn once,
    here, and closed over."""
    doc = _grid(3)
    _link(doc, 0, (0, 1, 2))
    _link(doc, 1, (0, 1))
    doc.flatten_layers()
    uids = [_cel(doc, 0, i).uid for i in range(3)]
    track_uid = doc.anim.tracks[0].uid

    assert doc.undo()
    assert doc.redo()

    assert [_cel(doc, 0, i).uid for i in range(3)] == uids
    assert doc.anim.tracks[0].uid == track_uid
    # And the partition came back with it, rather than being recomputed against
    # objects the snapshot restore replaced.
    assert _cel(doc, 0, 0) is _cel(doc, 0, 1)


def test_flatten_on_a_one_track_grid_pushes_nothing():
    doc = _grid(2)
    doc.merge_down(1)
    head = doc.history.head
    doc.flatten_layers()
    assert doc.history.head == head


def test_a_flattened_grid_still_stamps_every_frame():
    """Every cel in the document was replaced, which ``invalidate_all``
    deliberately does not assume -- so a stale cached flatten would show the
    unflattened frame in an onion skin."""
    doc = _grid(2)
    before = [doc.frame_stamp(frame.uid) for frame in doc.anim.frames]
    doc.flatten_layers()
    after = [doc.frame_stamp(frame.uid) for frame in doc.anim.frames]
    assert all(b < a for b, a in zip(before, after, strict=True))
