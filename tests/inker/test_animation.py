"""The animation grid, headless.

Phase 1's question is narrow and it is the one everything later rests on: can a
``Document`` present one frame of a grid as an ordinary ``LayerStack`` without
any of the machinery around it noticing, and can an edit recorded on one frame
still be undone once the playhead has moved somewhere else.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import animation
from warlock.studio.inker.animation import Animation, Frame, Tag, Track
from warlock.studio.inker.document import Document
from warlock.studio.inker.layers import Layer

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def _doc(width: int = 8, height: int = 8) -> Document:
    return Document.blank(width, height)


def _paint(doc: Document, colour: tuple[int, int, int, int]) -> None:
    """Write a solid square into the active layer, as one undo step."""
    weight = np.ones((4, 4), dtype=np.float32)
    assert doc.write_colour((0, 0, 4, 4), colour, weight)


def _add_frame(doc: Document, *, cel: bool = True) -> Frame:
    """A second frame, built by hand -- the edits that do this land in phase 2."""
    anim = doc.ensure_animation()
    frame = Frame()
    anim.frames.append(frame)
    if cel:
        width, height = doc.size
        for track in anim.tracks:
            anim.cels[(track.uid, frame.uid)] = Layer.empty(width, height, track.name)
    return frame


# --- becoming animated -------------------------------------------------------


def test_a_fresh_document_is_not_animated_and_nothing_below_changes():
    doc = _doc()
    assert doc.anim is None
    assert not doc.animated
    # The still path must not pay for the grid: no frame is stamped, because
    # there are no frames.
    doc.invalidate_all()
    assert doc._frame_stamps == {}


def test_becoming_animated_renumbers_nothing_and_copies_nothing():
    """The layers *become* the first frame's cels, identity and all.

    This is what lets a patch already on the undo stack keep working across the
    conversion, and it is why ``Track.of`` shares the layer's uid rather than
    minting one and inventing a mapping.
    """
    doc = _doc()
    doc.add_layer("Ink")
    originals = list(doc.stack)
    uids = [layer.uid for layer in originals]

    anim = doc.ensure_animation()

    assert [track.uid for track in anim.tracks] == uids
    assert len(anim.frames) == 1
    frame = anim.frames[0]
    for track, layer in zip(anim.tracks, originals, strict=True):
        assert anim.cels[(track.uid, frame.uid)] is layer


def test_ensure_animation_is_idempotent():
    doc = _doc()
    first = doc.ensure_animation()
    assert doc.ensure_animation() is first


def test_dropping_the_animation_leaves_the_visible_stack_alone():
    """The undo hook for the first "Add frame" -- one Ctrl+Z back to a still
    document, showing exactly the pixels that were on screen."""
    doc = _doc()
    _paint(doc, RED)
    before = doc.stack[0].pixels.copy()
    doc.ensure_animation()
    doc.drop_animation()

    assert doc.anim is None
    assert np.array_equal(doc.stack[0].pixels, before)
    assert doc._frame_stamps == {}


# --- materialisation ---------------------------------------------------------


def test_materialising_a_frame_preserves_the_active_index():
    doc = _doc()
    doc.add_layer("Ink")
    doc.add_layer("Shade")
    doc.set_active_layer(1)
    frame = _add_frame(doc)

    doc.set_current_frame(doc.anim.frame_index(frame.uid))

    assert doc.stack.active_index == 1
    assert len(doc.stack) == 3


def test_a_track_is_authoritative_over_the_cel_it_materialises():
    """Properties live on the row, once.

    A cel that kept its own opacity would composite differently on the frame it
    was drawn on than on a frame it was linked into -- the same layer, two
    answers.
    """
    doc = _doc()
    frame = _add_frame(doc)
    anim = doc.anim
    track = anim.tracks[0]
    track.name, track.opacity, track.visible, track.blend = "Ink", 0.25, False, "multiply"
    cel = anim.cels[(track.uid, frame.uid)]
    cel.opacity, cel.visible, cel.blend = 1.0, True, "normal"

    doc.set_current_frame(anim.frame_index(frame.uid))

    live = doc.stack[0]
    assert live is cel
    assert (live.name, live.opacity, live.visible, live.blend) == (
        "Ink",
        0.25,
        False,
        "multiply",
    )


def test_an_absent_cel_materialises_as_a_transparent_placeholder():
    doc = _doc()
    frame = _add_frame(doc, cel=False)
    doc.set_current_frame(doc.anim.frame_index(frame.uid))

    layer = doc.stack[0]
    assert layer.pixels.shape == (8, 8, 4)
    assert not layer.pixels.any()


def test_every_placeholder_shares_one_read_only_plane():
    """Sharing is what makes an empty grid free; read-only is what makes the
    sharing safe. A stray write would otherwise appear in every empty cel in
    the document at once, with nothing to say where it came from."""
    doc = _doc(64, 64)
    frame = _add_frame(doc, cel=False)
    doc.set_current_frame(doc.anim.frame_index(frame.uid))

    plane = doc.stack[0].pixels
    assert not plane.flags.writeable
    with pytest.raises(ValueError):
        plane[0, 0] = 1


def test_a_placeholder_keeps_its_uid_across_materialisations():
    doc = _doc()
    frame = _add_frame(doc, cel=False)
    index = doc.anim.frame_index(frame.uid)

    doc.set_current_frame(index)
    first = doc.stack[0].uid
    doc.set_current_frame(0)
    doc.set_current_frame(index)

    assert doc.stack[0].uid == first


def test_moving_the_playhead_pushes_no_history_step():
    """The playhead is view state, exactly as the active layer is.

    ``InkerDoc.dirty`` is a comparison against ``history.head``, so a step
    pushed here would make a document ask to be saved because the user looked
    at another frame.
    """
    doc = _doc()
    frame = _add_frame(doc)
    head = doc.history.head

    doc.set_current_frame(doc.anim.frame_index(frame.uid))
    doc.set_current_frame(0)

    assert doc.history.head == head


# --- linked cels -------------------------------------------------------------


def test_a_linked_cel_is_yielded_once_by_unique_cel_layers():
    """The double-transform guard. A rotate that walked slots instead would
    rotate a three-frame background three times."""
    doc = _doc()
    second = _add_frame(doc, cel=False)
    third = _add_frame(doc, cel=False)
    anim = doc.anim
    track = anim.tracks[0]
    shared = anim.cels[(track.uid, anim.frames[0].uid)]
    anim.cels[(track.uid, second.uid)] = shared
    anim.cels[(track.uid, third.uid)] = shared

    assert [id(layer) for layer in anim.unique_cel_layers()] == [id(shared)]
    assert anim.is_linked(track.uid, second.uid)
    assert len(anim.slots_of(shared)) == 3


def test_an_edit_to_a_linked_cel_is_visible_on_every_frame_it_occupies():
    """No propagation step: the frames hold the same object, so there is
    nothing to propagate and nothing to forget to propagate."""
    doc = _doc()
    second = _add_frame(doc, cel=False)
    anim = doc.anim
    track = anim.tracks[0]
    shared = anim.cels[(track.uid, anim.frames[0].uid)]
    anim.cels[(track.uid, second.uid)] = shared

    _paint(doc, RED)

    doc.set_current_frame(anim.frame_index(second.uid))
    assert tuple(doc.stack[0].pixels[0, 0]) == RED


def test_an_unlinked_cel_is_independent():
    doc = _doc()
    second = _add_frame(doc)
    anim = doc.anim
    _paint(doc, RED)

    doc.set_current_frame(anim.frame_index(second.uid))

    assert not doc.stack[0].pixels.any()


# --- undo across frames ------------------------------------------------------


def test_a_uid_resolves_to_a_cel_the_current_frame_does_not_show():
    doc = _doc()
    second = _add_frame(doc)
    anim = doc.anim
    off_frame = anim.cels[(anim.tracks[0].uid, second.uid)]

    assert doc.layer_by_uid(off_frame.uid) is off_frame


def test_an_unknown_uid_still_raises():
    doc = _doc()
    _add_frame(doc)
    with pytest.raises(KeyError):
        doc.layer_by_uid(-1)


def test_invalidating_an_off_frame_layer_stamps_rather_than_raising():
    """This is the line that used to be a ``KeyError`` out of the middle of an
    undo: ``LayerStack.index_of`` raises, and ``invalidate`` reached it for any
    patch addressed to a cel that is not on screen."""
    doc = _doc()
    second = _add_frame(doc)
    anim = doc.anim
    off_frame = anim.cels[(anim.tracks[0].uid, second.uid)]
    before = doc.frame_stamp(second.uid)

    doc.invalidate((0, 0, 4, 4), layer_uid=off_frame.uid)

    assert doc.frame_stamp(second.uid) > before


def test_undo_lands_on_the_right_cel_after_the_playhead_moved():
    """The whole point of addressing a patch by uid, extended one dimension.

    Paint on frame two, walk back to frame one, undo. The pixels that come back
    are frame two's, and frame one is untouched.
    """
    doc = _doc()
    second = _add_frame(doc)
    anim = doc.anim
    doc.set_current_frame(anim.frame_index(second.uid))
    _paint(doc, BLUE)
    painted = anim.cels[(anim.tracks[0].uid, second.uid)]
    assert tuple(painted.pixels[0, 0]) == BLUE

    doc.set_current_frame(0)
    assert doc.undo()

    assert not painted.pixels.any()
    assert not doc.stack[0].pixels.any()


def test_redo_after_moving_the_playhead_lands_there_too():
    doc = _doc()
    second = _add_frame(doc)
    anim = doc.anim
    doc.set_current_frame(anim.frame_index(second.uid))
    _paint(doc, BLUE)
    doc.set_current_frame(0)
    doc.undo()

    assert doc.redo()

    assert tuple(anim.cels[(anim.tracks[0].uid, second.uid)].pixels[0, 0]) == BLUE


# --- stamps ------------------------------------------------------------------


def test_a_write_stamps_exactly_the_frames_carrying_the_edited_cel():
    doc = _doc()
    linked = _add_frame(doc, cel=False)
    lone = _add_frame(doc)
    anim = doc.anim
    track = anim.tracks[0]
    shared = anim.cels[(track.uid, anim.frames[0].uid)]
    anim.cels[(track.uid, linked.uid)] = shared
    before = {f.uid: doc.frame_stamp(f.uid) for f in anim.frames}

    _paint(doc, RED)

    assert doc.frame_stamp(anim.frames[0].uid) > before[anim.frames[0].uid]
    assert doc.frame_stamp(linked.uid) > before[linked.uid]
    assert doc.frame_stamp(lone.uid) == before[lone.uid]


def test_a_track_property_stamps_every_frame():
    """A track is authoritative over every frame it appears in, so hiding one
    changes every cached flatten in the document."""
    doc = _doc()
    second = _add_frame(doc)
    anim = doc.anim
    before = {f.uid: doc.frame_stamp(f.uid) for f in anim.frames}

    assert doc.set_layer_props(0, visible=False)

    assert doc.frame_stamp(anim.frames[0].uid) > before[anim.frames[0].uid]
    assert doc.frame_stamp(second.uid) > before[second.uid]


def test_recompositing_the_view_stamps_nothing():
    """``invalidate_all`` is about the composite on screen, and most of what
    reaches it changes no pixels -- switching layer, switching frame, rebuilding
    after an edit that already stamped what it touched. Stamping there looked
    conservative and was the opposite of a cache: every step of the playhead
    threw away every frame's flatten."""
    doc = _doc()
    second = _add_frame(doc)
    anim = doc.anim
    before = {f.uid: doc.frame_stamp(f.uid) for f in anim.frames}

    doc.invalidate_all()
    doc.set_active_layer(0)
    doc.set_current_frame(anim.frame_index(second.uid))
    doc.set_current_frame(0)

    assert {f.uid: doc.frame_stamp(f.uid) for f in anim.frames} == before


# --- the small pieces --------------------------------------------------------


def test_a_frames_duration_is_clamped_at_construction():
    assert Frame(duration_ms=0).duration_ms == 1
    assert Frame(duration_ms=10**9).duration_ms == 60_000
    assert Frame(duration_ms="nonsense").duration_ms == Frame().duration_ms


def test_an_animations_total_duration_is_the_sum_of_its_frames():
    anim = Animation(frames=[Frame(duration_ms=40), Frame(duration_ms=60)])
    assert anim.duration_ms() == 100


def test_the_playhead_is_clamped_rather_than_raising():
    anim = Animation(tracks=[Track()], frames=[Frame(), Frame()])
    anim.current = 99
    assert anim.frame is anim.frames[1]
    assert anim.current == 1


def test_a_tag_is_an_inclusive_index_range():
    tag = Tag(name="walk", start=2, end=5)
    assert (tag.start, tag.end, tag.loop) == (2, 5, True)


# --- frames as edits ---------------------------------------------------------


def test_the_first_added_frame_undoes_all_the_way_back_to_a_still_document():
    """One Ctrl+Z, not two.

    Animating and adding the frame are one ``CompoundEdit``; without that, the
    first undo leaves a one-frame animation the user never asked for and the
    second one is needed to get rid of it.
    """
    doc = _doc()
    _paint(doc, RED)
    doc.add_frame()

    assert doc.animated and len(doc.anim.frames) == 2
    assert doc.undo()

    assert doc.anim is None
    assert tuple(doc.stack[0].pixels[0, 0]) == RED


def test_redoing_the_first_frame_restores_the_same_animation_object():
    """``AnimateEdit`` carries the animation rather than rebuilding it: a redo
    that re-derived one would mint a fresh frame uid and strand every cel key
    and stamp that names the old one."""
    doc = _doc()
    doc.add_frame()
    anim = doc.anim
    doc.undo()

    assert doc.redo()

    assert doc.anim is anim
    assert len(doc.anim.frames) == 2


def test_a_blank_frame_carries_no_cels():
    doc = _doc()
    _paint(doc, RED)
    frame = doc.add_frame()

    assert doc.anim.cel(doc.anim.tracks[0].uid, frame.uid) is None
    assert not doc.stack[0].pixels.any()


def test_a_linked_duplicate_shares_the_object_and_costs_nothing():
    doc = _doc()
    _paint(doc, RED)
    source = doc.stack[0]
    before_bytes = doc.history.bytes

    frame = doc.add_frame(link=True)

    assert doc.anim.cel(doc.anim.tracks[0].uid, frame.uid) is source
    assert doc.history.bytes == before_bytes


def test_a_copied_duplicate_is_independent_and_is_charged_for():
    doc = _doc()
    _paint(doc, RED)
    source = doc.stack[0]
    before_bytes = doc.history.bytes

    frame = doc.add_frame(copy=True)
    cel = doc.anim.cel(doc.anim.tracks[0].uid, frame.uid)

    assert cel is not source
    assert np.array_equal(cel.pixels, source.pixels)
    assert doc.history.bytes >= before_bytes + source.pixels.nbytes


def test_the_last_frame_cannot_be_removed():
    doc = _doc()
    doc.ensure_animation()
    assert not doc.remove_frame()


def test_removing_a_frame_and_undoing_it_puts_its_cels_back():
    doc = _doc()
    doc.add_frame(copy=True)
    doc.set_current_frame(1)
    _paint(doc, BLUE)
    cel = doc.anim.cel(doc.anim.tracks[0].uid, doc.anim.frames[1].uid)

    assert doc.remove_frame(1)
    assert len(doc.anim.frames) == 1
    assert doc.undo()

    assert doc.anim.cel(doc.anim.tracks[0].uid, doc.anim.frames[1].uid) is cel


def test_moving_a_frame_reorders_the_timeline_and_undoes():
    doc = _doc()
    second = doc.add_frame()
    third = doc.add_frame()
    order = [f.uid for f in doc.anim.frames]

    assert doc.move_frame(2, 0)
    assert [f.uid for f in doc.anim.frames] == [third.uid, order[0], second.uid]

    assert doc.undo()
    assert [f.uid for f in doc.anim.frames] == order


def test_an_undo_after_a_frame_reorder_still_lands_on_the_right_cel():
    """Frames are addressed by uid for exactly this. The patch was recorded
    against a cel, not against "frame two"."""
    doc = _doc()
    doc.add_frame()
    doc.set_current_frame(1)
    _paint(doc, BLUE)
    cel = doc.anim.cel(doc.anim.tracks[0].uid, doc.anim.frames[1].uid)
    doc.move_frame(1, 0)

    assert doc.undo()  # the move
    assert doc.undo()  # the paint

    assert not cel.pixels.any()


def test_a_duration_change_is_one_undoable_step():
    doc = _doc()
    doc.ensure_animation()
    before = doc.anim.frames[0].duration_ms

    assert doc.set_frame_duration(0, 250)
    assert doc.anim.frames[0].duration_ms == 250
    assert not doc.set_frame_duration(0, 250)

    assert doc.undo()
    assert doc.anim.frames[0].duration_ms == before


# --- autovivify --------------------------------------------------------------


def test_drawing_on_an_empty_frame_is_one_undo_step():
    """The cel that had to exist for the stroke to land and the stroke itself
    undo together -- a user who draws once presses Ctrl+Z once."""
    doc = _doc()
    frame = doc.add_frame()
    track = doc.anim.tracks[0]
    depth = len(doc.history)

    _paint(doc, RED)

    assert doc.anim.cel(track.uid, frame.uid) is not None
    assert len(doc.history) == depth + 1

    assert doc.undo()
    assert doc.anim.cel(track.uid, frame.uid) is None


def test_an_autovivified_cel_keeps_the_placeholder_uid():
    doc = _doc()
    doc.add_frame()
    placeholder_uid = doc.stack[0].uid

    _paint(doc, RED)

    assert doc.stack[0].uid == placeholder_uid


def test_a_write_that_changes_nothing_leaves_no_cel_and_no_step():
    """The "a step that changes nothing is not pushed" rule, extended to the
    cel: a brush-down with no drag must not make the document dirty."""
    doc = _doc()
    frame = doc.add_frame()
    track = doc.anim.tracks[0]
    head = doc.history.head

    weight = np.zeros((4, 4), dtype=np.float32)
    doc.write_colour((0, 0, 4, 4), RED, weight)

    assert doc.anim.cel(track.uid, frame.uid) is None
    assert doc.history.head == head


def test_a_brush_down_with_no_dab_leaves_no_cel():
    doc = _doc()
    frame = doc.add_frame()
    track = doc.anim.tracks[0]
    head = doc.history.head

    doc.begin_stroke((2.0, 2.0), RED, size=4)
    doc._stroke.dirty = None
    doc.end_stroke()

    assert doc.anim.cel(track.uid, frame.uid) is None
    assert doc.history.head == head


def test_committing_a_paste_onto_an_empty_cel_works_and_is_one_step():
    doc = _doc()
    _paint(doc, RED)
    doc.select_all()
    doc.copy()
    frame = doc.add_frame()
    track = doc.anim.tracks[0]
    depth = len(doc.history)

    assert doc.paste((0, 0))
    assert doc.commit_floating()

    assert doc.anim.cel(track.uid, frame.uid) is not None
    assert len(doc.history) == depth + 1
    assert doc.undo()
    assert doc.anim.cel(track.uid, frame.uid) is None


# --- cels --------------------------------------------------------------------


def test_clearing_a_cel_is_undoable():
    doc = _doc()
    _paint(doc, RED)
    doc.ensure_animation()
    track, frame = doc.anim.tracks[0], doc.anim.frames[0]
    cel = doc.anim.cel(track.uid, frame.uid)

    assert doc.clear_cel()
    assert doc.anim.cel(track.uid, frame.uid) is None
    assert not doc.stack[0].pixels.any()

    assert doc.undo()
    assert doc.anim.cel(track.uid, frame.uid) is cel


def test_linking_a_slot_points_it_at_another_frames_cel():
    doc = _doc()
    _paint(doc, RED)
    source = doc.stack[0]
    doc.add_frame()
    doc.set_current_frame(1)

    assert doc.link_cel(0)

    assert doc.anim.cel(doc.anim.tracks[0].uid, doc.anim.frames[1].uid) is source
    assert tuple(doc.stack[0].pixels[0, 0]) == RED


def test_unlinking_mints_its_copy_once_so_the_uid_survives_undo_and_redo():
    """The ``flatten_layers`` rule, one level up. A redo that copied again
    would hand back a layer with a new identity and strand every patch recorded
    against the first one."""
    doc = _doc()
    _paint(doc, RED)
    doc.add_frame(link=True)
    doc.set_current_frame(1)

    assert doc.unlink_cel()
    copy = doc.anim.cel(doc.anim.tracks[0].uid, doc.anim.frames[1].uid)
    uid = copy.uid

    doc.undo()
    doc.redo()

    again = doc.anim.cel(doc.anim.tracks[0].uid, doc.anim.frames[1].uid)
    assert again is copy
    assert again.uid == uid


def test_an_unlinked_cel_stops_following_the_one_it_came_from():
    doc = _doc()
    _paint(doc, RED)
    doc.add_frame(link=True)
    doc.set_current_frame(1)
    doc.unlink_cel()

    _paint(doc, BLUE)

    doc.set_current_frame(0)
    assert tuple(doc.stack[0].pixels[0, 0]) == RED


def test_unlinking_an_unlinked_cel_does_nothing():
    doc = _doc()
    _paint(doc, RED)
    doc.add_frame(copy=True)
    doc.set_current_frame(1)
    head = doc.history.head

    assert not doc.unlink_cel()
    assert doc.history.head == head


# --- restructuring is refused ------------------------------------------------


def test_merge_and_flatten_are_refused_on_an_animated_document():
    doc = _doc()
    doc.add_layer("Ink")
    assert doc.can_restructure

    doc.ensure_animation()

    assert not doc.can_restructure
    assert not doc.merge_down(1)
    head = doc.history.head
    doc.flatten_layers()
    assert doc.history.head == head
    assert len(doc.stack) == 2


# --- layer ops become track ops ----------------------------------------------


def test_adding_a_layer_adds_a_track_with_no_cels():
    doc = _doc()
    doc.add_frame()

    doc.add_layer("Ink")

    assert len(doc.anim.tracks) == 2
    assert doc.anim.tracks[1].name == "Ink"
    # Sparse: "empty on every frame" is the absence of keys, not a grid of
    # blank layers.
    assert not any(key[0] == doc.anim.tracks[1].uid for key in doc.anim.cels)
    assert len(doc.stack) == 2

    assert doc.undo()
    assert len(doc.anim.tracks) == 1


def test_duplicating_a_layer_copies_that_tracks_cels_on_every_frame():
    doc = _doc()
    _paint(doc, RED)
    doc.add_frame(copy=True)

    doc.duplicate_layer(0)

    copy_track = doc.anim.tracks[1]
    for frame in doc.anim.frames:
        cel = doc.anim.cel(copy_track.uid, frame.uid)
        assert cel is not None
        assert tuple(cel.pixels[0, 0]) == RED
    # Copies, not links -- painting the duplicate must not paint the original.
    assert doc.anim.cel(copy_track.uid, doc.anim.frames[0].uid) is not doc.anim.cel(
        doc.anim.tracks[0].uid, doc.anim.frames[0].uid
    )


def test_removing_a_layer_takes_its_cels_and_puts_them_back():
    doc = _doc()
    _paint(doc, RED)
    doc.add_layer("Ink")
    doc.add_frame(copy=True)
    track = doc.anim.tracks[0]
    cels = {f.uid: doc.anim.cel(track.uid, f.uid) for f in doc.anim.frames}

    assert doc.remove_layer(0)
    assert not any(key[0] == track.uid for key in doc.anim.cels)

    assert doc.undo()
    for frame_uid, cel in cels.items():
        assert doc.anim.cel(track.uid, frame_uid) is cel


def test_the_last_track_cannot_be_removed():
    doc = _doc()
    doc.ensure_animation()
    assert not doc.remove_layer(0)


def test_moving_a_layer_reorders_the_tracks():
    doc = _doc()
    doc.add_layer("Ink")
    doc.ensure_animation()
    order = [t.uid for t in doc.anim.tracks]

    assert doc.move_layer(0, 1)
    assert [t.uid for t in doc.anim.tracks] == order[::-1]

    assert doc.undo()
    assert [t.uid for t in doc.anim.tracks] == order


def test_a_layer_property_is_written_to_the_track_and_survives_a_frame_switch():
    """Onto the layer instead, it would last exactly until the next time that
    frame was rebuilt."""
    doc = _doc()
    doc.add_frame(copy=True)

    assert doc.set_layer_props(0, opacity=0.5)

    doc.set_current_frame(1)
    assert doc.stack[0].opacity == 0.5
    doc.set_current_frame(0)
    assert doc.stack[0].opacity == 0.5
    assert doc.anim.tracks[0].opacity == 0.5

    assert doc.undo()
    assert doc.anim.tracks[0].opacity == 1.0


# --- whole-canvas geometry over the grid -------------------------------------


def test_a_linked_cel_is_rotated_exactly_once():
    """Walking the slots instead of the distinct cels rotates a background
    linked across three frames three times -- and it looks plausible, because
    every frame agrees with every other."""
    from warlock.studio.inker import transform as tf

    doc = _doc(8, 4)
    weight = np.ones((2, 1), dtype=np.float32)
    doc.write_colour((0, 0, 1, 2), RED, weight)
    doc.add_frame(link=True)
    doc.add_frame(link=True)
    original = doc.stack[0].pixels.copy()

    doc.rotate90(1)

    assert np.array_equal(doc.stack[0].pixels, tf.rotate90(original, 1))


def test_undoing_a_rotation_restores_the_link_structure():
    """Restoring copies per slot would give two equal layers where the document
    had one shared object, and the break would only show up on the next stroke:
    one frame changes, the other does not."""
    doc = _doc()
    _paint(doc, RED)
    doc.add_frame(link=True)
    doc.rotate90(1)

    assert doc.undo()

    anim = doc.anim
    track = anim.tracks[0]
    first = anim.cel(track.uid, anim.frames[0].uid)
    second = anim.cel(track.uid, anim.frames[1].uid)
    assert first is second
    assert len(anim.slots_of(first)) == 2


def test_a_crop_resizes_every_cel_and_the_placeholders_with_them():
    doc = _doc(8, 8)
    _paint(doc, RED)
    doc.add_frame()  # blank: this frame is all placeholder

    assert doc.crop((0, 0, 4, 4))

    doc.set_current_frame(1)
    assert doc.stack[0].pixels.shape == (4, 4, 4)
    doc.set_current_frame(0)
    assert doc.stack[0].pixels.shape == (4, 4, 4)


def test_a_crop_on_a_grid_with_no_cels_at_all_still_resizes():
    doc = _doc(8, 8)
    doc.ensure_animation()
    doc.clear_cel()

    assert doc.crop((0, 0, 4, 4))

    assert doc.size == (4, 4)
    assert doc.stack[0].pixels.shape == (4, 4, 4)


# --- the frame flatten cache -------------------------------------------------


def test_a_frame_flatten_is_the_frames_pixels_not_the_playheads():
    doc = _doc()
    _paint(doc, RED)
    second = doc.add_frame()

    flat = doc.frame_flat(doc.anim.frames[0].uid)

    assert tuple(flat[0, 0]) == RED
    assert not doc.frame_flat(second.uid).any()


def test_a_frame_flatten_is_cached_until_that_frame_changes():
    """Keyed on the frame's own stamp, not on ``doc.rev``: rev moves for any
    change anywhere, so every onion-skinned neighbour would be recomposited on
    every dab drawn on the frame between them."""
    doc = _doc()
    _paint(doc, RED)
    second = doc.add_frame()
    first_uid = doc.anim.frames[0].uid
    cached = doc.frame_flat(first_uid)

    assert doc.frame_flat(first_uid) is cached

    # Draw on the *other* frame: frame one is untouched and must stay cached.
    doc.set_current_frame(1)
    _paint(doc, BLUE)
    assert doc.frame_flat(first_uid) is cached

    # Now draw on frame one itself.
    doc.set_current_frame(0)
    _paint(doc, BLUE)
    assert doc.frame_flat(first_uid) is not cached
    assert tuple(doc.frame_flat(first_uid)[0, 0]) == BLUE
    assert second is not None


def test_editing_a_linked_cel_invalidates_every_frame_it_appears_on():
    doc = _doc()
    _paint(doc, RED)
    linked = doc.add_frame(link=True)
    lone = doc.add_frame()
    before_linked = doc.frame_flat(linked.uid)
    before_lone = doc.frame_flat(lone.uid)

    doc.set_current_frame(0)
    _paint(doc, BLUE)

    assert doc.frame_flat(linked.uid) is not before_linked
    assert doc.frame_flat(lone.uid) is before_lone


def test_the_frame_cache_respects_its_byte_ceiling(monkeypatch):
    from warlock.studio.inker import document as document_mod

    doc = _doc(64, 64)
    one_frame = 64 * 64 * 4
    monkeypatch.setattr(document_mod, "FRAME_CACHE_BYTES", one_frame * 2)
    for _ in range(5):
        doc.add_frame()
    for frame in doc.anim.frames:
        doc.frame_flat(frame.uid)

    assert doc.frame_cache_bytes() <= one_frame * 2


def test_one_frame_over_the_whole_budget_is_still_returned(monkeypatch):
    """A large canvas must not end up caching nothing and recompositing on
    every draw, which is what evicting the entry just stored would mean."""
    from warlock.studio.inker import document as document_mod

    doc = _doc(64, 64)
    monkeypatch.setattr(document_mod, "FRAME_CACHE_BYTES", 1)
    doc.add_frame()
    uid = doc.anim.frames[0].uid

    assert doc.frame_flat(uid) is not None
    assert doc.frame_flat(uid) is doc.frame_flat(uid)


def test_a_still_document_has_no_frame_flatten():
    assert _doc().frame_flat(1) is None


# --- playback arithmetic -----------------------------------------------------


def _advance(durations, index, accum, dt, span=None, **kw):
    """``advance`` without its fourth return value.

    Every test below this line is about the timekeeping, which no direction
    changes; the ping-pong leg is what the fourth value carries and the tests
    that care about it call ``advance`` directly.
    """
    span = span or (0, len(durations) - 1, True)
    return animation.advance(durations, index, accum, dt, span, **kw)[:3]


def test_time_accumulates_rather_than_being_divided():
    """A 40 ms frame followed by a 400 ms one is not a rate."""
    index, accum, playing = _advance([40, 400], 0, 0.0, 30.0)
    assert (index, playing) == (0, True)
    assert accum == pytest.approx(30.0)

    index, accum, playing = _advance([40, 400], 0, 30.0, 20.0)
    assert index == 1
    assert accum == pytest.approx(10.0)


def test_a_long_tick_steps_as_many_frames_as_it_covers():
    """A ``while``, not an ``if``. With 10 ms frames and a 200 ms hitch, an
    ``if`` runs the clip at a twentieth speed whenever the machine is busy."""
    index, _accum, playing = _advance([10] * 8, 0, 0.0, 55.0)
    assert (index, playing) == (5, True)


def test_playback_wraps_a_looping_span():
    index, _accum, playing = _advance([10, 10, 10], 2, 0.0, 10.0)
    assert (index, playing) == (0, True)


def test_a_non_looping_span_stops_at_its_end():
    index, accum, playing = _advance([10, 10], 1, 0.0, 10.0, span=(0, 1, False))
    assert (index, accum, playing) == (1, 0.0, False)


def test_playback_stays_inside_its_span():
    index, _accum, playing = _advance([10] * 5, 3, 0.0, 10.0, span=(1, 3, True))
    assert (index, playing) == (1, True)


def test_an_index_outside_the_span_is_clamped_to_the_nearest_end():
    """Clamped, not reset to the start: entering a tag from beyond its end
    should carry on from the frame nearest where the playhead was."""
    assert _advance([10] * 5, 4, 0.0, 0.0, span=(1, 2, True))[0] == 2
    assert _advance([10] * 5, 0, 0.0, 0.0, span=(1, 2, True))[0] == 1


def test_advance_on_an_empty_timeline_stops():
    assert animation.advance([], 0, 0.0, 100.0, (0, -1, True)) == (
        0,
        0.0,
        False,
        True,
        0,
    )


def test_a_negative_delta_time_does_not_run_the_clip_backwards():
    index, accum, playing = _advance([10, 10], 0, 0.0, -50.0)
    assert (index, accum, playing) == (0, 0.0, True)


# --- tag direction -----------------------------------------------------------


def _walk(durations, span, direction, steps, index=None, forward=True):
    """The indices a clip visits over ``steps`` whole frames of time."""
    start, end, _loop = span
    index = start if index is None else index
    out = []
    for _ in range(steps):
        nxt, _accum, playing, forward, _cycles = animation.advance(
            durations, index, 0.0, durations[index], span,
            direction=direction, forward=forward,
        )
        # A stop that leaves the playhead where it was shows no new frame, so it
        # ends the walk rather than repeating the last entry.
        if not playing and nxt == index:
            break
        index = nxt
        out.append(index)
        if not playing:
            break
    return out


def test_a_reverse_tag_walks_its_span_backwards_and_wraps_to_the_end():
    assert _walk([10] * 4, (0, 3, True), "reverse", 5, index=3) == [2, 1, 0, 3, 2]


def test_a_reverse_tag_that_does_not_loop_stops_at_its_start():
    assert _walk([10] * 3, (0, 2, False), "reverse", 5, index=2) == [1, 0]


def test_a_ping_pong_swings_without_holding_either_end_twice():
    """The frame at each extreme is held for its own duration and no longer.

    Resuming *at* the end frame after turning around would show it for two
    frame-times, which is a visible hitch at both ends of every swing -- and the
    one thing a hand-drawn ping-pong is drawn to avoid.
    """
    assert _walk([10] * 4, (0, 3, True), "pingpong", 8) == [1, 2, 3, 2, 1, 0, 1, 2]


def test_a_ping_pong_that_does_not_loop_plays_out_and_back_once():
    assert _walk([10] * 3, (0, 2, False), "pingpong", 6) == [1, 2, 1, 0]


def test_a_ping_pongs_leg_survives_between_ticks():
    """The returned ``forward`` is the whole of that state, and a caller that
    dropped it would make every tick an outward one -- so the clip would reach
    the end and stay there, stepping between the last two frames forever."""
    index, _accum, _playing, forward, _cycles = animation.advance(
        [10] * 3, 2, 0.0, 10.0, (0, 2, True), direction="pingpong"
    )
    assert (index, forward) == (1, False)
    index, _accum, _playing, forward, _cycles = animation.advance(
        [10] * 3, index, 0.0, 10.0, (0, 2, True), direction="pingpong", forward=forward
    )
    assert (index, forward) == (0, False)


def test_a_one_frame_ping_pong_has_nowhere_to_turn_and_holds():
    assert _walk([10] * 3, (1, 1, True), "pingpong", 3) == [1, 1, 1]


def test_a_tag_with_a_direction_this_build_does_not_carry_plays_forward():
    """Coerced rather than refused: a tag arrives from a file as often as from
    the menu, and an unknown spelling must not fail the document."""
    assert animation.Tag(name="x", direction="sideways").direction == "forward"


def test_play_direction_and_loop_range_answer_about_the_same_tag():
    anim = animation.Animation(
        frames=[animation.Frame() for _ in range(6)],
        tags=[
            animation.Tag(name="all", start=0, end=5),
            animation.Tag(name="swing", start=2, end=4, direction="pingpong"),
        ],
    )
    # The innermost tag wins, and both questions have to say so.
    assert anim.loop_range(3) == (2, 4, True)
    assert anim.play_direction(3) == "pingpong"
    assert anim.play_direction(0) == "forward"


# --- loop ranges and tags ----------------------------------------------------


def test_with_no_tag_the_loop_is_the_whole_timeline():
    anim = Animation(tracks=[Track()], frames=[Frame(), Frame(), Frame()])
    assert anim.loop_range(1) == (0, 2, True)


def test_a_tag_containing_the_playhead_becomes_the_loop():
    anim = Animation(tracks=[Track()], frames=[Frame() for _ in range(6)])
    anim.tags.append(Tag(name="walk", start=2, end=4, loop=False))
    assert anim.loop_range(3) == (2, 4, False)
    assert anim.loop_range(0) == (0, 5, True)


def test_the_innermost_tag_wins():
    """Tags nest in practice -- a short "hit" inside a long "combat" -- and the
    narrower one is the one the user just clicked into."""
    anim = Animation(tracks=[Track()], frames=[Frame() for _ in range(10)])
    anim.tags.append(Tag(name="combat", start=0, end=9))
    anim.tags.append(Tag(name="hit", start=4, end=5))
    assert anim.active_tag(4).name == "hit"


def test_a_tag_range_is_clamped_to_the_timeline():
    anim = Animation(tracks=[Track()], frames=[Frame(), Frame()])
    anim.tags.append(Tag(name="stale", start=0, end=99))
    assert anim.loop_range(0) == (0, 1, True)


# --- the audit's findings ----------------------------------------------------
#
# Each of these pins one thing the grid got wrong, named after the way it went
# wrong rather than after the method, because the method is the fix and the
# behaviour is the contract.


def test_duplicating_a_track_keeps_its_links_instead_of_multiplying_it():
    """One copy per distinct cel, not one per slot.

    Copying per slot gave the duplicate three independent frames where the
    original had one shared cel -- the link silently gone, and three times the
    pixels. It only shows on the next stroke, when one frame of the duplicate
    changes and the others do not.
    """
    doc = _doc()
    second = _add_frame(doc, cel=False)
    third = _add_frame(doc, cel=False)
    anim = doc.anim
    track = anim.tracks[0]
    shared = anim.cels[(track.uid, anim.frames[0].uid)]
    anim.cels[(track.uid, second.uid)] = shared
    anim.cels[(track.uid, third.uid)] = shared

    doc.duplicate_layer(0)
    copy_track = anim.tracks[1]
    copies = [anim.cels[(copy_track.uid, frame.uid)] for frame in anim.frames]

    assert len({id(cel) for cel in copies}) == 1, "one object across all three frames"
    assert copies[0] is not shared, "and not the original's object"

    # And the link is a real one: a write to the duplicate's cel shows on every
    # frame of the duplicate and on none of the original's.
    copies[0].pixels[0, 0] = RED
    assert tuple(copies[2].pixels[0, 0]) == RED
    assert tuple(shared.pixels[0, 0]) != RED


def test_a_matte_cuts_every_frame_not_only_the_one_on_screen():
    """A cutout describes the drawing, not a moment of it.

    Mattes applied to the current frame's stack while ``_stamp_all`` announced
    that every frame had changed was the worst of both: the other frames kept
    their alpha and every cached flatten was thrown away saying otherwise.
    """
    doc = _doc()
    second = _add_frame(doc)
    anim = doc.anim
    for layer in anim.unique_cel_layers():
        layer.pixels[:] = 255

    alpha = np.zeros(doc.size[::-1], dtype=np.uint8)
    alpha[0:4, 0:4] = 255
    assert doc.apply_matte(alpha)

    off_frame = anim.cels[(anim.tracks[0].uid, second.uid)]
    assert int(off_frame.pixels[0, 0, 3]) == 255
    assert int(off_frame.pixels[7, 7, 3]) == 0


def test_a_matte_multiplies_a_linked_cel_once():
    """Twice would square it, which shows as a darker edge wherever the matte
    is soft rather than binary."""
    doc = _doc()
    second = _add_frame(doc, cel=False)
    anim = doc.anim
    track = anim.tracks[0]
    shared = anim.cels[(track.uid, anim.frames[0].uid)]
    anim.cels[(track.uid, second.uid)] = shared
    shared.pixels[:] = 255

    alpha = np.full(doc.size[::-1], 128, dtype=np.uint8)
    assert doc.apply_matte(alpha)
    assert int(shared.pixels[0, 0, 3]) == 128


def test_de_animating_forgets_the_frame_cache_as_well_as_the_stamps():
    """Stamps and cache go together or not at all.

    Clearing the stamps alone restarts each frame's counter at zero, so a frame
    uid that comes back matches a cache entry built before it went and shows
    pixels from another edit.
    """
    doc = _doc()
    _add_frame(doc)
    uid = doc.anim.frames[0].uid
    assert doc.frame_flat(uid) is not None
    assert doc.frame_cache_bytes() > 0

    doc.drop_animation()
    assert doc.frame_cache_bytes() == 0
    assert doc._frame_stamps == {}


def test_deleting_a_frame_drops_its_cached_flatten_and_reports_it():
    """The cache entry is memory and the report is a GL texture: a frame nobody
    asks for again is never released without one."""
    doc = _doc()
    doc.add_frame()
    anim = doc.anim
    gone = anim.frames[1].uid
    assert doc.frame_flat(gone) is not None
    doc.take_dropped_frames()

    assert doc.remove_frame(1)
    assert gone not in doc._frame_cache
    assert gone not in doc._frame_order
    assert doc.take_dropped_frames() == [gone]
    assert doc.take_dropped_frames() == [], "a drain, not a list that grows"


def test_removing_a_frame_charges_the_budget_only_for_what_it_released():
    """A cel still linked into the frame next door is alive whatever the history
    does with the step, and charging for it evicts real history to make room for
    a number describing no memory."""
    doc = _doc()
    doc.add_frame(link=True)
    doc.add_frame(copy=True)
    before = doc.history.bytes

    assert doc.remove_frame(1)
    assert doc.history.top.cost == 0
    assert doc.history.bytes == before

    assert doc.remove_frame(1)
    assert doc.history.top.cost > 0, "the unlinked frame's cel really did go"


def test_clearing_a_linked_cel_charges_nothing():
    doc = _doc()
    doc.add_frame(link=True)
    assert doc.clear_cel(track_index=0, frame_index=1)
    assert doc.history.top.cost == 0


def test_a_second_autovivified_cel_does_not_land_on_the_shared_blank_plane():
    """The convention is that every write commits before the next one
    autovivifies. When it did not hold, the second write was pointed at the
    read-only placeholder plane and raised out of the middle of a gesture."""
    doc = _doc()
    doc.add_frame()  # a blank second frame: every slot is a placeholder
    doc.add_layer()
    doc._ensure_cel_for(doc.stack[0].uid)
    doc._ensure_cel_for(doc.stack[1].uid)

    for layer in doc.stack:
        assert not doc.anim.is_placeholder(layer)
        layer.pixels[0, 0] = RED  # writable, which is the whole claim

    doc._discard_pending_cel()
    frame_uid = doc.anim.frame.uid
    assert all(key[1] != frame_uid for key in doc.anim.cels)


def test_an_animation_with_no_frames_says_so_rather_than_indexing():
    anim = Animation(tracks=[Track()], frames=[])
    with pytest.raises(IndexError):
        _ = anim.frame


# --- tags --------------------------------------------------------------------


def test_a_tag_is_created_clamped_and_undoable():
    doc = _doc()
    doc.add_frame()
    doc.add_frame()
    assert doc.add_tag("walk", 1, 99)
    tag = doc.anim.tags[0]
    assert (tag.name, tag.start, tag.end, tag.loop) == ("walk", 1, 2, True)

    doc.undo()
    assert doc.anim.tags == []
    doc.redo()
    assert doc.anim.tags[0].name == "walk"


def test_a_tag_with_its_ends_the_wrong_way_round_is_put_right():
    doc = _doc()
    doc.add_frame()
    doc.add_frame()
    doc.add_tag("back", 2, 0)
    assert (doc.anim.tags[0].start, doc.anim.tags[0].end) == (0, 2)


def test_editing_a_tag_to_what_it_already_says_pushes_no_step():
    """The rule every op here follows: dirty is a comparison against the head,
    so a step that changes nothing makes a saved document ask to be saved."""
    doc = _doc()
    doc.ensure_animation()
    doc.add_tag("idle", 0)
    head = doc.history.head
    assert not doc.set_tag(0, name="idle")
    assert doc.history.head == head


def test_undoing_a_tag_edit_cannot_be_written_through_by_a_later_one():
    """The step holds lists of ``Tag`` objects and the document installs copies;
    sharing them would let the next rename edit the step meant to reverse it."""
    doc = _doc()
    doc.ensure_animation()
    doc.add_tag("first", 0)
    doc.set_tag(0, name="second")
    doc.set_tag(0, name="third")
    doc.undo()
    assert doc.anim.tags[0].name == "second"
    doc.undo()
    assert doc.anim.tags[0].name == "first"


def test_removing_a_tag_is_by_position_and_undoes_whole():
    doc = _doc()
    doc.ensure_animation()
    doc.add_tag("a", 0)
    doc.add_tag("b", 0)
    assert doc.remove_tag(0)
    assert [tag.name for tag in doc.anim.tags] == ["b"]
    doc.undo()
    assert [tag.name for tag in doc.anim.tags] == ["a", "b"]


def test_tag_ops_on_a_still_document_do_nothing():
    doc = _doc()
    assert not doc.add_tag("x", 0)
    assert not doc.remove_tag(0)
    assert not doc.set_tag(0, name="x")


# --- directional layouts ----------------------------------------------------


def test_a_turnaround_layout_maps_four_frames_onto_a_2x2_grid():
    layout = animation.DirectionalLayout.of("turnaround")
    assert (layout.columns, layout.rows, layout.frame_count) == (2, 2, 4)
    assert [layout.cell(i) for i in range(4)] == [
        (0, 0, "front", 0, 0),
        (0, 1, "left", 90, 0),
        (1, 0, "right", 270, 0),
        (1, 1, "back", 180, 0),
    ]


def test_a_walk_layout_gives_each_direction_a_row_of_four():
    layout = animation.DirectionalLayout.of("walk")
    assert (layout.columns, layout.rows, layout.frame_count) == (4, 4, 16)
    assert layout.frames_per_direction == 4
    for index in range(16):
        row, col, direction, yaw, frame = layout.cell(index)
        assert row == index // 4 and col == index % 4
        assert direction == animation.DIRECTION_ORDER[index // 4]
        assert yaw == animation.DIRECTION_YAWS[direction]
        assert frame == index % 4


def test_an_unknown_kind_has_no_layout_rather_than_raising():
    """A kind a later build introduced must cost the document its grid, not
    its openability -- the rule ``Tag.__post_init__`` follows for direction."""
    assert animation.DirectionalLayout.of("isometric") is None
    assert animation.DirectionalLayout.of(None) is None


def test_a_layout_refuses_a_frame_outside_its_grid():
    layout = animation.DirectionalLayout.of("turnaround")
    with pytest.raises(IndexError):
        layout.cell(4)
    with pytest.raises(IndexError):
        layout.cell(-1)


def test_an_ordinary_animation_has_no_layout():
    doc = _doc()
    doc.ensure_animation()
    assert doc.anim.layout is None


def test_a_layout_survives_a_whole_canvas_undo():
    """Construction-time state, like ``Document.matte``: no V1 op edits it, and
    a snapshot restore mutates the existing Animation in place -- so this needs
    no snapshot change, which is exactly what this asserts."""
    doc = _doc()
    doc.ensure_animation()
    doc.anim.layout = animation.DirectionalLayout("turnaround")
    before = len(doc.anim.frames)
    doc.add_frame()
    assert len(doc.anim.frames) == before + 1
    doc.undo()
    assert len(doc.anim.frames) == before
    assert doc.anim.layout == animation.DirectionalLayout("turnaround")


# --- tag repeat counts (C7) --------------------------------------------------


def test_a_repeat_of_zero_is_exactly_todays_behaviour():
    """The default has to be a no-op, because every document already on disk
    reads back with it -- a repeat that changed how an old file played would be
    a migration wearing a trailing field's clothes."""
    assert Tag(name="t", start=0, end=2).repeat == 0
    plain = animation.advance([10] * 3, 2, 0.0, 10.0, (0, 2, True))
    counted = animation.advance([10] * 3, 2, 0.0, 10.0, (0, 2, True), repeat=0)
    assert plain[:4] == counted[:4]


def test_a_nonsense_repeat_is_coerced_rather_than_refused():
    assert Tag(name="t", repeat=-3).repeat == 0
    assert Tag(name="t", repeat="nope").repeat == 0  # type: ignore[arg-type]


def test_a_forward_span_counts_one_cycle_each_time_it_reaches_its_end():
    _index, _accum, playing, _forward, cycles = animation.advance(
        [10] * 3, 2, 0.0, 10.0, (0, 2, True)
    )
    assert (playing, cycles) == (True, 1)


def test_a_finite_repeat_stops_at_the_end_of_its_last_pass():
    index, _accum, playing, _forward, cycles = animation.advance(
        [10] * 3, 2, 0.0, 10.0, (0, 2, True), repeat=1
    )
    # At the *end* frame, not wrapped round to the start: a clip that has
    # finished should be showing its last drawing.
    assert (index, playing, cycles) == (2, False, 1)


def test_a_finite_repeat_keeps_playing_until_the_count_runs_out():
    index, _accum, playing, _forward, cycles = animation.advance(
        [10] * 3, 2, 0.0, 10.0, (0, 2, True), repeat=3, cycles=1
    )
    assert (index, playing, cycles) == (0, True, 2)
    index, _accum, playing, _forward, cycles = animation.advance(
        [10] * 3, 2, 0.0, 10.0, (0, 2, True), repeat=3, cycles=cycles
    )
    assert (index, playing, cycles) == (2, False, 3)


def test_a_reverse_span_counts_a_cycle_at_its_start():
    index, _accum, playing, _forward, cycles = animation.advance(
        [10] * 3, 0, 0.0, 10.0, (0, 2, True), direction="reverse", repeat=1
    )
    assert (index, playing, cycles) == (0, False, 1)


def test_a_ping_pong_cycle_is_one_out_and_back():
    """Turning round at the far end is half a swing, and counting it would make
    "play three times" mean one and a half."""
    span = (0, 2, True)
    walked = []
    index, forward, cycles = 0, True, 0
    for _ in range(6):
        index, _accum, playing, forward, cycles = animation.advance(
            [10] * 3, index, 0.0, 10.0, span,
            direction="pingpong", forward=forward, cycles=cycles,
        )
        walked.append((index, cycles))
        if not playing:
            break
    # 1, 2 (out), 1, 0 (back), then the step that *restarts* the swing is the
    # one that counts it -- the same point a forward span counts at, which is
    # the step that puts the playhead back where the span begins.
    assert walked == [(1, 0), (2, 0), (1, 0), (0, 0), (1, 1), (2, 1)]


def test_a_ping_pong_played_once_stops_where_it_started():
    span = (0, 2, True)
    index, forward, cycles, playing = 0, True, 0, True
    for _ in range(8):
        index, _accum, playing, forward, cycles = animation.advance(
            [10] * 3, index, 0.0, 10.0, span,
            direction="pingpong", forward=forward, repeat=1, cycles=cycles,
        )
        if not playing:
            break
    assert (index, playing, cycles) == (0, False, 1)


def test_a_finite_repeat_overrides_a_tag_already_set_to_play_once():
    """``loop_range`` hands back ``tag.loop`` verbatim, so a tag set to "once"
    and *then* given a count of three used to stop on its first wrap -- the
    flag deciding a question the count is the more specific answer to. Worse,
    ``export_tag`` writes the count into the GIF, so the file played three
    times while the editor played one."""
    span = (0, 2, False)
    index, forward, cycles = 0, True, 0
    seen = []
    for _ in range(12):
        index, _accum, playing, forward, cycles = animation.advance(
            [10] * 3, index, 0.0, 10.0, span, forward=forward, repeat=3, cycles=cycles
        )
        seen.append((index, playing, cycles))
        if not playing:
            break
    assert seen[-1] == (2, False, 3)
    assert len(seen) == 9  # three whole passes of three frames, not one


def test_a_repeat_that_is_already_used_up_plays_no_further_pass():
    """Only reachable when the count is *lowered* under a clip that has passed
    it. Stepping on would play one whole extra pass before the wrap check
    below noticed."""
    index, _accum, playing, _forward, cycles = animation.advance(
        [10] * 3, 1, 0.0, 10.0, (0, 2, True), repeat=2, cycles=5
    )
    assert (index, playing, cycles) == (1, False, 5)


def test_clearing_a_repeat_gives_the_loop_flag_its_answer_back():
    """The count *overrides* the flag rather than rewriting it, so a "once" tag
    is still a "once" tag once the count is cleared."""
    index, _accum, playing, _forward, _cycles = animation.advance(
        [10] * 3, 2, 0.0, 10.0, (0, 2, False), repeat=0
    )
    assert (index, playing) == (2, False)


def test_a_repeat_does_not_fall_through_past_its_span():
    """The documented divergence from Aseprite: a finite repeat stops inside
    the tag rather than carrying on into the frames after it."""
    index, _accum, playing, _forward, _cycles = animation.advance(
        [10] * 5, 2, 0.0, 10.0, (1, 2, True), repeat=1
    )
    assert (index, playing) == (2, False)


def test_the_repeat_in_force_comes_from_the_tag_under_the_playhead():
    anim = Animation(
        tracks=[Track()],
        frames=[Frame(), Frame(), Frame()],
        tags=[Tag(name="hit", start=1, end=2, repeat=4)],
    )
    assert anim.play_repeat(0) == 0
    assert anim.play_repeat(2) == 4


def test_a_repeat_survives_a_tag_edit_and_is_clamped_at_zero():
    doc = _doc()
    _add_frame(doc)
    assert doc.add_tag("walk", 0, 1)
    assert doc.set_tag(0, repeat=3)
    assert doc.anim.tags[0].repeat == 3
    assert doc.set_tag(0, repeat=-2)
    assert doc.anim.tags[0].repeat == 0
