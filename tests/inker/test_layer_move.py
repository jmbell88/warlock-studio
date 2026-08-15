"""Moving a layer's pixels with no selection: the move tool's third arm.

The session mirrors the filter session, and the three properties worth pinning
are the ones that make it a session rather than a sequence of writes: a preview
re-renders from the snapshot so a drag cannot compound, a commit is exactly one
undo step over the union of where the pixels were and are, and a cancel puts
them back having pushed nothing.

The fourth is about the grid: a *linked* cel is one object shared by several
frames, so moving it has to show on every one of them -- asserted with ``is``,
because two equal copies look identical in a pixel test and are the bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio import inker

SIZE = (16, 16)


def _doc():
    doc = inker.Document.blank(*SIZE)
    doc.stack.active.pixels[4:8, 4:8] = (255, 0, 0, 255)
    doc.invalidate_all()
    return doc


def _box(pixels):
    ys, xs = np.nonzero(pixels[..., 3])
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def test_a_move_translates_the_pixels_and_lands_as_one_step():
    doc = _doc()
    before = doc.stack.active.pixels.copy()
    assert doc.begin_layer_move()
    doc.preview_layer_move(3, 2)
    assert doc.commit_layer_move()
    assert _box(doc.stack.active.pixels) == (7, 6, 11, 10)
    assert doc.history.can_undo
    doc.undo()
    assert np.array_equal(doc.stack.active.pixels, before)
    assert not doc.history.can_undo


def test_a_preview_is_measured_from_the_snapshot_and_never_compounds():
    """The anti-compounding rule ``preview_filter`` already follows: a drag
    calls this every frame with the *total* offset, so accumulating per-frame
    deltas would make a slow drag travel further than a fast one."""
    doc = _doc()
    doc.begin_layer_move()
    for offset in (1, 2, 3, 4, 5):
        doc.preview_layer_move(offset, 0)
    doc.commit_layer_move()
    assert _box(doc.stack.active.pixels) == (9, 4, 13, 8)


def test_pixels_pushed_off_the_edge_are_cropped_rather_than_rolled():
    """A layer is canvas-sized and its edge *is* the canvas edge. Rolling would
    put a sprite's head on the far side of the frame, which reads as
    corruption; the tiled-wrap variant is deliberately unwired."""
    doc = _doc()
    doc.begin_layer_move()
    doc.preview_layer_move(-6, 0)
    doc.commit_layer_move()
    painted = doc.stack.active.pixels
    assert _box(painted) == (0, 4, 2, 8)
    assert int(painted[..., 3][:, -1].max()) == 0


def test_a_cancelled_move_restores_the_pixels_and_pushes_nothing():
    doc = _doc()
    before = doc.stack.active.pixels.copy()
    doc.begin_layer_move()
    doc.preview_layer_move(5, 5)
    assert doc.cancel_layer_move()
    assert np.array_equal(doc.stack.active.pixels, before)
    assert not doc.history.can_undo


def test_a_move_of_nothing_pushes_nothing():
    """``_commit_patch`` compares before against after, so pressing and
    releasing without dragging is a no-op rather than a step that dirties the
    document."""
    doc = _doc()
    doc.begin_layer_move()
    doc.preview_layer_move(0, 0)
    doc.commit_layer_move()
    assert not doc.history.can_undo


def test_moving_an_empty_layer_is_a_refusal_rather_than_a_step():
    doc = inker.Document.blank(*SIZE)
    doc.begin_layer_move()
    doc.preview_layer_move(4, 4)
    assert not doc.commit_layer_move()
    assert not doc.history.can_undo


def test_the_patch_is_the_union_of_where_the_pixels_were_and_are():
    """Rather than the whole canvas: a 16-pixel nudge of a sprite on a 2048
    square would otherwise cost 32 MiB of history for a few thousand pixels."""
    doc = _doc()
    doc.begin_layer_move()
    doc.preview_layer_move(3, 2)
    doc.commit_layer_move()
    assert doc.history.top.rect == (4, 4, 11, 10)


def test_the_patch_covers_erased_pixels_that_still_carry_colour():
    """The eraser cuts alpha and leaves RGB, so an erased region is
    ``(r, g, b, 0)``: invisible, outside any alpha-only box, and zeroed by the
    translate. A patch measured on alpha alone would not record it and the
    colours would be unrecoverable -- silently, because nothing on screen
    changed."""
    doc = _doc()
    ghost = doc.stack.active.pixels
    ghost[12:14, 12:14] = (7, 8, 9, 0)  # erased: colour kept, alpha gone
    before = ghost.copy()
    doc.begin_layer_move()
    doc.preview_layer_move(1, 0)
    doc.commit_layer_move()
    doc.undo()
    assert np.array_equal(doc.stack.active.pixels, before)


def test_a_content_locked_layer_refuses_to_open_a_session():
    """The lock is read with ``getattr`` because the flag lands beside this
    work rather than under it -- but the door is here now, so a move cannot be
    the one write that walks through it."""
    doc = _doc()
    doc.stack.active.content_lock = True
    doc.select(inker.SelectionMask.from_rect(SIZE, (4, 4, 8, 8)))
    doc.lift()
    assert not doc.begin_layer_move()
    assert doc.preview_layer_move(4, 4) is False
    # A refusal leaves the document as it found it: the float is still
    # floating, rather than landed as the side effect of a move that did not
    # happen.
    assert doc.floating is not None


def test_beginning_twice_commits_the_first_session_rather_than_reverting_it():
    """``end_layer_move``, and the reason it is not ``cancel``. A session still
    open when the next one begins ended somewhere this class cannot see, so its
    snapshot is of unknown age -- restoring from it would roll the layer back
    past whatever happened in between. Committing records what actually
    happened as a step and leaves the layer where it looks like it is."""
    doc = _doc()
    doc.begin_layer_move()
    doc.preview_layer_move(4, 0)
    doc.begin_layer_move()
    doc.preview_layer_move(1, 0)
    doc.commit_layer_move()
    # Four pixels and then one more, with a step apiece to walk back.
    assert _box(doc.stack.active.pixels) == (9, 4, 13, 8)
    doc.undo()
    assert _box(doc.stack.active.pixels) == (8, 4, 12, 8)
    doc.undo()
    assert _box(doc.stack.active.pixels) == (4, 4, 8, 8)


def test_an_orphaned_session_cannot_wipe_what_was_painted_after_it():
    """The catastrophic shape of the C12d bug, pinned at the engine door so no
    future caller can reach it either: a stale snapshot restored over the layer
    takes every stroke painted since with it, while those strokes' own
    ``PatchEdit``s stay in history describing pixels that are no longer there.
    ``end_layer_move`` commits, so nothing is rolled back over."""
    doc = _doc()
    doc.begin_layer_move()
    doc.preview_layer_move(4, 0)
    # Somebody else paints, with the session still open.
    doc.begin_stroke((13.0, 1.0), (0, 255, 0, 255), size=3, nib="square")
    doc.end_stroke()
    painted = doc.stack.active.pixels[1, 13].copy()
    assert int(painted[1]) == 255

    doc.begin_layer_move()
    assert doc.stack.active.pixels[1, 13].tolist() == painted.tolist()


def test_end_layer_move_is_a_no_op_with_nothing_open():
    doc = _doc()
    assert doc.end_layer_move() is False
    assert not doc.history.can_undo


def test_a_selection_does_not_scope_it():
    """A move is not a paint: it translates the layer, and the selection is
    what the *other* two arms of the move tool act on."""
    doc = _doc()
    doc.select(inker.SelectionMask.from_rect(SIZE, (0, 0, 4, 4)))
    doc.begin_layer_move()
    doc.preview_layer_move(2, 0)
    doc.commit_layer_move()
    assert _box(doc.stack.active.pixels) == (6, 4, 10, 8)


def test_a_floating_buffer_is_committed_before_a_move_opens():
    doc = _doc()
    doc.select(inker.SelectionMask.from_rect(SIZE, (4, 4, 8, 8)))
    doc.lift()
    assert doc.floating is not None
    doc.begin_layer_move()
    assert doc.floating is None


# --- the animation grid -----------------------------------------------------


def _linked():
    """A two-frame document whose one track holds the *same* cel on both."""
    doc = _doc()
    doc.add_frame()
    anim = doc.anim
    track, frames = anim.tracks[0], anim.frames
    cel = anim.cels[(track.uid, frames[0].uid)]
    anim.cels[(track.uid, frames[1].uid)] = cel
    doc.set_current_frame(0)
    return doc, track, frames


def test_moving_a_linked_cel_shows_on_every_frame_it_appears_on():
    """Written in place, so the two slots go on holding one object. Rebinding
    ``layer.pixels`` would restore two equal copies and silently break the
    link -- and the user would find out on the next stroke, when one frame
    changed and the other did not."""
    doc, track, frames = _linked()
    doc.begin_layer_move()
    doc.preview_layer_move(3, 0)
    assert doc.commit_layer_move()
    first = doc.anim.cels[(track.uid, frames[0].uid)]
    second = doc.anim.cels[(track.uid, frames[1].uid)]
    assert first is second
    assert _box(first.pixels) == (7, 4, 11, 8)


def test_the_session_survives_the_playhead_moving_underneath_it():
    """Addressed by uid, never by "the active layer": a cel's uid names it on
    every frame at once, which is the whole point of addressing that way."""
    doc, track, frames = _linked()
    doc.begin_layer_move()
    doc.set_current_frame(1)
    doc.preview_layer_move(2, 0)
    assert doc.commit_layer_move()
    assert _box(doc.anim.cels[(track.uid, frames[0].uid)].pixels) == (6, 4, 10, 8)


def test_a_move_on_an_empty_frame_autovivifies_nothing_it_cannot_use():
    """A brush-down with no drag must not leave a blank cel behind and a
    document asking to be saved; the same is true of a press on the move tool."""
    doc = inker.Document.blank(*SIZE)
    doc.add_frame()
    doc.set_current_frame(1)
    anim = doc.anim
    slots = dict(anim.cels)
    doc.begin_layer_move()
    doc.preview_layer_move(3, 3)
    doc.commit_layer_move()
    assert anim.cels == slots


@pytest.mark.parametrize("dx,dy", [(1, 0), (-1, 0), (0, 1), (0, -1), (8, -8)])
def test_a_nudge_of_any_direction_is_exactly_one_step(dx, dy):
    doc = _doc()
    doc.begin_layer_move()
    doc.preview_layer_move(dx, dy)
    doc.commit_layer_move()
    assert doc.history.can_undo
    doc.undo()
    assert _box(doc.stack.active.pixels) == (4, 4, 8, 8)
