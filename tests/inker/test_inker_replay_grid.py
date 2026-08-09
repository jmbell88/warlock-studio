"""``restore_snapshot`` refuses a snapshot that describes a different document.

A ``ReplayEdit`` recorded on an animated document carries the grid -- the
frames, the tracks, and which slots share one cel, as *indices* into its
snapshot list, which is the whole of how a linked cel survives an undo. Restored
onto a document that is no longer animated, the old code took the still branch:
it rebuilt the stack from the snapshot and dropped the grid entirely, with no
error, nothing to undo and every link gone.

The two are unreachable past each other through the ordinary stack -- becoming
animated is itself an ``AnimateEdit``, so it sits above any replay that followed
it and LIFO undo re-animates first -- which is exactly the argument for raising
rather than falling back. Getting here means an assumption elsewhere has already
broken, and the fallback answered by silently flattening the timeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker.document import Document

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def _animated() -> Document:
    doc = Document.blank(8, 4)
    doc.stack.active.pixels[:] = RED
    doc.add_frame(link=True)  # a second frame sharing the first frame's cel
    return doc


def test_a_grid_carrying_snapshot_is_refused_by_a_still_document():
    doc = _animated()
    doc.rotate90()
    edit = doc.history.top
    assert edit is not None and edit.grid is not None

    doc.anim = None  # what de-animating under the step would look like
    with pytest.raises(ValueError, match="animated"):
        doc.restore_snapshot(edit.snapshot, edit.size, edit.active, edit.grid)


def test_a_still_snapshot_is_refused_by_an_animated_document():
    """The same rule from the other side. A snapshot with no grid describes a
    document with one stack, and restoring it over a grid would leave the stack
    and the frames disagreeing about what the document holds."""
    doc = Document.blank(8, 4)
    doc.stack.active.pixels[:] = RED
    doc.rotate90()
    edit = doc.history.top
    assert edit is not None and edit.grid is None

    doc.add_frame()
    with pytest.raises(ValueError, match="still"):
        doc.restore_snapshot(edit.snapshot, edit.size, edit.active, None)


def test_the_refusal_changes_nothing():
    """A raise that had already half-restored the document would be worse than
    the silent discard it replaced."""
    doc = _animated()
    doc.rotate90()
    edit = doc.history.top
    assert edit is not None

    doc.anim = None
    stack_before = list(doc.stack)
    size_before = doc.size
    with pytest.raises(ValueError):
        doc.restore_snapshot(edit.snapshot, edit.size, edit.active, edit.grid)
    assert list(doc.stack) == stack_before
    assert doc.size == size_before


# --- and the paths that must keep working ------------------------------------


def test_an_animated_replay_still_undoes_and_keeps_its_links():
    """The reason the grid is carried at all: two slots holding one object come
    back as two slots holding one object, not as two equal copies."""
    doc = _animated()
    assert doc.anim is not None
    shared = doc.anim.cels[(doc.anim.tracks[0].uid, doc.anim.frames[0].uid)]
    assert doc.anim.cels[(doc.anim.tracks[0].uid, doc.anim.frames[1].uid)] is shared

    doc.rotate90()
    assert doc.size == (4, 8)
    doc.undo()
    assert doc.size == (8, 4)

    anim = doc.anim
    assert anim is not None
    first = anim.cels[(anim.tracks[0].uid, anim.frames[0].uid)]
    second = anim.cels[(anim.tracks[0].uid, anim.frames[1].uid)]
    assert first is second, "a linked cel must come back linked"
    assert np.array_equal(first.pixels[0, 0], np.array(RED, dtype=np.uint8))


def test_a_still_replay_still_undoes():
    doc = Document.blank(8, 4)
    doc.stack.active.pixels[:] = BLUE
    doc.rotate90()
    assert doc.size == (4, 8)
    doc.undo()
    assert doc.size == (8, 4)
    assert np.array_equal(doc.stack.active.pixels[0, 0], np.array(BLUE, dtype=np.uint8))
