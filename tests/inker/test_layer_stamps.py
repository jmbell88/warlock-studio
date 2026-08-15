"""Per-layer change counters, which is what a cel thumbnail is keyed on.

The frame stamps beside them answer "has this frame's flatten changed"; these
answer "has this *cel's picture* changed", and the difference is the whole
reason there are two. A thumbnail keyed on the frame stamp would re-shrink
every cel in a column each time one of them was drawn on, and one keyed on
``doc.rev`` would re-shrink every cel in the document.

The trap here is the mirror of the one ``invalidate_all`` records: stamping too
eagerly is not conservative, it is the opposite of a cache. So the assertions
below are as much about what does *not* move the counter as about what does.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker.document import Document

RED = (255, 0, 0, 255)


def _paint(doc: Document, colour: tuple[int, int, int, int] = RED) -> None:
    weight = np.ones((2, 2), dtype=np.float32)
    assert doc.write_colour((0, 0, 2, 2), colour, weight)


def _animated(frames: int = 2) -> Document:
    doc = Document.blank(4, 4)
    for index in range(frames):
        if index:
            doc.add_frame()
        _paint(doc, (10 * index, 0, 0, 255))
    return doc


def _cel(doc: Document, track: int, frame: int):
    anim = doc.anim
    return anim.cels[(anim.tracks[track].uid, anim.frames[frame].uid)]


def test_a_write_bumps_only_the_layer_it_was_written_to():
    doc = _animated(2)
    left, right = _cel(doc, 0, 0), _cel(doc, 0, 1)
    before = (doc.layer_stamp(left.uid), doc.layer_stamp(right.uid))

    doc.set_current_frame(0)
    _paint(doc, (0, 255, 0, 255))
    assert doc.layer_stamp(left.uid) > before[0]
    assert doc.layer_stamp(right.uid) == before[1]


def test_invalidate_all_stamps_no_layer():
    """The "stamps no frame" lesson, one level down. Most of what reaches
    ``invalidate_all`` -- switching layer, switching frame, rebuilding the view
    after an edit that already stamped what it touched -- changes no pixels at
    all, and stamping here would throw away every thumbnail on every click."""
    doc = _animated(2)
    layer = _cel(doc, 0, 0)
    before = doc.layer_stamp(layer.uid)
    doc.invalidate_all()
    assert doc.layer_stamp(layer.uid) == before


def test_moving_the_playhead_stamps_no_layer():
    doc = _animated(3)
    stamps = {
        _cel(doc, 0, index).uid: doc.layer_stamp(_cel(doc, 0, index).uid)
        for index in range(3)
    }
    doc.set_current_frame(2)
    doc.set_current_frame(0)
    assert {uid: doc.layer_stamp(uid) for uid in stamps} == stamps


def test_a_whole_grid_change_stamps_every_distinct_cel():
    doc = _animated(2)
    uids = [_cel(doc, 0, index).uid for index in range(2)]
    before = [doc.layer_stamp(uid) for uid in uids]
    # A track property is authoritative over every frame it appears on, so
    # every cel's picture really does change.
    doc.set_layer_props(0, visible=False)
    assert [doc.layer_stamp(uid) for uid in uids] == [n + 1 for n in before]


def test_a_linked_cel_is_stamped_once_per_whole_grid_change():
    """Once per *object*, not once per slot it occupies -- the counter is only
    ever compared against itself, but bumping it three times for a background
    linked across three frames would make the number mean nothing."""
    doc = _animated(1)
    doc.add_frame(link=True)
    doc.add_frame(link=True)
    shared = _cel(doc, 0, 0)
    assert _cel(doc, 0, 2) is shared
    before = doc.layer_stamp(shared.uid)
    doc.set_layer_props(0, opacity=0.5)
    assert doc.layer_stamp(shared.uid) == before + 1


def test_an_undo_stamps_the_cel_it_lands_on_even_off_screen():
    """The thumbnail of a frame the playhead has moved off must still catch up:
    an undo addresses a cel by uid and can be pressed from anywhere."""
    doc = _animated(2)
    doc.set_current_frame(0)
    _paint(doc, (0, 0, 255, 255))
    edited = _cel(doc, 0, 0)
    doc.set_current_frame(1)
    before = doc.layer_stamp(edited.uid)
    assert doc.history.undo(doc)
    assert doc.layer_stamp(edited.uid) > before


def test_a_still_document_still_counts_its_layer_writes():
    doc = Document.blank(4, 4)
    layer = doc.stack.active
    before = doc.layer_stamp(layer.uid)
    _paint(doc)
    assert doc.layer_stamp(layer.uid) > before
