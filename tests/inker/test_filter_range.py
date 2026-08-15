"""A filter over a rectangle of the timeline, headless.

This is deliberately *not* the filter session: a session is a live preview over
one layer that recomputes from a snapshot every frame a slider moves, and this
is a bounded, mask-weighted write over many cels with no preview at all. The
assertions that earn the file are the two that a shared implementation would
have got wrong -- a linked cel filtered once rather than once per slot, and an
empty cel left empty rather than autovivified into a filtered nothing.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker import filters
from warlock.studio.inker.document import Document
from warlock.studio.inker.selection import SelectionMask

GREY = (128, 128, 128, 255)
BRIGHTER = {"brightness": 0.5, "contrast": 0.0}


def _paint(doc: Document, colour: tuple[int, int, int, int] = GREY) -> None:
    weight = np.ones((4, 4), dtype=np.float32)
    assert doc.write_colour((0, 0, 4, 4), colour, weight)


def _clip(frames: int = 3, tracks: int = 1) -> Document:
    doc = Document.blank(4, 4)
    for _ in range(tracks - 1):
        doc.add_layer()
    for index in range(frames):
        if index:
            doc.add_frame()
        for track in range(tracks):
            doc.set_active_layer(track)
            _paint(doc)
    doc.history.clear()
    return doc


def _cel(doc: Document, track: int, frame: int):
    anim = doc.anim
    return anim.cels.get((anim.tracks[track].uid, anim.frames[frame].uid))


def test_a_range_filter_writes_every_cel_it_covers_as_one_step():
    doc = _clip(3)
    before = [_cel(doc, 0, i).pixels.copy() for i in range(3)]

    assert doc.filter_range("brightness / contrast", BRIGHTER, 0, 0, 0, 1)
    assert not np.array_equal(_cel(doc, 0, 0).pixels, before[0])
    assert not np.array_equal(_cel(doc, 0, 1).pixels, before[1])
    # Outside the range, untouched.
    assert np.array_equal(_cel(doc, 0, 2).pixels, before[2])
    assert len(doc.history) == 1

    assert doc.history.undo(doc)
    assert np.array_equal(_cel(doc, 0, 0).pixels, before[0])
    assert np.array_equal(_cel(doc, 0, 1).pixels, before[1])


def test_a_linked_cel_is_filtered_once_however_many_slots_hold_it():
    """Twice the blur is not the blur. Deduping by ``id()`` is the whole of
    what stops a background linked across ten frames being filtered ten
    times."""
    doc = _clip(1)
    doc.add_frame(link=True)
    doc.add_frame(link=True)
    doc.history.clear()
    shared = _cel(doc, 0, 0)
    assert _cel(doc, 0, 2) is shared

    once = filters.apply_named("brightness / contrast", shared.pixels, **BRIGHTER)
    assert doc.filter_range("brightness / contrast", BRIGHTER, 0, 0, 0, 2)
    assert np.array_equal(shared.pixels, once)
    # One cel, one patch: a compound of three would be three writes to the same
    # plane and an undo that only reversed the last of them.
    assert len(doc.history) == 1


def test_an_empty_cel_is_never_autovivified_by_a_filter():
    """A write path conjures a cel because a stroke has pixels to put down.
    A filter over nothing has none, so the slot stays empty."""
    doc = _clip(2)
    assert doc.clear_range(0, 0, 1, 1)
    doc.history.clear()
    assert doc.filter_range("brightness / contrast", BRIGHTER, 0, 0, 0, 1)
    assert _cel(doc, 0, 1) is None


def test_a_range_with_no_cels_at_all_pushes_nothing():
    doc = _clip(2)
    assert doc.clear_range(0, 0, 0, 1)
    doc.history.clear()
    assert not doc.filter_range("brightness / contrast", BRIGHTER, 0, 0, 0, 1)
    assert len(doc.history) == 0


def test_a_filter_that_changes_nothing_pushes_nothing():
    doc = _clip(2)
    assert not doc.filter_range(
        "brightness / contrast", {"brightness": 0.0, "contrast": 0.0}, 0, 0, 0, 1
    )
    assert len(doc.history) == 0


def test_the_selection_is_honoured_as_a_weight_not_a_rectangle():
    doc = _clip(2)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0:2, 0:4] = 255
    mask[2, 0:4] = 128  # a feathered row
    doc.mask = SelectionMask(mask)
    before = _cel(doc, 0, 0).pixels.copy()

    assert doc.filter_range("brightness / contrast", BRIGHTER, 0, 0, 0, 1)
    after = _cel(doc, 0, 0).pixels
    full = filters.apply_named("brightness / contrast", before, **BRIGHTER)
    # Inside the selection, the filter in full; outside it, nothing at all; and
    # the half-covered row lands strictly between the two.
    assert np.array_equal(after[0:2], full[0:2])
    assert np.array_equal(after[3], before[3])
    assert not np.array_equal(after[2], before[2])
    assert not np.array_equal(after[2], full[2])


def test_a_range_filter_touches_only_the_tracks_it_covers():
    doc = _clip(2, tracks=2)
    untouched = _cel(doc, 1, 0).pixels.copy()
    assert doc.filter_range("brightness / contrast", BRIGHTER, 0, 0, 0, 1)
    assert np.array_equal(_cel(doc, 1, 0).pixels, untouched)


def test_a_range_filter_stamps_the_cels_it_wrote_and_not_the_rest():
    doc = _clip(3)
    watched = _cel(doc, 0, 2)
    stamps = {i: doc.layer_stamp(_cel(doc, 0, i).uid) for i in range(3)}

    assert doc.filter_range("brightness / contrast", BRIGHTER, 0, 0, 0, 1)
    assert doc.layer_stamp(_cel(doc, 0, 0).uid) > stamps[0]
    assert doc.layer_stamp(watched.uid) == stamps[2]


def test_a_range_filter_commits_a_float_before_it_reads_the_cels():
    """Clamp -> ``commit_floating`` -> mutate, the order every op here follows.
    A floating buffer is pixels the user can see that no layer holds, so
    filtering around it filters a picture that is not the one on screen -- and
    the cel its commit conjures has to be in the target set, not missed by it.
    """
    from warlock.studio.inker.selection import FloatingBuffer

    doc = _clip(2)
    assert doc.clear_range(0, 0, 1, 1)  # frame 1 is empty
    doc.set_current_frame(1)
    plane = np.zeros((4, 4, 4), dtype=np.uint8)
    plane[:] = GREY
    doc.floating = FloatingBuffer(
        pixels=plane,
        mask=np.full((4, 4), 255, dtype=np.uint8),
        offset=(0, 0),
        layer_uid=doc.stack.active.uid,
    )
    doc.history.clear()

    assert doc.filter_range("brightness / contrast", BRIGHTER, 0, 0, 0, 1)
    assert doc.floating is None
    landed = _cel(doc, 0, 1)
    assert landed is not None
    # Brightened, i.e. the cel the commit conjured was filtered with the rest
    # rather than left as the only unfiltered frame in the range.
    assert int(landed.pixels[0, 0, 0]) > GREY[0]


def test_a_still_document_has_no_range_to_filter():
    doc = Document.blank(4, 4)
    _paint(doc)
    doc.history.clear()
    assert not doc.filter_range("brightness / contrast", BRIGHTER, 0, 0, 0, 0)
    assert len(doc.history) == 0
