"""A filter session names the layer it is filtering, and keeps naming it.

The session opened by ``begin_filter`` lives across frames -- the popup stays up
while the user drags a slider -- and it used to record only a rectangle and the
pixels under it. Commit and cancel then acted on whatever ``stack.active``
happened to be by the time the user pressed a button, so selecting another layer
or stepping the playhead mid-popup applied the filter to a layer that had never
been previewed, left the previewed one holding a preview nothing would take
back, and on an animated document could aim the write at the shared **read-only**
placeholder plane -- a ``ValueError`` out of the middle of a draw.

The fix is ``undo.py``'s own rule applied one level up: address the layer by
uid, resolve through ``layer_by_uid`` (which answers on any frame, exactly as
``PatchEdit`` does), and when the uid no longer names anything writable, cancel
the session cleanly instead of writing somewhere.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker.document import Document
from warlock.studio.inker.layers import Layer

FILTER = "brightness / contrast"
RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def _still() -> Document:
    """Two layers, each a flat colour, so "which layer got written" is a
    question a single pixel answers."""
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[:] = RED
    doc.add_layer(name="Second")
    doc.stack.active.pixels[:] = BLUE
    return doc


def _animated() -> Document:
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[:] = RED
    doc.add_frame()  # animates; the new frame holds no cel
    return doc


# --- a still document: the active layer moves under the popup ----------------


def test_the_filter_lands_on_the_layer_it_was_opened_on():
    doc = _still()
    filtered_uid = doc.stack.active.uid
    other = doc.stack[0]
    other_before = other.pixels.copy()

    assert doc.begin_filter() is not None
    doc.set_active_layer(0)  # the user clicks another layer with the popup up
    assert doc.preview_filter(FILTER, brightness=0.5) is True
    assert doc.commit_filter() is True

    assert np.array_equal(other.pixels, other_before), "the other layer is untouched"
    assert not np.array_equal(doc.stack.by_uid(filtered_uid).pixels[0, 0], BLUE)


def test_cancelling_puts_the_pixels_back_on_the_right_layer():
    doc = _still()
    filtered_uid = doc.stack.active.uid
    before = doc.stack.active.pixels.copy()
    other_before = doc.stack[0].pixels.copy()

    assert doc.begin_filter() is not None
    doc.preview_filter(FILTER, brightness=0.5)
    doc.set_active_layer(0)
    assert doc.cancel_filter() is True

    assert np.array_equal(doc.stack.by_uid(filtered_uid).pixels, before)
    assert np.array_equal(doc.stack[0].pixels, other_before)


def test_a_layer_deleted_under_the_popup_cancels_the_session():
    """Nothing left to put the pixels back into, so the session goes -- rather
    than the write landing on whichever layer inherited the active index."""
    doc = _still()
    survivor_before = doc.stack[0].pixels.copy()

    assert doc.begin_filter() is not None
    doc.preview_filter(FILTER, brightness=0.5)
    doc.remove_layer(doc.stack.active_index)

    assert doc.commit_filter() is False
    assert doc._filter is None
    assert np.array_equal(doc.stack[0].pixels, survivor_before)


# --- an animated document: the playhead moves under the popup ----------------


def test_the_playhead_moving_does_not_move_the_filter_to_another_cel():
    doc = _animated()
    doc.set_current_frame(0)
    cel = doc.stack.active
    assert doc.anim is not None and not doc.anim.is_placeholder(cel)

    assert doc.begin_filter() is not None
    doc.preview_filter(FILTER, brightness=0.5)
    doc.set_current_frame(1)  # an empty slot: a read-only placeholder plane
    assert doc.anim.is_placeholder(doc.stack.active)

    # It used to raise here, writing into the shared blank plane.
    assert doc.commit_filter() is True
    doc.set_current_frame(0)
    assert not np.array_equal(doc.stack.active.pixels[0, 0], RED)


def test_a_preview_survives_the_playhead_and_is_still_undoable_as_one_step():
    doc = _animated()
    doc.set_current_frame(0)
    head = doc.history.head
    before = doc.stack.active.pixels.copy()

    assert doc.begin_filter() is not None
    doc.preview_filter(FILTER, brightness=0.5)
    doc.set_current_frame(1)
    assert doc.commit_filter() is True
    assert doc.history.head != head

    doc.undo()
    doc.set_current_frame(0)
    assert np.array_equal(doc.stack.active.pixels, before)


def test_cancelling_after_the_playhead_moved_restores_the_cel_it_previewed():
    doc = _animated()
    doc.set_current_frame(0)
    before = doc.stack.active.pixels.copy()

    assert doc.begin_filter() is not None
    doc.preview_filter(FILTER, brightness=0.5)
    doc.set_current_frame(1)
    assert doc.cancel_filter() is True

    doc.set_current_frame(0)
    assert np.array_equal(doc.stack.active.pixels, before)


def test_a_session_opened_on_an_empty_slot_writes_to_its_own_cel():
    """``begin_filter`` autovivifies, so the session's uid names a real cel from
    the start -- and the blank plane every *other* empty slot shares is never
    written, which is the property that stops one filter appearing on every
    empty frame in the document at once."""
    doc = _animated()
    doc.set_current_frame(1)
    assert doc.anim is not None
    blank = doc.anim.blank_plane(*doc.size)

    assert doc.begin_filter() is not None
    doc.preview_filter(FILTER, brightness=0.5)
    assert doc.commit_filter() is True
    assert doc.stack.active.pixels is not blank
    assert not blank.flags.writeable


def test_the_session_records_a_uid_and_not_a_position():
    """The shape of the fix, pinned directly: a reorder must not retarget an
    open session, and the only way that can be true is if the session is not
    holding an index."""
    doc = _still()
    uid = doc.stack.active.uid
    assert doc.begin_filter() is not None
    assert doc._filter is not None
    assert doc._filter[2] == uid
    doc.move_layer(doc.stack.active_index, 0)
    assert doc._filter_layer() is doc.stack.by_uid(uid)
    doc.cancel_filter()


def test_a_placeholder_wearing_the_sessions_uid_is_not_written_to():
    """The one case a uid alone does not settle. Placeholder uids are stable per
    slot, so undoing the autovivify hands the empty slot the same number back --
    and its plane is the shared read-only one."""
    doc = _animated()
    doc.set_current_frame(1)
    assert doc.begin_filter() is not None
    uid = doc._filter[2]
    doc._discard_pending_cel()  # the autovivify undone under the popup
    assert doc.anim is not None
    assert doc.stack.active.uid == uid
    assert doc.anim.is_placeholder(doc.stack.active)

    assert doc._filter_layer() is None
    assert doc.commit_filter() is False
    assert doc._filter is None


def test_an_orphaned_cel_is_dropped_when_the_session_is_abandoned():
    """A cancelled session must not leave the cel it autovivified behind: an
    empty cel is a change to the document, and one nobody asked for."""
    doc = _animated()
    doc.set_current_frame(1)
    head = doc.history.head
    assert doc.begin_filter() is not None
    assert doc.anim is not None
    assert not doc.anim.is_placeholder(doc.stack.active)
    doc.cancel_filter()
    assert doc.anim.is_placeholder(doc.stack.active)
    assert doc.history.head == head


def test_layers_that_never_took_part_are_never_touched():
    """A three-layer document, filtered on the middle one, with the active layer
    moved twice under the popup."""
    doc = _still()
    doc.add_layer(name="Third")
    doc.stack.active.pixels[:] = (0, 255, 0, 255)
    doc.set_active_layer(1)
    target = doc.stack.active.uid
    untouched = {layer.uid: layer.pixels.copy() for layer in doc.stack if layer.uid != target}

    assert doc.begin_filter() is not None
    doc.preview_filter(FILTER, brightness=0.5)
    doc.set_active_layer(0)
    doc.preview_filter(FILTER, brightness=0.5)
    doc.set_active_layer(2)
    assert doc.commit_filter() is True

    for uid, pixels in untouched.items():
        assert np.array_equal(doc.stack.by_uid(uid).pixels, pixels), uid


def test_a_document_with_no_animation_still_takes_the_plain_path():
    """The still case, unchanged to the letter: the layer resolves, the filter
    lands, and one step is pushed."""
    doc = Document.blank(4, 4)
    doc.stack.active.pixels[:] = RED
    head = doc.history.head
    assert doc.begin_filter() == (0, 0, 4, 4)
    assert doc.preview_filter(FILTER, brightness=0.5) is True
    assert doc.commit_filter() is True
    assert doc.history.head != head
    assert isinstance(doc.stack.active, Layer)


# --- the session memo --------------------------------------------------------
#
# ``inker_bridge`` calls ``preview_filter`` on every frame the popup is up, on
# purpose: the combo above it can switch filters, and a preview that only ran on
# a slider move would leave the last filter's pixels under the new filter's
# controls. That made a 2048-square blur cost its full ~1.1 s *per frame*. The
# snapshot the session holds never changes, so the filtered array is a pure
# function of (name, params) within one session and is memoised. Only that half:
# the blend, the alpha lock and the invalidate still run every frame.


def _counting_apply(calls: list[tuple]):
    from warlock.studio.inker import filters as real

    def apply_named(name, before, **params):
        calls.append((name, tuple(sorted(params.items()))))
        return real.FILTERS[name][1](before, **params)

    return apply_named


def test_an_unchanged_filter_and_params_is_computed_once_not_once_per_frame(monkeypatch):
    from warlock.studio.inker import document as docmod

    calls: list[tuple] = []
    monkeypatch.setattr(docmod.filters, "apply_named", _counting_apply(calls))
    doc = _still()
    doc.begin_filter()
    for _ in range(10):
        assert doc.preview_filter(FILTER, brightness=0.5) is True
    assert len(calls) == 1


def test_changing_a_parameter_recomputes(monkeypatch):
    from warlock.studio.inker import document as docmod

    calls: list[tuple] = []
    monkeypatch.setattr(docmod.filters, "apply_named", _counting_apply(calls))
    doc = _still()
    doc.begin_filter()
    doc.preview_filter(FILTER, brightness=0.5)
    doc.preview_filter(FILTER, brightness=0.75)
    doc.preview_filter(FILTER, brightness=0.5)
    assert len(calls) == 3


def test_switching_filters_recomputes_which_is_why_the_frame_call_exists(monkeypatch):
    from warlock.studio.inker import document as docmod

    calls: list[tuple] = []
    monkeypatch.setattr(docmod.filters, "apply_named", _counting_apply(calls))
    doc = _still()
    doc.begin_filter()
    doc.preview_filter(FILTER, brightness=0.5)
    doc.preview_filter("blur", radius=2.0)
    assert [c[0] for c in calls] == [FILTER, "blur"]


def test_a_new_session_does_not_reuse_the_last_ones_pixels(monkeypatch):
    """The memo is keyed on (name, params) and nothing else, so it *must* be
    dropped when the snapshot underneath it changes -- otherwise the second
    session previews the first session's layer."""
    from warlock.studio.inker import document as docmod

    calls: list[tuple] = []
    monkeypatch.setattr(docmod.filters, "apply_named", _counting_apply(calls))
    doc = _still()
    doc.begin_filter()
    doc.preview_filter(FILTER, brightness=0.5)
    doc.commit_filter()
    doc.begin_filter()
    doc.preview_filter(FILTER, brightness=0.5)
    assert len(calls) == 2


def test_the_memo_does_not_skip_the_write_so_the_pixels_are_right_every_frame():
    """Ten identical previews must leave exactly what one leaves -- the cached
    half is the filter, not the blend or the alpha lock beneath it."""
    once = _still()
    once.begin_filter()
    once.preview_filter(FILTER, brightness=0.6)
    expected = once.stack.active.pixels.copy()

    many = _still()
    many.begin_filter()
    for _ in range(10):
        many.preview_filter(FILTER, brightness=0.6)
    assert np.array_equal(many.stack.active.pixels, expected)


def test_cancelling_drops_the_memo_too():
    doc = _still()
    doc.begin_filter()
    doc.preview_filter(FILTER, brightness=0.5)
    doc.cancel_filter()
    assert doc._filter_memo is None
