"""A conversion session names the planes it is previewing, and keeps naming them.

The filter session's suite, one dimension wider. ``begin_convert`` lives across
frames -- the popup stays up while the user tries three matrices -- and it
records every real cel on the current frame by **uid**, for the reason
``test_inker_filter_session`` states at length: an index stops naming the thing
it named the moment anything moves, and on an animated document the plane may be
on a frame the playhead has since left.

Two things differ from the filter session on purpose, and both are asserted
below. Empty cels are skipped rather than autovivified -- a mode change must not
populate the timeline. And a plane that goes away under the popup is *skipped*
rather than cancelling the session, because the other planes are still
previewing and still have to be put back.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker.document import Document

BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
RAMP = [BLACK, WHITE]
RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
GREY = (100, 100, 100, 255)


def _still() -> Document:
    """Two layers of flat mid-grey: a colour no entry of RAMP is, so "did this
    plane convert" is a question one pixel answers."""
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[:] = GREY
    doc.add_layer(name="Second")
    doc.stack.active.pixels[:] = (160, 160, 160, 255)
    return doc


def _animated() -> Document:
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[:] = GREY
    doc.add_frame()  # animates; the new frame holds no cel
    return doc


# --- the session itself ------------------------------------------------------


def test_a_preview_converts_every_layer_of_the_frame():
    doc = _still()
    assert doc.begin_convert() is True
    assert doc.preview_convert(RAMP, "nearest") is True
    for layer in doc.stack:
        assert set(np.unique(layer.pixels[..., 0]).tolist()) <= {0, 255}


def test_a_preview_pushes_nothing_and_changes_no_table():
    doc = _still()
    head = doc.history.head
    doc.begin_convert()
    doc.preview_convert(RAMP, "bayer4")
    assert doc.history.head == head
    assert doc.palette is None


def test_cancelling_puts_every_plane_back_exactly():
    doc = _still()
    before = [layer.pixels.copy() for layer in doc.stack]
    doc.begin_convert()
    doc.preview_convert(RAMP, "floyd-steinberg")

    assert doc.cancel_convert() is True

    for layer, pixels in zip(doc.stack, before, strict=True):
        assert np.array_equal(layer.pixels, pixels)
    assert doc._convert is None


def test_committing_is_one_undo_step_that_lands_on_the_previewed_pixels():
    doc = _still()
    doc.begin_convert()
    doc.preview_convert(RAMP, "bayer8")
    previewed = [layer.pixels.copy() for layer in doc.stack]
    depth = len(doc.history)

    assert doc.commit_convert(RAMP, "bayer8") is True

    assert len(doc.history) == depth + 1
    for layer, pixels in zip(doc.stack, previewed, strict=True):
        assert np.array_equal(layer.pixels, pixels)
    assert doc.palette == RAMP


def test_the_undo_lands_on_the_document_the_user_had_and_not_on_the_preview():
    """The reason the commit restores first: without it, the snapshot
    ``convert_to_palette`` takes would record the *preview* as the state to
    undo to, and one Ctrl+Z would land on a document that never existed."""
    doc = _still()
    original = [layer.pixels.copy() for layer in doc.stack]

    doc.begin_convert()
    doc.preview_convert(RAMP, "floyd-steinberg")
    doc.commit_convert(RAMP, "floyd-steinberg")
    doc.undo()

    for layer, pixels in zip(doc.stack, original, strict=True):
        assert np.array_equal(layer.pixels, pixels)
    assert doc.palette is None


def test_changing_the_method_previews_from_the_snapshot_and_not_from_the_last_preview():
    """Compounding is the bug this shape exists to stop: a second conversion of
    an already-converted plane is not what the second method would have done to
    the drawing."""
    direct = _still()
    direct.begin_convert()
    direct.preview_convert(RAMP, "bayer8")
    expected = [layer.pixels.copy() for layer in direct.stack]

    stepped = _still()
    stepped.begin_convert()
    stepped.preview_convert(RAMP, "nearest")
    stepped.preview_convert(RAMP, "floyd-steinberg")
    stepped.preview_convert(RAMP, "bayer8")

    for layer, pixels in zip(stepped.stack, expected, strict=True):
        assert np.array_equal(layer.pixels, pixels)


def test_opening_a_second_session_cancels_the_first():
    doc = _still()
    before = [layer.pixels.copy() for layer in doc.stack]
    doc.begin_convert()
    doc.preview_convert(RAMP, "bayer2")

    doc.begin_convert()

    for (_uid, snapshot), pixels in zip(doc._convert, before, strict=True):
        assert np.array_equal(snapshot, pixels), "the new session snapshotted the original"


def test_nothing_happens_without_a_session():
    doc = _still()
    assert doc.preview_convert(RAMP) is False
    assert doc.commit_convert(RAMP) is False
    assert doc.cancel_convert() is False


def test_an_unknown_method_previews_nothing_rather_than_raising():
    doc = _still()
    doc.begin_convert()
    assert doc.preview_convert(RAMP, "atkinson") is False


def test_an_empty_table_previews_nothing():
    doc = _still()
    doc.begin_convert()
    assert doc.preview_convert([], "nearest") is False


# --- the memo ----------------------------------------------------------------


def _counting_convert(calls: list[tuple]):
    from warlock.studio.inker import dither as real

    # The real function is captured *now*, before the monkeypatch replaces the
    # attribute -- looking it up through the module afterwards finds this
    # wrapper and recurses.
    original = real.convert

    def convert(pixels, palette, method="nearest"):
        calls.append((tuple(palette), method))
        return original(pixels, palette, method)

    return convert


def test_an_unchanged_table_and_method_is_computed_once_not_once_per_frame(monkeypatch):
    from warlock.studio.inker import _doc_paint

    calls: list[tuple] = []
    monkeypatch.setattr(_doc_paint.dither, "convert", _counting_convert(calls))
    doc = _still()
    doc.begin_convert()
    for _ in range(10):
        assert doc.preview_convert(RAMP, "bayer4") is True
    assert len(calls) == 2, "two layers, converted once each"


def test_changing_the_method_recomputes(monkeypatch):
    from warlock.studio.inker import _doc_paint

    calls: list[tuple] = []
    monkeypatch.setattr(_doc_paint.dither, "convert", _counting_convert(calls))
    doc = _still()
    doc.begin_convert()
    doc.preview_convert(RAMP, "nearest")
    doc.preview_convert(RAMP, "bayer4")
    doc.preview_convert(RAMP, "nearest")
    assert [method for _table, method in calls] == [
        "nearest", "nearest", "bayer4", "bayer4", "nearest", "nearest"
    ]


def test_changing_the_table_recomputes(monkeypatch):
    from warlock.studio.inker import _doc_paint

    calls: list[tuple] = []
    monkeypatch.setattr(_doc_paint.dither, "convert", _counting_convert(calls))
    doc = _still()
    doc.begin_convert()
    doc.preview_convert(RAMP, "nearest")
    doc.preview_convert([BLACK, RED, WHITE], "nearest")
    assert len(calls) == 4


def test_a_new_session_does_not_reuse_the_last_ones_pixels(monkeypatch):
    from warlock.studio.inker import _doc_paint

    calls: list[tuple] = []
    monkeypatch.setattr(_doc_paint.dither, "convert", _counting_convert(calls))
    doc = _still()
    doc.begin_convert()
    doc.preview_convert(RAMP, "bayer4")
    doc.cancel_convert()
    doc.begin_convert()
    doc.preview_convert(RAMP, "bayer4")
    assert len(calls) == 4


def test_cancelling_drops_the_memo_too():
    doc = _still()
    doc.begin_convert()
    doc.preview_convert(RAMP, "bayer4")
    doc.cancel_convert()
    assert doc._convert_memo is None


# --- an animated document ----------------------------------------------------


def test_an_empty_slot_is_skipped_rather_than_brought_into_existence():
    """A mode change must not populate the timeline, and there is nothing for a
    conversion to do to an empty cel anyway."""
    doc = _animated()
    doc.set_current_frame(1)
    assert doc.anim is not None
    assert doc.anim.is_placeholder(doc.stack.active)

    assert doc.begin_convert() is False
    assert doc._convert is None
    assert doc.anim.is_placeholder(doc.stack.active)


def test_the_playhead_moving_does_not_move_the_preview_to_another_cel():
    doc = _animated()
    doc.set_current_frame(0)
    before = doc.stack.active.pixels.copy()

    doc.begin_convert()
    doc.preview_convert(RAMP, "nearest")
    doc.set_current_frame(1)  # an empty slot: a read-only placeholder plane

    # It would raise here if the session addressed a position rather than a uid.
    assert doc.cancel_convert() is True
    doc.set_current_frame(0)
    assert np.array_equal(doc.stack.active.pixels, before)


def test_a_commit_from_another_frame_still_converts_the_whole_document():
    doc = _animated()
    doc.set_current_frame(0)
    doc.begin_convert()
    doc.preview_convert(RAMP, "nearest")
    doc.set_current_frame(1)

    assert doc.commit_convert(RAMP, "nearest") is True

    doc.set_current_frame(0)
    assert set(np.unique(doc.stack.active.pixels[..., 0]).tolist()) <= {0, 255}


def test_the_session_records_uids_and_not_positions():
    doc = _still()
    uids = [layer.uid for layer in doc.stack]
    doc.begin_convert()
    assert [uid for uid, _pixels in doc._convert] == uids
    doc.move_layer(0, 1)
    assert doc._convert_target(uids[0], doc._convert[0][1]) is doc.stack.by_uid(uids[0])
    doc.cancel_convert()


def test_a_layer_deleted_under_the_popup_is_skipped_and_the_rest_still_restore():
    doc = _still()
    survivor = doc.stack[0].uid
    before = doc.stack[0].pixels.copy()

    doc.begin_convert()
    doc.preview_convert(RAMP, "bayer4")
    doc.remove_layer(1)

    assert doc.cancel_convert() is True
    assert np.array_equal(doc.stack.by_uid(survivor).pixels, before)


def test_a_cel_cleared_under_the_popup_is_skipped_rather_than_written_to():
    doc = _animated()
    doc.set_current_frame(0)
    uid = doc.stack.active.uid
    doc.begin_convert()
    doc.preview_convert(RAMP, "nearest")
    doc.clear_cel()
    assert doc.anim is not None and doc.anim.is_placeholder(doc.stack.active)

    assert doc._convert_target(uid, doc._convert[0][1]) is None
    assert doc.cancel_convert() is True


def test_a_placeholder_is_never_written_to_even_wearing_a_recorded_uid():
    """The one case a uid alone does not settle: a placeholder's plane is the
    shared read-only blank, which raises on the first write. Placeholder uids
    are stable per slot, so a session cannot assume one is a cel."""
    doc = _animated()
    doc.set_current_frame(0)
    doc.begin_convert()
    doc.clear_cel()
    assert doc.anim is not None
    placeholder = doc.stack.active
    assert doc.anim.is_placeholder(placeholder)

    assert doc._convert_target(placeholder.uid, doc._convert[0][1]) is None
    assert not placeholder.pixels.flags.writeable
