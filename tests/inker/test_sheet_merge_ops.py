"""``merge_render``: what a re-rendered sheet is allowed to do to hand edits.

Built the way a real sheet arrives -- ``document_from_sheet`` over an atlas --
so the base digests are recorded by the importer rather than by the test.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import sheetmerge
from warlock.studio.inker.sheetin import document_from_sheet

CELL = 8
FRAMES = 4


def _cell(value: int) -> np.ndarray:
    out = np.zeros((CELL, CELL, 4), dtype=np.uint8)
    out[..., 3] = 255
    out[2:6, 2:6, 0] = value
    return out


def _atlas(values) -> np.ndarray:
    atlas = np.zeros((CELL, len(values) * CELL, 4), dtype=np.uint8)
    for i, value in enumerate(values):
        atlas[:, i * CELL : (i + 1) * CELL] = _cell(value)
    return atlas


def _sheet(values=(10, 20, 30, 40)):
    cells = [{"x": i * CELL, "y": 0, "w": CELL, "h": CELL} for i in range(len(values))]
    anim = {
        "tags": [{"name": "walk_front", "start": 0, "end": len(values) - 1, "loop": True}],
        "frames": [],
    }
    doc = document_from_sheet(
        _atlas(values), cells, anim, source={"job": "J", "sheet": "S"}
    )
    doc.history.clear()
    return doc


def _track(doc):
    return doc.anim.tracks[0].uid


def _cel(doc, frame: int):
    anim = doc.anim
    return anim.cels[(anim.tracks[0].uid, anim.frames[frame].uid)]


def _paint(doc, frame: int, value: int) -> None:
    """A hand edit, straight onto the cel -- the merge reads pixels, not history."""
    _cel(doc, frame).pixels[...] = _cell(value)


def _incoming(values) -> list[np.ndarray]:
    return [_cell(v) for v in values]


# -- the four outcomes ---------------------------------------------------------


def test_an_untouched_cell_takes_the_render():
    doc = _sheet()
    counts = doc.merge_render(_track(doc), _incoming([99, 20, 30, 40]))

    assert counts.taken == 1
    assert counts.agreed == 3
    assert _cel(doc, 0).pixels[3, 3, 0] == 99


def test_an_edited_cell_the_render_did_not_change_keeps_the_edit():
    doc = _sheet()
    _paint(doc, 1, 77)
    counts = doc.merge_render(_track(doc), _incoming([10, 20, 30, 40]))

    assert counts.kept == 1
    assert counts.taken == 0
    assert _cel(doc, 1).pixels[3, 3, 0] == 77


def test_a_cell_both_changed_is_a_conflict_and_the_edit_stands():
    """**The rule the whole feature exists for.** Nothing painted is ever
    overwritten silently."""
    doc = _sheet()
    _paint(doc, 2, 77)
    counts = doc.merge_render(_track(doc), _incoming([10, 20, 88, 40]))

    assert counts.conflicts == 1
    assert counts.taken == 0
    assert _cel(doc, 2).pixels[3, 3, 0] == 77, "the hand edit must survive"
    assert doc.anim.frames[2].uid in doc.sheet_base.conflicts


def test_a_cell_painted_into_agreement_is_not_a_conflict():
    """The user painted what the renderer has now caught up to. Asking anyone
    to arbitrate between two identical pictures would be a conflict in name."""
    doc = _sheet()
    _paint(doc, 3, 88)
    counts = doc.merge_render(_track(doc), _incoming([10, 20, 30, 88]))

    assert counts.conflicts == 0
    assert counts.agreed == 4
    assert doc.sheet_base.conflicts == set()


def test_the_counts_add_up_to_every_cell():
    doc = _sheet()
    _paint(doc, 1, 77)
    _paint(doc, 2, 77)
    counts = doc.merge_render(_track(doc), _incoming([99, 20, 88, 40]))

    assert counts.total == FRAMES
    assert (counts.taken, counts.kept, counts.conflicts) == (1, 1, 1)


# -- one step ------------------------------------------------------------------


def test_one_undo_restores_both_the_pixels_and_the_recorded_render():
    """A merge undone without its base restores the picture and leaves the
    document's idea of what the renderer gave it in the future -- and the next
    merge would then read every restored edit as untouched."""
    doc = _sheet()
    before_base = dict(doc.sheet_base.digests)
    doc.merge_render(_track(doc), _incoming([99, 20, 30, 40]))
    assert _cel(doc, 0).pixels[3, 3, 0] == 99

    doc.undo()

    assert _cel(doc, 0).pixels[3, 3, 0] == 10
    assert doc.sheet_base.digests == before_base


def test_a_merge_and_its_undo_are_a_single_step():
    doc = _sheet()
    _paint(doc, 2, 77)
    doc.history.clear()
    doc.merge_render(_track(doc), _incoming([99, 11, 88, 40]))

    doc.undo()
    assert _cel(doc, 0).pixels[3, 3, 0] == 10
    assert _cel(doc, 1).pixels[3, 3, 0] == 20
    assert doc.sheet_base.conflicts == set()


def test_a_second_merge_after_an_undo_classifies_from_the_restored_base():
    """The property the single step buys. If the base had stayed advanced, the
    restored edit would read as untouched and be overwritten."""
    doc = _sheet()
    _paint(doc, 2, 77)
    doc.history.clear()
    doc.merge_render(_track(doc), _incoming([10, 20, 88, 40]))
    doc.undo()

    counts = doc.merge_render(_track(doc), _incoming([10, 20, 88, 40]))
    assert counts.conflicts == 1
    assert _cel(doc, 2).pixels[3, 3, 0] == 77


def test_the_base_advances_to_the_render_even_for_a_kept_edit():
    """That is now what the renderer last gave us, which is what makes the
    *next* merge classify correctly."""
    doc = _sheet()
    _paint(doc, 1, 77)
    doc.merge_render(_track(doc), _incoming([10, 20, 30, 40]))

    frame = doc.anim.frames[1].uid
    assert doc.sheet_base.digests[frame] == sheetmerge.cell_digest(_cell(20))


def test_a_merge_that_changes_nothing_still_records_the_render():
    doc = _sheet()
    counts = doc.merge_render(_track(doc), _incoming([10, 20, 30, 40]))
    assert counts.wrote is False
    assert counts.agreed == FRAMES


# -- the refusals --------------------------------------------------------------


def test_a_document_with_no_recorded_render_is_refused_by_name():
    """Two-way is not three-way. Without a base there is no way to tell an edit
    from a render, so the op refuses rather than overwriting on a coin flip."""
    doc = _sheet()
    doc.sheet_base = None
    with pytest.raises(ValueError, match="no recorded render"):
        doc.merge_render(_track(doc), _incoming([99, 20, 30, 40]))


def test_a_frame_count_mismatch_is_refused_with_both_numbers():
    doc = _sheet()
    with pytest.raises(ValueError, match="4 frames and that sheet has 3"):
        doc.merge_render(_track(doc), _incoming([1, 2, 3]))


def test_cells_of_the_wrong_size_are_refused_by_name():
    doc = _sheet()
    wrong = [np.zeros((CELL * 2, CELL, 4), dtype=np.uint8) for _ in range(FRAMES)]
    with pytest.raises(ValueError, match="this document is"):
        doc.merge_render(_track(doc), wrong)


def test_an_alpha_locked_track_is_refused_rather_than_bypassed():
    """The lock is passed into ``masked_apply``, so a locked track would keep
    the old silhouette while the merge reported the render as taken."""
    doc = _sheet()
    doc.anim.tracks[0].alpha_lock = True
    with pytest.raises(ValueError, match="alpha lock"):
        doc.merge_render(_track(doc), _incoming([99, 20, 30, 40]))


def test_a_refused_merge_leaves_the_document_exactly_as_it_was():
    doc = _sheet()
    _paint(doc, 1, 77)
    before = _cel(doc, 1).pixels.copy()
    base = dict(doc.sheet_base.digests)

    with pytest.raises(ValueError):
        doc.merge_render(_track(doc), _incoming([1, 2, 3]))

    assert np.array_equal(_cel(doc, 1).pixels, before)
    assert doc.sheet_base.digests == base


def test_a_linked_cel_is_refused_rather_than_silently_dropped():
    """One cel serving two slots cannot take two different renders, and the
    edit builder dedupes by identity -- so one of them would vanish silently."""
    doc = _sheet()
    anim = doc.anim
    shared = anim.cels[(anim.tracks[0].uid, anim.frames[0].uid)]
    anim.cels[(anim.tracks[0].uid, anim.frames[1].uid)] = shared

    with pytest.raises(ValueError, match="linked cel"):
        doc.merge_render(_track(doc), _incoming([99, 88, 30, 40]))


# -- cells outside the re-rendered runs ----------------------------------------


def test_cells_the_worker_copied_through_classify_themselves():
    """The merge is never told what the subset was. A cell outside it arrives
    as the previous atlas's own pixels, so it compares equal to the base."""
    doc = _sheet()
    _paint(doc, 3, 77)
    # Only cell 0 was re-rendered; 1, 2 and 3 are copies of the old atlas.
    counts = doc.merge_render(_track(doc), _incoming([99, 20, 30, 40]))

    assert counts.taken == 1
    assert counts.kept == 1, "the edit outside the subset is kept, not re-taken"
    assert _cel(doc, 3).pixels[3, 3, 0] == 77
