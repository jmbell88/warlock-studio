"""Sheet corrections through the document: one step, uid-addressed, honest.

Built the way a real sheet arrives -- ``document_from_sheet`` over an atlas
plus an ``animation`` block of the sidecar's own shape -- so what is asserted
is the door a user reaches, not a fixture that happens to resemble it.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import mirror, sheetscope
from warlock.studio.inker.document import Document
from warlock.studio.inker.selection import SelectionMask
from warlock.studio.inker.sheetin import document_from_sheet

CELL = 16
DIRECTIONS = ("front", "left", "back", "right")
FRAMES = 3
BLUE = (40, 60, 200, 255)
SKIN = (200, 150, 120, 255)
RED = (220, 30, 30, 255)


def _figure(mirror_it: bool = False) -> np.ndarray:
    out = np.zeros((CELL, CELL, 4), dtype=np.uint8)
    out[2:6, 6:10] = SKIN
    out[3, 7] = (0, 0, 0, 255)  # one eye, off-centre
    out[6:13, 7:9] = BLUE
    out[7:9, 3:7] = BLUE
    out[7:9, 9:13] = BLUE
    return np.ascontiguousarray(out[:, ::-1]) if mirror_it else out


def _sheet(animations=("walk",)) -> Document:
    count = len(animations) * len(DIRECTIONS) * FRAMES
    columns = 8
    rows = (count + columns - 1) // columns
    atlas = np.zeros((rows * CELL, columns * CELL, 4), dtype=np.uint8)
    cells, tags, index = [], [], 0
    for animation in animations:
        for direction in DIRECTIONS:
            tags.append(
                {
                    "name": f"{animation}_{direction}",
                    "start": index,
                    "end": index + FRAMES - 1,
                    "loop": True,
                }
            )
            for _frame in range(FRAMES):
                x, y = (index % columns) * CELL, (index // columns) * CELL
                atlas[y : y + CELL, x : x + CELL] = _figure(direction == "right")
                cells.append({"x": x, "y": y, "w": CELL, "h": CELL})
                index += 1
    doc = document_from_sheet(atlas, cells, {"tags": tags, "frames": []})
    doc.history.clear()
    return doc


def _cel(doc: Document, frame: int):
    anim = doc.anim
    return anim.cels.get((anim.tracks[0].uid, anim.frames[frame].uid))


def _track_uid(doc: Document) -> int:
    return doc.anim.tracks[0].uid


def _dab(doc: Document, frame: int, xy=(4, 12)) -> np.ndarray:
    """Paint one red pixel on ``frame`` and return the mark weight for it."""
    before = _cel(doc, frame).pixels.copy()
    _cel(doc, frame).pixels[xy[1], xy[0]] = RED
    weight = mirror.changed_weight(before, _cel(doc, frame).pixels)
    assert weight is not None
    return weight


def test_a_sheet_document_knows_its_runs():
    doc = _sheet()
    assert doc.has_sheet()
    assert [run.direction for run in doc.sheet_runs()] == list(DIRECTIONS)
    assert not Document.blank(4, 4).has_sheet()


def test_a_patch_propagates_to_every_direction_as_one_step():
    doc = _sheet()
    source = 1  # walk_front frame 1
    weight = _dab(doc, source)
    targets = sheetscope.frames_for(doc.sheet_runs(), source, "directions")
    assert targets == [4, 7, 10]
    snapshot = {f: _cel(doc, f).pixels.copy() for f in range(12)}

    assert doc.propagate_patch(_track_uid(doc), source, targets, weight)
    for frame in targets:
        assert tuple(_cel(doc, frame).pixels[12, 4]) == RED
    # Only the marked pixel moved; the rest of every target is untouched.
    for frame in targets:
        expected = snapshot[frame].copy()
        expected[12, 4] = RED
        assert np.array_equal(_cel(doc, frame).pixels, expected)
    # Frames outside the scope are untouched.
    for frame in (0, 2, 5, 11):
        assert np.array_equal(_cel(doc, frame).pixels, snapshot[frame])

    assert len(doc.history) == 1
    assert doc.undo()
    for frame in targets:
        assert np.array_equal(_cel(doc, frame).pixels, snapshot[frame])
    assert doc.redo()
    assert tuple(_cel(doc, 7).pixels[12, 4]) == RED


def test_a_correction_that_changes_nothing_pushes_nothing():
    doc = _sheet()
    weight = np.zeros((CELL, CELL), dtype=np.uint8)
    weight[12, 4] = 255
    # The source pixel there is transparent, and so are the targets.
    assert not doc.propagate_patch(_track_uid(doc), 1, [4, 7], weight)
    assert len(doc.history) == 0
    # And a mark of nothing at all is a no-op rather than a full-cell write.
    assert not doc.propagate_patch(
        _track_uid(doc), 1, [4], np.zeros((CELL, CELL), dtype=np.uint8)
    )
    assert len(doc.history) == 0


def test_a_linked_cel_is_written_once_and_the_source_is_never_a_target():
    doc = _sheet()
    anim = doc.anim
    shared = _cel(doc, 4)
    anim.cels[(anim.tracks[0].uid, anim.frames[5].uid)] = shared
    weight = _dab(doc, 1)
    assert doc.propagate_patch(_track_uid(doc), 1, [4, 5, 1], weight)
    assert len(doc.history) == 1
    assert tuple(shared.pixels[12, 4]) == RED
    # The step holds one edit for the shared cel, not two.
    assert doc.undo()
    assert tuple(shared.pixels[12, 4]) != RED


def test_replacing_a_colour_across_the_sheet_recolours_every_cell():
    doc = _sheet()
    frames = sheetscope.frames_for(doc.sheet_runs(), 0, "sheet")
    assert doc.replace_colour_frames(_track_uid(doc), [0, *frames], BLUE, RED)
    for frame in range(12):
        pixels = _cel(doc, frame).pixels
        assert tuple(pixels[8, 7]) == RED
        assert tuple(pixels[2, 6]) == SKIN
    assert len(doc.history) == 1
    assert doc.undo()
    assert tuple(_cel(doc, 11).pixels[8, 7]) == BLUE


def test_a_recolour_honours_the_selection():
    doc = _sheet()
    mask = np.zeros((CELL, CELL), dtype=np.uint8)
    mask[6:13, 7:9] = 255  # the body column only
    doc.mask = SelectionMask(mask)
    assert doc.replace_colour_frames(_track_uid(doc), [0, 3], BLUE, RED)
    for frame in (0, 3):
        pixels = _cel(doc, frame).pixels
        assert tuple(pixels[8, 7]) == RED
        assert tuple(pixels[7, 4]) == BLUE  # the arm kept its colour


def test_shifting_a_selection_moves_it_on_every_frame_and_needs_a_selection():
    doc = _sheet()
    with pytest.raises(ValueError, match="select"):
        doc.shift_frames(_track_uid(doc), [0, 1], 1, 0)
    mask = np.zeros((CELL, CELL), dtype=np.uint8)
    mask[7:9, 3:7] = 255  # the left arm
    doc.mask = SelectionMask(mask)
    assert doc.shift_frames(_track_uid(doc), [0, 1, 2], 0, 2)
    for frame in (0, 1, 2):
        pixels = _cel(doc, frame).pixels
        assert tuple(pixels[9, 3]) == BLUE
        assert tuple(pixels[7, 3]) == (0, 0, 0, 0)
    assert len(doc.history) == 1
    assert not doc.shift_frames(_track_uid(doc), [0], 0, 0)


def test_a_mirror_onto_an_already_mirrored_view_changes_nothing_outside_the_face():
    doc = _sheet()
    sheet = doc.sheet_runs()
    left = 3  # walk_left frame 0
    right = sheetscope.counterpart(sheet, left)
    assert right == 9
    # The right view is the exact mirror of the left except for its eye,
    # which an artist drew independently -- so with the face excluded there
    # is nothing to write and no step is pushed.
    _cel(doc, right).pixels[3, 8] = SKIN
    _cel(doc, right).pixels[3, 7] = (0, 0, 0, 255)
    assert not doc.mirror_to(_track_uid(doc), left, right)
    assert len(doc.history) == 0
    # With no face excluded the eye differs and the write happens.
    assert doc.mirror_to(_track_uid(doc), left, right, face_fraction=0.0)
    assert len(doc.history) == 1
    assert np.array_equal(_cel(doc, right).pixels, mirror.mirrored(_cel(doc, left).pixels))


def test_a_fix_on_the_left_lands_mirrored_on_the_right_with_the_face_kept():
    doc = _sheet()
    left, right = 3, 9
    _cel(doc, left).pixels[10, 4] = RED  # a longer arm, well below the face
    _cel(doc, left).pixels[3, 7] = RED  # and a red eye, inside the face
    before_right = _cel(doc, right).pixels.copy()
    assert doc.mirror_to(_track_uid(doc), left, right)
    after = _cel(doc, right).pixels
    assert tuple(after[10, CELL - 1 - 4]) == RED
    # The right view's own eye is untouched.
    assert np.array_equal(after[2:6, :], before_right[2:6, :])
    assert doc.undo()
    assert np.array_equal(_cel(doc, right).pixels, before_right)


def test_a_whole_run_mirrors_as_one_step():
    doc = _sheet()
    for frame in (3, 4, 5):
        _cel(doc, frame).pixels[10 + (frame - 3), 4] = RED
    run = next(r for r in doc.sheet_runs() if r.direction == "left")
    assert doc.mirror_run(_track_uid(doc), run)
    assert len(doc.history) == 1
    for frame in (9, 10, 11):
        assert tuple(_cel(doc, frame).pixels[10 + (frame - 9), CELL - 1 - 4]) == RED
    assert doc.undo()
    assert tuple(_cel(doc, 9).pixels[10, CELL - 1 - 4]) != RED


def test_a_direction_with_no_mirror_has_no_counterpart_to_write():
    doc = _sheet()
    run = next(r for r in doc.sheet_runs() if r.direction == "front")
    assert not doc.mirror_run(_track_uid(doc), run)
    assert len(doc.history) == 0


def test_an_indexed_sheet_takes_the_index_patch_path():
    from warlock.studio.inker.undo import IndexPatchEdit

    doc = _sheet()
    palette = [(0, 0, 0, 255), BLUE, SKIN, RED, (0, 0, 0, 0)]
    assert doc.convert_to_indexed(palette)
    doc.history.clear()
    weight = _dab(doc, 1)
    assert doc.propagate_patch(_track_uid(doc), 1, [4], weight)
    step = doc.history.top
    edits = getattr(step, "edits", [step])
    assert any(isinstance(edit, IndexPatchEdit) for edit in edits)


def test_alpha_lock_keeps_the_target_transparent_where_it_was():
    doc = _sheet()
    doc.anim.tracks[0].alpha_lock = True
    weight = _dab(doc, 1, xy=(0, 0))  # a pixel on transparent background
    doc.propagate_patch(_track_uid(doc), 1, [4], weight)
    assert _cel(doc, 4).pixels[0, 0, 3] == 0
    # And a locked write onto an opaque pixel keeps that pixel's alpha too.
    weight = _dab(doc, 1, xy=(7, 8))
    assert doc.propagate_patch(_track_uid(doc), 1, [4], weight)
    assert tuple(_cel(doc, 4).pixels[8, 7]) == RED


def test_a_tilemap_is_refused_by_name():
    from warlock.studio.inker.tiles import TilemapCel

    doc = _sheet()
    anim = doc.anim
    target = _cel(doc, 4)
    fake = TilemapCel.__new__(TilemapCel)
    fake.__dict__.update(target.__dict__)
    anim.cels[(anim.tracks[0].uid, anim.frames[4].uid)] = fake
    weight = _dab(doc, 1)
    with pytest.raises(ValueError, match="tilemap"):
        doc.propagate_patch(_track_uid(doc), 1, [4], weight)


def test_an_unknown_track_is_refused_by_name():
    doc = _sheet()
    with pytest.raises(ValueError, match="track"):
        doc.replace_colour_frames(999, [0], BLUE, RED)
