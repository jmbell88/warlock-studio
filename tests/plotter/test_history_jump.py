"""Jumping to a position in the undo stack, which the history panel asks for.

The stack has had ``step_to`` since Inker's panel was built. What Plotter did
not have was a way to *call* it that is safe from this document -- and the
unsafe way is the obvious one: ``doc.history.step_to(doc, n)`` walks the
**stack's** undo and redo, not ``MapDoc``'s, so it commits no open stroke. A
jump made mid-drag would then step over paint that is on the layer and in no
step, leaving the cells ahead of the head. That is precisely the defect
``MapDoc.undo`` was written to prevent, and reaching past it is the one way to
reintroduce it.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.plotter.tilemap import MapDoc
from warlock.studio.tilegrid import gid

#: Adding the layer is itself an undoable step, so every count below is
#: measured from here rather than from zero.
BASE = 1


def _doc() -> MapDoc:
    doc = MapDoc(4, 4, 16, 16)
    doc.add_tile_layer("Ground")
    assert len(doc.history) == BASE
    return doc


def _paint(doc: MapDoc, column: int, row: int, value: int) -> None:
    layer = doc.tile_layers()[0]
    doc.write_region(layer.uid, column, row, np.array([[value]], gid.DTYPE))


def _cells(doc: MapDoc) -> np.ndarray:
    return doc.tile_layers()[0].data


def test_a_jump_walks_back_to_a_named_position_and_forward_again():
    doc = _doc()
    _paint(doc, 0, 0, 1)
    _paint(doc, 1, 0, 2)
    _paint(doc, 2, 0, 3)
    assert len(doc.history) == BASE + 3

    assert doc.step_history(BASE + 1) is True
    assert [int(value) for value in _cells(doc)[0, :3]] == [1, 0, 0]
    assert len(doc.history) == BASE + 1

    # And forward, through the stack's own redo rather than by splicing lists.
    assert doc.step_history(BASE + 3) is True
    assert [int(value) for value in _cells(doc)[0, :3]] == [1, 2, 3]


def test_a_jump_to_where_you_already_are_moves_nothing():
    doc = _doc()
    _paint(doc, 0, 0, 1)
    assert doc.step_history(BASE + 1) is False
    assert len(doc.history) == BASE + 1


def test_a_jump_is_clamped_rather_than_indexing_off_the_end():
    doc = _doc()
    _paint(doc, 0, 0, 1)
    doc.step_history(99)
    assert len(doc.history) == BASE + 1
    doc.step_history(-4)
    # Clamped to zero, which is "the map as opened" -- and this map was opened
    # with no layer at all, because adding one is a step like any other.
    assert len(doc.history) == 0
    assert doc.tile_layers() == []


def test_a_jump_commits_an_open_stroke_first():
    """The reason this is a method on the document and not a call at the panel.

    Reaching straight for ``doc.history.step_to`` steps over paint that is on
    the layer and in no step: the cells end up ahead of the head, and the very
    next undo reverses the step *before* the one the user is looking at.
    """
    doc = _doc()
    layer = doc.tile_layers()[0]
    _paint(doc, 0, 0, 1)
    doc.begin_stroke(layer.uid)
    doc.stroke_write(1, 1, np.array([[gid.compose(5)]], gid.DTYPE))

    # The stroke is live and unrecorded at this point...
    assert len(doc.history) == BASE + 1
    # ...and the jump commits it before moving, so the jump target counts it.
    assert doc.step_history(BASE + 2) is False
    assert len(doc.history) == BASE + 2
    assert int(_cells(doc)[1, 1]) == gid.compose(5)

    # One step back is now the stroke, and nothing is left ahead of the head.
    doc.step_history(BASE + 1)
    assert int(_cells(doc)[1, 1]) == 0
    assert int(_cells(doc)[0, 0]) == 1
