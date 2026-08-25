"""The paint bucket's three Aseprite options: refer, connectivity, grid.

Each is a property of *which pixels the flood may reach*, which is a property
of an array -- so each is asserted against one, without a window.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio import inker

RED = (255, 0, 0, 255)
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)


def _doc(size=(16, 16), colour=WHITE):
    pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    pixels[:, :] = colour
    return inker.Document.from_pixels(pixels)


def _at(doc, x, y):
    return tuple(int(v) for v in doc.composite[y, x])


# --- refer to ---------------------------------------------------------------


def _split_by_a_layer_above():
    """A white active layer with a black bar drawn on the layer *above* it."""
    doc = _doc((16, 16), WHITE)
    doc.add_layer()
    doc.stack.active.pixels[8, :] = BLACK
    doc.set_active_layer(0)
    doc.invalidate_all()
    return doc


def test_the_default_fill_refers_to_the_canvas_and_stops_at_a_layer_above():
    doc = _split_by_a_layer_above()
    doc.fill((0, 0), RED, thresh=0)
    assert tuple(int(v) for v in doc.stack[0].pixels[0, 0]) == RED
    assert tuple(int(v) for v in doc.stack[0].pixels[15, 0]) == WHITE


def test_referring_to_the_layer_crosses_what_is_only_on_the_canvas():
    doc = _split_by_a_layer_above()
    doc.fill((0, 0), RED, thresh=0, refer="layer")
    assert tuple(int(v) for v in doc.stack[0].pixels[15, 0]) == RED


def test_an_unknown_refer_is_refused_by_name():
    doc = _doc()
    with pytest.raises(ValueError, match="refer"):
        doc.fill((0, 0), RED, refer="sideways")


# --- connectivity -----------------------------------------------------------


def _two_rooms_touching_at_a_corner():
    doc = _doc((16, 16), BLACK)
    doc.stack[0].pixels[0:3, 0:3] = WHITE
    doc.stack[0].pixels[3:6, 3:6] = WHITE
    doc.invalidate_all()
    return doc


def test_four_connected_stops_at_a_diagonal_join():
    doc = _two_rooms_touching_at_a_corner()
    doc.fill((0, 0), RED, thresh=0)
    assert _at(doc, 0, 0) == RED
    assert _at(doc, 4, 4) == WHITE


def test_eight_connected_crosses_a_diagonal_join():
    doc = _two_rooms_touching_at_a_corner()
    doc.fill((0, 0), RED, thresh=0, eight_connected=True)
    assert _at(doc, 0, 0) == RED
    assert _at(doc, 4, 4) == RED


def test_the_wand_reads_the_same_connectivity_the_fill_does():
    """One predicate for both, which is the rule ``thresh`` already follows."""
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[:, :] = BLACK
    pixels[0:3, 0:3] = WHITE
    pixels[3:6, 3:6] = WHITE
    four = inker.magic_wand(pixels, (0, 0), tolerance=0)
    eight = inker.magic_wand(pixels, (0, 0), tolerance=0, eight_connected=True)
    assert four.mask[4, 4] == 0
    assert eight.mask[4, 4] == 255


# --- stop at grid -----------------------------------------------------------


def test_a_fill_stops_at_the_grid_lines_when_asked():
    doc = _doc((16, 16), WHITE)
    doc.fill((0, 0), RED, thresh=0, stop_at_grid=8)
    assert _at(doc, 7, 7) == RED
    assert _at(doc, 8, 7) == WHITE
    assert _at(doc, 9, 9) == WHITE


def test_the_grid_confines_a_fill_seeded_in_a_middle_cell():
    doc = _doc((16, 16), WHITE)
    doc.fill((9, 9), RED, thresh=0, stop_at_grid=8)
    assert _at(doc, 9, 9) == RED
    assert _at(doc, 15, 15) == RED
    assert _at(doc, 7, 7) == WHITE


def test_the_grid_confines_a_non_contiguous_fill_too():
    doc = _doc((16, 16), WHITE)
    doc.fill((0, 0), RED, thresh=0, contiguous=False, stop_at_grid=8)
    assert _at(doc, 7, 7) == RED
    assert _at(doc, 9, 9) == WHITE


def test_a_grid_wider_than_the_canvas_changes_nothing():
    doc = _doc((16, 16), WHITE)
    doc.fill((0, 0), RED, thresh=0, stop_at_grid=64)
    assert _at(doc, 15, 15) == RED


def test_the_wand_tool_honours_eight_connectivity_too():
    """The option is on both tools' bars, so both have to read it."""
    doc = _two_rooms_touching_at_a_corner()
    doc.select_wand((0, 0), tolerance=0, eight_connected=True)
    assert doc.mask is not None and doc.mask.contains((4, 4))
