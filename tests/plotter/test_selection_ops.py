"""Invert, grow and shrink: the two genuinely new engine functions of W3.4.

Pure, in ``plotter/tools.py``, with no new outward import -- the same rule the
rest of that module follows, and what lets these be checked as arrays rather
than through a frame.

The property worth stating: **shrink is grow of the outside**. A hand-written
erosion beside a hand-written dilation is two definitions of "next to", and the
day one of them gains a diagonal the pair stop being inverses.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.plotter import tools


def test_the_inverse_of_a_rectangle_is_not_a_rectangle():
    mask, rect = tools.invert_selection(None, (1, 1, 2, 2), 4, 4)
    assert mask.shape == (4, 4)
    assert not mask[1:3, 1:3].any()
    assert mask[0].all() and mask[3].all()
    # The bounding rect of what is left is the whole map, which is honest: the
    # selection now touches every edge.
    assert rect == (0, 0, 3, 3)


def test_inverting_nothing_selects_everything():
    mask, rect = tools.invert_selection(None, None, 3, 2)
    assert mask.all()
    assert rect == (0, 0, 2, 1)


def test_inverting_a_masked_selection_reads_the_mask():
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    inverted, _rect = tools.invert_selection(mask, (0, 0, 2, 2), 3, 3)
    assert not inverted[0, 0]
    assert inverted.sum() == 8


def test_grow_is_four_connected():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    grown = tools.grow_selection(mask)
    assert grown[1, 2] and grown[3, 2] and grown[2, 1] and grown[2, 3]
    assert not grown[1, 1], "a diagonal is not a neighbour on a tile map"
    assert grown.sum() == 5


def test_grow_by_two_is_grow_twice():
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True
    assert np.array_equal(
        tools.grow_selection(mask, 2), tools.grow_selection(tools.grow_selection(mask))
    )


def test_shrink_is_grow_of_the_outside():
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    shrunk = tools.shrink_selection(mask)
    assert shrunk.sum() == 4
    assert shrunk[2:4, 2:4].all()
    assert np.array_equal(shrunk, ~tools.grow_selection(~mask))


def test_shrinking_past_nothing_leaves_nothing():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    assert not tools.shrink_selection(mask, 2).any()


def test_growing_stops_at_the_edge():
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    grown = tools.grow_selection(mask, 5)
    assert grown.shape == (3, 3) and grown.all()


def test_zero_steps_change_nothing():
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    assert np.array_equal(tools.grow_selection(mask, 0), mask)
    assert np.array_equal(tools.shrink_selection(mask, 0), mask)
