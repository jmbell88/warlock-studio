"""Twelve toolbox slots, and the geometry of the strip that opens off one.

The reason this can be a unit test at all is the rule the flyout is built on:
**one function places the cells and hit-tests them**. Every hand-rolled flyout
has the same defect -- a picture drawn by one piece of arithmetic and a hit box
computed by another -- and the only fix that survives a change of cell size is
not having two.

The other property here is that key cycling is *additive*: ``TOOLS`` and
``TOOL_KEYS`` are untouched, so every letter still lands where it always did on
the first press, and a second press is a new fact rather than a moved one.
"""

from __future__ import annotations

import pytest

from warlock.studio import inker_state


def test_every_tool_is_in_exactly_one_group():
    tools = [tool for _key, _label, tools in inker_state.TOOL_GROUPS for tool in tools]
    assert sorted(tools) == sorted(tool for tool, _l, _k in inker_state.TOOLS)
    assert len(tools) == len(set(tools))


def test_the_toolbox_is_two_columns_of_six():
    assert len(inker_state.TOOL_GROUPS) == 12


def test_a_group_is_named_for_a_tool_in_it():
    """The rail's id and the group's first member share a name, which is what
    makes a group button's imgui id stable when a member is added."""

    for key, _label, members in inker_state.TOOL_GROUPS:
        assert key in members or key in {"effects", "path", "util"}


def test_the_first_press_of_a_letter_is_the_binding_it_always_had():
    for tool, _label, _key in inker_state.TOOLS:
        # From some tool outside the group -- the ordinary case.
        outside = "brush" if inker_state.GROUP_OF[tool] != "brush" else "wand"
        assert inker_state.cycle_in_group(outside, tool) == tool


def test_a_second_press_moves_along_the_group():
    assert inker_state.cycle_in_group("rect", "rect") == "ellipse"
    assert inker_state.cycle_in_group("ellipse", "rect") == "rect"
    # Three members wrap.
    assert inker_state.cycle_in_group("blur", "blur") == "smudge"
    assert inker_state.cycle_in_group("smudge", "blur") == "shade"
    assert inker_state.cycle_in_group("shade", "blur") == "blur"


def test_a_group_of_one_never_cycles():
    assert inker_state.cycle_in_group("wand", "wand") == "wand"


RECT = (100.0, 200.0, 30.0, 30.0)
CELL = (30.0, 30.0)


def test_the_strip_hangs_off_the_right_of_the_button():
    cells = inker_state.flyout_cells(RECT, CELL, 3)
    assert [cell[0] for cell in cells] == [130.0, 160.0, 190.0]
    # Vertically centred on the button; same height here, so flush with it.
    assert all(cell[1] == 200.0 for cell in cells)


def test_a_taller_button_centres_its_strip():
    cells = inker_state.flyout_cells((0.0, 0.0, 40.0, 50.0), (20.0, 30.0), 1)
    assert cells == [(40.0, 10.0, 20.0, 30.0)]


@pytest.mark.parametrize(
    ("mouse", "expected"),
    [
        ((131.0, 210.0), 0),
        ((165.0, 201.0), 1),
        ((195.0, 229.0), 2),
        # The button itself is not a cell of its own strip.
        ((110.0, 210.0), None),
        # Past the end, and above it.
        ((260.0, 210.0), None),
        ((131.0, 199.0), None),
    ],
)
def test_the_hit_test_reads_the_cells_it_drew(mouse, expected):
    assert inker_state.flyout_hit(RECT, CELL, 3, mouse) == expected


def test_the_hit_test_and_the_picture_cannot_disagree():
    """Every cell's own centre hits that cell, at any cell size."""

    for cell in ((30.0, 30.0), (17.0, 44.0), (64.0, 21.0)):
        cells = inker_state.flyout_cells(RECT, cell, 4)
        for index, (x, y, w, h) in enumerate(cells):
            centre = (x + w / 2, y + h / 2)
            assert inker_state.flyout_hit(RECT, cell, 4, centre) == index
