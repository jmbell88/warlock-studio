"""The layer list's alternating row bands, asserted as a list rather than pixels.

``striped_rows`` exists separately from ``row_plan`` so these claims can be
made at all: banding that goes wrong is a mistake about a *list*, and a list is
the only thing a headless test can hold it to. Every test here asserts a set of
positions and never touches imgui.

The two implementations a reader reaches for first are both wrong, and each has
a test named after the way it fails: parity straight off ``enumerate(plan)``
puts two bands either side of a group header, and parity off ``entry.index``
bands the stack rather than the rows on screen.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np

from warlock.studio import inker, theme, tokens
from warlock.studio.panes import inker_timeline


def _doc(count=4):
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    doc = inker.Document.from_pixels(pixels)
    for index in range(count - 1):
        doc.add_layer(f"L{index + 1}")
    return doc


def _kinds(plan):
    return [entry.kind for entry in plan]


def _calls_of(func):
    """Every name called in ``func``, comments and docstrings excluded."""
    tree = ast.parse(inspect.getsource(func).lstrip())
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
        elif isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


def test_a_layer_list_with_no_groups_alternates_from_the_second_row():
    """The top row stays clean: it abuts the frame headers, and a band there
    merges into the header row rather than reading as the first layer."""
    plan = inker_timeline.row_plan(_doc(4))
    assert _kinds(plan) == ["track"] * 4
    assert inker_timeline.striped_rows(plan) == frozenset({1, 3})


def test_a_group_header_never_takes_the_stripe():
    doc = _doc(4)
    doc.group_layers([1, 2])
    plan = inker_timeline.row_plan(doc)
    headers = {i for i, entry in enumerate(plan) if entry.kind == "group"}
    assert headers, "the fixture must actually produce a header"
    assert not (headers & inker_timeline.striped_rows(plan))


def test_a_header_between_two_layers_does_not_put_two_bands_side_by_side():
    """The claim the header rule exists for.

    The plan is track/group/track/track. A header that *consumed* a parity slot
    leaves positions 0 and 2 unbanded and 3 banded -- but the two rows the user
    sees either side of the folder are then both unbanded, and the two below it
    run together. Counting only tracks is what keeps the layers alternating as
    layers, which is the whole point of the feature.
    """
    doc = _doc(4)
    doc.group_layers([1, 2])
    plan = inker_timeline.row_plan(doc)
    assert _kinds(plan) == ["track", "group", "track", "track", "track"]
    striped = inker_timeline.striped_rows(plan)
    # Ranks 0,1,2,3 fall on positions 0,2,3,4.
    assert striped == frozenset({2, 4})
    tracks = [i for i, entry in enumerate(plan) if entry.kind == "track"]
    banded = [i in striped for i in tracks]
    assert banded == [False, True, False, True], "layers must alternate as layers"


def test_stripes_follow_the_rows_drawn_not_the_stack_index():
    """A fold makes the drawn rows a sparse subset of the stack.

    Six layers with the middle three folded away draws stack indices 0, 4 and
    5 -- so as *rows* they are the first, second and third and the second is
    banded, while as *stack indices* only 5 is odd and the band lands on the
    third. The fixture is deliberately not the four-layer one used elsewhere
    in this file: there the two spellings agree by coincidence, and a test
    whose name is this claim has to be able to fail on it.
    """
    doc = _doc(6)
    node = doc.group_layers([1, 2, 3])
    plan = inker_timeline.row_plan(doc, collapsed={node.uid})
    assert _kinds(plan) == ["track", "group", "track", "track"]
    assert [entry.index for entry in plan if entry.kind == "track"] == [0, 4, 5]

    striped = inker_timeline.striped_rows(plan)
    assert striped == frozenset({2})
    stack_parity = frozenset(
        position
        for position, entry in enumerate(plan)
        if entry.kind == "track" and entry.index % 2
    )
    assert stack_parity == frozenset({3})
    assert striped != stack_parity, "the fixture must separate the two spellings"


def test_a_filtered_list_stripes_the_rows_it_still_draws():
    doc = _doc(4)
    plan = inker_timeline.row_plan(doc, indices=[0, 2, 3])
    assert _kinds(plan) == ["track"] * 3
    assert inker_timeline.striped_rows(plan) == frozenset({1})


def test_the_timeline_stripe_and_the_table_zebra_are_one_number():
    """One zebra, one alpha. ``theme`` sets imgui's own ``table_row_bg_alt``
    and the timeline paints its bands with its own draw list, so the two
    cannot be made to agree by sharing a call -- only by sharing the token.
    Pinned at the source, the way ``tests/test_layout.py`` pins ``main.py``.
    """
    assert tokens.ROW_STRIPE_ALPHA == 0.4
    source = inspect.getsource(theme)
    at = source.find("table_row_bg_alt")
    assert at != -1, "the table zebra moved; this pin needs rewriting"
    call = source[at : at + 160]
    assert "tokens.ROW_STRIPE_ALPHA" in call
    assert "0.4" not in call, "a literal here is the drift this pins"

    assert "tokens.ROW_STRIPE_ALPHA" in inspect.getsource(inker_timeline._row_stripe)


def test_the_band_is_packed_the_way_every_other_tint_in_the_grid_is():
    """``_u32`` and not ``get_color_u32``: the grid paints inside
    ``begin_disabled`` blocks, where ``get_color_u32`` multiplies by
    ``style.Alpha``. Row rhythm is structure rather than state and must not
    fade out with the row while a save has the panel disabled."""
    assert "_u32(theme.ELEV_1" in inspect.getsource(inker_timeline._row_stripe)
    # Over the *called names* rather than the text: the docstring names the
    # rejected spelling on purpose, and a substring search would read the
    # explanation as the mistake.
    called = _calls_of(inker_timeline._row_stripe)
    assert "_u32" in called
    assert "get_color_u32" not in called


def test_the_band_submits_no_item_so_it_cannot_steal_the_row_s_clicks():
    """The band is geometry, not a widget.

    ``list_row`` needs ``set_next_item_allow_overlap`` because its surface is a
    real ``invisible_button``; this design avoids that whole class of problem
    by adding a draw command and no item, so the name's click, ``_reorder``'s
    drag source and ``_row_menu``'s context popup all still resolve to the
    items ``_track_row`` submits. If this ever grows a button the flag becomes
    load-bearing, so the assumption is written down where it would break.
    """
    called = _calls_of(inker_timeline._row_stripe)
    for id_bearing in ("button", "invisible_button", "selectable", "dummy"):
        assert id_bearing not in called
    assert "add_rect_filled" in called


def test_the_first_cell_s_x_is_measured_before_the_first_row_draws():
    """``geom["x0"]`` was seeded ``0.0`` and filled in by ``_cell``, so it was
    the window origin for the whole of the first track row of *every* draw --
    and ``_range_overlay``/``_track_overlay`` read it, drawing their accent box
    against the left edge of the window. The header measures it instead."""
    headers = inspect.getsource(inker_timeline._frame_headers)
    assert "x0 = imgui.get_cursor_screen_pos().x" in headers
    assert headers.rstrip().endswith("return x0")

    grid = inspect.getsource(inker_timeline._grid)
    assert "header_x0 = _frame_headers(" in grid
    assert '"x0": header_x0' in grid
    assert '"x0": 0.0' not in grid
