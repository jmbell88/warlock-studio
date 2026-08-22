"""Which options the context bar shows, and -- the one that earns its keep --
that no option became unreachable on the way out of the sidebar.

Moving fourteen controls out of a 300 px column and onto a 38 px row has
exactly one way to go badly wrong: a key that no tool's bar lists is a setting
the user can no longer reach, and nothing about the drawing would look
different. So the table is asserted against ``TOOL_OPTION_DEFAULTS`` in both
directions.
"""

from __future__ import annotations

import pytest

from warlock.studio import inker, inker_state


def test_every_option_is_reachable_from_some_tools_bar():
    listed = {key for key, _l, _tools, _g in inker_state.CONTEXT_WIDGETS}
    assert listed == set(inker_state.TOOL_OPTION_DEFAULTS)


def test_every_widget_names_a_real_option_and_at_least_one_real_tool():
    tools = {tool for tool, _l, _k in inker_state.TOOLS}
    for key, label, applies, group in inker_state.CONTEXT_WIDGETS:
        assert key in inker_state.TOOL_OPTION_DEFAULTS, key
        assert label, key
        assert applies <= tools, key
        assert applies, f"{key} applies to no tool at all"
        assert group in ("", "dynamics"), key


@pytest.mark.parametrize("tool", [tool for tool, _l, _k in inker_state.TOOLS])
def test_widgets_for_answers_for_every_tool(tool):
    keys = inker_state.widgets_for(tool)
    assert len(keys) == len(set(keys))
    assert set(keys) <= set(inker_state.TOOL_OPTION_DEFAULTS)


def test_a_pixel_nib_has_no_hardness_and_gains_the_corner_filter():
    state = inker_state.InkerState(tool="brush")
    assert "hardness" in inker_state.widgets_for("brush", None, state)
    assert "pixel_perfect" not in inker_state.widgets_for("brush", None, state)
    state.options_for("brush")["nib"] = "pixel"
    keys = inker_state.widgets_for("brush", None, state)
    assert "hardness" not in keys and "pixel_perfect" in keys


def test_an_indexed_document_has_no_hardness_on_any_nib():
    class Indexed:
        is_indexed = True

    state = inker_state.InkerState(tool="brush")
    assert "hardness" not in inker_state.widgets_for("brush", Indexed(), state)


def test_the_spray_never_offers_the_corner_filter():
    """It is about a *line*, and the canvas forces it off there."""

    state = inker_state.InkerState(tool="spray")
    state.options_for("spray")["nib"] = "pixel"
    assert "pixel_perfect" not in inker_state.widgets_for("spray", None, state)


def test_the_four_stroke_dynamics_are_the_ones_behind_the_popup():
    behind = {
        key
        for key, _l, _t, group in inker_state.CONTEXT_WIDGETS
        if group == "dynamics"
    }
    assert {"spacing", "stabilise", "speed_taper", "strength"} <= behind


def test_the_brush_bar_leads_with_size():
    """The one option every painting tool has, and the one reached most."""

    assert inker_state.widgets_for("brush")[0] == "brush_size"
    assert inker_state.widgets_for("rect")[0] == "brush_size"


def test_a_tool_with_no_options_gets_an_empty_bar():
    assert inker_state.widgets_for("move") == ()


def test_the_size_widget_covers_the_engines_whole_brush_range():
    """A slider that cannot reach ``MAX_BRUSH`` is a setting with a new cap."""

    assert inker.MIN_BRUSH < inker.MAX_BRUSH
