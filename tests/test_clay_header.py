"""Does Clay's viewport header fit, and does it hold what it claims to.

The one question a header can silently get wrong is whether it fits at the
window the app opens at. ``toolbar`` degrades rather than clips, so a bar that
does not fit does not *look* broken -- it looks like a bar whose every label is
a hover away, on the machine everybody uses. That is the failure this file
exists for, and it is measured rather than eyeballed.
"""

from __future__ import annotations

import pytest

from warlock.studio import clay_ops, clay_state, toolbar
from warlock.studio.panes import clay_header, clay_tools

# --- the tables ---------------------------------------------------------------


def test_every_element_mode_has_a_short_label():
    """The compact tier swaps the words for these, so a mode missing one would
    be a segment drawn with nothing in it."""
    for mode, _label, _key in clay_tools.MODE_BUTTONS:
        assert clay_header.MODE_SHORT.get(mode), mode
    assert set(clay_header.MODE_SHORT) == {
        mode for mode, _l, _k in clay_tools.MODE_BUTTONS
    }


def test_the_modes_the_header_offers_are_the_modes_the_ops_registry_knows():
    """A mode on the bar that no op declares is a mode with nothing in it."""
    assert {mode for mode, _l, _k in clay_tools.MODE_BUTTONS} == set(clay_ops.ALL_MODES)


def test_every_tool_has_a_glyph_or_falls_back_to_its_initial():
    for key, label, _shortcut in clay_state.TOOLS:
        drawn = clay_tools.TOOL_ICONS.get(key) or label[:1]
        assert drawn, key


def test_the_overlay_rows_name_real_switches():
    """``grid`` is a field and the rest live in ``state.overlays``; the two
    accessors are what let the popover be a loop over one table."""
    state = clay_state.ClayState()
    for key, label, tip in clay_header.OVERLAY_ROWS:
        assert label and tip
        before = clay_header.overlay_value(state, key)
        clay_header.set_overlay(state, key, not before)
        assert clay_header.overlay_value(state, key) is (not before)


def test_the_grid_keeps_its_own_field_rather_than_moving_into_the_dict():
    """One switch, one home. It is wired straight to ``ClayView.show_grid`` and
    a second copy in the dict is two places that can disagree."""
    state = clay_state.ClayState()
    clay_header.set_overlay(state, "grid", False)
    assert state.grid is False
    assert "grid" not in state.overlays


def test_every_axis_row_names_a_view_the_camera_has():
    from warlock.studio.viewer.camera import Camera

    for name, label, chord in clay_header.AXIS_ROWS:
        assert name in Camera.AXIS_VIEWS, name
        assert label and chord.startswith("Ctrl+")
    assert {name for name, _l, _c in clay_header.AXIS_ROWS} == set(Camera.AXIS_VIEWS), (
        "all six, not the three that had buttons -- the backs were reachable "
        "only by holding Shift, which nothing said"
    )


# The fit itself needs a live imgui context to measure a font in, and this
# file must not build one: two imgui contexts over the one GL context crash
# the process when they overlap, which is why every context in the suite is
# per-file and torn down with it. The three width tests live in
# ``tests/test_studio_smoke.py``, which already owns one.


@pytest.mark.parametrize(
    "popup", ["SNAP_POPUP", "PROPORTIONAL_POPUP", "OVERLAYS_POPUP", "VIEW_POPUP"]
)
def test_every_popup_has_its_own_name(popup):
    """Two popups sharing a name is one popup that opens when either is asked
    for, which imgui reports as neither working."""
    names = {
        getattr(clay_header, key)
        for key in ("SNAP_POPUP", "PROPORTIONAL_POPUP", "OVERLAYS_POPUP", "VIEW_POPUP")
    }
    assert len(names) == 4
    assert getattr(clay_header, popup) in names


def test_the_mode_and_tool_pills_are_fields_rather_than_items():
    """Items collapse into the overflow menu first, and a mode picker in a menu
    is a mode picker nobody can see the state of -- which is the one thing a
    mode picker is for."""
    state = clay_state.ClayState()
    keys = {item.key for item in clay_header._items(state)}
    assert "mode" not in keys and "tool" not in keys
    assert isinstance(clay_header._tool_field(state), toolbar.Field)
    assert clay_header._tool_field(state).priority == 0


def test_the_tool_pill_never_gets_narrower():
    """Four glyphs are already the smallest it can be, and ``Field`` says so by
    declaring one width twice rather than by a comment."""
    field = clay_header._tool_field(clay_state.ClayState())
    assert field.widths()[0] == field.widths()[1]
