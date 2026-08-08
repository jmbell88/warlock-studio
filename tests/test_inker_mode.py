"""Paint mode's state, without a window.

The tab arithmetic is where a multi-document editor actually goes wrong -- a
dirty flag that latches, a close that jumps you to the wrong tab, an "already
open" check that forks a second tab onto one file and lets the two race on
save. None of that needs GL, so none of it is left to the smoke test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from warlock.studio import inker, inker_state
from warlock.studio.inker_state import InkerDoc, InkerState, PaintView


def _tab(name="a.png", path=None, size=(16, 16)):
    doc = inker.Document.blank(*size)
    return InkerDoc(doc=doc, title=name, path=path, saved_head=doc.history.head)


def _state(*tabs):
    state = InkerState()
    for tab in tabs:
        state.add(tab)
    return state


# --- dirtiness --------------------------------------------------------------


def test_a_new_document_is_not_dirty_and_an_edit_makes_it_so():
    tab = _tab()
    assert not tab.dirty
    tab.doc.add_layer()
    assert tab.dirty


def test_dirty_is_a_comparison_so_undoing_back_to_the_saved_state_is_clean():
    """A latching flag would leave the tab claiming unsaved changes it has
    just undone, and asking to confirm a close that would lose nothing."""
    tab = _tab()
    tab.doc.flip("horizontal")
    assert tab.dirty
    tab.doc.undo()
    assert not tab.dirty


def test_marking_saved_records_the_position_the_encode_started_at():
    """An edit made while the file was being written is genuinely not in that
    file, so it must stay dirty -- which a boolean flag could not express."""
    tab = _tab()
    tab.doc.add_layer()
    head_at_submit = tab.doc.history.head
    tab.doc.add_layer()  # drawn while the save was in flight
    tab.mark_saved(head_at_submit)
    assert tab.dirty
    assert not tab.saving


def test_any_dirty_reports_across_every_tab():
    a, b = _tab("a"), _tab("b")
    state = _state(a, b)
    assert not state.any_dirty
    b.doc.add_layer()
    assert state.any_dirty


# --- tabs -------------------------------------------------------------------


def test_a_tab_label_carries_a_stable_id_so_a_rename_does_not_move_it():
    tab = _tab("untitled")
    before = tab.label.split("###")[1]
    tab.title = "barrel.ora"
    assert tab.label.split("###")[1] == before
    assert tab.label.startswith("barrel.ora###")


def test_two_documents_never_share_an_id():
    assert _tab().uid != _tab().uid


def test_adding_a_tab_makes_it_active():
    a, b = _tab("a"), _tab("b")
    state = _state(a, b)
    assert state.active is b


def test_closing_the_active_tab_lands_on_its_neighbour_not_the_first():
    a, b, c = _tab("a"), _tab("b"), _tab("c")
    state = _state(a, b, c)
    state.activate(b.uid)
    state.close(b.uid)
    assert state.active is c


def test_closing_the_last_tab_leaves_nothing_active():
    a = _tab("a")
    state = _state(a)
    assert state.close(a.uid)
    assert state.active is None
    assert not state.close("nope")


def test_cycling_wraps_and_does_nothing_with_one_document():
    a, b = _tab("a"), _tab("b")
    state = _state(a, b)
    state.activate(a.uid)
    state.cycle(1)
    assert state.active is b
    state.cycle(1)
    assert state.active is a
    state.cycle(-1)
    assert state.active is b

    solo = _state(_tab())
    only = solo.active
    solo.cycle(1)
    assert solo.active is only


def test_a_file_already_open_is_found_rather_than_opened_twice():
    """Two tabs over one path would race on save and silently lose one set of
    edits."""
    path = Path("/tmp/x.ora")
    tab = _tab(path=path)
    state = _state(tab, _tab(path=Path("/tmp/y.ora")))
    assert state.find_path(path) is tab
    assert state.find_path(Path("/tmp/z.ora")) is None


def test_a_linked_document_is_found_by_its_job():
    tab = _tab()
    tab.job_id = "abc123"
    state = _state(tab)
    assert state.find_job("abc123") is tab
    assert state.find_job("nope") is None
    assert tab.linked


def test_switching_tabs_abandons_a_half_finished_drag():
    a, b = _tab("a"), _tab("b")
    state = _state(a, b)
    state.drag_kind = "marquee"
    state.drag_anchor = (1.0, 1.0)
    state.activate(a.uid)
    assert state.drag_kind == "" and state.drag_anchor is None


# --- the view ---------------------------------------------------------------


def test_fitting_shows_the_whole_document_centred():
    view = PaintView()
    inker_state.fit(view, (100, 50), (200.0, 200.0))
    assert view.zoom == pytest.approx(2.0)
    assert view.pan == pytest.approx((0.0, 50.0))
    assert view.fitted


def test_a_round_trip_through_the_view_is_the_identity():
    view = PaintView(zoom=2.5, pan=(13.0, -7.0))
    screen = inker_state.to_screen(view, (10.0, 20.0), 4.0, 6.0)
    assert inker_state.to_image(view, (10.0, 20.0), *screen) == pytest.approx((4.0, 6.0))


def test_image_coordinates_stay_fractional_for_the_brush():
    """Rounding here would quantise every stroke to the zoom it was drawn at."""
    view = PaintView(zoom=4.0)
    x, y = inker_state.to_image(view, (0.0, 0.0), 3.0, 5.0)
    assert (x, y) == pytest.approx((0.75, 1.25))


def test_zooming_keeps_the_pixel_under_the_cursor_under_the_cursor():
    view = PaintView(zoom=1.0, pan=(0.0, 0.0))
    origin, mouse = (0.0, 0.0), (120.0, 80.0)
    before = inker_state.to_image(view, origin, *mouse)
    inker_state.zoom_about(view, origin, mouse, 3.0)
    assert inker_state.to_image(view, origin, *mouse) == pytest.approx(before)
    assert view.zoom > 1.0


def test_zoom_is_clamped_at_both_ends():
    view = PaintView(zoom=1.0)
    inker_state.zoom_about(view, (0.0, 0.0), (0.0, 0.0), 200.0)
    assert view.zoom == pytest.approx(inker_state.MAX_ZOOM)
    inker_state.zoom_about(view, (0.0, 0.0), (0.0, 0.0), -400.0)
    assert view.zoom == pytest.approx(inker_state.MIN_ZOOM)


def test_centring_sets_an_exact_zoom():
    view = PaintView()
    inker_state.centre(view, (100, 100), (400.0, 400.0), 1.0)
    assert view.zoom == pytest.approx(1.0)
    assert view.pan == pytest.approx((150.0, 150.0))


def test_the_view_belongs_to_the_document_not_to_the_app():
    """Switching tabs must not scroll you back to the origin of the other."""
    a, b = _tab("a"), _tab("b")
    a.view.pan = (100.0, 100.0)
    state = _state(a, b)
    state.activate(a.uid)
    assert state.active.view.pan == (100.0, 100.0)
    state.activate(b.uid)
    assert state.active.view.pan == (0.0, 0.0)


# --- tool settings ----------------------------------------------------------


def test_brush_stepping_accelerates_and_stays_in_range():
    assert inker_state.step_size(8, 1) > 8
    assert inker_state.step_size(200, 1) - 200 > inker_state.step_size(8, 1) - 8
    assert inker_state.step_size(inker.MIN_BRUSH, -1) == inker.MIN_BRUSH
    assert inker_state.step_size(inker.MAX_BRUSH, 1) == inker.MAX_BRUSH


def test_swapping_colours_is_its_own_inverse():
    state = InkerState()
    fg, bg = state.fg, state.bg
    state.swap_colours()
    assert (state.fg, state.bg) == (bg, fg)
    state.swap_colours()
    assert (state.fg, state.bg) == (fg, bg)


def test_a_swatch_is_added_once_and_the_row_is_bounded():
    state = InkerState()
    state.swatches = []
    state.add_swatch((1, 2, 3, 255))
    state.add_swatch((1, 2, 3, 255))
    assert state.swatches == [(1, 2, 3, 255)]
    for i in range(inker_state.MAX_SWATCHES + 5):
        state.add_swatch((i, 0, 0, 255))
    assert len(state.swatches) == inker_state.MAX_SWATCHES


def test_tool_settings_are_shared_across_documents():
    """Aseprite's convention, and the one users expect: switching tabs must not
    silently change your brush."""
    state = _state(_tab("a"), _tab("b"))
    state.brush_size = 40
    state.activate(state.docs[0].uid)
    assert state.brush_size == 40


def test_a_tool_remembers_its_own_options():
    """The whole of Ink2: sizing the eraser must not resize the brush.

    Written against ``state.brush_size`` rather than the dictionary because
    that attribute is what nine call sites say, and the property is the feature
    -- a test that went through ``options_for`` would pass with the property
    removed and every pane back to one shared size.
    """
    state = inker_state.InkerState()
    state.tool = "brush"
    state.brush_size = 12
    state.tool = "eraser"
    state.brush_size = 60
    state.hardness = 0.1

    state.tool = "brush"
    assert state.brush_size == 12
    assert state.hardness == inker_state.TOOL_OPTION_DEFAULTS["hardness"]
    state.tool = "eraser"
    assert (state.brush_size, state.hardness) == (60, 0.1)


def test_an_untouched_tool_starts_at_the_declared_defaults():
    state = inker_state.InkerState()
    for tool, _label, _key in inker_state.TOOLS:
        state.tool = tool
        for name, default in inker_state.TOOL_OPTION_DEFAULTS.items():
            assert getattr(state, name) == default, (tool, name)


def test_resetting_a_tool_forgets_only_that_tool():
    state = inker_state.InkerState()
    state.tool = "brush"
    state.brush_size = 40
    state.tool = "eraser"
    state.brush_size = 80

    state.reset_tool_options("eraser")
    assert state.brush_size == inker_state.TOOL_OPTION_DEFAULTS["brush_size"]
    state.tool = "brush"
    assert state.brush_size == 40


def test_the_canvas_settings_stay_app_level():
    """A grid that switched off because you picked the eraser would be a bug.

    The split is the decision in Ink2, so it is asserted rather than left to
    the comment: these names must *not* be per-tool options.
    """
    shared = {"symmetry", "grid", "grid_size", "fg", "bg", "feather_radius", "swatches"}
    assert shared.isdisjoint(inker_state.TOOL_OPTION_DEFAULTS)
    state = inker_state.InkerState()
    state.grid_size = 32
    state.tool = "eraser"
    assert state.grid_size == 32


def test_every_tool_has_a_shortcut_and_a_label():
    keys = [key for key, _, _ in inker_state.TOOLS]
    assert len(set(keys)) == len(keys)
    assert all(label and shortcut for _, label, shortcut in inker_state.TOOLS)


def test_every_painting_tool_maps_to_a_brush_mode_the_engine_knows():
    for tool in inker_state.PAINT_TOOLS:
        assert inker_state.BRUSH_MODES[tool] in inker.MODES


def test_the_tool_groups_do_not_overlap():
    groups = (inker_state.PAINT_TOOLS, inker_state.SHAPE_TOOLS, inker_state.SELECT_TOOLS)
    for i, first in enumerate(groups):
        for second in groups[i + 1 :]:
            assert not first & second


# --- recent files -----------------------------------------------------------


def test_recent_files_are_most_recent_first_deduplicated_and_bounded():
    state = InkerState()
    for i in range(inker_state.MAX_RECENT + 5):
        state.remember(Path(f"/tmp/f{i}.ora"))
    assert len(state.recent) == inker_state.MAX_RECENT
    state.remember(Path("/tmp/f0.ora"))
    assert state.recent[0] == str(Path("/tmp/f0.ora"))
    assert state.recent.count(str(Path("/tmp/f0.ora"))) == 1


def test_an_unsaved_document_contributes_nothing_to_the_recent_list():
    state = InkerState()
    state.remember(None)
    assert state.recent == []


def test_a_path_that_did_not_open_can_be_forgotten():
    state = InkerState()
    state.remember(Path("/tmp/gone.ora"))
    state.forget(str(Path("/tmp/gone.ora")))
    assert state.recent == []


# --- keys -------------------------------------------------------------------


def test_the_tool_shortcuts_and_the_tool_list_agree():
    from warlock.studio import inker_mode

    tools = {key for key, _, _ in inker_state.TOOLS}
    assert set(inker_mode.TOOL_KEYS.values()) == tools
    assert len(set(inker_mode.TOOL_KEYS)) == len(inker_mode.TOOL_KEYS)
