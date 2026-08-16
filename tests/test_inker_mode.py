"""Paint mode's state, without a window.

The tab arithmetic is where a multi-document editor actually goes wrong -- a
dirty flag that latches, a close that jumps you to the wrong tab, an "already
open" check that forks a second tab onto one file and lets the two race on
save. None of that needs GL, so none of it is left to the smoke test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from warlock.studio import inker, inker_mode, inker_state
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
    """The module defaults, which Plotter and Packwright both take."""
    view = PaintView(zoom=1.0)
    inker_state.zoom_about(view, (0.0, 0.0), (0.0, 0.0), 200.0)
    assert view.zoom == pytest.approx(inker_state.MAX_ZOOM)
    inker_state.zoom_about(view, (0.0, 0.0), (0.0, 0.0), -400.0)
    assert view.zoom == pytest.approx(inker_state.MIN_ZOOM)


def _inker_bounds() -> dict[str, float]:
    return {"lo": inker_state.INKER_MIN_ZOOM, "hi": inker_state.INKER_MAX_ZOOM}


def test_the_inker_bounds_are_narrower_than_the_shared_ones():
    """A tile map wants 5%; a drawing at 5% is a postage stamp."""
    assert inker_state.MIN_ZOOM < inker_state.INKER_MIN_ZOOM
    assert inker_state.INKER_MAX_ZOOM < inker_state.MAX_ZOOM


def test_zoom_is_clamped_to_the_inker_bounds_when_they_are_passed():
    view = PaintView(zoom=1.0)
    inker_state.zoom_step(view, (0.0, 0.0), (0.0, 0.0), 500.0, **_inker_bounds())
    assert view.zoom == pytest.approx(inker_state.INKER_MAX_ZOOM)
    inker_state.zoom_step(view, (0.0, 0.0), (0.0, 0.0), -500.0, **_inker_bounds())
    assert view.zoom == pytest.approx(inker_state.INKER_MIN_ZOOM)


def test_a_wheel_notch_is_five_percent_and_snapped_to_the_grid():
    view = PaintView(zoom=0.25)
    for expected in (0.30, 0.35, 0.40):
        inker_state.zoom_step(view, (0.0, 0.0), (0.0, 0.0), 1.0, **_inker_bounds())
        assert view.zoom == pytest.approx(expected)


def test_a_notch_joins_the_grid_rather_than_carrying_a_fitted_fraction():
    """A zoom arrived at by fitting is arbitrary; one notch makes it round."""
    view = PaintView(zoom=0.834)
    inker_state.zoom_step(view, (0.0, 0.0), (0.0, 0.0), 1.0, **_inker_bounds())
    assert view.zoom == pytest.approx(0.90)


def test_stepping_keeps_the_pixel_under_the_cursor_under_the_cursor():
    view = PaintView(zoom=1.0, pan=(0.0, 0.0))
    origin, mouse = (0.0, 0.0), (120.0, 80.0)
    before = inker_state.to_image(view, origin, *mouse)
    inker_state.zoom_step(view, origin, mouse, 3.0, **_inker_bounds())
    assert inker_state.to_image(view, origin, *mouse) == pytest.approx(before)
    assert view.zoom == pytest.approx(1.15)


def test_a_step_that_changes_nothing_leaves_the_pan_alone():
    view = PaintView(zoom=inker_state.INKER_MAX_ZOOM, pan=(13.0, -7.0))
    inker_state.zoom_step(view, (0.0, 0.0), (50.0, 50.0), 4.0, **_inker_bounds())
    assert view.pan == pytest.approx((13.0, -7.0))


def test_fitting_under_a_floor_centres_at_the_floor_and_overflows():
    """The stated cost of having a floor: a huge page does not shrink to meet it."""
    view = PaintView()
    inker_state.fit(view, (8000, 8000), (400.0, 400.0), **_inker_bounds())
    assert view.zoom == pytest.approx(inker_state.INKER_MIN_ZOOM)


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
#
# The list itself moved to ``studio/recents.py`` and ``tests/test_recents.py``
# owns its rules; what these check is that Inker still reaches it, through a
# settings object with the two methods that module uses and nothing else.


class _RecentCtx:
    def __init__(self) -> None:
        self.data: dict = {}

    @property
    def settings(self):
        return self

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value) -> None:
        self.data[key] = value



def test_recent_files_are_most_recent_first_deduplicated_and_bounded():
    from warlock.studio import recents

    ctx = _RecentCtx()
    for i in range(recents.MAX_RECENT + 5):
        inker_mode.remember_path(ctx, Path(f"/tmp/f{i}.ora"))
    assert len(inker_mode.recent_paths(ctx)) == recents.MAX_RECENT
    inker_mode.remember_path(ctx, Path("/tmp/f0.ora"))
    found = inker_mode.recent_paths(ctx)
    assert found[0] == str(Path("/tmp/f0.ora"))
    assert found.count(str(Path("/tmp/f0.ora"))) == 1


def test_an_unsaved_document_contributes_nothing_to_the_recent_list():
    ctx = _RecentCtx()
    inker_mode.remember_path(ctx, None)
    assert inker_mode.recent_paths(ctx) == []


def test_a_path_that_did_not_open_can_be_forgotten():
    ctx = _RecentCtx()
    inker_mode.remember_path(ctx, Path("/tmp/gone.ora"))
    inker_mode.forget_path(ctx, str(Path("/tmp/gone.ora")))
    assert inker_mode.recent_paths(ctx) == []


# --- keys -------------------------------------------------------------------


def test_the_tool_shortcuts_and_the_tool_list_agree():
    from warlock.studio import inker_mode

    tools = {key for key, _, _ in inker_state.TOOLS}
    assert set(inker_mode.TOOL_KEYS.values()) == tools
    assert len(set(inker_mode.TOOL_KEYS)) == len(inker_mode.TOOL_KEYS)


# --- palette files (Ink8) ----------------------------------------------------


class _PaletteCtx:
    """Enough of Ctx for the two palette entry points, running inline."""

    def __init__(self) -> None:
        from warlock.studio.state import AppState

        self.state = AppState()
        self.settings = _MemorySettings()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.result = None

    def submit(self, key, run, *args):
        self.submitted.append(key)
        self.result = run(*args)
        return True

    def toast(self, message, level="info") -> None:
        self.toasts.append((message, level))


class _MemorySettings:
    def __init__(self) -> None:
        self.store: dict = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value) -> None:
        self.store[key] = value


def test_importing_a_palette_never_opens_a_picker_on_the_frame_thread(monkeypatch):
    """The rule every dialog in this app follows: a native picker is modal to
    the OS and blocks until it is dismissed."""
    from warlock.studio import dialogs, inker_mode

    opened: list[str] = []

    def fake_open(title, filters=None):
        opened.append(title)
        return None

    monkeypatch.setattr(dialogs, "open_file", fake_open)
    ctx = _PaletteCtx()
    inker_mode.import_palette(ctx)
    assert ctx.submitted == ["inker-palette"]
    assert opened, "the picker ran, on the task thread the submit stands in for"


def test_an_imported_palette_adds_to_the_row_rather_than_replacing_it(monkeypatch, tmp_path):
    """A user who wanted the old ones gone can right-click them away; an
    import that wiped a session's palette has no way back."""
    from warlock.studio import dialogs, inker_mode

    path = tmp_path / "p.gpl"
    path.write_text("GIMP Palette\n10 20 30\n", encoding="utf-8")
    monkeypatch.setattr(dialogs, "open_file", lambda *a, **k: path)

    ctx = _PaletteCtx()
    state = inker_mode.ensure(ctx)
    state.swatches = [(1, 1, 1, 255)]
    inker_mode.import_palette(ctx)
    inker_mode.on_task_done(
        ctx, type("Done", (), {"key": "inker-palette", "result": ctx.result})()
    )
    assert state.swatches == [(1, 1, 1, 255), (10, 20, 30, 255)]


def test_a_cancelled_palette_picker_changes_nothing(monkeypatch):
    from warlock.studio import dialogs, inker_mode

    monkeypatch.setattr(dialogs, "open_file", lambda *a, **k: None)
    ctx = _PaletteCtx()
    state = inker_mode.ensure(ctx)
    before = list(state.swatches)
    inker_mode.import_palette(ctx)
    inker_mode.on_task_done(ctx, type("Done", (), {"key": "inker-palette", "result": None})())
    assert state.swatches == before
    assert not ctx.toasts


def test_exporting_builds_the_bytes_before_the_picker(monkeypatch, tmp_path):
    """``save_as``'s rule: serialising after an unbounded modal would write
    whatever the user changed while it was open."""
    from warlock.studio import dialogs, inker_mode
    from warlock.studio.inker import gpl

    out = tmp_path / "out.gpl"

    def fake_save(title, default, filters=None):
        # The state changes *while the picker is up*; the file must not.
        state.swatches = [(9, 9, 9, 255)]
        return out

    ctx = _PaletteCtx()
    state = inker_mode.ensure(ctx)
    state.swatches = [(1, 2, 3, 255)]
    monkeypatch.setattr(dialogs, "save_file", fake_save)
    inker_mode.export_palette(ctx)
    assert gpl.parse(out.read_text(encoding="utf-8")) == [(1, 2, 3, 255)]


# --- crash-safe autosave (Ink13) ---------------------------------------------


class _AutosaveCtx(_PaletteCtx):
    """A ctx with a config whose autosave directory is a tmp_path."""

    def __init__(self, root) -> None:
        super().__init__()
        self.svc = type("Svc", (), {"config": type("Cfg", (), {"autosave_dir": root})()})()
        self.confirms = _Confirms()

    def submit(self, key, run, *args):
        self.submitted.append(key)
        self.result = run(*args)
        return True


class _Confirms:
    def __init__(self) -> None:
        self.pending = None

    def ask(self, confirm) -> None:
        self.pending = confirm

    def dismiss(self) -> None:
        self.pending = None


def _dirty_tab(state, title="a"):
    from warlock.studio import inker

    doc = inker.Document.blank(8, 8)
    tab = InkerDoc(doc=doc, title=title, saved_head=doc.history.head)
    state.add(tab)
    doc.add_layer()  # one edit, so the tab is dirty
    return tab


def _sidecar(root, payload, title, kind="inker"):
    """Write the completion gate beside a payload, as the journal would."""
    import json as _json

    from warlock.studio import journal

    journal.meta_path(payload).write_text(
        _json.dumps(
            {
                "version": journal.VERSION,
                "kind": kind,
                "title": title,
                "uid": "pd9",
                "at": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return payload


def _leave_behind(root, title, kind="inker"):
    """A crash copy from a previous session: payload *and* sidecar."""
    payload = root / f"{title}-pd9.ora"
    payload.write_bytes(b"not really an ora")
    return _sidecar(root, payload, title, kind=kind)


def test_a_clean_document_is_never_journalled(tmp_path):
    from warlock.studio import inker, inker_mode, journal

    ctx = _AutosaveCtx(tmp_path)
    state = inker_mode.ensure(ctx)
    doc = inker.Document.blank(8, 8)
    state.add(InkerDoc(doc=doc, title="clean", saved_head=doc.history.head))
    journal.pump(ctx, now=10_000.0)
    assert ctx.submitted == []
    assert not list(tmp_path.glob("*.ora"))


def test_a_dirty_document_is_journalled_once_the_interval_has_passed(tmp_path):
    from warlock.studio import inker_mode, journal

    ctx = _AutosaveCtx(tmp_path)
    state = inker_mode.ensure(ctx)
    tab = _dirty_tab(state)

    journal.pump(ctx, now=0.0)
    assert ctx.submitted == [], "not immediately -- the interval starts at zero"
    journal.pump(ctx, now=journal.JOURNAL_SECONDS + 1.0)
    assert ctx.submitted == [f"journal:inker:{tab.uid}"]
    assert (tmp_path / tab.journal_name).exists()
    # And the sidecar, which is the completion gate: a payload without one was
    # interrupted mid-write and is never offered.
    assert journal.meta_path(tmp_path / tab.journal_name).exists()


def test_a_journal_entry_is_not_a_save(tmp_path):
    """It must not clear dirty, move the saved head or retitle the tab: all
    three would answer "where should this go" on the user's behalf."""
    from warlock.studio import inker_mode, journal

    ctx = _AutosaveCtx(tmp_path)
    state = inker_mode.ensure(ctx)
    tab = _dirty_tab(state)
    head, title = tab.saved_head, tab.title

    journal.pump(ctx, now=journal.JOURNAL_SECONDS + 1.0)
    assert tab.dirty
    assert tab.saved_head == head
    assert tab.title == title
    assert tab.path is None
    assert not tab.saving, "and it must never lock the editor"


def test_an_idle_document_is_not_rewritten_every_interval(tmp_path):
    """Compared against the history head rather than tracked with a flag, so
    an undo back to the journalled position is not a new edit either."""
    from warlock.studio import inker_mode, journal

    ctx = _AutosaveCtx(tmp_path)
    state = inker_mode.ensure(ctx)
    _dirty_tab(state)

    journal.pump(ctx, now=200.0)
    journal.pump(ctx, now=400.0)
    assert len(ctx.submitted) == 1

    state.active.doc.add_layer()
    journal.pump(ctx, now=600.0)
    assert len(ctx.submitted) == 2


def test_a_busy_document_is_skipped(tmp_path):
    """write_ora walks the stack; a second encode mid-save is the archive whose
    parts disagree about the canvas size."""
    from warlock.studio import inker_mode, journal

    ctx = _AutosaveCtx(tmp_path)
    state = inker_mode.ensure(ctx)
    tab = _dirty_tab(state)
    tab.saving = True
    journal.pump(ctx, now=10_000.0)
    assert ctx.submitted == []


def test_saving_for_real_drops_the_crash_copy(tmp_path):
    from warlock.studio import inker_mode, journal

    ctx = _AutosaveCtx(tmp_path)
    state = inker_mode.ensure(ctx)
    tab = _dirty_tab(state)
    journal.pump(ctx, now=10_000.0)
    path = tmp_path / tab.journal_name
    assert path.exists() and journal.meta_path(path).exists()

    inker_mode.drop_autosave(ctx, tab)
    assert not path.exists()
    assert not journal.meta_path(path).exists()
    assert tab.journal_name == ""


def test_dropping_a_copy_that_is_already_gone_does_not_raise(tmp_path):
    """Cleanup, not an edit."""
    from warlock.studio import inker_mode, journal

    ctx = _AutosaveCtx(tmp_path)
    state = inker_mode.ensure(ctx)
    tab = _dirty_tab(state)
    journal.pump(ctx, now=10_000.0)
    (tmp_path / tab.journal_name).unlink()
    inker_mode.drop_autosave(ctx, tab)


def test_recovery_is_offered_only_when_something_was_left_behind(tmp_path):
    from warlock.studio import inker_mode, journal

    ctx = _AutosaveCtx(tmp_path)
    inker_mode.ensure(ctx)
    assert journal.snapshot(ctx) == []

    # A fresh session over the same directory: the scan is one-shot per state,
    # so finding the new copy needs the next launch, which is the whole point.
    ctx = _AutosaveCtx(tmp_path)
    inker_mode.ensure(ctx)
    _leave_behind(tmp_path, "sketch")
    assert [r.title for r in journal.snapshot(ctx)] == ["sketch"]


def test_a_payload_with_no_sidecar_is_never_offered(tmp_path):
    """The completion gate. A copy interrupted between the two writes is a
    payload nothing knows the shape of, and offering it would hand the user a
    truncated archive as though it were their work."""
    from warlock.studio import inker_mode, journal

    ctx = _AutosaveCtx(tmp_path)
    inker_mode.ensure(ctx)
    (tmp_path / "half-pd9.ora").write_bytes(b"interrupted")
    assert journal.recoverable(ctx) == []
    assert journal.snapshot(ctx) == []


def test_a_sidecar_naming_a_payload_that_has_gone_is_skipped(tmp_path):
    from warlock.studio import inker_mode, journal

    ctx = _AutosaveCtx(tmp_path)
    inker_mode.ensure(ctx)
    _leave_behind(tmp_path, "sketch")
    (tmp_path / "sketch-pd9.ora").unlink()
    assert journal.recoverable(ctx) == []


def test_a_sidecar_from_a_version_nobody_understands_is_skipped(tmp_path):
    """A half-understood recovery is worse than none."""
    import json as _json

    from warlock.studio import inker_mode, journal

    ctx = _AutosaveCtx(tmp_path)
    inker_mode.ensure(ctx)
    _leave_behind(tmp_path, "sketch")
    side = journal.meta_path(tmp_path / "sketch-pd9.ora")
    side.write_text(_json.dumps({"version": 99, "kind": "inker"}), encoding="utf-8")
    assert journal.recoverable(ctx) == []


def test_declining_recovery_keeps_the_files(tmp_path):
    """"Not now" is not "delete my work" -- and on the home screen "not now" is
    simply never clicking Recover, so listing must touch nothing on disk."""
    from warlock.studio import inker_mode, journal

    ctx = _AutosaveCtx(tmp_path)
    inker_mode.ensure(ctx)
    left = _leave_behind(tmp_path, "sketch")
    assert len(journal.snapshot(ctx)) == 1
    assert left.exists()
    assert journal.meta_path(left).exists()


def test_a_kind_nothing_can_adopt_is_skipped_rather_than_deleted(tmp_path):
    """The mode may simply not be built into this run, and deleting somebody's
    work because this build does not understand it is the one outcome worse
    than not offering it."""
    from warlock.studio import inker_mode, journal

    ctx = _AutosaveCtx(tmp_path)
    inker_mode.ensure(ctx)
    left = _leave_behind(tmp_path, "mystery", kind="from-the-future")
    found = journal.recoverable(ctx)
    assert [r.kind for r in found] == ["from-the-future"]
    assert found[0].adoptable is False
    assert journal.adopt(ctx, found) == 0
    assert left.exists()


def test_a_recovered_document_opens_untitled_and_dirty(tmp_path):
    """The file it was copied from may still be on disk with its own contents,
    so adopting the path would arm Ctrl+S to overwrite something the user has
    not looked at."""
    from warlock.studio import inker, inker_mode, journal

    ctx = _AutosaveCtx(tmp_path)
    state = inker_mode.ensure(ctx)
    source = inker.Document.blank(8, 8)
    source.stack.active.pixels[:, :] = (1, 2, 3, 255)
    path = tmp_path / "sketch-pd9.ora"
    inker.write_ora(source, path)
    _sidecar(tmp_path, path, "sketch")

    journal.adopt(ctx, journal.recoverable(ctx))
    inker_mode.on_task_done(
        ctx,
        type("Done", (), {"key": "inker-recover:1", "result": ctx.result})(),
    )
    tab = state.active
    assert tab is not None
    assert tab.path is None
    assert tab.dirty
    assert "recovered" in tab.title
    assert tab.journal_name == path.name


# --- a custom new canvas (Ink7) ----------------------------------------------


def test_a_typed_size_is_clamped_rather_than_refused():
    """The snap rule: the fields are being *typed into*, and there is nothing
    useful for a refusal to show halfway through a number."""
    from warlock.studio import inker_mode

    assert inker_mode.clamp_canvas(0, -4) == (1, 1)
    assert inker_mode.clamp_canvas(1920, 1080) == (1920, 1080)
    assert inker_mode.clamp_canvas(99999, 8) == (inker_mode.NEW_MAX, 8)


def test_a_size_that_is_not_a_number_at_all_falls_back_to_one():
    from warlock.studio import inker_mode

    assert inker_mode.clamp_canvas(None, "x") == (1, 1)


def test_a_new_document_honours_a_non_square_size():
    """The whole of Ink7: a user who wanted 1920x1080 had to make a square and
    then resize it, which is two undo steps and a guess about the anchor."""
    from warlock.studio import inker_mode

    ctx = _PaletteCtx()
    doc = inker_mode.new_document(ctx, 1920, 1080)
    assert doc.doc.size == (1920, 1080)


def test_a_new_document_cannot_be_asked_for_a_gigabyte_by_one_stray_digit():
    from warlock.studio import inker_mode

    ctx = _PaletteCtx()
    doc = inker_mode.new_document(ctx, 20480, 20480)
    assert doc.doc.size == (inker_mode.NEW_MAX, inker_mode.NEW_MAX)


# --- canvas rotation and the flipped view (Ink9) ------------------------------
#
# Quarter turns only, and the tests say so as much as the code does: the whole
# licence for leaving every overlay in the pane axis-aligned is that an
# axis-aligned image rectangle comes out an axis-aligned screen rectangle.


ORIENTATIONS = [(r, f) for r in (0, 90, 180, 270) for f in (False, True)]


def _view(rotation=0, flipped=False, zoom=2.0, pan=(7.0, 11.0)):
    from warlock.studio import inker_state

    return inker_state.PaintView(zoom=zoom, pan=pan, rotation=rotation, flipped=flipped)


@pytest.mark.parametrize(("rotation", "flipped"), ORIENTATIONS)
def test_a_round_trip_is_the_identity_in_every_orientation(rotation, flipped):
    """The basis is orthonormal, so its transpose is its inverse -- exactly,
    for all eight of them, rather than to within a rounding error."""
    from warlock.studio import inker_state

    view = _view(rotation, flipped)
    screen = inker_state.to_screen(view, (10.0, 20.0), 4.0, 6.0)
    assert inker_state.to_image(view, (10.0, 20.0), *screen) == pytest.approx((4.0, 6.0))


@pytest.mark.parametrize(("rotation", "flipped"), ORIENTATIONS)
def test_a_quarter_turn_keeps_an_axis_aligned_rectangle_axis_aligned(rotation, flipped):
    """The whole licence for the pane's overlays staying as they were. If this
    ever fails, the grid, the marquee preview and the transform box are all
    quietly drawing the wrong shape."""
    from warlock.studio import inker_state

    view = _view(rotation, flipped)
    corners = [
        inker_state.to_screen(view, (0.0, 0.0), x, y)
        for x, y in ((0, 0), (8, 0), (8, 5), (0, 5))
    ]
    xs = sorted({round(p[0], 6) for p in corners})
    ys = sorted({round(p[1], 6) for p in corners})
    assert len(xs) == 2 and len(ys) == 2


@pytest.mark.parametrize(("rotation", "flipped"), ORIENTATIONS)
def test_the_orientation_preserves_distance(rotation, flipped):
    """Which is why the marching ants need no change to their arc-length
    arithmetic: the basis turns, and turning does not stretch."""
    import math

    from warlock.studio import inker_state

    view = _view(rotation, flipped, zoom=1.0, pan=(0.0, 0.0))
    a = inker_state.to_screen(view, (0.0, 0.0), 1.0, 2.0)
    b = inker_state.to_screen(view, (0.0, 0.0), 4.0, 6.0)
    assert math.dist(a, b) == pytest.approx(5.0)


def test_a_quarter_turn_swaps_the_extent_the_canvas_needs():
    from warlock.studio import inker_state

    upright = _view(0)
    turned = _view(90)
    (lo, hi) = inker_state.view_extent(upright, (100, 50))
    assert (hi[0] - lo[0], hi[1] - lo[1]) == (100.0, 50.0)
    (lo, hi) = inker_state.view_extent(turned, (100, 50))
    assert (hi[0] - lo[0], hi[1] - lo[1]) == (50.0, 100.0)


@pytest.mark.parametrize(("rotation", "flipped"), ORIENTATIONS)
def test_fitting_puts_the_whole_canvas_inside_the_pane_however_it_is_turned(
    rotation, flipped
):
    """The one thing rotation genuinely costs the layout: a turn puts part of
    the canvas at negative view coordinates, so the framing cannot assume the
    corner is at the origin."""
    from warlock.studio import inker_state

    view = _view(rotation, flipped)
    inker_state.fit(view, (100, 50), (200.0, 200.0))
    corners = [
        inker_state.to_screen(view, (0.0, 0.0), x, y)
        for x, y in ((0, 0), (100, 0), (100, 50), (0, 50))
    ]
    lo_x, hi_x = min(p[0] for p in corners), max(p[0] for p in corners)
    lo_y, hi_y = min(p[1] for p in corners), max(p[1] for p in corners)
    assert lo_x >= -1e-6 and hi_x <= 200.0 + 1e-6
    assert lo_y >= -1e-6 and hi_y <= 200.0 + 1e-6
    # Centred on both axes, and touching on the one that limited the zoom.
    assert lo_x == pytest.approx(200.0 - hi_x)
    assert lo_y == pytest.approx(200.0 - hi_y)
    assert min(lo_x, lo_y) == pytest.approx(0.0)


def test_a_flip_mirrors_left_to_right_and_is_its_own_inverse():
    from warlock.studio import inker_state

    view = _view(0, zoom=1.0, pan=(0.0, 0.0))
    right = inker_state.to_screen(view, (0.0, 0.0), 10.0, 0.0)
    inker_state.flip_view(view)
    assert inker_state.to_screen(view, (0.0, 0.0), 10.0, 0.0)[0] == pytest.approx(-right[0])
    inker_state.flip_view(view)
    assert not view.flipped


def test_rotating_cycles_through_the_four_and_keeps_the_zoom():
    """Re-centred through ``pending_zoom`` rather than by clearing ``fitted``,
    which would also re-scale and throw away a zoom the user chose."""
    from warlock.studio import inker_state

    view = _view(0, zoom=3.0)
    for expected in (90, 180, 270, 0):
        inker_state.rotate_view(view)
        assert view.rotation == expected
        assert view.pending_zoom == 3.0
        assert view.zoom == 3.0


def test_rotating_backwards_is_the_other_direction():
    from warlock.studio import inker_state

    view = _view(0)
    inker_state.rotate_view(view, -1)
    assert view.rotation == 270


def test_neither_rotation_nor_flip_is_an_edit():
    """No pixels move, so there is nothing to undo and nothing to save -- which
    is the reason both live on the view rather than on the document."""
    from warlock.studio import inker, inker_state

    doc = inker.Document.blank(8, 8)
    head, rev = doc.history.head, doc.rev
    view = inker_state.PaintView()
    inker_state.rotate_view(view)
    inker_state.flip_view(view)
    assert (doc.history.head, doc.rev) == (head, rev)


def test_a_rotation_off_the_quarter_lattice_reads_as_zero_everywhere():
    """Only code can put one on the view today, which is exactly why the two
    readers must agree about it: ``basis`` always answered such a value as 0,
    while ``rotate_view`` restated the lookup without the guard and raised a
    ValueError out of ``index()`` -- one bad state, two different verdicts."""
    from warlock.studio import inker_state

    view = _view(45)
    assert inker_state.basis(view) == inker_state.basis(_view(0))
    inker_state.rotate_view(view)
    assert view.rotation == 90
    assert view.pending_zoom == view.zoom


def test_an_over_wound_multiple_of_ninety_still_turns_from_where_it_reads():
    """The other side of the shared spelling: 450 is 90 on screen, and a turn
    from it lands on 180 rather than raising or restarting at zero."""
    from warlock.studio import inker_state

    view = _view(450)
    assert inker_state.basis(view) == inker_state.basis(_view(90))
    inker_state.rotate_view(view)
    assert view.rotation == 180


# --- importing an Aseprite file (Q-d) ----------------------------------------


class _ImportCtx:
    """Enough of Ctx for the import door, running its task inline."""

    def __init__(self) -> None:
        from warlock.studio.state import AppState

        self.state = AppState()
        self.settings = _MemorySettings()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str, str | None]] = []
        self.result = None

    def submit(self, key, run, *args):
        self.submitted.append(key)
        self.result = run(*args)
        return True

    def toast(self, message, level="info", action=None, action_arg=None) -> None:
        self.toasts.append((message, level, action))


def _ase_bytes() -> bytes:
    """The smallest Aseprite file: one layer, one frame, one flat cel."""
    import struct

    def string(text):
        raw = text.encode("utf-8")
        return struct.pack("<H", len(raw)) + raw

    def chunk(kind, payload):
        return struct.pack("<IH", len(payload) + 6, kind) + payload

    layer = chunk(
        0x2004,
        struct.pack("<HHHHHHB3s", 3, 0, 0, 0, 0, 0, 255, b"\0\0\0") + string("Art"),
    )
    cel = chunk(
        0x2005,
        struct.pack("<HhhBHh5s", 0, 0, 0, 255, 0, 0, b"\0" * 5)
        + struct.pack("<HH", 1, 1)
        + bytes((1, 2, 3, 255)),
    )
    body = layer + cel
    # 0xFFFF in the legacy chunk count and the real number in the DWORD behind
    # it -- the shape a real Aseprite 1.3 file has. The engine's own suite pins
    # both spellings; this fixture stands in for a file off somebody's disk, so
    # it is written the way that file would be.
    frame = struct.pack("<IHHHHI", len(body) + 16, 0xF1FA, 0xFFFF, 100, 0, 2) + body
    head = struct.pack(
        "<IHHHHHIHIIB3sHBBhhHH",
        0, 0xA5E0, 1, 1, 1, 32, 1, 100, 0, 0, 0, b"\0\0\0", 0, 1, 1, 0, 0, 0, 0,
    ) + b"\0" * 84
    whole = head + frame
    return struct.pack("<I", len(whole)) + whole[4:]


def test_an_imported_aseprite_document_points_at_no_file(tmp_path):
    """The whole read-only guarantee, and it is structural rather than a flag:
    this app reads the format and cannot write it, so a tab pointing at the
    source would let one Ctrl+S put ORA bytes over somebody's artwork."""
    from warlock.studio import inker_mode

    path = tmp_path / "hero.aseprite"
    path.write_bytes(_ase_bytes())
    result = inker_mode._load_aseprite(path)
    assert result["path"] is None
    assert result["format"] == "ora"
    assert result["title"] == "hero"
    assert result["doc"].path is None


def test_dropping_an_aseprite_file_imports_it_rather_than_refusing_it(tmp_path):
    """``.aseprite`` is deliberately not in ``OPENABLE`` -- that tuple is what
    the app can write back -- so the drop route has to know about it or the
    user is told the app cannot do something it can."""
    from warlock.studio import inker_mode

    path = tmp_path / "hero.ase"
    path.write_bytes(_ase_bytes())
    ctx = _ImportCtx()
    inker_mode.open_path(ctx, path)
    assert ctx.submitted and ctx.submitted[0].startswith("inker-open:aseprite")
    assert ctx.result["doc"].size == (1, 1)
    assert not ctx.toasts


def test_the_aseprite_picker_never_runs_on_the_frame_thread(monkeypatch):
    from warlock.studio import dialogs, inker_mode

    opened: list[str] = []

    def fake_open(title, filters=None):
        opened.append(title)
        return None

    monkeypatch.setattr(dialogs, "open_file", fake_open)
    ctx = _ImportCtx()
    inker_mode.ask_import_aseprite(ctx)
    assert ctx.submitted == ["inker-open:aseprite"]
    assert opened, "the picker ran, on the task thread the submit stands in for"


def test_what_an_import_dropped_is_one_toast_and_every_line_in_the_log():
    """A toast per warning is a stack of them for one file; silence is worse
    still, since a drawing that quietly lost something looks merely wrong."""
    from warlock.studio import inker_mode

    ctx = _ImportCtx()
    inker_mode._report_import_warnings(ctx, [])
    assert not ctx.toasts
    inker_mode._report_import_warnings(ctx, ["a colour profile was dropped", "and more"])
    assert len(ctx.toasts) == 1
    message, level, action = ctx.toasts[0]
    assert "colour profile" in message and "+1 more" in message
    assert (level, action) == ("warn", "log")
