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


def test_a_clean_document_is_never_autosaved(tmp_path):
    from warlock.studio import inker, inker_mode

    ctx = _AutosaveCtx(tmp_path)
    state = inker_mode.ensure(ctx)
    doc = inker.Document.blank(8, 8)
    state.add(InkerDoc(doc=doc, title="clean", saved_head=doc.history.head))
    inker_mode.pump_autosave(ctx, now=10_000.0)
    assert ctx.submitted == []
    assert not list(tmp_path.glob("*.ora"))


def test_a_dirty_document_is_autosaved_once_the_interval_has_passed(tmp_path):
    from warlock.studio import inker_mode

    ctx = _AutosaveCtx(tmp_path)
    state = inker_mode.ensure(ctx)
    tab = _dirty_tab(state)

    inker_mode.pump_autosave(ctx, now=0.0)
    assert ctx.submitted == [], "not immediately -- the interval starts at zero"
    inker_mode.pump_autosave(ctx, now=inker_mode.AUTOSAVE_SECONDS + 1.0)
    assert ctx.submitted == [f"inker-autosave:{tab.uid}"]
    assert (tmp_path / tab.autosave_name).exists()


def test_an_autosave_is_not_a_save(tmp_path):
    """It must not clear dirty, move the saved head or retitle the tab: all
    three would answer "where should this go" on the user's behalf."""
    from warlock.studio import inker_mode

    ctx = _AutosaveCtx(tmp_path)
    state = inker_mode.ensure(ctx)
    tab = _dirty_tab(state)
    head, title = tab.saved_head, tab.title

    inker_mode.pump_autosave(ctx, now=inker_mode.AUTOSAVE_SECONDS + 1.0)
    assert tab.dirty
    assert tab.saved_head == head
    assert tab.title == title
    assert tab.path is None
    assert not tab.saving, "and it must never lock the editor"


def test_an_idle_document_is_not_rewritten_every_interval(tmp_path):
    """Compared against the history head rather than tracked with a flag, so
    an undo back to the autosaved position is not a new edit either."""
    from warlock.studio import inker_mode

    ctx = _AutosaveCtx(tmp_path)
    state = inker_mode.ensure(ctx)
    _dirty_tab(state)

    inker_mode.pump_autosave(ctx, now=200.0)
    inker_mode.pump_autosave(ctx, now=400.0)
    assert len(ctx.submitted) == 1

    state.active.doc.add_layer()
    inker_mode.pump_autosave(ctx, now=600.0)
    assert len(ctx.submitted) == 2


def test_a_busy_document_is_skipped(tmp_path):
    """write_ora walks the stack; a second encode mid-save is the archive whose
    parts disagree about the canvas size."""
    from warlock.studio import inker_mode

    ctx = _AutosaveCtx(tmp_path)
    state = inker_mode.ensure(ctx)
    tab = _dirty_tab(state)
    tab.saving = True
    inker_mode.pump_autosave(ctx, now=10_000.0)
    assert ctx.submitted == []


def test_saving_for_real_drops_the_crash_copy(tmp_path):
    from warlock.studio import inker_mode

    ctx = _AutosaveCtx(tmp_path)
    state = inker_mode.ensure(ctx)
    tab = _dirty_tab(state)
    inker_mode.pump_autosave(ctx, now=10_000.0)
    path = tmp_path / tab.autosave_name
    assert path.exists()

    inker_mode.drop_autosave(ctx, tab)
    assert not path.exists()
    assert tab.autosave_name == ""


def test_dropping_a_copy_that_is_already_gone_does_not_raise(tmp_path):
    """Cleanup, not an edit."""
    from warlock.studio import inker_mode

    ctx = _AutosaveCtx(tmp_path)
    state = inker_mode.ensure(ctx)
    tab = _dirty_tab(state)
    inker_mode.pump_autosave(ctx, now=10_000.0)
    (tmp_path / tab.autosave_name).unlink()
    inker_mode.drop_autosave(ctx, tab)


def test_recovery_is_offered_only_when_something_was_left_behind(tmp_path):
    from warlock.studio import inker_mode

    ctx = _AutosaveCtx(tmp_path)
    assert inker_mode.offer_recovery(ctx) is False
    assert ctx.confirms.pending is None

    (tmp_path / "sketch-pd9.ora").write_bytes(b"not really an ora")
    assert inker_mode.offer_recovery(ctx) is True
    assert ctx.confirms.pending is not None
    assert "sketch" in ctx.confirms.pending.message


def test_declining_recovery_keeps_the_files(tmp_path):
    """"Not now" is not "delete my work"."""
    from warlock.studio import inker_mode

    ctx = _AutosaveCtx(tmp_path)
    left = tmp_path / "sketch-pd9.ora"
    left.write_bytes(b"x")
    inker_mode.offer_recovery(ctx)
    ctx.confirms.dismiss()
    assert left.exists()


def test_a_recovered_document_opens_untitled_and_dirty(tmp_path):
    """The file it was copied from may still be on disk with its own contents,
    so adopting the path would arm Ctrl+S to overwrite something the user has
    not looked at."""
    from warlock.studio import inker, inker_mode

    ctx = _AutosaveCtx(tmp_path)
    state = inker_mode.ensure(ctx)
    source = inker.Document.blank(8, 8)
    source.stack.active.pixels[:, :] = (1, 2, 3, 255)
    path = tmp_path / "sketch-pd9.ora"
    inker.write_ora(source, path)

    inker_mode.recover(ctx, [path])
    inker_mode.on_task_done(
        ctx,
        type("Done", (), {"key": "inker-recover:1", "result": ctx.result})(),
    )
    tab = state.active
    assert tab is not None
    assert tab.path is None
    assert tab.dirty
    assert "recovered" in tab.title
    assert tab.autosave_name == path.name
