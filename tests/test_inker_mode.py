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


def test_only_the_ceiling_is_the_inkers_own_now():
    """The floor was 25% and the reversal is deliberate.

    The old claim was that a tile map wants 5% while a drawing at 5% is a
    postage stamp nobody can nib -- true, and it cost Fit: a page too large to
    fit at 25% centred and *overflowed* the pane rather than shrinking to meet
    it, which is the one thing Fit means. Nobody nibs at 5% on purpose, and
    the wheel notch that reaches it also leaves it. The ceiling stays Inker's
    own: 32x is a tile map's magnifier.
    """
    assert inker_state.MIN_ZOOM == inker_state.INKER_MIN_ZOOM
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
    """The stated cost of having a floor: a huge page does not shrink to meet it.

    The document is 40 000 px rather than 8 000 because 8 000 into 400 is
    *exactly* the new 5% floor -- the assertion would still have passed while
    the overflow it names had stopped happening, which is a lie that stays
    green. It is the claim that matters, so the page grew instead.
    """
    view = PaintView()
    inker_state.fit(view, (40000, 40000), (400.0, 400.0), **_inker_bounds())
    assert view.zoom == pytest.approx(inker_state.INKER_MIN_ZOOM)
    # 40 000 at 5% is 2 000 px in a 400 px pane: it overflows, and that is the
    # cost being recorded rather than a bug.
    assert 40000 * view.zoom > 400.0


def test_the_zoom_presets_and_the_ladder_are_two_tables_on_purpose():
    """A "sync the two lists" tidy-up must fail here rather than land.

    The combo answers *"show me exactly this number"*, so 75% belongs on it
    even though a source pixel is then 0.75 screen pixels -- the user asked
    for it. The ladder answers *"the next honest scale"*, so 75% must stay off
    it, or +/- walks into banding unasked.
    """
    assert 0.75 in inker_state.ZOOM_PRESETS
    assert 0.75 not in inker_state.ZOOM_LADDER
    # Both are sorted, and both live inside the pane's own bounds -- a preset
    # the canvas would clamp away is a menu entry that lies.
    for table in (inker_state.ZOOM_PRESETS, inker_state.ZOOM_LADDER):
        assert list(table) == sorted(table)
        assert table[0] >= inker_state.INKER_MIN_ZOOM
        assert table[-1] <= inker_state.INKER_MAX_ZOOM
    # Every ladder rung is still whole either way up (the pixel-art rule).
    for rung in inker_state.ZOOM_LADDER:
        whole = rung if rung >= 1.0 else 1.0 / rung
        assert whole == pytest.approx(round(whole))


def test_a_preset_key_survives_the_half_percent():
    """``int(picked) / 100`` -- Plotter's spelling -- turns 12.5% into 12%."""
    assert inker_state.zoom_key(0.125) == "12.5"
    assert inker_state.zoom_key(1.0) == "100"
    keys = [inker_state.zoom_key(z) for z in inker_state.ZOOM_PRESETS]
    assert len(set(keys)) == len(keys), "two presets sharing a key pick the wrong one"


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

    boot = 5_000.0  # boot-relative, like the real time.monotonic()
    journal.pump(ctx, now=boot)
    assert ctx.submitted == [], "not immediately -- first sight arms the interval"
    journal.pump(ctx, now=boot + journal.JOURNAL_SECONDS + 1.0)
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

    journal.pump(ctx, now=1.0)  # first sight arms the debounce
    journal.pump(ctx, now=journal.JOURNAL_SECONDS + 1.0)
    assert ctx.submitted, "the copy was taken"
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
    journal.pump(ctx, now=9_000.0)  # arm
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
    journal.pump(ctx, now=9_000.0)  # arm
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


# --- a transform commits over the range (Wave 2) -----------------------------
#
# The preview only ever showed the active cel, so the commit is where the
# timeline's range takes effect -- Aseprite's timeline-target behaviour, and the
# visible range outline is what tells the user how far it will reach.


class _Recorder:
    """A document that only remembers which commit it was asked for."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.floating = object()

    def commit_floating(self) -> bool:
        self.calls.append(("plain",))
        return True

    def commit_floating_range(self, *rect) -> bool:
        self.calls.append(("range", rect))
        return True

    def cancel_floating(self) -> bool:
        self.calls.append(("cancel",))
        return True


def _transform_ctx(monkeypatch, rect):
    from types import SimpleNamespace

    doc = _Recorder()
    tab = SimpleNamespace(doc=doc, range_sel=rect)
    # ``transform_uid``/``get`` because ``end_transform`` resolves the owner by
    # uid now; an empty uid falls back to ``active``, which is this tab.
    state = SimpleNamespace(
        active=tab,
        transforming=True,
        transform_uid="",
        get=lambda uid: None,
        clear_drag=lambda: None,
    )
    monkeypatch.setattr(inker_mode, "ensure", lambda ctx: state)
    return doc


def test_ending_a_transform_with_no_range_commits_plain(monkeypatch):
    doc = _transform_ctx(monkeypatch, None)
    inker_mode.end_transform(object(), commit=True)
    assert doc.calls == [("plain",)]


def test_ending_a_transform_with_a_range_commits_over_it(monkeypatch):
    doc = _transform_ctx(monkeypatch, (0, 1, 2, 3))
    inker_mode.end_transform(object(), commit=True)
    assert doc.calls == [("range", (0, 1, 2, 3))]


def test_cancelling_a_transform_is_never_ranged(monkeypatch):
    """Nothing but the active cel was ever written, so there is nothing else
    to put back."""
    doc = _transform_ctx(monkeypatch, (0, 1, 2, 3))
    inker_mode.end_transform(object(), commit=False)
    assert doc.calls == [("cancel",)]


# --- the sheet door's grid suggestion (Part E1) -------------------------------


def _ruled(rows=3, cols=4, cell=16, sep=2, offset=0):
    """A sheet ruled off by dark separator lines, optionally inset."""
    import numpy as np

    height = offset * 2 + rows * cell + sep * (rows - 1)
    width = offset * 2 + cols * cell + sep * (cols - 1)
    out = np.zeros((height, width, 4), dtype=np.uint8)
    out[:, :, 3] = 255
    for r in range(rows):
        for c in range(cols):
            y = offset + r * (cell + sep)
            x = offset + c * (cell + sep)
            out[y:y + cell, x:x + cell, :3] = 200 + r * 5 + c
    return out


def test_a_uniform_detection_seeds_the_three_sheet_fields():
    assert inker_mode._suggest_grid(_ruled(cell=16, sep=2)) == ((16, 16), (0, 0), (2, 2))


def test_the_suggestion_carries_a_border_offset():
    cell, offset, padding = inker_mode._suggest_grid(_ruled(cell=16, sep=2, offset=3))
    assert cell == (16, 16)
    assert offset == (3, 3)
    assert padding == (2, 2)


def test_an_irregular_grid_seeds_nothing():
    """The popup's model is one cell size plus one offset plus one padding, and
    an irregular grid has no such spelling -- recomposing it is the Plotter
    door's move, not this one's."""
    import numpy as np

    sheet = np.zeros((49, 50, 4), dtype=np.uint8)
    sheet[:, :, 3] = 255
    # Rows of 15, 16, 17 -- within the CV gate, so the detector finds a grid,
    # but the segments are not one size.
    y = 0
    for height in (15, 16, 17):
        x = 0
        for width in (16, 16):
            sheet[y:y + height, x:x + width, :3] = 200
            x += width + 2
        y += height + 1
    from warlock.studio.tilegrid import slicing

    assert slicing.detect_grid(sheet) is not None
    assert inker_mode._suggest_grid(sheet) is None


def test_no_detection_suggests_nothing():
    import numpy as np

    flat = np.full((64, 64, 4), 200, dtype=np.uint8)
    flat[:, :, 3] = 255
    assert inker_mode._suggest_grid(flat) is None


class _Done:
    def __init__(self, key, result):
        self.key, self.result, self.message = key, result, ""


class _Settings:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value


class _Ctx:
    def __init__(self):
        self.state = type("S", (), {"inker": None, "mode": "home",
                                    "previous_mode": "home", "mode_observed": "home",
                                    "preview": {}})()
        self.settings = _Settings()

    def toast(self, message, kind="info"):
        pass


def test_a_none_detection_keeps_the_previous_field_values():
    """The three fields persist across imports on purpose -- a folder of sheets
    cut the same way is typed once -- so a failed detection must keep the last
    values rather than resetting them to the defaults."""
    import numpy as np

    ctx = _Ctx()
    state = inker_mode.ensure(ctx)
    state.sheet_cell = (24, 24)
    state.sheet_offset = (1, 1)
    state.sheet_padding = (3, 3)

    flat = np.full((64, 64, 4), 200, dtype=np.uint8)
    flat[:, :, 3] = 255
    inker_mode.on_task_done(
        ctx, _Done("inker-sheetin", {"atlas": flat, "title": "s", "suggest": None})
    )

    assert state.sheet_cell == (24, 24)
    assert state.sheet_offset == (1, 1)
    assert state.sheet_padding == (3, 3)


def test_a_detection_overwrites_the_field_values():
    ctx = _Ctx()
    state = inker_mode.ensure(ctx)
    state.sheet_cell = (24, 24)
    sheet = _ruled(cell=16, sep=2)
    inker_mode.on_task_done(
        ctx,
        _Done(
            "inker-sheetin",
            {"atlas": sheet, "title": "s", "suggest": inker_mode._suggest_grid(sheet)},
        ),
    )
    assert state.sheet_cell == (16, 16)
    assert state.sheet_padding == (2, 2)


# --- the transform's owner ----------------------------------------------------
#
# ``transforming`` was an app-level bool with no tab identity -- the exact bug
# shape ``convert_uid`` documents: the modal lives on one document, the state
# is shared by every tab, and Enter after a mid-transform switch committed the
# *new* tab's floating buffer while the owner's lifted pixels stayed stranded.


def test_a_tab_switch_cancels_the_transform_on_its_owner():
    import numpy as np

    a, b = _tab("a"), _tab("b")
    a.doc.stack.active.pixels[...] = 200
    before = a.doc.stack.active.pixels.copy()
    state = _state(a, b)
    state.activate(a.uid)
    assert a.doc.begin_transform()
    state.transforming = True
    state.transform_uid = a.uid

    state.activate(b.uid)

    assert not state.transforming
    assert state.transform_uid == ""
    assert a.doc.floating is None, "the owner's lift went back where it came from"
    # The pixels are exactly where they were lifted from -- the select step
    # ``begin_transform`` pushed remains (a selection op moves the head), but
    # no lifted pixel is stranded and no alpha-cut survives.
    assert np.array_equal(a.doc.stack.active.pixels, before)


def test_closing_the_owner_mid_transform_clears_the_mode():
    a, b = _tab("a"), _tab("b")
    state = _state(a, b)
    state.activate(a.uid)
    assert a.doc.begin_transform()
    state.transforming = True
    state.transform_uid = a.uid

    state.close(a.uid)

    assert not state.transforming
    assert state.transform_uid == ""


def test_end_transform_lands_on_the_owner_by_uid():
    from types import SimpleNamespace

    a = _tab("a")
    state = _state(a)
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))
    assert a.doc.begin_transform()
    state.transforming = True
    state.transform_uid = a.uid

    inker_mode.end_transform(ctx, commit=False)

    assert a.doc.floating is None
    assert not state.transforming
    assert state.transform_uid == ""


# --- the brush's slot claim, and the rename buffer ----------------------------


def test_swapping_colours_releases_the_palette_slot_claim():
    """X puts the background in hand, and the background never came from a
    slot -- a claim left standing would land the next stroke in the slot of a
    colour no longer held."""
    state = InkerState()
    state.set_fg((1, 2, 3, 255), slot=4)
    assert state.fg_slot == 4
    state.swap_colours()
    assert state.fg_slot is None


def test_a_tab_switch_drops_a_half_typed_tag_rename():
    """``tag_editing`` indexes the *active* document's tag list, so surviving
    a switch left the rename box open over another document's tag of the same
    number."""
    a, b = _tab("a"), _tab("b")
    state = _state(a, b)
    state.activate(a.uid)
    state.tag_editing = 2
    state.tag_name = "walk"
    state.activate(b.uid)
    assert state.tag_editing == -1
    assert state.tag_name == ""


# --- the filter session, and the save that used to capture it ---------------


def test_a_save_cancels_an_open_filter_preview():
    """``preview_filter`` writes into the layer every frame the popup is up and
    pushes nothing, so a save that did not settle the session serialised a
    filter the user had never approved -- and ``mark_saved`` then called the
    tab clean against it. ``_settle`` cancels the conversion preview for
    exactly this reason and had missed its twin."""
    import numpy as np

    tab = _tab()
    tab.doc.stack.active.pixels[...] = (10, 20, 30, 255)
    tab.doc.invalidate_all()
    before = tab.doc.stack.active.pixels.copy()
    head = tab.doc.history.head

    ctx = _Ctx()
    state = inker_mode.ensure(ctx)
    state.add(tab)

    tab.doc.begin_filter()
    state.filter_uid = tab.uid
    tab.doc.preview_filter("invert", red=True, green=True, blue=True)
    assert not np.array_equal(tab.doc.stack.active.pixels, before), "the preview shows"

    inker_mode._settle(ctx, tab)

    assert np.array_equal(tab.doc.stack.active.pixels, before), "the save puts them back"
    assert tab.doc.history.head == head, "and an unanswered preview is not a step"
    assert tab.doc._filter is None
    assert state.filter_uid == ""


def test_a_filter_session_is_settled_on_its_owner_not_the_active_tab():
    """The session lives on one document while ``InkerState`` is shared, so a
    settle asked about tab B must leave tab A's session alone -- the bug
    ``convert_uid`` was introduced to fix, in the popup it was cloned from."""
    import numpy as np

    a, b = _tab("a"), _tab("b")
    a.doc.stack.active.pixels[...] = (10, 20, 30, 255)
    a.doc.invalidate_all()
    ctx = _Ctx()
    state = inker_mode.ensure(ctx)
    state.add(a)
    state.add(b)

    a.doc.begin_filter()
    state.filter_uid = a.uid
    a.doc.preview_filter("invert", red=True, green=True, blue=True)
    previewed = a.doc.stack.active.pixels.copy()

    inker_mode._settle(ctx, b)
    assert a.doc._filter is not None, "another tab's session is left alone"
    assert state.filter_uid == a.uid
    assert np.array_equal(a.doc.stack.active.pixels, previewed)

    inker_mode._settle(ctx, a)
    assert a.doc._filter is None
    assert state.filter_uid == ""


# --- space-to-pan is a hold, and a hold has to be let go of ------------------


def _space(down=True):
    import pygame

    return type(
        "E", (), {"key": pygame.K_SPACE, "type": pygame.KEYDOWN if down else pygame.KEYUP,
                  "mod": 0}
    )()


def test_closing_the_last_tab_with_space_held_does_not_latch_the_pan():
    """``handle_key``'s "is there a document" guards used to sit in front of the
    Space branch, so the release was dropped and every left-drag panned instead
    of painting for the rest of the session."""
    ctx = _Ctx()
    state = inker_mode.ensure(ctx)
    tab = _tab()
    state.add(tab)

    assert inker_mode.handle_key(ctx, _space(down=True))
    assert state.space_held

    state.close(tab.uid)
    assert not state.docs
    inker_mode.handle_key(ctx, _space(down=False))
    assert not state.space_held


def test_a_tab_switch_lets_go_of_a_held_space():
    """The other route: ``main`` gates both key edges on ``_passes_text_field``,
    which answers no for a plain Space, so a release arriving while a text field
    has focus never reaches ``handle_key`` at all."""
    a, b = _tab("a"), _tab("b")
    state = _state(a, b)
    state.activate(a.uid)
    state.space_held = True
    state.activate(b.uid)
    assert not state.space_held


# --- a gesture belongs to the tab it started on ------------------------------


def test_a_tab_switch_drops_a_timeline_range_anchor():
    """``timeline_anchor`` is an index pair into the *active* tab's grid while
    ``range_sel`` lives on the tab, so clicking a cell in one document and
    Shift+clicking in another built the second one's range from the first's
    coordinates. The most reachable of the three -- it needs no chord at all."""
    a, b = _tab("a"), _tab("b")
    state = _state(a, b)
    state.activate(a.uid)
    state.timeline_anchor = (5, 20)
    state.activate(b.uid)
    assert state.timeline_anchor is None


def test_a_tab_switch_drops_an_open_eye_drag():
    """The pre-image is keyed by row index, so a drag carrying on over another
    tab's rows pushed one ``set_layers_props`` with a pre-image belonging partly
    to each -- an undo that restores the wrong values."""
    a, b = _tab("a"), _tab("b")
    state = _state(a, b)
    state.activate(a.uid)
    state.eye_drag = False
    state.eye_drag_was = {0: {"visible": True}}
    state.activate(b.uid)
    assert state.eye_drag is None
    assert state.eye_drag_was == {}


def test_a_tab_switch_drops_the_text_stamp_target():
    a, b = _tab("a"), _tab("b")
    state = _state(a, b)
    state.activate(a.uid)
    state.text_at = (12, 9)
    state.text_uid = a.uid
    state.activate(b.uid)
    assert state.text_uid == ""
    assert state.text_at == (0, 0)


def test_a_text_stamp_is_refused_for_another_tabs_press():
    """The door behind the popup's own close: a stamp is a write, and a write at
    coordinates taken from a different picture is refused rather than relied on
    being unreachable."""
    a, b = _tab("a"), _tab("b")
    state = _state(a, b)
    state.text_buffer = "hello"
    state.text_at = (2, 2)
    state.text_uid = a.uid
    ctx = _Ctx()
    ctx.state.inker = state
    assert inker_mode.stamp_text(ctx, state, b) is False
    assert b.doc.floating is None


def test_every_task_key_inker_submits_is_answered():
    """The guard the palette export needed. Both palette commands submitted
    under one key -- so ``tasks.submit`` refused the second, silently, since
    neither call site reads the bool -- and *neither* had a branch in
    ``on_task_done``: the key fell through to the uid-keyed tail, found no
    ``:``, and returned. An export reported neither success nor failure.

    A key is answered if ``on_task_done`` names it, or if it is keyed on a tab
    uid, which is what the tail resolves. Anything else is a command that can
    finish and say nothing."""
    import inspect
    import pathlib
    import re

    from warlock.studio import inker_mode

    root = pathlib.Path(inker_mode.__file__).resolve().parent
    handled = inspect.getsource(inker_mode.on_task_done)

    keys: set[tuple[str, str]] = set()
    for path in [root / "inker_mode.py", *sorted((root / "panes").glob("inker_*.py"))]:
        source = path.read_text(encoding="utf-8")
        for found in re.findall(r'submit\(\s*f?"(inker-[^"]*)"', source):
            keys.add((found.split(":", 1)[0], found))
    assert keys, "the scan has to find something or it proves nothing"

    for prefix, template in sorted(keys):
        if "{tab.uid}" in template:
            continue  # the uid-keyed tail resolves these
        assert f'"{prefix}"' in handled, f"{template} finishes and nothing answers it"
