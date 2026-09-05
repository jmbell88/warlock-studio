"""Using Warlock Studio feels and acts the same whichever mode is open.

The second consistency pass (2026-09-05), after the first closed the asset
clock, the document header, the cross-workspace verbs, the empty states, the
spacing tokens and the inspector's exits. Three sweeps -- behaviour, look and
wording, enforcement -- found the divergences below, and each test here is
one of them stated as a claim. Source scans where the failure being guarded
is a *call site* getting it wrong later, behaviour where a helper carries the
rule.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from warlock.studio import inker_state
from warlock.studio.panes import inker_canvas, packwright_preview, plotter_canvas

# --- the wheel (A1) ---------------------------------------------------------


def _view(zoom: float = 1.0) -> inker_state.PaintView:
    view = inker_state.PaintView()
    view.zoom = zoom
    view.pan = (0.0, 0.0)
    return view


def test_the_wheel_zooms_on_the_five_percent_lattice_in_every_canvas():
    """Plotter and Packwright zoomed multiplicatively on the backend-halved
    count and never reached 100% from 83%; Inker scrolled. One rule now."""
    view = _view(0.834)
    inker_state.wheel(view, (0.0, 0.0), (0.0, 0.0), 1.0)
    assert view.zoom == pytest.approx(0.90)
    inker_state.wheel(view, (0.0, 0.0), (0.0, 0.0), 2.0)
    assert view.zoom == pytest.approx(1.0)


def test_shift_and_the_wheel_scrolls_sideways_instead_of_zooming():
    view = _view()
    along = inker_state.wheel(view, (0.0, 0.0), (0.0, 0.0), -2.0, shift=True)
    assert along == -2.0
    assert view.zoom == pytest.approx(1.0)


def test_a_tilt_wheel_scrolls_sideways_with_the_opposite_sign():
    view = _view()
    along = inker_state.wheel(view, (0.0, 0.0), (0.0, 0.0), 0.0, 1.5)
    assert along == -1.5
    assert view.zoom == pytest.approx(1.0)


@pytest.mark.parametrize("pane", [inker_canvas, plotter_canvas, packwright_preview])
def test_every_two_d_canvas_asks_the_shared_wheel_rule(pane):
    """No canvas reads ``io.mouse_wheel`` into its own zoom any more."""
    source = inspect.getsource(pane)
    assert "inker_state.wheel(" in source
    assert not re.search(r"zoom_about\([^)]*mouse_wheel", source)


# --- the 3-D viewports (A2, A3) ---------------------------------------------


def test_alt_drag_orbits_in_the_pose_viewer_as_it_does_in_clay(monkeypatch):
    """``viewer_embed`` panned on Alt+drag (citing Maya) while ``_view_drag``
    orbited on it (saying it must never be reinterpreted). One gesture."""
    import pygame

    from warlock.studio import viewer_embed

    viewer = viewer_embed.Viewer.__new__(viewer_embed.Viewer)
    viewer._grab = None
    viewer._last_mouse = (0.0, 0.0)
    viewer.pose_mode = False
    viewer._deselect_on_click = False
    monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_ALT)
    assert viewer._press(1, (10.0, 10.0)) is True
    assert viewer._grab == "orbit"
    viewer._grab = None
    assert viewer._press(2, (10.0, 10.0)) is True
    assert viewer._grab == "pan"
    viewer._grab = None
    assert viewer._press(3, (10.0, 10.0)) is True
    assert viewer._grab is None, "the right button is not a second orbit"


def test_the_axis_view_keys_are_one_function_both_viewports_call():
    from warlock.studio import clay_mode, poser_mode

    assert "clay_mode.axis_view_key(" in inspect.getsource(poser_mode.handle_key)
    assert "axis_view_key(" in inspect.getsource(clay_mode)
    assert "look_along" not in inspect.getsource(poser_mode.handle_key)


# --- drops, tabs, the status bar (A4-A7) ------------------------------------


def test_every_mode_either_takes_a_drop_or_says_it_does_not():
    """Poser and Troupe were given a refusal on 2026-09-04; Muse, Review and
    Settings still fell through to Create, which switched the window."""
    from warlock.studio import main, modes

    source = inspect.getsource(main.App._on_drop)
    handled = set(re.findall(r'ctx\.state\.mode == "(\w+)"', source))
    starts = {"home", "library", "create"}
    covered = handled | set(main.DROP_REFUSALS) | starts
    assert covered == set(modes.KEYS), sorted(set(modes.KEYS) - covered)
    assert "DROP_REFUSALS[ctx.state.mode]" in source


def test_find_path_folds_case_in_every_mode(tmp_path):
    """``Level.WBLK`` from recents and ``level.wblk`` from a drop forked into
    two tabs in four of the five modes; Plotter alone normcased."""
    from types import SimpleNamespace

    from warlock.studio import (
        clay_state,
        docmodes,
        inker_state,
        packwright_state,
        plotter_state,
        sirens_state,
    )

    for module in (clay_state, inker_state, packwright_state, plotter_state, sirens_state):
        source = inspect.getsource(module)
        assert "docmodes.find_path(self.docs, path)" in source, module.__name__

    real = tmp_path / "level.wblk"
    real.write_bytes(b"")
    docs = [SimpleNamespace(path=real)]
    spelled = tmp_path / "LEVEL.WBLK"
    import os

    if os.path.normcase("A") == os.path.normcase("a"):
        assert docmodes.find_path(docs, spelled) is docs[0]
    assert docmodes.find_path(docs, real) is docs[0]
    assert docmodes.find_path(docs, tmp_path / "other.wblk") is None


def test_closing_a_tab_mid_save_is_refused_out_loud_in_every_mode():
    """Clay refused silently; the other four did not refuse at all, and the
    serialise task reads the live document on a task thread."""
    from types import SimpleNamespace

    from warlock.studio import (
        clay_mode,
        docmodes,
        inker_mode,
        packwright_mode,
        plotter_mode,
        sirens_mode,
    )

    for module in (clay_mode, inker_mode, packwright_mode, plotter_mode, sirens_mode):
        assert "docmodes.close_tab(ctx, state," in inspect.getsource(module), module.__name__

    toasts: list[tuple[str, str]] = []
    ctx = SimpleNamespace(toast=lambda text, level="info", **_: toasts.append((text, level)))
    tab = SimpleNamespace(uid="t1", saving=True, dirty=True, title="x")
    closed: list[str] = []
    state = SimpleNamespace(get=lambda uid: tab, close=closed.append)
    docmodes.close_tab(ctx, state, "t1", lambda _tab: None)
    assert closed == []
    assert toasts == [(docmodes.CLOSE_WHILE_SAVING, "info")]


def test_the_status_bar_reports_zoom_and_tool_for_every_canvas_mode():
    """Inker alone had a tool and a zoom item, from a branch of its own."""
    from types import SimpleNamespace

    from warlock.studio import status_bar

    class State:
        mode = "plotter"
        errors: list = []

    tab = SimpleNamespace(label="Untitled###pl1", dirty=False, view=SimpleNamespace(zoom=1.07))
    state = State()
    state.plotter = SimpleNamespace(active=tab, tool="stamp", docs=[tab])
    ctx = SimpleNamespace(
        state=state, cache=SimpleNamespace(jobs=[]), runtime=SimpleNamespace(checks=[])
    )
    items = {item.key: item.text for item in status_bar.items(ctx)}
    assert items["document"] == "Untitled"
    assert items["tool"] == "Stamp"
    assert items["zoom"] == "107%"
    assert "poser_mode.document_label" in inspect.getsource(status_bar.items)


# --- exports, the sheet, recovery, refusals, the write gate (A8-A12) ---------


def test_the_file_export_prints_the_chord_the_mode_binds():
    """The palette row said Ctrl+E for every mode; Ctrl+E is the *library*
    export in four of them and the file export is Ctrl+Shift+E."""
    from types import SimpleNamespace

    from warlock.studio import palette

    assert palette._doc_export_hint(SimpleNamespace(state=SimpleNamespace(mode="clay"))) == "Ctrl+E"
    for mode in ("inker", "plotter", "packwright", "sirens"):
        ctx = SimpleNamespace(state=SimpleNamespace(mode=mode))
        assert palette._doc_export_hint(ctx) == "Ctrl+Shift+E", mode


def test_sirens_binds_the_file_export_and_clay_no_longer_aliases_it():
    from warlock.studio import clay_mode, sirens_keys

    assert 'name == "e" and shift' in inspect.getsource(sirens_keys._ctrl_key)
    assert 'name == "e" and not shift' in inspect.getsource(clay_mode._ctrl_key)


@pytest.mark.parametrize("group", ["Clay", "Inker", "Plotter", "Packwright", "Sirens", "Poser"])
def test_the_sheet_says_ctrl_shift_z_redoes_wherever_it_does(group):
    from warlock.studio import shortcuts

    rows = dict(shortcuts.shortcut_sections())[group]
    text = " ".join(f"{keys} {what}" for keys, what in rows)
    assert "Ctrl+Shift+Z" in text
    if group != "Poser":
        assert "Ctrl+Shift+Tab" in text


def test_a_crash_copy_that_will_not_reopen_warns_the_same_way_in_every_mode():
    """Inker let the exception through, Clay raised an error, three warned."""
    from types import SimpleNamespace

    from warlock.studio import (
        clay_mode,
        inker_mode,
        journal,
        packwright_mode,
        plotter_mode,
        sirens_mode,
    )

    for module in (clay_mode, inker_mode, packwright_mode, plotter_mode, sirens_mode):
        assert "journal.adopt_failed(ctx," in inspect.getsource(module), module.__name__
    toasts: list = []
    journal.adopt_failed(SimpleNamespace(toast=lambda *a, **k: toasts.append((a, k))), "map")
    assert toasts == [(("A recovered map could not be reopened.", "warn"), {"action": "log"})]


def test_a_refusal_outside_inker_is_coalesced_and_carries_its_remedy():
    from types import SimpleNamespace

    from warlock.studio import clay_mode, docmodes, plotter_mode

    calls: list = []
    ctx = SimpleNamespace(toast_once=lambda *a: calls.append(a))
    docmodes.refuse(ctx, "Select some cells first.")
    docmodes.refuse(ctx, "Ground is locked.", action="unlock", action_arg="7")
    assert calls == [
        ("Select some cells first.", "error", None, None),
        ("Ground is locked.", "error", "unlock", "7"),
    ]
    # A fake without the coalescing twin still gets the sentence.
    plain: list = []
    docmodes.refuse(SimpleNamespace(toast=lambda *a, **k: plain.append((a, k))), "x")
    assert plain == [(("x", "error"), {})]
    assert "docmodes.refuse(" in inspect.getsource(clay_mode._toast)
    assert "docmodes.refuse(" in inspect.getsource(plotter_mode._locked_toast)


def test_export_waits_for_a_save_in_every_document_mode():
    """Inker alone refused Ctrl+E mid-save; the others exported the document a
    task thread was still serialising."""
    from types import SimpleNamespace

    from warlock.studio import (
        clay_mode,
        docmodes,
        inker_keys,
        packwright_mode,
        plotter_mode,
        sirens_keys,
    )

    for module in (clay_mode, inker_keys, packwright_mode, plotter_mode, sirens_keys):
        assert "e" in module._MUTATING_CTRL, module.__name__
        source = inspect.getsource(module)
        assert "docmodes.blocked_while_writing(tab, name, _MUTATING_CTRL)" in source
    saving = SimpleNamespace(saving=True)
    assert docmodes.blocked_while_writing(saving, "e")
    assert not docmodes.blocked_while_writing(saving, "c")
    playing = SimpleNamespace(saving=False, busy=True)
    assert docmodes.blocked_while_writing(playing, "z")


# --- feel the same: transports, verbs, glyphs, busy, bridges (B1-B13) --------


def _pane_sources() -> dict[str, str]:
    from warlock.studio import panes

    root = Path(panes.__file__).parent
    return {path.name: path.read_text(encoding="utf-8") for path in root.glob("*.py")}


def test_every_transport_is_the_one_helper():
    """Six spellings across seven panes -- "Pause" over a stop square, no
    glyph, the chord in the label, a primary Play."""
    from warlock.studio import widgets

    sources = _pane_sources()
    for name in (
        "sirens_transport.py",
        "inker_preview.py",
        "muse_player.py",
        "muse_results.py",
        "plotter_tileset_editor.py",
    ):
        assert "widgets.transport(" in sources[name], name
    for name in ("inker_timeline.py", "troupe_preview.py"):
        assert "widgets.transport_label(" in sources[name], name
    for name, text in sources.items():
        assert '"Pause"' not in text, name
        assert re.search(r'"Stop" if .* else "Play"', text) is None, name
    label, glyph, tip = widgets.transport_label(True)
    assert (label, tip) == ("Stop", "Stop (Space)")
    assert widgets.transport_label(False)[0] == "Play"
    assert widgets.transport_label(False, shortcut="")[2] == "Play"


def test_frame_counters_are_one_spelling():
    sources = _pane_sources()
    for name in ("inker_preview.py", "troupe_preview.py", "plotter_tileset_editor.py"):
        assert "widgets.frame_counter(" in sources[name], name
    for name in ("inker_preview.py", "troupe_preview.py", "plotter_tileset_editor.py"):
        assert 'f"frame {' not in sources[name], name
        assert 'imgui.text(f"{index + 1}/' not in sources[name], name


def test_delete_destroys_and_remove_detaches():
    """Sirens said Remove for an instrument it destroys; Packwright said
    Remove with a trash glyph for a source whose file stays on disk."""
    sources = _pane_sources()
    assert 'f"{icons.TRASH} Remove"' not in sources["sirens_instruments.py"]
    assert 'f"{icons.TRASH} Delete"' in sources["sirens_instruments.py"]
    assert 'f"{icons.TRASH} Delete"' in sources["sirens_effects.py"]
    assert 'f"{icons.MINUS} Remove"' in sources["packwright_sources.py"]
    for name, text in sources.items():
        assert "TRASH} Remove" not in text, name


def test_zoom_wears_the_zoom_glyphs_and_no_icon_is_an_empty_string():
    from warlock.studio import icons

    sources = _pane_sources()
    assert "icons.ZOOM_OUT}##palette-zoom-out" in sources["plotter_tileset.py"]
    assert "icons.ZOOM_IN}##palette-zoom-in" in sources["plotter_tileset.py"]
    for name in dir(icons):
        if name.isupper():
            assert getattr(icons, name) != "", f"icons.{name} draws nothing"


def test_a_shortcut_lives_in_the_tooltip_never_in_the_label():
    sources = _pane_sources()
    for name, text in sources.items():
        assert not re.search(r'f?"\{icons\.\w+\} [^"\n]*\((Ctrl\+\w+|Space|R)\)"', text), name
        assert "(Ctrl+E)" not in text and "(Space)\"" not in text and "Repack (R)" not in text, name


def test_every_bridge_has_one_primary_and_the_exits_heading():
    """Packwright's library export was a plain button where Clay's is the
    primary; Troupe's Send and Sirens' export had no primary at all; Plotter
    had no exit on screen and Inker's sat under "Generation"."""
    from warlock.studio import widgets

    sources = _pane_sources()
    for name in ("packwright_bridge.py", "sirens_bridge.py", "troupe_settings.py"):
        assert "widgets.primary_button(" in sources[name], name
    for name in (
        "clay_bridge.py",
        "packwright_bridge.py",
        "troupe_bridge.py",
        "plotter_bridge.py",
        "sirens_bridge.py",
        "inker_generate.py",
    ):
        heading = f'"{widgets.EXITS_HEADING}"'
        assert "widgets.exits()" in sources[name] or heading in sources[name], name
    assert 'widgets.section("Model file")' in sources["clay_bridge.py"]
    assert 'section("Document")' not in sources["clay_bridge.py"]
    for name, text in sources.items():
        assert 'role="primary"' not in text, name


def test_every_bridge_draws_the_one_history_block():
    """Four bridges drew the pair by hand, each saying Inker drew it twice;
    Inker drew it nowhere, and its popover stepped the stack directly."""
    from warlock.studio import inker_mode, packwright_mode, sirens_mode

    sources = _pane_sources()
    for name in (
        "clay_bridge.py",
        "plotter_bridge.py",
        "packwright_bridge.py",
        "sirens_bridge.py",
        "inker_generate.py",
    ):
        assert "widgets.history_block(" in sources[name], name
        assert "Nothing to undo yet." not in sources[name], name
    assert "widgets.history_rows(" in sources["inker_menu.py"]
    assert "history.step_to(" not in sources["inker_menu.py"]
    assert "widgets.recent_files(" in sources["inker_generate.py"]
    for module in (inker_mode, packwright_mode, sirens_mode):
        assert callable(module.step_history)


def test_busy_is_one_spelling():
    sources = _pane_sources()
    for name, text in sources.items():
        assert not re.search(r'widgets\.muted\((f?)"[A-Z][a-z]+ing\.\.\."\)', text), name


def test_no_pane_draws_a_raw_separator_or_a_heading_as_body_text():
    from warlock.studio import widgets

    sources = _pane_sources()
    for name, text in sources.items():
        assert "imgui.separator()" not in text, name
        for title in ("Map properties", "Go to coordinate", "New canvas", "Resize the map"):
            assert f'imgui.text("{title}"' not in text, (name, title)
    assert callable(widgets.divider) and callable(widgets.popup_title)
