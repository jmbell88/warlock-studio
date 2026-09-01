"""The context bar's sitting-wide controls, pressed rather than inspected.

``tests/inker/test_pattern_fill.py`` states the rule this file follows: *a
control that is drawn and wired to nothing is this codebase's most common
historical defect*, so every test here goes through a real press on the rect
the control actually occupies and then asserts the engine's state moved. A
test that called the writer directly would pass just as happily against a bar
with no button on it at all -- which is the exact shape of the defect that left
``canvas_options`` with zero callers and symmetry permanently off.

Real imgui, no GL: nothing is rendered, only laid out.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _ui_context import imgui_context

from warlock.studio import inker, inker_state, probe, widgets
from warlock.studio.inker import brush
from warlock.studio.panes import inker_context


@pytest.fixture
def ui(monkeypatch):
    """The shared imgui context; see ``_ui_context`` for why it is not a
    conftest fixture."""
    with imgui_context(monkeypatch) as imgui:
        yield imgui

def _scene(monkeypatch):
    state = inker_state.InkerState()
    state.set_tool("brush")
    tab = SimpleNamespace(
        doc=inker.Document.blank(32, 32),
        tiled="off",
        busy=False,
        uid="t",
        range_sel=None,
        view=inker_state.PaintView(),
    )
    state.docs.append(tab)
    state.active_uid = "t"
    written: list[str] = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=state),
        settings=SimpleNamespace(
            get=lambda _key: None, set=lambda _key, _value: written.append(_key)
        ),
        toast=lambda *a, **k: None,
        viewer=None,
    )
    monkeypatch.setattr(widgets, "FORCE_SECTIONS_OPEN", True)
    return ctx, state, tab, written


def _bar(ui, ctx, state, tab):
    def build():
        inker_context.draw(ctx, state, tab)

    return build


def _frame(imgui, build, *, pos=(-100.0, -100.0), down=False):
    io = imgui.get_io()
    io.add_mouse_pos_event(pos[0], pos[1])
    io.add_mouse_button_event(0, down)
    probe.begin_frame()
    imgui.new_frame()
    imgui.set_next_window_size((1000.0, 700.0))
    imgui.set_next_window_pos((0.0, 0.0))
    imgui.begin("##host")
    build()
    imgui.end()
    imgui.end_frame()
    return list(probe.FRAME_CONTROLS)


def _laid_out(imgui, build, frames=2):
    """Two frames: a fresh popup's first pass only measures, and a rect read
    from that pass sits in a window that is not yet visible -- which is the
    state in which a press lands on nothing."""
    out = []
    for _ in range(frames):
        out = _frame(imgui, build)
    return out


def _find(controls, label):
    found = [c for c in controls if c.label.startswith(label)]
    assert found, f"{label!r} not among {sorted({c.label for c in controls})}"
    return found[0]


def _click(imgui, build, control):
    """Press *and* release: imgui fires a button on the release inside it.

    ``Control.centre`` rather than the middle of ``rect``: imgui groups a
    checkbox with the text beside it, so the rect's centre can land on the
    label, where a click does nothing. That is exactly the false ``inert`` the
    probe grew ``hit`` to stop reporting.
    """
    _frame(imgui, build, pos=control.centre, down=True)
    _frame(imgui, build, pos=control.centre, down=False)


def test_the_bar_says_the_word_symmetry(ui, monkeypatch):
    """The complaint this whole change answers: four buttons labelled ``H``,
    ``V``, ``\\`` and ``/`` with nothing on screen naming them."""
    ctx, state, tab, _written = _scene(monkeypatch)
    controls = _laid_out(ui, _bar(ui, ctx, state, tab))
    _find(controls, "Sym")


def test_pressing_a_mirror_toggles_that_axis_and_only_that_axis(ui, monkeypatch):
    ctx, state, tab, _written = _scene(monkeypatch)
    build = _bar(ui, ctx, state, tab)
    controls = _laid_out(ui, build)
    _click(ui, build, _find(controls, "H##"))
    assert brush.axes_of(state.symmetry) == ("x",)
    controls = _laid_out(ui, build)
    _click(ui, build, _find(controls, "\\##"))
    assert brush.axes_of(state.symmetry) == ("x", "diag")
    controls = _laid_out(ui, build)
    _click(ui, build, _find(controls, "H##"))
    assert brush.axes_of(state.symmetry) == ("diag",)


def test_a_mirror_press_is_written_down(ui, monkeypatch):
    """The defect underneath the discoverability one: ``persist`` was called on
    every press and wrote no symmetry at all."""
    ctx, state, tab, written = _scene(monkeypatch)
    build = _bar(ui, ctx, state, tab)
    controls = _laid_out(ui, build)
    _click(ui, build, _find(controls, "H##"))
    assert "inker" in written


def test_the_symmetry_word_lights_up_when_a_mirror_is_on(ui, monkeypatch):
    """Collapsed, the word is the only thing that can say a mirror is armed."""
    ctx, state, tab, _written = _scene(monkeypatch)
    build = _bar(ui, ctx, state, tab)
    assert _find(_laid_out(ui, build), "Sym").selected is False
    state.symmetry = "x"
    assert _find(_laid_out(ui, build), "Sym").selected is True


def test_the_radial_count_is_reachable_from_the_bar(ui, monkeypatch):
    """It was behind a flip-horizontal glyph in another pane."""
    ctx, state, tab, _written = _scene(monkeypatch)
    build = _bar(ui, ctx, state, tab)
    controls = _laid_out(ui, build)
    _click(ui, build, _find(controls, "Sym"))
    controls = _laid_out(ui, build)
    _click(ui, build, _find(controls, "Radial"))
    assert "radial" in brush.axes_of(state.symmetry)


def test_the_mirrors_come_back_as_words_in_the_popover(ui, monkeypatch):
    """What collapses out of the row comes back with its full label, which is
    ``toolbar``'s own doctrine -- and ``\\`` is an honest character but still
    not a word."""
    ctx, state, tab, _written = _scene(monkeypatch)
    build = _bar(ui, ctx, state, tab)
    _click(ui, build, _find(_laid_out(ui, build), "Sym"))
    controls = _laid_out(ui, build)
    for word in inker_context._MIRROR_WORDS.values():
        _find(controls, word)


def test_reset_is_reachable_and_says_why_when_nothing_is_set(ui, monkeypatch):
    ctx, state, tab, _written = _scene(monkeypatch)
    build = _bar(ui, ctx, state, tab)
    _click(ui, build, _find(_laid_out(ui, build), "Sym"))
    reset = _find(_laid_out(ui, build), "Reset")
    assert reset.enabled is False
    assert reset.reason
    state.symmetry = "x+y"
    state.radial_count = 9
    controls = _laid_out(ui, build)
    _click(ui, build, _find(controls, "Reset"))
    assert state.symmetry == "none"
    assert state.radial_count == brush.DEFAULT_RADIAL


def test_the_view_aids_are_one_press_from_the_canvas(ui, monkeypatch):
    """Seven aids that were split between three unlabelled toolbox glyphs and
    the View menu, now behind one named door beside the drawing."""
    ctx, state, tab, _written = _scene(monkeypatch)
    build = _bar(ui, ctx, state, tab)
    controls = _laid_out(ui, build)
    _click(ui, build, _find(controls, "View"))
    controls = _laid_out(ui, build)
    for label in (
        "Grid",
        "Snap to grid",
        "Grid size",
        "Rulers",
        "Pixel grid",
        "Layer edges",
        "Tile numbers",
        "Tiled: off",
        "Tiled: both ways",
    ):
        _find(controls, label)


def test_pressing_the_grid_row_turns_the_grid_on_and_writes_it_down(ui, monkeypatch):
    """Asserted through the op, so this row and the View-menu row cannot
    disagree about what the grid is."""
    ctx, state, tab, written = _scene(monkeypatch)
    build = _bar(ui, ctx, state, tab)
    _click(ui, build, _find(_laid_out(ui, build), "View"))
    assert state.grid is False
    controls = _laid_out(ui, build)
    _click(ui, build, _find(controls, "Grid##"))
    assert state.grid is True
    assert "inker" in written


def test_pressing_a_tiled_row_sets_the_documents_tiling(ui, monkeypatch):
    """Pressed rather than inspected, which is this file's whole rule -- and it
    also pins the scope: tiling is a property of *this document*, so the row
    writes ``tab.tiled`` and not a session preference."""
    ctx, state, tab, _written = _scene(monkeypatch)
    build = _bar(ui, ctx, state, tab)
    _click(ui, build, _find(_laid_out(ui, build), "View"))
    controls = _laid_out(ui, build)
    _click(ui, build, _find(controls, "Tiled: both ways"))
    assert tab.tiled == "both"
    controls = _laid_out(ui, build)
    _click(ui, build, _find(controls, "Tiled: off"))
    assert tab.tiled == "off"


def test_rolling_the_seam_is_refused_with_a_sentence_when_untiled(ui, monkeypatch):
    """The op's own refusal reaches the popover rather than being written a
    second time here."""
    ctx, state, tab, _written = _scene(monkeypatch)
    build = _bar(ui, ctx, state, tab)
    _click(ui, build, _find(_laid_out(ui, build), "View"))
    row = _find(_laid_out(ui, build), "Roll the seam")
    assert row.enabled is False
    assert row.reason
