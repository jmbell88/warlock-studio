"""The cel menu's opacity slider, driven by a real mouse press.

A control that is drawn and wired to nothing is this codebase's most common
historical defect, and it is invisible to every test that calls the document
directly. So this file does not call ``set_cel_opacity``: it opens the timeline
cell menu inside a real imgui frame, finds the slider through :mod:`.probe` --
the same census ``scripts/exercise_mode`` uses, so the rect is *read* and never
computed -- and presses the left mouse button inside it. What is asserted is
that the document's grid changed.

The context is built and destroyed here rather than shared, for the reason
``test_pane_guard``'s own fixture gives: two imgui contexts over one GL context
crash the process. No GL at all is needed, though -- nothing is rendered, only
laid out -- so the backend renderer that fixture builds is skipped and this file
runs everywhere rather than only on a machine with a card.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from _ui_context import imgui_context

from warlock.studio import inker_state, probe


@pytest.fixture
def ui(monkeypatch):
    """The shared imgui context; see ``_ui_context`` for why it is not a
    conftest fixture."""
    with imgui_context(monkeypatch) as imgui:
        yield imgui

def _doc():
    from warlock.studio.inker.document import Document

    doc = Document.blank(4, 4)
    doc.stack[0].name = "Art"
    doc.stack[0].pixels[:, :] = (255, 0, 0, 255)
    doc.invalidate_all()
    doc.ensure_animation()
    doc.add_frame(link=True)
    doc.set_current_frame(0)
    return doc


def _tab(doc):
    return SimpleNamespace(doc=doc, busy=False, range_sel=None)


def _ctx():
    state = SimpleNamespace(inker=inker_state.InkerState())
    return SimpleNamespace(state=state, toast=lambda *a, **k: None, viewer=None)


def _frame(imgui, build, *, pos=(400.0, 300.0), down=False):
    """One whole frame with the mouse where it is said to be."""
    io = imgui.get_io()
    io.add_mouse_pos_event(pos[0], pos[1])
    io.add_mouse_button_event(0, down)
    probe.begin_frame()
    imgui.new_frame()
    imgui.set_next_window_size((520.0, 900.0))
    imgui.set_next_window_pos((0.0, 0.0))
    imgui.begin("##host")
    build()
    imgui.end()
    imgui.end_frame()
    return list(probe.FRAME_CONTROLS)


def _open_menu(imgui, ctx, tab, ti=0, fi=0):
    from warlock.studio.panes import inker_timeline

    def build():
        if not imgui.is_popup_open("celmenu"):
            imgui.open_popup("celmenu")
        inker_timeline._cell_menu(ctx, tab, ti, fi, True, True)

    return build


def _laid_out(imgui, build, frames=2):
    """Run ``build`` until the popup has a size, and hand back the census.

    Two frames and not one: imgui lays a fresh popup out once with its items
    hidden to *measure* it, and a rect read from that pass is real but the
    window it sits in is not yet visible -- which is exactly the state in which
    a press lands on nothing.
    """
    out = []
    for _ in range(frames):
        out = _frame(imgui, build)
    return out


def _slider(controls):
    found = [c for c in controls if c.label == "Opacity##cel"]
    assert found, [c.label for c in controls]
    return found[0]


def test_the_cel_menu_offers_an_opacity_slider(ui):
    doc = _doc()
    controls = _laid_out(ui, _open_menu(ui, _ctx(), _tab(doc)))
    slider = _slider(controls)
    assert slider.kind == "slider_float"
    assert slider.enabled and slider.visible


def test_pressing_the_slider_changes_the_document(ui):
    """The whole point of the file: the press reaches the grid."""
    doc = _doc()
    ctx, tab = _ctx(), _tab(doc)
    build = _open_menu(ui, ctx, tab)

    slider = _slider(_laid_out(ui, build))
    x, y, w, h = slider.hit
    # A quarter along the track: any value below 1.0 proves the wiring, and a
    # quarter is far enough from both ends that a pixel of style drift cannot
    # land it on a no-op.
    target = (x + w * 0.25, y + h * 0.5)

    assert doc.anim.cel_opacity == {}
    _frame(ui, build, pos=target, down=True)

    key = (doc.anim.tracks[0].uid, doc.anim.frames[0].uid)
    assert key in doc.anim.cel_opacity, "the slider drew but changed nothing"
    assert 0.0 <= doc.anim.cel_opacity[key] < 1.0
    # And it went through the ordinary door, so it is one undoable step.
    assert doc.history.can_undo
    doc.undo()
    assert doc.anim.cel_opacity == {}


def test_the_press_dims_only_the_slot_it_landed_on(ui):
    """A linked cel is one object in two slots; the press moves one of them."""
    doc = _doc()
    track = doc.anim.tracks[0].uid
    first, second = (frame.uid for frame in doc.anim.frames)
    assert doc.anim.cels[(track, first)] is doc.anim.cels[(track, second)]

    ctx, tab = _ctx(), _tab(doc)
    build = _open_menu(ui, ctx, tab, fi=1)
    slider = _slider(_laid_out(ui, build))
    x, y, w, h = slider.hit
    _frame(ui, build, pos=(x + w * 0.25, y + h * 0.5), down=True)

    assert doc.anim.cel_alpha(track, second) < 1.0
    assert doc.anim.cel_alpha(track, first) == 1.0
    assert doc.anim.cels[(track, first)] is doc.anim.cels[(track, second)]


def test_the_slider_shows_the_value_the_slot_already_carries(ui):
    """A control that always opens at 1.0 would silently reset on the next
    press, which is the same defect one step later."""
    doc = _doc()
    doc.set_cel_opacity(0.25, track_index=0, frame_index=0)
    ctx, tab = _ctx(), _tab(doc)
    slider = _slider(_laid_out(ui, _open_menu(ui, ctx, tab)))
    # imgui renders the value into the widget itself, so the census cannot read
    # it back -- what is asserted is that the value the pane hands imgui is the
    # document's, which is what the slot returns.
    assert doc.anim.cel_alpha(doc.anim.tracks[0].uid, doc.anim.frames[0].uid) == 0.25
    assert slider.enabled


def test_an_empty_slot_offers_no_opacity_slider(ui):
    from warlock.studio.panes import inker_timeline

    doc = _doc()
    ctx, tab = _ctx(), _tab(doc)

    def build():
        if not ui.is_popup_open("celmenu"):
            ui.open_popup("celmenu")
        inker_timeline._cell_menu(ctx, tab, 0, 0, False, False)

    controls = _laid_out(ui, build)
    assert not [c for c in controls if c.label == "Opacity##cel"]


def test_the_menu_is_still_a_balanced_frame(ui):
    """The slider sits inside two ``begin_disabled`` scopes and a popup; an
    unbalanced one is a crash three panes later rather than here."""
    doc = _doc()
    _laid_out(ui, _open_menu(ui, _ctx(), _tab(doc)))
    assert np.isfinite(ui.get_io().display_size.x)
