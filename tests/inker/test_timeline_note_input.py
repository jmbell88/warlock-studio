"""The timeline's Properties block, driven by a real mouse click.

``test_timeline_cel_opacity_input``'s file one wave later, and for its reason:
a control that is drawn and wired to nothing is this codebase's most common
historical defect, and it is invisible to every test that calls the document
directly. So nothing here calls ``set_cel_note``: the cel menu is opened inside
a real imgui frame, the swatch is found through :mod:`.probe` -- the same census
``scripts/exercise_mode`` uses, so the rect is *read* and never computed -- and
the mouse is pressed and released inside it.

Two findings from the two waves before this one are load-bearing here. A popup
reports ``visible=False`` on its first frame, so the layout runs twice before a
rect is read (Wave 10). And a **button fires on release**, not on press, so a
press-only frame does nothing at all (Wave 11's checkbox, one widget along) --
which is why :func:`_click` is two frames and the opacity file's single
``down=True`` frame would have quietly asserted nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.studio import inker_state, probe
from warlock.studio.inker.animation import Note


@pytest.fixture
def ui(monkeypatch):
    """``test_timeline_cel_opacity_input``'s fixture verbatim: an imgui context
    with the control census on, no renderer backend, torn down after."""
    from imgui_bundle import imgui

    from warlock.studio import theme

    previous = imgui.get_current_context()
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600.0, 950.0)
    io.delta_time = 1.0 / 60.0
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)
    monkeypatch.setattr(probe, "ENABLED", True)
    yield imgui
    imgui.destroy_context(ctx)
    if previous is not None:
        imgui.set_current_context(previous)


class _Prompts:
    """``dialogs.PromptQueue``'s surface, as much of it as the pane touches."""

    def __init__(self) -> None:
        self.asked: list = []

    def ask(self, prompt) -> None:
        self.asked.append(prompt)


def _doc():
    from warlock.studio.inker.document import Document

    doc = Document.blank(4, 4)
    doc.stack[0].name = "Art"
    doc.stack[0].pixels[:, :] = (255, 0, 0, 255)
    doc.invalidate_all()
    doc.ensure_animation()
    doc.add_frame(link=True)
    doc.add_tag("walk", 0, 1)
    doc.set_current_frame(0)
    return doc


def _tab(doc):
    return SimpleNamespace(doc=doc, busy=False, range_sel=None)


def _ctx():
    state = SimpleNamespace(inker=inker_state.InkerState())
    return SimpleNamespace(
        state=state, toast=lambda *a, **k: None, viewer=None, prompts=_Prompts()
    )


def _frame(imgui, build, *, pos=(400.0, 300.0), down=False):
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


def _laid_out(imgui, build, frames=2):
    """Run ``build`` until the popup has a size, and hand back the census."""
    out = []
    for _ in range(frames):
        out = _frame(imgui, build)
    return out


def _click(imgui, build, control):
    """Press *and release* inside a control's own rect.

    Two frames, because imgui's button and selectable both answer on the
    release. A single ``down=True`` frame leaves the widget merely held, which
    is exactly the state a test can mistake for "the control did nothing".
    """
    x, y, w, h = control.hit
    at = (x + w * 0.5, y + h * 0.5)
    _frame(imgui, build, pos=at, down=True)
    _frame(imgui, build, pos=at, down=False)


def _cell_menu(imgui, ctx, tab, ti=0, fi=0, has_cel=True, linked=True):
    from warlock.studio.panes import inker_timeline

    def build():
        if not imgui.is_popup_open("celmenu"):
            imgui.open_popup("celmenu")
        inker_timeline._cell_menu(ctx, tab, ti, fi, has_cel, linked)

    return build


def _row_menu(imgui, ctx, tab, index=0):
    from warlock.studio.panes import inker_timeline

    def build():
        if not imgui.is_popup_open("layer-menu"):
            imgui.open_popup("layer-menu")
        inker_timeline._row_menu(ctx, tab, tab.doc, index)

    return build


def _tag_menu(imgui, ctx, tab, index=0):
    from warlock.studio.panes import inker_timeline

    def build():
        if not imgui.is_popup_open("tagmenu"):
            imgui.open_popup("tagmenu")
        inker_timeline._tag_menu(ctx, tab, index, tab.doc.anim.tags[index])

    return build


def _named(controls, label):
    """By exact label, or by the ``##`` id where the visible half is an icon.

    The census records the label imgui was handed, glyph and all, so the clear
    swatch is ``"##notecolour-cel-none"`` rather than the id alone --
    matching on the suffix is what keeps a test naming the control instead of
    naming a codepoint out of ``icons.py``.
    """
    found = [c for c in controls if c.label == label or c.label.endswith(label)]
    assert found, sorted({c.label for c in controls})
    return found[0]


# --- the swatch reaches the document ----------------------------------------


def test_pressing_a_cel_swatch_colours_that_slot(ui):
    """The whole point of the file: the click reaches the grid."""
    from warlock.studio.panes.inker_timeline import NOTE_COLOURS

    doc = _doc()
    ctx, tab = _ctx(), _tab(doc)
    build = _cell_menu(ui, ctx, tab)
    swatch = _named(_laid_out(ui, build), "##notecolour-cel-blue")
    assert swatch.enabled and swatch.visible

    assert doc.anim.cel_notes == {}
    _click(ui, build, swatch)

    key = (doc.anim.tracks[0].uid, doc.anim.frames[0].uid)
    assert key in doc.anim.cel_notes, "the swatch drew but changed nothing"
    assert doc.anim.cel_note(*key).colour == dict(NOTE_COLOURS)["blue"]
    # And it went through the ordinary door, so it is one undoable step.
    assert doc.history.can_undo
    doc.undo()
    assert doc.anim.cel_notes == {}


def test_the_click_colours_only_the_slot_it_landed_on(ui):
    """A linked cel is one object in two slots; the click moves one of them."""
    doc = _doc()
    track = doc.anim.tracks[0].uid
    first, second = (frame.uid for frame in doc.anim.frames)
    assert doc.anim.cels[(track, first)] is doc.anim.cels[(track, second)]

    ctx, tab = _ctx(), _tab(doc)
    build = _cell_menu(ui, ctx, tab, fi=1)
    _click(ui, build, _named(_laid_out(ui, build), "##notecolour-cel-red"))

    assert doc.anim.cel_note(track, second).colour is not None
    assert doc.anim.cel_note(track, first).colour is None
    assert doc.anim.cels[(track, first)] is doc.anim.cels[(track, second)]


def test_the_clear_swatch_takes_a_colour_back_off(ui):
    """Every other swatch *sets*; without this one the first press could not be
    reversed except through undo."""
    doc = _doc()
    doc.set_cel_note(Note("kept", (1, 2, 3, 255)), 0, 0)
    ctx, tab = _ctx(), _tab(doc)
    build = _cell_menu(ui, ctx, tab)
    _click(ui, build, _named(_laid_out(ui, build), "##notecolour-cel-none"))

    key = (doc.anim.tracks[0].uid, doc.anim.frames[0].uid)
    assert doc.anim.cel_note(*key) == Note("kept")


def test_pressing_a_row_swatch_colours_the_track(ui):
    doc = _doc()
    ctx, tab = _ctx(), _tab(doc)
    build = _row_menu(ui, ctx, tab)
    _click(ui, build, _named(_laid_out(ui, build), "##notecolour-track-green"))
    assert doc.anim.tracks[0].note.colour is not None


def test_pressing_a_tag_swatch_colours_the_tag(ui):
    doc = _doc()
    ctx, tab = _ctx(), _tab(doc)
    build = _tag_menu(ui, ctx, tab)
    _click(ui, build, _named(_laid_out(ui, build), "##notecolour-tag-purple"))
    assert doc.anim.tags[0].note.colour is not None


# --- the text half ----------------------------------------------------------


def test_properties_asks_the_prompt_and_the_answer_reaches_the_document(ui):
    """The text entry is the app's own one-line prompt, so what this drives is
    the *menu item*: the click has to queue a question whose answer lands on
    the slot the menu was opened over."""
    doc = _doc()
    ctx, tab = _ctx(), _tab(doc)
    build = _cell_menu(ui, ctx, tab)
    _click(ui, build, _named(_laid_out(ui, build), "Properties...##cel"))

    assert len(ctx.prompts.asked) == 1, "the item drew but asked nothing"
    prompt = ctx.prompts.asked[0]
    assert prompt.label == "User data"
    prompt.on_accept("the anticipation frame")

    key = (doc.anim.tracks[0].uid, doc.anim.frames[0].uid)
    assert doc.anim.cel_note(*key).text == "the anticipation frame"
    assert doc.history.can_undo


def test_the_prompt_opens_showing_the_text_the_slot_already_carries(ui):
    doc = _doc()
    doc.set_cel_note(Note("already here"), 0, 0)
    ctx, tab = _ctx(), _tab(doc)
    build = _cell_menu(ui, ctx, tab)
    _click(ui, build, _named(_laid_out(ui, build), "Properties...##cel"))
    assert ctx.prompts.asked[0].value == "already here"


def test_setting_the_text_keeps_the_colour_that_was_already_there(ui):
    """The block hands a whole note to the door, so "change one half" has to be
    expressed by carrying the other -- which is the thing a keyword-and-sentinel
    signature would have got wrong."""
    doc = _doc()
    doc.set_cel_note(Note(colour=(9, 9, 9, 255)), 0, 0)
    ctx, tab = _ctx(), _tab(doc)
    build = _cell_menu(ui, ctx, tab)
    _click(ui, build, _named(_laid_out(ui, build), "Properties...##cel"))
    ctx.prompts.asked[0].on_accept("with a colour")

    key = (doc.anim.tracks[0].uid, doc.anim.frames[0].uid)
    assert doc.anim.cel_note(*key) == Note("with a colour", (9, 9, 9, 255))


# --- what is not offered ----------------------------------------------------


def test_an_empty_slot_offers_no_properties_block(ui):
    doc = _doc()
    ctx, tab = _ctx(), _tab(doc)
    controls = _laid_out(ui, _cell_menu(ui, ctx, tab, has_cel=False, linked=False))
    assert not [c for c in controls if c.label.startswith("Properties...##cel")]
    assert not [c for c in controls if c.label.startswith("##notecolour-cel")]


def test_a_still_documents_row_menu_offers_no_properties_block(ui):
    """A note lives on a ``Track`` and a still document has none, so the block
    is hidden rather than greyed -- there is nothing for it to promise."""
    from warlock.studio.inker.document import Document

    doc = Document.blank(4, 4)
    ctx, tab = _ctx(), _tab(doc)
    controls = _laid_out(ui, _row_menu(ui, ctx, tab))
    assert not [c for c in controls if c.label.startswith("##notecolour-track")]
    # The layer dialog's own entry is still there, and is named for what it is.
    assert _named(controls, "Layer properties...")


def test_the_row_menu_has_exactly_one_item_called_properties(ui):
    """Two items called Properties in one menu is a menu that answers neither
    question, which is why the blend/opacity/lock dialog was renamed."""
    doc = _doc()
    ctx, tab = _ctx(), _tab(doc)
    labels = [c.label for c in _laid_out(ui, _row_menu(ui, ctx, tab))]
    assert labels.count("Properties...##track") == 1
    assert "Properties..." not in labels


def test_a_busy_document_takes_no_note_edit(ui):
    """The block is inside the menu's ``begin_disabled(tab.busy)``, and what
    that is worth is asserted as behaviour rather than as census state: the
    census records each control's *own* ``enabled`` argument, not imgui's
    ambient disabled stack, so a press is the only thing that can tell the
    difference."""
    doc = _doc()
    ctx, tab = _ctx(), _tab(doc)
    tab.busy = True
    build = _cell_menu(ui, ctx, tab)
    _click(ui, build, _named(_laid_out(ui, build), "##notecolour-cel-blue"))
    assert doc.anim.cel_notes == {}
    assert not ctx.prompts.asked
