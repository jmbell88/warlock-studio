"""The text stamp, and the floating buffer it is delivered through (C14).

**The vendored Inter face is the fixture and a system font never is.** Every
assertion below about a shape -- how tall two lines are, that the crop is tight,
that the monochrome rasteriser produces no partial coverage -- is an assertion
about a specific set of outlines, so a test that reached into
``C:/Windows/Fonts`` would be a test that passes or fails depending on which
machine ran it, and would not run at all off Windows. The one thing that is
*not* asserted here is the system scan itself: it is a directory listing, and
what matters about it is only that the vendored face comes first, which
``inker_mode.font_choices`` states and this file does not need a real
``C:/Windows`` to see.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from warlock.studio import fonts, inker_mode, inker_state
from warlock.studio.inker.document import Document
from warlock.studio.inker.textstamp import MAX_SIZE, MIN_SIZE, text_stamp
from warlock.studio.panes import inker_canvas

FONT = str(fonts.FONT_DIR / "Inter-Regular.ttf")
RED = (255, 0, 0, 255)


def test_the_vendored_face_is_actually_there():
    """The guard every test in this file needs and none of them can make: a
    missing font is a ``None`` from every call below, and a file of tests
    asserting that ``None`` is returned would pass while checking nothing."""
    assert (fonts.FONT_DIR / "Inter-Regular.ttf").is_file()


# --- what comes out ---------------------------------------------------------


def test_a_stamp_is_rgba_pixels_in_the_colour_asked_for():
    out = text_stamp("Hi", FONT, 24, RED)
    assert out is not None
    assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 4
    # The colour is written across the whole array, transparent margin
    # included: straight-alpha compositing reads RGB wherever alpha is not
    # zero, and a black fringe under a red word is exactly the halo a
    # colour-only-under-the-ink stamp grows the first time it is scaled.
    assert np.all(out[..., 0] == 255)
    assert np.all(out[..., 1] == 0)
    assert np.all(out[..., 2] == 0)
    assert out[..., 3].max() == 255


def test_without_antialiasing_every_pixel_is_all_there_or_not_there():
    """The promise the pixel nibs make, kept by the text tool: an indexed
    document gains no colours from a stamp."""
    out = text_stamp("Hamburgefonstiv", FONT, 32, RED, antialias=False)
    assert out is not None
    assert set(np.unique(out[..., 3]).tolist()) == {0, 255}


def test_with_antialiasing_the_edges_are_partially_covered():
    """The other half of the same statement -- a rasteriser stuck in one mode
    would pass the test above by accident."""
    out = text_stamp("Hamburgefonstiv", FONT, 32, RED, antialias=True)
    assert out is not None
    alpha = out[..., 3]
    assert np.any((alpha > 0) & (alpha < 255))


def test_the_colours_alpha_scales_the_coverage():
    solid = text_stamp("O", FONT, 32, (0, 0, 255, 255), antialias=False)
    half = text_stamp("O", FONT, 32, (0, 0, 255, 128), antialias=False)
    assert solid is not None and half is not None
    assert solid.shape == half.shape
    assert set(np.unique(half[..., 3]).tolist()) == {0, 128}


def test_the_stamp_is_cropped_to_its_ink():
    """No transparent margin: the user positions what they can see, and a
    margin whose width depends on the font is a margin they have to guess at."""
    out = text_stamp("gjpq", FONT, 28, RED)
    assert out is not None
    alpha = out[..., 3]
    assert alpha[0].any() and alpha[-1].any()
    assert alpha[:, 0].any() and alpha[:, -1].any()


def test_a_descender_survives_the_crop():
    """The +1px slack is what this is really about: a hinted outline can reach
    past the measured box, and without the padding the bottom row of a *g* is
    rendered off the surface and silently lost."""
    flat = text_stamp("xxx", FONT, 28, RED)
    tailed = text_stamp("xgx", FONT, 28, RED)
    assert flat is not None and tailed is not None
    assert tailed.shape[0] > flat.shape[0]


def test_two_lines_are_about_twice_as_tall_as_one():
    one = text_stamp("Ay", FONT, 20, RED)
    two = text_stamp("Ay\nAy", FONT, 20, RED)
    assert one is not None and two is not None
    assert 1.8 * one.shape[0] < two.shape[0] < 2.6 * one.shape[0]
    # And no wider: the second line is the same word.
    assert two.shape[1] == one.shape[1]


def test_carriage_returns_are_newlines():
    """imgui gives "\\n"; a paste from a Windows text editor does not."""
    unix = text_stamp("a\nb", FONT, 20, RED)
    windows = text_stamp("a\r\nb", FONT, 20, RED)
    assert unix is not None and windows is not None
    assert np.array_equal(unix, windows)


# --- when it declines -------------------------------------------------------


@pytest.mark.parametrize(
    "text, font, size",
    [
        ("Hi", "no-such-font.ttf", 24),  # missing
        ("Hi", __file__, 24),  # present, not a font
        ("", FONT, 24),  # nothing typed
        ("   \n\n  ", FONT, 24),  # nothing but whitespace
        ("Hi", FONT, MIN_SIZE - 1),  # too small to be legible
        ("Hi", FONT, MAX_SIZE + 1),  # a mistyped size, not a request
    ],
)
def test_it_answers_none_rather_than_raising(text, font, size):
    """One answer for every way this can decline, because they are one thing
    from the caller's side: there is nothing to float, and the pane says so
    once. A raise would put a broken font file the user picked out of a system
    directory on the frame thread's exception path."""
    assert text_stamp(text, font, size, RED) is None


# --- delivery through the floating buffer -----------------------------------


def _stamped(doc: Document, at=(4, 6)):
    out = text_stamp("Hi", FONT, 16, RED, antialias=False)
    assert out is not None
    return out, doc.float_pixels(out, at)


def test_a_stamp_floats_commits_and_undoes_as_one_step():
    """The whole reason there are no text objects: placement, the move, the
    commit and the undo are the paste's, already written and already tested."""
    doc = Document.blank(64, 48)
    before = doc.stack.active.pixels.copy()
    pixels, floated = _stamped(doc)
    assert floated
    assert doc.floating is not None and doc.floating.offset == (4, 6)
    # Floating writes nothing: it is a buffer over the layer, not on it.
    assert np.array_equal(doc.stack.active.pixels, before)

    assert doc.commit_floating()
    assert doc.floating is None
    assert not np.array_equal(doc.stack.active.pixels, before)
    # And it landed where it was put.
    ys, xs = np.nonzero(pixels[..., 3])
    assert np.array_equal(doc.stack.active.pixels[6 + ys[0], 4 + xs[0]], np.array(RED, np.uint8))

    assert doc.history.can_undo
    doc.undo()
    assert np.array_equal(doc.stack.active.pixels, before)


def test_a_float_can_be_moved_before_it_lands():
    doc = Document.blank(64, 48)
    _stamped(doc)
    doc.move_floating(3, -2)
    assert doc.floating is not None and doc.floating.offset == (7, 4)


def test_the_buffer_is_grabbable_anywhere_in_its_box():
    """The mask is solid rather than the glyphs' own alpha, which is what
    ``put_clipboard`` does and for the same reason: the mask is what hit-tests
    a grab, and one cut to the outlines would mean clicking a letter's stem to
    move the word you just placed."""
    doc = Document.blank(64, 48)
    pixels, _ = _stamped(doc)
    height, width = pixels.shape[:2]
    assert doc.floating is not None
    assert doc.floating.contains((4, 6))
    assert doc.floating.contains((4 + width - 1, 6 + height - 1))


def test_floating_pixels_forget_the_redo_branch():
    """The paste rule. A float pushes no step of its own, so without the
    ``forget_redo`` a Ctrl+Y after a stamp replays an unrelated edit."""
    doc = Document.blank(64, 48)
    doc.select_all()
    doc.undo()
    assert doc.history.can_redo
    _stamped(doc)
    assert not doc.history.can_redo


def test_a_locked_layer_refuses_the_float():
    """The door is here rather than at ``commit_floating``, which is
    deliberately not refused: the buffer is bound to the active layer and would
    land on it."""
    doc = Document.blank(64, 48)
    doc.stack.active.locked = True
    out = text_stamp("Hi", FONT, 16, RED)
    assert out is not None
    assert not doc.float_pixels(out, (0, 0))
    assert doc.floating is None


def test_a_second_stamp_puts_the_first_one_down():
    """"Stamp again" means "put that one down", not "throw it away" -- the
    paste's behaviour, and the reason ``commit_floating`` is called rather than
    the buffer being overwritten."""
    doc = Document.blank(64, 48)
    _stamped(doc, (2, 2))
    empty = doc.stack.active.pixels.copy()
    _stamped(doc, (20, 20))
    assert doc.floating is not None and doc.floating.offset == (20, 20)
    assert not np.array_equal(doc.stack.active.pixels, empty), "the first one landed"


def test_a_stamp_is_refused_rather_than_reshaped():
    """``float_pixels`` is the delivery route for anything that *makes* pixels
    (image brushes next), so it checks what it is handed rather than trusting
    the one caller that exists today."""
    doc = Document.blank(16, 16)
    assert not doc.float_pixels(np.zeros((4, 4), np.uint8), (0, 0))
    assert not doc.float_pixels(np.zeros((4, 4, 3), np.uint8), (0, 0))
    assert not doc.float_pixels(np.zeros((0, 4, 4), np.uint8), (0, 0))
    assert doc.floating is None


# --- the fonts offered ------------------------------------------------------


def test_the_vendored_face_leads_the_list_and_is_the_default():
    """So the tool behaves identically on a machine with no fonts installed,
    and so the documented default is a file that ships in the wheel rather than
    whatever a directory listing happens to sort first."""
    choices = inker_mode.font_choices(refresh=True)
    assert choices, "the vendored face at least"
    assert choices[0][0].endswith("Inter-Regular.ttf")
    state = inker_state.InkerState()
    state.tool = "text"
    assert state.font == "", "the stored default names no file"
    assert inker_mode.font_path(state) == choices[0][0]
    state.font = "C:/somewhere/Else.ttf"
    assert inker_mode.font_path(state) == "C:/somewhere/Else.ttf"


# --- the canvas arm ---------------------------------------------------------
#
# ``_press`` on the text tool is a dispatch decision like the slice tool's:
# where the click landed, whether the popup opens at all, and -- the one that
# would look like a working editor -- that it leaves no dab and no drag behind.


@pytest.fixture
def pressed(monkeypatch):
    """``_press`` with imgui's two calls stubbed, and a toast recorder."""
    opened: list[str] = []
    monkeypatch.setattr(
        inker_canvas,
        "imgui",
        SimpleNamespace(
            get_io=lambda: SimpleNamespace(key_shift=False, key_alt=False),
            open_popup=opened.append,
        ),
    )
    toasts: list[tuple[str, str]] = []
    ctx = SimpleNamespace(toast=lambda text, kind="info": toasts.append((text, kind)))
    state = inker_state.InkerState(fg=RED)
    state.tool = "text"
    tab = SimpleNamespace(doc=Document.blank(32, 32), tiled="off", busy=False)

    def press(point, *, button: int = 0):
        state.drag_button = button
        inker_canvas._press(ctx, state, tab, point)

    return SimpleNamespace(
        state=state, tab=tab, ctx=ctx, press=press, opened=opened, toasts=toasts
    )


def test_a_click_records_the_spot_and_opens_the_box(pressed):
    pressed.press((7.0, 9.0))
    assert pressed.state.text_at == (7, 9)
    assert pressed.opened == [inker_canvas.TEXT_POPUP]
    # No dab, and no gesture left open: the arm returns before every paint
    # branch, which is the failure the slice tool found first.
    assert pressed.state.drag_kind == ""
    assert not pressed.tab.doc.stack.active.pixels.any()


def test_the_right_button_is_inert(pressed):
    """Reserved rather than spent, like the slice tool's."""
    pressed.press((7.0, 9.0), button=1)
    assert pressed.opened == []


def test_a_locked_layer_is_refused_before_the_box_opens(pressed):
    """Offering a font list and a size and only *then* saying the layer is
    locked is a form the app knew the answer to before it drew it."""
    pressed.tab.doc.stack.active.locked = True
    pressed.press((7.0, 9.0))
    assert pressed.opened == []
    assert pressed.toasts and "locked" in pressed.toasts[0][0]


def _indexed(tab) -> None:
    tab.doc.palette = [(0, 0, 0, 255), (255, 255, 255, 255)]


def test_an_indexed_document_starts_with_antialiasing_off(pressed):
    """A palette promises the file holds exactly those colours, and an
    antialiased rim is a row of blends that each snap to the nearest slot."""
    _indexed(pressed.tab)
    pressed.press((4.0, 4.0))
    assert pressed.state.aa is False


def test_every_other_document_starts_with_it_on(pressed):
    """The other half of the same statement -- a default stuck off would pass
    the test above by accident, and antialiased is what a painted reference
    wants."""
    pressed.press((4.0, 4.0))
    assert pressed.state.aa is True


def test_the_indexed_default_is_not_spent_by_the_first_popup(pressed, monkeypatch):
    """**The regression this pair exists for.** The rule was once "the text
    tool has no stored options entry", and ``options_for`` materialises every
    key of a tool the first time any one of them is read -- so one stamp on an
    RGB document made every indexed document for the rest of the session open
    with antialiasing on, while the manual promises it starts off with no
    session in the sentence.
    """
    pressed.press((4.0, 4.0))
    _driven(monkeypatch, pressed, _Popup())  # a full popup: reads all three
    assert pressed.state.aa is True
    _indexed(pressed.tab)
    pressed.press((4.0, 4.0))
    assert pressed.state.aa is False


def test_setting_the_box_yourself_stops_the_default_deciding(pressed, monkeypatch):
    """It is a starting point, not a rule: an indexed document with a
    soft-edged palette is a real thing, and a checkbox that unticks itself on
    the next click is a control the user cannot operate."""
    _indexed(pressed.tab)
    _driven(monkeypatch, pressed, _Popup(tick=True))
    assert pressed.state.aa is True and pressed.state.text_aa_touched
    pressed.press((4.0, 4.0))
    assert pressed.state.aa is True


def test_it_stops_deciding_in_both_directions(pressed, monkeypatch):
    """Unticking on an RGB document is as much an opinion as ticking on an
    indexed one, and the flag does not know which way round it was set."""
    _driven(monkeypatch, pressed, _Popup(tick=False))
    assert pressed.state.aa is False and pressed.state.text_aa_touched
    pressed.press((4.0, 4.0))
    assert pressed.state.aa is False


def test_resetting_the_tool_hands_the_default_back(pressed, monkeypatch):
    """``text_aa_touched`` is part of the tool's stored settings and only lives
    outside the dictionary because ``options_for`` cannot represent "unset", so
    a Reset that left it standing would not reset."""
    _driven(monkeypatch, pressed, _Popup(tick=True))
    _indexed(pressed.tab)
    pressed.state.reset_tool_options("text")
    assert pressed.state.text_aa_touched is False
    pressed.press((4.0, 4.0))
    assert pressed.state.aa is False


# --- OK ---------------------------------------------------------------------


def test_ok_floats_the_stamp_and_puts_the_move_tool_in_your_hand(pressed):
    """The Ctrl+V precedent: a stamp arrives floating, and the first thing
    anybody does with a word they have just placed is drag it into position."""
    state, tab = pressed.state, pressed.tab
    state.text_buffer = "Hi"
    state.text_at = (5, 6)
    assert inker_mode.stamp_text(pressed.ctx, state, tab)
    assert tab.doc.floating is not None and tab.doc.floating.offset == (5, 6)
    assert state.tool == "move"
    assert not pressed.toasts
    # In the foreground colour, which is what the popup says and the only
    # colour it offers.
    assert (tab.doc.floating.pixels[..., :3] == np.array(RED[:3], np.uint8)).all()


class _Popup:
    """imgui and ``widgets``, as much of each as ``_text_popup`` calls.

    The body is imgui code, which is exactly the kind that is only ever run on
    the frame thread and takes the process down when it is wrong -- a widget
    that does not exist, a ``begin`` without its ``end``. Driving it against a
    recorder is the cheapest way to find that in a test rather than in a
    screenshot.
    """

    def __init__(
        self, *, ok: bool = False, cancel: bool = False, tick: bool | None = None
    ) -> None:
        self.ok, self.cancel, self.tick = ok, cancel, tick
        self.calls: list[str] = []

    # -- imgui
    def begin_popup(self, name):
        self.calls.append("begin")
        return True

    def end_popup(self):
        self.calls.append("end")

    def close_current_popup(self):
        self.calls.append("close")

    def button(self, label, size=None):
        self.calls.append(f"button:{label}")
        return self.ok if label.startswith("OK") else self.cancel

    def checkbox(self, label, value):
        """``tick`` is the user setting the box: a *change*, to that value."""
        if self.tick is None:
            return False, value
        return True, self.tick

    def color_button(self, label, colour, flags=0, size=None):
        return False

    def same_line(self):
        pass

    def dummy(self, size):
        pass

    def ImVec4(self, *parts):  # noqa: N802 - imgui's own spelling
        return parts

    # -- widgets
    def multiline(self, label, value, height, cap):
        return value

    def combo(self, label, value, options, width=-1.0):
        return value

    def labeled_slider_int(self, label, value, low, high):
        assert low <= value <= high, "the slider cannot reach the stored size"
        return False, value

    def muted(self, text):
        pass

    def help_marker(self, text):
        pass

    def popup_chrome(self, **_kwargs):
        """Renderer-only chrome is a no-op in this interaction recorder."""
        pass


def _driven(monkeypatch, pressed, popup):
    """One frame of the popup body, with the recorder in force for that frame
    alone -- a test that draws a popup and then presses again needs the
    fixture's own imgui stub back afterwards."""
    with monkeypatch.context() as patched:
        patched.setattr(inker_canvas, "imgui", popup)
        patched.setattr(inker_canvas, "widgets", popup)
        inker_canvas._text_popup(pressed.ctx, pressed.state, pressed.tab)


def test_the_popup_body_draws_and_closes_its_own_scope(monkeypatch, pressed):
    popup = _Popup()
    _driven(monkeypatch, pressed, popup)
    assert popup.calls[0] == "begin" and popup.calls[-1] == "end"
    assert "close" not in popup.calls, "an untouched popup stays up"


def test_ok_in_the_popup_stamps_and_cancel_does_not(monkeypatch, pressed):
    pressed.state.text_buffer = "Hi"
    _driven(monkeypatch, pressed, _Popup(ok=True))
    assert pressed.tab.doc.floating is not None

    pressed.tab.doc.cancel_floating()
    _driven(monkeypatch, pressed, _Popup(cancel=True))
    assert pressed.tab.doc.floating is None


def test_an_empty_box_toasts_instead_of_floating_nothing(pressed):
    state, tab = pressed.state, pressed.tab
    state.text_buffer = "   "
    assert not inker_mode.stamp_text(pressed.ctx, state, tab)
    assert tab.doc.floating is None
    assert state.tool == "text", "and the tool stays put"
    assert pressed.toasts and pressed.toasts[0][1] == "warn"
