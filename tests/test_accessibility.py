"""Contrast and keyboard access, measured rather than asserted.

The audit's accessibility findings were all of the form "this looks fine and
measures badly", so the tests here measure. Every number in them is computed
from the palette that ships, which is what makes the file a gate rather than a
transcription: changing a token to something that fails fails here, and nobody
has to remember to re-run a colour picker.
"""

from __future__ import annotations

import colorsys
import re
from pathlib import Path

import pytest

from warlock.studio import tokens

SRC = Path(__file__).resolve().parents[1] / "src" / "warlock"


# -- the maths ---------------------------------------------------------------


def test_contrast_spans_the_whole_wcag_range() -> None:
    """The two anchors WCAG itself names, to catch a transposed coefficient."""
    assert tokens.contrast(0xFFFFFF, 0x000000) == pytest.approx(21.0, abs=0.01)
    assert tokens.contrast(0x7C6CF0, 0x7C6CF0) == pytest.approx(1.0, abs=0.001)


def test_contrast_is_symmetric() -> None:
    """Ratio is a property of a pair, not of which one is the ink."""
    assert tokens.contrast(0xE8E8EE, 0x0F1014) == tokens.contrast(0x0F1014, 0xE8E8EE)


def test_compositing_lands_on_the_endpoints() -> None:
    assert tokens.composite(0xFFFFFF, 0x000000, 1.0) == 0xFFFFFF
    assert tokens.composite(0xFFFFFF, 0x000000, 0.0) == 0x000000
    assert tokens.composite(0xFFFFFF, 0x000000, 0.5) == 0x808080


def test_compositing_is_what_makes_a_translucent_role_measurable() -> None:
    """The bug UX-18 named, reproduced as a number.

    MUTED *stored* clears the bar comfortably; MUTED drawn at the 0.6 alpha
    ``text_disabled`` carries does not. If these two ever agree, the composite
    step has been dropped and the whole file is measuring the wrong colour.
    """
    dark = tokens.PALETTES["dark"]
    stored = tokens.contrast(dark["MUTED"], dark["PANEL"])
    drawn = tokens.contrast(
        tokens.composite(dark["MUTED"], dark["PANEL"], 0.6), dark["PANEL"]
    )
    assert stored > tokens.CONTRAST_TEXT
    assert drawn < tokens.CONTRAST_TEXT
    assert drawn == pytest.approx(3.20, abs=0.01)


# -- the palette -------------------------------------------------------------


@pytest.mark.parametrize("palette", sorted(tokens.PALETTES))
@pytest.mark.parametrize("surface", tokens.COPY_SURFACES)
@pytest.mark.parametrize("role", ["TEXT", "MUTED"])
def test_every_copy_role_is_readable_on_every_surface(
    palette: str, surface: str, role: str
) -> None:
    """The two roles that carry sentences, against all four elevations.

    MUTED is in here because UX-18's fix was to draw it *opaque* rather than to
    replace it -- so the claim the fix rests on ("MUTED already qualifies") is
    the thing under test, in both themes, on the hovered card as well as the
    window floor.
    """
    colours = tokens.PALETTES[palette]
    ratio = tokens.contrast(colours[role], colours[surface])
    assert ratio >= tokens.CONTRAST_TEXT, (
        f"{palette}/{role} on {surface} is {ratio:.2f}:1, under "
        f"{tokens.CONTRAST_TEXT}:1"
    )


@pytest.mark.parametrize("palette", sorted(tokens.PALETTES))
@pytest.mark.parametrize("surface", tokens.COPY_SURFACES)
def test_the_accent_qualifies_as_a_control_boundary(palette: str, surface: str) -> None:
    """The focus ring is drawn in ACCENT, so it answers to SC 1.4.11's 3:1.

    Deliberately the looser bar: a ring is a boundary rather than copy. It is
    still measured on every surface, because the ring around a control on a
    hovered card is the dimmest case and the one a spot check misses.
    """
    colours = tokens.PALETTES[palette]
    ratio = tokens.contrast(colours["ACCENT"], colours[surface])
    assert ratio >= tokens.CONTRAST_UI, (
        f"{palette}/ACCENT on {surface} is {ratio:.2f}:1, under "
        f"{tokens.CONTRAST_UI}:1 -- the focus ring would be invisible"
    )


@pytest.mark.parametrize("palette", sorted(tokens.PALETTES))
def test_the_accent_is_told_apart_from_the_warning_colour(palette: str) -> None:
    """Contrast qualifies a colour against its *ground*; this is the other axis.

    Dark and light never had to answer for it -- indigo against amber separates
    itself -- but a palette whose accent is warm puts ACCENT and WARN in one
    family, and ``pixel``'s first pass landed them 2 degrees of hue apart. A
    focus ring and a warning badge were then the same colour under two names,
    which no contrast bar in this module would have caught: both cleared their
    own, against a surface, independently.

    Twelve degrees is where the pair stops reading as one hue at the sizes these
    are drawn -- a 2 px ring, a status dot -- and the saturation gap is asked
    for as well, because hue alone is the weakest of the three channels for
    anyone with a red-green deficiency. ``theme.STATUS_GLYPHS`` remains the
    reason legibility does not *rest* on this: every status says itself in a
    shape too.
    """
    colours = tokens.PALETTES[palette]

    def hue_and_saturation(value: int) -> tuple[float, float]:
        red, green, blue = (
            (value >> 16 & 0xFF) / 255,
            (value >> 8 & 0xFF) / 255,
            (value & 0xFF) / 255,
        )
        hue, saturation, _value = colorsys.rgb_to_hsv(red, green, blue)
        return hue * 360, saturation

    accent_hue, accent_sat = hue_and_saturation(colours["ACCENT"])
    warn_hue, warn_sat = hue_and_saturation(colours["WARN"])
    apart = abs(accent_hue - warn_hue)
    apart = min(apart, 360 - apart)
    assert apart >= 12.0, (
        f"{palette}: ACCENT and WARN are {apart:.1f} degrees apart -- a focus "
        f"ring and a warning read as the same colour"
    )
    if apart < 60.0:
        assert abs(accent_sat - warn_sat) >= 0.15, (
            f"{palette}: ACCENT and WARN are close in hue ({apart:.1f} degrees) "
            f"and within {abs(accent_sat - warn_sat):.2f} saturation of each other"
        )


@pytest.mark.parametrize("palette", sorted(tokens.PALETTES))
def test_status_colours_are_readable(palette: str) -> None:
    """A failure the user cannot read is a failure they cannot act on."""
    colours = tokens.PALETTES[palette]
    for role in ("OK", "ERR", "WARN"):
        ratio = tokens.contrast(colours[role], colours["PANEL"])
        assert ratio >= tokens.CONTRAST_TEXT, (
            f"{palette}/{role} on PANEL is {ratio:.2f}:1"
        )


# -- the rule -----------------------------------------------------------------

# ``imgui.text_disabled`` fades text to MUTED at 0.6 alpha. That is the correct
# rendering for a control nobody may operate and the wrong one for a sentence
# somebody has to read, and every call site in the tree was the second kind.
_DISABLED_TEXT = re.compile(r"\bimgui\.text_disabled\s*\(")


def test_no_source_file_draws_copy_with_the_disabled_text_role() -> None:
    """UX-18, as a rule rather than a one-off sweep.

    Text inside a ``begin_disabled`` block does not need this call either --
    imgui already fades the whole block -- so the ban is total, and
    ``widgets.secondary`` is the one way to draw a second-rank sentence.
    """
    offenders = [
        f"{path.relative_to(SRC)}:{i}"
        for path in SRC.rglob("*.py")
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _DISABLED_TEXT.search(line)
    ]
    assert not offenders, (
        "draw second-rank copy with widgets.secondary (UX-18): " + ", ".join(offenders)
    )


# -- text input (UX-19) -------------------------------------------------------


class _FakeIO:
    """Records what a backend hands imgui, without an imgui context."""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.keys: list[tuple[object, bool]] = []
        self.buttons: list[tuple[int, bool]] = []
        self.want_text_input = False

    def add_input_characters_utf8(self, text: str) -> None:
        self.text.append(text)

    def add_input_character(self, code: int) -> None:  # pragma: no cover
        self.text.append(chr(code))

    def add_key_event(self, key: object, down: bool) -> None:
        self.keys.append((key, down))

    def add_mouse_button_event(self, button: int, down: bool) -> None:
        from imgui_bundle import imgui

        # The real io asserts on this, and a failed assert crashes the frame
        # loop rather than failing a test, so the fake asserts it too.
        assert 0 <= int(button) < int(imgui.MouseButton_.count), button
        self.buttons.append((int(button), down))


class _FakeKey:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.rect: object = None

    def start_text_input(self) -> None:
        self.calls.append("start")

    def stop_text_input(self) -> None:
        self.calls.append("stop")

    def set_text_input_rect(self, rect: object) -> None:
        self.rect = rect


class _FakePygame:
    @staticmethod
    def Rect(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:  # noqa: N802
        return (x, y, w, h)

    def __init__(self) -> None:
        self.key = _FakeKey()


def _event(kind: object, **fields: object) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(type=kind, **fields)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> tuple[object, _FakeIO]:
    from warlock.studio import imgui_backend

    io = _FakeIO()
    monkeypatch.setattr(imgui_backend.imgui, "get_io", lambda: io)
    return imgui_backend, io


def test_committed_text_reaches_imgui_whole(backend: tuple[object, _FakeIO]) -> None:
    import pygame

    module, io = backend
    assert module.process_event(_event(pygame.TEXTINPUT, text="hello")) is True
    assert io.text == ["hello"]


def test_text_outside_the_bmp_survives(backend: tuple[object, _FakeIO]) -> None:
    """The concrete half of UX-19.

    The old path filtered on ``0 < ord(char) < 0x10000``, so every astral code
    point -- emoji, the CJK extensions, and every historic script -- was
    dropped silently on its way into a prompt.
    """
    import pygame

    module, io = backend
    module.process_event(_event(pygame.TEXTINPUT, text="a\U0001F600\U00020000"))
    assert io.text == ["a\U0001F600\U00020000"]


def test_a_composition_in_progress_does_not_reach_the_field(
    backend: tuple[object, _FakeIO],
) -> None:
    """TEXTEDITING is the pre-edit buffer, and imgui cannot render one."""
    import pygame

    module, io = backend
    assert module.process_event(_event(pygame.TEXTEDITING, text="か", start=0)) is True
    assert io.text == []


def test_a_keypress_no_longer_synthesises_characters(
    backend: tuple[object, _FakeIO],
) -> None:
    """Keys and characters are two facts and now arrive by two routes.

    Leaving the old ``KEYDOWN.unicode`` path in beside TEXTINPUT would type
    everything twice, which is the failure mode this asserts against.
    """
    import pygame

    module, io = backend
    module.process_event(_event(pygame.KEYDOWN, key=pygame.K_a, unicode="a"))
    assert io.text == []
    assert io.keys, "the key itself must still arrive"


def test_text_input_is_started_only_while_a_field_wants_it(
    backend: tuple[object, _FakeIO],
) -> None:
    module, io = backend
    fake = _FakePygame()
    module._text_input_on = None

    io.want_text_input = True
    assert module.sync_text_input(fake) is True
    assert fake.key.calls == ["start"]

    # Idempotent: re-starting SDL text input restarts an IME composition, so a
    # frame that changes nothing must call nothing.
    module.sync_text_input(fake)
    assert fake.key.calls == ["start"]

    io.want_text_input = False
    assert module.sync_text_input(fake) is False
    assert fake.key.calls == ["start", "stop"]


def test_the_ime_rect_is_placed_in_window_coordinates(
    backend: tuple[object, _FakeIO],
) -> None:
    module, _ = backend
    fake = _FakePygame()
    module.set_ime_rect((10.6, 20.2), (100.0, 24.0), fake)
    assert fake.key.rect == (10, 20, 100, 24)


# -- mouse buttons ------------------------------------------------------------


def test_the_three_named_buttons_reach_imgui(backend: tuple[object, _FakeIO]) -> None:
    import pygame
    from imgui_bundle import imgui

    module, io = backend
    for button in (1, 2, 3):
        assert module.process_event(_event(pygame.MOUSEBUTTONDOWN, button=button))
    assert io.buttons == [
        (int(imgui.MouseButton_.left), True),
        (int(imgui.MouseButton_.middle), True),
        (int(imgui.MouseButton_.right), True),
    ]


@pytest.mark.parametrize("button", [4, 5, 6, 7, 8, 9])
def test_a_button_imgui_cannot_name_is_dropped_not_shifted(
    backend: tuple[object, _FakeIO], button: int
) -> None:
    """A thumb click on the side of the mouse used to kill the frame loop.

    pygame numbers more buttons than imgui has -- 4/5 are the legacy wheel
    notches, 6/7 are SDL's X1/X2 back-and-forward pair -- and the old
    ``button - 1`` fallback handed imgui an index past ImGuiMouseButton_COUNT,
    which is an IM_ASSERT and therefore a RuntimeError out of the event pump.
    """
    import pygame

    module, io = backend
    assert module.process_event(_event(pygame.MOUSEBUTTONDOWN, button=button)) is False
    assert module.process_event(_event(pygame.MOUSEBUTTONUP, button=button)) is False
    assert io.buttons == []


# -- keyboard navigation (UX-02) ----------------------------------------------


def test_keyboard_navigation_is_enabled_for_the_whole_app() -> None:
    """The flag itself, read out of the source that sets it.

    A test rather than a comment because switching it back off would be a
    one-word change that no other test in the suite notices, and the whole of
    UX-02 rests on it.
    """
    source = (SRC / "studio" / "main.py").read_text(encoding="utf-8")
    assert "nav_enable_keyboard" in source


def test_the_key_map_covers_the_keyboard() -> None:
    """Not "some keys are mapped" but "the ordinary ones all are".

    The map used to hold the six letters the clipboard and undo chords needed.
    Anything else -- a digit, a function key, the letter a pane wanted for a
    tool -- was invisible to imgui, so no shortcut inside a text field and no
    nav step could ever read it.
    """
    import pygame

    from warlock.studio import imgui_backend

    for char in "abcdefghijklmnopqrstuvwxyz0123456789":
        key = getattr(pygame, f"K_{char}")
        assert key in imgui_backend._KEY_MAP, f"{char} is unmapped"
    for n in range(1, 13):
        assert getattr(pygame, f"K_F{n}") in imgui_backend._KEY_MAP, f"F{n} is unmapped"


def test_tab_is_never_reserved_away_from_navigation() -> None:
    """The half of the rule that makes traversal work everywhere.

    *Tab traverses, the arrows belong to the surface.* Putting Tab in the
    reserved set would give the five arrow-binding modes no focus traversal at
    all, which is the state UX-02 is about.
    """
    import pygame

    from warlock.studio import imgui_backend

    assert pygame.K_TAB not in imgui_backend._NAV_KEYS
    assert pygame.K_LEFT in imgui_backend._NAV_KEYS
    assert pygame.K_SPACE in imgui_backend._NAV_KEYS


def test_a_reserving_surface_keeps_the_arrows_from_imgui(
    backend: tuple[object, _FakeIO],
) -> None:
    """Home moving its Resume list must not also step a focus ring."""
    import pygame

    module, io = backend
    try:
        module.reserve_nav_keys(True)
        module.process_event(_event(pygame.KEYDOWN, key=pygame.K_DOWN))
        assert io.keys == []
    finally:
        module.reserve_nav_keys(False)


def test_a_reservation_yields_to_a_text_field(
    backend: tuple[object, _FakeIO],
) -> None:
    """Arrows are caret movement while somebody is typing.

    A mode that binds Up/Down for its list has no claim on them inside a
    prompt box -- and this is the case that makes the reservation safe to
    apply mode-wide rather than pane-by-pane.
    """
    import pygame

    module, io = backend
    try:
        module.reserve_nav_keys(True)
        io.want_text_input = True
        module.process_event(_event(pygame.KEYDOWN, key=pygame.K_LEFT))
        assert io.keys, "the caret still has to move"
    finally:
        module.reserve_nav_keys(False)


def test_modifiers_survive_a_reservation(backend: tuple[object, _FakeIO]) -> None:
    """Shift is never the surface's to claim.

    If a reservation swallowed modifiers too, imgui would read Shift+Tab as a
    plain Tab and backwards traversal would quietly become forwards.
    """
    import pygame

    module, io = backend
    try:
        module.reserve_nav_keys(True)
        module.process_event(_event(pygame.KEYDOWN, key=pygame.K_LSHIFT))
        assert io.keys, "Shift must reach imgui whatever the surface claims"
    finally:
        module.reserve_nav_keys(False)


def test_every_reserving_mode_is_a_real_mode() -> None:
    """A misspelled entry reserves nothing and says nothing.

    The failure this guards is silent in both directions: the name never
    matches ``state.mode``, so the arrows stay with imgui, and the symptom is
    a focus ring stepping behind a list -- which nobody would trace back to a
    typo in a frozenset.
    """
    from warlock.studio import modes

    unknown = sorted(modes.NAV_KEY_MODES - set(modes.KEYS))
    assert not unknown, f"not modes: {unknown}"


def test_the_reserving_modes_are_the_ones_that_bind_arrows() -> None:
    """The declaration against the source that does the binding.

    Read out of the mode modules rather than restated, so a mode that *starts*
    binding Up/Down without joining ``NAV_KEY_MODES`` fails here rather than
    at somebody's keyboard.
    """
    from warlock.studio import modes

    arrows = re.compile(r"K_(UP|DOWN|LEFT|RIGHT|SPACE)\b")
    binders = {
        mode
        for mode in modes.KEYS
        for path in [SRC / "studio" / f"{mode}_mode.py"]
        if path.exists() and arrows.search(path.read_text(encoding="utf-8"))
    }
    missing = sorted(binders - modes.NAV_KEY_MODES)
    assert not missing, (
        f"these modes bind arrows or Space but do not reserve them: {missing}"
    )


# -- display scale (UX-22) ----------------------------------------------------


class _ScaleApp:
    """Just enough of ``App`` to exercise ``_resample_display_scale``.

    The real one needs a GL context, a window and a service layer; what the
    method actually touches is four attributes and the token module, so the
    test builds those rather than the app.
    """

    from warlock.studio.main import App

    _resample_display_scale = App._resample_display_scale

    def __init__(self, monitor: float, ui_scale: float) -> None:
        from types import SimpleNamespace

        self._monitor_scale = monitor
        self._min_size = (0, 0)
        self.app_ctx = SimpleNamespace(
            dpi_scale=monitor,
            state=SimpleNamespace(fonts_dirty=False),
            settings=SimpleNamespace(get=lambda _k, ui=ui_scale: ui),
        )


@pytest.fixture
def quiet_rescale(monkeypatch: pytest.MonkeyPatch):
    """Stub the two things the method does to the outside world."""
    from warlock.studio import main, theme

    monkeypatch.setattr(theme, "apply", lambda _imgui: None)
    return main


def test_moving_to_another_monitor_rebuilds_the_scale(
    monkeypatch: pytest.MonkeyPatch, quiet_rescale
) -> None:
    """The whole of UX-22: a 100% -> 150% move used to change nothing."""
    from warlock.studio import dpi, tokens

    monkeypatch.setattr(dpi, "window_scale", lambda _p: 1.5)
    app = _ScaleApp(monitor=1.0, ui_scale=1.0)
    try:
        assert app._resample_display_scale() == pytest.approx(1.5)
        assert app.app_ctx.dpi_scale == pytest.approx(1.5)
        assert app.app_ctx.state.fonts_dirty, "the atlas is baked at the old size"
        assert app._min_size != (0, 0), "the resize floor follows the monitor"
    finally:
        tokens.set_scale(1.0)


def test_staying_on_the_same_monitor_rebuilds_nothing(
    monkeypatch: pytest.MonkeyPatch, quiet_rescale
) -> None:
    """WINDOWMOVED fires on every drag; almost none of them change the DPI.

    Rebuilding the font atlas on each would re-bake it whenever the window was
    nudged, which is a visible hitch for no reason.
    """
    from warlock.studio import dpi

    monkeypatch.setattr(dpi, "window_scale", lambda _p: 1.0)
    app = _ScaleApp(monitor=1.0, ui_scale=1.0)
    app._resample_display_scale()
    assert not app.app_ctx.state.fonts_dirty


def test_a_clamped_zoom_is_not_baked_in_by_moving(
    monkeypatch: pytest.MonkeyPatch, quiet_rescale
) -> None:
    """The trap in recovering the user's zoom by division.

    ``set_scale`` clamps the *product* to 4.0, so on a 250% display a stored
    2.0x zoom is really drawn at 1.6x. Dividing the scale in force by the old
    monitor scale to recover "the zoom" would read back 1.6, and each move
    between two such monitors would shrink the UI again -- permanently, since
    nothing ever writes it back up. Re-reading the stored preference is what
    makes the operation repeatable.
    """
    from warlock.studio import dpi, tokens

    try:
        monkeypatch.setattr(dpi, "window_scale", lambda _p: 2.5)
        app = _ScaleApp(monitor=1.0, ui_scale=2.0)
        app._resample_display_scale()
        # 2.0x is not offerable at 250%: the product ceiling leaves room for
        # 1.6x, so this is the clamp biting.
        assert pytest.approx(4.0) == tokens.SCALE

        # The discriminating move. Recovering the zoom by division would read
        # 4.0 / 2.5 == 1.6 and draw the 100% monitor at 1.6x; re-reading the
        # stored preference gives the 2.0x the user actually asked for. Both
        # implementations agree on every other value in this test, which is
        # what makes this the only assertion that proves anything.
        monkeypatch.setattr(dpi, "window_scale", lambda _p: 1.0)
        app._resample_display_scale()
        assert pytest.approx(2.0) == tokens.SCALE, "the clamp was baked in"
    finally:
        tokens.set_scale(1.0)


# -- responsive columns (UX-01) -----------------------------------------------


@pytest.fixture
def unscaled():
    """``fit`` reads ``tokens.SCALE`` through ``sp``; pin it and put it back."""
    from warlock.studio import tokens

    before = tokens.SCALE
    yield tokens.set_scale
    tokens.set_scale(before)


def test_a_roomy_window_gets_the_full_sidebar(unscaled) -> None:
    from warlock.studio import layout

    unscaled(1.0)
    assert layout.fit(2400.0, 8.0) == pytest.approx(layout.SIDEBAR_W)


def test_the_reported_failure_no_longer_overflows(unscaled) -> None:
    """UX-01 as the audit stated it, in numbers.

    A 1100-px window at 2x UI scale: two 300-design-px sidebars and a 300-px
    centre want 1800 physical px before gutters. Unconditionally reserving
    them is what pushed the inspector past the window edge, because the
    right-hand column is the one sized from what is left.
    """
    from warlock.studio import layout

    unscaled(2.0)
    spacing = 8.0
    room = 1100.0
    side = layout.fit(room, spacing)
    assert side * 2 + spacing * 2 <= room, "the sidebars alone must fit"
    centre = room - (side * 2 + spacing * 2)
    assert centre > 0, "and must leave something for the centre"


def test_the_sidebars_give_way_before_the_centre(unscaled) -> None:
    """The stated order: squeeze the sidebars, then the centre.

    Discriminating rather than decorative -- at a width where *something* has
    to give, this asserts which. An implementation that shrank the centre
    first would return the full sidebar here.
    """
    from warlock.studio import layout
    from warlock.studio.tokens import sp

    unscaled(1.0)
    # Room for the comfortable centre and two sidebars a little too narrow.
    room = sp(layout.SIDEBAR_W) * 2 + sp(layout.CENTRE_MIN) + 16.0 - 80.0
    side = layout.fit(room, 8.0)
    assert side < sp(layout.SIDEBAR_W), "the sidebar should have given way"
    assert side >= sp(layout.SIDEBAR_MIN)


def test_a_sidebar_is_never_squeezed_past_usefulness(unscaled) -> None:
    """Below SIDEBAR_MIN a form's labels wrap to a word a line.

    So the squeeze stops, and past that point it is the centre that shrinks --
    which ``centre_width``'s lower floor is what allows.
    """
    from warlock.studio import layout
    from warlock.studio.tokens import sp

    unscaled(2.0)
    assert layout.fit(200.0, 8.0) == pytest.approx(sp(layout.SIDEBAR_MIN))


def test_the_fit_is_unset_until_a_frame_measures_one() -> None:
    """Headless callers get the unconstrained width, not a fit against zero.

    ``tick`` stays pure for this reason -- it is called by tests with no imgui
    context, where reaching for a viewport is an access violation rather than
    an error -- so the measurement is a separate call and this is what the
    world looks like before it has happened.
    """
    from warlock.studio import layout
    from warlock.studio.tokens import sp

    assert layout.SIDEBAR_FIT is None or isinstance(layout.SIDEBAR_FIT, float)
    if layout.SIDEBAR_FIT is None:
        assert layout.sidebar_width() == pytest.approx(sp(layout.SIDEBAR_W))
