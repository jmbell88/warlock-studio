"""Contrast and keyboard access, measured rather than asserted.

The audit's accessibility findings were all of the form "this looks fine and
measures badly", so the tests here measure. Every number in them is computed
from the palette that ships, which is what makes the file a gate rather than a
transcription: changing a token to something that fails fails here, and nobody
has to remember to re-run a colour picker.
"""

from __future__ import annotations

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
        self.want_text_input = False

    def add_input_characters_utf8(self, text: str) -> None:
        self.text.append(text)

    def add_input_character(self, code: int) -> None:  # pragma: no cover
        self.text.append(chr(code))

    def add_key_event(self, key: object, down: bool) -> None:
        self.keys.append((key, down))


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
