"""The type ramp: three Inter faces with Lucide icons merged into each.

Loaded once at startup from the TTFs vendored under ``resources/fonts`` -- the
offline invariant applies to fonts as much as to model weights, so these ship
in the wheel and are never fetched. imgui 1.92 sizes fonts at ``push_font``
time, so each *face* is loaded once and the type scale is applied where text
is drawn, not in the atlas.

Headless tests never call :func:`load`; every helper here degrades to a no-op
on the default atlas font (icons render as the missing-glyph box, which the
smoke tests do not look at).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import tokens

FONT_DIR = Path(__file__).parent / "resources" / "fonts"

# Lucide is upem 1000 / ascent 1000 / descent 0, with its ink flush at x=0 and
# spanning y 0..~959; Inter's baseline sits at 0.801 of the line box (ascent
# 1984 of 1984+494 over a 2048 upem). imgui bakes each merged source against
# its *own* ascent-to-descent span and then draws it on the *destination*
# font's baseline, so an unnudged icon occupies -0.158..0.801 of a 0..1 line
# box -- 0.179 too high -- and sits 0.04 to the left inside its own advance.
# Every icon button in the app centres on the line box or the advance, so this
# one offset is what makes all of them land in the middle of their boxes.
ICON_OFFSET = (0.040, 0.179)

# ImFont handles, populated by load(). None means "run on imgui's default".
REGULAR: Any = None
MEDIUM: Any = None
SEMIBOLD: Any = None


def load(imgui: Any) -> None:
    """Build the atlas fonts. Call between context creation and first frame."""
    global REGULAR, MEDIUM, SEMIBOLD
    io = imgui.get_io()
    base = tokens.TEXT_BODY * tokens.SCALE

    def face(name: str) -> Any:
        font = io.fonts.add_font_from_file_ttf(str(FONT_DIR / name), base)
        merge = imgui.ImFontConfig()
        merge.merge_mode = True
        merge.glyph_offset = imgui.ImVec2(base * ICON_OFFSET[0], base * ICON_OFFSET[1])
        io.fonts.add_font_from_file_ttf(str(FONT_DIR / "lucide.ttf"), base, merge)
        return font

    # Regular first: the first atlas font is imgui's default, so every string
    # that never pushes a font still comes out in Inter.
    REGULAR = face("Inter-Regular.ttf")
    MEDIUM = face("Inter-Medium.ttf")
    SEMIBOLD = face("Inter-SemiBold.ttf")


@contextmanager
def push(imgui: Any, font: Any, size: float) -> Iterator[None]:
    """Push a face at a design-pixel size; no-op when fonts were never loaded."""
    if font is None:
        yield
        return
    imgui.push_font(font, size * tokens.SCALE)
    try:
        yield
    finally:
        imgui.pop_font()


def body(imgui: Any) -> Any:
    return push(imgui, REGULAR, tokens.TEXT_BODY)


def small(imgui: Any) -> Any:
    return push(imgui, REGULAR, tokens.TEXT_SMALL)


def label(imgui: Any) -> Any:
    """Medium weight at body size: buttons, field labels, card titles."""
    return push(imgui, MEDIUM, tokens.TEXT_BODY)


def title(imgui: Any) -> Any:
    return push(imgui, SEMIBOLD, tokens.TEXT_TITLE)
