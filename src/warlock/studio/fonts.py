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

#: Every file the atlas is built from, in the order :func:`load` reads them.
FACES = ("Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf", "lucide.ttf")


class FontsUnavailable(RuntimeError):
    """A vendored TTF is missing or unreadable.

    A named exception rather than whatever ``add_font_from_file_ttf`` does with
    a path that is not there, which is an ``IM_ASSERT`` surfacing as a bare
    ``RuntimeError`` with imgui's own wording in it. These files ship in the
    wheel, so the ways to reach this are a partial install, an antivirus
    quarantine and a half-copied directory -- all of which are worth naming,
    and none of which the user can act on from "assertion failed".
    """

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(
            f"missing font files in {FONT_DIR}: {', '.join(missing)}"
        )


def _check_files() -> None:
    """Refuse before imgui is asked for a file that is not there.

    ``load`` had no existence check at all, and it is reachable **mid-session**:
    the UI-scale slider re-bakes the atlas through :func:`reload`, so a font
    file that went away while the app was running took the frame loop with it.
    """
    missing = [name for name in FACES if not (FONT_DIR / name).is_file()]
    if missing:
        raise FontsUnavailable(missing)

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


def reload(imgui: Any) -> None:
    """Re-bake the atlas at the current ``tokens.SCALE`` (K99).

    **Between frames, never inside one.** ``clear_fonts`` invalidates every
    ``ImFont`` handle, and the ones this module holds are pushed and popped all
    over a frame -- rebuilding mid-frame would leave the rest of that frame
    drawing through freed pointers. ``main`` calls this before ``new_frame``.

    The reason it is needed at all is ``ICON_OFFSET``: the merged icon range's
    glyph offset is baked as an absolute pixel figure at *load* time, so at any
    scale but the one the atlas was built at, every icon sits off-centre in its
    button by a fraction of the difference. The glyph *shapes* would sharpen on
    their own -- imgui 1.92 rasterises per pushed size -- which is exactly why
    the old "text sharpens after a restart" note was only half the story.

    ``FontsUnavailable`` is raised *before* ``clear_fonts``, so a rebuild that
    cannot happen leaves the atlas it already had rather than an empty one. A
    failure after that point resets the three handles to ``None``, which is the
    documented headless state -- every helper here degrades onto imgui's own
    default font -- so the caller can report it and keep drawing.
    """
    _check_files()
    imgui.get_io().fonts.clear_fonts()
    global REGULAR, MEDIUM, SEMIBOLD
    try:
        load(imgui)
    except Exception:
        REGULAR = MEDIUM = SEMIBOLD = None
        raise


def load(imgui: Any) -> None:
    """Build the atlas fonts. Call between context creation and first frame."""
    global REGULAR, MEDIUM, SEMIBOLD
    _check_files()
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


# There is deliberately no ``body()`` helper. Regular is loaded first, so it
# *is* imgui's default font, and it is loaded at ``tokens.TEXT_BODY`` -- a
# ``push(REGULAR, TEXT_BODY)`` therefore pushes what is already in force. The
# one that existed had no callers, and the reason it never gained any is that
# there is nothing for it to do.


def small(imgui: Any) -> Any:
    return push(imgui, REGULAR, tokens.TEXT_SMALL)


def label(imgui: Any) -> Any:
    """Medium weight at body size: buttons, field labels, card titles."""
    return push(imgui, MEDIUM, tokens.TEXT_BODY)


def title(imgui: Any) -> Any:
    return push(imgui, SEMIBOLD, tokens.TEXT_TITLE)


def heading(imgui: Any) -> Any:
    """One region's name: a pane header, a manual chapter."""
    return push(imgui, SEMIBOLD, tokens.TEXT_HEADING)


def display(imgui: Any) -> Any:
    """The one loud thing on a screen. There is deliberately only ever one.

    SemiBold rather than a fourth vendored face: at 28 px Inter SemiBold is
    already emphatic, and Bold at display size reads as a warning rather than
    as a title against near-black. The call was made against the screenshot
    pass, which is the only way it could be made.
    """
    return push(imgui, SEMIBOLD, tokens.TEXT_DISPLAY)
