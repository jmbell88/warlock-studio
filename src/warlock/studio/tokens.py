"""Design tokens: the numbers the UI is drawn from.

One module owns spacing, radii, type sizes, the palette and the DPI scale so
that a pane never hard-codes a pixel value twice. Everything here is a plain
constant except :data:`SCALE`, which is sampled once at startup from the
monitor the window opens on (see :mod:`.dpi`) and multiplied in by the
:func:`sp` helper -- panes ask for *design* pixels and get *physical* pixels.
"""

from __future__ import annotations

# -- DPI ---------------------------------------------------------------------

# Physical-pixels-per-design-pixel. 1.0 on a 96 DPI monitor, 1.5 at 150 %.
# Set once by main.App.setup_window() before any font or style is built; a mid-session
# monitor change does not re-sample it (the atlas would need a rebuild).
SCALE = 1.0

# What SCALE itself is allowed to be, after the monitor and the user's zoom are
# multiplied together. A sanity clamp on the product, not a preference.
SCALE_RANGE = (0.5, 4.0)

# What the user's zoom alone is allowed to be, before the monitor's scale.
UI_SCALE_RANGE = (0.5, 2.0)


def set_scale(value: float) -> None:
    global SCALE
    SCALE = max(SCALE_RANGE[0], min(float(value), SCALE_RANGE[1]))


def ui_scale_bounds(monitor_scale: float) -> tuple[float, float]:
    """The zoom range actually offerable on a monitor of this scale.

    ``set_scale`` clamps the *product*, so on a 250 % display a requested 2x
    silently became 1.6x and the slider snapped back under the cursor. Bounding
    the control by what the product can hold means it can only ask for a value
    it will get.
    """
    base = float(monitor_scale) or 1.0
    lo = max(UI_SCALE_RANGE[0], SCALE_RANGE[0] / base)
    hi = min(UI_SCALE_RANGE[1], SCALE_RANGE[1] / base)
    # A monitor scaled past the product ceiling leaves no room to zoom at all;
    # a degenerate range would make the slider unusable rather than merely
    # limited, so it collapses to the one value that is honoured.
    return (lo, hi) if lo <= hi else (hi, hi)


def sp(n: float) -> float:
    """Design pixels -> physical pixels."""
    return n * SCALE


# -- spacing scale -----------------------------------------------------------

# Only the steps something actually spaces with. A scale carried out to every
# multiple "for completeness" is a menu of near-identical choices, which is how
# two panes come to sit 24 and 20 apart for no stated reason -- SP_6, SP_10 and
# RADIUS_L were exactly that and had no readers at all.
#
# The upper half (SP_5..SP_8) arrives with its readers, which is the same rule
# read the other way round: the landing screen was inventing ``sp(20)``,
# ``sp(40)`` and ``sp(48)`` inline because the scale stopped at 16, so the gaps
# that carry the app's first screen were the only ones no module owned.
SP_1 = 4
SP_2 = 8
SP_3 = 12
SP_4 = 16
SP_5 = 20
SP_6 = 24

# -- radii / strokes ---------------------------------------------------------

RADIUS_S = 4.0
RADIUS_M = 6.0
# The *surface* radius (UX.md Phase 2), and the split is what it is for: a
# control is a small rect and reads as a rounded rectangle at 4-6, while a
# surface -- a card, a modal, the palette window -- is large enough that the
# same radius reads as square. It arrives here now because its readers do:
# ``widgets.card``, ``dialogs``' two modals and the palette window. Phase 0
# deliberately left it out for exactly one frame longer than that.
RADIUS_L = 10.0
BORDER = 1.0
# There is deliberately still no SP_8. UX.md Phase 0's list named it and Phase
# 0's own note deferred it to "whatever gap turns out to want 32"; the section
# rhythm this phase settled on is SP_6 above a heading and SP_4 inside a form,
# and the two 32-ish numbers left in the tree (``landing``'s icon column,
# ``empty_state``'s vertical offset) are *positions* rather than gaps, which
# that note already permits to stay literal. A name with no reader is the exact
# state the comment above records deleting SP_6 and RADIUS_L from -- and which
# ``test_the_spacing_scale_carries_only_the_steps_in_use`` fails on.

# -- type scale --------------------------------------------------------------

TEXT_SMALL = 11.0
TEXT_BODY = 13.0
TEXT_TITLE = 16.0
# The loud end. Display type exists so a screen can have exactly one loud
# thing: with the ramp topping out at 16 the largest type in the app was a
# section heading, so the Home hero and a manual chapter title were the same
# size as the label above a combo. HEADING names a region, DISPLAY names a
# screen -- there is deliberately nothing between them and nothing above.
TEXT_HEADING = 20.0
TEXT_DISPLAY = 28.0

# -- motion ------------------------------------------------------------------

# Durations in seconds; motion.value() converts them to time constants.
DUR_FAST = 0.12
DUR_BASE = 0.20
# Mode-scale transitions: a whole screen changing is a longer move than a
# hover, and the same 0.20 that reads as instant on a pill reads as a glitch
# across a viewport.
DUR_SLOW = 0.30

# -- palette (sRGB hex) ------------------------------------------------------

# Near-black neutrals with one indigo accent: the Final Cut register. The
# elevation ramp replaces "everything is PANEL with a border" -- a surface says
# how high it sits by which step it fills with, not by drawing an outline.
#
# **Two palettes, one set of names** (M105). Every pane reads ``theme.ACCENT``
# and never a literal, so the whole of a theme is this table plus the module
# ``__getattr__`` in :mod:`.theme` that resolves the names through it live --
# module-level constants would have bound at import and a switch would have
# repainted the imgui style while leaving every hand-drawn rect on the old one.
#
# The light palette keeps the *roles* rather than inverting the numbers. An
# inverted dark theme puts near-white text on near-black cards and calls it
# light; here PANEL is still "the surface a form sits on" and ELEV_1/ELEV_2 are
# still steps *away from* the floor, which on a light ground means darker
# rather than lighter -- so a card still reads as raised. ACCENT keeps its hue
# and drops in lightness, because the same indigo that reads as a highlight on
# black is barely visible on white.
PALETTES: dict[str, dict[str, int]] = {
    "dark": {
        "BG": 0x0F1014,  # the window floor
        "PANEL": 0x16171C,  # sidebars, inspector: ELEV_0
        "ELEV_1": 0x1D1F26,  # cards, fields on a panel
        "ELEV_2": 0x252833,  # hovered/raised elements, popups
        "EDGE": 0x2A2D38,  # the one hairline
        "TEXT": 0xE8E8EE,
        "MUTED": 0x9A9DB0,
        "ACCENT": 0x7C6CF0,
        "OK": 0x4CC38A,
        "ERR": 0xE5484D,
        "WARN": 0xE5A03D,
    },
    "light": {
        "BG": 0xF4F4F7,
        "PANEL": 0xFBFBFD,
        "ELEV_1": 0xEDEDF2,
        "ELEV_2": 0xE1E1EA,
        "EDGE": 0xD2D2DC,
        "TEXT": 0x1B1C22,
        "MUTED": 0x5F6272,
        "ACCENT": 0x5344C7,
        "OK": 0x1D7F53,
        "ERR": 0xC2262B,
        "WARN": 0x9A6410,
    },
}

# Which one is in force. Module state, exactly as ``SCALE`` is, and read
# through :func:`colour` rather than copied into constants.
THEME = "dark"


def set_theme(name: str) -> str:
    """Switch palettes. -> the name actually applied.

    An unknown name falls back to dark rather than raising: this comes out of a
    settings file, and a value written by a build that shipped a third palette
    must not stop the window opening. The caller still has to re-run
    ``theme.apply`` -- imgui's style holds *copies* of these numbers.
    """
    global THEME
    THEME = name if name in PALETTES else "dark"
    return THEME


def colour(name: str) -> int:
    """One palette entry under the current theme."""
    return PALETTES[THEME][name]


# The palette names, so ``theme`` can answer for exactly these and let every
# other attribute error normally.
COLOUR_NAMES = frozenset(PALETTES["dark"])

# -- contrast ----------------------------------------------------------------
#
# WCAG 2.1's two bars, because the app has two kinds of thing to qualify: body
# copy at 4.5:1 (SC 1.4.3 AA) and a control's own boundary -- a focus ring, a
# checkbox outline -- at 3:1 (SC 1.4.11). Naming both is what stops the accent
# being judged against the stricter bar it does not need to meet and the
# secondary copy against the looser one it does.
CONTRAST_TEXT = 4.5
CONTRAST_UI = 3.0

# The surfaces copy is actually drawn on, darkest-to-lightest step of the
# elevation ramp. A role is qualified against *all* of them rather than against
# BG alone: MUTED clears 4.5:1 on the window floor and the same label on a
# hovered card is the one that fails, so the floor is the least informative
# place to measure.
COPY_SURFACES = ("BG", "PANEL", "ELEV_1", "ELEV_2")


def _linear(channel: float) -> float:
    """One 0..1 sRGB channel, linearised."""
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def luminance(value: int) -> float:
    """Relative luminance of a packed sRGB colour, per WCAG 2.1."""
    r, g, b = (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF
    return (
        0.2126 * _linear(r / 255.0)
        + 0.7152 * _linear(g / 255.0)
        + 0.0722 * _linear(b / 255.0)
    )


def contrast(fg: int, bg: int) -> float:
    """The contrast ratio between two packed colours, 1.0 (same) to 21.0."""
    a, b = luminance(fg), luminance(bg)
    hi, lo = (a, b) if a >= b else (b, a)
    return (hi + 0.05) / (lo + 0.05)


def composite(fg: int, bg: int, alpha: float) -> int:
    """``fg`` drawn over ``bg`` at ``alpha``, as a packed colour.

    Contrast is a property of what reaches the eye, so a translucent label has
    to be measured on the blend and never on the token it came from. This is
    the whole of UX-18: MUTED is *stored* at 6.67:1 on PANEL and, at the 0.6
    alpha ``text_disabled`` carries, *drawn* at 3.20:1. Compositing in sRGB
    rather than linear light because that is what the blend in the renderer
    does -- the number this returns is the one on the glass, not the one a
    correctly gamma-aware compositor would have produced.
    """
    alpha = min(max(alpha, 0.0), 1.0)
    out = 0
    for shift in (16, 8, 0):
        f, b = (fg >> shift) & 0xFF, (bg >> shift) & 0xFF
        out |= round(f * alpha + b * (1.0 - alpha)) << shift
    return out

# -- depth -------------------------------------------------------------------
#
# **One shadow, told everywhere** (UX.md Phase 2). There is no blur in the
# backend, so a blur is *approximated*: four concentric bands, each further out
# and fainter than the last, whose union has a soft edge where the old two
# hard-edged rects had a visible step. Each entry is ``(inner, outer, alpha)``
# in design px from the surface's own edge, and the bands tile -- band n's
# outer bound is band n+1's inner -- so there is no gap to read as a ring.
# The alphas fall off roughly quadratically, which is what the tail of a
# Gaussian does.
#
# ``widgets.shadow`` draws each band as a *stroked* rounded rect rather than a
# filled one, which is what lets the same recipe sit under a card (drawn before
# the surface) and under a popup window (drawn after it, because a window's
# background is painted by ``begin``): a stroke leaves the interior alone, so
# the second case does not darken the surface it is meant to lift.
SHADOW_STEPS: tuple[tuple[float, float, float], ...] = (
    (0.0, 1.5, 0.22),
    (1.5, 3.0, 0.14),
    (3.0, 5.5, 0.09),
    (5.5, 9.0, 0.05),
)

# How far off the surface behind it a thing sits, as a multiplier on the bands
# above. Three names rather than a number at each call site, because "a card
# rests, a popup floats, a modal blocks" is the physical story and the numbers
# are one reading of it. An unknown name reads as ``resting``: this comes from
# a call site rather than a file, but a shadow is not worth an exception.
ELEVATION: dict[str, float] = {"resting": 1.0, "raised": 1.7, "overlay": 2.6}

# How far the whole shadow is displaced downward, as a fraction of a band's
# outer bound: the light is above, so a raised surface casts below itself. Kept
# small -- a shadow a reader can measure is an object hanging in the air.
SHADOW_DROP = 0.35
