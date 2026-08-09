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
BORDER = 1.0
# There is deliberately still no RADIUS_L, and no SP_8. Both were written into
# UX.md Phase 0's list and both belong to Phase 2: the surface radius is read by
# cards, modals and the palette window when *they* change, and SP_8 by whatever
# gap turns out to want 32. Adding them here first would have put two names in
# this file with nothing reading them, which is the exact state the comment
# above records deleting them from -- and which
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

# Shadow alphas for the layered-rect elevation trick (no real blur in the
# backend): outer wide + inner tight.
SHADOW_OUTER = 0.22
SHADOW_INNER = 0.34
