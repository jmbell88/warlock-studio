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
# Set once by main.setup() before any font or style is built; a mid-session
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

SP_1 = 4
SP_2 = 8
SP_3 = 12
SP_4 = 16
SP_6 = 24
SP_10 = 40

# -- radii / strokes ---------------------------------------------------------

RADIUS_S = 4.0
RADIUS_M = 6.0
RADIUS_L = 10.0
BORDER = 1.0

# -- type scale --------------------------------------------------------------

TEXT_SMALL = 11.0
TEXT_BODY = 13.0
TEXT_TITLE = 16.0

# -- motion ------------------------------------------------------------------

# Durations in seconds; motion.value() converts them to time constants.
DUR_FAST = 0.12
DUR_BASE = 0.20

# -- palette (sRGB hex, dark only) -------------------------------------------

# Near-black neutrals with one indigo accent: the Final Cut register. The
# elevation ramp replaces "everything is PANEL with a border" -- a surface says
# how high it sits by which step it fills with, not by drawing an outline.
BG = 0x0F1014  # the window floor
PANEL = 0x16171C  # sidebars, inspector: ELEV_0
ELEV_1 = 0x1D1F26  # cards, fields on a panel
ELEV_2 = 0x252833  # hovered/raised elements, popups
EDGE = 0x2A2D38  # the one hairline
TEXT = 0xE8E8EE
MUTED = 0x9A9DB0
ACCENT = 0x7C6CF0
OK = 0x4CC38A
ERR = 0xE5484D
WARN = 0xE5A03D

# Shadow alphas for the layered-rect elevation trick (no real blur in the
# backend): outer wide + inner tight.
SHADOW_OUTER = 0.22
SHADOW_INNER = 0.34
