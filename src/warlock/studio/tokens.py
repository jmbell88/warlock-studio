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


def set_scale(value: float) -> None:
    global SCALE
    SCALE = max(0.5, min(float(value), 4.0))


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
