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
# Larger steps arrive only with their readers, which is the same rule read the
# other way round: named spacing that no layout uses is not part of the scale.
SP_1 = 4
SP_2 = 8
SP_3 = 12
SP_4 = 16
SP_6 = 24

# -- radii / strokes ---------------------------------------------------------

RADIUS_S = 4.0
# The *control* radius (the UI redesign, wave 1). 6 was the number a control gets
# when the ramp is read as "small, medium, large" rather than as two jobs: at
# 6 a button, a field and a combo all read as *slightly* softened rects, which
# is the register a developer tool draws in and not the one this program is
# aiming at. 8 is where a 28-32 px control reads as deliberately rounded
# without becoming a pill -- and it is what let ``theme.apply`` stop spelling
# the frame radius ``RADIUS_S + 1``, an inline arithmetic that existed because
# neither named step was the right one.
RADIUS_M = 8.0
# The *surface* radius (UX.md Phase 2), and the split is what it is for: a
# control is a small rect and reads as a rounded rectangle at 4-8, while a
# surface -- a card, a modal, the palette window -- is large enough that the
# same radius reads as square. It arrives here now because its readers do:
# ``widgets.card``, ``dialogs``' two modals and the palette window. Phase 0
# deliberately left it out for exactly one frame longer than that.
#
# 12 rather than 10 for the same reason the control step moved, read at the
# other end: the gap between the two is what says "this is a surface, that is
# a thing on it", and 6-against-10 was a difference a reader had to be told
# about. The pair has to stay ordered -- ``test_the_surface_radius_arrived_
# with_its_readers`` asserts it -- because a surface rounded *tighter* than
# the controls inside it inverts the depth story the elevation ramp tells.
RADIUS_L = 12.0
# The *region* radius. A third job, and named rather than folded into one of
# the two above for the reason the ramp is named by job at all: ``RADIUS_M``
# is what a control rounds at and ``RADIUS_L`` what a surface rounds at, and a
# pane -- a region of the window, holding both -- is neither. Naming it is what
# lets this number move without dragging tabs, grabs, cards and modals with it.
#
# Deliberately *tighter* than the control radius, which is the one place this
# file's ordering rule is knowingly inverted: the panes that visibly round are
# the PANEL sidebars, inspectors and sheets (a CONTENT pane is ``theme.BG``,
# the window colour, so it rounds invisibly), and at 5 those read as softened
# regions rather than as oversized cards. A pane is not a thing sitting *on*
# the window in the way a card sits on a pane, so the depth story ``RADIUS_M``
# and ``RADIUS_L`` tell each other is not the story being told here.
#
# Panes take plain imgui rounding and never the nine-patch: ``surfaces``
# refuses below ``MIN_RADIUS`` (8), so ``surfaces.sprite(5)`` returns None and
# ``widgets.surface_fill`` is a no-op at this radius *by design*. That is
# forced rather than chosen -- recorded here so nobody "upgrades" panes to the
# squircle later and quietly gets nothing.
RADIUS_PANE = 5.0
BORDER = 1.0

# -- controls / interaction -------------------------------------------------

# One vertical rhythm for every interactive control.  The regular height is
# used by forms and primary toolbars; compact is for dense editor chrome and
# table rows.  These are design pixels, like the spacing scale above, and are
# converted with ``sp`` at the point of use.
CONTROL_HEIGHT_REGULAR = 30.0
CONTROL_HEIGHT_COMPACT = 26.0
# Short names are useful when a caller is choosing a size rather than doing
# arithmetic, and keep the public vocabulary aligned with ``ControlSize``.
CONTROL_REGULAR = CONTROL_HEIGHT_REGULAR
CONTROL_COMPACT = CONTROL_HEIGHT_COMPACT

# Shared state treatment.  Hover is an elevation change; pressed is a short
# accent acknowledgement; persistent selection is a quiet wash plus a hard
# accent boundary; keyboard focus is a separate, unmissable ring.  Keeping the
# alphas here prevents each hand-drawn control from inventing its own strength.
DIVIDER_WIDTH = 1.0
FOCUS_RING_WIDTH = 2.0
SELECTION_BOUNDARY_WIDTH = 2.0
SELECTION_WASH_ALPHA = 0.14
HOVER_WASH_ALPHA = 0.10
PRESSED_WASH_ALPHA = 0.24
ERROR_WASH_ALPHA = 0.12
# There is deliberately still no SP_8. UX.md Phase 0's list named it and Phase
# 0's own note deferred it to "whatever gap turns out to want 32"; the section
# rhythm this phase settled on is SP_6 above a heading and SP_4 inside a form,
# and the two 32-ish numbers left in the tree (``landing``'s icon column,
# ``empty_state``'s vertical offset) are *positions* rather than gaps, which
# that note already permits to stay literal. A name with no reader is the exact
# state the comment above records deleting SP_6 and RADIUS_L from -- and which
# ``test_the_spacing_scale_carries_only_the_steps_in_use`` fails on.

# -- type scale --------------------------------------------------------------

# The quiet end. 11/13 was a dense-tool ramp: 13 px Inter at 100 % is a size
# somebody reads a *table* in, and 11 is below where secondary copy stays
# comfortable at all -- which is why the muted-text contrast work (UX-18) was
# fighting a legibility problem it could not win with colour alone. 12/14 is
# the register the rest of the program is drawn in (the UI redesign, wave 1); the
# atlas is baked at ``TEXT_BODY`` (see :mod:`.fonts`), so this is the size the
# default face loads at and every unpushed string comes out in.
#
# Raising these raises every *measured* height with them -- a row is text plus
# ``frame_padding`` -- so the fixed heights tuned against a 13 px body moved in
# the same commit (``library.CARD_HEIGHT``, ``COMPACT_HEIGHT``, the
# ``_footer_px`` seed). The lever if a block still reads cramped is
# ``item_spacing``, never a font size: leading is the space *between* lines and
# shrinking the glyphs to buy it trades the thing being read for the gap around
# it.
TEXT_SMALL = 12.0
TEXT_BODY = 14.0
TEXT_TITLE = 16.0
# The loud end. Display type exists so a screen can have exactly one loud
# thing: with the ramp topping out at 16 the largest type in the app was a
# section heading, so the Home hero and a manual chapter title were the same
# size as the label above a combo. HEADING names a region, DISPLAY names a
# screen -- there is deliberately nothing between them and nothing above.
TEXT_HEADING = 20.0
TEXT_DISPLAY = 28.0

# -- state -------------------------------------------------------------------

# How far a disabled thing fades. imgui carries this as a style var and has
# always defaulted it to 0.6, which is the number ``text_disabled``'s own alpha
# was chosen against (see :func:`composite` and UX-18) -- so naming it here and
# setting it explicitly in ``theme.apply`` changes nothing on screen. What it
# buys is that the figure is *stated* rather than inherited: ``begin_disabled``
# multiplies the global alpha by it, so a colour a widget pushed fades by
# exactly as much as the label drawn on top of it, and a widget that reasons
# about its own greyed fill (``widgets.primary_button``) can name the rule
# instead of copying the literal 0.6 or -- worse -- dimming a second time on
# top of imgui and landing at 0.36.
DISABLED_ALPHA = 0.6

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
# **Three palettes, one set of names** (M105). Every pane reads ``theme.ACCENT``
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
#
# The third is the pixel-editor register, and it exists because Inker is measured
# against the program its users already know in *format* and *behaviour* -- a
# reader, a writer, a corpus, twenty-four numbered divergences -- and, until now,
# in look not at all. So somebody who lives in a pixel editor opened the Inker
# into the cool indigo-on-near-black that belongs to the 3D half of the app.
# ``pixel`` answers that with the register those tools draw in: a warm neutral
# ramp against a near-black canvas surround, and amber where the accent goes.
# It keeps *dark*'s elevation direction (steps away from the floor read lighter),
# so it is a change of hue and temperature rather than a third set of rules.
#
# It is *not* an eyedropper copy. A real pixel editor's chrome is grey on grey
# and would fail several of the bars in this module outright; these values take
# the hues and the roles and lift the numbers until every one clears, which is
# why MUTED is well above where a screenshot would have put it.
PALETTES: dict[str, dict[str, int]] = {
    "dark": {
        "BG": 0x0F1014,  # the window floor
        "PANEL": 0x16171C,  # sidebars, inspector: ELEV_0
        "ELEV_1": 0x1D1F26,  # cards, fields on a panel
        "ELEV_2": 0x252833,  # hovered/raised elements, popups
        "EDGE": 0x2A2D38,  # the one hairline
        "DIVIDER": 0x242630,  # major pane boundaries only
        "TEXT": 0xE8E8EE,
        "MUTED": 0x9A9DB0,
        "ACCENT": 0x7C6CF0,
        "OK": 0x4CC38A,
        "ERR": 0xE5484D,
        "WARN": 0xE5A03D,
        # The two squares of the transparency checker. Not a copy or a control
        # role, so no contrast bar applies to them -- what they answer to is
        # each *other*: the pair has to read as a checker rather than as a flat
        # field, and the hard-coded pair these replace was 14 levels apart.
        # The guided tour's scrim and its ring. Two entries rather than
        # reusing ACCENT and BG, because what they answer to is each *other*
        # and to the thing underneath: the scrim has to read as "not this part"
        # without hiding it -- the reader is still meant to see the control the
        # ring is around, and to be able to click it -- and the ring has to
        # separate from whatever the scrim is over rather than from a panel.
        #
        # A literal here instead would fail the way the checkerboard did: alive
        # in one palette's range and invisible in another, found by a
        # screenshot rather than by a test.
        "TOUR_VEIL": 0x07080B,
        "TOUR_RING": 0x9A8CFF,
        # A toggle's knob, and the wash a hover or a press paints over the 3D
        # viewport. Two more entries rather than literals for the checkerboard's
        # reason, read at the other end: both were hard-coded white, which is
        # ~1.3:1 against the light theme's EDGE track (the knob) and invisible
        # over a light viewport (the wash). ``test_accessibility`` measures
        # ``PALETTES`` and nothing else, so a literal is a colour no test can
        # see.
        "KNOB": 0xFFFFFF,
        "WASH": 0xFFFFFF,
        "CHECKER_A": 0x44464F,
        "CHECKER_B": 0x2C2E35,
    },
    "light": {
        "BG": 0xF4F4F7,
        "PANEL": 0xFBFBFD,
        "ELEV_1": 0xEDEDF2,
        "ELEV_2": 0xE1E1EA,
        "EDGE": 0xD2D2DC,
        "DIVIDER": 0xDADAE2,
        "TEXT": 0x1B1C22,
        "MUTED": 0x5F6272,
        "ACCENT": 0x5344C7,
        "OK": 0x1D7F53,
        "ERR": 0xC2262B,
        "WARN": 0x9A6410,
        # Light on light, the way every other editor draws it. The pair these
        # replace lived in the dark palette's range only, so a light-theme
        # session drew a near-black checkerboard under a white window -- and
        # dark artwork on it could not be seen, which is the half that mattered.
        "TOUR_VEIL": 0x2B2C36,
        "TOUR_RING": 0x1B1470,
        # Dark on light, both of them: a white knob on the light EDGE track
        # is ~1.3:1, and a white wash over a light-theme viewport is nothing
        # at all.
        "KNOB": 0x1B1C22,
        "WASH": 0x1B1C22,
        "CHECKER_A": 0xFFFFFF,
        "CHECKER_B": 0xD6D6DE,
    },
    "pixel": {
        "BG": 0x1A1714,  # the canvas surround: the darkest thing on screen
        "PANEL": 0x232019,  # toolbox, timeline, sidebars: ELEV_0
        "ELEV_1": 0x2E2A22,
        "ELEV_2": 0x3A352B,
        "EDGE": 0x4A4438,
        "DIVIDER": 0x2A2620,
        "TEXT": 0xF2EDE3,  # warm off-white rather than dark's cool one
        "MUTED": 0xB9AE9B,
        # Amber, where dark and light both spend indigo. It is the one colour
        # that says "pixel editor" on its own, and it clears the 3:1 control
        # boundary on all four elevation steps with room to spare.
        #
        # A warm accent puts ACCENT and WARN in the same family, which the other
        # two palettes never have to solve -- indigo against amber separates
        # itself. The first pass here landed them 2 degrees of hue apart, so a
        # warning badge and a focus ring were the same colour with a different
        # name. They are pulled to opposite ends of the warm range instead: the
        # accent is orange and heavily saturated, WARN is yellow and washed out,
        # and the two differ in hue, saturation *and* lightness rather than in
        # any one of them. ``theme.STATUS_GLYPHS`` is still the reason this is a
        # legibility improvement rather than the thing legibility rests on --
        # every status says itself in a shape as well as a colour.
        "ACCENT": 0xF7921E,
        "OK": 0x8FCB6B,
        "ERR": 0xF2706B,
        "WARN": 0xE3C75E,
        # Warmed to match the ramp: a neutral-grey checker under a warm panel
        # reads as a colour cast rather than as the absence of pixels.
        "TOUR_VEIL": 0x0E0C09,
        "TOUR_RING": 0xFFAE4A,
        # Warm off-white, the palette's TEXT, rather than a flat white that
        # would be the one cold thing on the screen.
        "KNOB": 0xF2EDE3,
        "WASH": 0xF2EDE3,
        "CHECKER_A": 0x4F4A40,
        "CHECKER_B": 0x35312A,
    },
}

# Which one is in force. Module state, exactly as ``SCALE`` is, and read
# through :func:`colour` rather than copied into constants.
THEME = "dark"


def set_theme(name: str) -> str:
    """Switch palettes. -> the name actually applied.

    An unknown name falls back to dark rather than raising: this comes out of a
    settings file, and a value written by a build that shipped a palette this
    one does not carry must not stop the window opening. The caller still has
    to re-run ``theme.apply`` -- imgui's style holds *copies* of these numbers.
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


# --- side columns -------------------------------------------------------------

# The three legacy named sidebar sizes, and the range a dragged splitter may
# land in, in design pixels.
#
# **Here rather than in ``layout``**, which is where they were written and
# where ``layout.SIDEBAR_WIDTHS``/``PANEL_MIN``/``PANEL_MAX`` still name them:
# ``layouts`` is the persistence half of the same pair and cannot import
# ``layout`` (that is the direction the dependency runs), so it had re-spelled
# all three as literals -- ``{"narrow": 260.0, ...}`` and ``min(max(v, 220.0),
# 480.0)`` in three places. Two spellings of a clamp is a clamp that stops
# agreeing, and a splitter and the file it is saved into disagreeing about the
# ceiling is a width that will not round-trip.
SIDEBAR_WIDTHS: dict[str, float] = {
    "narrow": 260.0,
    "default": 300.0,
    "wide": 360.0,
}
PANEL_MIN = 220.0
PANEL_MAX = 480.0


def clamp_panel(value: float) -> float:
    """A side-column width, held inside the range a splitter may reach."""

    return min(max(float(value), PANEL_MIN), PANEL_MAX)


# --- floating surface widths --------------------------------------------------

# How wide a thing that floats over the app is, in design pixels. Named for the
# reason the spacing scale is: eight call sites had spelled ``sp(210)``,
# ``sp(320)``, ``sp(430)``, ``sp(520)`` and ``sp(720)`` inline, so "how wide is
# a popover" had eight answers and no way to change any of them together. Four
# steps and no more, and each arrived with its readers -- the same rule the
# spacing scale states, which is what keeps this from becoming a menu of
# near-identical widths.
#
# A tooltip-sized readout: the FPS meter, a one-line hover card.
SURFACE_W_TIP = 210.0
# A popover: a combo's drop-down, one control's options.
SURFACE_W_POPOVER = 320.0
# A card: the progress card, a toast with a body.
SURFACE_W_CARD = 430.0
# A sheet: the shortcuts reference, the news panel, a snippet listing. Paired
# with ``SURFACE_H_SHEET``, which is the only height on the scale because it is
# the only one a surface is ever *given* rather than fitted to.
SURFACE_W_SHEET = 520.0
SURFACE_H_SHEET = 720.0

# A picture a judgement is made about, in design pixels: Review's labelling
# cell and Home's Resume thumbnail. One number for both, because both answer
# "big enough to judge a composition by, small enough that a sidebar-width
# column fits three across at 100% scale" -- they were 132 and 136, which is
# two answers to one question and a difference nobody could see.
THUMB_CELL = 136.0

