"""The imgui style, built from the design tokens.

The palette lives in :mod:`.tokens` now (it grew an elevation ramp); the names
are re-exported here because ``theme.ACCENT`` is written all over the panes
and the palette is still conceptually "the theme". ``apply()`` is idempotent
on purpose -- every value is set absolutely and pre-multiplied by
``tokens.SCALE`` rather than run through ``style.scale_all_sizes``, which
compounds when called twice on one context (the test fixture applies the theme
per-session, but nothing should be one repeated call away from double-scaled
padding).
"""

from __future__ import annotations

from typing import Any

from . import icons, tokens


def __getattr__(name: str) -> int:
    """Resolve a palette name against the palette in force (M105).

    ``theme.ACCENT`` is written all over the panes, and before this it was a
    module constant bound at import -- so switching themes would have repainted
    imgui's own style and left every hand-drawn rect, badge and ring on the old
    colours. PEP 562 makes the same expression a live lookup, which is why the
    switch needed no edit at any of those call sites.

    Only the palette's own names resolve; anything else raises, as an
    attribute should. There used to be a ``RAISED`` alias for ``ELEV_2`` here
    -- "one step above PANEL", which predates the elevation ramp -- and it had
    no reader left anywhere in the tree. A colour name with no call site is a
    second vocabulary for the ramp, and the ramp is the vocabulary.
    """
    if name in tokens.COLOUR_NAMES:
        return tokens.colour(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Status to *palette name*, not to a colour: the values are resolved when a
# status is drawn, which is the whole point of the indirection above.
STATUS_COLORS = {
    "queued": "MUTED",
    "running": "ACCENT",
    "done": "OK",
    "error": "ERR",
    "cancelled": "MUTED",
}

# Glyphs rather than coloured dots alone: colour is the fast read, but a pill
# that only differs by hue is unreadable to a chunk of people and unprintable
# in a bug report.
#
# Lucide rather than the ASCII ``"..." / ">>" / "OK" / "!" / "x"`` these were
# (UX.md Phase 2). The double encoding is the point and survives the change --
# what improved is that the second channel is now a *shape* somebody can name:
# ``OK`` in green and ``!`` in red differ by two letters and a hue, where a
# check, a triangle and a circled cross differ by outline at any size and in
# any palette. Chosen for that: no two of the five share a silhouette, and the
# two that mean "this stopped" (error, cancelled) are deliberately the pair
# furthest apart, because a cancel is the user's decision and a failure is not.
STATUS_GLYPHS = {
    "queued": icons.ELLIPSIS,
    "running": icons.LOADER,
    "done": icons.CHECK,
    "error": icons.TRIANGLE_ALERT,
    "cancelled": icons.CIRCLE_X,
}


def rgba(value: int, alpha: float = 1.0) -> tuple[float, float, float, float]:
    return (
        ((value >> 16) & 0xFF) / 255.0,
        ((value >> 8) & 0xFF) / 255.0,
        (value & 0xFF) / 255.0,
        alpha,
    )


def mix(
    low: int, high: int, t: float, alpha: float = 1.0
) -> tuple[float, float, float, float]:
    """``low`` at ``t=0``, ``high`` at ``t=1``, interpolated in sRGB.

    Two palette entries rather than two rgba tuples, because the callers are
    animating between *roles* -- ELEV_1 to ELEV_2 as a card lifts, PANEL to
    ELEV_2 as a library row is chosen -- and a role resolves against the
    palette in force. Interpolating the encoded values is not physically
    correct, but every pair this is used on is one step of the elevation ramp
    apart, where the difference from a linear-light blend is under a level.
    """
    t = min(max(t, 0.0), 1.0)
    a, b = rgba(low), rgba(high)
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, alpha)


def status_color(status: str) -> tuple[float, float, float, float]:
    return rgba(tokens.colour(STATUS_COLORS.get(status, "MUTED")))


def apply(imgui: Any) -> None:
    """Set the global style. Called once, after the context and fonts exist."""
    sp = tokens.sp
    style = imgui.get_style()
    style.window_rounding = sp(tokens.RADIUS_M)
    style.child_rounding = sp(tokens.RADIUS_M)
    style.frame_rounding = sp(tokens.RADIUS_S + 1)
    style.popup_rounding = sp(tokens.RADIUS_M)
    style.grab_rounding = sp(tokens.RADIUS_S + 1)
    style.tab_rounding = sp(tokens.RADIUS_S + 1)
    style.scrollbar_rounding = sp(tokens.RADIUS_S + 1)
    # Depth comes from the elevation ramp, not outlines: a field is a lighter
    # fill on its panel, and the one hairline left is the child border.
    style.window_border_size = 0.0
    style.child_border_size = sp(tokens.BORDER)
    style.frame_border_size = 0.0
    style.popup_border_size = sp(tokens.BORDER)
    # Apple-scale gutters (UX.md Phase 2): 12 was the number a dense DAW panel
    # wants, and beside ``layout.PANE_PADDING``'s 5 it was the reason a form
    # read as pressed against the panel holding it. Both are steps of the scale
    # now -- SP_4 out here, SP_3 inside a pane -- so the two gutters are one
    # rhythm rather than two arbitrary numbers.
    style.window_padding = (sp(tokens.SP_4), sp(tokens.SP_4))
    style.frame_padding = (sp(9), sp(6))
    style.item_spacing = (sp(tokens.SP_2), sp(tokens.SP_2))
    style.item_inner_spacing = (sp(6), sp(5))
    style.cell_padding = (sp(tokens.SP_1), sp(3))
    style.indent_spacing = sp(tokens.SP_4)
    style.scrollbar_size = sp(10)
    style.grab_min_size = sp(10)
    style.separator_text_padding = (sp(tokens.SP_4), sp(3))

    c = imgui.Col_
    set_color = style.set_color_

    set_color(c.window_bg.value, rgba(tokens.colour("BG")))
    set_color(c.child_bg.value, rgba(tokens.colour("PANEL")))
    set_color(c.popup_bg.value, rgba(tokens.colour("ELEV_2"), 0.98))
    set_color(c.border.value, rgba(tokens.colour("EDGE")))
    set_color(c.border_shadow.value, (0, 0, 0, 0))
    set_color(c.text.value, rgba(tokens.colour("TEXT")))
    set_color(c.text_disabled.value, rgba(tokens.colour("MUTED"), 0.6))

    set_color(c.frame_bg.value, rgba(tokens.colour("ELEV_1")))
    set_color(c.frame_bg_hovered.value, rgba(tokens.colour("ELEV_2")))
    set_color(c.frame_bg_active.value, rgba(tokens.colour("EDGE")))

    set_color(c.title_bg.value, rgba(tokens.colour("PANEL")))
    set_color(c.title_bg_active.value, rgba(tokens.colour("PANEL")))
    set_color(c.title_bg_collapsed.value, rgba(tokens.colour("PANEL"), 0.7))
    set_color(c.menu_bar_bg.value, rgba(tokens.colour("PANEL")))

    set_color(c.button.value, rgba(tokens.colour("ELEV_2")))
    set_color(c.button_hovered.value, rgba(tokens.colour("ACCENT"), 0.75))
    set_color(c.button_active.value, rgba(tokens.colour("ACCENT")))

    set_color(c.header.value, rgba(tokens.colour("ELEV_2"), 0.9))
    set_color(c.header_hovered.value, rgba(tokens.colour("ACCENT"), 0.5))
    set_color(c.header_active.value, rgba(tokens.colour("ACCENT"), 0.75))

    # The focus ring imgui's own navigation draws (UX-02). ACCENT rather than
    # the near-white default, so it reads as the same "this is the live thing"
    # the rest of the app says in that colour -- and it is qualified against
    # WCAG 2.1 SC 1.4.11's 3:1 for a control boundary on every step of the
    # elevation ramp, in both palettes, by ``test_accessibility``. Full alpha:
    # a focus indicator faded for looks is the one piece of chrome whose whole
    # job is to be unmissable.
    set_color(c.nav_cursor.value, rgba(tokens.colour("ACCENT")))

    set_color(c.separator.value, rgba(tokens.colour("EDGE")))
    set_color(c.separator_hovered.value, rgba(tokens.colour("ACCENT"), 0.6))
    set_color(c.separator_active.value, rgba(tokens.colour("ACCENT")))

    set_color(c.check_mark.value, rgba(tokens.colour("ACCENT")))
    set_color(c.slider_grab.value, rgba(tokens.colour("ACCENT"), 0.85))
    set_color(c.slider_grab_active.value, rgba(tokens.colour("ACCENT")))

    set_color(c.scrollbar_bg.value, rgba(tokens.colour("BG"), 0.0))
    set_color(c.scrollbar_grab.value, rgba(tokens.colour("EDGE")))
    set_color(c.scrollbar_grab_hovered.value, rgba(tokens.colour("MUTED"), 0.5))
    set_color(c.scrollbar_grab_active.value, rgba(tokens.colour("ACCENT"), 0.7))

    set_color(c.tab.value, rgba(tokens.colour("PANEL")))
    set_color(c.tab_hovered.value, rgba(tokens.colour("ACCENT"), 0.6))
    set_color(c.tab_selected.value, rgba(tokens.colour("ELEV_2")))

    set_color(c.plot_histogram.value, rgba(tokens.colour("ACCENT")))
    set_color(c.plot_histogram_hovered.value, rgba(tokens.colour("ACCENT"), 0.8))
    set_color(c.table_header_bg.value, rgba(tokens.colour("PANEL")))
    set_color(c.table_border_strong.value, rgba(tokens.colour("EDGE")))
    set_color(c.table_border_light.value, rgba(tokens.colour("EDGE"), 0.5))
