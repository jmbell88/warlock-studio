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

from . import tokens
from .tokens import (  # noqa: F401  (re-exports: the panes read these off theme)
    ACCENT,
    BG,
    EDGE,
    ELEV_1,
    ELEV_2,
    ERR,
    MUTED,
    OK,
    PANEL,
    TEXT,
    WARN,
)

# Kept as an alias: "one step above PANEL" predates the elevation ramp.
RAISED = ELEV_2

STATUS_COLORS = {
    "queued": MUTED,
    "running": ACCENT,
    "done": OK,
    "error": ERR,
    "cancelled": MUTED,
}

# Glyphs rather than coloured dots alone: colour is the fast read, but a pill
# that only differs by hue is unreadable to a chunk of people and unprintable
# in a bug report.
STATUS_GLYPHS = {
    "queued": "...",
    "running": ">>",
    "done": "OK",
    "error": "!",
    "cancelled": "x",
}


def rgba(value: int, alpha: float = 1.0) -> tuple[float, float, float, float]:
    return (
        ((value >> 16) & 0xFF) / 255.0,
        ((value >> 8) & 0xFF) / 255.0,
        (value & 0xFF) / 255.0,
        alpha,
    )


def status_color(status: str) -> tuple[float, float, float, float]:
    return rgba(STATUS_COLORS.get(status, MUTED))


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
    style.window_padding = (sp(tokens.SP_3), sp(tokens.SP_3))
    style.frame_padding = (sp(9), sp(6))
    style.item_spacing = (sp(tokens.SP_2), sp(7))
    style.item_inner_spacing = (sp(6), sp(5))
    style.cell_padding = (sp(tokens.SP_1), sp(3))
    style.indent_spacing = sp(tokens.SP_4)
    style.scrollbar_size = sp(10)
    style.grab_min_size = sp(10)
    style.separator_text_padding = (sp(tokens.SP_4), sp(3))

    c = imgui.Col_
    set_color = style.set_color_

    set_color(c.window_bg.value, rgba(BG))
    set_color(c.child_bg.value, rgba(PANEL))
    set_color(c.popup_bg.value, rgba(ELEV_2, 0.98))
    set_color(c.border.value, rgba(EDGE))
    set_color(c.border_shadow.value, (0, 0, 0, 0))
    set_color(c.text.value, rgba(TEXT))
    set_color(c.text_disabled.value, rgba(MUTED, 0.6))

    set_color(c.frame_bg.value, rgba(ELEV_1))
    set_color(c.frame_bg_hovered.value, rgba(ELEV_2))
    set_color(c.frame_bg_active.value, rgba(EDGE))

    set_color(c.title_bg.value, rgba(PANEL))
    set_color(c.title_bg_active.value, rgba(PANEL))
    set_color(c.title_bg_collapsed.value, rgba(PANEL, 0.7))
    set_color(c.menu_bar_bg.value, rgba(PANEL))

    set_color(c.button.value, rgba(ELEV_2))
    set_color(c.button_hovered.value, rgba(ACCENT, 0.75))
    set_color(c.button_active.value, rgba(ACCENT))

    set_color(c.header.value, rgba(ELEV_2, 0.9))
    set_color(c.header_hovered.value, rgba(ACCENT, 0.5))
    set_color(c.header_active.value, rgba(ACCENT, 0.75))

    set_color(c.separator.value, rgba(EDGE))
    set_color(c.separator_hovered.value, rgba(ACCENT, 0.6))
    set_color(c.separator_active.value, rgba(ACCENT))

    set_color(c.check_mark.value, rgba(ACCENT))
    set_color(c.slider_grab.value, rgba(ACCENT, 0.85))
    set_color(c.slider_grab_active.value, rgba(ACCENT))

    set_color(c.scrollbar_bg.value, rgba(BG, 0.0))
    set_color(c.scrollbar_grab.value, rgba(EDGE))
    set_color(c.scrollbar_grab_hovered.value, rgba(MUTED, 0.5))
    set_color(c.scrollbar_grab_active.value, rgba(ACCENT, 0.7))

    set_color(c.tab.value, rgba(PANEL))
    set_color(c.tab_hovered.value, rgba(ACCENT, 0.6))
    set_color(c.tab_selected.value, rgba(ELEV_2))

    set_color(c.plot_histogram.value, rgba(ACCENT))
    set_color(c.plot_histogram_hovered.value, rgba(ACCENT, 0.8))
    set_color(c.table_header_bg.value, rgba(PANEL))
    set_color(c.table_border_strong.value, rgba(EDGE))
    set_color(c.table_border_light.value, rgba(EDGE, 0.5))
