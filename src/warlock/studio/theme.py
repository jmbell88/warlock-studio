"""The palette and the imgui style built from it.

The same nine colours the browser build used, so a screenshot of one app and a
screenshot of the other are recognisably the same product. They are defined
here in sRGB hex because that is how they were authored and how they appear in
the CSS this replaces; imgui wants linear-ish floats in 0..1 and gets them from
:func:`rgba`.
"""

from __future__ import annotations

from typing import Any

BG = 0x14151A
PANEL = 0x1D1F27
EDGE = 0x2C2F3A
TEXT = 0xE6E6EC
MUTED = 0x9A9DB0
ACCENT = 0x7C6CF0
OK = 0x4CC38A
ERR = 0xE5484D
WARN = 0xE5A03D

# One step lighter than PANEL, for a raised element on a panel (a card, a
# hovered row). Derived rather than a tenth named colour: it only ever means
# "PANEL, but this one".
RAISED = 0x24273180 & 0xFFFFFF

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
    """Set the global style. Called once, after the context exists."""
    style = imgui.get_style()
    style.window_rounding = 6.0
    style.child_rounding = 6.0
    style.frame_rounding = 5.0
    style.popup_rounding = 6.0
    style.grab_rounding = 5.0
    style.tab_rounding = 5.0
    style.scrollbar_rounding = 5.0
    style.window_border_size = 0.0
    style.child_border_size = 1.0
    style.frame_border_size = 1.0
    style.window_padding = (12, 12)
    style.frame_padding = (8, 5)
    style.item_spacing = (8, 7)
    style.item_inner_spacing = (6, 5)
    style.scrollbar_size = 12.0
    style.grab_min_size = 10.0

    c = imgui.Col_
    set_color = style.set_color_

    set_color(c.window_bg.value, rgba(BG))
    set_color(c.child_bg.value, rgba(PANEL))
    set_color(c.popup_bg.value, rgba(PANEL, 0.98))
    set_color(c.border.value, rgba(EDGE))
    set_color(c.border_shadow.value, (0, 0, 0, 0))
    set_color(c.text.value, rgba(TEXT))
    set_color(c.text_disabled.value, rgba(MUTED, 0.6))

    set_color(c.frame_bg.value, rgba(BG))
    set_color(c.frame_bg_hovered.value, rgba(EDGE, 0.8))
    set_color(c.frame_bg_active.value, rgba(EDGE))

    set_color(c.title_bg.value, rgba(PANEL))
    set_color(c.title_bg_active.value, rgba(PANEL))
    set_color(c.title_bg_collapsed.value, rgba(PANEL, 0.7))
    set_color(c.menu_bar_bg.value, rgba(PANEL))

    set_color(c.button.value, rgba(EDGE))
    set_color(c.button_hovered.value, rgba(ACCENT, 0.75))
    set_color(c.button_active.value, rgba(ACCENT))

    set_color(c.header.value, rgba(EDGE, 0.9))
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
    set_color(c.tab_selected.value, rgba(EDGE))

    set_color(c.plot_histogram.value, rgba(ACCENT))
    set_color(c.plot_histogram_hovered.value, rgba(ACCENT, 0.8))
    set_color(c.table_header_bg.value, rgba(PANEL))
    set_color(c.table_border_strong.value, rgba(EDGE))
    set_color(c.table_border_light.value, rgba(EDGE, 0.5))
