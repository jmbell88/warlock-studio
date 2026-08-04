"""Resizable pane widths, persisted between runs.

The three-column skeleton stays the product's shape -- there is no docking and
imgui's own ini file stays disabled -- but the two sidebars and the sidebar's
internal split are draggable, and what the user drags is remembered through
the same debounced :class:`Settings` machinery every other preference uses.

Widths are stored in *design* pixels (divided by ``tokens.SCALE`` on save,
multiplied on load), so a settings file carried between a 100% and a 150%
monitor keeps meaning "a sidebar this wide" rather than drifting.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from . import theme, tokens
from .tokens import sp

SIDEBAR_MIN, SIDEBAR_MAX = 280.0, 480.0
SHARE_MIN, SHARE_MAX = 0.25, 0.75
GRIP = 7.0  # hit-zone width in design px; the drawn line is 1px


class Layout:
    """The user's pane sizes, in design pixels."""

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        stored = settings.get("layout") or {}

        def width(key: str) -> float:
            try:
                value = float(stored.get(key, 340.0))
            except (TypeError, ValueError):
                value = 340.0
            return min(max(value, SIDEBAR_MIN), SIDEBAR_MAX)

        self.sidebar_w = width("sidebar_w")
        self.inspector_w = width("inspector_w")
        try:
            share = float(stored.get("settings_share", 0.55))
        except (TypeError, ValueError):
            share = 0.55
        self.settings_share = min(max(share, SHARE_MIN), SHARE_MAX)

    def save(self) -> None:
        self._settings.set(
            "layout",
            {
                "sidebar_w": round(self.sidebar_w, 1),
                "inspector_w": round(self.inspector_w, 1),
                "settings_share": round(self.settings_share, 3),
            },
        )


def splitter(split_id: str, *, vertical: bool = True, length: float = 0.0) -> float:
    """A drag handle between two panes. -> this frame's drag delta in design px.

    ``vertical`` is the *bar's* orientation: a vertical bar sits between
    columns and drags horizontally. The bar is an ``invisible_button`` with a
    1px line drawn down its middle, accent-coloured while hovered or dragged.
    """
    grip = sp(GRIP)
    if length <= 0:
        avail = imgui.get_content_region_avail()
        length = avail.y if vertical else avail.x
        length = max(length, 1.0)
    size = (grip, length) if vertical else (length, grip)
    pos = imgui.get_cursor_screen_pos()
    imgui.invisible_button(f"##split/{split_id}", size)
    active = imgui.is_item_active()
    hovered = imgui.is_item_hovered()
    if hovered or active:
        imgui.set_mouse_cursor(
            (imgui.MouseCursor_.resize_ew if vertical else imgui.MouseCursor_.resize_ns).value
        )
    colour = theme.rgba(theme.ACCENT) if (hovered or active) else theme.rgba(theme.EDGE, 0.6)
    draw = imgui.get_window_draw_list()
    if vertical:
        x = pos.x + grip * 0.5
        draw.add_line((x, pos.y), (x, pos.y + length), imgui.get_color_u32(colour), sp(1))
    else:
        y = pos.y + grip * 0.5
        draw.add_line((pos.x, y), (pos.x + length, y), imgui.get_color_u32(colour), sp(1))
    if not active:
        return 0.0
    delta = imgui.get_io().mouse_delta
    return (delta.x if vertical else delta.y) / tokens.SCALE
