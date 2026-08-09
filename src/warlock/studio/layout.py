"""The three-column skeleton's measurements.

There is no docking and imgui's own ini file stays disabled. The two sidebars
are a **fixed** ``SIDEBAR_W`` wide rather than draggable: they hold forms, and
a form has one width that reads well -- what a drag bought was a way to make
the app look broken. The one split that survives is the sidebar's *internal*
horizontal one (``settings_share``), because the two stacked scrollers there
genuinely trade against each other, and that is remembered through the same
debounced :class:`Settings` machinery every other preference uses.

Sizes are in *design* pixels, multiplied by ``tokens.SCALE`` at use, so they
keep meaning "a sidebar this wide" on a 150% monitor rather than drifting.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from . import motion, theme, tokens
from .tokens import sp

# The three sidebar widths on offer (M106), in design pixels. Three named
# options rather than a drag, for the reason the module docstring gives: a form
# has a width that reads well, and what a free drag bought was a way to make
# the app look broken. What it did *not* answer is that 300 reads well on a
# 1600-wide window and wastes a third of a 5120 one -- three sizes is enough to
# cover that without reopening the drag.
SIDEBAR_WIDTHS: dict[str, float] = {
    "narrow": 260.0,
    "default": 300.0,
    "wide": 360.0,
}

# The width in force. Module state, set by ``Layout`` at construction and when
# the option changes, exactly as ``tokens.SCALE`` is -- eight call sites read
# this directly and threading a Layout through all of them would put the
# measurement in eight places instead of one.
SIDEBAR_W = SIDEBAR_WIDTHS["default"]
# What ``SIDEBAR_W`` is on its way to. The two are the same number except
# during the ~200 ms after the user picks a different size; see :func:`tick`.
SIDEBAR_TARGET = SIDEBAR_W
# The motion key the width eases on, named here because ``set_sidebar`` has to
# forget it when it snaps.
_SIDEBAR_KEY = "layout/sidebar"


def set_sidebar(key: str, *, animate: bool = False) -> str:
    """Apply a named sidebar width. -> the key actually applied.

    An unknown key falls back to the default rather than raising: this is read
    from a settings file, and a value written by a build that offered a fourth
    size must not stop the window opening.

    ``animate`` is the split between the two callers and it is not cosmetic.
    The construction path (and every headless test) has to leave ``SIDEBAR_W``
    correct *immediately* -- there may be no frame loop at all -- so it snaps;
    the user changing the option in Settings goes through :func:`tick`, where
    the two columns slide to their new widths instead of jumping.
    """
    global SIDEBAR_W, SIDEBAR_TARGET
    key = key if key in SIDEBAR_WIDTHS else "default"
    SIDEBAR_TARGET = SIDEBAR_WIDTHS[key]
    if animate:
        # Stated rather than assumed: a key that is not live yet snaps on its
        # first sighting, so an animated change made before the first tick --
        # which is every change made in the same frame the option is read back
        # -- would jump. Seeding says where the slide starts.
        motion.seed(_SIDEBAR_KEY, SIDEBAR_W)
    else:
        SIDEBAR_W = SIDEBAR_TARGET
        # Forgotten rather than assigned: motion snaps a key's first sighting,
        # so dropping it is how the next tick starts settled at the new width
        # rather than easing from the old one.
        motion.forget(_SIDEBAR_KEY)
    return key


def tick() -> None:
    """Advance the sidebar width one frame. Called once, before anything reads
    ``SIDEBAR_W`` -- a half-eased width read by the left column and the settled
    one by the right would be two columns disagreeing about the same frame."""
    global SIDEBAR_W
    SIDEBAR_W = motion.value(_SIDEBAR_KEY, SIDEBAR_TARGET, duration=tokens.DUR_BASE)


PANE_PADDING = 5.0  # between a pane's border and its content
SHARE_MIN, SHARE_MAX = 0.25, 0.75
GRIP = 7.0  # hit-zone width in design px; the drawn line is 1px


class Layout:
    """The one pane proportion the user still controls."""

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        stored = settings.get("layout") or {}
        try:
            share = float(stored.get("settings_share", 0.55))
        except (TypeError, ValueError):
            share = 0.55
        self.settings_share = min(max(share, SHARE_MIN), SHARE_MAX)
        self.sidebar = set_sidebar(str(stored.get("sidebar", "default")))

    def set_sidebar_width(self, key: str) -> None:
        self.sidebar = set_sidebar(key, animate=True)
        self.save()

    def save(self) -> None:
        # Only the surviving keys: Settings.set replaces the whole dict, so the
        # stale sidebar_w/inspector_w a settings file may still carry are gone
        # the first time anything saves. ``sidebar`` is the *name* of a width,
        # never a number, so a stored value can never be a size this build does
        # not offer.
        self._settings.set(
            "layout",
            {"settings_share": round(self.settings_share, 3), "sidebar": self.sidebar},
        )


def pane_child(pane_id: str, size: tuple[float, float], window_flags: int = 0) -> bool:
    """A bordered pane child with ``PANE_PADDING`` between its edge and content.

    The padding is pushed and popped around ``begin_child`` alone: window
    padding latches at begin, so a popup or tooltip opened *inside* the pane
    keeps the theme's own padding rather than inheriting this one. Pushing it
    globally in theme.py would shrink every modal in the app.
    """
    pad = sp(PANE_PADDING)
    imgui.push_style_var(imgui.StyleVar_.window_padding.value, (pad, pad))
    open_ = imgui.begin_child(pane_id, size, imgui.ChildFlags_.borders.value, window_flags)
    imgui.pop_style_var()
    return open_


def centre_width() -> float:
    """What is left for the centre pane once the right sidebar is reserved.

    Called with the cursor between the left sidebar and the centre, so the
    available width still contains the centre, one item spacing and the right
    sidebar. The right sidebar itself is then sized ``(0, h)`` and resolves to
    ``SIDEBAR_W`` because this reserved exactly that.
    """
    spacing = imgui.get_style().item_spacing.x
    return max(imgui.get_content_region_avail().x - (sp(SIDEBAR_W) + spacing), sp(300))


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
    # Faded rather than switched: a hairline that snaps from edge-grey to full
    # accent as the pointer crosses it is the loudest hard cut left in the
    # workspace, and it fires on every pass over the column boundary whether or
    # not anybody meant to drag.
    lit = motion.value(
        f"splitter/{split_id}", 1.0 if (hovered or active) else 0.0, duration=tokens.DUR_FAST
    )
    colour = theme.mix(theme.EDGE, theme.ACCENT, lit, 0.6 + 0.4 * lit)
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
