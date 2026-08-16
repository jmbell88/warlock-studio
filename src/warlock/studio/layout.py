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

from contextlib import contextmanager
from enum import StrEnum
from typing import Any

from imgui_bundle import imgui

from . import motion, theme, tokens, widgets
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


# What a centre pane wants, and what it will accept before the sidebars start
# giving way instead (UX-01), both in design px. 300 was already the floor in
# ``centre_width`` -- what it was missing is that a floor which cannot be met
# does not stop being claimed, it just pushes the column after it off the
# window. The two numbers separate "this is the comfortable size" from "this
# is the size below which shrinking the centre is the wrong answer".
CENTRE_MIN = 300.0
CENTRE_FLOOR = 220.0

# How narrow a sidebar may be squeezed. Below this a form's labels wrap to one
# word a line, which is a pane nobody can use rather than a small one -- so
# this is where squeezing stops and the centre gives instead.
SIDEBAR_MIN = 200.0

# The width the two sidebars actually get this frame, in *physical* px. Module
# state, computed once by :func:`measure` for the same reason ``SIDEBAR_W`` is
# eased in ``tick``: the left column and the right must agree within a frame,
# and the right one is sized ``(0, h)`` from whatever the centre left behind.
#
# ``None`` until a frame has measured one. Its own sentinel rather than a
# plausible default, so that the headless callers -- tests, and anything asking
# what a sidebar is before there is a window -- get the unconstrained width
# rather than a fit computed against a viewport of zero.
SIDEBAR_FIT: float | None = None

# What the navigation rail has taken out of the window this frame, in physical
# px (the UI redesign, wave 3). Module state set by ``rail.tick`` immediately before
# :func:`tick`, for the reason ``SIDEBAR_W`` is: the sidebars are measured from
# what is left after the rail, and a number threaded through would be the same
# figure in two places.
#
# Zero by default, which is what makes every headless caller -- ``fit`` is pure
# and the tests drive it directly -- see the window it always saw. A rail that
# has never been drawn has taken nothing.
RAIL_RESERVED: float = 0.0


def fit(available: float, spacing: float) -> float:
    """How wide each sidebar can be, given the room. Physical px, pure.

    The whole of UX-01 is that this used to be unconditional. Three columns
    reserved two 300-design-px sidebars and floored the centre at 300 more, so
    at 1.5x the workspace demanded ~1350 physical px and at 2x ~1800 -- while
    the resize floor follows the *monitor's* scale alone and stays near 1100.
    (Deliberately: see ``main._min_window_size``. A floor that multiplied the
    user's zoom in demanded a window bigger than a 1080p display and refused to
    shrink, which is worse.) The arithmetic simply overflowed, and since the
    right-hand column is the one sized from the leftovers, the pane that fell
    off the edge was always the inspector.

    So the columns give way in a stated order: the sidebars narrow first, down
    to :data:`SIDEBAR_MIN`, and only then does the centre drop below
    :data:`CENTRE_MIN`. Somebody who enlarges the UI to read it ends up with
    three narrow columns rather than two comfortable ones and a third they
    cannot reach.
    """
    want = sp(SIDEBAR_W)
    if available >= want * 2 + sp(CENTRE_MIN) + spacing * 2:
        return want
    # What is left for the two sidebars once the centre keeps its comfortable
    # width -- which may be negative, hence the floor rather than a clamp.
    slack = (available - sp(CENTRE_MIN) - spacing * 2) / 2.0
    return max(min(slack, want), sp(SIDEBAR_MIN))


def tick() -> None:
    """Advance the sidebar width one frame. Called once, before anything reads
    ``SIDEBAR_W`` -- a half-eased width read by the left column and the settled
    one by the right would be two columns disagreeing about the same frame."""
    global SIDEBAR_W
    SIDEBAR_W = motion.value(_SIDEBAR_KEY, SIDEBAR_TARGET, duration=tokens.DUR_BASE)


def measure() -> float:
    """Settle this frame's sidebar fit. Called once, straight after :func:`tick`.

    Separate from ``tick`` rather than folded into it, and the reason is not
    tidiness: ``tick`` is pure arithmetic over the motion table and is called
    by headless tests that have no imgui context at all, where reaching for a
    viewport is an access violation rather than an error.

    Measured from the whole window rather than from the content region a column
    happens to be standing in: ``centre_width`` runs *after* the left sidebar is
    drawn, so recomputing there would fit the right sidebar against a width the
    left has already taken its share of, and the two columns would come out
    different widths.
    """
    global SIDEBAR_FIT
    style = imgui.get_style()
    room = imgui.get_main_viewport().work_size[0] - style.window_padding.x * 2
    if RAIL_RESERVED:
        # The rail and the gap after it, both taken off before the columns are
        # fitted. Measured rather than assumed because the rail eases between
        # two widths: fitting against the settled figure would leave the three
        # columns wrong for the ~200 ms of every expand.
        room -= RAIL_RESERVED + style.item_spacing.x
    SIDEBAR_FIT = fit(room, style.item_spacing.x)
    return SIDEBAR_FIT


# Between a pane's border and its content. 5 was the most utilitarian number in
# the codebase: it made a pane's contents start one pixel-and-a-bit from the
# hairline around them, which is what a mixer strip wants and not what a form
# does. SP_3 rather than the window's own SP_4 (UX.md Phase 2) because a pane is
# already inside the host window's gutter -- matching it would double the inset
# on the two sidebars, which are the width-constrained case.
PANE_PADDING = tokens.SP_3
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
        # Whether the navigation rail shows its labels. A *name* rather than a
        # width, exactly as ``sidebar`` is, so a stored value can never be a
        # size this build does not offer -- and anything unrecognised reads as
        # "icons", which is the state the rail is designed around rather than a
        # degraded version of the other one.
        self.rail = "labels" if str(stored.get("rail", "icons")) == "labels" else "icons"

    def set_sidebar_width(self, key: str) -> None:
        self.sidebar = set_sidebar(key, animate=True)
        self.save()

    def set_rail(self, key: str) -> None:
        self.rail = "labels" if key == "labels" else "icons"
        self.save()

    def save(self) -> None:
        # Only the surviving keys: Settings.set replaces the whole dict, so the
        # stale sidebar_w/inspector_w a settings file may still carry are gone
        # the first time anything saves. ``sidebar`` is the *name* of a width,
        # never a number, so a stored value can never be a size this build does
        # not offer -- and ``rail`` follows it for the same reason. Both have to
        # be written here every time, because the whole dict is replaced: a key
        # this method forgets is a preference that silently resets the next time
        # the *other* one changes.
        self._settings.set(
            "layout",
            {
                "settings_share": round(self.settings_share, 3),
                "sidebar": self.sidebar,
                "rail": self.rail,
            },
        )


class PaneRole(StrEnum):
    """Semantic surface role for a major region of a workspace."""

    SIDEBAR = "sidebar"
    CONTENT = "content"
    PRIMARY = "content"  # the plan's prose name; CONTENT reads better at calls
    INSPECTOR = "inspector"
    SHEET = "sheet"
    OVERLAY = "overlay"


class PaneEdge(StrEnum):
    """The single major boundary a pane owns, if any."""

    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


def _pane_fill(role: PaneRole) -> int:
    if role in (PaneRole.SIDEBAR, PaneRole.INSPECTOR, PaneRole.SHEET):
        return theme.PANEL
    if role is PaneRole.OVERLAY:
        return theme.ELEV_2
    return theme.BG


def _divider(edge: PaneEdge) -> None:
    if edge is PaneEdge.NONE:
        return
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    if edge is PaneEdge.LEFT:
        start, end = (low.x, low.y), (low.x, high.y)
    elif edge is PaneEdge.RIGHT:
        start, end = (high.x, low.y), (high.x, high.y)
    elif edge is PaneEdge.TOP:
        start, end = (low.x, low.y), (high.x, low.y)
    else:
        start, end = (low.x, high.y), (high.x, high.y)
    imgui.get_window_draw_list().add_line(
        start,
        end,
        imgui.get_color_u32(theme.rgba(theme.DIVIDER)),
        sp(tokens.DIVIDER_WIDTH),
    )


@contextmanager
def pane(
    pane_id: str,
    size: tuple[float, float],
    role: PaneRole | str = PaneRole.CONTENT,
    *,
    edge: PaneEdge | str = PaneEdge.NONE,
    window_flags: int = 0,
):
    """Begin a role-aware major pane and close it reliably.

    Child borders stay disabled.  A caller names at most the one edge that
    separates this pane from another major region, which avoids turning every
    stacked child into a bordered card.

    ``always_use_window_padding`` goes with that borderlessness rather than
    beside it: Dear ImGui zeroes a *borderless* child's window padding, so the
    ``PANE_PADDING`` pushed two lines below had no effect at all and every
    pane's content sat flush against the column edge. Two panes had already
    worked around it locally (``panes/app_settings``, ``panes/landing``) --
    those two draw into ``##content`` rather than through here, so they keep
    their own flag.
    """

    try:
        resolved_role = role if isinstance(role, PaneRole) else PaneRole(role)
    except ValueError:
        resolved_role = PaneRole.CONTENT
    try:
        resolved_edge = edge if isinstance(edge, PaneEdge) else PaneEdge(edge)
    except ValueError:
        resolved_edge = PaneEdge.NONE
    pad = sp(PANE_PADDING)
    imgui.push_style_var(imgui.StyleVar_.window_padding.value, (pad, pad))
    imgui.push_style_color(
        imgui.Col_.child_bg.value, imgui.ImVec4(*theme.rgba(_pane_fill(resolved_role)))
    )
    visible = imgui.begin_child(
        pane_id, size, imgui.ChildFlags_.always_use_window_padding.value, window_flags
    )
    imgui.pop_style_color()
    imgui.pop_style_var()
    try:
        # Every pane drawn through here groups its headings onto tinted blocks.
        # Here rather than in each pane because it is a property of *being a
        # pane*, and because the scope has to bracket the child exactly: it
        # splits this child's own draw list, which is why two panes can never
        # collide over one splitter. A pane that draws no ``widgets.section``
        # pays a split and a merge and gets no fills, which is free enough not
        # to be worth a predicate. See ``widgets.section_blocks``.
        with widgets.section_blocks():
            yield visible
    finally:
        imgui.end_child()
        _divider(resolved_edge)


def pane_child(
    pane_id: str, size: tuple[float, float], window_flags: int = 0
) -> bool:
    """Compatibility entry point for third-party panes and older tests.

    Studio code uses :func:`pane`; this retains the old begin/end shape for
    callers outside the migration boundary. It is intentionally borderless --
    and carries ``always_use_window_padding`` for that reason, exactly as
    :func:`pane` does.
    """

    pad = sp(PANE_PADDING)
    imgui.push_style_var(imgui.StyleVar_.window_padding.value, (pad, pad))
    visible = imgui.begin_child(
        pane_id, size, imgui.ChildFlags_.always_use_window_padding.value, window_flags
    )
    imgui.pop_style_var()
    return visible


def sidebar_width() -> float:
    """A sidebar's width this frame, in physical px. Narrowed to fit (UX-01).

    What the seven workspaces call instead of ``sp(SIDEBAR_W)``. Same value
    whenever there is room, which is the ordinary case; the difference only
    shows at high UI scale in a small window, which is exactly the case a user
    who enlarged the UI to read it is in.
    """
    return sp(SIDEBAR_W) if SIDEBAR_FIT is None else SIDEBAR_FIT


def centre_width() -> float:
    """What is left for the centre pane once the right sidebar is reserved.

    Called with the cursor between the left sidebar and the centre, so the
    available width still contains the centre, one item spacing and the right
    sidebar. The right sidebar itself is then sized ``(0, h)`` and resolves to
    :data:`SIDEBAR_FIT` because this reserved exactly that.

    The floor is :data:`CENTRE_FLOOR` and not ``CENTRE_MIN``, which is the
    other half of UX-01: a floor is a claim, and a claim that cannot be met
    does not fail -- it silently pushes the *next* column past the window edge.
    Reserving the sidebar's real width and flooring lower is what keeps the
    right-hand pane on screen when the arithmetic is tight.
    """
    spacing = imgui.get_style().item_spacing.x
    return max(
        imgui.get_content_region_avail().x - (sidebar_width() + spacing),
        sp(CENTRE_FLOOR),
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
