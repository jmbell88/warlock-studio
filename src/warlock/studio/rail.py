"""The navigation rail: a column of glyphs down the left edge.

What it replaces is a horizontal segmented control across the top of every
frame, holding thirteen destinations. Three things were wrong with that and
only one of them was fixable inside it.

**Thirteen is too many to be a row.** The switch had already been compacted to
glyph-only at 1.5 UI scale (a clipped segment is an unreachable mode), so on
any scaled display it was thirteen unlabelled pictures in a line -- and it was
drawn identically in every mode, which is to say it spent the top of the window
on navigation that never changes. A rail costs a 52 px column instead of a full
row, keeps its labels available on demand, and stops competing with the pane
title underneath it.

**The header carried three things that are not navigation.** A frame-rate and
VRAM readout (developer chrome, on screen permanently), a shortcuts button and
a power icon. The readout is gone -- F10 still raises ``overlay.fps_meter``,
which is the tool that was always the real answer -- and the quit icon is gone
with it: a destructive control one click from every mode was mitigated by a
confirm rather than fixed, and the window's own X is the path that carries the
unsaved-work chain.

**Grouping had to be spacing.** The old control derived its gaps from where
``WORK_MODES`` changed, which is a rule that renders correctly and explains
nothing. ``modes.RAIL_GROUPS`` is written out instead: the two ways in and the
places you look at work, the six creative workspaces, and Settings alone in the
footer. A column can carry that as *sections*, which is a stronger statement
than a wider gap in a row.

**The rail takes no arrow keys, deliberately.** ``modes.NAV_KEY_MODES`` names
five surfaces that bind the arrows themselves (Home's Resume list, the Library,
Review's unit stepper, Inker's and Plotter's pan), and a shell control that
claimed Up/Down would be a sixth claimant arbitrating against all of them. Mode
switching is a mouse action and a ``Ctrl+K`` action, which is the doctrine
``modes.py`` already states for the digits. If that is ever revisited, the shape
is a focus flag OR'd into the input door (``imgui_backend._NAV_KEYS``) rather
than a rail-specific key handler -- one arbitration point, not two.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from . import icons, modes, motion, theme, tokens
from . import layout as layout_mod
from .tokens import sp

# The two widths, in design px. 52 is a 44 px hit target plus its gutters --
# the smallest a glyph-only column can be while every item still clears the
# 24 px minimum a control is comfortably clickable at, with room for the
# selected pill to inset. 188 is what the longest label ("Packwright") needs
# beside that glyph without truncating, which is the only figure an expanded
# rail can honestly be: a rail that ellipsises its labels has spent the width
# and not bought the words.
RAIL_W = 52.0
RAIL_EXPANDED_W = 188.0

# One item's height, and the gap between the groups.
ITEM_H = 44.0
# What an item is allowed to be compressed to when the window is too short for
# every row at full height -- see :func:`fitted_height`. 24 design px is the
# figure the app already uses for "the minimum anything is comfortably
# clickable at" (it is why the old health dot grew a hit area at all, F58), so
# it is a floor with a citation rather than a number picked to make the
# arithmetic come out. Below it the column scrolls instead, because a target
# nobody can hit is not a smaller version of one they can.
MIN_ITEM_H = 24.0
GROUP_GAP = tokens.SP_3

# How far the selected pill is inset from the item's box on each side.
PILL_INSET = 3.0

_WIDTH_KEY = "layout/rail"
_PILL_KEY = "rail/pill"

# The eased width, in design px. Module state exactly as ``layout.SIDEBAR_W``
# is, and for its reason: the rail is drawn in one place but *measured* by
# another (``layout.measure``, before any column exists), so threading it would
# put one number in two places.
_WIDTH = [RAIL_W]

# What the rail asked for at host scope this frame. imgui popups are registered
# in the window that opens them, and the rail is a child -- so rather than
# discover that the hard way, the two popups the footer raises are flagged here
# and opened by ``main`` after ``end_child``. The same one-shot pattern
# ``state.shortcuts_requested`` already uses, for the same reason.
_wants: dict[str, bool] = {"diagnostics": False}


def request(name: str) -> None:
    """Ask ``main`` to open one of the rail's popups at host scope.

    Public because the rail's footer is not the only door: the health badge is
    drawn *only* when a check is failing, so on a healthy install the
    Issues list would be unreachable -- and "everything is fine" is
    exactly the claim somebody occasionally wants to see the evidence for. The
    palette's Issues command comes through here.
    """
    _wants[name] = True


def take(name: str) -> bool:
    """Consume a one-shot request. -> whether it was set."""
    asked = _wants.get(name, False)
    _wants[name] = False
    return asked


def expanded_fits() -> bool:
    """Whether the labelled rail leaves the three columns a usable width.

    The preference is untouched by this -- a window dragged narrow collapses the
    rail and dragging it back restores the labels, because what the user chose
    and what fits are two different facts. At ``main.MIN_SIZE`` and 1.5 UI scale
    the expanded rail does not fit, which is what this exists for: without it,
    the inspector is the column that falls off the edge (UX-01's whole story,
    one column further left).
    """
    if imgui.get_current_context() is None:
        return True
    style = imgui.get_style()
    need = (
        sp(RAIL_EXPANDED_W)
        + sp(layout_mod.SIDEBAR_MIN) * 2
        + sp(layout_mod.CENTRE_MIN)
        + style.item_spacing.x * 3
        + style.window_padding.x * 2
    )
    return imgui.get_main_viewport().work_size[0] >= need


def tick(layout: Any) -> float:
    """Settle the rail's width for this frame. -> design px.

    Called *before* ``layout.tick``/``layout.measure``, because how wide the
    sidebars can be is a fact about what is left after the rail has taken its
    column -- and a rail measured afterwards would leave the two sidebars
    disagreeing with the window by exactly its own width for one frame every
    time it was toggled.
    """
    want = RAIL_EXPANDED_W if layout.rail == "labels" and expanded_fits() else RAIL_W
    _WIDTH[0] = motion.value(_WIDTH_KEY, want, duration=tokens.DUR_BASE)
    layout_mod.RAIL_RESERVED = sp(_WIDTH[0])
    return _WIDTH[0]


def width() -> float:
    """This frame's rail width in physical px."""
    return sp(_WIDTH[0])


def _label_alpha() -> float:
    """How visible the labels are, from how far the rail has opened.

    Tied to the *drawn* width rather than to the preference, so the words fade
    in with the column instead of appearing at the end of the slide -- and so a
    forced collapse takes them away on the way rather than at the finish.
    """
    span = RAIL_EXPANDED_W - RAIL_W
    return min(max((_WIDTH[0] - RAIL_W) / span, 0.0), 1.0) if span else 0.0


def fitted_height(rows: int, gaps: float, avail: float) -> float:
    """How tall one item is drawn, given how many have to fit. Physical px.

    The vertical half of ``segmented_control``'s rule, and it exists for the
    same failure: eleven items at ``ITEM_H`` plus a footer need about 640
    design px, and at the resize floor on a 175 % display there are roughly
    370 -- so the bottom of the rail (Packwright, Settings, the expand toggle)
    was simply drawn past the end of the column, which clips. A clipped item is
    an unreachable mode whichever axis it runs off.

    So the items *compress* instead, down to :data:`MIN_ITEM_H`. Below that the
    column scrolls, which is the honest last resort rather than a design: a
    26 dp row is small and reachable, a 12 dp row is a target nobody can hit.
    Pure, so the arithmetic is testable at every window size.
    """
    if rows <= 0:
        return sp(ITEM_H)
    room = (avail - gaps) / rows
    return max(min(sp(ITEM_H), room), sp(MIN_ITEM_H))


def _item(
    key: str,
    label: str,
    icon: str,
    box: float,
    *,
    selected: bool,
    tooltip: str = "",
    height: float = 0.0,
) -> bool:
    """One row of the rail. -> whether it was clicked.

    An ``invisible_button`` with the glyph and label drawn under it by hand,
    rather than a button with a label: the two have to sit at fixed positions
    inside the box (the glyph centred in the collapsed column's width, so it
    does not move when the labels arrive) and imgui's own alignment cannot say
    "centre this one thing and left-align that one" inside a single item.
    """
    height = height or sp(ITEM_H)
    origin = imgui.get_cursor_screen_pos()
    clicked = imgui.invisible_button(f"rail/{key}", (box, height))
    hovered = imgui.is_item_hovered()
    focused = imgui.is_item_focused() and imgui.get_io().nav_visible
    lit = motion.value(
        f"rail/{key}/hover", 1.0 if (hovered or focused) else 0.0, duration=tokens.DUR_FAST
    )
    draw = imgui.get_window_draw_list()
    if lit > 0.0 and not selected:
        draw.add_rect_filled(
            (origin.x + sp(PILL_INSET), origin.y + sp(PILL_INSET)),
            (origin.x + box - sp(PILL_INSET), origin.y + height - sp(PILL_INSET)),
            imgui.get_color_u32(theme.rgba(theme.ELEV_1, lit)),
            sp(tokens.RADIUS_M),
        )
    alpha = 1.0 if selected else (0.85 if lit > 0.0 else 0.65)
    colour = imgui.get_color_u32(theme.rgba(theme.TEXT if selected else theme.MUTED, alpha))
    glyph_w = imgui.calc_text_size(icon)
    draw.add_text(
        (
            origin.x + (sp(RAIL_W) - glyph_w.x) * 0.5 - sp(tokens.SP_1),
            origin.y + (height - glyph_w.y) * 0.5,
        ),
        colour,
        icon,
    )
    words = _label_alpha()
    if words > 0.0:
        size = imgui.calc_text_size(label)
        draw.add_text(
            (origin.x + sp(RAIL_W) - sp(tokens.SP_1), origin.y + (height - size.y) * 0.5),
            imgui.get_color_u32(
                theme.rgba(theme.TEXT if selected else theme.MUTED, alpha * words)
            ),
            label,
        )
    # The accessible name, for as long as the glyph is the only thing drawn.
    # Suppressed once the label is legible, because a tooltip repeating a word
    # that is already on screen is noise -- unless the caller has something
    # more to say, which is what ``tooltip`` is for.
    if hovered and (tooltip or words < 0.5):
        imgui.set_tooltip(tooltip or label)
    if hovered:
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand.value)
    return clicked


def _health(ctx: Any) -> tuple[int, str] | None:
    """The failing-checks badge, or ``None`` when everything passes.

    A permanent "OK" badge is noise -- the app said "Everything checks out" in
    the corner of every frame -- so the footer carries this only when there is
    something to carry. The diagnostics list stays reachable when it is green
    through the palette, which is where a thing you occasionally want and never
    need belongs.
    """
    checks = list(getattr(ctx.runtime, "checks", []) or [])
    failing = [c for c in checks if not c.ok]
    if not failing and not ctx.state.errors:
        return None
    fatal = bool(ctx.state.errors) or any(c.fatal for c in failing)
    names = "\n".join(f"  {'!' if c.fatal else '-'} {c.name}" for c in failing[:8])
    more = f"\n  ... and {len(failing) - 8} more" if len(failing) > 8 else ""
    tip = "Click for details:" + (f"\n{names}{more}" if names else "")
    return (theme.ERR if fatal else theme.WARN, tip)


def draw(app: Any, ctx: Any) -> None:
    """The whole column, inside the host window. Nothing else is a header."""
    from .manual import render as manual_render

    box = width()
    pad = sp(tokens.SP_1)
    imgui.push_style_var(imgui.StyleVar_.window_padding.value, (pad, pad))
    open_ = imgui.begin_child("##rail", (box, 0), imgui.ChildFlags_.borders.value)
    imgui.pop_style_var()
    if not open_:
        imgui.end_child()
        return
    item_w = imgui.get_content_region_avail().x
    labels = {key: (label, icon) for key, label, icon in modes.MODES}
    body_groups = modes.RAIL_GROUPS[:-1]
    footer = list(modes.RAIL_GROUPS[-1])
    badge = _health(ctx)
    # Help and the expand toggle are always there; the badge is not.
    footer_rows = (["health"] if badge else []) + ["help", *footer, "expand"]
    rows = sum(len(g) for g in body_groups) + len(footer_rows)
    avail_h = imgui.get_content_region_avail().y
    # The gaps the column will actually contain: one between each pair of rows,
    # one wider one between the body's two groups, and one more between the
    # body and the footer.
    #
    # **The air goes before the items do.** A short window closes the
    # row-to-row spacing first and only then compresses the rows themselves,
    # which is ``layout.fit``'s ladder one axis over: give up the thing that is
    # only breathing room before giving up the thing that is the control. The
    # group gaps survive both, because they are what says the rail has
    # sections at all.
    group_gaps = sp(GROUP_GAP) * len(body_groups)
    gap = imgui.get_style().item_spacing.y
    if rows * sp(ITEM_H) + (rows - 1) * gap + group_gaps > avail_h:
        gap = 0.0
    item_h = fitted_height(rows, (rows - 1) * gap + group_gaps, avail_h)
    step = item_h + gap
    # Stated to imgui as well as used in the arithmetic, or the two disagree by
    # one spacing per row and the footer lands somewhere else entirely.
    imgui.push_style_var(
        imgui.StyleVar_.item_spacing.value, (imgui.get_style().item_spacing.x, gap)
    )

    # Where the selected pill is going, computed before anything is drawn: the
    # positions are arithmetic, so the target is known without waiting a frame
    # for the item to have been laid out. (``card`` and the glyph buttons read
    # last frame's hover because a *colour* is needed to draw the thing that
    # would tell you; a position is not in that bind.)
    top = imgui.get_cursor_screen_pos()
    start_y = imgui.get_cursor_pos_y()
    offsets: dict[str, float] = {}
    y = 0.0
    for index, group in enumerate(body_groups):
        if index:
            y += sp(GROUP_GAP)
        for key in group:
            offsets[key] = y
            y += step
    # ``y`` carries a trailing gap that belongs *between* rows; the body ends
    # at its last item's bottom edge, not one spacing past it.
    body_h = y - gap

    footer_h = len(footer_rows) * item_h + (len(footer_rows) - 1) * gap
    footer_top = max(body_h + sp(GROUP_GAP), avail_h - footer_h)
    for offset, key in enumerate(footer_rows):
        offsets[key] = footer_top + offset * step

    current = ctx.state.mode
    pill = motion.value(_PILL_KEY, offsets.get(current, 0.0), duration=tokens.DUR_BASE)
    if current in offsets:
        imgui.get_window_draw_list().add_rect_filled(
            (top.x + sp(PILL_INSET), top.y + pill + sp(PILL_INSET)),
            (top.x + item_w - sp(PILL_INSET), top.y + pill + item_h - sp(PILL_INSET)),
            imgui.get_color_u32(theme.rgba(theme.ELEV_2)),
            sp(tokens.RADIUS_M),
        )

    for index, group in enumerate(body_groups):
        if index:
            imgui.dummy((0, sp(GROUP_GAP) - gap))
        for key in group:
            label, icon = labels[key]
            if _item(key, label, icon, item_w, selected=key == current, height=item_h):
                app._set_mode(key)

    # Placed against the measured start rather than nudged from wherever the
    # body left the cursor: a delta accumulates whatever rounding the rows
    # above it collected, which is how a footer comes to sit six pixels past
    # the bottom of the column it is supposed to be pinned to.
    imgui.set_cursor_pos_y(start_y + footer_top)
    if badge:
        colour, tip = badge
        if _item("health",
            "Issues",
            icons.TRIANGLE_ALERT,
            item_w,
            selected=False,
            tooltip=tip,
            height=item_h,
        ):
            request("diagnostics")
        # The glyph again in the failing colour, over the muted one the shared
        # item drew: a badge that is the same grey as everything above it is a
        # badge nobody looks at, and colour is the fast read here even though
        # the shape carries it for anyone the colour does not reach.
        _tint_last(colour)
    if _item("help",
        "Manual",
        icons.BOOK_OPEN,
        item_w,
        selected=ctx.state.manual.open,
        tooltip="Manual (F1)",
        height=item_h,
    ):
        manual_render.toggle(ctx)
    for key in footer:
        label, icon = labels[key]
        if _item(key, label, icon, item_w, selected=key == current, height=item_h):
            app._set_mode(key)
    expanded = app.layout.rail == "labels"
    # ``ARROW_LEFT`` rather than a chevron pointing the other way: the vendored
    # lucide subset carries chevron-right and chevron-down and no chevron-left,
    # and a glyph that is not in the font is a blank square. Two families in one
    # toggle is the smaller cost -- both point the way the rail is about to move,
    # which is the whole of what the control has to say.
    if _item("expand",
        "Collapse" if expanded else "Expand",
        icons.ARROW_LEFT if expanded else icons.CHEVRON_RIGHT,
        item_w,
        selected=False,
        height=item_h,
    ):
        app.layout.set_rail("icons" if expanded else "labels")
    imgui.pop_style_var()
    imgui.end_child()


def _tint_last(colour: int) -> None:
    """Redraw the previous item's glyph slot in ``colour``.

    A small hack with a stated cost: it paints a filled circle rather than the
    glyph, because the glyph was already drawn by ``_item`` and a second copy
    over it would only thicken it. What this adds is the *colour*, which is the
    half a monochrome rail cannot say -- and it is a dot beside a triangle
    rather than instead of it, so the double encoding the status pills use
    holds here too.
    """
    lo = imgui.get_item_rect_min()
    imgui.get_window_draw_list().add_circle_filled(
        (lo.x + sp(RAIL_W) - sp(tokens.SP_2), lo.y + sp(tokens.SP_3)),
        sp(3.5),
        imgui.get_color_u32(theme.rgba(colour)),
        12,
    )


