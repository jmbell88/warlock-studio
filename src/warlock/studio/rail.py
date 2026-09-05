"""The compact workspace navigation rail.

The 44 px rail contains destinations only. Utilities live in the global menu
or shared status bar, leaving every row a full 44 px target with a tooltip and
focus treatment. Fresh installs use glyphs; a saved labelled preference is
preserved and temporarily collapses only when the editor would no longer fit.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from . import anchors, fonts, modes, motion, theme, tokens
from . import layout as layout_mod
from .tokens import sp

# The two widths, in design px. The compact column itself is one 44 px hit
# target wide. 188 is what the longest label ("Packwright") needs
# beside that glyph without truncating, which is the only figure an expanded
# rail can honestly be: a rail that ellipsises its labels has spent the width
# and not bought the words.
RAIL_W = 44.0
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

# Host-scoped popup requests. They originate in the global menu and avoid
# coupling those surfaces to main.
_wants: dict[str, bool] = {"layouts": False}


def request(name: str) -> None:
    """Ask ``main`` to open a shell popup at host scope."""
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
        # ``PANEL_MIN``, not the older ``SIDEBAR_MIN``: the columns are fitted
        # by ``layout.fit_widths`` now, which holds 220 dp a side and only goes
        # under it as a last resort. Asking 200 here promised the labels fit at
        # widths where the real columns still wanted 40 dp more each, and the
        # 20 dp it bought back is not worth a rail that opens into a squeeze.
        + sp(layout_mod.PANEL_MIN) * 2
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
    badge: str = "",
    height: float = 0.0,
    enabled: bool = True,
) -> bool:
    """One row of the rail. -> whether it was clicked.

    ``enabled=False`` dims the row and marks it, but **keeps the button live**.
    imgui shows nothing for a genuinely disabled item, and a rail item that
    neither opens nor explains itself is the worst of both -- so the click is
    still reported and the caller sends it somewhere useful (Settings, with the
    missing rows already ticked) rather than opening the mode.

    An ``invisible_button`` with the glyph and label drawn under it by hand,
    rather than a button with a label: the two have to sit at fixed positions
    inside the box (the glyph centred in the collapsed column's width, so it
    does not move when the labels arrive) and imgui's own alignment cannot say
    "centre this one thing and left-align that one" inside a single item.
    """
    height = height or sp(ITEM_H)
    origin = imgui.get_cursor_screen_pos()
    clicked = imgui.invisible_button(f"rail/{key}", (box, height))
    # Straight after the button, because that is the only moment imgui can be
    # asked where it went -- and every rail item is a thing a tour can point
    # at, so this is one mark rather than eleven at the call sites.
    anchors.mark(f"rail/{key}")
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
    if not enabled:
        # Dimmer than the dimmest live row, so "not available" reads as a
        # different state rather than as "not hovered".
        alpha *= 0.45
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
        text_x = origin.x + sp(RAIL_W) - sp(tokens.SP_1)
        draw.add_text(
            (text_x, origin.y + (height - size.y) * 0.5),
            imgui.get_color_u32(theme.rgba(theme.TEXT if selected else theme.MUTED, alpha * words)),
            label,
        )
        if badge:
            _badge(
                draw, badge, text_x + size.x + sp(tokens.SP_2),
                origin, box, height, alpha * words,
            )
    # The accessible name, for as long as the glyph is the only thing drawn.
    # Suppressed once the label is legible, because a tooltip repeating a word
    # that is already on screen is noise -- unless the caller has something
    # more to say, which is what ``tooltip`` is for.
    #
    # Every rail item now has something more to say (``modes.PURPOSE``), so the
    # collapsed rail prepends the label: without it, hovering a bare glyph gave
    # a sentence about a mode whose *name* was the one thing not on screen.
    if hovered and (tooltip or words < 0.5):
        text = tooltip or label
        if tooltip and words < 0.5:
            text = f"{label} — {tooltip}"
        imgui.set_tooltip(text)
    if hovered:
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand.value)
    return clicked


def _badge(
    draw: Any, text: str, x: float, origin: Any,
    box: float, height: float, alpha: float,
) -> None:
    """A small chip after a rail item's label -- "Experimental" and nothing else.

    Drawn onto the draw list rather than through ``widgets._chip``, for
    :func:`_item`'s own reason: this rail's layout is arithmetic over an
    ``invisible_button``, and a chip that emitted real imgui items would both
    advance the cursor inside the row and steal the hover the row is built
    around.

    **Skipped rather than clipped when it does not fit.** The rail narrows to a
    glyph column, and a chip half-drawn over the panel edge reads as a
    rendering fault; the tooltip carries the same word at every width, so
    nothing is lost by staying silent here.
    """
    with fonts.small(imgui):
        size = imgui.calc_text_size(text)
        pad_x, pad_y = sp(5), sp(2)
        width = size.x + pad_x * 2
        if x + width > origin.x + box - sp(tokens.SP_1):
            return
        top = origin.y + (height - (size.y + pad_y * 2)) * 0.5
        colour = theme.rgba(theme.ACCENT, alpha)
        draw.add_rect_filled(
            (x, top),
            (x + width, top + size.y + pad_y * 2),
            imgui.get_color_u32(theme.rgba(theme.ACCENT, alpha * 0.16)),
            (size.y + pad_y * 2) * 0.5,
        )
        draw.add_text((x + pad_x, top + pad_y), imgui.get_color_u32(colour), text)


def _caption(text: str, height: float) -> None:
    """A group's name in the expanded rail, in exactly ``height`` of column.

    Drawn onto the draw list under a bare ``dummy`` rather than emitted as
    ``imgui.text``, for :func:`_item`'s reason: this rail's vertical layout is
    arithmetic, and an item whose height is whatever the font happened to
    measure is a footer six pixels out of place. Left-aligned with the row
    labels, not with the glyphs -- it names the words, not the icons.
    """
    if not text:
        # The footer group's label is deliberately empty (``RAIL_GROUP_LABELS``
        # says why), and before the shell refactor the footer was drawn by a
        # separate path that never reached here. Now every group comes through
        # this loop, so an empty label has to cost nothing -- a bare ``dummy``
        # would spend a full caption row on a word that is not there, which
        # reads as an unexplained gap above Settings.
        return
    origin = imgui.get_cursor_screen_pos()
    with fonts.small(imgui):
        size = imgui.calc_text_size(text)
        imgui.get_window_draw_list().add_text(
            (
                origin.x + sp(RAIL_W) - sp(tokens.SP_1),
                origin.y + max(height - size.y, 0.0),
            ),
            imgui.get_color_u32(theme.rgba(theme.MUTED, 0.75 * _label_alpha())),
            text,
        )
    imgui.dummy((0, height))


def draw(app: Any, ctx: Any) -> None:
    """The whole column, inside the host window. Nothing else is a header."""
    # Local: panes sit above this module, and rail is drawn by main.
    from .panes import model_gate

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
    body_groups = modes.RAIL_GROUPS
    # The rail is destinations only. Manual, shortcuts, layouts, Settings and
    # the labels toggle live in the global menus/status bar.
    rows = sum(len(g) for g in body_groups)
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
    # **The group captions are the ladder's new first rung.** They are the most
    # air of anything here -- a word over a gap that already exists -- so they
    # are the first thing given up, before the row-to-row spacing and long
    # before an item is compressed. They are also tied to ``_label_alpha``: a
    # 52 dp column has no room for a word, and there the gap alone is right.
    with fonts.small(imgui):
        caption_h = imgui.get_text_line_height() + sp(tokens.SP_1)
    # ``caption_h`` plus one row spacing, because the dummy the caption draws
    # emits the spacing after it like every other item does -- budget one and
    # draw the other and the footer lands a row's air out of place.
    caption_step = caption_h + gap
    captions = _label_alpha() > 0.5
    body_h_wanted = rows * sp(ITEM_H) + (rows - 1) * gap + group_gaps
    if captions and body_h_wanted + caption_step * len(body_groups) > avail_h:
        captions = False
    caption_total = caption_step * len(body_groups) if captions else 0.0
    if body_h_wanted > avail_h:
        gap = 0.0
    # **And then the section gaps go too, last of all.** The ladder above gives
    # up the row-to-row air before it compresses the rows; this is its third
    # rung, and it exists because the second one has a floor with a citation
    # (:data:`MIN_ITEM_H`) that must not be argued down. Adding Troupe made the
    # rail fifteen rows, which at the resize floor on a 175% display wants
    # 23.85 design px a row against that 24 px floor -- four physical pixels of
    # overflow, and four physical pixels of overflow is an unreachable mode.
    #
    # A section marker is air, and air is what this ladder spends first. Giving
    # up some of it to keep every row at the floor is the same trade the two
    # rungs above make, not an exception to them -- and the gap only vanishes
    # entirely in a window shorter than any this app can be resized to.
    floor = rows * sp(MIN_ITEM_H)
    if floor + group_gaps > avail_h:
        group_gaps = max(avail_h - floor, 0.0)
    item_h = fitted_height(rows, (rows - 1) * gap + group_gaps + caption_total, avail_h)
    # The drawn gap follows the budget, or the rows are laid out to one figure
    # and drawn against another -- which is the footer landing somewhere else
    # entirely, one rung down.
    group_gap = group_gaps / len(body_groups) if body_groups else 0.0
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
    offsets: dict[str, float] = {}
    y = 0.0
    for index, group in enumerate(body_groups):
        if index:
            y += group_gap
        # Must agree with the drawing loop below, which skips ``_caption``
        # entirely for an empty label: budgeting a row here that nothing draws
        # puts the pill one caption-height below the item it is naming.
        if captions and modes.RAIL_GROUP_LABELS[index]:
            y += caption_step
        for key in group:
            offsets[key] = y
            y += step
    # ``y`` carries a trailing gap that belongs *between* rows; the body ends
    # at its last item's bottom edge, not one spacing past it.
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
            imgui.dummy((0, max(group_gap - gap, 0.0)))
        if captions and modes.RAIL_GROUP_LABELS[index]:
            _caption(modes.RAIL_GROUP_LABELS[index], caption_h)
        for key in group:
            label, icon = labels[key]
            # ``modes`` is the authority on all three, so a mode added there
            # arrives here with its sentence and its maturity already attached.
            note = modes.MATURITY_NOTE.get(key, "")
            purpose = modes.PURPOSE.get(key, "")
            # Why a mode cannot open, if it cannot. Answered from
            # ``ctx.model_rows`` rather than the disk, because this runs sixty
            # times a second; ``model_gate`` owns the answer so the grey here
            # and ``set_mode``'s refusal cannot disagree.
            blocked = model_gate.mode_reason(ctx, key)
            tooltip = f"{purpose} {note}".strip() if note else purpose
            if blocked:
                tooltip = f"{purpose} {blocked}".strip()
            if _item(
                key,
                label,
                icon,
                item_w,
                selected=key == current,
                tooltip=tooltip,
                badge="Download" if blocked else modes.MATURITY.get(key, ""),
                height=item_h,
                enabled=not blocked,
            ):
                if blocked:
                    # Greyed, but never a dead end: the click goes to the one
                    # place that can change the answer, with exactly these rows
                    # already ticked.
                    model_gate.request_install(ctx, model_gate.mode_block(ctx, key))
                else:
                    app._set_mode(key)

    imgui.pop_style_var()
    imgui.end_child()
