"""A row of actions that gives up its labels before it gives up its actions.

Every workspace in this app has a row of buttons across the top of a pane, and
every one of them was written as a chain of ``same_line()`` calls with no idea
how wide the pane it is drawn in actually is. That works at UI scale 1.0 on a
wide window, which is the only configuration the smoke suite renders, and it
fails the same way everywhere else: ``same_line`` past the content region does
not wrap, it *clips* -- so the Inker's file row loses "Export PNG" at 150 %, and
the timeline's seventeen-item transport loses its export buttons entirely. A
clipped button is not a cramped button; it is a feature the user cannot reach
and cannot see to ask about.

So the row measures itself and degrades, in the order a reader would choose:

1. every item drawn with its label;
2. a whole *priority group* drops to its glyph, with the label in a tooltip;
3. a whole priority group moves into an overflow menu, where the full labels
   come back.

Three rules make that honest.

**Tiering is per group and all-or-nothing**, which is ``segmented_control``'s
rule and for the same reason: a row where three buttons show labels and two
show glyphs is one control saying two kinds of thing, and *which* two would
change as the window is dragged. The group is ``Item.priority`` -- 0 is what
the row is for, and higher numbers are what it also offers.

**An item never abbreviates twice.** What goes into the menu goes in with its
full label; the menu is where the words come back, not a second, smaller place
to lose them.

**Pinned items never enter the menu.** Two kinds of thing are pinned: the
destructive one (a Delete hidden behind ``...`` is a Delete somebody finds by
accident) and the transport of a timeline (a play button that moves house when
the window is resized is not a transport). They collapse to glyphs and stop
there, which is why :func:`plan` can return a row that does not fit -- if
everything is pinned, there is nothing left to give, and drawing the truth
beats pretending.

The tier arithmetic is :func:`plan`, which is pure and takes numbers: no imgui
context, no fonts, no scale. That is what lets the collapse be tested at every
width rather than eyeballed at one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from imgui_bundle import imgui

from . import fonts, icons, widgets

# The three ways one item can be drawn. Strings rather than an enum because
# they are compared in tests by name and read in a failure message by a person.
FULL = "full"
ICON = "icon"
MENU = "menu"


@dataclass(frozen=True)
class Item:
    """One action in a row.

    ``key`` is the identity: it is what a click reports, and it is what the
    imgui id is built from (``label##bar/key``). Never the label -- a row that
    renames a button, or draws the same word twice, would otherwise silently
    merge two controls into one, and the label is exactly the part that changes
    between the full and compact renderings.
    """

    key: str
    label: str
    icon: str = ""
    tooltip: str = ""
    enabled: bool = True
    reason: str = ""
    danger: bool = False
    # The one thing this row is *for*, drawn in the accent (REDESIGN.md wave
    # 4.2). At most one per row, and it should almost always be pinned as well:
    # a row's answer moving into an overflow menu is the collapse doing the
    # opposite of its job. Ignored when ``danger`` is set -- red already says
    # "this is the consequential one", and an accent-filled destructive button
    # says two things at once.
    primary: bool = False
    # Higher collapses first. 0 is the row's reason for existing.
    priority: int = 0
    # Never hidden in the overflow menu; see the module docstring.
    pinned: bool = False


def plan(
    widths_full: Any,
    widths_icon: Any,
    priorities: Any,
    pinned: Any,
    avail: float,
    overflow_w: float,
    *,
    gap: float = 0.0,
) -> list[str]:
    """Which tier each item is drawn at, given the room. -> FULL/ICON/MENU.

    Pure arithmetic over parallel sequences, in physical pixels. ``gap`` is the
    spacing between two drawn items (imgui's ``item_spacing.x``); it is a
    keyword because the interesting cases in a test have no gap at all and the
    zero is the honest default for "just the widths".

    An item with no glyph is expected to be handed the *same* width for both
    tiers by the caller, which is what makes "this group drops to icons" mean
    "this group gets no narrower" for that item rather than "that item vanishes".
    """
    count = len(widths_full)
    tiers = [FULL] * count

    def spread() -> float:
        shown = [i for i in range(count) if tiers[i] != MENU]
        total = sum(
            (widths_full[i] if tiers[i] == FULL else widths_icon[i]) for i in shown
        )
        if shown:
            total += gap * (len(shown) - 1)
        if len(shown) != count:
            # The overflow button is an item too, and it is the one a row
            # forgets to budget for -- which is how a bar "fits" at exactly the
            # width where the ... it just grew hangs off the end.
            total += overflow_w + (gap if shown else 0.0)
        return total

    if spread() <= avail or not count:
        return tiers
    groups = sorted(set(priorities), reverse=True)
    for group in groups:
        for i in range(count):
            if priorities[i] == group:
                tiers[i] = ICON
        if spread() <= avail:
            return tiers
    for group in groups:
        for i in range(count):
            if priorities[i] == group and not pinned[i]:
                tiers[i] = MENU
        if spread() <= avail:
            return tiers
    return tiers


def _measure(items: Any) -> tuple[list[float], list[float]]:
    """(full, icon) widths, measured in the font each tier is drawn in."""
    square = imgui.get_frame_height()
    with fonts.label(imgui):
        full = [widgets.button_width(item.label) for item in items]
    icon = [square if item.icon else full[i] for i, item in enumerate(items)]
    return full, icon


def toolbar(
    bar_id: str,
    items: Any,
    on_click: Any = None,
    *,
    trailing: Any = None,
) -> str | None:
    """Draw the row. -> the key of the item clicked this frame, or None.

    ``on_click`` is called with that key as well, so a caller can either branch
    on the return or hand over a dispatcher; both shapes exist in the panes and
    neither is worth forcing.

    ``trailing`` is ``(width, draw)`` for the fixed things that live at the end
    of a row and cannot collapse -- a combo, a slider, a help marker. Its width
    is subtracted *before* the tiers are chosen and its ``draw`` is called
    after them, because a control that is measured after the row has already
    claimed the space is a control that gets clipped.
    """
    avail = imgui.get_content_region_avail().x
    gap = imgui.get_style().item_spacing.x
    overflow_w = imgui.get_frame_height()
    if trailing is not None:
        avail -= trailing[0] + gap
    full, icon = _measure(items)
    tiers = plan(
        full,
        icon,
        [item.priority for item in items],
        [item.pinned for item in items],
        avail,
        overflow_w,
        gap=gap,
    )
    clicked: str | None = None
    drawn = False
    for item, tier in zip(items, tiers, strict=True):
        if tier == MENU:
            continue
        if drawn:
            imgui.same_line()
        drawn = True
        ident = f"##{bar_id}/{item.key}"
        if tier == ICON and item.icon:
            # The label becomes the tooltip rather than disappearing: a glyph
            # with no name is the accessible-name failure ``icon_button``'s
            # required tooltip exists to prevent, and a compacted button is
            # exactly where it would come back.
            hit = widgets.icon_button(
                f"{item.icon}{ident}",
                item.tooltip or item.label,
                danger=item.danger,
                # A compacted primary keeps its frame where the others lose
                # theirs: borderless is what makes a row of glyphs read as
                # content rather than as boxes, and the one glyph that is the
                # answer still has to be findable among them.
                enabled=item.enabled,
                borderless=not item.primary,
            )
        elif item.danger:
            hit = widgets.destructive_button(
                f"{item.label}{ident}", enabled=item.enabled
            )
        elif item.primary:
            hit = widgets.primary_button(
                f"{item.label}{ident}",
                enabled=item.enabled,
                reason=item.reason,
                tooltip=item.tooltip,
            )
        else:
            hit = widgets.ghost_button(
                f"{item.label}{ident}",
                enabled=item.enabled,
                reason=item.reason,
                tooltip=item.tooltip,
            )
        if hit:
            clicked = item.key
    if MENU in tiers:
        if drawn:
            imgui.same_line()
        popup = f"{bar_id}/menu"
        if widgets.icon_button(f"{icons.ELLIPSIS}##{bar_id}/overflow", "More", borderless=True):
            imgui.open_popup(popup)
        if imgui.begin_popup(popup):
            for item, tier in zip(items, tiers, strict=True):
                if tier != MENU:
                    continue
                if imgui.menu_item(
                    f"{item.label}##{bar_id}/menu/{item.key}", "", False, item.enabled
                )[0]:
                    clicked = item.key
                note = item.reason if not item.enabled else item.tooltip
                if note and imgui.is_item_hovered(
                    imgui.HoveredFlags_.allow_when_disabled.value
                ):
                    imgui.set_tooltip(note)
            imgui.end_popup()
    if trailing is not None:
        if drawn or MENU in tiers:
            # ``same_line_or_wrap``, not ``same_line``: :func:`plan` can return
            # a row that does not fit -- an all-pinned row has nothing left to
            # give and says so rather than pretending -- and when it does, a
            # trailing block continued on that line starts past the content
            # edge and is clipped. Wrapping puts it on the next line instead,
            # which is a row that got taller rather than a control that
            # vanished. Found by the timeline's transport at 150 %, where five
            # pinned steps plus a pinned Delete overrun a 500 px strip.
            widgets.same_line_or_wrap(trailing[0])
        trailing[1]()
    if clicked is not None and on_click is not None:
        on_click(clicked)
    return clicked
