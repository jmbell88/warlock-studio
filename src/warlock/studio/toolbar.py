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

from dataclasses import InitVar, dataclass
from typing import Any

from imgui_bundle import imgui

from . import controls, fonts, icons, widgets
from .tokens import sp

# The three ways one item can be drawn. Strings rather than an enum because
# they are compared in tests by name and read in a failure message by a person.
FULL = "full"
ICON = "icon"
MENU = "menu"
ButtonRole = controls.ButtonRole


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
    role: controls.ButtonRole = controls.ButtonRole.SECONDARY
    # Explicit because selected is persistent state, not hover. Tool modes and
    # view toggles use the same wash/boundary as selectable rows.
    selected: bool = False
    # Constructor-only migration shims. Studio call sites use ``role``; these
    # keep older extensions and serialized test fixtures source-compatible.
    primary: InitVar[bool] = False
    danger: InitVar[bool] = False
    # Higher collapses first. 0 is the row's reason for existing.
    priority: int = 0
    # Never hidden in the overflow menu; see the module docstring.
    pinned: bool = False

    def __post_init__(self, primary: bool, danger: bool) -> None:
        if danger:
            object.__setattr__(self, "role", controls.ButtonRole.DESTRUCTIVE)
        elif primary:
            object.__setattr__(self, "role", controls.ButtonRole.PRIMARY)


@dataclass(frozen=True)
class Field:
    """A control that lives in a row and is not a button.

    A context bar is a row of *settings* -- a size, a mode, an opacity -- and
    the tier machinery above is exactly as right for them as it is for
    buttons: a row that keeps every field at full width and clips the last one
    is the same failure as a row that clips a button, and the fix is the same
    ordered degradation.

    ``draw`` is called with one argument, ``compact``, and draws the control
    at the width it was measured at. It is a callable rather than a widget
    spec because the controls a bar can hold are open-ended (a combo, a
    slider, a colour chip, a segmented control) and enumerating them here
    would make this module the place every new control has to be taught about.

    ``width``/``compact`` are design pixels, not measured: a field's natural
    size is a decision its author makes, not something a font metric can tell
    us. ``priority`` and ``pinned`` mean exactly what they mean on :class:`Item`.
    """

    key: str
    label: str
    draw: Any
    width: float = 120.0
    compact: float = 0.0
    priority: int = 0
    pinned: bool = False

    def widths(self) -> tuple[float, float]:
        """(full, compact) in physical pixels. Equal when no compact size is set."""

        full = sp(self.width)
        return full, (sp(self.compact) if self.compact else full)


@dataclass(frozen=True)
class Trailing:
    """The fixed block at the end of a row, with a tier of its own.

    A bare ``(width, draw)`` tuple still works and means "one width, never
    collapses", which is what a combo or a help marker wants. This exists for
    the case that tuple gets wrong: a block holding the *sitting's* settings --
    Inker's symmetry group and its view aids -- is measured and subtracted
    before :func:`plan` runs, so at one fixed width it wins unconditionally
    against the tool's own size, ink and nib. That is how a row of things
    nobody touches twice an hour crowds out the things a hand reaches for
    between strokes, which is the failure that deleted Inker's second bar on
    2026-08-23 and would have come back through this door.

    The ordering it establishes, measured rather than assumed (see
    :func:`trailing_compact`): **a label is cheaper to lose than a control.**
    The row gives up its words first -- a slider that reads ``129`` instead of
    ``129 px`` is still the slider, and its name is a hover away -- and the
    block gives up its controls only when even a fully compacted row cannot fit
    beside them, because a folded mirror is not a smaller mirror, it is a
    button that is no longer there.

    ``draw`` is called with the chosen tier, exactly as :class:`Field`'s is.
    """

    width: float
    compact: float
    draw: Any


def trailing_widths(trailing: Any) -> tuple[float, float, Any]:
    """``(full, compact, draw)`` for either shape of trailing block."""
    if isinstance(trailing, Trailing):
        return (trailing.width, trailing.compact, trailing.draw)
    width, draw = trailing
    return (width, width, draw)


def trailing_compact(
    row_min: float,
    avail: float,
    full_w: float,
    compact_w: float,
    *,
    gap: float = 0.0,
) -> bool:
    """Whether the trailing block must collapse. Pure, and takes numbers.

    ``row_min`` is what the items and fields need at their **smallest** tier,
    not their largest, and that is the whole of the policy: the block keeps its
    controls as long as the row can be made to fit beside them at all, and
    folds only when even a fully compacted row cannot.

    The other way round was tried first and measured wrong. At the app's
    default 1600x950 Inker's context bar has about 835 px -- the mode rail
    takes 70 that a sidebars-only sum misses -- while the brush's row wants 689
    and the symmetry block 217. Deciding from the row's *full* width therefore
    folded the mirrors away on the commonest tool in the commonest window,
    which is the opposite of what putting them on the bar was for. Deciding
    from its smallest width keeps them, and the price is paid where it is
    legible: a slider drops from ``129 px`` to ``129`` and stays a number.
    """
    return not (row_min + gap + full_w <= avail)


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
    fields: Any = (),
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

    ``fields`` are :class:`Field`s drawn after the items, in the same row and
    under the same tiering: a field's ``priority`` competes with the buttons'
    directly, so a bar can say "the size spinner goes before the Flip button"
    rather than "controls collapse before buttons do", which is a rule no user
    would predict. A field that reaches the overflow menu is drawn in it, with
    its label above it -- the menu is where the words come back.
    """
    avail = imgui.get_content_region_avail().x
    gap = imgui.get_style().item_spacing.x
    overflow_w = imgui.get_frame_height()
    entries = [*items, *fields]
    full, icon = _measure(items)
    for field in fields:
        wide, narrow = field.widths()
        full.append(wide)
        icon.append(narrow)
    trailing_draw = None
    trailing_w = 0.0
    trailing_tight = False
    if trailing is not None:
        trailing_full, trailing_small, trailing_draw = trailing_widths(trailing)
        # The block's tier is chosen *before* the row's, because its width is
        # what the row is then planned against -- and from the row's *smallest*
        # width, so the question asked is "can the row be made to fit beside
        # this at all", not "does it fit with every label on".
        trailing_tight = trailing_compact(
            sum(icon) + gap * max(len(icon) - 1, 0),
            avail,
            trailing_full,
            trailing_small,
            gap=gap,
        )
        trailing_w = trailing_small if trailing_tight else trailing_full
        avail -= trailing_w + gap
    tiers = plan(
        full,
        icon,
        [entry.priority for entry in entries],
        [entry.pinned for entry in entries],
        avail,
        overflow_w,
        gap=gap,
    )
    clicked: str | None = None
    drawn = False
    for index, (item, tier) in enumerate(zip(entries, tiers, strict=True)):
        if tier == MENU:
            continue
        if drawn:
            imgui.same_line()
        drawn = True
        if isinstance(item, Field):
            imgui.set_next_item_width(full[index] if tier == FULL else icon[index])
            item.draw(tier != FULL)
            continue
        ident = f"##{bar_id}/{item.key}"
        if tier == ICON and item.icon:
            # The label becomes the tooltip rather than disappearing: a glyph
            # with no name is the accessible-name failure ``icon_button``'s
            # required tooltip exists to prevent, and a compacted button is
            # exactly where it would come back.
            # One call for both states. A selected item used to abandon this
            # button for ``controls.button(role=ICON)`` purely to get the
            # selection paint -- which cost the compacted item the glyph
            # centring this button exists for, and left the two states drawn
            # by two different widgets. ``icon_button`` carries ``selected``
            # now, so the branch is gone rather than tidied.
            hit = widgets.icon_button(
                f"{item.icon}{ident}",
                item.tooltip or item.label,
                danger=item.role is controls.ButtonRole.DESTRUCTIVE,
                enabled=item.enabled,
                borderless=item.role is not controls.ButtonRole.PRIMARY,
                selected=item.selected,
            )
        elif item.selected:
            hit = controls.button(
                f"{item.label}{ident}",
                role=item.role,
                selected=True,
                enabled=item.enabled,
                reason=item.reason,
                tooltip=item.tooltip,
            )
        elif item.role is controls.ButtonRole.DESTRUCTIVE:
            hit = widgets.destructive_button(
                f"{item.label}{ident}", enabled=item.enabled
            )
        elif item.role is controls.ButtonRole.PRIMARY:
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
            widgets.popup_chrome(_imgui=imgui)
            for item, tier in zip(entries, tiers, strict=True):
                if tier != MENU:
                    continue
                if isinstance(item, Field):
                    widgets.field_label(item.label)
                    imgui.set_next_item_width(sp(item.width))
                    item.draw(False)
                    continue
                if controls.menu_item(
                    f"{item.label}##{bar_id}/menu/{item.key}", "", False, item.enabled
                )[0]:
                    clicked = item.key
                note = item.reason if not item.enabled else item.tooltip
                if note and imgui.is_item_hovered(
                    imgui.HoveredFlags_.allow_when_disabled.value
                ):
                    imgui.set_tooltip(note)
            imgui.end_popup()
    if trailing_draw is not None:
        if drawn or MENU in tiers:
            # ``same_line_or_wrap``, not ``same_line``: :func:`plan` can return
            # a row that does not fit -- an all-pinned row has nothing left to
            # give and says so rather than pretending -- and when it does, a
            # trailing block continued on that line starts past the content
            # edge and is clipped. Wrapping puts it on the next line instead,
            # which is a row that got taller rather than a control that
            # vanished. Found by the timeline's transport at 150 %, where five
            # pinned steps plus a pinned Delete overrun a 500 px strip.
            widgets.same_line_or_wrap(trailing_w)
        # A tuple's ``draw`` takes nothing and a ``Trailing``'s takes the tier,
        # which is what keeps every existing caller working unchanged.
        if isinstance(trailing, Trailing):
            trailing_draw(trailing_tight)
        else:
            trailing_draw()
    if clicked is not None and on_click is not None:
        on_click(clicked)
    return clicked
