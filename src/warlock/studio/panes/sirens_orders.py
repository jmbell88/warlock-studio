"""Sirens' left-bottom pane: the order list, and the patterns it points at.

**The order holds pattern uids, and this pane never shows one.** It draws each
entry's *position* and the pattern's name, because a uid is an implementation
detail the user has no way to act on -- and because the document's rule (uids,
never indices) exists precisely so that deleting a pattern cannot silently
repoint an order entry. A pane that displayed the number would be teaching the
user to rely on a number that is deliberately meaningless to them.

**Adding a pattern does not add it to the order**, and removing one from the
order does not delete it. They are two lists and the whole point of an order
list is that a pattern can appear in it more than once, or not at all.
"""

from __future__ import annotations

from typing import Any

from .. import controls, icons, sirens_mode, widgets
from ..manual import render as manual_render


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = sirens_mode.ensure(ctx)
    tab = state.active
    widgets.section("Order")
    manual_render.help_button(ctx, "sirens-orders")

    if tab is None:
        return

    doc = tab.doc
    editable = not tab.busy
    busy_why = "This song is being written; the buttons come back when it lands."

    width = widgets.grid_width(2)
    if widgets.disabled_button(
        f"{icons.PLUS} Pattern", editable, (width, 0), reason=busy_why
    ):
        pattern = doc.add_pattern()
        sirens_mode.request_rerender(ctx, tab)
        sirens_mode.set_caret(ctx, pattern=pattern.uid)
    imgui.same_line()
    if widgets.disabled_button(
        f"{icons.PLUS} To order", editable and bool(doc.patterns), (width, 0),
        reason=busy_why if editable else "There is no pattern to add yet.",
    ):
        current = state.pattern or doc.patterns[0].uid
        if doc.set_order(list(doc.order) + [current]):
            sirens_mode.request_rerender(ctx, tab)

    imgui.dummy((0, 4))
    _order(ctx, state, tab, editable)
    imgui.dummy((0, 8))
    widgets.section("Patterns")
    _patterns(ctx, state, tab, editable)


def _name_of(doc: Any, uid: int) -> str:
    pattern = doc.pattern(uid)
    if pattern is None:
        # An order entry whose pattern is gone. ``set_order`` refuses one, so
        # this is unreachable through the app -- but a hand-edited ``.wsng``
        # can carry it and a row that renders as an exception is worse than a
        # row that says what is wrong.
        return "(missing)"
    return pattern.name or f"Pattern {doc.patterns.index(pattern) + 1}"


def _order(ctx: Any, state: Any, tab: Any, editable: bool) -> None:
    from imgui_bundle import imgui

    doc = tab.doc
    if not doc.order:
        widgets.muted("Nothing in the order yet -- the song plays nothing.")
        return
    for index, uid in enumerate(list(doc.order)):
        selected = state.pattern == uid
        clicked = controls.selectable(
            f"{index:02d}  {_name_of(doc, uid)}###sirens-order-{index}", selected
        )[0]
        if clicked:
            sirens_mode.set_caret(ctx, pattern=uid)
        imgui.same_line()
        if widgets.disabled_button(
            f"{icons.TRASH}###sirens-order-drop-{index}", editable, (0, 0),
            reason="This song is being written.",
        ):
            order = list(doc.order)
            del order[index]
            if doc.set_order(order):
                sirens_mode.request_rerender(ctx, tab)
            break
    changed, value = controls.checkbox("Loop at the end", doc.loop_order >= 0)
    if changed and doc.set_song(loop_order=0 if value else -1):
        sirens_mode.request_rerender(ctx, tab)


def _patterns(ctx: Any, state: Any, tab: Any, editable: bool) -> None:
    from imgui_bundle import imgui

    doc = tab.doc
    if not doc.patterns:
        widgets.muted("No patterns yet.")
        return
    for pattern in list(doc.patterns):
        selected = state.pattern == pattern.uid
        if controls.selectable(
            f"{_name_of(doc, pattern.uid)}  ({pattern.rows} rows)"
            f"###sirens-pattern-{pattern.uid}",
            selected,
        )[0]:
            sirens_mode.set_caret(ctx, pattern=pattern.uid)
        if not selected:
            continue
        imgui.set_next_item_width(-1)
        changed, value = controls.slider_int(
            "Rows", pattern.rows, 1, 256, enabled=editable
        )
        if changed and doc.resize_pattern(pattern.uid, int(value)):
            sirens_mode.request_rerender(ctx, tab)
            sirens_mode.clamp_caret(ctx, tab)
