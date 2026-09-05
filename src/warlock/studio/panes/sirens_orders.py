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

**Every verb here is one ``SongDoc`` already had.** ``set_order`` is insert,
remove, move and retarget over a list of integers; ``loop_order`` is any entry
and not only the first; ``remove_pattern``, ``duplicate_pattern`` and
``rename_pattern`` are one call each. Until 2026-09-03 the pane offered two of
them, so an intro-then-loop song -- the ordinary shape of a game track -- could
not be expressed from the UI at all (the 2026-09-02 review, section 8).
"""

from __future__ import annotations

from typing import Any

from .. import anchors, controls, icons, sirens_mode, widgets
from ..manual import render as manual_render

_BUSY_WHY = "This song is being written; the buttons come back when it lands."
_ROW_WHY = "This entry is already at the end it would move to."


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    anchors.mark_window("sirens/orders")
    state = sirens_mode.ensure(ctx)
    tab = state.active
    widgets.section("Order")
    manual_render.help_button(ctx, "sirens-orders")

    if tab is None:
        return

    doc = tab.doc
    editable = not tab.busy

    width = widgets.grid_width(2)
    if widgets.disabled_button(
        f"{icons.PLUS} Pattern", editable, (width, 0), reason=_BUSY_WHY
    ):
        pattern = doc.add_pattern()
        sirens_mode.request_rerender(ctx, tab)
        sirens_mode.set_caret(ctx, pattern=pattern.uid)
    imgui.same_line()
    # **A sound effect's pattern is not a song pattern.** Adding an effect mints
    # a pattern of its own and points the grid at it, so with the caret there
    # this button used to append a coin pickup into the middle of the song and
    # say nothing.
    effect = sirens_mode.oneshot_name_for_caret(ctx, tab)
    addable = editable and bool(doc.patterns) and not effect
    if effect:
        add_why = (
            f"The grid is editing the sound effect {effect}, and an effect's "
            "pattern is not part of the song. Pick a song pattern first."
        )
    elif not doc.patterns:
        add_why = "There is no pattern to add yet."
    else:
        add_why = _BUSY_WHY
    if widgets.disabled_button(
        f"{icons.PLUS} To order", addable, (width, 0), reason=add_why
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


def moved_loop(loop: int, index: int, to: int) -> int:
    """Where a loop point ends up when the entry at ``index`` moves to ``to``.

    The loop is an *index* into the order list, so moving entries under it
    silently repoints it at whatever landed there -- a song that loops from
    somewhere the user never chose. Pure, so the arithmetic is assertable
    without a frame: the moved entry carries its own loop with it, and every
    entry the move stepped over shifts by one the other way.
    """
    if loop < 0 or index == to:
        return loop
    if loop == index:
        return to
    if index < loop <= to:
        return loop - 1
    if to <= loop < index:
        return loop + 1
    return loop


def _reorder(ctx: Any, tab: Any, index: int, to: int) -> None:
    """Move one order entry, and carry the loop point with it.

    Two ``SongDoc`` calls and one intention, which is why the loop follows
    here rather than being left for the user to notice and fix.
    """
    doc = tab.doc
    order = list(doc.order)
    if not (0 <= index < len(order) and 0 <= to < len(order)) or index == to:
        return
    after = moved_loop(doc.loop_order, index, to)
    order.insert(to, order.pop(index))
    if not doc.set_order(order):
        return
    if after != doc.loop_order:
        doc.set_song(loop_order=after)
    sirens_mode.request_rerender(ctx, tab)


def _order(ctx: Any, state: Any, tab: Any, editable: bool) -> None:
    from imgui_bundle import imgui

    doc = tab.doc
    if not doc.order:
        widgets.muted("Nothing in the order yet -- the song plays nothing.")
        return
    order = list(doc.order)
    looping = doc.loop_order >= 0
    for index, uid in enumerate(order):
        # The *entry*, not the pattern (S3): a chorus at 00 and 03 used to draw
        # both rows highlighted at once, because a uid cannot tell them apart.
        selected = state.pattern == uid and state.order_index in (None, index)
        mark = f"{icons.REFRESH} " if looping and doc.loop_order == index else ""
        if controls.selectable(
            f"{index:02d}  {mark}{_name_of(doc, uid)}###sirens-order-{index}", selected
        )[0]:
            sirens_mode.set_caret(ctx, pattern=uid, order_index=index)
        # The row's own verbs, in the order a person reaches for them: move it,
        # point it somewhere else, take it out. The loop below breaks after any
        # of them, because each rewrites the list being walked.
        imgui.same_line()
        if widgets.disabled_button(
            f"{icons.ARROW_UP}###sirens-order-up-{index}",
            editable and index > 0,
            (0, 0),
            reason=_BUSY_WHY if not editable else _ROW_WHY,
        ):
            _reorder(ctx, tab, index, index - 1)
            break
        imgui.same_line()
        if widgets.disabled_button(
            f"{icons.ARROW_DOWN}###sirens-order-down-{index}",
            editable and index < len(order) - 1,
            (0, 0),
            reason=_BUSY_WHY if not editable else _ROW_WHY,
        ):
            _reorder(ctx, tab, index, index + 1)
            break
        imgui.same_line()
        if _retarget(ctx, tab, index, uid, editable):
            break
        imgui.same_line()
        if widgets.disabled_button(
            f"{icons.TRASH}###sirens-order-drop-{index}", editable, (0, 0),
            reason=_BUSY_WHY,
        ):
            del order[index]
            if doc.set_order(order):
                sirens_mode.request_rerender(ctx, tab)
            break

    changed, value = controls.checkbox("Loop the song", looping)
    if changed and doc.set_song(loop_order=0 if value else -1):
        sirens_mode.request_rerender(ctx, tab)
    if not looping:
        widgets.muted_wrapped(
            "Off, the song plays once and its instruments ring out. On, it "
            "loops from whichever entry you pick -- an intro that plays once "
            "is the point of the choice."
        )
        return
    imgui.set_next_item_width(-1)
    changed, value = controls.slider_int(
        "Loop from", doc.loop_order, 0, max(0, len(order) - 1), enabled=editable
    )
    controls.fold_undo(doc.history)
    if changed and doc.set_song(loop_order=int(value)):
        sirens_mode.request_rerender(ctx, tab)
    if 0 <= doc.loop_order < len(order):
        widgets.muted(
            f"{icons.REFRESH} back to {doc.loop_order:02d}  "
            f"{_name_of(doc, order[doc.loop_order])}"
        )


def _retarget(ctx: Any, tab: Any, index: int, uid: int, editable: bool) -> bool:
    """The "point this entry at another pattern" menu. -> whether it changed.

    A menu rather than a second list to drag between: an order entry *is* a
    pattern reference, and the shortest way to say "bar 3 is the chorus after
    all" is to pick the chorus where bar 3 is drawn.
    """
    from imgui_bundle import imgui

    doc = tab.doc
    name = f"sirens-order-point-popup-{index}"
    if widgets.disabled_button(
        f"{icons.MOVE}###sirens-order-point-{index}", editable, (0, 0),
        reason=_BUSY_WHY,
    ):
        imgui.open_popup(name)
    changed = False
    if imgui.begin_popup(name):
        for pattern in list(doc.patterns):
            if controls.selectable(
                f"{_name_of(doc, pattern.uid)}###sirens-point-{index}-{pattern.uid}",
                pattern.uid == uid,
            )[0]:
                order = list(doc.order)
                order[index] = pattern.uid
                if doc.set_order(order):
                    sirens_mode.request_rerender(ctx, tab)
                    changed = True
        imgui.end_popup()
    return changed


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
        changed, name = controls.input_text(
            f"Name###sirens-pattern-name-{pattern.uid}",
            pattern.name,
            enabled=editable,
            commit=True,
        )
        if changed:
            doc.rename_pattern(pattern.uid, name)
        imgui.set_next_item_width(-1)
        changed, value = controls.slider_int(
            "Rows", pattern.rows, 1, 256, enabled=editable
        )
        controls.fold_undo(doc.history)
        if changed and doc.resize_pattern(pattern.uid, int(value)):
            sirens_mode.request_rerender(ctx, tab)
            sirens_mode.clamp_caret(ctx, tab)
        width = widgets.grid_width(2)
        if widgets.disabled_button(
            f"{icons.COPY} Duplicate###sirens-pattern-copy-{pattern.uid}",
            editable,
            (width, 0),
            reason=_BUSY_WHY,
        ):
            copy = doc.duplicate_pattern(pattern.uid)
            sirens_mode.set_caret(ctx, pattern=copy.uid)
            # Not added to the order: the two lists are separate by design, and
            # a copy that started playing in the middle of the song the moment
            # it was made would be a surprise, not a shortcut.
            sirens_mode.request_rerender(ctx, tab)
            break
        imgui.same_line()
        # **Delete says what it takes with it.** ``remove_pattern`` drops every
        # order entry naming this pattern in the same undo step, which is right
        # and is exactly the kind of thing a person should be told before the
        # press rather than after it.
        used = sum(1 for one in doc.order if one == pattern.uid)
        if widgets.disabled_button(
            f"{icons.TRASH} Delete###sirens-pattern-drop-{pattern.uid}",
            editable and len(doc.patterns) > 1,
            (width, 0),
            reason=_BUSY_WHY
            if not editable
            else "A song keeps at least one pattern.",
        ):
            sirens_mode.confirm_remove_pattern(ctx, tab, pattern.uid, used)
            break
        if used:
            widgets.muted(
                f"In the order {used} time(s); deleting takes those entries too."
            )
