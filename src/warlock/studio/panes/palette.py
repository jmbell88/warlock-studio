"""The command palette: Ctrl+K, type, Enter (I80).

Drawing only. Which commands exist, what a query matches and how the rows are
ordered all live in :mod:`..palette`, which is pure -- so this file has no list
of its own and cannot become a second one.

It is drawn as a modal popup from the frame loop's overlay pass rather than by
any mode, because it is the one surface that has to be reachable from all of
them, including the three that own their whole window.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import icons, theme, widgets
from .. import palette as palette_lib
from ..tokens import sp

POPUP = "command-palette"
# How far down the screen the box sits. A palette anchored to the exact centre
# jumps as the result list grows and shrinks under the cursor; anchored near
# the top, the input stays put and only the list below it moves.
TOP_FRACTION = 0.18
WIDTH = 560.0
LIST_HEIGHT = 320.0


def toggle(ctx: Any) -> None:
    """Ctrl+K. Opening always starts from an empty query -- see ``AppState``."""
    state = ctx.state
    state.palette_open = not state.palette_open
    state.palette_query = ""
    state.palette_index = 0


def close(ctx: Any) -> None:
    ctx.state.palette_open = False
    ctx.state.palette_query = ""
    ctx.state.palette_index = 0


def rows(ctx: Any) -> list[tuple[str, Any]]:
    """The palette's contents as ``(kind, item)``, in the order drawn.

    Assets after commands and only once something has been typed: the commands
    are a fixed vocabulary the user is learning, and burying them under eight
    asset names on every open would teach the opposite.
    """
    query = ctx.state.palette_query
    ranked = palette_lib.rank(query, palette_lib.commands(ctx), lambda c: c.label)
    found = [("command", command) for command in ranked]
    found += [("asset", job) for job in palette_lib.assets(ctx, query)]
    return found


def draw(ctx: Any) -> None:
    state = ctx.state
    if not state.palette_open:
        return
    appearing = not imgui.is_popup_open(POPUP)
    if appearing:
        imgui.open_popup(POPUP)

    # The rise is spent on the *window position* here, not on a cursor offset:
    # this is the one floating surface that already places itself every frame
    # (``Cond_.always``, so the input never jumps as the list grows), so the
    # whole panel can move instead of its contents sliding inside it.
    alpha, rise = widgets.popover_enter("palette", appearing)
    viewport = imgui.get_main_viewport()
    imgui.set_next_window_pos(
        (
            viewport.pos.x + viewport.size.x * 0.5,
            viewport.pos.y + viewport.size.y * TOP_FRACTION + rise,
        ),
        imgui.Cond_.always.value,
        (0.5, 0.0),
    )
    imgui.set_next_window_size((sp(WIDTH), 0))
    imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
    opened, _ = imgui.begin_popup_modal(
        POPUP,
        None,
        imgui.WindowFlags_.no_title_bar.value
        | imgui.WindowFlags_.no_move.value
        | imgui.WindowFlags_.always_auto_resize.value,
    )
    if not opened:
        # Dismissed by a click outside. The flag has to follow, or the next
        # frame reopens it.
        imgui.pop_style_var()
        close(ctx)
        return

    found = rows(ctx)
    # Clamped every frame rather than when the query changes: the list shrinks
    # as characters are typed, and an index left past the end would run
    # whatever happened to be last.
    if found:
        state.palette_index = max(0, min(state.palette_index, len(found) - 1))
    else:
        state.palette_index = 0

    imgui.set_next_item_width(-1)
    if imgui.is_window_appearing():
        imgui.set_keyboard_focus_here()
    state.palette_query = widgets.input_text(
        "##palette-query", state.palette_query, max_length=120, hint="Type a command..."
    )
    # The keys are read here rather than in ``App._shortcut``: the query box
    # holds the keyboard focus, so the frame loop is not dispatching shortcuts
    # at all while this is up.
    if imgui.is_key_pressed(imgui.Key.down_arrow):
        state.palette_index += 1
    if imgui.is_key_pressed(imgui.Key.up_arrow):
        state.palette_index -= 1
    if found:
        state.palette_index %= len(found)
    chosen = None
    if imgui.is_key_pressed(imgui.Key.enter) or imgui.is_key_pressed(imgui.Key.keypad_enter):
        chosen = found[state.palette_index] if found else None
    if imgui.is_key_pressed(imgui.Key.escape):
        imgui.close_current_popup()
        close(ctx)
        imgui.end_popup()
        imgui.pop_style_var()
        return

    if imgui.begin_child("palette-list", (0, sp(LIST_HEIGHT))):
        if not found:
            widgets.empty_state(
                icons.SEARCH,
                "No command matches.",
                "Try fewer letters, or Esc to close.",
            )
        group = ""
        for index, (kind, item) in enumerate(found):
            heading = "Assets" if kind == "asset" else item.group
            if heading != group:
                group = heading
                widgets.section(heading.lower())
            if _row(ctx, kind, item, index == state.palette_index):
                chosen = (kind, item)
    imgui.end_child()

    if chosen is not None:
        _run(ctx, chosen)
        imgui.close_current_popup()
        close(ctx)
    imgui.end_popup()
    imgui.pop_style_var()


def _row(ctx: Any, kind: str, item: Any, current: bool) -> bool:
    """One row. -> whether it was clicked. Scrolls itself into view when it is
    the one the arrow keys have landed on."""
    if kind == "asset":
        label = item.get("name") or item.get("prompt") or item["id"]
        hint = item["id"]
        enabled = True
        row_id = f"asset/{item['id']}"
    else:
        label = item.label
        hint = item.hint
        enabled = item.enabled(ctx)
        row_id = f"cmd/{item.key}"
    if current:
        imgui.set_scroll_here_y(0.5)
    imgui.begin_disabled(not enabled)
    clicked = imgui.selectable(f"{label}##{row_id}", current)[0]
    imgui.end_disabled()
    if hint:
        # Painted into the row rather than laid out beside it: a selectable is
        # full-width, so ``same_line`` after one leaves no room to right-align
        # into. The shortcut is a footnote and a column of them down the right
        # edge is what makes them skimmable.
        low = imgui.get_item_rect_min()
        high = imgui.get_item_rect_max()
        size = imgui.calc_text_size(hint)
        imgui.get_window_draw_list().add_text(
            (high.x - size.x - sp(6), low.y + (high.y - low.y - size.y) * 0.5),
            imgui.get_color_u32(theme.rgba(theme.MUTED)),
            hint,
        )
    return clicked and enabled


def _run(ctx: Any, chosen: tuple[str, Any]) -> None:
    kind, item = chosen
    if kind == "asset":
        from . import library

        library.run_action(ctx, item, "open")
        return
    if item.enabled(ctx):
        item.run(ctx)
