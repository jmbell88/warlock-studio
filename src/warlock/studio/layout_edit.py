"""Rearranging the workspace, drawn as one overlay over the whole viewport.

**Not a ``panes/`` file**, deliberately: it trips neither the help-button gates
nor the no-imgui-controls guard, and both would be false about it -- it is not
a pane, it is a picture of where the panes are.

It draws **nothing inside any pane**. ``layout.column`` records each pane's
rect into ``layout.FRAME_PANES`` as it draws, and this renders every handle and
drop bar into a single full-viewport window afterwards. One window, one draw
list: ``widgets.section_blocks``' double-split corruption -- which surfaces in
a *different* pane from the one that caused it -- is out of the picture by
construction rather than by care.

Splitters are suppressed while editing, because a handle and a drag target on
the same two pixels is a gesture nobody can aim; sizes are dragged in normal
mode, where the handles are the only thing there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Design px. How close to a pane's top or bottom edge a drop lands "before" or
#: "after" it rather than "onto" it.
EDGE_BAND = 24.0


@dataclass
class EditState:
    """What the editor is in the middle of. Lives on ``AppState``."""

    open: bool = False
    #: The slot being dragged, or "".
    dragging: str = ""
    #: Where it would land: ``(column_id, index)``, or None.
    target: tuple[str, int] | None = None
    #: Slots hidden in this session's edit, before it is committed.
    hidden: set[str] = field(default_factory=set)


def drop_index(rects: list[tuple[str, tuple[float, float, float, float]]], y: float) -> int:
    """Where a pointer at *y* would insert into a column. Pure.

    The whole of the placement rule, as arithmetic: above a pane's midpoint is
    before it, below is after it, and past the last pane is the end. Pure so
    every case -- an empty column, one pane, a pointer above the first -- is a
    plain assertion rather than a drag nobody can repeat.
    """

    for index, (_slot, rect) in enumerate(rects):
        middle = rect[1] + rect[3] * 0.5
        if y < middle:
            return index
    return len(rects)


def moved(order: list[str], slot: str, index: int) -> list[str]:
    """*order* with *slot* moved to *index*. Pure, and clamped.

    Removing first and then inserting is what makes a drag onto a pane's own
    place a no-op rather than a duplicate -- the index the pointer produced was
    computed against the list that still contained it.
    """

    out = [entry for entry in order if entry != slot]
    if slot in order:
        at = min(max(0, index if index <= order.index(slot) else index - 1), len(out))
        out.insert(at, slot)
    else:
        out.insert(min(max(0, index), len(out)), slot)
    return out


def toggle(state: Any) -> None:
    """Shift+W. Enter and leave the editor."""

    edit = ensure(state)
    edit.open = not edit.open
    edit.dragging = ""
    edit.target = None


def ensure(state: Any) -> EditState:
    """The editor's state, made on first use."""

    edit = getattr(state, "layout_edit", None)
    if edit is None:
        edit = EditState()
        state.layout_edit = edit
    return edit


def draw(app: Any, ctx: Any, viewport: Any) -> None:
    """The overlay: a handle on every movable pane, and a drop bar.

    Called from ``App._overlays``, after the workspace has drawn and recorded
    its rects.
    """
    from imgui_bundle import imgui

    from . import layout as layout_mod
    from . import skeletons, theme
    from .tokens import sp

    edit = ensure(ctx.state)
    if not edit.open:
        return
    columns = skeletons.for_mode(ctx, ctx.state.mode)
    if not columns:
        # A workspace whose columns are not data yet cannot be rearranged, and
        # saying so is better than an editor that appears and does nothing.
        _banner(ctx, "This workspace cannot be rearranged yet.")
        return
    rects = layout_mod.FRAME_PANES
    draw_list = imgui.get_foreground_draw_list()
    mouse = imgui.get_mouse_pos()
    hovered = ""
    for column in columns.values():
        for slot in column.live(ctx):
            rect = rects.get(slot.id)
            if rect is None:
                continue
            x, y, w, h = rect
            inside = x <= mouse.x < x + w and y <= mouse.y < y + h
            colour = theme.ACCENT if inside else theme.DIVIDER
            draw_list.add_rect(
                (x, y), (x + w, y + h), imgui.get_color_u32(theme.rgba(colour, 0.9)), sp(4)
            )
            label = slot.label if slot.movable else f"{slot.label} (fixed)"
            draw_list.add_text(
                (x + sp(8), y + sp(6)),
                imgui.get_color_u32(theme.rgba(theme.TEXT)),
                label,
            )
            if inside and slot.movable:
                hovered = slot.id
    if imgui.is_mouse_clicked(0) and hovered:
        edit.dragging = hovered
    if edit.dragging and not imgui.is_mouse_down(0):
        _commit(app, ctx, columns, edit, mouse)
        edit.dragging = ""
    _banner(
        ctx,
        "Drag a pane onto another to reorder it. Shift+W leaves; "
        "Settings > Advanced resets.",
    )


def _banner(ctx: Any, text: str) -> None:
    from imgui_bundle import imgui

    from . import theme
    from .tokens import sp

    draw_list = imgui.get_foreground_draw_list()
    size = imgui.calc_text_size(text)
    pad = sp(10)
    origin = (pad, pad)
    draw_list.add_rect_filled(
        (origin[0] - pad * 0.5, origin[1] - pad * 0.5),
        (origin[0] + size.x + pad * 0.5, origin[1] + size.y + pad * 0.5),
        imgui.get_color_u32(theme.rgba(theme.ELEV_2, 0.95)),
        sp(6),
    )
    draw_list.add_text(origin, imgui.get_color_u32(theme.rgba(theme.TEXT)), text)


def _commit(app: Any, ctx: Any, columns: Any, edit: EditState, mouse: Any) -> None:
    """Land a drag: work out the column and index, and record the arrangement."""

    from . import layout as layout_mod

    rects = layout_mod.FRAME_PANES
    for column in columns.values():
        live = [
            (slot.id, rects[slot.id])
            for slot in column.live(ctx)
            if slot.id in rects
        ]
        if not live:
            continue
        x = live[0][1][0]
        width = live[0][1][2]
        if not (x <= mouse.x < x + width):
            continue
        index = drop_index(live, mouse.y)
        order = moved([slot for slot, _rect in live], edit.dragging, index)
        arrangement = {
            other.id: [
                slot.id
                for slot in other.live(ctx)
            ]
            for other in columns.values()
        }
        arrangement[column.id] = order
        # Anything the drag took *out* of another column leaves it.
        for key, ids in arrangement.items():
            if key != column.id:
                arrangement[key] = [entry for entry in ids if entry != edit.dragging]
        library = getattr(app, "layouts", None)
        if library is not None:
            library.record(ctx.state.mode, arrangement, edit.hidden)
        return
