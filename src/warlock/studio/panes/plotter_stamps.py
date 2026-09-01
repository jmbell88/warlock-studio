"""Plotter's numbered stamps: nine slots, each with a picture and a name.

The slots existed and had no pane. They were stored with `Ctrl+Shift+<digit>`
and recalled with a bare digit, and that was the whole of the feature -- nothing
on screen said a slot was full, what was in it, or which of the nine you wanted.
A user came back to a map the next day with nine numbers and no way to tell them
apart short of pressing each one and looking at the brush.

Three things fix that, and only one of them is this pane. The stamps moved onto
the **document**, so they survive a close (``MapDoc.stamps``, and the reason a
map is their only honest home is written there). They gained a **name**. And
this draws both: a thumbnail of the block, the name as a field, and the two
verbs.

**The digits stay the recall gesture.** Storing happens nine times a session and
recalling hundreds, so the cheap gesture keeps the frequent job and this pane is
for the other three: naming one, seeing which is which, and clearing one out.
"""

from __future__ import annotations

from typing import Any

from .. import controls, icons, plotter_mode, widgets
from ..manual import render as manual_render
from ..tilegrid import gid as gidlib
from ..tokens import sp

#: What this pane refuses to shrink past, in design pixels.
STAMPS_FLOOR = 120.0

#: The slots, in the order the number row runs. Nine because there are nine
#: digits a hand reaches without moving -- ``0`` is deliberately not one: it
#: sits past the reach and would make the mapping "slot N is key N except the
#: last".
SLOTS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)

#: How big a slot's thumbnail is drawn, in design pixels.
THUMB = 40.0

#: How many cells of a stamp the thumbnail shows, at most, on each axis. A
#: 16x16 stamp drawn into 40 px would be two-and-a-half pixels per cell, which
#: is a smear; four across says "a block, roughly this shape" and stops.
THUMB_CELLS = 4


def on_tile_layer(ctx: Any) -> bool:
    """Whether a stamp can be recalled right now. The slot's ``when``.

    A stamp is a block of tiles, so the pane belongs to a tile layer the way the
    tileset palette does. On an object layer it would be nine controls that
    cannot act -- and greying all nine says less than not claiming the height.
    """

    from ..plotter_state import layer_kind

    state = getattr(ctx.state, "plotter", None)
    tab = None if state is None else state.active
    if tab is None:
        return False
    return layer_kind(tab.doc.active()) == "tile"


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = plotter_mode.ensure(ctx)
    tab = state.active
    widgets.section("Tile stamps")
    manual_render.help_button(ctx, "plotter-stamps")
    if tab is None:
        return

    editable = not tab.busy
    for slot in SLOTS:
        imgui.push_id(f"stamp-{slot}")
        _slot_row(ctx, state, tab, slot, editable)
        imgui.pop_id()


def _slot_row(ctx: Any, state: Any, tab: Any, slot: int, editable: bool) -> None:
    """One slot: the digit, a thumbnail, the name, and Store or Recall."""
    from imgui_bundle import imgui

    stamp = tab.doc.stamps.get(slot)
    imgui.align_text_to_frame_padding()
    widgets.muted(str(slot))
    imgui.same_line()
    _thumbnail(ctx, tab, stamp)
    imgui.same_line()

    if stamp is None:
        # An empty slot offers exactly one thing, and says what would fill it.
        if widgets.disabled_button(
            f"{icons.PLUS} Store",
            editable and state.brush is not None,
            (-1, 0),
            reason=(
                "This map is being written."
                if not editable
                else "Pick a tile or capture a block first -- then Ctrl+Shift+"
                f"{slot} stores it here too."
            ),
            tooltip=f"Put the brush in hand into slot {slot} (Ctrl+Shift+{slot})",
        ):
            plotter_mode.store_stamp(ctx, state, tab, slot)
        return

    imgui.set_next_item_width(-sp(56))
    typed = widgets.input_text(
        "##name", stamp.name, max_length=48, hint=f"Stamp {slot}"
    )
    if typed != stamp.name and editable:
        # Per keystroke, unlike the layer rename: a stamp name is a label rather
        # than a document-wide identifier, and the slot is already its address,
        # so the extra undo steps buy nothing the caret leaving would.
        tab.doc.rename_stamp(slot, typed)
    imgui.same_line()
    if controls.button(
        f"{icons.BRUSH}##recall",
        tooltip=f"Take this stamp into the hand (press {slot})",
    ):
        plotter_mode.recall_stamp(ctx, state, tab, slot)
    imgui.same_line()
    if widgets.disabled_button(
        f"{icons.X}##clear",
        editable,
        reason="This map is being written.",
        tooltip=f"Empty slot {slot}. Ctrl+Z brings it back.",
    ):
        tab.doc.clear_stamp(slot)


def _thumbnail(ctx: Any, tab: Any, stamp: Any) -> None:
    """The stamp's first few cells, drawn from the map's own tileset textures.

    Through the same texture the picker and the canvas use, so a slot shows the
    tiles it actually holds rather than a coloured square standing in for them.
    An empty slot draws the frame and nothing in it, which is what says "empty"
    without a word.
    """
    from imgui_bundle import imgui

    from .. import theme
    from . import plotter_textures

    side = sp(THUMB)
    origin = imgui.get_cursor_screen_pos()
    draw_list = imgui.get_window_draw_list()
    draw_list.add_rect(
        (origin.x, origin.y),
        (origin.x + side, origin.y + side),
        imgui.get_color_u32(theme.rgba(theme.MUTED, 0.5)),
        sp(2.0),
    )
    imgui.dummy((side, side))
    if stamp is None or stamp.cells is None or not stamp.cells.size:
        return

    doc = tab.doc
    rows = min(int(stamp.cells.shape[0]), THUMB_CELLS)
    columns = min(int(stamp.cells.shape[1]), THUMB_CELLS)
    step = side / max(rows, columns)
    for row in range(rows):
        for column in range(columns):
            value = int(stamp.cells[row, column])
            found = doc.resolve(value)
            if found is None:
                continue
            tileset, local = found
            index = next(
                (
                    at
                    for at, ref in enumerate(doc.tilesets)
                    if ref.tileset is tileset
                ),
                None,
            )
            if index is None:
                continue
            texture = plotter_textures.tileset_texture(
                ctx, tab.uid, index, tileset, doc.tileset_epoch
            )
            if texture is None:
                continue
            uv = tileset.uv(local)
            corners = _corner_uvs(uv, value)
            x0 = origin.x + column * step
            y0 = origin.y + row * step
            draw_list.add_image_quad(
                widgets.texture_ref(texture),
                (x0, y0),
                (x0 + step, y0),
                (x0 + step, y0 + step),
                (x0, y0 + step),
                corners[0],
                corners[1],
                corners[2],
                corners[3],
                imgui.get_color_u32((1.0, 1.0, 1.0, 1.0)),
            )


def _corner_uvs(uv: Any, value: int) -> Any:
    """The tile's UV corners with its flip flags applied.

    Through ``plotter_canvas``' own function rather than a second copy: a
    thumbnail that ignored the flags would show a stamp the wrong way round,
    which is exactly the mistake the stamp transforms exist to make visible.
    """
    from . import plotter_canvas

    mask = int(value)
    return plotter_canvas._corner_uvs(
        uv,
        bool(mask & gidlib.FLIP_H),
        bool(mask & gidlib.FLIP_V),
        bool(mask & gidlib.FLIP_D),
    )
