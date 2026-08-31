"""The tileset editor: a sheet over the centre pane, not a mode and not a doc.

Drawn *instead of* ``plotter_canvas`` when ``state.editing_tileset is not
None`` -- precisely the branch ``_review_workspace`` already takes, and
``PaneRole.SHEET`` already exists for it.

**Not a mode.** Adding one is a 21-place checklist including literal English
prose asserting the mode *count* in two documents. **Not a document kind**
either: that would teach ``state.active``, ``plotter_canvas.draw``, four panes,
the journal, the guard and the save path a second shape, for something that is
a *view* of a map's own tileset.

Three tabs, which are the three questions a tileset answers:

* **Tiles** -- the atlas at a readable zoom, and the per-tile form *moved* out
  of the 300 px sidebar, where a class name, a probability and a property table
  had about eleven characters of width each.
* **Collision** -- one tile at 8-16x, with shapes written through the already
  undoable ``doc.set_tile_meta``. This is the editor ``docs/COMPAT.md`` claimed
  existed for four months, which is why that note is rewritten in this commit.
* **Animation** -- the frame strip, with a real per-frame ``duration_ms``
  instead of the hard-coded 100 ms the sidebar's "Add frame" wrote, an order
  the arrows can change, and a preview that plays it through the same
  ``tileset.frame_at`` the canvas substitutes gids with.

**No tileset reordering, and the reason is not squeamishness**: order *is*
firstgid order, baked into every painted cell, so reordering means renumbering
the map. Tiled reorders its *tabs*, not its ids.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from imgui_bundle import imgui

from .. import controls, icons, plotter_mode, theme, widgets
from ..manual import render as manual_render
from ..tilegrid import tileset as tileset_lib
from ..tilegrid.tileset import TileEllipse, TileFrame, TileRect
from ..tokens import sp
from . import plotter_textures

#: How big one tile is drawn in the collision tab, in design px. Large enough
#: that a 16 px tile's corner is a target a mouse can hit: the shapes are in
#: *tile* pixels, so a 1 px handle at 1:1 is unusable by construction.
COLLISION_VIEW = 256.0

#: The tile grid's cell in the Tiles tab, in design px.
TILE_CELL = 48.0

TABS = ("Tiles", "Collision", "Animation")


def active(ctx: Any) -> bool:
    """Whether the sheet is what the centre pane should draw."""

    state = getattr(ctx.state, "plotter", None)
    if state is None or state.editing_tileset is None:
        return False
    tab = state.active
    return tab is not None and 0 <= state.editing_tileset < len(tab.doc.tilesets)


def draw(ctx: Any) -> None:
    state = plotter_mode.ensure(ctx)
    tab = state.active
    if tab is None or state.editing_tileset is None:
        return
    index = state.editing_tileset
    if not 0 <= index < len(tab.doc.tilesets):
        state.editing_tileset = None
        return
    ref = tab.doc.tilesets[index]

    widgets.section(f"Tileset -- {ref.tileset.name or 'untitled'}")
    manual_render.help_button(ctx, "plotter-tileset-editor")
    imgui.same_line()
    if widgets.ghost_button("Back to the map##tsback"):
        state.editing_tileset = None
        return
    widgets.muted(
        f"{len(ref.tileset)} tiles, {ref.tileset.tile_w} x {ref.tileset.tile_h} px, "
        f"first gid {ref.first_gid}"
    )

    changed, picked = controls.segmented_choice(
        "plotter-tileset-tabs",
        [(name, name) for name in TABS],
        state.tileset_tab if state.tileset_tab in TABS else TABS[0],
    )
    if changed:
        state.tileset_tab = picked
    which = state.tileset_tab if state.tileset_tab in TABS else TABS[0]
    imgui.separator()
    if which == "Tiles":
        _tiles_tab(ctx, state, tab, ref, index)
    elif which == "Collision":
        _collision_tab(ctx, state, tab, ref, index)
    else:
        _animation_tab(ctx, state, tab, ref, index)


# --- Tiles ------------------------------------------------------------------


def _tiles_tab(ctx: Any, state: Any, tab: Any, ref: Any, index: int) -> None:
    """The atlas at a readable size, and the per-tile form beside it."""

    _tile_grid(ctx, state, ref)
    imgui.separator()
    local = int(state.editing_tile)
    meta = ref.tileset.meta_of(local)
    widgets.muted(f"Tile {local}")

    class_name = widgets.input_text(
        "##ts-class", meta.class_name, max_length=64, hint="class"
    )
    changed, probability = controls.input_float("Probability", float(meta.probability))
    if class_name != meta.class_name or changed:
        tab.doc.set_tile_meta(
            index,
            local,
            dataclasses.replace(
                meta, class_name=class_name, probability=max(0.0, probability)
            ),
        )
    if meta.probability == 0.0:
        widgets.muted_wrapped(
            "Never chosen by a random brush, and always placeable by hand."
        )
    widgets.muted("Properties")
    from . import plotter_layers

    plotter_layers.property_editor(
        ctx,
        f"tsedit:{tab.uid}:{index}:{local}",
        meta.properties,
        lambda values: tab.doc.set_tile_meta(
            index, local, dataclasses.replace(meta, properties=values)
        ),
    )


def _tile_grid(ctx: Any, state: Any, ref: Any) -> None:
    """Every tile as a button, wrapped to the pane. Selection is one integer."""

    tileset = ref.tileset
    cell = sp(TILE_CELL)
    avail = max(1.0, imgui.get_content_region_avail().x)
    across = max(1, int(avail // (cell + imgui.get_style().item_spacing.x)))
    for local in range(len(tileset)):
        if local % across:
            imgui.same_line()
        selected = local == int(state.editing_tile)
        if controls.button(
            f"{local}##tsedit-tile{local}",
            (cell, cell),
            selected=selected,
            control_size=controls.ControlSize.COMPACT,
        ):
            state.editing_tile = local
    imgui.new_line()


# --- Collision --------------------------------------------------------------


def _collision_tab(ctx: Any, state: Any, tab: Any, ref: Any, index: int) -> None:
    """One tile, large, and the shapes on it.

    This is the editor ``COMPAT.md`` promised. The shapes go through
    ``doc.set_tile_meta``, which is already one undo step and already writes
    the sparse record's own rule -- an all-default record is removed rather
    than stored.
    """

    local = int(state.editing_tile)
    meta = ref.tileset.meta_of(local)
    tileset = ref.tileset
    side = sp(COLLISION_VIEW)
    origin = imgui.get_cursor_screen_pos()
    draw_list = imgui.get_window_draw_list()
    draw_list.add_rect_filled(
        (origin.x, origin.y),
        (origin.x + side, origin.y + side),
        imgui.get_color_u32(theme.rgba(theme.PANEL)),
    )
    scale = side / max(1, tileset.tile_w)
    for shape in meta.collision:
        x, y = float(shape.x) * scale, float(shape.y) * scale
        w, h = float(shape.w) * scale, float(shape.h) * scale
        colour = imgui.get_color_u32(theme.rgba(theme.ACCENT, 0.55))
        if isinstance(shape, TileEllipse):
            draw_list.add_ellipse(
                (origin.x + x + w / 2, origin.y + y + h / 2), (w / 2, h / 2), colour
            )
        else:
            draw_list.add_rect(
                (origin.x + x, origin.y + y), (origin.x + x + w, origin.y + y + h), colour
            )
    imgui.dummy((side, side))
    widgets.muted(
        f"{len(meta.collision)} shape(s), in tile pixels "
        f"({tileset.tile_w} x {tileset.tile_h})"
    )

    width = widgets.grid_width(3)
    if controls.button(f"Add box##tscol{local}", (width, 0)):
        _add_shape(tab, index, local, meta, TileRect)
    imgui.same_line()
    if controls.button(f"Add ellipse##tscole{local}", (width, 0)):
        _add_shape(tab, index, local, meta, TileEllipse)
    imgui.same_line()
    if widgets.disabled_button(
        f"Clear##tscolc{local}",
        bool(meta.collision),
        (width, 0),
        reason="This tile has no collision shapes.",
    ):
        tab.doc.set_tile_meta(index, local, dataclasses.replace(meta, collision=()))


def _add_shape(tab: Any, index: int, local: int, meta: Any, kind: Any) -> None:
    """A shape covering the whole tile, which is the one obviously editable
    starting size -- a zero-sized box is a shape you cannot grab."""

    tileset = tab.doc.tilesets[index].tileset
    shape = kind(x=0.0, y=0.0, w=float(tileset.tile_w), h=float(tileset.tile_h))
    tab.doc.set_tile_meta(
        index, local, dataclasses.replace(meta, collision=(*meta.collision, shape))
    )


# --- Animation --------------------------------------------------------------


#: How big the preview tile is drawn, in design px.
PREVIEW_VIEW = 128.0


def moved_frame(frames: Any, at: int, delta: int) -> tuple:
    """``frames`` with the one at ``at`` shifted by ``delta``. -> a new tuple.

    Returns the input unchanged when the move would fall off either end, which
    is what lets the caller compare and push nothing: reordering is one undo
    step *when something moved*, and a no-op that still wrote would put an empty
    step on the stack for every click on a disabled-looking arrow.

    Order is the whole content of an animation -- the frames name tile ids and
    the durations ride with them -- so this is a list operation and not a swap
    of two ids: swapping ids alone would leave each frame's duration behind.
    """
    frames = tuple(frames)
    target = at + int(delta)
    if not 0 <= at < len(frames) or not 0 <= target < len(frames):
        return frames
    moving = list(frames)
    moving.insert(target, moving.pop(at))
    return tuple(moving)


def playing_frame(state: Any, frames: Any, clock: Any = time.monotonic) -> int | None:
    """Which frame the preview is showing, or ``None`` when it is not playing.

    Off the wall clock and the moment Play was pressed, so the preview runs at
    the durations the tile actually carries rather than at the frame rate the
    app happens to be drawing -- and so pressing Play always starts at frame 0.
    """
    if not state.tileset_playing:
        return None
    elapsed = max(0.0, float(clock()) - float(state.tileset_play_at)) * 1000.0
    return tileset_lib.frame_at(frames, int(elapsed))


def _animation_tab(ctx: Any, state: Any, tab: Any, ref: Any, index: int) -> None:
    """The frame strip, with a duration per frame, an order and a preview.

    The sidebar wrote a hard-coded 100 ms and had no way to change it, so every
    animated tile in a map made here ran at exactly ten frames a second. The
    duration fixed that and left two things an animation editor cannot be
    without: the frames could not be **reordered** -- the only way to move one
    was to delete every frame after it and re-add them -- and nothing ever
    **played**, so the durations were numbers you typed and then went to the map
    to see. Both are here now, and the preview is the same
    ``tileset.frame_at`` the canvas substitutes gids through, so what plays here
    is what plays there.
    """

    local = int(state.editing_tile)
    meta = ref.tileset.meta_of(local)
    frames = meta.animation

    def write(replacement: Any) -> None:
        tab.doc.set_tile_meta(
            index, local, dataclasses.replace(meta, animation=tuple(replacement))
        )

    widgets.muted(f"Tile {local}: {len(frames)} frame(s)")
    showing = playing_frame(state, frames)
    _preview(ctx, state, tab, ref, index, frames, showing)

    for at, frame in enumerate(frames):
        imgui.push_id(at)
        imgui.set_next_item_width(sp(90))
        changed, duration = controls.input_int("ms", int(frame.duration_ms))
        if changed:
            edited = list(frames)
            edited[at] = TileFrame(local_id=frame.local_id, duration_ms=max(1, duration))
            write(edited)
        imgui.same_line()
        widgets.muted(f"tile {frame.local_id}" + ("  <" if at == showing else ""))
        imgui.same_line()
        # Up and down rather than a drag: the strip is a column of rows a few
        # pixels tall, and a drag reorder in one needs a hit band, an insertion
        # marker and a cancel. Two buttons say the same thing and can be aimed.
        if widgets.icon_button(
            f"{icons.ARROW_UP}##tsup{at}",
            "Move this frame earlier",
            borderless=True,
            enabled=at > 0,
        ):
            write(moved_frame(frames, at, -1))
        imgui.same_line()
        if widgets.icon_button(
            f"{icons.ARROW_DOWN}##tsdown{at}",
            "Move this frame later",
            borderless=True,
            enabled=at < len(frames) - 1,
        ):
            write(moved_frame(frames, at, 1))
        imgui.same_line()
        if widgets.icon_button(f"x##tsdel{at}", "Remove this frame", borderless=True):
            write([f for pos, f in enumerate(frames) if pos != at])
        imgui.pop_id()
    if controls.button(f"Add this tile as a frame##tsaddf{local}", (-1, 0)):
        write((*frames, TileFrame(local_id=local, duration_ms=100)))
    widgets.muted_wrapped(
        "Frames name local ids inside this tileset, which is what makes a "
        "tileset self-contained: a firstgid belongs to the map that loaded it."
    )


def _preview(
    ctx: Any,
    state: Any,
    tab: Any,
    ref: Any,
    index: int,
    frames: Any,
    showing: int | None,
) -> None:
    """The Play button and the tile it plays, at a readable size.

    Drawn even with nothing to play, because a Play button that appears only
    once an animation exists is a control the user meets for the first time
    after they have already guessed there is none.
    """
    tileset_ref = ref.tileset
    side = sp(PREVIEW_VIEW)
    at = showing if showing is not None else 0
    shown_id = int(frames[at].local_id) if frames else int(state.editing_tile)
    texture = plotter_textures.tileset_texture(
        ctx, tab.uid, index, tileset_ref, tab.doc.tileset_epoch
    )
    if texture is None:
        # No GL context: the headless suite. The same square the picture would
        # occupy, for ``plotter_tileset``'s reason.
        widgets.thumb_placeholder(side, icons.GRID, side)
    else:
        u0, v0, u1, v1 = tileset_ref.uv(shown_id)
        imgui.image(widgets.texture_ref(texture), (side, side), (u0, v0), (u1, v1))
    playing = bool(state.tileset_playing)
    if controls.button(
        ("Stop##tsplay" if playing else "Play##tsplay"),
        (widgets.grid_width(3), 0),
        selected=playing,
    ):
        state.tileset_playing = not playing
        # Restart from frame 0 every time, which is what makes Play readable as
        # "show me this animation" rather than "resume wherever the clock is".
        state.tileset_play_at = time.monotonic()
    imgui.same_line()
    if not frames:
        widgets.muted("No frames yet -- add one below.")
    elif showing is None:
        widgets.muted(f"Stopped. Tile {shown_id}.")
    else:
        widgets.muted(f"Frame {showing + 1} of {len(frames)}, tile {shown_id}.")
