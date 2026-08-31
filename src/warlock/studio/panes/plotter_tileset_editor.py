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
from ..tilegrid import picking
from ..tilegrid import tileset as tileset_lib
from ..tilegrid.tileset import TileEllipse, TileFrame, TilePolygon, TileRect
from ..tokens import sp
from . import plotter_textures

#: How big one tile is drawn in the collision tab, in design px. Large enough
#: that a 16 px tile's corner is a target a mouse can hit: the shapes are in
#: *tile* pixels, so a 1 px handle at 1:1 is unusable by construction.
COLLISION_VIEW = 256.0

#: The tile grid's cell in the Tiles tab, in design px.
TILE_CELL = 48.0

#: How big a resize handle or a polygon corner is *drawn*, in design px. The
#: radius it can be grabbed from is :data:`picking.GRAB_RADIUS`, which is
#: deliberately larger: a grip you can only hit dead centre is a grip that
#: works in a screenshot and not under a hand.
HANDLE_SIZE = 8.0

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
            # A shape selection names a position in *this* tile's tuple, so it
            # means nothing on the next one.
            state.tileset_shape = None
            state.clear_tileset_drag()
    imgui.new_line()


# --- Collision --------------------------------------------------------------


def _collision_tab(ctx: Any, state: Any, tab: Any, ref: Any, index: int) -> None:
    """One tile, large, and the shapes on it -- which are now *editable*.

    This is the editor ``COMPAT.md`` promised, and until this wave it was half
    of one: a shape could be added and cleared, but ``_add_shape`` hard-coded
    the geometry to the whole tile and the view under it was an
    ``imgui.dummy`` -- a picture, not a control. So every collision shape any
    map ever carried was the same full-tile box, and the format's ability to
    round-trip an arbitrary one had never been reachable from the app.

    The view is a real region now (see :func:`_collision_input`), and the
    geometry it edits comes back through the same ``doc.set_tile_meta`` door,
    with a drag arriving as **one** step through the session pair beside it.
    """

    local = int(state.editing_tile)
    meta = ref.tileset.meta_of(local)
    tileset = ref.tileset
    shapes = tuple(meta.collision)
    # The selection is a position in a tuple the document owns, so a shape
    # removed under it -- by an undo, by Clear, by the tile changing -- has to
    # drop the selection rather than leave it pointing at a neighbour.
    if state.tileset_shape is not None and not 0 <= int(state.tileset_shape) < len(shapes):
        state.tileset_shape = None
        state.clear_tileset_drag()
    chosen = state.tileset_shape

    side = sp(COLLISION_VIEW)
    origin = imgui.get_cursor_screen_pos()
    view = picking.TileView(
        origin=(origin.x, origin.y),
        side=side,
        tile_w=tileset.tile_w,
        tile_h=tileset.tile_h,
    )
    _collision_draw(view, shapes, chosen)
    # ``invisible_button`` rather than ``dummy``: a dummy is a rectangle of
    # nothing and imgui will not tell you whether the pointer is over it, which
    # is the first thing a gesture has to ask.
    imgui.invisible_button(f"##tscol-view{local}", (side, side))
    _collision_input(ctx, state, tab, index, local, view, imgui.is_item_hovered())

    widgets.muted(
        f"{len(shapes)} shape(s), in tile pixels "
        f"({tileset.tile_w} x {tileset.tile_h})"
    )

    width = widgets.grid_width(3)
    if controls.button(f"Add box##tscol{local}", (width, 0)):
        _add_shape(state, tab, index, local, meta, TileRect)
    imgui.same_line()
    if controls.button(f"Add ellipse##tscole{local}", (width, 0)):
        _add_shape(state, tab, index, local, meta, TileEllipse)
    imgui.same_line()
    if controls.button(f"Add polygon##tscolp{local}", (width, 0)):
        _add_shape(state, tab, index, local, meta, TilePolygon)

    if widgets.disabled_button(
        f"Clear##tscolc{local}",
        bool(shapes),
        (width, 0),
        reason="This tile has no collision shapes.",
    ):
        state.tileset_shape = None
        state.clear_tileset_drag()
        tab.doc.set_tile_meta(index, local, dataclasses.replace(meta, collision=()))
    if chosen is not None:
        imgui.same_line()
        if controls.button(f"Remove shape##tscolr{local}", (width, 0)):
            state.tileset_shape = None
            state.clear_tileset_drag()
            tab.doc.set_tile_meta(
                index,
                local,
                dataclasses.replace(
                    meta,
                    collision=tuple(
                        s for at, s in enumerate(shapes) if at != int(chosen)
                    ),
                ),
            )
            return
        shape = shapes[int(chosen)]
        x, y, w, h = picking.bounds(shape)
        kind = type(shape).__name__.removeprefix("Tile").lower()
        widgets.muted(
            f"Shape {int(chosen) + 1} of {len(shapes)}: {kind} at "
            f"{x:.0f}, {y:.0f} -- {w:.0f} x {h:.0f} px"
        )
    else:
        widgets.muted("Click a shape to select it.")
    widgets.muted_wrapped(
        "Drag a shape to move it and its square handles to resize it. A polygon "
        "is edited by its corners instead: drag one to move it, Ctrl+click an "
        "edge to add one, Alt+click a corner to remove it."
    )


def _collision_draw(view: Any, shapes: Any, chosen: int | None) -> None:
    """The tile, its shapes, and the handles of the selected one.

    The handle positions come from :func:`picking.box_handles` -- the same
    function :func:`_collision_input` arms them from -- so what is drawn here
    is exactly what is clickable there. Two lists would drift within a wave.
    """
    draw_list = imgui.get_window_draw_list()
    width, height = view.size
    draw_list.add_rect_filled(
        view.origin,
        (view.origin[0] + width, view.origin[1] + height),
        imgui.get_color_u32(theme.rgba(theme.PANEL)),
    )
    plain = imgui.get_color_u32(theme.rgba(theme.ACCENT, 0.55))
    lit = imgui.get_color_u32(theme.rgba(theme.ACCENT))
    for at, shape in enumerate(shapes):
        colour = lit if chosen is not None and at == int(chosen) else plain
        if isinstance(shape, TilePolygon):
            points = [view.to_screen(*p) for p in picking.vertices(shape)]
            for edge, start in enumerate(points):
                draw_list.add_line(start, points[(edge + 1) % len(points)], colour)
            continue
        x, y, w, h = picking.bounds(shape)
        top_left = view.to_screen(x, y)
        bottom_right = view.to_screen(x + w, y + h)
        if isinstance(shape, TileEllipse):
            draw_list.add_ellipse(
                (
                    (top_left[0] + bottom_right[0]) / 2,
                    (top_left[1] + bottom_right[1]) / 2,
                ),
                ((bottom_right[0] - top_left[0]) / 2, (bottom_right[1] - top_left[1]) / 2),
                colour,
            )
        else:
            draw_list.add_rect(top_left, bottom_right, colour)
    if chosen is None or not 0 <= int(chosen) < len(shapes):
        return
    shape = shapes[int(chosen)]
    grip = sp(HANDLE_SIZE) / 2.0
    for point in list(picking.box_handles(shape).values()) + list(
        picking.vertices(shape)
    ):
        at = view.to_screen(*point)
        draw_list.add_rect_filled(
            (at[0] - grip, at[1] - grip), (at[0] + grip, at[1] + grip), lit
        )


def _collision_input(
    ctx: Any, state: Any, tab: Any, index: int, local: int, view: Any, hovered: bool
) -> None:
    """Press, drag, release over one tile -- the whole collision gesture.

    Written as one dispatch for ``plotter_canvas._object_input``'s reason: the
    interesting rules (which grip wins, what a modifier means, when a step is
    pushed) live in the ordering, and a test that called a helper would assert
    around them rather than through them. ``tests/plotter/test_tile_collision``
    drives this with the shared synthetic pointer.

    Priority on the press is **handle, then vertex, then body, then empty**.
    Grips win because they sit *on* the outline and so overlap the body;
    checking the body first would make every one of them unreachable.
    """

    doc = tab.doc
    meta = doc.tilesets[index].tileset.meta_of(local)
    shapes = tuple(meta.collision)
    mouse = imgui.get_mouse_pos()
    screen = (mouse.x, mouse.y)
    point = view.to_tile(mouse.x, mouse.y)
    tile_w, tile_h = view.tile_w, view.tile_h

    if state.tileset_drag:
        chosen = state.tileset_shape
        if (
            imgui.is_mouse_released(0)
            or not imgui.is_mouse_down(0)
            or chosen is None
            or not 0 <= int(chosen) < len(shapes)
        ):
            # One step for the whole drag, pushed here. A release can also be
            # missed to focus loss, which is why the document closes its own
            # session at undo, redo and a history jump as well.
            doc.end_tile_meta_edit()
            state.clear_tileset_drag()
            return
        shape = shapes[int(chosen)]
        if state.tileset_drag == "move":
            grab = state.tileset_grab or (0.0, 0.0)
            x, y, _w, _h = picking.bounds(shape)
            replacement = picking.moved(
                shape, point[0] - grab[0] - x, point[1] - grab[1] - y, tile_w, tile_h
            )
        elif state.tileset_drag == "vertex":
            replacement = picking.with_vertex(
                shape, int(state.tileset_drag_vertex or 0), point, tile_w, tile_h
            )
        else:
            replacement = picking.resized(shape, state.tileset_drag, point, tile_w, tile_h)
        if replacement != shape:
            doc.live_tile_meta(
                dataclasses.replace(meta, collision=_swapped(shapes, chosen, replacement))
            )
        return

    if not (hovered and imgui.is_mouse_clicked(0)):
        return

    io = imgui.get_io()
    chosen = state.tileset_shape
    shape = (
        shapes[int(chosen)]
        if chosen is not None and 0 <= int(chosen) < len(shapes)
        else None
    )
    if shape is not None:
        handles = {
            name: view.to_screen(*at) for name, at in picking.box_handles(shape).items()
        }
        handle = picking.nearest_region(handles, screen, sp(picking.GRAB_RADIUS))
        if handle is not None:
            doc.begin_tile_meta_edit(index, local)
            state.tileset_drag = str(handle)
            return
        corners = {
            at: view.to_screen(*point_)
            for at, point_ in picking.vertex_regions(shape).items()
        }
        vertex = picking.nearest_region(corners, screen, sp(picking.GRAB_RADIUS))
        if vertex is not None:
            if io.key_alt:
                trimmed = picking.without_vertex(shape, int(vertex))
                if trimmed is None:
                    ctx.toast(
                        "A polygon needs at least three corners.", "error"
                    )
                    return
                doc.set_tile_meta(
                    index,
                    local,
                    dataclasses.replace(
                        meta, collision=_swapped(shapes, chosen, trimmed)
                    ),
                )
                return
            doc.begin_tile_meta_edit(index, local)
            state.tileset_drag = "vertex"
            state.tileset_drag_vertex = int(vertex)
            return
        if io.key_ctrl and isinstance(shape, TilePolygon):
            doc.set_tile_meta(
                index,
                local,
                dataclasses.replace(
                    meta,
                    collision=_swapped(
                        shapes,
                        chosen,
                        picking.inserted_vertex(shape, point, tile_w, tile_h),
                    ),
                ),
            )
            return

    found = picking.shape_at(shapes, point)
    if found is None:
        state.tileset_shape = None
        state.clear_tileset_drag()
        return
    state.tileset_shape = int(found)
    x, y, _w, _h = picking.bounds(shapes[int(found)])
    # The offset the pointer already has inside the shape, kept for the whole
    # drag: without it the shape's corner jumps to the pointer on the first
    # moved frame, which is a leap rather than a drag.
    state.tileset_grab = (point[0] - x, point[1] - y)
    doc.begin_tile_meta_edit(index, local)
    state.tileset_drag = "move"


def _swapped(shapes: Any, at: Any, replacement: Any) -> tuple:
    """``shapes`` with position ``at`` replaced. A new tuple every time: the
    shapes are frozen and the tileset is rebuilt around them."""
    edited = list(shapes)
    edited[int(at)] = replacement
    return tuple(edited)


def _add_shape(state: Any, tab: Any, index: int, local: int, meta: Any, kind: Any) -> None:
    """Add a whole-tile shape and *select* it.

    Selecting it is half the feature: the handles are drawn on the selected
    shape only, so a shape added and left unselected would be one the user has
    to discover they can click before they can discover they can drag it.
    """

    tileset = tab.doc.tilesets[index].tileset
    shape = picking.new_shape(kind, tileset.tile_w, tileset.tile_h)
    tab.doc.set_tile_meta(
        index, local, dataclasses.replace(meta, collision=(*meta.collision, shape))
    )
    state.tileset_shape = len(meta.collision)
    state.clear_tileset_drag()


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
