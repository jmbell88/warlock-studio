"""The tileset editor: a sheet over the centre pane, not a mode and not a doc.

Drawn *instead of* ``plotter_canvas`` when ``state.editing_tileset is not
None`` -- precisely the branch ``_review_workspace`` already takes, and
``PaneRole.SHEET`` already exists for it.

**Not a mode.** Adding one is a 21-place checklist including literal English
prose asserting the mode *count* in two documents. **Not a document kind**
either: that would teach ``state.active``, ``plotter_canvas.draw``, four panes,
the journal, the guard and the save path a second shape, for something that is
a *view* of a map's own tileset.

Four tabs, which are the four questions a tileset answers:

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
* **Terrain** -- the Wang sets, which until 2026-08-30 nothing in this app could
  make. ``WangColour``/``WangSet`` had round-tripped through ``.tsx`` and
  ``.wmap`` since they landed and the Terrain *tool* had painted with them the
  whole time, but the only way to get one was to import a file Tiled wrote:
  the single ``wangset`` reference anywhere in ``panes/`` was a read-only
  swatch enumeration. The tab is the author, and everything downstream of it
  -- the tool's picker, the constraint matcher, both exporters -- was already
  built and needed no plumbing at all.

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
from ..tilegrid import wang as wanglib
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

#: How big a Wang slot marker is *drawn*, in design px.
WANG_MARKER = 16.0

#: How near the pointer has to be to a Wang marker, in design px.
#:
#: Much larger than :data:`picking.GRAB_RADIUS`, and the difference is not
#: sloppiness: a collision handle is one of eight grips that can sit a single
#: tile pixel apart on a small shape, so it needs a *tight* radius or the wrong
#: one wins. The eight Wang markers are at fixed positions half a view apart --
#: :data:`COLLISION_VIEW` / 2, so 128 design px -- and there is nothing else on
#: the square to hit, so the useful radius is the one that makes a corner an
#: easy target. 56 is under half that spacing, so no two regions can overlap and
#: the centre of the tile still means "nothing".
WANG_GRAB = 56.0

#: What a new Wang colour is dressed in, cycled. Not white: two colours the
#: same colour is a set whose markers cannot be told apart, which is a set the
#: user has to name to read.
WANG_SWATCHES = ("#4f9d52", "#c8a165", "#3f7fd0", "#b0413e", "#8d6cab", "#d0c341")

TABS = ("Tiles", "Collision", "Animation", "Terrain")


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
    elif which == "Terrain":
        _terrain_tab(ctx, state, tab, ref, index)
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


# --- Terrain -----------------------------------------------------------------
#
# Authoring a Wang set, which is the one thing this app could not do to one.
#
# **No new data types and no new write door.** ``WangColour``/``WangSet`` and
# ``Tileset.wangsets`` already existed, round-tripped and painted;
# ``MapDoc.replace_tileset`` -- what the Inker polish trip and *Reload the
# image...* already come back through -- is the undoable door a tileset's own
# content changes by. So every gesture below is one pure edit from
# ``tilegrid.wang`` followed by one ``replace_tileset``, and there is no second
# path for a future reader to find and wonder about.
#
# One click is one undo step, which is this editor's existing granularity
# (``set_tile_meta`` per control on the other three tabs). Unlike Wave 7's
# collision drags these are *discrete* clicks, so none of the begin/live/end
# session machinery is needed or used.


def _wangset_at(state: Any, sets: Any) -> int:
    """Which set the tab is on, pulled back into range.

    Clamped on read rather than fixed up at every write: a set can vanish under
    the selection by an undo as easily as by the Delete button, and a tab that
    only corrected itself when *it* removed one would draw off the end of the
    list after a Ctrl+Z.
    """
    at = int(getattr(state, "tileset_wangset", 0))
    return at if 0 <= at < len(sets) else 0


def _write_wangsets(tab: Any, index: int, wangsets: Any) -> None:
    """The whole write path of this tab: a new tileset through the one door.

    The tileset is rebuilt rather than written through -- ``Tileset`` is frozen
    because the UI keys its texture upload on identity -- and
    ``replace_tileset`` keeps the firstgid, the tile count and the declared
    blob terrains, so nothing already painted moves.
    """
    ref = tab.doc.tilesets[int(index)]
    tab.doc.replace_tileset(
        int(index), dataclasses.replace(ref.tileset, wangsets=tuple(wangsets))
    )


def authoring_refusal(tileset: Any) -> str:
    """Why a hand-authored Wang set cannot go on this tileset, or ``""``.

    **One sentence, spelled once**, because two places say it: the door
    (:func:`create_wangset`, as a toast) and the tab (as the prose where the
    *Create* button would otherwise be). A greyed control with no reason is a
    defect in this codebase's own terms, and a refusal whose message lives
    only at the door is one the user meets *after* the gesture.

    The rule is ``Tileset.terrains`` -- the blob preset Warlock's own terrain
    generator declares. A tileset carrying one already has a working terrain
    brush, so nothing is lost by refusing a second; and a tileset carrying
    *both* is a state neither format can hold. ``tsx.write_tsx`` writes the
    preset's ``<wangsets>`` and the foreign model's ``<wangsets>`` from
    separate doors, each returning early only on its own emptiness, so a
    tileset with both emits **two** blocks -- which Tiled's schema does not
    allow and which breaks the exporter's byte-identical pin. And the reader
    (``tsx._wang_model_of``) drops the foreign model whenever ``terrains`` is
    set, so the hand-authored set would not survive its own file: merging the
    two into one block would round-trip to exactly the same silent loss.
    ``docs/COMPAT.md`` says there are no silently-dropped rows, and stopping
    the combination by name is what keeps that true.
    """
    if not tuple(getattr(tileset, "terrains", ()) or ()):
        return ""
    return (
        "This tileset carries a generated terrain set, which is already its "
        "terrain brush -- so a second, hand-authored set is refused here. A "
        "Tiled tileset holds one terrain block, and a set added on top of the "
        "generated one could not be written to a .tsx and would be dropped "
        "when the file was read back."
    )


def create_wangset(
    state: Any, tab: Any, index: int, kind: str = "corner", *, ctx: Any = None
) -> bool:
    """Add an empty Wang set to a tileset and select it. -> whether it landed.

    **The kind is chosen here and never afterwards**, deliberately. ``kind``
    decides which of the eight slots the set *uses* (``WangSet.slots``), so an
    editable kind would mean either carrying values in slots that no longer
    count -- which travel out to a ``.tsx`` and are read back by Tiled -- or
    clearing them on the switch, which is silent data loss one misclick away.
    Making a second set is the cheaper of the three.

    **A blob-preset tileset is refused by name**, for
    :func:`authoring_refusal`'s reason -- and the refusal is here rather than
    only in the tab because this is the *door*: the tab is what stops the
    gesture being offered, and this is what makes offering it impossible to
    get wrong. ``ctx`` is optional so the pure edit stays callable without one
    (which is how every other function in this section is tested); when it is
    given, the reason is said rather than swallowed.
    """
    tileset = tab.doc.tilesets[int(index)].tileset
    refusal = authoring_refusal(tileset)
    if refusal:
        if ctx is not None:
            ctx.toast(refusal, "error")
        return False
    sets = tuple(tileset.wangsets)
    fresh = wanglib.WangSet(name=f"Terrain {len(sets) + 1}", kind=str(kind))
    _write_wangsets(tab, index, (*sets, fresh))
    state.tileset_wangset = len(sets)
    state.tileset_wang_colour = 1
    return True


def delete_wangset(state: Any, tab: Any, index: int, at: int) -> None:
    """Remove one whole Wang set. Undoable, like every other write here.

    No usage refusal, and that is the difference between this and removing a
    *tileset*: a cell stores a gid, so dropping a tileset renumbers what every
    painted cell means, while a Wang set is only ever consulted to *choose* a
    gid. The cells it chose stay exactly as they are and simply stop growing
    new edges, which is a change the user can see and undo.
    """
    sets = tuple(tab.doc.tilesets[int(index)].tileset.wangsets)
    if not 0 <= int(at) < len(sets):
        return
    _write_wangsets(tab, index, tuple(s for i, s in enumerate(sets) if i != int(at)))
    state.tileset_wangset = max(0, int(at) - 1)


def add_wang_colour(state: Any, tab: Any, index: int, at: int) -> None:
    """One more colour on the selected set, and it goes into the hand.

    Selected for ``_add_shape``'s reason: a colour added and left unarmed is
    one the user has to discover they can click before they can discover they
    can paint a corner with it.
    """
    sets = tuple(tab.doc.tilesets[int(index)].tileset.wangsets)
    if not 0 <= int(at) < len(sets):
        return
    wangset = sets[int(at)]
    count = len(wangset.colours)
    fresh = wanglib.WangColour(
        name=f"Terrain {count + 1}", colour=WANG_SWATCHES[count % len(WANG_SWATCHES)]
    )
    _write_wangsets(tab, index, _swapped(sets, at, wanglib.with_colour(wangset, fresh)))
    state.tileset_wang_colour = count + 1


def remove_wang_colour(state: Any, tab: Any, index: int, at: int, colour: int) -> None:
    """Drop a colour, and with it every slot that named it.

    The renumbering is ``wang.without_colour``'s and is the whole reason this
    is not a list splice: a slot is a *position* in ``colours``.
    """
    sets = tuple(tab.doc.tilesets[int(index)].tileset.wangsets)
    if not 0 <= int(at) < len(sets):
        return
    wangset = sets[int(at)]
    if not 0 <= int(colour) < len(wangset.colours):
        return
    _write_wangsets(
        tab, index, _swapped(sets, at, wanglib.without_colour(wangset, int(colour)))
    )
    # The hand holds a 1-based colour number, and the numbers just moved.
    state.tileset_wang_colour = min(
        int(state.tileset_wang_colour), len(wangset.colours) - 1
    )


def set_wang_colour(
    state: Any, tab: Any, index: int, at: int, colour: int, entry: Any
) -> None:
    """A rename, a new swatch or a new probability on one colour."""
    sets = tuple(tab.doc.tilesets[int(index)].tileset.wangsets)
    if not 0 <= int(at) < len(sets):
        return
    edited = wanglib.with_colour_at(sets[int(at)], int(colour), entry)
    if edited != sets[int(at)]:
        _write_wangsets(tab, index, _swapped(sets, at, edited))


def rename_wangset(tab: Any, index: int, at: int, name: str) -> None:
    """The set's own name, which is what a Tiled user sees in the terrain bar."""
    sets = tuple(tab.doc.tilesets[int(index)].tileset.wangsets)
    if not 0 <= int(at) < len(sets) or sets[int(at)].name == str(name):
        return
    edited = dataclasses.replace(sets[int(at)], name=str(name))
    _write_wangsets(tab, index, _swapped(sets, at, edited))


def _terrain_tab(ctx: Any, state: Any, tab: Any, ref: Any, index: int) -> None:
    """The Wang sets on this tileset, their colours, and one tile's eight slots.

    **This tab draws its own tile strip**, and it is the only one of the four
    that does. The other three edit whichever tile the *Tiles* tab selected --
    the convention since the sheet landed -- and that is right for them: a
    collision shape or an animation is a property of one tile you went looking
    for. A Wang set is not. Authoring one means saying something about *every*
    tile in the set in a row, forty-seven of them for a blob-shaped one, and
    under the existing convention each of those would cost a trip to the Tiles
    tab and back. The strip is the same ``_tile_grid`` the Tiles tab draws,
    writing the same ``state.editing_tile``, so this is one selection shown in
    two places rather than a second selection that can disagree with the first:
    switch to Collision and you are on the tile you were just marking up.
    """

    tileset = ref.tileset
    sets = tuple(tileset.wangsets)
    _wangset_row(ctx, state, tab, index, sets)
    if not sets:
        # ``_wangset_row`` has already said why on a blob-preset tileset, and
        # "Create one to start" under that sentence would be an instruction to
        # press a button that is deliberately not there.
        if not authoring_refusal(tileset):
            widgets.muted_wrapped(
                "A terrain set says, per tile, which colour sits at each of its "
                "corners and edges -- and the Terrain tool then picks the tile whose "
                "corners match what is already around the cell. Create one to start."
            )
        return
    at = _wangset_at(state, sets)
    wangset = sets[at]
    imgui.separator()
    _wang_colours(ctx, state, tab, index, sets, at, wangset)
    if not wangset.colours:
        widgets.muted_wrapped(
            "Add a colour first: a slot is painted with a colour, so a set with "
            "none has nothing it can say about a tile."
        )
        return
    imgui.separator()
    widgets.muted("Tile")
    _tile_grid(ctx, state, ref)
    local = int(state.editing_tile)

    side = sp(COLLISION_VIEW)
    origin = imgui.get_cursor_screen_pos()
    view = picking.TileView(
        origin=(origin.x, origin.y),
        side=side,
        tile_w=tileset.tile_w,
        tile_h=tileset.tile_h,
    )
    _terrain_draw(ctx, tab, index, tileset, view, wangset, local)
    imgui.invisible_button(f"##tswang-view{local}", (side, side))
    _terrain_input(ctx, state, tab, index, local, view, imgui.is_item_hovered())

    wangid = wangset.wangid_of(local)
    used = [wangid[slot] for slot in wangset.slots]
    widgets.muted(
        f"Tile {local}: {sum(1 for value in used if value)} of {len(used)} "
        f"{wangset.kind} slot(s) set"
    )
    widgets.muted_wrapped(
        "Click a corner or an edge of the tile to put the colour in hand there, "
        "and Unset to clear one. A tile every one of whose slots is the same "
        "colour is that colour's interior -- the tile a click on the map lays "
        "down -- and the rest are what the neighbours choose between."
    )


def _wangset_row(ctx: Any, state: Any, tab: Any, index: int, sets: Any) -> None:
    """Which set is being edited, its name, and the two set-level buttons."""

    if sets:
        at = _wangset_at(state, sets)
        changed, picked = controls.segmented_choice(
            "plotter-wangset",
            [
                (str(slot), f"{entry.name or 'Terrain'} ({entry.kind})")
                for slot, entry in enumerate(sets)
            ],
            str(at),
        )
        if changed:
            state.tileset_wangset = int(picked)
            state.tileset_wang_colour = 1
            return
        name = widgets.input_text(
            "##tswang-name", sets[at].name, max_length=64, hint="set name"
        )
        if name != sets[at].name:
            rename_wangset(tab, index, at, name)

    width = widgets.grid_width(3)
    refusal = authoring_refusal(tab.doc.tilesets[int(index)].tileset)
    if refusal:
        # Said instead of drawn-and-disabled. A greyed *Create* button with no
        # sentence beside it is the shape this codebase treats as a defect in
        # its own right, and the sentence is the whole content of the refusal.
        # Delete stays reachable: a document written before this refusal
        # existed can carry both, and the way out of that state is to remove
        # the hand-authored set.
        widgets.muted_wrapped(refusal)
        if sets and controls.button("Delete this set##tswang-del", (width, 0)):
            delete_wangset(state, tab, index, _wangset_at(state, sets))
        return
    kind = str(getattr(state, "tileset_wang_kind", "corner"))
    if kind not in wanglib.WANG_KINDS:
        kind = "corner"
    changed, picked = controls.segmented_choice(
        "plotter-wangkind",
        [(entry, entry.title()) for entry in wanglib.WANG_KINDS],
        kind,
    )
    if changed:
        state.tileset_wang_kind = picked
        kind = picked
    if controls.button(f"Create a {kind} set##tswang-new", (width, 0)):
        create_wangset(state, tab, index, kind, ctx=ctx)
        return
    if sets:
        imgui.same_line()
        if controls.button("Delete this set##tswang-del", (width, 0)):
            delete_wangset(state, tab, index, _wangset_at(state, sets))
            return
    widgets.muted_wrapped(
        "A corner set decides a tile by its four corners, an edge set by its "
        "four sides, and a mixed set by all eight. The kind is fixed when the "
        "set is created, because changing it later would either strand values "
        "in slots that no longer count or throw them away."
    )


def _wang_colours(
    ctx: Any, state: Any, tab: Any, index: int, sets: Any, at: int, wangset: Any
) -> None:
    """The colour list, and which one is in the hand.

    ``Unset`` is a first-class entry rather than a right-click or a modifier:
    clearing a slot is as ordinary as setting one, and every gesture on this
    tab is then the same gesture with a different colour in hand.
    """
    from . import plotter_tools

    widgets.muted("Colours")
    width = widgets.grid_width(3)
    in_hand = int(getattr(state, "tileset_wang_colour", 1))
    if controls.button("Unset##tswang-c0", (width, 0), selected=in_hand == 0):
        state.tileset_wang_colour = 0
    for slot, colour in enumerate(wangset.colours):
        imgui.push_id(f"tswang-colour{slot}")
        fill = tuple(part / 255.0 for part in plotter_tools.hex_rgba(colour.colour))
        imgui.push_style_color(imgui.Col_.button.value, imgui.get_color_u32(fill))
        imgui.push_style_color(
            imgui.Col_.button_hovered.value, imgui.get_color_u32(fill)
        )
        pressed = controls.button(
            f"{colour.name or slot + 1}##pick", (width, 0), selected=in_hand == slot + 1
        )
        imgui.pop_style_color(2)
        if pressed:
            state.tileset_wang_colour = slot + 1
        imgui.same_line()
        name = widgets.input_text("##name", colour.name, max_length=64, hint="name")
        moved, value = controls.color_edit4(
            "##swatch",
            imgui.ImVec4(fill[0], fill[1], fill[2], 1.0),
            imgui.ColorEditFlags_.no_inputs.value
            | imgui.ColorEditFlags_.no_alpha.value
            | imgui.ColorEditFlags_.display_hex.value
            | imgui.ColorEditFlags_.picker_hue_bar.value,
        )
        weighed, probability = controls.input_float(
            "Probability", float(colour.probability)
        )
        if name != colour.name or moved or weighed:
            set_wang_colour(
                state,
                tab,
                index,
                at,
                slot,
                wanglib.WangColour(
                    name=name,
                    colour=_hex_text(value) if moved else colour.colour,
                    probability=max(0.0, probability),
                ),
            )
            imgui.pop_id()
            return
        if widgets.icon_button("x##drop", "Remove this colour", borderless=True):
            remove_wang_colour(state, tab, index, at, slot)
            imgui.pop_id()
            return
        imgui.pop_id()
    if controls.button("Add a colour##tswang-add", (width, 0)):
        add_wang_colour(state, tab, index, at)


def _hex_text(value: Any) -> str:
    """imgui's float colour as Tiled's ``#rrggbb``, which is what a set stores."""
    channels = (value.x, value.y, value.z) if hasattr(value, "x") else tuple(value)[:3]
    return "#" + "".join(
        f"{max(0, min(255, int(round(part * 255)))):02x}" for part in channels
    )


def _terrain_draw(
    ctx: Any, tab: Any, index: int, tileset: Any, view: Any, wangset: Any, local: int
) -> None:
    """The tile, large, with a marker at each slot the set uses.

    **The tile's own art is drawn under the markers**, which the Collision tab
    does not do and this one has to: a collision shape is judged against a
    silhouette you can hold in your head, but a Wang slot is a claim about
    *which corner of the picture is grass*, and marking that up against a blank
    square is guessing. Falls back to the same flat panel when there is no GL
    context, which is the headless suite.

    The marker positions come from :func:`wang.slot_points` -- the same
    function :func:`_terrain_input` builds its click regions from -- so what is
    drawn is exactly what is clickable.
    """
    from . import plotter_tools

    draw_list = imgui.get_window_draw_list()
    width, height = view.size
    low = view.origin
    high = (low[0] + width, low[1] + height)
    texture = plotter_textures.tileset_texture(
        ctx, tab.uid, index, tileset, tab.doc.tileset_epoch
    )
    if texture is None:
        draw_list.add_rect_filled(
            low, high, imgui.get_color_u32(theme.rgba(theme.PANEL))
        )
    else:
        u0, v0, u1, v1 = tileset.uv(int(local))
        draw_list.add_image(widgets.texture_ref(texture), low, high, (u0, v0), (u1, v1))
    outline = imgui.get_color_u32(theme.rgba(theme.ACCENT, 0.55))
    draw_list.add_rect(low, high, outline)
    wangid = wangset.wangid_of(local)
    radius = sp(WANG_MARKER) / 2.0
    for slot, point in wanglib.slot_points(
        view.tile_w, view.tile_h, wangset.slots
    ).items():
        centre = view.to_screen(*point)
        value = wangid[slot]
        fill = (
            tuple(
                part / 255.0
                for part in plotter_tools.hex_rgba(wangset.colours[value - 1].colour)
            )
            if value
            else theme.rgba(theme.PANEL, 0.75)
        )
        draw_list.add_circle_filled(centre, radius, imgui.get_color_u32(fill))
        # Every marker keeps its ring, set or not: the ring is what says "this
        # is a place you can click", and a slot only visible once it already
        # holds a colour is a control nobody finds.
        draw_list.add_circle(centre, radius, outline)


def _terrain_input(
    ctx: Any, state: Any, tab: Any, index: int, local: int, view: Any, hovered: bool
) -> None:
    """One click on one slot -- the whole Terrain gesture.

    Written as a dispatch beside ``_collision_input`` for its reason: the rule
    that matters is which region a press lands on, and a test calling a helper
    would assert around that rather than through it.
    ``tests/plotter/test_wang_authoring`` drives this with the shared synthetic
    pointer.

    No drag session, because there is nothing continuous here: a slot is one of
    a few discrete values and a click sets it. One click is one undo step.
    """
    if not (hovered and imgui.is_mouse_clicked(0)):
        return
    sets = tuple(tab.doc.tilesets[int(index)].tileset.wangsets)
    if not sets:
        return
    at = _wangset_at(state, sets)
    wangset = sets[at]
    colour = int(getattr(state, "tileset_wang_colour", 1))
    if colour > len(wangset.colours):
        # The hand can hold a colour an undo has taken away.
        return
    regions = {
        slot: view.to_screen(*point)
        for slot, point in wanglib.slot_points(
            view.tile_w, view.tile_h, wangset.slots
        ).items()
    }
    mouse = imgui.get_mouse_pos()
    slot = picking.nearest_region(regions, (mouse.x, mouse.y), sp(WANG_GRAB))
    if slot is None:
        return
    edited = wanglib.with_slot(wangset, int(local), int(slot), colour)
    if edited != wangset:
        _write_wangsets(tab, index, _swapped(sets, at, edited))
