"""Packwright's centre pane: the packed atlas, at whatever zoom.

Draw-only, and it is also **the mode's heartbeat**. There is no per-mode update
hook anywhere in the app, so the one thing that runs every frame is whichever
pane draws -- the ``motion.py`` idiom, and the same place Inker ticks playback
from. This pane calls ``packwright_mode.pump``, which repacks *only* when the
document says it is dirty and only when the runner accepts the submit.

A checkerboard behind the atlas rather than a flat colour: the whole question a
user brings to this pane is "where is the transparency", and a solid background
answers it wrongly at exactly one colour.
"""

from __future__ import annotations

from typing import Any

from .. import inker_state, packwright_mode, theme, widgets
from ..tokens import sp
from . import packwright_textures

CHECKER = 8


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = packwright_mode.ensure(ctx)
    _tabs(ctx, state)
    # The heartbeat, before the early returns: a document with nothing packed
    # yet is exactly the one that needs the first pack requested.
    packwright_mode.pump(ctx)

    tab = state.active
    if tab is None:
        _empty(ctx)
        return
    if tab.atlas is None:
        imgui.dummy((0, 40))
        if tab.pack_error:
            widgets.text_colored(theme.ERR, tab.pack_error)
        else:
            widgets.muted("Packing..." if tab.packing else "Add a sprite to see the atlas.")
        return

    view = tab.view
    avail = imgui.get_content_region_avail()
    region = (max(float(avail.x), 1.0), max(float(avail.y), 1.0))
    size_px = (int(tab.atlas.shape[1]), int(tab.atlas.shape[0]))
    if not view.fitted:
        inker_state.fit(view, size_px, region)
    if view.pending_zoom is not None:
        inker_state.centre(view, size_px, region, view.pending_zoom)
        view.pending_zoom = None

    origin = imgui.get_cursor_screen_pos()
    imgui.invisible_button("packwright-preview", region)
    hovered = imgui.is_item_hovered()
    draw_list = imgui.get_window_draw_list()
    draw_list.push_clip_rect(
        (origin.x, origin.y), (origin.x + region[0], origin.y + region[1]), True
    )
    lo = inker_state.to_screen(view, (origin.x, origin.y), 0, 0)
    hi = inker_state.to_screen(view, (origin.x, origin.y), size_px[0], size_px[1])
    _checkerboard(draw_list, lo, hi)

    texture = packwright_textures.atlas_texture(ctx, tab)
    if texture is not None:
        draw_list.add_image(widgets.texture_ref(texture), lo, hi)
    draw_list.add_rect(lo, hi, imgui.get_color_u32(theme.rgba(theme.EDGE)))
    _outlines(state, tab, draw_list, (origin.x, origin.y))
    draw_list.pop_clip_rect()

    _events(state, tab, (origin.x, origin.y), hovered)
    widgets.muted(
        f"{size_px[0]} x {size_px[1]} px  --  {int(view.zoom * 100)}%  --  "
        f"{len(tab.layout.frames) if tab.layout else 0} sprite(s)"
    )


def _tabs(ctx: Any, state: Any) -> None:
    from imgui_bundle import imgui

    if not state.docs:
        return
    # ``auto_select_new_tabs`` for ``inker_canvas``'s reason: without it, a
    # second opened document lands behind the first and "Open" looks inert.
    flags = (
        imgui.TabBarFlags_.reorderable.value
        | imgui.TabBarFlags_.auto_select_new_tabs.value
    )
    if imgui.begin_tab_bar("packwright-tabs", flags):
        for tab in list(state.docs):
            # imgui's own dot, not a ``"* "`` prefix -- see ``inker_canvas``.
            item_flags = imgui.TabItemFlags_.unsaved_document.value if tab.dirty else 0
            opened, keep = imgui.begin_tab_item(tab.label, True, item_flags)
            if opened:
                state.activate(tab.uid)
                imgui.end_tab_item()
            if not keep:
                packwright_mode.close_tab(ctx, tab.uid)
        imgui.end_tab_bar()


def _empty(ctx: Any) -> None:
    from imgui_bundle import imgui

    imgui.dummy((0, 40))
    imgui.text("Nothing open")
    widgets.muted("Start an atlas, open a .wpack, or drop images on the window.")
    imgui.dummy((0, 16))
    if imgui.button("New atlas", (240, 0)):
        packwright_mode.new_document(ctx)
    imgui.dummy((0, 8))
    if imgui.button("Open a file...", (240, 0)):
        packwright_mode.ask_open(ctx)


def _checkerboard(draw_list: Any, lo, hi) -> None:
    """Drawn as rectangles rather than as a tiled texture.

    A texture would be the right call at Inker's canvas sizes; an atlas preview
    is bounded by the pane, so the worst case here is a few hundred quads and
    the alternative is a second texture to create, register and release.
    """
    from imgui_bundle import imgui

    light = imgui.get_color_u32(theme.rgba(theme.ELEV_2))
    dark = imgui.get_color_u32(theme.rgba(theme.ELEV_1))
    draw_list.add_rect_filled(lo, hi, dark)
    step = sp(CHECKER)
    rows = int((hi[1] - lo[1]) // step) + 1
    columns = int((hi[0] - lo[0]) // step) + 1
    # Bounded: past this the pattern is finer than it is useful and the cost is
    # not. A flat fill is the honest fallback.
    if rows * columns > 4096:
        return
    for row in range(rows):
        for column in range(columns):
            if (row + column) % 2:
                continue
            x0 = lo[0] + column * step
            y0 = lo[1] + row * step
            draw_list.add_rect_filled(
                (x0, y0), (min(x0 + step, hi[0]), min(y0 + step, hi[1])), light
            )


def _outlines(state: Any, tab: Any, draw_list: Any, origin) -> None:
    from imgui_bundle import imgui

    if tab.layout is None:
        return
    view = tab.view
    by_key = {source.key: source.uid for source in tab.doc.sources}
    faint = imgui.get_color_u32(theme.rgba(theme.EDGE, 0.7))
    accent = imgui.get_color_u32(theme.rgba(theme.ACCENT))
    for frame in tab.layout.frames:
        selected = state.selected is not None and by_key.get(frame.key) == state.selected
        p0 = inker_state.to_screen(view, origin, frame.x, frame.y)
        p1 = inker_state.to_screen(view, origin, frame.x + frame.w, frame.y + frame.h)
        draw_list.add_rect(
            p0, p1, accent if selected else faint, 0.0, sp(2 if selected else 1)
        )


def _events(state: Any, tab: Any, origin, hovered: bool) -> None:
    from imgui_bundle import imgui

    io = imgui.get_io()
    view = tab.view
    if hovered and io.mouse_wheel:
        mouse = imgui.get_mouse_pos()
        inker_state.zoom_about(view, origin, (mouse.x, mouse.y), io.mouse_wheel)
    if hovered and imgui.is_mouse_dragging(2):
        delta = imgui.get_mouse_drag_delta(2)
        imgui.reset_mouse_drag_delta(2)
        view.pan = (view.pan[0] + delta.x, view.pan[1] + delta.y)
    if hovered and imgui.is_mouse_clicked(0) and tab.layout is not None:
        mouse = imgui.get_mouse_pos()
        x, y = inker_state.to_image(view, origin, mouse.x, mouse.y)
        by_key = {source.key: source.uid for source in tab.doc.sources}
        # Reversed, so a sprite drawn over another is reachable -- the rule the
        # map canvas follows for overlapping objects.
        for frame in reversed(tab.layout.frames):
            if frame.x <= x <= frame.x + frame.w and frame.y <= y <= frame.y + frame.h:
                state.selected = by_key.get(frame.key)
                return
        state.selected = None
