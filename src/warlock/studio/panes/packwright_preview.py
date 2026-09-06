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

from pathlib import Path
from typing import Any

from .. import docmodes, icons, inker_state, packwright_mode, theme, widgets
from ..tokens import sp
from . import overlay, packwright_textures

#: This pane's square, in design pixels. ``widgets.CHECKER``'s value, named
#: here because the preview has always had one and a call site that spelled
#: the number would be a third copy of it.
CHECKER = widgets.CHECKER


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
        if tab.pack_error:
            imgui.dummy((0, 40))
            widgets.text_colored(theme.ERR, tab.pack_error)
        elif tab.packing:
            imgui.dummy((0, 40))
            widgets.busy("Packing")
        else:
            # The one viewport that answered "nothing here yet" with a muted
            # sentence in the top-left corner while the other nine drew the
            # icon-title-hint form. One empty-state vocabulary (2026-09-05),
            # and this is the state with exactly one thing to do, so it is
            # also the sentence's own verb as a button.
            overlay.centred_empty(
                *overlay.PLACEHOLDERS["packwright"],
                action=overlay.action_for(ctx, "packwright"),
            )
        return
    if tab.pack_stale_why:
        # Above the picture rather than instead of it: the last good atlas is
        # still worth looking at, and what it needed was a mark saying it is
        # not the current one.
        widgets.text_colored(theme.ERR, f"{icons.TRIANGLE_ALERT} {tab.pack_stale_why}")

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
    widgets.checkerboard(draw_list, lo, hi, step=CHECKER)

    texture = packwright_textures.atlas_texture(ctx, tab)
    if texture is not None:
        draw_list.add_image(widgets.texture_ref(texture), lo, hi)
    draw_list.add_rect(lo, hi, imgui.get_color_u32(theme.rgba(theme.EDGE)))
    _outlines(state, tab, draw_list, (origin.x, origin.y))
    draw_list.pop_clip_rect()

    _events(state, tab, (origin.x, origin.y), hovered, region)
    widgets.muted(
        f"{size_px[0]} x {size_px[1]} px  --  {int(view.zoom * 100)}%  --  "
        f"{len(tab.layout.frames) if tab.layout else 0} sprite(s)"
    )
    note = _area_note(tab, size_px)
    if note:
        widgets.muted(note)


def _area_note(tab: Any, size_px: tuple[int, int]) -> str | None:
    """The shrink-to-fit line: what the packed atlas' area is next to the pile
    of source pixels it came from, in words rather than left for the user to
    do the division themselves.

    A pure function of the tab -- no imgui in it -- so it is worth testing on
    its own rather than only through the smoke suite that draws it. ``None``
    with nothing packed yet, or with every source at zero area (an empty
    document has neither), so the caller draws nothing rather than a
    percentage of zero.
    """
    if not tab.doc.sources:
        return None
    source_area = sum(source.sprite.width * source.sprite.height for source in tab.doc.sources)
    if source_area <= 0:
        return None
    ratio = (size_px[0] * size_px[1]) / source_area
    if ratio <= 0.995:
        return f"Packed to {ratio:.0%} of the source pixels' area."
    if ratio >= 1.005:
        return (
            f"Packed to {ratio:.0%} of the source pixels' area -- turn off "
            "power-of-two, or MaxRects, for a tighter fit."
        )
    return "Packed to the same area as the source pixels."


def _tabs(ctx: Any, state: Any) -> None:

    docmodes.tab_bar(
        ctx, state, "packwright-tabs", lambda tab: packwright_mode.close_tab(ctx, tab.uid)
    )


def _empty(ctx: Any) -> None:
    widgets.nothing_open(
        "Start an atlas, open a .wpack, or drop images on the window.",
        [
            ("New atlas", lambda: packwright_mode.new_document(ctx)),
            ("Open a file...", lambda: packwright_mode.ask_open(ctx)),
        ],
        # The same recents the bridge lists, here too -- every workspace's
        # empty screen offers the way back (2026-09-05).
        recent_paths=packwright_mode.recent_paths(ctx),
        on_open=lambda path: packwright_mode.open_path(ctx, Path(path)),
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
        _pivot_mark(draw_list, view, origin, frame, accent if selected else faint)


def _pivot_mark(draw_list: Any, view: Any, origin, frame: Any, colour: int) -> None:
    """A cross where this frame's anchor sits, if it has one.

    The preview half of the anchor control: a number in a field is not
    something anybody can aim, and this is the only surface that shows the
    sprite. Placed by the frame's *trim* exactly as ``texturepacker._pivot``
    normalises it -- the pivot is stored in the sprite's own untrimmed pixels
    and the frame is what survived the trim, so a cross drawn without that
    subtraction sits wherever the trim happened to cut.
    """
    if frame.pivot is None:
        return
    x = frame.x + float(frame.pivot[0]) - frame.trim[0]
    y = frame.y + float(frame.pivot[1]) - frame.trim[1]
    at = inker_state.to_screen(view, origin, x, y)
    arm = sp(5)
    draw_list.add_line((at[0] - arm, at[1]), (at[0] + arm, at[1]), colour, sp(1))
    draw_list.add_line((at[0], at[1] - arm), (at[0], at[1] + arm), colour, sp(1))
    draw_list.add_circle(at, arm * 0.6, colour, 12)


def _events(state: Any, tab: Any, origin, hovered: bool, region) -> None:
    from imgui_bundle import imgui

    from .. import imgui_backend

    io = imgui.get_io()
    view = tab.view
    if hovered and (io.mouse_wheel or io.mouse_wheel_h):
        mouse = imgui.get_mouse_pos()
        # The rule every 2-D canvas shares (``inker_state.wheel``): the wheel
        # zooms on the 5% lattice, Shift+wheel and a tilt wheel scroll
        # sideways. Until 2026-09-05 this pane zoomed multiplicatively on the
        # backend-halved count and never landed on a round percentage.
        along = inker_state.wheel(
            view, origin, (mouse.x, mouse.y),
            io.mouse_wheel / imgui_backend.WHEEL_SCALE,
            io.mouse_wheel_h / imgui_backend.WHEEL_SCALE,
            shift=bool(io.key_shift),
        )
        if along:
            view.pan = (view.pan[0] + inker_state.scroll_step(region[0]) * along, view.pan[1])
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
