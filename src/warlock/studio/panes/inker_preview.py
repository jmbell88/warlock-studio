"""The clip, playing, while you draw on it.

The timeline's Play button is the *document's* playhead: it locks the tab
(``busy``), draws into the canvas, and stopping it moves the document to
wherever the eye last was. That is the right behaviour for "watch this" and the
wrong one for the thing an animator actually does, which is run the cycle in a
corner of the screen and keep drawing.

So this pane carries a **second playhead**, entirely on the tab
(``preview_playing`` and friends), and the discipline that makes it free is
short: it never touches ``playing`` or ``saving``, and it never calls
``set_current_frame``. What it draws is ``Document.frame_flat`` through the
same ``inker_textures.frame_texture`` onion skinning already uses, so it adds
no GPU state at all -- a frame the preview shows and a frame the onion skin
shows are one texture.

Two smaller decisions.

**It draws upright.** ``PaintView``'s rotation and flip are aids for *drawing*
-- turning the page to get a curve right, checking a face mirrored -- and a
preview is what the animation will look like. Following the view would rotate
the thing being checked along with the check.

**It re-flattens at most four times a second.** While the shown frame is the
one being painted, its stamp moves on every dab, and a full re-flatten and
upload per dab is the B24 problem again. A quarter second is not visible in a
clip playing at ten frames a second and is the same throttle the layer and cel
thumbnails use.
"""

from __future__ import annotations

import time
from typing import Any

from imgui_bundle import imgui

from .. import inker_mode, theme, widgets
from ..manual import render as manual_render
from ..tokens import sp
from . import inker_textures

#: The pane's height at the top of the right column, in design pixels.
PREVIEW_H = 180.0

#: How often the shown frame may be re-flattened while its pixels keep
#: changing. See the module docstring.
REFRESH_SECONDS = 0.25

#: The multipliers the speed combo offers. A short ladder rather than a slider:
#: these are the ones an animator asks for by name, and a continuous control
#: invites hunting for a number that does not exist.
SPEEDS = ((0.25, "0.25x"), (0.5, "0.5x"), (1.0, "1x"), (2.0, "2x"), (4.0, "4x"))

SCOPE_LABELS = (("clip", "Whole clip"), ("tag", "Active tag"))


def draw(ctx: Any) -> None:
    from .. import inker_walk as walk_session

    state = ctx.state.inker
    tab = None if state is None else state.active
    widgets.section("Preview")
    manual_render.help_button(ctx, "inker-preview")
    if tab is None:
        return
    # **A walk cycle borrows this pane, and does not get one of its own.** It is
    # a clip that has no document yet -- so there is nothing here for the code
    # below to flatten -- and it is the thing a user watches continuously while
    # dragging a joint, which is exactly what this pane is for. Its own setup
    # panel is a fourteen-row list over four sliders, so a preview at the end of
    # that would be permanently below the fold. See ``inker_walk.draw_preview``.
    session = walk_session.session(state, tab)
    if session is not None:
        from . import inker_walk as walk_pane

        walk_pane.draw_preview(ctx, tab, session)
        return
    if tab.doc.anim is None:
        return
    _tick(tab)
    _transport(ctx, tab)
    _image(ctx, tab)


def _tick(tab: Any) -> None:
    """``delta_time``, for ``inker_timeline._tick``'s reason: it is the number
    imgui animates everything else on screen by, so a stalled frame stalls the
    preview by the same amount it stalls the rest of the window."""
    if tab.preview_playing:
        inker_mode.tick_preview(tab, imgui.get_io().delta_time * 1000.0)


def _transport(ctx: Any, tab: Any) -> None:
    anim = tab.doc.anim
    last = len(anim.frames) - 1
    index = max(0, min(int(tab.preview_index), last))

    # Never disabled, not even mid-save: nothing here writes to the document,
    # and a preview that stopped being playable because a file was being
    # written would be refusing the one thing it is safe to do.
    if widgets.transport("inker-preview", tab.preview_playing, size=(sp(72), 0), shortcut=""):
        inker_mode.toggle_preview(tab)
    imgui.same_line()
    widgets.frame_counter(index, len(anim.frames))

    imgui.same_line()
    speed = widgets.combo(
        "##previewspeed",
        _speed_key(tab.preview_speed),
        _speed_options(),
        sp(64),
        tooltip=(
            "Playback speed. A preview option rather than a document one -- the "
            "frame durations in the timeline are what gets exported."
        ),
    )
    tab.preview_speed = float(speed)

    scope = widgets.combo(
        "##previewscope",
        tab.preview_scope,
        list(SCOPE_LABELS),
        sp(110),
        tooltip=(
            "How much of the document plays here: the whole clip, or just the "
            "tag the playhead is inside."
        ),
    )
    if scope != tab.preview_scope:
        tab.preview_scope = scope
        # The cycle count belongs to the span that was playing; keeping it
        # across a scope change would make a tag with "play twice" stop
        # immediately because the whole-clip pass already counted.
        tab.preview_cycles = 0


def _speed_options() -> list[tuple[str, str]]:
    return [(f"{value}", label) for value, label in SPEEDS]


def _speed_key(value: float) -> str:
    """The combo key nearest the stored multiplier.

    Nearest rather than exact: the value is a float on the tab and a key is a
    string, and a combo whose current value matched nothing would silently
    reset the speed to the first entry on the next frame.
    """
    best = min(SPEEDS, key=lambda item: abs(item[0] - float(value)))
    return f"{best[0]}"


def _image(ctx: Any, tab: Any) -> None:
    """The shown frame, fitted into whatever room is left, upright."""
    anim = tab.doc.anim
    region = imgui.get_content_region_avail()
    if region.x < 1.0 or region.y < 1.0:
        return
    index = max(0, min(int(tab.preview_index), len(anim.frames) - 1))
    texture = _throttled(ctx, tab, anim.frames[index].uid)
    origin = imgui.get_cursor_screen_pos()
    if texture is None:
        imgui.dummy((region.x, region.y))
        return
    width, height = texture.size
    scale = min(region.x / max(width, 1), region.y / max(height, 1))
    draw_w, draw_h = width * scale, height * scale
    x = origin.x + (region.x - draw_w) * 0.5
    y = origin.y + (region.y - draw_h) * 0.5
    draw_list = imgui.get_window_draw_list()
    draw_list.add_rect_filled(
        (x, y),
        (x + draw_w, y + draw_h),
        imgui.color_convert_float4_to_u32(theme.rgba(theme.PANEL)),
    )
    draw_list.add_image(
        widgets.texture_ref(texture), (x, y), (x + draw_w, y + draw_h)
    )
    imgui.dummy((region.x, region.y))


def _throttled(ctx: Any, tab: Any, frame_uid: int) -> Any:
    """``frame_texture``, but at most four times a second.

    The existing texture is handed back untouched in between, which is what
    makes the throttle cost nothing: it is the same object the next call would
    return, one flatten and one upload later.
    """
    if ctx.viewer is None:
        return None
    key = f"inker_preview:{tab.uid}"
    now = time.monotonic()
    existing = ctx.state.preview.get(f"inker_tex:{tab.uid}:frame{frame_uid}")
    if existing is not None and now - float(ctx.state.preview.get(key) or 0.0) < (
        REFRESH_SECONDS
    ):
        return existing
    ctx.state.preview[key] = now
    return inker_textures.frame_texture(ctx, tab, frame_uid)
