"""The guided tour, drawn over whatever is on screen.

Point-and-wait: the tour rings a real control, says one thing about it, and
waits for the reader to use it. **It never clicks anything for them**, which is
what keeps a tour from mutating a document, a setting or the job queue -- and it
is also why nothing here reaches ``App._modal_open``. The app underneath has to
stay live.

Two consequences shape the whole file.

The scrim cannot be a mask. ``ImDrawData`` arrives as a flat list of command
lists with nothing on them saying which window each came from -- the same fact
``vibrancy`` is built around -- so there is no "everything except the
highlight" to dim. The dimming fills the viewport *around* holes instead, which
is exact where a mask would be a guess, and leaves the control genuinely
visible rather than approximately so.

There are two holes, and the second is the one nobody predicts: the scrim is on
the foreground draw list, which imgui paints above every window -- so without a
hole for it, the tour dims its own card.

And the scrim takes no input, for a better reason than a flag. It is draw-list
geometry rather than a window, so there is nothing there to hit: a click aimed
at the ringed control reaches the control, and the point-and-wait promise holds
without anything having to be careful about it. The card *is* a window and does
take input, because its buttons are the only part the reader is meant to press
here.

Positions are read from ``anchors``, never computed. The rail alone recomputes
every item's box each frame across a three-rung compression ladder, so a ring
that did its own arithmetic would drift the moment the window got short.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import anchors, controls, icons, theme, tokens, widgets
from ..manual import render as manual_render
from ..tokens import sp
from ..tour import find as find_tour

#: The card's width, in design px. Wide enough for two sentences at a readable
#: measure and narrow enough to sit beside a ringed control rather than over it.
CARD_W = 380

#: How far the scrim goes. Not opaque: the reader is meant to keep their
#: bearings, and a tour that blacked out the app would be a modal wearing a
#: different name.
VEIL_ALPHA = 0.55

#: Breathing room between the ringed control and the hole's edge.
HOLE_PAD = 6

_was_open = [False]

#: Last frame's card rect, so this frame's scrim can leave a hole for it.
#:
#: One frame stale, and the same staleness ``widgets.window_shadow`` documents
#: and accepts: the card is auto-height, so its rectangle does not exist until
#: it has drawn. It only moves when a step changes side, and on that one frame
#: the scrim's hole is where the card was rather than where it is -- which
#: costs a frame of dimmed text and never a frame of hidden text, because the
#: card is drawn after the scrim either way.
_card_rect: list[tuple[float, float, float, float] | None] = [None]


# -- what a step is waiting for -------------------------------------------
#
# One snapshot per frame, built here and read by name, so ``studio/tour`` stays
# free of imgui and ``service`` and its rules stay assertable headlessly.


def _inker_doc(ctx: Any) -> Any:
    inker = getattr(ctx.state, "inker", None)
    return getattr(inker, "active", None) if inker is not None else None


def satisfied(ctx: Any, name: str, arg: str | None) -> bool:
    """Whether the condition ``name`` holds right now.

    Every name in ``tour.steps.CONDITIONS`` must be answered here, and nothing
    else may be; ``tests/tour/test_tour_conditions.py`` asserts both directions.
    An unknown name reads as "never satisfied", which on a point-and-wait step
    is indistinguishable from the app being broken -- so it is a test failure
    rather than something to discover at runtime.
    """
    if name == "manual":
        return False
    if name == "mode_is":
        return ctx.state.mode == arg
    if name == "doc_open":
        if arg == "inker":
            return _inker_doc(ctx) is not None
        holder = getattr(ctx.state, str(arg), None)
        return bool(getattr(holder, "docs", None))
    if name == "tool_is":
        inker = getattr(ctx.state, "inker", None)
        return getattr(inker, "tool", None) == arg
    if name == "layers_at_least":
        doc = _inker_doc(ctx)
        if doc is None:
            return False
        try:
            return len(doc.doc.stack) >= int(arg or 0)
        except (AttributeError, TypeError, ValueError):
            return False
    if name == "animated":
        doc = _inker_doc(ctx)
        return doc is not None and getattr(doc.doc, "anim", None) is not None
    return False


#: The names :func:`satisfied` answers. Written out rather than derived from the
#: function, so the agreement test compares two independently authored lists
#: instead of one list against itself.
HANDLED: frozenset[str] = frozenset(
    {"manual", "mode_is", "doc_open", "tool_is", "layers_at_least", "animated"}
)


# -- running a tour --------------------------------------------------------


def start(ctx: Any, key: str) -> None:
    """Begin a tour. Unknown keys are ignored rather than raising."""

    if find_tour(key) is None:
        return
    ctx.state.tour.start(key)


def stop(ctx: Any) -> None:
    ctx.state.tour.stop()
    _card_rect[0] = None


def advance(ctx: Any, delta: int = 1) -> None:
    """Step forward or back, completing the tour when it runs off the end."""

    state = ctx.state.tour
    tour = find_tour(state.key)
    if tour is None:
        state.stop()
        return
    index = state.index + delta
    if index >= len(tour):
        state.complete()
        _remember(ctx)
        return
    state.index = max(0, index)
    state.satisfied = False


def is_open(ctx: Any) -> bool:
    return bool(getattr(ctx.state, "tour", None) and ctx.state.tour.running)


def _remember(ctx: Any) -> None:
    """Persist which tours have been finished, so Home can stop offering them."""

    settings = getattr(ctx, "settings", None)
    if settings is None:
        return
    try:
        settings.set("tours_finished", list(ctx.state.tour.finished))
    except Exception as exc:  # pragma: no cover - a settings write is best effort
        ctx.toast(f"Could not remember the finished tour: {exc}", "warn")


def restore(ctx: Any) -> None:
    """Read the finished-tour list back at startup."""

    settings = getattr(ctx, "settings", None)
    if settings is None:
        return
    try:
        saved = settings.get("tours_finished", []) or []
    except Exception:  # pragma: no cover - a missing key is not an error
        return
    ctx.state.tour.finished = tuple(str(key) for key in saved)


# -- drawing ---------------------------------------------------------------


def _hole(ctx: Any, step: Any) -> tuple[float, float, float, float] | None:
    """This frame's rect for the step's anchor, padded, or ``None``.

    ``None`` is an ordinary state rather than a failure: a control inside a
    collapsed section or behind another tab simply did not draw, and the card
    still has something to say about it.
    """
    if not step.anchor:
        return None
    found = anchors.rect(step.anchor)
    if found is None:
        return None
    pad = sp(HOLE_PAD)
    x, y, w, h = found
    return (x - pad, y - pad, w + pad * 2, h + pad * 2)


def _veil(viewport: Any, holes: list[tuple[float, float, float, float]]) -> None:
    """Dim the viewport except where the holes are.

    A horizontal-band decomposition rather than four rectangles around one
    hole, because there are two: the control being pointed at, and the card
    doing the pointing. The card needs one for a reason that is not obvious
    until you see it -- the scrim is on the *foreground* draw list, which imgui
    paints above every window including the card, so without a hole the tour
    dims its own text.

    Bands are cut at every hole edge and each band is filled only where no hole
    covers it, so no pixel is painted twice. That matters: two overlapping fills
    at 0.55 would leave a visibly darker patch wherever they crossed, and
    "darker where two rectangles happen to meet" is the kind of artefact that
    reads as a rendering bug rather than as a design.
    """
    draw = imgui.get_foreground_draw_list()
    colour = imgui.get_color_u32(theme.rgba(theme.TOUR_VEIL, VEIL_ALPHA))
    # The whole viewport, not the work area, which is what
    # ``App._transition_overlay`` covers for the same reason: the work area
    # excludes the setup banner, and a scrim that dims the app apart from one
    # bright strip along the top reads as the scrim having failed.
    x0 = viewport.pos.x
    y0 = viewport.pos.y
    x1 = x0 + viewport.size.x
    y1 = y0 + viewport.size.y
    edges = sorted({y0, y1} | {v for _x, y, _w, h in holes for v in (y, y + h) if y0 < v < y1})
    for top, bottom in zip(edges, edges[1:], strict=False):
        spans = sorted(
            (max(x0, hx), min(x1, hx + hw))
            for hx, hy, hw, hh in holes
            if hy < bottom and hy + hh > top and hx + hw > x0 and hx < x1
        )
        cursor = x0
        for left, right in spans:
            if left > cursor:
                draw.add_rect_filled((cursor, top), (left, bottom), colour)
            cursor = max(cursor, right)
        if cursor < x1:
            draw.add_rect_filled((cursor, top), (x1, bottom), colour)


def _ring(hole: tuple[float, float, float, float]) -> None:
    """The outline around the hole, on the same list as the scrim."""

    hx, hy, hw, hh = hole
    imgui.get_foreground_draw_list().add_rect(
        (hx, hy),
        (hx + hw, hy + hh),
        imgui.get_color_u32(theme.rgba(theme.TOUR_RING, 0.95)),
        sp(6),
        thickness=sp(2.0),
    )


def _card_pos(viewport: Any, hole: tuple[float, float, float, float] | None) -> tuple[float, float]:
    """Bottom-right by default, and out of the hole's way when there is one.

    Only the horizontal side is swapped. A card that also chased the hole
    vertically would jump the length of the window between two steps pointing
    at the top and bottom of the same pane, and a reader tracking a moving card
    is not reading it.
    """
    margin = sp(tokens.SP_4)
    y = viewport.work_pos.y + viewport.work_size.y - margin
    right = viewport.work_pos.x + viewport.work_size.x - margin
    if hole is not None:
        hx, _hy, hw, _hh = hole
        centre = viewport.work_pos.x + viewport.work_size.x * 0.5
        if hx + hw * 0.5 > centre:
            return (viewport.work_pos.x + margin + sp(CARD_W), y)
    return (right, y)


def draw(ctx: Any) -> None:
    """The whole tour: scrim, ring and card. Called once from ``App._overlays``."""

    state = getattr(ctx.state, "tour", None)
    if state is None or not state.running:
        _was_open[0] = False
        return
    tour = find_tour(state.key)
    step = tour.step(state.index) if tour is not None else None
    if tour is None or step is None:
        state.stop()
        _was_open[0] = False
        return

    appearing = not _was_open[0]
    _was_open[0] = True
    state.satisfied = satisfied(ctx, step.done.name, step.done.arg)

    viewport = imgui.get_main_viewport()
    hole = _hole(ctx, step)
    holes = [one for one in (hole, _card_rect[0]) if one is not None]
    _veil(viewport, holes)
    if hole is not None:
        _ring(hole)
    _card(ctx, viewport, tour, step, hole, appearing)


def _card(
    ctx: Any,
    viewport: Any,
    tour: Any,
    step: Any,
    hole: tuple[float, float, float, float] | None,
    appearing: bool,
) -> None:
    state = ctx.state.tour
    alpha, rise = widgets.popover_enter("tour", appearing)
    x, y = _card_pos(viewport, hole)
    imgui.set_next_window_pos((x, y + rise), imgui.Cond_.always.value, (1.0, 1.0))
    imgui.set_next_window_size((sp(CARD_W), 0))
    frosted = widgets.frosted()
    if frosted:
        imgui.set_next_window_bg_alpha(0.0)
    imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
    radius = widgets.push_surface_rounding()
    opened = imgui.begin(
        "##tour-card",
        None,
        imgui.WindowFlags_.no_title_bar.value
        | imgui.WindowFlags_.no_move.value
        | imgui.WindowFlags_.no_resize.value
        | imgui.WindowFlags_.no_collapse.value
        | imgui.WindowFlags_.always_auto_resize.value
        | imgui.WindowFlags_.no_saved_settings.value,
    )[0]
    widgets.pop_surface_rounding()
    if opened:
        widgets.window_shadow("overlay", radius=radius)
        if frosted:
            widgets.window_backdrop(radius=radius)
        _card_body(ctx, tour, step, state)
        pos = imgui.get_window_pos()
        size = imgui.get_window_size()
        # Padded, so the shadow under the card is not the one thing the scrim
        # still darkens -- a bright card with a dimmed halo reads as a seam.
        pad = sp(HOLE_PAD)
        _card_rect[0] = (pos.x - pad, pos.y - pad, size.x + pad * 2, size.y + pad * 2)
    imgui.end()
    imgui.pop_style_var()


def _card_body(ctx: Any, tour: Any, step: Any, state: Any) -> None:
    widgets.secondary(f"{tour.title} - {state.index + 1} of {len(tour)}")
    imgui.same_line()
    close_w = imgui.get_frame_height()
    imgui.set_cursor_pos_x(
        max(imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x - close_w, 0.0)
    )
    if widgets.icon_button(f"{icons.CIRCLE_X}##tour-close", "End the tour (Esc)", borderless=True):
        stop(ctx)
        return

    widgets.pane_title(step.title)
    for paragraph in step.body.split("\n\n"):
        imgui.text_wrapped(paragraph)
        imgui.dummy((0, sp(tokens.SP_1)))

    if step.done.name != "manual":
        widgets.secondary("Waiting for you." if not state.satisfied else "Done.")

    imgui.dummy((0, sp(tokens.SP_1)))
    if state.index > 0 and controls.button("Back##tour", role=controls.ButtonRole.GHOST):
        advance(ctx, -1)
        return
    if state.index > 0:
        imgui.same_line()
    label = "Next" if state.index + 1 < len(tour) else "Finish"
    # A satisfied condition offers the step's exit rather than taking it: the
    # reader has just done the thing and the card would otherwise vanish
    # mid-sentence, which reads as the app having lost their place.
    if widgets.primary_button(f"{label}##tour"):
        advance(ctx)
        return
    if step.chapter is not None:
        imgui.same_line()
        if controls.button("Read more##tour", role=controls.ButtonRole.GHOST):
            manual_render.open_at(ctx, step.chapter)
