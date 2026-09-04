"""Inker's playhead: the clip, the preview and the two tickers that move them.

Two playheads, deliberately. tick_playback moves the *document's* frame and is
what a save has to wait for; tick_preview moves a strip's own index and must
never touch playing, saving or set_current_frame -- which is why it is a clone
rather than a share, and why frame_durations is the one thing they do share.

Lifted out of ``studio/inker_mode`` on 2026-09-04 (T7 of the 2026-09-02
review), after every behavioural finding that touches it was closed, so the
move is code motion over tested behaviour rather than a rewrite.

``inker_mode`` is imported as a *module* and never ``from``-imported: every
attribute is resolved at call time, so this file and its parent may be
imported in either order. The parent serves these names back through a PEP
562 ``__getattr__``, which is what keeps ``inker_mode.export_png`` and the
rest working for every caller and every test.
"""

from __future__ import annotations

from typing import Any

from . import inker_mode
from .inker import animation
from .inker_state import InkerDoc


def toggle_play(ctx: Any, tab: InkerDoc | None = None) -> None:
    """Start or stop playback. Refused while a save is encoding."""
    tab = tab or inker_mode.active(ctx)
    if tab is None or tab.doc.anim is None or tab.saving:
        return
    if tab.playing:
        stop_play(tab)
        return
    state = ctx.state.inker
    if state is not None and (state.drag_kind or state.gesture_pts):
        # ``step_frame``'s guard, and for its reason: playback advances frames,
        # which rebuilds the layer stack, and an open paint drag holds a
        # ``StrokeState`` addressed into the stack it began on -- the next
        # ``stroke_to`` raises out of ``by_uid``. Stopping has no such problem
        # and is above this line, because a way out must always be available.
        return
    # The float is committed *before* playback rather than when it ends: while
    # playing, the canvas draws cached frame flattens, and a floating buffer is
    # in no layer and therefore in no flatten -- it would simply vanish for the
    # duration and reappear at the end.
    tab.doc.commit_floating()
    tab.playing = True
    tab.play_index = tab.doc.anim.current
    tab.play_accum_ms = 0.0
    # Every play starts on the outward leg. A ping-pong stopped halfway back and
    # resumed would otherwise carry on inwards from wherever it was, which reads
    # as the clip playing backwards for no reason anyone watching can see.
    tab.play_forward = True
    # Every play starts the repeat count over. A tag set to play three times
    # and stopped halfway must play three times again when it is started, not
    # remember that it already finished once -- the same argument
    # ``play_forward`` above makes about a ping-pong's leg.
    tab.play_cycles = 0


def stop_play(tab: InkerDoc) -> None:
    """Stop, and leave the playhead where the eye last saw it."""
    if not tab.playing:
        return
    tab.playing = False
    if not tab.saving:
        # ``set_current_frame`` rebuilds the layer stack, which is exactly the
        # structure an in-flight encode is walking -- the ``_MUTATING_CTRL``
        # rule, applied to the one mutation stopping playback itself makes.
        # Unreachable while the saves below stop playback before capturing and
        # ``toggle_play`` refuses to start during one; kept as the backstop,
        # at the cost of the playhead resting where play began.
        tab.doc.set_current_frame(tab.play_index)
    tab.play_accum_ms = 0.0


def frame_durations(tab: InkerDoc, anim: Any) -> list[int]:
    """The durations a playhead steps by, Constant Frame Rate included.

    **Constant Frame Rate** (6.7) is Aseprite's own playback switch: play every
    frame at one rate rather than at the durations the document stores. It is a
    *preview* setting and it does not touch the frames -- what an animator is
    asking is "what does this look like at 12 fps", not "make every frame
    83 ms", and answering the second would be an undoable edit to every frame.

    One function because there are two playheads. ``tick_preview`` is a clone of
    ``tick_playback`` rather than a share, deliberately -- it must never touch
    ``playing``, ``saving`` or ``set_current_frame`` -- but the clone had copied
    the plain duration list and not the switch above it, so turning Constant
    Frame Rate on left the timeline playing at 12 fps and the preview pane
    playing at the stored durations. Two playheads disagreeing about one clip is
    the drift a clone invites, so the one part they genuinely share lives here.
    """
    durations = [frame.duration_ms for frame in anim.frames]
    rate = getattr(tab, "constant_rate", 0)
    if rate:
        held = max(1, round(1000.0 / float(rate)))
        return [held] * len(durations)
    return durations


def tick_playback(tab: InkerDoc, dt_ms: float) -> None:
    """One frame's worth of time.

    Deliberately does *not* call ``set_current_frame``: that re-materialises the
    stack and recomposites the whole canvas, sixty times a second, to show a
    picture the frame cache already has. The playhead on the document stays put
    and the canvas draws ``play_index``'s cached flatten instead; stopping is
    the one moment the document catches up.
    """
    anim = tab.doc.anim
    if not tab.playing or anim is None:
        return
    index, accum, playing, forward, cycles = animation.advance(
        frame_durations(tab, anim),
        tab.play_index,
        tab.play_accum_ms,
        min(float(dt_ms), inker_mode.MAX_TICK_MS),
        anim.loop_range(tab.play_index),
        direction=anim.play_direction(tab.play_index),
        forward=tab.play_forward,
        repeat=anim.play_repeat(tab.play_index),
        cycles=tab.play_cycles,
    )
    tab.play_index, tab.play_accum_ms, tab.play_forward = index, accum, forward
    tab.play_cycles = cycles
    if not playing:
        stop_play(tab)


# --- the preview pane's second playhead --------------------------------------

#: Bounds on the preview's speed multiplier. A ceiling because past ×4 a clip
#: is a flicker rather than a preview, and a floor because a multiplier that
#: reaches zero is a stopped clip pretending to play.
MIN_PREVIEW_SPEED = 0.25
MAX_PREVIEW_SPEED = 4.0

def toggle_preview(tab: InkerDoc) -> None:
    """Start or stop the preview. Refused for nothing at all.

    Deliberately not gated on ``busy``: the preview neither edits the document
    nor moves its playhead, so there is nothing for a save or for canvas
    playback to be protected from -- and being able to watch the clip while
    drawing on it is the whole feature.
    """
    if tab.doc.anim is None:
        return
    if tab.preview_playing:
        tab.preview_playing = False
        return
    tab.preview_playing = True
    tab.preview_accum_ms = 0.0
    tab.preview_forward = True
    tab.preview_cycles = 0


def tick_preview(tab: InkerDoc, dt_ms: float) -> None:
    """One frame's worth of time for the *preview*'s playhead.

    A clone of :func:`tick_playback` rather than a share, and the difference is
    the point: this one **never touches ``tab.playing`` or ``tab.saving``** and
    never calls ``set_current_frame``. It reads ``anim`` and writes four fields
    on the tab, so a preview running while the user paints needs no gating
    change anywhere -- the canvas draws the document, the preview draws
    ``frame_flat``, which is the same read onion skinning already makes and is
    safe even during a save (``sheetout.snapshot``'s argument).

    The speed multiplier scales time **after** the stall clamp, so a two-second
    hitch is still treated as a stall at ×4 rather than as eight seconds of
    animation.
    """
    anim = tab.doc.anim
    if not tab.preview_playing or anim is None or not anim.frames:
        return
    last = len(anim.frames) - 1
    index = max(0, min(int(tab.preview_index), last))
    if tab.preview_scope == "tag":
        span = anim.loop_range(index)
        direction = anim.play_direction(index)
        repeat = anim.play_repeat(index)
    else:
        # The whole clip, looping, whatever tags happen to cover it. A preview
        # scoped to the clip that stopped at a non-looping tag's end would be
        # answering a question the scope switch just said no to.
        span, direction, repeat = (0, last, True), "forward", 0
    speed = max(MIN_PREVIEW_SPEED, min(float(tab.preview_speed), MAX_PREVIEW_SPEED))
    index, accum, playing, forward, cycles = animation.advance(
        frame_durations(tab, anim),
        index,
        tab.preview_accum_ms,
        min(float(dt_ms), inker_mode.MAX_TICK_MS) * speed,
        span,
        direction=direction,
        forward=tab.preview_forward,
        repeat=repeat,
        cycles=tab.preview_cycles,
    )
    tab.preview_index, tab.preview_accum_ms = index, accum
    tab.preview_forward, tab.preview_cycles = forward, cycles
    if not playing:
        tab.preview_playing = False


def step_frame(ctx: Any, delta: int, tab: InkerDoc | None = None) -> None:
    tab = tab or inker_mode.active(ctx)
    if tab is None or tab.doc.anim is None or tab.busy:
        return
    state = ctx.state.inker
    if state is not None and (state.drag_kind or state.gesture_pts):
        # ``set_current_frame`` rebuilds the layer stack, and an open paint
        # drag holds a ``StrokeState`` addressed into the stack it began on --
        # the next ``stroke_to``/``end_stroke`` raises out of ``by_uid``. A
        # multi-click gesture's vertices likewise belong to the frame they
        # were placed on. Refused like the ``_MUTATING_CTRL`` set: the gesture
        # finishes first.
        return
    anim = tab.doc.anim
    tab.doc.set_current_frame((anim.current + delta) % len(anim.frames))


def animate(ctx: Any, tab: InkerDoc | None = None) -> None:
    """The entry point: turn a still document into a two-frame animation."""
    tab = tab or inker_mode.active(ctx)
    if tab is None or tab.busy:
        return
    tab.doc.add_frame()
