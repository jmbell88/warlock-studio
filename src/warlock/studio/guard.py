"""A pane that raises loses its pane, not the session.

``App.run`` wraps setup *and* the whole frame loop in one ``except``, so until
now any exception from any pane's draw ended the process. Everything downstream
of that is good -- teardown journals first, so ``_report_crash``'s "your work is
safe" sentence is computed rather than promised -- but the user still lost the
window, every open document's view state, and whatever they were in the middle
of. And the trigger could be one wrong byte in ``settings.json``:
``widgets.section`` reads ``panels_open`` every frame in most panes, and a
truthy non-dict there is an ``AttributeError`` sixty times a second.

**Dear ImGui can be unwound, which is the only reason this is possible.** An
exception between ``new_frame`` and ``render`` leaves unmatched ``Begin``,
``BeginChild``, ``PushStyleVar``, ``PushID`` and the rest, and the next
``new_frame`` then fails on a stack imgui believes it owns. 1.91.6 added
``ErrorRecoveryStoreState``/``ErrorRecoveryTryToRecoverState`` for exactly this,
and imgui-bundle binds both. Store a mark, and on the way out of a failure imgui
closes every window, table, tab bar, tree, group, popup and stack pushed past
it, naming each one in its debug log.

Three things that had to be measured rather than assumed:

* ``io.config_error_recovery_enable_assert`` defaults to **True**, and under
  imgui-bundle an ``IM_ASSERT`` surfaces as a Python ``RuntimeError``. Left on,
  the recovery *itself* raises and the guard turns a survivable pane failure
  into the crash it exists to prevent. :func:`configure` turns it off, and a
  scan test refuses a ``create_context`` that does not.
* The **clip-rect stack is not part of** ``ImGuiErrorRecoveryState``. It lives
  on the ``ImDrawList``, and children share the enclosing window's draw list --
  so a pane that raised between ``push_clip_rect`` and its pop makes the
  *enclosing* ``end_child`` pop our rectangle instead of the window's, and every
  later pane in that host clips one level wrong. ``plotter_canvas`` and
  ``packwright_preview`` both push one around the code most likely to raise.
  :func:`enter` records the depth and :func:`recover` pops back to it by hand.
* Python's own ``finally`` blocks run *first*, on the way to this handler, so
  ``widgets.section_blocks``' channel merge and ``menus``' menu-bar end have
  already happened. The guard only ever faces the bare ``begin``/``end`` pairs
  written as plain calls, which is why unwinding them is enough.

The breaker is deliberately **not** cleared per frame, unlike ``FRAME_FAILURES``
beside it and unlike the three censuses this module's ``begin_frame`` joins
(``layout.FRAME_PANES``, ``anchors.FRAME_ANCHORS``, ``probe.FRAME_CONTROLS``).
A pane that fails does so every frame; forgetting that between frames would draw
the broken pane, catch, recover and re-announce sixty times a second.
"""

from __future__ import annotations

import logging
import os
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from imgui_bundle import imgui

log = logging.getLogger(__name__)

#: Consecutive failed *attempts* before a pane stops being re-drawn. Attempts
#: rather than frames: panes are not drawn every frame (the idle skip, a mode
#: change, a slot whose ``applies`` is False), so a frame count would trip a
#: rarely-drawn pane after a single bad afternoon. Three is about 50 ms of a
#: live workspace -- long enough that a genuinely transient failure (a job row
#: that vanished between the cache tick and the draw) recovers with no flicker a
#: person can see, short enough that a deterministic one trips before they can
#: read the half-drawn pane behind it.
TRIP_AFTER = 3

#: How many times "Try again" is offered before the button is dropped. A user
#: pressing it a fourth time is not going to be rewarded on the fifth, and a
#: button that never works is worse than no button.
RETRY_LIMIT = 3

#: Failures this guard will not swallow under any circumstances. Recovering from
#: these and then drawing a placeholder is not a credible claim: the process is
#: out of memory, out of stack, or imgui is in a state its own recovery cannot
#: describe.
FATAL: tuple[type[BaseException], ...] = (MemoryError, RecursionError, SystemError)

#: Re-raise everything instead of catching it. ``widgets.FORCE_SECTIONS_OPEN``'s
#: idiom, and load-bearing rather than a convenience: ``scripts/exercise_mode``
#: exists to find controls that are dead or crashing, and a guard that quietly
#: replaced a crash with a tidy placeholder would make it report a clean pass on
#: a broken pane. ``tests/conftest.py`` sets this autouse for the same reason --
#: twelve thousand tests must keep failing loudly.
STRICT = os.environ.get("WARLOCK_UI_STRICT") == "1"


@dataclass(frozen=True)
class Failure:
    """One surface's draw that raised. For a harness to read, not for the UI."""

    key: str
    title: str
    kind: str
    message: str
    traceback: str
    tripped: bool


@dataclass(frozen=True)
class Mark:
    """Where a surface's draw began, so a failure can be unwound back to it."""

    state: Any
    clips: int | None
    window: str


@dataclass
class _Breaker:
    fails: int = 0
    tripped: bool = False
    announced: bool = False
    retries: int = 0


#: Every failure this frame, cleared beside the other three censuses. Read by
#: ``scripts/exercise_mode`` and ``scripts/screenshot_modes`` so a swallowed
#: exception is still a finding rather than a green picture of a placeholder.
FRAME_FAILURES: list[Failure] = []

#: Every failure this *session*. Bounded by construction rather than by a cap: a
#: pane fails at most ``TRIP_AFTER`` times before its breaker opens, and only a
#: deliberate "Try again" reopens it.
HISTORY: list[Failure] = []

_BREAKERS: dict[str, _Breaker] = {}
_CTX: Any = None
_DRAWING_PLACEHOLDER = False


def configure() -> None:
    """Make imgui's error recovery usable. Call once, after ``create_context``.

    The assert is the one that matters and it defaults on; see the module
    docstring. The tooltip goes with it -- a debug affordance that would
    otherwise appear over a real user's broken pane. The **debug log stays on**:
    imgui names each thing it closed ("Missing EndChild()", "Missing
    EndTable()") and that line is the cheapest diagnosis of a pane failure there
    will ever be.
    """
    io = imgui.get_io()
    io.config_error_recovery_enable_assert = False
    io.config_error_recovery_enable_tooltip = False
    io.config_error_recovery_enable_debug_log = True


def begin_frame(ctx: Any = None) -> None:
    """Forget last frame's failures, and remember who to tell about this one's.

    Joins ``layout.begin_frame``/``anchors.begin_frame``/``probe.begin_frame`` in
    ``_build_ui``, but has to run *before* ``menus.draw`` rather than beside
    them, because the menu bar draws first and is itself guarded.
    """
    global _CTX

    FRAME_FAILURES.clear()
    _CTX = ctx


def enter(key: str) -> Mark | None:
    """Record where this surface's draw begins. ``None`` if it cannot be.

    The context check comes first for ``anchors.mark``'s reason: imgui's null
    checks are asserts compiled out of the release build, so calling in with no
    context is an access violation rather than an exception, and
    ``get_current_context`` is the one entry point safe to ask.
    """
    if imgui.get_current_context() is None:
        return None
    try:
        state = imgui.internal.ErrorRecoveryState()
        imgui.internal.error_recovery_store_state(state)
        window = imgui.internal.get_current_window().name
    except Exception:  # noqa: BLE001 -- a guard that fails to arm must not raise
        log.exception("could not take a recovery mark for %s", key)
        return None
    return Mark(state=state, clips=_clip_depth(), window=window)


def _clip_depth() -> int | None:
    """How deep the current window's clip-rect stack is, or ``None``.

    Defensive because this is the one thing here reaching past the public API:
    ``_clip_rect_stack`` is an implementation detail of ``ImDrawList`` that
    imgui-bundle happens to expose. If a future build stops exposing it the guard
    must lose the clip repair, not stop working.
    """
    try:
        return len(imgui.get_window_draw_list()._clip_rect_stack)
    except Exception:  # noqa: BLE001
        return None


def recover(mark: Mark | None, key: str, title: str, exc: BaseException) -> None:
    """Unwind imgui back to ``mark`` and record the failure. Re-raises to escalate.

    Every ``raise`` below is a case where carrying on would be a lie, and they
    are the whole safety argument: the caller's ``except Exception`` already
    excludes ``KeyboardInterrupt`` and ``SystemExit``, and what escapes here
    lands in ``App.run``'s handler, which journals the user's work and then
    reports the crash exactly as it does today.
    """
    if STRICT:
        raise exc
    if isinstance(exc, FATAL):
        raise exc
    if mark is None:
        # Nothing safe to unwind to: either there was no imgui context or arming
        # the guard failed, and both mean the stacks are unknown.
        raise exc
    if _DRAWING_PLACEHOLDER:
        # The placeholder itself raised. Recursing here is the one way this
        # module could loop forever, so it does not.
        raise exc
    try:
        imgui.internal.error_recovery_try_to_recover_state(mark.state)
        _pop_clips(mark.clips)
    except Exception:  # noqa: BLE001
        # The recovery raised. The stacks are now in a state nothing can
        # describe and ``imgui.render`` would fail anyway, so hand the original
        # exception on -- a bare ``raise`` here would re-raise the wrong one.
        # ``from None`` because the recovery's own failure is already in the log
        # and chaining it would put it at the *top* of the crash traceback, above
        # the pane failure that actually caused all this.
        log.exception("recovering %s failed; escalating the original failure", key)
        raise exc from None
    landed = _window_name()
    if landed != mark.window:
        # Recovery stopped somewhere other than where it started. This check is
        # cheap, and it is what makes the whole scheme safe rather than hopeful:
        # without it a mis-landed unwind draws a placeholder into somebody
        # else's window and corrupts the rest of the frame.
        log.error("recovering %s landed in %r, expected %r", key, landed, mark.window)
        raise exc
    _record(key, title, exc)


def _pop_clips(depth: int | None) -> None:
    """Pop back to the recorded clip depth. imgui does not do this one."""
    if depth is None:
        return
    draw_list = imgui.get_window_draw_list()
    # Bounded rather than ``while``: this repairs a leak, and a leak deep enough
    # to need more than sixteen pops is a bug this loop should not hide.
    for _ in range(16):
        if len(draw_list._clip_rect_stack) <= depth:
            return
        draw_list.pop_clip_rect()


def _window_name() -> str:
    try:
        return imgui.internal.get_current_window().name
    except Exception:  # noqa: BLE001
        return ""


def _record(key: str, title: str, exc: BaseException) -> None:
    breaker = _BREAKERS.setdefault(key, _Breaker())
    breaker.fails += 1
    failure = Failure(
        key=key,
        title=title,
        kind=type(exc).__name__,
        message=str(exc),
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        tripped=breaker.fails >= TRIP_AFTER,
    )
    FRAME_FAILURES.append(failure)
    HISTORY.append(failure)
    log.error("%s stopped drawing", title or key, exc_info=exc)
    if failure.tripped:
        breaker.tripped = True
    if breaker.tripped and not breaker.announced:
        breaker.announced = True
        _announce(title or key)


def _announce(title: str) -> None:
    """Say it once on the way past, and leave a line that outlives the toast.

    Two surfaces because they answer different questions. The toast is the
    interruption -- it expires in eight seconds and carries the "Open the log"
    action ``_toast_action`` already routes. ``note_error`` is the record: the
    doctor banner draws it, it dedupes on its own text, and it is cleared only by
    Dismiss, which is right, because the panel stays broken for the session.

    ``announced`` above is what makes "once" true. ``toast_once`` dedupes only
    against toasts still on screen and an error toast lives eight seconds -- so
    on its own it would speak again every eight seconds until the app closed.
    """
    ctx = _CTX
    if ctx is None:
        return
    try:
        ctx.toast(f"{title} stopped drawing. The rest of the app still works.", "error", "log")
        ctx.state.note_error(f"{title} stopped drawing. The details are in warlock.log.")
    except Exception:  # noqa: BLE001 -- an announcement must never be the failure
        log.exception("could not announce that %s stopped drawing", title)


def ok(key: str) -> None:
    """A clean draw. Forgets a failure that did not go on to trip the breaker."""
    breaker = _BREAKERS.get(key)
    if breaker is not None and not breaker.tripped:
        breaker.fails = 0


def tripped(key: str) -> bool:
    breaker = _BREAKERS.get(key)
    return breaker is not None and breaker.tripped


def retries(key: str) -> int:
    breaker = _BREAKERS.get(key)
    return breaker.retries if breaker is not None else 0


def reset(key: str | None = None) -> None:
    """Close a breaker and let the surface draw again, or reset every one."""
    if key is None:
        _BREAKERS.clear()
        return
    breaker = _BREAKERS.get(key)
    if breaker is None:
        return
    breaker.retries += 1
    breaker.fails = 0
    breaker.tripped = False
    breaker.announced = False


def placeholder(key: str, title: str) -> None:
    """What the user sees instead of the surface. Drawn where the surface was.

    No rect arithmetic and no promises. After a successful recovery the current
    window *is* the pane again, so this draws in it; and the hint says the rest
    of the app works rather than that nothing was lost, because
    ``_report_crash``'s rule applies here too -- a reassurance that cannot be
    computed is not one worth giving.

    "Try again" only closes the breaker. A "reset this panel" that discarded a
    document's in-memory edits would be data loss wearing recovery's name, and no
    pane has a reset hook for it to call.
    """
    global _DRAWING_PLACEHOLDER

    from . import icons, widgets

    offer_retry = retries(key) < RETRY_LIMIT
    hint = (
        "The rest of the app is still working. The details are in warlock.log."
        if offer_retry
        else "It has failed every time. Restart Warlock; the details are in warlock.log."
    )
    _DRAWING_PLACEHOLDER = True
    try:
        widgets.empty_state(
            icons.TRIANGLE_ALERT,
            f"{title or key} stopped drawing",
            hint,
            action=("Try again", lambda: reset(key)) if offer_retry else None,
        )
    except Exception:  # noqa: BLE001 -- reached only when the placeholder itself fails
        log.exception("could not draw the placeholder for %s", key)
    finally:
        _DRAWING_PLACEHOLDER = False


def failures() -> tuple[Failure, ...]:
    """This session's failures, for a harness to assert on."""
    return tuple(HISTORY)


@contextmanager
def surface(key: str, title: str = "", *, draw_placeholder: bool = True) -> Iterator[bool]:
    """Guard a region that is not a :func:`layout.pane`.

    The menu bar, the rail, the status bar, the whole content region and each
    overlay. ``yield``s False when the breaker is open, so a caller can skip its
    body the way ``pane`` does.

    ``draw_placeholder=False`` for the overlays, and it is not a preference.
    They are drawn at host scope *after* ``imgui.end()``, so there is no window
    to put a placeholder in -- ``empty_state`` would land in imgui's implicit
    debug window, which is worse than saying nothing. A failed overlay is
    announced by the toast and the doctor banner instead, and those are drawn by
    the overlays themselves, which is exactly why the toast call is one of the
    surfaces guarded separately rather than as a group.
    """
    mark = enter(key)
    live = not tripped(key)
    failed = False
    try:
        yield live
    except Exception as exc:  # noqa: BLE001 -- the whole point of this module
        recover(mark, key, title or key, exc)
        failed = True
    else:
        if live:
            ok(key)
    if draw_placeholder and (failed or tripped(key)):
        placeholder(key, title or key)


def run(
    key: str,
    fn: Callable[..., Any],
    /,
    *args: Any,
    title: str = "",
    on_failure: Callable[[], None] | None = None,
    draw_placeholder: bool = True,
    **kwargs: Any,
) -> bool:
    """Call ``fn`` under a guard. -> did it draw?

    ``on_failure`` is for a surface that latches something the placeholder cannot
    un-latch: the tour's scrim is on the foreground draw list and cannot be
    un-drawn, and a confirm queue that stops drawing still reports
    ``modal_open``, which would leave an invisible modal owning the keyboard.
    """
    drew = False
    with surface(key, title, draw_placeholder=draw_placeholder) as live:
        if live:
            try:
                fn(*args, **kwargs)
            except Exception:
                if on_failure is not None:
                    try:
                        on_failure()
                    except Exception:  # noqa: BLE001
                        log.exception("the failure handler for %s also failed", key)
                raise
            drew = True
    return drew
