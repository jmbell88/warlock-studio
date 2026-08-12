"""Per-frame animation values for the widget kit.

One primitive: :func:`value` eases a named float towards a target with a
frame-rate-independent exponential approach, using ``io.delta_time`` (which the
frame loop already sets). State lives here in a module dict keyed by explicit
string ids -- not on ``Ctx``, and not by ``imgui.get_id`` (a widget that moves
between windows would silently restart its animation). Nothing blocks, nothing
threads: this is arithmetic the frame thread does on the way past.

A key's first sighting snaps to its target rather than animating from zero, so
a newly built pane appears settled instead of assembling itself; animation
happens when a *target changes* on a key that is already live.

**Reduce-motion is honoured here and nowhere else.** Every animated surface in
the app -- the segment pill, hover, the mode crossfade, a popover's rise, the
sidebar width -- reads its number from this module, so collapsing every
duration to zero in one place is what turns all of them off at once. A consumer
that implemented its own easing would be a second switch to remember, which is
why the hand-rolled toast rise and ``drop_flash`` route through :func:`ease`
rather than through their own arithmetic.
"""

from __future__ import annotations

import math
from typing import Any

from imgui_bundle import imgui

from . import tokens

_STATE: dict[str, float] = {}
# Per key: the target it was last asked for, and the imgui frame that asked.
# Both halves are what :func:`animating` needs. The *frame* is the half that is
# easy to leave out and wrong to: a key whose widget stopped being drawn
# mid-move keeps a target it will never reach, and without the frame stamp the
# idle clamp would then hold the app at 60 fps forever over a card that is no
# longer on screen.
_TARGET: dict[str, float] = {}
_FRAME: dict[str, int] = {}

# Whether motion is suppressed app-wide (accessibility: vestibular
# sensitivity). Module state, exactly as ``tokens.SCALE`` and
# ``layout.SIDEBAR_W`` are, and for the same reason: dozens of call sites read
# this module and threading a flag through all of them would put the setting in
# dozens of places instead of one.
REDUCED = False


def set_reduced(value: bool) -> bool:
    """Turn all motion off (or back on). -> what is now in force."""
    global REDUCED
    REDUCED = bool(value)
    return REDUCED


def _clock() -> tuple[float, int] | None:
    """``(this frame's delta, this frame's number)``, or ``None`` off-context.

    ``fonts``' rule applied to the other half of the widget kit: a helper that
    animates has to degrade to a no-op where there is no imgui context, because
    a headless test that drives one widget through a stand-in imgui still
    reaches the real module in here. ``None`` means "snap", which is what a
    non-animating caller wants anyway.
    """
    try:
        return max(imgui.get_io().delta_time, 1e-4), imgui.get_frame_count()
    except Exception:
        return None


def value(key: str, target: float, *, duration: float = tokens.DUR_BASE) -> float:
    """The animated value for ``key`` this frame, moving towards ``target``."""
    clock = _clock()
    if REDUCED or clock is None:
        duration = 0.0
    current = _STATE.get(key)
    _TARGET[key] = target
    _FRAME[key] = clock[1] if clock is not None else -1
    if current is None or duration <= 0.0:
        _STATE[key] = target
        return target
    dt = clock[0]
    # duration/3: an exponential approach covers ~95% of the distance in three
    # time constants, so "duration" reads as "how long the move visibly takes".
    current += (target - current) * (1.0 - math.exp(-dt / (duration / 3.0)))
    if abs(target - current) < 1e-3:
        current = target
    _STATE[key] = current
    return current


def peek(key: str, default: float = 0.0) -> float:
    """This frame's value for ``key`` without advancing it or claiming it.

    For the widgets that have to *read* an animated value before they know what
    to set it to -- a button's hover colour is needed to draw the button, and
    whether it is hovered is only known once it has been drawn. Calling
    :func:`value` twice in one frame instead (once to read, once to set) steps
    the easing twice and, worse, steps it towards the wrong target first: the
    card's lift did exactly that, decaying towards 0 on the way past and then
    climbing back towards 1, so a hover took visibly longer to arrive than to
    leave for no reason anybody chose.

    Deliberately does *not* stamp ``_FRAME``: a peek is not a claim that
    anything is still moving, and :func:`animating` answers for the widgets
    that set targets.
    """
    return _STATE.get(key, default)


def animating() -> bool:
    """Whether anything drawn in the last frame is still moving (B11).

    The frame loop's idle clamp asks this: an active animation is a reason the
    screen can change with no input, which is exactly what the clamp's other
    conditions enumerate. Only keys touched by the *last* frame count, so a
    widget that left the screen mid-move cannot hold the app awake.
    """
    clock = _clock()
    if REDUCED or clock is None:
        return False
    frame = clock[1]
    # Distance *or* velocity: a spring's overshoot passes through zero distance
    # at full speed -- the exact moment its damping exists to produce -- so
    # "settled" is both under the floor and "still moving" is either over it.
    # The floors are :func:`spring`'s own settle floors, one judgement shared,
    # or the idle clamp sleeps through the half of the move it was woken for.
    return any(
        _FRAME.get(key) == frame
        and (
            abs(_STATE.get(key, target) - target) >= _SETTLE_DISTANCE
            or abs(_VELOCITY.get(key, 0.0)) >= _SETTLE_VELOCITY
        )
        for key, target in _TARGET.items()
    )


# -- easing ------------------------------------------------------------------
#
# The exponential approach above is the right primitive for a value chasing a
# target that can change under it (hover, a sliding pill). These are for the
# other shape: a move with a known start, a known end and a known length, where
# what is wanted is the *curve* -- a popover's rise, the mode crossfade, the
# splash fading out. Cubic rather than anything fancier because the difference
# above cubic is not visible at 200 ms, and every one of these is under 300.


def ease_out_cubic(t: float) -> float:
    """Fast, then settling. The curve for something arriving."""
    t = min(max(t, 0.0), 1.0)
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    """Symmetric. The curve for something moving from one place to another."""
    t = min(max(t, 0.0), 1.0)
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def ease(key: str, duration: float, *, curve: Any = ease_out_cubic) -> float:
    """A 0->1 ramp on its own clock, eased. 1.0 once it has finished.

    The counterpart to :func:`value` for a one-shot: the caller says when it
    started (by :func:`restart`\\ ing the key) and reads the progress. Under
    reduce-motion it is 1.0 on the first frame, which is what makes "no
    animation" mean *arrived* rather than *never appears*.
    """
    clock = _clock()
    if REDUCED or duration <= 0.0 or clock is None:
        _STATE[key] = duration
        _TARGET[key] = duration
        _FRAME[key] = clock[1] if clock is not None else -1
        return 1.0
    elapsed = _STATE.get(key)
    if elapsed is None:
        elapsed = duration
    elapsed = min(elapsed + clock[0], duration)
    _STATE[key] = elapsed
    _TARGET[key] = duration
    _FRAME[key] = clock[1]
    return curve(elapsed / duration)


# -- springs -----------------------------------------------------------------
#
# The third primitive, and the one the other two cannot be (UX.md Phase 5).
# :func:`value`'s exponential approach has no velocity: it decelerates into
# every target from wherever it is, so a target that changes mid-move is
# followed with a visible kink and a move that "arrives" does so by running out
# of distance. A spring carries velocity, which is what makes a re-aimed pill
# curve rather than corner, and -- being slightly under-damped -- what makes a
# popover settle *into* place instead of creeping up to it.
#
# Semi-implicit Euler with a substep ceiling rather than the analytic solution:
# the closed form is only closed for a *fixed* target, this module's whole
# subject is targets that change under a value, and a 4 ms substep is exact
# enough for a 200 ms move at a cost of at most five multiplies a frame.

# Angular frequency per unit of duration: one full undamped period over the
# duration asked for, so ``spring`` and ``value`` at the same duration take
# roughly the same time to become still.
_SPRING_OMEGA = 2.0 * math.pi

# Slightly under one. At 1.0 (critical) a spring never overshoots from rest,
# which is the whole of what this primitive was added for; below ~0.6 the
# overshoot reads as a bounce, which is a different and much louder feeling.
SPRING_DAMPING = 0.72

# The longest step the integrator will take at once. A frame that hitched to
# 250 ms is not a reason to integrate 250 ms of spring in one go: the explicit
# step is only stable while ``h * omega`` stays small, and the failure is not a
# slow spring but a divergent one.
_SPRING_STEP = 0.004

# The settle floors, shared with :func:`animating`: under both a spring is
# declared arrived, so over either it is still moving. One judgement in one
# place, because a clamp that used its own thresholds would disagree with the
# spring about the frame the move ends on.
_SETTLE_DISTANCE = 1e-3
_SETTLE_VELOCITY = 1e-2

_VELOCITY: dict[str, float] = {}


def spring(key: str, target: float, *, duration: float = tokens.DUR_BASE) -> float:
    """Like :func:`value`, but carrying velocity: it settles rather than eases.

    Under reduce-motion it snaps, like everything else in this module. It used
    to fall back to :func:`value` when ``effects.SPRINGS`` was off; that switch
    was folded away on 2026-08-12 (Phase 5's "folded away once trusted"), and
    reduce-motion -- which was always the accessibility answer here -- is the
    remaining way to turn the movement off.
    """
    clock = _clock()
    if REDUCED or clock is None or duration <= 0.0:
        _STATE[key] = target
        _VELOCITY[key] = 0.0
        _TARGET[key] = target
        _FRAME[key] = clock[1] if clock is not None else -1
        return target
    dt, frame = clock
    current = _STATE.get(key)
    _TARGET[key] = target
    _FRAME[key] = frame
    if current is None:
        # A key's first sighting snaps, as everywhere else here: a pane being
        # built for the first time should appear settled rather than assemble
        # itself out of nothing.
        _STATE[key] = target
        _VELOCITY[key] = 0.0
        return target
    omega = _SPRING_OMEGA / duration
    velocity = _VELOCITY.get(key, 0.0)
    steps = max(1, min(int(dt / _SPRING_STEP) + 1, 64))
    step = dt / steps
    for _ in range(steps):
        accel = -(omega * omega) * (current - target) - 2.0 * SPRING_DAMPING * omega * velocity
        velocity += accel * step
        current += velocity * step
    # Settled: a spring approaches its target asymptotically, so without a floor
    # it is *always* animating, which through ``animating()`` means the idle
    # clamp never idles again.
    if abs(target - current) < _SETTLE_DISTANCE and abs(velocity) < _SETTLE_VELOCITY:
        current, velocity = target, 0.0
    _STATE[key] = current
    _VELOCITY[key] = velocity
    return current


def restart(key: str) -> None:
    """Put a :func:`ease` key back at the start of its ramp."""
    _STATE[key] = 0.0


def seed(key: str, current: float) -> None:
    """State a key's value now, so its next :func:`value` eases *from* it.

    For the callers whose animated number lives somewhere else and is changed
    outside a frame -- ``layout.SIDEBAR_W`` is the one: without this, an
    animated change made before the first tick would snap, because a key that
    is not live yet has nothing to move from. Saying where it starts is more
    honest than requiring the caller to have ticked once first.

    Any velocity the key had is dropped: stating where a value *is* says
    nothing about where it was going, and a spring re-seeded to 0 while still
    carrying the velocity of its last arrival would leave at a speed nothing
    asked for.
    """
    _STATE[key] = current
    _VELOCITY.pop(key, None)


def forget(key: str) -> None:
    """Drop one key, so its next sighting snaps rather than animating."""
    _STATE.pop(key, None)
    _TARGET.pop(key, None)
    _FRAME.pop(key, None)
    _VELOCITY.pop(key, None)


def reset() -> None:
    """Forget everything; for tests and mode teardowns."""
    _STATE.clear()
    _TARGET.clear()
    _FRAME.clear()
    _VELOCITY.clear()
