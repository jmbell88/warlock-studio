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
"""

from __future__ import annotations

import math

from imgui_bundle import imgui

from . import tokens

_STATE: dict[str, float] = {}


def value(key: str, target: float, *, duration: float = tokens.DUR_BASE) -> float:
    """The animated value for ``key`` this frame, moving towards ``target``."""
    current = _STATE.get(key)
    if current is None or duration <= 0.0:
        _STATE[key] = target
        return target
    dt = max(imgui.get_io().delta_time, 1e-4)
    # duration/3: an exponential approach covers ~95% of the distance in three
    # time constants, so "duration" reads as "how long the move visibly takes".
    current += (target - current) * (1.0 - math.exp(-dt / (duration / 3.0)))
    if abs(target - current) < 1e-3:
        current = target
    _STATE[key] = current
    return current


def reset() -> None:
    """Forget everything; for tests and mode teardowns."""
    _STATE.clear()
