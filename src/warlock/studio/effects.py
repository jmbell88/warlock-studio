"""Which of the GPU-tier effects are switched on (UX.md Phase 5).

Four items ship behind four flags, which is what Phase 5 asks for in its own
words: "each item toggleable off (a config flag per item while it stabilizes,
folded away once trusted)". They live together in one stdlib-only module for
the reason ``motion.REDUCED`` lives in ``motion``: the consumers are scattered
-- a widget, a window, an easing primitive, the frame loop -- and threading a
settings object through all of them would put the switch in a dozen places
instead of one.

A flag being *on* is never a promise that the effect is available. Every
consumer degrades on its own to the pre-Phase-5 drawing when the GL side is
missing or has failed once (no renderer, no context, a texture that would not
build), because all four of these are appearance and none of them is worth a
traceback out of a frame. So the question a call site asks is always "is this
switched on **and** did it work", and the answer to the second half lives with
the code that does the work.

``REDUCE_MOTION`` is deliberately *not* here: it is an accessibility setting
rather than a rendering tier, it predates this module, and ``motion`` is where
every consumer already reads it from.
"""

from __future__ import annotations

from typing import Any

# The four, defaulting on: they are what the phase is, and a phase that ships
# switched off has not shipped. Each has exactly one settings key, spelled the
# same as the attribute in lower case -- see :data:`KEYS`.
SOFT_SHADOWS = True
VIBRANCY = True
SPRINGS = True
SQUIRCLES = True

# What each flag is called in the settings file and in the Settings pane, in
# the order the pane draws them. A table rather than four calls, because the
# pane, the loader and the test that keeps them in agreement are three readers
# of one list -- the argument ``landing.rows`` and ``palette.py`` both make.
KEYS: tuple[tuple[str, str, str], ...] = (
    (
        "soft_shadows",
        "Soft shadows",
        "Real blurred shadows under cards, popups and modals, from a texture "
        "built once at startup. Off falls back to concentric outlines.",
    ),
    (
        "vibrancy",
        "Translucent panels",
        "The command palette, the modals and the shortcuts popup blur what is "
        "behind them instead of covering it.",
    ),
    (
        "springs",
        "Spring motion",
        "The mode switch's pill and a popover's rise settle rather than "
        "easing to a stop. Reduce motion turns this off as well.",
    ),
    (
        "squircles",
        "Continuous corners",
        "Card corners are a superellipse rather than a circular arc -- a "
        "smaller difference than it sounds, and the most expensive of the four.",
    ),
)


def get(key: str) -> bool:
    """Whether one effect is on. Unknown names read as off."""
    return bool(globals().get(key.upper(), False)) if key.lower() in _NAMES else False


def set(key: str, value: bool) -> bool:  # noqa: A001 - the natural name for a flag table
    """Turn one effect on or off. -> what is now in force."""
    if key.lower() not in _NAMES:
        return False
    globals()[key.upper()] = bool(value)
    return bool(value)


def load(settings: Any) -> None:
    """Apply the stored flags, before the first frame is built.

    A missing key is the default rather than ``False``: a settings file written
    by a build that predates this module must not silently turn the phase off.
    """
    for key, _label, _help in KEYS:
        stored = settings.get(f"fx_{key}")
        if stored is not None:
            set(key, bool(stored))


def store(settings: Any, key: str) -> None:
    """Write one flag back. Prefixed, so it cannot collide with a pane's own key."""
    settings.set(f"fx_{key}", get(key))


_NAMES = frozenset(key for key, _label, _help in KEYS)
