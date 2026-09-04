"""What Muse remembers between frames.

Much smaller than ``sirens_state`` / ``inker_state`` / ``plotter_state``,
because **Muse holds no document.** The other document workspaces each own a file
format, a tab list, a dirty flag and an undo stack; a take is a job row that a
worker wrote, so the store owns it and there is nothing here to lose. What is
left is a form and a pointer at whatever is currently making a noise -- which is
Troupe's shape, the one other workspace whose subject is rows a worker
published.

``ensure`` and ``active`` live here rather than in ``muse_mode`` for the reason
they live in ``sirens_state``: they touch exactly one thing, ``ctx.state.muse``,
and neither knows that a task thread or a mixer exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The form's defaults, and the single place they are written down.
#:
#: They are deliberately *not* ``create_music_job``'s signature defaults
#: repeated: that function's defaults are what an API caller gets, this is what
#: a new user sees, and a test asserting the two agree would freeze a UI choice
#: to a door's choice. What must agree is that every value here is one the door
#: accepts, which is a bound and not a value.
DEFAULT_FORM: dict[str, Any] = {
    "prompt": "",
    "lyrics": "",
    "duration": 60.0,
    "count": 1,
    # The recipe half, drawn by the column rather than the bar. Their names are
    # the model's own, so what the user changes and what is asked for do not
    # need a translation between them.
    "infer_step": 60,
    "guidance_scale": 15.0,
    "scheduler_type": "euler",
    "cfg_type": "apg",
    "omega_scale": 10.0,
    # None means "a fresh one per take", which is what a generative mode
    # defaults to; a number makes the first take reproducible.
    "seed": None,
}


@dataclass
class MuseState:
    """The mode's whole memory."""

    #: The brief and the recipe, one flat dict, edited in place by the bar and
    #: the column. One dict rather than two so that a future "reuse this take's
    #: settings" is a copy rather than a merge of two halves.
    form: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_FORM))

    #: The job id currently being auditioned, or "". The *audio* is
    #: ``sirens_audio``'s -- it is tag-keyed and mode-agnostic, and Muse passes
    #: the job id as the tag -- so this is only what the results tray needs in
    #: order to draw one card's button as Stop rather than Play. Kept here
    #: rather than asked of the mixer every frame because a card has to know
    #: whether *it* is the one playing, which the mixer's tag answers and its
    #: "is anything playing" does not.
    playing_job: str = ""

    #: The take the tray has selected, or "". Space auditions this one, which
    #: is why it is state rather than a hover.
    selected_job: str = ""


def ensure(ctx: Any) -> MuseState:
    """The mode's state, built on first use."""
    state = ctx.state.muse
    if state is None:
        state = MuseState()
        ctx.state.muse = state
    return state


def active(ctx: Any) -> MuseState | None:
    """The state, or ``None``. Deliberately *not* through :func:`ensure`:
    asking what Muse holds must not create the state that says nothing."""
    return ctx.state.muse
