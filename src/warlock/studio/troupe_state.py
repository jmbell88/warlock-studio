"""Troupe's session state: which character, which sheet, and where the clock is.

Split from the controller for :mod:`.packwright_state`'s reason -- these touch
nothing but ``ctx.state.troupe``, so a pane can read them without importing the
half that talks to the service and the task runner.

**There is no document here, and that is the mode.** Inker, Clay, Plotter and
Packwright each hold something the user is editing and can lose; Troupe holds a
*selection* -- a character sheet that already exists on disk, in the job
directory that owns it. Which is why the mode registers no journal provider and
no palette Save: a crash costs the user the frame the preview happened to be
on. Everything else is a file, and it was published by a worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TroupeState:
    """What Troupe remembers between frames."""

    #: The mesh job whose sheets are being looked at, and one of its sheets.
    #: Both are ids rather than records: a record cached across a frame is a
    #: record that can outlive the file it describes.
    job_id: str = ""
    sheet_id: str = ""

    #: What the preview is playing. Names from the selected sheet's snapshot --
    #: not indices, which would silently re-point if its frame table changed.
    animation: str = "walk"
    direction: str = "front"

    #: **Paused by default.** This ran on open, and the argument for that was
    #: that a bad frame becomes obvious immediately and a preview you have to
    #: press play on is one nobody presses play on. Overturned on request
    #: 2026-08-23: a clip that is already moving when you arrive is one you
    #: have to stop before you can look at anything in it, and the first thing
    #: anyone does with a new sheet is look at a *frame* -- checking a hand, a
    #: silhouette, the direction the feet point. Stepping already implies
    #: looking, which is why ``step`` clears this; opening should mean the same
    #: thing.
    playing: bool = False

    #: Seconds accumulated inside the current frame, and which frame that is.
    #: Kept as a float clock rather than a frame counter driven by the frame
    #: rate: the durations are milliseconds per *animation* and the window runs
    #: at whatever it runs at, so counting draws would play a 60 ms run cycle
    #: at monitor speed on one machine and half that on another.
    clock: float = 0.0
    frame: int = 0

    #: Playback speed, as a multiplier. For looking at a run cycle one frame at
    #: a time without editing the clip.
    speed: float = 1.0

    #: How many screen pixels one sprite pixel is drawn as. Integer by
    #: construction -- a fractional scale is a filtered sprite, which is the one
    #: thing a pixel-art preview must never show.
    zoom: int = 6

    #: The new-character form, kept here rather than in ``state.preview`` so it
    #: survives a trip to another mode and back.
    form: dict[str, Any] = field(default_factory=dict)



def ensure(ctx: Any) -> TroupeState:
    """This mode's state, built on first use."""
    if ctx.state.troupe is None:
        ctx.state.troupe = TroupeState()
    return ctx.state.troupe
