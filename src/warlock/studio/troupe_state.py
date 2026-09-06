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

    #: The two view marks over the sprite, and they default differently on
    #: purpose. The checkerboard is **off**: a character sheet is judged on its
    #: colours first, and a pattern behind every frame is noise until the
    #: question is "where is the transparency". The pivot is **on**: it is the
    #: one thing on the picture that is not in the picture -- where the engine
    #: will put this sprite's feet -- and a user who does not know the mark
    #: exists will never turn it on to find out.
    checker: bool = False
    show_pivot: bool = True

    #: The new-character form, kept here rather than in ``state.preview`` so it
    #: survives a trip to another mode and back.
    form: dict[str, Any] = field(default_factory=dict)

    #: The throttled cast, and when it next goes stale. See
    #: ``troupe_mode.cast_and_pending``: the two walks behind it are SQL, and
    #: the pane that wants them draws every frame.
    cast_cache: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None
    cast_next: float = 0.0
    #: The same throttle for the two *disk* reads beside the SQL one. ``sheets``
    #: globs a job directory and reads a JSON sidecar per sheet, and
    #: ``active_sheet`` reads one -- both from pane draws, three to four times a
    #: frame between them, for a directory that changes only when a sheet is
    #: built. Keyed so a job or sheet change is read immediately rather than up
    #: to the interval later; see ``troupe_mode.invalidate_sheets``.
    sheets_cache: list[dict[str, Any]] | None = None
    sheets_key: str = ""
    sheets_next: float = 0.0
    sheet_cache: dict[str, Any] | None = None
    sheet_cache_key: tuple[str, str] = ("", "")
    sheet_cache_next: float = 0.0
    #: The pixel-art measurement for the selected sheet, keyed on its id. Read
    #: with a ``kind``-filtered 400-row page under the store's one lock, which
    #: this pane was doing *per frame* -- contending with the worker updating
    #: those very rows.
    pixel_report_cache: dict[str, Any] | None = None
    pixel_report_key: str = ""
    pixel_report_next: float = 0.0
    #: The mesh picker's list, and the file-existence answers behind it. See
    #: ``troupe_mode.sendable_meshes``: the predicate reads ``files``, which
    #: ``attach_files`` fills with one stat per listed name per row -- its own
    #: docstring calls that the frame loop's single largest syscall cost. The
    #: picker asked for it *per frame* for as long as its header was open.
    #: ``sendable_files`` is that helper's own ``{job: (stamp, names)}`` cache,
    #: owned here because the caller is required to own it.
    sendable_cache: list[dict[str, Any]] | None = None
    sendable_next: float = 0.0
    sendable_files: dict[str, Any] = field(default_factory=dict)



def ensure(ctx: Any) -> TroupeState:
    """This mode's state, built on first use."""
    if ctx.state.troupe is None:
        ctx.state.troupe = TroupeState()
    return ctx.state.troupe
