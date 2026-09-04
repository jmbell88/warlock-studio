"""Editing one of an instrument's four sequences: the arithmetic, with no pane.

This is the half of the envelope editor that decides *what* an edit is, rather
than what it looks like. It lived in ``panes/sirens_envelopes.py`` until
2026-09-04, which meant a test for "a fast drag fills the columns it skipped"
or "a marker cannot be dragged somewhere it stops being visible" had to import
imgui to ask. It answers to numbers alone, so it belongs down here beside the
:class:`~.instruments.Sequence` it edits -- the pane re-exports every name at
its old address.

**The two markers are constrained, not merely clamped.** ``release == 0`` makes
every value tail material and a held note silent, and a ``loop`` at or past the
release is a marker the graph cannot draw and the engine will not use. Both are
states the engine tolerates from a hand-edited file and neither is one a drag
should be able to create: an editor that lets you put a handle where it then
disappears is an editor you cannot trust the picture of.
"""

from __future__ import annotations

from dataclasses import replace

from . import instruments as inst
from . import notes

#: How many columns a graph draws when the sequence is shorter than that. An
#: empty sequence with one column would be a single bar you cannot drag a shape
#: into; sixteen is a bar at the default speed, which is the unit an envelope is
#: written in.
MIN_STEPS = 16

#: The editor's reach for the two signed sequences, which the engine does not
#: bound. One octave of arpeggio, because that is what a chord written on one
#: voice spans, and one semitone of pitch per tick, because a cent is a
#: hundredth of one. Neither is a limit: :func:`span` widens to fit whatever a
#: sequence already holds, so a file with a two-octave arpeggio is drawn whole.
ARPEGGIO_REACH = len(notes.SEMITONES)
PITCH_REACH = 100


def span(field: str, sequence: inst.Sequence) -> tuple[int, int]:
    """The low and high value one graph draws between.

    Volume and duty are the engine's own bounds and start at zero. Arpeggio and
    pitch are signed and centred, and their reach *grows* to hold a sequence
    that is already wider -- drawing a stored +18 semitones clipped at +12 would
    be the editor quietly disagreeing with the file.
    """
    if field == "volume":
        return 0, inst.MAX_VOLUME
    if field == "duty":
        return 0, inst.MAX_DUTY
    reach = ARPEGGIO_REACH if field == "arpeggio" else PITCH_REACH
    reach = max(reach, max((abs(int(v)) for v in sequence.values), default=0))
    return -reach, reach


def columns(sequence: inst.Sequence) -> int:
    """How many steps the graph shows. Never fewer than :data:`MIN_STEPS`, so
    there is somewhere to drag a shape into an empty sequence."""
    return max(len(sequence.values), min(MIN_STEPS, inst.MAX_SEQUENCE_LEN))


def painted(
    sequence: inst.Sequence, step: int, value: int, *, previous: int = -1
) -> inst.Sequence:
    """``sequence`` with ``step`` set to ``value``, extended if it has to be.

    Two things happen here that a plain assignment would not. **Painting past
    the end lengthens the sequence**, which is how an envelope is written: you
    drag right and it grows, rather than setting a length first and then filling
    it in. The steps that appear between hold the last value the sequence had,
    so extending a flat tone stays a flat tone.

    And **``previous`` fills the run**. A pointer crosses several columns in one
    frame, so painting only the column it happens to be over on a frame boundary
    leaves a comb of untouched steps behind a quick drag -- which reads as the
    editor dropping input.

    ``value`` arrives already inside the graph's range: :func:`value_at` reads
    it off the pointer against the span this sequence is drawn over, and a
    sequence does not carry which parameter it is, so this is the wrong place
    to ask.
    """
    values = [int(v) for v in sequence.values]
    step = max(0, min(int(step), inst.MAX_SEQUENCE_LEN - 1))
    first = step if previous < 0 else max(0, min(int(previous), inst.MAX_SEQUENCE_LEN - 1))
    lo, hi = min(first, step), max(first, step)
    if hi >= len(values):
        fill = values[-1] if values else int(value)
        values.extend([fill] * (hi + 1 - len(values)))
    values[lo : hi + 1] = [int(value)] * (hi - lo + 1)
    return replace(sequence, values=tuple(values))


def marker_bounds(sequence: inst.Sequence, grip: str) -> tuple[int, int]:
    """The lowest and highest step ``grip`` may sit on. -> ``(low, high)``.

    The floors are the two invisible states in the module docstring. A
    ``release`` starts at 1, because at 0 the whole sequence is tail and the
    held note is silent. A ``loop`` stops one step short of a live release,
    because the loop repeats the *held* half and a loop point inside the tail
    is a handle the graph stops drawing and the engine never reaches -- which is
    a marker the user set, cannot see, and would find again on the next save.

    ``high`` can fall below ``low`` -- a one-step sequence with a release on it
    has nowhere left for a loop. The caller reads that as "no room", which is
    the honest answer and the one :func:`moved` refuses on.
    """
    top = len(sequence.values) - 1
    if grip == "release":
        return 1, top
    release = int(sequence.release)
    if 0 <= release <= top:
        return 0, release - 1
    return 0, top


def moved(sequence: inst.Sequence, grip: str, step: int) -> inst.Sequence:
    """A marker dragged to ``step``, clamped into where it can be seen.

    A loop or release index outside the values is not an error the engine minds
    -- ``Sequence.index_at`` treats it as absent -- but it *is* a marker the
    user cannot see, which is worse than one that stops at the edge. The bounds
    are :func:`marker_bounds`; a drag with no room left inside them leaves the
    sequence alone rather than putting the handle somewhere it vanishes.
    """
    if not sequence.values:
        return sequence
    low, high = marker_bounds(sequence, grip)
    if high < low:
        return sequence
    step = max(low, min(int(step), high))
    return replace(sequence, **{grip: step})


def toggled(sequence: inst.Sequence, grip: str) -> inst.Sequence:
    """Turn a marker on or off, landing an enabled one where it can be seen.

    A release lands halfway rather than at zero, for :func:`marker_bounds`'
    reason. A loop lands at step 0, which is always inside its own bounds --
    a release it would have to clear is at 1 or above.
    """
    if getattr(sequence, grip) >= 0:
        return replace(sequence, **{grip: -1})
    if not sequence.values:
        return sequence
    if grip == "loop":
        return replace(sequence, loop=0)
    if len(sequence.values) < 2:
        # A one-step sequence has no held half to split off, so there is
        # nowhere legal for a release to land. Refused rather than put at 0.
        return sequence
    return replace(sequence, release=max(1, len(sequence.values) // 2))


def grabbed(sequence: inst.Sequence, offset: float, col_w: float, grip_w: float) -> str:
    """What a press ``offset`` pixels into the graph takes hold of.

    Release before loop, deliberately: the two can sit on adjacent steps, and
    the release is the one whose position changes what is heard while the key
    is down -- so it is the one a press between them should get.
    """
    for grip in ("release", "loop"):
        index = int(getattr(sequence, grip))
        if 0 <= index < len(sequence.values) and abs(index * col_w - offset) <= grip_w:
            return grip
    return "paint"


def step_at(offset: float, col_w: float, count: int) -> int:
    """Which column an x offset into the graph is over."""
    if col_w <= 0:
        return 0
    return max(0, min(int(offset // col_w), count - 1))


def value_at(offset: float, height: float, low: int, high: int) -> int:
    """Which value a y offset into the graph is at. Zero is the top."""
    if height <= 0:
        return high
    part = 1.0 - max(0.0, min(offset / height, 1.0))
    return int(round(low + part * (high - low)))


def resized(sequence: inst.Sequence, steps: int) -> inst.Sequence:
    """Lengthen or shorten a sequence, holding its last value into the new room.

    The markers come with it: a release point left at step 40 of a sequence
    shortened to 8 is a release the engine ignores and the editor cannot draw,
    which is the same invisible state :func:`moved` refuses to create. They go
    back through :func:`moved` rather than through a bare ``min``, so a release
    squeezed onto step 0 by a shortening -- and a loop that release would then
    swallow -- cannot be created here either.
    """
    steps = max(0, min(int(steps), inst.MAX_SEQUENCE_LEN))
    values = [int(v) for v in sequence.values]
    if steps > len(values):
        values.extend([values[-1] if values else 0] * (steps - len(values)))
    else:
        values = values[:steps]
    out = replace(sequence, values=tuple(values), loop=-1, release=-1)
    if not values:
        return out
    for grip in ("release", "loop"):
        index = int(getattr(sequence, grip))
        if index < 0:
            continue
        low, high = marker_bounds(out, grip)
        if high < low:
            # No room left for this marker in the shortened sequence. Dropped
            # rather than pinned to an edge that is not inside its own bounds:
            # a one-step sequence has no held half for a release to split off.
            continue
        out = replace(out, **{grip: max(low, min(index, high))})
    return out
