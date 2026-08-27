"""Sirens' envelope editor: the four sequences an instrument is made of.

**Why this is a pane and not four fields in the instrument list.** A FamiTracker
instrument is not an ADSR with four knobs -- it is four short lists of numbers
stepped once a tick, and the shape of the list *is* the sound. ``volume``
``[15,14,12,8,4,2,1,0]`` is a pluck and ``[15,15,15,15]`` on loop is an organ;
those two are indistinguishable as a row of spinboxes and obvious as two curves.
So each sequence is a bar graph you drag on, which is what the Phase 2
instrument pane's docstring said this would be.

**One draw list and one invisible button per graph**, which is
``sirens_patterns``' argument at a smaller scale: a 256-step sequence is 256
bars, and one imgui widget per bar is 256 ids and 256 hit-tests a frame for
rectangles that are never interactive individually. What a canvas spends its
effort not building is a widget tree.

**A drag is one undo step.** The gesture is opened and closed in
``sirens_mode`` -- ``begin_envelope_drag`` records ``len(doc.history)`` and
``end_envelope_drag`` folds the run with ``collapse_since`` -- because painting
a decay across twenty columns is twenty ``InstrumentEdit``s and nobody presses
Ctrl+Z twenty times to take back one drag. ``document.add_oneshot`` does the
same thing at the same place.

**``release`` splits the sequence, and this pane has to say so on screen.**
Everything before the release point is what a held note plays; everything from
it is the tail that plays after a note-off, and the tail never loops. That rule
is invisible in a list of numbers -- it is the one Phase 1 got wrong on its
first attempt -- so the tail is drawn on its own ground, in its own colour, with
its own label, and the loop's span is underlined along the bottom of the held
half. A reader should be able to see which half is which without being told.

**No new value ranges.** Volume and duty are bounded by the engine
(``instruments.MAX_VOLUME``, ``instruments.MAX_DUTY``) and every sequence by
``instruments.MAX_SEQUENCE_LEN``. Arpeggio and pitch are *not* bounded by it --
both are added to a float pitch and neither has a ceiling to read -- so what is
here is the editor's **reach** rather than a rule, and a sequence that already
holds more than the reach widens its own graph instead of being drawn clipped.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .. import controls, sirens_mode, theme, widgets
from ..manual import render as manual_render
from ..sirens import instruments as inst
from ..sirens import notes
from ..tokens import sp

#: ``(field, heading)`` for the four graphs, in the order they stack.
#: ``sirens_mode.ENVELOPE_FIELDS`` is the authority on the set; a test asserts
#: the two agree, so a fifth sequence in the engine cannot reach one and not
#: the other.
FIELDS: tuple[tuple[str, str], ...] = (
    ("volume", "Volume"),
    ("arpeggio", "Arpeggio"),
    ("pitch", "Pitch"),
    ("duty", "Duty"),
)

#: One graph's height and the marker strip's, in design pixels. Tall enough for
#: sixteen volume steps to be distinguishable from fifteen and short enough
#: that four of them and their headers fit a sidebar without scrolling.
GRAPH_H = 52.0

#: How many columns a graph draws when the sequence is shorter than that. An
#: empty sequence with one column would be a single bar you cannot drag a shape
#: into; sixteen is a bar at the default speed, which is the unit an envelope is
#: written in.
MIN_STEPS = 16

#: How near the pointer has to be to a marker line, in design pixels, to grab
#: it rather than paint. Two pixels reads as "the line does not move"; ten eats
#: enough of the graph that painting near a marker becomes impossible.
GRIP_W = 5.0

#: The editor's reach for the two signed sequences, which the engine does not
#: bound. One octave of arpeggio, because that is what a chord written on one
#: voice spans, and one semitone of pitch per tick, because a cent is a
#: hundredth of one. Neither is a limit: :func:`span` widens to fit whatever a
#: sequence already holds, so a file with a two-octave arpeggio is drawn whole.
ARPEGGIO_REACH = len(notes.SEMITONES)
PITCH_REACH = 100

#: What the column needs before its graphs stop being lines. Four graphs, four
#: headers and the caption.
ENVELOPES_FLOOR = 300.0


# --- the pure half ------------------------------------------------------------
#
# Everything below that decides *what* an edit is, rather than what it looks
# like, is a function of numbers alone. That is not tidiness: this box has no
# display, so a test for "a fast drag fills the columns it skipped" can only
# exist if the answer is reachable without an imgui frame.


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


def moved(sequence: inst.Sequence, grip: str, step: int) -> inst.Sequence:
    """A marker dragged to ``step``, clamped inside the values it points into.

    A loop or release index past the end of the values is not an error the
    engine minds -- ``Sequence.index_at`` treats it as absent -- but it *is* a
    marker the user cannot see, which is worse than one that stops at the edge.
    """
    if not sequence.values:
        return sequence
    step = max(0, min(int(step), len(sequence.values) - 1))
    return replace(sequence, **{grip: step})


def toggled(sequence: inst.Sequence, grip: str) -> inst.Sequence:
    """Turn a marker on or off, landing an enabled one where it can be seen.

    A release lands halfway rather than at zero: ``release == 0`` makes every
    value tail material and a held note silent, which is a state the engine
    accepts from a hand-edited file and not one a button should produce.
    """
    if getattr(sequence, grip) >= 0:
        return replace(sequence, **{grip: -1})
    if not sequence.values:
        return sequence
    if grip == "loop":
        return replace(sequence, loop=0)
    return replace(sequence, release=max(1, len(sequence.values) // 2))


def grabbed(sequence: inst.Sequence, offset: float, col_w: float, grip_w: float) -> str:
    """What a press ``offset`` pixels into the graph takes hold of.

    Release before loop, deliberately: the two can sit on the same step, and
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


def _baseline(low: int, high: int, height: float) -> float:
    """How far down the graph the value ``0`` sits -- the bottom for volume and
    duty, the middle for the two signed sequences."""
    if high == low:
        return height
    return height * (high - 0.0) / (high - low)


def _bar_top(value: int, low: int, high: int, height: float) -> float:
    if high == low:
        return height
    return height * (high - max(low, min(int(value), high))) / (high - low)


# --- drawing ------------------------------------------------------------------


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = sirens_mode.ensure(ctx)
    tab = state.active
    widgets.section("Envelopes")
    manual_render.help_button(ctx, "sirens-envelopes")

    if tab is None:
        return
    doc = tab.doc
    instrument = None if state.instrument is None else doc.instrument(state.instrument)
    if instrument is None:
        widgets.muted_wrapped(
            "Select an instrument above to shape the four sequences it ticks"
            " through: volume, arpeggio, pitch and duty."
        )
        return

    widgets.muted(f"{instrument.uid:02X}  {instrument.name or 'Instrument'}")
    widgets.muted_wrapped(
        "Drag in a graph to paint it. The release marker splits a sequence:"
        " everything left of it plays while the note is held and loops there;"
        " everything from it is the tail after a note-off, and the tail never"
        " repeats."
    )
    imgui.dummy((0, sp(4)))

    for field, label in FIELDS:
        _graph(ctx, state, tab, instrument, field, label)


def _graph(ctx: Any, state: Any, tab: Any, instrument: Any, field: str, label: str) -> None:
    from imgui_bundle import imgui

    sequence = getattr(instrument, field)
    _header(ctx, state, tab, instrument, field, label, sequence)

    low, high = span(field, sequence)
    count = columns(sequence)
    origin = imgui.get_cursor_screen_pos()
    width = max(imgui.get_content_region_avail().x, 1.0)
    height = sp(GRAPH_H)
    col_w = width / count
    draw_list = imgui.get_window_draw_list()

    ground = imgui.get_color_u32(theme.rgba(theme.ELEV_1))
    tail_ground = imgui.get_color_u32(theme.rgba(theme.WARN, 0.10))
    edge = imgui.get_color_u32(theme.rgba(theme.EDGE))
    held_bar = imgui.get_color_u32(theme.rgba(theme.ACCENT, 0.85))
    tail_bar = imgui.get_color_u32(theme.rgba(theme.WARN, 0.85))
    loop_line = imgui.get_color_u32(theme.rgba(theme.OK))
    release_line = imgui.get_color_u32(theme.rgba(theme.WARN))
    muted = imgui.get_color_u32(theme.rgba(theme.MUTED))

    draw_list.add_rect_filled(
        (origin.x, origin.y), (origin.x + width, origin.y + height), ground
    )
    release = int(sequence.release)
    split = release if 0 <= release < len(sequence.values) else -1
    if split >= 0:
        # The tail on its own ground. The single most legible way to say that
        # these two halves are played at different times.
        draw_list.add_rect_filled(
            (origin.x + split * col_w, origin.y),
            (origin.x + width, origin.y + height),
            tail_ground,
        )

    base = _baseline(low, high, height)
    if low < 0:
        # The zero line, which the signed graphs are read against: a bar that
        # goes down is a note bent flat, and without the line it is just a bar.
        draw_list.add_line(
            (origin.x, origin.y + base), (origin.x + width, origin.y + base), edge, 1.0
        )

    for index, value in enumerate(sequence.values):
        x0 = origin.x + index * col_w
        top = origin.y + _bar_top(int(value), low, high, height)
        bottom = origin.y + base
        colour = tail_bar if 0 <= split <= index else held_bar
        draw_list.add_rect_filled(
            (x0 + 1.0, min(top, bottom)), (x0 + max(col_w - 1.0, 1.0), max(top, bottom) + 1.0),
            colour,
        )

    _markers(
        draw_list,
        origin,
        sequence,
        col_w,
        height,
        split,
        loop=loop_line,
        release=release_line,
        muted=muted,
    )
    # ``add_rect`` is (p_min, p_max, col, rounding, thickness, flags) -- the
    # rounding and the thickness are adjacent and both floats, which is a pair
    # it is easy to fill in the other way round and get a TypeError only on the
    # frames that draw the rectangle.
    draw_list.add_rect(
        (origin.x, origin.y), (origin.x + width, origin.y + height), edge, 0.0, 1.0
    )

    imgui.invisible_button(f"sirens-env-{field}", (width, height))
    _input(ctx, state, tab, instrument, field, sequence, origin, col_w, height, count, low, high)
    imgui.dummy((0, sp(4)))


def _markers(
    draw_list: Any,
    origin: Any,
    sequence: inst.Sequence,
    col_w: float,
    height: float,
    split: int,
    *,
    loop: int,
    release: int,
    muted: int,
) -> None:
    """The two handles, and the span the loop repeats.

    The underline is the half of this that is not decoration: ``loop`` is where
    the *held* half jumps back to, and where it jumps back *from* is the release
    point or the end of the values -- which is arithmetic the reader should not
    have to do while looking at a curve.
    """
    count = len(sequence.values)
    end = split if split >= 0 else count
    index = int(sequence.loop)
    if 0 <= index < end:
        draw_list.add_line(
            (origin.x + index * col_w, origin.y + height - 2.0),
            (origin.x + end * col_w, origin.y + height - 2.0),
            loop,
            2.0,
        )
        draw_list.add_line(
            (origin.x + index * col_w, origin.y),
            (origin.x + index * col_w, origin.y + height),
            loop,
            1.5,
        )
        draw_list.add_text((origin.x + index * col_w + 2.0, origin.y), loop, "L")
    if split >= 0:
        draw_list.add_line(
            (origin.x + split * col_w, origin.y),
            (origin.x + split * col_w, origin.y + height),
            release,
            1.5,
        )
        draw_list.add_text((origin.x + split * col_w + 2.0, origin.y), release, "R")
    elif not count:
        draw_list.add_text((origin.x + 4.0, origin.y + height * 0.5 - 6.0), muted, "empty")


def _header(
    ctx: Any, state: Any, tab: Any, instrument: Any, field: str, label: str,
    sequence: inst.Sequence,
) -> None:
    """The one line over each graph: what it is, how long, and the two markers.

    The markers are buttons *and* handles: a drag moves one, and the button is
    how a sequence that has none gets one -- there is nothing on the graph to
    grab until it exists.
    """
    from imgui_bundle import imgui

    editable = not tab.busy
    imgui.text(label)
    imgui.same_line()
    imgui.set_next_item_width(sp(46))
    changed, steps = controls.drag_int(
        f"##sirens-env-steps-{field}",
        len(sequence.values),
        1.0,
        0,
        inst.MAX_SEQUENCE_LEN,
        enabled=editable,
        commit=True,
        tooltip="How many steps this sequence ticks through. Painting past the"
        " end lengthens it too.",
    )
    if changed and editable:
        sirens_mode.set_sequence(
            ctx, tab, instrument.uid, field, _resized(sequence, int(steps))
        )
    for grip, name in (("loop", "Loop"), ("release", "Tail")):
        index = int(getattr(sequence, grip))
        imgui.same_line()
        shown = f"{name} {index}" if index >= 0 else name
        if controls.small_button(
            f"{shown}##sirens-env-{grip}-{field}",
            selected=index >= 0,
            enabled=editable and bool(sequence.values),
            reason="This sequence has no steps yet." if not sequence.values else "",
            tooltip=(
                "Where the held half jumps back to when it runs off its end."
                if grip == "loop"
                else "Where the tail starts. Everything from here plays after"
                " the note is released, and never loops."
            ),
        ):
            sirens_mode.set_sequence(
                ctx, tab, instrument.uid, field, toggled(sequence, grip)
            )


def _resized(sequence: inst.Sequence, steps: int) -> inst.Sequence:
    """Lengthen or shorten a sequence, holding its last value into the new room.

    The markers come with it: a release point left at step 40 of a sequence
    shortened to 8 is a release the engine ignores and the editor cannot draw,
    which is the same invisible state :func:`moved` refuses to create.
    """
    steps = max(0, min(int(steps), inst.MAX_SEQUENCE_LEN))
    values = [int(v) for v in sequence.values]
    if steps > len(values):
        values.extend([values[-1] if values else 0] * (steps - len(values)))
    else:
        values = values[:steps]
    top = len(values) - 1
    return inst.Sequence(
        values=tuple(values),
        loop=min(int(sequence.loop), top),
        release=min(int(sequence.release), top),
    )


def _input(
    ctx: Any,
    state: Any,
    tab: Any,
    instrument: Any,
    field: str,
    sequence: inst.Sequence,
    origin: Any,
    col_w: float,
    height: float,
    count: int,
    low: int,
    high: int,
) -> None:
    """The gesture, in imgui's three moments: activated, active, deactivated.

    ``is_item_activated`` rather than ``is_mouse_clicked`` because the button
    owns the press: a click that started outside this graph and dragged over it
    must not begin painting here, which is the same rule that keeps a slider
    from grabbing a drag begun on its neighbour.
    """
    from imgui_bundle import imgui

    if tab.busy:
        return
    mine = state.env_field == field
    if imgui.is_item_activated():
        offset = imgui.get_mouse_pos().x - origin.x
        grip = grabbed(sequence, offset, col_w, sp(GRIP_W))
        sirens_mode.begin_envelope_drag(ctx, tab, field, grip)
        mine = state.env_field == field
    if not (mine and imgui.is_item_active()):
        if mine and imgui.is_item_deactivated():
            sirens_mode.end_envelope_drag(ctx, tab)
        return

    mouse = imgui.get_mouse_pos()
    step = step_at(mouse.x - origin.x, col_w, count)
    if state.env_grip in ("loop", "release"):
        after = moved(sequence, state.env_grip, step)
    else:
        value = value_at(mouse.y - origin.y, height, low, high)
        after = painted(sequence, step, value, previous=state.env_step)
    state.env_step = step
    sirens_mode.set_sequence(ctx, tab, instrument.uid, field, after)
    # No close here: ``is_item_active`` and ``is_item_deactivated`` are never
    # both true on one frame -- deactivation is the frame *after* the button
    # stopped being active -- so the gesture is closed by the branch above.
