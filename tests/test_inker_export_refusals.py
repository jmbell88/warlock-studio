"""The export door's own refusals, driven through the mode rather than the engine.

``sheetout`` already refuses every one of these, and its tests already prove it
does. What none of them prove is that the *door* refuses first -- and that is
the whole point of the checks in ``inker_mode._submit_export``: the engine's
``ValueError`` is a backstop that arrives **after** the user has picked a
filename, as an opaque task error with no obvious cause, so each of these is
re-asked on the frame thread before the save dialog opens.

A door check that silently stopped firing would leave every engine test green
and every one of these failures reachable again as a task error. This file is
what makes that a red test, and it asserts the three things the engine layer
cannot see:

* the toast is raised, and it is a ``warn``,
* **the tab's lock is put back** -- ``tab.saving`` is what ``busy`` refuses
  mutation on, so a refusal that forgot it would leave the document read-only
  until the tab was closed, and
* **nothing is submitted** -- no task key, so no save dialog ever opened.

Carried from the Aseprite parity programme's own Wave 4 owed list,
where the gap was named ("none of the arrange/merge/skip_empty/trim/padding/
extrude early refusals have a dedicated UI-level test") and re-named in Wave 5's
as still open.

One entry in that list has no door check and deliberately so: **trim**. It
composes with every layout and every arrange -- it changes the cell size, not
which cells exist -- so there is nothing for a door to refuse. It is exercised
here only as the negative control that says so.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from warlock.studio import inker, inker_mode
from warlock.studio.inker import sheetout
from warlock.studio.inker.animation import DirectionalLayout
from warlock.studio.inker_state import InkerDoc, InkerState

RED = (255, 0, 0, 255)


class _Ctx:
    """``test_inker_export_steps._Ctx``, and the same three things asked of it."""

    def __init__(self, state: Any) -> None:
        self.state = _AppState(state)
        self.toasts: list[tuple[str, str]] = []
        self.submitted: list[str] = []
        self.accept = True

    def toast(self, message: str, kind: str = "info", **_kw: Any) -> None:
        self.toasts.append((message, kind))

    def submit(self, key: str, run: Any) -> bool:
        self.submitted.append(key)
        self.run = run
        return self.accept


class _AppState:
    def __init__(self, inker_state: Any) -> None:
        self.inker = inker_state


def _animated(frames: int) -> Any:
    doc = inker.Document.blank(4, 4)
    weight = np.ones((2, 2), dtype=np.float32)
    doc.write_colour((0, 0, 2, 2), RED, weight)
    for _ in range(frames - 1):
        doc.add_frame(link=True)
    return doc


def _open(frames: int = 4, *, layout: str | None = None):
    doc = _animated(frames)
    if layout is not None:
        # Construction-time state, exactly as an imported sheet carries it --
        # no undo op edits a layout, so setting it directly is what loading one
        # does (``ora._read`` and ``sheetin.document_from_sheet`` both assign
        # ``anim.layout``).
        doc.anim.layout = DirectionalLayout.of(layout)
    tab = InkerDoc(doc=doc, title="walk.ora")
    state = InkerState()
    state.add(tab)
    return _Ctx(state), state, tab


def _run_to_submit(ctx: Any, state: Any, limit: int = 200) -> None:
    """Pump until the export leaves the stepper, however it leaves it."""
    for _ in range(limit):
        if state.export is None:
            return
        inker_mode.pump_export(ctx)
    raise AssertionError("the export never left the stepper")


def _refused(ctx: Any, tab: Any) -> None:
    """The three assertions every one of these refusals owes."""
    assert ctx.submitted == [], "a refused export must not reach a save dialog"
    assert tab.saving is False, "a refusal must put the tab's lock back"
    assert ctx.toasts, "a refusal must say why"
    assert ctx.toasts[-1][1] == "warn"


# -- padding / extrude ---------------------------------------------------


def test_padding_below_twice_extrude_is_refused_at_the_door():
    """``build`` raises this too, but only once the file has been named. Two
    neighbours extrude into one shared gutter, so the gutter has to be twice
    one extrusion; the door says which two numbers disagree."""
    ctx, state, tab = _open(2)
    state.export_padding = 1
    state.export_extrude = 1
    inker_mode.export_sheet(ctx, tab)
    _run_to_submit(ctx, state)
    _refused(ctx, tab)
    assert "Padding must be at least twice Extrude" in ctx.toasts[-1][0]
    # The numbers, not just the rule -- a message naming neither is one the
    # user has to go and work out for themselves.
    assert "1 x 2 = 2" in ctx.toasts[-1][0]
    assert "padding is 1" in ctx.toasts[-1][0]


def test_padding_exactly_twice_extrude_is_allowed():
    """The boundary, from the legal side: the rule is ``>=``, and an off-by-one
    here would refuse the tightest packing that is actually correct."""
    ctx, state, tab = _open(2)
    state.export_padding = 2
    state.export_extrude = 1
    inker_mode.export_sheet(ctx, tab)
    _run_to_submit(ctx, state)
    assert ctx.submitted == [f"inker-export:{tab.uid}"]


def test_the_padding_door_is_the_sheet_export_only():
    """A GIF has no atlas, so it has no gutters and nothing to refuse. Pinned
    because the check reads ``export.kind`` and dropping that condition would
    make a padding setting left over from a sheet export refuse every GIF."""
    ctx, state, tab = _open(2)
    state.export_padding = 1
    state.export_extrude = 1
    inker_mode.export_gif(ctx, tab)
    _run_to_submit(ctx, state)
    assert ctx.submitted == [f"inker-export:{tab.uid}"]


# -- the directional-layout trio -----------------------------------------


def test_a_frame_added_to_a_directional_sheet_is_refused_by_count():
    """Drawing a fifth frame in a four-frame turnaround is an *ordinary edit* --
    the layout is construction-time state and no op forbids it -- so the
    mismatch has to surface at export, naming both counts rather than the
    document's."""
    ctx, state, tab = _open(5, layout="turnaround")
    inker_mode.export_sheet(ctx, tab)
    _run_to_submit(ctx, state)
    _refused(ctx, tab)
    assert "turnaround sheet of 4 frames" in ctx.toasts[-1][0]
    assert "document has 5" in ctx.toasts[-1][0]


def test_arrange_over_a_directional_sheet_is_refused_at_the_door():
    """A directional grid already fixes its columns; an Arrange fixing them for
    a second reason would be a silent coin flip about which wins."""
    ctx, state, tab = _open(4, layout="turnaround")
    state.export_arrange = "rows"
    state.export_wrap = 2
    inker_mode.export_sheet(ctx, tab)
    _run_to_submit(ctx, state)
    _refused(ctx, tab)
    assert "keeps its own fixed grid" in ctx.toasts[-1][0]
    assert "Arrange back to Grid" in ctx.toasts[-1][0]


@pytest.mark.parametrize("option", ["export_merge", "export_skip_empty"])
def test_merge_and_skip_empty_over_a_directional_sheet_are_refused(option: str):
    """Both, separately, because they are refused by one condition and a change
    that dropped either side of the ``or`` would leave this green with one of
    them reachable. A directional cell is a pose at a yaw; there is nothing for
    a merge or an omission to act on that would not change what the grid claims."""
    ctx, state, tab = _open(4, layout="turnaround")
    setattr(state, option, True)
    inker_mode.export_sheet(ctx, tab)
    _run_to_submit(ctx, state)
    _refused(ctx, tab)
    assert "turn Merge and Skip empty off" in ctx.toasts[-1][0]


def test_trim_over_a_directional_sheet_is_allowed():
    """The negative control this file's docstring promises. ``trim`` changes the
    cell size, never which cells exist, so it composes with a fixed grid and has
    no door check -- and a future door that refused it would be wrong."""
    ctx, state, tab = _open(4, layout="turnaround")
    state.export_trim = True
    inker_mode.export_sheet(ctx, tab)
    _run_to_submit(ctx, state)
    assert ctx.submitted == [f"inker-export:{tab.uid}"]


def test_a_partial_span_drops_the_layout_and_with_it_every_layout_refusal():
    """``timing`` forces a partial span's layout to None -- half a directional
    sheet is a clip, not a smaller sheet of the same kind -- so an Arrange over
    a *range* export of the same document is legal where the whole-timeline one
    is refused. The refusals key off the layout, so this is the pin that they
    key off the layout the export actually got."""
    ctx, state, tab = _open(4, layout="turnaround")
    state.export_arrange = "rows"
    state.export_wrap = 2
    inker_mode.export_range(ctx, tab, "sheet", (0, 2))
    _run_to_submit(ctx, state)
    assert ctx.submitted == [f"inker-export:{tab.uid}"]


# -- {frame} in a split --------------------------------------------------


@pytest.mark.parametrize("kind", ["tag", "layer"])
def test_a_split_template_naming_frame_is_refused_before_the_lock(kind: str):
    """Wave 4's other named gap. ``filename_for`` refuses ``{frame}`` on an
    export that has no frame, and ``_split_stems`` is the call that reaches it
    for a per-tag or per-layer batch -- but the pin was at the ``sheetout``
    layer only, so nothing proved the split runners actually pass the user's
    template down that path.

    Refused in ``_begin_export``, *before* ``tab.saving`` is ever set: a
    collision (and a bad template) is a property of the labels alone, so it can
    be answered without locking anything or spreading a single flatten."""
    ctx, state, tab = _open(2)
    state.export_template = "{title}_{frame}"
    inker_mode._begin_export(
        ctx,
        tab,
        "sheet",
        legs=[inker_mode._Leg(uids=[], label="walk", split_kind=kind)],
    )
    assert ctx.submitted == []
    assert tab.saving is False
    assert state.export is None
    assert ctx.toasts[-1][1] == "warn"
    assert "no frame" in ctx.toasts[-1][0]
    assert "{frame}" in ctx.toasts[-1][0]


def test_a_split_template_colliding_two_labels_is_refused_before_the_lock():
    """The same door, the other rule it enforces. A template that ignores the
    label renders every leg the same name, which would have the batch overwrite
    itself one file at a time."""
    ctx, state, tab = _open(2)
    state.export_template = "{title}"
    inker_mode._begin_export(
        ctx,
        tab,
        "sheet",
        legs=[
            inker_mode._Leg(uids=[], label="walk", split_kind="tag"),
            inker_mode._Leg(uids=[], label="run", split_kind="tag"),
        ],
    )
    assert ctx.submitted == []
    assert tab.saving is False
    assert state.export is None
    assert "would both be called" in ctx.toasts[-1][0]


def test_frame_is_legal_in_a_png_sequence_template():
    """The positive control for the refusal above: ``{frame}`` is not a bad key,
    it is a key this *export* has no value for. A PNG sequence has one per
    file, so the same template is correct there -- which is why the refusal
    tests the export's shape rather than the template's spelling."""
    assert (
        sheetout.filename_for(
            sheetout.DEFAULT_FRAME_TEMPLATE, title="walk", frame=7
        )
        == "walk_0007"
    )
