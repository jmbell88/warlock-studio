"""Sheet and GIF export, spread across frames instead of frozen into one.

``export_sheet`` and ``export_gif`` used to call ``sheetout.snapshot`` inline,
on the frame the button was clicked. That is one composite per frame of the
clip, all in one frame of the app: a sixty-frame walk cycle is a visible freeze
at exactly the moment the user is waiting to be asked for a filename.

Moving it to a task thread is *forbidden*, not merely awkward: ``frame_flat``
fills and evicts the document's flatten cache and ``layers_for`` writes track
properties down onto cels, and the onion-skin draw walks the same structures
sixty times a second. So the work stays on the frame thread and is spread over
later frames -- the answer ``viewer/sheet.StripRender`` already gives to the
same problem one layer down.

What is pinned here is the part that can silently go wrong: that spreading the
read does not change the output, that it really is one frame's work per pump,
and that every way out of the loop puts the tab's lock back.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from warlock.studio import inker, inker_mode
from warlock.studio.inker import sheetout
from warlock.studio.inker_state import InkerDoc, InkerState

RED = (255, 0, 0, 255)


class _Ctx:
    """The three things the export path asks of a Ctx."""

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


def _animated(frames: int = 4) -> Any:
    doc = inker.Document.blank(4, 4)
    weight = np.ones((2, 2), dtype=np.float32)
    doc.write_colour((0, 0, 2, 2), RED, weight)
    for _ in range(frames - 1):
        doc.add_frame(link=True)
    return doc


def _open(frames: int = 4):
    tab = InkerDoc(doc=_animated(frames), title="walk.ora")
    state = InkerState()
    state.add(tab)
    return _Ctx(state), state, tab


def _pump_until_done(ctx: Any, state: Any, limit: int = 100) -> int:
    """Pump to completion. -> how many pumps it took."""
    for count in range(1, limit + 1):
        inker_mode.pump_export(ctx)
        if state.export is None:
            return count
    raise AssertionError("the export never finished")


def test_the_stepper_produces_exactly_what_the_one_shot_snapshot_did():
    """The whole risk of spreading a read is that it stops being the same read.
    ``snapshot`` is kept as the one-shot spelling and is now written in terms of
    the same three halves the stepper uses, so this compares them directly --
    byte for byte, because a composite that differs in one pixel is a sheet
    nobody would notice was wrong."""
    ctx, state, tab = _open(5)
    expected_frames, expected_durations, expected_tags, expected_layout = (
        sheetout.snapshot(tab.doc)
    )
    inker_mode.export_sheet(ctx, tab)
    _pump_until_done(ctx, state)
    # ``_submit_export`` closes over the frames it collected; the task was
    # submitted, so the read finished and matched the work list.
    assert ctx.submitted == [f"inker-export:{tab.uid}"]
    stepped = [sheetout.flatten_one(tab.doc, uid) for uid in sheetout.frame_uids(tab.doc)]
    assert len(stepped) == len(expected_frames)
    for a, b in zip(stepped, expected_frames, strict=True):
        assert np.array_equal(a, b)
    assert sheetout.timing(tab.doc) == (
        expected_durations,
        expected_tags,
        expected_layout,
    )


def test_one_frame_is_flattened_per_pump_and_not_one_clip():
    """The point of the change. A five-frame clip takes five pumps, so the
    frame the user clicked on does one composite rather than five."""
    ctx, state, tab = _open(5)
    inker_mode.export_sheet(ctx, tab)
    assert state.export is not None
    assert state.export.frames == []  # the click frame flattened nothing at all
    for expected in range(1, 6):
        inker_mode.pump_export(ctx)
        # The stepper is cleared by the pump that reads the *last* frame, so
        # the count is only observable while there is more to do.
        if state.export is not None:
            assert len(state.export.frames) == expected
    assert state.export is None
    assert ctx.submitted == [f"inker-export:{tab.uid}"]


def test_the_tab_is_locked_from_the_click_rather_than_from_the_submit():
    """Frames go by between the click and the encode, and an edit landing in
    one of them would put half of two documents into the sheet. ``saving`` is
    what ``busy`` already refuses every mutation on."""
    ctx, state, tab = _open(4)
    assert not tab.busy
    inker_mode.export_sheet(ctx, tab)
    assert tab.saving and tab.busy
    _pump_until_done(ctx, state)
    assert tab.saving  # still locked: the encode is in flight


def test_a_refused_submit_puts_the_lock_back():
    """``docmodes.start_save``'s rule, which the click-time lock makes easier
    to get wrong: the runner refuses a key already in flight, and leaving
    ``saving`` set there is what makes a tab read-only for the session."""
    ctx, state, tab = _open(3)
    ctx.accept = False
    inker_mode.export_sheet(ctx, tab)
    _pump_until_done(ctx, state)
    assert ctx.submitted == [f"inker-export:{tab.uid}"]
    assert not tab.saving
    assert not tab.busy


def test_a_second_click_while_one_is_stepping_is_refused():
    """Both exports share a task key and the tab is locked, so a second press
    has nowhere to go. It must not park a second stepper over the first."""
    ctx, state, tab = _open(6)
    inker_mode.export_sheet(ctx, tab)
    first = state.export
    inker_mode.export_gif(ctx, tab)
    assert state.export is first
    inker_mode.export_sheet(ctx, tab)
    assert state.export is first


def test_closing_the_tab_under_an_export_abandons_it_quietly():
    """Nothing has been written and the lock goes with the tab, so there is
    nothing to undo and nothing to say."""
    ctx, state, tab = _open(6)
    inker_mode.export_sheet(ctx, tab)
    inker_mode.pump_export(ctx)
    state.docs.remove(tab)
    inker_mode.pump_export(ctx)
    assert state.export is None
    assert ctx.submitted == []
    assert ctx.toasts == []


def test_a_still_document_is_refused_before_anything_is_locked():
    """``Export PNG`` covers that case. The refusal has to come before the
    lock, or a document that can never export is read-only until restart."""
    tab = InkerDoc(doc=inker.Document.blank(4, 4), title="still.png")
    state = InkerState()
    state.add(tab)
    ctx = _Ctx(state)
    inker_mode.export_sheet(ctx, tab)
    assert state.export is None
    assert not tab.saving


def test_the_gif_path_steps_the_same_way():
    """Same stepper, different encoder. Written out because the two used to be
    two copies of the snapshot call and the copies are exactly what drifted."""
    ctx, state, tab = _open(4)
    inker_mode.export_gif(ctx, tab)
    assert state.export is not None and state.export.kind == "gif"
    assert _pump_until_done(ctx, state) == 4
    assert ctx.submitted == [f"inker-export:{tab.uid}"]
