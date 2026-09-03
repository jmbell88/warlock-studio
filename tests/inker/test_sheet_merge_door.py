"""The Merge re-render door, end to end: press, task, adopt.

``tests/inker/test_sheet_merge_ops.py`` pins what ``merge_render`` is allowed
to do to a cel; nothing pinned the wiring around it, and the wiring is where it
was broken. The op handed a completion callback to ``ctx.submit`` as
``on_done=``, but the runner forwards its keyword arguments to the *task
function* (``tasks.submit``), so the load raised ``TypeError`` inside the pool
and no merge ever landed. The runner below has the real signature and forwards
the same way, which is the whole point: a door that invents a runner keyword
fails here rather than in front of a user.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from warlock.studio import inker_mode, inker_ops, inker_sheet
from warlock.studio.inker.sheetin import document_from_sheet
from warlock.studio.inker_state import InkerDoc, InkerState
from warlock.studio.state import AppState

CELL = 8
WORKER = "warlock-task-test"


def _cell(value: int) -> np.ndarray:
    out = np.zeros((CELL, CELL, 4), dtype=np.uint8)
    out[..., 3] = 255
    out[2:6, 2:6, 0] = value
    return out


def _atlas(values) -> np.ndarray:
    atlas = np.zeros((CELL, len(values) * CELL, 4), dtype=np.uint8)
    for i, value in enumerate(values):
        atlas[:, i * CELL : (i + 1) * CELL] = _cell(value)
    return atlas


def _doc(values=(10, 20, 30, 40), sheet="S1"):
    cells = [{"x": i * CELL, "y": 0, "w": CELL, "h": CELL} for i in range(len(values))]
    anim = {
        "tags": [{"name": "walk_front", "start": 0, "end": len(values) - 1, "loop": True}],
        "frames": [],
    }
    doc = document_from_sheet(
        _atlas(values), cells, anim, source={"job": "J", "sheet": sheet}
    )
    doc.history.clear()
    return doc


def _tab(doc=None, title="walk.ora"):
    doc = _doc() if doc is None else doc
    return InkerDoc(doc=doc, title=title, saved_head=doc.history.head)


def _cel(tab, frame: int):
    anim = tab.doc.anim
    return anim.cels[(anim.tracks[0].uid, anim.frames[frame].uid)]


@dataclass
class _Done:
    key: str
    result: Any = None
    tag: Any = None
    error: Any = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


class _Ctx:
    """``submit`` with the runner's real signature, run on a worker thread.

    ``fn(*args, **kwargs)`` is what ``TaskRunner.submit`` does, so a keyword
    the task function does not take is a ``TypeError`` here too -- and the
    thread name is how the test can say the decode did not run on the caller.
    """

    def __init__(self, state: InkerState) -> None:
        self.state = AppState()
        self.state.inker = state
        self.settings = _Settings()
        self.svc = object()
        self.submitted: list[str] = []
        self.dones: list[_Done] = []
        self.toasts: list[tuple[str, str]] = []
        self.threads: list[str] = []
        self.accept = True

    def submit(
        self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any
    ) -> bool:
        self.submitted.append(key)
        if not self.accept:
            return False
        box: dict[str, Any] = {}

        def go() -> None:
            self.threads.append(threading.current_thread().name)
            try:
                box["result"] = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
                box["error"] = exc

        worker = threading.Thread(target=go, name=WORKER)
        worker.start()
        worker.join()
        if "error" in box:
            raise box["error"]
        self.dones.append(_Done(key=key, result=box["result"]))
        return True

    def toast(self, message: str, level: str = "info", action: Any = None) -> None:
        self.toasts.append((message, level))

    def deliver(self) -> None:
        """What ``App._on_task_done`` does for an ``inker-`` key."""
        for done in self.dones:
            inker_mode.on_task_done(self, done)
        self.dones.clear()


class _Settings:
    def __init__(self) -> None:
        self.store: dict = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value) -> None:
        self.store[key] = value


@pytest.fixture
def wired(monkeypatch):
    """One sheet tab, and a re-render of it waiting on the service."""
    tab = _tab()
    state = InkerState()
    state.add(tab)
    ctx = _Ctx(state)
    monkeypatch.setattr(inker_mode, "newest_sheet_after", lambda *a: "S2")
    monkeypatch.setattr(
        inker_mode,
        "load_sheet_cells",
        lambda svc, job, sheet: [_cell(v) for v in (99, 20, 30, 40)],
    )
    return ctx, state, tab


def _press(ctx) -> bool:
    return inker_ops.run(ctx, inker_ops.get("sheet_merge"))


# -- the wiring ----------------------------------------------------------------


def test_the_press_submits_a_task_the_runner_can_actually_call(wired):
    """The regression. ``submit`` forwards its keywords to the task function,
    so a door that passes a completion callback as one never runs at all."""
    ctx, _state, _tab = wired

    assert _press(ctx) is True
    assert ctx.submitted and ctx.submitted[0].startswith("inker-merge:")
    assert ctx.dones[0].result["sheet"] == "S2"


def test_the_atlas_is_decoded_off_the_frame_thread(wired):
    ctx, _state, _tab = wired
    _press(ctx)
    assert ctx.threads == [WORKER]


def test_nothing_is_written_until_the_task_lands(wired):
    ctx, _state, tab = wired
    _press(ctx)
    assert _cel(tab, 0).pixels[3, 3, 0] == 10, "the press alone must not write"

    ctx.deliver()
    assert _cel(tab, 0).pixels[3, 3, 0] == 99


def test_landing_advances_the_recorded_sheet_id(wired):
    """On the base the merge left behind -- ``merge_render`` replaces
    ``sheet_base`` wholesale, so writing to the one the door read updates
    nothing and the same re-render is offered again forever."""
    ctx, _state, tab = wired
    _press(ctx)
    ctx.deliver()

    assert tab.doc.sheet_base.source["sheet"] == "S2"
    assert tab.doc.sheet_base.source["job"] == "J"


def test_undo_takes_the_sheet_id_back_with_the_pixels(wired):
    ctx, _state, tab = wired
    _press(ctx)
    ctx.deliver()
    tab.doc.undo()

    assert _cel(tab, 0).pixels[3, 3, 0] == 10
    assert tab.doc.sheet_base.source["sheet"] == "S1"


def test_the_landing_finds_its_own_tab_after_the_user_switches(wired):
    """The decode is unbounded and the tabs are not modal: ``inker-index``'s
    rule, which is why the key carries the uid."""
    ctx, state, tab = wired
    _press(ctx)
    other = _tab(_doc((1, 2, 3, 4), sheet="OTHER"), title="other.ora")
    state.add(other)
    assert state.active is other

    ctx.deliver()
    assert _cel(tab, 0).pixels[3, 3, 0] == 99
    assert other.doc.sheet_base.source["sheet"] == "OTHER"


def test_a_busy_document_is_told_rather_than_written_to(wired):
    """A save is walking the layer stack; ``merge_render`` pushes into it."""
    ctx, _state, tab = wired
    _press(ctx)
    tab.saving = True

    ctx.deliver()
    assert _cel(tab, 0).pixels[3, 3, 0] == 10
    assert any("busy" in message for message, _ in ctx.toasts)


def test_a_closed_tab_lands_nothing_and_does_not_raise(wired):
    ctx, state, tab = wired
    _press(ctx)
    state.close(tab.uid)

    ctx.deliver()
    assert _cel(tab, 0).pixels[3, 3, 0] == 10


def test_a_second_press_while_one_is_loading_says_so(wired):
    ctx, _state, _tab = wired
    _press(ctx)
    ctx.accept = False

    assert _press(ctx) is False
    assert any("already loading" in message for message, _ in ctx.toasts)


# -- the refusals the door owns ------------------------------------------------


def test_a_document_with_no_newer_sheet_says_so_and_submits_nothing(wired, monkeypatch):
    ctx, _state, _tab = wired
    monkeypatch.setattr(inker_mode, "newest_sheet_after", lambda *a: "")

    assert _press(ctx) is False
    assert ctx.submitted == []
    assert any("No newer sheet" in message for message, _ in ctx.toasts)


def test_a_sheet_that_does_not_record_its_job_is_refused_by_name(wired):
    ctx, _state, tab = wired
    tab.doc.sheet_base.source.clear()

    assert _press(ctx) is False
    assert ctx.submitted == []
    assert any("which job" in message for message, _ in ctx.toasts)


def test_land_merge_ignores_a_result_that_is_not_a_payload(wired):
    ctx, state, tab = wired
    assert inker_sheet.land_merge(ctx, state, _Done(key="inker-merge:x")) is False
    assert _cel(tab, 0).pixels[3, 3, 0] == 10
