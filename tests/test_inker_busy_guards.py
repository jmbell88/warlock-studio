"""The gates between a document mid-something and the hands that reach it.

Four seams, one rule: nothing may restructure the layer stack while a save is
encoding it or a gesture is open on it, and a task completing after an
unbounded dialog re-checks the tab it is about to mutate. Plus the recents
rule a linked document earns: its path is a job's *served* file, and offering
it back as a plain file arms Ctrl+S to overwrite the serve in place.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from warlock.studio import inker, inker_mode
from warlock.studio.inker.tiles import blank_strip
from warlock.studio.inker_state import InkerDoc, InkerState


class _Ctx:
    """Runs submits inline and records toasts, over a real ``InkerState``."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(inker=InkerState())
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        self.result = run(*args)
        return True

    def toast(self, text: str, level: str = "info", **_: Any) -> None:
        self.toasts.append((text, level))


def _animated_tab(**kwargs: Any) -> InkerDoc:
    doc = inker.Document.blank(8, 8)
    doc.ensure_animation()
    doc.add_frame()
    doc.set_current_frame(0)  # add_frame leaves the playhead on the new frame
    return InkerDoc(doc=doc, title="a", **kwargs)


# --- playback against the save ------------------------------------------------


def test_stopping_playback_mid_save_keeps_set_current_frame_off_the_stack():
    """``set_current_frame`` rebuilds the layer stack -- the very structure an
    in-flight encode is walking. The backstop: a stop landing while ``saving``
    leaves the document's playhead alone."""
    tab = _animated_tab()
    tab.playing, tab.play_index, tab.saving = True, 1, True
    inker_mode.stop_play(tab)
    assert not tab.playing
    assert tab.doc.anim.current == 0, "the stack was not rebuilt under the encode"

    tab.playing, tab.saving = True, False
    inker_mode.stop_play(tab)
    assert tab.doc.anim.current == 1, "an ordinary stop still settles the playhead"


def test_a_save_stops_playback_and_settles_the_playhead_first(tmp_path: Path):
    """The other half: a save arriving during playback stops it synchronously
    on the frame thread, so the encode reads a settled stack and playback's own
    stop can never land mid-write."""
    ctx = _Ctx()
    tab = _animated_tab(path=tmp_path / "a.ora", file_format="ora")
    ctx.state.inker.add(tab)
    tab.playing, tab.play_index = True, 1

    inker_mode.save(ctx, tab)

    assert not tab.playing
    assert tab.doc.anim.current == 1, "settled before the encode, not during it"
    assert ctx.submitted == [f"inker-save:{tab.uid}"]
    assert (tmp_path / "a.ora").exists()


# --- frame stepping against an open gesture ------------------------------------


def test_step_frame_is_refused_while_a_drag_is_open():
    """A paint drag holds a ``StrokeState`` addressed into the stack it began
    on; ``set_current_frame`` mid-drag rebuilds that stack and the next
    ``stroke_to`` raises out of ``by_uid``."""
    ctx = _Ctx()
    tab = _animated_tab()
    ctx.state.inker.add(tab)

    ctx.state.inker.drag_kind = "paint"
    inker_mode.step_frame(ctx, 1, tab)
    assert tab.doc.anim.current == 0, "refused while the gesture is open"

    ctx.state.inker.clear_drag()
    inker_mode.step_frame(ctx, 1, tab)
    assert tab.doc.anim.current == 1, "and served once it is not"


def test_step_frame_is_refused_while_a_multi_click_gesture_is_open():
    ctx = _Ctx()
    tab = _animated_tab()
    ctx.state.inker.add(tab)
    ctx.state.inker.gesture_pts = [(1.0, 1.0), (2.0, 2.0)]
    inker_mode.step_frame(ctx, 1, tab)
    assert tab.doc.anim.current == 0


# --- task completions re-check the tab -----------------------------------------


def test_a_tileset_import_landing_on_a_busy_tab_is_refused():
    """The picker is unbounded, so the tab it was opened for may be saving (or
    playing) by the time the decode lands -- and ``add_tileset`` pushes a
    history step into the stack the encode is walking."""
    ctx = _Ctx()
    tab = InkerDoc(doc=inker.Document.blank(8, 8), title="a")
    ctx.state.inker.add(tab)
    done = SimpleNamespace(
        key=f"inker-tileset-import:{tab.uid}", result={"tileset": blank_strip(4, 4)}
    )

    tab.saving = True
    inker_mode.on_task_done(ctx, done)
    assert tab.doc.tilesets == []

    tab.saving = False
    inker_mode.on_task_done(ctx, done)
    assert len(tab.doc.tilesets) == 1


# --- indexing clears the brush's slot claim -------------------------------------


def test_indexing_to_a_new_table_releases_the_brush_slot_claim():
    """``set_color_mode`` clears ``fg_slot`` and ``index_to`` did not: the slot
    the brush was claiming indexes a table that has just been replaced."""
    ctx = _Ctx()
    tab = InkerDoc(doc=inker.Document.blank(8, 8), title="a")
    ctx.state.inker.add(tab)
    ctx.state.inker.fg_slot = 3

    assert inker_mode.index_to(ctx, tab, [(0, 0, 0, 255), (255, 255, 255, 255)])
    assert ctx.state.inker.fg_slot is None


# --- a linked document stays out of recents -------------------------------------


class _AdoptCtx(_Ctx):
    def __init__(self) -> None:
        super().__init__()
        self._data: dict = {}

    @property
    def settings(self):
        return self

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value


def test_a_job_linked_adoption_stays_out_of_the_recent_list():
    """A linked tab's path is the job's served ``input.png``. A recents entry
    reopens it as a plain file, and a plain file's Ctrl+S writes PNG bytes over
    the served file in place -- past ``save_edited_image``'s backup and staged
    replace."""
    ctx = _AdoptCtx()
    state = ctx.state.inker
    inker_mode._adopt(
        ctx,
        state,
        inker.Document.blank(8, 8),
        path=Path("C:/jobs/j1/input.png"),
        job_id="j1",
        link_kind="reference-edit",
    )
    assert inker_mode.recent_paths(ctx) == []

    plain = Path("C:/pics/a.png")
    inker_mode._adopt(ctx, state, inker.Document.blank(8, 8), path=plain)
    assert inker_mode.recent_paths(ctx) == [str(plain)]
