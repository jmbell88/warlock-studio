"""The twelve "do these first" findings of the 2026-09-02 review, pinned.

Each test here is the regression for one entry of that review's section 0 --
the ones that lost data, crashed the session or made a documented feature
untrue. The entries themselves were struck from the findings file as they were
fixed, per the repository's rule that a built thing is deleted rather than
ticked; what stays is the test that keeps it fixed.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from warlock.studio import inker
from warlock.studio.undo import CompoundEdit, Edit, UndoStack

SRC = Path(__file__).resolve().parents[1] / "src" / "warlock" / "studio"


# --- 1. set_mode takes the AppState, never the context -----------------------


def test_no_caller_hands_set_mode_the_context():
    """``state.set_mode(state, key)`` reads ``state.mode``; handed the context
    it raised ``AttributeError`` outside every guard. Two callers did."""
    offenders = [
        path.relative_to(SRC)
        for path in SRC.rglob("*.py")
        if "set_mode(ctx, " in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


# --- 2. a Tiled export never writes onto the source image --------------------


def test_image_layers_export_under_minted_names_only():
    """A drive-absolute source read as "relative" through ``PurePosixPath``,
    and a relative one resolved beside the map: both were the user's original
    file, and the export overwrote it with PNG bytes."""
    from warlock.studio.plotter import tmx
    from warlock.studio.plotter.tilemap import MapDoc

    doc = MapDoc(4, 4, 16, 16)
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    doc.add_image_layer("sky", pixels=pixels, source="D:/pics/bg.jpg")
    doc.add_image_layer("unc", pixels=pixels, source="//server/share/bg.png")
    doc.add_image_layer("ground", pixels=pixels, source="bg.png")

    files = tmx.tmx_export(doc)
    images = [name for name in files if name != "map.tmx"]
    assert sorted(images) == ["bg.png", "images/00-sky.png", "images/01-unc.png"]
    assert "D:/pics" not in files["map.tmx"].decode("utf-8")


# --- 3. range ops honour the lock --------------------------------------------


def _clip(frames: int, tracks: int):
    """A clip with a painted cel on every slot -- ``test_timeline_ranges``'s."""
    doc = inker.Document.blank(4, 4)
    for _ in range(tracks - 1):
        doc.add_layer()
    weight = np.ones((2, 2), dtype=np.float32)
    for index in range(frames):
        if index:
            doc.add_frame()
        for track in range(tracks):
            doc.set_active_layer(track)
            assert doc.write_colour((0, 0, 2, 2), (10 * index + track, 0, 0, 255), weight)
    doc.history.clear()
    return doc


def _cel(doc, track: int, frame: int):
    anim = doc.anim
    return anim.cels.get((anim.tracks[track].uid, anim.frames[frame].uid))


def test_a_locked_track_is_not_painted_through_the_timeline():
    doc = _clip(2, 2)
    doc.set_layer_props(0, locked=True)
    before = [_cel(doc, 0, index).pixels.copy() for index in range(2)]
    green = np.array((0, 255, 0, 255), dtype=np.uint8)

    assert doc.fill_range((0, 255, 0, 255), 0, 1, 0, 1), "the unlocked track is painted"
    for index in range(2):
        assert np.array_equal(_cel(doc, 0, index).pixels, before[index]), "the lock held"
        assert (_cel(doc, 1, index).pixels == green).all()
    assert not doc.flip_range("h", 0, 0, 0, 1), "a range of only locked tracks does nothing"


# --- 4. collapse_since keys on serials, and a gesture defers eviction --------


class _Cheap(Edit):
    def __init__(self, cost: int = 0) -> None:
        self.cost = cost

    def undo(self, doc: Any) -> None:
        pass

    def redo(self, doc: Any) -> None:
        pass


def test_a_gesture_longer_than_the_depth_cap_folds_exactly_its_own_steps():
    """``collapse_since`` sliced by a recorded length while ``_evict`` popped
    from the front, so a gesture that ran past ``UNDO_MAX_DEPTH`` folded the
    wrong steps and evicted the work before it."""
    from warlock.studio.undo import UNDO_MAX_DEPTH

    stack = UndoStack()
    earlier = _Cheap()
    stack.push(earlier)
    mark = stack.mark()
    for _ in range(UNDO_MAX_DEPTH + 10):
        stack.push(_Cheap())
    assert stack.collapse_since(mark) is True
    assert len(stack) == 2
    assert stack._done[0] is earlier, "the step before the gesture survived"
    assert isinstance(stack.top, CompoundEdit)
    assert len(stack.top.edits) == UNDO_MAX_DEPTH + 10


def test_the_byte_budget_waits_for_the_gesture_to_close():
    stack = UndoStack(budget=100, hard=100_000)
    kept = _Cheap(cost=10)
    stack.push(kept)
    mark = stack.mark()
    for _ in range(30):
        stack.push(_Cheap(cost=10))
    assert stack.trimmed == 0, "nothing evicted mid-gesture"
    stack.collapse_since(mark)
    # Closed: the fold is one step of 300 bytes, over budget, but the floor is
    # UNDO_MIN_DEPTH steps and there are only two, so both stay.
    assert len(stack) == 2 and stack._done[0] is kept


def test_a_gesture_that_folds_nothing_still_closes():
    stack = UndoStack()
    mark = stack.mark()
    assert stack.collapse_since(mark) is False
    assert stack._open_gestures == 0


# --- 5. Clay: history and tab chords wait for the drag to end ----------------


def test_ctrl_z_during_a_live_clay_drag_is_swallowed(svc):
    import pygame
    from test_clay_mode import FakeCtx, _FakeDrag, _tab

    from warlock.studio import clay_mode

    ctx = FakeCtx(svc)
    tab = _tab(ctx)
    depth = len(tab.doc.history)
    ctx.clay_view = _FakeDrag()
    for name in ("z", "n", "TAB"):
        event = pygame.event.Event(
            pygame.KEYDOWN, key=getattr(pygame, f"K_{name}"), mod=pygame.KMOD_CTRL
        )
        assert clay_mode.handle_key(ctx, event) is True
    assert len(tab.doc.history) == depth
    assert ctx.state.clay.active is tab, "the tab did not change under the drag"
    assert len(ctx.state.clay.docs) == 1


# --- 6/7. Poser: key navigation is guarded, a save carries a copy ------------


def test_selecting_another_key_asks_before_discarding_an_edit():
    from test_poser_mode import _clip_ctx, _turned

    from warlock.studio import poser_mode

    ctx, state = _clip_ctx()
    editor = ctx.poser_viewer.editor
    editor.apply({"head": _turned()}, dirty=True)
    poser_mode.select_key(ctx, 1)
    assert len(ctx.confirms.asked) == 1
    assert state.key_index == 0, "nothing moved until the user answers"
    assert editor.pose()["head"] == pytest.approx(_turned())
    ctx.confirms.asked[0].on_confirm()
    assert state.key_index == 1


def test_scrubbing_over_an_unsaved_pose_refuses_in_words():
    from test_poser_mode import _clip_ctx, _turned

    from warlock.studio import poser_mode

    ctx, state = _clip_ctx()
    editor = ctx.poser_viewer.editor
    editor.apply({"head": _turned()}, dirty=True)
    poser_mode.scrub(ctx, 2)
    assert state.frame == -1
    assert editor.has_unsaved_edits()
    assert ctx.toasts, "the refusal is said"


def test_edits_made_while_a_clip_save_is_writing_survive_the_landing(monkeypatch):
    from test_poser_mode import _clip_ctx

    from warlock.service import clips as svc_clips
    from warlock.studio import poser_mode

    ctx, state = _clip_ctx()
    pending: dict[str, Any] = {}

    def defer(key, fn, *args, **kwargs):
        pending[key] = (fn, args, kwargs)
        return True

    ctx.submit = defer
    monkeypatch.setattr(
        svc_clips, "save", lambda svc, template, payload: {"template": template, **payload}
    )
    poser_mode.set_segment(ctx, 0, 5)
    poser_mode.save_clips(ctx)
    fn, args, kwargs = pending[poser_mode.CLIPS_SAVE_KEY]
    # The payload is a copy: an edit made now does not reach the writer.
    poser_mode.set_segment(ctx, 1, 7)
    result = fn(*args, **kwargs)
    written = [c for c in result["clips"] if c["name"] == state.clip][0]
    assert written["segments"][1] != 7
    poser_mode.on_task_done(ctx, SimpleNamespace(key=poser_mode.CLIPS_SAVE_KEY, result=result))
    assert state.clips_unsaved is True, "the later edit is still owed a save"
    assert state.open_clip()["segments"][1] == 7


# --- 8. Plotter: Delete/Cut/Copy respect the wand mask -----------------------


def test_delete_and_copy_apply_the_wand_mask():
    from test_plotter_mode import FakeCtx, _tab

    from warlock.studio import plotter_mode
    from warlock.studio.tilegrid import gid

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = ctx.state.plotter
    layer = tab.doc.tile_layers()[0]
    tab.doc.write_region(layer.uid, 0, 0, np.full((2, 2), 1, gid.DTYPE))
    mask = np.zeros((tab.doc.height, tab.doc.width), dtype=bool)
    mask[0, 0] = True
    state.set_selection((0, 0, 1, 1), mask)

    plotter_mode._copy(ctx, state, tab, cut=False)
    assert state.clipboard.tolist() == [[1, 0], [0, 0]]
    plotter_mode._delete(ctx, state, tab)
    assert layer.data[:2, :2].tolist() == [[0, 1], [1, 1]]


def test_select_all_drops_the_wand_mask():
    from test_plotter_mode import FakeCtx, _key, _tab

    from warlock.studio import plotter_mode

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = ctx.state.plotter
    mask = np.zeros((tab.doc.height, tab.doc.width), dtype=bool)
    state.set_selection((0, 0, 0, 0), mask)
    assert plotter_mode.handle_key(ctx, _key("a", ctrl=True)) is True
    assert state.select == (0, 0, tab.doc.width - 1, tab.doc.height - 1)
    assert state.select_mask is None


# --- 9. Inker: a landing honours the alpha lock ------------------------------


def test_a_paste_onto_an_alpha_locked_layer_keeps_its_alpha():
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    pixels[2:4, 2:4] = (255, 0, 0, 255)
    doc = inker.Document.from_pixels(pixels)
    doc.select_all()
    assert doc.copy()
    doc.set_layer_props(0, alpha_lock=True)
    assert doc.paste((4, 4))
    assert doc.commit_floating()
    alpha = doc.stack.active.pixels[..., 3]
    assert int(alpha[6, 6]) == 0, "alpha written through the lock"
    assert int(alpha[2, 2]) == 255


# --- 10. Inker: any export failure clears the lock ---------------------------


def test_pump_export_clears_the_lock_on_any_failure():
    from warlock.studio import inker_mode

    source = inspect.getsource(inker_mode.pump_export)
    assert "except Exception" in source
    assert "except (ValueError, IndexError, KeyError)" not in source


# --- 11. Sirens: a backward Bxx is the loop point ----------------------------


def test_a_backward_jump_ends_the_body_and_loops_rather_than_rendering_forever():
    from warlock.studio.sirens import document as D
    from warlock.studio.sirens import synth

    doc = D.new_song()
    first = doc.patterns[0]
    second = doc.add_pattern(rows=8)
    assert doc.set_order([first.uid, second.uid])
    doc.set_cell(second.uid, 0, 0, D.EFFECT, synth.FX_JUMP)
    doc.set_cell(second.uid, 0, 0, D.PARAM, 0)
    samples, loop = synth.render(doc)
    assert samples.shape[0] < 30 * synth.SAMPLE_RATE
    assert loop is not None and loop[0] == 0


# --- 12. a refused "submit" is reported and mutates nothing ------------------


def test_a_refused_generate_keeps_the_seed_and_the_history(svc):
    from test_ux_silent_refusals import _Ctx

    from warlock.studio.panes import settings_2d
    from warlock.studio.state import default_form_2d

    ctx = _Ctx()
    ctx.svc = svc

    def refuse(key, run, *a, **k):
        ctx.submitted.append(key)
        return False

    ctx.submit = refuse
    form = dict(default_form_2d())
    form["prompt"] = "a teapot"
    form["seed"] = 1234
    form["seed_locked"] = False
    settings_2d.generate(ctx, form)
    assert ctx.submitted == ["submit"], "the refusal is the runner's, not the form's"
    assert form["seed"] == 1234, "the seed rolled for a submit that never happened"
    assert ctx.state.prompts == []
    assert ctx.state.toasts, "the refusal is said"
