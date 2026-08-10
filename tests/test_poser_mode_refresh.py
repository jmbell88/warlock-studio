"""Poser's refresh flag: wanted-while-busy is re-armed, refused is not dropped.

The findings_dirty pathology, pinned at the seam it lives on: ``TaskRunner.submit``
refuses a key already in flight and nothing else re-arms it, so a refresh wanted
*while the list task runs* -- a save landing mid-list -- must be a flag that
``pump`` clears only when a submit is accepted. The ctx here scripts ``submit``'s
answers and runs nothing, which is exactly the refusal seam under test.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock.studio import poser_mode
from warlock.studio.viewer.pose import PoseEditor


class ScriptedCtx:
    """Records submits and answers them from a script; never runs the fn."""

    def __init__(self, script=None) -> None:
        self.svc = None
        self.state = SimpleNamespace(poser=None)
        self.submitted: list[str] = []
        # Answers handed out in order; empty means "accept everything".
        self.script: list[bool] = list(script or [])
        self.toasts: list = []
        self.rig_default = "humanoid"
        self.rigging_available = True
        self.poser_viewer = None

    def submit(self, key, fn, *args, **kwargs) -> bool:
        self.submitted.append(key)
        return self.script.pop(0) if self.script else True

    def busy(self, key) -> bool:
        return False

    def toast(self, message, level="info", *args) -> None:
        self.toasts.append((message, level))


class _Viewer:
    """The surface ``save`` touches, over an unbound (and so empty) editor."""

    def __init__(self) -> None:
        self.editor = PoseEditor()
        self.pose_mode = True

    def get_pose(self):
        return self.editor.pose()


def _list_result(template="humanoid"):
    return {"template": template, "poses": [], "presets": []}


def test_a_save_landing_mid_list_still_refreshes_once_the_list_lands():
    """The C4 case: refresh() during flight can submit nothing -- the flag has
    to survive until the key is free, and the landing is the moment it is."""
    ctx = ScriptedCtx()
    state = poser_mode.ensure(ctx)
    state.loading = True  # a list task is in flight

    poser_mode.on_task_done(ctx, SimpleNamespace(key=poser_mode.SAVE_KEY, result={"id": None}))
    assert state.refresh_dirty is True, "the wanted refresh is remembered, not dropped"
    assert poser_mode.LIST_KEY not in ctx.submitted, "and not submitted over the in-flight list"

    poser_mode.on_task_done(
        ctx, SimpleNamespace(key=poser_mode.LIST_KEY, result=_list_result())
    )
    assert ctx.submitted.count(poser_mode.LIST_KEY) == 1, "the landing re-read the library"
    assert state.refresh_dirty is False, "cleared only because the submit was accepted"
    assert state.loading is True


def test_a_refused_list_submit_is_re_armed_by_the_frame_pump():
    ctx = ScriptedCtx(script=[False, True])
    state = poser_mode.ensure(ctx)

    poser_mode.refresh(ctx)
    assert state.loading is False, "a refused submit must not leave the mode inert"
    assert state.refresh_dirty is True, "but the refresh is still wanted"

    poser_mode.pump(ctx)  # the per-frame pump (poser_library.draw) retries
    assert ctx.submitted.count(poser_mode.LIST_KEY) == 2
    assert state.refresh_dirty is False
    assert state.loading is True


def test_a_failed_list_re_pumps_a_refresh_wanted_meanwhile():
    """The failure path frees the key too; a refresh wanted mid-flight must
    not wait for a frame that may never draw."""
    ctx = ScriptedCtx()
    state = poser_mode.ensure(ctx)
    state.loading = True
    state.refresh_dirty = True

    poser_mode.on_task_failed(ctx, SimpleNamespace(key=poser_mode.LIST_KEY, message=""))
    assert ctx.submitted.count(poser_mode.LIST_KEY) == 1
    assert state.refresh_dirty is False
    assert state.loading is True


def test_a_clean_failure_does_not_resubmit_forever():
    """The no-loop half: the flag cleared at submit-accept, so a list that
    simply fails does not re-arm itself."""
    ctx = ScriptedCtx()
    state = poser_mode.ensure(ctx)
    poser_mode.refresh(ctx)  # accepted; flag cleared
    poser_mode.on_task_failed(ctx, SimpleNamespace(key=poser_mode.LIST_KEY, message=""))
    assert ctx.submitted.count(poser_mode.LIST_KEY) == 1
    assert state.refresh_dirty is False


def test_a_refused_save_submit_strands_nothing_and_says_so():
    """Dirty clears only on landing, so a refusal loses no edit -- but it must
    not be silence either, or the press reads as a broken button."""
    ctx = ScriptedCtx(script=[False])
    state = poser_mode.ensure(ctx)
    viewer = ctx.poser_viewer = _Viewer()
    state.poses = [{"id": "0123456789ab", "name": "Crouch"}]
    viewer.editor.current = "0123456789ab"
    viewer.editor.dirty = True

    poser_mode.save(ctx)
    assert ctx.submitted == [poser_mode.SAVE_KEY]
    assert viewer.editor.dirty is True, "the guard still stands in front of the exits"
    assert len(ctx.toasts) == 1

    poser_mode.save(ctx)  # script exhausted: accepted this time, and quietly
    assert ctx.submitted == [poser_mode.SAVE_KEY, poser_mode.SAVE_KEY]
    assert len(ctx.toasts) == 1


def test_a_refused_duplicate_submit_toasts():
    ctx = ScriptedCtx(script=[False])
    poser_mode.ensure(ctx)
    poser_mode.duplicate(ctx, "0123456789ab")
    assert ctx.submitted == [poser_mode.DUPLICATE_KEY]
    assert len(ctx.toasts) == 1
