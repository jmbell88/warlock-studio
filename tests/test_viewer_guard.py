"""The unsaved-pose question, once, over two viewers.

``poser_mode.guard`` and ``pose_panel.guard`` were two byte-identical copies of
one rule differing only in which viewer they read and what they call the work.
This is :func:`docmodes.viewer_guard` and the two delegating callers: what is
pinned is that the wording stays one sentence, and that each caller still reads
its *own* viewer -- which is the entire reason the two sessions can be open at
once without either discarding the other.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from warlock.studio import docmodes, poser_mode
from warlock.studio.panes import pose_panel


class _Confirms:
    def __init__(self) -> None:
        self.asked: list[Any] = []

    def ask(self, confirm: Any) -> None:
        self.asked.append(confirm)


def _ctx(*, viewer=None, poser_viewer=None) -> SimpleNamespace:
    return SimpleNamespace(viewer=viewer, poser_viewer=poser_viewer, confirms=_Confirms())


def _viewer(*, pose_mode=True, dirty=True, mode="pose") -> SimpleNamespace:
    return SimpleNamespace(
        pose_mode=pose_mode,
        editor=SimpleNamespace(mode=mode, has_unsaved_edits=lambda: dirty),
    )


# --- the shared rule ----------------------------------------------------------


def test_no_viewer_proceeds_without_asking():
    ctx = _ctx()
    ran: list[str] = []
    assert docmodes.viewer_guard(ctx, None, "pose", "quit", lambda: ran.append("go")) is True
    assert ran == ["go"] and ctx.confirms.asked == []


def test_a_viewer_that_is_not_in_pose_mode_proceeds():
    ctx = _ctx()
    viewer = _viewer(pose_mode=False)
    ran: list[str] = []
    assert docmodes.viewer_guard(ctx, viewer, "pose", "quit", lambda: ran.append("go")) is True
    assert ran == ["go"] and ctx.confirms.asked == []


def test_a_clean_editor_proceeds():
    ctx = _ctx()
    ran: list[str] = []
    assert (
        docmodes.viewer_guard(ctx, _viewer(dirty=False), "pose", "quit", lambda: ran.append("go"))
        is True
    )
    assert ran == ["go"] and ctx.confirms.asked == []


def test_a_dirty_editor_asks_once_and_moves_nothing_until_answered():
    ctx = _ctx()
    ran: list[str] = []
    assert (
        docmodes.viewer_guard(ctx, _viewer(), "pose", "mirror the pose", lambda: ran.append("go"))
        is False
    )
    assert ran == []
    assert len(ctx.confirms.asked) == 1
    confirm = ctx.confirms.asked[0]
    assert confirm.title == "Discard unsaved changes?"
    assert confirm.message == "Unsaved pose changes will be lost if you mirror the pose."

    confirm.on_confirm()
    assert ran == ["go"]


def test_the_noun_is_the_callers_and_reaches_the_sentence():
    ctx = _ctx()
    docmodes.viewer_guard(ctx, _viewer(), "joint", "leave edit mode", lambda: None)
    assert ctx.confirms.asked[0].message == (
        "Unsaved joint changes will be lost if you leave edit mode."
    )


# --- the two callers ----------------------------------------------------------


def test_the_inspector_guard_names_a_joint_in_joints_mode():
    """Joints mode is an inspector-only state, and a joint correction is a
    different thing to lose than a pose -- so the noun is computed at the
    caller rather than in the shared helper."""
    ctx = _ctx(viewer=_viewer(mode="joints"))
    assert pose_panel.guard(ctx, "leave edit mode", lambda: None) is False
    assert "Unsaved joint changes" in ctx.confirms.asked[0].message

    ctx = _ctx(viewer=_viewer(mode="pose"))
    pose_panel.guard(ctx, "leave edit mode", lambda: None)
    assert "Unsaved pose changes" in ctx.confirms.asked[0].message


def test_each_guard_reads_its_own_viewer_and_only_its_own():
    """The mutual exclusion, from both sides. A dirty Poser session must not
    make the inspector ask, and a dirty inspector session must not make the
    Poser ask -- no edit can live in both, and one question per act is the
    whole doctrine."""
    ctx = _ctx(viewer=_viewer(), poser_viewer=None)
    ran: list[str] = []
    assert poser_mode.guard(ctx, "quit", lambda: ran.append("poser")) is True
    assert pose_panel.guard(ctx, "quit", lambda: ran.append("pose")) is False
    assert ran == ["poser"]

    ctx = _ctx(viewer=None, poser_viewer=_viewer())
    ran = []
    assert pose_panel.guard(ctx, "quit", lambda: ran.append("pose")) is True
    assert poser_mode.guard(ctx, "quit", lambda: ran.append("poser")) is False
    assert ran == ["pose"]


def test_both_panes_delegate_rather_than_keeping_a_copy():
    """The docmodes-pin idiom: a fourth copy that happens to agree today is
    what this replaced."""
    import inspect

    for source in (
        inspect.getsource(poser_mode.guard),
        inspect.getsource(pose_panel.guard),
    ):
        assert "viewer_guard" in source
        assert "dialogs.Confirm" not in source
