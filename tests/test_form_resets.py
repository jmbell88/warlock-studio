"""Both Create forms can be put back to their defaults, and both ask first.

The 3D pane had no *Reset...* while the 2D pane did, which is the asymmetry
these tests are really about: both forms accumulate overrides across a session
and only one of them offered a way out of that. So the pair is asserted
together -- a reset that exists on one pane and not the other is exactly the
state this file exists to catch coming back.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from warlock.studio import dialogs
from warlock.studio.panes import settings_2d, settings_3d
from warlock.studio.state import DEFAULT_FORM_3D, default_form_2d


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(
            form_2d=default_form_2d(),
            form_3d=dict(DEFAULT_FORM_3D),
            source_job="job-1",
            preview={"anything": 1},
            preview_dirty_at=0.0,
        )
        self.confirms = dialogs.ConfirmQueue()
        self.toasts: list[str] = []

    def toast(self, text: str, level: str = "info", *a, **kw) -> None:
        self.toasts.append(text)


# --- the 3D form ------------------------------------------------------------


def test_resetting_the_model_form_restores_every_default():
    ctx = _Ctx()
    ctx.state.form_3d.update(
        platform="high", size_m=2.5, mesh_seed=1234, candidates=3, rig=True
    )
    settings_3d._reset(ctx)
    assert ctx.state.form_3d == DEFAULT_FORM_3D


def test_resetting_the_model_form_does_not_alias_the_default_dict():
    """``DEFAULT_FORM_3D`` is module state; a form aliased onto it would edit
    the default for the rest of the process."""
    ctx = _Ctx()
    settings_3d._reset(ctx)
    ctx.state.form_3d["platform"] = "poisoned"
    assert DEFAULT_FORM_3D["platform"] == ""


def test_resetting_the_model_form_keeps_the_chosen_source():
    """The source is not part of this form -- it lives on ``state.source_job``
    -- and dropping it would be a larger action than the button offers."""
    ctx = _Ctx()
    settings_3d._reset(ctx)
    assert ctx.state.source_job == "job-1"


def test_resetting_the_model_form_says_so():
    ctx = _Ctx()
    settings_3d._reset(ctx)
    assert ctx.toasts and "default" in ctx.toasts[0]


# --- the 2D form, which is the precedent ------------------------------------


def test_resetting_the_image_form_restores_every_default():
    ctx = _Ctx()
    ctx.state.form_2d.update(prompt="a knight")
    settings_2d._reset(ctx)
    assert ctx.state.form_2d["prompt"] == default_form_2d()["prompt"]


def test_resetting_the_image_form_drops_the_stale_preview():
    """The preview describes the form that produced it."""
    ctx = _Ctx()
    settings_2d._reset(ctx)
    assert ctx.state.preview == {}


# --- both are behind a confirm ----------------------------------------------


def _row_source(fn) -> str:
    return inspect.getsource(fn)


def test_both_resets_are_behind_a_confirm_dialog():
    """Read off the source, because drawing the row needs a live imgui context.

    What is being pinned is the *shape*: a form reset is unrecoverable (the 2D
    seed is rerolled, so even retyping the prompt does not get you back), so
    neither button may act on the click that draws it.
    """
    for source in (_row_source(settings_3d._reset_row), _row_source(settings_2d._reset_row)):
        assert "dialogs.Confirm(" in source
        assert "on_confirm=" in source
        assert "_reset(ctx)" in source
