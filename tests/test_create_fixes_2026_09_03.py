"""The three Create-mode defects closed on 2026-09-03, each pinned by behaviour.

1. An emptied Avoid box reaches the door as ``""`` -- the explicit "no negative
   prompt" ``guidance.normalize`` already understood -- rather than being
   folded back into the default by an ``or None`` at every door.
2. The form's default is the default text, so the user can see what they are
   deleting (the manual promised this since chapter 22 was written).
3. The mesh seed rerolls after an *accepted* Make 3D unless it is locked --
   the engine is deterministic in its seed, so the old behaviour produced the
   identical mesh on a second press.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock.guidance import DEFAULT_NEGATIVE_PROMPT
from warlock.studio import state as state_mod
from warlock.studio.panes import settings_2d, settings_3d


def test_the_form_starts_with_the_default_negative_prompt_visible():
    form = state_mod.default_form_2d()
    assert form["negative_prompt"] == DEFAULT_NEGATIVE_PROMPT


def test_an_emptied_avoid_box_is_sent_as_an_explicit_empty_string():
    form = state_mod.default_form_2d()
    form["prompt"] = "a chest"
    form["negative_prompt"] = ""
    assert settings_2d.submit_kwargs(form)["negative_prompt"] == ""
    form["negative_prompt"] = "blurry"
    assert settings_2d.submit_kwargs(form)["negative_prompt"] == "blurry"


def test_the_door_keeps_an_explicit_empty_negative_prompt(svc):
    from warlock.service import jobs as svc_jobs

    none = svc_jobs.create_job(
        svc, kind="text", prompt="a chest", output="reference", negative_prompt=""
    )
    assert svc.store.get(none["id"])["params"]["negative_prompt"] == ""
    default = svc_jobs.create_job(svc, kind="text", prompt="a chest", output="reference")
    assert svc.store.get(default["id"])["params"]["negative_prompt"] == DEFAULT_NEGATIVE_PROMPT


def _ctx(accept: bool) -> SimpleNamespace:
    state = SimpleNamespace(
        form_3d=dict(state_mod.DEFAULT_FORM_3D),
        filters=SimpleNamespace(kind="all"),
        clear_field_errors=lambda: None,
    )
    toasts: list[str] = []
    return SimpleNamespace(
        state=state,
        svc=object(),
        submit=lambda *a, **k: accept,
        toast=toasts.append,
        toasts=toasts,
    )


def test_an_accepted_make_3d_rerolls_the_mesh_seed_unless_locked():
    ctx = _ctx(accept=True)
    ctx.state.form_3d["mesh_seed"] = 0
    settings_3d.submit_promotion(ctx, "job", {}, force=False)
    assert ctx.state.form_3d["mesh_seed"] > 0

    ctx.state.form_3d["mesh_seed"] = 1234
    ctx.state.form_3d["mesh_seed_locked"] = True
    settings_3d.submit_promotion(ctx, "job", {}, force=False)
    assert ctx.state.form_3d["mesh_seed"] == 1234


def test_a_refused_make_3d_keeps_the_seed():
    ctx = _ctx(accept=False)
    ctx.state.form_3d["mesh_seed"] = 1234
    settings_3d.submit_promotion(ctx, "job", {}, force=False)
    assert ctx.state.form_3d["mesh_seed"] == 1234
    assert ctx.toasts  # the collision was said, not swallowed


def test_reroll_mesh_seed_respects_the_lock():
    form = {"mesh_seed": 7, "mesh_seed_locked": True}
    settings_3d.reroll_mesh_seed(form)
    assert form["mesh_seed"] == 7
    form["mesh_seed_locked"] = False
    settings_3d.reroll_mesh_seed(form)
    assert form["mesh_seed"] != 7
