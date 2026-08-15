"""Exactly one "Open in Inker" is on screen at a time, and which one it is.

The decision this pins shipped, so it is stated here rather than cited to a
roadmap section: the duplicate is resolved by *ownership*, not by removing
either affordance. Both shipped in 9e and both act on
``ctx.job()`` -- the *same* asset -- so over the reference there were two buttons
for one action on one object, one directly above the image and one in the
sidebar.

So: **in a mode with a viewport toolbar over the reference, the
toolbar owns it; everywhere else the inspector does** -- ``overlay.offers_inker``
is the toolbar's gate and ``inspector.offers_inker`` is written as its
*complement*, so exactly one is on screen in every mode. The toolbar's button sits
against the pixels it edits, which is where the intent forms; the inspector's
covers the cases with no such toolbar -- a reference selected at the Mesh stage,
where ``overlay.toolbar`` deliberately hides the button because the thing on
screen is a mesh.

This is not the F10 resolution. Keeping two readouts was defensible there
because they said different amounts about different things (an always-on summary
and a detailed panel). Two identical buttons invoking one function on one job
say nothing different, so the argument does not transfer.
"""

from __future__ import annotations

from typing import Any

from warlock.studio.panes import inspector
from warlock.studio.state import AppState


class FakeCtx:
    def __init__(self, svc: Any, mode: str, stage: str = "reference") -> None:
        self.svc = svc
        self.state = AppState()
        self.state.mode = mode
        self.state.create_stage = stage
        self.state.selected = None

    def job(self) -> Any:
        return None


def _reference(svc):
    job_id = svc.store.create("text", "a chest", {}, stage="reference", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"png-not-really")
    job = svc.store.get(job_id)
    job["files"] = ["input.png"]
    return job


def test_the_viewport_toolbar_owns_the_inker_button_at_the_reference_stage(svc):
    """Where the reference is the thing on screen."""
    from warlock.studio import inker_mode

    ctx = FakeCtx(svc, "create", "reference")
    job = _reference(svc)

    assert inker_mode.can_edit_job(ctx, job) is True
    assert inspector.offers_inker(ctx, job) is False


def test_the_inspector_offers_it_where_no_toolbar_does(svc):
    """The Mesh stage's toolbar hides the button on purpose -- the thing on
    screen is a mesh, and the camera controls beside it do not apply to a
    picture. So a reference selected there has no other way in."""
    ctx = FakeCtx(svc, "create", "mesh")
    job = _reference(svc)

    assert inspector.offers_inker(ctx, job) is True


def test_neither_offers_it_for_something_that_cannot_be_edited(svc):
    ctx = FakeCtx(svc, "create", "mesh")
    mesh = svc.store.create("image", "a chest", {}, stage="model", status="done")
    job = svc.store.get(mesh)
    job["files"] = ["model.glb"]

    assert inspector.offers_inker(ctx, job) is False


def test_the_toolbar_and_the_inspector_agree_about_which_of_them_it_is(svc):
    """The two conditions are complementary rather than merely different: for
    every mode, *exactly* one of them offers an editable reference. The
    inspector's gate is written as the complement of the toolbar's rather than
    as a second reading of the mode, which is what makes both halves true."""
    from warlock.studio import create_stages, modes
    from warlock.studio.panes import overlay

    job = _reference(svc)
    for mode in modes.KEYS:
        for stage in create_stages.STAGES:
            ctx = FakeCtx(svc, mode, stage)
            offers = [overlay.offers_inker(ctx, job), inspector.offers_inker(ctx, job)]
            assert sum(offers) == 1, (mode, stage)
