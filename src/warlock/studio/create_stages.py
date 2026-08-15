"""Create mode's stages: what they are, which one an asset has reached, which
ones you may travel to, and the one function that travels.

**A stage is a position, not an object.** There is no stage record anywhere --
no column, no params key, no file. A stage is *derived* from the job row in
front of you (plus, later, the rig sidecar), and where the user currently is
is one volatile field on ``AppState`` that is never persisted. That is the
whole design: the pipeline already threads itself through ``parent_id`` /
``rerun_of`` / ``candidate_group`` and the ``DERIVED_PARAMS`` discipline, so a
second representation of "how far along is this" would be a second thing to
keep in step with the first, and the first is the one the corpus is keyed on.

**The stage names are UI names and must never leak into a record.** ``mesh``
is deliberately not ``model``: ``verdicts.STAGES`` spells the reconstruction
stage ``model`` and that string is in the stored corpus, so the two vocabularies
are kept apart on purpose. A test pins that no ``service`` module imports this
one.

The module imports nothing from imgui and draws nothing. The rail that renders
it is :func:`warlock.studio.widgets.stage_rail`; the panes each stage maps to
are wired in :mod:`.main`.
"""

from __future__ import annotations

from typing import Any

from ..service.validation import not_done_message
from . import icons

# The stages, in the order the rail draws them and the order an asset passes
# through them. **Ordered and monotone**: :func:`reached` walks this list and
# stops at the first stage a job has not got to, which is only correct while a
# later stage genuinely implies every earlier one. Every entry so far does --
# there is no mesh without a reference and no rig without a mesh.
#
# The tuple *grows* across REDESIGN.md wave 5's steps (reference/mesh at 5.2b,
# +rig/pose at 5.3, +export at 5.4) and a test pins that it only ever grows:
# shipping a rail with two segments while the rig and pose panels are still
# inspector tabs is honest, and shipping five segments of which three are dead
# is not.
STAGES: tuple[str, ...] = ("reference", "mesh")

# What each segment is called. Sentence case, one word each: the rail is a
# breadcrumb across the top of a settings column and has room for exactly that.
LABELS: dict[str, str] = {
    "reference": "Reference",
    "mesh": "Mesh",
}

# The compact face of each segment, for the rail's all-or-nothing fallback at
# narrow widths. **Moved, not re-picked**: Reference and Mesh take the glyphs
# the 2D and 3D modes wore in the rail, so a user who learned the pictures
# keeps them across the merge.
ICONS: dict[str, str] = {
    "reference": icons.IMAGE,
    "mesh": icons.BOX,
}

# The job-row ``stage`` values the Reference stage is *about*. Not a synonym
# for ``service.files.EDITABLE_STAGES`` even though the pair currently agree:
# that list answers "can Inker open this", which is a question about pixels,
# and this one answers "is this row a picture rather than a mesh".
IMAGE_STAGES = ("reference", "tile")


def _reached_reference(job: Any, rig_meta: Any) -> bool:
    """Every job has a reference behind it, including a mesh: a model job
    carries its own ``input.png`` (the promotion copies it), which is why the
    Reference stage can show a mesh's source without walking ``parent_id``."""
    return job is not None


def _reached_mesh(job: Any, rig_meta: Any) -> bool:
    return _stage_of(job) == "model"


# One predicate per stage, in ``STAGES`` order. A table rather than a chain of
# ifs because growth is then *adding a row* -- the alternative had every step of
# wave 5 editing the same nested conditional, which is how the two spellings of
# a gate drift apart.
_REACHED: dict[str, Any] = {
    "reference": _reached_reference,
    "mesh": _reached_mesh,
}


def _stage_of(job: Any) -> str | None:
    return job.get("stage") if isinstance(job, dict) else None


def reached(job: Any, rig_meta: Any = None) -> str:
    """The furthest stage ``job`` has got to. -> a key of :data:`STAGES`.

    ``rig_meta`` is the selected asset's ``rig.json`` as
    :func:`panes.inspector.rig_meta` returns it (None when there is none). It is
    threaded through every predicate rather than only the ones that read it
    today, so 5.3's rig stage is a row in :data:`_REACHED` and not a signature
    change at four call sites.

    With no job at all this is the first stage: an empty Create mode is a form
    waiting for a prompt, which is exactly what Reference is.
    """
    out = STAGES[0]
    for stage in STAGES:
        if not _REACHED[stage](job, rig_meta):
            break
        out = stage
    return out


def shows(stage: str, job: Any) -> bool:
    """Whether ``stage``'s panes can say something true about ``job``.

    The question :func:`go` asks before it moves the selection. Not the same as
    "has this job reached the stage": Reference shows a *mesh* perfectly well,
    because the mesh job carries the reference it was made from -- so walking
    back down the rail never disturbs what is selected, and only walking
    forward can.
    """
    if job is None:
        return False
    if stage == "reference":
        return "input.png" in (job.get("files") or [])
    if stage == "mesh":
        return _stage_of(job) == "model"
    return False


def available(stage: str, job: Any, ctx: Any = None) -> str | None:
    """Why ``stage`` cannot be entered from ``job``, or None when it can.

    A *reason*, not a bool, because the rail's third segment state is
    blocked-with-reason and a segment that merely vanishes answers the user's
    question by deleting it. The wording is the service's own, verbatim, so the
    tooltip on the disabled segment and the toast from the refusal it is
    predicting are one sentence and not two paraphrases.

    ``ctx`` is unused by the two stages that exist at this step and is in the
    signature because 5.3's rig gate reads ``ctx.rigging_available``. An
    optional parameter rather than a required one so the pure tests can call
    this with a job and nothing else.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {list(STAGES)}")
    if stage == "reference":
        # Never blocked. It is the front of the pipeline and the only stage
        # that is reachable with nothing selected at all -- an empty prompt
        # form is a legitimate place to stand.
        return None
    if stage == "mesh":
        if job is None:
            # Also legitimate: the mesh form takes an uploaded image, so it
            # does not need a selected reference to be worth opening.
            return None
        row_stage = _stage_of(job)
        if row_stage == "tile":
            # ``_jobs_resubmit.promote_to_model``'s own sentence. A tile is
            # seamless texture; there is no subject in it to reconstruct, and
            # the promotion refuses in exactly these words.
            return "a tile has no subject to reconstruct"
        if row_stage == "reference" and job.get("status") != "done":
            return not_done_message("That reference", str(job.get("status") or ""))
        return None
    return None


def go(ctx: Any, stage: str, *, select: str | None = None) -> None:
    """**The one stage switch.** Every path into Create mode comes through here.

    One function rather than a ``state.create_stage = ...`` at each call site,
    for the reason ``state.set_mode`` is one function: the switch has three
    obligations that are easy to honour in four places and forget in the fifth
    -- it may have to move the selection, it has to leave Esc a way back, and
    (from 5.3) every exit from the pose stage has to go through
    ``pose_panel.guard`` or an unsaved pose is discarded by a click on a
    breadcrumb.

    ``select`` names an asset to bring along explicitly (Home's Resume list and
    the library's Open both know which one). Without it the selection is left
    exactly where it is *unless* the target stage could not show it, which is
    the only case where standing still would put a panel and a viewport on
    screen describing different objects.
    """
    from . import state as state_mod
    from .panes import library

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {list(STAGES)}")
    if select is not None:
        library.select(ctx, select)
    elif not shows(stage, ctx.job()):
        target = _along_lineage(ctx, stage)
        if target is not None:
            library.select(ctx, target)
    ctx.state.create_stage = stage
    state_mod.set_mode(ctx.state, _MODE_OF[stage])


# Stage -> the ``AppState.mode`` that draws it. **This table is temporary**:
# wave 5.2b collapses ``2d`` and ``3d`` into one ``create`` mode, at which
# point every value here becomes ``"create"`` and the dispatch moves onto the
# stage itself. It exists at this step so that ``go`` is already the only
# switch before the modes are the thing that changes -- the flip then has one
# table to rewrite instead of a dozen call sites to find.
_MODE_OF: dict[str, str] = {
    "reference": "2d",
    "mesh": "3d",
}


def _along_lineage(ctx: Any, stage: str) -> str | None:
    """The nearest job the target stage *can* show, following the lineage.

    Forward (reference -> mesh) is the promotion edge: the mesh job records the
    reference as its ``parent_id``, so the child is found by scanning the loaded
    page for it. The newest child wins -- a reference that has been promoted
    three times is three meshes, and the last one made is the one the user was
    just looking at.

    Backward needs nothing, because :func:`shows` says Reference can show a
    mesh; the walk is only ever asked for in the forward direction, and returns
    None when there is nothing there rather than inventing a selection.
    """
    job = ctx.job()
    if job is None or stage != "mesh":
        return None
    parent = job["id"]
    best: dict[str, Any] | None = None
    for row in getattr(ctx.cache, "jobs", []) or []:
        if row.get("parent_id") != parent or row.get("stage") != "model":
            continue
        if best is None or str(row.get("created_at") or "") >= str(best.get("created_at") or ""):
            best = row
    return None if best is None else str(best["id"])
