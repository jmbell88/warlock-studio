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
# The tuple *grows* across the UI redesign wave 5's steps (reference/mesh at 5.2b,
# +rig/pose at 5.3, +export at 5.4) and a test pins that it only ever grows:
# shipping a rail with two segments while the rig and pose panels are still
# inspector tabs is honest, and shipping five segments of which three are dead
# is not.
STAGES: tuple[str, ...] = ("reference", "mesh", "rig", "pose", "export")

# What each segment is called. Sentence case, one word each: the rail is a
# breadcrumb across the top of a settings column and has room for exactly that.
LABELS: dict[str, str] = {
    "reference": "Reference",
    "mesh": "Mesh",
    "rig": "Rig",
    "pose": "Pose",
    "export": "Export",
}

# The compact face of each segment, for the rail's all-or-nothing fallback at
# narrow widths. **Moved, not re-picked**: Reference and Mesh take the glyphs
# the 2D and 3D modes wore in the rail, so a user who learned the pictures
# keeps them across the merge.
ICONS: dict[str, str] = {
    "reference": icons.IMAGE,
    "mesh": icons.BOX,
    "rig": icons.BONE,
    # The glyph the Poser wears in the shell rail. The stage is not the Poser
    # -- it is the inspector's editor over the shared viewer -- but a skeleton
    # in a stance is what both are about, and two pictures for one idea is how
    # a user comes to believe they are two.
    "pose": icons.PERSON_STANDING,
    "export": icons.DOWNLOAD,
}

# The job-row ``stage`` values the Reference stage is *about*. Not a synonym
# for ``service.files.EDITABLE_STAGES`` even though the pair currently agree:
# that list answers "can Inker open this", which is a question about pixels,
# and this one answers "is this row a picture rather than a mesh".
# "tilesheet" joins the two for the reason the sentence above gives and no
# other: sixty-four tiles in a grid is a picture. It is deliberately *not* in
# ``service.files.EDITABLE_STAGES`` (a sheet is cut up by Packwright, not
# painted over in Inker) and not in any promote/remesh gate, which is what keeps
# a grid of tiles from being offered two minutes of trellis.
IMAGE_STAGES = ("reference", "tile", "tilesheet")


def _reached_reference(job: Any, rig_meta: Any, poses: Any) -> bool:
    """Every job has a reference behind it, including a mesh: a model job
    carries its own ``input.png`` (the promotion copies it), which is why the
    Reference stage can show a mesh's source without walking ``parent_id``."""
    return job is not None


def _reached_mesh(job: Any, rig_meta: Any, poses: Any) -> bool:
    return _stage_of(job) == "model"


def _reached_rig(job: Any, rig_meta: Any, poses: Any) -> bool:
    """The file, first: ``rig.glb`` is listed on the row, so this costs no
    read at all on the frame the rail is drawn. ``rig.json`` is the fallback
    for a rig whose GLB has not been listed yet -- the two land together, but
    the row is refreshed on a timer and the sidecar is read on demand."""
    return "rig.glb" in ((job or {}).get("files") or []) or rig_meta is not None


def _reached_export(job: Any, rig_meta: Any, poses: Any) -> bool:
    """Whether there is anything to export.

    It used to be **never**, and the reasoning was about a different question:
    ``save_artifact`` copies a file to a directory of the user's choosing and
    the job row records none of it, so "has the user finished exporting" has
    no honest answer and a check drawn on a guess would mean nothing.

    But that is not the question the rail asks. The rail asks *what has this
    asset got*, one segment at a time, and it now draws its labels rather than
    five anonymous icons -- so a final step that can never tick reads as work
    still outstanding on every finished asset there will ever be. Answer the
    rail's question instead: the export grid is stage-keyed and never empty
    (:func:`panes.widgets.artifacts_for`), so this is reached the moment the
    asset is one the grid has entries for. Note that :func:`reached` stops at
    the first unreached stage, which is what keeps this from ticking on a bare
    reference: Export is only asked about once mesh, rig and pose have all
    answered yes.
    """
    from . import widgets

    return job is not None and bool(widgets.artifacts_for(job))


def _reached_pose(job: Any, rig_meta: Any, poses: Any) -> bool:
    """A *saved* pose, not the ability to make one.

    The one predicate that cannot be answered from the row: poses are files
    under ``<job>/poses/`` and no artifact of theirs is in ``files.LISTED``.
    So the evidence is passed in -- ``state.preview["poses"]``, which
    ``_refresh_rig_side_data`` already fills off-thread for the selected job.

    Ticking this off the rig instead would put a check beside Pose the moment
    a rig landed, which says a step is behind you that has not been taken.
    """
    return bool(poses)


# One predicate per stage, in ``STAGES`` order. A table rather than a chain of
# ifs because growth is then *adding a row* -- the alternative had every step of
# wave 5 editing the same nested conditional, which is how the two spellings of
# a gate drift apart.
_REACHED: dict[str, Any] = {
    "reference": _reached_reference,
    "mesh": _reached_mesh,
    "rig": _reached_rig,
    "pose": _reached_pose,
    "export": _reached_export,
}


def _stage_of(job: Any) -> str | None:
    return job.get("stage") if isinstance(job, dict) else None


def reached(job: Any, rig_meta: Any = None, poses: Any = None) -> str | None:
    """The furthest stage ``job`` has got to, or None. -> a key of :data:`STAGES`.

    Three pieces of evidence, and every predicate takes all three whether it
    reads them or not, so a later stage is a *row* in :data:`_REACHED` rather
    than a signature change at four call sites. ``rig_meta`` is the asset's
    ``rig.json`` (:func:`panes.inspector.rig_meta`); ``poses`` is its saved
    pose list, which is the one fact no artifact on the row carries.

    **None with no job**, rather than the first stage. This is the value the
    rail ticks its segments off against, and an empty Create mode had a check
    beside Reference -- claiming a step was finished on a screen where nothing
    had been generated at all. Standing at a stage and having completed it are
    different facts; ``state.create_stage`` is the first one.
    """
    out: str | None = None
    for stage in STAGES:
        if not _REACHED[stage](job, rig_meta, poses):
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
    if stage == "export":
        # Everything exports: a reference has its cutouts and its palette, a
        # tile has the wrapped view, a mesh has the eight mesh artifacts. So
        # the last segment never moves the selection either.
        return True
    # Mesh, Rig and Pose are all *about the model job*: the rig and its poses
    # are written beside the ``model.glb`` they were fitted to, because the rig
    # belongs to the mesh. So one answer for the three, and the rail walks the
    # last three segments without ever moving the selection.
    return _stage_of(job) == "model"


def available(stage: str, job: Any, ctx: Any = None) -> str | None:
    """Why ``stage`` cannot be entered from ``job``, or None when it can.

    A *reason*, not a bool, because the rail's third segment state is
    blocked-with-reason and a segment that merely vanishes answers the user's
    question by deleting it. The wording is the service's own, verbatim, so the
    tooltip on the disabled segment and the toast from the refusal it is
    predicting are one sentence and not two paraphrases.

    ``ctx`` is read only by the two stages that need Blender. Optional rather
    than required so the pure tests can call this with a job and nothing else --
    and a missing ctx is treated as "rigging is available", because refusing a
    stage on the strength of an absent object would be a gate nobody chose.
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
        if row_stage == "tilesheet":
            # ``promote_to_model``'s own sentence for this row, verbatim: that
            # door refuses every stage but "reference" and names them all with
            # the general case. It is deliberately *not* ``rerun_job``'s "a tile
            # sheet has no subject to reconstruct" -- that is the remesh door's
            # wording, and the Mesh segment predicts the promotion.
            #
            # Needed explicitly, because the arms below fall through to None:
            # right for a "model" row -- the Mesh segment is the stage you are
            # standing on -- and wrong for a grid of tiles. A tile sheet
            # reaching Create at all is deliberate (IMAGE_STAGES carries it,
            # because a sheet is a picture); this is where the promise that the
            # promote gates stay shut is kept.
            return "this job is not a reference"
        if row_stage == "reference" and job.get("status") != "done":
            return not_done_message("That reference", str(job.get("status") or ""))
        return None
    if stage in ("rig", "pose"):
        if ctx is not None and not getattr(ctx, "rigging_available", True):
            # Hidden would be worse: without Blender the whole feature is
            # absent, and a user who never sees the segment concludes the app
            # cannot rig at all. ``pose_panel``'s own sentence.
            noun = "Rigging" if stage == "rig" else "Posing"
            return f"{noun} needs Blender, which is not installed."
        if not _reached_mesh(job, None, None) or "model.glb" not in (
            (job or {}).get("files") or []
        ):
            # ``service.rig.create_rig``'s refusal, verbatim.
            return "job has no finished mesh to rig"
        if stage == "pose" and not _reached_rig(job, None, None):
            # ``service.rig.save_joints``'s refusal, verbatim. The Rig segment
            # beside it is open, which is where this sends you.
            return "job is not rigged"
        return None
    if stage == "export":
        if job is None:
            return "Nothing is selected to export."
        if job.get("status") in ("queued", "running"):
            # The downloads grid's own words for the same fact, so the segment
            # and the buttons behind it do not offer two accounts of it.
            return "not finished yet"
        return None
    return None


def stage_for(job: Any) -> str:
    """Which stage opens ``job``. -> a key of :data:`STAGES`.

    The routing every "open this asset" surface applies -- Home's Resume list,
    the library's Open, a card dropped on the window. One function because
    there were four copies of the mode ternary it replaces, and four copies of
    a rule is four places for it to be wrong about a tile.
    """
    return "reference" if _stage_of(job) in IMAGE_STAGES else "mesh"


def in_create(state: Any) -> bool:
    """Whether the app is in Create mode at all.

    The indirection wave 5.2a of the UI redesign existed for: every ``mode == "2d" or
    mode == "3d"`` read in the panes was rewritten to ask this, and 5.2b then
    changed the answer *here* rather than hunting a string through nine files.
    """
    return state.mode == MODE


def at(state: Any, stage: str) -> bool:
    """Whether the user is standing at ``stage`` right now.

    Two questions, and both halves matter: the panes ask this to decide what to
    draw, and a stage is only the thing being drawn while Create is the mode.
    Without the mode half, the inspector in Poser would grow the mesh stage's
    quality section because ``create_stage`` still said ``mesh``.
    """
    return state.mode == MODE and state.create_stage == stage


def go(ctx: Any, stage: str, *, select: str | None = None, follow: bool = True) -> None:
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

    ``follow=False`` turns that walk off, for the two arrivals that are about
    to *make* something: a dropped image and a promotion both carry their own
    source, and jumping the selection onto a mesh this reference already has
    would describe the wrong asset while the form builds another.
    """
    from .panes import pose_panel

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {list(STAGES)}")
    if at(ctx.state, "pose") and stage != "pose":
        # **Every** exit from the pose stage, in one place. A pose lives in the
        # viewer until it is saved, so a click on a breadcrumb is exactly as
        # destructive as closing the window -- and the guard is asked here
        # rather than in the rail because the rail is not the only caller (a
        # library card's Open, Home's Resume and the palette all land through
        # ``go`` too).
        pose_panel.guard(ctx, "leave the pose stage", lambda: _switch(ctx, stage, select, follow))
        return
    _switch(ctx, stage, select, follow)


def _switch(ctx: Any, stage: str, select: str | None, follow: bool) -> None:
    """:func:`go` past the pose guard. Separate so the guard can defer it into
    a confirm's ``on_confirm`` without re-asking itself on the way through."""
    from . import state as state_mod
    from .panes import library, pose_panel

    if at(ctx.state, "pose") and stage != "pose":
        # Leaving the editor as well as the stage. Without this the viewer
        # stays in pose mode, and ``_sync_viewer`` returns early while it is --
        # so the Mesh stage would go on showing ``rig.glb`` for the rest of the
        # session.
        pose_panel.leave(ctx)
    if select is not None:
        library.select(ctx, select)
    elif follow and not shows(stage, _current(ctx)):
        target = _along_lineage(ctx, stage)
        if target is not None:
            library.select(ctx, target)
    ctx.state.create_stage = stage
    state_mod.set_mode(ctx.state, MODE)


# The ``AppState.mode`` every stage is drawn in. One value since wave 5.2b --
# that is the merge. It is a named constant rather than a literal at the four
# places that need it because "which mode is Create" and "which stage am I on"
# are two facts, and the flip was survivable precisely because only one of them
# was ever spelled out.
MODE = "create"


def _along_lineage(ctx: Any, stage: str) -> str | None:
    """The nearest job the target stage *can* show, following the lineage.

    Forward is the promotion edge: the mesh job records the reference as its
    ``parent_id``, so the child is found by scanning the loaded page for it.
    The newest child wins -- a reference that has been promoted three times is
    three meshes, and the last one made is the one the user was just looking
    at. One walk for Mesh, Rig and Pose, because the rig and its poses are
    written beside the ``model.glb``: the model job is what all three need.

    Backward needs nothing, because :func:`shows` says Reference can show a
    mesh; the walk is only ever asked for in the forward direction, and returns
    None when there is nothing there rather than inventing a selection.
    """
    job = _current(ctx)
    if job is None or stage == "reference":
        return None
    parent = job["id"]
    best: dict[str, Any] | None = None
    for row in getattr(ctx.cache, "jobs", []) or []:
        if row.get("parent_id") != parent or row.get("stage") != "model":
            continue
        if best is None or str(row.get("created_at") or "") >= str(best.get("created_at") or ""):
            best = row
    return None if best is None else str(best["id"])


def parent(ctx: Any, job: Any) -> Any:
    """The reference ``job`` was promoted from, or None.

    Read off the loaded page rather than the store: this is asked once per
    frame by a pane, and a job whose parent has scrolled out of the window
    simply has no link -- which is honest, because the link's whole purpose is
    to *go* there, and the library cannot select a row it is not holding.
    """
    parent_id = (job or {}).get("parent_id")
    if not parent_id:
        return None
    getter = getattr(getattr(ctx, "cache", None), "get", None)
    return getter(parent_id) if callable(getter) else None


def promotions(ctx: Any, job: Any) -> list[Any]:
    """The meshes made from ``job``, newest first.

    The other direction of :func:`parent`, and the reason both exist: the
    lineage is already threaded through ``parent_id`` and nothing on screen
    said so, so a reference and the three meshes made from it were four
    unrelated cards in a list sorted by date.
    """
    job_id = (job or {}).get("id")
    if not job_id:
        return []
    rows = [
        row
        for row in getattr(getattr(ctx, "cache", None), "jobs", []) or []
        if row.get("parent_id") == job_id and row.get("stage") == "model"
    ]
    rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    return rows


def _current(ctx: Any) -> Any:
    """The selected job, from a ctx that may not have one.

    The palette and the profile sheet both call :func:`go` with the app ctx,
    and their tests hand it a stub carrying a state and nothing else. A stage
    switch is not the place to require a job cache.
    """
    getter = getattr(ctx, "job", None)
    return getter() if callable(getter) else None
