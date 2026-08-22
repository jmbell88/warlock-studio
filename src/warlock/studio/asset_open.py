"""Where a job row opens, and the one function that goes there.

**A row is not always an asset.** ``rig``, ``sheet``, ``pixel_sheet``,
``sprite_synthesis``, ``charsheet`` and ``retexture`` are *products of another
asset*: every one of them is minted with ``params["source_job"]`` and writes
its artifacts into that job's directory, never into its own -- which is the
same fact ``service._jobs_lifecycle.dependent_jobs`` is built on ("a rig or a
sheet is a separate job row whose artifacts land beside the ``model.glb`` they
were made from"), read from the other side. Their own job directory is never
created, none of what they write is in ``service.files.LISTED`` so
``job["files"]`` is empty, and ``db.Store.create`` gives them ``stage='model'``
because that is the column default and nothing in ``src/`` ever calls
``set_stage``.

Routing such a row by its *stage* therefore lands on Create's Mesh stage
holding a row with no mesh, no files, no thumbnail and a downloads grid of
eight disabled buttons -- the blank screen a finished sprite sheet's "Show"
toast opened onto, while the sheet itself sat one directory over. Routing it by
its *kind* lands on the asset that actually holds the artifact, at the surface
that draws it.

**One copy of the rule.** ``create_stages.stage_for`` exists because there were
four copies of a mode ternary; this exists because there were then four copies
of ``go(ctx, stage_for(job), select=job["id"])``, and all four were wrong about
a follow-up in the same way. ``stage_for`` answers "which stage shows this
asset" and is still the right question for an asset; this answers "where does
this row open", which is a different question the moment a row can be a product
of something else.

Kept out of ``create_stages`` deliberately: that module is *Create's* stages,
and a ``charsheet`` opens in **Troupe**, which is not a stage. An edge from
there to ``troupe_mode`` would quietly make Create's stage module the
navigation layer for the whole app under a name that says otherwise.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class Route(NamedTuple):
    """Where one row opens. Pure data, so :func:`route` needs no ``ctx``."""

    #: ``"create"`` or ``"troupe"``.
    mode: str
    #: A key of ``create_stages.STAGES``; ``""`` when the mode is not Create.
    stage: str
    #: The row to *select* -- the source job, for a follow-up.
    job_id: str
    #: Which sheet or draft inside that surface, or ``""``.
    detail: str
    #: An inspector section to force open on arrival, or ``""``.
    section: str


#: The section keys :func:`open_asset` asks ``widgets.request_open`` for. Named
#: here rather than at the two headers because a ``request_open`` for a key no
#: header registers matches nothing and merely accumulates for the life of the
#: process -- a bug this codebase has already shipped once (see the comment in
#: ``main`` about the 2D pane's Reference block). One spelling, in the module
#: that asks.
SPRITES_SECTION = "inspector/sprites"
SHEET_SECTION = "inspector/sheet"

#: Which Create stage each follow-up's *product* is drawn at. The surface, not
#: the row: a rig lands as ``rig.glb`` beside the mesh, a sheet and a pixel
#: sheet are drawn by ``panes.sheet_panel`` which ``inspector._STAGE_SECTIONS``
#: hangs off Pose, a sprite draft by ``panes.sprite_panel`` which hangs off
#: Reference, and a retexture rewrites the mesh's own skin.
#:
#: ``charsheet`` is deliberately absent: it opens in Troupe, not in Create.
FOLLOWUP_STAGES: dict[str, str] = {
    "rig": "rig",
    "sheet": "pose",
    "pixel_sheet": "pose",
    "sprite_synthesis": "reference",
    "retexture": "mesh",
}

#: Which section holds each follow-up's product, where one has to be opened.
#: Both headers default shut, and arriving from a "Show" toast at a collapsed
#: section is the blank screen again one level down: the user is told the thing
#: is ready and the pane holding it is closed.
FOLLOWUP_SECTIONS: dict[str, str] = {
    "sprite_synthesis": SPRITES_SECTION,
    "sheet": SHEET_SECTION,
    "pixel_sheet": SHEET_SECTION,
}

#: Which params key names the individual artifact inside the surface.
FOLLOWUP_DETAIL_KEYS: dict[str, str] = {
    "sprite_synthesis": "draft_id",
    "sheet": "sheet_id",
    "pixel_sheet": "sheet_id",
    "charsheet": "sheet_id",
}


def route(job: Any) -> Route:
    """Where a "Show" or an "Open" on ``job`` lands.

    Pure: no ``ctx``, no imgui, no I/O, so the rule can be tested as a table
    rather than by driving a window.
    """
    from . import create_stages

    if not isinstance(job, dict):
        return Route("create", "reference", "", "", "")
    params = job.get("params") or {}
    kind = str(job.get("kind") or "")
    source = str(params.get("source_job") or "")
    detail = str(params.get(FOLLOWUP_DETAIL_KEYS.get(kind, "")) or "")

    # A follow-up whose ``source_job`` is missing -- an old row, or a params
    # blob edited by hand -- falls through to the ordinary arm rather than
    # routing nowhere. The blank stage is still better than a click that does
    # nothing, and no door any current version ships can mint one.
    if source:
        if kind == "charsheet":
            return Route("troupe", "", source, detail, "")
        stage = FOLLOWUP_STAGES.get(kind)
        if stage is not None:
            return Route("create", stage, source, detail, FOLLOWUP_SECTIONS.get(kind, ""))

    return Route("create", create_stages.stage_for(job), str(job.get("id") or ""), "", "")


def open_asset(ctx: Any, job_or_id: Any) -> None:
    """Open a row wherever its artifacts actually are. **The one door.**

    Takes a row or an id, because the toast has an id and the panes have the
    row, and resolving that here is what keeps the call sites down to one line
    each.
    """
    # Lazily: this module is imported by panes and by main, while ``widgets``
    # pulls in imgui at module scope and this file must stay importable by a
    # test with no GL context.
    from . import create_stages, troupe_mode, widgets
    from .panes import library

    job = job_or_id if isinstance(job_or_id, dict) else ctx.cache.get(job_or_id)
    if job is None:
        # The row has fallen off the loaded page. Selecting by id still works
        # and is exactly what the toast used to do; there are no params to
        # route without, so this is the honest floor rather than a guess.
        library.select(ctx, str(job_or_id))
        return

    target = route(job)
    if target.mode == "troupe":
        if troupe_mode.open_sheet(ctx, target.job_id, target.detail):
            return
        # Only reachable if the sidecar has gone -- it is written last, as the
        # completion marker -- but Troupe would otherwise draw "No character on
        # screen", which is the blank arrival this module exists to stop.
        ctx.toast("That character sheet is not on disk any more.", "error")
        create_stages.go(ctx, "pose", select=target.job_id)
        return

    create_stages.go(ctx, target.stage, select=target.job_id)
    if target.section:
        widgets.request_open(target.section)
    if target.section == SPRITES_SECTION and target.detail:
        # Which of several drafts is the one that just landed.
        # ``rigging.list_sprite_drafts`` is documented oldest-first, so the new
        # one is at the *bottom* of a list the user did not watch grow.
        ctx.state.preview["sprite_focus"] = target.detail
