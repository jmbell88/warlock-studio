"""Submitting a finished job's settings again: rerun, promote, keep.

Split out of ``service/jobs.py``, which had grown to 1,446 lines over five
unrelated subjects; ``jobs.py`` stays as the facade every caller still imports
and calls by attribute.

Every door here mints a **new row** from an old one, which is what makes
``DERIVED_PARAMS`` load-bearing rather than tidy: anything the worker recorded
about the artifacts of the source job is stripped on the way through, or a
reroll ships wearing a verdict, a mesh audit or a re-texture report measured
from a different mesh. The other shared rule is admission -- these are extra
doors onto a mesh job, and each one runs ``check_weights``/``check_vram``
again rather than trusting the source row, because the source was admitted
against whatever was on disk when *it* was submitted and a promotion can be
days later.

``promote_candidates`` calls ``promote_to_model`` directly rather than through
the facade; the binding is early, and no test redirects it.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from typing import Any

from .. import guidance
from . import matte
from ._jobs_create import _normalize_guidance, resolve_profile
from .core import WarlockService
from .errors import Conflict, Invalid
from .validation import (
    ALLOWED_RESOLUTIONS,
    CONDITIONING_PARAMS,
    DERIVED_PARAMS,
    MAX_MESH_CANDIDATES,
    check_job_id,
    check_seed,
    check_vram,
    check_weights,
    not_done_message,
    random_seed,
    valid_template,
)

log = logging.getLogger(__name__)


def rerun_job(
    svc: WarlockService,
    job_id: str,
    *,
    mode: str = "reroll",
    seed: int | None = None,
) -> dict[str, Any]:
    """Resubmit a finished job, either from scratch or from its reference image.

    The two loops this closes:

    * ``reroll`` -- same prompt and guidance, new seed. Generation is
      deterministic in the seed, so pressing Generate twice on an unchanged
      form used to produce the identical mesh; this is the "that's close, give
      me another" button.
    * ``remesh`` -- reuse the existing input.png and rerun only the 3D stage.
      When SDXL drew a good reference but trellis made a poor mesh there was no
      way to retry the second half without paying for the first, including the
      VRAM handoff in exclusive mode.

    Both reduce to creating an ordinary job -- remesh is just an image job
    whose input.png is copied across -- so the worker, the queue and the
    progress model need no special case.
    """
    check_job_id(job_id)
    if mode not in ("reroll", "remesh"):
        raise Invalid("mode must be 'reroll' or 'remesh'", field="mode")
    check_seed("seed", seed)
    source = svc.require_job(job_id)

    if mode == "remesh" and source["kind"] == "tile_sheet":
        # Refused by name, and only in this mode: a reroll of a tile sheet is
        # exactly meaningful -- the whole request is a prompt, a tile size and a
        # seed, and "draw me another" is the reason the seed is stored. What
        # cannot happen is a reconstruction: a remesh is an image job at stage
        # "model", so a sheet taken through here would buy two minutes of
        # trellis turning a grid of tiles into a lumpy plane.
        # ``promote_to_model``'s sentence for a tile, restated for the thing a
        # tile sheet is sixty-four of.
        raise Invalid("a tile sheet has no subject to reconstruct")

    if mode == "remesh" and source["stage"] == "tile":
        # The other door onto a reconstruction, and it has to be shut for the
        # same reason promote_to_model's is: a remesh is an image job at stage
        # "model", so a tile taken through here would buy two minutes of
        # trellis turning a texture into a lumpy plane. Rerolling one is fine
        # -- it stays a tile.
        raise Invalid("a tile has no subject to reconstruct")
    if source["params"].get("built"):
        # Both modes, and before the input.png check below -- which would
        # otherwise refuse this with the reference message, telling the user to
        # go and edit an image that has never existed. There is no generator
        # behind a built asset: a reroll has no seed to change and a remesh has
        # no reference to reconstruct from. The way to get a different mesh is
        # to open the document and change it.
        raise Invalid(
            "this asset was built rather than generated, so there is nothing to "
            "regenerate; open it in Clay to change the mesh"
        )
    kind = "image" if mode == "remesh" else source["kind"]
    src_png = svc.job_dir(job_id) / "input.png"
    if kind == "image" and not src_png.exists():
        raise Invalid("source job has no reference image to reuse")
    if mode == "reroll" and kind == "image" and source["stage"] == "reference":
        # A hand-drawn or imported reference: there is no generator behind it,
        # so a new seed changes nothing about the pixels, and the row this
        # would mint (kind=image, stage=reference) is the one combination
        # create_job itself refuses. The worker's reference-stage early-return
        # lives inside its text branch, so such a row used to fall straight
        # through and pay two minutes of trellis for "give me another image".
        raise Invalid(
            "this reference was made by hand, so there is nothing to reroll; "
            "remesh it to build a mesh from it"
        )

    # Derived values describe the *source* run, not this one: keeping them
    # would make the new job claim a composed prompt it never used and a
    # quality score for a mesh that doesn't exist yet.
    params = {k: v for k, v in source["params"].items() if k not in DERIVED_PARAMS}
    if kind == "text":
        # hand_edited is a statement about input.png, not about the run, which
        # is why it is not in DERIVED_PARAMS: a remesh and a promotion *copy*
        # that file, so the flag is still true for them. A text reroll
        # regenerates it from the prompt, so carrying it would claim a hand
        # edit of pixels nobody has touched.
        params.pop("hand_edited", None)
    if mode == "remesh":
        # A remesh is an image job: SDXL never runs, so a carried-over
        # conditioning selection would describe a run that cannot happen.
        for key in CONDITIONING_PARAMS:
            params.pop(key, None)
    fresh = seed if seed is not None else random_seed()
    params["seed"] = fresh
    if mode == "remesh":
        # The reference is being reused verbatim; only the 3D stage rerolls.
        params["reference_seed"] = source["params"].get(
            "reference_seed", source["params"].get("seed", fresh)
        )
        params["mesh_seed"] = fresh
    else:
        params["reference_seed"] = fresh
        params["mesh_seed"] = fresh
    params["rerun_of"] = job_id
    # A reroll of a reference-stage job is "try another": it must stop at the
    # reference again, not fall through to the default "model" stage and
    # silently pay for a trellis run the user never asked for. remesh always
    # finishes at a mesh, so it keeps the default.
    stage = source["stage"] if mode == "reroll" else "model"

    if kind == "pixel_sheet":
        # ``sheet_id`` sits on DERIVED_PARAMS for the *sheet* render kind,
        # where the worker records it about the atlas it produced. On this
        # kind it is an input the door validated -- which sheet the restyle
        # depicts -- so the strip above must not cost it, or the minted job
        # dispatches straight into "sheet_id is not a sheet id: ''".
        params["sheet_id"] = source["params"].get("sheet_id")
    if kind == "sprite_synthesis":
        # Reroll only: remesh forced ``kind`` to "image" above (and a sprite
        # job has no input.png, so it is refused there anyway).
        from .. import rigging

        # The worker samples from ``seed_a``/``seed_b``; the generic ``seed``
        # above is never read by this kind, so copying the pair verbatim made
        # "give me another" reproduce a byte-identical sheet. Same shape as
        # the door: the first seed honours a pinned request, the second is
        # drawn until distinct.
        params["seed_a"] = fresh
        second = random_seed()
        while second == fresh:
            second = random_seed()
        params["seed_b"] = second
        # A fresh trio, never the source's: ``draft_id`` names the files this
        # job publishes into the *source reference's* directory, and it is
        # minted at the door rather than recorded by the worker -- so it is
        # not in DERIVED_PARAMS and used to be copied verbatim. Carried over,
        # a cancelled reroll made ``_discard_artifacts`` delete the original
        # job's published trio, and a finished one silently overwrote it.
        params["draft_id"] = rigging.new_id()

    # A remesh of a reference is the third door onto a mesh job (create_job
    # and promote_to_model are the other two), and the only one that used to
    # skip admission: the reference was admitted against the SDXL cost alone,
    # and the trellis cost was never checked against the plan.
    #
    # Both halves of admission, in create_job's order. The weights check is not
    # redundant with the original job's: the source row was admitted against
    # whatever was on disk when it was submitted, and a reroll can be days
    # later, against a models directory the user has since pruned or moved. It
    # matters most for ``style_lora``, which fails *silently* at load -- the
    # reroll would finish, look nothing like the run it was rerolling, and
    # write a row claiming a style that never ran. That row is corpus evidence
    # (style_lora is in VECTOR_PARAMS), so the cost of skipping this is not one
    # bad picture but a poisoned neighbourhood of them. Unconditional for every
    # kind, as at create_job's door -- check_weights makes the text-only
    # decision itself.
    check_weights(svc, kind, params)
    check_vram(svc, kind, stage, params)
    if kind == "tile_sheet":
        # The same door ``create_tile_sheet`` holds, held again on the way back
        # in -- the ``retexture`` precedent below, for the reason the long
        # comment above gives about ``style_lora``. ``check_weights`` is
        # text-only, so without this a reroll days later, against a models
        # directory the user has since pruned, would finish and come back a
        # grid of flat photographs while the row claimed a pixel-art LoRA that
        # never loaded. Gated on the *stored* reference rather than the params'
        # adapter, because ``carry_ref`` below is what decides whether the new
        # row will actually have one.
        from .tilesheets import _check_weights as _check_sheet_weights

        _check_sheet_weights(
            svc, with_reference=(svc.job_dir(job_id) / "ref.png").exists()
        )
    if kind == "sprite_synthesis":
        # The same door ``create_sprite_synthesis`` holds, held again on the
        # way back in -- the tile-sheet precedent above, for the same reason:
        # ``check_weights`` is text-only, and both adapters plus the pixel
        # LoRA are mandatory for this kind. A reroll can be days later,
        # against a models directory the user has since pruned; without this
        # it dispatches into a runtime failure instead of a refusal that
        # names the download.
        from .sprites import _check_weights as _check_sprite_weights

        _check_sprite_weights(svc)
    if params.get("sprite_sheet"):
        # The sprite arm's half of the same rule, and it needs saying separately
        # because the job in front of this function is not the one the check is
        # about: ``params["sprite_sheet"]`` is a *request for a follow-up*, and
        # the row that would load these weights is minted by the worker minutes
        # later, in ``_maybe_queue_sprite_sheet``, which cannot refuse anything.
        # ``create_job`` holds this door on the way in for that reason; a reroll
        # is the other way in, days later and against a models directory the
        # user may have pruned since. Without it the character redraws, the
        # sheet is queued behind it, and the pair fails at dispatch -- which
        # reads as a bug rather than as a download nobody has done.
        from ._jobs_create import _check_sprite_sheet

        _check_sprite_sheet(svc, params["sprite_sheet"])
    if kind == "retexture":
        # The same door ``retexture_job`` holds, held again on the way back in.
        # A stored row's ``base_model`` outlives the door that admitted it -- a
        # row written before the family check existed, or one whose checkpoint
        # the registry has since re-declared -- and a reroll that skipped this
        # would spend the six Blender views and a ~16 GiB load to reach a
        # runtime refusal, exactly as the original did (MDL-15).
        from .. import models
        from ._jobs_rework import _check_retexture_family

        # ``.get``, not ``[]``: a stored row may name a checkpoint the registry
        # no longer carries, which is the tolerance ``Worker._resolve_base_key``
        # owns for every t2i stage. An unknown key is that path's problem, not a
        # family refusal's.
        stored = models.BASE_MODELS.get(str(params.get("base_model") or ""))
        if stored is not None:
            _check_retexture_family(stored)

    new_id = uuid.uuid4().hex[:12]
    new_dir = None
    # A reroll reruns SDXL, so its conditioning reference has to come with it
    # -- the first time a *text* rerun needs a directory before the row, which
    # is why the except below now covers a case it never did.
    src_ref = svc.job_dir(job_id) / "ref.png"
    carry_ref = mode == "reroll" and src_ref.exists()
    try:
        if kind == "image" or carry_ref:
            # Before the row exists, for the same reason create_job does it:
            # next_queued can otherwise claim the job in the gap and find no
            # input.png on disk.
            new_dir = svc.job_dir(new_id)
            new_dir.mkdir(parents=True, exist_ok=True)
            if kind == "image":
                shutil.copyfile(src_png, new_dir / "input.png")
            if carry_ref:
                shutil.copyfile(src_ref, new_dir / "ref.png")
        svc.store.create(kind, source["prompt"], params, new_id, stage=stage)
    except Exception:
        # The other half of writing the dir first: a row that exists owns its
        # directory, so only an insert (or copy) that never landed cleans up
        # after itself.
        if new_dir is not None:
            shutil.rmtree(new_dir, ignore_errors=True)
        raise
    svc.wake_worker()
    return {"id": new_id, "seed": params["seed"]}


def promote_to_model(
    svc: WarlockService,
    job_id: str,
    *,
    mesh_seed: int | None = None,
    platform: str | None = None,
    resolution: int | None = None,
    size_m: float | None = None,
    bg_removal: str | None = None,
    profile: str | None = None,
    custom_triangles: int | None = None,
    rig: bool | None = None,
    rig_template: str | None = None,
    reference_prep: bool | None = None,
    force: bool = False,
    candidate_group: str | None = None,
    candidate_index: int = 0,
) -> dict[str, Any]:
    """Run the 3D stage from a reference the user approved.

    The child is an ordinary image job whose input.png is the parent's, which
    is the same reduction remesh already makes -- so the queue, the progress
    model and cancellation need no special case. What is new is parent_id,
    which is what lets the history show a reference and its attempts as one
    lineage instead of unrelated rows.

    Everything but mesh_seed is an *override*: omitted means "keep what the
    reference recorded". They exist because the 3D pane owns the mesh-side
    decisions -- the reference was made without any of them being asked -- and
    they are validated with the same helpers create_job uses, in the same
    order, so an unusable value costs the request rather than two minutes of
    GPU.

    ``resolution`` is the exception to "omitted means keep what the reference
    recorded": see the comment below the params copy -- a reference's stored
    resolution belongs to the 2D pane's platform select, not to the mesh.
    """
    check_seed("mesh_seed", mesh_seed)
    if resolution is not None and resolution not in ALLOWED_RESOLUTIONS:
        raise Invalid(
            f"resolution must be one of {sorted(ALLOWED_RESOLUTIONS)}", field="resolution"
        )
    source = svc.require_job(job_id)
    if source["stage"] != "reference":
        raise Invalid(
            "a tile has no subject to reconstruct"
            if source["stage"] == "tile"
            else "this job is not a reference"
        )
    if source["status"] != "done":
        raise Invalid(not_done_message("That reference", source["status"]))
    src_png = svc.job_dir(job_id) / "input.png"
    if not src_png.exists():
        raise Invalid("reference has no image")
    # A soft check, and only against what the reference stage already
    # measured: the mesh stage is where the two minutes of GPU are, so a
    # reference that cannot reconstruct should be refused before it is spent.
    # Bypassable because the rules are heuristics about composition, not
    # facts -- the 3D pane sends force behind a confirm.
    report = source["params"].get("reference_report") or {}
    if not force and report.get("ok") is False:
        raise Invalid(
            " ".join(report.get("reasons") or ["this reference cannot reconstruct"])
        )

    params = {
        k: v
        for k, v in source["params"].items()
        if k not in DERIVED_PARAMS and k not in CONDITIONING_PARAMS
    }
    # rerun_of is provenance of the *reference*: this job's lineage is
    # parent_id, and carrying it would claim the mesh is a rerun of a
    # reference it never was.
    params.pop("rerun_of", None)
    # No ref.png is copied either: the promotion is an image job, and the
    # conditioning already did its work in the reference this promotes.

    overrides = {
        k: v
        for k, v in (("platform", platform), ("size_m", size_m), ("bg_removal", bg_removal))
        if v is not None
    }
    # Re-normalized as a whole rather than patched in place: platform implies
    # the resolution the worker sends to trellis, so a stored resolution from
    # the old platform has to be dropped and re-derived rather than left to
    # contradict the new one.
    #
    # And the inherited one is dropped even with *no* platform override, which
    # is the 512 trap: ``platform`` is one params key behind two selects, and
    # the value a reference carries is the 2D pane's prompt fragment -- "2d"
    # meaning flat and readable, whose 512 is a *reference* resolution. Copying
    # it wholesale reconstructed every promotion of a 2D-styled reference at
    # half the geometry resolution, invisibly. So the model side decides,
    # exactly as ``studio/review_mode.capture_base`` decides it for a sweep:
    # the 3D platform wins, an explicit resolution survives. The 2D taxonomy
    # value itself is left alone -- this is the geometry resolution, not a
    # relabelling of the prompt the reference was drawn from.
    raw = {**params, **overrides}
    raw.pop("resolution", None)
    if resolution is not None:
        raw["resolution"] = int(resolution)
    elif "platform" not in overrides:
        raw["resolution"] = guidance.PLATFORMS[guidance.DEFAULT_PLATFORM].resolution
    params.update(_normalize_guidance(svc, raw))
    resolve_profile(svc, params, profile, custom_triangles)
    if reference_prep is not None:
        params["reference_prep"] = bool(reference_prep)
    if rig is not None:
        # An explicit false has to clear an inherited rig request, or a
        # reference generated with rigging on would rig every promotion of it
        # whatever the 3D pane says.
        if rig:
            params["rig"] = True
            params["rig_template"] = valid_template(rig_template, svc.config.rig_template)
        else:
            params.pop("rig", None)
            params.pop("rig_template", None)

    # After the guidance normalize, because it *overrides* the matte mode that
    # normalize just defaulted: a reference whose alpha is a cutout somebody
    # approved must not be re-cut by the server. Read off the file rather than
    # off params, because the alpha is the evidence -- a hand edit in Inker
    # writes it into input.png and records nothing.
    matte.approve(params, src_png)

    params["mesh_seed"] = mesh_seed if mesh_seed is not None else random_seed()
    params["seed"] = params["mesh_seed"]

    # The other door onto a mesh job, and the expensive one -- a promotion is
    # the two-minute reconstruction with nothing cheap in front of it.
    check_vram(svc, "image", "model", params)

    new_id = uuid.uuid4().hex[:12]
    new_dir = svc.job_dir(new_id)
    new_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(src_png, new_dir / "input.png")
        svc.store.create(
            "image",
            source["prompt"],
            params,
            new_id,
            stage="model",
            parent_id=job_id,
            # Columns, never params keys: this function copies the source's
            # params, so a membership key here would be inherited by every
            # later promotion of the same reference. See migration 6.
            candidate_group=candidate_group,
            candidate_index=candidate_index,
        )
    except Exception:
        # As in create_job: the dir is written first so the worker can never
        # claim a job with no input.png, so a failed insert (or a copy that
        # died mid-file) has to remove it.
        shutil.rmtree(new_dir, ignore_errors=True)
        raise
    svc.wake_worker()
    return {"id": new_id, "parent": job_id, "mesh_seed": params["mesh_seed"]}


def promote_candidates(
    svc: WarlockService,
    job_id: str,
    *,
    count: int = 1,
    mesh_seed: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Promote one reference into ``count`` competing meshes.

    Reliability, not variety: trellis is deterministic in its seed and its
    failure mode is a lottery -- the same reference reconstructs cleanly at one
    seed and comes back with a hole through the shoulder at another. Asking for
    three and keeping the best one is the cheapest answer to that, and it costs
    nothing new anywhere else because each candidate is an *ordinary* mesh job
    through :func:`promote_to_model`: same validation, same VRAM admission per
    job, same directory-before-row ordering, same worker path.

    ``count == 1`` is exactly the old call and mints no group at all -- a group
    of one is a picker asking which of one, and a row hidden from the library
    until somebody answers.

    **The seed rule.** Candidate 0 keeps the requested ``mesh_seed`` so a
    pinned seed still reproduces; the rest draw fresh ones. This is the idiom
    ``create_job`` already applies to reference candidates, for the same
    reason: fanning out from a seed the user chose, rather than replacing it.

    **Admission is all-or-nothing**, the rule ``service.sweeps`` states -- a
    refused submit must not leave a partial run nobody asked for. It needs no
    validation pass of its own here, and duplicating one would be the drift
    ``sweeps._check_unit`` has to work to avoid: every candidate is the *same*
    job but for its mesh seed, and ``promote_to_model`` performs every one of
    its checks before it writes a directory or a row. So a refusal can only
    land on candidate 0, with nothing yet written. Anything failing later is a
    genuine failure (a full disk, a DB error), and gets the best-effort
    rollback below -- through ``delete_if_not_running``, because ``create``
    wakes the worker and candidate 0 may already be inside trellis.
    """
    if not 1 <= count <= MAX_MESH_CANDIDATES:
        raise Invalid(
            f"candidates must be between 1 and {MAX_MESH_CANDIDATES}", field="count"
        )
    check_seed("mesh_seed", mesh_seed)
    if count == 1:
        result = promote_to_model(svc, job_id, mesh_seed=mesh_seed, **kwargs)
        return {**result, "ids": [result["id"]], "group": None}

    group = uuid.uuid4().hex[:12]
    ids: list[str] = []
    try:
        for index in range(count):
            # Candidate 0 keeps what was asked for (None included -- then
            # promote_to_model draws it, which is the unpinned case); the rest
            # fan out from it.
            seed = mesh_seed if index == 0 else random_seed()
            result = promote_to_model(
                svc,
                job_id,
                mesh_seed=seed,
                candidate_group=group,
                candidate_index=index,
                **kwargs,
            )
            ids.append(result["id"])
    except Exception:
        for made in ids:
            if svc.store.delete_if_not_running(made):
                shutil.rmtree(svc.job_dir(made), ignore_errors=True)
        raise
    return {"id": ids[0], "ids": ids, "group": group, "parent": job_id}


def keep_candidate(svc: WarlockService, job_id: str) -> dict[str, Any]:
    """Settle a candidate group: this one is the keeper.

    The whole group leaves it, winner and losers alike, in one statement --
    which is what makes "hidden from the library" a state nothing can be
    stranded in. The losers are *named*, never deleted: the caller offers that
    as its own confirmed action through the ordinary delete path, and a user
    who declines is left with three ordinary assets rather than two invisible
    ones.
    """
    check_job_id(job_id)
    job = svc.require_job(job_id)
    group = job.get("candidate_group")
    if not group:
        raise Conflict("that job is not an undecided candidate")
    losers = [m["id"] for m in svc.store.candidate_jobs(group) if m["id"] != job_id]
    svc.store.resolve_candidates(group)
    return {"id": job_id, "group": group, "losers": losers}
