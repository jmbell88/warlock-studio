"""Creating, resubmitting, editing and removing jobs."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from .. import guidance, rigging
from .core import WarlockService
from .errors import Conflict, Failed, Invalid, NotFound, TooLarge
from .files import ImageTooLarge, attach_files, measure_storage, to_png
from .validation import (
    ALLOWED_RESOLUTIONS,
    CONDITIONING_PARAMS,
    DERIVED_PARAMS,
    MAX_JOB_NAME,
    MAX_LIST_LIMIT,
    MAX_PROMPT,
    MAX_REFERENCE_COUNT,
    MAX_UPLOAD_BYTES,
    check_job_id,
    check_seed,
    normalize_tags,
    random_seed,
    valid_template,
)

log = logging.getLogger(__name__)


def _normalize_guidance(raw: dict[str, Any]) -> dict[str, Any]:
    try:
        return guidance.normalize(raw)
    except ValueError as exc:
        raise Invalid(str(exc)) from exc


def _resolve_profile(params: dict[str, Any], profile: str | None, custom_triangles: int | None):
    """Validate a triangle budget and record it. Validated at submit time for
    the same reason a rig template is: an unusable budget should cost the
    request, not the two minutes of GPU that precede the optimize step."""
    if profile is None:
        return
    from ..pipelines import optimize

    try:
        optimize.resolve(profile, custom_triangles)
    except ValueError as exc:
        raise Invalid(str(exc), field="profile") from exc
    params["profile"] = profile
    if custom_triangles is not None:
        params["custom_triangles"] = custom_triangles


def create_job(
    svc: WarlockService,
    *,
    kind: str,
    prompt: str | None = None,
    seed: int = 42,
    reference_seed: int | None = None,
    mesh_seed: int | None = None,
    resolution: int | None = None,
    size_m: float | None = None,
    lora_weight: float | None = None,
    ip_scale: float | None = None,
    control_scale: float | None = None,
    control_end: float | None = None,
    bg_removal: str | None = None,
    negative_prompt: str | None = None,
    rig: bool = False,
    rig_template: str | None = None,
    reference_prep: bool | None = None,
    profile: str | None = None,
    custom_triangles: int | None = None,
    image: bytes | None = None,
    reference: bytes | None = None,
    output: str = "model",
    count: int = 1,
    guidance_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Queue one job, or ``count`` reference candidates of one.

    ``image`` is the raw upload; the caller may hand over at most
    MAX_UPLOAD_BYTES + 1 bytes, which is all that is needed to know it is over.
    """
    config = svc.config
    if kind not in ("text", "image"):
        raise Invalid("kind must be 'text' or 'image'", field="kind")
    if output not in ("reference", "model"):
        raise Invalid("output must be 'reference' or 'model'", field="output")
    if output == "reference" and kind != "text":
        # An image job's reference is the upload; there is nothing to approve.
        raise Invalid("only text jobs can stop at a reference", field="output")
    if not 1 <= count <= MAX_REFERENCE_COUNT:
        raise Invalid(f"count must be between 1 and {MAX_REFERENCE_COUNT}", field="count")
    if count > 1 and output != "reference":
        # N meshes per submit is minutes of GPU each; only the cheap 4-step
        # reference stage is worth batching.
        raise Invalid("count > 1 requires output=reference", field="count")
    # An explicit resolution overrides the platform preset; the UI no longer
    # sends one, but the API keeps accepting it.
    if resolution is not None and resolution not in ALLOWED_RESOLUTIONS:
        raise Invalid(
            f"resolution must be one of {sorted(ALLOWED_RESOLUTIONS)}", field="resolution"
        )
    if kind == "text" and not (prompt and prompt.strip()):
        raise Invalid("text jobs require a prompt", field="prompt")
    if prompt is not None and len(prompt) > MAX_PROMPT:
        raise Invalid(f"prompt must be at most {MAX_PROMPT} characters", field="prompt")
    if kind == "image" and image is None:
        raise Invalid("image jobs require an image upload", field="image")
    if reference is not None and kind != "text":
        # An image job never touches SDXL, so there is nothing for a
        # conditioning reference to condition.
        raise Invalid("only text jobs take a conditioning reference", field="reference")
    for name, value in (
        ("seed", seed),
        ("reference_seed", reference_seed),
        ("mesh_seed", mesh_seed),
    ):
        check_seed(name, value)

    # Validated up front: a rejected request must not leave an input.png behind.
    params = _normalize_guidance(
        {
            **(guidance_fields or {}),
            "size_m": size_m,
            "resolution": resolution,
            "lora_weight": lora_weight,
            "ip_scale": ip_scale,
            "control_scale": control_scale,
            "control_end": control_end,
            "bg_removal": bg_removal,
            "negative_prompt": negative_prompt,
        }
    )
    # Checked after normalize so an unknown adapter key still fails first, and
    # before anything is written: a conditioning selection with no image to
    # condition on would otherwise reach the worker and be silently dropped.
    if reference is None and (params.get("ip_adapter") or params.get("control")):
        raise Invalid(
            "conditioning needs a reference image", field="reference"
        )
    # One seed used to drive both stages, so "keep this reference, try another
    # mesh" was impossible without also redrawing the image. seed remains the
    # fallback for both so old rows are unchanged.
    params["seed"] = seed
    params["reference_seed"] = seed if reference_seed is None else reference_seed
    params["mesh_seed"] = seed if mesh_seed is None else mesh_seed
    _resolve_profile(params, profile, custom_triangles)
    if reference_prep is not None:
        # Written only when asked for, so an un-set job keeps following
        # queue.DEFAULT_REFERENCE_PREP rather than being pinned to whatever
        # today's default happens to be.
        params["reference_prep"] = bool(reference_prep)
    if rig:
        # Validated now rather than 90 seconds later: an unusable template
        # should cost the request, not the whole generation that precedes the
        # rig. The worker queues the follow-up job when the mesh lands.
        params["rig"] = True
        params["rig_template"] = valid_template(rig_template, config.rig_template)

    def _decode(raw: bytes, field: str) -> bytes:
        if len(raw) > MAX_UPLOAD_BYTES:
            raise TooLarge(f"{field} upload is over 20 MB")
        try:
            return to_png(raw)
        except ImageTooLarge as exc:
            raise Invalid(str(exc), field=field) from exc
        except Exception as exc:
            raise Invalid(f"could not decode uploaded {field}", field=field) from exc

    normalized = _decode(image, "image") if image is not None else None
    # Same caps, same order, same pre-write window as input.png.
    normalized_ref = _decode(reference, "reference") if reference is not None else None

    # Write the file before the row exists: the worker's next_queued() poll can
    # otherwise claim an image job in the gap and find no input.png on disk yet.
    # count > 1 only ever happens for text/reference jobs (normalized is None
    # then), but count == 1 still carries an ordinary image job through this
    # same loop, so the ordering must hold for every candidate. The flip side
    # of that ordering is the except below: a failed insert must remove the dir
    # it already wrote, or every DB hiccup leaves an orphan directory nothing
    # will ever list or prune by id.
    ids: list[str] = []
    made_dirs: list[Path] = []
    try:
        for i in range(count):
            candidate = dict(params)
            if i > 0:
                # Candidate 0 keeps the requested seed so a pinned seed still
                # reproduces; the rest fan out from it. mesh_seed follows
                # reference_seed/seed here even though these rows are still
                # stage="reference" and never reach the mesh stage as-is: the
                # settings panel renders all three seeds together, and a
                # candidate whose seed/reference_seed changed but whose
                # mesh_seed silently didn't would read as a bug, not as the
                # "seed is the legacy fallback for both stages" contract a few
                # lines up. promote_to_model overwrites mesh_seed on promotion
                # and reroll rewrites all of them, so this has no effect on
                # what actually gets meshed -- it only keeps the displayed
                # values internally consistent.
                candidate["reference_seed"] = random_seed()
                candidate["seed"] = candidate["reference_seed"]
                candidate["mesh_seed"] = candidate["reference_seed"]
            job_id = uuid.uuid4().hex[:12]
            if normalized is not None or normalized_ref is not None:
                job_dir = config.job_dir(job_id)
                job_dir.mkdir(parents=True, exist_ok=True)
                if normalized is not None:
                    (job_dir / "input.png").write_bytes(normalized)
                if normalized_ref is not None:
                    # Every candidate gets its own copy: they are independent
                    # rows, and prune deletes one dir without touching another.
                    (job_dir / "ref.png").write_bytes(normalized_ref)
                made_dirs.append(job_dir)
            svc.store.create(kind, prompt, candidate, job_id, stage=output)
            ids.append(job_id)
    except Exception:
        # Only the candidates whose insert never landed: a row that exists owns
        # its directory, and deleting it here would orphan the row.
        for job_dir in made_dirs[len(ids) :]:
            shutil.rmtree(job_dir, ignore_errors=True)
        raise
    svc.wake_worker()
    return {"id": ids[0], "ids": ids}


def import_reference(
    svc: WarlockService,
    image: bytes,
    *,
    prompt: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Mint a finished reference row from pixels that were painted, not generated.

    No worker run: the image already exists, so queueing one would spend two
    minutes of GPU reproducing what the user just drew. The row is created
    ``done`` at stage ``reference``, which is exactly what promote_to_model
    consumes -- so "paint something, make a mesh of it" needs no new path
    through the queue.

    ``hand_edited`` is set for the same reason save_edited_image sets it:
    ``params["recipe"]`` normally claims a seed and a model produced this
    image, and here nothing did. The reference report *is* measured, because
    promote_to_model's quality gate reads it and a missing report would let a
    reference through that cannot reconstruct.
    """
    from ..pipelines import reference

    if len(image) > MAX_UPLOAD_BYTES:
        raise TooLarge("image upload is over 20 MB")
    try:
        normalized = to_png(image)
    except ImageTooLarge as exc:
        raise Invalid(str(exc), field="image") from exc
    except Exception as exc:
        raise Invalid("could not decode the painted image", field="image") from exc
    if prompt is not None and len(prompt) > MAX_PROMPT:
        raise Invalid(f"prompt must be at most {MAX_PROMPT} characters", field="prompt")

    params: dict[str, Any] = {
        "seed": 0,
        "reference_seed": 0,
        "mesh_seed": random_seed(),
        "hand_edited": True,
        "imported": True,
    }
    job_id = uuid.uuid4().hex[:12]
    job_dir = svc.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / "input.png"
    # Written before the row, and cleaned up if the insert fails: the same
    # ordering create_job uses, for the same reason.
    dest.write_bytes(normalized)
    try:
        params["reference_report"] = reference.measure_file(dest).as_dict()
        svc.store.create(
            "image", prompt or "", params, job_id, stage="reference", status="done"
        )
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    if name:
        svc.store.set_meta(job_id, name=name[:MAX_JOB_NAME])
    return {"id": job_id}


def list_jobs(
    svc: WarlockService, limit: int = 100, before: tuple[float, str] | None = None
) -> list[dict[str, Any]]:
    """One page of history, newest first. ``before`` is the (created_at, id) of
    the last row of the previous page; MAX_LIST_LIMIT stays the ceiling on a
    single read, so a longer history is reached by paging rather than by asking
    for more at once."""
    limit = max(1, min(limit, MAX_LIST_LIMIT))
    jobs = svc.store.list(limit, before)
    for job in jobs:
        attach_files(job, svc.job_dir(job["id"]))
        svc.attach_progress(job)
    return jobs


def get_job(svc: WarlockService, job_id: str) -> dict[str, Any]:
    job = svc.require_job(job_id)
    attach_files(job, svc.job_dir(job_id))
    svc.attach_progress(job)
    return job


def storage(svc: WarlockService) -> dict[str, Any]:
    """How much disk the generated assets are using.

    Jobs and their artifacts accumulate forever otherwise -- at 5-20 MB per GLB
    that is real disk within weeks of regular use.
    """
    return measure_storage(svc.config.data_dir)


def prune_jobs(svc: WarlockService, keep: int = 20) -> dict[str, Any]:
    """Delete everything but the newest ``keep`` jobs. Never touches a running one."""
    if keep < 0:
        raise Invalid("keep must be >= 0", field="keep")
    # Paged with a keyset cursor rather than one MAX_LIST_LIMIT read: a history
    # longer than a single page used to be un-prunable past its first 5000
    # rows, which is exactly the history that needs pruning. Deleting rows the
    # walk has already passed doesn't disturb the cursor.
    deleted = 0
    seen = 0
    cursor: tuple[float, str] | None = None
    while True:
        page = svc.store.list(MAX_LIST_LIMIT, cursor)
        if not page:
            break
        cursor = (page[-1]["created_at"], page[-1]["id"])
        for job in page:
            seen += 1
            if seen <= keep or job["status"] == "running":
                continue
            svc.store.delete(job["id"])
            shutil.rmtree(svc.job_dir(job["id"]), ignore_errors=True)
            deleted += 1
    return {"deleted": deleted}


def update_job(svc: WarlockService, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Rename, retag or (un)favourite a job."""
    check_job_id(job_id)
    name = payload.get("name")
    if name is not None:
        name = str(name).strip()
        if len(name) > MAX_JOB_NAME:
            raise Invalid(f"name must be at most {MAX_JOB_NAME} characters", field="name")
    tags = normalize_tags(payload["tags"]) if "tags" in payload else None
    favorite = None
    if "favorite" in payload:
        # Demanded rather than coerced: bool("false") is True, so a caller that
        # sent the string form used to favourite the job it meant to unfavourite.
        if not isinstance(payload["favorite"], bool):
            raise Invalid("favorite must be true or false", field="favorite")
        favorite = payload["favorite"]

    if not svc.store.set_meta(job_id, name=name, tags=tags, favorite=favorite):
        raise NotFound("no such job")
    return get_job(svc, job_id)


def delete_job(svc: WarlockService, job_id: str) -> dict[str, Any]:
    job = svc.require_job(job_id)
    if job["status"] == "running":
        raise Conflict("cancel the job before deleting it")
    svc.store.delete(job_id)
    shutil.rmtree(svc.job_dir(job_id), ignore_errors=True)
    return {"ok": True}


def cancel_job(svc: WarlockService, job_id: str) -> dict[str, Any]:
    job = svc.require_job(job_id)
    if job["status"] == "cancelled":
        # Idempotent success: some earlier request (possibly this exact race)
        # already cancelled it. Only a genuinely terminal done/error status
        # below is "too late" and worth refusing.
        return {"ok": True}
    if job["status"] not in ("queued", "running"):
        raise Conflict(f"job is {job['status']}")
    if job["status"] == "running" and svc.worker is not None:
        svc.call_on_loop(lambda: svc.worker.request_cancel(job_id))
    # Atomic: if the worker's own terminal write (done/error) landed first,
    # this is a no-op and the job's real outcome stands instead of being
    # retroactively overwritten to "cancelled". The DB-level JobStore.finish()
    # conditional write (queue.py) is what actually closes the lost-cancel race
    # for a job that was 'queued' here but got claimed before this reached the
    # DB -- not request_cancel, which only matters for a job already running.
    if not svc.store.cancel(job_id):
        # Could be "already cancelled" (idempotent success -- this call's own
        # effect already landed, e.g. via the race above) or "already
        # done/error" (genuinely too late).
        current = svc.store.get(job_id)
        if current and current["status"] == "cancelled":
            return {"ok": True}
        raise Conflict("job already finished")
    return {"ok": True}


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

    kind = "image" if mode == "remesh" else source["kind"]
    src_png = svc.job_dir(job_id) / "input.png"
    if kind == "image" and not src_png.exists():
        raise Invalid("source job has no reference image to reuse")

    # Derived values describe the *source* run, not this one: keeping them
    # would make the new job claim a composed prompt it never used and a
    # quality score for a mesh that doesn't exist yet.
    params = {k: v for k, v in source["params"].items() if k not in DERIVED_PARAMS}
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

    new_id = uuid.uuid4().hex[:12]
    new_dir = None
    # A reroll reruns SDXL, so its conditioning reference has to come with it
    # -- the first time a *text* rerun needs a directory before the row, which
    # is why the except below now covers a case it never did.
    src_ref = svc.job_dir(job_id) / "ref.png"
    carry_ref = mode == "reroll" and src_ref.exists()
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
    try:
        svc.store.create(kind, source["prompt"], params, new_id, stage=stage)
    except Exception:
        # The other half of writing the dir first: a row that exists owns its
        # directory, so only an insert that never landed cleans up after itself.
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
    size_m: float | None = None,
    bg_removal: str | None = None,
    profile: str | None = None,
    custom_triangles: int | None = None,
    rig: bool | None = None,
    rig_template: str | None = None,
    reference_prep: bool | None = None,
    force: bool = False,
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
    """
    check_seed("mesh_seed", mesh_seed)
    source = svc.require_job(job_id)
    if source["stage"] != "reference":
        raise Invalid("this job is not a reference")
    if source["status"] != "done":
        raise Invalid(f"reference is {source['status']}")
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
    # No ref.png is copied either: the promotion is an image job, and the
    # conditioning already did its work in the reference this promotes.

    overrides = {
        k: v
        for k, v in (("platform", platform), ("size_m", size_m), ("bg_removal", bg_removal))
        if v is not None
    }
    if overrides:
        # Re-normalized as a whole rather than patched in place: platform
        # implies the resolution the worker sends to trellis, so a stored
        # resolution from the old platform has to be dropped and re-derived
        # rather than left to contradict the new one.
        raw = {**params, **overrides}
        if "platform" in overrides:
            raw.pop("resolution", None)
        params.update(_normalize_guidance(raw))
    _resolve_profile(params, profile, custom_triangles)
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

    params["mesh_seed"] = mesh_seed if mesh_seed is not None else random_seed()
    params["seed"] = params["mesh_seed"]

    new_id = uuid.uuid4().hex[:12]
    new_dir = svc.job_dir(new_id)
    new_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_png, new_dir / "input.png")
    try:
        svc.store.create(
            "image", source["prompt"], params, new_id, stage="model", parent_id=job_id
        )
    except Exception:
        # As in create_job: the dir is written first so the worker can never
        # claim a job with no input.png, so a failed insert has to remove it.
        shutil.rmtree(new_dir, ignore_errors=True)
        raise
    svc.wake_worker()
    return {"id": new_id, "parent": job_id, "mesh_seed": params["mesh_seed"]}


def optimize_job(
    svc: WarlockService,
    job_id: str,
    *,
    profile: str | None = None,
    custom_triangles: int | None = None,
) -> dict[str, Any]:
    """Rebuild model.glb from source.glb at a different triangle budget.

    Inline rather than on the queue, under the same per-artifact lock the
    STL/OBJ exports use: gltfpack is a two-second subprocess, and putting it
    behind the serial GPU queue would make it wait on a trellis run.

    A terminal status is required. The two writers are otherwise genuinely
    concurrent -- this rewrites model.glb while the worker's own
    _optimize/_apply_scale/_audit_mesh are still writing it and recording what
    they measured -- and no lock here can help, because the worker's half
    doesn't take one. Refusing beats racing.
    """
    import contextlib

    from ..pipelines import optimize, postprocess

    job = svc.require_job(job_id)
    if job["status"] in ("queued", "running"):
        raise Conflict(f"job is {job['status']}; re-optimize it once it finishes")
    job_dir = svc.job_dir(job_id)
    source = job_dir / "source.glb"
    if not source.exists():
        raise Invalid("this job has no source reconstruction to re-optimize")
    # The configured default, not a hardcoded tier: every named tier needs a
    # gltfpack that isn't vendored yet, so a bare call used to explode.
    profile = profile or svc.config.mesh_profile
    try:
        budget = optimize.resolve(profile, custom_triangles)
    except ValueError as exc:
        raise Invalid(str(exc), field="profile") from exc

    with svc.convert_lock(job_id, "optimize"):
        try:
            result = optimize.run(
                source,
                job_dir / "model.glb",
                target_triangles=budget,
                exe=svc.config.gltfpack_exe,
            )
        except optimize.OptimizeError as exc:
            raise Failed(str(exc)) from exc
        # Derived artifacts describe the old mesh; drop them the moment the new
        # model.glb lands, and under each artifact's own lock so an in-flight
        # conversion of the old mesh can't rename a stale copy into place after
        # the unlink.
        for name in ("model.stl", "model_obj.zip", "model.fbx", "collision.glb", "textures.zip"):
            with svc.convert_lock(job_id, name), contextlib.suppress(OSError):
                (job_dir / name).unlink()
        # The optimizer rewrote the node graph, so the grounding transform went
        # with it and has to be reapplied. Failure is logged and swallowed,
        # same rule as the queue path (_apply_scale): the new GLB is already on
        # disk and serving it unnormalized beats an error that leaves the
        # caches and params half-updated.
        transform = None
        try:
            transform = postprocess.normalize_glb(
                job_dir / "model.glb",
                float(job["params"]["size_m"]) if job["params"].get("size_m") else None,
            )
        except Exception:
            log.exception("normalize failed after optimize for job %s", job_id)

    changes: dict[str, Any] = {"profile": profile, "optimize": result}
    if custom_triangles is not None:
        changes["custom_triangles"] = custom_triangles
    # The old audit/report describe a mesh that no longer exists.
    drop = ["mesh_audit", "mesh_report"]
    if transform is None:
        drop += ["transform", "scale_factor"]
    else:
        changes["transform"] = transform
        changes["scale_factor"] = transform["scale"]
    # Merged rather than written from the copy read at the top: params is one
    # JSON blob, and a full-blob write from a stale read silently discards
    # anything committed in between.
    svc.store.merge_params(job_id, changes, remove=tuple(drop))
    return {
        "ok": True,
        "optimize": result,
        "transform": transform,
        "stale": stale_rig_artifacts(job_dir),
    }


def stale_rig_artifacts(job_dir: Path) -> list[str]:
    """What still describes the old mesh after a retarget.

    Reported, not deleted: the rig and its poses are user work, and a warning
    the caller can act on beats silently destroying them over a triangle
    retarget.
    """
    out = [n for n in ("rig.glb", "rig.json") if (job_dir / n).exists()]
    if out:
        out += sorted(
            f"{rigging.POSE_DIR_NAME}/{p.name}" for p in rigging.pose_dir(job_dir).glob("*.glb")
        )
        out += sorted(
            f"{rigging.SHEET_DIR_NAME}/{p.name}" for p in rigging.sheet_dir(job_dir).glob("*.png")
        )
    return out
