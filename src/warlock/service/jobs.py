"""Creating, resubmitting, editing and removing jobs."""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from .. import guidance, models, rigging
from . import matte
from . import verdicts as verdicts_mod
from .core import WarlockService
from .errors import Conflict, Failed, Invalid, NotFound, TooLarge, invalid_from
from .files import ImageTooLarge, attach_files, dir_size, measure_storage, to_png
from .validation import (
    ALLOWED_RESOLUTIONS,
    CONDITIONING_PARAMS,
    DERIVED_PARAMS,
    MAX_JOB_NAME,
    MAX_LIST_LIMIT,
    MAX_MESH_BYTES,
    MAX_MESH_CANDIDATES,
    MAX_PROMPT,
    MAX_REFERENCE_COUNT,
    MAX_UPLOAD_BYTES,
    check_glb,
    check_job_id,
    check_seed,
    check_trellis_band,
    check_trellis_tex_res,
    check_vram,
    check_weights,
    normalize_tags,
    not_done_message,
    random_seed,
    valid_template,
)

log = logging.getLogger(__name__)


def _normalize_guidance(svc: WarlockService, raw: dict[str, Any]) -> dict[str, Any]:
    """``guidance.normalize`` with this host's matte gate applied and its
    ValueError translated. Takes the service purely for the gate: guidance is
    pure and cannot look at ``birefnet.gguf`` itself."""
    try:
        return guidance.normalize(
            raw, bg_default=guidance.default_bg_removal(svc.config.trellis_models_dir)
        )
    except ValueError as exc:
        raise invalid_from(exc, "Those generation settings are not usable") from exc


def _resolve_profile(
    svc: WarlockService,
    params: dict[str, Any],
    profile: str | None,
    custom_triangles: int | None,
) -> int | None:
    """Validate a triangle budget, record it, and return it. Validated at submit
    time for the same reason a rig template is: an unusable budget should cost
    the request, not the two minutes of GPU that precede the optimize step.

    A tier that needs gltfpack is refused outright while the binary is absent,
    not just downgraded: the worker's fallback ships the raw copy silently, so
    the job would finish ``done`` wearing a profile param the mesh never saw --
    and ``profile`` is in ``findings.VECTOR_PARAMS``, so every verdict on it
    would credit a tier that never ran. The UI never offers these tiers
    without the binary; this closes the API, sweep and retarget doors too.

    The budget is returned so ``optimize_job`` -- which runs the optimizer right
    there rather than queueing it -- can use this one implementation instead of
    a second, divergent copy of the same two checks. ``None`` back means "no
    reduction": the ``raw`` tier, and also a caller that named no profile at all,
    which are the same instruction to ``optimize.run``.
    """
    if profile is None:
        return None
    from ..pipelines import optimize

    try:
        target = optimize.resolve(profile, custom_triangles)
    except ValueError as exc:
        raise invalid_from(exc, "That triangle budget is not usable", field="profile") from exc
    if target is not None and not svc.config.gltfpack_exe.exists():
        raise Invalid(
            f"the '{profile}' budget needs gltfpack, which is not installed "
            f"(expected at {svc.config.gltfpack_exe}); use profile 'raw', or "
            "vendor the binary and retry",
            field="profile",
        )
    params["profile"] = profile
    if custom_triangles is not None:
        params["custom_triangles"] = custom_triangles
    return target


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
    trellis_band: int | None = None,
    trellis_tex_res: int | None = None,
    image: bytes | None = None,
    reference: bytes | None = None,
    output: str = "model",
    count: int = 1,
    guidance_fields: dict[str, Any] | None = None,
    sweep_id: str | None = None,
    sweep_unit: str = "",
) -> dict[str, Any]:
    """Queue one job, or ``count`` reference candidates of one.

    ``image`` is the raw upload; the caller may hand over at most
    MAX_UPLOAD_BYTES + 1 bytes, which is all that is needed to know it is over.
    """
    config = svc.config
    if kind not in ("text", "image"):
        raise Invalid("kind must be 'text' or 'image'", field="kind")
    if output not in ("reference", "model", "tile"):
        raise Invalid("output must be 'reference', 'model' or 'tile'", field="output")
    if output in ("reference", "tile") and kind != "text":
        # An image job's reference is the upload; there is nothing to approve.
        # And a tile is generated by definition -- an uploaded one is just an
        # image, and nothing here would do anything to it.
        raise Invalid(f"only text jobs can produce a {output}", field="output")
    if not 1 <= count <= MAX_REFERENCE_COUNT:
        raise Invalid(f"count must be between 1 and {MAX_REFERENCE_COUNT}", field="count")
    if count > 1 and output == "model":
        # N meshes per submit is minutes of GPU each; only the cheap 4-step
        # image stages are worth batching.
        raise Invalid("count > 1 requires output=reference or output=tile", field="count")
    if count > 1 and sweep_id:
        # A sweep unit is one job: it is what a verdict is filed against and
        # what a config vector describes. N candidates behind one unit label
        # would make both ambiguous.
        raise Invalid("a sweep unit is a single job", field="count")
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
    check_trellis_band(trellis_band)
    check_trellis_tex_res(trellis_tex_res)

    # Validated up front: a rejected request must not leave an input.png behind.
    params = _normalize_guidance(
        svc,
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
        },
    )
    # Checked after normalize so an unknown adapter key still fails first, and
    # before anything is written: a conditioning selection with no image to
    # condition on would otherwise reach the worker and be silently dropped.
    if reference is None and (params.get("ip_adapter") or params.get("control")):
        raise Invalid(
            "conditioning needs a reference image", field="reference"
        )
    # Same place and the same reason: a tile's seamlessness is circular padding
    # over Conv2d, which a DiT has none of. Refused rather than degraded --
    # patching only what a non-SDXL pipe does have (its VAE) yields an image
    # whose latent never wrapped, which looks seamless in a thumbnail and seams
    # in a material.
    if output == "tile":
        base = models.BASE_MODELS.get(str(params.get("base_model") or ""))
        if base is not None and base.family != models.FAMILY_SDXL:
            raise Invalid(
                f"base_model {base.key!r} cannot generate a seamless tile; "
                f"pick one of {models.tile_bases()}",
                field="base_model",
            )
    # One seed used to drive both stages, so "keep this reference, try another
    # mesh" was impossible without also redrawing the image. seed remains the
    # fallback for both so old rows are unchanged.
    params["seed"] = seed
    params["reference_seed"] = seed if reference_seed is None else reference_seed
    params["mesh_seed"] = seed if mesh_seed is None else mesh_seed
    _resolve_profile(svc, params, profile, custom_triangles)
    if reference_prep is not None:
        # Written only when asked for, so an un-set job keeps following
        # queue.DEFAULT_REFERENCE_PREP rather than being pinned to whatever
        # today's default happens to be. The 3D pane *always* asks -- its
        # checkbox is on screen, and pinning what the user can see is the
        # deliberate choice there (settings_3d.promote_kwargs) -- so the
        # follow-the-default path is the API's and the sweeps', not the UI's.
        params["reference_prep"] = bool(reference_prep)
    for key, value in (
        ("trellis_band", trellis_band),
        ("trellis_tex_res", trellis_tex_res),
    ):
        # Written only when asked for, the reference_prep pattern: an unset job
        # keeps following the config, rather than being pinned to whatever
        # today's default happens to be. These are *inputs*, so they stay out of
        # DERIVED_PARAMS -- a reroll reproducing the server config the mesh was
        # made under is correct provenance, not an inherited verdict.
        #
        # The known limitation: with None meaning "unset", an explicit "auto
        # band" is inexpressible when the config default is a number. Today's
        # default is auto, so nothing can currently want it.
        if value is not None:
            params[key] = int(value)
    if rig:
        # Validated now rather than 90 seconds later: an unusable template
        # should cost the request, not the whole generation that precedes the
        # rig. The worker queues the follow-up job when the mesh lands.
        params["rig"] = True
        params["rig_template"] = valid_template(rig_template, config.rig_template)

    # Last of the door checks and still before any write, so a refused job
    # leaves no input.png: the projected peak is a function of the normalized
    # params (conditioning, resolution), which is why it cannot run any earlier.
    check_weights(svc, kind, params)
    check_vram(svc, kind, "model" if output == "model" else output, params)

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
    if normalized is not None and output == "model":
        # An upload that already carries a matte -- Clay's "send to 3D", a
        # drawing Inker sent straight over, a cutout from anywhere -- is a
        # matte somebody made, and the server would re-cut it under today's
        # default. Only a mesh job asks: nothing downstream of a reference or a
        # tile mattes anything. See service/matte.py for the exe's own rule.
        matte.approve(params, normalized)
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
                # Recorded before the payload writes, not after: a write that
                # dies mid-file (disk full) must still get its dir removed.
                made_dirs.append(job_dir)
                if normalized is not None:
                    (job_dir / "input.png").write_bytes(normalized)
                if normalized_ref is not None:
                    # Every candidate gets its own copy: they are independent
                    # rows, and prune deletes one dir without touching another.
                    (job_dir / "ref.png").write_bytes(normalized_ref)
            svc.store.create(
                kind,
                prompt,
                candidate,
                job_id,
                stage=output,
                sweep_id=sweep_id,
                sweep_unit=sweep_unit,
            )
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
    authored: str | None = None,
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

    ``authored`` names the mode that has a document beside this row -- today
    ``"plotter"`` or ``"packwright"``. It is what lets the library offer
    *Edit in Plotter* from the **cached row alone**, with no stat on the frame
    thread: a reopen has no fallback (unlike *Edit in Clay*, which imports
    ``model.glb`` when there is no ``.wblk``), so the menu has to know whether
    the source is there before it offers to open it. It is an *input*, like
    ``built``, so it stays out of ``DERIVED_PARAMS`` -- a reroll of one of these
    is already refused, and a promotion carries a true statement about where the
    picture came from.
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
    if authored:
        params["authored"] = str(authored)
    job_id = uuid.uuid4().hex[:12]
    job_dir = svc.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / "input.png"
    # Written before the row, and cleaned up if the write or the insert fails:
    # the same ordering create_job uses, for the same reason.
    try:
        dest.write_bytes(normalized)
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


def import_mesh(
    svc: WarlockService,
    glb: bytes,
    *,
    name: str | None = None,
    prompt: str | None = None,
    size_m: float | None = None,
) -> dict[str, Any]:
    """Mint a finished model row from geometry that was built, not reconstructed.

    Modelled line for line on :func:`import_reference`, which mints a finished
    *reference* row from pixels that were painted: no worker run, because the
    artifact already exists and queueing one would spend two minutes of GPU
    reproducing what the user just made.

    The payoff is out of all proportion to the size of this function. Rigging,
    posing, sprite sheets, the triangle retarget and every mesh export are pure
    functions of ``model.glb``, so a row created here inherits all of them
    without one of those paths learning that Clay mode exists.

    ``source.glb`` is the authored mesh and ``model.glb`` derives from it, which
    is the existing invariant rather than a new one -- and it is what keeps
    ``optimize_job``'s retarget working on a built asset, since a retarget
    re-derives from the source.
    """
    from ..pipelines import postprocess

    if len(glb) > MAX_MESH_BYTES:
        raise TooLarge("mesh is over 100 MB")
    check_glb(glb)
    if prompt is not None and len(prompt) > MAX_PROMPT:
        raise Invalid(f"prompt must be at most {MAX_PROMPT} characters", field="prompt")

    params: dict[str, Any] = {
        "seed": 0,
        "mesh_seed": random_seed(),
        # An input, not a derived value, so it stays out of DERIVED_PARAMS: it
        # is a statement about where the geometry came from, and it is what
        # rerun_job reads to refuse a regenerate that has nothing to regenerate.
        "built": True,
        "imported": True,
    }
    if size_m is not None:
        params["size_m"] = float(size_m)

    job_id = uuid.uuid4().hex[:12]
    job_dir = svc.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    source = job_dir / "source.glb"
    model = job_dir / "model.glb"
    # Written before the row, and cleaned up if the write or the insert fails:
    # the same ordering create_job and import_reference use, for the same
    # reason -- a disk-full mid-write must not leave a truncated orphan.
    try:
        source.write_bytes(glb)
        shutil.copyfile(source, model)
        try:
            params["transform"] = postprocess.normalize_glb(
                model, float(size_m) if size_m else None
            )
            params["scale_factor"] = params["transform"]["scale"]
        except Exception:
            # Logged and swallowed, as the worker's own grounding step is: the
            # GLB is already on disk, and a mesh trimesh cannot parse must not
            # fail a job that has produced one. Grounding runs on every asset
            # regardless of whether a size was asked for.
            log.exception("normalize failed for built asset %s; leaving the mesh as-is", job_id)
        try:
            from .. import meshreport

            params["mesh_report"] = meshreport.build(model, target_size_m=size_m)
        except Exception:
            log.exception("mesh report failed for built asset %s", job_id)
        svc.store.create("image", prompt or "", params, job_id, stage="model", status="done")
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    if name:
        svc.store.set_meta(job_id, name=name[:MAX_JOB_NAME])
    return {"id": job_id}


def list_jobs(
    svc: WarlockService,
    limit: int = 100,
    before: tuple[float, str] | None = None,
    *,
    files_cache: dict | None = None,
) -> list[dict[str, Any]]:
    """One page of history, newest first. ``before`` is the (created_at, id) of
    the last row of the previous page; MAX_LIST_LIMIT stays the ceiling on a
    single read, so a longer history is reached by paging rather than by asking
    for more at once.

    ``files_cache`` is handed straight to ``attach_files`` and is what makes
    this affordable to call twice a second from the frame loop -- see there.
    """
    limit = max(1, min(limit, MAX_LIST_LIMIT))
    jobs = svc.store.list(limit, before)
    for job in jobs:
        attach_files(job, svc.job_dir(job["id"]), cache=files_cache)
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


def storage_sizes(svc: WarlockService) -> dict[str, int]:
    """The same walk, per job directory -- what incremental accounting needs."""
    from .files import storage_sizes as _sizes

    return _sizes(svc.config.data_dir)


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
    # Read once, outside the walk: it is a whole-table scan, and a prune of a
    # long history would otherwise repeat it per page.
    retained = retained_job_ids(svc)
    kept = 0
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
            if job["id"] in retained:
                # Counted, not silently skipped: a prune that reports "deleted
                # 40" having kept 12 has described a smaller act than it
                # performed in the one direction that matters least, and a
                # larger reclaim than it achieved in the one that matters most.
                kept += 1
                continue
            # Skipped rather than refused, unlike delete_job: pruning is a bulk
            # reclaim, and one asset with a rig in flight is no reason to keep
            # the other two hundred.
            if worker_is_inside(svc, job["id"]) or dependent_jobs(svc, job["id"]):
                continue
            # Conditional in the DB, not against this page's snapshot: a
            # queued job can be claimed in the gap, and deleting it then
            # rmtrees a directory a live reconstruction is writing into.
            if not svc.store.delete_if_not_running(job["id"]):
                continue
            shutil.rmtree(svc.job_dir(job["id"]), ignore_errors=True)
            deleted += 1
    return {"deleted": deleted, "kept": kept}


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


def worker_is_inside(svc: WarlockService, job_id: str) -> bool:
    """Whether the GPU worker is still running this job, whatever the row says.

    The row is not enough. ``cancel_job`` writes ``cancelled`` immediately and
    only *asks* the worker to stop, so between that write and the worker
    unwinding there is a window in which the status says the job is over and
    the reconstruction is still writing into its directory. ``current_job_id``
    is cleared in the worker's own ``finally``, after the last write, which
    makes it the only honest answer to "is it safe to rmtree this".
    """
    worker = getattr(svc, "worker", None)
    return worker is not None and worker.current_job_id == job_id


def dependent_jobs(svc: WarlockService, job_id: str) -> list[str]:
    """Unfinished jobs that write into ``job_id``'s directory.

    A rig or a sheet is a *separate* job row whose artifacts land beside the
    ``model.glb`` they were made from -- the rig belongs to the mesh -- so
    deleting the mesh while one runs lets ``finalize_rig`` rename into a
    directory that no longer exists, and recreates it as an orphan. The target
    job's own status says nothing about this: it is ``done``, which is exactly
    why a rig could be queued for it.
    """
    return [
        job["id"]
        for job in svc.store.active_jobs()
        if (job.get("params") or {}).get("source_job") == job_id
    ]


def retained_job_ids(svc: WarlockService) -> set[str]:
    """Jobs a *bulk* delete must skip, because their files are the evidence.

    ``verdicts.vector`` is denormalized so that what a review taught outlives
    the assets it was taught on, and ``delete_sweep`` says as much: "the assets
    are disposable, what was learned from them is not". That is exactly true of
    one case and false of two, and the difference is whether the claim can be
    reconstructed from the row alone.

    * **A model-stage reject** is fully carried by its row. The finding *is*
      "this vector produced a bad mesh"; the mesh adds nothing. Still disposable.
    * **An accept** is not. Its value is the artifact -- ``tiercheck`` qualifies
      a gltfpack tier against accepted ``source.glb`` files, and the mesh probe
      is fitted to accepted meshes. A row saying "this was good" with no mesh
      behind it cannot serve either.
    * **An image label of either class** is not, and this is the half that is
      easy to get wrong. ``judge.fit`` embeds *pixels*, and it refuses below
      ``MIN_PER_CLASS`` of **each** class -- so the rejected references are
      training data every bit as much as the accepted ones, and deleting them
      is deleting half a corpus.

    This is not hypothetical. On 2026-08-09 the database held 117 verdicts, of
    which **100 named job directories that no longer existed** -- every one of
    them destroyed by a button whose confirmation truthfully said the verdicts
    would be kept. They were. The pixels were not, and the pixels were what
    three separate blocked items needed.

    Per-job deletion is deliberately unaffected: :func:`delete_job` and
    :func:`trash_job` are one deliberate act on one named asset, which is the
    escape hatch when somebody really does want an accepted job gone. What is
    guarded is the three bulk paths, where the asset is not on screen and the
    count is the only thing the user sees.

    The code below is unchanged by migration 10 and reads the same column it
    always did -- but ``"accept"`` is the *derived usable cut* now (grade >= +3),
    written from the grade by one owner, ``vectors.verdict_for_grade``. This is
    one of the four readers that split survives for. There is deliberately no
    per-grade retention tier: which grades are worth keeping is a policy nobody
    has asked for, and the cut is the only threshold the scale has.
    """
    keep: set[str] = set()
    for verdict in svc.store.latest_verdicts():
        if verdict["verdict"] == "accept" or verdict["stage"] in verdicts_mod.IMAGE_STAGES:
            keep.add(verdict["job_id"])
    return keep


def _refuse_if_busy(svc: WarlockService, job_id: str) -> None:
    """Raise ``Conflict`` if anything is still writing into this job's dir."""
    if worker_is_inside(svc, job_id):
        raise Conflict("cancel the job before deleting it")
    if dependent_jobs(svc, job_id):
        raise Conflict("a rig or sprite sheet for this asset is still running")


def trash_job(svc: WarlockService, job_id: str) -> dict[str, Any]:
    """Move a job to the trash (J91): the row is marked, nothing is removed.

    The same refusals as :func:`delete_job`, and that is the point rather than
    symmetry for its own sake. Trashing a job whose rig is still running would
    make it vanish from the library while a subprocess kept writing into its
    directory, and the user would have no way to explain what they were
    watching -- the refusal is about the *filesystem*, and the filesystem does
    not care that this delete is reversible.
    """
    check_job_id(job_id)
    job = svc.require_job(job_id)
    if job["status"] == "running":
        raise Conflict("cancel the job before deleting it")
    _refuse_if_busy(svc, job_id)
    if job["status"] == "queued":
        # A queued job has to be *cancelled*, not merely hidden. Deleting one
        # used to remove the row, so the worker's poll never saw it again; a
        # trashed row is still a row, and without this the worker would pick up
        # a job the user has thrown away, spend two minutes of GPU on it and
        # write a mesh into a directory nothing shows. Best-effort: a claim
        # landing in the gap is caught by the conditional write below, which
        # refuses and leaves the job running and untrashed.
        svc.store.cancel(job_id)
    if not svc.store.set_deleted_if_not_running(job_id, time.time()):
        raise Conflict("cancel the job before deleting it")
    return {"ok": True}


def restore_job(svc: WarlockService, job_id: str) -> dict[str, Any]:
    """Take a job back out of the trash."""
    check_job_id(job_id)
    if not svc.store.set_deleted_if_not_running(job_id, None):
        raise NotFound("no such job")
    return {"ok": True}


def empty_trash(svc: WarlockService) -> dict[str, Any]:
    """Delete every trashed job for real.

    Reads the whole trash rather than a page (``store.trashed``): "empty"
    that left the older half behind while reporting success would be the worst
    possible reading of the word. Each row still goes through the same
    conditional delete as :func:`delete_job` -- a trashed job cannot be running,
    but it can have been *restored and resubmitted* between the read and the
    write, and that job's directory is not this call's to remove.

    A job carrying evidence (:func:`retained_job_ids`) is kept even here, and
    "empty" therefore stops being literally true -- which is the lesser of two
    wrongs, because the trash is where a labelled reference most plausibly sits.
    The rejected references are half a probe's training set and nothing on the
    card can regenerate a *specific* one.
    """
    deleted = 0
    retained = retained_job_ids(svc)
    kept = 0
    for job in svc.store.trashed():
        if job["id"] in retained:
            kept += 1
            continue
        if worker_is_inside(svc, job["id"]) or dependent_jobs(svc, job["id"]):
            # Skipped rather than refused, as prune does: one asset with a rig
            # in flight is no reason to keep the other two hundred.
            continue
        if not svc.store.delete_if_not_running(job["id"]):
            continue
        shutil.rmtree(svc.job_dir(job["id"]), ignore_errors=True)
        deleted += 1
    return {"deleted": deleted, "kept": kept}


def trash_size(svc: WarlockService) -> dict[str, Any]:
    """How much the trash is holding. Blocking -- call from a task thread."""
    rows = svc.store.trashed()
    total = 0
    for job in rows:
        try:
            total += dir_size(svc.job_dir(job["id"]))
        except OSError:
            # A directory that has gone is a job whose files somebody removed
            # by hand; the row is still restorable-in-name and the figure is
            # advisory, so this is not worth failing the measurement over.
            continue
    return {"count": len(rows), "bytes": total}


def delete_job(svc: WarlockService, job_id: str) -> dict[str, Any]:
    job = svc.require_job(job_id)
    if job["status"] == "running":
        raise Conflict("cancel the job before deleting it")
    _refuse_if_busy(svc, job_id)
    # Re-checked inside the delete statement: the status read above is a
    # snapshot, and the worker's claim() can land in the gap.
    if not svc.store.delete_if_not_running(job_id):
        raise Conflict("cancel the job before deleting it")
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

    # A remesh of a reference is the third door onto a mesh job (create_job
    # and promote_to_model are the other two), and the only one that used to
    # skip admission: the reference was admitted against the SDXL cost alone,
    # and the trellis cost was never checked against the plan.
    check_vram(svc, kind, stage, params)

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
    _resolve_profile(svc, params, profile, custom_triangles)
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
    from . import files

    job = svc.require_job(job_id)
    if job["status"] in ("queued", "running"):
        raise Conflict(f"job is {job['status']}; re-optimize it once it finishes")
    job_dir = svc.job_dir(job_id)
    source = job_dir / "source.glb"
    if not source.exists():
        raise Invalid("this job has no source reconstruction to re-optimize")
    # The configured default, not a hardcoded tier: a named tier needs the
    # vendored gltfpack, and `raw` is what Config.mesh_profile still defaults to.
    profile = profile or svc.config.mesh_profile
    # Through the one implementation, not a second copy of it: this used to
    # resolve the budget itself, which raised a fieldless Invalid the UI could
    # not point at anything and -- the half that mattered -- skipped the
    # gltfpack-presence refusal. With a broken WARLOCK_GLTFPACK a retarget then
    # shipped the raw copy while params["profile"] named a tier that never ran,
    # and `profile` is in VECTOR_PARAMS, so the corpus learned it.
    budget = _resolve_profile(svc, {}, profile, custom_triangles)

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
        # Derived artifacts describe the old mesh; drop them once model.glb is
        # *finished*, and under each artifact's own lock so an in-flight
        # conversion of the old mesh can't rename a stale copy into place after
        # the unlink. Deleting before the normalize left a multi-second window
        # in which an STL or OBJ export -- which takes only its own artifact's
        # lock, never this one -- rebuilt itself from the ungrounded mesh and
        # cached that answer indefinitely.
        for name in files.DERIVED:
            with svc.convert_lock(job_id, name), contextlib.suppress(OSError):
                (job_dir / name).unlink()

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


def retexture_job(
    svc: WarlockService,
    job_id: str,
    prompt: str,
    *,
    strength: float | None = None,
    texture_size: int | None = None,
    seed: int | None = None,
    base_model: str | None = None,
) -> dict[str, Any]:
    """Queue a new surface for a finished mesh, from a prompt.

    ``optimize_job``'s near-sibling in everything it *refuses* -- a terminal
    status is required, ``source.glb`` is never touched, and the exports that
    describe the old skin go -- and its opposite in where the work runs. A
    retarget is a two-second gltfpack subprocess and belongs inline; a
    re-texture is six SDXL passes around two Blender ops, so it takes the queue
    for the reason every other GPU path does: it needs the resident pipe, and a
    TaskRunner thread racing the worker for VRAM is the OOM that only
    reproduces under load. Every refusal is still *here* rather than in the
    worker, exactly as ``create_pixel_sheet`` states: a mesh with no atlas to
    replace should cost the request, not a place in the queue and a minute of
    GPU.

    The ``Conflict`` is the same one and for the same reason: the worker's own
    ``_optimize``/``_apply_scale`` write ``model.glb`` without taking a lock, so
    a re-texture queued against a running job would be a second writer with no
    ordering between them. Refusing beats racing.
    """
    from ..pipelines import retexture

    check_job_id(job_id)
    job = svc.require_job(job_id)
    if job["status"] in ("queued", "running"):
        raise Conflict(f"job is {job['status']}; re-texture it once it finishes")
    job_dir = svc.job_dir(job_id)
    if not (job_dir / "model.glb").exists():
        raise Invalid("this job has no mesh to re-texture")
    text = (prompt or "").strip()
    if not text:
        raise Invalid("describe the surface you want", field="prompt")
    if len(text) > MAX_PROMPT:
        raise TooLarge(f"prompt is longer than {MAX_PROMPT} characters", field="prompt")

    value = models.DEFAULT_IMG2IMG_STRENGTH if strength is None else float(strength)
    if not models.IMG2IMG_STRENGTH_MIN <= value <= models.IMG2IMG_STRENGTH_MAX:
        raise Invalid(
            f"strength must be between {models.IMG2IMG_STRENGTH_MIN} "
            f"and {models.IMG2IMG_STRENGTH_MAX}",
            field="strength",
        )
    # None means "match the mesh's own atlas", resolved by the worker against
    # the file rather than here: it is a property of the GLB, and reading it at
    # the door would open a 26 MB file to answer a question the run is about to
    # ask anyway. It rides in params as an absent key, which is also what makes
    # a job row written before this existed still mean what it said.
    size = None if texture_size is None else int(texture_size)
    if size is not None and size not in retexture.TEXTURE_SIZES:
        raise Invalid(
            f"texture_size must be one of {list(retexture.TEXTURE_SIZES)}",
            field="texture_size",
        )
    if seed is not None:
        check_seed("seed", seed)
    base_key = str(base_model or svc.config.t2i_model)
    if base_key not in models.BASE_MODELS:
        raise Invalid(f"unknown base model {base_key!r}", field="base_model")

    params = {
        # Inputs, every one of them: what the re-texture recorded about the
        # atlas it produced goes on the *mesh's* row under "retexture", which is
        # in DERIVED_PARAMS. Nothing here is derived, so a rerun copies it
        # verbatim.
        "source_job": job_id,
        "strength": value,
        "seed": random_seed() if seed is None else int(seed),
        "base_model": base_key,
    }
    if size is not None:
        params["texture_size"] = size
    # At the door and before the row exists, as everywhere: six img2img passes
    # through one resident pipe is a real budget question beside a warm trellis.
    check_vram(svc, "retexture", "model", params)
    check_weights(svc, "text", params)
    new_id = svc.store.create("retexture", text, params, uuid.uuid4().hex[:12])
    svc.wake_worker()
    return {"id": new_id, "source_job": job_id, "stale": stale_surface_artifacts(job_dir)}


def stale_surface_artifacts(job_dir: Path) -> list[str]:
    """What a re-texture will invalidate, so the panel can say so beforehand.

    Deliberately *not* ``stale_rig_artifacts``, and the difference is the whole
    point: a rig, its poses and its sheets reference geometry, and a re-texture
    changes no geometry -- so none of them appears here, and
    ``tests/test_retexture.py`` asserts that rather than leaving it to be
    "fixed" later. What does appear is the exports that carry the skin.

    Reported rather than merely deleted after the fact, the rule the retarget
    panel already follows: a warning before the button beats a list afterwards.
    """
    from ..pipelines import retexture

    return [n for n in retexture.SURFACE_DERIVED if (job_dir / n).exists()]


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
