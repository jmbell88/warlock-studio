"""Making a job exist: the front door, and the two import doors beside it.

Split out of ``service/jobs.py``, which had grown to 1,446 lines over five
unrelated subjects; ``jobs.py`` stays as the facade every caller still imports
and calls by attribute, so a ``monkeypatch.setattr(svc_jobs, "create_job", ...)``
lands exactly as it always did.

The invariant these three share, and the reason they are one module: **the
directory is created before the row**. ``next_queued`` can otherwise claim a
job in the gap and find no ``input.png`` on disk. Every one of them therefore
carries the same shape -- validate everything first, make the directory, write
the row, and delete the directory again if the row write raises.

The function-body imports are deferred deliberately, not accidentally: they
pull in torch, cv2 and trimesh, and this module is imported at app startup.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from .. import guidance, models
from . import matte
from .core import WarlockService
from .errors import Invalid, TooLarge, invalid_from
from .files import ImageTooLarge, to_png
from .validation import (
    ALLOWED_RESOLUTIONS,
    MAX_JOB_NAME,
    MAX_MESH_BYTES,
    MAX_PROMPT,
    MAX_REFERENCE_COUNT,
    MAX_UPLOAD_BYTES,
    check_glb,
    check_seed,
    check_trellis_band,
    check_trellis_tex_res,
    check_vram,
    check_weights,
    note_degraded,
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


def resolve_profile(
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


# The private spelling stayed importable: it is the name three modules in this
# package and one outside it have always called, and renaming a cross-module
# call site is not what promoting a helper is for.
_resolve_profile = resolve_profile


def _check_troupe(svc: WarlockService, block: Any) -> dict[str, Any]:
    """Troupe's follow-up options, validated at *this* door.

    Delegated whole to ``service.troupe``, which owns the numbers and is in
    the same layer -- the drift argument that makes a pipeline restate a
    constant does not apply between two service modules. It is a function here
    rather than an inline call so the import stays local: ``service.troupe``
    imports ``pipelines.spritesynth`` for the guide variants, and this module
    is on the import path of every job the app creates.
    """
    from . import troupe as svc_troupe

    return svc_troupe.check_troupe(svc, block)


def _check_sprite_sheet(svc: WarlockService, block: Any) -> dict[str, Any]:
    """The follow-up sprite sheet's options, validated at *this* door.

    Validated here rather than when the follow-up is queued, and that is the
    whole point of the function. The worker mints the follow-up row itself
    (``_maybe_queue_sprite_sheet``, the shape ``_maybe_queue_rig`` established),
    so it never passes through ``sprites.create_sprite_synthesis`` -- and a bad
    option or a missing ControlNet discovered *there* would be a refusal an hour
    later, on a row the user never submitted, after the reference they did
    submit had already been paid for.

    Delegating to ``sprites``' own constants rather than restating them: the two
    are in the same layer, so the drift argument that makes ``grounds.py``
    restate the plotter's numbers does not apply.
    """
    from . import sprites as svc_sprites

    entries = dict(block or {})
    sheet_type = str(entries.get("sheet_type") or svc_sprites.DEFAULT_SPRITE_SHEET_TYPE)
    if sheet_type not in svc_sprites.SPRITE_SHEET_TYPES:
        raise Invalid(
            f"sheet_type must be one of {list(svc_sprites.SPRITE_SHEET_TYPES)}",
            field="sheet_type",
        )
    try:
        logical = int(
            entries.get("logical_size") or svc_sprites.DEFAULT_SPRITE_LOGICAL_SIZE
        )
    except (TypeError, ValueError):
        raise Invalid("logical_size must be a whole number", field="logical_size") from None
    try:
        colors = int(entries.get("colors") or svc_sprites.DEFAULT_SPRITE_COLORS)
    except (TypeError, ValueError):
        raise Invalid("colors must be a whole number", field="colors") from None
    if logical not in svc_sprites.SPRITE_LOGICAL_SIZES:
        raise Invalid(
            f"logical_size must be one of {list(svc_sprites.SPRITE_LOGICAL_SIZES)}",
            field="logical_size",
        )
    if colors not in svc_sprites.SPRITE_COLOR_CHOICES:
        raise Invalid(
            f"colors must be one of {list(svc_sprites.SPRITE_COLOR_CHOICES)}",
            field="colors",
        )
    # The weights the *follow-up* will load, refused now. Both adapters are
    # mandatory for a synthesis -- the pose guide is the ControlNet and the
    # identity is the IP-Adapter -- so a host missing either would draw the
    # character, queue the sheet and fail it, which reads as a bug rather than
    # as a download the user has not done.
    svc_sprites._check_weights(svc)
    # And the VRAM the follow-up will need, for the same reason: the direct
    # door (``sprites.create_sprite_synthesis``) checks it, and this is the
    # only other way a ``sprite_synthesis`` row comes to exist. A card that
    # fits the character but not the checkpoint-plus-both-adapters sum would
    # otherwise draw the reference and fail the sheet at dispatch, with no
    # remedy in sight. ``base_model`` is the only param the estimate reads for
    # this kind -- both adapters are priced unconditionally.
    check_vram(
        svc,
        "sprite_synthesis",
        "model",
        {"base_model": svc_sprites.SPRITE_BASE_MODEL},
    )
    return {"sheet_type": sheet_type, "logical_size": logical, "colors": colors}


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
    sprite_sheet: dict[str, Any] | None = None,
    troupe: dict[str, Any] | None = None,
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
    if sprite_sheet is not None:
        # The prompt-driven sprite sheet, expressed the way the rig checkbox
        # already is: a *flag on the reference job*, honoured by the worker once
        # the picture it needs exists. Not a job kind of its own, because the
        # thing being asked for is genuinely two steps -- draw the character,
        # then imagine its other three sides -- and the first step is exactly an
        # ordinary reference. Chaining it that way means the character is a row
        # the user can keep, reroll and edit even if the sheet is a disaster.
        if output != "reference":
            raise Invalid(
                "a sprite sheet is drawn from a reference, so output must be "
                "'reference'",
                field="output",
            )
        if count > 1:
            # N characters each spawning two more generations is 3N passes from
            # one button. The Sheet output pins count to 1; this is the door
            # holding the same line for the API.
            raise Invalid(
                "a sprite sheet is drawn from one reference at a time",
                field="count",
            )
        # Checked here, at the top of the door, and the *result* carried down to
        # the params write below rather than the check being made twice: it
        # walks the model registry and stats the weight files, which is a real
        # cost to pay once and a pointless one to pay twice for one submit.
        sprite_block = _check_sprite_sheet(svc, sprite_sheet)
    if troupe is not None:
        # Troupe's request for a follow-up, expressed the way the rig checkbox
        # and the sprite sheet already are: a *flag on the reference job*,
        # honoured once the picture it needs exists. Not a job kind of its own,
        # because the thing being asked for is genuinely a chain with a human
        # gate in the middle of it -- draw the character, approve it, and only
        # then spend the reconstruction -- and the first link is exactly an
        # ordinary reference.
        if output != "reference":
            raise Invalid(
                "a character sheet starts from a reference you approve, so "
                "output must be 'reference'",
                field="output",
            )
        if count > 1:
            # N characters each spawning a mesh, a rig and 256 rendered frames
            # is not a batch, it is an afternoon. The sprite path holds the
            # same line for the same reason.
            raise Invalid(
                "a character sheet is built from one reference at a time",
                field="count",
            )
        # Checked here, at the top of the door, and the result carried down to
        # the params write: it stats weight files and reads a palette, which is
        # a real cost to pay once and a pointless one to pay twice.
        troupe_block = _check_troupe(svc, troupe)
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
    if image is not None and kind != "image":
        # The symmetric refusal: a text job's picture is its ``reference``,
        # and silently accepting an ``image`` upload here would write an
        # input.png the text pipeline never reads (the comment beside the
        # pre-write loop below assumes exactly this cannot happen).
        raise Invalid("only image jobs take an image upload", field="image")
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
    resolve_profile(svc, params, profile, custom_triangles)
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
    if sprite_sheet is not None:
        # The block validation returned, at the top of the door, for the reason
        # the rig template is checked there: the refusal should cost the request
        # rather than the generation that precedes the follow-up. Stored as one
        # nested block so ``_maybe_queue_sprite_sheet`` reads one key, and so
        # VECTOR_PARAMS -- an allowlist of flat settings -- cannot pick any of
        # it up by accident.
        params["sprite_sheet"] = sprite_block
    if troupe is not None:
        # One nested block, for the reason the sprite block gives: a follow-up
        # request is not a flat setting, and VECTOR_PARAMS is an allowlist of
        # flat settings that must not pick any of this up by accident.
        from . import troupe as svc_troupe

        params["troupe"] = troupe_block
        # The rig is not optional for a character sheet -- every cell is a
        # posed frame -- so it is set here rather than left to a checkbox the
        # user could clear without knowing what it cost them. The template is
        # pinned for the same reason ``TROUPE_TEMPLATE`` exists: it is the only
        # one the clip library carries a walk for.
        params["rig"] = True
        params["rig_template"] = svc_troupe.TROUPE_TEMPLATE
        # Measured off the mesh's own vertices rather than fitted to its
        # bounding box: the reference is a constrained T-pose by construction
        # here, and the shipped humanoid template is an A-pose that mis-fits
        # one badly enough to skin the arms to the chest. Where the
        # measurement refuses -- a mesh that is in neither pose --
        # ``jointfit`` says so and the rig falls back to the template, which
        # is the fallback the plan asks for: the user corrects the joints in
        # Poser and re-runs the sheet from the direct door.
        params["rig_joints"] = "measured"
        # Pinned, deliberately not inherited: the reference stage conditions on
        # a rendered pose guide, so it needs a base that can run a ControlNet.
        params["base_model"] = svc_troupe.TROUPE_BASE_MODEL
        # Set *after* the "conditioning needs a reference image" check above,
        # and that is the point: this job's hint is not derived from a
        # reference at all. ``guide`` tells ``_q_generate._conditioning`` to
        # draw ``spritesynth.render_tpose_guide`` straight into control.png --
        # the guide is already line art in canny space, and running the
        # detector over it would return two lines where it means one.
        params["control"] = "canny"
        params["control_hint_source"] = "guide"
        params["guide_variant"] = troupe_block["variant"]
        params.setdefault("control_scale", models.CONTROLNETS["canny"].default_scale)
        params.setdefault("control_end", models.CONTROLNETS["canny"].default_end)

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
    *Open in Plotter* from the **cached row alone**, with no stat on the frame
    thread: a reopen has no fallback (unlike *Open in Clay*, which imports
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
        except Exception as exc:
            # Logged and swallowed, as the worker's own grounding step is: the
            # GLB is already on disk, and a mesh trimesh cannot parse must not
            # fail a job that has produced one. Grounding runs on every asset
            # regardless of whether a size was asked for.
            #
            # Recorded as well as logged, and for the reason ART-01 gives: this
            # row is inserted with ``status="done"``, so without the note the
            # user has a successful-looking asset whose pivot and scale are the
            # engine's, and no way at all to find that out.
            log.exception("normalize failed for built asset %s; leaving the mesh as-is", job_id)
            note_degraded(
                params,
                "normalize",
                f"the mesh was not centred, grounded"
                f"{' or resized' if size_m else ''} ({exc}); its pivot and scale "
                f"are the engine's",
            )
        try:
            from .. import meshreport

            params["mesh_report"] = meshreport.build(model, target_size_m=size_m)
        except Exception as exc:
            log.exception("mesh report failed for built asset %s", job_id)
            note_degraded(
                params,
                "report",
                f"the mesh could not be measured ({exc}); size, triangle count "
                f"and watertightness are unknown for this asset",
            )
        svc.store.create("image", prompt or "", params, job_id, stage="model", status="done")
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    if name:
        svc.store.set_meta(job_id, name=name[:MAX_JOB_NAME])
    return {"id": job_id}
