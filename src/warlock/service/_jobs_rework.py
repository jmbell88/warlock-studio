"""Reworking a finished asset in place: retarget its mesh, restyle its skin.

Split out of ``service/jobs.py``, which had grown to 1,446 lines over five
unrelated subjects; ``jobs.py`` stays as the facade every caller still imports
and calls by attribute.

What separates these from a rerun is that **no new job row is minted for the
mesh** -- the asset keeps its identity and its history, and what changes is
one artifact of it. That is also what makes them the dangerous pair: they
write onto files that are being served, so both refuse a job that is queued or
running, both stage their writes, and both are followed by deleting the
derived exports that no longer describe the thing on disk. The two ``stale_*``
helpers are that list, stated rather than globbed.

``optimize_job`` runs the optimizer inline rather than queueing it, which is
why it shares ``resolve_profile`` with ``create_job`` instead of re-deriving
the budget: one implementation of the gltfpack gate, or a job finishes wearing
a tier that never ran.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from .. import models, rigging
from ._jobs_create import resolve_profile
from .core import WarlockService
from .errors import Conflict, Failed, Invalid, TooLarge
from .validation import (
    MAX_PROMPT,
    check_job_id,
    check_seed,
    check_vram,
    check_weights,
    random_seed,
)

log = logging.getLogger(__name__)


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
    budget = resolve_profile(svc, {}, profile, custom_triangles)

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
