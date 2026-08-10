"""Rigs, joint adjustment and poses.

A pose is a bone -> local rotation map saved against a job's rig, stored as a
file in that job's directory. No job kind and no DB row: a pose is small,
instant to write, and belongs to the mesh the same way its rig does. Baking one
into a GLB is the only slow part, and that is derived on demand and cached,
exactly like the STL and OBJ exports.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from .. import doctor, rigging
from .core import WarlockService
from .errors import Conflict, Failed, Invalid, NotFound, invalid_from
from .validation import check_job_id, check_pose_id, valid_template

log = logging.getLogger(__name__)


def rig_templates(svc: WarlockService) -> dict[str, Any]:
    """Which skeletons a rig request may ask for, plus whether rigging works at
    all. The UI hides the rig controls when available is false rather than
    offering a button that can only fail."""
    check = doctor.blender_check()
    return {
        "available": check.ok,
        "detail": check.detail,
        "default": svc.config.rig_template,
        "templates": rigging.catalog(),
    }


def template_presets(key: str) -> dict[str, Any]:
    """The shipped pose library for a skeleton.

    Read-only and job-independent: applying one saves an ordinary pose through
    the normal save path, so a preset and a hand-made pose are the same thing
    by the time anything else sees them.
    """
    try:
        return {"poses": rigging.preset_poses(key)}
    except ValueError as exc:
        raise invalid_from(
            exc, "That skeleton has no pose library", field="rig_template"
        ) from exc


def create_rig(svc: WarlockService, job_id: str, *, template: str | None = None) -> dict[str, Any]:
    """Queue a rig for a finished job's mesh.

    A queue job rather than an inline call: automatic weights on a 300k-face
    mesh is minutes of CPU, and going through the queue buys cancellation,
    progress and history for free -- and guarantees it never overlaps a trellis
    or SDXL run.
    """
    source = svc.require_job(job_id)
    if source["kind"] == "rig":
        raise Invalid("cannot rig a rig job; rig its source mesh")
    if source["status"] != "done" or not (svc.job_dir(job_id) / "model.glb").exists():
        raise Invalid("job has no finished mesh to rig")
    params = {"source_job": job_id, "template": valid_template(template, svc.config.rig_template)}
    new_id = svc.store.create("rig", source["prompt"], params, uuid.uuid4().hex[:12])
    svc.wake_worker()
    return {"id": new_id, "source_job": job_id, "template": params["template"]}


def adjust_joints(svc: WarlockService, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Re-rig a mesh with joints the user moved.

    A queued rig job rather than an inline call, for the same reason the
    original rig is: skinning is minutes of CPU and must never overlap a
    trellis run.
    """
    source = svc.require_job(job_id)
    job_dir = svc.job_dir(job_id)
    rig = rigging.read_rig(job_dir)
    if rig is None or not (job_dir / "model.glb").exists():
        raise Invalid("job is not rigged")
    try:
        template = rigging.get_template(str(rig.get("template") or svc.config.rig_template))
        bones = rigging.validate_joints(payload, template)
    except ValueError as exc:
        raise invalid_from(exc, "Those joint positions cannot be used") from exc

    params = {
        "source_job": job_id,
        "template": template.key,
        "bones": bones,
        "adjusted": True,
    }
    new_id = svc.store.create("rig", source["prompt"], params, uuid.uuid4().hex[:12])
    svc.wake_worker()
    return {"id": new_id, "source_job": job_id}


def get_rig(svc: WarlockService, job_id: str) -> dict[str, Any]:
    check_job_id(job_id)
    rig = rigging.read_rig(svc.job_dir(job_id))
    if rig is None:
        raise NotFound("job is not rigged")
    return rig


def _rig_bones(svc: WarlockService, job_id: str) -> list[str]:
    bones = rigging.rig_bone_names(svc.job_dir(job_id))
    if bones is None:
        raise NotFound("job is not rigged")
    return bones


def list_poses(svc: WarlockService, job_id: str) -> dict[str, Any]:
    check_job_id(job_id)
    bones = _rig_bones(svc, job_id)
    return {"bones": bones, "poses": rigging.list_poses(svc.job_dir(job_id))}


def save_pose(svc: WarlockService, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Create a pose, or overwrite one by passing its id."""
    check_job_id(job_id)
    known = _rig_bones(svc, job_id)
    job_dir = svc.job_dir(job_id)
    try:
        pose = rigging.validate_pose(payload, known)
    except ValueError as exc:
        raise invalid_from(exc, "That pose cannot be saved") from exc

    pose_id = str(payload["id"]) if payload.get("id") else None
    if pose_id is not None:
        check_pose_id(pose_id)
        if not rigging.pose_path(job_dir, pose_id).exists():
            raise NotFound("no such pose")
        # Under the pose's bake lock: an in-flight bake of the old rotations
        # must finish (and be deleted here) before the new rotations land, or
        # the stale GLB gets cached under this id.
        with svc.convert_lock(job_id, f"pose:{pose_id}"):
            return rigging.save_pose(job_dir, pose, pose_id)
    if len(rigging.list_poses(job_dir)) >= rigging.MAX_POSES:
        raise Conflict(f"a job may hold at most {rigging.MAX_POSES} poses")
    return rigging.save_pose(job_dir, pose, pose_id)


def delete_pose(svc: WarlockService, job_id: str, pose_id: str) -> dict[str, Any]:
    check_job_id(job_id)
    check_pose_id(pose_id)
    # Under the same lock the overwrite and the bake take: deleting while a
    # bake is in flight let the bake recreate <pose_id>.glb after its .json
    # was gone, leaving an orphan GLB nothing could ever reach or clean up.
    with svc.convert_lock(job_id, f"pose:{pose_id}"):
        if not rigging.delete_pose(svc.job_dir(job_id), pose_id):
            raise NotFound("no such pose")
    return {"ok": True}


def _pose_bake_spec(job_dir: Path, pose_id: str, pose: dict[str, Any]) -> dict[str, Any]:
    """The bake spec, with a snapshot's root translation scaled onto this rig.

    A pose applied from the global library can carry ``root_translation`` in
    character-height units; the rig's own height comes from rig.json's bounds
    and the offset lands on its root bone. Absent or zero yields today's spec
    exactly -- ``pose_spec`` adds no keys -- and a rig.json that cannot answer
    (pre-template, unreadable) costs the offset, never the bake.
    """
    root = pose.get("root_translation")
    if root and any(float(v) for v in root):
        rig = rigging.read_rig(job_dir) or {}
        bounds, bone = rig.get("bounds"), rig.get("root")
        if bounds and bone:
            return rigging.pose_spec(
                job_dir,
                pose_id,
                pose["bones"],
                root_bone=str(bone),
                root_offset=rigging.root_offset_world(root, bounds),
            )
        log.warning("pose %s carries a root offset but rig.json cannot scale it", pose_id)
    return rigging.pose_spec(job_dir, pose_id, pose["bones"])


def posed_model(svc: WarlockService, job_id: str, pose_id: str) -> Path:
    """The rig with one saved pose baked into it.

    Derived on first request and cached beside the pose, under the same
    per-artifact lock the STL/OBJ exports use -- posing runs Blender, and two
    callers asking at once should wait for one subprocess, not start two.
    """
    check_job_id(job_id)
    check_pose_id(pose_id)
    job_dir = svc.job_dir(job_id)
    path = rigging.pose_glb_path(job_dir, pose_id)
    # Existence is only checked under the lock: the bake writes the served
    # filename in place over seconds, so a lock-free exists() would let a
    # second reader open a truncated GLB mid-export. The pose is *read* under
    # the same lock for the same reason -- a delete landing between the read
    # and the bake would otherwise recreate the GLB with no .json beside it.
    with svc.convert_lock(job_id, f"pose:{pose_id}"):
        pose = rigging.read_pose(job_dir, pose_id)
        if pose is None:
            raise NotFound("no such pose")
        if not path.exists():
            if not (job_dir / "rig.glb").exists():
                raise NotFound("job is not rigged")
            spec = _pose_bake_spec(job_dir, pose_id, pose)
            try:
                rigging.run_worker(spec, timeout=svc.config.pose_timeout)
            except rigging.BlenderError as exc:
                log.error("posing %s/%s failed: %s", job_id, pose_id, exc)
                raise Failed("could not bake this pose") from exc
    return path
