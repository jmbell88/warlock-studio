"""The global pose library: CRUD, applying a pose to an asset, the preview.

The storage rules live in :mod:`warlock.poselib`; this module is what the Poser
panes and the asset Pose panel call, and it owns exactly two things the pure
half cannot: the store-wide lock, and the snapshot semantics of *applying* a
library pose to a job.

**Every library write happens under one lock** -- ``convert_lock("poser",
"store")``, a sentinel key the per-artifact table is happy to hold -- because
case-insensitive name uniqueness and the 500-pose cap are cross-file
invariants: list, check the name, check the cap and write must be one hold, or
two saves racing each other both pass the check and both land. Reads take no
lock at all; ``os.replace`` gives a concurrent reader old-or-new, never torn.

**Applying is a copy, and that is the provenance model.** ``apply_library_pose``
snapshots the record into the job's own ``poses/`` directory through the
ordinary ``rigging.save_pose`` path, so editing or deleting the library pose
afterwards can never change what an asset bakes -- immutability is automatic
rather than enforced.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .. import doctor, poselib, rigging
from .core import WarlockService
from .errors import Conflict, Failed, Invalid, NotFound, invalid_from
from .validation import check_pose_id

log = logging.getLogger(__name__)


def _record_or_not_found(svc: WarlockService, pose_id: str) -> dict[str, Any]:
    check_pose_id(pose_id)
    record = poselib.read_record(svc.config, pose_id)
    if record is None:
        raise NotFound("no such pose")
    return record


def _check_name_free(
    svc: WarlockService, template: str, name: str, exclude: str | None = None
) -> None:
    if poselib.find_name(svc.config, template, name, exclude=exclude) is not None:
        raise Conflict(f'a pose named "{name}" already exists for this skeleton', field="name")


def _check_cap(svc: WarlockService) -> None:
    if len(poselib.list_records(svc.config)) >= poselib.MAX_LIBRARY_POSES:
        raise Conflict(f"the pose library may hold at most {poselib.MAX_LIBRARY_POSES} poses")


def list_library(svc: WarlockService, template: str | None = None) -> dict[str, Any]:
    return {"poses": poselib.list_records(svc.config, template)}


def read_library_pose(svc: WarlockService, pose_id: str) -> dict[str, Any]:
    return _record_or_not_found(svc, pose_id)


def create_library_pose(svc: WarlockService, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        record = poselib.validate_record(payload)
    except ValueError as exc:
        raise invalid_from(exc, "That pose cannot be saved") from exc
    with svc.convert_lock("poser", "store"):
        _check_name_free(svc, record["template"], record["name"])
        _check_cap(svc)
        return poselib.save_record(svc.config, record)


def update_library_pose(
    svc: WarlockService, pose_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """Overwrite a pose in place. The template is immutable: a pose's bones
    are meaningless on any other skeleton, so 'changing' it is authoring a new
    pose, which is what duplicate is for."""
    existing = _record_or_not_found(svc, pose_id)
    asked = payload.get("template")
    if asked is not None and str(asked) != existing["template"]:
        raise Invalid("a pose's skeleton cannot be changed", field="template")
    merged = {
        "template": existing["template"],
        "name": payload.get("name", existing["name"]),
        "bones": payload.get("bones", existing["bones"]),
        "root_translation": payload.get(
            "root_translation", existing.get("root_translation")
        ),
    }
    try:
        record = poselib.validate_record(merged)
    except ValueError as exc:
        raise invalid_from(exc, "That pose cannot be saved") from exc
    with svc.convert_lock("poser", "store"):
        _check_name_free(svc, record["template"], record["name"], exclude=pose_id)
        return poselib.save_record(svc.config, record, pose_id)


def rename_library_pose(svc: WarlockService, pose_id: str, name: str) -> dict[str, Any]:
    return update_library_pose(svc, pose_id, {"name": name})


def duplicate_library_pose(
    svc: WarlockService, pose_id: str, name: str | None = None
) -> dict[str, Any]:
    source = _record_or_not_found(svc, pose_id)
    with svc.convert_lock("poser", "store"):
        if name is None:
            taken = [
                str(r.get("name", ""))
                for r in poselib.list_records(svc.config, source["template"])
            ]
            name = poselib.next_copy_name(str(source["name"]), taken)
        try:
            record = poselib.validate_record(dict(source, name=name))
        except ValueError as exc:
            raise invalid_from(exc, "That pose cannot be saved") from exc
        _check_name_free(svc, record["template"], record["name"])
        _check_cap(svc)
        return poselib.save_record(svc.config, record)


def delete_library_pose(svc: WarlockService, pose_id: str) -> dict[str, Any]:
    check_pose_id(pose_id)
    with svc.convert_lock("poser", "store"):
        if not poselib.delete_record(svc.config, pose_id):
            raise NotFound("no such pose")
    return {"ok": True}


def library_for_job(svc: WarlockService, job_id: str) -> dict[str, Any]:
    """The library poses applicable to one rigged job, plus which skeleton.

    What the asset Pose panel's "Library poses" section reads. The template
    comes off rig.json rather than being caller-supplied, because the file is
    the only place the selection on screen can learn how its mesh was bound --
    and a job with no rig (or a pre-template rig.json) gets an empty answer
    rather than an error, since the section simply has nothing to offer.
    """
    from .validation import check_job_id

    check_job_id(job_id)
    rig = rigging.read_rig(svc.job_dir(job_id)) or {}
    template = str(rig.get("template") or "")
    if not template:
        return {"template": None, "poses": []}
    return {"template": template, "poses": poselib.list_records(svc.config, template)}


def apply_library_pose(svc: WarlockService, job_id: str, pose_id: str) -> dict[str, Any]:
    """Snapshot a library pose into a rigged job's own poses/ directory.

    No bake happens here: the snapshot goes through ``rigging.save_pose`` like
    any hand-made pose, and the GLB derives lazily on first request through the
    untouched ``posed_model`` path. ``source_pose`` rides along as provenance
    -- which library pose, as of which edit -- and ``root_translation`` is
    copied so the bake can scale it onto this rig's own height.
    """
    svc.require_job(job_id)
    job_dir = svc.job_dir(job_id)
    rig = rigging.read_rig(job_dir)
    if rig is None:
        raise NotFound("job is not rigged")
    record = _record_or_not_found(svc, pose_id)

    if str(rig.get("template") or "") != record["template"]:
        label = rigging.get_template(record["template"]).label
        raise Conflict(f"that pose is for the {label} skeleton")

    known = [b["name"] for b in rig.get("bones", [])]
    try:
        pose = rigging.validate_pose(
            {"name": record["name"], "bones": record["bones"]}, known
        )
    except ValueError as exc:
        raise invalid_from(exc, "That pose cannot be applied") from exc

    if len(rigging.list_poses(job_dir)) >= rigging.MAX_POSES:
        raise Conflict(f"a job may hold at most {rigging.MAX_POSES} poses")
    return rigging.save_pose(
        job_dir,
        pose,
        extra={
            "root_translation": record.get("root_translation") or [0.0, 0.0, 0.0],
            "source_pose": {
                "id": record["id"],
                "name": record["name"],
                "updated": record.get("updated"),
            },
        },
    )


def template_preview(svc: WarlockService, template: str) -> Path:
    """The armature-only preview GLB for one template, built on first request.

    Cached under ``data_dir/poser/previews/`` and invalidated by the sidecar
    (template bytes, Blender version, PREVIEW_EPOCH -- see ``poselib``). The
    build lands in a temp name and is renamed in with the sidecar written
    last, the ``finalize_rig`` ordering: the sidecar is the completion marker,
    so a half-written GLB is never both present and claimed valid.
    """
    try:
        key = rigging.get_template(str(template or "")).key
    except ValueError as exc:
        raise invalid_from(exc, "That skeleton is not available", field="rig_template") from exc
    check = doctor.blender_check()
    if not check.ok:
        raise Failed(check.detail)

    path = poselib.preview_path(svc.config, key)
    with svc.convert_lock("poser", f"preview:{key}"):
        if poselib.preview_valid(svc.config, key, check.detail):
            return path
        directory = poselib.preview_dir(svc.config)
        directory.mkdir(parents=True, exist_ok=True)
        # The .glb suffix is load-bearing: Blender's exporter appends one to
        # any filepath without it (the RIG_GLB_TMP rule).
        tmp = directory / f".{key}.tmp.glb"
        spec = rigging.armature_spec(key, tmp, directory)
        try:
            rigging.run_worker(spec, timeout=svc.config.pose_timeout)
        except rigging.BlenderError as exc:
            log.error("building the %s pose preview failed: %s", key, exc)
            raise Failed("could not build the pose preview") from exc
        os.replace(tmp, path)
        poselib.write_preview_sidecar(svc.config, key, check.detail)
    return path
