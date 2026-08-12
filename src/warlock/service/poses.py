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

import contextlib
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
    """The semantic read door for every operation that *uses* a saved pose.

    Validated on the way in, not only on the way out: the library is a
    directory of files any other program can edit, and a record that had lost
    its ``bones`` reached the panes as a KeyError rather than as a refusal.

    Two callers deliberately do not come through here. ``delete_library_pose``
    must never validate -- a corrupt record that cannot be deleted is worse
    than one that cannot be applied -- and ``list_library`` must not either,
    because a record the user cannot see is a record they cannot delete.

    The validated fields overwrite the stored ones; ``created``/``updated``
    survive because they are storage's, not the payload's. ``id`` is pinned to
    the id that was addressed, which is the filename.
    """
    check_pose_id(pose_id)
    record = poselib.read_record(svc.config, pose_id)
    if record is None:
        raise NotFound("That pose is not in the library.")
    try:
        validated = poselib.validate_record(record)
    except ValueError as exc:
        raise invalid_from(exc, "That saved pose could not be read") from exc
    return dict(record, **validated, id=pose_id)


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
    # The whole read-modify-write is one hold, not just the write. A merge is
    # only meaningful against the record it merged from: reading outside the
    # lock lets a concurrent edit land between the read and the save, and
    # because the save writes the *whole* record, the fields this caller did
    # not touch are silently rolled back to what they were when it read. That
    # is the same rule merge_params follows in the job store.
    with svc.convert_lock("poser", "store"):
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
        _check_name_free(svc, record["template"], record["name"], exclude=pose_id)
        return poselib.save_record(svc.config, record, pose_id)


def rename_library_pose(svc: WarlockService, pose_id: str, name: str) -> dict[str, Any]:
    return update_library_pose(svc, pose_id, {"name": name})


def duplicate_library_pose(
    svc: WarlockService, pose_id: str, name: str | None = None
) -> dict[str, Any]:
    # Read inside the hold, for update_library_pose's reason: the copy is a
    # function of the record it was taken from, and a delete or an edit landing
    # between the read and the save would duplicate a pose that no longer
    # exists in that form.
    with svc.convert_lock("poser", "store"):
        source = _record_or_not_found(svc, pose_id)
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
    # Deliberately not through _record_or_not_found: a record too corrupt to
    # validate must still be removable, or the only way out of a bad file is a
    # file manager.
    check_pose_id(pose_id)
    with svc.convert_lock("poser", "store"):
        try:
            deleted = poselib.delete_record(svc.config, pose_id)
        except OSError as exc:
            # The wrap lives here, not in poselib: the pure half may not import
            # service, and a locked file is a real failure, not a bug report.
            log.error("deleting library pose %s failed: %s", pose_id, exc)
            raise Failed(
                "That pose could not be deleted; a file may be locked by another program."
            ) from exc
        if not deleted:
            raise NotFound("That pose is not in the library.")
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
        raise NotFound("That asset is not rigged yet.")
    record = _record_or_not_found(svc, pose_id)

    if str(rig.get("template") or "") != record["template"]:
        label = rigging.get_template(record["template"]).label
        raise Conflict(f"That pose was authored for the {label} skeleton.")

    known = [b["name"] for b in rig.get("bones", [])]
    try:
        pose = rigging.validate_pose(
            {"name": record["name"], "bones": record["bones"]}, known
        )
    except ValueError as exc:
        raise invalid_from(exc, "That pose cannot be applied") from exc

    # Under the same job-wide lock ``rig.save_pose`` takes, and for the identical
    # reason: the cap is a check-then-write, so the count and the write that
    # depends on it must happen under one hold. This is the *other* door onto
    # that write, and it held nothing -- which voided the sibling's guarantee
    # rather than merely lacking one of its own. Two callers (one from the rig
    # panel, one applying a library pose) both read ``MAX_POSES - 1`` and both
    # save, and the job ends up over its cap with nothing to notice (CON-03).
    #
    # The key is ``"poses"``, matching ``rig.save_pose`` exactly: what is guarded
    # is the *set*, not any one pose, and a different string here would be two
    # locks that never exclude each other.
    with svc.convert_lock(job_id, "poses"):
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
            os.replace(tmp, path)
        except rigging.BlenderError as exc:
            log.error("building the %s pose preview failed: %s", key, exc)
            raise Failed("could not build the pose preview") from exc
        except OSError as exc:
            # The rename is the one step after Blender succeeded, and it fails
            # for reasons Blender never sees -- an antivirus holding the temp,
            # a full disk. Same sentence: the preview is what was not built.
            log.error("renaming the %s pose preview into place failed: %s", key, exc)
            raise Failed("could not build the pose preview") from exc
        finally:
            # A Blender that died part way through leaves its staging GLB
            # behind, and this directory is never swept: previews are keyed by
            # template, so a failed build for a template that is then fixed
            # would strand one dotfile per attempt for the life of the install.
            # Inside the replace's try, not after it, so a failed rename is
            # cleaned up too.
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
        poselib.write_preview_sidecar(svc.config, key, check.detail)
    return path
