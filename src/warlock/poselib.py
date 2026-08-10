"""The global pose library: reusable poses authored in Poser, host side.

A library pose is a complete bone -> local-quaternion map authored against one
of the shipped skeleton templates, stored under ``data_dir/poser/`` rather than
in any job's directory -- it belongs to no mesh. Applying one to an asset goes
through ``service.poses.apply_library_pose``, which *snapshots* it into the
job's own ``poses/`` directory: the copy is the provenance record, so editing
or deleting the library pose afterwards can never change what an asset bakes.

Retargeting is sound because a stored quaternion is the absolute glTF
node-local rotation (``blender_worker._apply_pose`` sets
``basis = rest^-1 @ q``, so the exported node rotation is ``q`` verbatim) and a
record carries *every* bone: applying a complete map pins each bone's
orientation relative to its parent's posed frame regardless of either rig's
rest frames, leaving the target rig to contribute only translations.

Pure in the ``rigging.py`` sense: no bpy, no ``service``, no ``studio``.
Everything here is decidable with the filesystem and the template registry.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from . import rigging

log = logging.getLogger(__name__)

# The canonical armature's bounding box: the unit box fit_template normalizes
# against, so place() is the identity on the template's own landmarks and the
# preview skeleton is exactly one character-height tall. Load-bearing twice
# over -- it is what frames the Poser camera, and it is what makes a stored
# root_translation mean character-height units literally.
UNIT_LO = (-0.5, -0.5, 0.0)
UNIT_HI = (0.5, 0.5, 1.0)

# A root offset past two character heights is a fat-fingered gizmo drag, not a
# pose. Blender axes, character-height units, per component.
MAX_ROOT_TRANSLATION = 2.0

# Same ceiling and same reasoning as rigging.MAX_POSES: far above any
# hand-authored set, low enough that a scripted client cannot turn the library
# into a million files.
MAX_LIBRARY_POSES = 500

# Bumped by hand whenever blender_worker._build_armature or ._export changes in
# a way that alters the preview GLB: the sidecar records it, so every cached
# armature rebuilds on the next request rather than serving last release's
# bone frames.
PREVIEW_EPOCH = 1


class RecordError(ValueError):
    """A refused pose record, naming the field it came from.

    ``guidance.GuidanceError``'s shape for the same reason: a ``ValueError``
    subclass so nothing that already catches ``ValueError`` changes, with
    ``field`` carried so ``service.errors.invalid_from`` can pass the address
    through instead of the message arriving pointed at nothing.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


# --- paths ------------------------------------------------------------------
#
# data_dir/poser/ sits beside the job directories. A collision with a job id is
# impossible -- ids are 12 lowercase hex -- but service.files.storage_sizes
# counts it as one more "job dir", exactly as it already does autosave/; the
# miscount is a byte total under a different heading, not a correctness issue.


def poser_dir(config: Any) -> Path:
    return Path(config.data_dir) / "poser"


def library_dir(config: Any) -> Path:
    return poser_dir(config) / "poses"


def preview_dir(config: Any) -> Path:
    return poser_dir(config) / "previews"


def pose_path(config: Any, pose_id: str) -> Path:
    """The record for one library pose. Raises on an id that isn't ours.

    The ``rigging.pose_path`` rule: this is the only place a caller-supplied
    pose id becomes a path, and ``dir / pose_id`` does no sanitising of its own.
    """
    if not rigging.is_valid_id(pose_id):
        raise ValueError(f"malformed pose id {pose_id!r}")
    return library_dir(config) / f"{pose_id}.json"


def preview_path(config: Any, template_key: str) -> Path:
    """The cached armature-only preview GLB for one template."""
    rigging.get_template(template_key)  # the registry is the path sanitiser
    return preview_dir(config) / f"{template_key}.glb"


def preview_sidecar_path(config: Any, template_key: str) -> Path:
    return preview_path(config, template_key).with_suffix(".json")


# --- records ----------------------------------------------------------------


def validate_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a pose-record payload, or raise :class:`RecordError`.

    Everything a record may carry, checked in the order a refusal is most
    useful: the template (nothing else means anything without it), the name,
    the bones against that template's skeleton, then the root translation --
    finite-checked *before* range-checked, because ``abs(nan) <= 2.0`` is
    False and the range message would then blame a value that is not a number.
    """
    template_key = str(payload.get("template") or "")
    try:
        template = rigging.get_template(template_key)
    except ValueError as exc:
        raise RecordError(str(exc), field="template") from None

    name = str(payload.get("name") or "").strip()
    if not name:
        raise RecordError("pose requires a name", field="name")
    if len(name) > rigging.MAX_POSE_NAME:
        raise RecordError(
            f"pose name must be at most {rigging.MAX_POSE_NAME} characters", field="name"
        )

    known = [b["name"] for b in template.bones]
    try:
        pose = rigging.validate_pose({"name": name, "bones": payload.get("bones")}, known)
    except ValueError as exc:
        raise RecordError(str(exc)) from None
    # Exactly the template's bone set, not a subset: the module docstring's
    # retargeting argument stands on a record carrying *every* bone, and a
    # partial record applied over a posed editor keeps whatever the missing
    # bones were doing. Poser's own saves are ``get_pose()``, which is always
    # complete, so this refuses only a hand-built payload.
    missing = [b for b in known if b not in pose["bones"]]
    if missing:
        raise RecordError(
            f"pose is missing {len(missing)} of this skeleton's {len(known)} bones",
            field="bones",
        )

    raw = payload.get("root_translation")
    if raw is None:
        # Absent means "no offset", never an error: every record written before
        # the field existed reads as zeros, and so does a pose that never
        # touched the root.
        root = [0.0, 0.0, 0.0]
    else:
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise RecordError("root_translation must be a 3-vector", field="root_translation")
        try:
            root = [float(v) for v in raw]
        except (TypeError, ValueError):
            raise RecordError(
                "root_translation is not numeric", field="root_translation"
            ) from None
        if not all(math.isfinite(v) for v in root):
            raise RecordError("root_translation is not numeric", field="root_translation")
        if any(abs(v) > MAX_ROOT_TRANSLATION for v in root):
            raise RecordError(
                f"root_translation components must be within "
                f"+/-{MAX_ROOT_TRANSLATION} character heights",
                field="root_translation",
            )

    return {
        "name": pose["name"],
        "template": template.key,
        "bones": pose["bones"],
        "root_translation": root,
    }


def save_record(
    config: Any, record: dict[str, Any], pose_id: str | None = None
) -> dict[str, Any]:
    """Write a validated record and return what was stored.

    Passing an existing ``pose_id`` overwrites in place, keeping ``created``
    and bumping ``updated`` -- the same edit-vs-accumulate rule
    ``rigging.save_pose`` follows. The write is staged and renamed for the same
    reason too: a save that died mid-write must not truncate the only copy.
    """
    pose_id = pose_id or rigging.new_id()
    path = pose_path(config, pose_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_record(config, pose_id) if path.exists() else None
    now = time.time()
    stored = {
        "version": 1,
        "id": pose_id,
        "name": record["name"],
        "template": record["template"],
        "bones": record["bones"],
        "root_translation": record.get("root_translation") or [0.0, 0.0, 0.0],
        "created": existing.get("created", now) if existing else now,
        "updated": now,
    }
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{pose_id}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(stored, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return stored


def read_record(config: Any, pose_id: str) -> dict[str, Any] | None:
    path = pose_path(config, pose_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.exception("unreadable library pose at %s", path)
        return None


def list_records(config: Any, template: str | None = None) -> list[dict[str, Any]]:
    """Every library pose, sorted by name. A corrupt file costs itself, not
    the list -- the ``rigging.list_poses`` precedent."""
    directory = library_dir(config)
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.json")):
        if not rigging.is_valid_id(path.stem):
            continue
        record = read_record(config, path.stem)
        if record is None:
            continue
        if template is not None and record.get("template") != template:
            continue
        records.append(record)
    # Case-insensitive by name so "crouch" and "Crouch low" sort together, with
    # the id as the final tie-break so the order is total and stable.
    records.sort(key=lambda r: (str(r.get("name", "")).casefold(), str(r.get("id", ""))))
    return records


def delete_record(config: Any, pose_id: str) -> bool:
    path = pose_path(config, pose_id)
    if not path.exists():
        return False
    path.unlink()
    return True


def find_name(
    config: Any, template: str, name: str, exclude: str | None = None
) -> str | None:
    """The id of an existing pose on this template with this name, or None.

    Case-insensitive via ``casefold`` and nothing more: nothing in this repo
    normalizes Unicode, and an NFC-vs-NFD near-miss slipping past is a
    duplicate name, not a corrupted library. ``exclude`` is how a rename or an
    overwrite skips colliding with itself.
    """
    wanted = name.strip().casefold()
    for record in list_records(config, template):
        if record.get("id") == exclude:
            continue
        if str(record.get("name", "")).strip().casefold() == wanted:
            return str(record["id"])
    return None


# The clay/ops._next_name idiom, copied rather than imported: this module (and
# the service layer above it) may not import studio, and the four lines are one
# judgement -- count up, never prefix -- that both sides state independently.
_SUFFIX = re.compile(r"^(?P<base>.*)\.(?P<n>\d{3})$")


def next_copy_name(name: str, taken: Any = ()) -> str:
    """``Crouch`` -> ``Crouch.001`` -> ``Crouch.002``, skipping names in use."""
    used = {str(t).strip().casefold() for t in taken}
    match = _SUFFIX.match(name)
    base = match.group("base") if match else name
    start = int(match.group("n")) if match else 0
    for n in range(start + 1, start + len(used) + 3):
        candidate = f"{base}.{n:03d}"
        if candidate.strip().casefold() not in used:
            return candidate
    return f"{base}.{start + len(used) + 3:03d}"


def mirror_root_translation(v: Any) -> list[float]:
    """Reflect a root offset across the subject's YZ plane: (x, y, z) -> (-x, y, z).

    The positional half of ``rigging.mirror_quaternion``'s reflection --
    Blender axes, X the mirror normal -- so mirroring a pose that steps left
    steps right.
    """
    x, y, z = (float(c) for c in v)
    return [-x, y, z]


# --- the preview cache ------------------------------------------------------
#
# One armature-only GLB per template, built by the same Blender code path as a
# real rig (op_armature) so its bone frames can never disagree with a bake.
# Cached because building one is a Blender subprocess; the sidecar is the
# completion marker (written last, the finalize_rig ordering) and the whole of
# the invalidation rule.


def template_digest(template_key: str) -> str | None:
    """SHA-256 of the template file's bytes, or None when it cannot be read.

    File bytes rather than the parsed structure: the file is what op_armature's
    fit reads through the registry, and hashing the source is the cheapest
    answer that can never miss a change. Template files are named by their key
    -- enforced at registry load (``rigging._load_templates``), not merely a
    convention. None rather than a raise for a file the cached registry still
    names but the disk no longer holds: the registry loads once per process,
    so the file can vanish underneath it, and a cache-validity question must
    answer "stale", never throw out of a preview request.
    """
    rigging.get_template(template_key)
    path = rigging.TEMPLATE_DIR / f"{template_key}.json"
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        log.exception("unreadable skeleton template at %s", path)
        return None


def read_preview_sidecar(config: Any, template_key: str) -> dict[str, Any] | None:
    path = preview_sidecar_path(config, template_key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.exception("unreadable preview sidecar at %s", path)
        return None


def preview_valid(config: Any, template_key: str, blender_version: str) -> bool:
    """Whether the cached preview GLB still answers for this template.

    Three ways to go stale, each checked: the template file changed (the
    skeleton is different), Blender changed (its exporter may emit different
    frames), or PREVIEW_EPOCH was bumped (our own armature/export code did).
    A missing GLB fails regardless of what the sidecar claims.
    """
    if not preview_path(config, template_key).exists():
        return False
    sidecar = read_preview_sidecar(config, template_key)
    if sidecar is None:
        return False
    digest = template_digest(template_key)
    if digest is None:
        # An unreadable template can neither confirm nor deny the cache; and
        # a sidecar that recorded None must not match it -- both answers here
        # are "rebuild", never a raise out of template_preview.
        return False
    return (
        sidecar.get("template_sha256") == digest
        and sidecar.get("blender_version") == blender_version
        and sidecar.get("preview_epoch") == PREVIEW_EPOCH
    )


def write_preview_sidecar(config: Any, template_key: str, blender_version: str) -> None:
    """Written last, after the GLB is in place -- it is the completion marker."""
    path = preview_sidecar_path(config, template_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "template_sha256": template_digest(template_key),
        "blender_version": blender_version,
        "preview_epoch": PREVIEW_EPOCH,
    }
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{template_key}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
