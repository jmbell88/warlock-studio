"""Artifact naming, gating and the two image paths in and out of a job dir."""

from __future__ import annotations

import contextlib
import os
import shutil
from pathlib import Path
from typing import Any

from .errors import Conflict, Invalid, NotFound, TooLarge
from .validation import (
    MAX_IMAGE_PIXELS,
    MAX_THUMB_BYTES,
    MAX_UPLOAD_BYTES,
    check_job_id,
)

# The complete artifact allowlist. It is also the export allowlist: the point
# is that a caller-supplied name never becomes a path component without
# passing through this dict first.
MEDIA = {
    "model.glb": "model/gltf-binary",
    # The trellis response model.glb is derived from, kept downloadable so a
    # user can take the full-density reconstruction if they want it.
    "source.glb": "model/gltf-binary",
    "input.png": "image/png",
    # The three conditioning images, which together answer "why does the mesh
    # look like that": what the user supplied, what trellis was actually
    # handed, and what the ControlNet actually saw.
    "ref.png": "image/png",
    "reference.png": "image/png",
    "control.png": "image/png",
    "model.stl": "model/stl",
    "model_obj.zip": "application/zip",
    "collision.glb": "model/gltf-binary",
    "model.fbx": "application/octet-stream",
    "textures.zip": "application/zip",
    "rig.glb": "model/gltf-binary",
    "thumb.png": "image/png",
    # The traceback errors.write_error_log already writes per job. The DB only
    # ever holds the one-line friendly sentence, so without this the actual
    # failure is on disk and unreachable from the UI.
    "error.log": "text/plain; charset=utf-8",
}

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
# An ORA is a zip; "PK\x03\x04" is as far as a magic-byte check can go, and the
# ``mimetype`` entry inside is what actually identifies one.
ORA_MAGIC = b"PK\x03\x04"

# A layered document is legitimately several times the flat image it exports:
# ten layers of a 20 MB reference is not a mistake. Still bounded, because this
# is the one path that writes an arbitrary-sized blob into a job directory.
MAX_PAINT_BYTES = 20 * MAX_UPLOAD_BYTES


class ImageTooLarge(ValueError):
    """The upload decodes to more pixels than the pipeline will accept."""


def to_png(data: bytes) -> bytes:
    """Re-encode any uploaded image as PNG; trellis.cpp only decodes PNG/JPEG.

    Alpha is preserved only when the source already had it, so a pre-matted
    upload (RGBA/LA/PA, or a palette image with a transparency entry) keeps
    its alpha for the server's bg-removal auto-detection, without forcing an
    opaque photo through the same path.
    """
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        # Checked from the header, before any pixel is decoded: a flat 20 MP
        # PNG is a few hundred KB on disk and hundreds of MB decoded, and
        # PIL's own bomb guard doesn't bite until ~178 MP.
        if im.width * im.height > MAX_IMAGE_PIXELS:
            raise ImageTooLarge(
                f"image is {im.width}x{im.height}; the limit is {MAX_IMAGE_PIXELS:,} pixels"
            )
        has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
        out = io.BytesIO()
        im.convert("RGBA" if has_alpha else "RGB").save(out, "PNG")
        return out.getvalue()


def save_thumbnail(svc: Any, job_id: str, data: bytes) -> dict[str, Any]:
    """Store a rendered preview of the mesh beside its job.

    Rendered by the viewer rather than by a pipeline: the viewport already has
    the model loaded and framed when the user first opens it, so the snapshot
    is free -- while a Blender render would need a place on the serial GPU
    queue for something purely cosmetic.

    The magic-byte check is the whole validation: this is written under a fixed
    filename inside a job directory that already exists, so the only thing
    worth refusing is a body that is not the image it claims to be.
    """
    check_job_id(job_id)
    if svc.store.get(job_id) is None:
        raise NotFound("no such job")
    if len(data) > MAX_THUMB_BYTES:
        raise TooLarge("thumbnail too large")
    if not data.startswith(PNG_MAGIC):
        raise Invalid("thumbnail must be a PNG")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "thumb.png").write_bytes(data)
    return {"ok": True}


# The untouched generated image, kept the first time a hand edit overwrites
# input.png. Deliberately absent from MEDIA and LISTED: it is an internal
# backup, never listed and never downloadable, and it goes away with the job
# directory for free.
ORIGINAL = "input.orig.png"

# The layered working file behind a hand-edited reference. Absent from MEDIA and
# LISTED for the same reason ORIGINAL is: it is internal working state, never
# served and never downloadable, and it goes away with the job directory for
# free. input.png stays the one name every consumer reads -- this only exists so
# that reopening an edited reference brings its layers back instead of a
# flattened image.
PAINT_WORKING = "paint.ora"


def paint_working_path(svc: Any, job_id: str) -> Path:
    check_job_id(job_id)
    return svc.job_dir(job_id) / PAINT_WORKING


def paint_working_status(svc: Any, job_id: str) -> dict[str, Any]:
    """Whether a layered working file exists and is newer than input.png.

    The mtime comparison is the whole rule. A revert, a regenerate or a remesh
    rewrites input.png without touching paint.ora, which would otherwise
    resurrect the layers of an image that is no longer there -- so an older
    working file is treated as stale rather than as the truth.
    """
    check_job_id(job_id)
    job_dir = svc.job_dir(job_id)
    working = job_dir / PAINT_WORKING
    flat = job_dir / "input.png"
    if not working.exists():
        return {"exists": False, "fresh": False}
    try:
        fresh = not flat.exists() or working.stat().st_mtime >= flat.stat().st_mtime
    except OSError:
        fresh = False
    return {"exists": True, "fresh": bool(fresh)}


def save_paint_working(svc: Any, job_id: str, data: bytes) -> dict[str, Any]:
    """Store the layered source beside the reference it flattens to."""
    _editable_reference(svc, job_id)
    if len(data) > MAX_PAINT_BYTES:
        raise TooLarge("layered document too large")
    if not data.startswith(ORA_MAGIC):
        raise Invalid("the layered source must be an OpenRaster file")
    dest = svc.job_dir(job_id) / PAINT_WORKING
    tmp = dest.with_suffix(".ora.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return {"ok": True}


def discard_paint_working(svc: Any, job_id: str) -> None:
    """Drop the layers. Called when a revert makes them describe pixels that
    are no longer the reference."""
    check_job_id(job_id)
    with contextlib.suppress(OSError):
        (svc.job_dir(job_id) / PAINT_WORKING).unlink(missing_ok=True)


def _editable_reference(svc: Any, job_id: str) -> tuple[dict[str, Any], Path]:
    """The gates the 2D editor and promote_to_model agree on."""
    check_job_id(job_id)
    job = svc.store.get(job_id)
    if job is None:
        raise NotFound("no such job")
    if job["stage"] != "reference":
        raise Invalid("this job is not a reference")
    if job["status"] != "done":
        raise Invalid(f"reference is {job['status']}")
    src = svc.job_dir(job_id) / "input.png"
    if not src.exists():
        raise Invalid("reference has no image")
    return job, src


def _remeasure(svc: Any, job_id: str, src: Path, *, hand_edited: bool) -> None:
    """Re-run the reference measurement over the pixels that are now on disk.

    promote_to_model refuses a reference whose stored report says it cannot
    reconstruct, and that report was measured from the *generated* pixels. An
    edit -- or a revert -- makes it a verdict about an image that no longer
    exists, so it is recomputed here. ``hand_edited`` rides along because
    ``params["recipe"]`` claims a seed and a model produced this image, which
    after an edit is no longer the whole truth.
    """
    from ..pipelines import reference

    changes: dict[str, Any] = {"reference_report": reference.measure_file(src).as_dict()}
    remove: tuple[str, ...] = ()
    if hand_edited:
        changes["hand_edited"] = True
    else:
        remove = ("hand_edited",)
    # merge_params, not set_params: this runs off the frame thread while the
    # worker may be writing other keys on the same row.
    svc.store.merge_params(job_id, changes, remove=remove)


def reference_edit_status(svc: Any, job_id: str) -> dict[str, Any]:
    """Whether this job can be opened in the 2D editor, and whether it has a
    backup to revert to. Two stats; safe from the frame thread."""
    check_job_id(job_id)
    job = svc.store.get(job_id)
    job_dir = svc.job_dir(job_id)
    editable = (
        job is not None
        and job["stage"] == "reference"
        and job["status"] == "done"
        and (job_dir / "input.png").exists()
    )
    return {"editable": bool(editable), "has_original": (job_dir / ORIGINAL).exists()}


def save_edited_image(svc: Any, job_id: str, data: bytes) -> dict[str, Any]:
    """Overwrite a reference's input.png with a hand-edited version.

    In place rather than beside, because every consumer of a reference --
    promote, remesh, export, the thumbnail -- reads that one name; a second
    "edited.png" would mean teaching all of them which to prefer. The original
    is preserved once, on the first save, so the edit is still undoable after
    the session that made it is gone.
    """
    _, dest = _editable_reference(svc, job_id)
    if len(data) > MAX_UPLOAD_BYTES:
        raise TooLarge("image too large")
    if not data.startswith(PNG_MAGIC):
        raise Invalid("the edited image must be a PNG")
    _check_pixels(data)

    original = dest.parent / ORIGINAL
    if not original.exists():
        # Once, and never clobbered: a second save must not make the *first*
        # edit the thing "Revert to original" restores.
        shutil.copyfile(dest, original)
    # Staged: promote_to_model and remesh copy input.png with a bare copyfile,
    # so a direct write_bytes onto a served name is a torn read waiting to
    # happen.
    tmp = dest.with_suffix(".png.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    _remeasure(svc, job_id, dest, hand_edited=True)
    return {"ok": True}


def revert_reference(svc: Any, job_id: str) -> dict[str, Any]:
    """Put the untouched generated image back, consuming the backup."""
    _, dest = _editable_reference(svc, job_id)
    original = dest.parent / ORIGINAL
    if not original.exists():
        raise Conflict("this reference has no unedited original")
    os.replace(original, dest)
    _remeasure(svc, job_id, dest, hand_edited=False)
    return {"ok": True}


def _check_pixels(data: bytes) -> None:
    """The header pixel cap, without decoding. Same guard to_png applies."""
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.width, im.height
    except Exception as exc:
        raise Invalid("that is not a readable image") from exc
    if width * height > MAX_IMAGE_PIXELS:
        raise TooLarge(f"image is {width}x{height}; the limit is {MAX_IMAGE_PIXELS:,} pixels")


def measure_storage(data_dir: Path) -> dict[str, Any]:
    """Total bytes and directory count under data_dir. Blocking; call off the
    frame thread."""
    total = 0
    dirs = 0
    if data_dir.exists():
        for entry in data_dir.iterdir():
            if not entry.is_dir():
                continue
            dirs += 1
            for f in entry.rglob("*"):
                # Symlinks and vanishing files (a concurrent delete) shouldn't
                # abort the whole measurement.
                with contextlib.suppress(OSError):
                    if f.is_file():
                        total += f.stat().st_size
    return {"job_dirs": dirs, "bytes": total}


# Everything that is a pure function of model.glb: derivable exactly when
# model.glb itself is ready, and never independently of it.
DERIVED = ("model.stl", "model_obj.zip", "collision.glb", "textures.zip", "model.fbx")

# The order attach_files lists them in. Derived artifacts are deliberately
# absent: they are produced on request, so listing them would claim a file that
# usually isn't on disk.
LISTED = (
    "input.png",
    "ref.png",
    "reference.png",
    "control.png",
    "model.glb",
    "source.glb",
    "rig.glb",
    "thumb.png",
    "error.log",
)


def ready(job: dict[str, Any], job_dir: Path, name: str) -> bool:
    """Whether ``name`` may be served/exported for this job. The one place the
    rules live.

    They used to be restated in five callers and had drifted apart -- the
    listing gated model.glb on status while the file route and the exporter
    served it on mere existence, which is a half-written mesh handed to a
    concurrent reader.
    """
    path = job_dir / name
    if name in ("model.glb", "source.glb"):
        # Gated on status, not just existence: the worker still writes to
        # model.glb after the file first appears (queue.py:_apply_scale), and
        # source.glb is the reconstruction the same run produced.
        return job.get("status") == "done" and path.exists()
    if name == "rig.glb":
        # Gated on rig.json, not on its own existence and not on this job's
        # status. The rig lands in the *source* job's directory, so this job is
        # usually already 'done' while a separate rig job is still writing --
        # and the worker writes rig.json last, which makes it the completion
        # marker for the pair.
        return (job_dir / "rig.json").exists() and path.exists()
    if name in DERIVED:
        # Derivable, not present: the caller still has to produce it.
        return ready(job, job_dir, "model.glb")
    # input.png, ref/reference/control.png, thumb.png and error.log are each
    # written in one call (the last three through a temp-and-rename) and are
    # complete the moment they exist -- error.log before the row is even marked
    # failed, thumb.png by the viewer long after the job finished.
    return path.exists()


def attach_files(job: dict[str, Any], job_dir: Path) -> None:
    job["files"] = [n for n in LISTED if ready(job, job_dir, n)]
