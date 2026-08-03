"""Artifact naming, gating and the two image paths in and out of a job dir."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from .errors import Invalid, NotFound, TooLarge
from .validation import MAX_IMAGE_PIXELS, MAX_THUMB_BYTES, check_job_id

# The complete artifact allowlist. It is also the export allowlist: the point
# is that a caller-supplied name never becomes a path component without
# passing through this dict first.
MEDIA = {
    "model.glb": "model/gltf-binary",
    # The trellis response model.glb is derived from, kept downloadable so a
    # user can take the full-density reconstruction if they want it.
    "source.glb": "model/gltf-binary",
    "input.png": "image/png",
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


def attach_files(job: dict[str, Any], job_dir: Path) -> None:
    # model.glb is gated on status, not just existence: the worker still
    # writes to it after the file first appears on disk (see
    # queue.py:_apply_scale), so "exists" alone can expose a half-written file
    # to a concurrent reader.
    files = []
    if (job_dir / "input.png").exists():
        files.append("input.png")
    if job["status"] == "done" and (job_dir / "model.glb").exists():
        files.append("model.glb")
    # rig.glb is gated on rig.json, not on its own existence and not on this
    # job's status. The rig lands in the *source* job's directory, so this job
    # is usually already 'done' while a separate rig job is still writing --
    # and the worker writes rig.json last, which makes it the completion
    # marker for the pair.
    if (job_dir / "rig.json").exists() and (job_dir / "rig.glb").exists():
        files.append("rig.glb")
    # Ungated on status: the thumbnail is written by the viewer after the job
    # finished, and it is complete the moment it exists.
    if (job_dir / "thumb.png").exists():
        files.append("thumb.png")
    # Written in one call by errors.write_error_log before the row is marked
    # failed, so its existence is its own completion marker. Listed so the UI
    # can offer the traceback behind the one-line message it shows.
    if (job_dir / "error.log").exists():
        files.append("error.log")
    job["files"] = files
