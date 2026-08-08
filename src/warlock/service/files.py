"""Artifact naming, gating and the two image paths in and out of a job dir."""

from __future__ import annotations

import contextlib
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .errors import Conflict, Invalid, NotFound, TooLarge
from .validation import (
    MAX_IMAGE_PIXELS,
    MAX_THUMB_BYTES,
    MAX_UPLOAD_BYTES,
    check_job_id,
    not_done_message,
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
    # The deformation battery rendered against the rig: the poses a rig is
    # reviewed in, one atlas, written by the rig job into the mesh's own
    # directory the way rig.glb is. Its sidecar is deliberately not here --
    # nothing outside the app reads it, and MEDIA is an export allowlist.
    "rig_qa.png": "image/png",
    "thumb.png": "image/png",
    # The 2D exports. Derived from input.png on a finished reference exactly
    # the way the mesh exports derive from model.glb -- so every reference
    # already on disk gains them, which is the whole reason they are derived
    # rather than produced by a second kind of job.
    #
    # Each pixel size is its own literal name because MEDIA is the allowlist
    # that keeps a caller-supplied string off the filesystem: a pixel_{n}.png
    # pattern would put the number back in the caller's hands.
    "icon.png": "image/png",
    "sprite.png": "image/png",
    "pixel_32.png": "image/png",
    "pixel_64.png": "image/png",
    "pixel_128.png": "image/png",
    # A tile's own export, and the only one the cutouts are replaced by: the
    # texture rolled by half in both axes, so what was the wrap seam runs
    # through the middle of the frame where a discontinuity is visible.
    "wrap_preview.png": "image/png",
    "manifest.json": "application/json",
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
MAX_INKER_BYTES = 20 * MAX_UPLOAD_BYTES


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
#
# The filename keeps its pre-rename spelling on purpose: it is a compatibility
# contract with every asset directory already on disk, and the mtime-staleness
# rule below only works while there is exactly one name to compare against.
INKER_WORKING = "paint.ora"


def inker_working_path(svc: Any, job_id: str) -> Path:
    check_job_id(job_id)
    return svc.job_dir(job_id) / INKER_WORKING


def inker_working_status(svc: Any, job_id: str) -> dict[str, Any]:
    """Whether a layered working file exists and is newer than input.png.

    The mtime comparison is the whole rule. A revert, a regenerate or a remesh
    rewrites input.png without touching paint.ora, which would otherwise
    resurrect the layers of an image that is no longer there -- so an older
    working file is treated as stale rather than as the truth.
    """
    check_job_id(job_id)
    job_dir = svc.job_dir(job_id)
    working = job_dir / INKER_WORKING
    flat = job_dir / "input.png"
    if not working.exists():
        return {"exists": False, "fresh": False}
    try:
        fresh = not flat.exists() or working.stat().st_mtime >= flat.stat().st_mtime
    except OSError:
        fresh = False
    return {"exists": True, "fresh": bool(fresh)}


def save_inker_working(svc: Any, job_id: str, data: bytes) -> dict[str, Any]:
    """Store the layered source beside the reference it flattens to."""
    _editable_reference(svc, job_id)
    if len(data) > MAX_INKER_BYTES:
        raise TooLarge("layered document too large")
    if not data.startswith(ORA_MAGIC):
        raise Invalid("the layered source must be an OpenRaster file")
    dest = svc.job_dir(job_id) / INKER_WORKING
    tmp = dest.with_suffix(".ora.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return {"ok": True}


def discard_inker_working(svc: Any, job_id: str) -> None:
    """Drop the layers. Called when a revert makes them describe pixels that
    are no longer the reference."""
    check_job_id(job_id)
    with contextlib.suppress(OSError):
        (svc.job_dir(job_id) / INKER_WORKING).unlink(missing_ok=True)


# The authored Clay document behind a built asset, following the paint.ora
# precedent exactly: absent from MEDIA and LISTED, never served, never
# downloadable, and it goes away with the job directory for free. model.glb
# stays the one name every consumer reads; this exists only so that reopening a
# built asset brings its objects back instead of a single frozen mesh.
CLAY_SOURCE = "build.wblk"  # the on-disk name predates the Clay rename

# A .wblk is scene.json plus one npz per object. Geometry compresses, and Phase
# 1 documents are a handful of primitives -- but a Phase 3 subdivided scene is
# the case this bounds, and it is bounded on the same reasoning as every other
# ceiling here rather than left open because today's files are small.
MAX_CLAY_SOURCE_BYTES = 5 * MAX_UPLOAD_BYTES


def clay_source_path(svc: Any, job_id: str) -> Path:
    check_job_id(job_id)
    return svc.job_dir(job_id) / CLAY_SOURCE


def clay_source_status(svc: Any, job_id: str) -> dict[str, Any]:
    """Whether an authored source exists for this asset.

    **No staleness rule**, unlike :func:`inker_working_status`, and the
    difference is not an oversight. That rule exists because a revert, a
    regenerate or a remesh rewrites ``input.png`` behind the layers, leaving
    them describing an image that is gone. Nothing does that here: a build
    export writes the GLB and the ``.wblk`` in one operation, and the one thing
    that later rewrites ``model.glb`` -- a triangle retarget -- does not make
    the authored document wrong. It makes the served mesh a *derivative* of it,
    which is the whole point of keeping the source.
    """
    check_job_id(job_id)
    return {"exists": (svc.job_dir(job_id) / CLAY_SOURCE).exists()}


def save_clay_source(svc: Any, job_id: str, data: bytes) -> dict[str, Any]:
    """Store the authored document beside the mesh it exported to.

    Written through a temp and ``os.replace``, as the layered source is: the
    file is read whole by the reader and a torn one is a document that will not
    open. The caller writes the GLB first and this second, so a crash between
    the two leaves the sidecar absent rather than lying about a mesh it did not
    produce.
    """
    check_job_id(job_id)
    if svc.store.get(job_id) is None:
        raise NotFound("no such job")
    if len(data) > MAX_CLAY_SOURCE_BYTES:
        raise TooLarge("clay document too large")
    if not data.startswith(ORA_MAGIC):
        # The same four bytes: a .wblk is a zip, as an .ora is.
        raise Invalid("the clay source must be a .wblk archive")
    dest = svc.job_dir(job_id) / CLAY_SOURCE
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".wblk.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return {"ok": True}


def _editable_reference(svc: Any, job_id: str) -> tuple[dict[str, Any], Path]:
    """The gates the 2D editor and promote_to_model agree on."""
    check_job_id(job_id)
    job = svc.store.get(job_id)
    if job is None:
        raise NotFound("no such job")
    if job["stage"] != "reference":
        raise Invalid("this job is not a reference")
    if job["status"] != "done":
        raise Invalid(not_done_message("That reference", job["status"]))
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
    # Touched, because a restore is the one write here that would otherwise
    # arrive wearing an *older* timestamp than the pixels it replaces: the
    # backup was copied when the first edit was made, and shutil.copyfile does
    # not preserve mtimes, so the restored file carries that moment rather than
    # this one. fresh_2d compares every derived export against this mtime, so
    # without the touch a revert would leave the exports of the edit looking
    # current -- the exact staleness the comparison exists to catch, in the
    # only direction where the content changes and the clock goes backwards.
    os.utime(dest)
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
        # The formats, named (E53). "That is not a readable image" is true of a
        # PDF, a .psd, a truncated download and a file the user renamed to
        # .png, and it tells apart none of them -- while the one thing the user
        # can act on is knowing what would have worked. Spelled from the same
        # list the file picker offers, so the dialog and the refusal agree.
        raise Invalid(
            "That is not a readable image. PNG, JPEG, WebP and BMP are accepted."
        ) from exc
    if width * height > MAX_IMAGE_PIXELS:
        raise TooLarge(f"image is {width}x{height}; the limit is {MAX_IMAGE_PIXELS:,} pixels")


def dir_size(path: Path) -> int:
    """One directory's recursive byte count; 0 when it does not exist."""
    total = 0
    try:
        walk = path.rglob("*")
    except OSError:
        return 0
    for f in walk:
        # Symlinks and vanishing files (a concurrent delete) shouldn't
        # abort the whole measurement.
        with contextlib.suppress(OSError):
            if f.is_file():
                total += f.stat().st_size
    return total


def storage_sizes(data_dir: Path) -> dict[str, int]:
    """``{directory name: bytes}`` for every job directory under data_dir.

    The decomposed form of :func:`measure_storage`, kept so an incremental
    accounting (a job finishing re-measures its own directory only) shares
    the walk with the full one rather than restating it.
    """
    sizes: dict[str, int] = {}
    if data_dir.exists():
        for entry in data_dir.iterdir():
            if entry.is_dir():
                sizes[entry.name] = dir_size(entry)
    return sizes


def measure_storage(data_dir: Path) -> dict[str, Any]:
    """Total bytes and directory count under data_dir. Blocking; call off the
    frame thread."""
    sizes = storage_sizes(data_dir)
    return {"job_dirs": len(sizes), "bytes": sum(sizes.values())}


# Everything that is a pure function of model.glb: derivable exactly when
# model.glb itself is ready, and never independently of it.
DERIVED = ("model.stl", "model_obj.zip", "collision.glb", "textures.zip", "model.fbx")

# Everything that is a pure function of a *reference's* input.png. Kept apart
# from DERIVED rather than merged into it: the two sets have different sources,
# different readiness rules and different jobs they apply to, and one tuple
# would have to be filtered at every use anyway.
REFERENCE_2D = (
    "icon.png",
    "sprite.png",
    "pixel_32.png",
    "pixel_64.png",
    "pixel_128.png",
    "manifest.json",
)

# And what a *tile's* input.png can produce, which is deliberately almost none
# of the above. Every cutout is the operation of lifting a subject off its
# background, and a seamless texture is background: an icon of one is the whole
# frame with a matte guessed over it, and a sprite of one is a trim box around
# nothing. What a tile has instead is the wrapped view -- the only export that
# says something true about it that the PNG itself does not.
TILE_2D = ("wrap_preview.png", "manifest.json")

# The union: what ``fresh_2d`` and ``derivable_2d`` answer about a *name*,
# independently of the job asking. The per-stage split above is what decides
# whether a given job may ask.
# Composed rather than written out, and that is not tidiness: the two halves
# overlap in manifest.json and a hand-written union is only ever right by
# coincidence. A tile-only artifact added to TILE_2D and missed here would be
# ``ready`` -- derived_2d_for says so -- while get_file's derivation gate and
# fresh_2d, which both key on this tuple, would never produce it: an enabled
# button that answers NotReady for ever.
DERIVED_2D = REFERENCE_2D + tuple(n for n in TILE_2D if n not in REFERENCE_2D)


def derived_2d_for(stage: str | None) -> tuple[str, ...]:
    """Which 2D exports this stage's input.png can produce.

    One function rather than a condition restated wherever the question comes
    up: it is asked by ``ready``, by the Export tab's grid and by the pane's
    copy of the derivability rule, and those three drifting apart is a button
    that lights up and then produces an error toast.
    """
    if stage == "tile":
        return TILE_2D
    if stage == "reference":
        return REFERENCE_2D
    return ()

# Which pixel-art size each artifact name means. The names are literals for the
# allowlist's sake; this is where they get their number back.
PIXEL_ARTIFACTS = {"pixel_32.png": 32, "pixel_64.png": 64, "pixel_128.png": 128}

# The palette caps a pixel artifact may be quantized to (0 = no cap). One
# source for the service's validation and the inspector's combo, like
# ALLOWED_RESOLUTIONS.
PIXEL_COLOR_CHOICES = (0, 8, 16, 32, 64)

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
    "rig_qa.png",
    "thumb.png",
    "error.log",
)


def fresh_2d(job_dir: Path, name: str) -> bool:
    """Whether a derived 2D artifact still describes the input.png on disk.

    An mtime comparison, and deliberately the same idiom -- for the same
    reason -- as ``inker_working_status``'s. ``input.png`` has several writers:
    ``save_edited_image``, ``revert_reference``, and the Inker's linked save,
    which writes input.png *first* precisely so that this comparison decides
    staleness. Invalidating by unlinking beside each of them is a rule every
    writer added later has to remember, and the one that forgets serves an icon
    of pixels that no longer exist -- forever, because ``get_file`` caches on
    existence. A file older than its source cannot be forgotten about, because
    the question is asked at the only moment that matters: when somebody wants
    to serve it.

    A name that is not derived from input.png is always fresh. This is a
    freshness rule and not an existence check, so any caller may ask it of any
    name -- but for a 2D artifact "does not exist" and "is stale" are the same
    answer, which is what lets ``get_file`` treat both as "derive it".
    """
    if name not in DERIVED_2D:
        return True
    path = job_dir / name
    source = job_dir / "input.png"
    try:
        if not path.exists():
            return False
        if not source.exists():
            # Nothing left to be stale against. ``ready`` has already refused
            # the whole set when input.png is missing, so this is reachable
            # only in a race, and the artifact is the better answer than a
            # spurious re-derivation that has no source to read.
            return True
        return path.stat().st_mtime_ns >= source.stat().st_mtime_ns
    except OSError:
        return False


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
    if name == "rig_qa.png":
        # Gated on its sidecar for the reason rig.glb is gated on rig.json: the
        # atlas is written first and the sidecar last, by a *different* job
        # than the one this directory belongs to, so existence alone can hand a
        # reader an atlas that is still being packed.
        return (job_dir / "rig_qa.json").exists() and path.exists()
    if name == "rig.glb":
        # Gated on rig.json, not on its own existence and not on this job's
        # status. The rig lands in the *source* job's directory, so this job is
        # usually already 'done' while a separate rig job is still writing --
        # and the worker writes rig.json last, which makes it the completion
        # marker for the pair.
        return (job_dir / "rig.json").exists() and path.exists()
    if name in DERIVED_2D:
        # A reference's or a tile's pixels, and only those: a mesh job's
        # input.png is the picture it was reconstructed *from*, so an icon
        # derived from it would quietly claim to be an export of the mesh. The
        # two image stages take different halves of the set -- see
        # ``derived_2d_for`` -- so the stage decides the name as well as the
        # permission.
        return (
            name in derived_2d_for(job.get("stage"))
            and job.get("status") == "done"
            and (job_dir / "input.png").exists()
        )
    if name in DERIVED:
        # Derivable, not present: the caller still has to produce it.
        return ready(job, job_dir, "model.glb")
    # input.png, ref/reference/control.png, thumb.png and error.log are each
    # written in one call (the last three through a temp-and-rename) and are
    # complete the moment they exist -- error.log before the row is even marked
    # failed, thumb.png by the viewer long after the job finished.
    return path.exists()


def unready_reason(job: dict[str, Any], job_dir: Path, name: str) -> str:
    """Why ``name`` is not servable, as a sentence (E51).

    Explanatory only. :func:`ready` stays the gate and the authority -- this is
    called on the refusal path, having already been refused, so a sentence that
    disagreed with it would be a slightly wrong explanation rather than a wrong
    permission. Written as a separate function for exactly that reason: folding
    it into ``ready`` would put string formatting inside the per-row, per-name
    listing loop that ``attach_files`` exists to keep cheap.

    "file not ready" was the whole message before, on all three refusal paths at
    once, and it is the least actionable sentence in the app: the job is queued,
    the mesh is still being written, the rig has not been made, the export has
    to be derived first and the model has no textures to derive it from are five
    different situations with five different responses.
    """
    status = job.get("status")
    derived_from_a_run = (
        name in ("model.glb", "source.glb") or name in DERIVED or name in DERIVED_2D
    )
    if status in ("queued", "running", "error", "cancelled") and derived_from_a_run:
        return not_done_message(f"{name} is not available yet: this job", str(status))
    if name == "rig.glb" and not (job_dir / "rig.json").exists():
        return "This asset has not been rigged yet."
    if name == "rig_qa.png" and not (job_dir / "rig_qa.json").exists():
        return "The deformation sheet is only made when a rig is."
    if name in DERIVED_2D and name not in derived_2d_for(job.get("stage")):
        return f"{name} is only offered for a reference or a tile, not for a mesh."
    if name in DERIVED_2D and not (job_dir / "input.png").exists():
        return f"{name} is derived from the reference image, which is not on disk."
    if name in DERIVED and not (job_dir / "model.glb").exists():
        return f"{name} is derived from the mesh, which is not on disk."
    return f"{name} is not on disk for this job."


# How old a job directory's mtime must be before that mtime is trusted as a
# cache key. Above Windows' 15.6 ms default system-clock tick with room to
# spare, and a tenth of the 500 ms list refresh, so at most one extra listing
# per write. See the racily-clean paragraph in ``attach_files``.
MTIME_RACE_NS = 50_000_000


def attach_files(job: dict[str, Any], job_dir: Path, *, cache: dict | None = None) -> None:
    """List which of ``LISTED`` this job actually has.

    ``cache`` is an optional ``{job_id: (stamp, names)}`` the caller owns, and
    it exists because this is the frame loop's single largest syscall cost:
    ``LISTED`` is ten names and ``ready`` stats one or two files for each, so
    a two-hundred row page costs upwards of two thousand ``stat`` calls -- twice
    a second, on the thread that must not block, growing without limit as
    "load more" widens the window.

    The stamp is ``(status, the job directory's own mtime)``. Sound because
    every name here is answered by *existence*: a file appearing or being
    removed adds or removes a directory entry, which is what moves a
    directory's mtime -- and a status change is the other thing that can change
    the answer (a queued job's ``model.glb`` is not servable). One stat per row
    instead of ten. The names are copied out, because a caller that edits
    ``job["files"]`` must not edit what the next tick will hand somebody else.

    **But a directory's mtime has a resolution, and it is coarse.** Windows
    updates it from the system clock, whose tick is 15.6 ms unless something has
    asked for better; measured here, adding a file left the mtime *unchanged*
    155 times out of 200 (see ``docs/measurements/2026-08-07-directory-mtime-
    granularity.md``). So a write that lands after this listing but still inside
    the stamped mtime's tick is invisible to the stamp -- and not for one tick,
    but **forever**, because every later comparison keeps matching. That is
    exactly the case the second half of the stamp exists for: a rig lands in the
    *source* job's directory while that job stays ``done``.

    The answer is git's racily-clean rule. A stamp is only stored once its mtime
    is comfortably in the past; a directory touched moments ago is answered
    correctly and simply not remembered, so the next tick asks the disk again.
    **The clock is read after the listing, not before, and that ordering is the
    proof**: the hazard needs a write later than our listing yet still in the
    mtime's tick, so if the listing itself already finished more than one tick
    after the mtime, no such write exists. Costs one extra listing per write, on
    one row, since ticks are 500 ms apart and the window is 50.
    """
    if cache is None:
        job["files"] = [n for n in LISTED if ready(job, job_dir, n)]
        return
    try:
        stamp: tuple[Any, int | None] = (job.get("status"), job_dir.stat().st_mtime_ns)
    except OSError:
        # No directory at all yet -- a text job has none until the worker
        # writes into it. That is a perfectly good stamp of its own: the answer
        # is "no files", and the moment a directory appears the stamp changes.
        # Cached rather than fallen through, because a queued job is exactly
        # the row that sits on screen being re-listed twice a second.
        stamp = (job.get("status"), None)
    hit = cache.get(job["id"])
    if hit is not None and hit[0] == stamp:
        job["files"] = list(hit[1])
        return
    names = [n for n in LISTED if ready(job, job_dir, n)]
    job["files"] = list(names)
    # None races with nothing: a directory appearing changes the stamp from
    # None to a number whatever the clock did, so that one is always storable.
    # A backwards clock step makes the difference negative, which declines to
    # cache -- slower, never wrong, which is the right way round.
    if stamp[1] is None or time.time_ns() - stamp[1] > MTIME_RACE_NS:
        cache[job["id"]] = (stamp, names)
