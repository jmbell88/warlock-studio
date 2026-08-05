"""The lazy-derivation engine behind every artifact download.

Everything but the GLB, the reference PNG and the rig is a pure function of
``model.glb``, produced on first request and cached beside it. One lock per
(job, artifact) means two callers asking at once wait for one conversion
rather than starting two -- which for the FBX path is two Blender subprocesses
writing the same filename.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .. import rigging
from . import files
from .core import WarlockService
from .errors import Failed, NotFound, NotReady
from .files import MEDIA

log = logging.getLogger(__name__)


def get_file(svc: WarlockService, job_id: str, name: str) -> Path:
    """The path of one artifact, deriving it first if that is possible.

    Blocking: a cold STL of a 300k-face mesh is seconds and an FBX is a Blender
    subprocess, so this must never be called from the frame thread.
    """
    if name not in MEDIA:
        raise NotFound("unknown file")
    # The row is fetched, not just the id checked: readiness is a fact about
    # the job (a mesh is only servable once it is *done*), and an orphaned
    # directory used to serve files for a job that no longer exists.
    job = svc.require_job(job_id)
    job_dir = svc.job_dir(job_id)
    path = job_dir / name
    glb = job_dir / "model.glb"
    if not files.ready(job, job_dir, name):
        raise NotReady("file not ready")
    # FBX needs a Blender subprocess rather than a trimesh call, so it does not
    # fit the `derived` dict below -- but it takes the same per-artifact lock,
    # for the same reason. Existence checked only under the lock: Blender
    # writes the served filename in place, so a lock-free exists() during the
    # first export would open a truncated FBX. The trimesh conversions below
    # don't need this -- postprocess stages them and renames atomically.
    if name == "model.fbx" and glb.exists():
        with svc.convert_lock(job_id, name):
            if not path.exists():
                spec = rigging.fbx_spec(glb, path, job_dir)
                try:
                    # FBX export is import-plus-export like a pose bake, so it
                    # reuses pose_timeout rather than gaining a knob.
                    rigging.run_worker(spec, timeout=svc.config.pose_timeout)
                except rigging.BlenderError as exc:
                    log.error("fbx export for %s failed: %s", job_id, exc)
                    raise Failed("could not export FBX") from exc

    from ..pipelines import postprocess

    # Everything that is a pure function of model.glb, derived on first request
    # and cached. Each entry takes (glb, out_path).
    derived = {
        "model.stl": postprocess.glb_to_stl,
        "model_obj.zip": postprocess.glb_to_obj_zip,
        "collision.glb": postprocess.glb_to_collision,
        "textures.zip": postprocess.glb_to_textures_zip,
    }
    if not path.exists() and name in derived and glb.exists():
        convert = derived[name]
        with svc.convert_lock(job_id, name):
            # Re-checked inside the lock: whoever waited here was waiting for
            # exactly this file, so the second caller gets the finished
            # artifact instead of exporting a 300K-face mesh again.
            if not path.exists():
                try:
                    convert(glb, path)
                except ValueError as exc:
                    # "this model has no textures" is a fact about the mesh,
                    # not a fault -- the artifact simply cannot exist.
                    raise NotReady(str(exc)) from exc
    if not path.exists() and name in files.DERIVED_2D:
        _derive_2d(svc, job, job_id, job_dir, name)
    if not path.exists():
        raise NotReady("file not ready")
    return path


def derivable(name: str) -> bool:
    """Whether ``name`` can be produced from model.glb on demand.

    The UI uses this to decide between a save button and a "derive, then save"
    one, without having to know which conversions exist.
    """
    return name in files.DERIVED


def derivable_2d(name: str) -> bool:
    """Whether ``name`` can be produced from a reference's input.png."""
    return name in files.DERIVED_2D


# The manifest is written under its own lock, always taken *inside* the
# artifact's. Two derivations of different artifacts genuinely race for it, and
# a consistent order is the only thing standing between that and a deadlock --
# nothing anywhere may take the manifest lock first.
MANIFEST = "manifest.json"


def _derive_2d(svc: WarlockService, job: dict, job_id: str, job_dir: Path, name: str) -> None:
    """Produce one 2D export from the reference's input.png.

    Blocking, like every other derivation here: the matte is a model or a
    flood fill and the quantize is real work, so this must never be called
    from the frame thread.
    """
    from PIL import Image

    from ..pipelines import asset2d, matting

    source = job_dir / "input.png"
    if not source.exists():
        raise NotReady("this job has no reference image")
    if name == MANIFEST:
        # Nothing to compute: the manifest is written by the artifacts
        # themselves, so asking for it before any of them exist means writing
        # the header alone. Deliberately *without* an enclosing artifact lock
        # of its own -- the manifest's artifact lock and its manifest lock are
        # the same lock, and convert_lock hands out a plain, non-reentrant
        # threading.Lock, so wrapping the call would deadlock against itself.
        # _write_manifest takes it, and the ordering rule is untouched: the
        # manifest lock is still the innermost thing anyone holds.
        _write_manifest(svc, job, job_id, job_dir, None, None)
        return
    with svc.convert_lock(job_id, name):
        # Re-checked inside the lock, for the reason the mesh exports give:
        # whoever waited here was waiting for exactly this file.
        if (job_dir / name).exists():
            return
        with Image.open(source) as image:
            image.load()
            mask, matte = matting.mask(image, svc.config)
            try:
                if name == "icon.png":
                    out, meta = asset2d.icon(image, mask)
                elif name == "sprite.png":
                    out, meta = asset2d.sprite(image, mask)
                else:
                    out, meta = asset2d.pixel(image, mask, size=files.PIXEL_ARTIFACTS[name])
            except asset2d.NoSubject as exc:
                # A fact about the image, not a fault -- the same shape
                # glb_to_textures_zip's "this model has no textures" takes.
                raise NotReady(str(exc)) from exc
        meta["matte"] = matte
        meta["alpha"] = asset2d.alpha_report(out)
        # Staged and renamed: a concurrent reader of an artifact this job
        # derived a moment ago must never see a half-written PNG.
        tmp = job_dir / f".{name}.tmp"
        out.save(tmp, "PNG")
        os.replace(tmp, job_dir / name)
        _write_manifest(svc, job, job_id, job_dir, name, meta)


def _write_manifest(
    svc: WarlockService,
    job: dict,
    job_id: str,
    job_dir: Path,
    name: str | None,
    meta: dict | None,
) -> None:
    """Merge one artifact's metadata into the job's manifest.

    Read-modify-write under the manifest's own lock, because that is what it
    is: several artifacts derived concurrently each add their own entry, and
    a whole-file write from a stale read would drop whichever landed first.
    """
    import json

    from ..pipelines import asset2d

    path = job_dir / MANIFEST
    with svc.convert_lock(job_id, MANIFEST):
        try:
            manifest = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            manifest = {}
        if not isinstance(manifest, dict):
            manifest = {}
        manifest.setdefault("version", 1)
        manifest["job"] = job_id
        manifest["prompt"] = job.get("prompt") or ""
        manifest["recipe"] = asset2d.recipe_hash(
            (job.get("params") or {}).get("recipe", {}).get("reference")
        )
        artifacts = manifest.setdefault("artifacts", {})
        if name is not None and meta is not None:
            artifacts[name] = meta
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp, path)
