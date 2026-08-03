"""The lazy-derivation engine behind every artifact download.

Everything but the GLB, the reference PNG and the rig is a pure function of
``model.glb``, produced on first request and cached beside it. One lock per
(job, artifact) means two callers asking at once wait for one conversion
rather than starting two -- which for the FBX path is two Blender subprocesses
writing the same filename.
"""

from __future__ import annotations

import logging
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
    if not path.exists():
        raise NotReady("file not ready")
    return path


def derivable(name: str) -> bool:
    """Whether ``name`` can be produced from model.glb on demand.

    The UI uses this to decide between a save button and a "derive, then save"
    one, without having to know which conversions exist.
    """
    return name in files.DERIVED
