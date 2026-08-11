"""Retarget a reconstruction to a triangle budget with a vendored gltfpack.

The trellis response is ~290k triangles and 22 MB, which is a source mesh, not a
game asset. gltfpack simplifies it without re-running the reconstruction, which
is the whole point: a re-target is a two-second subprocess, and a trellis run is
two minutes of GPU.

The flags are not negotiable and each earns its place:

* ``-si <ratio>`` -- the simplification ratio. gltfpack takes a ratio, not a
  triangle count, so the caller's budget is divided by the source count here.
* ``-noq`` -- no quantisation. Quantised attributes need KHR_mesh_quantization,
  which some importers list as required and refuse the file over.
* ``-ke`` / ``-km`` -- keep extras and materials. Without them the material
  assignment (and therefore both PBR textures) can be dropped on merge.

Like ``trellis-server.exe`` the binary is vendored and pinned; nothing here
downloads anything. Missing it is not fatal -- the ``raw`` profile is always
available and is what every job did before this existed.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .. import winjob

log = logging.getLogger(__name__)

# Named budgets. None means "ship the reconstruction untouched".
PROFILES: dict[str, int | None] = {
    "draft": 20_000,
    "standard": 50_000,
    "detailed": 100_000,
    "raw": None,
}

CUSTOM_MIN = 5_000
CUSTOM_MAX = 200_000

DEFAULT_TIMEOUT = 300.0


class OptimizeError(RuntimeError):
    """gltfpack was missing, failed, timed out, or produced an unusable file."""


def staged_copy(source: Path, dest: Path) -> None:
    """Copy ``source`` onto ``dest`` via a temp file and an atomic rename.

    ``dest`` here is model.glb, which the file route serves on mere existence
    once a job is done -- and POST /optimize runs on done jobs. A plain
    copyfile truncates ``dest`` before writing, so a concurrent reader could
    observe a half-written file; this way it sees the old file or the new
    one, never a mixture. Same idiom as postprocess._staged, kept local so
    this module never has to import trimesh-heavy postprocess.
    """
    fd, raw = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(raw)
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, dest)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def run(
    source: Path,
    dest: Path,
    *,
    target_triangles: int | None,
    exe: Path,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Write an optimized copy of ``source`` to ``dest``.

    ``dest`` is only created on success -- a half-written or rejected output must
    never end up as the model the user downloads.
    """
    if target_triangles is None:
        # No budget asked for, so nothing here needs to know the count: this is
        # a copy either way. Counting meant a full trimesh load of a ~22 MB,
        # ~290k-face GLB, serially on the queue, purely to record a number the
        # mesh report measures again a step later -- and `raw` is the profile
        # every job runs today. None, not a guess: the field says "not
        # measured" rather than claiming a figure nothing produced.
        staged_copy(source, dest)
        return {
            "requested": None,
            "achieved": None,
            "source_triangles": None,
            "bytes": dest.stat().st_size,
        }
    source_triangles = _triangles(source)
    if source_triangles <= target_triangles:
        # Already inside the budget. Copying is honest: running the simplifier
        # to a ratio above 1.0 is a no-op that still re-encodes the file.
        staged_copy(source, dest)
        return {
            "requested": target_triangles,
            # A byte-for-byte copy has exactly the source's count; loading the
            # copy through trimesh just to re-measure it is wasted work.
            "achieved": source_triangles,
            "source_triangles": source_triangles,
            "bytes": dest.stat().st_size,
        }
    if not exe.exists():
        raise OptimizeError(
            f"gltfpack not found at {exe}; use the 'raw' profile or set WARLOCK_GLTFPACK"
        )

    ratio = max(min(target_triangles / max(source_triangles, 1), 1.0), 0.0)
    tmp = dest.with_suffix(".glb.opt.tmp")
    argv = [
        str(exe),
        "-i", str(source),
        "-o", str(tmp),
        "-si", f"{ratio:g}",
        "-noq",
        "-ke",
        "-km",
    ]
    try:
        # winjob.run rather than subprocess.run, for the same reason every
        # other child is in the job object: a hard kill of the app must not
        # leave a gltfpack behind holding a half-written .glb.opt.tmp.
        proc = winjob.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        tmp.unlink(missing_ok=True)
        raise OptimizeError(f"gltfpack timed out after {timeout:.0f}s") from exc
    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise OptimizeError(
            f"gltfpack exited {proc.returncode}: {(proc.stderr or proc.stdout)[:500]}"
        )

    # Every other exit from this function unlinks the staging file; the tail
    # did not, so a _triangles() that raised (a mesh trimesh cannot parse) left
    # a .glb.opt.tmp beside the served model for the next reader to find.
    try:
        achieved = _triangles(tmp)
        if achieved <= 0:
            raise OptimizeError("gltfpack produced a mesh with no triangles")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    log.info(
        "optimized %s: %d -> %d triangles (asked %d)",
        source.name, source_triangles, achieved, target_triangles,
    )
    return {
        "requested": target_triangles,
        "achieved": achieved,
        "source_triangles": source_triangles,
        "bytes": dest.stat().st_size,
    }


def resolve(profile: str, custom: int | None = None) -> int | None:
    """A profile name (or 'custom' plus a count) -> a triangle budget."""
    if profile == "custom":
        if custom is None or not CUSTOM_MIN <= custom <= CUSTOM_MAX:
            raise ValueError(f"custom triangles must be {CUSTOM_MIN}-{CUSTOM_MAX}")
        return custom
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {sorted(PROFILES)}")
    return PROFILES[profile]


def _triangles(path: Path) -> int:
    import trimesh

    loaded = trimesh.load(path, process=False)
    mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
    return int(len(mesh.faces))
