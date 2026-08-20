"""Bulk export: many jobs' artifacts into one zip, or into a project folder."""

from __future__ import annotations

import contextlib
import os
import secrets
import shutil
import zipfile
from pathlib import Path
from typing import Any

from . import files
from .core import WarlockService
from .errors import Invalid, NotFound
from .files import MEDIA
from .validation import ARTIFACT_HEALTH, check_job_id


def export_names(names_wanted: list[str] | None) -> list[str]:
    """The requested artifact names, defaulting to the GLB.

    The allowlist *is* files.MEDIA: the point is that a caller-supplied name
    never becomes a path component without passing through it first -- which is
    also why the parameter is not called ``files``, the name of the module that
    allowlist comes out of. ``field="files"`` on the refusal is unchanged: that
    is the name of the *control*, which is what an error has to point at.
    """
    names = [f for f in (names_wanted or []) if f] or ["model.glb"]
    unknown = [n for n in names if n not in MEDIA]
    if unknown:
        raise Invalid(f"unknown file(s): {sorted(unknown)}", field="files")
    return names


def collect(svc: WarlockService, ids: list[str], names: list[str]) -> list[tuple[str, Path]]:
    """-> (arcname, path) for every requested file that is ready to serve.

    Silently skips what is missing rather than failing the batch: a selection
    of ten jobs where one never produced an OBJ should still deliver nine, and
    the zip's contents say which. Readiness, not existence -- exporting a
    running job's model.glb would zip a file the worker is still writing.
    """
    out: list[tuple[str, Path]] = []
    for job_id in ids:
        check_job_id(job_id)
        job = svc.store.get(job_id)
        if job is None:
            continue
        job_dir = svc.job_dir(job_id)
        for name in names:
            path = job_dir / name
            # fresh_2d as well as ready: this is a serving path that never
            # derives, so without it a batch would zip an icon left over from
            # a reference the user has since edited. It answers True for every
            # name that is not a 2D export, so nothing else changes. The pixel
            # palette knob is deliberately not consulted here -- a palette
            # mismatch is a preference, not staleness, so a batch ships the
            # last-derived palette.
            if path.exists() and files.ready(job, job_dir, name) and files.fresh_2d(job_dir, name):
                out.append((f"{job_id}/{name}", path))
    return out


def bulk_export(
    svc: WarlockService,
    ids: list[str],
    names_wanted: list[str] | None,
    dest_zip: Path,
) -> dict[str, Any]:
    """Zip the named artifacts of several jobs into ``dest_zip``.

    Derived artifacts are *not* generated on demand here -- a batch export
    should not be able to kick off twenty Blender subprocesses.

    The selection parameter is ``names_wanted`` and not ``files`` because this
    module imports a module called ``files`` and reads it in ``collect`` a few
    lines up: the shorter name shadowed it for the length of both functions, so
    one added line reaching for ``files.ready`` here would have got a list of
    strings and an AttributeError.
    """
    names = export_names(names_wanted)
    members = collect(svc, ids, names)
    if not members:
        raise NotFound("nothing to export")
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    # Through a temp sibling, like ``_staged_copy`` below and for its reason:
    # the destination is wherever the user pointed the save dialog -- possibly
    # a watched project folder -- and writing the zip in place would leave a
    # torn archive there for the length of the build (SVC-06).
    tmp = dest_zip.with_name(f".{dest_zip.name}.{secrets.token_hex(4)}.tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for arcname, path in members:
                zf.write(path, arcname)
        os.replace(tmp, dest_zip)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
    return {"path": str(dest_zip), "files": len(members)}


def export_to_folder(
    svc: WarlockService, ids: list[str], names_wanted: list[str] | None
) -> dict[str, Any]:
    """Copy the same selection into WARLOCK_EXPORT_DIR."""
    if svc.config.export_dir is None:
        raise NotFound("no export folder configured (set WARLOCK_EXPORT_DIR)")
    names = export_names(names_wanted)
    members = collect(svc, ids, names)
    if not members:
        raise NotFound("nothing to export")
    copied = 0
    for arcname, path in members:
        dest = svc.config.export_dir / arcname
        dest.parent.mkdir(parents=True, exist_ok=True)
        _staged_copy(path, dest)
        copied += 1
    return {
        "copied": copied,
        "dir": str(svc.config.export_dir),
        # Named rather than counted: "3 of 12 assets are degraded" is not
        # something a user can act on, and which ones is (ART-01).
        "degraded": degraded_ids(svc, ids),
    }


def degraded_ids(svc: WarlockService, ids: list[str]) -> list[str]:
    """Which of these jobs had a canonical post-processing step fail.

    Normalization and the mesh report are non-fatal by design -- see
    ``validation.ARTIFACT_HEALTH`` -- so a job can be ``done`` and still carry a
    ``model.glb`` whose pivot and scale are the engine's rather than this
    project's. An export is the moment that stops being an internal detail: the
    file is on its way into a game project, where a wrong pivot is a manual
    fixup on every import and nothing anywhere would say why.
    """
    out: list[str] = []
    for job_id in ids:
        job = svc.store.get(job_id)
        if job and (job.get("params") or {}).get(ARTIFACT_HEALTH):
            out.append(job_id)
    return out


def _staged_copy(source: Path, dest: Path) -> None:
    """Copy through a temp sibling and rename. Never truncates the destination.

    ``WARLOCK_EXPORT_DIR`` exists to be *watched* -- it is a game project's
    assets folder -- and ``shutil.copyfile`` truncates its target before writing
    a byte, so a hot-reloading engine could read a torn GLB for the length of
    the copy. Outside the letter of the staged-writes invariant, which is about
    files this app serves, and squarely inside its reasoning (SVC-06).
    """
    tmp = dest.with_name(f".{dest.name}.{secrets.token_hex(4)}.tmp")
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, dest)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
