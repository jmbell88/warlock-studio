"""Bulk export: many jobs' artifacts into one zip, or into a project folder."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any

from . import files
from .core import WarlockService
from .errors import Invalid, NotFound
from .files import MEDIA
from .validation import check_job_id


def export_names(files: list[str] | None) -> list[str]:
    """The requested artifact names, defaulting to the GLB.

    The allowlist *is* files.MEDIA: the point is that a caller-supplied name
    never becomes a path component without passing through it first.
    """
    names = [f for f in (files or []) if f] or ["model.glb"]
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
            # name that is not a 2D export, so nothing else changes.
            if path.exists() and files.ready(job, job_dir, name) and files.fresh_2d(job_dir, name):
                out.append((f"{job_id}/{name}", path))
    return out


def bulk_export(
    svc: WarlockService,
    ids: list[str],
    files: list[str] | None,
    dest_zip: Path,
) -> dict[str, Any]:
    """Zip the named artifacts of several jobs into ``dest_zip``.

    Derived artifacts are *not* generated on demand here -- a batch export
    should not be able to kick off twenty Blender subprocesses.
    """
    names = export_names(files)
    members = collect(svc, ids, names)
    if not members:
        raise NotFound("nothing to export")
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in members:
            zf.write(path, arcname)
    return {"path": str(dest_zip), "files": len(members)}


def export_to_folder(
    svc: WarlockService, ids: list[str], files: list[str] | None
) -> dict[str, Any]:
    """Copy the same selection into WARLOCK_EXPORT_DIR."""
    if svc.config.export_dir is None:
        raise NotFound("no export folder configured (set WARLOCK_EXPORT_DIR)")
    names = export_names(files)
    members = collect(svc, ids, names)
    if not members:
        raise NotFound("nothing to export")
    copied = 0
    for arcname, path in members:
        dest = svc.config.export_dir / arcname
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        copied += 1
    return {"copied": copied, "dir": str(svc.config.export_dir)}
