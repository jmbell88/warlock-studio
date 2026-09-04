"""Is the library intact, and give me a copy of the part that cannot be redone.

``~/.warlock`` holds every asset this tool has ever made, and until now nothing
could answer either question. ``prune``, ``trash`` and ``empty_trash`` change
the library; ``warlock doctor --verify`` re-hashes the *models*, which are
downloads and can be fetched again. The user's own work had no equivalent.

**What can go wrong is not hypothetical.** ``_jobs_lifecycle.retained_job_ids``
records the incident that motivates most of this: on 2026-08-09 the store held
117 verdicts, of which **100 named job directories that no longer existed** --
destroyed by a button whose confirmation truthfully said the verdicts would be
kept. They were; the pixels were not, and the pixels were what three blocked
items needed. Nothing reported that at the time and nothing would report it
today. :func:`verify` is where that class of finding now surfaces.

**Read-only, all of it.** Nothing here deletes, moves or repairs. A verify that
tidied up would be a verify nobody dares run on a library they care about, and
the two operations that *do* remove things already exist with their own
refusals and their own undo. What this produces is a report; acting on it is
:func:`~warlock.service._jobs_lifecycle.clean_jobs`' job, or the user's.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import files
from .core import WarlockService
from .errors import Invalid
from .validation import JOB_ID_RE

log = logging.getLogger(__name__)

#: The artifact whose absence means a finished job produced nothing, keyed by
#: the job's stage.
#:
#: Keyed on *stage* and not on *kind* because stage is what ``files.ready``
#: itself keys on, and a second answer to "what is this job's output" would be
#: a second thing to keep in step. A stage that is not in here is not checked
#: at all -- a guess would produce findings against jobs that are perfectly
#: intact, and a verify that cries wolf is one nobody reads twice.
PRIMARY = {
    "model": "model.glb",
    "reference": "input.png",
    "tile": "input.png",
    "tilesheet": "input.png",
    # Its own stage string rather than reusing "model": db.py's queries load
    # that one with mesh-verdict semantics, and a music row picking those up
    # would be graded against a scale that has no meaning for audio.
    "music": "track.wav",
}

#: How many rows one page of the walk reads. The same keyset-cursor shape
#: ``prune_jobs`` uses and for its reason: a history longer than one page must
#: still be walkable, and this one is read-only so nothing moves under it.
_PAGE = 500


def _walk(svc: WarlockService):
    """Every job row, newest first, in pages."""
    cursor: tuple[float, str] | None = None
    while True:
        page = svc.store.list(limit=_PAGE, before=cursor)
        if not page:
            return
        yield from page
        last = page[-1]
        cursor = (last["created_at"], last["id"])


def verify(svc: WarlockService) -> dict[str, Any]:
    """Check the library against itself. -> a report; changes nothing.

    Five questions, and each one is a way the store and the disk can disagree
    without anything having complained at the time:

    ``missing_dirs``
        A finished job whose directory is gone. Its artifacts are unrecoverable
        and the row still lists, offers exports and reads as an asset.

    ``orphan_dirs``
        A directory with no row. Invisible to the library and to the prune, so
        it is disk nothing will ever reclaim. Sized, because that is the number
        that decides whether the user cares.

    ``missing_artifacts``
        A finished job whose directory exists but whose *output* is not there.
        Through ``files.ready`` rather than ``Path.exists``, so this asks the
        same question the exporter and the library ask -- a half-written mesh
        is not a mesh here either.

    ``stale_verdicts``
        A verdict naming a job whose directory is gone: the 2026-08-09 incident,
        as a standing check. The row is not wrong -- a verdict deliberately
        outlives its asset -- but a *model* accept whose ``source.glb`` has gone
        can no longer serve the thing it was kept for.

    ``unreadable_params``
        A row whose settings blob will not parse. The store answers ``{}`` for
        these on purpose (see ``JobStore.unreadable_params``), which makes this
        the only place the damage is visible.

    A job that writes into *another* job's directory -- a rig, a sheet, a
    re-texture, anything carrying ``source_job`` -- is excluded from the two
    directory questions rather than reported. It never owned a directory of its
    own, so "missing" would be true of every one of them and mean nothing.
    """
    data_dir = Path(svc.config.data_dir)
    missing_dirs: list[dict[str, Any]] = []
    missing_artifacts: list[dict[str, Any]] = []
    known: set[str] = set()
    checked = 0

    for job in _walk(svc):
        job_id = job["id"]
        known.add(job_id)
        checked += 1
        if (job.get("params") or {}).get("source_job"):
            continue
        if job.get("status") != "done":
            # Only a finished job promises an artifact. A queued one has no
            # directory yet by design, and a failed one is *expected* to have
            # produced nothing -- reporting either would bury the real finding.
            continue
        job_dir = data_dir / job_id
        if not job_dir.is_dir():
            missing_dirs.append(_row(job))
            continue
        name = PRIMARY.get(job.get("stage") or "")
        if name is not None and not files.ready(job, job_dir, name):
            missing_artifacts.append({**_row(job), "artifact": name})

    orphan_dirs = []
    for entry in _job_shaped_dirs(data_dir):
        if entry.name in known:
            continue
        orphan_dirs.append({"id": entry.name, "bytes": files.dir_size(entry)})

    stale_verdicts = []
    for verdict in svc.store.latest_verdicts():
        job_id = verdict["job_id"]
        if not (data_dir / job_id).is_dir():
            stale_verdicts.append(
                {"id": job_id, "stage": verdict["stage"], "verdict": verdict["verdict"]}
            )

    unreadable = svc.store.unreadable_params()
    findings = (
        len(missing_dirs)
        + len(orphan_dirs)
        + len(missing_artifacts)
        + len(stale_verdicts)
        + len(unreadable)
    )
    return {
        "ok": findings == 0,
        "checked": checked,
        "findings": findings,
        "missing_dirs": missing_dirs,
        "orphan_dirs": orphan_dirs,
        "orphan_bytes": sum(d["bytes"] for d in orphan_dirs),
        "missing_artifacts": missing_artifacts,
        "stale_verdicts": stale_verdicts,
        "unreadable_params": list(unreadable),
    }


def _job_shaped_dirs(data_dir: Path) -> list[Path]:
    """Directories under ``data_dir`` that are named like a job.

    ``JOB_ID_RE`` rather than an exclusion list. A job directory is twelve hex
    characters and nothing else under here is -- ``autosave``, ``warlock.log``,
    ``crash.log``, ``trellis.log`` and ``session.marker`` all fail the pattern
    on their own -- so the rule needs no maintenance as the app grows siblings,
    which is exactly what an exclusion list would need and would not get.
    """
    try:
        entries = sorted(data_dir.iterdir())
    except OSError:
        return []
    return [e for e in entries if e.is_dir() and JOB_ID_RE.match(e.name)]


def _row(job: dict[str, Any]) -> dict[str, Any]:
    """The identifying half of a job, for a report line."""
    return {
        "id": job["id"],
        "kind": job.get("kind", ""),
        "stage": job.get("stage", ""),
        "name": job.get("name") or (job.get("prompt") or "")[:60],
    }


def backup(
    svc: WarlockService, dest: Path, *, include_assets: bool = False
) -> dict[str, Any]:
    """Copy the store to ``dest``, and the asset tree too if asked. -> a summary.

    **The store is the default and the assets are not**, which is a deliberate
    asymmetry rather than a half-finished function. They are different kinds of
    thing:

    * The store is a few megabytes and holds what cannot be recreated from the
      files -- every prompt, seed, model fingerprint, name, tag and verdict.
      Losing it turns a library into a directory of anonymous GLBs. It is small
      enough to copy on a whim and valuable enough to copy often.
    * The asset tree is the generated output, routinely tens of gigabytes (a
      real library measured 99 GB), and it is ordinary files that the user's own
      backup already handles better than this can. Copying it from inside the
      app, by default, would turn a one-second operation into an hour-long one
      and fill a disk without warning.

    So ``include_assets`` exists, and it is off. A caller that wants everything
    says so.

    The store goes through ``JobStore.backup_to`` -- sqlite's online backup,
    because this connection is in WAL mode and a file copy is a snapshot missing
    every transaction since the last checkpoint. Assets go through
    ``export._staged_copy``, the same temp-then-replace this codebase uses
    everywhere a destination might be watched or re-read.
    """
    dest = Path(dest)
    if dest.exists() and not dest.is_dir():
        raise Invalid("the backup destination is a file, not a folder", field="dest")
    dest.mkdir(parents=True, exist_ok=True)
    store_path = dest / Path(svc.config.db_path).name
    written = svc.store.backup_to(store_path)
    out: dict[str, Any] = {
        "dir": str(dest),
        "store": str(store_path),
        "store_bytes": written,
        "assets": 0,
        "asset_bytes": 0,
        "included_assets": include_assets,
    }
    if not include_assets:
        return out

    from .export import _staged_copy

    data_dir = Path(svc.config.data_dir)
    copied = 0
    total = 0
    for job_dir in _job_shaped_dirs(data_dir):
        for source in sorted(job_dir.rglob("*")):
            if not source.is_file():
                continue
            target = dest / "assets" / source.relative_to(data_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                _staged_copy(source, target)
            except OSError:
                # One unreadable file does not fail the backup. The count and
                # the log are what say so -- the alternative is a two-hour copy
                # that aborts on its last file and leaves the user with nothing.
                log.exception("backup: could not copy %s", source)
                continue
            copied += 1
            total += target.stat().st_size
    out["assets"] = copied
    out["asset_bytes"] = total
    return out
