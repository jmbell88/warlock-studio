"""A job's life after it exists: prune, edit, trash, restore, delete, cancel.

Split out of ``service/jobs.py``, which had grown to 1,446 lines over five
unrelated subjects; ``jobs.py`` stays as the facade every caller still imports
and calls by attribute.

The rule that ties these together is **what a job is allowed to take with it**.
A rig, a sheet and a re-texture all write into the *source* job's directory, so
deleting a mesh is never a single-row operation: ``dependent_jobs`` is what
finds the rows that describe artifacts about to vanish, ``retained_job_ids``
is what stops prune from collecting a job something else still points at, and
``_refuse_if_busy`` is what keeps any of it from racing the worker. Trash is a
move rather than a delete for the same reason it is everywhere else -- the
undo is the feature.

``MAX_LIST_LIMIT`` is read through the facade at call time; see the comment on
``prune_jobs``.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import time
from typing import Any

from . import verdicts as verdicts_mod
from ._jobs_list import get_job
from .core import WarlockService
from .errors import Conflict, Invalid, NotFound
from .files import dir_size
from .validation import (
    MAX_JOB_NAME,
    check_job_id,
    normalize_tags,
)

log = logging.getLogger(__name__)


def prune_jobs(svc: WarlockService, keep: int = 20) -> dict[str, Any]:
    """Delete everything but the newest ``keep`` jobs. Never touches a running one."""
    if keep < 0:
        raise Invalid("keep must be >= 0", field="keep")
    # Paged with a keyset cursor rather than one MAX_LIST_LIMIT read: a history
    # longer than a single page used to be un-prunable past its first 5000
    # rows, which is exactly the history that needs pruning. Deleting rows the
    # walk has already passed doesn't disturb the cursor.
    # Through the facade, for _jobs_list.list_jobs' reason exactly: the ceiling
    # is patched on ``service.jobs`` by tests, and this is the second reader.
    from . import jobs as _facade

    deleted = 0
    seen = 0
    # Read once, outside the walk: it is a whole-table scan, and a prune of a
    # long history would otherwise repeat it per page.
    retained = retained_job_ids(svc)
    kept = 0
    cursor: tuple[float, str] | None = None
    while True:
        page = svc.store.list(_facade.MAX_LIST_LIMIT, cursor)
        if not page:
            break
        cursor = (page[-1]["created_at"], page[-1]["id"])
        for job in page:
            seen += 1
            if seen <= keep or job["status"] == "running":
                continue
            if job["id"] in retained:
                # Counted, not silently skipped: a prune that reports "deleted
                # 40" having kept 12 has described a smaller act than it
                # performed in the one direction that matters least, and a
                # larger reclaim than it achieved in the one that matters most.
                kept += 1
                continue
            # Skipped rather than refused, unlike delete_job: pruning is a bulk
            # reclaim, and one asset with a rig in flight is no reason to keep
            # the other two hundred.
            if worker_is_inside(svc, job["id"]) or dependent_jobs(svc, job["id"]):
                continue
            # Conditional in the DB, not against this page's snapshot: a
            # queued job can be claimed in the gap, and deleting it then
            # rmtrees a directory a live reconstruction is writing into.
            if not svc.store.delete_if_not_running(job["id"]):
                continue
            shutil.rmtree(svc.job_dir(job["id"]), ignore_errors=True)
            deleted += 1
    return {"deleted": deleted, "kept": kept}


def clean_jobs(svc: WarlockService) -> dict[str, Any]:
    """Delete every job, trashed or not, and every orphaned job directory.

    **This deliberately ignores :func:`retained_job_ids`, and that is the whole
    feature.** Prune and empty-trash skip a job carrying evidence -- an accept,
    or an image label of either class -- for the measured reason set out there,
    and the result is that there has never been a way to say "remove
    everything". A user reclaiming a disk, handing a machine on, or starting a
    corpus over is not asking for a reclaim that quietly keeps the largest
    meshes on it. So this one keeps nothing, and the confirmation in
    ``studio/panes/library.py`` says so in as many words: the accepted meshes
    and the labelled references go, the verdict rows survive with nothing
    behind them, and ``tiercheck`` and ``judge.fit`` will have less to read
    afterwards. Do not "fix" this back into a retention check -- that is
    :func:`prune_jobs`, which is still there and still guards them.

    What survives is everything that is not a job: the global pose library, the
    style-anchor profiles, Inker's autosaves, the settings, the logs, and
    ``jobs.sqlite`` itself -- dropping the database would take the verdicts
    corpus with it, and the corpus is the one thing here that cannot be
    regenerated at any price.

    Refused outright rather than skipped-per-job while anything is queued or
    running. Prune skips a busy job because it is a bulk *reclaim* and one rig
    in flight is no reason to keep two hundred other assets; "delete
    everything" that left three behind would have failed at the only thing it
    claims to do, and the honest answer is to make the user stop the queue.
    """
    active = svc.store.active_jobs()
    if active:
        raise Conflict(
            f"{len(active)} job(s) are still queued or running -- "
            f"cancel or wait for them before cleaning the library"
        )
    # And the worker's own answer, for ``worker_is_inside``'s reason: a
    # cancelled row is terminal in the DB while the reconstruction is still
    # unwinding, so the rows above can all be finished and a directory still be
    # under a live write.
    worker = getattr(svc, "worker", None)
    if worker is not None and worker.current_job_id:
        raise Conflict("cancel the job before deleting it")

    from . import jobs as _facade

    deleted = 0
    seen: set[str] = set()
    cursor: tuple[float, str] | None = None
    while True:
        page = svc.store.list(_facade.MAX_LIST_LIMIT, cursor)
        if not page:
            break
        cursor = (page[-1]["created_at"], page[-1]["id"])
        for job in page:
            seen.add(job["id"])
    # ``store.list`` already returns trashed rows, but it is a *page walk* over
    # a table this call is deleting from, and ``trashed`` is the one read that
    # is defined to return all of them. Union rather than trust either alone.
    for job in svc.store.trashed():
        seen.add(job["id"])

    for job_id in seen:
        # Conditional in the DB for prune_jobs' reason: the refusal above is a
        # snapshot, and a submit landing in the gap must not have its directory
        # removed underneath it.
        if not svc.store.delete_if_not_running(job_id):
            continue
        shutil.rmtree(svc.job_dir(job_id), ignore_errors=True)
        deleted += 1

    # And then the directories no row names. These are what a hand-deleted row
    # or an interrupted delete leaves behind, and they are invisible to every
    # other path in this file -- which is why a library can measure larger than
    # the sum of the jobs in it. Job-shaped names only (12 hex characters), so
    # ``poser/``, ``profiles/`` and ``autosave/`` cannot be caught by it.
    orphans = 0
    try:
        entries = list(svc.config.data_dir.iterdir())
    except OSError:
        entries = []
    for entry in entries:
        if not entry.is_dir() or entry.name in seen:
            continue
        try:
            check_job_id(entry.name)
        except NotFound:
            continue
        if svc.store.get(entry.name) is not None:
            continue
        shutil.rmtree(entry, ignore_errors=True)
        orphans += 1

    log.info("clean_jobs removed %d job(s) and %d orphan director(ies)", deleted, orphans)
    return {"deleted": deleted, "orphans": orphans}


def update_job(svc: WarlockService, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Rename, retag or (un)favourite a job."""
    check_job_id(job_id)
    name = payload.get("name")
    if name is not None:
        name = str(name).strip()
        if len(name) > MAX_JOB_NAME:
            raise Invalid(f"name must be at most {MAX_JOB_NAME} characters", field="name")
    tags = normalize_tags(payload["tags"]) if "tags" in payload else None
    favorite = None
    if "favorite" in payload:
        # Demanded rather than coerced: bool("false") is True, so a caller that
        # sent the string form used to favourite the job it meant to unfavourite.
        if not isinstance(payload["favorite"], bool):
            raise Invalid("favorite must be true or false", field="favorite")
        favorite = payload["favorite"]

    if not svc.store.set_meta(job_id, name=name, tags=tags, favorite=favorite):
        raise NotFound("no such job")
    return get_job(svc, job_id)


def worker_is_inside(svc: WarlockService, job_id: str) -> bool:
    """Whether the GPU worker is still running this job, whatever the row says.

    The row is not enough. ``cancel_job`` writes ``cancelled`` immediately and
    only *asks* the worker to stop, so between that write and the worker
    unwinding there is a window in which the status says the job is over and
    the reconstruction is still writing into its directory. ``current_job_id``
    is cleared in the worker's own ``finally``, after the last write, which
    makes it the only honest answer to "is it safe to rmtree this".
    """
    worker = getattr(svc, "worker", None)
    return worker is not None and worker.current_job_id == job_id


def dependent_jobs(svc: WarlockService, job_id: str) -> list[str]:
    """Unfinished jobs that write into ``job_id``'s directory.

    A rig or a sheet is a *separate* job row whose artifacts land beside the
    ``model.glb`` they were made from -- the rig belongs to the mesh -- so
    deleting the mesh while one runs lets ``finalize_rig`` rename into a
    directory that no longer exists, and recreates it as an orphan. The target
    job's own status says nothing about this: it is ``done``, which is exactly
    why a rig could be queued for it.
    """
    return [
        job["id"]
        for job in svc.store.active_jobs()
        if (job.get("params") or {}).get("source_job") == job_id
    ]


def retained_job_ids(svc: WarlockService) -> set[str]:
    """Jobs a *bulk* delete must skip, because their files are the evidence.

    ``verdicts.vector`` is denormalized so that what a review taught outlives
    the assets it was taught on, and ``delete_sweep`` says as much: "the assets
    are disposable, what was learned from them is not". That is exactly true of
    one case and false of two, and the difference is whether the claim can be
    reconstructed from the row alone.

    * **A model-stage reject** is fully carried by its row. The finding *is*
      "this vector produced a bad mesh"; the mesh adds nothing. Still disposable.
    * **An accept** is not. Its value is the artifact -- ``tiercheck`` qualifies
      a gltfpack tier against accepted ``source.glb`` files, and the mesh probe
      is fitted to accepted meshes. A row saying "this was good" with no mesh
      behind it cannot serve either.
    * **An image label of either class** is not, and this is the half that is
      easy to get wrong. ``judge.fit`` embeds *pixels*, and it refuses below
      ``MIN_PER_CLASS`` of **each** class -- so the rejected references are
      training data every bit as much as the accepted ones, and deleting them
      is deleting half a corpus.

    This is not hypothetical. On 2026-08-09 the database held 117 verdicts, of
    which **100 named job directories that no longer existed** -- every one of
    them destroyed by a button whose confirmation truthfully said the verdicts
    would be kept. They were. The pixels were not, and the pixels were what
    three separate blocked items needed.

    Per-job deletion is deliberately unaffected: :func:`delete_job` and
    :func:`trash_job` are one deliberate act on one named asset, which is the
    escape hatch when somebody really does want an accepted job gone. What is
    guarded is the three bulk paths, where the asset is not on screen and the
    count is the only thing the user sees.

    The code below is unchanged by migration 10 and reads the same column it
    always did -- but ``"accept"`` is the *derived usable cut* now (grade >= +3),
    written from the grade by one owner, ``vectors.verdict_for_grade``. This is
    one of the four readers that split survives for. There is deliberately no
    per-grade retention tier: which grades are worth keeping is a policy nobody
    has asked for, and the cut is the only threshold the scale has.
    """
    keep: set[str] = set()
    for verdict in svc.store.latest_verdicts():
        if verdict["verdict"] == "accept" or verdict["stage"] in verdicts_mod.IMAGE_STAGES:
            keep.add(verdict["job_id"])
    return keep


def _refuse_if_busy(svc: WarlockService, job_id: str) -> None:
    """Raise ``Conflict`` if anything is still writing into this job's dir."""
    if worker_is_inside(svc, job_id):
        raise Conflict("cancel the job before deleting it")
    if dependent_jobs(svc, job_id):
        raise Conflict("a rig or sprite sheet for this asset is still running")


def trash_job(svc: WarlockService, job_id: str) -> dict[str, Any]:
    """Move a job to the trash (J91): the row is marked, nothing is removed.

    The same refusals as :func:`delete_job`, and that is the point rather than
    symmetry for its own sake. Trashing a job whose rig is still running would
    make it vanish from the library while a subprocess kept writing into its
    directory, and the user would have no way to explain what they were
    watching -- the refusal is about the *filesystem*, and the filesystem does
    not care that this delete is reversible.
    """
    check_job_id(job_id)
    job = svc.require_job(job_id)
    if job["status"] == "running":
        raise Conflict("cancel the job before deleting it")
    _refuse_if_busy(svc, job_id)
    if job["status"] == "queued":
        # A queued job has to be *cancelled*, not merely hidden. Deleting one
        # used to remove the row, so the worker's poll never saw it again; a
        # trashed row is still a row, and without this the worker would pick up
        # a job the user has thrown away, spend two minutes of GPU on it and
        # write a mesh into a directory nothing shows.
        #
        # Through ``cancel_job`` rather than the bare ``store.cancel``, which is
        # the correction CON-05 names. The comment here used to say a claim
        # landing in the gap was "caught by the conditional write below, which
        # refuses" -- and that is not what happened. The row was still `queued`
        # when the store's atomic cancel ran, so the cancel *succeeded* and the
        # job was duly trashed, with a correct final state. What never happened
        # was ``request_cancel``: the worker had already claimed the row, saw no
        # cancel flag, and burned the full two-minute reconstruction before its
        # ``finish()`` returned False. The outcome was right and the GPU time
        # was wasted. ``cancel_job`` is the function that knows to signal a
        # running job, and routing through it costs one extra status read.
        #
        # ``Conflict`` is swallowed on purpose: it means the job reached a
        # terminal state in the gap, which is not a reason to refuse *trashing*
        # something that has finished.
        with contextlib.suppress(Conflict):
            cancel_job(svc, job_id)
    if not svc.store.set_deleted_if_not_running(job_id, time.time()):
        raise Conflict("cancel the job before deleting it")
    return {"ok": True}


def restore_job(svc: WarlockService, job_id: str) -> dict[str, Any]:
    """Take a job back out of the trash."""
    check_job_id(job_id)
    if not svc.store.set_deleted_if_not_running(job_id, None):
        raise NotFound("no such job")
    return {"ok": True}


def empty_trash(svc: WarlockService) -> dict[str, Any]:
    """Delete every trashed job for real.

    Reads the whole trash rather than a page (``store.trashed``): "empty"
    that left the older half behind while reporting success would be the worst
    possible reading of the word. Each row still goes through the same
    conditional delete as :func:`delete_job` -- a trashed job cannot be running,
    but it can have been *restored and resubmitted* between the read and the
    write, and that job's directory is not this call's to remove.

    A job carrying evidence (:func:`retained_job_ids`) is kept even here, and
    "empty" therefore stops being literally true -- which is the lesser of two
    wrongs, because the trash is where a labelled reference most plausibly sits.
    The rejected references are half a probe's training set and nothing on the
    card can regenerate a *specific* one.
    """
    deleted = 0
    retained = retained_job_ids(svc)
    kept = 0
    for job in svc.store.trashed():
        if job["id"] in retained:
            kept += 1
            continue
        if worker_is_inside(svc, job["id"]) or dependent_jobs(svc, job["id"]):
            # Skipped rather than refused, as prune does: one asset with a rig
            # in flight is no reason to keep the other two hundred.
            continue
        if not svc.store.delete_if_not_running(job["id"]):
            continue
        shutil.rmtree(svc.job_dir(job["id"]), ignore_errors=True)
        deleted += 1
    return {"deleted": deleted, "kept": kept}


def trash_size(svc: WarlockService) -> dict[str, Any]:
    """How much the trash is holding. Blocking -- call from a task thread."""
    rows = svc.store.trashed()
    total = 0
    for job in rows:
        try:
            total += dir_size(svc.job_dir(job["id"]))
        except OSError:
            # A directory that has gone is a job whose files somebody removed
            # by hand; the row is still restorable-in-name and the figure is
            # advisory, so this is not worth failing the measurement over.
            continue
    return {"count": len(rows), "bytes": total}


def delete_job(svc: WarlockService, job_id: str) -> dict[str, Any]:
    job = svc.require_job(job_id)
    if job["status"] == "running":
        raise Conflict("cancel the job before deleting it")
    _refuse_if_busy(svc, job_id)
    # Re-checked inside the delete statement: the status read above is a
    # snapshot, and the worker's claim() can land in the gap.
    if not svc.store.delete_if_not_running(job_id):
        raise Conflict("cancel the job before deleting it")
    shutil.rmtree(svc.job_dir(job_id), ignore_errors=True)
    return {"ok": True}


def cancel_job(svc: WarlockService, job_id: str) -> dict[str, Any]:
    job = svc.require_job(job_id)
    if job["status"] == "cancelled":
        # Idempotent success: some earlier request (possibly this exact race)
        # already cancelled it. Only a genuinely terminal done/error status
        # below is "too late" and worth refusing.
        return {"ok": True}
    if job["status"] not in ("queued", "running"):
        raise Conflict(f"job is {job['status']}")
    if job["status"] == "running" and svc.worker is not None:
        svc.call_on_loop(lambda: svc.worker.request_cancel(job_id))
    # Atomic: if the worker's own terminal write (done/error) landed first,
    # this is a no-op and the job's real outcome stands instead of being
    # retroactively overwritten to "cancelled". The DB-level JobStore.finish()
    # conditional write (queue.py) is what actually closes the lost-cancel race
    # for a job that was 'queued' here but got claimed before this reached the
    # DB -- not request_cancel, which only matters for a job already running.
    if not svc.store.cancel(job_id):
        # Could be "already cancelled" (idempotent success -- this call's own
        # effect already landed, e.g. via the race above) or "already
        # done/error" (genuinely too late).
        current = svc.store.get(job_id)
        if current and current["status"] == "cancelled":
            return {"ok": True}
        raise Conflict("job already finished")
    return {"ok": True}
