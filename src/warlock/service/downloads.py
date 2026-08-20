"""Fetching a model's weights, on the user's explicit say-so, out of process.

The narrow exception to the offline invariant, and every part of its narrowness
lives here. The generation pipeline never calls anything in this module; only a
button in the app-Settings pane does. Nothing here downloads either -- it plans
(``warlock.fetch``, pure), refuses (disk), spawns
``python -m warlock.pipelines.fetch_worker``, and reads that child's progress
lines. The variable that would make *this* process online-capable is set in the
child's own environment and nowhere else.

Blocking by contract, like every other multi-second call in the service layer:
the pane dispatches it through ``TaskRunner``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .. import fetch as fetch_mod
from .. import vram, winjob
from .core import WarlockService
from .errors import Conflict, Failed, Invalid, NotFound, ServiceError

log = logging.getLogger(__name__)

# Wall-clock ceiling for one repository fetch. Generous because 16 GB over a
# slow line is genuinely long, bounded because a child parked on a stalled
# socket holds a task-pool worker forever.
FETCH_TIMEOUT = 4 * 60 * 60.0

Progress = Callable[[float, str], None]


def worker_argv() -> list[str]:
    """How the child is started. A function rather than an inline literal so a
    test can put a stub in its place and exercise this half -- the spawn, the
    stdin hand-over, the progress lines, the result file -- without any test in
    this project ever making a network call."""
    return [sys.executable, "-m", "warlock.pipelines.fetch_worker"]


def rows(svc: WarlockService) -> list[dict[str, Any]]:
    """Every downloadable registry entry with a real presence flag.

    A flag rather than the word "missing" inside a label. The pane used to
    infer missing-ness from the substring, which is brittle in the obvious way
    and, less obviously, silently gave the style LoRAs no marking at all --
    nothing was putting the word into their labels.
    """
    config = svc.config
    plan_ = getattr(svc, "vram_plan", None)
    out: list[dict[str, Any]] = []
    for entry in fetch_mod.entries():
        jobs = fetch_mod.plan(config, [entry])
        row: dict[str, Any] = {
            "row_key": entry.row_key,
            "kind": entry.kind,
            "key": entry.key,
            "label": entry.label,
            "present": entry.is_present(config),
            "size_gib": fetch_mod.total_gib(jobs),
            "downloadable": bool(jobs),
        }
        # Only base models have a device footprint, and only when a plan has
        # been resolved -- a headless caller and every test that builds a
        # service by hand have ``vram_plan is None``, and a badge invented from
        # no measurement is worse than no badge. The key is *absent* rather than
        # None in that case, so the pane's ``.get`` reads one way.
        if plan_ is not None and entry.kind == "base":
            row["vram"] = vram.fits(plan_, entry.spec)
        # What the row is, beyond a label: which repositories it comes from,
        # what a base model costs on the card as a *number* rather than only
        # the three-state word, and an adapter's trained trigger words -- all
        # of it already on the spec, all of it previously reachable only by
        # reading MODELS.md beside the app.
        repos = tuple(dict.fromkeys(f.repo_id for f in entry.fetch if f.repo_id))
        if repos:
            row["repos"] = repos
        need = getattr(entry.spec, "vram_gib", None)
        if entry.kind == "base" and isinstance(need, int | float):
            row["vram_gib"] = float(need)
        trigger = str(getattr(entry.spec, "trigger", "") or "")
        if trigger:
            row["trigger"] = trigger
        # What removing *this* row alone would free, and whether it would free
        # anything at all. Only for rows that are here -- there is nothing to
        # offer against a model that is not installed -- and computed rather
        # than guessed, because one of four recipes over one checkpoint frees
        # 0.8 GB and not 7, and a button that said 7 would be a lie.
        #
        # Path arithmetic only (no stat, no walk), ~17 entries, on the
        # task-done path. Cheap enough; measure again if the pane hitches.
        if row["present"]:
            removal = fetch_mod.removal_plan(config, [entry])
            row["removable"] = bool(removal.paths)
            row["freed_gib"] = removal.freed_gib
        out.append(row)
    return out


def needed_rows(svc: WarlockService, row_keys: Sequence[str]) -> list[dict[str, Any]]:
    """Which of these rows are *not* on this host, in ``rows()``'s shape.

    The question a locked feature asks: "what does this need that I haven't
    got". Resolution goes through ``fetch_mod.find`` and raises ``NotFound`` on
    an unknown row exactly as ``plan_for`` does, because a feature naming a row
    key the registry has never heard of is a bug in the feature and must not
    quietly return "nothing missing".

    Sizes here are **per row and not deduped** -- each is what that row alone
    would fetch. A caller quoting one figure to a user composes this with
    ``plan_for``/``fetch.total_gib`` over the returned keys instead, which is
    what makes a sprite sheet on a fresh host quote ~13 GB rather than the ~28
    a naive sum of four rows sharing one 7 GiB checkpoint would produce.
    """
    config = svc.config
    out: list[dict[str, Any]] = []
    for row_key in row_keys:
        entry = fetch_mod.find(row_key)
        if entry is None:
            raise NotFound(f"No such model: {row_key}")
        if entry.is_present(config):
            continue
        jobs = fetch_mod.plan(config, [entry])
        out.append(
            {
                "row_key": entry.row_key,
                "kind": entry.kind,
                "key": entry.key,
                "label": entry.label,
                "present": False,
                "size_gib": fetch_mod.total_gib(jobs),
                "downloadable": bool(jobs),
            }
        )
    return out


def needed_keys(svc: WarlockService, row_keys: Sequence[str]) -> tuple[str, ...]:
    """``needed_rows`` reduced to the keys, for a refusal's ``rows=``."""
    return tuple(row["row_key"] for row in needed_rows(svc, row_keys))


def needed_gib(svc: WarlockService, row_keys: Sequence[str]) -> float:
    """What installing exactly the missing ones costs, deduped across them.

    The composition ``needed_rows`` deliberately does not do for you: shared
    weights are counted once, so this is the number a pane may put on a button.
    """
    keys = list(needed_keys(svc, row_keys))
    return fetch_mod.total_gib(plan_for(svc, keys)) if keys else 0.0


def plan_for(svc: WarlockService, row_keys: list[str]) -> list[fetch_mod.Job]:
    """The deduped set of fetches these rows need. Raises on an unknown row.

    Deduped across the whole selection rather than per row, which is the only
    reason ticking all four SDXL 1.0 recipes asks for 7 GB rather than 28.
    """
    chosen: list[fetch_mod.Entry] = []
    for row_key in row_keys:
        entry = fetch_mod.find(row_key)
        if entry is None:
            raise NotFound(f"No such model: {row_key}")
        chosen.append(entry)
    return fetch_mod.plan(svc.config, chosen)


def download(
    svc: WarlockService,
    row_keys: list[str],
    *,
    on_progress: Progress | None = None,
    timeout: float = FETCH_TIMEOUT,
) -> dict[str, Any]:
    """Fetch everything these rows need. Blocking; raises ``Invalid`` on refusal.

    The disk check is at the door and refuses the *whole* plan, the way sweep
    admission does: half a plan downloaded is a set of model directories some
    of which are complete and none of which the user asked for individually.
    """
    jobs = plan_for(svc, row_keys)
    if not jobs:
        raise Invalid("There is nothing to download for that.")
    refusal = fetch_mod.disk_refusal(jobs)
    if refusal is not None:
        raise Invalid(refusal)

    with _maintenance("downloading a model"):
        return _download(jobs, on_progress, timeout)


def _maintenance(what: str) -> Any:
    """The exclusive model-store lease, as a service-layer refusal.

    Every mutation of the store holds this for its whole duration, which is two
    guarantees in one. Against the *worker*, it is the thing that was missing:
    the operations it excludes run in ``asyncio.to_thread``, so nothing on the
    event loop could ever have serialised against them (MDL-01). Against
    *itself*, it is the backend mutation lock that the Settings pane's disabled
    buttons only ever imitated -- a headless caller, a future pane or a shutdown
    race shares no UI convention (MDL-11).

    Still process-local. A second Warlock on the same home is RUN-01's problem
    and wants an OS-level lock; this would not have seen it.
    """
    from .. import leases

    @contextlib.contextmanager
    def _held() -> Any:
        try:
            with leases.MODELS.maintain():
                yield
        except TimeoutError as exc:
            raise Conflict(
                f"Warlock is still using the image model, so {what} would not be "
                f"safe. Wait for the current job to finish and try again."
            ) from exc

    return _held()


# What an interrupted publish leaves behind. Deterministic names (see
# ``pipelines/fetch_worker._move_into``), so they can be found and removed --
# which is the point: a hard kill during a publish strands them, and they are
# invisible to every presence probe, so the disk quietly holds a second copy of
# a checkpoint forever.
_STAGING_SUFFIXES = (".fetch.part", ".fetch.bak")


def _sweep_staging(jobs: list[fetch_mod.Job], *, spare: set[str] | None = None) -> None:
    """Remove staging trees an interrupted fetch left behind. Never raises.

    Opportunistic and at the *start* of the next download, exactly as
    ``_sweep_trash`` is for uninstall: the leak is bounded by "the user fetches
    again", it costs one directory listing per destination, and a sweep that
    raised would turn recoverable clutter into a permanent refusal.

    ``spare`` is the set of staging trees an *open journal* is protecting, and
    it is load-bearing rather than tidy. Without it the sweep and the recovery
    race in exactly the wrong direction: the sweep sees ``.thing.fetch.part``,
    correctly identifies it as the litter of an interrupted fetch, and deletes
    the only copy of the files the rollback was about to put back.
    """
    spared = spare or set()
    for parent in {job.dest.parent for job in jobs}:
        try:
            entries = list(parent.iterdir())
        except OSError:
            continue
        for path in entries:
            if not path.name.endswith(_STAGING_SUFFIXES):
                continue
            if str(path) in spared:
                log.info("keeping staging an open transaction still needs: %s", path)
                continue
            log.warning("removing staging left by an interrupted fetch: %s", path)
            with contextlib.suppress(OSError):
                shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink()


def disk_usage(svc: WarlockService) -> dict[str, Any]:
    """What the model store is actually holding, measured. Blocking.

    Every figure the Models list shows is the registry's *declared*
    ``size_gib``, so the one number nobody could see was how much disk the
    weights are really using -- and Storage measured job directories and the
    trash and nothing else, which on a full install is the smaller half.

    A walk, so it goes through ``TaskRunner`` like the library's own.
    """
    root = Path(svc.config.t2i_model_root)
    total = 0
    files = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
                files += 1
        except OSError:
            # A file a fetch is mid-way through renaming is the normal state of
            # a download, not an error -- ``fetch_worker._dir_bytes``'s rule.
            continue
    return {"root": str(root), "bytes": total, "files": files}


def sweep_staging(svc: WarlockService) -> list[str]:
    """Remove staging trees no open transaction needs. -> what it removed.

    ``_sweep_staging`` runs at the start of the *next* download, which is a
    bound on the leak and not a fix for it: a user who cancels a 16 GB fetch
    and never downloads again keeps 16 GB of invisible ``.fetch.part`` -- it
    is under no presence probe and shows in no storage view. The pane calls
    this when it opens, so "cancelled" and "reclaimed" are the same visit.

    Never raises, for ``_sweep_staging``'s reason: clutter must not become a
    refusal.
    """
    from .. import publish

    root = Path(svc.config.t2i_model_root)
    jobs = fetch_mod.plan(svc.config, list(fetch_mod.entries()))
    parents = {job.dest.parent for job in jobs} | {root}
    spared = publish.staged_dirs(root)
    removed: list[str] = []
    for parent in parents:
        try:
            entries = list(parent.iterdir())
        except OSError:
            continue
        for path in entries:
            if not path.name.endswith(_STAGING_SUFFIXES) or str(path) in spared:
                continue
            log.warning("removing staging left by an interrupted fetch: %s", path)
            with contextlib.suppress(OSError):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink()
                removed.append(path.name)
    return removed


def recover(config: Any) -> list[str]:
    """Roll back a publish the process did not live to finish. -> what it undid.

    Called once on the startup path, after ``reconcile_startup`` and before the
    staging sweep -- that order is the whole of it. The sweep would otherwise
    delete the staging trees this needs, and reconcile is about *jobs* rather
    than about the model store, so the two do not interact.

    Never raises. An interrupted download must not make the app unlaunchable,
    which would be the one failure worse than the interrupted download.
    """
    from .. import publish

    root = Path(config.t2i_model_root)
    try:
        undone = publish.recover(root)
    except Exception:  # pragma: no cover - publish.recover suppresses its own
        log.exception("could not roll back an interrupted download")
        return []
    if undone:
        log.warning(
            "rolled back an interrupted download; these are as they were before "
            "it started: %s",
            ", ".join(undone),
        )
        fetch_mod.bump_store_generation()
    return undone


def _download(
    jobs: list[fetch_mod.Job], on_progress: Progress | None, timeout: float
) -> dict[str, Any]:
    """Stage the whole selection, then publish all of it. MDL-10.

    "Install these four" is one decision, and it used to be four. Each child
    downloaded *and* published its own repository, so a failure on the third
    left two installed that nobody had asked for individually -- and the
    refusal said so, because saying otherwise would have been a lie.

    Now the children stage only (``publish: False``), and nothing is installed
    until every one of them has landed and verified its bytes. A failure during
    the staging phase is the cheap case: the staging trees are removed and the
    disk is exactly as it was, because nothing ever moved.

    The publishing phase is the one that needs the journal. Unwinding inside
    Python already rolls a publish back, but a process *killed* mid-publish
    cannot unwind at all -- so what it is about to do is written down first,
    and ``recover`` reads that back on the next launch. See
    :mod:`warlock.publish`.

    The disk door already charges for this: ``disk_refusal`` is computed over
    the whole plan plus headroom, and every staging tree is a sibling of its
    destination, so staging all of them at once needs the space the door
    already required.
    """
    from .. import publish

    root = Path(jobs[0].dest).parent
    _sweep_staging(jobs, spare=publish.staged_dirs(root))
    total = fetch_mod.total_gib(jobs) or float(len(jobs))
    done_gib = 0.0
    staged: list[tuple[fetch_mod.Job, dict[str, Any]]] = []

    # --- phase one: stage everything, install nothing -----------------------
    try:
        for index, job in enumerate(jobs, start=1):
            share = job.size_gib or (total / len(jobs))

            def report(percent: float, label: str, _done=done_gib, _share=share) -> None:
                if on_progress is None:
                    return
                # Capped below 100 and below the publish's own share: the move
                # is fast but it is not instant, and a bar that reached the end
                # before the files existed is what made "downloaded" and
                # "installed" look like one moment.
                overall = 98.0 * (_done + _share * percent / 100.0) / total
                on_progress(min(overall, 98.0), label)

            report(0.0, f"{job.repo_id} ({index} of {len(jobs)})")
            result = _run_worker(job, on_progress=report, timeout=timeout, publish=False)
            staged.append((job, result))
            done_gib += share
    except ServiceError as exc:
        # Nothing has moved, so there is nothing to undo -- only staging trees
        # to drop. This is the whole benefit of the two phases: the message can
        # now say "nothing was installed" and be true.
        for _job, result in staged:
            shutil.rmtree(result.get("staged") or "", ignore_errors=True)
        _sweep_staging(jobs)
        raise Invalid(
            f"{exc.message} Nothing was installed: the whole selection is "
            f"downloaded before any of it is put in place, so your models are "
            f"exactly as they were. Try again."
        ) from exc

    # --- phase two: publish, under a journal --------------------------------
    if on_progress is not None:
        on_progress(98.0, "Installing")
    entries = [
        {"repo_id": job.repo_id, "staging": result.get("staged"), "dest": str(job.dest)}
        for job, result in staged
    ]
    publish.begin(root, entries)
    fetched: list[str] = []
    try:
        for job, result in staged:
            staging = Path(result["staged"])
            names = publish.move_into(staging, job.dest)
            # Recorded *as it happens*, because the crash this survives happens
            # during the loop: the journal has to be true at every moment of it
            # rather than only at the ends.
            publish.note_published(root, str(job.dest), names)
            _write_manifest_for(job, staging, names)
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(publish.backup_dir(job.dest), ignore_errors=True)
            fetched.append(job.repo_id)
    except BaseException:
        # In-process failure: undo what this loop did, in reverse, and leave
        # the journal for a recovery only if we cannot finish the undo here.
        publish.recover(root)
        _sweep_staging(jobs)
        fetch_mod.bump_store_generation()
        raise
    publish.finish(root)
    # Once, at the end, and now that is the honest place for it: before the
    # transaction, publication was per repo and the counter had to keep up with
    # it. Everything in this selection became visible at the same moment.
    fetch_mod.bump_store_generation()
    if on_progress is not None:
        on_progress(100.0, "")
    return {"fetched": fetched}


def _write_manifest_for(job: fetch_mod.Job, staging: Path, names: list[str]) -> None:
    """The completion marker, written by the parent now that it publishes.

    Same record the child used to write and for the same reason -- a manifest
    is what makes "this directory was finished by a Warlock download" a
    question with an answer -- and written last, after the publish, so it
    remains a completion marker rather than an intention.

    Never raises: a manifest that could not be written must not fail a download
    that succeeded. Its absence already means "unknown", which would be true.
    """
    from .. import hashes, publish

    path = job.dest / fetch_mod.MANIFEST_NAME
    record = {
        "repo_id": job.repo_id,
        "revision": job.revision,
        "files": sorted(names),
        "digests": hashes.digest_files(job.dest, names),
    }
    existing = publish.read_json(path) or {}
    repos = existing.get("repos")
    repos = dict(repos) if isinstance(repos, dict) else {}
    repos[job.repo_id] = record
    publish.write_json(path, {"repos": repos})


# A directory being deleted is renamed to a sibling with this prefix first, so
# the removal is atomic from a reader's point of view: the model is either fully
# there or fully gone, never a half-emptied checkpoint that ``present`` calls
# downloaded. ``rmtree`` of a 7 GiB tree is not atomic and can fail halfway;
# ``os.rename`` within one volume is.
TRASH_PREFIX = ".trash-"

# States that mean the queue may still ask for these weights. Conservative and
# whole-queue rather than per-model: a queued job's ``base_model`` names a
# checkpoint, but a running one may already have loaded a style LoRA, an
# adapter and a ControlNet that its row does not mention, and deleting a file
# out from under a live pipe is not a failure with a good message.
_LIVE_STATUSES = ("queued", "running")


def _sweep_trash(root: Path) -> None:
    """Remove whatever an interrupted uninstall left behind. Never raises.

    Opportunistic, at the *start* of the next uninstall rather than on a timer:
    the leak is bounded by "the user removes another model", it costs one
    directory listing, and a sweep that raised would turn a recoverable mess
    into a permanent refusal.
    """
    try:
        stale = [p for p in root.iterdir() if p.name.startswith(TRASH_PREFIX)]
    except OSError:
        return
    for path in stale:
        with contextlib.suppress(OSError):
            shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink()


async def _unload_if_idle(worker: Any) -> bool:
    """Drop the resident pipe, but only if nothing is running. Loop-side.

    The re-check and the unload have to be the same loop-side callable, because
    what makes the answer trustworthy is that no other coroutine runs between
    them: ``current_job_id`` is set and cleared on this thread. Checking from
    the service thread and unloading in a second hop would reintroduce exactly
    the gap it is here to close.

    Returns False when a job is in flight, so the caller can refuse instead of
    deleting weights a live pipe is reading.
    """
    if worker.current_job_id is not None:
        return False
    await worker.unload_text2image()
    return True


def uninstall(
    svc: WarlockService,
    row_keys: list[str],
    *,
    on_progress: Progress | None = None,
) -> dict[str, Any]:
    """Delete these models' weights. Blocking; refuses rather than half-deletes.

    The mirror of ``download`` and the same posture at the door: the whole
    selection is refused, or the whole selection runs. What it may delete is
    ``fetch.removal_plan``'s answer and nothing else -- reference-counted over
    the registry, so uninstalling one of four recipes over one checkpoint frees
    only that recipe's own adapter.

    Removing the *default* base model is deliberately allowed. It degrades to
    the friendly refusal every other missing checkpoint gets, which names the
    model and offers Settings → Models; a special case here would be a rule the
    user cannot see and cannot undo.
    """
    chosen: list[fetch_mod.Entry] = []
    for row_key in row_keys:
        entry = fetch_mod.find(row_key)
        if entry is None:
            raise NotFound(f"No such model: {row_key}")
        chosen.append(entry)

    # ``active_jobs()`` and not ``list(limit=MAX_LIST_LIMIT)``: the paged read
    # this replaced returned the newest 5,000 rows and filtered them in Python,
    # so on a library past that a *queued* row older than the page was simply
    # invisible and its weights were deleted out from under it. The purpose-built
    # query is unbounded, asks the store the question directly, and is cheap --
    # one job runs at a time and a queue is a handful of rows (MDL-01).
    if svc.store.active_jobs():
        raise Conflict(
            "Jobs are still queued or running; wait for them to finish before "
            "removing a model."
        )

    removal = fetch_mod.removal_plan(svc.config, chosen)
    if removal.blocked:
        raise Invalid(" ".join(removal.blocked))
    if not removal.paths:
        raise Invalid(
            "There is nothing to remove: every file those models use is shared "
            "with another model that would still need it."
        )

    # Before anything is unlinked, and tolerant of a service with no worker (a
    # headless tool, a test): on Windows a mapped safetensors file cannot be
    # deleted, and this process is the one holding the mapping.
    #
    # The check above is a *snapshot*: nothing stops a create_job landing in the
    # gap between it and here, being claimed, and putting a generate() in flight
    # while this unloads the pipe underneath it -- the loop is free the whole
    # time ``_generate`` is awaiting its ``to_thread``. So the same question is
    # asked again from inside the callable that runs *on the loop thread*, where
    # ``current_job_id`` cannot change under the read, and the unload is
    # abandoned rather than racing it (MDL-01).
    if svc.worker is not None and not svc.call_on_loop(
        lambda: _unload_if_idle(svc.worker)
    ):
        raise Conflict(
            "A job started while the model was being removed. Nothing has "
            "been deleted -- wait for it to finish and try again."
        )

    with _maintenance("removing a model"):
        return _uninstall(svc, removal, on_progress)


def _uninstall(
    svc: WarlockService, removal: Any, on_progress: Progress | None
) -> dict[str, Any]:
    root = svc.config.t2i_model_root
    _sweep_trash(root)
    if on_progress is not None:
        on_progress(0.0, "removing")
    removed: list[str] = []
    for index, path in enumerate(removal.paths, start=1):
        _remove_one(path, root)
        removed.append(str(path))
        # Same reasoning as ``download``'s: per unlink, so a raise partway
        # through cannot leave a cache believing in weights that are gone.
        fetch_mod.bump_store_generation()
        if on_progress is not None:
            on_progress(100.0 * index / len(removal.paths), path.name)
    if on_progress is not None:
        on_progress(100.0, "")
    return {"removed": removed, "freed_gib": removal.freed_gib}


def _remove_one(path: Path, root: Path) -> None:
    """Delete one claim, crash-atomically for a directory. Missing is fine.

    An absent path is not an error: ``removal_plan`` is pure and answers about
    what a row *stands on*, not about what happens to be there, so a partially
    downloaded model has claims with nothing behind them.
    """
    try:
        if path.is_dir():
            trash = root / f"{TRASH_PREFIX}{secrets.token_hex(6)}"
            os.rename(path, trash)
            shutil.rmtree(trash)
        elif path.exists():
            path.unlink()
    except PermissionError as exc:
        raise Failed(
            f"Could not remove {path.name}: another process holds these files. "
            "Close any running jobs, or restart Warlock, and try again."
        ) from exc
    except OSError as exc:
        raise Failed(f"Could not remove {path.name}: {exc}") from exc


# How much of a dead child's stderr to quote. A traceback's last few lines are
# what name the failure; the whole thing belongs in the log, not in a toast.
STDERR_TAIL = 400

# How many stderr lines the pump retains. Bounded because huggingface_hub can
# chatter for hours and only the tail is ever quoted; generous enough that a
# whole traceback survives whatever preceded it.
_STDERR_KEEP_LINES = 200


def _stderr_tail(lines: deque[str], reader: threading.Thread) -> str:
    """The end of the child's stderr, or "". Never raises and never blocks long.

    ``lines`` is the pump thread's bounded buffer; the pump hits EOF when the
    child exits, so the short join only covers a kill racing its last read.
    """
    reader.join(timeout=2.0)
    text = "".join(lines).strip()
    if not text:
        return ""
    log.error("fetch worker stderr:%s%s", chr(10), text)
    return text[-STDERR_TAIL:].replace(chr(10), " ")


def _kill_and_reap(proc: subprocess.Popen[str]) -> None:
    """Kill the child and collect it, without being able to hang doing so.

    ``rigging.run_worker``'s rule, and it belongs here for the same reason: a
    kill without a wait leaves the child unreaped and the pump thread blocked
    on a pipe that never closes -- but an *unbounded* wait means the caller
    that was already timing out or unwinding now blocks indefinitely on a
    child refusing to die. The bound is the point; on Windows a killed process
    that is stuck in an uninterruptible driver call is exactly the case, and
    this one holds a socket. It cannot outlive the app either way -- it is in
    the kill-on-close job -- so giving up after ten seconds costs nothing.
    """
    proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=10.0)
    winjob.untrack(proc.pid)


def _run_worker(
    job: fetch_mod.Job, *, on_progress: Progress, timeout: float, publish: bool = True
) -> dict[str, Any]:
    """One child, one repository. Raises ``Invalid`` with the child's own words.

    The spec goes over stdin and the answer comes back through a file, matching
    ``rigging.run_worker`` -- stdout carries progress lines and a stray print
    from ``huggingface_hub`` must not be able to corrupt the result.

    ``publish=False`` stops the child with its staging tree complete and
    verified, for the parent to install as part of a transaction (MDL-10). The
    default stays True so a caller that wants one repository installed on its
    own -- and the tests that exercise that path -- is unchanged.
    """
    with tempfile.TemporaryDirectory(prefix="warlock-fetch-") as scratch:
        result_path = Path(scratch) / "result.json"
        spec = job.spec()
        spec["result_path"] = str(result_path)
        spec["publish"] = publish
        # ``stderr=PIPE``, not DEVNULL. A child that dies *before* ``main()`` --
        # a broken venv, an import error, the ``json.loads(sys.stdin.read())``
        # that sits outside its own try -- writes no result file, so the only
        # report was "the fetch worker exited with code 1" and its traceback
        # went to the void. ``rigging.run_worker`` already keeps a tail of the
        # child's stderr for exactly this reason (SVC-05). Written here rather
        # than inline in the call because ``tests/test_vram.py``'s spawn scan
        # reads a 15-line window after each ``Popen(`` looking for the
        # ``winjob.assign`` below, and a comment block inside the arguments
        # pushed it out of range.
        proc = subprocess.Popen(
            worker_argv(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        # The same kill-on-close job every other child goes in. A fetch holds a
        # socket and writes gigabytes; one that outlived a hard kill of the app
        # would go on filling the disk with nothing on screen to say so.
        winjob.assign(proc.pid)
        # And tracked: this is the child with a four-hour ceiling, so it is the
        # one most likely to still be running when something asks everything to
        # stop (MDL-02, and the Cancel button MDL-14 asks for).
        winjob.track(proc.pid, f"fetch {spec.get('repo_id', '')}")
        assert proc.stdin is not None and proc.stdout is not None
        assert proc.stderr is not None

        # stderr is drained from the start, not read after exit: a child whose
        # stderr outgrows the OS pipe buffer blocks on its next write and never
        # reaches the exit that read was waiting on -- a deadlock the four-hour
        # timeout turns into a four-hour hang. The pump keeps only a bounded
        # tail, which is all ``_stderr_tail`` ever quoted.
        err_lines: deque[str] = deque(maxlen=_STDERR_KEEP_LINES)

        def _pump_err(stream: Any) -> None:
            try:
                for raw in stream:
                    err_lines.append(raw)
            except (OSError, ValueError):
                pass

        err_reader = threading.Thread(target=_pump_err, args=(proc.stderr,), daemon=True)
        err_reader.start()

        def _send() -> None:
            try:
                proc.stdin.write(json.dumps(spec))
                proc.stdin.close()
            except OSError:
                # A worker that died before draining stdin says far more
                # through its exit code and its result file than this would.
                pass

        writer = threading.Thread(target=_send, daemon=True)
        writer.start()

        # stdout is drained on a helper thread so the *whole* fetch has a
        # deadline, not just the wait() after EOF -- ``rigging.run_worker``'s
        # pattern, and here for a sharper reason: the child's _Sampler emits a
        # progress line every half second whether or not bytes are arriving, so
        # a child parked on a stalled socket never closes stdout at all. Reading
        # it inline meant `timeout` was never consulted and the task-pool worker
        # was held forever.
        lines: queue.Queue[str | None] = queue.Queue()

        def _pump(stream: Any) -> None:
            try:
                for raw in stream:
                    lines.put(raw)
            finally:
                lines.put(None)

        reader = threading.Thread(target=_pump, args=(proc.stdout,), daemon=True)
        reader.start()

        deadline = time.monotonic() + timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(proc.args, timeout)
                try:
                    raw = lines.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    continue
                if raw is None:
                    break
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(payload, dict):
                    # A bare JSON scalar is valid JSON; huggingface_hub prints
                    # into the same stream the progress lines go down.
                    continue
                try:
                    percent = float(payload.get("percent") or 0.0)
                except (TypeError, ValueError):
                    # A non-numeric percent is a malformed line, not a reason
                    # to abandon a download that is otherwise working.
                    continue
                on_progress(percent, str(payload.get("label") or ""))
            code = proc.wait(timeout=max(deadline - time.monotonic(), 0.0))
            winjob.untrack(proc.pid)
        except subprocess.TimeoutExpired:
            _kill_and_reap(proc)
            raise Invalid(f"The download of {job.repo_id} timed out.") from None
        except BaseException:
            _kill_and_reap(proc)
            raise

        result: dict[str, Any] = {}
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except ValueError:
                result = {}
    if not result.get("ok"):
        detail = result.get("error") or f"the fetch worker exited with code {code}"
        if not result.get("error"):
            # No result file means the child died before it could write one, so
            # its stderr is the only thing that knows why.
            tail = _stderr_tail(err_lines, err_reader)
            if tail:
                detail += f": {tail}"
        log.warning("fetch of %s failed: %s", job.repo_id, detail)
        raise Invalid(f"Could not download {job.repo_id}: {detail}")
    return result
