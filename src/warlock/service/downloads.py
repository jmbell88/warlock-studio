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


def _sweep_staging(jobs: list[fetch_mod.Job]) -> None:
    """Remove staging trees an interrupted fetch left behind. Never raises.

    Opportunistic and at the *start* of the next download, exactly as
    ``_sweep_trash`` is for uninstall: the leak is bounded by "the user fetches
    again", it costs one directory listing per destination, and a sweep that
    raised would turn recoverable clutter into a permanent refusal.

    Unwinding inside Python already rolls a publish back (``_move_into``); this
    is for the case that cannot -- the process being killed mid-publish (MDL-10).
    """
    for parent in {job.dest.parent for job in jobs}:
        try:
            entries = list(parent.iterdir())
        except OSError:
            continue
        for path in entries:
            if not path.name.endswith(_STAGING_SUFFIXES):
                continue
            log.warning("removing staging left by an interrupted fetch: %s", path)
            with contextlib.suppress(OSError):
                shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink()


def _download(
    jobs: list[fetch_mod.Job], on_progress: Progress | None, timeout: float
) -> dict[str, Any]:
    _sweep_staging(jobs)
    total = fetch_mod.total_gib(jobs) or float(len(jobs))
    done_gib = 0.0
    fetched: list[str] = []
    for index, job in enumerate(jobs, start=1):
        share = job.size_gib or (total / len(jobs))

        def report(percent: float, label: str, _done=done_gib, _share=share) -> None:
            if on_progress is None:
                return
            overall = 100.0 * (_done + _share * percent / 100.0) / total
            on_progress(min(overall, 99.0), label)

        report(0.0, f"{job.repo_id} ({index} of {len(jobs)})")
        try:
            _run_worker(job, on_progress=report, timeout=timeout)
        except ServiceError as exc:
            # Repositories publish one at a time, so a failure here leaves the
            # earlier ones *installed* -- the docstring's "the whole selection
            # is refused, or the whole selection runs" is true of the disk check
            # at the door and not of this loop (MDL-10). Making it true would
            # mean staging the entire selection and publishing through a
            # journal, which is the Phase 4A transaction; what is fixed here is
            # the *lying*: the refusal now says exactly what did land, so a user
            # can see that a checkpoint arrived and its required adapter did
            # not, rather than being told the whole thing failed and left alone
            # to guess at gigabytes.
            landed = ", ".join(fetched) if fetched else "nothing"
            raise Invalid(
                f"{exc.message} Already installed before this failed: {landed}. "
                f"Try the remaining ones again -- what downloaded is kept."
            ) from exc
        fetched.append(job.repo_id)
        done_gib += share
        # Per repo, not once at the end: publication is per repo (see MDL-10 on
        # why that is not yet a transaction), so after this line those weights
        # really are on disk and a resident pipe really is out of date. A raise
        # from a later repo must not leave the counter behind what the store
        # already holds.
        fetch_mod.bump_store_generation()
    if on_progress is not None:
        on_progress(100.0, "")
    return {"fetched": fetched}


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


def _stderr_tail(proc: subprocess.Popen[str]) -> str:
    """The end of the child's stderr, or "". Never raises and never blocks long.

    The pipe is already at EOF by the time this is called -- the process has
    exited -- so the read returns immediately.
    """
    stream = proc.stderr
    if stream is None:
        return ""
    try:
        text = stream.read() or ""
    except (OSError, ValueError):
        return ""
    text = text.strip()
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
    job: fetch_mod.Job, *, on_progress: Progress, timeout: float
) -> dict[str, Any]:
    """One child, one repository. Raises ``Invalid`` with the child's own words.

    The spec goes over stdin and the answer comes back through a file, matching
    ``rigging.run_worker`` -- stdout carries progress lines and a stray print
    from ``huggingface_hub`` must not be able to corrupt the result.
    """
    with tempfile.TemporaryDirectory(prefix="warlock-fetch-") as scratch:
        result_path = Path(scratch) / "result.json"
        spec = job.spec()
        spec["result_path"] = str(result_path)
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
            tail = _stderr_tail(proc)
            if tail:
                detail += f": {tail}"
        log.warning("fetch of %s failed: %s", job.repo_id, detail)
        raise Invalid(f"Could not download {job.repo_id}: {detail}")
    return result
