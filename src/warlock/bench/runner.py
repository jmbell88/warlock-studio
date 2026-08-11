"""Running a suite: submit, wait, copy, render, repeat.

Threading, exactly -- this is the part that has to agree with docs/INVARIANTS.md's
three-thread invariant rather than route around it:

* ``run`` executes on the process main thread, which here is a may-block
  thread equivalent to a TaskRunner worker. It never touches the asyncio loop
  directly and never imports bpy into this process.
* It stands up ``studio.runtime.Runtime``, which is already headless and
  already constructs the Worker on its own loop. Zero new wiring.
* Submission is ``service.jobs.create_job``, which wakes the worker through
  ``loop.call_soon_threadsafe`` as it always does.
* Waiting is ``store.get`` polled from this thread -- exactly what the pygame
  frame loop already does against the same lock-guarded connection. No second
  sqlite connection, no new primitive.
* The harness never writes to params. It only reads; everything it records
  goes into the run directory, so the last-write-wins hazard on ``set_params``
  cannot arise.

One job in flight at a time. The queue is serial, so batching buys nothing --
and a Ctrl-C mid-run would otherwise leave 159 queued rows for the app to chew
through at next launch.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import manifest as manifest_mod
from . import recipe as recipe_mod
from . import suite as suite_mod
from . import views as views_mod

log = logging.getLogger(__name__)

POLL_INTERVAL = 0.5
# A reference is seconds and a mesh is minutes; this is the "it has hung"
# ceiling, not an expectation.
JOB_TIMEOUT = 3600.0

ITEMS_FILE = "items.jsonl"

# Copied into every item directory when present. source.glb is deliberately
# absent: it is the largest artifact by far and nothing measures it, so it
# rides behind --keep-source.
ARTIFACTS = ("input.png", "reference.png", "control.png", "model.glb")
SOURCE_ARTIFACT = "source.glb"


@dataclass(frozen=True, slots=True)
class Unit:
    """One (item, seed) pair -- the atom a run is made of."""

    item: Any
    seed: int

    @property
    def key(self) -> str:
        return f"{self.item.id}--s{self.seed}"


def units(suite: Any, items: Iterable[Any], seeds: Iterable[int]) -> list[Unit]:
    return [Unit(item=i, seed=s) for i in items for s in seeds]


def run_dir_name(started: str, recipe_key: str, suite_key: str) -> str:
    return f"{started}-{recipe_key}-{suite_key}"


def select(
    suite: Any, *, categories: tuple[str, ...] = (), ids: tuple[str, ...] = (), limit: int = 0
) -> list[Any]:
    chosen = suite.filter(categories=categories, ids=ids)
    return chosen[:limit] if limit > 0 else chosen


def plan_run(
    config: Any,
    suite: Any,
    recipe: Any,
    *,
    stage: str,
    started: str,
    categories: tuple[str, ...] = (),
    ids: tuple[str, ...] = (),
    limit: int = 0,
    seeds: tuple[int, ...] = (),
    run_dir: Path | None = None,
) -> tuple[Path, list[Unit], dict[str, Any]]:
    """Everything decided before a single job is submitted.

    Separate from ``run`` so ``--dry-run`` is the same code path minus the
    execution, rather than a second implementation of the selection rules that
    could disagree with it.
    """
    items = select(suite, categories=categories, ids=ids, limit=limit)
    use_seeds = seeds or suite.seeds
    todo = units(suite, items, use_seeds)
    directory = run_dir or (
        Path(config.bench_dir) / "runs" / run_dir_name(started, recipe.key, suite.key)
    )
    doc = manifest_mod.build_manifest(
        config,
        suite,
        recipe,
        stage=stage,
        items=[i.id for i in items],
        seeds=list(use_seeds),
        started=started,
    )
    return directory, todo, doc


def read_items(run_dir: Path) -> list[dict[str, Any]]:
    """Every recorded unit, oldest first, torn lines skipped.

    A unit re-run by a later resume appends a second record under the same key;
    callers that want one row per unit take the last, which is the one that
    describes the artifacts currently on disk.
    """
    path = run_dir / ITEMS_FILE
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            # A torn last line is what a Ctrl-C mid-write leaves.
            continue
        if isinstance(record, dict) and "key" in record:
            out.append(record)
    return out


def latest_items(run_dir: Path) -> dict[str, dict[str, Any]]:
    """One record per unit key -- the newest, so a retry supersedes its
    failure rather than sitting beside it."""
    return {record["key"]: record for record in read_items(run_dir)}


def completed(run_dir: Path) -> set[str]:
    """The unit keys ``--resume`` should *not* re-run.

    Only the ones that finished. A unit that errored or was cancelled by the
    Ctrl-C path recorded a terminal row like any other, and keying on mere
    presence marked it permanently done -- so the one thing a resume is for,
    picking up after a failure, was the one thing it could not do.
    """
    return {
        key
        for key, record in latest_items(run_dir).items()
        if record.get("status") == "done"
    }


def retryable(run_dir: Path) -> set[str]:
    """Units that were recorded but did not finish."""
    return {
        key
        for key, record in latest_items(run_dir).items()
        if record.get("status") != "done"
    }


def append_item(run_dir: Path, record: dict[str, Any]) -> None:
    with (run_dir / ITEMS_FILE).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def _resolve_vram(config: Any) -> None:
    """Write the auto-selected VRAM mode onto the config, as Runtime would."""
    from .. import vram

    plan = vram.plan(
        exclusive=config.vram_exclusive,
        budget_gib=config.vram_budget_gib,
        total_gib=config.vram_total_gib,
        device=vram.probe(),
        explicit=config.vram_exclusive_explicit,
    )
    config.vram_exclusive = plan.exclusive
    config.vram_budget_gib = plan.budget_gib
    config.vram_exclusive_explicit = plan.explicit


def run(
    config: Any,
    *,
    suite_key: str = suite_mod.DEFAULT_SUITE,
    recipe_key: str,
    stage: str = "reference",
    started: str,
    categories: tuple[str, ...] = (),
    ids: tuple[str, ...] = (),
    limit: int = 0,
    seeds: tuple[int, ...] = (),
    render: bool = False,
    keep_source: bool = False,
    resume: Path | None = None,
    on_event: Callable[[str], None] | None = None,
) -> Path:
    """Execute a run and return its directory."""
    from ..studio.runtime import Runtime

    say = on_event or (lambda msg: log.info("%s", msg))
    # Before the manifest is built, not when Runtime starts: vram_exclusive is
    # a recorded field, and a run whose manifest says "auto" cannot be compared
    # with -- or resumed by -- one that says which mode auto actually chose.
    _resolve_vram(config)
    suite = suite_mod.load(suite_key)
    recipe = recipe_mod.load(recipe_key)

    already: set[str] = set()
    if resume is not None:
        try:
            stored = manifest_mod.read_manifest(resume)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"{resume}: not a bench run directory "
                f"({manifest_mod.FILENAME} is unreadable: {exc})"
            ) from None
        # The selection is the *stored* run's, read back from its manifest.
        # Re-planning from this invocation's flags meant resuming a
        # ``--limit 5`` run without repeating the flag silently expanded it to
        # the whole suite -- hours of GPU nobody asked for, announced as
        # "5 of 160 already done". assert_resumable cannot catch it either: it
        # deliberately excludes the item and seed lists, because a resume
        # legitimately covers a subset.
        if categories or ids or limit or seeds:
            say("resume: ignoring the selection flags; using the run's own")
        run_dir, todo, doc = plan_run(
            config, suite, recipe,
            stage=stage, started=started,
            ids=tuple(stored.get("items") or ()),
            seeds=tuple(stored.get("seeds") or ()),
            run_dir=resume,
        )
        # Refuse before a single job is submitted: a run half-measured against
        # two checkpoints is worse than no run.
        manifest_mod.assert_resumable(stored, doc)
        already = completed(resume)
        again = retryable(resume)
        say(f"resuming {resume.name}: {len(already)} of {len(todo)} already done")
        if again:
            say(f"retrying {len(again)} unit(s) that did not finish")
    else:
        run_dir, todo, doc = plan_run(
            config, suite, recipe,
            stage=stage, started=started, categories=categories, ids=ids,
            limit=limit, seeds=seeds, run_dir=None,
        )
        manifest_mod.write_manifest(run_dir, doc)

    pending = [u for u in todo if u.key not in already]
    if not pending:
        say("nothing to do")
        return run_dir

    with Runtime(config) as svc:
        for n, unit in enumerate(pending, 1):
            say(f"[{n}/{len(pending)}] {unit.key}")
            try:
                record = _run_unit(
                    svc, config, run_dir, recipe, unit,
                    stage=stage, render=render, keep_source=keep_source,
                )
            except KeyboardInterrupt:
                say("interrupted; the in-flight job has been cancelled")
                raise
            append_item(run_dir, record)
    return run_dir


def _run_unit(
    svc: Any,
    config: Any,
    run_dir: Path,
    recipe: Any,
    unit: Unit,
    *,
    stage: str,
    render: bool,
    keep_source: bool = False,
) -> dict[str, Any]:
    from ..service import jobs as svc_jobs

    item_dir = run_dir / "items" / unit.key
    item_dir.mkdir(parents=True, exist_ok=True)

    kwargs = recipe_mod.job_kwargs(recipe, unit.item, unit.seed, stage=stage)
    started = time.monotonic()
    job_id = svc_jobs.create_job(svc, **kwargs)["id"]
    try:
        job = _await_job(svc, job_id)
    except BaseException:
        # Includes KeyboardInterrupt: the queue is serial, so leaving a job
        # running would block whatever the user does next.
        with _suppressed():
            svc_jobs.cancel_job(svc, job_id)
        raise

    elapsed = time.monotonic() - started
    _copy_artifacts(config.job_dir(job_id), item_dir, keep_source=keep_source)
    (item_dir / "job.json").write_text(json.dumps(job, indent=2, default=str), encoding="utf-8")

    record = {
        "key": unit.key,
        "item": unit.item.id,
        "category": unit.item.category,
        "seed": unit.seed,
        "job": job_id,
        "status": job.get("status"),
        "error": job.get("error"),
        "seconds": round(elapsed, 2),
    }
    if render and job.get("status") == "done" and (item_dir / "model.glb").exists():
        record["views"] = _render_views(item_dir)
    return record


def _render_views(item_dir: Path) -> bool:
    try:
        views_mod.render_views(item_dir / "model.glb", item_dir / "views")
        return True
    except Exception:
        # A render failure costs this item its automatic score, not the run:
        # the mesh is on disk and the human comparison still works.
        log.exception("view render failed for %s", item_dir.name)
        return False


def _copy_artifacts(job_dir: Path, item_dir: Path, *, keep_source: bool = False) -> None:
    """Copied, never referenced: a run has to survive ``prune_jobs``."""
    names = (*ARTIFACTS, SOURCE_ARTIFACT) if keep_source else ARTIFACTS
    for name in names:
        src = job_dir / name
        if src.exists():
            shutil.copyfile(src, item_dir / name)


def _await_job(svc: Any, job_id: str, timeout: float = JOB_TIMEOUT) -> dict[str, Any]:
    """Poll the store from this thread until the job reaches a terminal state.

    The same read the pygame frame loop makes against the same connection --
    JobStore serialises every statement behind its RLock, so this is safe and
    needs no new primitive.
    """
    deadline = time.monotonic() + timeout
    while True:
        job = svc.store.get(job_id)
        if job is None:
            raise RuntimeError(f"job {job_id} vanished")
        if job["status"] in ("done", "error", "cancelled"):
            return job
        if time.monotonic() > deadline:
            raise TimeoutError(f"job {job_id} did not finish within {timeout:.0f}s")
        time.sleep(POLL_INTERVAL)


def _suppressed():
    import contextlib

    return contextlib.suppress(Exception)


PURGED_FILE = "purged.json"


def _media_files(run_dir: Path) -> list[Path]:
    """Every file :func:`purge_media` would delete, as it exists right now.

    One definition, used by both the measurement and the deletion, so the size
    a user is shown before confirming is by construction the set that goes.
    """
    out: list[Path] = []
    items = run_dir / "items"
    if not items.is_dir():
        return out
    for item_dir in sorted(items.iterdir()):
        if not item_dir.is_dir():
            continue
        for name in (*ARTIFACTS, SOURCE_ARTIFACT):
            path = item_dir / name
            if path.is_file():
                out.append(path)
        views = item_dir / "views"
        if views.is_dir():
            out.extend(sorted(p for p in views.glob("*.png") if p.is_file()))
    return out


def media_bytes(run_dir: Path) -> int:
    """What :func:`purge_media` would free, in bytes."""
    total = 0
    for path in _media_files(run_dir):
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def is_purged(run_dir: Path) -> bool:
    return (Path(run_dir) / PURGED_FILE).is_file()


def purge_media(run_dir: Path) -> dict[str, Any]:
    """Delete a run's pixels and meshes, keeping everything that says what it
    measured. Idempotent; returns the ``purged.json`` it writes.

    **What this costs, plainly.** ``score.py`` re-scores retroactively from the
    rendered views on disk, so a purged run keeps whatever ``scores.json`` it
    already has but can never be scored with a *new* metric -- the mesh and the
    reference image are gone. What survives is everything a comparison is built
    from: ``items.jsonl``, ``manifest.json``, ``scores.json``, each unit's
    ``views/views.json``, and the archived ``job.json`` (which carries
    ``params["mesh_report"]`` and ``params["mesh_audit"]``).

    This is disk space, not tidiness: a run's item dirs hold a GLB, up to four
    PNGs and eight rendered views each, times as many units as the run had.
    """
    run_dir = Path(run_dir)
    files = _media_files(run_dir)
    freed = 0
    removed: list[str] = []
    for path in files:
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError:
            log.exception("could not remove %s", path)
            continue
        freed += size
        removed.append(path.relative_to(run_dir).as_posix())

    doc: dict[str, Any] = {
        "purged_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "freed_bytes": freed,
        "files": removed,
    }
    if is_purged(run_dir):
        # A second purge is a no-op that must not erase the first one's tally:
        # what the run freed is a fact about the run, not about this call.
        try:
            previous = json.loads((run_dir / PURGED_FILE).read_text("utf-8"))
        except (OSError, ValueError):
            previous = {}
        if isinstance(previous, dict):
            doc["freed_bytes"] = int(previous.get("freed_bytes") or 0) + freed
            doc["files"] = list(previous.get("files") or ()) + removed
            doc["purged_at"] = previous.get("purged_at") or doc["purged_at"]
    (run_dir / PURGED_FILE).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def prune_runs(config: Any, keep: int) -> list[str]:
    """Delete all but the newest ``keep`` runs. Directory names sort by
    timestamp, which is why the timestamp leads them."""
    root = Path(config.bench_dir) / "runs"
    if not root.exists():
        return []
    runs = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
    removed = []
    for path in runs[max(0, keep):]:
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path.name)
    return removed
