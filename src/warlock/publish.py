"""Moving a staged download into place, and the journal that makes it undoable.

Pure in the ``vram.py`` sense -- stdlib only, no imports from ``service``,
``queue`` or ``studio``. Two callers need it and they are in different
processes: the fetch child stages, and the parent publishes.

**Why publishing moved out of the child.** MDL-10: a selection of models is one
decision -- "install these" -- and it used to be N independent ones. Each child
downloaded *and* published its own repository, so a failure on the third of four
left two installed that nobody had asked for individually, and the refusal said
so because saying otherwise would have been a lie. Making it a transaction means
staging everything first and publishing only once every download has landed,
which is only possible if the publish is a step the parent takes.

**The journal is what covers the gap the language cannot.** Unwinding inside
Python already rolls a publish back: :func:`move_into` restores every file it
moved and everything it overwrote. What it cannot survive is the process
*dying* mid-publish -- a hard kill, a power cut -- and at that moment the disk
holds some of the new files, some of the old ones in a backup tree, and a
staging directory with the rest. So the parent writes down what it is about to
do before it does any of it, and :func:`recover` reads that back on the next
launch and puts the disk where it was.

Rolling *back* rather than forward, deliberately. Completing an interrupted
publish would need the staged bytes to still be trustworthy, and the one thing
known about a process that died mid-write is that its last write may be torn.
Undoing is decidable from the journal alone.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Written into a staging tree by the child, read by the parent. Its presence is
#: what distinguishes "this download finished and is waiting to be published"
#: from "this download was interrupted", which the staging directory's mere
#: existence cannot say.
PUBLISH_NAME = ".warlock-publish.json"

#: Written beside the model root by the parent, before the first publish of a
#: transaction and deleted after the last. Its presence at startup means a
#: publish was interrupted.
JOURNAL_NAME = ".warlock-txn.json"


def backup_dir(dest: Path, staging: Path | None = None) -> Path:
    """Where :func:`move_into` parks whatever it is about to overwrite.

    A deterministic sibling, because ``recover`` has to find it on a later run
    with nothing but the destination's name to go on.
    """
    dest = Path(dest)
    if staging is None:
        return dest.parent / f".{dest.name}.fetch.bak"
    staging = Path(staging)
    suffix = ".fetch.part"
    name = staging.name
    name = (
        f"{name[:-len(suffix)]}.fetch.bak"
        if name.endswith(suffix)
        else f"{name}.fetch.bak"
    )
    return staging.parent / name


def planned_names(staging: Path) -> list[str]:
    """The names :func:`move_into` would publish out of ``staging``, in order.

    Split out so the *plan* and the *doing* cannot disagree. The journal is
    written from this before the first file moves, which is what makes an
    interrupted publish recoverable: ``undo_into`` skips a name that is not
    at the destination, so the intended list undoes exactly the prefix that
    happened.

    Recording it afterwards -- which is what ``note_published`` alone did --
    left the entry with no ``published`` list at all when the process was
    killed *inside* the loop. Recovery then moved nothing back out of the
    destination, restored only the backup tree (which covers replaced files,
    never added ones) and deleted the staging tree that held the other copy.
    The destination kept an arbitrary prefix of a model with no manifest, and
    for an adapter whose presence probe sorts early that reads as installed.

    Forward slashes, because these names go into JSON and come back out in a
    later process that may not be this one's platform.
    """
    staging = Path(staging)
    out: list[str] = []
    for src in sorted(staging.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(staging)
        if rel.name == PUBLISH_NAME:
            # The child's completion marker belongs to the staging tree and
            # must not be published into a model directory.
            continue
        out.append(str(rel).replace("\\", "/"))
    return out


def move_into(staging: Path, dest: Path) -> list[str]:
    """Move the staged tree into ``dest``: all of it, or none of it.

    Per file rather than one directory rename, because a destination
    legitimately already exists and is shared: ``loras/`` holds every adapter,
    and a second fetch into it must add files rather than replace the folder.

    Which is exactly what makes the rollback load-bearing rather than tidy. A
    per-file move that fails partway -- a full disk, a file another process has
    open -- leaves the half-populated directory the staging tree exists to
    prevent, and every presence probe in ``warlock.fetch`` answers "is this
    here" from a handful of filenames, so such a directory reads as a finished
    download forever. So each move is undone in reverse: the file back into
    staging, where the caller's rmtree removes it with the rest of the failed
    download, and anything it overwrote back out of the backup tree.

    The backup tree is *left in place* on success by the caller's decision, not
    this function's: it removes it, because a completed publish has nothing to
    restore. What matters here is that its name is deterministic
    (:func:`backup_dir`), so a recovery on a later launch can find it.
    """
    staging, dest = Path(staging), Path(dest)
    backup = backup_dir(dest, staging)
    moved: list[tuple[Path, Path]] = []
    saved: list[tuple[Path, Path]] = []
    names: list[str] = []
    try:
        for name in planned_names(staging):
            src = staging / name
            rel = Path(name)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                # os.replace destroys what is already there, and "left as it
                # was" has to include that: a second fetch into loras/ writes
                # names an earlier, different download put there.
                kept = backup / rel
                kept.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, kept)
                saved.append((kept, target))
            # replace(), not move(): an interrupted earlier attempt can leave a
            # file of the same name, and shutil.move onto an existing file raises
            # on some platforms and silently differs on others.
            os.replace(src, target)
            moved.append((src, target))
            names.append(name)
    except BaseException:
        # Suppressed, every one of them: a rollback that raises would replace
        # the real failure with a second one and tell the user nothing about
        # either. Moved files first -- a restore needs its target free.
        for src, target in reversed(moved):
            with contextlib.suppress(OSError):
                os.replace(target, src)
        restored_all = True
        for kept, target in reversed(saved):
            try:
                os.replace(kept, target)
            except OSError:
                # Still suppressed -- but remembered. A file another process
                # holds open (the realistic Windows failure; finalize_rig
                # retries for exactly it) cannot be restored right now, and
                # the backup tree holds its only copy.
                restored_all = False
        if restored_all:
            shutil.rmtree(backup, ignore_errors=True)
        # else: leave the backup in place. Its name is deterministic
        # (backup_dir), so recovery on a later launch can still restore from
        # it -- deleting it here would destroy the only copy of whatever the
        # loop above could not put back.
        raise
    return names


def undo_into(staging: Path, dest: Path, names: list[str]) -> bool:
    """Reverse a publish that the process did not live to finish.

    The inverse of :func:`move_into`, driven by the journal rather than by
    in-memory state, because there is none: this runs in a later process.

    A file named in the journal is moved back to staging if it is there to move
    -- so a publish that got halfway is undone halfway, which is exactly the
    half that happened -- and then whatever the backup tree holds is restored
    over the top. Both directions are needed: the first puts back files that
    were *added*, the second puts back files that were *replaced*. Returns
    whether every required move succeeded; failures remain for a later retry.
    """
    staging, dest = Path(staging), Path(dest)
    complete = True
    failed: set[Path] = set()
    for name in names:
        # ``names`` comes back off disk, out of ``.warlock-txn.json``, which a
        # crashed run left behind and anything with local write access could
        # have edited since. A member spelled ``../..`` or ``C:/Windows/x``
        # would have this loop move a file the publish never touched. Skipped
        # rather than raised: recovery runs at startup and one bad line in a
        # stale journal may not stop the rest of it being undone.
        if not _contained(dest, name) or not _contained(staging, name):
            log.warning("skipping %r in a journal: not a name inside the tree", name)
            complete = False
            continue
        target = dest / name
        back = staging / name
        # A previous recovery attempt may already have moved this file back.
        # In that case the target is the restored original and must not be
        # moved out again when a retained journal is retried.
        if back.is_file():
            continue
        if not target.is_file():
            continue
        try:
            back.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, back)
        except OSError:
            complete = False
            failed.add(Path(name))
    backup = backup_dir(dest, staging)
    if backup.is_dir():
        for kept in sorted(backup.rglob("*")):
            if kept.is_dir():
                continue
            rel = kept.relative_to(backup)
            # Do not overwrite a new file that could not be moved back. The
            # backup is its only copy of the original, so retaining both is
            # what lets the next startup try again without data loss.
            if rel in failed:
                continue
            target = dest / rel
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(kept, target)
            except OSError:
                complete = False
    return complete


def _contained(root: Path, name: str) -> bool:
    """Whether ``root / name`` stays under ``root``.

    The containment rule ``sirens_io._under`` and ``inker_mode._under`` state
    for export paths, asked as a question rather than a refusal because the one
    caller is crash recovery and wants to carry on past a bad line.
    """
    try:
        return root.resolve() in (root / name).resolve().parents
    except OSError:
        return False


def write_json(path: Path, payload: Any) -> None:
    """Write JSON through a temp file and one rename. Never raises.

    ``tmp`` plus ``os.replace``, the staged-write rule this repo applies to
    everything it serves: a journal half-written by the crash it exists to
    survive would be worse than none, because ``recover`` would read it.
    """
    path = Path(path)
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)


def _write_required(path: Path, payload: Any) -> None:
    """Atomically write transaction state, propagating every I/O failure.

    Completion manifests are advisory and use :func:`write_json`; the journal
    is the only copy of the information needed to undo a crash and therefore
    must exist before publication is allowed to begin.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any] | None:
    """One JSON object, or None for missing, unreadable or malformed."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def journal_path(root: Path) -> Path:
    return Path(root) / JOURNAL_NAME


def begin(root: Path, entries: list[dict[str, Any]]) -> Path:
    """Record what is about to be published. Returns the journal's path.

    Called with every staging tree already complete and verified, and *before*
    the first file moves. Everything after this point is recoverable; anything
    that fails before it leaves nothing installed, because nothing was.
    """
    path = journal_path(root)
    _write_required(path, {"version": 1, "entries": entries})
    return path


def note_published(root: Path, staging: str, dest: str, names: list[str]) -> None:
    """Record which files one entry actually published, as it happens.

    Written per destination rather than once at the end, which is the whole
    point: the crash this survives happens *during* the loop, so the journal
    has to be true at every moment of it rather than only at the ends.
    """
    path = journal_path(root)
    data = read_json(path)
    if data is None:
        raise OSError(f"publish journal is missing or unreadable: {path}")
    found = False
    for entry in data.get("entries") or []:
        if (
            isinstance(entry, dict)
            and entry.get("staging") == staging
            and entry.get("dest") == dest
        ):
            entry["published"] = names
            found = True
    if not found:
        raise OSError(f"publish journal has no entry for {dest}")
    _write_required(path, data)


def finish(root: Path) -> None:
    """The transaction committed. Delete the journal; nothing to recover."""
    journal_path(root).unlink(missing_ok=True)


def staged_dirs(root: Path) -> set[str]:
    """The staging trees an open journal is protecting, as absolute strings.

    The staging sweep spares these. Without it, a recovery and a sweep race in
    exactly the wrong direction: the sweep sees ``.thing.fetch.part``, calls it
    the litter of an interrupted fetch, and deletes the only copy of the files
    the rollback was about to put back.
    """
    path = journal_path(root)
    data = read_json(path)
    if not data:
        # A present but unreadable journal must make the sweep conservative.
        # Its staging paths cannot be recovered from JSON, but deleting every
        # candidate would guarantee that recovery can never be attempted.
        if path.exists():
            try:
                return {str(p) for p in Path(root).iterdir()}
            except OSError:
                return set()
        return set()
    return {
        str(Path(entry["staging"]))
        for entry in (data.get("entries") or [])
        if isinstance(entry, dict) and entry.get("staging")
    }


def recover(root: Path) -> list[str]:
    """Roll back an interrupted publish. -> the destinations put back.

    Never raises: this runs on the startup path, and a recovery that took the
    app down with it would make an interrupted download unrecoverable by
    launching.
    """
    root = Path(root)
    data = read_json(journal_path(root))
    if not data:
        return []
    undone: list[str] = []
    cleanups: list[tuple[Path, Path]] = []
    complete = True
    for entry in data.get("entries") or []:
        if not isinstance(entry, dict):
            complete = False
            continue
        dest = entry.get("dest")
        staging = entry.get("staging")
        if not isinstance(dest, str) or not dest or not isinstance(staging, str) or not staging:
            complete = False
            continue
        published = entry.get("published") or []
        if not isinstance(published, list) or not all(
            isinstance(name, str) for name in published
        ):
            complete = False
            continue
        staging_path, dest_path = Path(staging), Path(dest)
        try:
            restored = undo_into(staging_path, dest_path, published)
        except OSError:
            restored = False
        if restored:
            undone.append(str(dest))
            cleanups.append((staging_path, backup_dir(dest_path, staging_path)))
        else:
            complete = False
    if not complete:
        return []
    try:
        # The journal is the retry marker. Remove it before cleanup: if its
        # unlink fails, staging and backups retain the idempotence evidence a
        # later recovery needs.
        finish(root)
    except OSError:
        return []
    for staging, backup in cleanups:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)
    return undone
