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
import os
import shutil
from pathlib import Path
from typing import Any

#: Written into a staging tree by the child, read by the parent. Its presence is
#: what distinguishes "this download finished and is waiting to be published"
#: from "this download was interrupted", which the staging directory's mere
#: existence cannot say.
PUBLISH_NAME = ".warlock-publish.json"

#: Written beside the model root by the parent, before the first publish of a
#: transaction and deleted after the last. Its presence at startup means a
#: publish was interrupted.
JOURNAL_NAME = ".warlock-txn.json"


def backup_dir(dest: Path) -> Path:
    """Where :func:`move_into` parks whatever it is about to overwrite.

    A deterministic sibling, because ``recover`` has to find it on a later run
    with nothing but the destination's name to go on.
    """
    dest = Path(dest)
    return dest.parent / f".{dest.name}.fetch.bak"


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
    backup = backup_dir(dest)
    moved: list[tuple[Path, Path]] = []
    saved: list[tuple[Path, Path]] = []
    names: list[str] = []
    try:
        for src in sorted(staging.rglob("*")):
            if src.is_dir():
                continue
            rel = src.relative_to(staging)
            if rel.name == PUBLISH_NAME:
                # The child's completion marker belongs to the staging tree and
                # must not be published into a model directory.
                continue
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
            names.append(str(rel).replace("\\", "/"))
    except BaseException:
        # Suppressed, every one of them: a rollback that raises would replace
        # the real failure with a second one and tell the user nothing about
        # either. Moved files first -- a restore needs its target free.
        for src, target in reversed(moved):
            with contextlib.suppress(OSError):
                os.replace(target, src)
        for kept, target in reversed(saved):
            with contextlib.suppress(OSError):
                os.replace(kept, target)
        shutil.rmtree(backup, ignore_errors=True)
        raise
    return names


def undo_into(staging: Path, dest: Path, names: list[str]) -> None:
    """Reverse a publish that the process did not live to finish. Never raises.

    The inverse of :func:`move_into`, driven by the journal rather than by
    in-memory state, because there is none: this runs in a later process.

    A file named in the journal is moved back to staging if it is there to move
    -- so a publish that got halfway is undone halfway, which is exactly the
    half that happened -- and then whatever the backup tree holds is restored
    over the top. Both directions are needed: the first puts back files that
    were *added*, the second puts back files that were *replaced*.
    """
    staging, dest = Path(staging), Path(dest)
    for name in names:
        target = dest / name
        if not target.is_file():
            continue
        back = staging / name
        with contextlib.suppress(OSError):
            back.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, back)
    backup = backup_dir(dest)
    if backup.is_dir():
        for kept in sorted(backup.rglob("*")):
            if kept.is_dir():
                continue
            target = dest / kept.relative_to(backup)
            with contextlib.suppress(OSError):
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(kept, target)
        shutil.rmtree(backup, ignore_errors=True)


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
    write_json(path, {"version": 1, "entries": entries})
    return path


def note_published(root: Path, dest: str, names: list[str]) -> None:
    """Record which files one entry actually published, as it happens.

    Written per destination rather than once at the end, which is the whole
    point: the crash this survives happens *during* the loop, so the journal
    has to be true at every moment of it rather than only at the ends.
    """
    data = read_json(journal_path(root)) or {"version": 1, "entries": []}
    for entry in data.get("entries") or []:
        if entry.get("dest") == dest:
            entry["published"] = names
    write_json(journal_path(root), data)


def finish(root: Path) -> None:
    """The transaction committed. Delete the journal; nothing to recover."""
    with contextlib.suppress(OSError):
        journal_path(root).unlink(missing_ok=True)


def staged_dirs(root: Path) -> set[str]:
    """The staging trees an open journal is protecting, as absolute strings.

    The staging sweep spares these. Without it, a recovery and a sweep race in
    exactly the wrong direction: the sweep sees ``.thing.fetch.part``, calls it
    the litter of an interrupted fetch, and deletes the only copy of the files
    the rollback was about to put back.
    """
    data = read_json(journal_path(root))
    if not data:
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
    for entry in data.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        dest = entry.get("dest")
        staging = entry.get("staging")
        if not dest or not staging:
            continue
        undo_into(Path(staging), Path(dest), list(entry.get("published") or []))
        shutil.rmtree(staging, ignore_errors=True)
        undone.append(str(dest))
    finish(root)
    return undone
