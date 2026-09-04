"""Move a pre-``~/.warlock`` library out of the source tree, once.

Every root used to be computed off the checkout -- ``PROJECT_ROOT / "assets"``,
``/ "bench"``, ``/ "palettes"``, ``/ "models"`` -- so a user's generated work
lived *inside* the repository: invisible to anyone who had not cloned it,
destroyed by a `git clean`, and gone on the next reinstall. :mod:`warlock.config`
now points all four at ``~/.warlock``, and this module is what makes that a
change of address rather than a change of subject.

**Pure, in the ``vram.py`` sense.** Stdlib only -- no ``service``, no ``queue``,
no ``studio``. ``config.get_config()`` calls :func:`run` and everything in the
app calls ``get_config``, so an import here is an import in every process the
project has, including the Blender worker and the offline fetch subprocess.

**The order is copy, verify, delete, and it is not negotiable.** The move is
cross-volume in the case it was written for (a checkout on D:, a home on C:),
so it is a byte copy and not a rename -- there is no atomic form of it. What
there *is* is an order in which no failure loses data: nothing is removed until
both sides have been recounted and agree, and the copy lands in a staging
directory beside the destination so a crash halfway through leaves a
``.incoming`` to delete rather than a half-populated library that the next start
would mistake for a finished move.

Three preconditions, all checked before the first byte is copied:

* **Nothing else is live.** A second Warlock writing into ``assets/`` while its
  contents are copied out from under it produces a library that is missing
  whatever it wrote. Tested by taking ``BEGIN EXCLUSIVE`` on the legacy
  ``jobs.sqlite`` -- a real lock rather than ``session.marker``, which survives
  a crash and would block the migration forever.
* **The destination volume has room.** Measured against the legacy trees plus
  10%, because the failure mode otherwise is a disk filled to zero by a copy
  that then cannot be rolled back cheaply.
* **The user has not already chosen.** Any root whose own ``WARLOCK_*``
  variable is set is left exactly where the user pointed it, and so is one
  whose destination *is* its source (``WARLOCK_HOME`` aimed back at the
  checkout). ``WARLOCK_NO_MIGRATE`` turns the whole thing off without examining
  anything.

``WARLOCK_MIGRATE_KEEP=1`` runs the copy and the verify and then keeps the
source, for a user who would rather delete 95 GB by hand than trust this.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sqlite3
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover -- import cycle; config imports this module
    from .config import Config

from .config import PROJECT_ROOT

# The name of the breadcrumb left in the new home. "Where did my library go" is
# a question asked in the place it went to, so that is where the answer lives.
BREADCRUMB = "MIGRATED.txt"

# How much slack over the measured size the destination volume must have.
SPACE_MARGIN = 1.1

# What the one migration this process performed actually moved. Recorded
# because :func:`run` is called from inside ``config.get_config()``, which runs
# before ``studio.main._setup_logging`` has a file handler -- the progress lines
# go to stderr at the time, and this is what lets the log say the same thing
# once there is somewhere to say it.
MOVED: list[str] = []

# (legacy directory name under PROJECT_ROOT, Config field, the field's own
# environment variable). Smallest first, so a failure on ``models`` -- the 95 GB
# one, and the only one likely to run out of room -- still leaves the half the
# user actually looks at already home.
_ROOTS: tuple[tuple[str, str, str], ...] = (
    ("assets", "data_dir", "WARLOCK_DATA_DIR"),
    ("bench", "bench_dir", "WARLOCK_BENCH_DIR"),
    ("palettes", "palette_dir", "WARLOCK_PALETTE_DIR"),
    ("models", "t2i_model_root", "WARLOCK_T2I_ROOT"),
)


class MigrationError(RuntimeError):
    """The move could not be made safely. Nothing has been deleted."""


def _tree_size(root: Path) -> tuple[int, int]:
    """``(files, bytes)`` under ``root``, following no symlinks."""
    files = 0
    total = 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                stack.append(Path(entry.path))
            else:
                files += 1
                with contextlib.suppress(OSError):
                    total += entry.stat(follow_symlinks=False).st_size
    return files, total


def _is_empty(path: Path) -> bool:
    try:
        return not any(os.scandir(path))
    except OSError:
        return True


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"  # pragma: no cover -- unreachable, the loop ends at TB


def _pending(config: Config) -> list[tuple[Path, Path]]:
    """The (legacy, destination) pairs that still have to move."""
    out: list[tuple[Path, Path]] = []
    for name, field_name, env in _ROOTS:
        if os.environ.get(env):
            # The user has already said where this one goes.
            continue
        legacy = PROJECT_ROOT / name
        dest = getattr(config, field_name)
        if legacy.resolve() == dest.resolve():
            # WARLOCK_HOME aimed back at the checkout: this *is* home.
            continue
        if not legacy.is_dir() or _is_empty(legacy):
            continue
        if dest.exists() and not _is_empty(dest):
            # A populated destination is a library in its own right. Merging is
            # not a thing this can get right unattended -- two jobs.sqlite files
            # have no join -- so it declines and says nothing further.
            continue
        out.append((legacy, dest))
    return out


@contextlib.contextmanager
def _no_live_writer() -> Iterator[None]:
    """Hold the legacy store exclusively for the whole migration.

    The check this replaces opened the database, ran ``BEGIN EXCLUSIVE`` /
    ``ROLLBACK``, and *closed the connection* -- proving only that nothing was
    live at that instant, and then dropping the guarantee before a cross-volume
    copy of tens of gigabytes began. A second Warlock launched during those
    minutes passed its own identical precondition, opened the legacy store, and
    wrote into ``assets/``; this process then finished copying, verified against
    the sizes it measured *before* those writes, and ``rmtree``'d the legacy
    root on top of them. The recount catches added files, but an in-place
    modification of equal size is invisible to it, so the good case is a loud
    failure and the bad case is a silently split library (RUN-02).

    So the transaction stays open across every ``_move`` -- the measure, the
    copy and the verify. The ``rmtree`` of the legacy trees happens *after* it
    is released, deliberately: the hold is an open handle on the legacy
    ``jobs.sqlite``, and Windows will not unlink a tree that contains one. That
    ordering is safe because by then the destinations are published and
    ``_pending`` skips a populated destination -- a legacy tree that survives
    costs disk space, never a split library. The single-instance lock (RUN-01) now makes
    a second Warlock on the same home impossible in the first place; this is the
    same guarantee taken at the level of the thing actually being moved, which
    still holds when the two processes have different homes and one of them is
    migrating the shared legacy directory.
    """
    db = PROJECT_ROOT / "assets" / "jobs.sqlite"
    if not db.exists():
        yield
        return
    try:
        conn = sqlite3.connect(str(db), timeout=0)
    except sqlite3.Error as exc:  # pragma: no cover -- a corrupt or unreadable file
        raise MigrationError(f"cannot open {db}: {exc}") from exc
    try:
        try:
            conn.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            raise MigrationError(
                f"another Warlock process is using {db} -- close it and start again "
                f"(moving a live library would lose whatever it is writing)"
            ) from exc
        except sqlite3.DatabaseError:
            # Not a database at all -- truncated, foreign, or a leftover of some
            # older layout. Nothing can be holding it as one, which is the only
            # question this asks, so the move goes ahead and copies the file
            # verbatim like every other byte in the tree. No transaction to
            # hold, and nothing to roll back.
            yield
            return
        try:
            yield
        finally:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
    finally:
        conn.close()


def _require_space(config: Config, needed: int) -> None:
    required = int(needed * SPACE_MARGIN)
    try:
        free = shutil.disk_usage(config.home.anchor).free
    except OSError as exc:  # pragma: no cover -- an unreadable volume root
        raise MigrationError(f"cannot measure free space on {config.home.anchor}: {exc}") from exc
    if free < required:
        raise MigrationError(
            f"not enough room on {config.home.anchor} to move the library into "
            f"{config.home}: {_human(required)} required, {_human(free)} free. "
            f"Set WARLOCK_HOME={PROJECT_ROOT} to keep everything where it is, or "
            f"point one root elsewhere with its own variable "
            f"(WARLOCK_DATA_DIR, WARLOCK_T2I_ROOT, ...)."
        )


def _move(legacy: Path, dest: Path, files: int, total: int, *, remove: bool = True) -> None:
    """Copy ``legacy`` to ``dest`` via a staging directory, verify, then delete.

    ``remove=False`` stops after publishing the destination and leaves the
    legacy tree in place for the caller to delete. That is what :func:`run`
    passes, because the exclusive hold it takes on the legacy ``jobs.sqlite``
    keeps an open handle on a file *inside* ``legacy`` -- and on Windows a file
    with an open handle cannot be unlinked, so deleting under the lock would
    silently leave the store behind (``rmtree(..., ignore_errors=True)``) and
    the next start would see a non-empty legacy root and decline to migrate
    again. The copy and the verify are the window that needs protecting; by the
    time the destination is published, a writer that opens the legacy store is
    writing into an orphan.
    """
    staging = dest.parent / f"{dest.name}.incoming"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(staging, ignore_errors=True)
    try:
        shutil.copytree(legacy, staging, symlinks=True)
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise MigrationError(f"could not copy {legacy} to {dest}: {exc}") from exc

    copied_files, copied_bytes = _tree_size(staging)
    if (copied_files, copied_bytes) != (files, total):
        shutil.rmtree(staging, ignore_errors=True)
        raise MigrationError(
            f"the copy of {legacy} did not match the original "
            f"({copied_files} files / {copied_bytes} bytes copied, "
            f"{files} / {total} expected). Nothing has been deleted."
        )

    if dest.exists():
        # Empty -- _pending skipped it otherwise -- and os.replace will not
        # rename onto an existing directory even so.
        try:
            dest.rmdir()
        except OSError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise MigrationError(f"could not clear {dest}: {exc}") from exc
    os.replace(staging, dest)

    if not remove or os.environ.get("WARLOCK_MIGRATE_KEEP") == "1":
        return
    shutil.rmtree(legacy, ignore_errors=True)


def _breadcrumb(config: Config, moved: list[tuple[Path, Path]]) -> None:
    lines = [f"Warlock moved its data here on {datetime.now():%Y-%m-%d %H:%M}."]
    for legacy, dest in moved:
        lines.append(f"  {legacy}  ->  {dest}")
    lines.append("")
    try:
        config.home.mkdir(parents=True, exist_ok=True)
        with (config.home / BREADCRUMB).open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError:
        # Advisory. A note that could not be written is not a reason to fail a
        # move that has already succeeded.
        pass


def _carry_the_database(config: Config, moved: list[tuple[Path, Path]]) -> None:
    """Put the migrated ``jobs.sqlite`` where a custom ``WARLOCK_DB`` points.

    ``_ROOTS`` moves the legacy ``assets`` tree as one unit keyed on
    ``WARLOCK_DATA_DIR``, and ``db_path`` normally sits *inside* it, so the
    store rides along for free. It does not have to: pointing ``WARLOCK_DB`` at
    a fast volume over a spinning-disk library is a documented, advertised
    setup. A user in that setup with an unmigrated legacy tree got their assets
    moved correctly, the old ``jobs.sqlite`` moved with them, and ``JobStore``
    then opening the custom path, finding nothing, and silently creating an
    empty database -- every job's files on disk, unread, and the whole verdict
    corpus with them. Nothing warned.

    Copied rather than refused, because the user's intent is unambiguous and a
    hard stop at startup over a config they deliberately set helps nobody. Only
    ever onto a path that does not exist: a database already there is a library
    in its own right, and merging two of them is the thing ``_pending`` already
    declines to attempt.

    Through ``sqlite3.Connection.backup``, the same as ``JobStore.backup_to`` --
    not ``shutil.copy2``. The store the mover just relocated runs in WAL mode,
    so the committed database is the ``.sqlite`` file *plus* whatever is still
    in its ``-wal`` sidecar; a plain byte copy of the ``.sqlite`` alone silently
    drops every transaction since the last checkpoint. ``_ROOTS`` moving
    ``assets`` as a directory tree carries the ``-wal``/``-shm`` files along for
    free, but this path copies just the one named file onto a location
    ``WARLOCK_DB`` chose, so it has to take a consistent snapshot itself.
    """
    store = config.db_path
    if store.exists():
        return
    for _legacy, dest in moved:
        candidate = dest / "jobs.sqlite"
        if not candidate.is_file() or candidate.resolve() == store.resolve():
            continue
        try:
            store.parent.mkdir(parents=True, exist_ok=True)
            # Staged, like every other write onto a served name: a half-copied
            # sqlite file at the path the app is about to open is worse than no
            # file at all, which at least reads as "new library".
            tmp = store.with_name(store.name + ".migrating")
            source = sqlite3.connect(str(candidate))
            try:
                target = sqlite3.connect(str(tmp))
                try:
                    source.backup(target)
                finally:
                    target.close()
            finally:
                source.close()
            os.replace(tmp, store)
        except (OSError, sqlite3.Error) as exc:  # pragma: no cover -- a full or read-only volume
            raise MigrationError(
                f"the library moved to {dest}, but its job history could not be "
                f"copied to {store}: {exc}"
            ) from exc
        print(f"warlock: job history copied to {store}.", file=sys.stderr, flush=True)
        return


def run(config: Config) -> list[str]:
    """Move any legacy root into ``config.home``. -> the roots that moved.

    Idempotent by construction: a successful run leaves nothing behind at the
    legacy paths, so every later call is four ``is_dir()`` checks.
    """
    if os.environ.get("WARLOCK_NO_MIGRATE"):
        return []
    try:
        pending = _pending(config)
    except OSError:  # pragma: no cover -- an unreadable PROJECT_ROOT
        return []
    if not pending:
        return []

    moved: list[tuple[Path, Path]] = []
    # The exclusive hold spans the measure, the copy and the verify; the
    # deletes wait below until it is released -- see ``_no_live_writer`` for
    # why the boundary sits there. Measuring inside it as well is not
    # incidental: sizes taken before the lock could already be stale by the time
    # the copy starts, and the verify compares against them.
    with _no_live_writer():
        sizes = [_tree_size(legacy) for legacy, _ in pending]
        _require_space(config, sum(total for _, total in sizes))

        for (legacy, dest), (files, total) in zip(pending, sizes, strict=True):
            # stderr rather than the log: this runs inside get_config(), which is
            # called long before studio.main installs a file handler, and a 95 GB
            # copy with no output at all is indistinguishable from a hang.
            print(
                f"warlock: moving {legacy} ({_human(total)}) to {dest} -- "
                f"this happens once.",
                file=sys.stderr,
                flush=True,
            )
            _move(legacy, dest, files, total, remove=False)
            print(f"warlock: moved {dest.name}.", file=sys.stderr, flush=True)
            moved.append((legacy, dest))

    _carry_the_database(config, moved)

    # Outside the hold, and deliberately last. The destinations are already
    # published, so nothing is at risk here except the disk space the legacy
    # trees occupy -- and the open handle on the legacy ``jobs.sqlite`` has to
    # be gone before Windows will let it be unlinked. A failure to delete is not
    # a failed migration: ``_pending`` skips a root whose destination is
    # populated, so a leftover legacy tree costs space and nothing else.
    if os.environ.get("WARLOCK_MIGRATE_KEEP") != "1":
        for legacy, _dest in moved:
            shutil.rmtree(legacy, ignore_errors=True)

    _breadcrumb(config, moved)
    MOVED.extend(str(dest) for _, dest in moved)
    return [str(dest) for _, dest in moved]
