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


def _require_no_live_writer() -> None:
    db = PROJECT_ROOT / "assets" / "jobs.sqlite"
    if not db.exists():
        return
    try:
        conn = sqlite3.connect(str(db), timeout=0)
    except sqlite3.Error as exc:  # pragma: no cover -- a corrupt or unreadable file
        raise MigrationError(f"cannot open {db}: {exc}") from exc
    try:
        conn.execute("BEGIN EXCLUSIVE")
        conn.execute("ROLLBACK")
    except sqlite3.OperationalError as exc:
        raise MigrationError(
            f"another Warlock process is using {db} -- close it and start again "
            f"(moving a live library would lose whatever it is writing)"
        ) from exc
    except sqlite3.DatabaseError:
        # Not a database at all -- truncated, foreign, or a leftover of some
        # older layout. Nothing can be holding it as one, which is the only
        # question this function asks, so the move goes ahead and copies the
        # file verbatim like every other byte in the tree.
        pass
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


def _move(legacy: Path, dest: Path, files: int, total: int) -> None:
    """Copy ``legacy`` to ``dest`` via a staging directory, verify, then delete."""
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

    if os.environ.get("WARLOCK_MIGRATE_KEEP") == "1":
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

    _require_no_live_writer()

    sizes = [_tree_size(legacy) for legacy, _ in pending]
    _require_space(config, sum(total for _, total in sizes))

    moved: list[tuple[Path, Path]] = []
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
        _move(legacy, dest, files, total)
        print(f"warlock: moved {dest.name}.", file=sys.stderr, flush=True)
        moved.append((legacy, dest))

    _breadcrumb(config, moved)
    MOVED.extend(str(dest) for _, dest in moved)
    return [str(dest) for _, dest in moved]
