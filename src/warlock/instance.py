"""One Warlock per writable resource set, enforced by OS locks.

There was a check before this, and its shape is the finding: startup read a
``session.marker`` file, saw a live pid in it, wrote a **log warning**, and
carried on. The second instance then shared the job database and the engine
port with the first -- the manual's own troubleshooting chapter said it "will
lose fights over both" -- and nothing on screen said so. Worse, the marker is an
ordinary file either process can overwrite or delete, so it could not even be
trusted for the crash attribution it was actually written for.

Three properties an OS lock has that a marker file cannot:

* **A crash releases it.** The kernel drops the lock when the handle closes,
  however the process ended. A marker left behind by a hard kill is
  indistinguishable from a live instance, which is why the old code had to ask
  "is that pid alive" and then guess.
* **It cannot be won twice.** Two processes racing at startup both see an empty
  directory; only one gets the lock.
* **It is not a hint.** The second instance stops, with a dialog, before it has
  touched the store.

Scoped to the home, configured job database and model root rather than to the
machine. Two homes are independent only while those overrideable resources are
independent too; pointing both at one external SQLite file or model directory
must still converge on the same lock.

This module is imported before ``migrate`` and before any store is opened, so it
must stay dependency-free -- stdlib only, no config import at module scope.
"""

from __future__ import annotations

import errno
import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

LOCK_NAME = "instance.lock"
DB_LOCK_SUFFIX = ".warlock-db.lock"
MODEL_LOCK_NAME = ".warlock-models.lock"


def _canonical(path: Path) -> Path:
    """An absolute, symlink-resolved resource name without requiring existence."""
    try:
        return Path(path).resolve(strict=False)
    except OSError:
        return Path(path).absolute()


def lock_paths(config: Any) -> tuple[Path, ...]:
    """Canonical lock files for the home, database and shared model root.

    The lock file lives beside (DB) or inside (models) the protected resource,
    so two homes that spell the same resource through different relative paths
    still converge on one OS lock after symlink resolution.
    """
    home = _canonical(Path(config.home))
    database = _canonical(Path(config.db_path))
    models = _canonical(Path(config.t2i_model_root))
    candidates = (
        home / LOCK_NAME,
        database.parent / f".{database.name}{DB_LOCK_SUFFIX}",
        models / MODEL_LOCK_NAME,
    )
    seen: set[str] = set()
    out: list[Path] = []
    for path in candidates:
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            out.append(path)
    return tuple(out)


class InstanceLock:
    """An exclusive, OS-level lock on one resource lock file.

    Held for the life of the process. ``release`` exists for tests and for an
    orderly shutdown; nothing depends on it being called, which is the point.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None
        self.failure: str | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self, *, allow_unsafe: bool = False) -> bool:
        """Take the lock. False if another process already holds it.

        Never raises. Failure to open or operate the lock fails closed: sharing
        a database and model roots is unsafe unless the caller explicitly opts
        into recovery mode with ``allow_unsafe``. Lock contention is not
        overridable and leaves ``failure`` as ``None``; an infrastructure
        failure records a user-facing reason there.
        """
        if self._fd is not None:
            return True
        self.failure = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            log.warning("could not open the instance lock at %s", self.path, exc_info=True)
            self.failure = f"The instance lock could not be opened: {self.path}"
            return allow_unsafe
        try:
            if not _lock(fd):
                os.close(fd)
                return False
        except OSError:
            log.warning("instance locking is unavailable on this host", exc_info=True)
            os.close(fd)
            self.failure = "The operating system could not lock the Warlock data directory."
            return allow_unsafe
        self._fd = fd
        # Deliberately left empty. The obvious nicety -- writing ``pid=NNN`` for
        # somebody reading the directory -- cannot work here: Windows locks the
        # byte range, so any reader of that byte gets PermissionError while the
        # lock is held, and a file that cannot be read is a worse label than no
        # label. ``session.marker`` beside it already records the pid, the start
        # time and the version for exactly that audience.
        global _current
        _current = self
        return True

    def release(self) -> None:
        """Give the lock back. Safe to call when it was never taken."""
        global _current
        if _current is self:
            _current = None
        fd, self._fd = self._fd, None
        if fd is None:
            return
        with _suppressed():
            _unlock(fd)
        with _suppressed():
            os.close(fd)


class InstanceLocks:
    """Acquire several instance locks as one all-or-none startup guard."""

    def __init__(self, paths: Any) -> None:
        self.locks = tuple(InstanceLock(Path(path)) for path in paths)
        self.blocked: InstanceLock | None = None

    @property
    def held(self) -> bool:
        return bool(self.locks) and all(lock.held for lock in self.locks)

    @property
    def path(self) -> Path | None:
        return self.blocked.path if self.blocked is not None else None

    @property
    def failure(self) -> str | None:
        return self.blocked.failure if self.blocked is not None else None

    def acquire(self, *, allow_unsafe: bool = False) -> bool:
        """Take every lock, releasing earlier ones if any later lock refuses."""
        self.blocked = None
        acquired: list[InstanceLock] = []
        for lock in self.locks:
            if lock.acquire(allow_unsafe=allow_unsafe):
                if lock.held:
                    acquired.append(lock)
                continue
            self.blocked = lock
            for held in reversed(acquired):
                held.release()
            return False
        return True

    def release(self) -> None:
        for lock in reversed(self.locks):
            lock.release()


# The lock this process is holding, if any. Set on a successful acquire so a
# later caller can ask "do *we* already have it" without trying to take it
# again -- which is not a question an OS lock answers: re-locking the same file
# from the same process is either allowed (and proves nothing) or refused (and
# would read as "somebody else has it"), depending on the platform. Doctor's
# single-instance row is the caller that needs the distinction.
_current: InstanceLock | None = None


def held_by_us() -> bool:
    """Whether this process is holding an instance lock right now."""
    return _current is not None and _current.held


def _suppressed() -> Any:
    import contextlib

    return contextlib.suppress(OSError, ValueError)


if sys.platform == "win32":

    def _lock(fd: int) -> bool:
        import msvcrt

        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            # EACCES/EDEADLOCK here is the answer, not a failure: somebody else
            # holds it. Distinguished from "locking does not work at all" by
            # the fact that the call reached the kernel and was refused.
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return False
            raise
        return True

    def _unlock(fd: int) -> None:
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:  # pragma: no cover - the app is Windows-only; this keeps tests portable

    def _lock(fd: int) -> bool:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise
        return True

    def _unlock(fd: int) -> None:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def alert(title: str, message: str) -> None:
    """Say something to the user with no window, no GL and no imgui.

    Startup refusals happen before any of that exists, and a message only in
    ``warlock.log`` is a message nobody reads -- the whole complaint about the
    behaviour this replaces. Falls back to stderr wherever a native box is not
    available, so the text always goes *somewhere* a person might look.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            # MB_OK | MB_ICONWARNING | MB_SETFOREGROUND | MB_TOPMOST
            ctypes.windll.user32.MessageBoxW(None, message, title, 0x30 | 0x10000 | 0x40000)
            return
        except Exception:  # noqa: BLE001 -- a dialog must never be the failure
            log.warning("could not show a native dialog", exc_info=True)
    print(f"{title}: {message}", file=sys.stderr)


#: ``MessageBoxW``'s answer for Yes. Named because the number alone at a call
#: site is the kind of constant a reader has to go and look up.
IDYES = 6


def alert_fatal(title: str, message: str, *, log_dir: Any = None) -> bool:
    """Report a crash on the way out, and offer the log. -> did they say yes?

    UX-06. A crash used to be a traceback in a file and a window that simply
    vanished: the app disappeared, and the one artefact that could explain it
    was somewhere the user had no reason to know about. This is the whole fix,
    and it is deliberately small -- a native box, because by the time it runs
    the GL context and imgui are gone or untrustworthy.

    **MB_YESNO rather than MB_OK.** "Something went wrong, here is a wall of
    text" is a dialog people dismiss; "Open the log folder?" is a question with
    an action behind it, and the folder rather than the file because the log's
    name is not something to have to know. Declining is free and is the
    default nothing is lost by choosing.

    ``log_dir`` is opened with the shell on Yes -- deliberately not a
    subprocess and deliberately outside the kill-on-close job, for the reason
    ``Ctx.open_log`` states: the user's file manager is not ours to kill.

    Never raises. It runs in a ``finally`` on the way out of a process that is
    already failing, and a dialog that threw there would replace a reported
    crash with an unreported one.
    """
    answered = False
    if sys.platform == "win32":
        try:
            import ctypes

            # MB_YESNO | MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST
            flags = 0x4 | 0x10 | 0x10000 | 0x40000
            answered = (
                ctypes.windll.user32.MessageBoxW(None, message, title, flags) == IDYES
            )
        except Exception:  # noqa: BLE001 -- a dialog must never be the failure
            log.warning("could not show a native crash dialog", exc_info=True)
            print(f"{title}: {message}", file=sys.stderr)
    else:
        print(f"{title}: {message}", file=sys.stderr)
    if answered and log_dir is not None:
        try:
            import os

            os.startfile(str(log_dir))  # noqa: S606 -- the shell, on purpose
        except Exception:  # noqa: BLE001
            log.warning("could not open the log folder", exc_info=True)
    return answered
