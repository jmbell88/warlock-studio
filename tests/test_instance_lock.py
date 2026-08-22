"""One Warlock per home, enforced by the OS.

The behaviour this replaces read a ``session.marker`` file, saw a live pid, and
wrote a **log warning** -- then carried on into a shared job database and a
shared engine port, with the manual itself admitting the second instance "will
lose fights over both" (RUN-01).

The tests below are about the three properties a marker file cannot have: a
crash releases it, it cannot be won twice, and losing it stops startup rather
than annotating it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

from warlock import instance


def test_the_first_holder_wins_and_a_second_is_refused(tmp_path):
    path = tmp_path / instance.LOCK_NAME
    first = instance.InstanceLock(path)
    second = instance.InstanceLock(path)
    assert first.acquire() is True
    assert first.held is True
    try:
        # Same process, a different handle: the same question a second launch
        # asks, and it must get the same answer.
        assert second.acquire() is False
        assert second.held is False
    finally:
        first.release()
    # And once released it is free again -- an app that quits and relaunches
    # must not be locked out by its own previous run.
    assert second.acquire() is True
    second.release()


def test_releasing_is_idempotent_and_safe_when_never_taken(tmp_path):
    lock = instance.InstanceLock(tmp_path / instance.LOCK_NAME)
    lock.release()  # never acquired
    assert lock.acquire() is True
    lock.release()
    lock.release()
    assert lock.held is False


def test_lock_infrastructure_failure_fails_closed(tmp_path, monkeypatch):
    def broken(_fd: int) -> bool:
        raise OSError(5, "locking unavailable")

    monkeypatch.setattr(instance, "_lock", broken)
    lock = instance.InstanceLock(tmp_path / instance.LOCK_NAME)
    assert lock.acquire() is False
    assert lock.held is False
    assert lock.failure


def test_lock_infrastructure_failure_requires_explicit_unsafe_override(tmp_path, monkeypatch):
    def broken(_fd: int) -> bool:
        raise OSError(5, "locking unavailable")

    monkeypatch.setattr(instance, "_lock", broken)
    lock = instance.InstanceLock(tmp_path / instance.LOCK_NAME)
    assert lock.acquire(allow_unsafe=True) is True
    assert lock.held is False
    assert lock.failure


def test_the_lock_file_is_a_lock_and_not_a_record(tmp_path):
    """It stays empty on purpose.

    Writing ``pid=NNN`` into it for a human reading the directory looks obvious
    and cannot work: Windows locks the byte *range*, so while the lock is held
    any reader of that byte gets ``PermissionError`` -- a label nobody can read
    is worse than none. ``session.marker`` carries the pid, the start time and
    the version for that audience, and is a plain readable file.
    """
    path = tmp_path / instance.LOCK_NAME
    lock = instance.InstanceLock(path)
    assert lock.acquire()
    try:
        assert path.exists()
        assert path.stat().st_size == 0
    finally:
        lock.release()


def test_held_by_us_tracks_the_live_lock(tmp_path):
    """Doctor's single-instance row needs "do *we* have it", which no OS lock
    answers: re-locking one file from one process proves nothing on some
    platforms and reads as a conflict on others."""
    assert instance.held_by_us() is False
    lock = instance.InstanceLock(tmp_path / instance.LOCK_NAME)
    assert lock.acquire()
    try:
        assert instance.held_by_us() is True
    finally:
        lock.release()
    assert instance.held_by_us() is False


def test_resource_lock_paths_converge_for_two_homes_sharing_external_roots(tmp_path):
    shared_db = tmp_path / "shared" / "jobs.sqlite"
    shared_models = tmp_path / "shared-models"
    one = SimpleNamespace(
        home=tmp_path / "home-one",
        db_path=shared_db,
        t2i_model_root=shared_models,
    )
    two = SimpleNamespace(
        home=tmp_path / "home-two",
        db_path=shared_db.parent / "." / shared_db.name,
        t2i_model_root=shared_models / "child" / "..",
    )

    first = instance.lock_paths(one)
    second = instance.lock_paths(two)
    assert first[0] != second[0]
    assert first[1:] == second[1:]
    assert first[1].name.endswith(instance.DB_LOCK_SUFFIX)
    assert first[2].name == instance.MODEL_LOCK_NAME


def test_shared_database_or_model_root_refuses_a_second_home(tmp_path):
    shared_db = tmp_path / "shared" / "jobs.sqlite"
    shared_models = tmp_path / "shared-models"

    def paths(home):
        config = SimpleNamespace(
            home=home, db_path=shared_db, t2i_model_root=shared_models
        )
        return instance.lock_paths(config)

    first = instance.InstanceLocks(paths(tmp_path / "home-one"))
    second = instance.InstanceLocks(paths(tmp_path / "home-two"))
    assert first.acquire()
    try:
        assert second.acquire() is False
        assert second.path == paths(tmp_path / "home-two")[1]
        assert all(not lock.held for lock in second.locks)
    finally:
        first.release()

    assert second.acquire() is True
    second.release()


def test_a_dead_holders_lock_is_free_again(tmp_path):
    """The property the marker file could never have. A hard kill leaves a
    marker behind that is indistinguishable from a live instance -- which is why
    the old code had to ask "is that pid alive" and then guess. The kernel drops
    an OS lock however the process ended.
    """
    path = tmp_path / instance.LOCK_NAME
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(__import__("pathlib").Path(instance.__file__).parents[2])!r})
        from warlock import instance
        lock = instance.InstanceLock(__import__('pathlib').Path({str(path)!r}))
        assert lock.acquire()
        print("held", flush=True)
        import time; time.sleep(60)
        """
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "held"
        # While it lives, we are locked out.
        ours = instance.InstanceLock(path)
        assert ours.acquire() is False
        # Killed outright -- no teardown, no release call, exactly like a crash.
        child.kill()
        child.wait(timeout=10)
        # Polled rather than asserted once: ``wait()`` returns when the process
        # is gone, but Windows closes its handles (and so drops the lock)
        # asynchronously afterwards. What is under test is that it is released
        # *without anyone releasing it*, not how many milliseconds that takes.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if ours.acquire():
                break
            time.sleep(0.05)
        assert ours.held is True, "a killed holder's lock was never released"
        ours.release()
    finally:
        if child.poll() is None:  # pragma: no cover - only if the kill failed
            child.kill()
            child.wait(timeout=10)
