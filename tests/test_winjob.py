"""The job object is the thing that stops a parent crash from leaving a
16 GB trellis-server holding port 17971. Its contract is unusual: it is
*best-effort* by design, so the tests that matter are the ones proving it
never turns a failure of its own into a failed spawn."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from warlock import winjob

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Job Objects")


@pytest.fixture(autouse=True)
def _fresh_job(monkeypatch):
    # The handle is process-wide and cached; a test that forces a failure must
    # not poison the cache for the next one.
    monkeypatch.setattr(winjob, "_job", None)


@windows_only
def test_a_live_child_is_assigned_to_the_job():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        assert winjob.assign(proc.pid) is True
    finally:
        proc.kill()
        proc.wait(timeout=10)


@windows_only
def test_the_job_handle_is_created_once_and_reused():
    first = winjob._ensure_job()
    assert first is not None
    assert winjob._ensure_job() == first


def test_assigning_a_pid_that_cannot_be_opened_returns_false_rather_than_raising():
    # 0 is never a real pid, so OpenProcess fails: the status quo ante (a child
    # that outlives a parent crash), not a reason to fail the spawn.
    assert winjob.assign(0) is False


def test_assign_is_a_no_op_when_the_job_is_unavailable(monkeypatch):
    monkeypatch.setattr(winjob, "_ensure_job", lambda: None)
    assert winjob.assign(4321) is False


def test_everything_is_a_no_op_off_windows(monkeypatch):
    monkeypatch.setattr(winjob.sys, "platform", "linux")
    assert winjob._ensure_job() is None
    assert winjob.assign(4321) is False
    assert winjob.listener_pid(17971) is None
    assert winjob.image_path(4321) is None
    assert winjob.terminate(4321) is False


# --- identifying whatever holds the port -------------------------------------


@windows_only
def test_a_listening_socket_is_traced_back_to_this_process():
    """The bind precheck could only say "in use"; the whole point of reclaiming
    an orphan is being able to name who is holding it."""
    import socket

    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        assert winjob.listener_pid(port) == os.getpid()
    finally:
        sock.close()
    assert winjob.listener_pid(port) is None


@windows_only
def test_a_pid_names_its_own_executable():
    """The only evidence that an orphan is ours rather than a stranger's."""
    path = winjob.image_path(os.getpid())
    assert path is not None and "python" in path.lower()


@windows_only
def test_a_dead_pid_has_no_image_and_cannot_be_terminated():
    assert winjob.image_path(999_999_999) is None
    assert winjob.terminate(999_999_999) is False
    assert winjob.image_path(0) is None
    assert winjob.terminate(0) is False


@windows_only
def test_terminating_really_kills_the_process():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        assert winjob.terminate(proc.pid) is True
        assert proc.wait(timeout=10) is not None
    finally:
        if proc.poll() is None:  # pragma: no cover - only if terminate failed
            proc.kill()
            proc.wait(timeout=10)


def test_the_child_registry_is_a_plain_add_remove_snapshot():
    """Tracking is separate from the job object and answers a different
    question. The job object says "these die when this process does"; the
    registry says "these are alive *now*", which is what a shutdown that leaves
    the interpreter running needs in order to stop them (MDL-02)."""
    before = winjob.tracked()
    winjob.track(999_999_001, "a fetch")
    winjob.track(999_999_002, "a bake")
    try:
        live = winjob.tracked()
        assert live[999_999_001] == "a fetch"
        assert live[999_999_002] == "a bake"
        # A snapshot, not the live dict: mutating what you got back must not
        # reach into the registry.
        live.clear()
        assert winjob.tracked()

        winjob.untrack(999_999_001)
        assert 999_999_001 not in winjob.tracked()
        # Untracking something unknown is a no-op, not an error: every reap
        # path calls it and several of them run during an unwind.
        winjob.untrack(999_999_001)
    finally:
        winjob.untrack(999_999_001)
        winjob.untrack(999_999_002)
    assert winjob.tracked() == before


def test_terminating_the_tracked_set_empties_it_even_for_dead_pids():
    """Best-effort by contract, like every other call in this module. A pid that
    has already exited is not a reason to raise on the way out of a shutdown --
    but it must still leave the registry, or the next shutdown tries again."""
    winjob.track(999_999_003, "already gone")
    try:
        winjob.terminate_tracked()
        assert 999_999_003 not in winjob.tracked()
    finally:
        winjob.untrack(999_999_003)


@windows_only
def test_terminate_tracked_actually_stops_a_live_child():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        winjob.track(proc.pid, "a sleeper")
        assert proc.pid in winjob.terminate_tracked()
        assert proc.wait(timeout=10) is not None
        assert proc.pid not in winjob.tracked()
    finally:
        winjob.untrack(proc.pid)
        if proc.poll() is None:  # pragma: no cover - only if terminate failed
            proc.kill()
            proc.wait(timeout=10)
