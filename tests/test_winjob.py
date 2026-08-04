"""The job object is the thing that stops a parent crash from leaving a
16 GB trellis-server holding port 17971. Its contract is unusual: it is
*best-effort* by design, so the tests that matter are the ones proving it
never turns a failure of its own into a failed spawn."""

from __future__ import annotations

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
