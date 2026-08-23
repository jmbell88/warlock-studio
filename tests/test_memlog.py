"""memlog is the instrumentation that makes the next commit-exhaustion crash
attributable, so what it has to guarantee is that it never raises and never
lies: it is called from inside `finally` blocks and from the frame loop's
ticker, where an exception would be worse than the missing reading."""

from __future__ import annotations

import sys

import pytest

from warlock import memlog

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="Win32 memory counters"
)


@windows_only
def test_process_memory_reports_a_plausible_private_commit():
    proc = memlog.process_memory()
    assert proc is not None
    # A live CPython with pytest loaded is comfortably over 10 MiB and nowhere
    # near a terabyte; the point is to catch a unit error (pages vs bytes),
    # not to pin a number.
    assert 0.01 < proc.private < 1024
    assert 0.01 < proc.working_set < 1024


@windows_only
def test_system_commit_is_reported_in_gib_not_pages(real_system_memory):
    # ``real_system_memory`` undoes the suite-wide roomy pin: this test is about
    # what the real reader returns, and a pinned one would pass it by fiat.
    sysmem = memlog.system_memory()
    assert sysmem is not None
    assert 0 < sysmem.commit_total < sysmem.commit_limit
    # 4 GiB is below any machine that runs this app; the pages-as-bytes bug
    # would land here at ~1/4096th of the true figure.
    assert sysmem.commit_limit > 4
    assert 0.0 < sysmem.commit_fraction < 1.0


def test_commit_fraction_does_not_divide_by_a_zero_limit():
    assert memlog.SystemMemory(commit_total=1.0, commit_limit=0.0).commit_fraction == 0.0


def test_summary_returns_none_rather_than_raising_when_both_readings_fail(monkeypatch):
    monkeypatch.setattr(memlog, "process_memory", lambda: None)
    monkeypatch.setattr(memlog, "system_memory", lambda: None)
    assert memlog.summary() is None


def test_summary_survives_one_half_being_unavailable(monkeypatch):
    monkeypatch.setattr(memlog, "process_memory", lambda: None)
    monkeypatch.setattr(
        memlog,
        "system_memory",
        lambda: memlog.SystemMemory(commit_total=40.0, commit_limit=80.0),
    )
    assert memlog.summary() == "commit 40.0/80.0 GiB (50%)"


def test_readings_are_none_off_windows(monkeypatch, real_system_memory):
    # The real reader for the same reason as above: the platform guard under
    # test lives inside it, and the suite-wide pin does not have one.
    monkeypatch.setattr(memlog.sys, "platform", "linux")
    assert memlog.process_memory() is None
    assert memlog.system_memory() is None


# --- children ----------------------------------------------------------------


@windows_only
def test_child_commit_reads_a_real_process():
    """The reading that stops a subprocess being invisible in the log.

    A matting worker measured 6.56 GiB of private commit on 2026-08-21 while
    the app's own idle-tick line said 15.9 GiB -- so the log under-reported
    Warlock's charge against the ceiling by 40%, and the sweep that would have
    freed the largest single piece of it looked unimportant. ``os.getpid()``
    stands in for a child here: what is being pinned is that a *pid* resolves
    to a plausible figure at all, in GiB rather than bytes or pages.
    """
    import os

    total = memlog.children_private([os.getpid()])
    assert total is not None
    assert 0.01 < total < 1024


@windows_only
def test_child_commit_skips_a_pid_that_is_gone_rather_than_failing():
    """A child reaped between ``tracked()`` and this call is the ordinary case,
    not an error: the sweep that kills them runs on the same loop this reads
    from. An unknown pid contributes nothing and the rest still count."""
    import os

    # 0xFFFFFFF0 is not a live pid on any machine; OpenProcess simply fails.
    mixed = memlog.children_private([os.getpid(), 0xFFFFFFF0])
    alone = memlog.children_private([os.getpid()])
    assert mixed is not None and alone is not None
    assert abs(mixed - alone) < 0.5


def test_child_commit_of_nothing_is_zero_not_none():
    """None means "cannot read"; an empty child set is a *known* zero, and the
    caller formats the two differently."""
    assert memlog.children_private([]) == 0.0


def test_child_commit_is_none_off_windows(monkeypatch):
    monkeypatch.setattr(memlog.sys, "platform", "linux")
    assert memlog.children_private([1234]) is None


@windows_only
def test_summary_names_child_commit_when_there_is_any():
    """The log line is the artifact; a figure that never reaches it is not
    instrumentation. ``memlog`` takes pids rather than importing ``winjob``,
    which is what keeps it stdlib-only and callable from a ``finally``."""
    import os

    line = memlog.summary(children=[os.getpid()])
    assert line is not None
    assert "children" in line, line
    assert "commit" in line


@windows_only
def test_summary_omits_the_child_clause_when_there_are_none():
    """An idle session with no subprocess must not grow a "children 0.0 GiB"
    column: the line is read by eye across thousands of ticks."""
    line = memlog.summary(children=[])
    assert line is not None
    assert "children" not in line
    line_default = memlog.summary()
    assert line_default is not None
    assert "children" not in line_default


# --- the reading that was understating the app by a third --------------------


@windows_only
def test_child_commit_counts_the_interpreter_behind_the_trampoline():
    """The defect the 2026-08-22 session log showed, as a test.

    `sys.executable` under a uv venv is a trampoline that spawns the real
    interpreter as its own child. Summing over the pids `Popen` returned reads
    the ~0.8 MB shim and misses everything the worker actually holds -- which
    is how an idle tick printed `children 0.0 GiB` while a BiRefNet worker held
    6.3 GiB, at the moment the app was deciding whether to admit the next job
    (docs/measurements/2026-08-22-trampoline-child-pids.md).

    Asserted as a comparison rather than an absolute: what must hold is that
    the job-derived set sees the weight and the Popen-derived one does not.
    """
    import subprocess

    from warlock import winjob

    hold = (
        "import sys, time; buf = bytearray(400 * 1024 * 1024);"
        " sys.stdout.write('up' + chr(10)); sys.stdout.flush(); time.sleep(30)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", hold],
        stdout=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        winjob.assign(proc.pid)
        winjob.track(proc.pid, "test holder")
        assert proc.stdout.readline().strip() == "up"

        measured = memlog.children_private(winjob.measured_pids())
        assert measured is not None
        # 400 MiB is 0.39 GiB; allow generous slack for the interpreter itself
        # and for a machine that trims aggressively.
        assert measured > 0.3, f"the 400 MiB holder was not counted: {measured}"

        if winjob.job_pids():
            # Only meaningful where a trampoline is genuinely in play. On an
            # installer layout the two sets are equal and there is nothing to
            # compare -- which is correct behaviour, not a skip-worthy gap.
            popen_only = memlog.children_private([proc.pid])
            assert popen_only is not None
            if popen_only < 0.3:
                assert measured > popen_only, (
                    "the job-derived reading must see what the Popen pid hides"
                )
    finally:
        winjob.untrack(proc.pid)
        proc.kill()
        proc.wait(timeout=10)
