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
