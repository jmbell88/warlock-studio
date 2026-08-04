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
def test_system_commit_is_reported_in_gib_not_pages():
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


def test_readings_are_none_off_windows(monkeypatch):
    monkeypatch.setattr(memlog.sys, "platform", "linux")
    assert memlog.process_memory() is None
    assert memlog.system_memory() is None
