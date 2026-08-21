"""Host-memory readings, stdlib only.

The 2026-08-03 crash was a *commit* exhaustion (Windows Resource-Exhaustion
Detector event 2004), not a CUDA OOM: python.exe was reported at 138-153 GB of
virtual memory across seven separate sessions while the GPU sat at 32 GB free.
Nothing in the app recorded host memory, so the cause had to be inferred.
These readings are what makes the next occurrence attributable.

stdlib only on purpose. psutil happens to import in the current venv but is not
declared in pyproject, so relying on it would make instrumentation that exists
precisely for a rare crash the first thing to break on a clean install. The two
Win32 calls needed are three ctypes structs.

Every function returns None off Windows or when the call fails -- logging must
never be the thing that raises inside a `finally`.
"""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Iterable
from ctypes import wintypes
from dataclasses import dataclass

_GIB = 1024**3


@dataclass(frozen=True)
class ProcessMemory:
    """Per-process figures, in GiB."""

    working_set: float
    """Resident physical memory."""
    private: float
    """PrivateUsage -- private *commit*, the number that hits the commit limit.

    This, not working_set, is what the pagefile has to back.
    """


@dataclass(frozen=True)
class SystemMemory:
    """System-wide commit accounting, in GiB."""

    commit_total: float
    commit_limit: float

    @property
    def commit_fraction(self) -> float:
        return self.commit_total / self.commit_limit if self.commit_limit else 0.0


if sys.platform == "win32":

    class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    class _PERFORMANCE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("CommitTotal", ctypes.c_size_t),
            ("CommitLimit", ctypes.c_size_t),
            ("CommitPeak", ctypes.c_size_t),
            ("PhysicalTotal", ctypes.c_size_t),
            ("PhysicalAvailable", ctypes.c_size_t),
            ("SystemCache", ctypes.c_size_t),
            ("KernelTotal", ctypes.c_size_t),
            ("KernelPaged", ctypes.c_size_t),
            ("KernelNonpaged", ctypes.c_size_t),
            ("PageSize", ctypes.c_size_t),
            ("HandleCount", wintypes.DWORD),
            ("ProcessCount", wintypes.DWORD),
            ("ThreadCount", wintypes.DWORD),
        ]


def process_memory() -> ProcessMemory | None:
    """This process's working set and private commit, or None."""
    if sys.platform != "win32":
        return None
    try:
        counters = _PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        # restype is load-bearing: GetCurrentProcess returns the pseudo-handle
        # (HANDLE)-1, and with ctypes' default c_int restype that becomes a
        # 32-bit -1 which is passed to a 64-bit HANDLE parameter as
        # 0x00000000FFFFFFFF -- a different, invalid handle. Every call then
        # fails with ERROR_INVALID_HANDLE and the reading silently goes None.
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        handle = kernel32.GetCurrentProcess()
        # GetProcessMemoryInfo lives in psapi.dll but is forwarded from
        # kernel32 as K32GetProcessMemoryInfo on Vista+; psapi.dll is the
        # spelling that works on every supported build.
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            return None
    except (OSError, AttributeError):
        return None
    return ProcessMemory(
        working_set=counters.WorkingSetSize / _GIB,
        private=counters.PrivateUsage / _GIB,
    )


def system_memory() -> SystemMemory | None:
    """System commit charge and limit, or None."""
    if sys.platform != "win32":
        return None
    try:
        info = _PERFORMANCE_INFORMATION()
        info.cb = ctypes.sizeof(info)
        ok = ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb)
        if not ok:
            return None
    except (OSError, AttributeError):
        return None
    # The Commit* fields are in pages, not bytes -- PageSize is in the same
    # struct precisely because it varies.
    page = info.PageSize
    return SystemMemory(
        commit_total=info.CommitTotal * page / _GIB,
        commit_limit=info.CommitLimit * page / _GIB,
    )


#: ``PROCESS_QUERY_LIMITED_INFORMATION``. Deliberately not
#: ``PROCESS_QUERY_INFORMATION``: the limited right is enough for
#: ``GetProcessMemoryInfo`` and is grantable for a process this one did not
#: necessarily create, so a reading never depends on the handle inheritance of
#: whichever code path spawned the child.
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def children_private(pids: Iterable[int]) -> float | None:
    """Total private commit, in GiB, of the given pids. None off Windows.

    The app's own ``process_memory`` cannot see a subprocess, and Warlock's
    subprocesses are not incidental: the BiRefNet matting worker was measured
    at **6.56 GiB of private commit** on 2026-08-21 while the idle-tick line
    reported 15.9 GiB for the app -- so the log understated what this
    application was charging against the commit limit by 40%, and the idle
    sweep that kills that child looked like a rounding error rather than the
    largest single thing it could give back.

    A pid that cannot be opened contributes nothing rather than failing the
    whole reading: children are killed by the same idle sweep whose effect this
    measures, so one disappearing between ``winjob.tracked()`` and this call is
    the ordinary case and not an error. An empty set is ``0.0`` and never
    ``None`` -- "no children" and "cannot read" are different facts and the
    caller renders them differently.

    Never raises, for the module's standing reason: it is called from ``finally``
    blocks and from the frame loop.
    """
    if sys.platform != "win32":
        return None
    total = 0.0
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = wintypes.HANDLE
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        for pid in pids:
            handle = kernel32.OpenProcess(
                _PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
            )
            if not handle:
                continue
            try:
                counters = _PROCESS_MEMORY_COUNTERS_EX()
                counters.cb = ctypes.sizeof(counters)
                if psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb
                ):
                    total += counters.PrivateUsage / _GIB
            finally:
                kernel32.CloseHandle(handle)
    except (OSError, AttributeError, TypeError, ValueError):
        return None
    return total


def summary(children: Iterable[int] | None = None) -> str | None:
    """A one-line host-memory summary for the log, or None if unavailable.

    Three halves now, and they answer three questions: `private` says whether
    *we* are the leak, `children` says whether one of our subprocesses is, and
    `commit` says how close the machine is to the wall the crash hit.

    ``children`` is a set of pids -- ``winjob.tracked()`` at every call site --
    rather than something this module looks up itself, because ``memlog``
    imports nothing from the package and is called from inside ``finally``
    blocks where an import cycle or a lock would be the worse failure.

    The clause is omitted entirely when there are no children, so an idle
    session does not grow a permanent "children 0.0 GiB" column on a line that
    is read by eye across thousands of ticks.
    """
    proc = process_memory()
    sysmem = system_memory()
    if proc is None and sysmem is None:
        return None
    parts = []
    if proc is not None:
        parts.append(f"private {proc.private:.1f} GiB, ws {proc.working_set:.1f} GiB")
    if children is not None:
        kids = children_private(children)
        if kids:
            parts.append(f"children {kids:.1f} GiB")
    if sysmem is not None:
        parts.append(
            f"commit {sysmem.commit_total:.1f}/{sysmem.commit_limit:.1f} GiB"
            f" ({sysmem.commit_fraction * 100:.0f}%)"
        )
    return "; ".join(parts)
