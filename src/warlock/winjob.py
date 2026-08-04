"""Kill-on-close Windows Job Objects for child processes.

Warlock spawns two long-lived children that own GPU memory: trellis-server.exe
(~16 GB VRAM plus a CUDA context) and the Blender rig worker. Neither was in a
job object, so when the parent died -- which it did, repeatedly, to commit
exhaustion -- the children survived.

That is not merely untidy. An orphaned trellis-server keeps port 17971, and
ensure_started's health poll cannot tell the orphan from the server it just
spawned (/health returns a bare ok with no identity field), so the fresh server
dies on bind while generates get silently routed to a stale process. The
trellis.log record of the crash shows exactly that: 30 startup banners against
28 generate requests, in two distinct restart storms.

Assigning each child to a job with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE makes the
kernel do the cleanup. The job handle is closed when the parent exits by any
means, including a hard kill, and the children go with it.

Every function is a no-op off Windows and swallows failures: a child that could
not be assigned is the status quo ante, not a reason to fail the job.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes

log = logging.getLogger(__name__)

_JobObjectExtendedLimitInformation = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001

if sys.platform == "win32":

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(wintypes.ULONG)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


_job: int | None = None


def _ensure_job() -> int | None:
    """The process-wide kill-on-close job handle, created on first use.

    One job for all children rather than one each: the semantics wanted are
    "everything Warlock spawned dies with Warlock", and a single handle cannot
    get out of sync with itself.
    """
    global _job
    if _job is not None:
        return _job
    if sys.platform != "win32":
        return None
    try:
        kernel32 = ctypes.windll.kernel32
        # Handle-returning calls need an explicit restype: ctypes defaults to
        # c_int, which truncates a 64-bit HANDLE to its low 32 bits.
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            log.warning("CreateJobObject failed (%s)", ctypes.get_last_error())
            return None
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            handle,
            _JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            log.warning("SetInformationJobObject failed; not using a job object")
            kernel32.CloseHandle(handle)
            return None
    except (OSError, AttributeError):
        log.warning("job objects unavailable on this host", exc_info=True)
        return None
    _job = handle
    return _job


def assign(pid: int) -> bool:
    """Put `pid` in the kill-on-close job. True if it took.

    Best-effort by contract. A False here means the child will outlive a parent
    crash exactly as it did before -- worth a log line, never worth failing the
    spawn that was about to succeed.
    """
    job = _ensure_job()
    if job is None:
        return False
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    handle = kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
    if not handle:
        log.warning("could not open pid %d to assign it to the job object", pid)
        return False
    try:
        ok = bool(kernel32.AssignProcessToJobObject(job, handle))
        if not ok:
            log.warning("AssignProcessToJobObject failed for pid %d", pid)
        return ok
    finally:
        kernel32.CloseHandle(handle)
