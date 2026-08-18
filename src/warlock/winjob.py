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
import subprocess
import sys
import threading
from ctypes import wintypes
from typing import Any

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
# Callers genuinely span threads (warlock-loop, TaskRunner workers, the main
# thread's doctor probe); an unlocked create-and-store can mint two jobs and
# leak one, leaving armed() reporting on a handle some children are not in.
_job_lock = threading.Lock()


def _ensure_job() -> int | None:
    """The process-wide kill-on-close job handle, created on first use.

    One job for all children rather than one each: the semantics wanted are
    "everything Warlock spawned dies with Warlock", and a single handle cannot
    get out of sync with itself.
    """
    with _job_lock:
        return _ensure_job_locked()


def _ensure_job_locked() -> int | None:
    global _job
    if _job is not None:
        return _job
    if sys.platform != "win32":
        return None
    try:
        kernel32 = ctypes.windll.kernel32
        # Full prototypes, not only the restypes. A handle-returning call needs
        # an explicit ``restype`` because ctypes defaults to ``c_int``, which
        # truncates a 64-bit HANDLE to its low 32 bits -- but the *arguments*
        # need declaring for the same reason: without ``argtypes`` ctypes
        # guesses from the Python value, and a handle that happens to fit in 32
        # bits is passed as an int while one above 2^31 is not. It works until
        # the machine is busy enough to hand out a large handle, which is the
        # worst possible failure schedule (RUN-04).
        _declare(kernel32)
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            # GetLastError directly: windll builds functions with
            # use_last_error=False, so ctypes.get_last_error() always reads 0.
            # Read *immediately*, before any other call can overwrite it.
            log.warning("CreateJobObject failed (%s)", kernel32.GetLastError())
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
            error = kernel32.GetLastError()
            log.warning(
                "SetInformationJobObject failed (%s); not using a job object", error
            )
            kernel32.CloseHandle(handle)
            return None
    except (OSError, AttributeError):
        log.warning("job objects unavailable on this host", exc_info=True)
        return None
    _job = handle
    return _job


def _declare(kernel32: Any) -> None:
    """Give every handle-taking call this module makes a full signature.

    Idempotent and cheap -- ctypes caches the function objects, so re-declaring
    is assigning the same attributes again. Declared in one place rather than
    beside each call so a new call site cannot quietly inherit the guesswork.
    """
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.GetLastError.restype = wintypes.DWORD
    kernel32.GetLastError.argtypes = []
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]


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
    _declare(kernel32)
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


# Every child this process has spawned and not yet reaped, by pid. The job
# object already guarantees they die when *this* process does -- but shutdown is
# reached on paths where the interpreter carries on, and a 4-hour fetch child or
# a Blender bake parked on a socket is then simply left running with nothing
# holding a handle to it (MDL-02). The registry is what lets shutdown say "and
# stop those too" instead of trusting an exit that may not come.
#
# Weak in intent, strong in practice: entries are removed on reap, and a pid
# that has already exited is a no-op to terminate. A dict rather than a set so
# the reason a child is running can be logged when one has to be killed.
_children: dict[int, str] = {}
_children_lock = threading.Lock()


def track(pid: int, what: str) -> None:
    """Record a live child so ``terminate_tracked`` can find it."""
    with _children_lock:
        _children[pid] = what


def untrack(pid: int) -> None:
    """Forget a child that has been reaped. Safe for an unknown pid."""
    with _children_lock:
        _children.pop(pid, None)


def tracked() -> dict[int, str]:
    """A snapshot of the live children, for diagnostics and shutdown."""
    with _children_lock:
        return dict(_children)


def terminate_tracked(what_prefix: str | None = None) -> list[int]:
    """Kill every tracked child, or only those whose recorded reason starts
    with ``what_prefix``. -> the pids that were asked to stop.

    The filter exists for the download Cancel button, which must stop the
    fetch child without taking a live Blender bake or the persistent matting
    worker with it; shutdown keeps the no-argument kill-everything semantics.

    Best-effort, like everything else here: a child that has already exited,
    or that this process cannot open, is not a reason to raise on the way out
    of a shutdown. The job object remains the backstop for a hard kill.
    """
    killed: list[int] = []
    for pid, what in tracked().items():
        if what_prefix is not None and not what.startswith(what_prefix):
            continue
        log.info("terminating tracked child %d (%s)", pid, what)
        if terminate(pid):
            killed.append(pid)
        untrack(pid)
    return killed


def armed() -> bool:
    """True if the kill-on-close job exists and children are being assigned.

    ``_ensure_job`` failing was previously visible only as a log line, which
    makes a host where the guarantee quietly did not take indistinguishable
    from one where it did. This is what the doctor row calls.
    """
    return _ensure_job() is not None


def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """``subprocess.run``, but the child is in the kill-on-close job.

    ``subprocess.run`` spawns and waits in one call, which leaves no moment to
    assign the pid -- so a short-lived child (the bpy probe, gltfpack) outlived
    a hard kill of the parent exactly as the long-lived ones used to. This is
    the same three steps `run` performs, with the assignment in the gap.

    ``capture_output``/``text``/``timeout`` behave as they do on
    ``subprocess.run``; on a timeout the child is killed and
    ``TimeoutExpired`` is raised, matching it.
    """
    timeout = kwargs.pop("timeout", None)
    if kwargs.pop("capture_output", False):
        kwargs.setdefault("stdout", subprocess.PIPE)
        kwargs.setdefault("stderr", subprocess.PIPE)
    kwargs.setdefault("creationflags", getattr(subprocess, "CREATE_NO_WINDOW", 0))
    with subprocess.Popen(argv, **kwargs) as proc:
        assign(proc.pid)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


# --- identifying (and reclaiming) whatever holds a port ----------------------
#
# The job object makes an orphan impossible going forward, but it cannot undo
# one: a trellis-server stranded by a crash that predates the fix still holds
# 17971, and the only thing the app could say about it was "run Get-Process
# yourself". These three answer "who is that, and is it ours" -- the same
# ctypes / no-op-off-Windows / swallow-failures contract as everything above,
# and psutil is not a dependency.

_TCP_TABLE_OWNER_PID_ALL = 5
_AF_INET = 2
_MIB_TCP_STATE_LISTEN = 2
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_INSUFFICIENT_BUFFER = 122

# How many times the TCP table is re-sized before giving up. The table is
# live: connections open between the sizing call and the reading one, and the
# second call then answers ERROR_INSUFFICIENT_BUFFER with the new size rather
# than filling the buffer. Retrying is the documented contract; not retrying
# meant a busy machine silently degraded "who holds this port" to "nobody
# knows", which is the answer the whole section exists to stop giving. Bounded
# because a machine whose table grows on every attempt is not going to settle.
_TCP_TABLE_ATTEMPTS = 5


if sys.platform == "win32":

    class _MIB_TCPROW_OWNER_PID(ctypes.Structure):
        _fields_ = [
            ("dwState", wintypes.DWORD),
            ("dwLocalAddr", wintypes.DWORD),
            ("dwLocalPort", wintypes.DWORD),
            ("dwRemoteAddr", wintypes.DWORD),
            ("dwRemotePort", wintypes.DWORD),
            ("dwOwningPid", wintypes.DWORD),
        ]


def listener_pid(port: int) -> int | None:
    """The pid listening on 127.0.0.1:`port`, or None if it cannot be found."""
    if sys.platform != "win32":
        return None
    try:
        iphlpapi = ctypes.windll.iphlpapi
        size = wintypes.DWORD(0)
        # First call sizes the buffer; it is expected to fail.
        iphlpapi.GetExtendedTcpTable(
            None, ctypes.byref(size), False, _AF_INET, _TCP_TABLE_OWNER_PID_ALL, 0
        )
        for _ in range(_TCP_TABLE_ATTEMPTS):
            buf = ctypes.create_string_buffer(size.value)
            rc = iphlpapi.GetExtendedTcpTable(
                buf, ctypes.byref(size), False, _AF_INET, _TCP_TABLE_OWNER_PID_ALL, 0
            )
            if rc == 0:
                break
            if rc != _ERROR_INSUFFICIENT_BUFFER:
                log.debug("GetExtendedTcpTable failed (%d)", rc)
                return None
            # The table grew between the two calls, and `size` now holds what it
            # takes today. Round again rather than reporting no owner.
        else:
            log.debug("GetExtendedTcpTable kept growing; giving up")
            return None
        count = ctypes.cast(buf, ctypes.POINTER(wintypes.DWORD)).contents.value
        rows = ctypes.cast(
            ctypes.byref(buf, ctypes.sizeof(wintypes.DWORD)),
            ctypes.POINTER(_MIB_TCPROW_OWNER_PID),
        )
        for i in range(count):
            row = rows[i]
            # dwLocalPort is in network byte order in the low 16 bits.
            local = ((row.dwLocalPort & 0xFF) << 8) | ((row.dwLocalPort >> 8) & 0xFF)
            if row.dwState == _MIB_TCP_STATE_LISTEN and local == port:
                return int(row.dwOwningPid)
    except (OSError, AttributeError, ValueError):
        log.debug("could not read the TCP table", exc_info=True)
    return None


def image_path(pid: int) -> str | None:
    """The full path of `pid`'s executable, or None.

    The only thing that can distinguish an orphan we may kill from a stranger's
    process that happens to hold the port.
    """
    if sys.platform != "win32" or pid <= 0:
        return None
    try:
        kernel32 = ctypes.windll.kernel32
        # The full _declare, not a lone OpenProcess restype: CloseHandle,
        # TerminateProcess and QueryFullProcessImageNameW were only typed if
        # _ensure_job had happened to run first, and an undeclared HANDLE
        # argument truncates through the default c_int above 2^31 -- the
        # exact failure schedule documented on _ensure_job_locked.
        _declare(kernel32)
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            size = wintypes.DWORD(32768)
            buf = ctypes.create_unicode_buffer(size.value)
            if not kernel32.QueryFullProcessImageNameW(
                handle, 0, buf, ctypes.byref(size)
            ):
                return None
            return buf.value or None
        finally:
            kernel32.CloseHandle(handle)
    except (OSError, AttributeError):
        log.debug("could not read the image path of pid %d", pid, exc_info=True)
        return None


def terminate(pid: int) -> bool:
    """Kill `pid`. True if the call succeeded."""
    if sys.platform != "win32" or pid <= 0:
        return False
    try:
        kernel32 = ctypes.windll.kernel32
        _declare(kernel32)  # image_path's reasoning: never inherit the guesswork
        handle = kernel32.OpenProcess(_PROCESS_TERMINATE, False, pid)
        if not handle:
            log.warning("could not open pid %d to terminate it", pid)
            return False
        try:
            return bool(kernel32.TerminateProcess(handle, 1))
        finally:
            kernel32.CloseHandle(handle)
    except (OSError, AttributeError):
        log.warning("could not terminate pid %d", pid, exc_info=True)
        return False
