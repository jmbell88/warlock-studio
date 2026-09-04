"""Reading a request line from a worker's stdin without ever blocking on it.

Shared by every worker that has to read stdin *while* it is busy -- which today
means ``text2image_worker`` and ``music_worker``, both of which must notice a
``cancel`` in the middle of a diffusion loop. A worker that reads only on its
main loop (``matting_worker``, ``blender_worker``) does not need any of this and
should keep iterating stdin directly.

It lives on its own because the Win32 plumbing below is ~60 lines of
``ctypes.windll.kernel32`` with nothing pipeline-specific in it, and a second
hand-copy is where copying stops being defensible -- the same argument
``test_every_subprocess_spawn_is_in_the_kill_on_close_job`` already makes for
``winjob.assign``. Everything *else* about a worker (``_Server``, ``MARKER``,
``handle``, ``serve``, ``main``) stays duplicated per worker on purpose, so that
two worker modules diff cleanly against each other.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from ctypes import wintypes
from typing import Any

STDIN_POLL_SECONDS = 0.05
"""How often the reader thread asks whether a request has arrived.

Only reached while the pipe is empty. It bounds how late a *cancel* can be
noticed, and 50 ms is far below the duration of the diffusion step a cancel
interrupts -- while being long enough that an idle child costs nothing.
"""

_STD_INPUT_HANDLE = -10


def peek_stdin(handle: Any) -> int:
    """Bytes already readable on the stdin pipe, or -1 if it is finished."""
    avail = wintypes.DWORD(0)
    ok = ctypes.windll.kernel32.PeekNamedPipe(
        handle, None, 0, None, ctypes.byref(avail), None
    )
    return int(avail.value) if ok else -1


def lines_from(stdin: Any) -> Any:
    """Yield stdin lines *without ever leaving a read pending*.

    **This is not a style choice; a blocking read here deadlocks the process.**
    Measured 2026-08-22, minimally reproducible: a daemon thread parked in a
    read on the inherited stdin pipe stops the main thread's very next native
    extension import dead. `import numpy` takes 0.1 s with no such thread and
    never returns with one -- the faulthandler dump shows the main thread inside
    `_bootstrap_external.create_module`, loading numpy's `_multiarray_umath`
    DLL, indefinitely.

    Controls place the cause exactly: an *idle* second thread is fine, and a
    second thread blocked reading a *regular file* is fine. Every way of
    blocking on the pipe fails identically -- `for line in sys.stdin`,
    `readline()`, `sys.stdin.buffer.readline()` and a bare `os.read(0, ...)` --
    so it is below Python's IO layer, in the interaction between a pending
    synchronous pipe read and the Windows image loader.

    `matting_worker` never met this because it reads stdin *on its main loop*
    and imports before ever blocking. A worker with a cancel op cannot: a cancel
    has to be read while a generate is running, which is precisely a concurrent
    reader. So the reader peeks first and reads only what is already there,
    leaving no outstanding read for an import to trip over. Verified against the
    same reproduction: numpy and torch import at full speed, and a line written
    during the import still arrives.

    Falls back to plain iteration off Windows and for anything that is not the
    real stdin -- the in-process tests drive `serve` with a `StringIO`, which
    has no pipe to peek at and no loader to deadlock.
    """
    if sys.platform != "win32":
        yield from stdin
        return
    try:
        if stdin.fileno() != 0:
            yield from stdin
            return
    except (AttributeError, OSError, ValueError):
        yield from stdin
        return

    kernel32 = ctypes.windll.kernel32
    kernel32.GetStdHandle.restype = wintypes.HANDLE
    kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    handle = kernel32.GetStdHandle(_STD_INPUT_HANDLE)
    buf = b""
    while True:
        available = peek_stdin(handle)
        if available < 0:
            # The parent closed its end: a broken pipe is how this loop is
            # meant to finish, not an error to report.
            return
        if available == 0:
            time.sleep(STDIN_POLL_SECONDS)
            continue
        try:
            chunk = os.read(0, available)
        except OSError:
            return
        if not chunk:
            return
        buf += chunk
        while b"\n" in buf:
            line, _, buf = buf.partition(b"\n")
            yield line.decode("utf-8", "replace") + "\n"
