"""The non-blocking stdin reader shared by the workers that need one.

Moved here from ``test_t2i_worker.py`` when ``music_worker`` became its second
caller: the deadlock these two tests pin is a property of
``pipelines/_workerio.py``, not of the image pipeline that first met it.
"""

from __future__ import annotations

import sys

import pytest

# --- the deadlock the polling reader exists to avoid --------------------------


def _run_reader_probe(args: list[str], timeout: float) -> list[str]:
    """Spawn the reproduction harness, write a line, collect its stages."""
    import subprocess
    import threading as _threading
    from pathlib import Path

    script = Path(__file__).parent / "fixtures" / "stdin_reader_deadlock.py"
    proc = subprocess.Popen(
        [sys.executable, str(script), *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    stages: list[str] = []
    finished = _threading.Event()

    def _read() -> None:
        for line in proc.stdout:
            line = line.strip()
            if line:
                stages.append(line)
            if line.startswith("SEEN"):
                finished.set()

    _threading.Thread(target=_read, daemon=True).start()
    try:
        proc.stdin.write("during-the-import\n")
        proc.stdin.flush()
        finished.wait(timeout=timeout)
    finally:
        proc.kill()
        proc.wait(timeout=10)
    return stages


@pytest.mark.skipif(sys.platform != "win32", reason="the deadlock is a Win32 one")
def test_the_polling_reader_does_not_deadlock_a_native_import():
    """A daemon thread reading stdin must not stop the main thread importing.

    Measured 2026-08-22: with a *blocking* reader parked on the inherited pipe,
    `import numpy` never returns -- the faulthandler dump shows the main thread
    inside the loader, creating `_multiarray_umath`. Every blocking flavour
    fails alike (`for line in stdin`, `readline()`, `buffer.readline()`, a bare
    `os.read`), while an idle thread and a thread reading a regular file are
    both fine, so the trigger is a *pending* read on the pipe.

    The worker cannot simply read on its main loop the way `matting_worker`
    does -- a cancel has to be read while a generate is running -- so the reader
    peeks and reads only what has already arrived.
    """
    stages = _run_reader_probe([], timeout=60)
    assert "START" in stages
    assert "IMPORTED" in stages, f"the import never finished: {stages}"
    # And it is still a working reader, not merely a harmless one.
    assert "SEEN during-the-import" in stages, stages


@pytest.mark.skipif(sys.platform != "win32", reason="the deadlock is a Win32 one")
def test_the_naive_blocking_reader_still_reproduces_the_deadlock():
    """The guard above is only meaningful while the thing it guards is real.

    If a future Python or Windows fixes the underlying interaction, this fails
    and `lines_from` can go back to being four lines. Until then it is the
    evidence that the polling is load-bearing rather than superstition.
    """
    # 12 s against an import that takes ~0.1 s when it is not deadlocked: a
    # hundredfold margin, and the whole cost this test adds to the suite.
    stages = _run_reader_probe(["--blocking"], timeout=12)
    assert "START" in stages
    assert "IMPORTED" not in stages, (
        "a blocking stdin reader no longer deadlocks the import; "
        "lines_from's polling may no longer be necessary"
    )
