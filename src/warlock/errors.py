"""Turn raw exceptions into a sentence a user can act on.

The full traceback always goes to the job's error.log; only the short,
friendly sentence goes in the DB and the UI.
"""

from __future__ import annotations

import errno
import subprocess
import traceback
from pathlib import Path

import httpx

# How much of an unrecognised exception reaches the DB and the inspector. Two
# lines of wrapped text: enough to carry a real sentence, short enough that a
# child process's whole stderr cannot become the job's error message.
MAX_MESSAGE = 200


def _one_line(text: str) -> str:
    """The first line, whitespace collapsed, bounded.

    The fallback used to be ``str(exc)`` verbatim, which is fine for a sentence
    and wrong for everything else: a child process's error carries its whole
    stderr, and that went into the DB column and straight onto the inspector --
    the exact "traceback fragment in the UI" that ``studio/tasks.py`` takes
    care to keep out of a toast.
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    line = " ".join(lines[0].split())
    if len(line) > MAX_MESSAGE:
        line = line[: MAX_MESSAGE - 3].rstrip() + "..."
    return line


def _subprocess_message(exc: subprocess.SubprocessError) -> str:
    """A helper program's failure, named by the program rather than by argv.

    Both the Blender worker and gltfpack are spawned with a full path and a
    long argument list, and ``str(CalledProcessError)`` repeats the whole of it
    -- which says nothing a user can act on and buries what did.
    """
    cmd = getattr(exc, "cmd", None)
    first = cmd[0] if isinstance(cmd, (list, tuple)) and cmd else cmd
    name = Path(str(first)).name if first else "A helper program"
    if isinstance(exc, subprocess.TimeoutExpired):
        return f"{name} did not finish in time. See warlock.log."
    code = getattr(exc, "returncode", None)
    exit_part = f" (exit {code})" if code is not None else ""
    return f"{name} failed{exit_part}. See warlock.log for what it printed."


def friendly(exc: Exception) -> str:
    text = str(exc).lower()
    # Before the substring checks: the host running out of commit and the card
    # running out of VRAM are different resources with different remedies, and
    # telling the second user to drop to 512 sends them after the wrong one.
    if isinstance(exc, MemoryError):
        return (
            "Out of system memory. Windows spills an oversized GPU allocation "
            "into host RAM rather than refusing it, so close other "
            "applications or set WARLOCK_VRAM_EXCLUSIVE=1."
        )
    if "out of memory" in text or "cuda oom" in text:
        return (
            "GPU out of memory — try resolution 512, close other GPU apps, "
            "or set WARLOCK_VRAM_EXCLUSIVE=1."
        )
    if "invalid glb" in text or "glb with no meshes" in text:
        return (
            "The 3D engine returned an invalid model — it may have run out of "
            "memory or crashed mid-reconstruction. See assets/trellis.log."
        )
    if isinstance(exc, httpx.TransportError):
        return "The 3D engine stopped unexpectedly. See assets/trellis.log."
    if isinstance(exc, subprocess.SubprocessError):
        return _subprocess_message(exc)
    if isinstance(exc, OSError):
        if exc.errno == errno.ENOSPC:
            return (
                "The drive Warlock writes to is full. Free some space, or "
                "prune old assets from the library."
            )
        if isinstance(exc, PermissionError):
            target = Path(str(exc.filename)).name if exc.filename else ""
            what = f" {target}" if target else ""
            return (
                f"Warlock could not write{what} — another program may have the "
                "file open. Close it and try again."
            )
    return _one_line(str(exc)) or exc.__class__.__name__


def write_error_log(job_dir: Path, exc: Exception) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "error.log").write_text(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    )
