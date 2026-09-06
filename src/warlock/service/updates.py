"""Asking whether there is a newer Warlock, on the user's say-so, out of process.

The third narrow exception to the offline invariant, and the same shape as the
first two: nothing here touches the network. It plans, refuses, spawns
``python -m warlock.pipelines.update_worker``, and reads that child's progress
lines. Reachable from two buttons in the app-Settings pane, from an *opt-in*
startup check, and from nothing on the job path.

Blocking by contract, like every other multi-second call in the service layer:
the pane dispatches it through ``TaskRunner``.

**Why this refuses less than ``packs`` does.** A pack install writes into the
``site-packages`` the app is running out of, so it earns two whole-plan
refusals before it may start. This writes one file into a staging directory
under the user's Warlock home that nothing else reads, and *never runs it*: the
pane offers a button that opens the verified installer, and the user decides.
So Cancel is safe at every point -- there is no commit phase, and the worst a
kill leaves behind is a ``.part`` nothing reads as a downloaded installer.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import installed_version, winjob
from .core import WarlockService
from .errors import Invalid

log = logging.getLogger(__name__)

# Two small GETs, plus the spawn. Bounded because a child parked on a stalled
# socket holds a task-pool worker forever; generous because a first request to
# a cold host over a slow line is seconds.
CHECK_TIMEOUT = 120.0

# One installer over a domestic line. ``fetch``'s reasoning with a smaller
# number, because this is hundreds of megabytes rather than sixteen gigabytes.
DOWNLOAD_TIMEOUT = 2 * 60 * 60.0

_STDERR_KEEP_LINES = 40

#: ``percent, label``. No ``phase``: unlike a pack install there is no
#: "cancel is no longer safe" line to be on either side of.
Progress = Callable[[float, str], None]

#: The reason the child is tracked under, so Cancel can stop *this* download
#: without taking a live Blender bake or the matting worker with it.
TRACK_REASON = "update"

_NUMBER = re.compile(r"\d+")


def worker_argv() -> list[str]:
    """How the child is started. A function rather than an inline literal so a
    test can put a stub in its place and exercise this half -- the spawn, the
    stdin hand-over, the progress lines, the result file -- without any test in
    this project reaching the network."""
    return [sys.executable, "-m", "warlock.pipelines.update_worker"]


def staging_dir(svc: WarlockService) -> Path:
    """Where a downloaded installer waits to be run.

    Under the user's Warlock home beside the wheel cache, and for the same
    reason: it survives the reinstall it exists to perform, so a user who
    downloads an installer and then closes the app still has it.
    """
    return svc.config.home / "updates"


def _version_tuple(text: str) -> tuple[int, ...]:
    """``"v0.0.37"`` -> ``(0, 0, 37)``. Tolerant, because a tag is typed.

    Not semver and deliberately not ``packaging``: these versions are plain
    sequential ``MAJOR.MINOR.PATCH`` integers (see ``warlock.__version__``'s
    own comment), so a tuple comparison over the numbers in the string is the
    whole of the question, and a dependency here would be one more thing the
    installer has to carry to answer it.
    """
    return tuple(int(found) for found in _NUMBER.findall(text or ""))


def _is_newer(latest: str | None, current: str) -> bool:
    """Whether ``latest`` is a version worth offering over ``current``.

    False for None, for equal, and for a *lower* published version -- the last
    of which is the case that matters on a machine running a build newer than
    the last release, where offering a "newer" version would be offering a
    downgrade.
    """
    if not latest:
        return False
    them, us = _version_tuple(latest), _version_tuple(current)
    if not them:
        return False
    return them > us


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(svc: WarlockService, *, timeout: float = CHECK_TIMEOUT) -> dict[str, Any]:
    """Ask the release feed what the latest version is. Blocking.

    The answer always carries ``current`` and ``available`` so the pane draws
    from one dict rather than re-deriving the comparison it is showing.
    """
    result = _run_worker(
        {"mode": "check"},
        on_progress=lambda _p, _l: None,
        timeout=timeout,
    )
    current = installed_version()
    latest = result.get("latest")
    result["current"] = current
    result["available"] = _is_newer(str(latest) if latest else None, current)
    return result


def download(
    svc: WarlockService,
    info: dict[str, Any],
    *,
    on_progress: Progress | None = None,
    timeout: float = DOWNLOAD_TIMEOUT,
) -> dict[str, Any]:
    """Fetch the installer this check described, verified against its digest.

    ``info`` is a ``check`` result rather than four loose arguments, because the
    digest and the URL have to come from the *same* answer: pairing a URL from
    one check with a digest from another is the one way this could verify
    something nobody published together.
    """
    for field in ("installer_url", "installer_name", "sha256"):
        if not str(info.get(field) or ""):
            raise Invalid("That update has nothing to download; check for updates again.")
    return _run_worker(
        {
            "mode": "download",
            "installer_url": str(info["installer_url"]),
            "installer_name": str(info["installer_name"]),
            "size_bytes": int(info.get("size_bytes") or 0),
            "sha256": str(info["sha256"]),
            "dest_dir": str(staging_dir(svc)),
        },
        on_progress=on_progress or (lambda _p, _l: None),
        timeout=timeout,
    )


def staged_installer(svc: WarlockService, info: dict[str, Any]) -> Path | None:
    """The already-downloaded installer for ``info``, if it is there and right.

    Verified rather than merely present, and that is the point: a ``.part``
    renamed by hand, a half-written file left by a disk that filled, or an
    installer from a different release with the same name are all things the
    pane would otherwise offer as "ready" and the user would double-click. The
    digest is what makes "ready" mean anything.
    """
    name = str(info.get("installer_name") or "")
    digest = str(info.get("sha256") or "").lower()
    if not name or not digest:
        return None
    found = staging_dir(svc) / name
    if not found.is_file():
        return None
    try:
        return found if _sha256(found) == digest else None
    except OSError:
        return None


def _kill_and_reap(proc: subprocess.Popen[str]) -> None:
    with contextlib.suppress(Exception):
        proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=10)
    winjob.untrack(proc.pid)


def _run_worker(
    spec: dict[str, Any], *, on_progress: Progress, timeout: float
) -> dict[str, Any]:
    """One child, one question. ``packs._run_worker``'s protocol exactly.

    The spec goes over stdin and the answer comes back through a file: stdout
    carries progress lines, and a stray print must not be able to corrupt the
    result. stderr is drained from the start rather than read after exit,
    because a child whose stderr outgrows the OS pipe buffer blocks on its next
    write and never reaches the exit that read was waiting on.
    """
    with tempfile.TemporaryDirectory(prefix="warlock-update-") as scratch:
        result_path = Path(scratch) / "result.json"
        spec = dict(spec)
        spec["result_path"] = str(result_path)
        proc = subprocess.Popen(
            worker_argv(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        # The same kill-on-close job every other child goes in. This one holds
        # a socket and writes hundreds of megabytes; an orphan of it would fill
        # the disk with nothing on screen to say so.
        winjob.assign(proc.pid)
        winjob.track(proc.pid, TRACK_REASON)
        assert proc.stdin is not None and proc.stdout is not None
        assert proc.stderr is not None

        err_lines: deque[str] = deque(maxlen=_STDERR_KEEP_LINES)

        def _pump_err(stream: Any) -> None:
            try:
                for raw in stream:
                    err_lines.append(raw)
            except (OSError, ValueError):
                pass

        threading.Thread(target=_pump_err, args=(proc.stderr,), daemon=True).start()

        def _send() -> None:
            try:
                proc.stdin.write(json.dumps(spec))
                proc.stdin.close()
            except OSError:
                pass

        threading.Thread(target=_send, daemon=True).start()

        lines: queue.Queue[str | None] = queue.Queue()

        def _pump(stream: Any) -> None:
            try:
                for raw in stream:
                    lines.put(raw)
            finally:
                lines.put(None)

        threading.Thread(target=_pump, args=(proc.stdout,), daemon=True).start()

        deadline = time.monotonic() + timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Drain what is already queued before calling this a
                    # timeout: the deadline can elapse in the same poll tick
                    # the child's stdout closes, and a download that finished
                    # must not be reported as having timed out.
                    try:
                        raw = lines.get_nowait()
                    except queue.Empty:
                        raise subprocess.TimeoutExpired(proc.args, timeout) from None
                else:
                    try:
                        raw = lines.get(timeout=min(remaining, 1.0))
                    except queue.Empty:
                        continue
                if raw is None:
                    break
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(payload, dict):
                    continue
                try:
                    percent = float(payload.get("percent") or 0.0)
                except (TypeError, ValueError):
                    continue
                on_progress(percent, str(payload.get("label") or ""))
            code = proc.wait(timeout=max(deadline - time.monotonic(), 1.0))
            winjob.untrack(proc.pid)
        except subprocess.TimeoutExpired:
            _kill_and_reap(proc)
            raise Invalid("The update check timed out.") from None
        except BaseException:
            _kill_and_reap(proc)
            raise

        result: dict[str, Any] = {}
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except ValueError:
                result = {}
    if not result.get("ok"):
        detail = result.get("error") or f"the update worker exited with code {code}"
        if not result.get("error"):
            tail = " ".join(x.strip() for x in list(err_lines)[-4:] if x.strip())
            if tail:
                detail += f": {tail}"
        log.warning("update worker failed: %s", detail)
        raise Invalid(f"Could not check for updates: {detail}")
    return result
