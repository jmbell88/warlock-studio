"""Fetching a model's weights, on the user's explicit say-so, out of process.

The narrow exception to the offline invariant, and every part of its narrowness
lives here. The generation pipeline never calls anything in this module; only a
button in the app-Settings pane does. Nothing here downloads either -- it plans
(``warlock.fetch``, pure), refuses (disk), spawns
``python -m warlock.pipelines.fetch_worker``, and reads that child's progress
lines. The variable that would make *this* process online-capable is set in the
child's own environment and nowhere else.

Blocking by contract, like every other multi-second call in the service layer:
the pane dispatches it through ``TaskRunner``.
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import fetch as fetch_mod
from .. import winjob
from .core import WarlockService
from .errors import Invalid, NotFound

log = logging.getLogger(__name__)

# Wall-clock ceiling for one repository fetch. Generous because 16 GB over a
# slow line is genuinely long, bounded because a child parked on a stalled
# socket holds a task-pool worker forever.
FETCH_TIMEOUT = 4 * 60 * 60.0

Progress = Callable[[float, str], None]


def worker_argv() -> list[str]:
    """How the child is started. A function rather than an inline literal so a
    test can put a stub in its place and exercise this half -- the spawn, the
    stdin hand-over, the progress lines, the result file -- without any test in
    this project ever making a network call."""
    return [sys.executable, "-m", "warlock.pipelines.fetch_worker"]


def rows(svc: WarlockService) -> list[dict[str, Any]]:
    """Every downloadable registry entry with a real presence flag.

    A flag rather than the word "missing" inside a label. The pane used to
    infer missing-ness from the substring, which is brittle in the obvious way
    and, less obviously, silently gave the style LoRAs no marking at all --
    nothing was putting the word into their labels.
    """
    config = svc.config
    out: list[dict[str, Any]] = []
    for entry in fetch_mod.entries():
        jobs = fetch_mod.plan(config, [entry])
        out.append(
            {
                "row_key": entry.row_key,
                "kind": entry.kind,
                "key": entry.key,
                "label": entry.label,
                "present": entry.is_present(config),
                "size_gib": fetch_mod.total_gib(jobs),
                "downloadable": bool(jobs),
            }
        )
    return out


def plan_for(svc: WarlockService, row_keys: list[str]) -> list[fetch_mod.Job]:
    """The deduped set of fetches these rows need. Raises on an unknown row.

    Deduped across the whole selection rather than per row, which is the only
    reason ticking all four SDXL 1.0 recipes asks for 7 GB rather than 28.
    """
    chosen: list[fetch_mod.Entry] = []
    for row_key in row_keys:
        entry = fetch_mod.find(row_key)
        if entry is None:
            raise NotFound(f"No such model: {row_key}")
        chosen.append(entry)
    return fetch_mod.plan(svc.config, chosen)


def download(
    svc: WarlockService,
    row_keys: list[str],
    *,
    on_progress: Progress | None = None,
    timeout: float = FETCH_TIMEOUT,
) -> dict[str, Any]:
    """Fetch everything these rows need. Blocking; raises ``Invalid`` on refusal.

    The disk check is at the door and refuses the *whole* plan, the way sweep
    admission does: half a plan downloaded is a set of model directories some
    of which are complete and none of which the user asked for individually.
    """
    jobs = plan_for(svc, row_keys)
    if not jobs:
        raise Invalid("There is nothing to download for that.")
    refusal = fetch_mod.disk_refusal(jobs)
    if refusal is not None:
        raise Invalid(refusal)

    total = fetch_mod.total_gib(jobs) or float(len(jobs))
    done_gib = 0.0
    for index, job in enumerate(jobs, start=1):
        share = job.size_gib or (total / len(jobs))

        def report(percent: float, label: str, _done=done_gib, _share=share) -> None:
            if on_progress is None:
                return
            overall = 100.0 * (_done + _share * percent / 100.0) / total
            on_progress(min(overall, 99.0), label)

        report(0.0, f"{job.repo_id} ({index} of {len(jobs)})")
        _run_worker(job, on_progress=report, timeout=timeout)
        done_gib += share
    if on_progress is not None:
        on_progress(100.0, "")
    return {"fetched": [job.repo_id for job in jobs]}


def _run_worker(
    job: fetch_mod.Job, *, on_progress: Progress, timeout: float
) -> dict[str, Any]:
    """One child, one repository. Raises ``Invalid`` with the child's own words.

    The spec goes over stdin and the answer comes back through a file, matching
    ``rigging.run_worker`` -- stdout carries progress lines and a stray print
    from ``huggingface_hub`` must not be able to corrupt the result.
    """
    with tempfile.TemporaryDirectory(prefix="warlock-fetch-") as scratch:
        result_path = Path(scratch) / "result.json"
        spec = job.spec()
        spec["result_path"] = str(result_path)
        proc = subprocess.Popen(
            worker_argv(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        # The same kill-on-close job every other child goes in. A fetch holds a
        # socket and writes gigabytes; one that outlived a hard kill of the app
        # would go on filling the disk with nothing on screen to say so.
        winjob.assign(proc.pid)
        assert proc.stdin is not None and proc.stdout is not None

        def _send() -> None:
            try:
                proc.stdin.write(json.dumps(spec))
                proc.stdin.close()
            except OSError:
                # A worker that died before draining stdin says far more
                # through its exit code and its result file than this would.
                pass

        writer = threading.Thread(target=_send, daemon=True)
        writer.start()

        # stdout is drained on a helper thread so the *whole* fetch has a
        # deadline, not just the wait() after EOF -- ``rigging.run_worker``'s
        # pattern, and here for a sharper reason: the child's _Sampler emits a
        # progress line every half second whether or not bytes are arriving, so
        # a child parked on a stalled socket never closes stdout at all. Reading
        # it inline meant `timeout` was never consulted and the task-pool worker
        # was held forever.
        lines: queue.Queue[str | None] = queue.Queue()

        def _pump(stream: Any) -> None:
            try:
                for raw in stream:
                    lines.put(raw)
            finally:
                lines.put(None)

        reader = threading.Thread(target=_pump, args=(proc.stdout,), daemon=True)
        reader.start()

        deadline = time.monotonic() + timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(proc.args, timeout)
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
                on_progress(float(payload.get("percent") or 0.0), str(payload.get("label") or ""))
            code = proc.wait(timeout=max(deadline - time.monotonic(), 0.0))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise Invalid(f"The download of {job.repo_id} timed out.") from None
        except BaseException:
            proc.kill()
            proc.wait()
            raise

        result: dict[str, Any] = {}
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except ValueError:
                result = {}
    if not result.get("ok"):
        detail = result.get("error") or f"the fetch worker exited with code {code}"
        log.warning("fetch of %s failed: %s", job.repo_id, detail)
        raise Invalid(f"Could not download {job.repo_id}: {detail}")
    return result
