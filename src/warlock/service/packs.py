"""Installing an optional dependency pack, on the user's say-so, out of process.

The second narrow exception to the offline invariant, and the same shape as the
first: nothing here downloads and nothing here installs. It reads the manifest
the build produced (``warlock.packs``, pure), refuses (disk, conflicts, a
missing manifest), spawns ``python -m warlock.pipelines.pack_worker``, and
reads that child's progress lines. Reachable from a button and from nothing on
the job path.

Blocking by contract, like every other multi-second call in the service layer:
the pane dispatches it through ``TaskRunner``.

**Why this refuses more than ``downloads`` does.** A model download writes into
a directory the app only ever reads weights from; a pack install writes into
the ``site-packages`` the app is *running out of*. So two whole-plan refusals
run before the spawn, in ``fetch.disk_refusal``'s all-or-nothing style:

* a plan that would **replace** a distribution the runtime already has
  (``packs.conflicts``). Pack and base come out of one lock, so every wheel is
  either absent or already at the pack's own version; anything else means the
  two were built from different locks, and installing it would re-version a
  package the running application has already imported.
* a plan that does not fit on the two volumes it writes to, which are
  routinely two drives -- the wheel cache under the user's Warlock home and
  the application runtime.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from .. import packs as packs_mod
from .. import winjob
from ..config import PROJECT_ROOT
from .core import WarlockService
from .errors import Invalid, NotFound

log = logging.getLogger(__name__)

# Wall-clock ceiling for one pack: the download, which can be gigabytes over a
# slow line, plus pip unpacking them. ``fetch``'s four hours for the same
# reason -- generous because it is genuinely long, bounded because a child
# parked on a stalled socket holds a task-pool worker forever.
PACK_TIMEOUT = 4 * 60 * 60.0

_STDERR_KEEP_LINES = 40

Progress = Callable[[float, str], None]


def worker_argv() -> list[str]:
    """How the child is started. A function rather than an inline literal so a
    test can put a stub in its place and exercise this half -- the spawn, the
    stdin hand-over, the progress lines, the result file -- without any test in
    this project downloading or installing anything."""
    return [sys.executable, "-m", "warlock.pipelines.pack_worker"]


def manifest_path() -> Path:
    """Where the build left ``packs.json``.

    Beside ``pyproject.toml`` in the checkout-shaped runtime, which is the same
    root every vendored native binary resolves against (DST-01). Derived rather
    than configurable: the manifest describes *this build's* packs, so a
    manifest from somewhere else is not a setting, it is a mismatch.
    """
    return PROJECT_ROOT / packs_mod.MANIFEST_NAME


def cache_dir(svc: WarlockService) -> Path:
    """Where collected wheels live: under the user's Warlock home, with their
    work, because they survive a reinstall and are worth keeping -- a pack
    reinstalled after an app upgrade re-downloads nothing whose digest still
    matches."""
    return svc.config.home / "packs"


def load() -> packs_mod.Manifest:
    """The manifest, or a refusal naming what is absent.

    A build with no manifest is a *supported* state -- a source checkout that
    has never run ``scripts/make_packs.py`` is exactly that -- so this is a
    plain refusal with the remedy in it rather than an error the pane has to
    translate.
    """
    path = manifest_path()
    if not path.is_file():
        raise NotFound(
            "This build carries no pack manifest, so packs cannot be "
            "installed. In a source checkout the extras are installed with "
            "uv instead; see the pack a mode needs in Settings."
        )
    try:
        return packs_mod.load_manifest(path)
    except packs_mod.ManifestError as exc:
        raise Invalid(f"The pack manifest is unusable: {exc}") from exc


def installed_versions() -> dict[str, str]:
    """Every distribution in the running runtime, canonical name to version.

    Read here rather than in ``warlock.packs`` because it is a fact about a
    process, not about a plan: the pure half takes it as an argument, which is
    what lets every refusal below be tested against a runtime that does not
    exist.
    """
    found: dict[str, str] = {}
    for dist in distributions():
        try:
            name = dist.metadata["Name"]
            version = dist.version
        except Exception:  # noqa: BLE001 -- one broken dist-info is not a failure
            continue
        if name and version:
            found[packs_mod.canonical_name(name)] = version
    return found


def rows(svc: WarlockService) -> list[dict[str, Any]]:
    """Every pack with a real presence flag and what installing it would cost.

    A flag rather than the word "missing" inside a label, for the reason
    ``downloads.rows`` carries one: the pane used to test a label for a
    substring and silently mislabelled every row the substring never reached.

    Works without a manifest. A checkout that has never generated one still
    wants Settings to say which packs are present and what each unlocks -- the
    part that is unavailable is *installing*, and that is what ``manifest``
    being false says.
    """
    try:
        found = load()
    except (NotFound, Invalid):
        found = None
    have = installed_versions()
    out: list[dict[str, Any]] = []
    for pack in packs_mod.PACKS:
        plan = packs_mod.plan(found, [pack.key]) if found is not None else []
        pending = packs_mod.to_install(plan, have)
        out.append(
            {
                "key": pack.key,
                "label": pack.label,
                "summary": pack.summary,
                "modes": list(pack.modes),
                "present": packs_mod.installed(pack),
                "missing": packs_mod.missing(pack),
                "manifest": found is not None,
                "wheels": len(pending),
                "download_gib": packs_mod.gib(packs_mod.total_bytes(pending)),
                "installed_gib": packs_mod.gib(packs_mod.installed_bytes(plan)),
                "install_hint": pack.install_hint,
            }
        )
    return out


def plan_for(keys: Sequence[str]) -> list[packs_mod.Wheel]:
    """Every wheel these packs need, each once. Raises on an unknown key."""
    return packs_mod.plan(load(), keys)


def refusal(svc: WarlockService, keys: Sequence[str]) -> str | None:
    """Why this install must not start, or None. Refusing is the point.

    Conflicts first and disk second, deliberately: a plan that would re-version
    the runtime is wrong however much room there is for it, and reporting the
    smaller problem of the two would send the user to free up space for an
    install that must not run at all.
    """
    plan = plan_for(keys)
    have = installed_versions()
    clashing = packs_mod.conflicts(plan, have)
    if clashing:
        listed = "; ".join(clashing[:3]) + ("..." if len(clashing) > 3 else "")
        return (
            "This pack was built against a different version of the "
            f"application and cannot be installed into this one ({listed}). "
            "Reinstall Warlock to get packs that match it."
        )
    return packs_mod.disk_refusal(
        packs_mod.to_install(plan, have),
        cache_dir=cache_dir(svc),
        install_dir=Path(sys.prefix),
    )


def install(
    svc: WarlockService,
    keys: Sequence[str],
    *,
    on_progress: Progress | None = None,
    timeout: float = PACK_TIMEOUT,
    collect_only: bool = False,
) -> dict[str, Any]:
    """Collect and install these packs. Blocking; raises with the child's words.

    ``collect_only`` downloads without installing, which is what a build that
    wants the wheels beside an installer asks for, and what a test can exercise
    without writing into ``site-packages``.
    """
    chosen = packs_mod.chosen_packs(keys)
    plan = plan_for([pack.key for pack in chosen])
    have = installed_versions()
    said = refusal(svc, [pack.key for pack in chosen])
    if said:
        raise Invalid(said)
    pending = packs_mod.to_install(plan, have)
    if not pending and not collect_only:
        # Everything the pack carries is already at the pack's own version.
        # Not an error and not a spawn: an install with nothing to install is
        # what running it twice looks like, and it must be cheap.
        return {"ok": True, "collected": [], "installed": [], "already": True}
    spec: dict[str, Any] = {
        "pack_dir": str(cache_dir(svc)),
        "wheels": [
            {
                "filename": w.filename,
                "url": w.url,
                "bundled": w.bundled,
                "sha256": w.sha256,
                "size_bytes": w.size_bytes,
            }
            for w in pending
        ],
        "probe": sorted({name for pack in chosen for name in pack.probe}),
        "collect_only": collect_only,
    }
    return _run_worker(spec, on_progress=on_progress or (lambda _p, _l: None), timeout=timeout)


def _kill_and_reap(proc: subprocess.Popen[str]) -> None:
    with contextlib.suppress(Exception):
        proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=10)
    winjob.untrack(proc.pid)


def _run_worker(
    spec: dict[str, Any], *, on_progress: Progress, timeout: float
) -> dict[str, Any]:
    """One child, one install. ``downloads._run_worker``'s protocol exactly.

    The spec goes over stdin and the answer comes back through a file: stdout
    carries progress lines, and a stray print from pip must not be able to
    corrupt the result. stderr is drained from the start rather than read after
    exit, because a child whose stderr outgrows the OS pipe buffer blocks on
    its next write and never reaches the exit that read was waiting on -- and
    pip is chattier than most.
    """
    with tempfile.TemporaryDirectory(prefix="warlock-pack-") as scratch:
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
        # a socket, writes gigabytes and then writes into site-packages; an
        # orphan of it is the worst orphan in the tree.
        winjob.assign(proc.pid)
        winjob.track(proc.pid, "pack install")
        assert proc.stdin is not None and proc.stdout is not None
        assert proc.stderr is not None

        err_lines: deque[str] = deque(maxlen=_STDERR_KEEP_LINES)

        def _pump_err(stream: Any) -> None:
            try:
                for raw in stream:
                    err_lines.append(raw)
            except (OSError, ValueError):
                pass

        err_reader = threading.Thread(target=_pump_err, args=(proc.stderr,), daemon=True)
        err_reader.start()

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
                    # the child's stdout closes, and an install that finished
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
            raise Invalid("The pack install timed out.") from None
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
        detail = result.get("error") or f"the pack worker exited with code {code}"
        if not result.get("error"):
            tail = " ".join(x.strip() for x in list(err_lines)[-4:] if x.strip())
            if tail:
                detail += f": {tail}"
        log.warning("pack install failed: %s", detail)
        raise Invalid(f"Could not install the pack: {detail}")
    return result
