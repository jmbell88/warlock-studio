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
from importlib import invalidate_caches
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

# ``percent, label, phase``. ``phase`` is the worker's own word for which
# side of the "can this still be cancelled" line it is on (H02) -- see
# ``pack_worker.PHASE_DOWNLOAD`` / ``PHASE_COMMIT`` -- rather than something
# a caller infers from the percent it happens to be looking at when it asks.
Progress = Callable[[float, str, str], None]


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


def bundled_dir() -> Path:
    """Where the installer left the wheels that cannot be downloaded at all.

    ``docopt``, ``mojimoji`` and ``unidic-lite`` publish no Windows wheel, so
    ``installer/build.ps1`` compiles them and stages them here, beside the
    manifest that pins their digests. Derived from ``PROJECT_ROOT`` for
    ``manifest_path``'s reason: they are part of *this build*, not a setting,
    and a bundled wheel from another build is a mismatch rather than a choice.

    Absent in a source checkout, which is correct and not an error -- a
    checkout installs the extras with uv and never reads this.
    """
    return PROJECT_ROOT / "packs"


def cache_dir(svc: WarlockService) -> Path:
    """Where collected wheels live: under the user's Warlock home, with their
    work, because they survive a reinstall and are worth keeping -- a pack
    reinstalled after an app upgrade re-downloads nothing whose digest still
    matches."""
    return svc.config.home / "packs"


def _selection_path(svc: WarlockService) -> Path:
    """Where the user's chosen packs are recorded, outside the runtime (M02).

    ``installer/warlock.iss`` deletes ``python\\Lib\\site-packages`` wholesale
    on an upgrade -- correctly, since a pack built against the old lock is not
    guaranteed to import under the new one -- which used to take the record of
    what had been installed with it: an upgrade left Create or Muse silently
    off with no saved selection to restore from and no sign anything had
    changed. This sits beside the wheel cache, under the user's Warlock home,
    which the installer never touches.
    """
    return cache_dir(svc) / "selected.json"


def selected_packs(svc: WarlockService) -> list[str]:
    """The pack keys this install has ever successfully installed. -> keys.

    Read by the pane to offer "Restore packs" after an upgrade removed them
    (M02): a key can be here and not ``installed_versions()`` at the same time
    only when something outside this process's control -- an upgrade's
    site-packages wipe -- took it away.
    """
    try:
        raw = json.loads(_selection_path(svc).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [str(key) for key in raw.get("packs") or [] if packs_mod.find(str(key))]


def _record_selected(svc: WarlockService, keys: Sequence[str]) -> None:
    """Add ``keys`` to the persisted selection. Best-effort: a write failure
    here must not fail an install that otherwise succeeded."""
    path = _selection_path(svc)
    have = set(selected_packs(svc))
    have.update(keys)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"packs": sorted(have)}), encoding="utf-8")
    except OSError:
        log.warning("could not record %r as installed packs", sorted(keys))


def packs_to_restore(svc: WarlockService) -> list[str]:
    """Previously-installed packs an upgrade has since removed. -> keys.

    Empty on an ordinary machine, including one that has never installed a
    pack at all -- this is only non-empty right after an upgrade wiped
    ``site-packages`` out from under a configured install (M02).
    """
    missing = []
    for key in selected_packs(svc):
        pack = packs_mod.find(key)
        if pack is not None and not packs_mod.installed(pack):
            missing.append(key)
    return missing


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


def unresolved(keys: Sequence[str]) -> list[str]:
    """Which of these packs' modules still do not import *in this process*.

    ``pack_worker.verify`` already proved them in the child, which is the
    honest place to prove an install finished -- but the app the user is
    looking at is this process, and this one had already cached "absent" for
    every name. ``install`` invalidates those caches, so this is normally
    empty; when it is not, the pane has something true to say ("restart
    Warlock") rather than leaving a mode grey after a successful install.
    """
    invalidate_caches()
    missing: list[str] = []
    for pack in packs_mod.chosen_packs(keys):
        for name in packs_mod.missing(pack):
            if name not in missing:
                missing.append(name)
    return missing


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
    probe = sorted({name for pack in chosen for name in pack.probe})
    if not pending and not collect_only:
        # Everything the pack carries is already at the pack's own version --
        # by distribution metadata alone. That is not proof it is usable
        # (M01, reproduced: matching metadata alone yields zero repair
        # candidates for a damaged install), so this path is probed exactly as
        # a fresh install is, in the same disposable child, rather than waved
        # through because nothing looked pending.
        _run_worker(
            {
                "pack_dir": str(cache_dir(svc)),
                "bundled_dir": str(bundled_dir()),
                "wheels": [],
                "probe": probe,
                "probe_only": True,
            },
            on_progress=on_progress or (lambda _p, _l, _ph: None),
            timeout=timeout,
        )
        invalidate_caches()
        return {"ok": True, "collected": [], "installed": [], "already": True}
    spec: dict[str, Any] = {
        "pack_dir": str(cache_dir(svc)),
        # The read-only half of the pair: the cache is where wheels are
        # downloaded to, this is where the three that cannot be downloaded
        # already are. Two directories rather than one because the cache lives
        # with the user's work and survives a reinstall, and a bundled wheel
        # belongs to the build -- an upgrade replaces it and must not have to
        # find it under a home directory it does not own.
        "bundled_dir": str(bundled_dir()),
        "wheels": _wheel_payload(pending),
        "probe": probe,
        "collect_only": collect_only,
    }
    result = _run_worker(
        spec, on_progress=on_progress or (lambda _p, _l, _ph: None), timeout=timeout
    )
    # This process has already asked ``find_spec`` about every module in the
    # pack and been told, truthfully at the time, that it was absent -- and the
    # answer is cached in ``sys.path_importer_cache`` against a site-packages
    # directory whose contents just changed underneath it. Without this the
    # pane would redraw the row it just installed as still missing, and the
    # only remedy on offer would be a restart the install does not need.
    invalidate_caches()
    if not collect_only:
        # Outside the runtime, so it survives the upgrade that would otherwise
        # erase every trace that this pack was ever chosen (M02).
        _record_selected(svc, [pack.key for pack in chosen])
    return result


def repair(
    svc: WarlockService,
    keys: Sequence[str],
    *,
    on_progress: Progress | None = None,
    timeout: float = PACK_TIMEOUT,
) -> dict[str, Any]:
    """Reinstall these packs' pinned wheels even though nothing looks pending.

    ``install`` treats a distribution whose metadata matches the pin as done;
    that is the fast path for the common case and exactly wrong for a
    distribution the probe has just shown is not actually importable (M01) --
    a prior pip run killed mid-unpack, a Windows update that broke a vendored
    DLL, leave a ``dist-info`` that looks finished. Repair skips ``to_install``
    and hands the worker the pack's whole wheel list with
    ``--force-reinstall``, so pip overwrites what is there instead of skipping
    it a second time for the same reason it was skipped the first.
    """
    chosen = packs_mod.chosen_packs(keys)
    plan = plan_for([pack.key for pack in chosen])
    said = refusal(svc, [pack.key for pack in chosen])
    if said:
        raise Invalid(said)
    spec: dict[str, Any] = {
        "pack_dir": str(cache_dir(svc)),
        "bundled_dir": str(bundled_dir()),
        "wheels": _wheel_payload(plan),
        "probe": sorted({name for pack in chosen for name in pack.probe}),
        "collect_only": False,
        "force_reinstall": True,
    }
    result = _run_worker(
        spec, on_progress=on_progress or (lambda _p, _l, _ph: None), timeout=timeout
    )
    invalidate_caches()
    _record_selected(svc, [pack.key for pack in chosen])
    return result


def _wheel_payload(wheels: Sequence[packs_mod.Wheel]) -> list[dict[str, Any]]:
    """The worker's own shape for a wheel list -- shared by ``install`` and
    ``repair`` so the two cannot describe the same wheel two different ways."""
    return [
        {
            "filename": w.filename,
            "url": w.url,
            "bundled": w.bundled,
            "sha256": w.sha256,
            "size_bytes": w.size_bytes,
        }
        for w in wheels
    ]


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
                on_progress(
                    percent,
                    str(payload.get("label") or ""),
                    str(payload.get("phase") or ""),
                )
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
