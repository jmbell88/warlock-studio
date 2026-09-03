"""Health, the prompt preview and the trellis log tail."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import asdict
from typing import Any

from .. import doctor, guidance
from .core import WarlockService
from .errors import NotFound

log = logging.getLogger(__name__)

# Health used to run the full doctor suite -- a socket bind, a disk stat and a
# dozen path probes -- on every call, and the UI calls it continuously. None of
# those answers can change second to second, so they are cached for a few
# seconds, keyed on trellis_running because the port check's answer depends on
# it.
HEALTH_TTL = 5.0

# Tail size for the shared trellis log. errors.friendly() tells the user to
# look at it by name; this is how they can. Bounded because the server logs
# every stage of every run into one file that grows without limit.
TRELLIS_LOG_TAIL = 64 * 1024

_health_lock = threading.Lock()


def cached_checks(
    svc: WarlockService, trellis_running: bool, *, force: bool = False
) -> list[doctor.Check]:
    """``force`` skips the TTL, for the one caller that knows the disk changed.

    The TTL exists because none of these answers can change second to second --
    which stopped being true the moment a download could put weights on disk
    while the app runs. A fetch that finished inside the window would otherwise
    leave the pane reporting the model missing for another five seconds and
    the toast saying it had arrived.
    """
    # The cache hangs off the service, not this module: it describes one
    # process's config and store, and a module-level dict would outlive them
    # and answer for the wrong one.
    cache = svc.health_cache
    now = time.monotonic()
    with _health_lock:
        if not force and (
            cache.get("running") == trellis_running
            and now - cache.get("at", 0.0) < HEALTH_TTL
        ):
            return cache["checks"]
        static = None if force else cache.get("static")
    # The static half -- every path probe, the torch import, the bpy answer --
    # is computed once and reused; a poll re-runs only the four volatile rows
    # (port, disk, VRAM, job object). ``force`` recomputes everything, because
    # its one caller knows the disk just changed (a finished download).
    if static is None:
        static = doctor.static_checks(svc.config)
    checks = doctor.run_checks(svc.config, trellis_running=trellis_running, static=static)
    with _health_lock:
        cache.update(at=now, running=trellis_running, checks=checks, static=static)
    return checks


def current_checks(svc: WarlockService, *, force: bool = False) -> list[doctor.Check]:
    """The doctor's rows as of now, through the TTL cache.

    What the header health dot polls from a task thread: the trellis flag is
    sampled at call time (two attribute reads, safe from any thread), and
    ``cached_checks`` keeps the re-probe down to once per ``HEALTH_TTL``.
    """
    worker = svc.worker
    running = bool(worker is not None and worker.trellis.running)
    return cached_checks(svc, running, force=force)


def health(svc: WarlockService) -> dict[str, Any]:
    """Everything about the process's state in one dict.

    **The app does not call this**, and that is not an oversight left over from
    the HTTP layer: each half of it has a cheaper reader now. The header dot
    polls ``current_checks``, a dead worker is read straight off
    ``runtime.fatal``, and ``export_dir`` is handed to panes on ``Ctx``, so
    going through here would rebuild all four to show one. It survives as the
    single "what is the state of everything" answer -- what a diagnostics dump
    or a future headless probe wants, and what the API tests assert against.
    """
    worker = svc.worker
    running = bool(worker is not None and worker.trellis.running)
    checks = cached_checks(svc, running)
    return {
        "ok": bool(worker is not None and worker.alive and worker.fatal is None),
        "worker_alive": bool(worker is not None and worker.alive),
        "fatal": str(worker.fatal) if worker is not None and worker.fatal else None,
        "trellis_running": running,
        # Off unless WARLOCK_EXPORT_DIR is set. The library pane reads the same
        # answer from ``ctx.export_dir``, which the runtime fills at startup.
        "export_dir": str(svc.config.export_dir) if svc.config.export_dir else None,
        "checks": [asdict(c) for c in checks],
    }


def guidance_catalog(svc: WarlockService) -> dict[str, Any]:
    """Taxonomy for the design-guidance selects, so the UI has one source.

    Takes the service for the matte gate alone: ``catalog()["defaults"]`` is
    what the generate form initialises from, so it has to be the value a submit
    would pick on *this* host, not the preference in the constant.

    The ``sweeps`` sub-dict is an *addition* rather than a change: every
    existing consumer reads a named key off this dict and none enumerates it,
    so a new key is invisible to all of them. What it holds is the handful of
    sweep axis params the guidance catalog has nothing to say about -- they are
    ``create_job`` kwargs rather than taxonomy fields, so their legal values
    live in ``validation`` and ``pipelines.optimize`` instead. Gathered here
    rather than read directly by the Review pane for the reason the taxonomy is:
    a pane that imported ``pipelines.optimize`` for a list of profile names
    would be a pane reaching past the service layer, and the numbers would then
    have two readers to keep in agreement.
    """
    from ..pipelines import optimize
    from . import validation

    return guidance.catalog(
        bg_default=guidance.default_bg_removal(svc.config.trellis_models_dir)
    ) | {
        "sweeps": {
            "resolution": sorted(validation.ALLOWED_RESOLUTIONS),
            "profile": list(optimize.PROFILES),
            "trellis_band_range": [
                validation.MIN_TRELLIS_BAND,
                validation.MAX_TRELLIS_BAND,
            ],
            "trellis_tex_res_range": [
                validation.MIN_TRELLIS_TEX_RES,
                validation.MAX_TRELLIS_TEX_RES,
            ],
            "trellis_max_tokens_range": [1, validation.MAX_TRELLIS_MAX_TOKENS],
            "custom_triangles_range": [optimize.CUSTOM_MIN, optimize.CUSTOM_MAX],
        }
    }


def trellis_log(svc: WarlockService) -> dict[str, Any]:
    """The tail of the shared trellis log, as text.

    Not a file handle: the file is append-only and unbounded, and the point is
    the last few pages of it -- which is what a user chasing "The 3D engine
    stopped unexpectedly" actually needs.
    """
    path = svc.config.data_dir / "trellis.log"
    if not path.exists():
        raise NotFound("no trellis log yet")
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - TRELLIS_LOG_TAIL))
        text = fh.read().decode("utf-8", "replace")
    return {"path": str(path), "text": text}
