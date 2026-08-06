"""Health, the prompt preview and the trellis log tail."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict
from typing import Any

from .. import doctor, guidance, models
from .core import WarlockService
from .errors import Invalid, NotFound
from .validation import MAX_PROMPT

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


def cached_checks(svc: WarlockService, trellis_running: bool) -> list[doctor.Check]:
    # The cache hangs off the service, not this module: it describes one
    # process's config and store, and a module-level dict would outlive them
    # and answer for the wrong one.
    cache = svc.health_cache
    now = time.monotonic()
    with _health_lock:
        if (
            cache.get("running") == trellis_running
            and now - cache.get("at", 0.0) < HEALTH_TTL
        ):
            return cache["checks"]
    checks = doctor.run_checks(svc.config, trellis_running=trellis_running)
    with _health_lock:
        cache.update(at=now, running=trellis_running, checks=checks)
    return checks


def current_checks(svc: WarlockService) -> list[doctor.Check]:
    """The doctor's rows as of now, through the TTL cache.

    What the header health dot polls from a task thread: the trellis flag is
    sampled at call time (two attribute reads, safe from any thread), and
    ``cached_checks`` keeps the re-probe down to once per ``HEALTH_TTL``.
    """
    worker = svc.worker
    running = bool(worker is not None and worker.trellis.running)
    return cached_checks(svc, running)


def health(svc: WarlockService) -> dict[str, Any]:
    worker = svc.worker
    running = bool(worker is not None and worker.trellis.running)
    checks = cached_checks(svc, running)
    return {
        "ok": bool(worker is not None and worker.alive and worker.fatal is None),
        "worker_alive": bool(worker is not None and worker.alive),
        "fatal": str(worker.fatal) if worker is not None and worker.fatal else None,
        "trellis_running": running,
        # The UI reveals its "save to project" action off this alone -- the
        # feature is off unless WARLOCK_EXPORT_DIR is set.
        "export_dir": str(svc.config.export_dir) if svc.config.export_dir else None,
        "checks": [asdict(c) for c in checks],
    }


def guidance_catalog() -> dict[str, Any]:
    """Taxonomy for the design-guidance selects, so the UI has one source."""
    return guidance.catalog()


def prompt_preview(
    svc: WarlockService, raw: dict[str, Any], prompt: str = "", *, tile: bool = False
) -> dict[str, Any]:
    """The composed prompt and its token/chunk cost, before submission.

    tokens/chunks are best-effort -- null when transformers isn't installed or
    the base model's weights aren't downloaded, the same degrade-not-fail
    pattern doctor.py uses. Blocking: loading the tokenizers reads from disk.

    ``tile`` has to be threaded through rather than inferred from ``raw``: the
    output kind is not a guidance field, and without it the preview of a tile
    would show the single-centred-object framing the job will not use, which
    is worse than no preview at all.
    """
    from ..pipelines import prompt as prompt_pipeline
    from ..pipelines.text2image import Text2Image

    try:
        params = guidance.normalize(raw)
    except ValueError as exc:
        raise Invalid(str(exc)) from exc

    style = models.STYLE_LORAS.get(params.get("style_lora") or "")
    # Same gate the real run applies (text2image.generate only prepends a
    # trigger for an adapter that actually loaded): a LoRA missing on disk must
    # not show its trigger in the preview and then drop it at run time.
    trigger = (
        style.trigger
        if style and (svc.config.t2i_model_root / "loras" / style.filename).exists()
        else ""
    )
    if len(prompt) > MAX_PROMPT:
        raise Invalid(f"prompt must be at most {MAX_PROMPT} characters", field="prompt")
    positive = prompt_pipeline.build(prompt, params, trigger=trigger, tile=tile)

    tokens = chunks = None
    try:
        spec = models.BASE_MODELS[params["base_model"]]
        t2i = Text2Image(
            spec,
            svc.config.t2i_model_root,
            svc.config.t2i_turbo_dir
            if params["base_model"] == models.DEFAULT_BASE_MODEL
            else None,
        )
        tokenizers = prompt_pipeline.load_tokenizers(t2i.model_dir)
        tokens = prompt_pipeline.count(positive, tokenizers)
        chunks = len(prompt_pipeline.chunk(positive, tokenizers))
    except (ImportError, OSError):
        pass  # transformers not installed, or this base model's weights aren't downloaded

    return {
        "prompt": positive,
        "negative_prompt": params["negative_prompt"],
        "tokens": tokens,
        "chunks": chunks,
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
