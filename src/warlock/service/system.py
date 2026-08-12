"""Health, the prompt preview and the trellis log tail."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import asdict
from typing import Any

from .. import doctor, fetch, guidance, models
from .core import WarlockService
from .errors import Invalid, NotFound, invalid_from
from .validation import MAX_PROMPT

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
    """
    return guidance.catalog(
        bg_default=guidance.default_bg_removal(svc.config.trellis_models_dir)
    )


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
        params = guidance.normalize(
            raw, bg_default=guidance.default_bg_removal(svc.config.trellis_models_dir)
        )
    except ValueError as exc:
        raise invalid_from(exc, "Those generation settings are not usable") from exc

    style = models.STYLE_LORAS.get(params.get("style_lora") or "")
    # Same gate the real run applies (text2image.generate only prepends a
    # trigger for an adapter that actually loaded): a LoRA missing on disk must
    # not show its trigger in the preview and then drop it at run time.
    #
    # Presence is the only question left here. Fitness needs no test: normalize
    # ran above in this same function and refuses a cross-family pair, so a
    # style that reaches this line is one the chosen base can take.
    # Through ``fetch.present`` rather than the path expression it wraps. Every
    # presence probe in the codebase lives in ``fetch``, and this was the one
    # that had reached for ``loras / filename`` itself -- so a change to where
    # or how a LoRA is stored would have moved every check but this one, and the
    # preview would have gone on promising a trigger the run then dropped.
    trigger = style.trigger if style and fetch.present(svc.config, "lora", style) else ""
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
            if params["base_model"] == models.T2I_DIR_MODEL
            else None,
        )
        tokenizers = prompt_pipeline.load_tokenizers(t2i.model_dir, spec.family)
        tokens = prompt_pipeline.count(positive, tokenizers)
        chunks = len(prompt_pipeline.chunk(positive, tokenizers))
    except Exception:
        # Every failure, not the two that were anticipated. This is a *preview*:
        # the token count is a nicety beside a prompt box that refreshes as the
        # user types, and the only correct response to not having it is to leave
        # it out. The narrow ``(ImportError, OSError)`` covered "not installed"
        # and "not downloaded" but not a corrupt tokenizer directory, which
        # raises ValueError or JSONDecodeError out of transformers -- turning a
        # live preview into an error toast on every keystroke (SVC-07).
        log.debug("prompt preview could not count tokens", exc_info=True)

    return {
        "prompt": positive,
        "negative_prompt": params["negative_prompt"],
        "tokens": tokens,
        "chunks": chunks,
        # What the taxonomy is contributing that argues with itself, or with
        # the brief (P124). Computed here rather than in the pane because it is
        # a fact about the *normalized* params -- the same ones the job will
        # compose from -- and because ``guidance`` is where the fragments that
        # do the arguing are written.
        "conflicts": guidance.colour_conflicts(params, prompt),
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
