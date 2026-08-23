"""Single-worker GPU job queue.

One job runs at a time. By default an SDXL-class image pipe (~7 GB; the
default checkpoint is ``sdxl_cfg``, SDXL 1.0 at full CFG, with Turbo as the
fast option) and the trellis server (~16 GB) coexist in VRAM — neither is
stopped for the other. **The image pipe is unloaded when the stage that loaded
it ends**, though (see ``_release_t2i`` for why the host, not the card, decides
that); trellis is a subprocess with a far larger startup cost and is evicted on
its own idle timeout. With Config.vram_exclusive set
(WARLOCK_VRAM_EXCLUSIVE=1, for small cards or a resident Flux), text jobs
instead use the sequential handoff: the trellis server is stopped before the
image model loads, and the image model is unloaded before trellis restarts.

Cancellation has no HTTP counterpart on trellis-server.exe (it exposes exactly
/generate and /health) and aborting the client request does not stop the GPU.
The only mechanism that actually frees VRAM mid-run is killing the subprocess
(TrellisServer.stop()), which this module already does for the VRAM handoff.
A cancel during the text2image stage instead sets a threading.Event that the
diffusers step callback checks every step (see pipelines/text2image.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import errors, fetch, leases, memlog, models, rigging, vectors, vram, winjob
from ._q_generate import GenerateOps
from ._q_jobs import JobOps
from ._q_mesh import MeshPostOps
from ._q_rig import RigOps
from ._q_sprite import SpriteOps
from ._q_tilesheet import TileSheetOps
from ._q_troupe import TroupeOps
from .config import Config
from .db import JobStore
from .pipelines import pose2d, reference
from .pipelines.trellis import TrellisServer, TrellisStopFailed
from .progress import ProgressBus, TrellisProgressParser

log = logging.getLogger(__name__)

POLL_INTERVAL = 1.0
# The poll once a full interval has passed with nothing waking the worker
# (C38): the backstop against a missed wake() stays, it just stops paying an
# executor hop plus a sqlite query every second of an idle session. Any
# enqueue calls wake(), which is observed immediately whichever timeout is in
# force, so the only thing this can delay is a *missed* wake -- by a few
# seconds instead of one.
IDLE_POLL_INTERVAL = 5.0
SHUTDOWN_TIMEOUT = 20.0

# Labels for the text-to-image phases, which have no trace of their own.
T2I_PHASES = {
    "load": ("t2i_load", "Loading image model"),
    # Attaching a ControlNet/IP-Adapter is a few seconds of loading, so it maps
    # onto the existing load phase rather than adding one -- the progress model
    # and its ETA need no change.
    "condition": ("t2i_load", "Loading conditioning"),
    "sample": ("t2i_sample", "Drawing reference image"),
}

# Whether reference.prepare recentres and rescales the subject, or merely
# measures it. False on purpose: upstream TRELLIS crops and pads in its own
# preprocessing and the vendored exe very likely does too, in which case
# normalising host-side over-zooms. This phase ships the report, the rejection
# rules and reference.png as an audit artifact *before* it ships the
# transform. Flip the default only after a sweep over 3 occupancy values x 3
# subjects scored with meshaudit.hole_fraction -- the same way
# Config.trellis_band was settled.
DEFAULT_REFERENCE_PREP = False

# The deformation QA sheet's grid. Four views rather than a sprite sheet's
# eight, and one size up from the sprite default: this is a picture a person
# looks at once to decide whether the weights hold, so a cell has to be big
# enough to read a collapsed elbow in, and the back three-quarter views say
# nothing a front and a side do not. Four battery poses at these numbers is 16
# EEVEE frames, which is seconds -- the rig itself is minutes.
DEFORM_QA_FRAME_SIZE = 256
DEFORM_QA_YAWS = 4


def _stage_link(source: Path, dest: Path) -> None:
    """Make ``dest`` hold ``source``'s bytes -- by hard link when possible.

    The remesh-retry staging copies two whole GLBs per kept attempt (C37); a
    hard link is free and sound here because **every** writer of source.glb
    and model.glb replaces the directory entry rather than rewriting the
    inode: trellis' ``_atomic_write``, ``optimize.staged_copy``, gltfpack's
    own ``tmp.replace(dest)`` and ``postprocess._staged`` all stage-and-rename,
    so a linked keep can never be scribbled on by the next attempt. The link
    itself goes through a temp name and ``os.replace`` so ``dest`` is never
    half-there, and a filesystem that refuses links falls back to the copy.
    """
    tmp = dest.with_name(dest.name + ".lnk.tmp")
    with contextlib.suppress(OSError):
        tmp.unlink()
    try:
        os.link(source, tmp)
        os.replace(tmp, dest)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink()
        # Staged, not written in place. ``copyfile`` truncates the destination
        # first, so the fallback -- which exists precisely for the filesystems
        # where ``os.link`` fails wholesale, an exFAT or network
        # ``WARLOCK_DATA_DIR`` -- was scribbling on a served ``model.glb`` for
        # the length of a copy, and a crash mid-copy left a torn file on a job
        # about to be marked ``done``. The docstring above promised
        # ``os.replace`` semantics that only the link branch had (CON-04).
        copy_tmp = dest.with_name(dest.name + f".copy.{secrets.token_hex(4)}.tmp")
        try:
            shutil.copyfile(source, copy_tmp)
            os.replace(copy_tmp, dest)
        finally:
            with contextlib.suppress(OSError):
                copy_tmp.unlink()


def _publish_text(path: Path, text: str) -> None:
    """Write ``text`` onto ``path`` through a staging file, never in place.

    ``_deform_qa``'s idiom, factored out because four sidecars need it for one
    reason: each of them is a *completion marker* -- the file the service keys
    "this artifact is ready" on -- and on a re-run the previous run's marker
    already says ready while this run's ``write_text`` is truncating and
    refilling it. A reader in that window gets a torn document that nothing
    marks as suspect. Renaming keeps the marker's promise true at every
    instant, and a failure leaves the previous marker intact rather than
    half-overwritten.

    Synchronous: every caller is on the loop thread and dispatches it through
    ``asyncio.to_thread`` like every other blocking call here.
    """
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def _fresh_seed() -> int:
    """A new 31-bit seed for a retry.

    Local rather than imported from service.validation: queue.py imports only
    top-level modules by design, and this is the same three-line contract --
    31-bit so it round-trips through an sqlite INTEGER unchanged.
    """
    return secrets.randbelow(2**31)


@dataclass
class _Cancel:
    job_id: str
    event: threading.Event = field(default_factory=threading.Event)
    #: Set by a stage the moment it *publishes* onto a served name. After that
    #: the artifact exists and a later cancel cannot un-publish it -- the
    #: cleanup deliberately never touches served files -- so recording the row
    #: as "cancelled" would leave a real rig on disk under a row saying it
    #: never happened, and every follow-up gated on ``status == "done"`` would
    #: silently never fire. A committed stage finishes normally; what a cancel
    #: still buys is skipping the work that comes *after* the publish.
    committed: bool = False

    def commit(self) -> None:
        self.committed = True

    @property
    def stopping(self) -> bool:
        """Whether a stage should stop early. False once committed is not a
        cancel being ignored: the tail after a publish is bookkeeping and QA,
        both of which are cheap to finish and expensive to have half-done."""
        return self.event.is_set()


def vram_gib() -> tuple[float, float] | None:
    """-> (allocated, reserved) GiB held by *this process's* torch, or None.

    Looked up through sys.modules rather than imported. Two reasons, both
    load-bearing:

    * Importing torch takes seconds, and this is called from the event loop.
      An image job on a machine with torch installed but not yet loaded would
      stall the whole app just to log a number.
    * The number would be zero anyway. memory_allocated() only sees this
      process's allocations, so it is meaningful exactly when the SDXL
      pipeline is loaded -- which is precisely when torch is already imported.
      trellis-server's ~16 GB lives in a separate process and never appears
      here regardless.
    """
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    gib = 1024**3
    return (torch.cuda.memory_allocated() / gib, torch.cuda.memory_reserved() / gib)


def _resident_t2i_gib(base_key: str | None) -> float:
    """What the resident pipe is worth when torch cannot be asked, in GiB.

    The dispatch credit's fallback used a flat SDXL_GIB, which is wrong the
    moment the resident base is an OFFLOAD entry -- the 10 GiB klein pipe
    credited at 7 would refuse a job the card actually holds. The registry
    already declares each base's footprint, so the fallback reads it, keeping
    SDXL_GIB only for a key the registry no longer carries -- the tolerance
    _generate already applies to a stored base_model.
    """
    spec = models.BASE_MODELS.get(base_key or "")
    return spec.vram_gib if spec is not None else vram.SDXL_GIB


def _log_mem(when: str) -> None:
    """Log VRAM *and* host memory at a stage boundary.

    The VRAM invariant (stop-before-load, unload-before-next-start) has one
    failure mode: an OOM that only reproduces under load. Nothing used to
    record whether the memory actually came back, so a regression was
    invisible until a user hit it. These lines are the record.

    The host half was added after the 2026-08-03 crash, which was commit
    exhaustion with the GPU nearly empty -- a failure the VRAM line alone
    could not have shown. Paired at the same boundaries, the two answer the
    question that matters: does memory come back after a job, or only grow.
    """
    mem = vram_gib()
    if mem is not None:
        log.info("vram %s: %.2f GiB allocated, %.2f GiB reserved", when, *mem)
    # ``winjob.measured_pids()`` rather than nothing: the matting worker is a
    # child, and at 6.5 GiB of private commit it was the largest single charge
    # this app made that no ``host ...`` line in this log had ever shown.
    #
    # And rather than ``tracked()``, which is the pids ``Popen`` returned: under
    # a uv venv those are trampolines, so the line read 0.8 MB of shim and
    # printed ``children 0.0 GiB`` beside a worker holding 6.3 GiB -- on the
    # very tick that then refused the next job for want of commit
    # (docs/measurements/2026-08-22-trampoline-child-pids.md).
    host = memlog.summary(children=winjob.measured_pids())
    if host is not None:
        log.info("host %s: %s", when, host)
    pressure = commit_fraction()
    if pressure is not None and pressure >= COMMIT_CEILING:
        log.critical(
            "host commit at %.0f%% %s -- at or past the ceiling the "
            "2026-08-03 crash hit; further jobs will be refused",
            pressure * 100,
            when,
        )


def _host_peak_gib(spec: Any) -> float:
    """What loading this checkpoint charges host commit, in GiB.

    The spec's declared ``host_peak_gib`` when it has one -- which is the
    offloaded entries, where the host figure is the large one and ``vram_gib``
    is small precisely because of it. Otherwise ``vram_gib``: a resident load
    reads the checkpoint into host memory and hands it to the device, so the
    host charge peaks at roughly the same size even though it does not stay.
    """
    declared = float(getattr(spec, "host_peak_gib", 0.0) or 0.0)
    return declared if declared > 0 else float(getattr(spec, "vram_gib", 0.0) or 0.0)


COMMIT_MARGIN_GIB = 2.0
"""Commit left over after the load this check is standing in front of.

Not a safety factor on the model's own figure -- that is what ``host_peak_gib``
being a peak is for -- but room for everything else the process does while the
weights are being read: the reader's buffers, the allocator's slack, and the
rest of the app carrying on. Crossing the commit limit is not an exception a
job can fail on; it is Windows ending the process.
"""

COMMIT_CEILING = 0.90
"""System commit fraction past which the worker stops taking jobs.

Windows kills the process (Resource-Exhaustion event 2004) rather than raising
something a job can fail on, so the only useful place to stand is *before* the
next allocation. It needs no watchdog thread: the stage boundaries _log_mem
already runs at are exactly the moments the number can be acted on.
"""


_ALWAYS_CONDITIONED = object()
"""Stand-in ``cond`` for a caller whose every pass is conditioned.

``_pixel_sheet`` and ``_retexture`` build a fresh ``Conditioning`` per band and
per view, inside the loop, so there is no single object to hand ``_needs_handoff``
at the point the decision is made -- and passing ``None`` would silently drop the
one term that is *always* true of them. A named sentinel says that out loud;
the predicate only ever asks whether it is None.
"""


def _needs_handoff(
    spec: models.BaseModel | None,
    cond: Any,
    *,
    exclusive: bool,
    trellis_running: bool,
) -> bool:
    """Must trellis be stopped before this image model loads?

    Three independent reasons, and the predicate has one owner because every
    t2i stage asks it through ``_acquire_t2i`` -- a second spelling is how
    they would come to disagree.

    * ``exclusive`` -- WARLOCK_VRAM_EXCLUSIVE, the original reason.
    * conditioning beside a *running* trellis -- a ControlNet plus the
      CLIP-ViT-H encoder is ~6 GiB over the plain budget, which does not fit
      beside a resident server on a 32 GiB card. Asked as "is trellis actually
      holding VRAM" rather than exempting the reference stage, because trellis
      stays warm for ``trellis_idle_timeout`` after the previous model job.
    * an ``OFFLOAD`` checkpoint -- unconditionally, whatever the flag says.
      ``enable_model_cpu_offload()`` keeps the whole ~16 GiB checkpoint in host
      RAM for the life of the pipe, so its VRAM figure is honest and irrelevant:
      what collides with trellis is *commit*, and under WDDM trellis' ~16 GiB
      device allocation is charged against the same limit. Measured on this
      machine as Python private commit climbing 24.4 -> 45.2 GiB across three
      FLUX.2 klein jobs, with system commit at 99%. "Coexist" was written when
      every image model was RESIDENT and says nothing about this shape.

    The accepted cost of the third term is a trellis restart per offloaded text
    job. ``vram.offloaded_base`` is the accounting half of the same rule.
    """
    if exclusive:
        return True
    if cond is not None and trellis_running:
        return True
    return spec is not None and spec.residency == models.OFFLOAD


# --- sprite synthesis helpers -----------------------------------------------
#
# Module-level and blocking, so ``_sprite_synthesis`` can push each of them
# through one ``asyncio.to_thread`` call: the matte is a model or a flood fill,
# the assembly is sixteen flood fills and a median cut, and neither belongs on
# the loop that drives dispatch and cancellation. They live here rather than in
# ``pipelines/spritesynth`` because each is a *sequence* of that module's pure
# steps plus this process's config -- which is the worker's business, not the
# pure module's.


def _sprite_source(src: Path, config: Config) -> tuple[Any, Any, str]:
    """The reference as a cutout, its measurement, and which matte produced it.

    The one place the heavy matting model is allowed to run on this path. The
    cutout feeds both halves of the job: the IP-Adapter image (so the adapter
    conditions on the character rather than on the background it happened to be
    drawn against) and the front-cell paste, which needs real alpha.
    """
    import numpy as np
    from PIL import Image

    from .pipelines import matting

    with Image.open(src) as opened:
        opened.load()
        flat = opened.convert("RGBA") if reference.has_alpha(opened) else opened.convert("RGB")
    found, matte_source = matting.mask(flat, config)
    rgb = np.asarray(flat.convert("RGB"))
    alpha = np.where(np.asarray(found), 255, 0).astype(np.uint8)
    cut = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    # Measured on the cutout, not on the file: ``front_fits`` is asking whether
    # the *subject* is clean and single, and the alpha channel is the answer we
    # just computed rather than a second, weaker guess at it.
    return (cut, reference.measure(cut), matte_source)


def _sprite_ip_image(cut: Any) -> Any:
    """The reference framed for the IP-Adapter: subject only, on neutral grey.

    Cropped to the subject so the adapter's sixteen patch tokens are spent on
    the character instead of on empty margin, and composited onto mid-grey for
    ``pixelsheet.BAND_BACKGROUND``'s reason -- an RGB conversion of a
    transparent background reads as black and puts a dark rim in the tokens.
    """
    from PIL import Image

    from .pipelines import pixelsheet, spritesynth

    side = spritesynth.ATLAS_PX
    box = cut.getbbox()
    subject = cut.crop(box) if box else cut
    scale = min(side / max(1, subject.width), side / max(1, subject.height)) * 0.9
    nw = max(1, round(subject.width * scale))
    nh = max(1, round(subject.height * scale))
    subject = subject.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (side, side), pixelsheet.BAND_BACKGROUND)
    canvas.paste(subject, ((side - nw) // 2, (side - nh) // 2), subject)
    return canvas


def _sprite_assemble(
    atlas: Any,
    geom: Any,
    logical: int,
    colors: int,
    source_rgba: Any,
    source_report: Any,
) -> tuple[Any, dict[str, Any]]:
    """One generated atlas turned into a publishable candidate.

    The order is the whole argument: warnings are measured on what the model
    actually drew (before anything is nudged), the baseline is shared before
    the front cell is replaced (so the paste has a line to stand on), and the
    front paste happens before the reduction and the palette (so the one cell
    that is definitely the right character is not also the one that does not
    match the sheet).
    """
    from .pipelines import pixelsheet, spritesynth

    matted, took = spritesynth.matte_cells(atlas, geom)
    warnings = spritesynth.structural_warnings(matted, geom, took)
    aligned = spritesynth.baseline_align(matted, geom)

    front_preserved = False
    front_note = ""
    if geom.kind == "turnaround":
        front = geom.cells[0]
        ok, front_note = spritesynth.front_fits(
            source_report, reference.measure(aligned.crop(front.box))
        )
        if ok:
            aligned = spritesynth.preserve_front(aligned, geom, source_rgba)
            front_preserved = True

    reduced = spritesynth.reduce_atlas(aligned, geom, logical)
    try:
        quantized, palette = pixelsheet.quantize_shared(reduced, colors)
    except ValueError as exc:
        # quantize_shared refuses an atlas with no opaque pixels at all, which
        # here means the model drew nothing anywhere -- worth a sentence naming
        # that rather than the sheet-shaped message it was written for.
        raise RuntimeError(
            "the generated sprite sheet came out empty in every cell"
        ) from exc
    return (
        quantized,
        {
            "palette": palette,
            "warnings": warnings,
            "front_preserved": front_preserved,
            "front_note": front_note,
        },
    )


def _require_commit_headroom(when: str, remedy: str, need_gib: float = 0.0) -> None:
    """Refuse to go on if host commit cannot take what comes next.

    Raises RuntimeError. The enforcing half of what ``_log_mem`` only records.
    It was one check, at dispatch, before ``_generate`` -- and nothing re-asked
    it across the whole stop-trellis / load-image-model / generate / unload /
    start-trellis sequence, which is precisely where the charge moves.

    Two questions, not one. The percentage ceiling is the standing "is this
    machine already in trouble" reading. ``need_gib`` is the other half and the
    one MDL-04 is about: FLUX.2 klein is ~16 GiB of *host* weights under CPU
    offload, and a machine sitting at 80-89% commit passes the ceiling while
    having far less than 16 GiB of commit left. It is then admitted and crosses
    the limit during checkpoint allocation -- on Windows, plausibly the OS
    terminating the process, which is the exact failure this check exists to
    prevent. A percentage is not a quantity, so the bytes the next operation
    needs have to be asked for by name.

    ``need_gib`` of 0 keeps the old behaviour exactly, for the call sites that
    are not about to allocate anything in particular.

    ``when`` is a clause appended to "committed" (with its own leading space,
    or empty), and ``remedy`` is the rest of the sentence, because the two call
    sites have genuinely different news: refusing at dispatch costs nothing,
    and refusing between the stages costs a reconstruction the user has already
    waited for the image half of.
    """
    pressure = commit_fraction()
    if pressure is not None and pressure >= COMMIT_CEILING:
        raise RuntimeError(
            f"host memory is {pressure * 100:.0f}% committed{when}, at or past the "
            f"{COMMIT_CEILING * 100:.0f}% ceiling. {remedy}"
        )
    if need_gib <= 0:
        return
    sysmem = memlog.system_memory()
    if sysmem is None:
        return
    free = sysmem.commit_limit - sysmem.commit_total
    want = need_gib + COMMIT_MARGIN_GIB
    if free < want:
        raise RuntimeError(
            f"loading this model needs about {need_gib:.1f} GiB of host memory "
            f"(plus a {COMMIT_MARGIN_GIB:.1f} GiB margin){when}, and only "
            f"{free:.1f} GiB of the commit limit is free. {remedy}"
        )


def _landmark_bones(
    config: Config, source_dir: Path, template_key: str
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    """Where the subject's joints are, read off the image the mesh was made
    from. -> ``(normalized bones, how they were found)``, or ``(None, None)``.

    The mesh *is* ``input.png`` reconstructed, so a landmark's position in that
    image is the joint's position on the mesh in X and Z -- which is the whole
    premise, and the reason the subject bbox comes from ``reference.measure``
    rather than from a person detector: the composition gate has already
    established that this image holds exactly one subject and has already
    measured its silhouette, so a second network to find a box that is known
    would be cost with no answer attached.

    Blocking. Every caller dispatches it through ``asyncio.to_thread``.

    ``(None, None)`` on anything at all -- an unreadable PNG, no weights, a
    detection that fails a sanity gate -- and the caller then rigs exactly as
    it did before this existed. Nothing here is allowed to raise: a better
    skeleton is an optimisation, and ``_audit_mesh``'s rule applies with more
    force here, because the artifact this would fail is one the user asked for
    and the improvement is one they did not.
    """
    try:
        image = source_dir / "input.png"
        report = reference.measure_file(image)
        if report.bbox is None:
            return (None, None)
        keypoints = pose2d.detect(image, report.bbox, config)
        if keypoints is None:
            return (None, None)
        bones = pose2d.refit(rigging.get_template(template_key), keypoints, report.bbox)
        if bones is None:
            return (None, None)
        # The confidences of the landmarks the fit actually rests on, not of
        # all seventeen: an unseen ear says nothing about whether the knees
        # were found, and averaging it in would flatter exactly the detections
        # worth doubting.
        scores = sorted(k.score for k in keypoints if k.name in pose2d.REQUIRED_KEYPOINTS)
        return (
            bones,
            {
                "method": "pose2d",
                "model": models.DEFAULT_POSE_MODEL,
                # Both, because they answer different questions: the mean says
                # how well the figure was read, the minimum says whether any
                # one joint is a guess -- and one bad joint is what a rig
                # actually breaks on.
                "confidence": round(sum(scores) / len(scores), 3),
                "confidence_min": round(scores[0], 3),
            },
        )
    except Exception:
        log.exception("could not read joint landmarks from %s; using the bbox fit", source_dir)
        return (None, None)


def _observe_finished(store: JobStore, job_id: str) -> bool:
    """Append the machine-evidence row for a model job that reached a terminal
    status. -> whether a row was written.

    Reads the row fresh rather than trusting an in-memory job dict: the audit
    and report were committed via ``set_params``/``merge_params`` and the
    remesh-restore path rebinds params, so the row is the record. A job with
    nothing measured writes nothing -- an empty metrics row would be a bucket
    diluting every mean it joins. ``import_mesh`` and the retarget re-audit
    deliberately have no call site here: the corpus is about what generation
    settings produce, and a hand-imported or rewritten mesh measures neither.

    "Nothing measured" is a smaller set than it was. A job refused at the
    composition gate has no mesh and so no audit and no report, but the
    refusal is itself a reading of the settings that drew the reference, and
    ``vectors.observation_metrics`` returns it -- so the *status* is not the
    filter here and never was. ``_process`` calls this for ``done`` and
    ``error`` alike and lets the metrics decide.
    """
    job = store.get(job_id)
    if job is None:
        return False
    if job.get("stage") != "model" or job.get("kind") not in ("text", "image"):
        return False
    params = job.get("params") or {}
    metrics = vectors.observation_metrics(params)
    if not metrics:
        return False
    seed = params.get("seed")
    store.add_observation(
        job_id,
        sweep_id=job.get("sweep_id"),
        sweep_unit=job.get("sweep_unit") or "",
        seed=seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
        prompt_hash=vectors.prompt_hash(job.get("prompt")),
        vector=vectors.config_vector(job),
        metrics=metrics,
    )
    return True


def _sheet_root_offsets(
    records: list[dict[str, Any]], rig_meta: dict[str, Any] | None
) -> tuple[dict[tuple[Any, int], list[float]], Any]:
    """(pose id, frame) -> world root offset for the records that carry one,
    plus the root bone they land on.

    A pose snapshotted from the global library can carry a root offset, and a
    sheet built from it must not silently disagree with the pose's own bake --
    one meaning per pose. Rows without one gain no keys, so every pre-existing
    cell dict is byte-identical to what it always was.

    Clip records *do* reach this now. ``sheetlib.interpolate`` used to refuse an
    endpoint carrying an offset and interpolates it instead, which is what a
    walk cycle's vertical bob is made of. Nothing here had to change to allow
    it: the map is keyed by ``(pose id, frame)`` and every frame of a clip
    shares an id but has its own frame number, so each frame already gets its
    own entry.

    A rig.json that cannot answer (pre-template, unreadable) costs the offset,
    never the sheet -- and says so, one warning per pose, the same sentence
    service.rig._pose_bake_spec logs for the same case. Matched by hand rather
    than shared, because queue.py may not import service: two spellings of one
    sentence, accepted, and this comment is the tie between them.
    """
    roots: dict[tuple[Any, int], list[float]] = {}
    offset_records = [
        r for r in records if any(float(v) for v in (r.get("root_translation") or ()))
    ]
    if not offset_records:
        return roots, None
    meta = rig_meta or {}
    bounds, root_bone = meta.get("bounds"), meta.get("root")
    if not (bounds and root_bone):
        for r in offset_records:
            log.warning(
                "pose %s carries a root offset but rig.json cannot scale it",
                r.get("id"),
            )
        return roots, None
    for r in offset_records:
        roots[(r.get("id"), r.get("frame", 0))] = rigging.root_offset_world(
            r["root_translation"], bounds
        )
    return roots, root_bone


def commit_fraction() -> float | None:
    """System commit charge as a fraction of the limit, or None off Windows."""
    sysmem = memlog.system_memory()
    return None if sysmem is None else sysmem.commit_fraction


class Worker(GenerateOps, RigOps, TroupeOps, SpriteOps, TileSheetOps, MeshPostOps, JobOps):
    def __init__(self, config: Config, store: JobStore) -> None:
        self.config = config
        self.store = store
        self.trellis = TrellisServer(
            config.trellis_server_exe,
            config.trellis_models_dir,
            config.trellis_port,
            log_path=config.data_dir / "trellis.log",
            webp=config.trellis_webp,
            tex_res=config.trellis_tex_res,
            band=config.trellis_band,
        )
        self._text2image = None  # lazy: torch/diffusers may not be installed
        # Which base model the resident pipe is, so _get_text2image can tell a
        # cache hit from a swap.
        self._t2i_key: str | None = None
        # The model store's generation when that pipe was built. A pipe freezes
        # its adapter set at load() time, so weights installed or removed since
        # make it stale even though the key still matches (MDL-05).
        self._t2i_generation: int = -1
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        # Set by wake() when something enqueues a job, so an idle dispatch loop
        # starts it immediately instead of waiting out the poll interval.
        self._wake = asyncio.Event()
        self.current_job_id: str | None = None
        self._cancel: _Cancel | None = None
        # The live Blender subprocess, if a rig job is running. Set from the
        # worker thread inside run_worker's on_start and read from the event
        # loop in request_cancel -- a plain attribute assignment either way,
        # and only ever non-None for the one job the queue is running.
        self._blender: Any = None
        # When the last job finished, and whether the host-side inference caches
        # have already been dropped since. Unlike trellis and the image pipe,
        # BiRefNet/pose/DINO carry no last_used of their own, so eviction is
        # driven off the last finished job and throttled to one pass per idle
        # timeout -- see ``_maybe_evict_caches`` for why this is a timestamp and
        # not the boolean latch it used to be (MDL-12).
        self._last_job_at: float = time.monotonic()
        self._caches_evicted_at: float = 0.0
        # How to take the lock guarding one derived artifact of one job, as
        # (job_id, name) -> a context manager. The worker holds no service and
        # may not import one, so ``studio.runtime`` injects
        # ``WarlockService.convert_lock`` here; unset, it is a null lock, which
        # is exactly the pre-existing behaviour of every write the worker makes
        # to a model.glb (and is why ``optimize_job`` refuses a job that is
        # queued or running rather than trying to interleave with it).
        #
        # Only ``_retexture`` uses it, and only to *delete* the exports that
        # describe the skin it just replaced -- the one place the worker touches
        # a finished job's derived artifacts, and so the one place an in-flight
        # conversion could otherwise rename a stale copy back into existence
        # after the unlink.
        self.artifact_lock: Any = lambda _job_id, _name: contextlib.nullcontext()
        self.fatal: BaseException | None = None
        self.progress = ProgressBus()
        self._parser = TrellisProgressParser(self._emit_progress)
        self.trellis.on_line = self._parser.feed

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="gpu-worker")
        self._task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            # Log with the traceback, store without it. A stored exception
            # keeps its __traceback__, which keeps every frame of the dying
            # call stack alive -- and those frames hold the pipeline and its
            # tensors. self.fatal is never cleared, so that retention lasts for
            # process life. Only the exception itself is ever read (main.py
            # renders str(fatal)); the traceback belongs in the log.
            #
            # The order is load-bearing and used to be inverted.
            # ``with_traceback`` mutates in place and returns *self*, so
            # stripping first left ``exc_info=exc`` with nothing to format --
            # the single log line for a dead GPU worker carried no stack at all.
            log.critical("gpu worker task died", exc_info=exc)
            self.fatal = exc.with_traceback(None)

    def wake(self) -> None:
        """Tell an idle dispatch loop there is work now.

        Called from the routes that insert a row, on the event loop the worker
        runs on -- so this is a plain Event.set(), not a threadsafe hop. The
        POLL_INTERVAL timeout in _run stays as the backstop: a caller that
        forgets to wake costs a second of latency, not a stuck queue.
        """
        self._wake.set()

    async def _wait_for_work(self, timeout: float = POLL_INTERVAL) -> bool:
        """Sleep until a job is enqueued, shutdown is asked for, or ``timeout``
        elapses -- whichever comes first. -> whether something woke us (rather
        than the timeout), which is what lets the idle loop back its DB poll
        off (C38)."""
        waiters = [
            asyncio.ensure_future(self._stop.wait()),
            asyncio.ensure_future(self._wake.wait()),
        ]
        try:
            await asyncio.wait(
                waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for waiter in waiters:
                waiter.cancel()
        woke = self._wake.is_set() or self._stop.is_set()
        # Cleared only after the wait, never before it: a wake() that landed
        # while next_queued() was in its thread must still be observed here
        # rather than swallowed into a full poll interval of sleep.
        self._wake.clear()
        return woke

    async def request_cancel(self, job_id: str) -> None:
        """No-op unless job_id is the job currently running."""
        if job_id != self.current_job_id or self._cancel is None:
            return
        self._cancel.event.set()
        snapshot = self.progress.snapshot()
        phase = snapshot["phase"] if snapshot else None
        if phase == "trellis":
            # The only real "abort" trellis-server.exe has: kill it. The
            # in-flight client.post then dies with a TransportError, which
            # _process below turns into a cancelled status because the
            # cancel event is already set.
            #
            # Suppressed: a stop that cannot confirm death must not turn a
            # cancel into an exception on the loop thread. stop() has already
            # logged it at critical and kept the handle, so the next
            # ensure_started reaps or refuses.
            with contextlib.suppress(TrellisStopFailed):
                await asyncio.to_thread(self.trellis.stop)
        elif phase in ("rig", "sheet", "views", "project"):
            # Same story as trellis: bpy is inside a C weighting solve (or an
            # EEVEE render) and checks nothing, so killing the subprocess is
            # the only abort.
            proc = self._blender
            if proc is not None and proc.poll() is None:
                with contextlib.suppress(OSError):
                    proc.kill()
        # t2i_sample: the diffusers step callback checks the event itself.
        # t2i_load: not interruptible; the event is checked once between
        # load() and sampling in Text2Image.generate().

    async def shutdown(self) -> None:
        self._stop.set()
        # Before the cancel and before the grace period, so a load already in
        # flight on a worker thread learns not to publish. Cancelling the task
        # below only cancels the coroutine awaiting it (MDL-02).
        pipe = self._text2image
        if pipe is not None:
            with contextlib.suppress(AttributeError):
                pipe.close()
        if self.current_job_id is not None:
            await self.request_cancel(self.current_job_id)
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=SHUTDOWN_TIMEOUT)
            except TimeoutError:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
        # Suppressed: a raise here strands the shutdown with the image pipe
        # still loaded and the runtime half torn down. The critical log inside
        # stop() is the record.
        with contextlib.suppress(TrellisStopFailed):
            await asyncio.to_thread(self.trellis.stop)
        # Shutdown used to stop trellis and leave SDXL loaded. Harmless when
        # the process exits immediately after -- but shutdown() is also reached
        # on paths that keep the interpreter alive, and the pipeline's several
        # GB of host-pinned staging memory has no other release point.
        #
        # Deliberately ``unload`` rather than ``_evict_t2i``: this is the end of
        # the worker's life, not a reclaim, and the pipe object is still the
        # record of what ran. Clearing the reference here breaks every caller
        # that inspects a shut-down worker.
        #
        # Behind the lease, and the ordering is the whole point (MDL-02).
        # Cancelling the task above cancels a *coroutine*; it does not stop the
        # thread that coroutine was awaiting, so after the grace period expires
        # a ``from_pretrained`` may still be running. Reading ``.loaded`` then
        # sees False, the unload is skipped, and the load publishes a fully
        # resident pipe *afterwards* -- up to ~16 GiB of host commit held for
        # the life of an interpreter that shutdown was supposed to leave clean.
        # Waiting for the lease is waiting for that thread, so by the time the
        # check below runs, the answer is final.
        await asyncio.to_thread(self._unload_under_lease)

    def _unload_under_lease(self) -> None:
        """Wait out any in-flight model operation, then drop the pipe. Blocking.

        ``maintain`` rather than ``use``: this must be the only model operation
        running, and its whole purpose is to observe a state nothing else can
        still be changing. A timeout here is not fatal -- the process is going
        away -- but it is worth a line, because it means a model thread outlived
        the shutdown that was meant to join it.
        """
        try:
            with leases.MODELS.maintain(timeout=SHUTDOWN_TIMEOUT):
                pipe = self._text2image
                if pipe is not None and pipe.loaded:
                    pipe.unload()
        except TimeoutError:
            log.warning(
                "a model operation was still running at shutdown; the image "
                "pipeline was left loaded rather than torn down under it"
            )

    async def _run(self) -> None:
        # Widened after an unwoken timeout, reset by anything happening (C38).
        wait = POLL_INTERVAL
        while not self._stop.is_set():
            try:
                job = await asyncio.to_thread(self.store.next_queued)
                if job is None:
                    await self._maybe_evict_idle()
                    woke = await self._wait_for_work(wait)
                    wait = POLL_INTERVAL if woke else IDLE_POLL_INTERVAL
                    continue
                wait = POLL_INTERVAL
                await self._process(job)
            except Exception:
                # A crash here used to kill the worker permanently and
                # silently -- next_queued or a DB hiccup would strand every
                # future job in 'queued' forever with no error surfaced.
                log.exception("worker loop iteration failed")
                # Bounded, or a *persistent* failure (disk full, corrupt DB
                # page) spins this loop flat out, writing a traceback per
                # pass to the disk that caused it.
                await asyncio.sleep(POLL_INTERVAL)
            if self._stop.is_set():
                break

    async def _maybe_evict_idle(self) -> None:
        # Every eviction goes through a thread. The queue being idle does not
        # make them cheap: stop() blocks for up to ~25 s if the server ignores
        # SIGTERM, unload() pays a gc.collect() plus empty_cache(), and
        # matting.unload() is now a subprocess kill. On the event loop any one
        # of them freezes the progress snapshot the frame loop reads, and every
        # other job with it.
        if (
            self.trellis.running
            and time.monotonic() - self.trellis.last_used > self.config.trellis_idle_timeout
        ):
            log.info("evicting idle trellis-server")
            # Suppressed: eviction is advisory and runs with no job in flight,
            # so a server that will not die is a logged fact rather than an
            # exception out of the worker loop. The next dispatch's
            # _check_resources still sees the VRAM it is holding.
            with contextlib.suppress(TrellisStopFailed):
                await asyncio.to_thread(self.trellis.stop)
        # In exclusive mode the per-job finally already unloaded it, so
        # loaded is never True here and this branch is inert.
        if (
            self._text2image is not None
            and self._text2image.loaded
            and time.monotonic() - self._text2image.last_used > self.config.trellis_idle_timeout
        ):
            log.info("evicting idle SDXL pipeline")
            await self._evict_t2i()
        await self._maybe_evict_caches()

    async def _evict_t2i(self, *, forget: bool = False) -> None:
        """Give back the resident image pipe's memory. A no-op when there is none.

        Its own method because three callers now want it for three reasons,
        and they differ in exactly one thing. Stale-pipe eviction
        (``_evict_stale_t2i``) forgets, because a different key or generation
        is about to replace the object. The idle sweep keeps the pipe object:
        it is the record of what ran, and reloading the same key is what it is
        for. ``forget`` additionally drops both references, which an *uninstall*
        needs -- the files this pipe was built from are about to stop existing,
        so a next ``_get_text2image`` that matched the key and handed the object
        back would hand back a pipe pointing at a deleted checkpoint.
        """
        pipe = self._text2image
        if forget:
            self._text2image = None
            self._t2i_key = None
        # ``.loaded`` is checked, not just ``is not None``: since every t2i
        # stage gives its checkpoint back when it ends (see ``_release_t2i``),
        # the pipe this finds is usually *already* unloaded and kept only for
        # its identity. ``Text2Image._unload`` returns immediately on
        # ``_pipe is None``, so the call would be harmless -- but it is still a
        # thread hop and the models lease for nothing, and it would make
        # "the switch unloads the previous pipe exactly once" false by one.
        if pipe is not None and getattr(pipe, "loaded", True):
            await asyncio.to_thread(pipe.unload)

    async def _evict_stale_t2i(self, base_key: str) -> bool:
        """Unload the resident image pipe if the next job cannot reuse it.

        Stale means the wrong base (``_t2i_key != base_key``) or the wrong
        store generation -- the weights on disk changed while the pipe was
        warm, so what it loaded is no longer what the store holds. The pipe's
        adapter set is fixed at load() time and never revisited, so a style
        LoRA installed since would be silently dropped at generate with a
        "not downloaded" warning that is not true -- and the job would finish
        looking successful, having recorded a style that never ran (MDL-05).
        Rebuilding is the honest answer and costs one reload of a checkpoint
        the user just changed the disk under.

        This runs *before* ``_require_commit_headroom`` in ``_acquire_t2i``,
        because the check would otherwise charge host commit the unload is
        about to give back: a live klein load was refused at 13.2 GiB free
        while ~12 GiB of the shortfall was the resident dreamshaper-xl pipe
        the switch was guaranteed to unload one call later. The headroom check
        then *re-measures* after the unload rather than crediting it
        arithmetically -- arenas can retain part of what was freed -- and the
        paired ``_log_mem`` lines are the record of how much commit actually
        came back.

        Returns True when an unload happened, False on a warm hit or no pipe.
        """
        if self._text2image is None:
            return False
        generation = fetch.store_generation()
        if self._t2i_generation != generation:
            log.info(
                "model store changed (generation %s -> %s); reloading %s",
                self._t2i_generation,
                generation,
                self._t2i_key,
            )
        elif self._t2i_key != base_key:
            log.info("switching image model %s -> %s", self._t2i_key, base_key)
        else:
            return False
        _log_mem(f"before unloading {self._t2i_key}")
        await self._evict_t2i(forget=True)
        _log_mem("after stale t2i unload")
        return True

    async def unload_text2image(self) -> None:
        """Drop the resident image model, unconditionally. A no-op when there
        is none.

        Public and on the loop thread, because the one caller that is not this
        worker is ``service.downloads.uninstall``: Windows refuses to delete a
        safetensors file another process has mapped, and this process is that
        other process. It reaches this through ``svc.call_on_loop`` -- nothing
        but the loop thread may touch the pipe.
        """
        await self._evict_t2i(forget=True)

    async def _maybe_evict_caches(self) -> None:
        """Drop the host-side inference caches an idle session is not using.

        Three session-long caches whose release functions existed and were
        called from nothing in ``src/``: BiRefNet (a child process holding
        ~4 GB of working set and **6.5 GiB of private commit**, measured
        2026-08-21), the 2D pose model, and ``bench.metrics``' DINOv2 -- which
        is the one of the three that can hold *VRAM*, because its cache key
        carries the device and the benchmark path resolves ``None`` to cuda.

        Stamped off the last finished job rather than off each cache, because
        none of the three records a last-used of its own and an unload of an
        empty cache is a no-op worth no thread at all.

        Throttled by *time*, not latched by a bool. The latch it replaces was
        set on the first idle pass and cleared only when a GPU job completed --
        so a cache populated after that pass by a path that is not a GPU job
        stayed resident indefinitely. Matting is exactly such a path: an export
        or a matte preview loads it through ``TaskRunner``
        (``service/derive.py``, ``service/matte.py``), never through the queue,
        and its child -- the largest single charge this sweep can give back --
        then sat there until the app closed (MDL-12).

        A repeat pass over three empty caches is one thread hop per idle
        timeout -- ten minutes apart, three no-op function calls -- which is the
        cost the latch was avoiding and is not worth a correctness hole.
        """
        now = time.monotonic()
        idle = self.config.trellis_idle_timeout
        if now - self._last_job_at <= idle or now - self._caches_evicted_at <= idle:
            return
        self._caches_evicted_at = now
        log.info("evicting idle host inference caches")

        def _drop() -> None:
            from .bench import metrics
            from .pipelines import matting

            for unload in (matting.unload, pose2d.unload, metrics.unload):
                # One that fails must not strand the other two: these are
                # advisory releases on an idle queue, exactly as _audit_mesh is
                # advisory on a finished job.
                try:
                    unload()
                except Exception:  # noqa: BLE001
                    log.exception("evicting an idle inference cache failed")

        await asyncio.to_thread(_drop)

    # --- progress plumbing ---

    def _step_progress(
        self, job_id: str, phase: str, label: str, step: int, total: int
    ) -> None:
        """One diffusion step's worth of progress, for whichever phase asked.

        The two callers differ only in their phase and their label; the
        arithmetic (and the ``step``/``step_total`` fields the bar reads to
        draw "step n/N") is the same sentence twice.
        """
        self.progress.update(
            job_id,
            phase=phase,
            label=label,
            inner=step / max(total, 1),
            inner_next=(step + 1) / max(total, 1),
            nominal=1.0,
            detail=f"step {step}/{total}",
            step=step,
            step_total=total,
        )

    def _note_blender(self, proc: Any) -> None:
        """``rigging.run_worker``'s ``on_start``: remember the live Popen.

        Every Blender stage passed the same two-line closure. There is no
        polite abort -- bpy is inside a C solve and checks nothing -- so
        holding the handle is the only thing that lets ``request_cancel`` stop
        one.
        """
        self._blender = proc

    def _resolve_base_key(self, params: dict[str, Any], *, default: str | None = None) -> str:
        """Which checkpoint a stage should load, from params it may not trust.

        Two chains, and the difference is deliberate. With a ``default``, that
        default *is* the requirement: the pixel-sheet and sprite stages need a
        CFG-capable base for ``PIXEL_SHEET_LORA``, so falling back through
        ``config.t2i_model`` -- which may name a turbo or klein checkpoint --
        would restyle bare. Silent, too: the caller that asked for a specific
        default already knows what it is getting.

        Without one, the general chain: what the row asked for, then the host's
        configured model, then the registry's own default, warning at each
        step. Params can predate a registry entry being renamed or removed, and
        ``WARLOCK_T2I_MODEL`` itself can name a key that does not exist -- so
        falling back beats failing a job the user cannot fix, and beats a bare
        KeyError in the worker.
        """
        if default is not None:
            base_key = str(params.get("base_model") or default)
            return base_key if base_key in models.BASE_MODELS else default
        base_key = str(params.get("base_model") or self.config.t2i_model)
        if base_key not in models.BASE_MODELS:
            log.warning("unknown base_model %r; using %s", base_key, self.config.t2i_model)
            base_key = self.config.t2i_model
        if base_key not in models.BASE_MODELS:
            log.warning("unknown t2i_model %r; using %s", base_key, models.DEFAULT_BASE_MODEL)
            base_key = models.DEFAULT_BASE_MODEL
        return base_key

    async def _acquire_t2i(self, spec, base_key: str, cond: Any = _ALWAYS_CONDITIONED):
        """The VRAM handoff every t2i stage makes, asked once. -> (pipe, handoff)

        Three reasons, one owner -- see ``_needs_handoff``. When it says yes:
        sequential handoff, because both models cannot fit -- free the VRAM
        held by the 3D server before the image model loads. Threaded because
        ``stop()`` can block for up to ~20 s, and it fires at the exact moment
        the user starts watching the progress bar.

        The stale pipe (wrong base, or wrong store generation) is evicted
        after the stop -- keeping the stop-before-load ordering intact -- and
        *before* the commit-headroom check, so the check measures a host that
        has already given the old pipe's weights back rather than charging
        them against the load that replaces them.

        ``cond`` defaults to ``_ALWAYS_CONDITIONED`` because the img2img stages
        build a fresh ``Conditioning`` per band and per view -- there is no
        single object to hand the predicate, and every one of their passes is
        conditioned. Two callers pass a real value instead: ``_generate`` hands
        over its actual ``cond``, and ``_tile_sheet`` hands over the grid
        guide's.

        No caller passes ``None`` today -- the one that did was the ground set,
        deleted on 2026-08-18 -- and the rule is written down anyway because it
        is the one that is wrong by *omission*. A stage conditioned on nothing
        but its prompt has to say ``None`` explicitly rather than take the
        default: taking the default is not a conservative choice, it stops a
        warm trellis the accounting in ``vram.estimate`` has already priced as
        co-resident, so the two halves of the same rule disagree.
        """
        handoff = _needs_handoff(
            spec,
            cond,
            exclusive=bool(self.config.vram_exclusive),
            trellis_running=self.trellis.running,
        )
        if handoff:
            await asyncio.to_thread(self.trellis.stop)
            _log_mem("after trellis stop")
        await self._evict_stale_t2i(base_key)
        # Immediately before the load, and after the handoff and the stale
        # eviction have given back whatever they were going to: this is the
        # last moment the answer is still about the allocation that is about
        # to happen. Skipped only when a *loaded* pipe survived the eviction --
        # that is exactly the warm same-key same-generation pipe, no load is
        # coming, and refusing would fail a job for memory it is not about to
        # ask for. Everything else means a load is coming: None (including the
        # generation-forced reload of a same-key pipe, which used to skip this
        # check entirely), and the retained-but-unloaded object ``_generate``'s
        # handoff teardown keeps around -- ``.loaded`` is what answers "is a
        # pipe resident" (see that teardown's comment), and gating on the
        # object alone let every reuse of it re-allocate ~16 GiB unchecked.
        if self._text2image is None or not self._text2image.loaded:
            _require_commit_headroom(
                f" before loading {spec.label}",
                "Close other applications, or use a smaller image model.",
                need_gib=_host_peak_gib(spec),
            )
        return await self._get_text2image(base_key), handoff

    async def _release_t2i(self, t2i: Any, spec: Any) -> None:
        """Give the checkpoint back, if this job was the reason it was resident.

        Every caller does this in a ``finally``, so a cancelled stage does not
        leave 7 GB resident. The offload term is the same rule ``_acquire_t2i``
        applies: ``trim()`` would give back the CUDA pool, and an offloaded
        pipe's ~16 GiB is host weights that only a full ``unload`` releases --
        and this pipe must not still be holding them when trellis restarts.

        **Unconditional since 2026-08-21.** It used to fire only under
        ``vram_exclusive`` or an OFFLOAD spec, on the reasoning that a card
        with room for both models should not throw away a checkpoint the next
        job may want warm. That reasoning is about the *card*, and the
        constraint that actually bit is the *host*: a resident pipe is ~7 GiB
        of VRAM whose WDDM backing, plus the pipe's own arenas, is charged
        against system commit -- and ``_require_commit_headroom`` refuses the
        next job on a percentage of that commit. A 63.5 GiB machine with a
        14.2 GiB pagefile sat at 96% while holding a pipe for a job that had
        finished ten minutes of idle ago, with 24 GiB of RAM free.

        So the trade is taken the other way: the checkpoint goes back when the
        stage that loaded it ends, and a back-to-back job pays one reload
        rather than the queue holding memory against a job that may never come.
        ``spec`` stays in the signature -- the offload distinction still
        decides *how much* is being given back, and the argument above is
        written in terms of it.

        ``_generate``'s teardown is still deliberately *not* this -- see the
        comment at its own ``finally``, which reaches the same conclusion by a
        different route because it hands straight on to the mesh stage.

        The pipe is unloaded and **kept**, which is the second thing that
        changed. Forgetting it was right while this only ran on the exclusive
        and offload paths, where the point was to make room for something
        different; as the ordinary end of every t2i stage it would throw away
        the cache entry on every job and construct a second ``Text2Image`` for
        the next one at the same base. ``_generate``'s handoff branch already
        argues this and already unloads without clearing: ``.loaded`` is what
        every reader asks -- the idle sweep, the dispatch credit,
        ``_evict_stale_t2i`` -- and ``load()`` rebuilds from ``_pipe is None``,
        so an unloaded instance is reusable rather than stale. Both teardown
        families now do the same thing for the same reason.
        """
        await asyncio.to_thread(t2i.unload)

    def _check_resources(self, job: dict[str, Any]) -> None:
        """The dispatch-time half of admission control.

        service.validation.check_vram refuses a job the *card* cannot hold at
        submit time. This catches what has changed since: another application
        that took VRAM in the meantime, or a host whose commit charge has
        climbed to the wall while the job sat in the queue. Failing here costs
        a job; not failing here has cost the machine.
        """
        _require_commit_headroom(
            "",
            "Close other applications or restart Warlock before running this job.",
        )
        need, image_term = vram.estimate_job_parts(
            job, exclusive=bool(self.config.vram_exclusive)
        )
        if need <= 0:
            return
        device = vram.device_memory()
        if device is None:
            return
        # Against *free*, not total: whatever we already hold is part of the
        # budget this job runs inside. But `need` prices every model from
        # zero, and free has already been debited for anything resident -- so
        # what is resident *and* counted in `need` must be credited back, or
        # it is charged twice. The double charge is how a warm SDXL pipe made
        # every follow-up job on a 32 GiB card refuse itself. The WDDM caveat
        # in vram.py still applies: this is the secondary check behind the
        # submit-time one, not a replacement for it.
        headroom = device.free_gib
        if self.trellis.running:
            # Counted in `need` for both stages under coexist, and given back
            # by the handoff's stop() before anything loads under exclusive.
            # No kind gate needed: need <= 0 already returned for rig jobs.
            headroom += vram.TRELLIS_GIB
        if image_term > 0 and self._text2image is not None and self._text2image.loaded:
            # Gated on "did the estimate charge for a checkpoint", not on the
            # job's kind. The kind list this replaced said `text` alone, while
            # `vram.estimate` prices pixel_sheet, sprite_synthesis and
            # retexture with a full image-model term too -- so the natural
            # 2D-then-3D flow on a 32 GiB coexist card refused a sprite
            # synthesis by ~3 GiB for reusing the pipe it was being charged
            # for, and waiting out the 600 s idle eviction "fixed" it, which
            # reads as a phantom VRAM leak (MDL-06). An image-kind job still
            # gets nothing, because its estimate carries no checkpoint term and
            # `image_term` is therefore 0.
            #
            # Measured reserved, not SDXL_GIB: reserved is what free was
            # actually debited by (7.52 GiB for sdxl-base-1.0 against the
            # flat 7.0 estimate). A base switch unloads the old pipe before
            # loading the new (_acquire_t2i via _evict_stale_t2i), so the
            # credit holds even when the resident base is not the one this
            # job wants.
            # The unmeasurable case reads the registry: a flat SDXL_GIB is
            # 3 GiB short of the offloaded klein entry's declared peak.
            mem = vram_gib()
            headroom += mem[1] if mem is not None else _resident_t2i_gib(self._t2i_key)
        if need > headroom:
            # The submit-time refusal's remedies, shared rather than restated
            # (N113): this is the check that fires *after* the user has waited
            # in the queue, and it used to be the one with the least to say
            # about what to change.
            raise RuntimeError(
                vram.dispatch_shortfall_message(
                    need,
                    headroom,
                    device.free_gib,
                    job.get("params") or {},
                    exclusive=bool(self.config.vram_exclusive),
                )
            )

    async def _process(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        claimed = await asyncio.to_thread(self.store.claim, job_id)
        if not claimed:
            # Cancelled or deleted between next_queued() and here.
            return
        self.current_job_id = job_id
        self._cancel = _Cancel(job_id)
        # A cold trellis server loads ~8 GB inside its first stage. Text jobs
        # that hand off stop the server outright; the rest leave a warm server
        # warm. A rig job never touches trellis, so it is never cold regardless
        # of the server's state.
        #
        # The offload term is here as well as in _needs_handoff because an
        # offloaded text job now *always* leaves trellis cold, whatever the flag
        # says, and the progress bar's nominal timing should say so rather than
        # promising a warm-server ETA it cannot meet. Conditioning is
        # deliberately not a term: it only forces a handoff when trellis is
        # already running, in which case the first disjunct is False and the
        # job is cold either way -- and unlike the other two, whether cond
        # exists is not knowable from the row without preparing it.
        cold = job["kind"] in ("text", "image") and (
            not self.trellis.running
            or (
                job["kind"] == "text"
                and (
                    self.config.vram_exclusive
                    or vram.offloaded_base(job.get("params") or {})
                )
            )
        )
        self.progress.begin(job_id, job["kind"], cold=cold)
        error: str | None = None
        # Whether the job got past the door. Only work that *ran* may re-arm the
        # idle clock below -- see that ``finally`` for the loop this closes.
        admitted = False
        try:
            self._check_resources(job)
            admitted = True
            await self._generate(job)
        except Exception as exc:
            if not self._cancel.event.is_set():
                log.exception("job %s failed", job_id)
                # The verdict first, the log file second. Writing the log can
                # itself raise -- a full or read-only disk is one of the ways
                # a job fails in the first place -- and doing it first left
                # `error` None, so the finally below recorded the job as
                # *done*: a successful-looking job with no model.glb and no
                # message anywhere.
                error = errors.friendly(exc)
                with contextlib.suppress(OSError):
                    errors.write_error_log(self.config.job_dir(job_id), exc)
                # Both strings are extracted by now, so the traceback has no
                # readers left -- and it is holding every frame between here and
                # the raise, which on a failed conditioned job means the
                # ControlNet (~2.5 GB) and the CLIP-ViT-H encoder (~1.2 GB)
                # still have a live reference while the reclaim passes below
                # run. The drop-every-reference doctrine had this one gap on the
                # error path; ``_on_task_done`` already does exactly this to the
                # task-side equivalent (MDL-16).
                exc.__traceback__ = None
                # And one pool pass now that nothing references those frames.
                # The teardown inside ``_conditioned`` already ran a collect,
                # but it ran *while the exception was propagating through it*,
                # so the conditioning tensors were still reachable and it freed
                # nothing. This is a trim and never an unload: a failed job is
                # not a reason to throw away a checkpoint the next job wants
                # warm, which is the same line ``_release_t2i`` draws.
                t2i = self._text2image
                if t2i is not None and t2i.loaded:
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(t2i.trim)
        finally:
            try:
                if self._cancel.event.is_set() and not self._cancel.committed:
                    await asyncio.to_thread(self.store.set_status, job_id, "cancelled")
                    # Through a thread, like every DB write in this same
                    # ``finally``. For a cancelled re-texture this ``rmtree``s
                    # ~24 rendered and baked images, and it was running on the
                    # ``warlock-loop`` thread that hosts dispatch and
                    # cancellation -- so the unlinks blocked the very loop that
                    # answers the next cancel (CON-07).
                    await asyncio.to_thread(self._discard_artifacts, job)
                else:
                    status = "error" if error is not None else "done"
                    finished = await asyncio.to_thread(
                        self.store.finish, job_id, status, error
                    )
                    if not finished:
                        # A cancel landed between claim() succeeding and this
                        # write (before self._cancel existed to observe it, or
                        # the API route's atomic cancel() won the DB race) --
                        # the DB already says cancelled; don't overwrite it and
                        # don't leave a viewable artifact behind.
                        await asyncio.to_thread(self._discard_artifacts, job)
                    elif status == "done":
                        # The rig first: it is work the user asked for, and
                        # recording the observation introduced the only await
                        # between the terminal write and this call. A shutdown
                        # landing on that await raises CancelledError -- which
                        # _record_observation deliberately does not catch --
                        # and would have skipped the auto-rig for a job already
                        # marked done. A diagnostic must not be able to cost
                        # that; in this order the worst it can cost is itself.
                        await self._maybe_queue_rig(job)
                        # Beside the rig and before the observation, for the
                        # same reason and with the same ordering argument: it
                        # is work the user asked for, and a diagnostic must not
                        # be able to cost it. The two never both fire -- one is
                        # a mesh job's follow-up and the other a reference's --
                        # so their order relative to each other says nothing.
                        await self._maybe_queue_sprite_sheet(job)
                        # Beside the other two and *after* the rig, which is
                        # the whole ordering: the queue is serial and FIFO, so
                        # a charsheet row minted here is claimed after the rig
                        # row minted above it has finished. The three never
                        # all fire -- a rig follows a mesh, a sprite sheet a
                        # reference, a character sheet a promoted mesh.
                        await self._maybe_queue_charsheet(job)
                        # And the fourth: the sheet half of "send this mesh to
                        # Troupe", which fires on the *rig* row the door minted
                        # rather than on a mesh. It cannot overlap the three
                        # above -- they are guarded on a mesh, a reference and
                        # a promoted mesh respectively, and this on a rig.
                        await self._maybe_queue_sheet_after_rig(job)
                        await self._record_observation(job_id)
                    else:
                        # An errored job too, and it is not a consolation
                        # prize: a job refused at the composition gate is the
                        # *measurement* -- "this checkpoint draws character
                        # sheets" is a fact about its settings, and the 17
                        # refusals of the 2026-08-07 rogue sweep recorded it
                        # nowhere and died with prune_jobs. A job that cleared
                        # the gate and then failed in trellis matters for the
                        # same rate, from the other side: a mean over refusals
                        # alone is 1.0 by construction.
                        #
                        # No rig: there is no mesh to rig. And nothing at all
                        # on a cancel, which is the branch above -- that is the
                        # user changing their mind rather than a measurement,
                        # and it is the status that discards the artifacts a
                        # row would be describing.
                        await self._record_observation(job_id)
            finally:
                # Unconditionally, and in a nest of its own: the terminal write
                # can raise (`database is locked`, a full disk) and these four
                # lines are how the worker lets go of the job. Skipping them
                # left current_job_id naming a job that was no longer running,
                # a stale _Cancel event, and a ProgressBus entry that never
                # ended -- so every later job's trellis output was reported
                # against the dead one, and prune/delete refused to touch it.
                self.current_job_id = None
                self._cancel = None
                self._blender = None
                # The idle clock the host-side cache eviction runs off, re-armed
                # here rather than at the top: a job may well have populated one
                # of those caches (a 2D export mattes, a bench run embeds), so
                # the throttle has to reopen on every finish, not only on the
                # ones that did.
                #
                # **Only for a job that was admitted**, and that condition is
                # the whole point. A job refused by ``_check_resources`` has
                # loaded nothing and touched no cache, so re-arming for it made
                # the refusal postpone the cleanup that would lift the refusal:
                # "host memory is 96% committed" invites a retry, each retry
                # pushed eviction another ``trellis_idle_timeout`` away, and the
                # 6.5 GiB BiRefNet child that eviction exists to kill stayed
                # resident for as long as the user kept trying. The app could
                # not recover on its own from the one state it most needed to.
                # A refusal is not a finish.
                if admitted:
                    self._last_job_at = time.monotonic()
                    self._caches_evicted_at = 0.0
                self.progress.end(job_id)
