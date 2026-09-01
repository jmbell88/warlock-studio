"""Input caps and id guards, applied at the door.

Every constant here bounds something a caller supplies. They live in one
module because the interesting property is that they are *all* checked before
anything is written -- a rejected request must not leave an input.png, a job
row or a place in the queue behind.
"""

from __future__ import annotations

import re
import secrets
from typing import Any

from .. import rigging, vram
from .errors import Invalid, NotFound, invalid_from

ALLOWED_RESOLUTIONS = {512, 1024, 1536}

# A submit may ask for several reference candidates at once. Bounded because
# each is a real queued job holding a place in the serial worker.
MAX_REFERENCE_COUNT = 8

# And a promotion may ask for several *mesh* candidates. Far smaller, because
# the two are not the same purchase: a reference candidate is four steps of
# SDXL, and a mesh candidate is roughly two minutes of the serial GPU worker --
# so three of them is already six minutes during which nothing else runs. The
# generic ``count`` refusal for output="model" in ``create_job`` stands
# unchanged; candidates are the deliberate exception and go through their own
# entry point (``jobs.promote_candidates``), which is what keeps "N meshes per
# submit" from being reachable by accident from every other caller.
MAX_MESH_CANDIDATES = 3

# Ceiling for list() and the internal full-history reads (prune). Bounded so a
# caller can't ask the single sqlite connection for everything at once and
# stall every other reader behind it.
MAX_LIST_LIMIT = 5000

# Exactly what create_job generates: uuid4().hex[:12].
JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")

# The reference upload arrives as raw bytes; nothing between it and the disk
# but these two numbers. Bytes are checked before decode, pixels after (a flat
# 20 MP PNG is tiny on disk and enormous decoded -- the classic decompression
# bomb).
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000

# A GLB arriving at ``import_mesh``. Larger than the image ceiling because the
# thing being handed over is geometry with its textures inside it -- a
# reconstruction routinely runs to tens of megabytes -- and smaller than the
# layered-document ceiling because, unlike an .ora, nothing here is a stack of
# full-canvas buffers. The check is on the bytes, before anything is decoded or
# written: this is the door, and refusing at the door is what stops a job
# directory existing for a request that was never going to succeed.
MAX_MESH_BYTES = 100 * 1024 * 1024

# The first four bytes of a binary glTF: "glTF", little-endian, as a u32.
GLB_MAGIC = b"glTF"

# A rendered snapshot of the viewport at list size.
MAX_THUMB_BYTES = 512 * 1024

# Matches guidance.MAX_NEGATIVE_PROMPT: the prompt ends up in an SDXL text
# encoder that truncates far earlier anyway, so anything longer is noise or a
# mistake -- refuse it rather than store it forever.
MAX_PROMPT = 1000


def check_prompt(text: str | None, *, field: str = "prompt") -> None:
    """Refuse a prompt past ``MAX_PROMPT``. One spelling for the seven doors
    that used to carry the comparison by hand -- a rule change here reaches
    all of them, where before it reached whichever were remembered."""
    if text is not None and len(text) > MAX_PROMPT:
        raise Invalid(f"{field} must be at most {MAX_PROMPT} characters", field=field)

# Seeds are 31-bit end to end: _random_seed generates them, and
# sqlite/torch/trellis all take this range without surprises.
MAX_SEED = 2**31 - 1

MAX_JOB_NAME = 120
MAX_TAGS = 20
MAX_TAG_LEN = 32

# Everything the worker records on a finished job about its *artifacts*. A
# rerun or a promotion must not inherit these -- they describe the source
# run's mesh, and a new job wearing the old job's mesh_report claims a quality
# verdict about a mesh that doesn't exist yet. Keep in sync with what
# queue.py writes into params.
# **``mesh_seed``/``reference_seed`` are deliberately not here.** They are
# worker-recorded provenance and would otherwise qualify, but a stripped seed
# reads as "no seed was requested" and every door would then mint a fresh one
# by accident rather than on purpose. Instead each reroll/promotion path in
# ``service/_jobs_resubmit.py`` overwrites them explicitly. That compensation
# lives in a different file from this list, so
# ``tests/test_jobs_resubmit.py::test_every_door_that_copies_params_rerolls_the_seeds``
# ties the two ends together: a future reroll-shaped door that copies
# ``source["params"]`` and forgets the override would reproduce the previous
# run's seed -- a reroll that looks like it ran and returns a byte-identical
# result.
DERIVED_PARAMS = (
    "composed_prompt",
    "scale_factor",
    "mesh_audit",
    "mesh_report",
    "optimize",
    "transform",
    "weighting",
    "weighting_reason",
    "bone_count",
    # Which battery was rendered against this rig's artifacts. A rerun would
    # otherwise claim a QA sheet of a mesh it has not bound yet.
    "deform_qa",
    # Recorded by the *sheet* render worker about the atlas it produced. For
    # the ``pixel_sheet`` kind the same key is an input -- which sheet the
    # restyle depicts, written at the door -- so ``rerun_job`` re-seeds it
    # from the source row for that kind after this strip.
    "sheet_id",
    "cells",
    "reference_report",
    # Advisory, and about this run's pixels -- a reroll inheriting it would
    # claim a seam verdict about an image it is about to replace. The tile
    # *flag* is deliberately not here: that is an input, like output, and
    # stripping it would turn a rerolled tile into an object.
    "seam_report",
    # Advisory, and about *this* run's image: a reroll that inherited it would
    # wear a verdict about pixels it is about to replace.
    "rank",
    # Provenance for a reroll that already happened. It describes this run's
    # attempts, so a rerun inheriting it would claim retries it never made.
    "reference_attempts",
    # The same, for the mesh half: it describes this run's reconstructions, so
    # a rerun inheriting it would claim remeshes it never made.
    "mesh_attempts",
    # Operational evidence about an automatic rig/sheet insert after this run
    # finished. A rerun gets its own follow-up attempt and must not inherit the
    # earlier row's database failure as though it happened again.
    "followup_failures",
    "control_hint",
    # What the character sheet's pixel-art pass measured about *this* atlas:
    # orphans cleaned, colours used, which palette it chose. A rerun wearing it
    # would claim a colour count for pixels it has not rendered yet.
    "pixel_report",
    "recipe",
    # What the tile-sheet worker recorded about the sheet it painted -- the
    # finished pixel size, the palette it quantized to, the base and LoRA it
    # actually ran with. Derived by definition, so a reroll must start clean
    # rather than wear a report about the sheet it is replacing.
    "sheet_report",
    # A compiled tileset plan the *legacy grid* worker used to write here. It
    # has no writer any more -- a single guided generation has no per-cell
    # prompt and lays no Wang role, so the block described a structure the sheet
    # did not have (see ``_q_tilesheet``'s module docstring). It is listed here
    # rather than forgotten because rows drawn before 2026-08-29 still carry
    # one: they keep it, nothing reads it, and this is what stops a reroll of
    # such a row from copying the old claim onto a new sheet with nothing left
    # to re-derive it.
    "tile_plan",
    # What a re-texture did to this mesh's surface -- coverage, view count, the
    # recipe that restyled it. Recorded on the mesh's own row because that is
    # where a reader asks what its skin is, and derived, so a reroll must not
    # inherit a coverage figure about an atlas it is about to reconstruct from
    # scratch.
    "retexture",
    # What a remesh did to this mesh -- face count, quad fraction, which
    # path produced the surface, and the tiercheck verdict against the mesh
    # it replaced. Derived for ``retexture``'s reason exactly.
    "remesh",
    # What a training run produced -- the manifest it registered and its
    # final loss. A rerun trains again and registers its own.
    "lora_result",
    # Which canonical post-processing steps failed on *this* run. Derived by
    # definition -- it is a statement about the artifacts this reconstruction
    # produced -- so a reroll must start clean or it would wear a degradation
    # that belongs to the run it replaced. See ``ARTIFACT_HEALTH`` and
    # ``_q_mesh._note_degraded``.
    "degraded",
)

# The params key that records swallowed post-processing failures, named once so
# the worker that writes it and the readers that surface it cannot drift.
#
# ART-01: normalization (requested size, X/Z centring, floor grounding) and the
# mesh report are wrapped in catch-everything handlers -- deliberately, because
# a mesh that trimesh cannot parse must not fail a job whose ``source.glb`` is a
# perfectly good reconstruction. But the job then stayed ``done`` with nothing
# anywhere except a log line, so a user could export a visibly successful asset
# with the wrong pivot or the wrong scale and have no way to find out. The
# non-fatal decision is right; the silence was not.
ARTIFACT_HEALTH = "degraded"


def note_degraded(params: dict[str, Any], step: str, detail: str) -> None:
    """Record that a canonical post-processing step did not run.

    The service-layer half of ``_q_mesh._note_degraded``. Two copies rather than
    one import, because the queue layer may not import ``service`` and the
    service layer must not reach into a queue private -- the same shape as
    ``VECTOR_PARAMS`` living in ``vectors.py``. A test pins the key and the
    behaviour together so the pair cannot drift.

    Steps accumulate rather than overwrite: a mesh can fail more than one, and
    the second failure is not a correction of the first.
    """
    health = params.get(ARTIFACT_HEALTH)
    if not isinstance(health, dict):  # a hand-edited row; start fresh
        health = {}
    # **Rebound, never mutated in place**, which is the rule ``_optimize``,
    # ``_apply_scale`` and ``_audit_mesh`` already follow and the reason
    # ``_q_generate``'s remesh loop can snapshot the winning attempt with a
    # shallow ``dict(params)``. That loop's own comment names this exact
    # hazard -- "a callee that grew an in-place nested mutation would defeat
    # it" -- and ``setdefault`` plus ``health[step] =`` was that callee: with
    # ``WARLOCK_MESH_RETRIES`` set, attempt 2's failures wrote straight through
    # the shared dict into ``best["params"]["degraded"]``, so the job that
    # shipped attempt 1's mesh carried attempt 2's health record.
    params[ARTIFACT_HEALTH] = {**health, step: detail}

# The conditioning selection itself, which is an *input* rather than a derived
# value -- so it survives a reroll, unlike DERIVED_PARAMS. It does not survive
# a promotion or a remesh: those are `image` jobs that never touch SDXL, and a
# row claiming an IP-Adapter that cannot have run is a lie about provenance.
CONDITIONING_PARAMS = (
    "ip_adapter",
    "ip_scale",
    "control",
    "control_scale",
    "control_end",
    # Troupe's T-pose guide: which figure was drawn into control.png, and the
    # fact that it was drawn rather than derived. Conditioning by definition,
    # and here for the same reason as the five above -- a promotion and a
    # remesh are image jobs where SDXL never runs, so carrying these would
    # describe a run that cannot happen. The ``troupe`` block itself is *not*
    # here: it is the request for the sheet, and the promoted job is exactly
    # the one that has to honour it.
    "control_hint_source",
    "guide_variant",
)


def check_glb(data: bytes, field: str = "glb") -> None:
    """Refuse bytes that are not a binary glTF carrying at least one mesh.

    Belt and braces where the caller is Clay mode -- we author those bytes --
    but "a caller-supplied input is bounded at the door" is the invariant and
    this is the door. Both refusals are worth having on their own terms: bytes
    that are not a GLB would mint a ``done`` row whose ``model.glb`` no reader
    can open, and a GLB with no mesh in it would mint one whose every
    downstream export -- STL, OBJ, FBX, collision, sheet -- produced an empty
    file, each failing far from here with a message about its own format.

    Deliberately structural rather than semantic: the JSON chunk is parsed and
    the mesh list is looked at, and nothing is decoded. A mesh that is *bad* is
    the mesh report's business, not the door's.
    """
    from ..glbio import read_glb

    if not data.startswith(GLB_MAGIC):
        raise Invalid("that file is not a binary glTF (.glb)", field=field)
    try:
        doc, _buffer = read_glb(data)
    except Exception as exc:
        raise Invalid("that .glb could not be read", field=field) from exc
    meshes = doc.get("meshes") or []
    if not any(mesh.get("primitives") for mesh in meshes):
        raise Invalid("that .glb has no mesh in it", field=field)


def check_seed(name: str, value: int | None) -> None:
    if value is None:
        return
    # The type check is the point: `0 <= 1.5 <= MAX_SEED` and `0 <= True` are
    # both true, so a float or a bool used to reach the pipeline as a seed.
    # bool is a subclass of int and has to be excluded by hand.
    if not isinstance(value, int) or isinstance(value, bool):
        raise Invalid(f"{name} must be a whole number", field=name)
    if not 0 <= value <= MAX_SEED:
        raise Invalid(f"{name} must be between 0 and {MAX_SEED}", field=name)


# The two trellis-server launch settings a job may now pin. Bounds are the
# exe's own: --band is a narrow-band width in voxels (1 is degenerate, past ~64
# the "narrow" band is the whole volume and the run will not fit), and
# --tex-res is a texture edge in pixels. Both are checked at the door for the
# reason every other cap here is: the cost of a bad one is a trellis-server
# that fails to start two minutes into a queue, not an error the caller sees.
MIN_TRELLIS_BAND = 1
MAX_TRELLIS_BAND = 64
MIN_TRELLIS_TEX_RES = 128
MAX_TRELLIS_TEX_RES = 4096


def check_trellis_band(value: int | None) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise Invalid("trellis_band must be a whole number", field="trellis_band")
    if not MIN_TRELLIS_BAND <= value <= MAX_TRELLIS_BAND:
        raise Invalid(
            f"trellis_band must be between {MIN_TRELLIS_BAND} and {MAX_TRELLIS_BAND}",
            field="trellis_band",
        )


def check_trellis_tex_res(value: int | None) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise Invalid("trellis_tex_res must be a whole number", field="trellis_tex_res")
    if not MIN_TRELLIS_TEX_RES <= value <= MAX_TRELLIS_TEX_RES:
        raise Invalid(
            f"trellis_tex_res must be between {MIN_TRELLIS_TEX_RES} and "
            f"{MAX_TRELLIS_TEX_RES}",
            field="trellis_tex_res",
        )


def vram_plan(svc: Any) -> vram.Plan:
    """The resolved plan, or one derived from the config for a Runtime-less caller."""
    plan = getattr(svc, "vram_plan", None)
    if plan is not None:
        return plan
    return vram.plan(
        exclusive=svc.config.vram_exclusive,
        budget_gib=svc.config.vram_budget_gib,
        total_gib=svc.config.vram_total_gib,
        device=vram.device_memory(),
        explicit=svc.config.vram_exclusive_explicit,
    )


def check_vram(svc: Any, kind: str, stage: str, params: dict[str, Any]) -> None:
    """Refuse a job the card cannot hold, before anything is written.

    Here rather than in the worker because the worker is two minutes too late
    and, on Windows, does not reliably fail at all: overcommitted VRAM spills
    into shared system memory and becomes host commit, so the symptom is the
    machine dying rather than the job erroring. The remedy has to name a thing
    the user can change, which is why the message is built from the job's own
    parameters.
    """
    plan = vram_plan(svc)
    if not plan.enforced:
        return
    need = vram.estimate(kind, stage, params, exclusive=plan.exclusive)
    if not plan.fits(need):
        # The one refusal in this module with no ``field=``, and deliberately.
        # A field points the UI at the control to change, and this refusal has
        # no single one: ``vram.remedies`` offers whichever of the ControlNet
        # toggle, the IP-Adapter toggle, the resolution preset and an
        # environment variable actually apply to *this* job, usually more than
        # one and sometimes only the env var, which is not a control at all.
        # Naming any one of them would put the error beside a widget that is
        # not the answer while the sentence beside it lists three that are.
        raise Invalid(vram.shortfall_message(need, plan, params))


def install_remedy(label: str, download: str) -> str:
    """The "here is how to get it" half of every missing-weights refusal.

    One spelling, in one place, because there are four of these refusals and
    they used to lead with the terminal. The in-app route exists now and is the
    one most people can act on without leaving the window, so it goes first;
    the ``hf download`` line stays, because it is the only route on a headless
    box and because doctor prints the same string.

    Plain ``->`` rather than the arrow the manual's prose uses: this ends up in
    a toast, and toasts are plain text.
    """
    return (
        f'Install "{label}" in Settings -> Models, '
        f"or download it with:\n  {download}"
    )


def check_base_model_weights(
    svc: Any, base: Any, *, rows: tuple[str, ...] | None = None
) -> None:
    """Refuse a base model that is not on this host, by name and with its
    ``hf download`` line.

    Its own function because three doors need exactly this refusal and they do
    not agree on how the model was chosen: a text job reads it off ``params``, a
    sprite synthesis and a pixel sheet pin theirs to a constant. Sharing the
    check rather than the call site is what keeps the *sentence* one thing --
    the field is ``base_model`` at every door, which is the name of the control
    the UI puts the error beside.

    ``base`` is a ``models.BaseModel``; ``None`` is accepted and is a no-op, for
    the text door's sake, where an unknown key means the registry never claimed
    to know these weights and there is nothing to say about them.

    ``rows`` overrides what the refusal carries for an "install what this needs"
    button. Defaulted, it is this checkpoint's own row, which is the whole
    answer for a text job. A door standing in front of a *feature* -- sprites,
    pixel sheets -- passes the feature's whole missing set instead, so the user
    installs once rather than being refused three times running.
    """
    if base is None:
        return
    from .. import fetch

    ok, missing_lora = fetch.base_model_state(svc.config, base)
    if ok:
        return
    what = (
        f"its step-distillation LoRA is missing at {missing_lora}"
        if missing_lora is not None
        else "its weights are not downloaded"
    )
    raise Invalid(
        f"The image model {base.label!r} cannot run: {what}. "
        f"{install_remedy(base.label, fetch.download_text(svc.config, 'base', base))}",
        field="base_model",
        rows=(f"base:{base.key}",) if rows is None else rows,
    )


def check_weights(svc: Any, kind: str, params: dict[str, Any]) -> None:
    """Refuse a job whose selected weights are not on this host (F55).

    Beside ``check_vram`` and for exactly its reason: the worker is the wrong
    place to find this out. Without the guard the job queues, waits its turn
    behind whatever else is running, loads, and dies with a diffusers traceback
    that names a directory -- and the row goes to ``error`` carrying a
    ``base_model`` the user then has to work out was never downloaded. The
    doctor has known the answer, and the exact ``hf download`` line that fixes
    it, since it was written; this is that answer, at the door.

    Only text jobs, because only they touch SDXL: an image job's reference is
    its upload and no image model is loaded at all.

    Deliberately *not* checked here: the trellis exe and the GGUF weights. Those
    are the two fatal doctor rows, and a host missing them has a red banner at
    startup rather than a per-job surprise -- re-refusing every submit would say
    the same thing a second time in a worse place.
    """
    if kind != "text":
        return
    from .. import fetch, models

    check_base_model_weights(svc, models.BASE_MODELS.get(str(params.get("base_model") or "")))
    # The three optional selections, each refused the same way. A style LoRA is
    # the one that would otherwise fail *silently* rather than loudly --
    # ``_load_loras`` skips a missing style adapter -- so the job would finish,
    # look wrong, and carry a ``style_lora`` param claiming a style that never
    # ran. That row would then join the findings corpus as evidence about it.
    #
    # ``normalize`` stores the resolved *key*, not the spec object, so each
    # field is looked back up in the registry it came from -- the same tables
    # ``guidance`` validated it against, so a key that reaches here is known.
    optional = (
        ("style_lora", "lora", models.STYLE_LORAS),
        ("ip_adapter", "adapter", models.IP_ADAPTERS),
        ("control", "control", models.CONTROLNETS),
    )
    for field, kindname, table in optional:
        spec = table.get(str(params.get(field) or ""))
        if spec is None or fetch.present(svc.config, kindname, spec):
            continue
        raise Invalid(
            f"{spec.label!r} is selected but not downloaded. "
            f"{install_remedy(spec.label, fetch.download_text(svc.config, kindname, spec))}",
            field=field,
            rows=(f"{kindname}:{spec.key}",),
        )


def random_seed() -> int:
    """A fresh seed for a re-roll. 31-bit so it round-trips through an sqlite
    INTEGER (and a JS number, for as long as anything speaks JSON) unchanged."""
    return secrets.randbelow(2**31)


def check_job_id(job_id: str) -> None:
    """Reject anything that isn't a generated job id, before it reaches the FS.

    config.job_dir() is a bare data_dir / job_id join with no sanitisation, so
    every entry point that builds a path from a caller-supplied id needs this.
    Only get_file could actually be steered today -- the others gate on a DB
    lookup first -- but the check costs nothing and removes the class rather
    than the instance.
    """
    if not job_id:
        # The same split ``derive.get_file`` makes for an empty file name
        # (O117): "no such job" is deliberately ambiguous between a malformed
        # id and a real one that has been pruned, because saying which leaks
        # what the store holds. An empty id is neither -- it is a request with a
        # blank where an id should be, and it says nothing about any job.
        raise NotFound("no job was named")
    if not JOB_ID_RE.match(job_id):
        raise NotFound("no such job")


def check_pose_id(pose_id: str) -> None:
    """Same guard, same reasoning, for the ids that name files inside a job dir."""
    if not rigging.is_valid_id(pose_id):
        raise NotFound("no such pose")


def check_sheet_id(sheet_id: str) -> None:
    if not rigging.is_valid_id(sheet_id):
        raise NotFound("no such sheet")


def check_sprite_draft_id(draft_id: str) -> None:
    if not rigging.is_valid_id(draft_id):
        raise NotFound("no such sprite draft")


def valid_template(key: str | None, default: str) -> str:
    """A known skeleton template key. None falls back to the config default."""
    try:
        return rigging.get_template(key or default).key
    except ValueError as exc:
        raise invalid_from(exc, "That skeleton is not available", field="rig_template") from exc


# What a job's status *means* to someone waiting on it, rather than the word
# the row stores (E50). ``reference is queued`` is accurate and useless: it
# names a state machine the user has never been shown and says nothing about
# what to do. Keyed on the status column's own vocabulary so an unknown value
# falls through to the word itself rather than to a wrong sentence.
STATUS_SENTENCES = {
    "queued": "is still waiting in the queue",
    "running": "is still being generated",
    "error": "failed, so it has no image",
    "cancelled": "was cancelled, so it has no image",
}


def not_done_message(subject: str, status: str) -> str:
    """-> "<subject> <what that status means>", e.g. for a refused promotion."""
    return f"{subject} {STATUS_SENTENCES.get(status, f'is {status}')}."


def normalize_tags(raw: Any) -> str:
    """A list or a comma string -> a sorted, deduped, lowercase csv.

    Normalized on the way in rather than at every read: a filter that has to
    case-fold and trim on each keystroke over a thousand rows is the kind of
    thing that makes a UI feel broken, and 'Prop' and 'prop ' being two tags
    is the kind of thing that makes a workshop unsearchable.
    """
    if raw is None:
        return ""
    items = raw if isinstance(raw, list) else str(raw).split(",")
    tags = sorted({t.strip().lower() for t in (str(i) for i in items) if t.strip()})
    if len(tags) > MAX_TAGS:
        raise Invalid(f"at most {MAX_TAGS} tags", field="tags")
    if any(len(t) > MAX_TAG_LEN for t in tags):
        raise Invalid(f"a tag may be at most {MAX_TAG_LEN} characters", field="tags")
    return ",".join(tags)
