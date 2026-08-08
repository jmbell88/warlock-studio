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
from .errors import Invalid, NotFound

ALLOWED_RESOLUTIONS = {512, 1024, 1536}

# A submit may ask for several reference candidates at once. Bounded because
# each is a real queued job holding a place in the serial worker.
MAX_REFERENCE_COUNT = 8

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
    "control_hint",
    "recipe",
)

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
        raise Invalid(vram.shortfall_message(need, plan, params))


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
    if not JOB_ID_RE.match(job_id):
        raise NotFound("no such job")


def check_pose_id(pose_id: str) -> None:
    """Same guard, same reasoning, for the ids that name files inside a job dir."""
    if not rigging.is_valid_id(pose_id):
        raise NotFound("no such pose")


def check_sheet_id(sheet_id: str) -> None:
    if not rigging.is_valid_id(sheet_id):
        raise NotFound("no such sheet")


def valid_template(key: str | None, default: str) -> str:
    """A known skeleton template key. None falls back to the config default."""
    try:
        return rigging.get_template(key or default).key
    except ValueError as exc:
        raise Invalid(str(exc), field="rig_template") from exc


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
