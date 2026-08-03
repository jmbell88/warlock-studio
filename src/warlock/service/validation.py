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

from .. import rigging
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
    "bone_count",
    "sheet_id",
    "cells",
)


def check_seed(name: str, value: int | None) -> None:
    if value is not None and not 0 <= value <= MAX_SEED:
        raise Invalid(f"{name} must be between 0 and {MAX_SEED}", field=name)


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
