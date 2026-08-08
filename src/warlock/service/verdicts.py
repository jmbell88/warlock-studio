"""Accept/Reject verdicts on a job, and the one function that records them.

Verdicts used to be an append-only JSONL file beside a sweep run directory,
keyed by a unit key that only meant anything inside that run. They are rows
now, keyed by job id, which is what lets *any* finished asset be judged --
daily use and sweep units feeding one pool.

**The vector is snapshotted onto the row on purpose.** It is denormalized and
that is the point: ``prune_jobs`` deletes job rows and their directories, and
the learning corpus has to outlive the assets it was learned from. A verdict
that only pointed at a job id would evaporate the first time a workshop was
tidied up -- taking with it every finding the user spent an afternoon
recording.

**The AI-judge seam is unchanged.** ``source`` is a free string, not an enum:
``latest_verdicts`` keys on (job_id, source, stage), so a future judge writing
``ai:<model>`` sits beside a human's verdict rather than overwriting it, and
``unverdicted_models(source="ai:...")`` lets a judge run resume exactly as a
human review session does.

**And ``stage`` says which question a verdict answers.** Three values: ``model``
(is this a good mesh -- every verdict recorded before migration 7, and the
default), ``reference`` (is this a good 2D asset) and ``blank`` (will this
reconstruct). The last two are labels *about an image*, and they are two
questions rather than one refinement of a question: in 2D mode the image is the
deliverable, so a dramatic plate with pillars and a cast shadow is a better
asset and a worse blank. They therefore coexist on one row, which is why intent
is a column and not a naming convention on ``source``.

That distinction reshapes one rule rather than adding to it. The
``status == 'done'`` gate is about **the artifact a verdict judges**, not about
the status word: a mesh verdict needs a mesh, and an image label needs the
image. A job refused at the composition gate has one -- the gate runs after the
picture is drawn -- and those refusals are the most informative negatives a
blank probe can be trained on, so refusing to label them would throw away the
best rows in the corpus. What has not moved an inch is the mesh side: an accept
filed against a mesh that never existed poisons the corpus permanently, because
the vector snapshot outlives the job.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..vectors import prompt_hash
from . import findings
from .core import WarlockService
from .errors import Invalid

VERDICTS = ("accept", "reject")

# Rejection reasons a reviewer picks from -- short and mesh/render-specific,
# not free text, so the findings table can tally them. Meaningful on reject
# only, but not refused on accept: a reviewer who mis-clicks a reason before
# switching to Accept should not be blocked by it.
REASONS = ("holes", "bad-shape", "bad-texture", "wrong-style", "broken")

SOURCE_HUMAN = "human"

# Which question a verdict answers. ``model`` first because it is the default
# and every row written before migration 7 is one.
STAGES = ("model", "reference", "blank")

# The image stages: a label about a picture rather than about a mesh.
IMAGE_STAGES = ("reference", "blank")

# Where an image label's subject is, in the order to look. A text job writes
# reference.png (what trellis actually saw); an upload has only its input.
# Mirrors ``review_mode.REFERENCE_NAMES``, which is the same question asked by
# the pane rather than by the service.
IMAGE_NAMES = ("reference.png", "input.png")


def record_verdict(
    svc: WarlockService,
    job_id: str,
    *,
    verdict: str,
    reasons: Iterable[str] = (),
    source: str = SOURCE_HUMAN,
    stage: str = "model",
) -> dict[str, Any]:
    """File one verdict against a job, with a snapshot of its config vector."""
    if verdict not in VERDICTS:
        raise Invalid(f"verdict must be one of {list(VERDICTS)}", field="verdict")
    if stage not in STAGES:
        raise Invalid(f"stage must be one of {list(STAGES)}", field="stage")
    reason_list = [str(r) for r in reasons]
    unknown = [r for r in reason_list if r not in REASONS]
    if unknown:
        raise Invalid(
            f"unknown reason(s) {unknown}; expected one of {list(REASONS)}", field="reasons"
        )
    if not source:
        raise Invalid("source must be non-empty", field="source")

    job = svc.require_job(job_id)
    if stage in IMAGE_STAGES:
        # An image label judges the picture, so the picture is what has to
        # exist. Deliberately not gated on ``done``: a job refused at the
        # composition gate is *errored* and has its reference on disk, and those
        # refusals are the most informative negatives a blank probe can learn
        # from -- throwing them away would leave the probe trained only on
        # images the hand-written rules already liked.
        if not any((svc.job_dir(job_id) / name).exists() for name in IMAGE_NAMES):
            raise Invalid("that job has no reference image to judge")
    elif job["status"] != "done":
        # A verdict is a judgement about artifacts, and the vector snapshot is
        # permanent -- it outlives the job on purpose. Filing one against a
        # queued or failed unit poisons the corpus with accepts for meshes
        # that never existed.
        raise Invalid(f"job is {job['status']}; a verdict needs a finished asset")
    # The sweep context rides along denormalized, like the vector: a matched
    # pair (same sweep, same seed, one param differing) must still be pairable
    # after delete_sweep has taken the job rows.
    params = job.get("params") or {}
    seed = params.get("seed")
    row_id = svc.store.add_verdict(
        job_id,
        source=source,
        verdict=verdict,
        reasons=reason_list,
        vector=findings.config_vector(job),
        sweep_id=job.get("sweep_id"),
        sweep_unit=job.get("sweep_unit") or "",
        seed=seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
        prompt_hash=prompt_hash(job.get("prompt")),
        stage=stage,
    )
    return {
        "id": row_id,
        "job": job_id,
        "verdict": verdict,
        "reasons": reason_list,
        "stage": stage,
    }
