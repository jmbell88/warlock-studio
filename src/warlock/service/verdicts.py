"""Verdicts on a job -- a graded mesh judgement or a binary image label -- and
the one function that records either.

**A mesh verdict is a grade, not a bit.** An integer -5..+5, where +5 is
game-useable as-is and -5 is nothing recoverable. The binary corpus it replaces
failed at its own purpose: 3 accepts against 81 rejects on 2026-08-07, in which a
slab with no geometry, a smeared texture and a mesh a modeller would fix in five
minutes are all the same row. A bit can say a mesh failed and can never say how
close it came. The scale, its +-3 backfill and its ``grade >= +3`` usable cut are
argued in ``docs/measurements/2026-08-09-grade-scale.md``.

**``verdict`` survives as a derived column with exactly one writer.** This
function is that writer, via ``vectors.verdict_for_grade``, which is what leaves
prune retention, the judge's label reads, the ``latest_verdicts`` SQL and every
findings-v3 reader unchanged while there is still only one place the cut lives. A
caller may not pass a ``verdict`` for a mesh at all -- two ways to say one thing
is how they come to disagree.

**Tags are optional at every grade.** The five bad spellings are the old
``REASONS`` tuple, frozen: the stored corpus carries those exact strings in the
``reasons`` column and there is no alias table, so a rename would not migrate
evidence but split it. What changed is the concept -- from "reasons a reviewer
rejected" to "what is true of this mesh" -- which is why a good tag on a negative
grade, or a bad one on a positive, is legal rather than a contradiction to refuse.

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

from ..vectors import (
    BAD_TAGS,
    BINARY_GRADES,
    GOOD_TAGS,
    GRADE_MAX,
    GRADE_MIN,
    TAGS,
    USABLE_GRADE,
    prompt_hash,
    verdict_for_grade,
)
from . import findings
from .core import WarlockService
from .errors import Invalid

__all__ = [
    "BAD_TAGS",
    "BINARY_GRADES",
    "GOOD_TAGS",
    "GRADE_MAX",
    "GRADE_MIN",
    "IMAGE_NAMES",
    "IMAGE_STAGES",
    "SOURCE_HUMAN",
    "STAGES",
    "TAGS",
    "USABLE_GRADE",
    "VERDICTS",
    "record_verdict",
    "verdict_for_grade",
]

# The two values the derived ``verdict`` column ever takes, and the two an image
# label is recorded with directly. A mesh verdict no longer names one at the
# door: it names a grade, and ``verdict_for_grade`` derives this.
VERDICTS = ("accept", "reject")

SOURCE_HUMAN = "human"

# Which question a verdict answers. ``model`` first because it is the default
# and every row written before migration 7 is one.
STAGES = ("model", "reference", "blank")

# The image stages: a label about a picture rather than about a mesh.
IMAGE_STAGES = ("reference", "blank")

# Where an image label's subject is, in the order to look. A text job writes
# reference.png (what trellis actually saw); an upload has only its input.
# The one spelling. ``studio.review_mode`` used to carry an identical copy and
# read both -- the judge.STAGES hazard, minus the test that makes it survivable
# -- and now imports this.
IMAGE_NAMES = ("reference.png", "input.png")


def record_verdict(
    svc: WarlockService,
    job_id: str,
    *,
    verdict: str | None = None,
    grade: int | None = None,
    reasons: Iterable[str] = (),
    source: str = SOURCE_HUMAN,
    stage: str = "model",
) -> dict[str, Any]:
    """File one verdict against a job, with a snapshot of its config vector.

    **The two stages take different arguments, and each refuses the other's.**
    A mesh verdict names a ``grade`` and the ``verdict`` column is derived from
    it; an image label names a ``verdict`` and keeps ``grade`` NULL. Passing the
    wrong one is an ``Invalid`` rather than a silent preference, because there is
    exactly one door in and that is what stops the derived column from ever
    disagreeing with the grade it was derived from.
    """
    if stage not in STAGES:
        raise Invalid(f"stage must be one of {list(STAGES)}", field="stage")

    if stage in IMAGE_STAGES:
        # Binary, permanently: these feed binary logistic probes, so a grade
        # would be thresholded straight back to a bit, and the two-key loop is
        # what makes a hundred-image pass viable at all.
        if grade is not None:
            raise Invalid(
                f"a {stage} label is binary; it takes a verdict, not a grade", field="grade"
            )
        if verdict not in VERDICTS:
            raise Invalid(f"verdict must be one of {list(VERDICTS)}", field="verdict")
    else:
        # A mesh verdict is graded. Refusing a passed ``verdict`` outright rather
        # than checking it against the derivation: two ways to say one thing is
        # how they come to disagree, and the caller has no business asserting the
        # cut.
        if verdict is not None:
            raise Invalid(
                "a mesh verdict is graded; pass grade, not verdict", field="verdict"
            )
        if not isinstance(grade, int) or isinstance(grade, bool):
            # ``bool`` is an ``int`` in Python and ``True`` would file a +1.
            raise Invalid(
                f"grade must be an integer in {GRADE_MIN}..{GRADE_MAX}", field="grade"
            )
        if not GRADE_MIN <= grade <= GRADE_MAX:
            raise Invalid(
                f"grade must be in {GRADE_MIN}..{GRADE_MAX}, not {grade}", field="grade"
            )
        verdict = verdict_for_grade(grade)

    # Tags are legal at *every* grade, which is the change of meaning migration
    # 10 came with: these used to be reasons a reviewer rejected, and are now
    # descriptions of what is true of the mesh. One namespace, because the two
    # vocabularies are disjoint strings.
    reason_list = [str(r) for r in reasons]
    unknown = [r for r in reason_list if r not in TAGS]
    if unknown:
        raise Invalid(
            f"unknown tag(s) {unknown}; expected one of {list(TAGS)}", field="reasons"
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
        grade=grade,
    )
    return {
        "id": row_id,
        "job": job_id,
        "verdict": verdict,
        "grade": grade,
        "reasons": reason_list,
        "stage": stage,
    }
